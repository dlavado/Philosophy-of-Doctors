

from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch 
from torch import nn
from torch.nn import functional as F

from core.models.FC_Classifier import Classifier_OutLayer
from core.models.lit_modules.lit_wrapper import LitWrapperModel
from torchmetrics.functional import accuracy



class Gaussian_Kernel(nn.Module):

    def __init__(self, kernel_size=(3,3), factor=1, stddev=1, mean=0) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.factor = factor
        self.stddev = stddev
        self.mean = mean
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.kernel = self.compute_kernel()

    def gaussian(self, x:torch.Tensor, epsilon=1e-8) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[0]-1)/2, (self.kernel_size[1]-1)/2], dtype=torch.float, requires_grad=True, device=self.device)

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        gauss_dist = x_c_norm**2 - (self.mean + epsilon)**2 

        return self.factor*torch.exp((gauss_dist**2) * (-1 / (2*(self.stddev + epsilon)**2)))
    
    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - torch.sum(tensor) / torch.prod(torch.tensor(self.kernel_size)) 
    
    def compute_kernel(self):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[0], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2) # Nx2 vector form of the indices
       
       
        kernel = self.gaussian(floor_idxs)
        kernel = self.sum_zero(kernel)
        kernel = torch.t(kernel).view(1, *self.kernel_size) # CxHxW

        #assert kernel.requires_grad

        return kernel



class IENEO(nn.Module):

    def __init__(self, num_gaussians:int, kernel_size:tuple) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.num_gaussians = num_gaussians
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # centers are the means of the gaussians
        self.mus = nn.Parameter(torch.rand(num_gaussians, requires_grad=True, device=self.device))

        # sigmas are the standard deviations of the gaussians, so they must be positive
        self.sigmas = nn.Parameter(torch.rand(num_gaussians, requires_grad=True, device=self.device))

        self.gaussians = nn.ModuleList([Gaussian_Kernel(self.kernel_size, mean=self.mus[i], stddev=self.sigmas[i]) for i in range(num_gaussians)])

        # convex combination weights of the gaussians
        self.lambdas = torch.rand(num_gaussians, requires_grad=True, device=self.device)
        self.lambdas = torch.softmax(self.lambdas, dim=0) # make sure they sum to 1
        self.lambdas = nn.Parameter(self.lambdas)


    def maintain_convexity(self):
        self.lambdas = nn.Parameter(torch.softmax(self.lambdas, dim=0)) # make sure they sum to 1

    def forward(self, x):

        kernels = torch.stack([gauss.compute_kernel() for gauss in self.gaussians])
        # apply the kernels to the input

        conv = F.conv2d(x, kernels, padding='same')

        # apply the convex combination weights
        # unsqueeze to regain the channel dimension
        cvx_comb = torch.sum(conv*self.lambdas[None, :, None, None], dim=1).unsqueeze(1) # BxCxHxW
        
        return cvx_comb


class IENEO_Fam(nn.Module):

    def __init__(self, num_operators:int, num_gaussians:int, kernel_size:tuple) -> None:
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        self.num_operators = num_operators
        self.num_gaussians = num_gaussians
        self.kernel_size = kernel_size

        # centers are the means of the gaussians
        self.mus = nn.Parameter(torch.rand((num_operators, num_gaussians), requires_grad=True, device=self.device))

        # sigmas are the standard deviations of the gaussians, so they must be positive
        self.sigmas = nn.Parameter(torch.rand((num_operators, num_gaussians), requires_grad=True, device=self.device))

        # convex combination weights of the gaussians
        self.lambdas = torch.rand((num_operators, num_gaussians), requires_grad=True, device=self.device)
        self.lambdas = torch.softmax(self.lambdas, dim=1) # make sure they sum to 1
        self.lambdas = nn.Parameter(self.lambdas)     


    def maintain_convexity(self):
        with torch.no_grad():
            self.lambdas[:, -1] = 1 - torch.sum(self.lambdas[:, :-1], dim=1)
        assert self.lambdas.requires_grad

    def compute_kernel(self):
        return torch.stack([Gaussian_Kernel(self.kernel_size, stddev=self.sigmas[ij % self.num_operators, ij % self.num_gaussians], 
                                               mean=self.mus[ij % self.num_operators, ij % self.num_gaussians]).compute_kernel() 
                                               for ij in range(self.num_operators*self.num_gaussians)])

    def forward(self, x):

        self.kernels = self.compute_kernel()
        # kernels.shape = (num_operators*num_gaussians, 1, kernel_size[0], kernel_size[1])
        
        # print(kernels.shape, kernels.device)

        # apply the kernels to the input
        conv = F.conv2d(x, self.kernels) # shape = (B, num_operators*num_gaussians, H, W)
        conv = conv.view(conv.shape[0], self.num_operators, self.num_gaussians, *conv.shape[2:]) # shape = (B, num_operators, num_gaussians, H, W)

        # print(conv.shape)
        # apply the convex combination weights
        # unsqueeze to regain the channel dimension
        cvx_comb = torch.sum(conv*self.lambdas[None, :, :, None, None], dim=2)

        # print(cvx_comb.shape)
        # input("Press Enter to continue...")

        return cvx_comb





class IENEO_Layer(nn.Module):


    def __init__(self, hidden_dim=128, kernel_size=(3, 3), gauss_hull_size=10) -> None:
        """
        IENEO Layer is a layer that applies a gaussian hull to the input and then max pools it

        A gaussian hull is a convex combination of gaussian kernels, where the weights of the convex combination are learned
        This grants the layer equivariance with respect to isomorphisms, so Euclidean transformations (i.e., translations, rotations, reflections, etc.)

        Parameters
        ----------

        hidden_dim: int
            The number of gaussians hulls to  instantiate
        
        kernel_size: tuple
            The size of the gaussian kernels to use

        gauss_hull_num: int
            The number of gaussians to use in each gaussian hull
        """
        super().__init__()
    
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ieneo = IENEO_Fam(hidden_dim, gauss_hull_size, kernel_size).to(self.device)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def get_cvx_coeffs(self):
        return self.ieneo.lambdas

    
    def maintain_convexity(self):
        self.ieneo.maintain_convexity()
    
    def forward(self, x):
        x = self.ieneo(x)
        x = self.bn(x)
        # x = self.relu(x)
        x = self.maxpool(x)
        return x
        



class IENEONet(nn.Module):


    def __init__(self, in_channels=1, hidden_dim=128,ghost_sample:torch.Tensor = None, gauss_hull_size=5, kernel_size=(3,3), num_classes=10):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_extractor = IENEO_Layer(hidden_dim, kernel_size, gauss_hull_size).to(device)
        ghost_shape = self.feature_extractor(ghost_sample.to(device)).shape
        self.classifier = Classifier_OutLayer(torch.prod(torch.tensor(ghost_shape[1:])), num_classes).to(device)

    def print_cvx_combination(self):
        print(f"mean cvx combination: {torch.sum(self.feature_extractor.get_cvx_coeffs(), dim=1).mean()}")

    def maintain_convexity(self):
        self.feature_extractor.maintain_convexity()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    

class Lit_IENEONet(LitWrapperModel):
    
    def __init__(self, 
                 in_channels=1, 
                 hidden_dim=128, 
                 ghost_sample:torch.Tensor = None,
                 gauss_hull_size=5,
                 kernel_size=(3,3),
                 num_classes=10,
                 optimizer_name = 'adam', 
                 learning_rate=0.01, 
                 metric_initializer=None):

        model = IENEONet(in_channels, hidden_dim, ghost_sample, gauss_hull_size, kernel_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer)

    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)


    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # self.model.maintain_convexity()
        self.model.print_cvx_combination()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    
        