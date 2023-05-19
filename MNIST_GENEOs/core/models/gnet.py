

from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch 
from torch import nn
from torch.nn import functional as F

from core.models.FC_Classifier import Classifier_OutLayer
from core.models.lit_wrapper import LitWrapperModel
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

        



class IENEO_Layer(nn.Module):


    def __init__(self, hidden_dim=128, kernel_size=(3, 3), gauss_hull_num=20) -> None:
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
        self.gauss_block = self.gaussian_block(hidden_dim, kernel_size, gauss_hull_num)
        self.norm_pool = nn.Sequential(
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(),
                            nn.MaxPool2d(2, 2)
                        ).to(self.device)
    
    def gaussian_block(self, out_channels, kernel_size, gauss_hull_num):
        return nn.ModuleList([IENEO(gauss_hull_num, kernel_size).to(self.device) for _ in range(out_channels)]).to(self.device)

        
    
    def maintain_convexity(self):
        self.gauss_block[0].maintain_convexity()
    
    def forward(self, x):
        gaussians = torch.cat([gauss(x) for gauss in self.gauss_block], dim=1) # BxCxHxW

        return self.norm_pool(gaussians)
        



class IENEONet(nn.Module):


    def __init__(self, in_channels=1, hidden_dim=128,ghost_sample:torch.Tensor = None, kernel_size=(3,3), num_classes=10):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_extractor = IENEO_Layer(hidden_dim, kernel_size).to(device)
        ghost_shape = self.feature_extractor(ghost_sample.to(device)).shape
        self.classifier = Classifier_OutLayer(torch.prod(torch.tensor(ghost_shape[1:])), num_classes).to(device)


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    

class Lit_IENEONet(LitWrapperModel):
    
    def __init__(self, 
                 in_channels=1, 
                 hidden_dim=128, 
                 ghost_sample:torch.Tensor = None,
                 kernel_size=(3,3),
                 num_classes=10,
                 optimizer_name = 'adam', 
                 learning_rate=0.01, 
                 metric_initializer=None):

        model = IENEONet(in_channels, hidden_dim, ghost_sample, kernel_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer)


    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)

        # print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is not None:
        #             print(name, param.data.shape, param.device, param.grad.device)
        #         else:
        #             print(name, param.data.shape, param.device, "NONE")

        # print(x.device, y.device, logits.device, self.criterion(logits, y).device)
        # input("Press Enter to continue...")
        
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

        return loss, preds, y
    
    def training_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "train")
        if self.train_metrics is not None:
            self.train_metrics(preds, y).update()
        return {"loss": loss}
    
    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        #self.model.feature_extractor.maintain_convexity()
        return super().training_step_end(step_output)
    
    def training_epoch_end(self, outputs) -> None:
        if self.train_metrics is not None:
            self._epoch_end_metric_logging(self.train_metrics, 'train', print_metrics=False)

    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "val")
        acc = accuracy(preds, y)
        if self.val_metrics is not None:
            self.val_metrics(preds, y).update()
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs) -> None: 
        if self.val_metrics is not None: # On epoch metric logging
           self._epoch_end_metric_logging(self.val_metrics, 'val', print_metrics=True)
    
    def test_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "test")
        if self.test_metrics is not None:
            self.test_metrics(preds, y).update()
        return {"test_loss": loss}
        