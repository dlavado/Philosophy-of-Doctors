

import torch 
from torch import nn



class Gaussian_Hull(nn.Module):

    def __init__(self, num_gaussians=10) -> None:
        super().__init__()


        # centers are the means of the gaussians
        self.mu = nn.Parameter(torch.rand(num_gaussians, requires_grad=True))

        # sigmas are the standard deviations of the gaussians, so they must be positive
        self.sigmas = nn.Parameter(torch.rand(num_gaussians, requires_grad=True))

        # convex combination weights of the gaussians
        self.lambdas = nn.Parameter(torch.rand(num_gaussians, requires_grad=True))
        self.lambdas = torch.softmax(self.lambdas, dim=0) # make sure they sum to 1



    def forward(self, x):

        x = x.repeat(self.mu.shape[0], 1)

        return torch.sum(self.lambdas*self.gaussian(x, self.mu, self.sigmas), dim=1)

    def gaussian(self, x, mu, sigma):
        """
        Gaussian function

        Parameters
        ----------

        x: torch.Tensor
            Input tensor
        
        mu: torch.Tensor
            Mean of the gaussian
        
        sigma: torch.Tensor
            Standard deviation of the gaussian
        """
        
        return torch.exp(-torch.pow(x - mu, 2.) / (2 * torch.pow(sigma, 2.)))


class IENEO_Layer(nn.Module):


    def __init__(self) -> None:
        super().__init__()



class IENEO_net(nn.Module):


    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        