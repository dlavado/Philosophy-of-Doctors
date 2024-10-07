# %%

import torch


import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
from core.models.GENEONets.geneos.GENEO_kernel_torch import GENEO_kernel

class negSpherev2(GENEO_kernel):

    def __init__(self, name, kernel_size, plot=False, **kwargs):
        """
        GENEO kernel that encodes a negative sphere.\n

        Required
        --------

        radius - float \in ]0, kernel_size[1]] :
        sphere's radius;

        Optional
        --------

        sigma - float:
        sigma variable for the gaussian distribution when assigning weights to the kernel;


        Returns
        -------
            3D torch tensor with the negative sphere kernel 
        """

        super().__init__(name, kernel_size, angles=kwargs.get('angles', None))

        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the sphere.")

        if  kwargs.get('neg_factor') is None:
            raise KeyError("Provide a negative factor for each sphere weight.")


        self.radius = kwargs['radius'].to(self.device)
        self.neg_factor = kwargs['neg_factor'].to(self.device)

        if plot:
            print("--- Neg. Sphere Kernel ---")
            print(f"radius = {self.radius:.4f}; {type(self.radius)};")
            print(f"neg_factor = {self.neg_factor:.4f}; {type(self.neg_factor)};")

        self.sigma = kwargs.get('sigma', 1)
      

  
    def mandatory_parameters():
        return ['radius', 'neg_factor']

    def geneo_parameters():
        return negSpherev2.mandatory_parameters() + ['sigma']

    def geneo_random_config(name="neg_sphere", kernel_size=None):
        rand_config = GENEO_kernel.geneo_random_config(name, kernel_size)

        geneo_params = {
            'radius' : torch.randint(1, rand_config['kernel_size'][1], (1,))[0], #int \in [1, kernel_size[1]] ,
            'neg_factor': torch.randint(1, 10, (1,))[0]/10, #float \in [0, 1]
            'sigma' : torch.randint(5, 10, (1,))[0] / 10 #float \in [0, 1]
        }

        rand_config['geneo_params'] = geneo_params

        rand_config['non_trainable'] = []

        return rand_config


    def gaussian(self, x:torch.Tensor, rad=None, sig=None, epsilon=1e-8) -> torch.Tensor:
        shape = torch.tensor(self.kernel_size, dtype=torch.float, device=self.device, requires_grad=True)
        center = (shape - 1) / 2

        if rad is None:
            rad = self.radius
        if sig is None:
            sig = self.sigma

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        gauss_dist = x_c_norm**2 #- (self.radius + epsilon)**2 

        return sig*torch.exp((gauss_dist**2) * (-1 / (2*(rad + epsilon)**2)))

    def sum_negfactor(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - (torch.sum(tensor) + self.neg_factor) / self.volume
    
    
    def compute_kernel(self):

        idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[0], dtype=torch.float, device=self.device, requires_grad=True),
                            torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True)
                            )
            ).T.reshape(-1, 3)

        kernel = self.gaussian(idxs)
        kernel = (-self.neg_factor)*kernel
        kernel = self.sum_negfactor(kernel)
        kernel = torch.t(kernel).view(self.kernel_size)

        if self.angles is not None:
            kernel = self.rotate_tensor(self.angles, kernel)            

        return kernel

    


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.transforms import Compose
    from core.datasets.torch_transforms import Voxelization, ToTensor, ToFullDense
    from scripts import constants as const
    from utils import voxelization as Vox
    from utils import pcd_processing as eda
    import numpy as np

    sphere = negSpherev2('cy', (6, 6, 6), radius=torch.tensor(5), 
                                          sigma=torch.tensor(1), 
                                          neg_factor=torch.tensor(0.5))
    kernel = sphere.compute_kernel(True)


