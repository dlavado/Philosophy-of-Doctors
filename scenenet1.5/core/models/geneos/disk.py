

import torch

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
from core.models.geneos.GENEO_kernel_torch import GENEO_kernel


class Disk(GENEO_kernel):



    def __init__(self, name, kernel_size, plot=False, **kwargs):
        """
        GENEO kernel that encodes a disk.\n 


        Parameters
        ----------

        radius - float \in ]0, kernel_size[1]] :
            disk's radius;

        height - float \in ]0, kernel_size[0]] :
            disk's height;

        Optional
        --------

        sigma - float:
            scalar    
        """
        super().__init__(name, kernel_size)  


        self.radius = kwargs['radius']
        if self.radius is None:
            raise KeyError("Provide a radius for the disk in the kernel.")

        self.radius = self.radius.to(self.device)

        self.height = kwargs['height']
        if self.height is None:
            raise KeyError("Provide a height for the disk in the kernel.")

        self.height = self.height.to(self.device).to(torch.int)

        self.sigma = kwargs.get('sigma', 1)


    def mandatory_parameters():
        return ['radius']
    
    def geneo_parameters():
        return disk.mandatory_parameters() + ['sigma']
    
    def geneo_random_config(name="disk", kernel_size=None):
        rand_config = GENEO_kernel.geneo_random_config(name, kernel_size)

        k_size = rand_config['kernel_size']

        geneo_params = {
            'radius' : torch.randint(1, k_size[1], (1,))[0] / 2, #float \in [0.5, kernel_size[1]/2]
            'height' : torch.randint(0, k_size[0]-1, (1,))[0] #int \in [1, kernel_size[0]]
        }

        rand_config['geneo_params'] = geneo_params
        rand_config['non_trainable'] = ['height']

        return rand_config
    
    def gaussian(self, x:torch.Tensor, epsilon=1e-8) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2], dtype=torch.float, device=self.device, requires_grad=True)

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        gauss_dist = x_c_norm**2 #- (self.radius + epsilon)**2 

        return self.sigma*torch.exp((gauss_dist**2) * (-1 / (2*(self.radius + epsilon)**2)))
    
    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - torch.sum(tensor) / torch.prod(torch.tensor(self.kernel_size[1:]))
    

    def compute_kernel(self):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2)
        
        floor_vals = self.gaussian(floor_idxs)
        
        floor_vals = self.sum_zero(floor_vals)
        floor_vals = torch.t(floor_vals).view(self.kernel_size[1:])

        kernel = torch.zeros_like(floor_vals, device=self.device)

        if self.height == 0:
            kernel = torch.cat([floor_vals[None], torch.zeros((self.kernel_size[0] - 1, *self.kernel_size[1:]), device=self.device)], dim=0)
        
        elif self.height == self.kernel_size[0] - 1:
            kernel = torch.cat([torch.zeros((self.kernel_size[0] - 2, *self.kernel_size[1:]), device=self.device), floor_vals[None]], dim=0)

        else:

            kernel = torch.cat([torch.zeros((self.height - 1, *self.kernel_size[1:]), device=self.device), floor_vals[None]], dim=0)
            kernel = torch.cat([kernel, torch.zeros((self.kernel_size[0] - self.height, *self.kernel_size[1:]), device=self.device)], dim=0)

        return kernel
    


if __name__ == '__main__':
    import torch.nn as nn

    disk = Disk('disk', [4, 3, 3], radius=torch.tensor(1.0), height=torch.tensor(3.0), plot=True)
    

