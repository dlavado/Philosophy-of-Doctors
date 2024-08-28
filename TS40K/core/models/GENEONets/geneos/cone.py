
# %%


import torch

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
sys.path.insert(3, '../../../..')
from core.models.GENEONets.geneos.GENEO_kernel_torch import GENEO_kernel


class Cone(GENEO_kernel):

    def __init__(self, name, kernel_size, plot=False, **kwargs):
        """
        GENEO kernel that encodes a cone.\n


        Required
        --------
        `cone_radius` - float \in  ]0, kernel_size[1]]:
        cone's base radius

        `cone_inc` - float \in ]0, 1[
        cone's inclination

        Optional
        --------
        `epsilon` - float \in [0, 1]:
        threshold below and beyond the radius to consider in the pattern's definition.
        x^2 + y^2 <= (radius + epsilon)^2 && x^2 + y^2 >= (radius - epsilon)^2;

        `sigma` - float:
        sigma variable for the gaussian distribution when assigning weights to the kernel;

        """

        super().__init__(name, kernel_size, angles=kwargs.get('angles', None))  

        if kwargs.get('apex') is None:
            raise KeyError("Provide a height for the cone.")

        if kwargs.get('cone_inc') is None:
            raise KeyError("Provide an inclination for the cone.")


        self.apex = kwargs['apex'].to(self.device)
        self.cone_inc = kwargs['cone_inc'].to(self.device)

        if kwargs.get('cone_radius') is None:
            self.cone_radius = torch.tensor(kernel_size[1]-1, device=self.device)
        else:
            self.cone_radius = kwargs['cone_radius'].to(self.device)
        
        self.sigma = kwargs.get('sigma', 1)
        self.epsilon = kwargs.get('epsilon', 1e-8)

        if plot:
            print("--- Cone Kernel ---")
            print(f"apex = {self.apex:.4f}; {type(self.apex)};")
            print(f"cone_radius = {self.cone_radius:.4f}; {type(self.cone_radius)};")
            print(f"cone_inc = {self.cone_inc:.4f}; {type(self.cone_inc)};")
            print(f"sigma = {self.sigma:.4f}; {type(self.sigma)};")  
            print(f"epsilon = {self.epsilon:.4f}; {type(self.epsilon)};")
    
    
    def mandatory_parameters():
        return ['apex', 'cone_radius','cone_inc']

    def geneo_parameters():
        return Cone.mandatory_parameters() + ['sigma']

    def geneo_random_config(name="cone", kernel_size=None):
        rand_config = GENEO_kernel.geneo_random_config(name, kernel_size)

        k_size = rand_config['kernel_size']

        geneo_params = {
            'apex': torch.randint(0, k_size[0]-1, (1,))[0], 
            'cone_radius' : torch.randint(1, k_size[1], (1,))[0] / 2,
            'cone_inc' : torch.rand(1,)[0], #float \in [0, 1]
            'sigma' : torch.randint(5, 10, (1,))[0] / 5 #float \in [1, 2]
        }

        rand_config['geneo_params'] = geneo_params

        rand_config['non_trainable'] = ['apex']

        return rand_config


    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - torch.sum(tensor) / torch.prod(torch.tensor(self.kernel_size[1:])) 


    def gaussian(self, x:torch.Tensor, rad=None, sig=None, epsilon=1e-8) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2], dtype=torch.float, device=self.device, requires_grad=True)

        if rad is None:
            rad = self.cone_radius
        if sig is None:
            sig = self.sigma

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        gauss_dist = x_c_norm**2 #- (self.radius + epsilon)**2 

        return sig*torch.exp((gauss_dist**2) * (-1 / (2*(rad + epsilon)**2)))

    def compute_kernel(self):

        hc = min([self.apex.to(torch.int).item(), self.kernel_size[0]]) # Even though the apex is a model parameter, it is used as an index, so it must be converted to an int
        cone_height = torch.tensor(self.kernel_size[0]) - hc

        # zero values till apex height
        kernel = torch.zeros((hc, self.kernel_size[1], self.kernel_size[2]), device=self.device, requires_grad=True)

        floor_idxs = torch.stack(
                        torch.meshgrid(torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                                    torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True))
                    ).T.reshape(-1, 2)

        self.cone_inc = torch.clamp(self.cone_inc, 0, 0.499) # tan is not defined for 90 degrees

        for h in range(cone_height -1, -1, -1): # to -1 since range is non inclusive of the right bound: [start, end)
            height_vals = self.gaussian(floor_idxs, rad=self.cone_radius*h*torch.tan(self.cone_inc*torch.pi), sig=None)
            height_vals = self.sum_zero(height_vals)
            height_vals = torch.t(height_vals).view(1, self.kernel_size[1], self.kernel_size[2])  
            kernel = torch.cat((height_vals, kernel), dim=0)


        if self.angles is not None:
            kernel = self.rotate_tensor(self.angles, kernel)

        return kernel
    
  


# %%
if __name__ == "__main__":
    from torch import nn
    # from scripts import constants as const
    from utils import voxelization as Vox
    # from utils import pcd_processing as eda
    # from torchvision.transforms import Compose
    # from core.datasets.torch_transforms import Voxelization, ToTensor, ToFullDense

    # from core.datasets.ts40k import TS40K

    #build_data_samples([DATA_SAMPLE_DIR], SAVE_DIR)
    # vxg_size = (64, 64, 64)
    # composed = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size, vox_size=None),
    #                     ToTensor(), 
    #                     ToFullDense(apply=(True, True))])
    
    # ts40k = TS40K(dataset_path=const.TS40K_PATH, transform=composed)

    # vox, vox_gt = ts40k[0]
    # vox, vox_gt = vox.to(torch.float), vox_gt.to(torch.float)
    # print(vox.shape)
    # Vox.plot_voxelgrid(vox.numpy()[0])
    # Vox.plot_voxelgrid(vox_gt.numpy()[0])
    
    # cone = cone_kernel('cy', (9, 9, 9), plot = True, radius=torch.tensor(1.0),  
    #                                     sigma=torch.tensor(2.0), 
    #                                     apex=nn.Parameter(torch.tensor(0.0)), 
    #                                     cone_radius=torch.tensor(0.6),
    #                                     cone_inc=nn.Parameter(torch.tensor(0.0)))

    cone = Cone('arrow', (9, 7, 7), plot = False, 
                                    sigma=torch.tensor(3.0), 
                                    apex=nn.Parameter(torch.tensor(4.0)), 
                                    cone_radius=torch.tensor(4),
                                    cone_inc=nn.Parameter(torch.tensor(0.1)))
    
    # Vox.plot_voxelgrid(vox[0])
  
    # cone.convolution(vox.view((1, *vox.shape)).to(cone.device))

    kernel = cone.compute_kernel()
    print(kernel.shape)
    #print(kernel)
    Vox.plot_voxelgrid(kernel.cpu().detach().numpy(), color_mode='density')

# %%
