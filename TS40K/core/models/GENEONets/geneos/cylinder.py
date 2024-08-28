



import torch

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
sys.path.insert(3, '../../../..')
from core.models.GENEONets.geneos.GENEO_kernel_torch import GENEO_kernel


class cylinder_kernel(GENEO_kernel):

    def __init__(self, name, kernel_size, plot=False, **kwargs):
        """
        Creates a 3D torch tensor that demonstrates a cylinder.\n

        Parameters
        ----------
        radius - float:
        radius of the cylinder's base; radius <= kernel_size[1];

        sigma - float:
        variance for the gaussian function when assigning weights to the kernel;

        Returns
        -------
            3D torch tensor with the cylinder kernel 
        """

        super().__init__(name, kernel_size)  


        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cylinder in the kernel.")

        self.radius = kwargs['radius'].to(self.device)

        if plot:
            print("--- Cylinder Kernel ---")
            print(f"radius = {self.radius:.4f}; {type(self.radius)};")

        self.sigma = kwargs.get('sigma', 1)
        

    def gaussian(self, x:torch.Tensor, epsilon=0) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2], dtype=torch.float, device=self.device, requires_grad=True)

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        circle_x = x_c_norm**2 - (self.radius + epsilon)**2 

        return torch.exp((circle_x**2) * (-1 / (2*self.sigma**2))) # shape = (N, 1)
 
    def compute_kernel(self):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2)
        
        floor_vals = self.gaussian(floor_idxs)
        
        floor_vals = self.sum_zero(floor_vals)
        floor_vals = torch.t(floor_vals).view(self.kernel_size[1:])            
        #assert floor_vals.requires_grad
    
        kernel = torch.tile(floor_vals, (self.kernel_size[0], 1, 1))
    
        return kernel 

       
    def mandatory_parameters():
        return ['radius']
    
    def geneo_parameters():
        return cylinder_kernel.mandatory_parameters() + ['sigma']

    def geneo_random_config(name="cylinder", kernel_size=None):
        rand_config = GENEO_kernel.geneo_random_config(name, kernel_size)

        geneo_params = {
            'radius' : torch.randint(1, rand_config['kernel_size'][1], (1,))[0] / 2 ,
            'sigma' : torch.randint(5, 10, (1,))[0] / 5 #float \in [1, 2]
        }   

        rand_config['geneo_params'] = geneo_params

        return rand_config




class cylinderv2(GENEO_kernel):

    def __init__(self, name, kernel_size, plot=False, **kwargs):
        """
        Creates a 3D torch tensor that demonstrates a cylinder.\n

        Parameters
        ----------
        radius - float:
        radius of the cylinder's base; radius <= kernel_size[1];

        sigma - float:
        variance for the gaussian function when assigning weights to the kernel;

        Returns
        -------
            3D torch tensor with the cylinder kernel 
        """

        super().__init__(name, kernel_size, angles=kwargs.get('angles', None))

        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cylinder in the kernel.")

        self.radius = kwargs['radius'].to(self.device)

        if plot:
            print("--- Cylinder Kernel ---")
            print(f"radius = {self.radius:.4f}; {type(self.radius)};")

        self.sigma = kwargs.get('sigma', 1)


    def gaussian(self, x:torch.Tensor, epsilon=1e-8) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2], dtype=torch.float, device=self.device, requires_grad=True)

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        gauss_dist = x_c_norm**2 #- (self.radius + epsilon)**2 

        return torch.exp((gauss_dist**2) * (-1 / (2*(self.radius + epsilon)**2)))


    def compute_kernel(self, plot=False):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2)

        floor_vals = self.gaussian(floor_idxs)
        floor_vals = self.sum_zero(floor_vals)
        floor_vals = torch.t(floor_vals).view(self.kernel_size[1:])            
        #assert floor_vals.requires_grad
    
        kernel = torch.tile(floor_vals, (self.kernel_size[0], 1, 1))

        # if self.angles is not None:
        #     kernel = self.rotate_tensor(self.angles, kernel)

        return kernel 
    
    def mandatory_parameters():
        return ['radius']
    
    def geneo_parameters():
        return cylinderv2.mandatory_parameters() + ['sigma']

    def geneo_random_config(name="cylinder", kernel_size=None):
        rand_config = GENEO_kernel.geneo_random_config(name, kernel_size)

        geneo_params = {
            'radius' : torch.randint(1, rand_config['kernel_size'][1], (1,))[0] / 2 ,
            'sigma' : torch.randint(5, 10, (1,))[0] / 5 #float \in [1, 2]
        }   

        rand_config['geneo_params'] = geneo_params

        return rand_config





# %%
if __name__ == "__main__":
    from utils import voxelization as Vox

    #rot_angles = torch.tensor([0.0, 0.0, 0.5], requires_grad=True, device='cuda')

    # cy = cylinder_kernel('cy', (6, 6, 6), radius=torch.tensor(2), sigma=torch.tensor(2))
    cy = cylinderv2('cy', (9, 6, 6), radius=torch.tensor(1.0), sigma=torch.tensor(1), angles=None)

    cy_kernel = cy.compute_kernel(False)
    print(cy_kernel)

    cy.plot_kernel(cy_kernel)
    Vox.plot_voxelgrid(cy_kernel.cpu().detach().numpy(), color_mode='density')


    # cy_rotated_kernel = cy.rotate_tensor(torch.tensor([0., 0., 0.6]), cy_kernel)

    # print(cy_rotated_kernel)


