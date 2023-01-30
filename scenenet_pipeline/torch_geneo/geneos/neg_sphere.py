# %%
import itertools
import time
from matplotlib import cm
import numpy as np
import sympy as smp
import sympy.vector as smpv
import sympy.physics.vector as spv
import sympytorch as spt
from scipy import integrate as intg
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import torch


import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from VoxGENEO import Voxelization as Vox
from EDA import EDA_utils as eda
from torch_geneo.datasets.ts40k import ToTensor, torch_TS40K
from torch_geneo.geneos.GENEO_kernel_torch import GENEO_kernel_torch

class neg_sphere_kernel(GENEO_kernel_torch):

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

        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the sphere.")

        if  kwargs.get('neg_factor') is None:
            raise KeyError("Provide a negative factor for each sphere weight.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.radius = kwargs['radius'].to(self.device)
        self.neg_factor = kwargs['neg_factor'].to(self.device)

        if plot:
            print("--- Neg. Sphere Kernel ---")
            print(f"radius = {self.radius:.4f}; {type(self.radius)};")
            print(f"neg_factor = {self.neg_factor:.4f}; {type(self.neg_factor)};")

        self.sigma = 1
        if kwargs.get('sigma') is not None:
            self.sigma = kwargs['sigma']
            if plot:
                print(f"sigma = {self.sigma:.4f}; {type(self.sigma)};")           

        self.plot = plot
        
        super().__init__(name, kernel_size)  

  
    def mandatory_parameters():
        return ['radius', 'neg_factor']

    def geneo_parameters():
        return neg_sphere_kernel.mandatory_parameters() + ['sigma']

    def geneo_random_config(name='GENEO_rand'):
        rand_config = GENEO_kernel_torch.geneo_random_config()

        # vol = torch.prod(torch.tensor(rand_config['kernel_size']))

        geneo_params = {
            'radius' : torch.randint(1, rand_config['kernel_size'][1], (1,))[0], #int \in [1, kernel_size[1]] ,
            'neg_factor': torch.randint(1, 10, (1,))[0]/10, #float \in [0, 1]
            'sigma' : torch.randint(5, 10, (1,))[0] / 10 #float \in [0, 1]
        }

        rand_config['geneo_params'] = geneo_params

        rand_config['non_trainable'] = []

        rand_config['name'] = 'neg'

        return rand_config

    def geneo_smart_config(name="Smart_Neg_Sphere"):

        config = {
            'name' : name,
            'kernel_size': (9, 6, 6),
            'plot': False,
            'non_trainable' : [],
            
            'geneo_params' : {
                                'radius' : torch.tensor(3.0) ,
                                'sigma' :  torch.tensor(2.0),
                                'neg_factor' :  torch.tensor(0.5)
                             }
        }

        return config

    def gaussian(self, x:torch.Tensor, epsilon=0) -> torch.Tensor:
        shape = torch.tensor(self.kernel_size, dtype=torch.float, device=self.device, requires_grad=True)
        center = (shape - 1) / 2

        x_c = x - center # Nx3
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        circle_x = x_c_norm**2 - (self.radius + epsilon )**2 

        return torch.exp((circle_x**2) * (-1 / (2*self.sigma**2)))

    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - torch.sum(tensor) / self.volume
    
 
    def compute_kernel(self, plot=False):

        idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[0], dtype=torch.float, device=self.device, requires_grad=True),
                            torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True)
                            )
            ).T.reshape(-1, 3)

        # idxs = list(itertools.product(list(range(self.kernel_size[0])), list(range(self.kernel_size[1])), list(range(self.kernel_size[2]))))
        # idxs = torch.from_numpy(np.array([*idxs], dtype=np.float)).to(self.device) # (z*x*y, 3) 
        # idxs.requires_grad_()

        kernel = self.gaussian(idxs)
        kernel = self.sum_zero(kernel) - self.neg_factor
        kernel = torch.t(kernel).view(self.kernel_size)            

        if plot:
            print(f"kernel shape = {kernel.shape}")
            print(f"kernel sum = {torch.sum(kernel)}")
            Vox.plot_voxelgrid(kernel.detach().cpu().numpy())
        return kernel

class negSpherev2(neg_sphere_kernel):

    def __init__(self, name, kernel_size, **kwargs):
        super().__init__(name, kernel_size, **kwargs)


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

    # def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
    #     return tensor - (torch.sum(tensor) - 1) / self.volume 
    
    
 
    def compute_kernel(self, plot=False):

        idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[0], dtype=torch.float, device=self.device, requires_grad=True),
                            torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True)
                            )
            ).T.reshape(-1, 3)

        # idxs = list(itertools.product(list(range(self.kernel_size[0])), list(range(self.kernel_size[1])), list(range(self.kernel_size[2]))))
        # idxs = torch.from_numpy(np.array([*idxs], dtype=np.float)).to(self.device) # (z*x*y, 3) 
        # idxs.requires_grad_()

        # kernel = -1*self.gaussian(idxs)
        # kernel = self.sum_zero(kernel) - self.neg_factor
        kernel = self.gaussian(idxs)
        kernel = self.sum_zero(kernel)
        kernel = (-self.neg_factor)*torch.relu(kernel)
        kernel = torch.t(kernel).view(self.kernel_size)            

        if plot:
            print(f"kernel shape = {kernel.shape}")
            print(f"kernel sum = {torch.sum(kernel)}")
            Vox.plot_voxelgrid(kernel.detach().cpu().numpy())
        return kernel

    


def sphere3D():
    from matplotlib import cm
    from matplotlib.colors import Normalize
    N = 200
    stride=1
    fig = plt.figure(figsize=(16, 20))
    ax = fig.gca(projection='3d')
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    mask = (x >= 0) & (y <= 0) & (z >= 0)
    x[mask] = 0
    y[mask] = 0
    z[mask] = 0
    
    rad = np.sqrt(x**2 + y**2 + z**2)
    print(np.min(rad))
    rad = rad / np.max(rad)
    print(rad.shape)
    colors = rad >= 0.5
    cmap = cm.get_cmap("coolwarm")

    ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride, facecolors=cmap(colors))
    #ax.set_axis_off()
    plt.show()
# %%
if __name__ == "__main__":

    ROOT_PROJECT = "/home/didi/VSCode/lidar_thesis"

    DATA_SAMPLE_DIR = ROOT_PROJECT + "/Data_sample"
    SAVE_DIR = ROOT_PROJECT + "/dataset/torch_dataset"
    print(DATA_SAMPLE_DIR)

    #build_data_samples([DATA_SAMPLE_DIR], SAVE_DIR)
    ts40k = torch_TS40K(dataset_path=SAVE_DIR, transform=ToTensor())

    vox, vox_gt = ts40k[2]
    vox, vox_gt = vox.to(torch.float), vox_gt.to(torch.float)
    print(vox.shape)
    # Vox.plot_voxelgrid(vox.numpy()[0])
    # Vox.plot_voxelgrid(vox_gt.numpy()[0])


    
    # %%

    sphere = negSpherev2('cy', (9, 9, 9), radius=torch.tensor(3), 
                                          sigma=torch.tensor(2), 
                                          neg_factor=torch.tensor(1))
    kernel = sphere.compute_kernel(True)

    #cy.visualize_kernel()
    # %%
    sphere.convolution(vox.view((1, *vox.shape)).to(sphere.device))

    # %%
    type(sphere.kernel)

# %%
