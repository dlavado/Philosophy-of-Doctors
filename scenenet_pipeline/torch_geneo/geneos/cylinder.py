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

import IPython.display as disp
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


class cylinder_kernel(GENEO_kernel_torch):

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cylinder in the kernel.")

        self.radius = kwargs['radius'].to(self.device)

        if plot:
            print("--- Cylinder Kernel ---")
            print(f"radius = {self.radius:.4f}; {type(self.radius)};")

        self.sigma = 1
        if kwargs.get('sigma') is not None:
            self.sigma = kwargs['sigma']
            if plot:
                print(f"sigma = {self.sigma:.4f}; {type(self.sigma)};")           

        self.plot = plot
        
        super().__init__(name, kernel_size)  

    # def compute_kernel_(self, plot=False) -> torch.Tensor:

    #     center = np.array([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2])
    #     bounds = np.array([[0, self.kernel_size[1] - 1], [0, self.kernel_size[2] -1]])

    #     x1, x2 = smp.symbols('x1, x2')
    #     C = smpv.CoordSys3D('C')
    #     x = x1*C.i + x2*C.j
    #     disp.display(x)
    #     x = smp.Matrix([x1, x2]) # symbolic point in the floor kernel

    #     # circumference definition
    #     x_c = x - center.reshape(-1, 1)
    #     #disp.display(x_c)
    #     # disp.display(x_c.norm())
    #     x_c_norm = x_c.norm()**2
    #     #disp.display(x_c_norm)
    #     circ_x = x_c_norm - self.radius**2
    #     x_plus = x_c_norm - (self.radius + self.epsilon)**2
    #     x_minus = x_c_norm - (self.radius - self.epsilon)**2

    #     #disp.display(circ_x)
    #     gauss = smp.exp((circ_x**2)*(-1 / (2*self.sigma**2)))
    #     gauss_plus = smp.exp((x_plus**2)*(-1 / (2*self.sigma**2))) 
    #     gauss_minus = smp.exp((x_minus**2)*(-1 / (2*self.sigma**2)))

    #     disp.display(gauss)
    #     # disp.display(gauss_plus)
    #     # disp.display(gauss_minus)
    #     # disp.display(x_c_norm.subs([(x1, 2.5), (x2, 2.5)]).evalf())
    #     gauss_num = smp.lambdify([x1, x2], gauss)
    #     gauss_plus_num = smp.lambdify([x1, x2], gauss_plus)
    #     gauss_minus_num = smp.lambdify([x1, x2], gauss_minus)


    #     plot_R2func(gauss_num, bounds[0], bounds[1])
    #     # plot_R2func(gauss_plus_num, bounds[0], bounds[1])
    #     # plot_R2func(gauss_minus_num, bounds[0], bounds[1])
        
    #     K, _ = intg.dblquad(gauss_num, bounds[0][0], bounds[0][1], lambda x: bounds[1][0], lambda x: bounds[1][1])
    #     print(f"Gaussian integral value = {K}") 
    #     factor = K/(self.kernel_size[1]*self.kernel_size[2])
    #     gauss_zero = smp.exp((circ_x**2)*(-1 / (2*self.sigma**2))) - factor
    #     gauss_zero_num = smp.lambdify([x1, x2], gauss_zero)
    #     # K, _ = intg.dblquad(gauss_zero_num, bounds[0][0], bounds[0][1], lambda x: bounds[1][0], lambda x: bounds[1][1])
    #     # print(K)
    #     plot_R2func(gauss_zero_num, bounds[0], bounds[1])

    #     gauss_vec = np.vectorize(gauss_num)

    #     #floor_idxs holds all the idx combinations of the 'floor' of the kernel 
    #     floor_idxs = list(itertools.product(list(range(self.kernel_size[1])), list(range(self.kernel_size[2]))))
    #     floor_idxs = np.array([*floor_idxs]) # (x*y, 2) 

    #     floor = np.array(gauss_vec(floor_idxs[:, 0], floor_idxs[:, 1])).reshape(self.kernel_size[1:])
    #     print(f"floor sum = {np.sum(floor)}")
    #     while np.sum(floor) > 0.05:
    #         floor = floor - np.sum(floor)/(self.kernel_size[1]*self.kernel_size[2])
    #         print(f"floor sum = {np.sum(floor)}")
    #     kernel = np.full(self.kernel_size, floor)
    #     print(f"kernel sum = {np.sum(kernel)}")
        
    #     assert np.isclose(np.sum(kernel), 0), np.sum(kernel)

    #     # gauss_plus_vec = np.vectorize(gauss_plus_num)
    #     # gauss_minus_vec = np.vectorize(gauss_minus_num)
    #     # floor_plus = np.array(gauss_plus_vec(floor_idxs[:, 0], floor_idxs[:, 1])).reshape(self.kernel_size[1:])
    #     # floor_minus = np.array(gauss_minus_vec(floor_idxs[:, 0], floor_idxs[:, 1])).reshape(self.kernel_size[1:])
        
    #     #floor = np.maximum(floor_plus, floor_minus)
    #     # print(f"floor sum = {np.sum(floor)}")
    #     # plt.imshow(floor, cmap=cm.coolwarm)
    #     #kernel = np.full(self.kernel_size, floor)
        
    #     if plot:
    #         Vox.plot_voxelgrid(kernel)

    #     return torch.from_numpy(kernel.astype(np.float))


    # def dep(self, plot=False)-> torch.Tensor:

    #     center = torch.tensor([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2], dtype=torch.float, requires_grad=True)

    #     floor_idxs = list(itertools.product(list(range(self.kernel_size[1])), list(range(self.kernel_size[2]))))
    #     floor_idxs = torch.from_numpy(np.array([*floor_idxs], dtype=np.float)) # (x*y, 2) 
    #     floor_idxs.requires_grad_()
    #     assert floor_idxs.requires_grad


    #     x1, x2 = smp.symbols('x1, x2')
    #     x = smp.Matrix([x1, x2]) # symbolic point in the floor kernel

    #     # circumference definition
    #     x_c = x - center.reshape(-1, 1)
    #     x_c_norm = x_c.norm()**2
    #     circ_x = x_c_norm - self.radius**2
    #     x_plus = x_c_norm - (self.radius + self.epsilon)**2
    #     x_minus = x_c_norm - (self.radius - self.epsilon)**2

    #     # gaussian functions
    #     gauss = smp.exp((circ_x**2)*(-1 / (2*self.sigma**2)))
    #     gauss_minus = smp.exp((x_minus**2)*(-1 / (2*self.sigma**2))) 
    #     gauss_plus = smp.exp((x_plus**2)*(-1 / (2*self.sigma**2)))

    #     if plot:
    #         disp.display(gauss)

    #     expr = [gauss, gauss_minus, gauss_plus]
    #     expr = [gauss]

    #     g = spt.SymPyModule(expressions=expr)

    #     floor_vals = g(x1=floor_idxs[:, 0], x2 = floor_idxs[:, 1])
    #     assert floor_vals.requires_grad

    #     if plot:
    #         print(f"floor values = {floor_vals.shape}; {type(floor_vals)}")
        

    #     floor_vals = torch.t(floor_vals).view(len(expr), self.kernel_size[1], self.kernel_size[2])
    #     floor_vals = torch.max(floor_vals, dim=0)[0] # values
        
    #     kernel = torch.tile(floor_vals, (self.kernel_size[0], 1, 1))
    #     assert kernel.shape == self.kernel_size
    #     assert kernel.requires_grad
    #     assert torch.equal(kernel[0], floor_vals[0])
       

    #     if plot:
    #         print(f"floor vals shape = {floor_vals.shape}")
    #         print(f"kernel shape = {kernel.shape}")
    #         print(f"kernel sum = {torch.sum(kernel)}")
    #         Vox.plot_voxelgrid(kernel.detach().numpy())
    #     return kernel



    def gaussian(self, x:torch.Tensor, epsilon=0) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2], dtype=torch.float, device=self.device, requires_grad=True)

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        circle_x = x_c_norm**2 - (self.radius + epsilon)**2 

        return torch.exp((circle_x**2) * (-1 / (2*self.sigma**2)))

    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - torch.sum(tensor) / torch.prod(torch.tensor(self.kernel_size[1:])) 
 
    def compute_kernel(self, plot=False):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2)

        # floor_idxs = list(itertools.product(list(range(self.kernel_size[1])), list(range(self.kernel_size[2]))))
        # floor_idxs = torch.from_numpy(np.array([*floor_idxs], dtype=np.float)).to(self.device) # (x*y, 2) 
        # floor_idxs.requires_grad_()
        # assert floor_idxs.requires_grad

        floor_vals = self.gaussian(floor_idxs)
        
        floor_vals = self.sum_zero(floor_vals)
        floor_vals = torch.t(floor_vals).view(self.kernel_size[1:])            
        #assert floor_vals.requires_grad
    
        kernel = torch.tile(floor_vals, (self.kernel_size[0], 1, 1))
        # assert kernel.shape == self.kernel_size
        # assert kernel.requires_grad
        # assert torch.equal(kernel[0], floor_vals)
        # assert torch.sum(kernel) <= 1e-10 or torch.sum(kernel) <= -1e-10 # weight sum == 0

        if plot:
            print(f"floor values = {floor_vals.shape}; {type(floor_vals)}")
            print(f"kernel shape = {kernel.shape}")
            print(f"kernel sum = {torch.sum(kernel)}")
            Vox.plot_voxelgrid(kernel.cpu().detach().numpy())
        return kernel

       
    def mandatory_parameters():
        return ['radius']
    
    def geneo_parameters():
        return cylinder_kernel.mandatory_parameters() + ['sigma']

    def geneo_random_config(name='GENEO_rand'):
        rand_config = GENEO_kernel_torch.geneo_random_config()

        geneo_params = {
            'radius' : torch.randint(1, rand_config['kernel_size'][1], (1,))[0] / 2 ,
            'sigma' : torch.randint(5, 10, (1,))[0] / 5 #float \in [1, 2]
        }   

        rand_config['geneo_params'] = geneo_params
        rand_config['name'] = 'cylinder'

        return rand_config

    def geneo_smart_config(name="Smart_Cylinder"):

        config = {
            'name' : name,
            'kernel_size': (9, 6, 6),
            'plot': False,
            'non_trainable' : [],

            'geneo_params' : {
                                'radius' :  torch.tensor(1.0) ,
                                'sigma' :  torch.tensor(2.0)
                             }
        }


        return config





class cylinderv2(cylinder_kernel):

    def __init__(self, name, kernel_size, plot=False, **kwargs):
        super().__init__(name, kernel_size, plot, **kwargs)


    def gaussian(self, x:torch.Tensor, epsilon=1e-8) -> torch.Tensor:
        center = torch.tensor([(self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2], dtype=torch.float, device=self.device, requires_grad=True)

        x_c = x - center # Nx2
        x_c_norm = torch.linalg.norm(x_c, dim=1, keepdim=True) # Nx1
        gauss_dist = x_c_norm**2 #- (self.radius + epsilon)**2 

        return self.sigma*torch.exp((gauss_dist**2) * (-1 / (2*(self.radius + epsilon)**2)))


    def compute_kernel(self, plot=False):

        floor_idxs = torch.stack(
                torch.meshgrid(torch.arange(self.kernel_size[1], dtype=torch.float, device=self.device, requires_grad=True), 
                            torch.arange(self.kernel_size[2], dtype=torch.float, device=self.device, requires_grad=True))
            ).T.reshape(-1, 2)

        # floor_idxs = list(itertools.product(list(range(self.kernel_size[1])), list(range(self.kernel_size[2]))))
        # floor_idxs = torch.from_numpy(np.array([*floor_idxs], dtype=np.float)).to(self.device) # (x*y, 2) 
        # floor_idxs.requires_grad_()
        # assert floor_idxs.requires_grad

        floor_vals = self.gaussian(floor_idxs)
        floor_vals = self.sum_zero(floor_vals)
        floor_vals = torch.t(floor_vals).view(self.kernel_size[1:])            
        #assert floor_vals.requires_grad
    
        kernel = torch.tile(floor_vals, (self.kernel_size[0], 1, 1))
        # assert kernel.shape == self.kernel_size
        # assert kernel.requires_grad
        # assert torch.equal(kernel[0], floor_vals)
        # assert torch.sum(kernel) <= 1e-10 or torch.sum(kernel) <= -1e-10 # weight sum == 0

        if plot:
            print(f"floor values = {floor_vals.shape}; {type(floor_vals)}")
            print(f"kernel shape = {kernel.shape}")
            print(f"kernel sum = {torch.sum(kernel)}")
            print(floor_vals)
            Vox.plot_voxelgrid(kernel.cpu().detach().numpy())
        return kernel 



def plot_R2func(func, lim_x1, lim_x2, cmap=cm.coolwarm):
    x1_lin = np.linspace(lim_x1[0], lim_x1[1], 100)
    x2_lin = np.linspace(lim_x2[0], lim_x2[1], 100)
    x1_lin, x2_lin = np.meshgrid(x1_lin, x2_lin)
    g = func(x1_lin, x2_lin)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_lin, x2_lin, g, cmap=cmap)
    plt.show()

# %%
if __name__ == "__main__":

    ROOT_PROJECT = "/home/didi/VSCode/lidar_thesis"

    SAVE_DIR = ROOT_PROJECT + "/dataset/torch_dataset"

    #build_data_samples([DATA_SAMPLE_DIR], SAVE_DIR)
    ts40k = torch_TS40K(dataset_path=SAVE_DIR, transform=ToTensor())

    vox, vox_gt = ts40k[2]
    vox, vox_gt = vox.to(torch.float), vox_gt.to(torch.float)
    print(vox.shape)
    Vox.plot_voxelgrid(vox.numpy()[0])
    Vox.plot_voxelgrid(vox_gt.numpy()[0])
    
    # %%

    cy = cylinder_kernel('cy', (6, 6, 6), radius=torch.tensor(2), sigma=torch.tensor(2))
    cy = cylinderv2('cy', (6, 7, 7), radius=torch.tensor(2.5), sigma=torch.tensor(5))
    #kernel = cy.compute_kernel_(True)

    cy.visualize_kernel()
    # %%
    cy.convolution(vox.view((1, *vox.shape)).to(cy.device),plot=True)

    # %%
    type(cy.kernel)

# %%
