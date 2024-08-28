


from abc import abstractmethod
import time
import torch
import torch.nn.functional as F
import json
import numpy as np

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
# from utils import voxelization as Vox
import core.models.GENEONets.geneos.diff_rotation_transform as drt



class GENEO_kernel:
    """
    Initialization class for GENEO kernels.

    Kernels are built on top of the convolution operation as 3D arrays.

    * kernel shape in (z, x, y)
    """

    def __init__(self, name, kernel_size, plot=False, angles=None, **kwargs):
        self.name = name
        self.kernel_size = kernel_size
        self.plot = plot
        self.angles = angles
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.sign = 1 if torch.any(torch.rand(1) > 0.5) else -1 # random sign for the kernel
        self.sign = 1
        self.volume = torch.prod(torch.tensor(self.kernel_size, device=self.device))


    @abstractmethod
    def compute_kernel(self) -> torch.Tensor:
        """
        Returns a 3D GENEO kernel in torch format
        """
        return

    def convolution(self, tensor:torch.Tensor, plot=True) -> torch.Tensor:
        """
        Convolves the kernel with the tensor data passed as argument.

        Attention: tensor should be in format [B, 1, data], where B is the batch number.

        plot ? Visualize convolution output
        """
        start = time.time()
        conv = F.conv3d(tensor, self.kernel.view(1, 1, *self.kernel.shape), padding='same')
        end = time.time() - start

        if plot:
            print(f"Elapsed time for convolution: {end}")
            # for i in range(tensor.shape[0]): # for each sample in the batch
            #     Vox.plot_voxelgrid(conv[i][0].detach().cpu().numpy())

        return conv


    def plot_kernel(self, kernel):
        print(f"\n{'*'*50}")
        print(f"kernel shape = {kernel.shape}")
        print(f"kernel sum = {torch.sum(kernel)}")
        # Vox.plot_voxelgrid(kernel.cpu().detach().numpy(), color_mode='density')

    @staticmethod
    def mandatory_parameters():
        return []

    @staticmethod
    def geneo_parameters():
        return []

    @staticmethod
    def geneo_config_from_json(filename):
        """
        Returns a GENEO configuration based on a json file
        """
        with open(filename) as json_file:
            return json.load(json_file)

    @staticmethod
    def geneo_smart_config():
        """
        Returns a tailored GENEO configuration
        """
        return

    @staticmethod
    def geneo_random_config(name='GENEO_rand', kernel_size=(9, 9, 9)):
        """
        Returns a random GENEO configuration
        """
        config = {
            'name' : name,
            'kernel_size': kernel_size,
            'plot': False,
        }

        geneo_params = {}

        for param in GENEO_kernel.geneo_parameters():
            geneo_params[param] = torch.randint(0, 10, (1,))[0]/5 # float \in [0, 2]

        config['geneo_params'] = geneo_params

        config['non_trainable'] = []

        return config
    
    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - torch.sum(tensor) / torch.prod(torch.tensor(self.kernel_size[1:])) 
    
    def rotate_tensor(self, angles:torch.Tensor, data):
        """
        Rotate a tensor along the x, y, and z axes by the given angles.
        
        Parameters
        ----------
        angles - torch.Tensor: 
            Tensor of shape (3,) containing rotation angles for the z, x, and y axes.
            These are in the range [0, 2] and represent a fraction of pi.
        data - torch.Tensor: 
            Input 3D tensor to be rotated.
            
        Returns
        -------
        rotated_data - torch.Tensor: 
            Rotated tensor
        """

        angles = angles * 180 # convert to degrees
        angle_z, angle_x, angle_y = angles
        interpolation = 'bilinear'
        if angle_z != 0:
            data = drt.rotation_3d(data, 0, angle_z, expand=False, interpolation=interpolation)
        if angle_x != 0:
            data = drt.rotation_3d(data, 1, angle_x, expand=False, interpolation=interpolation)
        if angle_y != 0:
            data = drt.rotation_3d(data, 2, angle_y, expand=False, interpolation=interpolation)
    
        return data
    

    def plot_geneo_kernel(self, kernel):
        """
        Plot the kernel in a 3D voxel grid.
        """
        print(f"\n{'*'*50}")
        print(f"kernel shape = {kernel.shape}")
        print(f"kernel sum = {torch.sum(kernel)}")


        if isinstance(grid, torch.Tensor):
            grid = grid.numpy()

        z, x, y = grid.nonzero()

        xyz = np.empty((len(z), 4))
        idx = 0
        for i, j, k in zip(x, y, z):
            xyz[idx] = [int(i), int(j), int(k), grid[k, i, j]]
            idx += 1

        if len(xyz) == 0:
            return
        
        uq_classes = np.unique(xyz[:, -1])
        class_colors = np.empty((len(xyz), 3))

        for i, c in enumerate(uq_classes):
        # [-1, 0[ - blue; ~0 white; ]0, 1] - red
            if c < 0:
                class_colors[i] = [1+c, 1+c, 1]
            else:
                class_colors[i] = [1, 1-c, 1-c]

        
        


   