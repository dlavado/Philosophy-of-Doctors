


from abc import abstractmethod
from numpy import number
import torch

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')

import core.models.GENEONets.geneos.diff_rotation_transform as drt



class GIB_Stub:
    """
    Abstract class for Geometric Inductive Bias operators.
    """

    def __init__(self, kernel_reach:int, angles=None, **kwargs):
        """
        Initializes the GIB kernel.

        Parameters
        ----------

        `kernel_reach` - int:
            The kernel's neighborhood reach in Geometric space.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sign = 1 if torch.any(torch.rand(1) > 0.5) else -1 # random sign for the kernel

        self.angles = angles

        self.kernel_reach = kernel_reach
        
        # variables to compute the integral of the GIB function within the kernel reach
        self.n_samples = 1e4
        self.ndims = 3
        self.montecarlo_points = torch.rand((int(self.n_samples), self.ndims), device=self.device) * 2 * self.kernel_reach - self.kernel_reach
        mask_inside = torch.linalg.norm(self.montecarlo_points, dim=1) <= self.kernel_reach
        self.montecarlo_points = self.montecarlo_points[mask_inside]

        self.epsilon = 1e-8 # small value to avoid division by zero
        self.intensity = 1 # intensity of the gaussian function

    def _to_parameter(self, value):
        """
        Converts the input value to a torch Parameter.
        """
        if isinstance(value, torch.Tensor):
            return torch.nn.Parameter(value)
        elif isinstance(value, int) or isinstance(value, float):
            return torch.nn.Parameter(torch.tensor(value, dtype=torch.float))
        else:
            raise ValueError("Input value must be a torch.Tensor")
        
    def _to_tensor(self, value):
        """
        Converts the input value to a torch Tensor.
        """
        import numpy as np
        if isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, int) or isinstance(value, float):
            return torch.tensor(value, dtype=torch.float)
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value).float()
        elif isinstance(value, list) or isinstance(value, tuple):
            return torch.tensor(value, dtype=torch.float)
        else:
            raise ValueError("Input value must be a torch.Tensor")
        

    @abstractmethod
    def compute_integral(self) -> torch.Tensor:
        """
        Computes an integral approximation of the gaussian function within the kernel_reach.

        Parameters
        ----------
        n_samples - int:
            Number of samples to use for the integral approximation using Monte Carlo.

        Returns
        -------
        `integral` - torch.Tensor:
            Tensor of shape (1,) representing the integral of the gaussian function within the kernel reach.
        """

    @abstractmethod
    def gaussian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function for the given input tensor.
        """
    

    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor by subtracting the integral of the resulting gaussian function within the kernel reach.

        Parameters
        ----------
        `tensor` - torch.Tensor:
            Tensor of shape (N,) representing the values of the kernel; this tensor directly results from the gaussian function.

        Returns
        -------
        `tensor` - torch.Tensor:
            Tensor of shape (N,) representing the normalized values of the kernel.
        """
        integral = self.compute_integral().to(tensor.device)
        return tensor - integral / self.n_samples


    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Returns a 3D GENEO kernel in torch format
        """


    @staticmethod
    def mandatory_parameters():
        return []

    @staticmethod
    def geneo_parameters():
        return []

    @staticmethod
    def geneo_random_config(kernel_reach:int):
        """
        Returns a random GENEO configuration
        """
        config = {
            'kernel_reach': kernel_reach   
        }
        geneo_params = {}

        for param in GIB_Stub.geneo_parameters():
            geneo_params[param] = torch.randint(0, 10, (1,))[0]/5 # float \in [0, 2]

        config['geneo_params'] = geneo_params

        config['non_trainable'] = []

        return config
    
   
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
        


   