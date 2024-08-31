


from abc import abstractmethod
import torch

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')


GIB_PARAMS = "gib_params"
NON_TRAINABLE = "non_trainable"
KERNEL_REACH = "kernel_reach"


class GIB_Stub(torch.nn.Module):
    """
    Abstract class for Geometric Inductive Bias operators.
    """

    def __init__(self, kernel_reach:float, angles=None, **kwargs):
        """
        Initializes the GIB kernel.

        Parameters
        ----------

        `kernel_reach` - int:
            The kernel's neighborhood reach in Geometric space.
        """
        super(GIB_Stub, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sign = 1 if torch.any(torch.rand(1) > 0.5) else -1 # random sign for the kernel

        self.angles:torch.Tensor = angles
        self.kernel_reach = kernel_reach
        
        # variables to compute the integral of the GIB function within the kernel reach
        self.n_samples = 1e4
        self.ndims = 3
        self.montecarlo_points = torch.rand((int(self.n_samples), self.ndims), device=self.device) * 2 * self.kernel_reach - self.kernel_reach
        mask_inside = torch.linalg.norm(self.montecarlo_points, dim=1) <= self.kernel_reach
        self.montecarlo_points = self.montecarlo_points[mask_inside]

        self.epsilon = 1e-8 # small value to avoid division by zero
        self.intensity = 1 # intensity of the gaussian function
        

    @abstractmethod
    def compute_integral(self) -> torch.Tensor:
        """
        Computes an integral approximation of the gaussian function within the kernel_reach.
       
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
    def forward(self, points:torch.Tensor, query_idxs:torch.Tensor, supports_idxs:torch.Tensor) -> torch.Tensor:
        """
        Computes a Cylinder GIB on the query points given the support points.

        Parameters
        ----------
        `points` - torch.Tensor:
            Tensor of shape (N, 3) representing the point cloud.

        `query_idxs` - torch.Tensor[int]:
            Tensor of shape (M,) representing the indices of the query points in `points`. With M <= N.

        `supports_idxs` - torch.Tensor[int]:
            Tensor of shape (M, K) representing the indices of the support points for each query point. With K <= N.

        Returns
        -------
        `q_output` - torch.Tensor:
            Tensor of shape (M,) representing the output of the GIB on the query points.    
        """


    @staticmethod
    def mandatory_parameters():
        return []

    @staticmethod
    def gib_parameters():
        return []

    @staticmethod
    def gib_random_config(kernel_reach:int):
        """
        Returns a random GENEO configuration
        """
        config = {
                KERNEL_REACH: kernel_reach   
        }
        gib_params = {
            'intensity' : torch.randint(5, 10, (1,))[0]/5, # float \in [0, 1]
            'angles'    : torch.zeros(3)
        }

        for param in GIB_Stub.gib_parameters():
            gib_params[param] = torch.randint(0, 10, (1,))[0]/5 # float \in [0, 2]

        config[GIB_PARAMS] = gib_params
        config[NON_TRAINABLE] = []

        return config
    
    
   
    def rotate(self, points:torch.Tensor) -> torch.Tensor:
        """
        Rotate a tensor along the x, y, and z axes by the given angles.
        
        Parameters
        ----------
        `angles` - torch.Tensor:
            Tensor of shape (3,) containing rotation angles for the x, y, and z axes.
            These are nromalized in the range [-1, 1] and represent angles_normalized = angles / pi.

        points - torch.Tensor:
            Tensor of shape (N, 3) representing the 3D points to rotate.
            
        Returns
        -------
        points - torch.Tensor:
            Tensor of shape (N, 3) containing the rotated
        """
        if self.angles is None:
            return points
        
        from core.models.GENEONets.geneos.diff_rotation_transform import rotate_points
        angles = torch.tanh(self.angles) # convert to range [-1, 1]
        return rotate_points(angles, points)
        


   