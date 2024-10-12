


from abc import abstractmethod
import torch

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')


GIB_PARAMS = "gib_params"
NON_TRAINABLE = "non_trainable"
KERNEL_REACH = "kernel_reach"



###############################################################
#                          GIB Utils                          #
###############################################################

def to_parameter(value):
    """
    Converts the input value to a torch.nn.Parameter.
    """
    if isinstance(value, torch.Tensor):
        return torch.nn.Parameter(value, requires_grad=True)
    elif isinstance(value, int) or isinstance(value, float):
        return torch.nn.Parameter(torch.tensor(value, dtype=torch.float), requires_grad=True)
    
    raise ValueError("Input value must be a torch.Tensor")
    
def to_tensor(value):
    """
    Converts the input value to a torch.Tensor.
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
 
    raise ValueError("Input value must be a torch.Tensor")


###############################################################
#                           GIB Stub                          #
###############################################################

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
    def forward(self, points:torch.Tensor, q_points:torch.Tensor, supports_idxs:torch.Tensor) -> torch.Tensor:
        """
        Computes a Cylinder GIB on the query points given the support points.

        Parameters
        ----------
        `points` - torch.Tensor:
            Tensor of shape (N, 3) representing the point cloud.

        `q_points` - torch.Tensor:
            Tensor of shape (M, 3) representing the query points.

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
        


   

if __name__ == "__main__":

    from core.neighboring.radius_ball import k_radius_ball
    from core.neighboring.knn import torch_knn
    from core.sampling.FPS import Farthest_Point_Sampling
    from core.models.GENEONets.geneos import cylinder, disk, cone, ellipsoid
    
    # # generate some points, query points, and neighbors. For the neighbors, I want to test two scenarios: 
    # # 1) where the neighbors are at radius distance from the query points
    # # 2) where the neighbors are very distance fromt the query points, at least 2*radius distance
    # points = torch.rand((100_000, 3), device='cuda')
    # query_idxs = farthest_point_pooling(points, 20)
    # q_points = points[query_idxs]
    # num_neighbors = 20
    # # neighbors = k_radius_ball(q_points, points, 0.2, 10, loop=True)
    # _, neighbors_idxs = torch_knn(q_points, points, num_neighbors)

    # # print(points.device, q_points.device, neighbors_idxs.device)

    # gib_setup = {
    #     'cy': 2,
    #     'cone': 2,
    #     'disk': 2,
    #     'ellip': 2
    # }

    # gib_layer = GIB_Layer(gib_setup, kernel_reach=16, num_observers=2)

    # gib_weights = gib_layer(points, query_idxs, neighbors_idxs)

    # print(gib_layer.get_cvx_coefficients())
    # print(gib_weights.shape)


    # input("Press Enter to continue...")


    from core.datasets.TS40K import TS40K_FULL_Preprocessed
    from utils import constants as C
    import utils.pointcloud_processing as eda

    ts40k = TS40K_FULL_Preprocessed(
        C.TS40K_FULL_PREPROCESSED_PATH,
        split='fit',
        sample_types=['tower_radius'],
        transform=None,
        load_into_memory=False
    )

    pcd, y = ts40k[0]
    pcd = pcd.to('cuda')
    num_query_points = 10000
    num_neighbors = 10
    kernel_reach = 0.1

    # max dist between points:
    # c_dist = torch.cdist(pcd, pcd)
    # print(torch.max(c_dist), torch.min(c_dist), torch.mean(c_dist)) # tensor(1.5170, device='cuda:0') tensor(0., device='cuda:0') tensor(0.4848, device='cuda:0')
    fps = Farthest_Point_Sampling(num_points=num_query_points)
    query_idxs = fps(pcd, num_query_points)
    # support_idxs = torch_knn(pcd[query_idxs], pcd[query_idxs], num_neighbors)[1]
    support_idxs = k_radius_ball(pcd[query_idxs], pcd, kernel_reach, num_neighbors, loop=False)

    gib_setup = {
        'cy': 1,
        # 'cone': 1,
        # 'disk': 1,
        # 'ellip': 1
    }

    # gib_layer = GIB_Layer(gib_setup, kernel_reach=0.1, num_observers=1)

    # effectively retrieve the body of towers
    # gib_layer = cylinder.Cylinder(kernel_reach=kernel_reach, radius=0.02)

    # disk is a generalization of the cylinder with the width parameter;
    # small width -> can be used to detect ground / surface
    # large width -> can be used to detect towers
    # gib_layer = disk.Disk(kernel_reach=kernel_reach, radius=0.02, width=0.1)

    # cone can be used to detect arboreal structures such as medium vegetation with small apex
    # gib_layer = cone.Cone(kernel_reach=kernel_reach, radius=0.05, inc=torch.tensor([0.1], device='cuda'), apex=1, intensity=1.0)

    # Ideally and ellipsoid can be used to detect the ground and power lines by assuming different radii
    gib_layer = ellipsoid.Ellipsoid(kernel_reach=kernel_reach, radii=torch.tensor([0.1, 0.1, 0.001], device='cuda'))
    

    gib_weights = gib_layer(pcd, query_idxs, support_idxs)


    # eda.plot_pointcloud(
    #     pcd.cpu().detach().numpy(),
    #     y.cpu().detach().numpy(),
    #     use_preset_colors=True
    # )

    # print(gib_weights.shape)

    colors = eda.weights_to_colors(gib_weights.cpu().detach().numpy(), cmap='seismic')
    eda.plot_pointcloud(
        pcd[query_idxs].cpu().detach().numpy(), 
        classes=None,
        rgb=colors
    )