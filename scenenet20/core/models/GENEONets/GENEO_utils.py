from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys


sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')

from core.models.GENEONets.geneos.GIB_Stub import GIB_Stub, NON_TRAINABLE, GIB_PARAMS
from core.models.GENEONets.geneos import cylinder, disk, cone, ellipsoid


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
#                          GIB Layer                          #
###############################################################

class GIB_Operator(nn.Module):

    def __init__(self, gib_class:GIB_Stub, kernel_reach:float=None, **kwargs):
        super(GIB_Operator, self).__init__()  

        self.gib_class = gib_class
        self.kernel_reach = kernel_reach
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if len(kwargs) > 0:
            self.init_from_kwargs(kwargs)
        else:
            self.random_init()

        self.gib = self.gib_class(kernel_reach=self.kernel_reach, **self.gib_params)

    
    def random_init(self):
        config = self.gib_class.gib_random_config(self.kernel_reach)
        self.gib_params = {}

        for param_name in config[GIB_PARAMS]:
            t_param = to_tensor(config[GIB_PARAMS][param_name])
            self.gib_params[param_name] = nn.Parameter(t_param, requires_grad = not param_name in config[NON_TRAINABLE])

        self.gib_params = nn.ParameterDict(self.gib_params)


    def init_from_kwargs(self, kwargs):
        self.gib_params = {}
        for param_name in self.gib_class.mandatory_parameters():
            self.gib_params[param_name] = to_parameter(kwargs[param_name])

        self.gib_params = nn.ParameterDict(self.gib_params)


    def forward(self, points:torch.Tensor, query_idxs:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the GIB on the query points given the support points.

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
        return self.gib(points, query_idxs, support_idxs)


###############################################################
#                           GIB-Net                           #
###############################################################

class GIB_Layer(nn.Module):

    def __init__(self, gib_dict:dict, kernel_reach:int, num_observers:int=1):
        """
        Instantiates a GIB-Layer Module with GIBs and their cvx coefficients.

        Parameters
        ----------
        `gib_dict` - dict[str, int]:
            Mappings that contain the number of GIBs of each kind (the key) to initialize;
            keys \in ['cy', 'cone', 'disk', 'ellip']

        `kernel_reach` - int:
            The kernel's neighborhood reach in Geometric space.

        `num_observers` - int:
            number os GENEO observers to form the output of the Module
        
        """
        super(GIB_Layer, self).__init__()
        
        if gib_dict is None or gib_dict == {}:
            geneo_keys = ['cy', 'cone', 'disk', 'ellip']
            self.gib_dict = {
                g : torch.randint(1, 64, (1,))[0] for g in geneo_keys
            }
        else:
            self.gib_dict = gib_dict

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_observers = num_observers

        self.gibs:Mapping[str, GIB_Operator] = nn.ModuleDict()

        # --- Initializing GIBs ---
        for key in self.gib_dict:
            if key == 'cy':
                g_class = cylinder.Cylinder
            elif key == 'disk':
                g_class = disk.Disk
            elif key == 'cone':
                g_class = cone.Cone
            elif g_class == 'ellip':
                g_class = ellipsoid.Ellipsoid

            for i in range(self.gib_dict[key]):
                self.gibs[f'{key}_{i}'] = GIB_Operator(g_class, kernel_reach=kernel_reach).to(self.device)

        # --- Initializing Convex Coefficients ---
        self.lambdas = torch.randn((len(self.gibs), num_observers), device=self.device) # shape (num_gibs, num_observers)
        self.maintain_convexity() # make sure the coefficients are convex
        self.lambdas = to_parameter(self.lambdas)


    def maintain_convexity(self):
        self.lambdas = torch.softmax(self.lambdas, dim=0)

    def get_cvx_coefficients(self) -> torch.Tensor:
        return self.lambdas

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    def _compute_gib_outputs(self, points:torch.Tensor, query_idxs:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:

        q_outputs = torch.zeros((len(query_idxs), len(self.gibs)), dtype=points.dtype, device=points.device)
        for i, gib_key in enumerate(self.gibs):
            q_outputs[:, i] = self.gibs[gib_key](points, query_idxs, support_idxs)

        return q_outputs
    
    def _compute_observers(self, q_outputs:torch.Tensor) -> torch.Tensor:
        # --- Convex Combination ---
        # for each query point, compute the convex combination of the outputs of the GIBs
        return q_outputs @ self.lambdas # shape (M, num_gibs) @ (num_gibs, num_observers) = (M, num_observers)
    

    
    def forward(self, points:torch.Tensor, query_idxs:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the GIB-Layer on the query points given the support points.

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
        `q_outputs` - torch.Tensor:
            Tensor of shape (M, num_observers) representing the output of the GIB-Layer on the query points.
        """
        q_outputs = self._compute_gib_outputs(points, query_idxs, support_idxs) # shape (M, num_gibs)
        q_outputs = self._compute_observers(q_outputs) # shape (M, num_observers)
        return q_outputs
    


if __name__ == "__main__":
    from core.neighboring.radius_ball import k_radius_ball
    from core.neighboring.knn import torch_knn
    from core.pooling.farthest_point import farthest_point_pooling
    
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

    query_idxs = farthest_point_pooling(pcd, num_query_points)
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