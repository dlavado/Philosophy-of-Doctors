from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys


sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')


from core.neighboring.neighbors import Neighboring_Method
from core.sampling.query_points_strat import Query_Points
from core.pooling.grid_poooling import grid_pooling_batch, _cluster_to_spoints



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


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]

    where B is the batch size, N is the number of points, C is the number of channels, and L is the number of neighbors
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError("Input dimension not supported")
        

class Neighboring(nn.Module):

    def __init__(self, 
                 neighborhood_strategy:str, 
                 num_neighbors:int, 
                 **kwargs
                ) -> None:
        
        super(Neighboring, self).__init__()

        self.neighbor = Neighboring_Method(neighborhood_strategy, num_neighbors, **kwargs)

    def forward(self, x, q_points) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (B, N, 3)

        q_points : torch.Tensor
            query points of shape (B, Q, 3)

        Returns
        -------
        s_points_idxs : torch.Tensor
            support points of shape (B, Q, k)
        """
        return self.neighbor(x, q_points)






###############################################################
# Build Graph Pyramid for GIBLi
###############################################################

@torch.no_grad()
def build_fps_graph_pyramid(
    points: torch.Tensor,
    num_layers: int,
    sampling_ratio: float,
    num_neighbors: List[int],
    neighborhood_strategy: str,
    neighborhood_kwargs: Dict,
    neighborhood_kwarg_update: Dict,
    )-> Dict:

    """
    Build a Graph Pyramid using Farthest Point Sampling (FPS)


    Parameters
    ----------
    `points` : torch.Tensor
        input tensor of shape (B, N, 3)

    `num_layers` : int
        number of layers in the graph pyramid, i.e., in the GIBLi model

    `sampling_ratio` : float \in (0, 1)
        the ratio of points to sample at each layer

    `num_neighbors` : List[int]
        the number of neighbors to consider at each layer;
        thus, len(num_neighbors) = num_layers - 1

    `neighborhood_strategy` : str \in {'knn', 'dbscan', 'radius_ball'}
        the strategy to use for selecting neighbors

    `neighborhood_kwargs` : Dict
        additional hyperparameters for the selected strategy;
        if the strategy is 'knn', then there is no need for additional hyperparameters;
        if the strategy is 'dbscan', then the hyperparameters are `eps` and `min_points`;
        if the strategy is 'radius_ball', then the hyperparameter is `radius`;

        for example, if the strategy is 'radius_ball', then neighborhood_kwargs = {'radius': 0.1}

    `neighborhood_kwarg_update` : Dict
        update the hyperparameters for the selected strategy in each layer;
        for example, if the strategy is 'radius_ball', then neighborhood_kwarg_update = {'radius': 2.0}, so that the search radius is doubled in each layer;

    Returns
    -------

    `graph_pyramid` : Dict
        a dictionary containing the following
        - `points_list` : List[torch.Tensor]
            a list of tensors containing the points at each layer, each with shape (B, Q, 3), where Q is updated according to the sampling ratio
        - `neighbors_idxs_list` : List[torch.Tensor]
            a list of tensors containing the indices of the neighbors at each layer, each with shape (B, Q, k), where k is the number of neighbors at each layer
        - `subsampling_idxs_list` : List[torch.Tensor]
            a list of tensors containing the indices of the subsampled points at each layer, each with shape (B, Q, k), where k is the number of neighbors at each layer
        - `upsampling_idxs_list` : List[torch.Tensor]
            a list of tensors containing the indices of the upsampled points at each layer, each with shape (B, Q, k), where k is the number of neighbors at each layer
    """

    points_list = []
    neighbors_idxs_list = []
    subsampling_idxs_list = []
    upsampling_idxs_list = []

    points_list.append(points[..., :3])
    num_q_points = int(points.shape[0] * sampling_ratio)
    for _ in range(num_layers):
        points = Query_Points('fps', num_q_points=num_q_points)(points)
        points_list.append(points) # (B, Q, 3)
        num_q_points = int(num_q_points * sampling_ratio)


    for i in range(num_layers):
        curr_points = points_list[i]

        neighborhood_finder = Neighboring(neighborhood_strategy, num_neighbors[i], **neighborhood_kwargs)
        upsample_kwargs = {}
        for key, value in neighborhood_kwargs.items():
            if key in neighborhood_kwarg_update:
                upsample_kwargs[key] = value * neighborhood_kwarg_update[key]
        
        upsample_neighborhood_finder = Neighboring(neighborhood_strategy, num_neighbors[i+1], **upsample_kwargs)

        s_points_idxs = neighborhood_finder(curr_points, curr_points) # (B, Q[i], k[i]); where Q is the num of q_points and k is the num of neighbors at ith layer
        neighbors_idxs_list.append(s_points_idxs)


        if i < num_layers - 1:
            next_points = points_list[i+1] # (B, Q[i+1], 3)

            sub_points_idxs = neighborhood_finder(curr_points, next_points) # (B, Q[i+1], k[i])
            subsampling_idxs_list.append(sub_points_idxs)


            up_points_idxs = upsample_neighborhood_finder(next_points, curr_points) # (B, Q[i], k[i+1])
            upsampling_idxs_list.append(up_points_idxs)

        
        neighborhood_kwargs = upsample_kwargs


    return {
        'points_list': points_list,
        'neighbors_idxs_list': neighbors_idxs_list,
        'subsampling_idxs_list': subsampling_idxs_list,
        'upsampling_idxs_list': upsampling_idxs_list
    }
            




def build_full_grid_graph_pyramid(
    points: torch.Tensor,
    num_layers: int,
    voxel_size: float,
    )-> Dict:
    """
    Build a Graph Pyramid using Grid Sampling

    Parameters
    ----------
    `points` : torch.Tensor
        input tensor of shape (B, N, 3)

    `num_layers` : int
        number of layers in the graph pyramid, i.e., in the GIBLi model

    `num_neighbors` : List[int]
        the number of neighbors to consider at each layer;
        thus, len(num_neighbors) = num_layers - 1

    Returns
    -------
    `graph_pyramid` : Dict
        a dictionary containing the following
        - `points_list` : List[torch.Tensor]
            a list of tensors containing the points at each layer, each with shape (B, Q, 3), where Q is updated according to the sampling ratio
        - `neighbors_idxs_list` : List[torch.Tensor]
            a list of tensors containing the indices of the neighbors at each layer, each with shape (B, Q, k), where k is the number of neighbors at each layer
        - `subsampling_idxs_list` : List[torch.Tensor]
            a list of tensors containing the indices of the subsampled points at each layer, each with shape (B, Q, k), where k is the number of neighbors
        - `upsampling_idxs_list` : List[torch.Tensor]
            a list of tensors containing the indices of the upsampled points at each layer, each with shape (B, Q, k), where k is the number of neighbors
    """


    points_list:List[torch.Tensor] = []
    neighbors_idxs_list:List[torch.Tensor] = []
    subsampling_idxs_list:List[torch.Tensor] = []
    upsampling_idxs_list:List[torch.Tensor] = []

    points = points[..., :3]
    og_voxel_size = voxel_size

    points_list.append(points)

    for _ in range(num_layers):
        points, clusters = grid_pooling_batch(points, voxel_size, None) # clusters is in format (B, N), where N is the number of points;
        points_list.append(points)

        # convert clusters shape to (B, Q, C), where Q is the number of centroids and C is the max number of points in each centroid; pad with -1
        s_points_idxs = _cluster_to_spoints(clusters)
        neighbors_idxs_list.append(s_points_idxs)

        voxel_size *= 2 # double the voxel size for the next layer

    
    for i in range(num_layers):
        if i < num_layers - 1:

            subsampling_idxs_list.append(neighbors_idxs_list[i+1])

            rev_i = num_layers - i - 1
            upsampling_idxs_list.append(neighbors_idxs_list[rev_i])


    # final upsampling layer
    #TODO

    return {
        'points_list': points_list,
        'neighbors_idxs_list': neighbors_idxs_list,
        'subsampling_idxs_list': subsampling_idxs_list,
        'upsampling_idxs_list': upsampling_idxs_list
    }



def build_neighbor_grid_graph_pyramid(
    points: torch.Tensor,
    num_layers: int,
    voxel_size: float,
    num_neighbors: List[int],
    neighborhood_strategy: str,
    neighborhood_kwargs: Dict,
    neighborhood_kwarg_update: Dict,
    )-> Dict:
    """
    Build a Graph Pyramid using Grid Sampling and Neighboring Strategy

    Parameters
    ----------
    `points` : torch.Tensor
        input tensor of shape (B, N, 3)

    `num_layers` : int
        number of layers in the graph pyramid, i.e., in the GIBLi model

    `voxel_size` : float
        the size of the voxel grid

    `num_neighbors` : List[int]
        the number of neighbors to consider at each layer;
        thus, len(num_neighbors) = num_layers - 1

    `neighborhood_strategy` : str \in {'knn', 'dbscan', 'radius_ball'}
        the strategy to use for selecting neighbors

    `neighborhood_kwargs` : Dict
        additional hyperparameters for the selected strategy;
        if the strategy is 'knn', then there is no need for additional hyperparameters;
        if the strategy is 'dbscan', then the hyperparameters are `eps` and `min_points`;
        if the strategy is 'radius_ball', then the hyperparameter is `radius`;

        for example, if the strategy is 'radius_ball', then neighborhood_kwargs = {'radius': 0.1}

    `neighborhood_kwarg_update` : Dict
        update the hyperparameters for the selected strategy in each layer;
        for example, if the strategy is 'radius_ball', then neighborhood_kwarg_update = {'radius': 2.0}, so that the search radius is doubled in each layer;

    Returns
    -------
    `graph_pyramid` : Dict
        a dictionary containing the following
        - `points_list` : List[torch.Tensor]
            a list of tensors containing the points at each layer, each with shape (B, Q, 3), where Q is updated according to the sampling ratio
        - `neighbors_idxs_list` : List[torch.Tensor]
            a list of tensors containing the indices of the neighbors at each layer, each with shape (B, Q, k), where k is the number of neighbors at each layer
        - `subsampling_idxs_list` : List[torch.Tensor]
            a list of tensors containing the
    """

    points_list:List[torch.Tensor] = []
    neighbors_idxs_list:List[torch.Tensor] = []
    subsampling_idxs_list:List[torch.Tensor] = []
    upsampling_idxs_list:List[torch.Tensor] = []

    points = points[..., :3]

    points_list.append(points)

    for _ in range(num_layers):
        points = Query_Points('grid', voxel_size=voxel_size)(points)
        points_list.append(points)

        voxel_size *= 2 # double the voxel size for the next layer

    
    for i in range(num_layers):
        curr_points = points_list[i]

        neighborhood_finder = Neighboring(neighborhood_strategy, num_neighbors[i], **neighborhood_kwargs)
        upsample_kwargs = {}
        for key, value in neighborhood_kwargs.items():
            if key in neighborhood_kwarg_update:
                upsample_kwargs[key] = value * neighborhood_kwarg_update[key]
        
        upsample_neighborhood_finder = Neighboring(neighborhood_strategy, num_neighbors[i+1], **upsample_kwargs)

        s_points_idxs = neighborhood_finder(curr_points, curr_points)
        neighbors_idxs_list.append(s_points_idxs)


        if i < num_layers - 1:
            next_points = points_list[i+1] # (B, Q[i+1], 3)

            sub_points_idxs = neighborhood_finder(curr_points, next_points) # (B, Q[i+1], k[i])
            subsampling_idxs_list.append(sub_points_idxs)


            up_points_idxs = upsample_neighborhood_finder(next_points, curr_points) # (B, Q[i], k[i+1])
            upsampling_idxs_list.append(up_points_idxs)

        
        neighborhood_kwargs = upsample_kwargs


    return {
        'points_list': points_list,
        'neighbors_idxs_list': neighbors_idxs_list,
        'subsampling_idxs_list': subsampling_idxs_list,
        'upsampling_idxs_list': upsampling_idxs_list
    }





    








