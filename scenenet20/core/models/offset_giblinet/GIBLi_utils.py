from typing import Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
sys.path.insert(3, '../../../..')
from core.models.giblinet.neighboring.neighbors import Neighboring_Method
from core.models.giblinet.sampling.query_points_strat import Query_Points



class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of ([B], N, C)

    where B is the batch size, N is the number of points, C is the number of channels
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        output = self.norm(input.contiguous().permute(0, 2, 1))
        return output.permute(0, 2, 1).contiguous()
        
        # if input.dim() == 3:
        #     return (
        #         self.norm(input.contiguous().transpose(1, 2))
        #         .transpose(1, 2)
        #         .contiguous()
        #     )
        # elif input.dim() == 2:
        #     return self.norm(input)
        # else:
        #     raise NotImplementedError

class Neighboring(nn.Module):

    def __init__(self, 
                 neighborhood_strategy:str, 
                 num_neighbors:int, 
                 **kwargs
                ) -> None:
        
        super(Neighboring, self).__init__()

        self.neighbor = Neighboring_Method(neighborhood_strategy, num_neighbors, **kwargs)

    def forward(self, q_points, support) -> torch.Tensor:
        """
        Parameters
        ----------
        `q_points` : torch.Tensor
            query points of shape (B, Q, 3)

        `support` : torch.Tensor
            input tensor of shape (B, N, 3)

        Returns
        -------
        `neighbor_idxs` : torch.Tensor
            neighbor indices of shape (B, Q, k)
        """
        return self.neighbor(q_points, support)



###############################################################
# Build Graph Pyramid for GIBLi
###############################################################


class BuildGraphPyramid(nn.Module):
    
        def __init__(self, 
                    num_layers:int,
                    graph_strategy:str,
                    sampling_factor:float,
                    num_neighbors:Union[int, List[int]],
                    neighborhood_strategy:str,
                    neighborhood_kwargs:Dict,
                    neighborhood_kwarg_update:Dict,
                    **kwargs
                    ) -> None:
            
            super(BuildGraphPyramid, self).__init__()
    
            self.num_layers = num_layers
            self.graph_strategy = graph_strategy
            self.sampling_factor = sampling_factor
            if isinstance(num_neighbors, int):
                self.num_neighbors = [num_neighbors] * num_layers
            else:
                self.num_neighbors = num_neighbors
            self.neighborhood_strategy = neighborhood_strategy
            self.neighborhood_kwargs = neighborhood_kwargs
            self.neighborhood_kwarg_update = neighborhood_kwarg_update

            if graph_strategy == 'grid':
                self.voxel_size = kwargs['voxel_size']


        def forward(self, points:torch.Tensor) -> Dict:
            if self.graph_strategy == 'fps':
                return build_fps_graph_pyramid(
                    points, 
                    self.num_layers, 
                    1 / self.sampling_factor, 
                    self.num_neighbors, 
                    self.neighborhood_strategy, 
                    self.neighborhood_kwargs, 
                    self.neighborhood_kwarg_update
                )
            elif self.graph_strategy == 'grid':
                return build_grid_graph_pyramid(
                    points, 
                    self.num_layers, 
                    self.voxel_size,
                    self.sampling_factor,
                    self.num_neighbors, 
                    self.neighborhood_strategy, 
                    self.neighborhood_kwargs, 
                    self.neighborhood_kwarg_update
                )
            else:
                raise ValueError("Query Strategy not supported")
               
    
      
@torch.no_grad()
def build_fps_graph_pyramid(
    points: torch.Tensor,
    num_layers: int,
    sampling_ratio: float,
    num_neighbors: List[int],
    neighborhood_strategy: str,
    neighborhood_kwargs: Dict = None,
    neighborhood_kwarg_update: Dict = None,
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
        thus, len(num_neighbors) = num_layers

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
    subsampling_list = []
    upsampling_list = []

    points = points[..., :3]
    num_q_points = points.shape[1] if points.ndim == 3 else points.shape[0]
    points_list.append(points)

    for _ in range(num_layers - 1):
        num_q_points = int(num_q_points * sampling_ratio)
        points = Query_Points('fps', num_points=num_q_points)(points)
        points_list.append(points) # (B, Q, 3)
        # print(points.shape)

    for i in range(num_layers):
        curr_points = points_list[i]

        neighborhood_finder = Neighboring(neighborhood_strategy, num_neighbors[i], **neighborhood_kwargs)

        neighbor_idxs = neighborhood_finder(curr_points, curr_points) # (B, Q[i], k[i]); where Q is the num of q_points and k is the num of neighbors at ith layer
        neighbors_idxs_list.append(neighbor_idxs)
        # print(f"Layer {i}: {neighbor_idxs.shape=}")

        upsample_kwargs = {}
        for key, value in neighborhood_kwargs.items():
            if key in neighborhood_kwarg_update:
                upsample_kwargs[key] = value * neighborhood_kwarg_update[key]
        # print(upsample_kwargs)

        if i < num_layers - 1:
            next_points = points_list[i+1] # (B, Q[i+1], 3)

            upsample_neighborhood_finder = Neighboring(neighborhood_strategy, num_neighbors[i+1], **upsample_kwargs)

            # I want the neighbor_idxs of the current layer wrt the next layer. This way, the features for the next layer can just be aggregated from these indices
            sub_points_idxs = neighborhood_finder(next_points, curr_points) # (B, Q[i+1], k[i]); the indices are from points[i]
            subsampling_list.append(sub_points_idxs)

            # up_points_idxs are the indices of the next_layer (i+1) wrt the current layer (i); 
            # Thus, for each point in the current layer (i), we find the k[i+1] nearest neighbors in the next layer (i+1) in order to to interpolate the features of the latter;
            up_points_idxs = upsample_neighborhood_finder(curr_points, next_points) # (B, Q[i], k[i+1]); the indices are from points[i+1]
            upsampling_list.append(up_points_idxs)

            # print(f"Layer {i}: {sub_points_idxs.shape=}, {up_points_idxs.shape=}")
        
        neighborhood_kwargs = upsample_kwargs


    return {
        'points_list': points_list,
        'neighbors_idxs_list': neighbors_idxs_list,
        'subsampling_idxs_list': subsampling_list,
        'upsampling_idxs_list': upsampling_list
    }
            


@torch.no_grad()
def build_grid_graph_pyramid(
    points: torch.Tensor,
    num_layers: int,
    voxel_size: float,
    voxel_factor:int,
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
    
    voxel_size = voxel_size.to(points.device)

    points_list.append(points)
    for _ in range(num_layers - 1):
        points = Query_Points('grid', voxel_size=voxel_size)(points)
        points_list.append(points)

        voxel_size *= voxel_factor
        # print(f"{points.shape=}")
    
    for i in range(num_layers):
        curr_points = points_list[i]

        neighborhood_finder = Neighboring(neighborhood_strategy, num_neighbors[i], **neighborhood_kwargs)
        upsample_kwargs = {}

        s_points_idxs = neighborhood_finder(curr_points, curr_points)
        neighbors_idxs_list.append(s_points_idxs)
        # print(f"Layer {i}: {s_points_idxs.shape=}")

        for key, value in neighborhood_kwargs.items():
            if key in neighborhood_kwarg_update:
                upsample_kwargs[key] = value * neighborhood_kwarg_update[key]

        if i < num_layers - 1:
            upsample_neighborhood_finder = Neighboring(neighborhood_strategy, num_neighbors[i+1], **upsample_kwargs)

            next_points = points_list[i+1] # (B, Q[i+1], 3)

            sub_points_idxs = neighborhood_finder(next_points, curr_points) # (B, Q[i+1], k[i])
            subsampling_idxs_list.append(sub_points_idxs)

            up_points_idxs = upsample_neighborhood_finder(curr_points, next_points) # (B, Q[i], k[i+1])
            upsampling_idxs_list.append(up_points_idxs)

            # print(f"Layer {i}: {sub_points_idxs.shape=}, {up_points_idxs.shape=}")
        
        neighborhood_kwargs = upsample_kwargs


    return {
        'points_list': points_list,
        'neighbors_idxs_list': neighbors_idxs_list,
        'subsampling_idxs_list': subsampling_idxs_list,
        'upsampling_idxs_list': upsampling_idxs_list
    }


 

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    sys.path.append('../../')
    from utils import constants
    from utils import pointcloud_processing as eda
    from core.datasets.TS40K import TS40K_FULL_Preprocessed, TS40K_FULL


    ts40k = TS40K_FULL_Preprocessed(
        constants.TS40K_FULL_PREPROCESSED_PATH, 
        split='fit', 
        sample_types=['tower_radius', '2_towers'], 
        transform=None, 
        load_into_memory=False
    )

    NUM_LAYERS = 5
    SAMPLING_RATIO = 0.5
    NUM_NEIGHBORS = [10, 20, 30, 40, 50]
    # NEIGHBORHOOD_STRATEGY = 'radius_ball'
    # NEIGHBORHOOD_KWARGS = {'radius': 0.1}
    # NEIGHBORHOOD_KWARG_UPDATE = {'radius': 2.0}
    # NEIGHBORHOOD_STRATEGY = 'dbscan'
    # NEIGHBORHOOD_KWARGS = {'eps': 0.1, 'min_points': 10}
    # NEIGHBORHOOD_KWARG_UPDATE = {'eps': 2.0, 'min_points': 2}
    NEIGHBORHOOD_STRATEGY = 'knn'
    NEIGHBORHOOD_KWARGS = {}
    NEIGHBORHOOD_KWARG_UPDATE = {}


    points, labels = ts40k[0]
    points = points.unsqueeze(0)

    print(points.shape)

    # graph_pyramid = build_fps_graph_pyramid(
    #     points, 
    #     NUM_LAYERS, 
    #     SAMPLING_RATIO, 
    #     NUM_NEIGHBORS, 
    #     NEIGHBORHOOD_STRATEGY, 
    #     NEIGHBORHOOD_KWARGS, 
    #     NEIGHBORHOOD_KWARG_UPDATE
    # )
    VOXEL_SIZE = torch.tensor((0.05, 0.05, 0.05))
    VOXEL_FACTOR = 1.2
    graph_pyramid = build_grid_graph_pyramid(
        points, 
        NUM_LAYERS, 
        VOXEL_SIZE,
        VOXEL_FACTOR,
        NUM_NEIGHBORS, 
        NEIGHBORHOOD_STRATEGY, 
        NEIGHBORHOOD_KWARGS, 
        NEIGHBORHOOD_KWARG_UPDATE
    )

    for key, value in graph_pyramid.items():
        print(key, len(value), value[0].shape) 



