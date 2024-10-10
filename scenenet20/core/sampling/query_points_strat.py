


import torch
import sys
sys.path.append('../')
sys.path.append('../../')
from core.sampling.FPS import Farthest_Point_Sampling
from core.pooling.grid_poooling import GridPooling

class Query_Points:


    def __init__(self, sampling_strategy:str, **kwargs) -> None:
        """
        This class is used to select a preferred strategy for selecting query points from a point cloud data.

        The Options for sampling_strategy are:
        - `grid` - Grid Pooling, with `grid_size` as the hyperparameter to be passed in kwargs.
        - `fps` - Farthest Point Sampling, with `num_q_points` as the hyperparameter to be passed in kwargs.
        """
        if sampling_strategy == 'grid':
            self.sampling = GridPooling(**kwargs)
        elif sampling_strategy == 'fps':
            self.sampling = Farthest_Point_Sampling(**kwargs)
        else:
            raise NotImplementedError(f"Sampling strategy {sampling_strategy} not implemented.")
        


    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        q_points = self.sampling(x)
        return q_points[..., :3]
