


import torch
import sys
sys.path.append('../')
sys.path.append('../../')
from core.sampling.grid_sampling import Grid_Sampling
from core.sampling.FPS import Farthest_Point_Sampling


class Query_Points:


    def __init__(self, sampling_strategy:str, num_q_points, **kwargs) -> None:
        if sampling_strategy == 'grid':
            self.sampling = Grid_Sampling(**kwargs)
        elif sampling_strategy == 'fps':
            self.sampling = Farthest_Point_Sampling(num_q_points)
        else:
            raise NotImplementedError(f"Sampling strategy {sampling_strategy} not implemented.")
        
        self.num_q_points = num_q_points


    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        q_points = self.sampling(x)
        return q_points[..., :3]
