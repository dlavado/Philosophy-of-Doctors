

import torch
import sys
sys.path.append('../')
sys.path.append('../../')
from core.neighboring.dbscan import DBSCAN_Neighboring
from core.neighboring.knn import KNN_Neighboring
from core.neighboring.radius_ball import RadiusBall_Neighboring

class Neighboring_Method:


    def __init__(self,
                 neighborhood_strategy:str,
                 num_neighbors:int,
                 **kwargs   
            ) -> None:
        
        """
        This class is used to select a preferred neighboring strategy for a given point cloud data.

        The Options for neighborhood_strategy are:
        - `knn` - K-Nearest Neighbors, with `num_neighbors` as the number of neighbors to consider.
        - `dbscan` - DBSCAN, with `eps` and `min_points` as the hyperparameters to be passed in kwargs.
        - `radius_ball` - Radius Ball, with `radius` as the hyperparameter to be passed in kwargs.

        Parameters
        ----------
        neighborhood_strategy - str
            the strategy to use for selecting neighbors

        num_neighbors - int
            the number of neighbors to select
        
        **kwargs
            additional hyperparameters for the selected strategy
        """
        
        if neighborhood_strategy == 'knn':
            self.neighbor = KNN_Neighboring(num_neighbors)
        elif neighborhood_strategy == 'dbscan':
            self.neighbor = DBSCAN_Neighboring(eps=kwargs['eps'], min_points=kwargs['min_points'], k=num_neighbors)
        elif neighborhood_strategy == 'radius_ball':
            self.neighbor = RadiusBall_Neighboring(radius=kwargs['radius'], k=num_neighbors)
        else:
            raise NotImplementedError(f"Neighborhood strategy {neighborhood_strategy} not implemented.")
        

    def __call__(self, q_points:torch.Tensor, support:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        q_points - torch.Tensor
            query points of shape (B, Q, 3)

        support - torch.Tensor
            input tensor of shape (B, N, 3)

        Returns
        -------
        neighbor_idxs - torch.Tensor
            neighbor indices of shape (B, Q, k)
        """

        neigh_idxs = self.neighbor(q_points, support)
        return neigh_idxs

        


