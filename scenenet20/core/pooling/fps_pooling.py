

import torch
from torch_cluster import fps


import sys
sys.path.append('../')
sys.path.append('../../')
from core.neighboring.neighbors import Neighboring_Method


class FPSPooling_Module(torch.nn.Module):

    def __init__(self, neighboring_strategy, 
                 num_neighbors, 
                 pooling_factor,
                 neigh_kwargs,
                 feat_mapping='max'
                ) -> None:
        
        super(FPSPooling_Module, self).__init__()

        self.fps_pooling = FPSPooling(neighboring_strategy, pooling_factor, num_neighbors, neigh_kwargs, feat_mapping)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.fps_pooling(x)


class FPSPooling:

    def __init__(self, 
                 neighboring_strategy, 
                 pooling_factor,
                 num_neighbors, 
                 neigh_kwargs,
                 feat_mapping='max'
                ) -> None:
        
        self.neighbor = Neighboring_Method(neighboring_strategy, num_neighbors, **neigh_kwargs)

        self.pooling_factor = pooling_factor
        self.feat_mapping = feat_mapping
        self.agg_coords = False
    
    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x - torch.Tensor
            input tensor of shape (N, 3 + F)

        Returns
        -------
        pooled_points - torch.Tensor
            pooled points of shape (N/pooling_factor, 3 + F)
        """
        q_points = fps_sampling(x, pooling_factor=self.pooling_factor) # (q_points, 3)

        s_point_idxs = self.neighbor(x[..., :3], q_points) # (q_points, num_neighbors)
        s_point_feats = x[s_point_idxs, 3:] # (q_points, num_neighbors, F)
        print(q_points.shape, s_point_idxs.shape, s_point_feats.shape)
        
        # aggregate features of support points at each query point; pooled_points shape = (q_points, F)
        if self.feat_mapping == 'max':
            pooled_points = s_point_feats.max(dim=1)[0]
        elif self.feat_mapping == 'min':
            pooled_points = s_point_feats.min(dim=1)[0]
        elif self.feat_mapping == 'mean':
            pooled_points = s_point_feats.mean(dim=1)
        elif self.feat_mapping == 'sum':
            pooled_points = s_point_feats.sum(dim=1)

        if self.agg_coords:
            s_point_coords = x[s_point_idxs, :3]
            pooled_coords = s_point_coords.mean(dim=1)
            pooled_points = torch.cat([pooled_coords, pooled_points], dim=-1)

        return torch.cat([q_points, pooled_points], dim=-1)

@torch.jit.script
def fps_sampling(x:torch.Tensor, pooling_factor:float=0.0, num_points:int=0) -> torch.Tensor:
        """
        Parameters
        ----------
        x - torch.Tensor
            input tensor of shape (B, N, F)

        Returns
        -------
        q_points - torch.Tensor
            query points of shape (B, N/pooling_factor, 3)
        """
        
        assert pooling_factor > 0.0 or num_points > 0, f"Either pooling_factor {pooling_factor} or num_points {num_points} must be provided."

        pointcloud = x[..., :3]

        if pointcloud.ndim == 3: # if the input is a batch of point clouds
            B, N, F = pointcloud.shape
            batch_vector = torch.arange(pointcloud.shape[0], device=pointcloud.device).repeat_interleave(pointcloud.shape[1])
            pointcloud = pointcloud.view(-1, pointcloud.shape[-1]) # shape = (B*N, 3 + F)
        else:
            B = None
            N = pointcloud.shape[0]
            batch_vector = None

        if pooling_factor == 0.0:
            ratio = num_points / N
        else:
            ratio = 1 / pooling_factor

        fps_indices = fps(pointcloud, batch=batch_vector, ratio=ratio, random_start=True)
        q_points = pointcloud[fps_indices]

        if B is not None:
            q_points = q_points.view(B, -1, q_points.shape[-1])

        return q_points


if __name__ == "__main__":
    # Define the point cloud
    points = torch.rand((100_000, 5))
    
    # Perform farthest point pooling
    pooling = FPSPooling(neighboring_strategy='knn', pooling_factor=2, num_neighbors=10, neigh_kwargs={'k':10})

    pooled_indices = pooling(points)
    
    print(f"Original point cloud shape: {points.shape}")
    print(f"Pooled point cloud shape: {pooled_indices.shape}")