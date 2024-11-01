
import torch
from typing import Union, Tuple, Optional

class Farthest_Point_Sampling:

    def __init__(self, num_points: int) -> None:
        self.num_points = num_points


    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x - torch.Tensor
            input tensor of shape ([B], N, 3 + F)

        Returns
        -------
        q_points - torch.Tensor
            query point coords of shape ([B], num_points, 3)
        """
        import torch_fpsample
        # from torch_cluster import fps

        pointcloud = x[..., :3]

        if pointcloud.ndim == 3: # if the input is a batch of point clouds
            B, _, F = pointcloud.shape
            batch_vector = torch.arange(pointcloud.shape[0], device=pointcloud.device).repeat_interleave(pointcloud.shape[1])
            pointcloud = pointcloud.reshape(-1, pointcloud.shape[-1]) # shape = (B*N, 3 + F)
            self.num_points = self.num_points * B
        else:
            B = None
            batch_vector = None

        start_idx = torch.randint(0, pointcloud.shape[0], (1,)).item()
        fps_indices = torch_fpsample.sample(pointcloud.cpu(), self.num_points, start_idx=start_idx)[1].to(pointcloud.device)
        q_points = x[fps_indices]

        if B is not None:
            q_points = q_points.reshape(B, -1, q_points.shape[-1])
            # fps_indices = fps_indices.view(B, -1)

        return q_points

      

