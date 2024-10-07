
import torch
from typing import Union, Tuple, Optional

class Farthest_Point_Sampling:

    def __init__(self, num_points: Union[int, torch.Tensor], fps_labels=True) -> None:
        self.num_points = num_points # if tensor, then it is the batch size and corresponds to dim 0 of the input tensor
        self.fps_labels = fps_labels # if True, then the labels are also sampled with the point cloud


    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        from torch_cluster import fps

        if self.fps_labels:
            if isinstance(sample, tuple): 
                pointcloud, labels = sample 
            else:
                pointcloud, labels = sample[..., :-1], sample[..., -1]
        else:
            pointcloud, target = sample
            labels = torch.zeros_like(target)

        if pointcloud.ndim == 3: # if the input is a batch of point clouds
            B, _, F = pointcloud.shape
            batch_vector = torch.arange(pointcloud.shape[0], device=pointcloud.device).repeat_interleave(pointcloud.shape[1])
            pointcloud = pointcloud.view(-1, pointcloud.shape[-1]) # shape = (B*N, 3 + F)
        else:
            B = None
            batch_vector = None

        fps_indices = fps(pointcloud, batch=batch_vector, ratio=self.num_points/pointcloud.shape[0], random_start=True)

        pointcloud = pointcloud[fps_indices] # shape = (N, 3 + F + 1)

        if pointcloud.shape[0] < self.num_points: # if the number of points in the point cloud is less than the number of points to sample
            pointcloud = torch.cat([pointcloud, pointcloud[torch.randint(0, pointcloud.shape[0] - 1, size=(self.num_points - pointcloud.shape[0],))]], dim=0)
        elif pointcloud.shape[0] > self.num_points:
            pointcloud = pointcloud[:self.num_points] # shape = (N, 3 + F)


        if B is not None:
            pointcloud = pointcloud.view(B, -1, F)
            labels = labels.view(B, -1)

        if self.fps_labels:
            return pointcloud, labels[fps_indices]
        else:
            return pointcloud, target
        

