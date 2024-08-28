
from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np


import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from utils import pointcloud_processing as eda


class Dict_to_Tuple:
    def __init__(self, omit:Union[str, list]=None) -> None:
        self.omit = omit

    def __call__(self, sample:dict):
        return tuple([sample[key] for key in sample.keys() if key not in self.omit])

class Add_Batch_Dim:

    def __call__(self, sample) -> Any:
        sample = list(sample)
        return tuple([s.unsqueeze(0) for s in sample])

class ToTensor:
    def __call__(self, sample):
        sample = list(sample)
        return tuple([torch.from_numpy(s.astype(np.float64)) for s in sample])

class To:

    def __init__(self, dtype:torch.dtype=torch.float32) -> None:
        self.dtype = dtype

    def __call__(self, sample):
        sample = list(sample)
        return tuple([s.to(self.dtype) for s in sample])

class ToDevice:

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, sample):
        if isinstance(sample, Sequence):
            return tuple([s.to(self.device) for s in sample])
        elif isinstance(sample, dict):
            return {key: value.to(self.device) for key, value in sample.items()}
        else:
            return sample.to(self.device)


class EDP_Labels:
    def __call__(self, sample) -> Any:
        pcd, labels, *args = sample

        labels = self.edp_labels(labels)

        return pcd, labels, *args
    
    def edp_labels(self, labels:torch.Tensor) -> torch.Tensor:

        #cast each label to its corresponding EDP label
        new_labels = torch.tensor([eda.DICT_NEW_LABELS[label.item()] if label.item() >= 0 else label.item() for label in labels.squeeze()]).reshape(labels.shape)
        # print(f"labels NEW unique: {torch.unique(new_labels)}, labels shape: {new_labels.shape}")
        return new_labels
    

class Normalize_Labels:

    def __call__(self, sample) -> Any:
        """
        Normalize the labels to be between [0, num_classes-1]
        """

        pointcloud, labels, pt_locs = sample

        labels = self.normalize_labels(labels)

        return pointcloud, labels, pt_locs
    
    def normalize_labels(self, labels:torch.Tensor) -> torch.Tensor:
        """

        labels - tensor with shape (P,) and values in [0, C -1] not necessarily contiguous
        

        transform the labels to be between [0, num_classes-1] with contiguous values
        """

        unique_labels = torch.unique(labels)
        num_classes = unique_labels.shape[0]
        
        labels = labels.unsqueeze(-1) # shape = (P, 1)
        labels = (labels == unique_labels).float() # shape = (P, C)
        labels = labels * torch.arange(num_classes).to(labels.device) # shape = (P, C)
        labels = labels.sum(dim=-1).long() # shape = (P,)
       
        return labels


class Ignore_Label:

    def __init__(self, ignore_label:int) -> None:
        self.ignore_label = ignore_label

    def __call__(self, sample) -> Any:
        """
        Ignore the points with the ignore label
        """

        pointcloud, labels = sample

        mask = labels == self.ignore_label

        # if pointcloud.ndim >= 3:
        #     pointcloud[mask[None]] = -1 # only if 
        labels[mask] = -1 # ignore the points with the ignore label

        return pointcloud, labels   

class Random_Point_Sampling:

    def __init__(self, num_points: Union[int, torch.Tensor]) -> None:
        self.num_points = num_points

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample `num_points` from the point cloud
        """

        if isinstance(sample, tuple):
            pointcloud, labels = sample
        else:
            pointcloud, labels = sample[:, :, :-1], sample[:, :, -1]

        if pointcloud.shape[1] < self.num_points:
            random_indices = torch.randint(0, pointcloud.shape[1] - 1, size=(self.num_points - pointcloud.shape[1],))

            pointcloud = torch.cat([pointcloud, pointcloud[:, random_indices]], dim=1)
            labels = torch.cat([labels, labels[:, random_indices]], dim=1)
        
        else:
            random_indices = torch.randperm(pointcloud.shape[1])[:self.num_points]
            pointcloud = pointcloud[:, random_indices]
            labels = labels[:, random_indices]


        return pointcloud, labels
    

class Inverse_Density_Sampling:
    """
    Inverse Density Sampling:
    1. calcule the neighbors of each 3D point within a ball of radius `ball_radius`
    2. order the point indices by the number of neighbors
    3. the `num_points` points with the least number of neighbors are sampled
    """

    def __init__(self, num_points, ball_radius) -> None:
        self.num_points = num_points
        self.ball_radius = ball_radius

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(sample, tuple):
            pointcloud, labels = sample
        else: # torch tensor
            if sample.ndim == 3: # batched point clouds
                pointcloud, labels = sample[:, :, :-1], sample[:, :, -1]
            else:
                pointcloud, labels = sample[:, :-1], sample[:, -1] # preprocessed sample

        idis_pointcloud = torch.empty((pointcloud.shape[0], self.num_points, pointcloud.shape[2]), device=pointcloud.device)
        idis_labels = torch.empty((pointcloud.shape[0], self.num_points), dtype=torch.long, device=pointcloud.device)

        if pointcloud.ndim == 3: # batched point clouds
            for i in range(pointcloud.shape[0]):
                knn_indices = self.inverse_density_sampling(pointcloud[i], self.num_points, self.ball_radius)
                idis_pointcloud[i] = pointcloud[i, knn_indices]
                idis_labels[i] = labels[i, knn_indices]
        else:
            # print(f"pointcloud shape: {pointcloud.shape}, labels shape: {labels.shape}")
            knn_indices = self.inverse_density_sampling(pointcloud, self.num_points, self.ball_radius)
            idis_pointcloud = pointcloud[:, knn_indices]
            idis_labels = labels[:, knn_indices]

        # print(f"idis_pointcloud shape: {idis_pointcloud.shape}, idis_labels shape: {idis_labels.shape}")

        return idis_pointcloud.squeeze(), idis_labels.squeeze()
    
    def inverse_density_sampling(self, pointcloud:torch.Tensor, num_points:int, ball_radius:float) -> torch.Tensor:
        from torch_cluster import radius

        pointcloud = pointcloud.squeeze() # shape = (B, P, 3) -> (P, 3)
        # print(f"pointcloud shape: {pointcloud.shape}")

        indices = radius(pointcloud, pointcloud, r=ball_radius, max_num_neighbors=pointcloud.shape[0]) # shape = (2, P^2)

        #print(f"indices shape: {indices.shape}")
        #print(f"indices: {indices}")
        
        # count the number of neighbors for each point
        num_neighbors = torch.bincount(indices[0], minlength=pointcloud.shape[0]) # shape = (P,)

        #print(f"num_neighbors shape: {num_neighbors.shape}; \nnum_neighbors: {num_neighbors}")

        # select the `num_points` points with the least number of neighbors
        knn_indices = torch.argsort(num_neighbors, dim=-1)[:num_points]

        #print(f"knn_indices shape: {knn_indices.shape}; \nknn_indices: {knn_indices}")

        return knn_indices

        

        
class Normalize_PCD:

    def __call__(self, sample) -> torch.Tensor:
        """
        Normalize the point cloud to have zero mean and unit variance.
        """
        pointcloud, labels = sample
        pointcloud = self.normalize(pointcloud)
        return pointcloud, labels
    

    def normalize(self, pointcloud:torch.Tensor) -> torch.Tensor:
        """
         (x - min(x)) / (max(x) - min(x))
        """

        pointcloud = pointcloud.float()

        if pointcloud.dim() == 3: # batched point clouds
            min_x = pointcloud.min(dim=1, keepdim=True).values
            max_x = pointcloud.max(dim=1, keepdim=True).values
            pointcloud = (pointcloud - min_x) / (max_x - min_x)
        
        else: # single point cloud
            min_x = pointcloud.min(dim=0, keepdim=True).values
            max_x = pointcloud.max(dim=0, keepdim=True).values
            pointcloud = (pointcloud - min_x) / (max_x - min_x)

        return pointcloud

    def standardize(self, pointcloud:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        `pointcloud` - torch.Tensor with shape ((B), P, 3)
            Point cloud to be normalized; Batch dim is optional
        """

        pointcloud = pointcloud.float()

        if pointcloud.dim() == 3: # batched point clouds
            centroid = pointcloud.mean(dim=1, keepdim=True)
            pointcloud = pointcloud - centroid
            max_dist:torch.Tensor = torch.sqrt((pointcloud ** 2).sum(dim=-1)).max(dim=1) # shape = (batch_size,)
            pointcloud = pointcloud / max_dist.values[:, None, None]

        else: # single point cloud
            centroid = pointcloud.mean(dim=0)
            pointcloud = pointcloud - centroid 
            max_dist = torch.sqrt((pointcloud ** 2).sum(dim=-1)).max()
            pointcloud = pointcloud / max_dist

        return pointcloud