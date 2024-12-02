
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
    
    
class Add_Normal_Vector:

    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add the normal vector to the point cloud
        """
        if isinstance(sample, torch.Tensor):
            pointcloud = sample
            normals = self.calc_normals(pointcloud)
            pointcloud = torch.cat([pointcloud, normals], dim=-1)
            return pointcloud
        else:
            pointcloud, *args = sample
            normals = self.calc_normals(pointcloud)
            pointcloud = torch.cat([pointcloud, normals], dim=-1)

            return pointcloud, *args
    
    def calc_normals(self, pointcloud:torch.Tensor) -> torch.Tensor:
        """
        Calculate the normal vector of the point cloud
        """
        normals = eda.estimate_normals(pointcloud.numpy(force=True))
        normals = torch.from_numpy(normals).to(pointcloud.device).to(pointcloud.dtype)
        return normals


class Repeat_Points:

    def __init__(self, num_points:int) -> None:
        """
        Repeat the points in the point cloud until the number of points is equal to the number of points to sample;

        Useful for batch training.
        """
        self.num_points = num_points

    def __call__(self, sample) -> Any:

        pointcloud, labels = sample
        if pointcloud.ndim == 3: # batched point clouds
           point_dim = 1
        else:
            point_dim = 0
        if pointcloud.shape[point_dim] < self.num_points:
            # duplicate the points until the number of points is equal to the number of points to sample
            random_indices = torch.randint(0, pointcloud.shape[point_dim] - 1, size=(self.num_points - pointcloud.shape[point_dim],))

            if pointcloud.ndim == 3:    
                pointcloud = torch.cat([pointcloud, pointcloud[:, random_indices]], dim=point_dim)
                labels = torch.cat([labels, labels[:, random_indices]], dim=point_dim)
            else:
                pointcloud = torch.cat([pointcloud, pointcloud[random_indices]], dim=point_dim)
                labels = torch.cat([labels, labels[random_indices]], dim=point_dim)

        return pointcloud, labels
    

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