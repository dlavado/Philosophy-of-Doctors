
from typing import Any, Tuple, Union
import torch
import numpy as np
import torch.nn.functional as F
from utils import voxelization as Vox
from pytorch3d.ops import sample_farthest_points


class Dict_to_Tuple:

    def __init__(self, omit:Union[str, list]=None) -> None:

        self.omit = omit

    def __call__(self, sample:dict):

        return tuple([sample[key] for key in sample.keys() if key not in self.omit])


class ToTensor:

    def __call__(self, sample):
        sample = list(sample)
        return tuple([torch.from_numpy(s.astype(np.float64)) for s in sample])



class ToFullDense:
    """
    Transforms a Regression Dataset into a Belief Dataset.

    Essentially, any voxel that has tower points is given a belief of 1,
    in order to maximize the towers' geometry.
    For the input, the density is normalized to 1, so empty voxels have a value
    of 0 and 1 otherwise.

    It requires a discretization of raw LiDAR Point Clouds in Torch format.
    """

    def __init__(self, apply=[True, True]) -> None:
        
        self.apply = apply
    
    def densify(self, tensor:torch.Tensor):
        return (tensor > 0).to(tensor)

    def __call__(self, sample:torch.Tensor):

        vox, gt = [self.densify(tensor) if self.apply[i] else tensor for i, tensor in enumerate(sample) ]
        
        return vox, gt



class Voxelization:

    def __init__(self, keep_labels, vox_size:Tuple[int]=None, vxg_size:Tuple[int]=None) -> None:
        """
        Voxelizes raw LiDAR 3D point points in `numpy` (N, 3) format 
        according to the provided discretization

        Parameters
        ----------
        `vox_size` - Tuple of 3 Ints:
            Size of the voxels to dicretize the point clouds
        `vxg_size` - Tuple of 3 Ints:
            Size of the voxelgrid used to discretize the point clouds

        One of the two parameters need to be provided, `vox_size` takes priority

        Returns
        -------
        A Voxelized 3D point cloud in Density/Probability mode
        """
        
        if vox_size is None and vxg_size is None:
            ValueError("Voxel size or Voxelgrid size must be provided")


        self.vox_size = vox_size
        self.vxg_size = vxg_size
        self.keep_labels = keep_labels


    def __call__(self, sample:np.ndarray):
        
        pts, labels = sample

        voxeled_xyz = Vox.hist_on_voxel(pts, voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size)
        voxeled_gt = Vox.reg_on_voxel(pts, labels, self.keep_labels, voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size)

        return voxeled_xyz[None], voxeled_gt[None] # vox-point-density, vox-tower-prob
    

class Voxelization_withPCD:

    def __init__(self, keep_labels=None, vox_size:Tuple[int]=None, vxg_size:Tuple[int]=None) -> None:
        """
        Voxelizes raw LiDAR 3D point points in `numpy` (N, 3) format 
        according to the provided discretization

        Parameters
        ----------
        `vox_size` - Tuple of 3 Ints:
            Size of the voxels to dicretize the point clouds
        `vxg_size` - Tuple of 3 Ints:
            Size of the voxelgrid used to discretize the point clouds

        One of the two parameters need to be provided, `vox_size` takes priority

        Returns
        -------
        A Voxelized 3D point cloud in Density/Probability mode
        """
        
        if vox_size is None and vxg_size is None:
            ValueError("Voxel size or Voxelgrid size must be provided")


        self.vox_size = vox_size
        self.vxg_size = vxg_size
        self.keep_labels = keep_labels


    def __call__(self, sample:np.ndarray):
        
        pts, labels = sample

        if pts.dim() == 3: # batched point clouds
            pts = pts[0] # decapsulate the batch dimension
            labels = labels[0]

        vox, gt, pt_locs = Vox.voxelize_input_pcd(pts, labels, 
                                                self.keep_labels,
                                                voxel_dims=self.vox_size, 
                                                voxelgrid_dims=self.vxg_size)

        return vox, gt, pt_locs
    

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
    

class Farthest_Point_Sampling:

    def __init__(self, num_points: Union[int, torch.Tensor]) -> None:
        self.num_points = num_points # if tensor, then it is the batch size and corresponds to dim 0 of the input tensor

    def __call__(self, sample) -> torch.Tensor:

        # print(f"Sample: {sample[0].shape}, {sample[1].shape}")

        pointcloud, labels = sample

        data = torch.cat([pointcloud, labels[:, :, None]], dim=-1) # add labels to the point cloud, shape = (B, P, 4)

        pointcloud = sample_farthest_points(data, K=self.num_points, random_start_point=True)[0] # return the sampled points, not the indices
        
        pointcloud, labels = pointcloud[:, :, :3], pointcloud[:, :, 3] # remove the labels from the point cloud, shape = (B, P, 3)

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


