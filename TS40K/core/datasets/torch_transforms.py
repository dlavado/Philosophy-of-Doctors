
from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np
from torch_cluster import fps


import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from utils import voxelization as Vox
from utils import pointcloud_processing as eda


class Dict_to_Tuple:

    def __init__(self, omit:Union[str, list]=None) -> None:

        self.omit = omit

    def __call__(self, sample:dict):

        return tuple([sample[key] for key in sample.keys() if key not in self.omit])


class ToTensor:

    def __call__(self, sample):
        sample = list(sample)
        return tuple([torch.from_numpy(s.astype(np.float64)) for s in sample])


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


    def __call__(self, sample:torch.Tensor):
        
        pts, labels = sample
        point_feats = None

        if pts.dim() == 3: # batched point clouds
            pts = pts[0] # decapsulate the batch dimension
            labels = labels[0]

        if pts.shape[1] > 3: # if the point cloud has features
            point_feats = pts[:, 3:] # keep the features
            pts = pts[:, :3] # keep only the xyz coordinates  

        vox, gt, pt_locs = Vox.torch_voxelize_input_pcd(
            pts, labels, self.keep_labels,
            voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size
        )
        
        if point_feats is not None:
            pt_locs = torch.concatenate([pt_locs, point_feats], dim=-1) # add the features to the point locations
        
        return vox, gt, pt_locs
    

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
    

class Farthest_Point_Sampling:

    def __init__(self, num_points: Union[int, torch.Tensor]) -> None:
        self.num_points = num_points # if tensor, then it is the batch size and corresponds to dim 0 of the input tensor


    def __call__(self, sample) -> Tuple[torch.Tensor, torch.Tensor]:

        # print(f"Sample: {sample[0].shape}, {sample[1].shape}")
        if isinstance(sample, tuple): 
            pointcloud, labels = sample
        else:
            pointcloud, labels = sample[:, :, :-1], sample[:, :, -1]

        if pointcloud.shape[1] < self.num_points: # if the number of points in the point cloud is less than the number of points to sample
            # print(f"sample shape {pointcloud.shape} < num_points {self.num_points}")
            # duplicate the points until the number of points is equal to the number of points to sample
            random_indices = torch.randint(0, pointcloud.shape[1] - 1, size=(self.num_points - pointcloud.shape[1],))

            pointcloud = torch.cat([pointcloud, pointcloud[:, random_indices]], dim=1)
            labels = torch.cat([labels, labels[:, random_indices]], dim=1)

        # ply = eda.np_to_ply(pointcloud[0].detach().cpu().numpy())
        # eda.color_pointcloud(ply, labels[0].detach().cpu().numpy())
        # eda.visualize_ply([ply])

        # add labels to the point cloud, shape = (B, P, 4)
        data = torch.concat([pointcloud, labels.unsqueeze(-1)], dim=-1)

        # pointcloud = self.farthest_point_sampling_with_features(data, self.num_points) # shape = (B, N, 3 + F)
        with torch.no_grad():
            indices = fps(data[0], batch=None, ratio=self.num_points/data.shape[1], random_start=True) # shape = (B, N, 3 + F)
            pointcloud = data[0, indices] # shape = (N, 3 + F)

            if pointcloud.shape[0] < self.num_points: # if the number of points in the point cloud is less than the number of points to sample
                pointcloud = torch.cat([pointcloud, pointcloud[torch.randint(0, pointcloud.shape[0] - 1, size=(self.num_points - pointcloud.shape[0],))]], dim=0)
            elif pointcloud.shape[0] > self.num_points:
                pointcloud = pointcloud[:self.num_points] # shape = (N, 3 + F)

        pointcloud, labels = pointcloud[:, :-1], pointcloud[:, -1]

        # ply = eda.np_to_ply(pointcloud.detach().cpu().numpy())
        # eda.color_pointcloud(ply, labels.detach().cpu().numpy())
        # eda.visualize_ply([ply])

        # print(f"Sampled point cloud shape: {pointcloud.shape}, labels shape: {labels.shape}")
        # print(f"Sampled point cloud unique labels: {torch.unique(labels)}")
        return pointcloud, labels
    
    # code stolen from pytorch3d cuz their library does not install
    def sample_farthest_points_naive(self,
        points: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        K: Union[int, list, torch.Tensor] = 50,
        random_start_point: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iterative farthest point sampling algorithm [1] to subsample a set of
        K points from a given pointcloud. At each iteration, a point is selected
        which has the largest nearest neighbor distance to any of the
        already selected points.

        Farthest point sampling provides more uniform coverage of the input
        point cloud compared to uniform random sampling.

        [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
            on Point Sets in a Metric Space", NeurIPS 2017.

        Args:
            points: (N, P, D) array containing the batch of pointclouds
            lengths: (N,) number of points in each pointcloud (to support heterogeneous
                batches of pointclouds)
            K: samples required in each sampled point cloud (this is typically << P). If
                K is an int then the same number of samples are selected for each
                pointcloud in the batch. If K is a tensor is should be length (N,)
                giving the number of samples to select for each element in the batch
            random_start_point: bool, if True, a random point is selected as the starting
                point for iterative sampling.

        Returns:
            selected_points: (N, K, D), array of selected values from points. If the input
                K is a tensor, then the shape will be (N, max(K), D), and padded with
                0.0 for batch elements where k_i < max(K).
            selected_indices: (N, K) array of selected indices. If the input
                K is a tensor, then the shape will be (N, max(K), D), and padded with
                -1 for batch elements where k_i < max(K).
        """
        N, P, D = points.shape
        device = points.device

        # Validate inputs
        if lengths is None:
            lengths = torch.full((N,), P, dtype=torch.int64, device=device)
        else:
            if lengths.shape != (N,):
                raise ValueError("points and lengths must have same batch dimension.")
            if lengths.max() > P:
                raise ValueError("Invalid lengths.")

        # TODO: support providing K as a ratio of the total number of points instead of as an int
        if isinstance(K, int):
            K = torch.full((N,), K, dtype=torch.int64, device=device)
        elif isinstance(K, list):
            K = torch.tensor(K, dtype=torch.int64, device=device)

        if K.shape[0] != N:
            raise ValueError("K and points must have the same batch dimension")

        # Find max value of K
        max_K = torch.max(K)

        # List of selected indices from each batch element
        all_sampled_indices = []

        for n in range(N):
            # Initialize an array for the sampled indices, shape: (max_K,)
            sample_idx_batch = torch.full(
                # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
                #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
                (max_K,),
                fill_value=-1,
                dtype=torch.int64,
                device=device,
            )

            # Initialize closest distances to inf, shape: (P,)
            # This will be updated at each iteration to track the closest distance of the
            # remaining points to any of the selected points
            closest_dists = points.new_full(
                # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
                #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
                (lengths[n],),
                float("inf"),
                dtype=torch.float32,
                device = device,
            )

            # Select a random point index and save it as the starting point
            selected_idx = torch.randint(0, lengths[n] - 1, device=device) if random_start_point else 0
            sample_idx_batch[0] = selected_idx

            # If the pointcloud has fewer than K points then only iterate over the min
            # pyre-fixme[6]: For 1st param expected `SupportsRichComparisonT` but got
            #  `Tensor`.
            # pyre-fixme[6]: For 2nd param expected `SupportsRichComparisonT` but got
            #  `Tensor`.
            k_n = min(lengths[n], K[n])

            # Iteratively select points for a maximum of k_n
            for i in range(1, k_n):
                # Find the distance between the last selected point
                # and all the other points. If a point has already been selected
                # it's distance will be 0.0 so it will not be selected again as the max.
                dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

                # If closer than currently saved distance to one of the selected
                # points, then updated closest_dists
                closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

                # The aim is to pick the point that has the largest
                # nearest neighbour distance to any of the already selected points
                selected_idx = torch.argmax(closest_dists)
                sample_idx_batch[i] = selected_idx

            # Add the list of points for this batch to the final list
            all_sampled_indices.append(sample_idx_batch)

        all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

        # Gather the points
        all_sampled_points = self._masked_gather(points, all_sampled_indices)

        # Return (N, max_K, D) subsampled points and indices
        return all_sampled_points, all_sampled_indices
    
    def _masked_gather(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Helper function for torch.gather to collect the points at
        the given indices in idx where some of the indices might be -1 to
        indicate padding. These indices are first replaced with 0.
        Then the points are gathered after which the padded values
        are set to 0.0.

        Args:
            points: (N, P, D) float32 tensor of points
            idx: (N, K) or (N, P, K) long tensor of indices into points, where
                some indices are -1 to indicate padding

        Returns:
            selected_points: (N, K, D) float32 tensor of points
                at the given indices
        """

        if len(idx) != len(points):
            raise ValueError("points and idx must have the same batch dimension")

        N, P, D = points.shape

        if idx.ndim == 3:
            # Case: KNN, Ball Query where idx is of shape (N, P', K)
            # where P' is not necessarily the same as P as the
            # points may be gathered from a different pointcloud.
            K = idx.shape[2]
            # Match dimensions for points and indices
            idx_expanded = idx[..., None].expand(-1, -1, -1, D)
            points = points[:, :, None, :].expand(-1, -1, K, -1)
        elif idx.ndim == 2:
            # Farthest point sampling where idx is of shape (N, K)
            idx_expanded = idx[..., None].expand(-1, -1, D)
        else:
            raise ValueError("idx format is not supported %s" % repr(idx.shape))

        idx_expanded_mask = idx_expanded.eq(-1)
        idx_expanded = idx_expanded.clone()
        # Replace -1 values with 0 for gather
        idx_expanded[idx_expanded_mask] = 0
        # Gather points
        selected_points = points.gather(dim=1, index=idx_expanded)
        # Replace padded values
        selected_points[idx_expanded_mask] = 0.0
        return selected_points

        
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


