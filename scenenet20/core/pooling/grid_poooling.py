import torch
import torch_cluster
import torch_scatter

from typing import Union


def _compute_grid_size(points: torch.Tensor, num_voxels: torch.Tensor) -> torch.Tensor:
    """
    Computes the grid size based on the bounding box of the points and the desired number of voxels.
    
    Parameters
    ----------
    points : torch.Tensor
        Tensor of shape (N, 3) representing the coordinates of the points.
    
    num_voxels : torch.Tensor
        Tensor of shape (3,) representing the desired number of voxels in the x, y, and z dimensions.
    
    Returns
    -------
    grid_size : torch.Tensor
        Tensor of shape (3,) representing the size of the grid cells in the x, y, and z dimensions.
    """
    min_coords = points.min(dim=0)[0]
    max_coords = points.max(dim=0)[0]
    bbox_size = max_coords - min_coords
    
    grid_size = bbox_size / num_voxels
    return grid_size

def grid_pooling(points: torch.Tensor, voxel_size:Union[None, torch.Tensor]=None, voxelgrid_size:Union[None, torch.Tensor]=None, feat_mapping: str = 'mean') -> torch.Tensor:
    """
    Aggregates points within each voxel of a grid using a specified function.

    Either `voxel_size` or `voxelgrid_size` must be provided.  

    Parameters
    ----------
    `points` : torch.Tensor
        Tensor of shape (N, 3 + C) where N is the number of points and C is the number of additional channels.

    `voxel_size` : (3,) torch.Tensor
        size of a voxel in each dimension.

    `voxelgrid_size` : (3,) torch.Tensor
        size of the entire voxel_grid in each dimension.       

    `feat_mapping` : str or dict[int, str]
        Aggregation function to apply to the points in each voxel. Can be one of {'mean', 'max', 'min', 'sum'}.7
        if a dict is provided, it should map the index of the feature to the aggregation function to be applied to that feature.

    Returns
    -------
    `aggregated_points` : torch.Tensor
        Tensor of shape (M, 3 + C) where M is the number of unique voxels.
    """

    if voxel_size is None and voxelgrid_size is None:
        raise ValueError("Either 'voxel_size' or 'voxelgrid_size' must be provided.")
    
    pos = points[:, :3]       # Positions
    features = points[:, 3:]  # Additional features

    if voxel_size is None:
        voxel_size = _compute_grid_size(pos, voxelgrid_size)
        print(f"voxel_size: {voxel_size}")

    # Get cluster assignments
    cluster = torch_cluster.grid_cluster(pos, size=voxel_size)

    # Define aggregation functions
    agg_funcs = {
        'mean': torch_scatter.scatter_mean,
        'max': torch_scatter.scatter_max,
        'min': torch_scatter.scatter_min,
        'sum': torch_scatter.scatter_sum,
    }

    if isinstance(feat_mapping, dict):
        if features.numel() == 0:
            raise ValueError("No features provided to apply aggregation functions to.")
        
        aggregated_points = agg_funcs['mean'](pos, cluster, dim=0) # positions are always aggregated using the mean
        
        for feat_idx, func_name in sorted(feat_mapping.items()):
            if feat_idx >= features.shape[1]:
                raise ValueError(f"Feature index {feat_idx} out of bounds. Must be less than {features.shape[1]}.")
            if func_name not in agg_funcs:
                raise ValueError(f"Invalid aggregation function '{func_name}'. Choose from {list(agg_funcs.keys())}.")
            
            agg_func = agg_funcs[func_name]
            agg_feat = agg_func(features[:, feat_idx], cluster, dim=0)
            if isinstance(agg_feat, tuple):
                agg_feat = agg_feat[0]
            aggregated_points = torch.cat([aggregated_points, agg_feat.view(-1, 1)], dim=1)
    else: # feat_mapping is a str; Apply the same aggregation function to all features
        
        if feat_mapping not in agg_funcs:
            raise ValueError(f"Invalid feat_mapping '{feat_mapping}'. Choose from {list(agg_funcs.keys())}.") 

        agg_func = agg_funcs[feat_mapping]

        # Aggregate positions and features
        agg_pos = agg_func(pos, cluster, dim=0)
        if isinstance(agg_pos, tuple):  # For max and min operations which return (values, indices)
            agg_pos = agg_pos[0]

        if features.numel() > 0:
            agg_features = agg_func(features, cluster, dim=0)
            if isinstance(agg_features, tuple):
                agg_features = agg_features[0]
            aggregated_points = torch.cat([agg_pos, agg_features], dim=1)
        else:
            aggregated_points = agg_pos

    return aggregated_points


if __name__ == '__main__':
    # Generate random points with additional features
    num_points = 200_000
    num_features = 2
    # points = torch.randint(0, 10, (num_points, 3 + num_features)).float()
    points = torch.rand(num_points, 3 + num_features)

    grid_size = torch.tensor([0.1, 0.1, 0.1])
    voxelgrid_size = torch.tensor([64.0, 64.0, 64.0])

    aggregated_points = grid_pooling(points, grid_size, None, feat_mapping={-1 : 'max', -2:'min'})

    print(f"Original number of points: {points.shape}")
    print(f"Aggregated number of points: {aggregated_points.shape}")

    print(torch.max(points, dim=0)[0][-1])
    print(torch.max(aggregated_points, dim=0)[0][-1])
