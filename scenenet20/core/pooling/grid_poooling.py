import torch
import torch_cluster
import torch_scatter

from typing import Union, Tuple


class GridPooling_Module(torch.nn.Module):

    def __init__(self, grid_size: Tuple[float, float, float], feat_mapping:Union[str,dict]='max') -> None:
        super(GridPooling_Module, self).__init__()
        self.grid_pooling = GridPooling(grid_size, feat_mapping)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.grid_pooling(x)


class GridPooling:
    def __init__(self, voxel_size: Tuple[float, float, float], feat_mapping:Union[str,dict]='max') -> None:
        self.voxel_size = voxel_size
        self.feat_mapping = feat_mapping
    
    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        """
        Aggregates points within each voxel of a grid using a specified function.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ([B], N, 3 + C) where N is the number of points and C is the number of additional channels.
        
        Returns
        -------
        aggregated_points : torch.Tensor
            Tensor of shape ([B], M, 3 + C) where M is the number of unique voxels.
        """
        return grid_pooling_batch(x, self.voxel_size, None, self.feat_mapping)[0]
    

def _cluster_to_spoints(cluster: torch.Tensor) -> torch.Tensor:
    """
    Converts cluster assignments to support point format.
    That is, cluster has a shape ([B], N) where N is the number of points and the values are the cluster assignments.
    The support points format has a shape ([B], Q, k) where Q is the number of query points and k is the number of neighbors.
    Use -1 to pad the neighbors if the number of neighbors is less than k.

    Parameters
    ----------
    `cluster` : torch.Tensor
        Tensor of shape ([B], N) containing the cluster assignments for each point.

    Returns
    -------
    `s_points` : torch.Tensor
        Tensor of shape ([B], Q, k) containing the indices of the k nearest neighbors of each query point.
    """
    
    if cluster.dim() == 1:
        q_num_points = cluster.max().item() + 1
        counts = torch.bincount(cluster, minlength=q_num_points)
        k_max_neighbors = counts.max().item()
        s_points = torch.full((q_num_points, k_max_neighbors), -1, dtype=torch.long)

        indices = torch.arange(cluster.size(0))
        for i in range(q_num_points):
            neighbors = indices[cluster == i]
            s_points[i, :neighbors.size(0)] = neighbors

    else:
        batch_size = cluster.size(0)
        q_num_points = cluster.max().item() + 1
        counts = torch.bincount(cluster.view(-1), minlength=q_num_points)
        k_max_neighbors = counts.max().item()
        s_points = torch.full((batch_size, q_num_points, k_max_neighbors), -1, dtype=torch.long)

        for i in range(q_num_points):
            neighbors = torch.where(cluster == i)  # Get (batch_idx, point_idx)
            for b in range(batch_size):
                batch_neighbors = neighbors[1][neighbors[0] == b]
                s_points[b, i, :batch_neighbors.size(0)] = batch_neighbors

    return s_points


def spoints_to_cluster(s_points: torch.Tensor) -> torch.Tensor:
    """
    Converts support point format to cluster assignments (also known as pack_mode).
    tensor of shape ([B], N), where N is the number of points, and each value is the corresponding cluster assignment.

    Parameters
    ----------
    `s_points` : torch.Tensor
        Tensor of shape ([B], Q, k) containing the indices of the k nearest neighbors of each query point.

    Returns
    -------
    `cluster` : torch.Tensor
        Tensor of shape ([B], N) containing the cluster assignments for each point.
    """

    if s_points.dim() == 2:
        cluster = torch.full((s_points.max().item() + 1,), -1, dtype=torch.long)

        for cluster_id in range(s_points.size(0)):
            points_in_cluster = s_points[cluster_id, :]
            points_in_cluster = points_in_cluster[points_in_cluster != -1]
            cluster[points_in_cluster] = cluster_id

    else:
        batch_size = s_points.size(0)
        cluster = torch.full((batch_size, s_points.max().item() + 1), -1, dtype=torch.long)

        for b in range(batch_size):
            for cluster_id in range(s_points.size(1)):
                points_in_cluster = s_points[b, cluster_id, :]
                points_in_cluster = points_in_cluster[points_in_cluster != -1]
                cluster[b, points_in_cluster] = cluster_id

    return cluster

    


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

    `cluster` : torch.Tensor
        Tensor of shape (N,) containing the cluster assignments for each point.
        for example, if cluster[i] = j, then points[i] belongs to the j-th cluster, where j is in the range [0, M).
    """

    if voxel_size is None and voxelgrid_size is None:
        raise ValueError("Either 'voxel_size' or 'voxelgrid_size' must be provided.")
    
    pos = points[:, :3]       # Positions
    features = points[:, 3:]  # Additional features

    if voxel_size is None:
        voxel_size = _compute_grid_size(pos, voxelgrid_size)
        print(f"voxel_size: {voxel_size}")

    # Get cluster assignments; cluster.shape = (N,)
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

    return aggregated_points, cluster


def grid_pooling_batch(points: torch.Tensor, voxel_size:Union[None, torch.Tensor]=None, voxelgrid_size:Union[None, torch.Tensor]=None, feat_mapping: str = 'mean') -> torch.Tensor:
    """
    Aggregates points within each voxel of a grid using a specified function.

    Either `voxel_size` or `voxelgrid_size` must be provided.  

    Parameters
    ----------
    `points` : torch.Tensor
        Tensor of shape ([B], N, 3 + C) where B is the batch size, N is the number of points in the point cloud and C is the number of additional channels.

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
        Tensor of shape ([B], M, 3 + C) where B is the batch size and M is the number of unique voxels.

    `cluster` : torch.Tensor
        Tensor of shape ([B], N) containing the cluster assignments for each point.
        for example, if cluster[b, i] = j, then points[b, i] belongs to the j-th cluster, where j is in the range [0, M).
    """

    if voxel_size is None and voxelgrid_size is None:
        raise ValueError("Either 'voxel_size' or 'voxelgrid_size' must be provided.")
    
    for i in range(points.shape[0]):
        aggregated_points, cluster = grid_pooling(points[i], voxel_size, voxelgrid_size, feat_mapping)
        if i == 0:
            agg_points = aggregated_points.unsqueeze(0)
            clusters = cluster.unsqueeze(0)
        else:
            agg_points = torch.cat([agg_points, aggregated_points.unsqueeze(0)], dim=0)
            clusters = torch.cat([clusters, cluster.unsqueeze(0)], dim=0)

    return agg_points, clusters


if __name__ == '__main__':
    # Generate random points with additional features
    num_points = 200_000
    num_features = 2
    # points = torch.randint(0, 10, (num_points, 3 + num_features)).float()
    points = torch.rand(num_points, 3 + num_features)

    grid_size = torch.tensor([0.1, 0.1, 0.1])
    voxelgrid_size = torch.tensor([64.0, 64.0, 64.0])

    aggregated_points, cluster = grid_pooling(points, grid_size, None, feat_mapping={-1 : 'max', -2:'min'})

    print(f"Original number of points: {points.shape}")
    print(f"Aggregated number of points: {aggregated_points.shape}")
    print(f"Cluster assignments: {cluster.shape}")
    
    s_points = _cluster_to_spoints(cluster)
    print(f"Support points: {s_points.shape}")
    back_to_cluster = spoints_to_cluster(s_points)

    assert torch.equal(cluster, back_to_cluster)

    for i in range(s_points.size(0)):
        cluster_1_indices = torch.where(cluster == i)[0]  # Indices of points in cluster 1
        s_points_1 = s_points[i]
        s_points_1_non_negative = s_points_1[s_points_1 != -1]  # Get valid support points
        
        matching = torch.equal(torch.sort(cluster_1_indices).values, torch.sort(s_points_1_non_negative).values)

        # print(f"Cluster 1 indices: {cluster_1_indices.size()}")
        # print(f"Support points in s_points[1]: {s_points_1_non_negative.size()}")
        # print(f"Do they match? {matching}")
        assert matching

    # Batched grid pooling

    batch_size = 4
    points = torch.rand(batch_size, num_points, 3 + num_features)

    aggregated_points, clusters = grid_pooling_batch(points, grid_size, None, feat_mapping={-1 : 'max', -2:'min'})

    print(f"Original number of points: {points.shape}")
    print(f"Aggregated number of points: {aggregated_points.shape}")
    print(f"Cluster assignments: {clusters.shape}")

    s_points = _cluster_to_spoints(clusters)
    print(f"Support points: {s_points.shape}")

    back_to_cluster = spoints_to_cluster(s_points)
    assert torch.equal(clusters, back_to_cluster)

    for i in range(s_points.size(1)):
        cluster_1_indices = torch.where(clusters == i)[1]
        s_points_1 = s_points[:, i]
        s_points_1_non_negative = s_points_1[s_points_1 != -1]
        matching = torch.equal(torch.sort(cluster_1_indices).values, torch.sort(s_points_1_non_negative).values)
        assert matching