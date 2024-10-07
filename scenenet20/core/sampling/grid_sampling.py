

import torch
import torch_scatter


def grid_sampling(points, grid_size=(0.05, 0.05, 0.05), feat_mapping='max'):
    """
    Grid sampling - downsample point cloud by aggregating points in a voxel grid and
    applying a mapping function to the points in each voxel, such as mean or max.


    Parameters
    ----------
    points - torch.Tensor
        Tensor of shape (N, 3 + C) where N is the number of points in the point cloud and C is the number of additional features.

    grid_size - tuple
        Tuple of length 3 representing the size of the voxels in the x, y, and z dimensions.

    feat_mapping - str
        Mapping function to apply to the points in each voxel. Can be either {'mean', 'max', 'min', 'sum', 'mode'}
    
    Returns
    -------
    aggregated_points - torch.Tensor
        Tensor of shape (M, 3 + C) where M is the number of voxels in the grid.
    """

    assert len(grid_size) == 3, "Grid size must be a tuple of length 3."

    if points.shape[0] == 0:
        return points

    voxel_indices = (points[:, :3] / torch.tensor(grid_size, device=points.device)).long()

    # Unique voxel indices
    _, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
    print(f"inverse_indices: {inverse_indices.shape}")
    
    # Aggregate points within each voxel using torch_scatter.scatter_mean
    aggregated_points = torch_scatter.scatter_mean(points, inverse_indices, dim=0)
    print(f"aggregated_points: {aggregated_points.shape}")

    # for the additional channels, if any, apply the feat_mapping function
    if points.shape[1] > 3:
        if feat_mapping == 'mean':
            aggregated_points[:, 3:] = torch_scatter.scatter_mean(points[:, 3:], inverse_indices, dim=0)
        elif feat_mapping == 'min':
            aggregated_points[:, 3:] = torch_scatter.scatter_min(points[:, 3:], inverse_indices, dim=0)[0]
        elif feat_mapping == 'sum':
            aggregated_points[:, 3:] = torch_scatter.scatter_sum(points[:, 3:], inverse_indices, dim=0)
        elif feat_mapping == 'mode':
            aggs = [points[inverse_indices == i, 3:] for i in range(inverse_indices.max() + 1)]
            modes = torch.stack([torch.mode(agg, dim=0)[0] for agg in aggs])
            aggregated_points[:, 3:] = modes
        else:
            aggregated_points[:, 3:] = torch_scatter.scatter_max(points[:, 3:], inverse_indices, dim=0)[0]

    return aggregated_points


if __name__ == "__main__":
    points = torch.tensor([
        [0.1, 0.1, 0.1, 1.0],
        [0.15, 0.1, 0.1, 1.0],
        [0.2, 0.2, 0.2, 1.0],
        [0.9, 0.9, 0.9, 1.0]
        ], device='cuda')
    
    print(points.shape)

    grid_size = (0.1, 0.1, 0.1)
    sampled_points = grid_sampling(points, grid_size, feat_mapping='mode')
    print(sampled_points)

    ######
    import sys
    sys.path.append('../')
    sys.path.append('../../')
    from utils import constants
    from utils import pointcloud_processing as eda
    from core.datasets.TS40K import TS40K_FULL_Preprocessed, TS40K_FULL


    ts40k = TS40K_FULL_Preprocessed(
        constants.TS40K_FULL_PREPROCESSED_PATH, 
        split='fit', 
        sample_types=['tower_radius', '2_towers'], 
        transform=None, 
        load_into_memory=False
    )

    ts40k = TS40K_FULL(constants.TS40K_FULL_PATH, 
                       split='fit', 
                       sample_types=['tower_radius'], 
                       task='sem_seg', transform=None, 
                       load_into_memory=False)

    sample = ts40k[0]
    points, labels = sample[0].squeeze(), sample[1].squeeze()


    concat = torch.cat([points, labels.reshape(-1, 1)], dim=1) # (N, 3 + C)

    sampled_points = grid_sampling(concat, grid_size, feat_mapping='mode')

    print(points.shape)
    print(sampled_points.shape)
    eda.plot_pointcloud(points.numpy(), labels.numpy(), window_name='Original Point Cloud', use_preset_colors=True)
    eda.plot_pointcloud(sampled_points[:, :3].cpu().numpy(), sampled_points[:, 3].cpu().numpy(), window_name='Sampled Point Cloud', use_preset_colors=True)

