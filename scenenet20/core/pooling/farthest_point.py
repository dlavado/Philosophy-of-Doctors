

import torch

def farthest_point_pooling(points:torch.Tensor, num_points:int) -> torch.Tensor:
    """
    Farthest point pooling - downsample point cloud by iteratively selecting the farthest point from the set of points and removing it.
    
    Parameters
    ----------
    `points` : torch.Tensor
        Tensor of shape (N, 3 + C) where N is the number of points in the point cloud and C is the number of additional features.
    
    `num_points` : int
        Number of points (M) to keep after pooling.
    
    Returns
    -------
    pooled_indices : torch.Tensor
        Tensor of shape (M, 3) where M is the number of points to keep after pooling.
    """
    
    if points.shape[0] == 0:
        return points
    
    # Initialize the list of indices of the pooled points
    pooled_indices = []
    
    # Select the first point randomly
    idx = torch.randint(0, points.shape[0], (1,))
    pooled_indices.append(idx.item())
    
    # Compute the distance of each point to the selected point
    dists = torch.norm(points - points[idx], dim=1)
    
    # Repeat the process until the desired number of points is reached
    while len(pooled_indices) < num_points:
        # Select the farthest point from the set of points
        idx = torch.argmax(dists)
        pooled_indices.append(idx.item())
        
        # Update the distances of the points to the selected point
        dists = torch.min(dists, torch.norm(points - points[idx], dim=1))
    
    return torch.tensor(pooled_indices)


if __name__ == "__main__":
    # Define the point cloud
    points = torch.rand((100_000, 3))
    
    # Perform farthest point pooling
    pooled_indices = farthest_point_pooling(points, 1000)
    
    print(f"Original point cloud shape: {points.shape}")
    print(f"Pooled point cloud shape: {pooled_indices.shape}")