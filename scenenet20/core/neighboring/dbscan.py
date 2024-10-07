

import torch

def dbscan_cluster(q_points: torch.Tensor, s_points: torch.Tensor, eps: float, min_points: int, k: int):
    """
    Clusters points using DBSCAN algorithm in PyTorch and returns the k nearest neighbors.

    Parameters
    ----------
    q_points : torch.Tensor
        Query points of shape (B, N, 3), where B is the batch size, N is the number of query points;
    s_points : torch.Tensor
        Support points of shape (B, M, 3), where B is the batch size, M is the number of support points;
    eps : float
        Maximum distance between two points for them to be considered neighbors.
    min_points : int
        Minimum number of points required to form a dense region (core point).
    k : int
        The maximum number of neighbors to return for each query point.

    Returns
    -------
    clusters : torch.Tensor
        Tensor of shape (B, N, k) representing the indices of the `k` nearest neighbors of each query point. If a query point cannot form a neighborhood, the row will be filled with -1s.
    """
    
    B, N, C = q_points.shape
    _, M, _ = s_points.shape

    # Initialize tensor for clusters (neighborhood indices) and noise
    clusters = torch.full((B, N, k), -1, dtype=torch.long)  # -1 will denote no neighbors found

    # Compute pairwise distances between query points and support points
    dist_matrix = torch.cdist(q_points, s_points)  # Shape (B, N, M)
    
    for b in range(B):  # Iterate over batch
        for i in range(N):  # Iterate over query points
            # Find neighbors within epsilon distance
            neighbors = torch.where(dist_matrix[b, i] <= eps)[0]  # Get neighbor indices

            if neighbors.numel() < min_points:
                # Not enough neighbors to form a cluster, leave the row as -1s
                continue

            # If neighbors are found, fill up to k neighbors
            selected_neighbors = neighbors[:k]  # Select up to k neighbors
            clusters[b, i, :selected_neighbors.numel()] = selected_neighbors

    return clusters


def find_nearest_neighbors(q_points: torch.Tensor, s_points: torch.Tensor, k: int):
    """
    Finds the k-nearest neighbors to each query point using a strategy complementary to farthest point sampling.

    Parameters
    ----------
    q_points : torch.Tensor
        Query points of shape (B, N, C), where B is the batch size, N is the number of query points, and C is the number of dimensions.
    s_points : torch.Tensor
        Support points of shape (B, M, C), where B is the batch size, M is the number of support points, and C is the number of dimensions.
    k : int
        The maximum number of neighbors to find for each query point.

    Returns
    -------
    neighbors : torch.Tensor
        Tensor of shape (B, N, k) representing the indices of the k nearest neighbors for each query point.
    """
    
    B, N, C = q_points.shape
    _, M, _ = s_points.shape

    # Initialize tensor for neighbors
    neighbors = torch.full((B, N, k), -1, dtype=torch.long)

    # Compute pairwise distances between query points and support points
    # dist_matrix = torch.cdist(q_points, s_points)  # Shape (B, N, M)
    
    for b in range(B):  # Iterate over batches
        for i in range(N):  # Iterate over query points
            # List to store the set of neighbors (starting with the closest point)
            selected_neighbors = []

            # Calculate initial distances between the query point and all support points
            q_point = q_points[b, i].unsqueeze(0)  # Shape (1, C)
            dist_matrix = torch.cdist(q_point, s_points[b]).squeeze(0)  # Shape (M,)

            # Find the closest point initially and store its index
            first_neighbor = torch.argmin(dist_matrix)
            selected_neighbors.append(first_neighbor.item())

            # Maintain the running minimum distance from the selected set to all support points
            min_dists = dist_matrix.clone()  # Shape (M,)

            for _ in range(1, k):
                # Get distances from the latest selected neighbor to all points
                latest_neighbor = selected_neighbors[-1]
                new_dists = torch.cdist(s_points[b, latest_neighbor].unsqueeze(0), s_points[b]).squeeze(0)  # Shape (M,)
                
                # Update the running minimum distance for all points
                min_dists = torch.minimum(min_dists, new_dists)

                # Mask the selected neighbors to avoid re-selecting them
                min_dists[selected_neighbors] = float('inf')
                
                # Select the next closest point to the current selected set
                next_neighbor = torch.argmin(min_dists)
                selected_neighbors.append(next_neighbor.item())


            # Save the selected neighbors (ensuring k neighbors)
            neighbors[b, i, :] = torch.tensor(selected_neighbors[:k])

    return neighbors



if __name__ == '__main__':
    import sys
    sys.path.append('../')
    sys.path.append('../../')
    from utils import constants
    from utils import pointcloud_processing as eda
    from core.datasets.TS40K import TS40K_FULL_Preprocessed, TS40K_FULL
    from core.sampling.FPS import Farthest_Point_Sampling




    ts40k = TS40K_FULL_Preprocessed(
        constants.TS40K_FULL_PREPROCESSED_PATH, 
        split='fit', 
        sample_types=['tower_radius', '2_towers'], 
        transform=None, 
        load_into_memory=False
    )

    NUM_Q_POINTS = 5000
    fps = Farthest_Point_Sampling(NUM_Q_POINTS)


    sample = ts40k[0]
    points, labels = sample[0], sample[1]

    query_points, q_labels = fps(torch.concat([points, labels.reshape(-1, 1)], dim=1))
    print(query_points.shape, q_labels.shape)
    print(torch.unique(labels))
    query_points

    eps = 0.1
    min_points = 10
    k = 100

    # clusters = dbscan_cluster(query_points.unsqueeze(0), points.unsqueeze(0), eps, min_points, k)

    clusters = find_nearest_neighbors(query_points.unsqueeze(0), points.unsqueeze(0), k) # (B, NUM_Q_POINTS, k)

    print(clusters.shape)

    eda.plot_pointcloud(points.numpy(), labels.numpy(), window_name='Original Point Cloud', use_preset_colors=True)

    # plot a few clusters
    for i in range(10):
        i = torch.randint(0, clusters.shape[1], (1,)).item()
        cluster = clusters[0, i].numpy()
        cluster = cluster[cluster != -1]
        print(cluster)
        if len(cluster) == 0:
            continue
        eda.plot_pointcloud(points[cluster].numpy(), labels[cluster].numpy(), window_name=f'Cluster {i}', use_preset_colors=True)



    









