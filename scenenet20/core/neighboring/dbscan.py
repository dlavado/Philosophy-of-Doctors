

import torch
from pykeops.torch import LazyTensor

class DBSCAN_Neighboring:

    def __init__(self, eps, min_points, k) -> None:
        self.eps = eps
        self.min_points = min_points
        self.k = k


    def __call__(self, q_points:torch.Tensor, support:torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        q_points - torch.Tensor
            query points of shape (B, Q, 3)

        support - torch.Tensor
            input tensor of shape (B, N, 3)

        Returns
        -------
        s_points - torch.Tensor
            support points of shape (B, Q, k)
        """
        neighbors = dbscan_cluster(q_points, support, self.eps, self.min_points, self.k)
        return neighbors


def dbscan_cluster(q_points: torch.Tensor, s_points: torch.Tensor, eps: float, min_points: int, k: int):
    """
    Clusters points using DBSCAN algorithm in PyTorch

    Parameters
    ----------
    `q_points` : torch.Tensor
        Query points of shape (B, N, 3), where B is the batch size, N is the number of query points;
    `s_points` : torch.Tensor
        Support points of shape (B, M, 3), where B is the batch size, M is the number of support points;
    `eps` : float
        Maximum distance between two points for them to be considered neighbors.
    `min_points` : int
        Minimum number of points required to form a dense region (core point).
    `k` : int
        The maximum number of neighbors to return for each query point.

    Returns
    -------
    clusters : torch.Tensor
        Tensor of shape (B, N, k) representing the indices of the `k` nearest neighbors of each query point. If a query point cannot form a neighborhood, the row will be filled with -1s.
    """
    
    B, N, C = q_points.shape
    _, M, _ = s_points.shape
    
    MAX_ITERS = 5

    # Initialize tensor for clusters (neighborhood indices) and noise
    clusters = torch.full((B, N, k), -1, dtype=torch.long)  # -1 will denote no neighbors found

    # Compute pairwise distances between query points and support points
    dist_matrix = torch.cdist(q_points, s_points)  # Shape (B, N, M)
    support_dist = torch.cdist(s_points, s_points) # Shape (B, M, M)
    print(f"support_dist: {support_dist.shape}")    
    
    for b in range(B):  # Iterate over batch
        for i in range(N):  # Iterate over query points
            # Find neighbors within epsilon distance
            neighbors = torch.nonzero(dist_matrix[b, i] <= eps).squeeze()  # Get neighbor indices
            print(f"neighbors {i}: {neighbors.shape}")
            iters = 0
            visited = torch.zeros(M, dtype=torch.bool)

            if not neighbors.size():  # If no neighbors found
                continue
            
            while 0 < neighbors.numel() < k and iters < MAX_ITERS:
                unvisited_neighbors = neighbors[~visited[neighbors]]
                print(f"cluster {i} --- unvisited neighbors: {unvisited_neighbors.shape}")
                expanded_neighbors = torch.where(support_dist[b, unvisited_neighbors] <= eps)[1]
                print(f"cluster {i} --- expanded neighbors: {expanded_neighbors.shape}")
                if expanded_neighbors.numel() == 0:
                    break
                neighbors = torch.cat((neighbors, expanded_neighbors)).unique()
                iters += 1    
                visited[neighbors] = True
                
            print(f"cluster {i} --- final neighbors: {neighbors.shape}")
                
            if neighbors.numel() < min_points:  # If not enough neighbors to form a cluster
                continue

            selected_neighbors = neighbors[:k]
            clusters[b, i, :selected_neighbors.numel()] = selected_neighbors

    return clusters



def keops_dbscan_cluster(q_points: torch.Tensor, s_points: torch.Tensor, eps: float, min_points: int, k: int):
    """
    Clusters points using an epsilon radius with KeOps to find `k` nearest neighbors, expanding as necessary.

    Parameters
    ----------
    q_points : torch.Tensor
        Query points of shape (B, N, C).
    s_points : torch.Tensor
        Support points of shape (B, M, C).
    eps : float
        Maximum distance between two points for them to be considered neighbors.
    min_points : int
        Minimum number of points required to form a cluster.
    k : int
        Number of nearest neighbors to retrieve for each query point.

    Returns
    -------
    clusters : torch.Tensor
        Tensor of shape (B, N, k) representing the indices of the `k` nearest neighbors for each query point.
        If fewer than `k` neighbors are found within eps distance, the row will be padded with -1s.
    """
    # Initialization
    B, N, _ = q_points.shape
    clusters = torch.full((B, N, k), -1, dtype=torch.long, device=q_points.device)

    # Compute pairwise distances in KeOps and expand neighbor lists
    for b in range(B):
        q_batch = LazyTensor(q_points[b].unsqueeze(-2)[..., :3])  # (N, 1, 3); xyz
        s_batch = LazyTensor(s_points[b].unsqueeze(-3)[..., :3])  # (1, M, 3); xyz
        
        dist_matrix = (q_batch - s_batch).norm2()  # (N, M), squared distances
        dist_to_eps = (dist_matrix - eps**2).relu() # (N, M), if dist_matrix <= eps set to zero, else set to dist_matrix
        
        
        for i in range(N):
            # Find neighbors within epsilon distance
            eps_neighbors = dist_to_eps.argKmin(k, dim=0) # Get neighbor indices 
            
            print(eps_neighbors.shape)
            if eps_neighbors.numel() < min_points: # no points to form a cluster
                continue
            
            # Expand neighbors until at least `k` neighbors are gathered
            while eps_neighbors.numel() < k:
                expanded_neighbors = dist_to_eps[eps_neighbors].argKmin(k - eps_neighbors.numel(), dim=0)

                # Concatenate and unique
                eps_neighbors = torch.cat((eps_neighbors, expanded_neighbors)).unique()

            # Update clusters with indices of neighbors
            print(eps_neighbors.shape) # shape = (k)
            clusters[b, i, :k] = eps_neighbors[:k]

    return clusters


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
    
    ts40k = TS40K_FULL(
        constants.TS40K_FULL_PATH, 
        split='fit', 
        sample_types=['tower_radius', '2_towers'], 
        transform=None, 
        load_into_memory=False
    )

    NUM_Q_POINTS = 5000
    fps = Farthest_Point_Sampling(NUM_Q_POINTS)


    sample = ts40k[10]
    points, labels = sample[0], sample[1]
    
    print(points.shape, labels.shape)
    eda.plot_pointcloud(points.cpu().numpy()[0], classes=labels.cpu().long().numpy()[0], window_name='Original Point Cloud', use_preset_colors=True)


    query_points = fps(torch.concat([points, labels.unsqueeze(-1)], dim=-1).squeeze(0))
    query_points, q_labels = query_points[..., :-1], query_points[..., -1]
    print(query_points.shape, q_labels.shape)
    print(torch.unique(labels))
    
    eps = 0.03
    min_points = 10
    k = 200
    
    
    clusters = dbscan_cluster(query_points.unsqueeze(0), points.unsqueeze(0), eps, min_points, k)
    
    # clusters = keops_dbscan_cluster(query_points.unsqueeze(0), points.unsqueeze(0), eps, min_points, k)

    
    print(clusters.shape)

    # plot a few clusters
    for i in range(1000):
        i = torch.randint(0, clusters.shape[1], (1,)).item()
        cluster = clusters[0, i].numpy()
        cluster = cluster[cluster != -1]
        if 5 not in labels[cluster].unique():
            continue
        print(f"number of dbscan neighbors: {len(cluster)}")
        # print(cluster)
        if len(cluster) == 0:
            continue
        eda.plot_pointcloud(points[cluster].numpy(), labels[cluster].numpy(), window_name=f'Cluster {i}', use_preset_colors=True)



    









