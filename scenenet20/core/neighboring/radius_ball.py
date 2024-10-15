

import torch
import torch_cluster

import sys
sys.path.append('../')
sys.path.append('../../')

from core.neighboring.conversions import batch_to_pack, lengths_to_batchvector, pack_to_batch
from core.neighboring.knn import torch_knn


class RadiusBall_Neighboring:

    def __init__(self, radius, k, loop=True, pad_value=-1) -> None:
        self.radius = radius
        self.k = k
        self.loop = loop
        self.pad_value = pad_value

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
        graph - torch.Tensor
            support points of shape (B, Q, k)
        """
        return k_radius_ball(q_points, support, self.radius, self.k, self.loop, self.pad_value)
        


def naive_radius_search(q_points, s_points, radius, neighbor_limit):
    """Radius search in naive implementation.

    Args:
        q_points (Tensor): query points (N, 3).
        s_points (Tensor): support points (M, 3).
        radius (float): radius radius.
        neighbor_limit (int): neighbor radius.

    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to M if not exist.
    """
    knn_distances, knn_indices = torch_knn(q_points, s_points, neighbor_limit)
    knn_indices = knn_indices.squeeze(0)
    knn_distances = knn_distances.squeeze(0)
    knn_masks = torch.gt(knn_distances, radius)
    knn_indices.masked_fill_(knn_masks, s_points.shape[0])
    return knn_indices


def radius_search(q_points, s_points, radius, neighbor_limit, batch_q=None, batch_s=None, loop=False):
    """
    Build Radius Graph for each query point in the query set.

    Parameters
    ----------
    q_points : Tensor
        Query points (N, 3).

    s_points : Tensor
        Support points (M, 3).

    radius : float
        Ball radius.

    neighbor_limit : int
        Maximum number of neighbors.

    batch_q : Tensor, optional
        Batch indices for query points (N).

    batch_s : Tensor, optional
        Batch indices for support points (M).

    loop : bool, optional
        If False, query points can't be neighbors of themselves.

    Returns
    -------

    pairs : Tensor
        Pairs of indices (2, N*neighbor_limit); the num_columns is not fixed and can be less than N*neighbor_limit if for a q_point there are less than neighbor_limit neighbors. 
        The first row contains the indices of the support points.
        The second row contains the indices of the query points.
    """

    q, s = torch_cluster.radius(s_points, q_points, radius, batch_s, batch_q, max_num_neighbors=neighbor_limit, num_workers=12)

    if not loop:
        mask = s != q
        s, q = s[mask], q[mask]
    
    pairs = torch.stack([s, q], dim=0) # (2, N*num_neighbors)
    return pairs


def pairs_to_tensor(pairs:torch.Tensor, pad_value=-1, num_points=None):
    """
    Convert pairs of indices to a tensor.

    Parameters
    ----------
    `pairs` : Tensor
        Pairs of indices in pack_mode (2, B\*N\*neighbor_limit).

    `pad_value` : int, optional
        Value to pad the tensor when the number of neighbors is less than neighbor_limit.

    `num_points` : int, optional
        Number of query points. 
        If None, it is inferred from the pairs tensor. However, functionality is not guaranteed.

    Returns
    -------
    `graph` : Tensor
        Tensor of shape (B\*N, neighbor_limit) with the indices of the neighbors in pack_mode.
    """
    # num of neighbors for each query point
    q_points, counts = torch.unique(pairs[1], return_counts=True) # (N,), 
    neighbor_limit = counts.max().item()

    if num_points is None:
        num_points = q_points.shape[0]
    else:
        q_points = torch.arange(num_points, device=pairs.device) # some query points may not have neighbors

    graph = torch.full((num_points, neighbor_limit), pad_value, dtype=torch.long, device=pairs.device)

    masks = pairs[1].unsqueeze(1) == q_points.unsqueeze(0) # (N, B*N*neighbor_limit)
    neighbors = torch.where(masks)[0] # (B*N*neighbor_limit)

    num_neighbors = masks.sum(dim=0)  # (B*N*neighbor_limit)

    # tensor with the possible indices of the neighbors
    index_tensor = torch.arange(neighbor_limit, device=pairs.device).unsqueeze(0).expand(-1, neighbor_limit) # (N, neighbor_limit)

    # this masks the indices of the neighbors that are not valid
    mask = index_tensor < num_neighbors.unsqueeze(1)
    graph[mask] = neighbors

    return graph


def k_radius_ball(q_points:torch.Tensor, s_points:torch.Tensor, radius:float, neighbor_limit:int, loop=False, pad_value=-1):
    """
    Build Radius Graph for each query point in the query set.

    Parameters
    ----------
    q_points : Tensor
        Query points ([B], N, 3).

    s_points : Tensor
        Support points ([B], M, 3).

    radius : float
        Ball radius.

    neighbor_limit : int
        Maximum number of neighbors.

    loop : bool, optional
        If False, query points can't be neighbors of themselves.

    pad_value : int, optional
        Value to pad the tensor when the number of neighbors is less than neighbor_limit.

    Returns
    -------
    graph : Tensor
        Tensor of shape (B, N, neighbor_limit) with the indices of the neighbors.
    """
    
    if q_points.dim() == 2:
        q_points = q_points.unsqueeze(0)
        s_points = s_points.unsqueeze(0)
        keepdim = True
    else:
        keepdim = False

    num_points = q_points.shape[1] # N
    batch_size = q_points.shape[0] # B

    q_points, q_lengths = batch_to_pack(q_points) # (N*B, 3), (B)
    s_points, s_lengths = batch_to_pack(s_points) # (M*B, 3), (B)
    q_batch_vector = lengths_to_batchvector(q_lengths) # (N), batch_vector is the indices of the items in the batch
    s_batch_vector = lengths_to_batchvector(s_lengths) # (M)

    pairs = radius_search(q_points, s_points, radius, neighbor_limit, q_batch_vector, s_batch_vector, loop) # (2, B*N*neighbor_limit)
    print(pairs.shape)
    graph = pairs_to_tensor(pairs, pad_value, num_points=num_points*batch_size) # (N*B, neighbor_limit)
    graph, _ = pack_to_batch(graph, q_lengths) # (B, N, neighbor_limit)
     
    if keepdim:
        graph = graph.squeeze(0)

    return graph


if __name__ == "__main__":
    # disable warnings
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Example usage with a larger point cloud
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],
        [0.15, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.25, 0.25, 0.25],
        [0.3, 0.3, 0.3],
        [0.35, 0.35, 0.35],
        [0.4, 0.4, 0.4],
        [0.45, 0.45, 0.45],
        [0.5, 0.5, 0.5],
        [0.55, 0.55, 0.55],
        [0.6, 0.6, 0.6],
        [0.65, 0.65, 0.65],
        [0.7, 0.7, 0.7],
        [0.75, 0.75, 0.75],
        [0.8, 0.8, 0.8],
        [0.85, 0.85, 0.85],
        [0.9, 0.9, 0.9],
        [0.95, 0.95, 0.95],
        [1.0, 1.0, 1.0]
    ], device='cuda')

    query_points = torch.tensor([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
        [0.5, 0.5, 0.5],
        [0.6, 0.6, 0.6],
        [0.7, 0.7, 0.7],
        [0.8, 0.8, 0.8],
        [0.9, 0.9, 0.9],
        [1.0, 1.0, 1.0]
    ], device='cuda')

    points = points.reshape(2, 10, 3)
    query_points = query_points.reshape(2, 5, 3)
    radius = 0.2
    neighbor_limit = 5

    graph = k_radius_ball(query_points, points, radius, neighbor_limit, loop=True)

    print(graph.shape)
    print(graph)

    input("Press any key to continue...")

    points, p_lengths = batch_to_pack(points)
    query_points, q_lengths = batch_to_pack(query_points)

    q_batch_vector = lengths_to_batchvector(q_lengths)
    p_batch_vector = lengths_to_batchvector(p_lengths)
    print(q_batch_vector)

   
    graph = radius_search(query_points, points, radius, neighbor_limit, q_batch_vector, p_batch_vector, loop=True)

    graph = pairs_to_tensor(graph)

    print(graph.shape)
    print(graph)

    input("Press any key to continue...")


    ######
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

    num_points = 5000
    fps = Farthest_Point_Sampling(num_points)


    sample = ts40k[0]
    points, labels = sample[0], sample[1]


    concat = torch.cat([points, labels.reshape(-1, 1)], dim=1) # (N, 3 + C)

    print(concat.shape)

    query_points, labels = fps(concat.unsqueeze(0))

    print(query_points.shape, labels.shape)
    

    radius = 0.2
    neighbor_limit = 16

    graph = radius_search(query_points[:, :3], concat[:, :3], radius, neighbor_limit, loop=True)

    print(graph.shape)

    graph = pairs_to_tensor(graph)

    
    print(graph)