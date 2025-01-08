import torch
from torch import Tensor
from typing import Tuple


class KNN_Neighboring:
    def __init__(self, k:int) -> None:
        self.k = k

    def __call__(self, q_points:Tensor, x:Tensor) -> torch.Tensor:
        """
        Parameters
        ----------

        x - torch.Tensor
            input tensor of shape (B, N, 3)

        q_points - torch.Tensor
            query points of shape (B, Q, 3)

        Returns
        -------
        s_points - torch.Tensor
            support points of shape (B, Q, k)
        """
        
        # return torch_knn(q_points, x, self.k)[1]
        return keops_knn(q_points.contiguous(), x.contiguous(), self.k)[1]


def torch_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """
    kNN without using KeOps.

    Parameters
    ----------
    `q_points` - torch.Tensor
        query points of shape (B, N, C)
    `s_points` - torch.Tensor
        support points of shape (B, M, C)
    `k` - int
        number of neighbors to consider

    Returns
    -------
    knn_distance - torch.Tensor
        distances to the k nearest neighbors of shape (B, N, k)
    knn_indices - torch.Tensor
        indices in `s_points` of the k nearest neighbors of shape (B, N, k)
    """
    keepdim = False
    if q_points.dim() == 2:
        q_points = q_points.unsqueeze(0)
        s_points = s_points.unsqueeze(0)
        keepdim = True

    num_batch_dims = q_points.dim() - 2
   
    dij = torch.cdist(s_points, q_points, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")

    # Find k nearest neighbors
    try:
        knn = torch.topk(dij, k, dim=num_batch_dims, largest=False, sorted=True) # tuple with (values, indices), both of shape (*, k, N)
    except Exception as e:
        print(f"Error: {e}")
        raise e
    
    dists = knn.values.permute(0, 2, 1) # (*, N, k)
    indices = knn.indices.permute(0, 2, 1) # (*, N, k)

    if keepdim:
        return dists.squeeze(0), indices.squeeze(0)

    return dists, indices


def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """kNN with PyKeOps.

    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)

    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    from pykeops.torch import LazyTensor
    num_batch_dims = q_points.dim() - 2
    xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=num_batch_dims + 1)  # (*, N, K)
    return knn_distances, knn_indices





if __name__ == '__main__':
    
    import pointops as pops
    
    
    # Define the point cloud
    points = torch.rand((100_000, 3)).cuda()    
    # offset = torch.tensor([points.size(0)]).cuda()
    # offset = torch.tensor([1000]*100).cuda() # batched offset;
    batch = torch.cat([torch.full((1000,), i, device=points.device) for i in range(100)], dim=0)
    offset = pops.batch2offset(batch)
    feats  = torch.rand((100_000, 1)).cuda()
    print(f"{points.shape=} {offset.shape=} {feats.shape=}")
    
    q_points = torch.rand((1000, 3)).cuda()
    q_batch = torch.cat([torch.full((10,), i, device=points.device) for i in range(100)], dim=0)
    q_offset = pops.batch2offset(q_batch)
    print(f"{q_points.shape=} {q_offset.shape=}")
    
    k = 100
    
    # Perform kNN
    ref_idx, _ = pops.knn_query(k, points, offset, q_points, q_offset)
    
    print(f"{ref_idx.shape=} {points[ref_idx].shape=}")
    
    # knn and groups with features?
    pts, idx = pops.knn_query_and_group(feats, points, offset, q_points, q_offset, nsample=k, with_xyz=False)
    
    print(f"{pts.shape=} {idx.shape=}")
    
    
    #######
    
    fps_idxs = pops.farthest_point_sampling(points, offset, q_offset)
    print(f"{fps_idxs.shape=} {points[fps_idxs].shape=}")
    
    # select the 1st pointcloud
    fps_idxs = fps_idxs[q_batch == 0]
    print(f"{fps_idxs.shape=} {points[fps_idxs].shape=}")
    
    ########
    
    
    
    
    
