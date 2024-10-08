import torch
from torch import Tensor
from typing import Tuple


class KNN_Neighboring:
    def __init__(self, k:int) -> None:
        self.k = k

    def __call__(self, x:Tensor, q_points:Tensor) -> torch.Tensor:
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
        
        return torch_knn(q_points, x, self.k)[1]


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

