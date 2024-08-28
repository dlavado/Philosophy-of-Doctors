import torch
from torch import Tensor
from typing import Tuple

def torch_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """
    kNN without using KeOps.

    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)

    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """

    num_batch_dims = q_points.dim() - 2
   
    dij = torch.cdist(s_points, q_points, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")

    # Find k nearest neighbors
    try:
        knn = torch.topk(dij, k, dim=num_batch_dims, largest=False, sorted=True) # tuple with (values, indices), both of shape (*, k, N)
    except Exception as e:
        # print("q_points.shape: ", q_points.shape, "s_points.shape: ", s_points.shape)
        # print("dij.shape: ", dij.shape)
        # print("k: ", k)
        raise e
    
    # from knn I want the select the N closest points to each query point
    # knn.values.shape: (*, N, M)

    # print("")
    # print(f"{k =}, {q_points.shape =}, {s_points.shape =}")
    # print(f"{dij.shape =}")
    # print("knn.shape: ", knn.values.shape, "knn.indices.shape: ", knn.indices.shape)

    return knn.values.permute(0, 2, 1), knn.indices.permute(0, 2, 1) # (*, N, k), (*, N, k)