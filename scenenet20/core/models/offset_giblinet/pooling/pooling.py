

import torch

@torch.jit.script
def _gather_points(x, indices):
    """
    Specific case of torch.gather for support-point indices.
    
    Parameters
    ----------
    
    x - torch.Tensor
        input tensor of shape (N, F)
        
    indices - torch.Tensor
        indices of shape (M, k) containing M*k support-point indices.
        
    Returns
    -------
    torch.Tensor
        tensor of shape (M, k, F) containing the support-point features.
    """
    mask = indices == -1
   
    indices = torch.where(mask, torch.zeros_like(indices), indices)
    support_points = x[indices]
    support_points[mask] = 0 # features of non-existing neighbors are set to 0
        
    return support_points

@torch.jit.script
def local_pooling(feats, neighbor_idxs):
    """
    Max pooling from neighbors

    Parameters
    ----------

    `feats` : torch.Tensor
        Tensor of shape ([B], N, C) representing the input features.

    `neighbor_idxs` : torch.Tensor
        Tensor of shape ([B], M, k) containing the indices of the k neighbors of each point and M is the number of query points (M <= N).

    Returns
    -------
    `pooled_feats` : torch.Tensor
        Tensor of shape ([B], M, C) representing the pooled features.
    """
    pool_dim = 2 if feats.dim() == 3 else 1
    neighbor_feats = _gather_points(feats, neighbor_idxs)  # shape: ([B], M, k, C)
    # print(f"{neighbor_feats.shape=}")
    pooled_feats = neighbor_feats.max(dim=pool_dim)[0]

    return pooled_feats




def naive_local_pooling(feats, neighbor_idxs, feat_mapping='max'):
    """
    Max pooling from neighbors

    Parameters
    ----------

    `feats` : torch.Tensor
        Tensor of shape ([B], N, C) representing the input features.

    `neighbor_idxs` : torch.Tensor
        Tensor of shape ([B], M, k) containing the indices of the k neighbors of each point and M is the number of query points (M <= N).

    `feat_mapping` : str
        Feature mapping strategy. One of ['max', 'min', 'mean', 'sum'].

    Returns
    -------
    `pooled_feats` : torch.Tensor
        Tensor of shape ([B], M, C) representing the pooled features.
    """
    pool_dim = 2 if feats.dim() == 3 else 1
    neighbor_feats = _gather_points(feats, neighbor_idxs)  # shape: ([B], M, k, C)
    # print(f"{neighbor_feats.shape=}")
    
    # pooled_feats shape: ([B], M, C)
    if feat_mapping == 'max':
        pooled_feats = neighbor_feats.max(dim=pool_dim)[0]
    elif feat_mapping == 'min':
        pooled_feats = neighbor_feats.min(dim=pool_dim)[0]
    elif feat_mapping == 'mean':
        pooled_feats = neighbor_feats.mean(dim=pool_dim)
    elif feat_mapping == 'sum':
        pooled_feats = neighbor_feats.sum(dim=pool_dim)

    return pooled_feats