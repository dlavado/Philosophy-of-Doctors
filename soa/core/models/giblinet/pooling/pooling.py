

import torch

@torch.jit.script
def _gather_points(x, indices):
    """
    Specific case of torch.gather for support-point indices.
    
    Parameters
    ----------
    
    x - torch.Tensor
        input tensor of shape ([B], N, F)
        
    indices - torch.Tensor
        indices of shape ([B], M, k) containing M*k support-point indices.
        
    Returns
    -------
    torch.Tensor
        tensor of shape ([B], M, k, F) containing the support-point features.
    """
    
    batched = x.dim() == 3
    if not batched:
        x = x.unsqueeze(0)
        indices = indices.unsqueeze(0)
    
    B, M, K = indices.shape
    F = x.shape[-1]

    mask = indices == -1
    #flat_supports_idxs = indices.reshape(B, -1).to(torch.long) # (B, M*K)
    # indices -1 are ignored
    flat_supports_idxs = torch.where(mask, 0, indices).reshape(B, -1).to(torch.long).contiguous() # (B, M*K)
    #flat_supports_idxs[flat_supports_idxs == -1] = 0
    support_points = torch.gather(x, 1, flat_supports_idxs.unsqueeze(-1).expand(-1, -1, F)).to(x.dtype) # (B, M*K, F)

    # Reshape back to (B, M, K, F)
    support_points = support_points.reshape(B, M, K, F).contiguous()
    support_points[mask] = 0 # features of non-existing neighbors are set to 0
    
    if not batched:
        support_points = support_points.squeeze(0)
        
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