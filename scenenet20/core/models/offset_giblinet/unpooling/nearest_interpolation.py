import torch
# import torch.nn.functional as F
# from torch.autograd import Function

@torch.jit.script
def interpolation(curr_points:torch.Tensor, skip_points:torch.Tensor, upsampling_idxs:torch.Tensor):
    """
    Performs interpolation from current points to skip points using neighbor indices.
    
    Args:
        curr_points: Tensor of shape (B, M, 3 + C), representing input coordinates and features of the current layer.
        skip_points: Tensor of shape (B, N, 3 + C), representing input coordinates and features of the skip connection. (only the coords are used)
        upsampling_idxs: Tensor of shape (B, N, K), representing indices of curr_points that are neighbors of skip_points.
    
    Returns:
        new_feat: Interpolated features at the skip_points (B, N, C).
    """
    B, M, _ = curr_points.shape
    B, N, K = upsampling_idxs.shape
        
    mask = upsampling_idxs == -1
    upsampling_idxs[mask] = 0 # -1 is used to ignore non-existing neighbors, but when gathering, it should be 0
    
    neighbors = torch.gather(
        curr_points.unsqueeze(1).expand(B, N, M, curr_points.shape[-1]), 
        2, upsampling_idxs.to(dtype=torch.long).unsqueeze(-1).expand(B, N, K, curr_points.shape[-1])
    ).contiguous().to(curr_points.dtype)  # (B, N, K, 3 + C)
    
    # print(f"{neighbors.dtype=} {skip_points.dtype=} {curr_points.dtype=}")
    
    # mask non-existing neighbors
    neighbors[mask.unsqueeze(-1).expand(B, N, K, curr_points.shape[-1])] = 0

    # Compute distances between skip points and their neighbors
    dist = torch.norm(skip_points[..., :3].unsqueeze(2) - neighbors[..., :3], dim=-1)  # (B, N, K)
    dist = 1.0 / (dist + 1e-8)  # Avoid division by zero   
    weight = dist / torch.sum(dist, dim=-1, keepdim=True)  # (B, N, K)
    # Compute the weighted sum of neighbor features
    new_feat = torch.sum(neighbors[..., 3:] * weight.unsqueeze(-1), dim=2, dtype=curr_points.dtype)  # (B, N, C)
    
    return new_feat # (B, N, C)



def interpolation2(curr_points:torch.Tensor, skip_points:torch.Tensor, c_offset, s_offset, k:int=3) -> torch.Tensor:
    
    from pointops import interpolation2
    c_p, c_f = curr_points[..., :3], curr_points[..., 3:]
    new_feat = interpolation2(c_p, skip_points, c_f, c_offset, s_offset, k)
    
    return new_feat


