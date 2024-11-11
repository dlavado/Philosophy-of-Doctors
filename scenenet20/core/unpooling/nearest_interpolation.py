import torch
import torch.nn.functional as F
from torch.autograd import Function

def interpolation(curr_points, skip_points, upsampling_idxs):
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
    C = curr_points.shape[-1] - 3  # Number of features (excluding coordinates)
    
    # Extract the coordinates and features from the input points
    curr_coords, curr_feat = curr_points[..., :3], curr_points[..., 3:]  # (B, M, 3), (B, M, C)
    skip_coords = skip_points[..., :3]  # (B, N, 3)
    
    mask = upsampling_idxs == -1
    upsampling_idxs[mask] = 0

    # Gather neighbor features using the upsampling_idxs
    neighbor_feat = torch.gather(
        curr_feat.unsqueeze(1).expand(B, N, M, C),  # Expand current features to (B, N, M, C)
        2, upsampling_idxs.unsqueeze(-1).expand(B, N, K, C)  # Gather features (B, N, K, C)
    )  # (B, N, K, C)
    
    neighbor_feat[mask.unsqueeze(-1).expand(B, N, K, C)] = 0

    # Compute distances between skip points and their neighbors
    neighbor_coords = torch.gather(
        curr_coords.unsqueeze(1).expand(B, N, M, 3),  # Expand current coords to (B, N, M, 3)
        2, upsampling_idxs.unsqueeze(-1).expand(B, N, K, 3)  # Gather coords (B, N, K, 3)
    )  # (B, N, K, 3)
    
    neighbor_coords[mask.unsqueeze(-1).expand(B, N, K, 3)] = 0
    
    # print("\n\n\n")
    # print(f"{neighbor_coords.shape=} {neighbor_feat.shape=} {skip_coords.shape=}")

    dist = torch.norm(skip_coords.unsqueeze(2) - neighbor_coords, dim=-1)  # (B, N, K)
    dist_recip = 1.0 / (dist + 1e-8)  # Avoid division by zero
    # if torch.isnan(dist_recip).any():
    #     print("interpolation")
    #     print(f"{dist_recip=}")
    norm = torch.sum(dist_recip, dim=-1, keepdim=True)  # (B, N, 1)
    weight = dist_recip / norm  # (B, N, K)
    # if torch.isnan(weight).any():
    #     print("interpolation")
    #     print(f"{weight=}")

    # Compute the weighted sum of neighbor features
    new_feat = torch.sum(neighbor_feat * weight.unsqueeze(-1), dim=2)  # (B, N, C)

    return new_feat # (B, N, C)


# class Interpolation(Function):
#     @staticmethod
#     def forward(ctx, curr_points, skip_points, upsampling_idxs):
#         """
#         Forward pass for interpolation.
#         Args:
#             curr_points: Tensor of shape (B, M, 3 + C), representing input coordinates and features of the current layer.
#             skip_points: Tensor of shape (B, N, 3 + C), representing input coordinates and features of the skip connection.
#             upsampling_idxs: Tensor of shape (B, N, K), representing the indices of the curr_points that are neighbors of the skip_points.
#         Returns:
#             new_feat: Interpolated features at the skip_points (B, N, C).
#         """
#         B, M, _ = curr_points.shape
#         B, N, K = upsampling_idxs.shape
#         C = curr_points.shape[-1] - 3  # Number of features (excluding coordinates)

#         # Extract the coordinates and features from the input points
#         curr_coords, curr_feat = curr_points[:, :, :3], curr_points[:, :, 3:]  # (B, M, 3), (B, M, C)
#         skip_coords, skip_feat = skip_points[:, :, :3], skip_points[:, :, 3:]  # (B, N, 3), (B, N, C)

#         # Gather neighbor features using the upsampling_idxs
#         neighbor_feat = torch.gather(
#             curr_feat.unsqueeze(1).expand(B, N, M, C),  # Expand current features to (B, N, M, C)
#             2, upsampling_idxs.unsqueeze(-1).expand(B, N, K, C)  # Gather features (B, N, K, C)
#         )  # (B, N, K, C)

#         # Compute distances between skip points and their neighbors
#         neighbor_coords = torch.gather(
#             curr_coords.unsqueeze(1).expand(B, N, M, 3),  # Expand current coords to (B, N, M, 3)
#             2, upsampling_idxs.unsqueeze(-1).expand(B, N, K, 3)  # Gather coords (B, N, K, 3)
#         )  # (B, N, K, 3)

#         dist = torch.norm(skip_coords.unsqueeze(2) - neighbor_coords, dim=-1)  # (B, N, K)
#         dist_recip = 1.0 / (dist + 1e-8)  # Avoid division by zero
#         norm = torch.sum(dist_recip, dim=-1, keepdim=True)  # (B, N, 1)
#         weight = dist_recip / norm  # (B, N, K)

#         # Compute the weighted sum of neighbor features
#         new_feat = torch.sum(neighbor_feat * weight.unsqueeze(-1), dim=2)  # (B, N, C)

#         # Save tensors for backward pass
#         ctx.save_for_backward(upsampling_idxs, weight, curr_feat)

#         return torch.cat([skip_coords, new_feat], dim=-1)  # (B, N, 3 + C)

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass for interpolation, computing gradients for the input features.
#         Args:
#             grad_output: Gradient of the output with respect to loss (B, N, 3 + C).
#         Returns:
#             grad_curr_points: Gradient of the input current points (B, M, 3 + C).
#             None: No gradient for skip_points.
#             None: No gradient for upsampling_idxs.
#         """
#         upsampling_idxs, weight, curr_feat = ctx.saved_tensors
#         B, N, K = upsampling_idxs.shape
#         C = curr_feat.shape[-1]

#         # Extract the gradient for the feature part
#         grad_new_feat = grad_output[:, :, 3:]  # (B, N, C)

#         # Initialize gradient for the current points' features
#         grad_curr_feat = torch.zeros_like(curr_feat)  # (B, M, C)

#         # Propagate gradients to the current features
#         for i in range(K):
#             grad_curr_feat.index_add_(
#                 1, upsampling_idxs[:, :, i].view(-1),
#                 (grad_new_feat * weight[:, :, i].unsqueeze(-1)).view(-1, C)
#             )

#         # No gradients for coords (assuming the features are the target for optimization)
#         return torch.cat([torch.zeros_like(curr_feat[:, :, :3]), grad_curr_feat], dim=-1), None, None


# interpolation = Interpolation.apply
