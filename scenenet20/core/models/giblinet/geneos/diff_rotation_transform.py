


import torch


def build_rotarion_matrix(angles:torch.Tensor) -> torch.Tensor: 
    """
    Build a rotation matrix from the given angles.
    See https://en.wikipedia.org/wiki/Rotation_matrix#:~:text=General%203D%20rotations for more information.

    Parameters
    ----------
    `angles` - torch.Tensor:
        Tensor of shape (3,) containing rotation angles for the x, y, and z axes.
        These are normalized in the range [0, 2] and represent their fraction of pi.
    """

    angles = angles * torch.pi
    t_x, t_y, t_z = angles

    cos_tx, sin_tx = torch.cos(t_x), torch.sin(t_x)
    cos_ty, sin_ty = torch.cos(t_y), torch.sin(t_y)
    cos_tz, sin_tz = torch.cos(t_z), torch.sin(t_z)

    R = torch.stack([
        torch.stack([cos_ty * cos_tz, cos_tz * sin_tx * sin_ty - cos_tx * sin_tz, cos_tx * cos_tz * sin_ty + sin_tx * sin_tz]),
        torch.stack([cos_ty * sin_tz, cos_tx * cos_tz + sin_tx * sin_ty * sin_tz, -cos_tz * sin_tx + cos_tx * sin_ty * sin_tz]),
        torch.stack([-sin_ty, cos_ty * sin_tx, cos_tx * cos_ty])
    ])

    return R

def rotate_points(angles:torch.Tensor, points:torch.Tensor) -> torch.Tensor:
    """
    Rotate a tensor along the x, y, and z axes by the given angles.

    Parameters
    ----------
    `angles` - torch.Tensor:
        Tensor of shape (3,) containing rotation angles for the x, y, and z axes.
        These are nromalized in the range [-1, 1] and represent angles_normalized = angles / pi.

    `points` - torch.Tensor:
        Tensor of shape (N, 3) representing the 3D points to rotate.

    Returns
    -------
    `points` - torch.Tensor:
        Tensor of shape (N, 3) containing the rotated
    """

    R = build_rotarion_matrix(angles).contiguous()
    # print(f"{R.device=} {points.device=}")
    return torch.matmul(points.contiguous(), R.T)




##################################################################################################
# Multiple Angles


@torch.jit.script
def build_rotation_matrices(angles: torch.Tensor) -> torch.Tensor:
    angles = angles * torch.pi
    t_x, t_y, t_z = angles[:, 0], angles[:, 1], angles[:, 2]

    cos_tx, sin_tx = torch.cos(t_x), torch.sin(t_x)
    cos_ty, sin_ty = torch.cos(t_y), torch.sin(t_y)
    cos_tz, sin_tz = torch.cos(t_z), torch.sin(t_z)

    R = torch.stack([
        torch.stack([cos_ty * cos_tz, cos_tz * sin_tx * sin_ty - cos_tx * sin_tz, cos_tx * cos_tz * sin_ty + sin_tx * sin_tz], dim=-1),
        torch.stack([cos_ty * sin_tz, cos_tx * cos_tz + sin_tx * sin_ty * sin_tz, -cos_tz * sin_tx + cos_tx * sin_ty * sin_tz], dim=-1),
        torch.stack([-sin_ty, cos_ty * sin_tx, cos_tx * cos_ty], dim=-1)
    ], dim=-2)

    return R

@torch.jit.script
def rotate_points_batch(angles: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Rotate a batch of tensors along the x, y, and z axes by the given angles.

    Parameters
    ----------
    `angles` - torch.Tensor:
        Tensor of shape (G, 3) containing rotation angles for the x, y, and z axes for G rotations.
        These are normalized in a range [0, 2] and represent their fraction of pi.

    `points` - torch.Tensor:
        Tensor of shape (..., N, 3) representing the 3D points to rotate.

    Returns
    -------
    `points` - torch.Tensor:
        Tensor of shape (..., G, N, 3) containing the rotated points for G batches.
    """
    R = build_rotation_matrices(angles.contiguous()) # shape (G, 3, 3)
    # points = points.unsqueeze(-3)  # (..., 1, N, 3)
    # points = torch.matmul(points, R.transpose(-1, -2))  # (..., G, N, 3)
    # print(points.shape)
    # return points
    # flat points
    points_shape = points.size()
    points = points.view(-1, points_shape[-1]).contiguous()  # (*, 3)
    points = torch.matmul(points, R.transpose(-1, -2)) # (*, 3) x (G, 3, 3) -> (*, G, 3)
    # implictit shape recovery can be achieved but not in script mode
    points = points.view(points_shape[0], points_shape[1], R.size(0), points_shape[2], -1)  # (..., G, N, 3)
    return points.contiguous()


if __name__ == '__main__':
    
    num_gibs = 15
    angles = torch.rand((num_gibs, 3)) # random angles
    
    points = torch.ones((4, 4, 10, 3)) # 4 batches, 4 neighbors, 10 3D points
    
    rotated_points = rotate_points_batch(angles, points)
    
    # print(rotated_points.shape)  # torch.Size([4, 4, 3, 10, 3])