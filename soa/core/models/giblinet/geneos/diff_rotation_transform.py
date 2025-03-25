


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
def update_rotation_matrices_inplace(angles: torch.Tensor, R: torch.Tensor) -> None:
    angles = angles * torch.pi
    t_x, t_y, t_z = angles.unbind(dim=-1) # slices the tensor along the last dimension

    cos_tx, sin_tx = torch.cos(t_x), torch.sin(t_x)
    cos_ty, sin_ty = torch.cos(t_y), torch.sin(t_y)
    cos_tz, sin_tz = torch.cos(t_z), torch.sin(t_z)

    R[:, 0, 0].copy_(cos_ty * cos_tz)
    R[:, 0, 1].copy_(cos_tz * sin_tx * sin_ty - cos_tx * sin_tz)
    R[:, 0, 2].copy_(cos_tx * cos_tz * sin_ty + sin_tx * sin_tz)

    R[:, 1, 0].copy_(cos_ty * sin_tz)
    R[:, 1, 1].copy_(cos_tx * cos_tz + sin_tx * sin_ty * sin_tz)
    R[:, 1, 2].copy_(-cos_tz * sin_tx + cos_tx * sin_ty * sin_tz)

    R[:, 2, 0].copy_(-sin_ty)
    R[:, 2, 1].copy_(cos_ty * sin_tx)
    R[:, 2, 2].copy_(cos_tx * cos_ty)

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
    # update_rotation_matrices_inplace(angles.contiguous(), R)  # shape (G, 3, 3)
    points_shape = points.size()
    points = points.view(-1, points_shape[-1]).contiguous()  # (*, 3)
    points = points @ R.transpose(-1, -2)  # (G, 3, *) x (G, 3, 3) -> (G, 3, *)
    # implictit shape recovery can be achieved but not in script mode
    points = points.view(points_shape[0], points_shape[1], R.size(0), points_shape[2], -1)  # (..., G, N, 3)
    return points.contiguous()


@torch.jit.script
def rotate(points:torch.Tensor, angles:torch.Tensor) -> torch.Tensor:
    """
    Rotate a tensor along the x, y, and z axes by the angles of each GIB in the collection.
    
    Parameters
    ----------
    `points` - torch.Tensor:
        Tensor of shape (N, 3) representing the 3D points to rotate.
        
    `angles` - torch.Tensor:
        Tensor of shape (G, 3) containing rotation angles for the x, y, and z axes for each GIB in the collection.
        These are normalized in the range [-1, 1] and represent angles_normalized = angles
        
    Returns
    -------
    `points` - torch.Tensor:
        Tensor of shape (G, N, 3) containing the rotated
    """
    # convert self.angles to an acceptable range [0, 2]:
    # this is equivalent to angles = self.angles % 2
    angles = torch.fmod(angles, 2) # rotations higher than 2\pi are equivalent to rotations within 2\pi
    # angles = angles + (angles < 0).float() * 2 
    
    angles = 2 - torch.relu(-angles) # convert negative angles to positive
    return rotate_points_batch(angles, points)


if __name__ == '__main__':
    import time
    
    # print(rotated_points.shape)  # torch.Size([4, 4, 3, 100_000, 3])
    num_gibs = 1000
    angles = torch.rand((num_gibs, 3))  # random angles
    points = torch.ones((4, 16, 10_000, 3))  # 4 batches, 16 neighbors, 100K points each

    # Time the rotation matrix construction
    start_build = time.time()
    R = build_rotation_matrices(angles)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_build = time.time()
    print(f"Time to build rotation matrices: {end_build - start_build:.6f} seconds")

    # Time the full operation
    start_full = time.time()
    rotated_points = rotate_points_batch(angles, points)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_full = time.time()
    print(f"Total time for rotate_points_batch: {end_full - start_full:.6f} seconds")