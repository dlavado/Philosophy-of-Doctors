


import torch
import torch.nn.functional as F


def build_rotarion_matrix(angles:torch.Tensor) -> torch.Tensor: 
    """
    Build a rotation matrix from the given angles.
    See https://en.wikipedia.org/wiki/Rotation_matrix#:~:text=General%203D%20rotations for more information.

    Parameters
    ----------
    `angles` - torch.Tensor:
        Tensor of shape (3,) containing rotation angles for the x, y, and z axes.
        These are normalized in the range [-1, 1] and represent angles_normalized = angles / pi.
    """

    angles = angles * 180 # convert to degrees
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

    R = build_rotarion_matrix(angles)
    return torch.matmul(points, R.T)

if __name__ == '__main__':
    pass