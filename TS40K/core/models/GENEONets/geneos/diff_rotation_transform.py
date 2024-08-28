


import torch
import torch.nn.functional as F


def rotate(data, interpolation='bilinear', angle=0.0, expand=False):
    """
    Rotate the input tensor.

    Args:
        data (torch.Tensor): Input tensor to be rotated. It should have shape (N, C, D, H, W).
        interpolation (str, optional): Interpolation mode for rotation. Default is 'bilinear'.
        angle (float, optional): Angle of rotation in degrees.
        expand (bool, optional): If True, expands the output tensor to make it large enough to hold the entire rotated image.
                                 If False, the output tensor will have the same size as the input tensor. Default is False.
        fill (float, optional): Value to fill the area outside the rotated image. Default is 0.

    Returns:
        torch.Tensor: Rotated tensor.
    """
    # Compute rotation matrix
    theta = torch.stack([torch.cos(angle), -torch.sin(angle), torch.tensor(0, device=data.device),
                          torch.sin(angle), torch.cos(angle), torch.tensor(0, device=data.device)])
    theta = theta.view(2, 3).to(data.device)

    data = data.unsqueeze(0)
    grid = F.affine_grid(theta.unsqueeze(0), data.size(), align_corners=False)

    rotated_data = F.grid_sample(data, grid, mode=interpolation, align_corners=False)

    if not expand:
        rotated_data = rotated_data.squeeze(0)

    return rotated_data



def rotation_3d(X, axis, theta, expand=False, interpolation='bilinear'):
    """
    The rotation is based on torchvision.transforms.functional.rotate, which is originally made for a 2d image rotation
    :param X: the data that should be rotated, a torch.tensor or an ndarray, with lenx * leny * lenz shape.
    :param axis: the rotation axis based on the keynote request. 0 for x axis, 1 for y axis, and 2 for z axis.
    :param expand:  (bool, optional) – Optional expansion flag. If true, expands the output image to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
    :param theta: the rotation angle, Counter-clockwise rotation, [-180, 180] degrees.
    :return: rotated tensor.
    """

    print(theta.grad_fn)

    if axis == 0:
        X = rotate(X, interpolation=interpolation, angle=theta, expand=expand)
    elif axis == 1:
        X = X.permute((1, 0, 2))
        X = rotate(X, interpolation=interpolation, angle=theta, expand=expand)
        X = X.permute((1, 0, 2))
    elif axis == 2:
        X = X.permute((2, 1, 0))
        X = rotate(X, interpolation=interpolation, angle=-theta, expand=expand)
        X = X.permute((2, 1, 0))
    else:
        raise Exception('Not invalid axis')
    return X.squeeze(0)


if __name__ == '__main__':
    # Example usage
    x = torch.rand(3, 64, 64)  # Assuming input image size is 64x64 and has 3 channels
    angle = torch.nn.Parameter(torch.tensor([0.0, 0.0, 90.0]))
    x.requires_grad = True
    rotated_x = rotation_3d(x, 2, angle[-1], expand=False)
    print(rotated_x.shape)

    # Compute gradients
    loss = torch.mean(rotated_x)  # Example loss
    loss.backward()

    # Access gradients with respect to the rotation angle
    print("Gradient w.r.t. angle:", angle.grad)