# Modified from [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).
import math

import torch
import torch.nn as nn
from torch import Tensor

from core.models.kpconv.easy_kpconv.ops.index_select import index_select
from core.models.kpconv.easy_kpconv.layers.kpconv_utils import load_kernels


class KPConv(nn.Module):
    """Rigid KPConv.

    Paper: https://arxiv.org/abs/1904.08889.

    Args:
         in_channels (int): The number of the input channels.
         out_channels (int): The number of the output channels.
         kernel_size (int): The number of kernel points.
         radius (float): The radius used for kernel point init.
         sigma (float): The influence radius of each kernel point.
         bias (bool, optional): If True, use bias. Default: False.
         dimension (int, optional): The dimension of the point space. Default: 3.
         inf (float, optional): The value of infinity to generate the padding point. Default: 1e6.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        groups: int = 1,
        bias: bool = False,
        dimension: int = 3,
        inf: float = 1e6,
    ):
        """Initialize a rigid KPConv."""
        super().__init__()

        assert in_channels % groups == 0, "in_channels must be divisible by groups."
        assert out_channels % groups == 0, "out_channels must be divisible by groups."
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups

        # Save parameters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.groups = groups
        self.dimension = dimension
        self.inf = inf
        self.in_channels_per_group = in_channels_per_group
        self.out_channels_per_group = out_channels_per_group

        # Initialize weights
        if self.groups == 1:
            weights = torch.zeros(size=(kernel_size, in_channels, out_channels))
        else:
            weights = torch.zeros(size=(kernel_size, groups, in_channels_per_group, out_channels_per_group))
        self.weights = nn.Parameter(weights)

        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,)))
        else:
            self.register_parameter("bias", None)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()  # (N, 3)
        self.register_buffer("kernel_points", kernel_points)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def initialize_kernel_points(self) -> Tensor:
        """Initialize the kernel point positions in a sphere."""
        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed="center")
        return torch.from_numpy(kernel_points).float()

    def forward(self, q_points: Tensor, s_points: Tensor, s_feats: Tensor, neighbor_indices: Tensor) -> Tensor:
        """KPConv forward.

        Args:
            s_feats (Tensor): (N, C_in)
            q_points (Tensor): (M, 3)
            s_points (Tensor): (N, 3)
            neighbor_indices (LongTensor): (M, H)

        Returns:
            q_feats (Tensor): (M, C_out)
        """
        # print("\n\n", q_points.shape, s_points.shape, s_feats.shape, neighbor_indices.shape)

        padded_s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0)  # (N, 3) -> (N+1, 3)
        neighbors = index_select(padded_s_points, neighbor_indices, dim=0)  # (N+1, 3) -> (M, H, 3)
        neighbors = neighbors - q_points.unsqueeze(1)  # (M, H, 3)

        # print(f"{neighbor_indices.shape=}")

        # Get Kernel point influences
        neighbors = neighbors.unsqueeze(2)  # (M, H, 3) -> (M, H, 1, 3)
        differences = neighbors - self.kernel_points  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
        sq_distances = torch.sum(differences ** 2, dim=3)  # (M, H, K)
        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)  # (M, H, K)
        neighbor_weights = torch.transpose(neighbor_weights, 1, 2)  # (M, H, K) -> (M, K, H)

        # apply neighbor weights
        # print(s_feats.shape)
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
        neighbor_feats = index_select(padded_s_feats, neighbor_indices, dim=0)  # (N+1, C) -> (M, H, C)
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)

        # apply convolutional weights
        if self.groups == 1:
            # standard conv
            # print("\n\n", weighted_feats.shape, self.weights.shape)
            output_feats = torch.einsum("mkc,kcd->md", weighted_feats, self.weights)
        else:
            # group conv
            weighted_feats = weighted_feats.view(-1, self.kernel_size, self.groups, self.in_channels_per_group)
            output_feats = torch.einsum("mkgc,kgcd->mgd", weighted_feats, self.weights)
            output_feats = output_feats.view(-1, self.out_channels)

        # density normalization
        neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)  # (M, H)
        neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)  # (M,)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        output_feats = output_feats / neighbor_num.unsqueeze(1)

        # NOTE: normalization with only positive neighbors works slightly better than all neighbors
        # neighbor_num = torch.sum(torch.lt(neighbor_indices, s_points.shape[0]), dim=-1)  # (M,)
        # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        # output_feats = output_feats / neighbor_num.unsqueeze(1)

        # add bias
        if self.bias is not None:
            output_feats = output_feats + self.bias

        return output_feats

    def extra_repr(self) -> str:
        param_strings = [
            f"kernel_size={self.kernel_size}",
            f"in_channels={self.in_channels}",
            f"out_channels={self.out_channels}",
            f"radius={self.radius:g}",
            f"sigma={self.sigma:g}",
            f"bias={self.bias is not None}",
        ]
        if self.dimension != 3:
            param_strings.append(f"dimension={self.dimension}")
        format_string = ", ".join(param_strings)
        return format_string
