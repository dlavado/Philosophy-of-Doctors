import torch.nn as nn

from core.models.kpconv.easy_kpconv.ops.pooling import local_maxpool_pack_mode
from core.models.kpconv.easy_kpconv.layers.basic_layers import (
    build_act_layer,
    build_norm_layer_pack_mode,
    check_bias_from_norm_cfg,
    LayerConfig,
)
from core.models.kpconv.easy_kpconv.layers.kpconv import KPConv
from core.models.kpconv.easy_kpconv.layers.unary_block import UnaryBlockPackMode


class KPConvBlock(nn.Module):
    """KPConv block with normalization and activation.

    Args:
        in_channels (int): dimension input features
        out_channels (int): dimension input features
        kernel_size (int): number of kernel points
        radius (float): convolution radius
        sigma (float): influence radius of kernel points
        dimension (int=3): dimension of input
        norm_cfg (str|dict|None='GroupNorm'): normalization config
        act_cfg (str|dict|None='LeakyReLU'): activation config
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        groups: int = 1,
        dimension: int = 3,
        norm_cfg: LayerConfig = "GroupNorm",
        act_cfg: LayerConfig = "LeakyReLU",
    ):
        super().__init__()

        bias = check_bias_from_norm_cfg(norm_cfg)
        self.conv = KPConv(
            in_channels, out_channels, kernel_size, radius, sigma, groups=groups, bias=bias, dimension=dimension
        )

        self.norm = build_norm_layer_pack_mode(out_channels, norm_cfg)
        self.act = build_act_layer(act_cfg)

    def forward(self, q_points, s_points, s_feats, neighbor_indices):
        q_feats = self.conv(q_points, s_points, s_feats, neighbor_indices)
        q_feats = self.norm(q_feats)
        q_feats = self.act(q_feats)
        return q_feats


class KPResidualBlock(nn.Module):
    """KPConv residual bottleneck block.

    Args:
        in_channels (int): dimension input features
        out_channels (int): dimension input features
        kernel_size (int): number of kernel points
        radius (float): convolution radius
        sigma (float): influence radius of each kernel point
        dimension (int=3): dimension of input
        strided (bool=False): strided or not
        norm_cfg (str|dict|None='GroupNorm'): normalization config
        act_cfg (str|dict|None='LeakyReLU'): activation config
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        groups: int = 1,
        dimension: int = 3,
        strided: bool = False,
        norm_cfg: LayerConfig = "GroupNorm",
        act_cfg: LayerConfig = "LeakyReLU",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4
        # print("mid_channels: ", mid_channels, "out_channels: ", out_channels)
        self.unary1 = UnaryBlockPackMode(in_channels, mid_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv = KPConvBlock(
            mid_channels,
            mid_channels,
            kernel_size,
            radius,
            sigma,
            groups=groups,
            dimension=dimension,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.unary2 = UnaryBlockPackMode(mid_channels, out_channels, norm_cfg=norm_cfg, act_cfg=None)

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlockPackMode(in_channels, out_channels, norm_cfg=norm_cfg, act_cfg=None)
        else:
            self.unary_shortcut = nn.Identity()

        self.act = build_act_layer(act_cfg)

    def forward(self, q_points, s_points, s_feats, neighbor_indices):
        residual = self.unary1(s_feats)
        residual = self.conv(q_points, s_points, residual, neighbor_indices)
        residual = self.unary2(residual)

        if self.strided:
            shortcut = local_maxpool_pack_mode(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        q_feats = residual + shortcut
        q_feats = self.act(q_feats)

        return q_feats
    
    
    
###################################################################################################

from core.models.giblinet.GIBLi_parts import GIBLiLayer
from core.models.giblinet.conversions import pack_to_batch, batch_to_pack

class GIBLiKPResidualBlock(nn.Module):
    """KPConv residual bottleneck block.

    Args:
        in_channels (int): dimension input features
        out_channels (int): dimension input features
        kernel_size (int): number of kernel points
        radius (float): convolution radius
        sigma (float): influence radius of each kernel point
        dimension (int=3): dimension of input
        strided (bool=False): strided or not
        norm_cfg (str|dict|None='GroupNorm'): normalization config
        act_cfg (str|dict|None='LeakyReLU'): activation config
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        groups: int = 1,
        dimension: int = 3,
        strided: bool = False,
        norm_cfg: LayerConfig = "GroupNorm",
        act_cfg: LayerConfig = "LeakyReLU",
        ### gib params
        gib_dict={},
        num_observers=[8, 8],
        kernel_reach=0.1,
        neighbor_size=[8, 16],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4
        
        self.unary1 =  GIBLiLayer(in_channels=in_channels, out_channels=mid_channels, gib_dict=gib_dict, num_observers=num_observers, kernel_reach=kernel_reach, neighbor_size=neighbor_size) 
        out_gibli_channels = mid_channels + sum(num_observers)
        self.gibli_proj = nn.Linear(out_gibli_channels, mid_channels, bias=False)
        # self.unary1 = UnaryBlockPackMode(in_channels, mid_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.conv = KPConvBlock(
            mid_channels,
            mid_channels,
            kernel_size,
            radius,
            sigma,
            groups=groups,
            dimension=dimension,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.unary2 = UnaryBlockPackMode(mid_channels, out_channels, norm_cfg=norm_cfg, act_cfg=None)

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlockPackMode(in_channels, out_channels, norm_cfg=norm_cfg, act_cfg=None)
        else:
            self.unary_shortcut = nn.Identity()

        self.act = build_act_layer(act_cfg)
        
        
    def build_input_gibli(self, coord, feat, lengths):
        batched_coord, mask = pack_to_batch(coord, lengths)
        input_dict = {
            "coord": batched_coord,
            "feat": pack_to_batch(feat, lengths)[0],
            "mask": mask,
        }
        return input_dict

    def forward(self, q_points, s_points, s_feats, neighbor_indices, lengths):
        
        gibli_dict = self.build_input_gibli(s_points, s_feats, lengths)
        gibli_out = self.gibli_proj(self.unary1(gibli_dict))
        residual = batch_to_pack(gibli_out, gibli_dict['mask'])[0]        
        residual = self.conv(q_points, s_points, residual, neighbor_indices)
        residual = self.unary2(residual)

        if self.strided:
            shortcut = local_maxpool_pack_mode(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        q_feats = residual + shortcut
        q_feats = self.act(q_feats)

        return q_feats # (M, C)
