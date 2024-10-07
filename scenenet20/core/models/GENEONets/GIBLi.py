from ast import Tuple
from typing import List, Mapping
from git import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys


sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')

from core.models.GENEONets.GIB_utils import GIB_Layer



#################################################################
# GIBLi Utils
#################################################################
class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]

    where B is the batch size, N is the number of points, C is the number of channels, and L is the number of neighbors
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError
        
class Sampling_and_Neighboring:

    def __init__(self,
                 sampling_strategy:str, 
                 neighborhood_strategy:str, 
                ) -> None:
        # TODO
        self.sampling = sampling_strategy
        self.neighbor = neighborhood_strategy

    def __call__(self, x:torch.Tensor) -> torch.Any:

        q_points = self.sampling(x)
        s_points = self.neighbor(x, q_points)

        return q_points, s_points
        

#################################################################
# GIBLi Blocks
#################################################################

class GIB_Block(nn.Module):

    def __init__(self, gib_dict, feat_channels, num_observers, kernel_size) -> None:
        super(GIB_Block, self).__init__()

        self.gib = GIB_Layer(gib_dict, kernel_size, num_observers)
        self.gib_norm = PointBatchNorm(num_observers)

        self.mlp = nn.Linear(feat_channels + num_observers, feat_channels + num_observers, bias=False)
        self.mlp_norm = PointBatchNorm(feat_channels + num_observers)

        self.act = nn.ReLU(inplace=True)


    def forward(self, coords, q_points, s_points, feats) -> torch.Tensor:

        gib_out = self.gib(coords, q_points, s_points)
        gib_out = self.act(self.gib_norm(gib_out))

        mlp_out = torch.cat([feats, gib_out], dim=-1)
        mlp_out = self.mlp(mlp_out)
        mlp_out = self.act(self.mlp_norm(mlp_out))

        return mlp_out
    

class GIB_Sequence(nn.Module):

    def __init__(self, 
                 num_layers, 
                 gib_dict, 
                 feat_channels, 
                 num_observers, 
                 kernel_size,
                 samp_neigh:Sampling_and_Neighboring
                ) -> None:
        
        super(GIB_Sequence, self).__init__()

        self.samp_neigh = samp_neigh

        self.gib_blocks = nn.ModuleList()
        for _ in range(num_layers):
            gib_block = GIB_Block(gib_dict, feat_channels, num_observers, kernel_size)
            self.gib_blocks.append(gib_block)

        

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        coords = x[..., :3] # (B, N, 3)
        if x.shape[-1] == 3:
            feats = coords
        else:
            feats = x[..., 3:]  # (B, N, C)

        q_coords, s_coords = self.samp_neigh(coords)

        for gib_block in self.gib_blocks:
            feats = gib_block(coords, q_coords, s_coords, feats)

        return q_coords, feats





#################################################################
# GIBLi: Geometric Inductive Bias Library for Point Clouds
#################################################################

class GeometricInductiveBias(nn.Module):

    def __init__(self, 
                in_channels:int,
                num_classes:int,
                num_layers:int,
                num_observers:int,
                kernel_size:Union[int, Tuple[int, int, int]],
                gib_dict:dict,
                neighborhood_strategy:str,
                neighborhood_size:int,
                pooling_strategy:str,
                pooling_factor:int,
                skip_connections:bool,
                upsampling_strategy:str,
                ) -> None:
        
        super(GeometricInductiveBias, self).__init__()

        self.gib_dict = gib_dict
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.neighborhood_strategy = neighborhood_strategy
        self.neighborhood_size = neighborhood_size
        self.pooling_strategy = pooling_strategy
        self.pooling_factor = pooling_factor
        self.skip_connections = skip_connections
        self.upsampling_strategy = upsampling_strategy

        self.gibs = nn.ModuleList()
        for _ in range(num_layers):
            gib_layer = GIB_Layer(gib_dict, kernel_size, num_observers)
            self.gibs.append(gib_layer)
    


if __name__ == "__main__":
    import sys
    import os
   
    
    # make random torch data to test the model
    x = torch.rand((2, 1, 32, 32, 32)).cuda() # (batch, c, z, x, y)


    