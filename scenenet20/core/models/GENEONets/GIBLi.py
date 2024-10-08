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
from core.pooling.fps_pooling import FPSPooling_Module
from core.pooling.grid_poooling import GridPooling_Module
from core.neighboring.neighbors import Neighboring_Method



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
            raise NotImplementedError("Input dimension not supported")
        

class Neighboring(nn.Module):

    def __init__(self, 
                 neighborhood_strategy:str, 
                 num_neighbors:int, 
                 **kwargs
                ) -> None:
        
        super(Neighboring, self).__init__()

        self.neighbor = Neighboring_Method(neighborhood_strategy, num_neighbors, **kwargs)

    def forward(self, x, q_points) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (B, N, 3)

        q_points : torch.Tensor
            query points of shape (B, Q, 3)

        Returns
        -------
        s_points_idxs : torch.Tensor
            support points of shape (B, Q, k)
        """
        return self.neighbor(x, q_points)


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


    def forward(self, coords, q_points, s_idxs, feats) -> torch.Tensor:

        gib_out = self.gib(coords, q_points, s_idxs) # (B, Q, num_observers)
        gib_out = self.act(self.gib_norm(gib_out)) # (B, Q, num_observers)

        mlp_out = torch.cat([feats, gib_out], dim=-1) # (B, Q, C + num_observers)
        mlp_out = self.mlp(mlp_out) # (B, Q, C + num_observers)
        mlp_out = self.act(self.mlp_norm(mlp_out)) # (B, Q, C + num_observers)

        return mlp_out
    

class GIB_Sequence(nn.Module):

    def __init__(self, 
                 num_layers, 
                 gib_dict, 
                 feat_channels, 
                 num_observers, 
                 kernel_size,
                ) -> None:
        
        super(GIB_Sequence, self).__init__()

        self.gib_blocks = nn.ModuleList()
        for _ in range(num_layers):
            gib_block = GIB_Block(gib_dict, feat_channels, num_observers, kernel_size)
            feat_channels = feat_channels + num_observers
            self.gib_blocks.append(gib_block)
        

    def forward(self, x, q_coords, s_idxs) -> torch.Tensor:

        coords, feats = x[..., :3], x[..., 3:]

        for gib_block in self.gib_blocks:
            feats = gib_block(coords, q_coords, s_idxs, feats)

        # feats shape: (B, Q, feat_channels + num_observers*num_layers) 

        return feats
    

class GIB_Down(nn.Module):

    def __init__(self, 
                 num_layers, 
                 gib_dict, 
                 feat_channels, 
                 num_observers, 
                 kernel_size,
                 pooling_strategy:str,
                 pooling_kwargs,
                 neighborhood_strategy:str,
                 num_neighbors:int,
                 neighborhood_kwargs,
                ) -> None:
        
        super(GIB_Down, self).__init__()

        self.gib = GIB_Sequence(num_layers, gib_dict, feat_channels, num_observers, kernel_size)

        self.neighbor = Neighboring(neighborhood_strategy, num_neighbors, **neighborhood_kwargs)

        if pooling_strategy == 'fps':
            self.pooling = FPSPooling_Module(neighborhood_strategy, num_neighbors, neighborhood_kwargs, **pooling_kwargs)
        elif pooling_strategy == 'grid':
            self.pooling = GridPooling_Module(**pooling_kwargs)
        else:
            raise NotImplementedError(f"Pooling strategy {pooling_strategy} not implemented.")

    def forward(self, x) -> torch.Tensor:

        x_pooled = self.pooling(x) # (B, N//pooling_factor, C)
        s_points_idxs = self.neighbor(x, x_pooled) # (B, N//pooling_factor, k)

        return x_pooled, self.gib(x, x_pooled, s_points_idxs) # (B, N//pooling_factor, C + num_observers*num_layers)


class Unpool_wSkip(nn.Module):

    def __init__(self,  
                in_channels,
                skip_channels,
                out_channels,
                bias=True,
                skip=True,
                backend="map") -> None:
    
        super(Unpool_wSkip, self).__init__()

        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x, x_skip, mapping) -> torch.Tensor:
            x_coords, x_feats = x[..., :3], x[..., 3:]
            x_skip_coords, x_skip_feats = x_skip[..., :3], x_skip[..., 3:]

            if self.backend == "map":
                feat = self.proj(x_feats)[mapping]

            elif self.backend == "interp":
                pass
            if self.skip:
                feat += self.proj_skip(x_skip_feats)



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


    