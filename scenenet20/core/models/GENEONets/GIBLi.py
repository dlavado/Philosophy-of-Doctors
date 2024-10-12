from ast import Tuple
from typing import List, Mapping, Dict
from git import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys


sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')

from core.models.GENEONets.GIBLi_parts import GIB_Sequence, Unpool_wSkip
from core.models.GENEONets.GIBLi_utils import BuildGraphPyramid


#################################################################
# GIBLi: Geometric Inductive Bias Library for Point Clouds
#################################################################

class GeometricInductiveBias(nn.Module):

    def __init__(self, 
                in_channels:int,
                num_classes:int,
                num_levels:int,
                out_gib_channels:int,
                num_observers:int,
                kernel_size:Union[int, Tuple[int, int, int]],
                gib_dict:dict,
                neighborhood_strategy:str,
                neighborhood_size:int,
                neighborhood_kwargs:Dict,
                neighborhood_update_kwargs:Dict,
                skip_connections:bool,
                graph_strategy:str,
                graph_pooling_factor:int,
                ) -> None:
        
        super(GeometricInductiveBias, self).__init__()

        self.skip_connections = skip_connections
        self.num_levels = num_levels

        # Build the graph pyramid
        self.graph_pyramid = BuildGraphPyramid(num_levels,
                                    graph_strategy,
                                    graph_pooling_factor,
                                    neighborhood_size,
                                    neighborhood_strategy,
                                    neighborhood_kwargs,
                                    neighborhood_update_kwargs,
                                    voxel_size = torch.tensor([0.05, 0.05, 0.05]).cuda() if graph_strategy=="grid" else None
                                )

        # Build the GIB layers
        self.gib_neigh_encoders = nn.ModuleList()
        self.gib_pooling_encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        enc_channels = []
        for i in range(num_levels):
            f_channels = in_channels if i == 0 else out_gib_channels*(i-1)
            out_channels = out_gib_channels*(i+1)
            gib_seq = GIB_Sequence(num_layers=(i+1), gib_dict=gib_dict, feat_channels=f_channels, num_observers=num_observers, kernel_size=kernel_size, out_gib_channels=out_channels)
            self.gib_neigh_encoders.append(gib_seq)
            enc_channels.append(out_channels)

        for i in range(num_levels - 1):
            # Build pooling layers
            f_channels = enc_channels[i]
            # maintain the same number of channels for the pooling encoder
            gib_seq = GIB_Sequence(num_layers=(i+1), gib_dict=gib_dict, feat_channels=f_channels, num_observers=num_observers, kernel_size=kernel_size, out_gib_channels=f_channels)
            self.gib_pooling_encoders.append(gib_seq)

            # Build unpooling layers
            unpool = Unpool_wSkip(feat_channels=enc_channels[i+1], skip_channels=enc_channels[i], out_channels=enc_channels[i], skip=skip_connections)
            self.decoders.append(unpool)


    def forward(self, x:torch.Tensor) -> torch.Tensor:

        coords, feats = x[..., :3], x[..., 3:]

        # Build the graph pyramid
        graph_pyramid_dict = self.graph_pyramid(coords)

        point_list = graph_pyramid_dict['points_list'] # shape of 0th element: (batch, num_points, 3)
        neighbors_idxs_list = graph_pyramid_dict['neighbors_idxs_list'] # shape of 0th element: (B, Q[0], neighborhood_size[0]) idxs from points[0]
        subsampling_idxs_list = graph_pyramid_dict['subsampling_idxs_list'] # shape of 0th element: (B, Q[1], neighborhood_size[1]) idxs from points[0]
        upsampling_idxs_list = graph_pyramid_dict['upsampling_idxs_list'] # shape of 0th element: (B, Q[-1], neighborhood_size[-1]) idxs from points[-2]

        level_feats = []

        for i in range(self.num_levels): # encoding phase
            feats = self.gib_neigh_encoders[i]((coords, feats), point_list[i], neighbors_idxs_list[i])
            level_feats.append(feats) # save the features for skip connections

            if i < self.num_levels - 1:
                feats = self.gib_pooling_encoders[i]((coords, feats), point_list[i], subsampling_idxs_list[i]) # pooling

            coords = point_list[i+1]


        for i in range(self.num_levels - 1, -1, -1): # decoding phase
            curr_points = point_list[i]
            curr_feats = level_feats[i]
            skip_feats = level_feats[i-1] if i > 0 else None
            skip_coords = point_list[i-1] if i > 0 else None
            feats = self.decoders[i]((curr_points, curr_feats), (skip_feats, skip_coords), upsampling_idxs_list[i])


        

            




        
    


if __name__ == "__main__":
    import sys
    import os
   
    
    # make random torch data to test the model
    x = torch.rand((2, 1, 32, 32, 32)).cuda() # (batch, c, z, x, y)


    