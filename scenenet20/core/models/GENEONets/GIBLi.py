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

from core.models.GENEONets.GIBLi_parts import GIB_Sequence, Unpool_wSkip, PointBatchNorm
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
                kernel_size:float,
                gib_dict:dict,
                neighborhood_strategy:str,
                neighborhood_size:Union[int, List[int]],
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
        if isinstance(neighborhood_size, int):
            # if the neighborhood size is an integer, then increase the its size by a factor of 1.5 for each level
            neighborhood_size = [int(neighborhood_size + (neighborhood_size/2)*i) for i in range(num_levels)]

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
            gib_seq = GIB_Sequence(num_layers=(i+1), gib_dict=gib_dict, feat_channels=f_channels, num_observers=num_observers, kernel_size=kernel_size, out_channels=out_channels)
            self.gib_neigh_encoders.append(gib_seq)
            enc_channels.append(out_channels)

        for i in range(num_levels - 1):
            # Build pooling layers
            f_channels = enc_channels[i]
            # maintain the same number of channels for the pooling encoder
            gib_seq = GIB_Sequence(num_layers=(i+1), gib_dict=gib_dict, feat_channels=f_channels, num_observers=num_observers, kernel_size=kernel_size, out_channels=f_channels)
            self.gib_pooling_encoders.append(gib_seq)

            # Build unpooling layers
            unpool = Unpool_wSkip(feat_channels=enc_channels[i+1], skip_channels=enc_channels[i], out_channels=enc_channels[i], skip=skip_connections, backend='interp')
            self.decoders.append(unpool)

        self.seg_head = nn.Sequential(
            nn.Linear(enc_channels[-1], enc_channels[-1]),
            PointBatchNorm(enc_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(enc_channels[-1], num_classes)
        )


    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # coords, feats = x[..., :3], x[..., 3:]

        if x.shape[-1] > 3:
            feats = x[..., 3:]
        else:
            feats = x
        coords = x[..., :3]

        # Build the graph pyramid
        graph_pyramid_dict = self.graph_pyramid(coords)

        point_list = graph_pyramid_dict['points_list'] # shape of 0th element: (batch, num_points, 3)
        neighbors_idxs_list = graph_pyramid_dict['neighbors_idxs_list'] # shape of 0th element: (B, Q[0], neighborhood_size[0]) idxs from points[0]
        subsampling_idxs_list = graph_pyramid_dict['subsampling_idxs_list'] # shape of 0th element: (B, Q[1], neighborhood_size[1]) idxs from points[0]
        upsampling_idxs_list = graph_pyramid_dict['upsampling_idxs_list'] # shape of 0th element: (B, Q[-1], neighborhood_size[-1]) idxs from points[-2]

        level_feats = []

        for i in range(self.num_levels): # encoding phase
            
            if i > 0: ###### Pooling phase ######
                feats = self.gib_pooling_encoders[i]((coords, feats), point_list[i], subsampling_idxs_list[i - 1]) # pooling
                print(f"Pooling {i}: {feats.shape}")
            
            ###### Encoding phase ######
            feats = self.gib_neigh_encoders[i]((coords, feats), point_list[i], neighbors_idxs_list[i])
            print(f"Encoding {i}: {feats.shape}")
            level_feats.append(feats) # save the features for skip connections

            coords = point_list[i+1]

        curr_latent_feats = level_feats[-1] # N 
        curr_coords = point_list[-1] # N
        ###### Decoding phase ######
        for i in reversed(range(len(upsampling_idxs_list))): # there are num_levels - 1 unpooling layers
            skip_feats  = level_feats[i]
            skip_coords = point_list[i]

            curr_latent_points = self.decoders[i]((curr_coords, curr_latent_feats), (skip_coords, skip_feats), upsampling_idxs_list[i])
            print(f"Decoding {i}: {curr_latent_points.shape}")
            curr_coords, curr_latent_feats = curr_latent_points[..., :3], curr_latent_points[..., 3:]

        ###### Segmentation phase ######
        seg_logits = self.seg_head(curr_latent_feats)
        print(seg_logits.shape)
        return seg_logits



        


if __name__ == "__main__":
    import sys
    import os
    import torchsummary
   
    
    # make random torch data to test the model
    x = torch.rand(1, 10000, 6).cuda()

    # define the model
    in_channels = 3
    num_classes = 10
    num_levels = 4
    out_gib_channels = 64
    num_observers = 16
    kernel_size = 0.1
    gib_dict = {
        'cy' : 2,
        'ellip': 2,
        'disk': 2
    }

    neighborhood_strategy = "knn"
    neighborhood_size = 4
    neighborhood_kwargs = {}
    neighborhood_update_kwargs = {}

    skip_connections = True
    graph_strategy = "fps"
    graph_pooling_factor = 2


    model = GeometricInductiveBias(in_channels, num_classes, num_levels, out_gib_channels, num_observers, kernel_size, gib_dict, neighborhood_strategy, neighborhood_size, neighborhood_kwargs, neighborhood_update_kwargs, skip_connections, graph_strategy, graph_pooling_factor).cuda()
    
    # print the model summary
    # torchsummary.summary(model, x.shape)

    pred = model(x)
    print(pred.shape) # should be (1, 10) for 10 classes