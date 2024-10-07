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

from core.models.GENEONets.GENEO_utils import GENEO_Layer



class GeometricInductiveBiasModel(nn.Module):

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
        
        super(GeometricInductiveBiasModel, self).__init__()

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
            gib_layer = GENEO_Layer(gib_dict, in_channels, num_observers, kernel_size)
            self.gibs.append(gib_layer)
    


if __name__ == "__main__":
    import sys
    import os
   
    
    # make random torch data to test the model
    x = torch.rand((2, 1, 32, 32, 32)).cuda() # (batch, c, z, x, y)


    