from typing import List, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os


sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.models.GENEONets.geneos.GIB_Stub import GIB_Stub
from core.models.GENEONets.geneos import cylinder, neg_sphere, arrow, disk, cone, ellipsoid



###############################################################
#                         GENEO Layer                         #
###############################################################

class GIB_Operator(nn.Module):

    def __init__(self, geneo_class:GIB_Stub, kernel_reach:tuple=None, **kwargs):
        super(GIB_Operator, self).__init__()  

        self.geneo_class = geneo_class
        self.kernel_reach = kernel_reach
        
        if len(kwargs) > 0:
            self.init_from_kwargs(**kwargs)
        else:
            self.init_from_config()

    
    def init_from_config(self):

        config = self.geneo_class.geneo_random_config()

        self.geneo_params = {}
        for param in config['geneo_params']:
            if isinstance(config['geneo_params'][param], torch.Tensor):
                t_param = config['geneo_params'][param].to(dtype=torch.float)
            else:
                t_param = torch.tensor(config['geneo_params'][param], dtype=torch.float)
            t_param = nn.Parameter(t_param, requires_grad = not param in config['non_trainable'])
            self.geneo_params[param] = t_param

        self.geneo_params = nn.ParameterDict(self.geneo_params)


    def init_from_kwargs(self, **kwargs):
        self.geneo_params = {}
        for param in self.geneo_class.mandatory_parameters():
            self.geneo_params[param] = nn.Parameter(torch.tensor(kwargs[param], dtype=torch.float))

        self.geneo_params = nn.ParameterDict(self.geneo_params)

    def compute_kernel(self) -> torch.Tensor:
       """
       TODO
       """

    def forward(self, x:torch.Tensor) -> torch.Tensor:
      """
      TODO
      """



###############################################################
#                         SCENE-Nets                          #
###############################################################

class GENEO_Layer(nn.Module):

    def __init__(self, gib_dict:dict, kernel_reach:int, num_observers:int=1):
        """
        Instantiates a GENEO-Layer Module with specific GENEOs and their respective cvx coefficients.

        Parameters
        ----------
        `gib_dict` - dict:
            Mappings that contain the number of GENEOs of each kind (the key) to initialize

        `kernel_reach` - int:
            The kernel's neighborhood reach in Geometric space.

        `num_observers` - int:
            number os GENEO observers to form the output of the Module
        
        """
        super(GENEO_Layer, self).__init__()
        
        if gib_dict is None or gib_dict == {}:
            geneo_keys = ['cy', 'arrow', 'cone', 'neg', 'disk', 'ellip']
            self.gib_dict = {
                g : torch.randint(1, 64, (1,))[0] for g in geneo_keys
            }
        else:
            self.gib_dict = gib_dict

        self.num_observers = num_observers

        self.gibs:Mapping[str, GIB_Operator] = nn.ModuleDict()

        # --- Initializing GENEOs ---
        for key in self.gib_dict:
            if key == 'cy':
                g_class = cylinder.Cylinder
            elif key == 'arrow':
                g_class = arrow.arrow
            elif key == 'neg':
                g_class = neg_sphere.negSpherev2
            elif key == 'disk':
                g_class = disk.Disk
            elif key == 'cone':
                g_class = cone.Cone
            elif g_class == 'ellip':
                g_class = ellipsoid.Ellipsoid

            for i in range(self.gib_dict[key]):
                self.gibs[f'{key}_{i}'] = GIB_Operator(g_class, kernel_reach=kernel_reach)

        # --- Initializing Convex Coefficients ---
        self.lambdas = torch.rand((num_observers, len(self.gibs)))
        self.lambdas = torch.softmax(self.lambdas, dim=1) # make sure they sum to 1
        self.lambdas = nn.Parameter(self.lambdas, requires_grad=True)   


    def maintain_convexity(self):
        self.lambdas = torch.softmax(self.lambdas, dim=1)

    def get_cvx_coefficients(self) -> torch.Tensor:
        return self.lambdas

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        conv = self._perform_conv(x)

        conv_pred = self._observer_cvx_combination(conv)
        conv_pred = torch.relu(torch.tanh(conv_pred)) # (batch, num_observers, z, x, y)

        return conv_pred
    



if __name__ == "__main__":
    pass