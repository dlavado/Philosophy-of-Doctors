
import torch
import torch.nn as nn
from typing import Mapping, Tuple, Union

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.models.GENEONets.geneos.GIB_Stub import GIB_Stub, NON_TRAINABLE, GIB_PARAMS, to_parameter, to_tensor
from core.models.GENEONets.geneos import cylinder, disk, cone, ellipsoid
from core.models.GENEONets.GIBLi_utils import PointBatchNorm
from core.unpooling.nearest_interpolation import interpolation

###############################################################
#                          GIB Layer                          #
###############################################################

class GIB_Operator(nn.Module):

    def __init__(self, gib_class:GIB_Stub, kernel_reach:float=None, **kwargs):
        super(GIB_Operator, self).__init__()  

        self.gib_class = gib_class
        self.kernel_reach = kernel_reach
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if len(kwargs) > 0:
            self.init_from_kwargs(kwargs)
        else:
            self.random_init()

        self.gib = self.gib_class(kernel_reach=self.kernel_reach, **self.gib_params)
    
    def random_init(self):
        config = self.gib_class.gib_random_config(self.kernel_reach)
        self.gib_params = {}

        for param_name in config[GIB_PARAMS]:
            t_param = to_tensor(config[GIB_PARAMS][param_name])
            self.gib_params[param_name] = nn.Parameter(t_param, requires_grad = not param_name in config[NON_TRAINABLE])

        self.gib_params = nn.ParameterDict(self.gib_params)


    def init_from_kwargs(self, kwargs):
        self.gib_params = {}
        for param_name in self.gib_class.mandatory_parameters():
            self.gib_params[param_name] = to_parameter(kwargs[param_name])

        self.gib_params = nn.ParameterDict(self.gib_params)


    def forward(self, points:torch.Tensor, q_coords:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the GIB on the query points given the support points.

        Parameters
        ----------
        `points` - torch.Tensor:
            Tensor of shape (N, 3) representing the point cloud.

        `q_coords` - torch.Tensor:
            Tensor of shape (M, 3) representing the query points.

        `supports_idxs` - torch.Tensor[int]:
            Tensor of shape (M, K) representing the indices of the support points for each query point. With K <= N.

        Returns
        -------
        `q_output` - torch.Tensor:
            Tensor of shape (M,) representing the output of the GIB on the query points.
        """
        return self.gib(points, q_coords, support_idxs)


class GIB_Layer(nn.Module):

    def __init__(self, gib_dict:dict, kernel_reach:int, num_observers:int=1):
        """
        Instantiates a GIB-Layer Module with GIBs and their cvx coefficients.

        Parameters
        ----------
        `gib_dict` - dict[str, int]:
            Mappings that contain the number of GIBs of each kind (the key) to initialize;
            keys \in ['cy', 'cone', 'disk', 'ellip']

        `kernel_reach` - int:
            The kernel's neighborhood reach in Geometric space.

        `num_observers` - int:
            number os GENEO observers to form the output of the Module
        
        """
        super(GIB_Layer, self).__init__()
        
        if gib_dict is None or gib_dict == {}:
            geneo_keys = ['cy', 'cone', 'disk', 'ellip']
            self.gib_dict = {
                g : torch.randint(1, 64, (1,))[0] for g in geneo_keys
            }
        else:
            self.gib_dict = gib_dict

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_observers = num_observers

        self.gibs:Mapping[str, GIB_Operator] = nn.ModuleDict()

        # --- Initializing GIBs ---
        for key in self.gib_dict:
            if key == 'cy':
                g_class = cylinder.Cylinder
            elif key == 'disk':
                g_class = disk.Disk
            elif key == 'cone':
                g_class = cone.Cone
            elif g_class == 'ellip':
                g_class = ellipsoid.Ellipsoid

            for i in range(self.gib_dict[key]):
                self.gibs[f'{key}_{i}'] = GIB_Operator(g_class, kernel_reach=kernel_reach).to(self.device)


        # --- Initializing Convex Coefficients ---
        self.lambdas = torch.randn((len(self.gibs), num_observers), device=self.device) # shape (num_gibs, num_observers)
        self.maintain_convexity() # make sure the coefficients are convex
        self.lambdas = to_parameter(self.lambdas)

    def maintain_convexity(self):
        self.lambdas = torch.softmax(self.lambdas, dim=0)

    def get_cvx_coefficients(self) -> torch.Tensor:
        return self.lambdas

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    def _compute_gib_outputs(self, points:torch.Tensor, q_points:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:

        if points.dim() == 2:
            points = points.unsqueeze(0)
            q_points = q_points.unsqueeze(0)
            support_idxs = support_idxs.unsqueeze(0)
            batched = False
        else:
            batched = True

        print(f"{points.shape=}, {q_points.shape=}, {support_idxs.shape=}")


        q_outputs = torch.zeros((points.shape[0], q_points.shape[1], len(self.gibs)), dtype=points.dtype, device=points.device) # shape (B, M, num_gibs)
        # print(f"{q_outputs.shape=}")
        for i, gib_key in enumerate(self.gibs):
            gib_output = self.gibs[gib_key](points, q_points, support_idxs) # shape: ([B], M)
            # print(f"{gib_output.shape=}")
            q_outputs[:, :, i] = gib_output
            

        if not batched:
            q_outputs = q_outputs.squeeze(0)

        return q_outputs
    
    def _compute_observers(self, q_outputs:torch.Tensor) -> torch.Tensor:
        # --- Convex Combination ---
        # for each query point, compute the convex combination of the outputs of the GIBs
        return q_outputs @ self.lambdas # shape (M, num_gibs) @ (num_gibs, num_observers) = (M, num_observers)
    
    
    def forward(self, points:torch.Tensor, q_coords:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the GIB-Layer on the query points given the support points.

        Parameters
        ----------
        `points` - torch.Tensor:
            Tensor of shape (N, 3) representing the point cloud.

        `q_coords` - torch.Tensor:
            Tensor of shape (M, 3) representing the query points; M <= N.

        `supports_idxs` - torch.Tensor[int]:
            Tensor of shape (M, K) representing the indices of the support points for each query point. With K <= N.

        Returns
        -------
        `q_outputs` - torch.Tensor:
            Tensor of shape (M, num_observers) representing the output of the GIB-Layer on the query points.
        """
        q_outputs = self._compute_gib_outputs(points, q_coords, support_idxs) # shape (M, num_gibs)
        q_outputs = self._compute_observers(q_outputs) # shape (M, num_observers)
        return q_outputs
    

    def forward_batch(self, points:torch.Tensor, q_coords:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the GIB-Layer on the query points given the support points.

        Parameters
        ----------
        `points` - torch.Tensor:
            Tensor of shape (B, N, 3) representing the batch of point clouds.

        `q_coords` - torch.Tensor:
            Tensor of shape (B, M, 3) representing the batch of query points; M <= N.

        `supports_idxs` - torch.Tensor[int]:
            Tensor of shape (B, M, K) representing the indices of the support points for each query point. With K <= N.

        Returns
        -------
        `q_outputs` - torch.Tensor:
            Tensor of shape (B, M, num_observers) representing the output of the GIB-Layer on the query points.
        """
        B, M, _ = q_coords.shape
        q_outputs = torch.zeros((B, M, len(self.gibs)), dtype=points.dtype, device=points.device)
        
        for i, gib_key in enumerate(self.gibs):
            for b in range(B):
                q_outputs[b, :, i] = self.gibs[gib_key](points[b], q_coords[b], support_idxs[b])

        q_outputs = q_outputs @ self.lambdas
        return q_outputs
    
    

###############################################################
#                        GIBLi Blocks                         #
###############################################################

class GIB_Block(nn.Module):

    def __init__(self, gib_dict, feat_channels, num_observers, kernel_size, out_channels=None) -> None:
        super(GIB_Block, self).__init__()

        self.gib = GIB_Layer(gib_dict, kernel_size, num_observers)
        self.gib_norm = PointBatchNorm(num_observers)

        if out_channels is None:
            out_channels = feat_channels + num_observers

        self.mlp = nn.Linear(feat_channels + num_observers, out_channels, bias=False)
        self.mlp_norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)


    def forward(self, points, q_points, neighbor_idxs) -> torch.Tensor:
        coords, feats = points

        gib_out = self.gib(coords, q_points, neighbor_idxs) # (B, Q, num_observers)
        gib_out = self.act(self.gib_norm(gib_out)) # (B, Q, num_observers)

        print(f"{feats.shape=}, {gib_out.shape=}")
        mlp_out = torch.cat([feats, gib_out], dim=-1) # (B, Q, out_channels)
        mlp_out = self.mlp(mlp_out) # (B, Q, out_channels)
        mlp_out = self.act(self.mlp_norm(mlp_out)) # (B, Q, out_channels)

        return mlp_out # (B, Q, out_channels)
    

class GIB_Sequence(nn.Module):

    def __init__(self, 
                 num_layers, 
                 gib_dict, 
                 feat_channels, 
                 num_observers:Union[int, list],
                 kernel_size,
                 out_channels:Union[int, list]=None
                ) -> None:
        
        super(GIB_Sequence, self).__init__()

        self.gib_blocks = nn.ModuleList()
        self.feat_channels = feat_channels
        self.num_layers = num_layers

        if isinstance(num_observers, int):
            num_observers = [num_observers] * num_layers
        
        out_channels = feat_channels if out_channels is None else out_channels
        if isinstance(out_channels, int):
            out_channels = [out_channels] * num_layers
        
        
        for i in range(num_layers):
            num_obs = num_observers[i]
            out_c = out_channels[i]
                
            gib_block = GIB_Block(gib_dict, feat_channels, num_obs, kernel_size, out_c)
            self.gib_blocks.append(gib_block)

            feat_channels = out_c

    def forward(self, points, q_coords, neighbor_idxs) -> torch.Tensor:
        coords, feats = points

        for gib_block in self.gib_blocks:
            # 1st iteration: x = (coords, feats); 2nd iteration: x = (coords, new_feats); ...
            # 1st iteration feats.shape = (B, N, F) or (B, N, out_channels[0]); 2nd iteration feats.shape = (B. N, F) or (B, N, out_channels[1])
            feats = gib_block((coords, feats), q_coords, neighbor_idxs)

        return feats # shape
    

###############################################################
#                       Unpooling Blocks                     #
###############################################################


class Unpool_wSkip(nn.Module):
    
    def __init__(self,  
                feat_channels,
                skip_channels,
                out_channels,
                bias=True,
                skip=True,
                backend="max") -> None:
    
        super(Unpool_wSkip, self).__init__()

        self.skip = skip
        self.backend = backend
        assert self.backend in ["max", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(feat_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, curr_points: Tuple[torch.Tensor, torch.Tensor], skip_points: Tuple[torch.Tensor, torch.Tensor], upsampling_idxs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        `curr_points` - Tuple[torch.Tensor, torch.Tensor]:
            Tuple containing two tensors:
                - coords: Tensor of shape (B, M, 3) representing the input coordinates of the current layer
                - feats: Tensor of shape (B, M, C) representing the input features of the current layer

        `skip_points` - Tuple[torch.Tensor, torch.Tensor]:
            Tuple containing two tensors:
                - coords: Tensor of shape (B, N, 3) representing the input coordinates of the skip connection
                - feats: Tensor of shape (B, N, C) representing the input features of the skip connection

        `upsampling_idxs` - torch.Tensor[int]:
            Tensor of shape (B, N, K) representing the indices of the curr_points that are neighbors of the skip_points

        Returns
        -------
        `output` - torch.Tensor:
            Tensor of shape (B, N, 3 + 2*out_channels) representing the output of the Unpooling Layer
        """

        # Extract the coordinates and features from the input points
        curr_coords, curr_feat = curr_points
        skip_coords, skip_feat = skip_points

        curr_points = torch.cat([curr_coords, curr_feat], dim=-1) # (B, M, 3 + C)
        skip_points = torch.cat([skip_coords, skip_feat], dim=-1) # (B, N, 3 + C)

        if self.backend == 'interp':
            inter_feats = interpolation(curr_points, skip_points, upsampling_idxs)
        else: #use max pooling
            B, M, _ = curr_points.shape
            B, N, K = upsampling_idxs.shape
            C = curr_points.shape[-1] - 3
            inter_feats = torch.gather(
                curr_feat.unsqueeze(1).expand(B, N, M, C),  # Expand current features to (B, N, M, C)
                2, upsampling_idxs.unsqueeze(-1).expand(B, N, K, C)  # Gather features (B, N, K, C)
            ) # shape (B, N, K, C)
            inter_feats = torch.max(inter_feats, dim=2)[0] # shape (B, N, C) # max pooling
        
        inter_feats = self.proj(inter_feats) # (B, N, out_channels)
        
        if self.skip:
            # summing or concatenating the skip features? Herein lies the question
            inter_feats = torch.cat([inter_feats, self.proj_skip(skip_feat)], dim=-1) # (B, N, 2*out_channels)

        return torch.cat([skip_coords, inter_feats], dim=-1) # (B, N, 3 + 2*out_channels)

        
        



if __name__ == "__main__":
    pass