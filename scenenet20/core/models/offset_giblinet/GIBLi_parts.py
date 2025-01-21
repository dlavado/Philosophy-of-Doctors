
import torch
import torch.nn as nn
from typing import Mapping, Tuple, Union, List, Dict

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.models.giblinet.geneos.GIB_Stub import GIB_Stub, GIBCollection, NON_TRAINABLE, GIB_PARAMS, to_parameter, to_tensor
from core.models.giblinet.geneos import cylinder, disk, cone, ellipsoid
from core.models.giblinet.GIBLi_utils import PointBatchNorm
from core.models.giblinet.unpooling.nearest_interpolation import interpolation
from core.models.giblinet.pooling.pooling import local_pooling
from core.models.giblinet.conversions import compute_centered_support_points
from core.models.giblinet.geneos.diff_rotation_transform import rotate_points_batch


@torch.jit.script
def rotate(points:torch.Tensor, angles:torch.Tensor) -> torch.Tensor:
    """
    Rotate a tensor along the x, y, and z axes by the angles of each GIB in the collection.
    
    Parameters
    ----------
    `points` - torch.Tensor:
        Tensor of shape (N, 3) representing the 3D points to rotate.
        
    `angles` - torch.Tensor:
        Tensor of shape (G, 3) containing rotation angles for the x, y, and z axes for each GIB in the collection.
        These are normalized in the range [-1, 1] and represent angles_normalized = angles
        
    Returns
    -------
    `points` - torch.Tensor:
        Tensor of shape (G, N, 3) containing the rotated
    """
    # convert self.angles to an acceptable range [0, 2]:
    # this is equivalent to angles = self.angles % 2
    angles = torch.fmod(angles, 2) # rotations higher than 2\pi are equivalent to rotations within 2\pi
    # angles = angles + (angles < 0).float() * 2 
    
    angles = 2 - torch.relu(-angles) # convert negative angles to positive
    return rotate_points_batch(angles, points)

###############################################################
#                          GIB Layer                          #
###############################################################

class GIB_Operator_Collection(nn.Module):
    
    def __init__(self, gib_class:GIBCollection, num_gibs, kernel_reach:float=None, **kwargs):
        super(GIB_Operator_Collection, self).__init__()
        
        self.gib_class = gib_class
        self.kernel_reach = kernel_reach
        self.num_gibs = num_gibs
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if len(kwargs) > 0:
            self.gib_params = self.init_from_kwargs(kwargs)
        else:
            self.gib_params = self.random_init(num_gibs, kernel_reach)
            
        self.gib:GIBCollection = torch.jit.script(self.gib_class(kernel_reach=self.kernel_reach, num_gibs=num_gibs, **self.gib_params))
        
        
    def random_init(self, num_gibs, kernel_reach):
        config = self.gib_class.gib_random_config(num_gibs, kernel_reach)
        gib_params = {}

        for param_name in config[GIB_PARAMS]:
            t_param = to_tensor(config[GIB_PARAMS][param_name]).contiguous()
            gib_params[param_name] = nn.Parameter(t_param, requires_grad = not param_name in config[NON_TRAINABLE])

        return nn.ParameterDict(gib_params)


    def init_from_kwargs(self, kwargs):
        gib_params = {}
        for param_name in self.gib_class.mandatory_parameters():
            gib_params[param_name] = to_parameter(kwargs[param_name])

        return nn.ParameterDict(gib_params)
    
    def get_gib_params(self) -> List[torch.Tensor]:
        params = []
        for param in self.gib_class.mandatory_parameters():
            params.append(self.gib_params[param])
        return params
    
    def _prepped_forward(self, s_centered:torch.Tensor, valid_mask:torch.Tensor, batched:bool, mc_points:torch.Tensor) -> torch.Tensor:
        return self.gib._prepped_forward(s_centered, valid_mask, batched, mc_points)
        
        
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
            Tensor of shape (M, G) representing the output of the GIB on the query points, where G is the number of GIBs.
        """
        return self.gib(points, q_coords, support_idxs)



class GIB_Layer_Coll(nn.Module):

    def __init__(self, 
                 gib_dict:Dict[str, int], 
                 kernel_reach:int, 
                 num_observers:int=1):
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
        super(GIB_Layer_Coll, self).__init__()
        
        if gib_dict is None or gib_dict == {}:
            geneo_keys = ['cy', 'cone', 'disk', 'ellip']
            self.gib_dict = {
                g : torch.randint(1, 64, (1,))[0] for g in geneo_keys
            }
        else:
            self.gib_dict = gib_dict

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_observers = num_observers
        self.total_gibs = 0
        
        self.gibs:Dict[str, GIB_Operator_Collection] = nn.ModuleDict()

        # --- Initializing GIBs ---
        for key in self.gib_dict:
            if key == 'cy':
                g_class = cylinder.CylinderCollection
            elif key == 'disk':
                g_class = disk.DiskCollection
            elif key == 'cone':
                g_class = cone.ConeCollection
            elif key == 'ellip':
                g_class = ellipsoid.EllipsoidCollection
            else:
                raise ValueError(f"Invalid GIB Type: `{key}`, must be one of ['cy', 'cone', 'disk', 'ellip']")
            
            
            num_gibs = self.gib_dict[key]
            if num_gibs > 0:
                self.gibs[key] = torch.jit.script(GIB_Operator_Collection(g_class, num_gibs=num_gibs, kernel_reach=kernel_reach))
                self.total_gibs += num_gibs
                
          
        # angles for each GIB in the layer; each GIB has 3 angles for rotation along the x, y, and z axes   
        self.angles = nn.Parameter(torch.randn((self.total_gibs, 3)) * 0.01)

        # --- Initializing Convex Coefficients ---
        if num_observers > 1:
            self.lambdas = torch.randn((self.total_gibs, num_observers)) # shape (num_gibs, num_observers)
            self.maintain_convexity() # make sure the coefficients are convex
            self.lambdas = to_parameter(self.lambdas)
        elif num_observers == 0: # no cvx comb, only a linear combination of the GIBs
            self.lambdas = nn.Linear(self.total_gibs, self.total_gibs, bias=False)
        else: # no observers, only the output of the GIBs
            self.lambdas = to_tensor(0.0)   

    def maintain_convexity(self):
        if self.num_observers > 1:
            self.lambdas = to_parameter(torch.softmax(self.lambdas, dim=0))

    def get_cvx_coefficients(self) -> torch.Tensor:
        return self.lambdas
    
    def get_gib_params(self) -> List[torch.Tensor]:
        params = []
        for _, gib_coll in self.gibs.items():
            params.extend(gib_coll.get_gib_params())
        return params

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    def apply_rotations(self, s_centered:torch.Tensor) -> torch.Tensor:
        # self.angles = None # for testing purposes
        if self.angles is not None:
            s_centered = rotate(s_centered, self.angles) # (B, M, G, K, 3), where G is the number of GIBs, we rotate each GIB separately according to their respective angles
        else:
            s_centered = s_centered.unsqueeze(2).expand(-1, -1, self.total_gibs, -1, -1) # (B, M, G, K, 3), expand the tensor to maintain the same shape as the angles tensor
            
        return s_centered.contiguous()
    
    
    def _compute_gib_outputs(self, coords: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor, mc_points:torch.Tensor) -> torch.Tensor:
        # Ensure tensors are batched
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            q_points = q_points.unsqueeze(0)
            support_idxs = support_idxs.unsqueeze(0)

        # Prepare support vectors
        s_centered, valid_mask, batched = compute_centered_support_points(coords, q_points, support_idxs)
        s_centered = self.apply_rotations(s_centered) # (B, M, G, K, 3)

        ###### time efficient sol; mem inefficient due to empty alloc ######
        q_outputs = torch.empty((q_points.shape[0], q_points.shape[1], self.total_gibs), device=coords.device).contiguous() # shape (B, M, G)

        offset = 0
        for gib_coll in self.gibs.values():
            gib_output_dim = gib_coll.num_gibs
            support = s_centered[:, :, offset : offset + gib_output_dim] # (B, M, G_i, K, 3)
            mc = mc_points[offset : offset + gib_output_dim] # (G_i, num_samples, 3)
            gib_outputs = gib_coll._prepped_forward(support, valid_mask, batched, mc)  # ([B], Q, output_dim)
            q_outputs[:, :, offset : offset + gib_output_dim] = gib_outputs
            offset += gib_output_dim

        if not batched:
            q_outputs = q_outputs.squeeze(0)

        return q_outputs
    
    
    def _jit_compute_gib_outputs(self, coords: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor, mc_points) -> torch.Tensor:
        
        # Ensure tensors are batched
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            q_points = q_points.unsqueeze(0)
            support_idxs = support_idxs.unsqueeze(0)

        # Prepare support vectors
        s_centered, valid_mask, batched = compute_centered_support_points(coords, q_points, support_idxs)
        s_centered = self.apply_rotations(s_centered)
        
        # futures will store the fork calls for each GIB in the Layer
        futures : List[torch.jit.Future[torch.Tensor]] = []
        offsets : List[int] = []
        
        offset = 0
        for gib_coll in self.gibs.values():
            gib_output_dim = gib_coll.num_gibs
            support = s_centered[:, :, offset : offset + gib_output_dim]  # (B, M, G_i, K, 3)
            mc = mc_points[offset : offset + gib_output_dim]  # (G_i, num_samples, 3)
            futures.append(torch.jit.fork(gib_coll.gib._prepped_forward, support, valid_mask, batched, mc))
            offsets.append(gib_output_dim)
            offset += gib_output_dim
            
        offset = 0
        q_outputs = torch.empty((q_points.shape[0], q_points.shape[1], self.total_gibs), device=coords.device).contiguous() # shape (B, M, G)
        for future, gib_output_dim in zip(futures, offsets):
            gib_outputs = torch.jit.wait(future)  # Retrieve the computed result
            q_outputs[:, :, offset : offset + gib_output_dim] = gib_outputs
            offset += gib_output_dim
            
        return q_outputs
        
    
    def _compute_observers(self, q_outputs:torch.Tensor) -> torch.Tensor:
        # --- Convex Combination ---
        # for each query point, compute the convex combination of the outputs of the GIBs
        return q_outputs @ self.lambdas # shape (M, num_gibs) @ (num_gibs, num_observers) = (M, num_observers)
    
    
    def forward(self, points:torch.Tensor, q_coords:torch.Tensor, support_idxs:torch.Tensor, mc_points) -> torch.Tensor:
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
            
        `mc_points` - torch.Tensor:
            Tensor of shape (None, Big_N, 3) representing the Monte Carlo points for integral approximation.

        Returns
        -------
        `q_outputs` - torch.Tensor:
            Tensor of shape (M, num_observers) representing the output of the GIB-Layer on the query points.
        """
        q_outputs = self._compute_gib_outputs(points, q_coords, support_idxs, mc_points) # shape (M, num_gibs)
        # q_outputs = self._jit_compute_gib_outputs(points, q_coords, support_idxs, mc_points) # shape (M, num_gibs)
        if self.num_observers > 1:
            q_outputs = self._compute_observers(q_outputs) # shape (M, num_observers)
        return q_outputs
    

###############################################################
#                        GIBLi Blocks                         #
###############################################################

class GIB_Block(nn.Module):

    def __init__(self, 
                 gib_dict:Dict[str, int], 
                 feat_channels:int,
                 num_observers:int,
                 kernel_size:int,
                 out_channels=None, 
                 strided=False
                ) -> None:
        super(GIB_Block, self).__init__()
        self.gib = torch.jit.script(GIB_Layer_Coll(gib_dict, kernel_size, num_observers))
        num_observers = self.gib.total_gibs if num_observers == -1 else num_observers
        self.gib_norm = PointBatchNorm(num_observers)
        self.strided = strided

        if out_channels is None:
            out_channels = feat_channels + num_observers

        self.mlp = nn.Linear(feat_channels + num_observers, out_channels, bias=False)
        self.mlp_norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def maintain_convexity(self):
        self.gib.maintain_convexity()
        
    def get_cvx_coefficients(self) -> torch.Tensor:
        return self.gib.get_cvx_coefficients()
    
    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gib.get_gib_params()

    def forward(self, points, q_points, neighbor_idxs, mc_points) -> torch.Tensor:
        """
        Parameters
        ----------

        `points` - Tuple[torch.Tensor, torch.Tensor]:
            Tuple containing two tensors:
                - coords: Tensor of shape (B, N, 3) representing the point cloud
                - feats: Tensor of shape (B, N, F) representing the features of the point cloud

        `q_points` - torch.Tensor:
            Tensor of shape (B, Q, 3) representing the query points

        `neighbor_idxs` - torch.Tensor[int]:
            Tensor of shape (B, Q, K) representing the indices of the neighbors of each query point in the tensor `coords`
            
        `mc_points` - torch.Tensor:
            Tensor of shape (None, Big_N, 3) representing the Monte Carlo points for integral approximation.
        """
        coords, feats = points

        gib_out = self.gib(coords, q_points, neighbor_idxs, mc_points) # (B, Q, num_observers)
        
        gib_out = self.act(self.gib_norm(gib_out)) # (B, Q, num_observers)

        if self.strided: # if the query poin # if torch.isnan(out).any():
            feats = local_pooling(feats, neighbor_idxs) # (B, Q, feat_channels)

        # print(f"{feats.shape=}, {gib_out.shape=}")
        mlp_out = torch.cat([feats, gib_out], dim=-1) # (B, Q, out_channels)
        mlp_out = self.mlp(mlp_out) # (B, Q, out_channels)
        mlp_out = self.act(self.mlp_norm(mlp_out)) # (B, Q, out_channels)

        return mlp_out # (B, Q, out_channels)
    

class GIB_Sequence(nn.Module):

    def __init__(self, 
                 num_layers:int, 
                 gib_dict:Dict[str, int], 
                 feat_channels:int, 
                 num_observers:Union[int, list],
                 kernel_size:float,
                 out_channels:Union[int, list]=None,
                 strided:bool=False,
                ) -> None:
        
        super(GIB_Sequence, self).__init__()
    

        self.gib_blocks = nn.ModuleList()
        self.feat_channels = feat_channels
        self.num_layers = num_layers

        if isinstance(num_observers, int):
            num_observers = [num_observers] * num_layers
            
        total_gibs = sum([gib_dict[k] for k in gib_dict])    
        if num_observers == -1: # if -1, then the number of observers is equal to the number of GIBs
            num_observers = total_gibs
        
        out_channels = feat_channels + num_observers if out_channels is None else out_channels
        if isinstance(out_channels, int):
            out_channels = [out_channels] * num_layers
        
        for i in range(num_layers):
            num_obs = num_observers[i]
            out_c = out_channels[i]
            if i > 0:
                strided = False # only the first layer of the Sequence is strided
            gib_block = GIB_Block(gib_dict, feat_channels, num_obs, kernel_size, out_c, strided)
            self.gib_blocks.append(gib_block)

            feat_channels = out_c
            
        # --- Initializing Monte Carlo Points ---
        # These points are used for integral approximation for each GIB in the Sequence.
        num_samples = 1000 # number of Monte Carlo points
        self.montecarlo_points = torch.rand((num_samples, 3), device='cuda') * 2 * kernel_size - kernel_size # \in [-kernel_reach, kernel_reach]
        self.montecarlo_points = self.montecarlo_points[torch.norm(self.montecarlo_points, dim=-1) <= kernel_size]
        self.montecarlo_points = self.montecarlo_points.unsqueeze(0).expand(total_gibs, -1, -1) # shape (1, num_samples, 3)
        # this is not compatible with model saving/loading due to the random shape of the tensor
        # self.montecarlo_points = to_parameter(self.montecarlo_points, requires_grad=False) # shape (num_samples, 3)


    def maintain_convexity(self):
        for gib_block in self.gib_blocks:
            gib_block.maintain_convexity()
            
    def get_gib_params(self) -> List[torch.Tensor]:
        params = []
        for gib_block in self.gib_blocks:
            params.extend(gib_block.get_gib_params())
        return params
            
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return [gib_block.get_cvx_coefficients() for gib_block in self.gib_blocks]

    def forward(self, points, q_coords, neighbor_idxs) -> torch.Tensor:
        coords, feats = points
        
        for gib_block in self.gib_blocks:
            # 1st it: feats = (B, Q, feat_channels + num_observers)
            feats = gib_block((coords, feats), q_coords, neighbor_idxs, self.montecarlo_points)

        return feats # shape (B, Q, out_channels)
    

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
                concat:bool=True,
                backend="max") -> None:
    
        super(Unpool_wSkip, self).__init__()

        self.skip = skip
        self.backend = backend
        self.concat = concat
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

    def forward(self, curr_points: Tuple[torch.Tensor, torch.Tensor], 
                skip_points: Tuple[torch.Tensor, torch.Tensor], 
                upsampling_idxs: torch.Tensor,
            ) -> torch.Tensor:
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
            
        `concat` - bool:
            Whether to concatenate the skip features to the interpolated features or to sum them.
            If sum, the skip_fea

        Returns
        -------
        `output` - torch.Tensor:
            Tensor of shape (B, N, 3 + 2*out_channels) representing the output of the Unpooling Layer
        """

        # Extract the coordinates and features from the input points
        curr_coords, inter_feats = curr_points
        skip_coords, skip_feats = skip_points

        curr_points = torch.cat([curr_coords, inter_feats], dim=-1).to(curr_coords.dtype) # (B, M, 3 + C)
        skip_points = torch.cat([skip_coords, skip_feats], dim=-1).to(skip_coords.dtype) # (B, N, 3 + C)
        

        if self.backend == 'interp':
            # print(f"{curr_points.shape=}, {skip_points.shape=}, {upsampling_idxs.shape=}")
            inter_feats = interpolation(curr_points, skip_points, upsampling_idxs)
        else: #use max pooling
            B, M, _ = curr_points.shape
            B, N, K = upsampling_idxs.shape
            C = curr_points.shape[-1] - 3
            inter_feats = torch.gather(
                inter_feats.unsqueeze(1).expand(B, N, M, C),  # Expand current features to (B, N, M, C)
                2, upsampling_idxs.unsqueeze(-1).expand(B, N, K, C)  # Gather features (B, N, K, C)
            ) # shape (B, N, K, C)
            inter_feats = torch.max(inter_feats, dim=2)[0] # shape (B, N, C) # max pooling
            

        # print(f"{curr_points.dtype=}, {skip_points.dtype=}, {upsampling_idxs.dtype=} {inter_feats.dtype=}")
        inter_feats = self.proj(inter_feats).to(inter_feats.dtype) # (B, N, out_channels)
       
        
        if self.skip:
            # summing or concatenating the skip features? Herein lies the question
            skip_feats = self.proj_skip(skip_feats).to(inter_feats.dtype) # (B, N, out_channels)
            if self.concat:
                inter_feats = torch.cat([skip_feats, inter_feats], dim=-1) # (B, N, 2*out_channels)
            else:
                inter_feats = inter_feats + skip_feats

        return torch.cat([skip_coords, inter_feats], dim=-1) # (B, N, 3 + 2*out_channels) or (B, N, 3 + out_channels)
    
    
    
    
    
class Decoder(nn.Module):
    
    def __init__(self, 
                feat_channels,
                skip_channels,
                out_channels,
                bias=True,
                skip=True,
                concat:bool=True,
                backend="max",
                num_layers=1, 
                gib_dict:Dict[str, int]={}, 
                num_observers=16,
                kernel_size=0.2,
            ) -> None:
        super(Decoder, self).__init__()
        
        self.unpool = Unpool_wSkip(feat_channels, skip_channels, out_channels, bias, skip, concat, backend)
        f_channels = out_channels*2 if concat else out_channels
        self.gib_seq = GIB_Sequence(num_layers, gib_dict, f_channels, num_observers, kernel_size, out_channels, strided=False)
        
    
    def maintain_convexity(self):
        self.gib_seq.maintain_convexity()
        
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.gib_seq.get_cvx_coefficients()
    
    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gib_seq.get_gib_params()
   
        
    def forward(self, curr_points, skip_points, upsampling_idxs, skip_neigh_idxs):
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
            
        `skip_neigh_idxs` - torch.Tensor[int]:
            Tensor of shape (B, N, K) representing the indices of the skip_points that are neighbors of the skip_points

        Returns
        -------
        `output` - torch.Tensor:
            Tensor of shape (B, N, out_channels) representing the output of the Decoder
        """
        out = self.unpool(curr_points, skip_points, upsampling_idxs) # output shape: (B, N, 3 + 2*out_channels)
        
        # if torch.isnan(out).any():
        #     print("Decoder")
        #     print(f"{out=}")
        #     print(f"{curr_points[0].shape=}, {skip_points[0].shape=}, {upsampling_idxs.shape=}")
        #     print(f"{curr_points[1].shape=}, {skip_points[1].shape=}")
        #     print(f"{out.shape=}")
        
        # intermediate features not used to save memory   
        out = self.gib_seq((out[..., :3], out[..., 3:]), out[..., :3], skip_neigh_idxs) # output shape: (B, N, out_channels)
        
        return out # (B, N, out_channels)
    

        
        



if __name__ == "__main__":
    pass