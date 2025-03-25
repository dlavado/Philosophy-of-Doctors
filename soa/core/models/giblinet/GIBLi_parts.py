
import torch
import torch.nn as nn
from typing import Mapping, Tuple, Union, List, Dict

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')

from core.models.giblinet.geneos.GIB_Stub import GIB_Stub, GIBCollection, NON_TRAINABLE, GIB_PARAMS, to_parameter, to_tensor
from core.models.giblinet.geneos import cylinder, disk, cone, ellipsoid
from core.models.giblinet.GIBLi_utils import PointBatchNorm, print_gpu_memory
from core.models.giblinet.unpooling.nearest_interpolation import interpolation
from core.models.giblinet.pooling.pooling import local_pooling
from core.models.giblinet.conversions import compute_centered_support_points
from core.models.giblinet.geneos.diff_rotation_transform import rotate



###############################################################
#                        GIBLi Utils                          #
###############################################################



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
        # self.gib:GIBCollection = self.gib_class(kernel_reach=self.kernel_reach, num_gibs=num_gibs, **self.gib_params)
        
        
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
    
    
    @torch.jit.export
    def _prepped_forward(self, s_centered:torch.Tensor, valid_mask:torch.Tensor, mc_points:torch.Tensor) -> torch.Tensor:
        return self.gib._prepped_forward(s_centered, valid_mask, mc_points)
        
        
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
        
class GIB_Layer(nn.Module): 

    def __init__(self, gib_dict:Dict[str, int], kernel_reach:int, num_observers:int=1):
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

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_observers = num_observers

        self.gibs:Mapping[str, GIB_Operator] = nn.ModuleDict()
        
        self.total_gibs = 0
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
                self.gibs[f'{key}_{i}'] = GIB_Operator(g_class, kernel_reach=kernel_reach)#.to(self.device)
                self.total_gibs += 1

        
         # --- Initializing Convex Coefficients ---
        if num_observers > 1:
            self.lambdas = torch.randn((self.total_gibs, num_observers)) # shape (num_gibs, num_observers)
            self.maintain_convexity() # make sure the coefficients are convex
            self.lambdas = to_parameter(self.lambdas)
        elif num_observers == 0: # no cvx comb, only a linear combination of the GIBs
            self.lambdas = nn.Linear(self.total_gibs, self.total_gibs, bias=False)
        else: # no observers, only the output of the GIBs
            self.lambdas = to_tensor(1.0)   

    def maintain_convexity(self):
        self.lambdas = to_parameter(torch.softmax(self.lambdas, dim=0))

    def get_cvx_coefficients(self) -> torch.Tensor:
        return self.lambdas

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    def _compute_gib_outputs(self, coords:torch.Tensor, q_points:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:
        
        batched = coords.dim() == 3
        
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            q_points = q_points.unsqueeze(0)
            support_idxs = support_idxs.unsqueeze(0)  

        q_outputs = torch.empty((q_points.shape[0], q_points.shape[1], self.total_gibs), device=coords.device).contiguous() # shape (B, M, G)
        # print(f"{q_outputs.shape=}")
        for i, gib_key in enumerate(self.gibs):
            
            gib_output = self.gibs[gib_key](coords.contiguous(), q_points.contiguous(), support_idxs.contiguous()) # shape: ([B], M)
            # print(f"{gib_output.shape=}")
            q_outputs[:, :, i] = gib_output
            
        if not batched:
            q_outputs = q_outputs.squeeze(0)

        return q_outputs
    
    def _compute_observers(self, q_outputs:torch.Tensor) -> torch.Tensor:
        # --- Convex Combination ---
        # for each query point, compute the convex combination of the outputs of the GIBs
        if self.num_observers > 1:
            return q_outputs @ self.lambdas # shape (M, num_gibs) @ (num_gibs, num_observers) = (M, num_observers)

        return q_outputs
    
    
    def forward(self, points:torch.Tensor, q_coords:torch.Tensor, support_idxs:torch.Tensor, mc_points:torch.Tensor) -> torch.Tensor:
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

class GIB_Layer_Coll(nn.Module):

    def __init__(self, 
                 gib_dict:Dict[str, int], 
                 kernel_reach:int, 
                 num_observers:int=1
                ):
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
            if 'cy' in key:
                g_class = cylinder.CylinderCollection
            elif 'disk' in key:
                g_class = disk.DiskCollection
            elif 'cone' in key:
                g_class = cone.ConeCollection
            elif 'ellip' in key:
                g_class = ellipsoid.EllipsoidCollection
            else:
                raise ValueError(f"Invalid GIB Type: `{key}`, must be one of ['cy', 'cone', 'disk', 'ellip']")
            
            num_gibs = self.gib_dict[key]
            if num_gibs > 0:
                # self.gibs[key] = torch.jit.script(GIB_Operator_Collection(g_class, num_gibs=num_gibs, kernel_reach=kernel_reach))
                self.gibs[key] = GIB_Operator_Collection(g_class, num_gibs=num_gibs, kernel_reach=kernel_reach)
                self.total_gibs += num_gibs
                
        # angles for each GIB in the layer; each GIB has 3 angles for rotation along the x, y, and z axes   
        self.angles = nn.Parameter(torch.randn((self.total_gibs, 3)) * 0.01)

        # --- Initializing Convex Coefficients ---
        if num_observers > 1:
            self.lambdas = torch.randn((self.total_gibs, num_observers)) # shape (num_gibs, num_observers)
            self.maintain_convexity() # make sure the coefficients are convex
            self.lambdas = to_parameter(self.lambdas)
        # elif num_observers == 0: # no cvx comb, only a linear combination of the GIBs
        #     self.lambdas = nn.Linear(self.total_gibs, self.total_gibs, bias=False)
        else: # no observers, only the output of the GIBs
            self.lambdas = to_tensor(1.0)   

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

        # Prepare support vectors
        # print(f"{support_idxs.dtype=}")
        with torch.no_grad():
            s_centered, valid_mask = compute_centered_support_points(coords, q_points, support_idxs)
        
        # print_gpu_memory()
        s_centered = self.apply_rotations(s_centered) # (B, M, G, K, 3)
        # print(f"{s_centered.shape=}")
        # print_gpu_memory()
        ###### time efficient sol; mem inefficient due to empty alloc ######
        q_outputs = torch.empty((q_points.shape[0], q_points.shape[1], self.total_gibs), device=coords.device).contiguous() # shape (B, M, G)

        offset = 0
        for _, gib_coll in self.gibs.items():
            gib_output_dim = gib_coll.num_gibs
            support = s_centered[:, :, offset : offset + gib_output_dim] # (B, M, G_i, K, 3)
            # mc = mc_points[offset : offset + gib_output_dim] # (G_i, num_samples, 3)
            gib_outputs = gib_coll._prepped_forward(support, valid_mask, mc_points)  # ([B], Q, output_dim)
            q_outputs[:, :, offset : offset + gib_output_dim] = gib_outputs # (B, M, G_i)
            offset += gib_output_dim
            
        # del gib_outputs, support
        # torch.cuda.empty_cache()
            
        ###### mem efficient sol; time inefficient due to cat ######
        # q_outputs_list = []
        # for gib_coll in self.gibs.values():
        #     gib_outputs = gib_coll._prepped_forward(s_centered, valid_mask, batched, mc_points)  # ([B], Q, output_dim)
        #     q_outputs_list.append(gib_outputs)
        # q_outputs = torch.cat(q_outputs_list, dim=-1)  # ([B], Q, total_gibs)
            
        return q_outputs
    
    
    def _jit_compute_gib_outputs(self, coords: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor, mc_points) -> torch.Tensor:
        
        # Ensure tensors are batched
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            q_points = q_points.unsqueeze(0)
            support_idxs = support_idxs.unsqueeze(0)

        # Prepare support vectors
        s_centered, valid_mask, batched = compute_centered_support_points(coords, q_points, support_idxs)
        # s_centered = self.apply_rotations(s_centered)
        
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

class GIB_Block_wMLP(nn.Module):

    def __init__(self, 
                 gib_dict:Dict[str, int], 
                 feat_channels:int,
                 num_observers:int,
                 kernel_size:int,
                 out_channels=None, 
                 strided=False
                ) -> None:
        super(GIB_Block_wMLP, self).__init__()
        self.gib = torch.jit.script(GIB_Layer_Coll(gib_dict, kernel_size, num_observers))
        # self.gib = GIB_Layer_Coll(gib_dict, kernel_size, num_observers)
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
        # print(f"{gib_out.shape=}")
        # gib_out = self.gib_norm(gib_out) # (B, Q, num_observers)

        if self.strided: # if the query poin # if torch.isnan(out).any():
        #     print("Decoder")
        #     print(f"{out=}")
        #     print(f"{curr_points[0].shape=}, {skip_points[0].shape=}, {upsampling_idxs.shape=}")
        #     print(f"{curr_points[1].shape=}, {skip_points[1].shape=}")
        #     print(f"{out.shape=}")
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
            # gib_block = GIB_Block(gib_dict, out_c, kernel_size, strided)
            gib_block = GIB_Block_wMLP(gib_dict, feat_channels, num_obs, kernel_size, out_c, strided)
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
        
        mc_points = self.montecarlo_points
        for gib_block in self.gib_blocks:
            # 1st it: feats = (B, Q, feat_channels + num_observers)
            feats = gib_block((coords, feats), q_coords, neighbor_idxs, mc_points)
            mc_points = mc_points * 2 # here we increase the volume of the mc_points

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
    

        



######################################################################################
# GIBLi Layer
######################################################################################
from core.models.giblinet.GIBLi_utils import Neighboring, GridPool
from core.models.giblinet.conversions import build_batch_tensor, batch_to_packed, offset2batch, batchvector_to_lengths, batch_to_pack, pack_to_batch


class GIBLiLayer(nn.Module):
    
    def __init__(self, 
                in_channels:int,
                gib_dict:Dict[str, int],
                num_observers:Union[int, List[int]],
                kernel_size:float,
                neighbor_size:Union[int, List[int]]=4,
                out_channels:int=16,
            ) -> None:
        """
        Instantiates a GIBLiLayer Module with GIBs and their cvx coefficients.
        
        Parameters
        ----------
        
        `in_channels` - int:
            The number of input features to the GIBLiLayer.
            
        `gib_dict` - dict[str, int]:
            Mappings that contain the number of GIBs of each kind (the key) to initialize;
            
        `num_observers` - int:
            number os GENEO observers to form the output of the Module
            If num_observers is a list, then we instantiate multiple GIB Layers with different number of observers.
            
        `kernel_size` - float:
            The kernel's neighborhood reach in Geometric space.
            
        `neighbor_size` - int or List[int]:
            The number of neighbors to consider for each point in the point cloud.
            If a list, then we instantiate multiple GIB Layers with different number of neighbors.
            
        `out_channels` - int:
            The number of output features of the self features; 
            This will then be concatenated with the GIB features
        """
        
        super(GIBLiLayer, self).__init__()
        
        if isinstance(neighbor_size, int):
            self.neighboring_strategies = [Neighboring("knn", neighbor_size)]
        else:
            self.neighboring_strategies = [Neighboring("knn", k) for k in neighbor_size]
            
        if isinstance(num_observers, int):
            num_observers = [num_observers for _ in range(len(self.neighboring_strategies))]
            
        assert len(num_observers) == len(self.neighboring_strategies), "The number of observers must be equal to the number of neighboring strategies"
        
        num_samples = 1000
        self.mc_points = torch.rand((num_samples, 3), device='cuda')
        self.mc_points = to_parameter(self.mc_points, requires_grad=False)
        # the mc_weights serve as the weights for the monte carlo integration;
        # that is, the weights essentially determine the distance of mc_points from the center, and, thus, the volume of the neighborhhod
        # self.mc_weights = nn.Parameter(torch.rand(num_samples, device='cuda'))
            
        # self.gibs = [torch.jit.script(GIB_Layer_Coll(gib_dict, kernel_size*(i+1), ob)) for i, ob in enumerate(num_observers)]
        self.gibs = [GIB_Layer_Coll(gib_dict, kernel_size*(i+1), ob) for i, ob in enumerate(num_observers)]
        self.gibs = nn.ModuleList(self.gibs)
        self.num_observers = num_observers
        
        self.mlp = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.bn = PointBatchNorm(out_channels)
    
    def maintain_convexity(self):
        for gib in self.gibs:
            gib.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        params = []
        for gib in self.gibs:
            params.extend(gib.get_gib_params())
        return params

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return [gib.get_cvx_coefficients() for gib in self.gibs]

    def forward(self, data_dict) -> torch.Tensor:
        """
        data_dict: Dict[str, torch.Tensor]
            A dictionary containing the following keys:
                - 'coords': torch.Tensor of shape (B, N, 3)
                - 'feats': torch.Tensor of shape (B, N, F)
                
        Returns
        -------
        `output` - torch.Tensor:
            Tensor of shape (B, N, out_channels + sum(num_observers))
        """
       
        coords = data_dict['coord']
        # feats = data_dict['feat']
        
        out = None
        
        for i, (neigh, gib) in enumerate(zip(self.neighboring_strategies, self.gibs)):
            # print mem usage
            with torch.no_grad():
                supp = neigh(coords, coords)
            feats = gib(coords, coords, supp, self.mc_points) # shape (B, N, num_observers)
            
            dist_weight = (len(self.neighboring_strategies) - i) / len(self.neighboring_strategies)
            feats = feats * dist_weight
            
            if out is None:
                out = feats
            else:
                out = torch.cat((out, feats), dim=-1)
                     
        x_out = self.act(self.bn(self.mlp(data_dict['feat'])))
        
        return torch.cat((x_out, out), dim=-1)
    
    
    
class GIBLiBlock(nn.Module):
    
    def __init__(self, 
                in_channels:int,
                sota_class:object,
                sota_input_format:str, # this can be: ['batch', 'ptv1', 'kpconv']
                gib_dict:Dict[str, int],
                num_observers:Union[int, List[int]],
                kernel_size:float,
                neighbor_size:Union[int, List[int]]=4,
                embed_channels:int=16,
                out_channels:int=16,
                sota_args = None,
            ):
        """
        Instantiates a GIBLiBlock Module with a GIBLiLayer and a SOTA Module.
        
        Parameters
        ----------
        
        `in_channels` - int:
            The number of input features to the GIBLiBlock.
            
        `sota_class` - object:
            The SOTA Module to be used in the GIBLiBlock.
            
        `sota_input_format` - str:
            The input format of the SOTA Module; this can be one of ['batch', 'ptv1', 'kpconv']
            
        `gib_dict` - dict[str, int]:
            Mappings that contain the number of GIBs of each kind (the key) to initialize;
            
        `num_observers` - int:
            number os GENEO observers to form the output of the Module
            If num_observers is a list, then we instantiate multiple GIB Layers with different number of observers.
            
        `kernel_size` - float:
            The kernel's neighborhood reach in Geometric space.
            
        `neighbor_size` - int or List[int]:
            The number of neighbors to consider for each point in the point cloud.
            If a list, then we instantiate multiple GIB Layers with different number of neighbors.
            
        `embed_channels` - int:
            The number of output features of the query points features;
            
        `out_channels` - int:
            The number of output features of the self features;
            
        `sota_args` - dict:
            The arguments to be passed to the SOTA Module.
            The out_channels need to be compatible with the `out_channels` argument.
        """
        super().__init__()
        
        self.gibli_layer = GIBLiLayer(in_channels, gib_dict, num_observers, kernel_size, neighbor_size, embed_channels)
        
        self.sota_input_format = sota_input_format
        sota_in_channels = embed_channels + sum(self.gibli_layer.num_observers)
        self.sota_module = sota_class(sota_in_channels, **sota_args)
        
        self.point_norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    
    def process_input_for_gibli(self, data_dict) -> Dict[str, torch.Tensor]:
        
        coords_dim = data_dict['coord'].dim()
        
        if coords_dim == 3: # batched input
            return data_dict
        # else its packed input        
        
        x, mask = build_batch_tensor(data_dict['coord'], data_dict['offset'])
        feat, _ = build_batch_tensor(data_dict['feat'], data_dict['offset'])
        
        out_dict = {
            'coord': x,
            'offset': data_dict['offset'],
            'feat' : feat,
            'mask': mask
        }
        
        return out_dict
    
    def process_input_for_sota(self, data_dict, gibli_dict, gibli_out):
        
        if self.sota_input_format == 'ptv1' or self.sota_input_format == 'kpconv':
            gibli_out = batch_to_packed(gibli_out, gibli_dict['mask'])
            
        if self.sota_input_format == 'kpconv':
            data_dict['lengths'] = batchvector_to_lengths(offset2batch(data_dict["offset"]))
    
        
        data_dict['feat'] = gibli_out
        
        
    def forward(self, data_dict) -> torch.Tensor:
        """
        data_dict: Dict[str, torch.Tensor]
            A dictionary containing the following keys:
                - 'coords': torch.Tensor of shape (B, N, 3)
                - 'feats': torch.Tensor of shape (B, N, F)
                ....
        """
        gibli_dict = self.process_input_for_gibli(data_dict)
        x = self.gibli_layer(gibli_dict)
        
        if self.sota_input_format == 'batch': # regular torch module
            x = self.sota_module(x)
        else:
            self.process_input_for_sota(data_dict, gibli_dict, x)
            x = self.sota_module(data_dict)
            
        x = self.act(self.point_norm(x))
        
        if self.sota_input_format == 'batch':
            x, offset = batch_to_packed(x, gibli_dict['mask'])

        return {
            'coord' : data_dict['coord'],
            'feat' : x,
            'offset' : offset if self.sota_input_format == 'batch' else gibli_dict['offset']
        }
    
    
class DownBlock(nn.Module):
    
    def __init__(self, 
                #### Grid Pooling ####
                in_channels:int,
                embed_channels:int,
                grid_size:int,
                #### GIBLi Layer ####
                sota_class:object,
                sota_input_format:str, # this can be: ['batch', 'ptv1', 'kpconv']
                gib_dict:Dict[str, int],
                num_observers:Union[int, List[int]],
                kernel_size:float,
                neighbor_size:Union[int, List[int]]=4,
                out_feat_dim:int=16,
                out_channels:int=16,
                sota_args = None,
            ):
        
        super().__init__()
                
        self.grid_pool = GridPool(in_channels, 
                                  embed_channels,
                                  grid_size
                                )
        
        self.gibli_block = GIBLiBlock(embed_channels, 
                                      sota_class, 
                                      sota_input_format, 
                                      gib_dict, 
                                      num_observers, 
                                      kernel_size, 
                                      neighbor_size, 
                                      out_feat_dim, 
                                      out_channels,
                                      sota_args
                                    )
        
        
    def process_input_for_pooling(self, data_dict):
        # pooling requires the input to be in the ptv1 format
        
        coord_dims = data_dict['coord'].dim()
        
        if coord_dims == 3: # batched input
            coord, offset = batch_to_packed(data_dict['coord'], None)
            feat, _ = batch_to_packed(data_dict['feat'], None)
            data_dict['offset'] = offset
            data_dict['coord'] = coord
            data_dict['feat'] = feat
        

        
    def forward(self, data_dict) -> torch.Tensor:
        self.process_input_for_pooling(data_dict)
        pool_dict = self.grid_pool(data_dict)
        
        gibli_dict = self.gibli_block(pool_dict) 
        
        return gibli_dict, pool_dict['cluster']
    

class Upsample_pops(nn.Module):
    
    def __init__(self,  
                feat_channels,
                skip_channels,
                out_channels,
                bias=True,
                skip=True,
                concat:bool=True,
                backend="interp") -> None:
    
        super(Upsample_pops, self).__init__()

        self.skip = skip
        self.backend = backend
        self.concat = concat
        assert self.backend in ["max", "interp", 'map']

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

    def forward(self, curr_dict, skip_dict, upsampling_idxs) -> torch.Tensor:
        """
        
        Parameters
        ----------
        
        `curr_dict` - Dict[str, torch.Tensor]:
            A dictionary containing the following keys:
                - 'coord': torch.Tensor of shape (B*M, 3)
                - 'feat': torch.Tensor of shape (B*M, C)
                - 'offset': torch.Tensor of shape (B)
                
        `skip_dict` - Dict[str, torch.Tensor]:
            A dictionary containing the following keys:
                - 'coord': torch.Tensor of shape (B*N, 3)
                - 'feat': torch.Tensor of shape (B*N, C)
                - 'offset': torch.Tensor of shape (B)
                
        `upsampling_idxs` - torch.Tensor[int]:
            Tensor of shape (B*N,) representing the indices of the curr_points that are neighbors of the skip_points
        """
        
        import pointops as pops
        
        coord, feat, offset = curr_dict['coord'], curr_dict['feat'], curr_dict['offset']
        skip_coord, skip_feat, skip_offset = skip_dict['coord'], skip_dict['feat'], skip_dict['offset']

        if self.backend == "map" and upsampling_idxs is not None:
            feat = self.proj(feat)[upsampling_idxs]
        elif self.backend == 'max':
            #TODO
            ValueError("Max pooling not yet implemented")
        else:
            feat = pops.interpolation(
                coord, skip_coord, self.proj(feat), offset, skip_offset
            )
        
        if self.skip:
            if self.concat:
                feat = torch.cat([feat, self.proj_skip(skip_feat)], dim=-1)
            else:
                feat = feat + self.proj_skip(skip_feat)
                   
        return {
            'coord': skip_coord,
            'feat': feat,
            'offset': skip_offset
        }    
        
    
class Decoder_pops(nn.Module):
    
    def __init__(self, 
                ### Unpooling Layer ###
                feat_channels,
                skip_channels,
                unpool_out_channels,
                bias=True,
                skip=True,
                concat:bool=True,
                backend="interp",
                ### GIBLi Layer ###
                gib_dict:Dict[str, int]={}, 
                num_observers=16,
                kernel_size=0.2,
                sota_class:object=None,
                sota_input_format:str='batch',
                neighbor_size:Union[int, List[int]]=4,
                out_feat_dim:int=16,
                out_channels:int=16,
                sota_args=None,
            ) -> None:
        super(Decoder_pops, self).__init__()
        
        self.unpool = Upsample_pops(feat_channels, skip_channels, unpool_out_channels, bias, skip, concat, backend)
        f_channels = unpool_out_channels*2 if concat else unpool_out_channels
        self.gibli_block = GIBLiBlock(f_channels, sota_class, sota_input_format, gib_dict, num_observers, kernel_size, neighbor_size, out_feat_dim, out_channels, sota_args)
        
    
    def maintain_convexity(self):
        self.gibli_block.gibli_layer.maintain_convexity()
        
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.gibli_block.gibli_layer.get_cvx_coefficients()
    
    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gibli_block.gibli_layer.get_gib_params()
   
        
    def forward(self, curr_dict, skip_dict, upsampling_idxs):
        """
        Parameters
        ----------
        `curr_dict` - Dict[str, torch.Tensor]:
            A dictionary containing the following keys:
                - 'coord': torch.Tensor of shape (B*M, 3)
                - 'feat': torch.Tensor of shape (B*M, C)
                - 'offset': torch.Tensor of shape (B)

        `skip_dict` - Dict[str, torch.Tensor]:
            A dictionary containing the following keys:
                - 'coord': torch.Tensor of shape (B*N, 3)
                - 'feat': torch.Tensor of shape (B*N, C)
                - 'offset': torch.Tensor of shape (B)

        `upsampling_idxs` - torch.Tensor[int]:
            Tensor of shape (B*N, K) representing the indices of the curr_dict that are neighbors of the skip_points    

        Returns
        -------
        `output` - torch.Tensor:
            Tensor of shape (B*N, out_channels) representing the output of the Decoder
        """
        out_dict = self.unpool(curr_dict, skip_dict, upsampling_idxs) # output shape: (B*N, 3 + 2*out_channels)

        out_dict = self.gibli_block(out_dict) # output shape: (B*N, out_channels)
        return out_dict # (B*N, out_channels)
    



# #####################################
# # GIBLi SOTA
# #####################################

# from core.models.pointcept.pointcept.models.point_transformer_v2.point_transformer_v2m2_base import BlockSequence
# from core.models.pointcept.pointcept.models.point_transformer_v3.point_transformer_v3m1_base import Block
# from core.models.pointcept.pointcept.models.point_transformer.point_transformer_seg import PointTransformerLayer




if __name__ == "__main__":
    # test gibli layer
    import gc
    from core.datasets.TS40K import TS40K_FULL_Preprocessed
    import utils.constants as C
    
    
    def test_gpu_memory_usage(model, inputs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Clear cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Track initial memory usage
        initial_mem = torch.cuda.memory_allocated(device)

        # Forward pass
        output = model(inputs)
        print(f"{output.shape=}")

        # Memory after forward pass
        forward_mem = torch.cuda.memory_allocated(device)
        forward_peak_mem = torch.cuda.max_memory_allocated(device)

        # Backward pass
        loss = output.sum()  # Dummy loss function
        loss.backward()

        # Memory after backward pass
        backward_mem = torch.cuda.memory_allocated(device)
        backward_peak_mem = torch.cuda.max_memory_allocated(device)

        print(f"Initial Memory: {initial_mem / 1e6:.2f} MB")
        print(f"After Forward: {forward_mem / 1e6:.2f} MB")
        print(f"Peak After Forward: {forward_peak_mem / 1e6:.2f} MB")
        print(f"After Backward: {backward_mem / 1e6:.2f} MB")
        print(f"Peak After Backward: {backward_peak_mem / 1e6:.2f} MB")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # gib_dict = {}
    
    # # testing with single gibs
    # for i in range(16):
    #     for gib in ['cy', 'cone', 'disk', 'ellip']:
    #         gib_dict[gib + str(i)] = 1

    gib_dict = {
        'cy': 4,
        'cone': 4,
        'disk': 4,
        'ellip': 4
    }
    
    print(f"{gib_dict=}")
    
    num_observers = [16, 16, 16]
    kernel_size = 0.2
    neighbor_size = [16, 16, 16]
    out_feat_dim = 16
    
    # gibli_layer = GIBLiLayer(3, gib_dict, num_observers, kernel_size, neighbor_size, out_feat_dim)
    gibli_block = GIBLiBlock(3, torch.nn.Linear, 'batch', gib_dict, num_observers, kernel_size, neighbor_size, out_feat_dim, out_channels=out_feat_dim, sota_args={'out_features': out_feat_dim})
    down_block = DownBlock(3, 16, 0.05, torch.nn.Linear, 'batch', gib_dict, num_observers, kernel_size, neighbor_size, out_feat_dim, sota_args={'out_features': out_feat_dim})
    
    # gibli_layer = gibli_layer.to('cuda')
    gibli_block = gibli_block.to('cuda')
    down_block = down_block.to('cuda')
    
    
   ### test upsample
   
    ts40k = TS40K_FULL_Preprocessed(
        dataset_path=C.TS40K_FULL_PREPROCESSED_PATH,
        split='fit',
        load_into_memory=False
    )
    
    
    input_dict = ts40k[0]
    for key in input_dict:
        input_dict[key] = input_dict[key].to('cuda')
        
    
    down_dict, upsample_idxs = down_block(input_dict)
    
    for key in input_dict:
        if key == 'offset':
            print(f"{key=}, {input_dict[key].shape=}, {input_dict[key]}")
        else:
            print(f"{key=}, {input_dict[key].shape=}")
            
    print("\n\n")
    
    for key in down_dict:
        if key == 'offset':
            print(f"{key=}, {down_dict[key].shape=}, {down_dict[key]}")
        else:
            print(f"{key=}, {down_dict[key].shape=}")
    print(f"{upsample_idxs.shape=}")
        
    upsample = Upsample_pops(16, 3, 16, backend='map')
    upsample = upsample.to('cuda')
    
    up_dict = upsample(down_dict, input_dict, upsample_idxs)
    
    for key in up_dict:
        if key == 'offset':
            print(f"{key=}, {up_dict[key].shape=}, {up_dict[key]}")
        else:
            print(f"{key=}, {up_dict[key].shape=}")
            
            
    ### test decoder
    
    decoder = Decoder_pops(16, 3, 16, backend='map', gib_dict=gib_dict, num_observers=16, kernel_size=0.2, sota_class=torch.nn.Linear, sota_input_format='batch', neighbor_size=16, out_feat_dim=32, out_channels=8, sota_args={'out_features': 8})
    decoder = decoder.to('cuda')
    
    output = decoder(down_dict, input_dict, upsample_idxs)
    
    for key in output:
        if key == 'offset':
            print(f"{key=}, {output[key].shape=}, {output[key]}")
        else:
            print(f"{key=}, {output[key].shape=}")
    
    
    

    
  
    
    
    
    