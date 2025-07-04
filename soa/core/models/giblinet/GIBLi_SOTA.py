
"""

In this file we define GIBLi models integrated with different core blocks from the SOTA.

"""
import torch
from torch import nn
from typing import Any, Dict, Union, List

import sys
sys.path.append('../../../')

from core.models.pointcept.pointcept.models.point_transformer import Bottleneck
from core.models.pointcept.pointcept.models.point_transformer_v2.point_transformer_v2m2_base import BlockSequence
from core.models.pointcept.pointcept.models.point_transformer_v3.point_transformer_v3m1_base import Block
from core.models.kpconv.easy_kpconv.layers.kpconv_blocks import KPResidualBlock
from core.models.pointnet.models.pointnet_sem_seg import PointNetEncoder
from core.models.pointnet.models.pointnet2_sem_seg import PointNetSetAbstraction
from core.models.pointcept.pointcept.models.utils import Point

from core.models.giblinet.GIBLi_utils import Neighboring, GridPool
from core.models.giblinet.GIBLi_parts import GIBLiLayer, Upsample_pops, PointBatchNorm
from core.models.giblinet.conversions import batch2offset, batchvector_to_lengths, lengths_to_batchvector, build_batch_tensor, batch_to_packed, offset2batch, pack_to_batch, batch_to_pack



###########################################################
# UTILS
###########################################################

def build_input_gibli(data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # point = data_dict['point']
    batched_coord, mask = build_batch_tensor(data_dict['coord'], data_dict['offset'])
    input_dict = {
        "coord": batched_coord,
        "feat": build_batch_tensor(data_dict['feat'], data_dict['offset'])[0],
        "mask": mask,
        }
    return input_dict
        
        
class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

        
        
###########################################################
# STUBs
###########################################################       
        
class GIBLiSequenceStub(nn.Module):
    """
    General GIBLi Sequence stub, where we have a genelarized GIBLi block and uniformed input and output. 
    """
    
    
    def __init__(self, 
                 in_channels:int,
                 depth:int,
                 #### gib params
                 sota_class:nn.Module=None,
                 sota_kwargs:Dict[str, Any]={},
                 sota_update_kwargs:Dict[str, Any]={},
                ):
        
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            block = sota_class(in_channels=in_channels, **sota_kwargs)
            
            self.blocks.append(block)
            in_channels = sota_kwargs['out_channels']
            
            if i < depth - 1:
                for key, value in sota_update_kwargs.items():
                    sota_kwargs[key] = value[i]
                    
                    
    def maintain_convexity(self):
        for block in self.blocks:
            block.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        params = []
        for block in self.blocks:
            params.extend(block.get_gib_params())
        return params

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        params = []
        for block in self.blocks:
            params.extend(block.get_cvx_coefficients())
        return params
                

    def forward(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for block in self.blocks:
            data_dict = block(data_dict)
        return data_dict
          

class GIBLiDownStub(nn.Module):
    
    def __init__(self, 
                 #### Grid Pooling ####
                 in_channels:int,
                 embed_channels:int,
                 grid_size:int,
                 #### GIBLi Sequence ####
                 depth:int,
                 sota_class:nn.Module=None,
                 sota_kwargs:Dict[str, Any]={},
                 sota_update_kwargs:Dict[str, Any]={},
                 ):
        super().__init__()
        
        self.grid_pool = GridPool(in_channels=in_channels, out_channels=embed_channels, grid_size=grid_size)
        
        self.encoder = GIBLiSequenceStub(in_channels=embed_channels, depth=depth, sota_class=sota_class, sota_kwargs=sota_kwargs, sota_update_kwargs=sota_update_kwargs)
        
    def maintain_convexity(self):
       self.encoder.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        return self.encoder.get_gib_params()

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.encoder.get_cvx_coefficients()
        
        
    def forward(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        data_dict = self.grid_pool(data_dict)
        skip_idxs = data_dict['cluster']
        data_dict = self.encoder(data_dict)
        
        return data_dict, skip_idxs
    
    
    
class GIBLiUpStub(nn.Module):
    
    def __init__(self, 
                 #### Upsample args ####
                 feat_channels,
                 skip_channels,
                 unpool_out_channels,
                 bias=True,
                 skip=True,
                 concat:bool=True,
                 backend="interp",
                 #### GIBLi Sequence ####
                 depth:int=4,
                 sota_class:nn.Module=None,
                 sota_kwargs:Dict[str, Any]={},
                 sota_update_kwargs:Dict[str, Any]={},
                 ):
        super().__init__()
        
        self.unpool = Upsample_pops(feat_channels, skip_channels, unpool_out_channels, bias, skip, concat, backend)
        f_channels  = unpool_out_channels*2 if concat else unpool_out_channels
        self.block  = GIBLiSequenceStub(in_channels=f_channels, depth=depth, sota_class=sota_class, sota_kwargs=sota_kwargs, sota_update_kwargs=sota_update_kwargs)

        out_channels = sota_kwargs['out_channels']
        if isinstance(out_channels, list):
            out_channels = out_channels[-1]

        self.residual_proj = nn.Linear(f_channels, out_channels) if f_channels != out_channels else nn.Identity()

    def forward(self, curr_dict, skip_dict, skip_idxs) -> Dict[str, torch.Tensor]: 
        data_dict = self.unpool(curr_dict, skip_dict, skip_idxs)
        residual = self.residual_proj(data_dict['feat'])
        data_dict = self.block(data_dict)
        data_dict['feat'] += residual  # residual connection
        return data_dict
    
    def maintain_convexity(self):
       self.block.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        return self.block.get_gib_params()

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.block.get_cvx_coefficients()




class GIBLiNetStub(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 num_levels: int,
                 grid_size: Union[float, List[float]],
                 embed_channels: Union[int, List[int]],
                 out_channels: Union[int, List[int]],
                 depth: int,
                 sota_class: nn.Module = None,
                 sota_kwargs: Dict[str, Any] = {},
                 sota_update_kwargs: Dict[str, Any] = {},
                 bias: bool = True,
                 skip: bool = True,
                 concat: bool = False,
                 backend: str = "interp",
                 ) -> None:
        
        super(GIBLiNetStub, self).__init__()
        
        self.embed_channels = embed_channels if isinstance(embed_channels, list) else [embed_channels * (i + 1) for i in range(num_levels)]
        self.out_channels = out_channels if isinstance(out_channels, list) else [out_channels * (i + 1) for i in range(num_levels)]
        self.grid_size = grid_size if isinstance(grid_size, list) else [grid_size * (i + 1) for i in range(num_levels)]
        self.num_levels = num_levels
        self.in_channels = in_channels

        #### Project the feature space to higher dimensions #### 
        out_update = sota_update_kwargs.get('out_channels', [])
        if len(out_update) == 0:
            out_update = [self.out_channels[0]] * (depth - 1)
        else:
            out_update[depth-1-1] = self.out_channels[0]
        sota_update_kwargs['out_channels'] = out_update
        sota_kwargs['out_channels'] = self.out_channels[0]
        
        self.initial_encoding = GIBLiSequenceStub(
            in_channels, depth, sota_class, sota_kwargs=sota_kwargs, sota_update_kwargs=sota_update_kwargs
        )
        
        self.last_decoder = GIBLiUpStub(
            feat_channels=self.out_channels[0], skip_channels=self.out_channels[0], unpool_out_channels=self.embed_channels[0], 
            bias=bias, skip=skip, concat=concat, backend=backend, 
            depth=depth, sota_class=sota_class, sota_kwargs=sota_kwargs, sota_update_kwargs=sota_update_kwargs
        )
        
        self.gib_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        in_channels = self.out_channels[0]
        
        for i in range(num_levels):
            # Out channels update
            out_update = sota_update_kwargs.get('out_channels', [])
            if len(out_update) == 0:
                out_update = [self.out_channels[i]] * (depth - 1)
            else:
                out_update[depth-1-1] = self.out_channels[i] # else we update the last out_channel update to be the supposed out_channel
            sota_update_kwargs['out_channels'] = out_update
            sota_kwargs['out_channels'] = self.embed_channels[i]
            
            gib_block = GIBLiDownStub(
                in_channels=in_channels, embed_channels=self.embed_channels[i], grid_size=self.grid_size[i],
                depth=depth, sota_class=sota_class, sota_kwargs=sota_kwargs, sota_update_kwargs=sota_update_kwargs
            )
            in_channels = self.out_channels[i]
            
            self.gib_blocks.append(gib_block)
            
            if i < num_levels - 1:
                dec = GIBLiUpStub(
                    feat_channels=self.out_channels[i+1], skip_channels=self.out_channels[i], unpool_out_channels=self.embed_channels[i], 
                    bias=bias, skip=skip, concat=concat, backend=backend, 
                    depth=depth, sota_class=sota_class, sota_kwargs=sota_kwargs, sota_update_kwargs=sota_update_kwargs
                )
                self.decoders.append(dec)
        
        self.seg_head = nn.Sequential(
            nn.Linear(self.out_channels[0], self.out_channels[0]),
            nn.BatchNorm1d(self.out_channels[0]),
            nn.GELU(),
            nn.Linear(self.out_channels[0], num_classes)
        ) if num_classes > 0 else nn.Identity()
        
        
    def maintain_convexity(self):
        for block in self.gib_blocks:
           block.maintain_convexity()
    
        for block in self.decoders:
            block.maintain_convexity()
            
        self.last_decoder.maintain_convexity()
        self.initial_encoding.maintain_convexity()     
        

    def get_gib_params(self) -> List[torch.Tensor]:
        params = []
        for block in self.gib_blocks:
            params.extend(block.get_gib_params())
        for block in self.decoders:
            params.extend(block.get_gib_params())
        params.extend(self.last_decoder.get_gib_params())
        params.extend(self.initial_encoding.get_gib_params())
        return params

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        params = []
        for block in self.gib_blocks:
            params.extend(block.get_cvx_coefficients())
        for block in self.decoders:
            params.extend(block.get_cvx_coefficients())
        params.extend(self.last_decoder.get_cvx_coefficients())
        params.extend(self.initial_encoding.get_cvx_coefficients())
        return params

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_dict = self.initial_encoding(input_dict)
        level_dicts = [input_dict]
        upsample_list = []
        
        # print(f"Level {0} Encoding; {input_dict['feat'].shape}")
        
        for i in range(len(self.gib_blocks)):
            input_dict, upsample_idxs = self.gib_blocks[i](input_dict)
            # print(f"Level {i + 1} Encoding; {input_dict['feat'].shape}")
            if i < self.num_levels - 1:
                level_dicts.append(input_dict)
            upsample_list.append(upsample_idxs)
        
        for i in reversed(range(len(self.decoders))):
            input_dict = self.decoders[i](input_dict, level_dicts[i+1], upsample_list[i+1])
            # print(f"Level {i} Decoding; {input_dict['feat'].shape}")
        
        input_dict = self.last_decoder(input_dict, level_dicts[0], upsample_list[0])
        # print(f"Level {-1} Decoding; {input_dict['feat'].shape}")
        seg_logits = self.seg_head(input_dict['feat'])
        
        return seg_logits

 
          
###########################################################
# SOTA GIBLi BLOCKS
###########################################################

class GIBLiBlockPTV1(nn.Module):
    
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 #### gib params
                 gib_dict:Dict[str, int],
                 num_observers:Union[int, List[int]],
                 kernel_reach:float,
                 neighbor_size:Union[int, List[int]]=4,
                 feat_enc_channels:int=16,
                 ### ptv1 params
                 shared_channels:int=1,
                 num_neighbors:int=16,
                ):
        super().__init__()
        
        
        self.gibli_layer = GIBLiLayer(in_channels=in_channels, 
                                gib_dict=gib_dict,
                                num_observers=num_observers,
                                kernel_reach=kernel_reach,
                                neighbor_size=neighbor_size,
                                out_channels=feat_enc_channels,
                            )
        
        sota_in_channels = feat_enc_channels + sum(self.gibli_layer.num_observers)
        
        self.gibli_proj = MLP(sota_in_channels,
                              sota_in_channels,
                              in_channels,
                              act_layer=nn.GELU,
                              drop=0.2)
        
        self.act = nn.GELU()        
        self.norm1 = PointBatchNorm(in_channels)
        
        self.pt_layer = Bottleneck(in_channels, in_channels, shared_channels, num_neighbors)
        
        self.sota_proj = MLP(in_channels, out_channels, out_channels, act_layer=nn.GELU, drop=0.2)
        self.norm2 = PointBatchNorm(out_channels)       
        
    def maintain_convexity(self):
       self.gibli_layer.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_gib_params()

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_cvx_coefficients()
        
    def forward(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        data_dict : Dict[str, torch.Tensor]
            A dictionary containing the input data tensors.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the output data tensors.
        """
        # Assuming daat_dict is in ptv1 format
        gibli_dict = build_input_gibli(data_dict)
        gibli_out = self.gibli_layer(gibli_dict)
        gibli_out, offset = batch_to_packed(gibli_out, gibli_dict['mask'])
        
        gibli_out = data_dict['feat'] + self.gibli_proj(gibli_out) # residual connection
        gibli_out = self.act(self.norm1(gibli_out))
        
        pt_out = self.pt_layer((data_dict['coord'], gibli_out, offset))
        
        pt_out[1] = self.act(self.norm2(self.sota_proj(pt_out[1])))
        
        # print(pt_out[1].shape)
        
        return {
            'coord': pt_out[0],
            'feat': pt_out[1],
            'offset': pt_out[2],
        }
        
        
class GIBLiBlockPTV2(nn.Module):
    
    def __init__(self,
                 in_channels:int,
                 #### gib params
                 gib_dict:Dict[str, int],
                 num_observers:Union[int, List[int]],
                 kernel_reach:float,
                 neighbor_size:Union[int, List[int]],
                 feat_enc_channels:int,
                 out_channels:int,
                 ### ptv2 params
                 depth,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 enable_checkpoint=False
                ):
        super().__init__()
        
        
        self.gibli_layer = GIBLiLayer(in_channels=in_channels, 
                                gib_dict=gib_dict,
                                num_observers=num_observers,
                                kernel_reach=kernel_reach,
                                neighbor_size=neighbor_size,
                                out_channels=feat_enc_channels,
                            )
        
        sota_in_channels = feat_enc_channels + sum(self.gibli_layer.num_observers)
        
        self.gibli_proj = MLP(sota_in_channels, sota_in_channels, in_channels, act_layer=nn.GELU, drop=0.2)
        
        self.act = nn.GELU()        
        self.norm1 = PointBatchNorm(in_channels)
        
        self.pt_layer = BlockSequence(
                            depth=depth,
                            embed_channels=in_channels,
                            groups=groups,
                            neighbours=neighbours,
                            qkv_bias=qkv_bias,
                            pe_multiplier=pe_multiplier,
                            pe_bias=pe_bias,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=drop_path_rate,
                            enable_checkpoint=enable_checkpoint
                        )  
        
        self.sota_proj = MLP(in_channels, out_channels, out_channels, act_layer=nn.GELU, drop=0.2)
        self.norm2 = PointBatchNorm(out_channels)
        
    def maintain_convexity(self):
       self.gibli_layer.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_gib_params()

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_cvx_coefficients()
        
        
    def forward(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        gibli_dict = build_input_gibli(data_dict)
        gibli_out = self.gibli_layer(gibli_dict)   
        gibli_out, offset = batch_to_packed(gibli_out, gibli_dict['mask'])
        
        gibli_out = data_dict['feat'] + self.gibli_proj(gibli_out) # residual connection
        gibli_out = self.act(self.norm1(gibli_out))
        
        # print(data_dict['coord'].shape, gibli_out.shape, offset.shape)
        pt_out = self.pt_layer((data_dict['coord'], gibli_out, offset))
        pt_out[1] = self.act(self.norm2(self.sota_proj(pt_out[1])))
        
        return {
            'coord': pt_out[0],
            'feat': pt_out[1],
            'offset': pt_out[2],
        }
        
        
class GIBLiBlockPTV3(nn.Module):
    
    
    def __init__(self,
                 in_channels:int,
                 #### gib params
                 gib_dict:Dict[str, int],
                 num_observers:Union[int, List[int]],
                 kernel_reach:float,
                 neighbor_size:Union[int, List[int]],
                 feat_enc_channels:int,
                 out_channels:int,
                 ### ptv3 params
                 num_heads,
                 grid_size=0.01,
                 patch_size=48,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_path=0.0,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 pre_norm=True,
                 order_index=0,
                 cpe_indice_key=None,
                 enable_rpe=False,
                 enable_flash=False,
                 upcast_attention=True,
                 upcast_softmax=True,
                ) -> None:
        
        
        super().__init__()
        
        self.gibli_layer = GIBLiLayer(in_channels=in_channels, 
                                gib_dict=gib_dict,
                                num_observers=num_observers,
                                kernel_reach=kernel_reach,
                                neighbor_size=neighbor_size,
                                out_channels=feat_enc_channels,
                            )
        self.grid_size = grid_size
        sota_in_channels = feat_enc_channels + sum(self.gibli_layer.num_observers)
        self.gibli_proj = MLP(sota_in_channels, sota_in_channels, in_channels, act_layer=nn.GELU, drop=0.2)
        self.act = nn.GELU()
        self.norm1 = PointBatchNorm(in_channels)
        
        num_heads = num_heads if in_channels % num_heads == 0 else 1
        
        self.pt_layer = Block(
                            channels=in_channels,
                            num_heads=num_heads,
                            patch_size=patch_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=drop_path,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=order_index,
                            cpe_indice_key=cpe_indice_key,
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax
                        )
        
        self.sota_proj = MLP(in_channels, out_channels, out_channels, act_layer=nn.GELU, drop=0.2)
        self.norm2 = PointBatchNorm(out_channels)  
        
    def maintain_convexity(self):
       self.gibli_layer.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_gib_params()

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_cvx_coefficients()     
        
        
        
    def forward(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
         
        gibli_dict = build_input_gibli(data_dict)
        gibli_out = self.gibli_layer(gibli_dict)   
        gibli_out, offset = batch_to_packed(gibli_out, gibli_dict['mask'])
        
        gibli_out = data_dict['feat'] + self.gibli_proj(gibli_out) # residual connection
        gibli_out = self.act(self.norm1(gibli_out))
        
        # print(data_dict['coord'].shape, gibli_out.shape, offset.shape)
        point = Point(
            coord=data_dict['coord'],
            feat=gibli_out,
            offset=offset,
            grid_size=self.grid_size,
        )
        point.serialization(order=("z", "z-trans"), shuffle_orders=True)
        point.sparsify()
        
        pt_out = self.pt_layer(point)
        pt_out.feat = self.act(self.norm2(self.sota_proj(pt_out.feat)))
        
        return {
            'coord': pt_out.coord,
            'feat': pt_out.feat,
            'offset': pt_out.offset,
        }
        
        
class GIBLiBlockKPConv(nn.Module):
    
    
    def __init__(self, 
                 in_channels:int,
                 #### gib params
                 gib_dict:Dict[str, int],
                 num_observers:Union[int, List[int]],
                 kernel_reach:float,
                 neighbor_size:Union[int, List[int]],
                 feat_enc_channels:int,
                 ### kpconv params
                 out_channels: int,
                 kernel_size: int,
                 radius: float,
                 sigma: float,
                 groups: int = 1,
                 dimension: int = 3,
                 strided: bool = False,
                 kpconv_neighbors=16,
                ):
            
            super().__init__()
            
            self.gibli_layer = GIBLiLayer(in_channels=in_channels, 
                                    gib_dict=gib_dict,
                                    num_observers=num_observers,
                                    kernel_reach=kernel_reach,
                                    neighbor_size=neighbor_size,
                                    out_channels=feat_enc_channels,
                                )
            
            sota_in_channels = feat_enc_channels + sum(self.gibli_layer.num_observers)
            self.gibli_proj = MLP(sota_in_channels, sota_in_channels, in_channels, act_layer=nn.GELU, drop=0.2)
            self.act = nn.GELU()
            self.norm1 = PointBatchNorm(in_channels)
            
            self.neigh_strat = Neighboring('knn', kpconv_neighbors)
            
            self.kpconv_layer = KPResidualBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                radius=radius,
                                sigma=sigma,
                                groups=groups,
                                dimension=dimension,
                                strided=strided,
                            )
            
            
    def maintain_convexity(self):
       self.gibli_layer.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_gib_params()

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_cvx_coefficients()
            
    def prep_input_to_kpconv(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        lengths = batchvector_to_lengths(offset2batch(data_dict["offset"]))
        
        coords, mask = pack_to_batch(data_dict['coord'], lengths)
        
        neigh_idxs = self.neigh_strat(coords, coords)
        
        return {
            'q_points': data_dict['coord'],
            's_points': data_dict['coord'],
            's_feats': data_dict['feat'],
            'neighbor_indices': batch_to_pack(neigh_idxs, mask)[0],
        }, lengths
        
        
                  
    def forward(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        gibli_dict = build_input_gibli(data_dict)
        
        gibli_out = self.gibli_layer(gibli_dict)   
        gibli_out, offset = batch_to_packed(gibli_out, gibli_dict['mask'])        
        gibli_out = data_dict['feat'] + self.gibli_proj(gibli_out) # residual connection
        gibli_out = self.act(self.norm1(gibli_out))
        
        data_dict['feat'] = gibli_out
        data_dict['offset'] = offset
        
        kpconv_dict, lengths = self.prep_input_to_kpconv(data_dict)
        # print(kpconv_dict['q_points'].shape, kpconv_dict['s_points'].shape, kpconv_dict['s_feats'].shape, kpconv_dict['neighbor_indices'].shape)
        kpconv_out = self.kpconv_layer(**kpconv_dict)
        
        return {
            'coord': data_dict['coord'],
            'feat': kpconv_out,
            'offset': batch2offset(lengths_to_batchvector(lengths)),
        }


class GIBLiBlockPointNet(nn.Module):
    
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 #### gib params
                 gib_dict:Dict[str, int],
                 num_observers:Union[int, List[int]],
                 kernel_reach:float,
                 neighbor_size:Union[int, List[int]],
                 feat_enc_channels:int,
                 ### pointnet params
                 feature_transform=False,
                ):
        super().__init__()
        
        
        self.gibli_layer = GIBLiLayer(in_channels=in_channels, 
                                gib_dict=gib_dict,
                                num_observers=num_observers,
                                kernel_reach=kernel_reach,
                                neighbor_size=neighbor_size,
                                out_channels=feat_enc_channels,
                            )
        
        sota_in_channels = feat_enc_channels + sum(self.gibli_layer.num_observers)
        self.gibli_proj = MLP(sota_in_channels, sota_in_channels, in_channels, act_layer=nn.GELU, drop=0.2)
        self.act = nn.GELU()
        self.norm1 = PointBatchNorm(in_channels)
        
        self.encoder = PointNetEncoder(
                            channel=3+in_channels, # coords also count
                            global_feat=False,
                            feature_transform=feature_transform
                        )
        
        encoder_out = 256 + 64 # hard coded from self.encoder
        
        self.sota_proj = MLP(encoder_out, out_channels, out_channels, act_layer=nn.GELU, drop=0.2)
        self.norm2 = PointBatchNorm(out_channels)       
        
    def maintain_convexity(self):
       self.gibli_layer.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_gib_params()

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_cvx_coefficients()
        
        
    def forward(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        gibli_dict = build_input_gibli(data_dict)
        
        gibli_out = self.gibli_layer(gibli_dict)               
        gibli_out = gibli_dict['feat'] + self.gibli_proj(gibli_out) # residual connection
        gibli_out = self.act(self.norm1(gibli_out))
        
        gibli_out = torch.cat([gibli_dict['coord'], gibli_out], dim=-1)
        gibli_out = gibli_out.transpose(1, 2)
        # print(gibli_out.shape)
        
        pointnet_out = self.encoder(gibli_out)[0]
        # print(pointnet_out.shape)
        pointnet_out = pointnet_out.transpose(1, 2)
        
    
        pointnet_out, offset = batch_to_packed(pointnet_out, gibli_dict['mask'])
        pointnet_out = self.act(self.norm2(self.sota_proj(pointnet_out)))
        
        return {
            'coord': data_dict['coord'],
            'feat':  pointnet_out,
            'offset': offset,
        }
        
        
class GIBLiBlockPointNet2(nn.Module):
    
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 #### gib params
                 gib_dict:Dict[str, int],
                 num_observers:Union[int, List[int]],
                 kernel_reach:float,
                 neighbor_size:Union[int, List[int]],
                 feat_enc_channels:int,
                 ### pointnet2 params
                 npoint=512,
                 radius=0.2,
                 nsample=64,
                 mlp=[64, 64, 128],
                ):
        super().__init__()
        
        
        self.gibli_layer = GIBLiLayer(in_channels=in_channels, 
                                gib_dict=gib_dict,
                                num_observers=num_observers,
                                kernel_reach=kernel_reach,
                                neighbor_size=neighbor_size,
                                out_channels=feat_enc_channels,
                            )
        
        sota_in_channels = feat_enc_channels + sum(self.gibli_layer.num_observers)
        self.gibli_proj = MLP(sota_in_channels, sota_in_channels, in_channels, act_layer=nn.GELU, drop=0.2)
        self.act = nn.GELU()
        self.norm1 = PointBatchNorm(in_channels)
        
        mlp[-1] = out_channels
        
        self.encoder = PointNetSetAbstraction(
                            npoint=npoint,
                            radius=radius,
                            nsample=nsample,
                            in_channel=3 + in_channels, # coords count
                            mlp=mlp,
                            group_all=False,
                        )  
        
    def maintain_convexity(self):
       self.gibli_layer.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_gib_params()

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.gibli_layer.get_cvx_coefficients()
        
        
    def forward(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        gibli_dict = build_input_gibli(data_dict)
        
        gibli_out = self.gibli_layer(gibli_dict)               
        gibli_out = gibli_dict['feat'] + self.gibli_proj(gibli_out) # residual connection
        gibli_out = self.act(self.norm1(gibli_out))
        
        # print(f"{gibli_out.shape=} {gibli_dict['coord'].shape=}")
        pointnet2_coord, pointnet2_feat = self.encoder(gibli_dict['coord'].transpose(1, 2), gibli_out.transpose(1, 2))
        # print(f"{pointnet2_coord.shape=} {pointnet2_feat.shape=}")
        
        # to packed
        pointnet2_coord = pointnet2_coord.transpose(1, 2)
        pointnet2_feat = pointnet2_feat.transpose(1, 2)
        pointnet2_coord, offset = batch_to_packed(pointnet2_coord, gibli_dict['mask'])
        pointnet2_feat, _ = batch_to_packed(pointnet2_feat, gibli_dict['mask'])
        
        return {
            'coord': pointnet2_coord,
            'feat': pointnet2_feat,
            'offset': offset,
        }
    


###########################################################
# GIBLi Net SOTA
###########################################################

class GIBLiNetPTV1(nn.Module):
    def __init__(self, 
            in_channels=3,
            num_classes=6,
            num_levels=4,
            grid_size=0.1,
            embed_channels=[16, 16, 32, 64],
            out_channels=[32, 32, 64, 128],
            depth=2,
            sota_kwargs={},
            sota_update_kwargs={}
        ):
        super(GIBLiNetPTV1, self).__init__()
        
        sota_kwargs_defaults = {
            ### gibli params
            'gib_dict': {'cy': 8, 'cone': 8, 'ellip': 8, 'disk': 8},
            'num_observers': [8, 8],
            'kernel_reach': 0.1,
            'neighbor_size': [8, 16],
            'feat_enc_channels': 16,
            ### ptv1 params
            'shared_channels': 1,
            'num_neighbors': 8,
        }
        sota_kwargs_defaults.update(sota_kwargs)
        sota_kwargs = sota_kwargs_defaults
        
        self.model = GIBLiNetStub(
            in_channels=in_channels,
            num_classes=num_classes,
            ### U-NET Structure
            num_levels=num_levels,
            grid_size=grid_size,
            embed_channels=embed_channels,
            out_channels=out_channels,
            ### SOTA MODULE
            depth=depth,
            sota_class=GIBLiBlockPTV1,
            sota_kwargs=sota_kwargs,
            sota_update_kwargs=sota_update_kwargs
        )
        
    def forward(self, data_dict:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(data_dict)
    
    def maintain_convexity(self):
        self.model.maintain_convexity()
    
    def get_gib_params(self) -> List[torch.Tensor]:
        return self.model.get_gib_params()
    
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.model.get_cvx_coefficients()
    

class GIBLiNetPTV2(nn.Module):
    def __init__(self, 
            in_channels=3,
            num_classes=6,
            num_levels=4,
            grid_size=0.1,
            embed_channels=[16, 16, 32, 64],
            out_channels=[32, 32, 64, 128],
            depth=2,
            sota_kwargs:Dict[str, Any]={},
            sota_update_kwargs={} ### sota update kwargs should contain lists with the kwargs update at each depth, 
        ):
        super(GIBLiNetPTV2, self).__init__()
        
        # defaults
        sota_kwargs_defaults = {
            ### gibli params
            'gib_dict': {'cy': 8, 'cone': 8, 'ellip': 8, 'disk': 8},
            'num_observers': [8, 8],
            'kernel_reach': 0.1,
            'neighbor_size': [8, 16],
            'feat_enc_channels': 16,
            ### ptv2 params
            'depth': 2,
            'groups': 1,
            'neighbours': 8,
            'qkv_bias': True,
            'pe_multiplier': False,
            'pe_bias': True,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.0,
            'enable_checkpoint': False,
        }
        sota_kwargs_defaults.update(sota_kwargs)
        sota_kwargs = sota_kwargs_defaults
        
        self.model = GIBLiNetStub(
            in_channels=in_channels,
            num_classes=num_classes,
            ### U-NET Structure
            num_levels=num_levels,
            grid_size=grid_size,
            embed_channels=embed_channels,
            out_channels=out_channels,
            ### SOTA MODULE
            depth=depth,
            sota_class=GIBLiBlockPTV2,
            sota_kwargs=sota_kwargs,
            sota_update_kwargs=sota_update_kwargs
        )
        
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(data_dict)
    
    def maintain_convexity(self):
        self.model.maintain_convexity()
    
    def get_gib_params(self) -> List[torch.Tensor]:
        return self.model.get_gib_params()
    
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.model.get_cvx_coefficients()


class GIBLiNetPTV3(nn.Module):
    def __init__(self, 
                in_channels=3,
                num_classes=6,
                num_levels=4,
                grid_size=0.1,
                embed_channels=[16, 16, 32, 64],
                out_channels=[32, 32, 64, 128],
                depth=2,
                sota_kwargs:Dict[str, Any]={},
                sota_update_kwargs={}
            ):
        super(GIBLiNetPTV3, self).__init__()
        
        # defaults
        sota_kwargs_defaults = {
            ### gibli params
            'gib_dict': {'cy': 8, 'cone': 8, 'ellip': 8, 'disk': 8},
            'num_observers': [8, 8],
            'kernel_reach': 0.1,
            'neighbor_size': [8, 16],
            'feat_enc_channels': 16,
            ### ptv3 params
            'num_heads': 4,
            'patch_size': 48,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'qk_scale': None,
            'attn_drop': 0.0,
            'proj_drop': 0.0,
            'drop_path': 0.0,
            'norm_layer': nn.LayerNorm,
            'act_layer': nn.GELU,
            'pre_norm': True,
            'order_index': 0,
            'cpe_indice_key': None,
            'enable_rpe': False,
            'enable_flash': False,
            'upcast_attention': True,
            'upcast_softmax': True,
        }
        sota_kwargs_defaults.update(sota_kwargs)
        sota_kwargs = sota_kwargs_defaults
                
        self.model = GIBLiNetStub(
            in_channels=in_channels,
            num_classes=num_classes,
            ### U-NET Structure
            num_levels=num_levels,
            grid_size=grid_size,
            embed_channels=embed_channels,
            out_channels=out_channels,
            ### SOTA MODULE
            depth=depth,
            sota_class=GIBLiBlockPTV3,
            sota_kwargs=sota_kwargs,
            sota_update_kwargs=sota_update_kwargs
        )
        
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(data_dict)
    
    def maintain_convexity(self):
        self.model.maintain_convexity()
    
    def get_gib_params(self) -> List[torch.Tensor]:
        return self.model.get_gib_params()
    
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.model.get_cvx_coefficients()


class GIBLiNetKPConv(nn.Module):
    def __init__(self, 
            in_channels=3,
            num_classes=6,
            num_levels=4,
            grid_size=0.1,
            embed_channels=[16, 16, 32, 64],
            out_channels=[32, 32, 64, 128],
            depth=2,
            sota_kwargs:Dict[str, Any]={},
            sota_update_kwargs:Dict[str, Any]={}
        ):
        super(GIBLiNetKPConv, self).__init__()
        
        sota_kwargs_defaults = {
            ### gibli params
            'gib_dict': {'cy': 8, 'cone': 8, 'ellip': 8, 'disk': 8},
            'num_observers': [8, 8],
            'kernel_reach': 0.1,
            'neighbor_size': [8, 16],
            'feat_enc_channels': 16,
            ### kpconv params
            'out_channels': 16,
            'kernel_size': 3,
            'radius': 0.1,
            'sigma': 0.1,
            'groups': 1,
            'dimension': 3,
            'strided': False,
            'kpconv_neighbors': 16,
        }
        sota_kwargs_defaults.update(sota_kwargs)
        sota_kwargs = sota_kwargs_defaults
        
        self.model = GIBLiNetStub(
            in_channels=in_channels,
            num_classes=num_classes,
            ### U-NET Structure
            num_levels=num_levels,
            grid_size=grid_size,
            embed_channels=embed_channels,
            out_channels=out_channels,
            ### SOTA MODULE
            depth=depth,
            sota_class=GIBLiBlockKPConv,
            sota_kwargs=sota_kwargs,
            sota_update_kwargs=sota_update_kwargs
        )
        
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(data_dict)
    
    def maintain_convexity(self):
        self.model.maintain_convexity()
    
    def get_gib_params(self) -> List[torch.Tensor]:
        return self.model.get_gib_params()
    
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.model.get_cvx_coefficients()


class GIBLiNetPointNet(nn.Module):
    def __init__(self, 
            in_channels=3,
            num_classes=6,
            num_levels=4,
            grid_size=0.1,
            embed_channels=[16, 16, 32, 64],
            out_channels=[32, 32, 64, 128],
            depth=2,
            sota_kwargs:Dict[str, Any]={},
            sota_update_kwargs:Dict[str, Any]={},
        ):
        super(GIBLiNetPointNet, self).__init__()
        
        sota_kwargs_defaults = {        
            ### gibli params
            'gib_dict': {'cy': 8, 'cone': 8, 'ellip': 8, 'disk': 8},
            'num_observers': [8, 8],
            'kernel_reach': 0.1,
            'neighbor_size': [8, 16],
            'feat_enc_channels': 16,
            ### pointnet params
            'feature_transform': False,
        }
        sota_kwargs_defaults.update(sota_kwargs)
        sota_kwargs = sota_kwargs_defaults
        
        self.model = GIBLiNetStub(
            in_channels=in_channels,
            num_classes=num_classes,
            ### U-NET Structure
            num_levels=num_levels,
            grid_size=grid_size,
            embed_channels=embed_channels,
            out_channels=out_channels,
            ### SOTA MODULE
            depth=depth,
            sota_class=GIBLiBlockPointNet,
            sota_kwargs=sota_kwargs,
            sota_update_kwargs=sota_update_kwargs
        )
        
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(data_dict)
    
    def maintain_convexity(self):
        self.model.maintain_convexity()
    
    def get_gib_params(self) -> List[torch.Tensor]:
        return self.model.get_gib_params()
    
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.model.get_cvx_coefficients()


class GIBLiNetPointNet2(nn.Module):
    def __init__(self, 
                    in_channels=3,
                    num_classes=6,
                    num_levels=4,
                    grid_size=0.1,
                    embed_channels=[16, 16, 32, 64],
                    out_channels=[32, 32, 64, 128],
                    depth=2,
                    sota_kwargs:Dict[str, Any]={},
                    sota_update_kwargs:Dict[str, Any]={}):
        super(GIBLiNetPointNet2, self).__init__()
            
        sota_kwargs_defaults = {        
            ### gibli params
            'gib_dict': {'cy': 8, 'cone': 8, 'ellip': 8, 'disk': 8},
            'num_observers': [8, 8],
            'kernel_reach': 0.1,
            'neighbor_size': [8, 16],
            'feat_enc_channels': 16,
            ### pointnet2 params
            'npoint': -1,
            'radius': 0.2,
            'nsample': 8,
            'mlp': [64, 64],
        }
        sota_kwargs_defaults.update(sota_kwargs)
        sota_kwargs = sota_kwargs_defaults
 
        self.model = GIBLiNetStub(
            in_channels=in_channels,
            num_classes=num_classes,
            ### U-NET Structure
            num_levels=num_levels,
            grid_size=grid_size,
            embed_channels=embed_channels,
            out_channels=out_channels,
            ### SOTA MODULE
            depth=depth,
            sota_class=GIBLiBlockPointNet2,
            sota_kwargs=sota_kwargs,
            sota_update_kwargs=sota_update_kwargs
        )
        
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(data_dict)
    
    def maintain_convexity(self):
        self.model.maintain_convexity()
    
    def get_gib_params(self) -> List[torch.Tensor]:
        return self.model.get_gib_params()
    
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.model.get_cvx_coefficients()



if __name__ == '__main__':
    
    from core.datasets.TS40K import TS40K_FULL_Preprocessed
    from core.lit_modules.lit_ts40k import LitTS40K_FULL_Preprocessed
    from utils.constants import TS40K_FULL_PREPROCESSED_PATH
    from torchinfo import summary
    
    
    def test_module_gradients(model, dataloader, loss_fn=torch.nn.MSELoss()):
        model.train()  # Ensure we're in training mode
        batch = next(iter(dataloader))  # Get a sample batch
        
        # Move batch to the same device as the model
        device = next(model.parameters()).device
        batch = batch.to(device) if isinstance(batch, torch.Tensor) else {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        model.zero_grad()
        outputs = model(batch)
        if isinstance(outputs, dict):
            outputs = outputs['feat']
            
        loss = loss_fn(outputs, torch.randn_like(outputs))        
        loss.backward()
        
        print("Model parameters with gradients:")
        grad_norms = {
            n: (p.grad.norm().item() if p.grad is not None else 0.0)
            for n, p in model.named_parameters()
        }
        
        for n, g in grad_norms.items():
            print(f"{n}: {g:.2e}")
    
    dataset = TS40K_FULL_Preprocessed(TS40K_FULL_PREPROCESSED_PATH,
                                      sample_types='all', load_into_memory=False)
    
    lit_dataset = LitTS40K_FULL_Preprocessed(
        TS40K_FULL_PREPROCESSED_PATH,
        batch_size=1,
        sample_types='all',
    )
    
    
    lit_dataset.setup('fit')
    dl_ts40k = lit_dataset.train_dataloader()
    
    
    # model = GIBLiNetPTV1(
    #     in_channels=3,
    #     num_classes=6,
    #     num_levels=4,
    #     grid_size=0.1,
    #     embed_channels=[16, 16, 32, 64],
    #     out_channels=[32, 32, 64, 128],
    #     depth=2,
    #     sota_kwargs={}
    # ).cuda()
    
    model = GIBLiSequenceStub(
        in_channels=3,
        depth=1,
        sota_class=GIBLiBlockPTV1,
        sota_kwargs={
            'gib_dict': {'cy': 8, 'cone': 8, 'ellip': 8, 'disk': 8},
            'num_observers': [8],
            'kernel_reach': 0.1,
            'neighbor_size': [8],
            'feat_enc_channels': 16,
            'shared_channels': 1,
            'num_neighbors': 8,
            'out_channels': 6,
        }
        ).cuda()

    # Initialize weights with Xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    
    ### print model summary
    summary(model)
    
    
    test_module_gradients(model, dl_ts40k)
    
    ####
    sample_dict = next(iter(dl_ts40k))
    sample_dict = {key: value.cuda() for key, value in sample_dict.items()}
    out = model(sample_dict)
    out = out if isinstance(out, torch.Tensor) else out['feat']
    print(f"{out.shape=}")
    