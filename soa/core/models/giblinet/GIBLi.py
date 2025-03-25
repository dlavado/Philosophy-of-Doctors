from typing import List, Dict, Union
import torch
import torch.nn as nn

import sys


sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')

from core.models.giblinet.GIBLi_parts import GIB_Sequence, PointBatchNorm, Decoder, GIB_Layer_Coll
from core.models.giblinet.GIBLi_utils import BuildGraphPyramid, Neighboring


#################################################################
# GIBLi: Geometric Inductive Bias Library for Point Clouds
#################################################################

class GIBLiLayer(nn.Module):
    
    def __init__(self, 
                in_channels:int,
                gib_dict:Dict[str, int],
                num_observers:Union[int, List[int]],
                kernel_size:float,
                neighbor_size:Union[int, List[int]]=4,
                out_feat_dim:int=16,
            ) -> None:
        
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
        # the mc_weights serve as the weights for the monte carlo integration;
        # that is, the weights essentially determine the distance of mc_points from the center, and, thus, the volume of the neighborhhod
        # self.mc_weights = nn.Parameter(torch.rand(num_samples, device='cuda'))
            
        self.gibs = [GIB_Layer_Coll(gib_dict, kernel_size, ob) for ob in num_observers]
        
        self.mlp = nn.Linear(in_channels, out_feat_dim)
        self.act = nn.ReLU(inplace=True)
    

    def forward(self, data_dict) -> torch.Tensor:
        """
        data_dict: Dict[str, torch.Tensor]
            A dictionary containing the following keys:
                - 'coords': torch.Tensor of shape (B, N, 3)
                - 'feats': torch.Tensor of shape (B, N, F)
        """
        # this assumes that the input is of shape (B, N, 3 + F), where N is the number of points and F is the number of features (F >= 0).
        # the batch_dim is necessary for now
       
        coords = data_dict['coords']
        feats = data_dict['feats']
        
        out = None
        
        for i, (neigh, gib) in enumerate(zip(self.neighboring_strategies, self.gibs)):
            feats = gib(coords, coords, neigh(coords, coords), self.mc_points) # shape (B, N, num_observers)
            
            dist_weight = (len(self.neighboring_strategies) - i) / len(self.neighboring_strategies)
            feats = feats * dist_weight
            
            if out is None:
                out = feats
            else:
                out = torch.cat((out, feats), dim=-1)
                
        x_out = self.act(self.mlp(out))
        
        return torch.cat((x_out, out), dim=-1)

    def maintain_convexity(self):
        self.gib_seq.maintain_convexity()

    def get_gib_params(self) -> List[torch.Tensor]:
        return self.gib_seq.get_gib_params()

    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        return self.gib_seq.get_cvx_coefficients()
    
    
    
    
class GIBLiNet(nn.Module):

    def __init__(self, 
                in_channels:int,
                num_classes:int,
                num_levels:int,
                out_gib_channels:Union[int, List[int]],
                num_observers:int,
                kernel_size:float,
                gib_dict:Dict[str, int],
                skip_connections:bool,
                pyramid_builder:BuildGraphPyramid
                ) -> None:
        
        super(GIBLiNet, self).__init__()

        self.skip_connections = skip_connections
        self.num_levels = num_levels
        
        if isinstance(out_gib_channels, int):
            out_gib_channels = [out_gib_channels*i for i in range(1, num_levels + 1)]
        else:
            assert len(out_gib_channels) == num_levels, "The number of out_gib_channels must be equal to the number of levels"

        self.pyramid_builder = pyramid_builder

        # Build the GIB layers
        self.gib_neigh_encoders = nn.ModuleList()
        self.gib_pooling_encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        enc_channels = []
        f_channels = in_channels
        for i in range(num_levels):
            out_channels = out_gib_channels[i]
            # print(f"GIBLi Building Encoding Level {i} with {f_channels} input channels and {out_channels} out channels")
            gib_seq = GIB_Sequence(num_layers=(i+1), 
                                   gib_dict=gib_dict, 
                                   feat_channels=f_channels, 
                                   num_observers=num_observers, 
                                   kernel_size=kernel_size, 
                                   out_channels=out_channels
                                )
            f_channels = out_channels
            self.gib_neigh_encoders.append(gib_seq)
            enc_channels.append(out_channels)

        unpool_concat_channels = True
        
        for i in range(num_levels - 1):
            # Build pooling layers
            f_channels = enc_channels[i]
            # print(f"GIBLi Building Pooling Level {i} with {f_channels} channels")
            # maintain the same number of channels for the pooling encoder
            gib_seq = GIB_Sequence(num_layers=(i+1), 
                                   gib_dict=gib_dict, 
                                   feat_channels=f_channels, 
                                   num_observers=num_observers, 
                                   kernel_size=kernel_size, 
                                   out_channels=f_channels, 
                                   strided=True
                                )
            self.gib_pooling_encoders.append(gib_seq)

            # Build unpooling layers
            dec = Decoder(feat_channels=enc_channels[i+1], 
                          skip_channels=enc_channels[i], 
                          unpool_out_channels=enc_channels[i], 
                          skip=skip_connections,
                          concat=unpool_concat_channels,
                          backend='interp',
                          num_layers=(i+1),
                          gib_dict=gib_dict,
                          num_observers=num_observers,
                          kernel_size=kernel_size,
                        )
            
            
            self.decoders.append(dec)

        self.seg_head = nn.Sequential(
            nn.Linear(enc_channels[0], enc_channels[0]),
            PointBatchNorm(enc_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(enc_channels[0], num_classes)
        )
        
        
        
    def maintain_convexity(self):
        for i in range(self.num_levels):
            self.gib_neigh_encoders[i].maintain_convexity()

            if i < self.num_levels - 1:
                self.gib_pooling_encoders[i].maintain_convexity()
                self.decoders[i].maintain_convexity()
                
                
    def get_gib_parameters(self) -> List[torch.Tensor]:
        gib_params = []
        for i in range(self.num_levels):
            gib_params.extend(self.gib_neigh_encoders[i].get_gib_params())

            if i < self.num_levels - 1:
                gib_params.extend(self.gib_pooling_encoders[i].get_gib_params())
                gib_params.extend(self.decoders[i].get_gib_params())
                
                
        # print(f"GIB parameters:\n {gib_params}")
                
        return gib_params
                
                
    def get_cvx_coefficients(self) -> List[torch.Tensor]:
        # get the convex coefficients for every GIB layer
        cvx_coefs = []
        for i in range(self.num_levels):
            cvx_coefs.extend(self.gib_neigh_encoders[i].get_cvx_coefficients())

            if i < self.num_levels - 1:
                cvx_coefs.extend(self.gib_pooling_encoders[i].get_cvx_coefficients())
                cvx_coefs.extend(self.decoders[i].get_cvx_coefficients())
        
        return cvx_coefs
    
    def gibli_forward(self, x:torch.Tensor, graph_pyramid_dict=None) -> torch.Tensor:
        
        if x.shape[-1] > 3:
            feats = x #x[..., 3:]
        else:
            feats = x # we consider the coords as features
        coords = x[..., :3]
        
        # print(f"{coords.shape=} --- {feats.shape=}")

        # Build the graph pyramid
        if graph_pyramid_dict is None:
            graph_pyramid_dict = self.pyramid_builder(coords)
            
        # for key, val in graph_pyramid_dict.items():
        #     print(f"{key=}...", end="")
        #     for t in val:
        #        print(f"{t.shape=}", end="")
        #     print()

        point_list = graph_pyramid_dict['points_list'] # shape of 0th element: (batch, num_points, 3)
        neighbors_idxs_list = graph_pyramid_dict['neighbors_idxs_list'] # shape of 0th element: (B, Q[0], neighborhood_size[0]) idxs from points[0]
        subsampling_idxs_list = graph_pyramid_dict['subsampling_idxs_list'] # shape of 0th element: (B, Q[1], neighborhood_size[1]) idxs from points[0]
        upsampling_idxs_list = graph_pyramid_dict['upsampling_idxs_list'] # shape of 0th element: (B, Q[-1], neighborhood_size[-1]) idxs from points[-2]

        level_feats = []

        for i in range(self.num_levels): # encoding phase
            
            if i > 0: ###### Pooling phase ######
                # print(f"Pooling {i}...")
                # print(f"\tfeats.shape={tuple(feats.shape)} \n\tpoint_list[{i}].shape={tuple(point_list[i].shape)} \n\tsubsampling_idxs_list[{i-1}].shape={tuple(subsampling_idxs_list[i-1].shape)}")
                feats = self.gib_pooling_encoders[i - 1]((coords, feats), point_list[i], subsampling_idxs_list[i - 1]) # pooling
                # print(f"\t{feats.shape}\n")
                
                coords = point_list[i] # update the coordinates
            
            ###### Encoding phase ######
            # print(f"Encoding {i}...")
            # print(f"{coords.dtype=}, {feats.dtype=}, {neighbors_idxs_list[i].dtype=}")
            # print(f"\tfeats.shape={tuple(feats.shape)} \n\tpoint_list[{i}].shape={tuple(point_list[i].shape)} \n\tneighbors_idxs_list[{i}].shape={tuple(neighbors_idxs_list[i].shape)}")
            feats = self.gib_neigh_encoders[i]((coords, feats), point_list[i], neighbors_idxs_list[i])
            # print(f"\t {feats.shape=}\n")
            level_feats.append(feats) # save the features for skip connections

            # coords = point_list[i+1]

        curr_latent_feats = level_feats[-1] # N 
        curr_coords = point_list[-1] # N
        ###### Decoding phase ######
        for i in reversed(range(len(upsampling_idxs_list))): # there are num_levels - 1 unpooling layers
            skip_coords, skip_feats, skip_neighbors_idxs = point_list[i], level_feats[i], neighbors_idxs_list[i]
            # print(f"{skip_coords.dtype=}, {skip_feats.dtype=}, {skip_neighbors_idxs.dtype=}")
            # print(f"{curr_coords.dtype=}, {curr_latent_feats.dtype=}, {upsampling_idxs_list[i].dtype=}")
            # print(f"Decoding {i + 1}...")
            # print(f"\tcurr_coords.shape={tuple(curr_coords.shape)} \n\tcurr_latent_feats.shape={tuple(curr_latent_feats.shape)} \n\tskip_coords.shape={tuple(skip_coords.shape)} \n\tskip_feats.shape={tuple(skip_feats.shape)} \n\tupsampling_idxs_list[{i}].shape={tuple(upsampling_idxs_list[i].shape)}")
            curr_latent_feats = self.decoders[i]((curr_coords, curr_latent_feats), (skip_coords, skip_feats), upsampling_idxs_list[i], skip_neighbors_idxs)
            curr_coords = skip_coords
            #print(f"\t{curr_latent_feats.shape=}\n")
            
        return curr_latent_feats


    def forward(self, x:torch.Tensor, graph_pyramid_dict=None) -> torch.Tensor:
        # this assumes that the input is of shape (B, N, 3 + F), where N is the number of points and F is the number of features (F >= 0).
        # the batch_dim is necessary for now
       
        curr_latent_feats = self.gibli_forward(x, graph_pyramid_dict)

        ###### Segmentation phase ######
        seg_logits = self.seg_head(curr_latent_feats)
        # print(seg_logits.shape)
        
        torch.cuda.empty_cache()
        
        return seg_logits




#######################################################################
# GIBLi with pops

from core.models.giblinet.GIBLi_parts import DownBlock, Decoder_pops, GIBLiBlock



class GIBLiNet_pops(nn.Module):
    
    def __init__(self,
                 in_channels:int,
                 num_classes:int,
                 num_levels:int,
                 ### Down Block args
                 sota_class:object,
                 sota_input_format:str, # this can be: ['batch', 'ptv1', 'kpconv']
                 grid_size:Union[float, List[float]],
                 gib_dict:Dict[str, int],
                 num_observers:Union[int, List[int]],
                 kernel_size:float,
                 neighbor_size:Union[int, List[int]]=4,
                 embed_channels:Union[int, List[int]]=16,
                 out_channels:Union[int, List[int]]=16,
                 sota_kwargs:Dict[str, object]=None,
                 sota_update_kwargs:Dict[str, object]=None,
                 ### Decoder args
                 bias=True,
                 skip=True,
                 concat:bool=False,
                 backend="interp",
                ) -> None:
        
        
        super(GIBLiNet_pops, self).__init__()
        
        self.embed_channels = embed_channels if isinstance(embed_channels, list) else [embed_channels*(i + 1) for i in range(num_levels)]
        self.out_channels = out_channels if isinstance(out_channels, list) else [out_channels*(i + 1) for i in range(num_levels)]
        self.grid_size = grid_size if isinstance(grid_size, list) else [grid_size*(i + 1) for i in range(num_levels)]

        self.gib_blocks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.num_levels = num_levels
        self.in_channels = in_channels
        
        self.first_gib_block = GIBLiBlock(in_channels,
                                        sota_class,
                                        sota_input_format,
                                        gib_dict,
                                        num_observers=num_observers,
                                        kernel_size=kernel_size,
                                        neighbor_size=neighbor_size,
                                        embed_channels=self.embed_channels[0],
                                        out_channels=self.out_channels[0],
                                        sota_args=sota_kwargs
                                    )
        
        self.last_decoder = Decoder_pops(self.embed_channels[0],
                                        self.embed_channels[0],
                                        self.embed_channels[0],
                                        bias,
                                        skip,
                                        concat,
                                        backend,
                                        #### gib 
                                        gib_dict,
                                        num_observers,
                                        kernel_size,
                                        sota_class,
                                        sota_input_format,
                                        neighbor_size,
                                        self.embed_channels[0],
                                        self.out_channels[0],
                                        sota_kwargs
                                    )
        
        
        in_channels = self.out_channels[0]        
        
        for i in range(num_levels):
            gib_block = DownBlock(in_channels,
                                self.embed_channels[i],
                                self.grid_size[i],
                                sota_class,
                                sota_input_format,
                                gib_dict,
                                num_observers,
                                kernel_size,
                                neighbor_size,
                                self.embed_channels[i],
                                self.out_channels[i],
                                sota_kwargs,
                            )
            
            self.gib_blocks.append(gib_block)
            in_channels = self.out_channels[i]
            
            if i < num_levels - 1:
                dec  = Decoder_pops(self.embed_channels[i+1],
                                    self.embed_channels[i],
                                    self.embed_channels[i],
                                    bias,
                                    skip,
                                    concat,
                                    backend,
                                    #### gib 
                                    gib_dict,
                                    num_observers,
                                    kernel_size,
                                    sota_class,
                                    sota_input_format,
                                    neighbor_size,
                                    self.embed_channels[i],
                                    self.out_channels[i],
                                    sota_kwargs
                                    )
                self.decoders.append(dec)

                # if there is still a next layer, update SOTA kwargs
                for key in sota_update_kwargs: # update the kwargs if necessary
                    if isinstance(sota_update_kwargs[key], list): # per-level update
                        sota_kwargs[key] = sota_update_kwargs[key][i]*sota_kwargs[key]
                    else: # global update
                        sota_kwargs[key] = sota_update_kwargs[key]*sota_kwargs[key] 
                    
            
        self.seg_head = nn.Sequential(
            nn.Linear(self.out_channels[0], self.out_channels[0]),
            PointBatchNorm(self.out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels[0], num_classes)
        ) if num_classes > 0 else nn.Identity()
        
        
        
    def forward(self, input_dict:Dict[str, torch.Tensor]) -> torch.Tensor:
        
        input_dict = self.first_gib_block(input_dict)
          
        # print(f"First GIB")
        # for key, val in input_dict.items():
        #     print(f"{key=}, {val.shape=}")
        # print()
        
        level_dicts = [input_dict]
        upsample_list = []
        for i in range(len(self.gib_blocks)):
            input_dict, upsample_idxs = self.gib_blocks[i](input_dict)
            
            # print(f"Encoding {i}")
            # for key, val in input_dict.items():
            #     print(f"{key=}, {val.shape=}")
            # print()
            
            if i < self.num_levels - 1:
                level_dicts.append(input_dict)
            
            upsample_list.append(upsample_idxs)
            
        for i in reversed(range(len(self.decoders))):
            input_dict = self.decoders[i](input_dict, level_dicts[i+1], upsample_list[i+1])
            # print(f"Decoding {i}")
            # for key, val in input_dict.items():
            #     print(f"{key=}, {val.shape=}")
            # print()
        
        input_dict = self.last_decoder(input_dict, level_dicts[0], upsample_list[0])
            
        seg_logits = self.seg_head(input_dict['feat'])
        return seg_logits




if __name__ == "__main__":
    import sys
    sys.path.append('../')
    sys.path.append('../../')
    from utils import constants as C
    from core.datasets.TS40K import TS40K_FULL_Preprocessed
    from core.lit_modules.lit_ts40k import LitTS40K_FULL_Preprocessed


    ts40k = TS40K_FULL_Preprocessed(
        C.TS40K_FULL_PREPROCESSED_PATH,
        split='fit',
        sample_types=['tower_radius', '2_towers'],
        transform=None,
        load_into_memory=False
    )

    # sample = ts40k[0]
    # points, labels = sample[0], sample[1]
    # print(points.shape, labels.shape)
    
    # # make random torch data to test the model
    # # x = torch.rand(1, 10000, 6)#.cuda()

    # # define the model
    # in_channels = 3
    # num_classes = 10
    # num_levels = 4
    # out_gib_channels = 16
    # num_observers = 16
    # kernel_size = 0.1
    # gib_dict = {
    #     'cy' : 2,
    #     'ellip': 2,
    #     'disk': 2,
    #     'cone': 2
    # }

    # neighborhood_strategy = "knn"
    # neighborhood_size = 4
    # neighborhood_kwargs = {}
    # neighborhood_update_kwargs = {}

    # skip_connections = True
    # graph_strategy = "fps"
    # graph_pooling_factor = 2


    # # model = GIBLiNet(in_channels, 
    # #                 num_classes, 
    # #                 num_levels, 
    # #                 out_gib_channels, 
    # #                 num_observers, 
    # #                 kernel_size, 
    # #                 gib_dict, 
    # #                 neighborhood_strategy, 
    # #                 neighborhood_size, 
    # #                 neighborhood_kwargs, 
    # #                 neighborhood_update_kwargs, 
    # #                 skip_connections, 
    # #                 graph_strategy, 
    # #                 graph_pooling_factor
    # #             ).cuda()

    # # pred = model(points.unsqueeze(0).cuda())
    # # print(pred.shape) # should be (1, 10_000, 10) for 10 classes
    
    # neighboring = Neighboring(neighborhood_strategy, neighborhood_size, **neighborhood_kwargs)
    # glayer = GIBLiLayer(in_channels, 16, num_observers, kernel_size, gib_dict, neighboring, gib_layers=1).cuda()
    
    # pred = glayer(points.unsqueeze(0).cuda())
    
    # print(pred.shape)
    
    
    #### test GIBLiNet_pops
    
    lit_ts40k = LitTS40K_FULL_Preprocessed(
        data_dir=C.TS40K_FULL_PREPROCESSED_PATH,
        batch_size=4,
        sample_types='all',
        transform=None,
        transform_test=None,
        num_workers=8,
        val_split=0.1,
        load_into_memory=False,
        use_full_test_set=False
    )
    
    gib_dict = {
        'cy' : 8,
        'ellip': 8,
        'disk': 8,
        'cone': 8
    }
    
    gibli = GIBLiNet_pops(
        in_channels=3,
        num_classes=5,
        num_levels=4,
        sota_class=torch.nn.Linear,
        sota_input_format='batch',
        grid_size=[0.01, 0.02, 0.04, 0.08],
        gib_dict=gib_dict,
        num_observers=16,
        kernel_size=0.1,
        neighbor_size=[4, 16],
        embed_channels=[8, 16, 32, 32],
        out_channels=[8, 16, 32, 32],
        sota_kwargs={'out_features': 8},
        sota_update_kwargs={'out_features': [2, 2, 1]},
        backend='map'
    ).to('cuda')
    
    # input_dict = ts40k[0]
    # for key, val in input_dict.items():
    #     input_dict[key] = val.to('cuda')    
    # pred = gibli(input_dict)
    # print(f"{pred.shape=}")
    
    
    lit_ts40k.setup('fit')
    
    train_dl = lit_ts40k.train_dataloader()
    
    for i, batch in enumerate(train_dl):
        batch = {key: val.to('cuda') for key, val in batch.items()}
        print(f"{batch['coord'].shape=}, {batch['offset'].shape=}, {batch['segment'].shape=}")   
        pred = gibli(batch)
        print(f"{pred.shape=}")
        break
    
    