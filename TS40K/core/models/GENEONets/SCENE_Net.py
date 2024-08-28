from typing import List, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys


sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')

import utils.voxelization as Vox
from core.datasets.torch_transforms import Normalize_PCD
from core.models.GENEONets.GENEO_utils import GENEO_Layer, GENEO_Transposed


class SceneNet_multiclass(nn.Module):
    """

    SceneNet 1.5 - 3D Convolutional Neural Network for Scene Classification based on GENEOs
    multiclass is accomplished by adding an MLP classifier after the GENEO feature extraction
    """


    def __init__(self, 
                geneo_num:Mapping[str, int]=None, 
                num_observers:List=[10],
                kernel_size:tuple=None,
                hidden_dims:List[int]=None,
                num_classes:int=2) -> None:
        
        super(SceneNet_multiclass, self).__init__()

        self.geneo_layers:List[GENEO_Layer] = []
        for num_o in num_observers:
            self.geneo_layers.append(GENEO_Layer(geneo_num, num_o, kernel_size)) 

        self.geneo_layers = nn.ModuleList(self.geneo_layers)
        
        # self.scenenet = GENEO_Layer(geneo_num, num_observers, kernel_size)

        self.feat_dim = 3 + (sum(num_observers) + sum(geneo_num.values())*len(num_observers))*2

        self.convs = []
        self.bns = []
        self.activation = nn.ReLU()
        self.num_classes = num_classes

        if hidden_dims is not None and len(hidden_dims) > 0:
            self.convs.append(nn.Conv1d(self.feat_dim, hidden_dims[0], kernel_size=1))
            self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

            for i in range(len(hidden_dims)-1):
                self.convs.append(nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=1))
                self.bns.append(nn.BatchNorm1d(hidden_dims[i+1]))
            
            self.convs = nn.ModuleList(self.convs)
            self.bns = nn.ModuleList(self.bns)
            
            self.out_conv = nn.Conv1d(hidden_dims[-1], num_classes, kernel_size=1)

    def get_geneo_params(self):
        g_params = nn.ParameterDict()
        for scene in self.geneo_layers:
            g_params.update(scene.get_geneo_params())
        return g_params

    def get_cvx_coefficients(self):
        cvx_coeffs = nn.ParameterDict()
        for scene in self.geneo_layers:
            cvx_coeffs.update(scene.get_cvx_coefficients())
        return cvx_coeffs
    
    def maintain_convexity(self):
        for scene in self.geneo_layers:
            scene.maintain_convexity()


    def feature_extraction(self, x:torch.Tensor, pt_loc: torch.Tensor) -> torch.Tensor:
        num_points = pt_loc.shape[1]
        
        observer_pts = None
        conv_pts = None
        for scene_layer in self.geneo_layers:
            conv = scene_layer._perform_conv(x) # (batch, num_geneos, z, x, y)
            observer = scene_layer._observer_cvx_combination(conv) # (batch, num_observers, z, x, y)
            if observer_pts is None:
                observer_pts = Vox.vox_to_pts(observer, pt_loc) # (batch, P, num_observers[0])
                conv_pts = Vox.vox_to_pts(conv, pt_loc) # (batch, P, num_geneos)
            else:
                observer_pts = torch.cat([observer_pts, Vox.vox_to_pts(observer, pt_loc)], dim=2) # (batch, P, num_observers*1^T)
                conv_pts = torch.cat([conv_pts, Vox.vox_to_pts(conv, pt_loc)], dim=2) # (batch, P, num_geneos*1^T)


        global_feat_descriptor = torch.cat([conv_pts, observer_pts], dim=2) # (batch, P, sum(num_geneos) + sum(num_observers))
        global_feat_descriptor = global_feat_descriptor.transpose(2,1).contiguous() # (batch, num_observers*1^T, P)
        global_feat_descriptor = torch.max_pool1d(global_feat_descriptor, kernel_size=num_points, padding=0, stride=1) # (batch, num_observers*1^T, 1)
        global_feat_descriptor = global_feat_descriptor.squeeze(2).unsqueeze(1) # (batch, 1, num_observers*1^T + num_geneos*1^T)

        x = torch.cat([pt_loc, conv_pts, observer_pts], dim=2) # (batch, P, 3 + num_observers*1^T + num_geneos*1^T)
        x = torch.cat([x, global_feat_descriptor.repeat(1, num_points, 1)], dim=2) # (batch, P, 3 + (num_observers*1^T + num_geneos*1^T)*2)
        x = x.transpose(2,1).contiguous() # (batch, 3 + (num_observers*1^T + num_geneos*1^T)*2, P)

        # print(f"x transposed = shape: {x.shape}, req_grad: {x.requires_grad}")
        return x

    def forward(self, x:torch.Tensor, pt_loc:torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------

        `x`: torch.Tensor
            Input tensor of shape (batch, z, x, y)

        `pt_loc`: torch.Tensor
            Input tensor of shape (batch, P, 3) where P is the number of points

        Returns
        -------

        `pts`: torch.Tensor
            Output tensor of shape (batch, P, num_classes)
        """

        batch_size = x.shape[0]

        x = self.feature_extraction(x, pt_loc)


        ###################### MLP HEAD ######################
        x = x.to(torch.float32)

        for conv, bn in zip(self.convs, self.bns):
            # print(f"x = shape: {x.shape}, req_grad: {x.requires_grad}, type: {x.dtype}")
            # print(f"conv = shape: {conv.weight.shape}, req_grad: {conv.weight.requires_grad}, type: {conv.weight.dtype}")
            x = self.activation(bn(conv(x))) # (batch, hidden_dims[i+1], P)
            # x = self.activation(conv(x)) # (batch, hidden_dims[i+1], P)
        
        x = self.out_conv(x) # (batch, num_classes, P)

        # print(f"x = shape: {x.shape}, req_grad: {x.requires_grad}")

        return x
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------

        `model_output`: torch.Tensor
            Output tensor of shape (batch, num_classes, P)

        Returns
        -------

        `pts`: torch.Tensor
            Output tensor of shape (batch, P)
        """

        return torch.argmax(model_output.permute(0, 2, 1), dim=2) # (batch, P)

    
    def _compute_loss(pred:torch.Tensor, target:torch.Tensor, weights:torch.Tensor=None) -> torch.Tensor:
        """

        Parameters
        ----------

        `pred`: torch.Tensor
            Prediction tensor of shape (batch, P, num_classes)

        `target`: torch.Tensor
            Target tensor of shape (batch, P)

        `weights`: torch.Tensor
            Weights tensor of shape (batch, P)

        Returns
        -------

        `loss`: torch.Tensor
            Loss tensor of shape (batch, P)
        """

        if weights is None:
            weights = torch.ones_like(target) # equal weights

        loss = F.nll_loss(pred, target, reduction='none')
        loss = torch.sum(loss*weights, dim=1)

        return loss



###############################################################
#                       SCENENet UNET                         #
###############################################################

class SceneNet_Unet(nn.Module):

    def __init__(self, 
                geneo_num:Mapping[str, int]=None, 
                num_observers:int=1,
                scene_layers = 1,
                kernel_size:tuple=None,
                hidden_dims:List[int]=None,
                num_classes:int=2) -> None:
        
        super(SceneNet_Unet, self).__init__()

        self.geneo_layers:List[GENEO_Layer] = []
        for _ in range(scene_layers):
            self.geneo_layers.append(GENEO_Layer(geneo_num, num_observers, kernel_size)) 
            num_observers = num_observers*2

        self.geneo_layers = nn.ModuleList(self.geneo_layers)


    def get_geneo_params(self):
        return [scene.get_geneo_params() for scene in self.geneo_layers]

    def get_cvx_coefficients(self):
        return [scene.get_cvx_coefficients() for scene in self.geneo_layers]
    
    def maintain_convexity(self):
        for scene in self.geneo_layers:
            scene.maintain_convexity()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------

        `x`: torch.Tensor
            Input tensor of shape (batch, z, x, y)

        Returns
        -------

        `pts`: torch.Tensor
            Output tensor of shape (batch, P, num_classes)
        """

        for scene_layer in self.geneo_layers:
            print(f"x = shape: {x.shape}, req_grad: {x.requires_grad}")
            x = scene_layer(x)

        return x
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------

        `model_output`: torch.Tensor
            Output tensor of shape (batch, num_classes, P)

        Returns
        -------

        `pts`: torch.Tensor
            Output tensor of shape (batch, P)
        """

        return torch.argmax(model_output.permute(0, 2, 1), dim=2) # (batch, P)

    
    def _compute_loss(pred:torch.Tensor, target:torch.Tensor, weights:torch.Tensor=None) -> torch.Tensor:
        """

        Parameters
        ----------

        `pred`: torch.Tensor
            Prediction tensor of shape (batch, P, num_classes)

        `target`: torch.Tensor
            Target tensor of shape (batch, P)

        `weights`: torch.Tensor
            Weights tensor of shape (batch, P)

        Returns
        -------

        `loss`: torch.Tensor
            Loss tensor of shape (batch, P)
        """

        if weights is None:
            weights = torch.ones_like(target) # equal weights

        loss = F.nll_loss(pred, target, reduction='none')
        loss = torch.sum(loss*weights, dim=1)

        return loss







###############################################################
#                       SCENENet + CNN                        #
###############################################################


class SceneNet_multiclass_CNN(SceneNet_multiclass):


    def __init__(self, 
                geneo_num:Mapping[str, int]=None, 
                num_observers:int=1,
                kernel_size:tuple=None,
                MLP_hidden_dims:List[int]=None,
                num_classes:int=2,
                cnn_out_channels:int=4,
                cnn_kernel_size:int=2,
            ) -> None:
        
        super(SceneNet_multiclass_CNN, self).__init__(geneo_num, num_observers, kernel_size, MLP_hidden_dims, num_classes)
        
        cnn_in_channels = sum(num_observers) + sum(geneo_num.values())*len(num_observers)
        
        self.conv1 = torch.nn.Conv3d(cnn_in_channels, cnn_out_channels, kernel_size=cnn_kernel_size, padding='same', groups=1)

        self.feat_dim = 3 + (cnn_in_channels + cnn_out_channels)*2

        self.convs[0] = nn.Conv1d(self.feat_dim, MLP_hidden_dims[0], kernel_size=1) # change the input channels to the MLP
    

    def forward(self, x:torch.Tensor, pt_loc:torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------

        `x`: torch.Tensor
            Input tensor of shape (batch, z, x, y)

        `pt_loc`: torch.Tensor
            Input tensor of shape (batch, P, 3) where P is the number of points

        Returns
        -------

        `pts`: torch.Tensor
            Output tensor of shape (batch, P, num_classes)
        """

        num_points = pt_loc.shape[1]
        
        observer_vox = None
        conv_vox = None
        for scene_layer in self.geneo_layers:
            conv = scene_layer._perform_conv(x) # (batch, num_geneos, z, x, y)
            observer = scene_layer._observer_cvx_combination(conv) # (batch, num_observers, z, x, y)
            if observer_vox is None:
                observer_vox = observer # (batch, P, num_observers[0])
                conv_vox = conv # (batch, P, num_geneos)
            else:
                observer_vox = torch.cat([observer_vox, observer], dim=1) 
                conv_vox = torch.cat([conv_vox, conv], dim=1)

        print(f"observer_vox = shape: {observer_vox.shape}, req_grad: {observer_vox.requires_grad}")
        cvx_lambda = self.geneo_layers[0].get_cvx_coefficients()
        cvx_lambda = cvx_lambda['lambda']
        print(f"cvx_lambda = shape: {cvx_lambda.shape}, req_grad: {cvx_lambda.requires_grad}")
        print(f"cvx coefficients describe: min: {cvx_lambda.min()}; max: {cvx_lambda.max()}; mean: {cvx_lambda.mean()}; std: {cvx_lambda.std()}")
        print(f"cvx_lambda = {cvx_lambda}")
        for i in range(len(observer_vox[0])):
            Vox.plot_voxelgrid(observer_vox[0][i].detach().cpu().numpy(), color_mode='density', plot=True)
        input("Press Enter to Continue...")

        feats = torch.cat([conv, observer], dim=1) # (batch, num_geneos*len(num_observers) + sum(num_observers), z, x, y)

        conv_features = self.conv1(feats) # (batch_size, out_channels, D, H, W)

        conv_vox = Vox.vox_to_pts(conv_features, pt_loc) # (batch, P, out_channels)
        observer_vox = Vox.vox_to_pts(feats, pt_loc) # (batch, P, num_observers)

        ### build global feature descriptor
        global_feat_descriptor = torch.cat([conv_vox, observer_vox], dim=2)
        global_feat_descriptor = global_feat_descriptor.transpose(2,1).contiguous() # (batch, out_channels + num_observers, P)
        global_feat_descriptor = torch.max_pool1d(global_feat_descriptor, kernel_size=num_points, padding=0, stride=1) # (batch, out_channels + num_observers, 1)

        global_feat_descriptor = global_feat_descriptor.squeeze(2).unsqueeze(1) # (batch, 1, out_channels + num_observers)

        x = torch.cat([pt_loc, conv_vox, observer_vox], dim=2) # (batch, P, 3 + out_channels + num_observers)
        x = torch.cat([x, global_feat_descriptor.repeat(1, num_points, 1)], dim=2) # (batch, P, 3 + (out_channels + num_observers)*2)

        x = x.transpose(2,1).contiguous() # (batch, 3 + (out_channels + num_observers)*2, P)

        ###################### MLP HEAD ######################
        x = x.to(torch.float32)
        for conv, bn in zip(self.convs, self.bns):
            # print(f"x = shape: {x.shape}, req_grad: {x.requires_grad}, type: {x.dtype}")
            # print(f"conv = shape: {conv.weight.shape}, req_grad: {conv.weight.requires_grad}, type: {conv.weight.dtype}")
            x = torch.relu(bn(conv(x))) # (batch, hidden_dims[i+1], P)
        
        x = self.out_conv(x) # (batch, num_classes, P)

        return x



###############################################################
#                     SCENENet PreProcess                     #
###############################################################

class SceneNet_PreBackbone(SceneNet_multiclass):


    def __init__(self, 
                geneo_num:Mapping[str, int]=None, 
                num_observers:int=1,
                kernel_size:tuple=None,
            ) -> None:
        
        super(SceneNet_PreBackbone, self).__init__(geneo_num, num_observers, kernel_size, [], 0)
        self.feat_dim = self.feat_dim - 3 # remove the 3 coordinates
        
    

    def forward(self, x:torch.Tensor, pt_loc:torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------

        `x`: torch.Tensor
            Input tensor of shape (batch, z, x, y)

        `pt_loc`: torch.Tensor
            Input tensor of shape (batch, P, 3) where P is the number of points

        Returns
        -------

        `pts`: torch.Tensor
            Output tensor of shape (batch, P, num_classes)
        """
        
        x = self.feature_extraction(x, pt_loc)
        x = x.permute(0, 2, 1) # (batch, P, 3 + (num_observers*1^T + num_geneos*1^T)*2)
        return x

if __name__ == "__main__":
    import sys
    import os
   
    
    # make random torch data to test the model
    x = torch.rand((2, 1, 32, 32, 32)).cuda() # (batch, c, z, x, y)

    trans = GENEO_Transposed(in_channels=1, out_channels=4, kernel_size=2, stride=2).cuda()

    out = trans(x)

    print(out.shape)


    #gnet = SceneNet15_Unet({'cy': 5, 'neg': 5, 'arrow': 5}, num_observers=4, scene_layers=3, kernel_size=(9, 6, 6)).cuda()
    #print(gnet(x).shape)
    

    