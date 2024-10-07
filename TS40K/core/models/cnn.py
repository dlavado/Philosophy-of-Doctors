



import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

import utils.voxelization as Vox



class CNN_Baseline(nn.Module):

    def __init__(self,
                in_channels, 
                out_channels, 
                kernel_size=3, 
                num_groups=1,
                padding='same', 
                MLP_hidden_dims=[], 
                num_classes=10
                ) -> None:
        
        super(CNN_Baseline, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=num_groups)

        self.convs = []
        self.bns = []
        self.num_classes = num_classes

        self.feat_dim = 3 + out_channels*2


        self.convs.append(nn.Conv1d(self.feat_dim, MLP_hidden_dims[0], kernel_size=1))
        self.bns.append(nn.BatchNorm1d(MLP_hidden_dims[0]))

        for i in range(len(MLP_hidden_dims)-1):
            self.convs.append(nn.Conv1d(MLP_hidden_dims[i], MLP_hidden_dims[i+1], kernel_size=1))
            self.bns.append(nn.BatchNorm1d(MLP_hidden_dims[i+1]))
        
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        
        self.out_conv = nn.Conv1d(MLP_hidden_dims[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, pt_loc: torch.Tensor):
        conv_features = self.conv1(x) # (batch_size, out_channels, D, H, W)

        conv_pts = Vox.vox_to_pts(conv_features, pt_loc)

        global_feat_descriptor = conv_pts # (batch, P, out_channels)
        global_feat_descriptor = global_feat_descriptor.transpose(2,1).contiguous() # (batch, out_channels, P)
        
        num_points = pt_loc.shape[1]
        global_feat_descriptor = torch.max_pool1d(global_feat_descriptor, kernel_size=num_points, padding=0, stride=1) # (batch, out_channels, 1)
        
        global_feat_descriptor = global_feat_descriptor.squeeze(2).unsqueeze(1) # (batch, 1, out_channels)
        
        x = torch.cat([pt_loc, conv_pts], dim=2) # (batch, P, 3 + out_channels)
        x = torch.cat([x, global_feat_descriptor.repeat(1, num_points, 1)], dim=2) # (batch, P, 3 + out_channels*2)
        x = x.transpose(2,1).contiguous() # (batch, 3 + out_channels*2, P)

        ####### MLP HEAD #######
        x = x.to(torch.float32)

        for conv, bn in zip(self.convs, self.bns):
            # print(f"x = shape: {x.shape}, req_grad: {x.requires_grad}, type: {x.dtype}")
            # print(f"conv = shape: {conv.weight.shape}, req_grad: {conv.weight.requires_grad}, type: {conv.weight.dtype}")
            x = torch.relu(bn(conv(x))) # (batch, hidden_dims[i+1], P)
        
        x = self.out_conv(x) # (batch, num_classes, P)

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
    

