

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    """
    Focal loss for binary classification introduced in:
        Lin et al. Focal Loss for Dense Object Detection. ICCV 2017 in https://arxiv.org/abs/1708.02002.

    The gamma parameter controls the focus of the loss. When gamma is 0, the loss is equivalent to cross entropy.
    When gamma is larger than 0, the loss is more focused on hard examples.
    """

    def __init__(self, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.reduction = reduction


    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss