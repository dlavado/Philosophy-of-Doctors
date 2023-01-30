
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.5
BETA = 1
GAMMA = 2


class TverskyLoss(nn.Module):
    """

    Tversky loss for binary classification introduced in: 
        Tversky loss function for image segmentation using 3D FCDN. 2017 in https://arxiv.org/abs/1706.05721.

    in the case of alpha=beta=0.5 the Tversky index simplifies to be the same as the Dice coefficient, which is also equal to the F1 score

    With alpha=beta=1, Equation 2 produces Tanimoto coefficient, and setting alpha+beta=1 produces the set of Fβ scores.
    Larger betas weigh recall higher than precision, and vice versa for smaller betas.

    Parameters:
    ----------
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        smooth: smooth factor to avoid division by zero.
    """


    def __init__(self):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for binary classification introduced in:
        Focal Tversky loss: a novel loss function and DSC (Dice score) maximization approach for lesion segmentation. 2019 in https://arxiv.org/abs/1810.07842.

    Parameters:
    ----------
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        gamma: controls the penalty for easy examples.
    """
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky