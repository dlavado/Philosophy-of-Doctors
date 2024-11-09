

import torch


def resolve_segmentation_criterion(criterion_name, ignore_index=-1, **kwargs) -> torch.nn.Module:
    from segmentation_models_pytorch import losses as L

    criterion_name = criterion_name.lower()
    if criterion_name == 'dice':
        # kwargs: ---
        return L.DiceLoss(mode=L.MULTICLASS_MODE, ignore_index=ignore_index, **kwargs)
    elif criterion_name == 'jaccard':
        # kwargs: 
        return L.JaccardLoss(mode=L.MULTICLASS_MODE, ignore_index=ignore_index, **kwargs)
    elif criterion_name == 'tversky':
        # kwargs: alpha=0.5, beta=0.5, gamma=1.0
        return L.TverskyLoss(mode=L.MULTICLASS_MODE, ignore_index=ignore_index, **kwargs)
    elif criterion_name == 'focal':
        # kwargs: gamma=2.0
        return L.FocalLoss(mode=L.MULTICLASS_MODE, ignore_index=ignore_index, **kwargs)
    else:
        raise NotImplementedError(f'Criterion {criterion_name} not implemented')
    
    
    
class SegLossWrapper(torch.nn.Module):
    """
    Wrapper for Image segmentation loss functions
    """
    
    def __init__(self, criterion_name:str, ignore_index=-1, **kwargs) -> None:
        super(SegLossWrapper, self).__init__()
        
        self.criterion = resolve_segmentation_criterion(criterion_name, ignore_index, **kwargs)
        
    def forward(self, model_output:torch.Tensor, target:torch.Tensor) -> torch.Tensor:

        # model output shape = (B*P, C) -> (1, B*P, 1, C)
        # target shape = (B*P) -> (1, B*P, 1)
        
        
        model_output = model_output.unsqueeze(0).permute(0, 2, 1).unsqueeze(-1) # (1, C, B*P, 1), this is the shape expected by the loss function
        target = target.unsqueeze(0).unsqueeze(-1) # (1, B*P, 1)
        
        return self.criterion(model_output, target)
        