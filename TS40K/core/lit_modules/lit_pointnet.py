


import torch
from lit_modules.lit_model_wrappers import LitWrapperModel

from models.pointnet.models.pointnet2_sem_seg import get_model as get_pointnet2_model
from models.pointnet.models.pointnet2_sem_seg import get_loss as get_pointnet2_loss
from models.pointnet.models.pointnet_sem_seg import get_model as get_pointnet_model
from models.pointnet.models.pointnet_sem_seg import get_loss as get_pointnet_loss






class LitPointNet(LitWrapperModel):


    def __init__(self, 
                 model:str,
                 optimizer_name: str, 
                 num_classes = 10,
                 learning_rate=0.01, 
                 metric_initializer=None, 
                 **kwargs):
        
        if model == 'pointnet':
            model = get_pointnet_model(num_classes)
            criterion = get_pointnet_loss()
        elif model == 'pointnet2':
            model = get_pointnet2_model(num_classes)
            criterion = get_pointnet2_loss()
        else:
            raise ValueError(f"Unknown model {model}")
        
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, **kwargs)


    def forward(self, x):
        return self.model(x)
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)
    

    
    


    
