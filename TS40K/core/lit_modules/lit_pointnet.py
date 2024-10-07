


from typing import Tuple
import torch
from core.lit_modules.lit_model_wrappers import LitWrapperModel

from core.models.pointnet.models.pointnet2_sem_seg import get_model as get_pointnet2_model
from core.models.pointnet.models.pointnet2_sem_seg import get_loss as get_pointnet2_loss
from core.models.pointnet.models.pointnet_sem_seg import get_model as get_pointnet_model
from core.models.pointnet.models.pointnet_sem_seg import get_loss as get_pointnet_loss



class LitPointNet(LitWrapperModel):
    # https://github.com/yanx27/Pointnet_Pointnet2_pytorch


    def __init__(self, 
                 model:str,
                 criterion:torch.nn.Module,
                 optimizer_name: str, 
                 num_classes = 10,
                 num_channels=3,
                 learning_rate=0.01, 
                 ignore_index=-1,
                 metric_initializer=None, 
                 **kwargs):
        
        if model == 'pointnet':
            model = get_pointnet_model(num_classes, num_channels=num_channels)
            if criterion is None:
                criterion = get_pointnet_loss()
        elif model == 'pointnet2':
            model = get_pointnet2_model(num_classes, num_channels=num_channels)
            if criterion is None:
                criterion = get_pointnet2_loss()
        else:
            raise ValueError(f"Unknown model {model}")
        
        super().__init__(model, criterion, optimizer_name, learning_rate, None)

        self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
        self.val_metrics = metric_initializer(num_classes=num_classes)
        self.test_metrics = metric_initializer(num_classes=num_classes)

        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)
    
    def forward_model_output(self, x:torch.Tensor) -> torch.Tensor:
        # run the model and return the model output in (B, N, C) format
        return self.model(x)[0]
    
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        # print(x.shape, y.shape)
        out, trans_feat = self(x)

        # print(x.shape, trans_feat.shape)
        # print(out.shape, y.shape)

        # out (B, N, C) to (B*N, C) ; y (B, N) to (B*N)
        out = out.reshape(-1, out.shape[-1])
        y = y.reshape(-1).to(torch.long)

        loss = self.criterion(out, y, trans_feat)

        # print(f"\nLoss: {loss}\n")

        preds = self.prediction(out)

        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y)
                    if isinstance(met, torch.Tensor):
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        return loss, preds, y 
    

    def _epoch_end_metric_logging(self, metrics, prefix, print_metrics=False):
        metric_res = metrics.compute()
        if print_metrics:
            print(f'{"="*10} {prefix} metrics {"="*10}')
        for metric_name, metric_val in metric_res.items():
            if print_metrics:
                # if metric is per class
                if isinstance(metric_val, torch.Tensor) and metric_val.ndim > 0: 
                    print(f'\t{prefix}_{metric_name}: {metric_val}; mean: {metric_val[1:].mean():.4f}')
                else:
                    print(f'\t{prefix}_{metric_name}: {metric_val:.4f}')

        metrics.reset()
    

    
    

from core.models.GENEONets.SCENE_Net import SceneNet_PreBackbone
    
class LitPointNet_wSCENENet(LitPointNet):


    def __init__(self, 
                 model:str,
                 criterion:torch.nn.Module,
                 optimizer_name: str, 
                 geneo_num = None,
                 num_observers = [1],
                 kernel_size = (9, 7, 7),
                 num_classes = 10,
                 num_channels=3,
                 learning_rate=0.01, 
                 metric_initializer=None, 
                 **kwargs):
        
        self.scenenet = SceneNet_PreBackbone(geneo_num, num_observers, kernel_size)
        
        super().__init__(model, criterion, optimizer_name, num_classes, self.scenenet.feat_dim, learning_rate, metric_initializer, **kwargs)


    def forward(self, x, pt_loc):
        return self.model(self.scenenet(x, pt_loc))   
    
    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y, pt_locs = batch
        # print(x.shape, y.shape)
        out, trans_feat = self(x, pt_locs)

        # print(x.shape, trans_feat.shape)
        # print(out.shape, y.shape)

        # out (B, N, C) to (B*N, C) ; y (B, N) to (B*N)
        out = out.reshape(-1, out.shape[-1])
        y = y.reshape(-1).to(torch.long)

        loss = self.criterion(out, y, trans_feat)

        # print(f"\nLoss: {loss}\n")

        preds = self.prediction(out)

        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y)
                    if isinstance(met, torch.Tensor):
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        return loss, preds, y 
