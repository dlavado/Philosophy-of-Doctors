

from typing import List, Union, Dict, Any

from core.models.giblinet.GIBLi_SOTA import GIBLiNetPTV1, GIBLiNetPTV2, GIBLiNetPTV3, GIBLiNetKPConv, GIBLiNetPointNet, GIBLiNetPointNet2, GIBLiNetStub
from core.lit_modules.lit_model_wrappers import LitWrapperModel
from core.criterions.elastic_net_reg import ElasticNetRegularization
from core.criterions.geneo_loss import GENEORegularizer

import torch


def disable_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
        m.eval()
        m.track_running_stats = False


class Lit_GIBLiSOTA(LitWrapperModel):
    
    
    def __init__(self, 
                 in_channels,
                 num_classes,
                 model_name:str,
                 num_levels: int,
                 grid_size: Union[float, List[float]],
                 embed_channels: Union[int, List[int]],
                 out_channels: Union[int, List[int]],
                 depth: int,
                 sota_kwargs: Dict[str, Any] = {},
                 sota_update_kwargs: Dict[str, Any] = {},
                 ###### LitWrapperModel parameters ######
                 criterion=None, 
                 optimizer_name='adam', 
                 ignore_index=-1,
                 learning_rate=0.01, 
                 metric_initializer=None, 
                 **kwargs):
        
        if model_name == 'ptv1':
            model_class = GIBLiNetPTV1
        elif model_name == 'ptv2':
            model_class = GIBLiNetPTV2
        elif model_name == 'ptv3':
            model_class = GIBLiNetPTV3
        elif model_name == 'kpconv':
            model_class = GIBLiNetKPConv
        elif model_name == 'pointnet':
            model_class = GIBLiNetPointNet
        elif model_name == 'pointnet2':
            model_class = GIBLiNetPointNet2
            
        
        model:GIBLiNetStub = model_class(in_channels=in_channels,
                                num_classes=num_classes,
                                num_levels=num_levels,
                                grid_size=grid_size,
                                embed_channels=embed_channels,
                                out_channels=out_channels,
                                depth=depth,
                                sota_kwargs=sota_kwargs,
                                sota_update_kwargs=sota_update_kwargs
                            )
        
    
        super().__init__(model, criterion, optimizer_name, learning_rate, None, num_classes=num_classes, **kwargs)
        
        self.model.apply(self.init_kaiming_weights)
        
        self.geneo_reg = GENEORegularizer(0.01)
        self.elastic_reg = ElasticNetRegularization(0.01, 0.5)
        
        # self.model.apply(disable_batchnorm)
    
        if metric_initializer is not None:
            self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.val_metrics = metric_initializer(num_classes=num_classes)
            self.test_metrics = metric_initializer(num_classes=num_classes)
            
            
    def forward(self, data_dict:dict) -> torch.Tensor:
        # for k, v in data_dict.items():
        #     print(f"{k} shape = {v.shape}")
        return self.model(data_dict) # out shape = (batch_size*num_points, num_classes)
    
    def prediction(self, model_output):
        # model_output shape = (batch_size*num_points, classes)
        return torch.argmax(model_output, dim=-1)
    
    
    def elastic_loss(self):
        return self.elastic_reg(self.model.get_cvx_coefficients())
    
    def geneo_loss(self):
        return self.geneo_reg.positive_regularizer(self.model.get_gib_params())
    
    def evaluate(self, data_dict, stage=None, metric=None, prog_bar=True, logger=True):

        out = self(data_dict) # out shape = (batch_size*num_points, num_classes)
        y = data_dict['segment'].to(torch.long)

        # print(f"out shape = {out.shape}, y shape = {y.shape}")
    
        data_fid = self.criterion(out, y)
        elastic_loss = self.elastic_loss()
        geneo_loss = self.geneo_loss()
        loss = data_fid + elastic_loss + geneo_loss
        preds = self.prediction(out) # preds shape = (batch_size*num_points)
        
        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_data_fid", data_fid, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_elastic_loss", elastic_loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_geneo_loss", geneo_loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    metric_val = metric_val
                    met = metric_val(preds, y.reshape(-1))
                    if met.numel() > 1: 
                        if stage == 'val':   
                            for i, m in enumerate(met.tolist()):
                                self.log(f"class_{i}_{metric_name}", m, on_epoch=True, on_step=False, prog_bar=False, logger=logger)
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=logger)

        return loss, preds, y
    
    
    # def on_after_backward(self):
    #     for name, param in self.model.named_parameters():
    #         # if 'gib_params' in name and 'disk' in name:
    #         if param.grad is not None:
    #             print(f"{name},  grad.norm={param.grad.norm().item():.2e}")
    #         else:
    #             print(f"{name} has .grad to None")
        
    #     # print(f"Memory after backward: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
    #     # print(torch.cuda.memory_summary())
    #     return

    