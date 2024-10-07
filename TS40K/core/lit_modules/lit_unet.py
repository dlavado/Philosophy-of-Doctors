




from typing import Any, List, Tuple
import torch

import sys
from torchmetrics import MetricCollection

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
from core.models.unet3d.model import get_model
from core.lit_modules.lit_model_wrappers import LitWrapperModel
from core.criterions.elastic_net_reg import ElasticNetRegularization
from utils.my_utils import pointcloud_to_wandb


class LitUNet(LitWrapperModel):

    def __init__(self, 
                model_config:dict=None,
                num_classes:int=10, 
                ignore_index:int=-1,
                criterion:torch.nn.Module=None, 
                optimizer:str=None, 
                learning_rate=1e-2, 
                metric_initializer=None,
            ):
    
        if model_config is None:
            model_config = {
                'name' : 'ResidualUNetSE3D',
                'in_channels': 1,
                'out_channels': 6,
                'final_sigmoid': False,
                'f_maps': 64,
                'layer_order': 'cbrd',
                'num_groups': 8,
                'num_levels': 4,
                'is_segmentation': True,
                'conv_padding': 1,
                'conv_upscale': 2,
                'upsample': 'default',
                'dropout_prob': 0.1,
                'is_geneo': True
            }

        model = get_model(model_config)
        super().__init__(model, criterion, optimizer, learning_rate, None)
        
        if metric_initializer is not None:
            self.train_metrics:MetricCollection = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.val_metrics:MetricCollection = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.test_metrics:MetricCollection = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)

        self.save_hyperparameters()
        self.logged_batch = False
        self.gradient_check = False
        self.elastic_reg = ElasticNetRegularization(alpha=0.001, l1_ratio=0.5)

    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.model.predict(model_output)
    
    
    def forward(self, x):
        return self.model(x)
    
    def on_after_backward(self) -> None:
        # if self.model.is_geneo:
        #    for name, param in self.model.named_parameters():
        #         if 'lambdas' in name:
        #             grad = None if param.grad is None else torch.norm(param.grad.data, 2)
        #             print(f'{name} -- value: {torch.norm(param.data, 2):.5f}; grad: {grad}')
        #         elif 'geneo' in name and param.data.item() < 0:
        #             print(f'{name} -- value: {param.data.item()}; grad: {torch.norm(param.grad.data, 2)}')
        # for name, param in self.model.named_parameters():
        #     if 'geneo_layer' in name:
        #         print(f'\t{name.split(".")[-1]} -- value: {torch.norm(param.data, 2):.5f}; grad_fn: {param.grad_fn}; grad: {param.grad}')
        return super().on_after_backward()
    
    
    def on_before_backward(self, loss: torch.Tensor) -> None:
        self.log('gradient L2 norm', torch.norm(loss, 2))
        if not self.gradient_check:
            print(f'\n{"="*10} Model Values & Gradients {"="*10}')
            print(f'L1/L2 Norms of the gradient: {torch.norm(loss, 1)}, {torch.norm(loss, 2)}')
            self.gradient_check = True
        return super().on_before_backward(loss)            
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self.model(x)
        y = y.long()
        loss = self.criterion(out, y) + self.elastic_reg(self.model.get_cvx_coefficients().parameters())
        preds = self.prediction(out)

        #print(f"evaluate: {out.shape}, {y.shape}, {preds.shape}")
        #print(f"out: {out.shape}, {out.requires_grad}, {out.grad_fn}")

        # for name, param in self.model.named_parameters():
        #     if 'geneo_layer' in name:
        #         print(f'\t{name} -- value: {torch.norm(param.data, 2):.5f}; grad: {param.grad}')
          
        # print(f'x: {x.shape}, y: {y.shape}, pts_locs: {pts_locs.shape}, preds: {preds.shape}')
    
        if stage:
            on_step = True
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y)
                    if isinstance(met, torch.Tensor):
                        met = torch.mean(met[met > 0]) #if a metric is zero for a class, it is not included in the mean
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        return loss, preds, y
    

    def _log_pointcloud_wandb(self, pcd, input=None, gt=None, prefix='run'):
        point_clouds = pointcloud_to_wandb(pcd, input, gt)
        self.logger.experiment.log({f'{prefix}_point_cloud': point_clouds})  

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.model.maintain_convexity()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        # cvx_coeffs = self.model.get_cvx_coefficients()
        # print(f'\n\n{"="*10} cvx coefficients {"="*10}')
        # for name, coeff in cvx_coeffs.items():
        #     for i in range(coeff.shape[0]):
        #         if torch.any(coeff[i] < 0) or torch.any(coeff[i] > 0.5):
        #             print(f'\t{name}_obs{i}:\n\t {coeff[i]}')

        self.logged_batch = False
        self.gradient_check = False


    def get_geneo_params(self):
        return self.model.get_geneo_params()

    def get_cvx_coefficients(self):
        return self.model.get_cvx_coefficients()