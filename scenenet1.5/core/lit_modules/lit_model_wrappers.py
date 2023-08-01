from typing import Any, List, Tuple
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl


import sys
from torch.optim.optimizer import Optimizer

from torchmetrics import MetricCollection
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
import utils.voxelization as Vox
from core.models.SCENE_Net import SceneNet, SceneNet_multiclass
from utils.scripts_utils import ParseKwargs, pointcloud_to_wandb



class LitWrapperModel(pl.LightningModule):
    """
    Generic Pytorch Lightning wrapper for Pytorch models that defines the logic for training, validation,testing and prediciton. 
    It also defines the logic for logging metrics and losses.    
    
    Parameters
    ----------

    `model` - torch.nn.Module:
        The model to be wrapped.
    
    `criterion` - torch.nn.Module:
        The loss function to be used

    `optimizer` - str:
        The Pytorch optimizer to be used for training.
        Note: str must be \in {'Adam', 'SGD', 'RMSprop'}

    `metric_initilizer` - function:
        A function that returns a TorchMetric object. The metric object must have a reset() and update() method.
        The reset() method is called at the end of each epoch and the update() method is called at the end of each step.
    """

    def __init__(self, model:torch.nn.Module, criterion:torch.nn.Module, optimizer_name:str, learning_rate=1e-2, metric_initializer=None, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = criterion


        if metric_initializer is not None:
            self.train_metrics:MetricCollection = metric_initializer(num_classes=kwargs['num_classes'])
            self.val_metrics:MetricCollection = metric_initializer(num_classes=kwargs['num_classes'])
            self.test_metrics:MetricCollection = metric_initializer(num_classes=kwargs['num_classes'])
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
    
        self.save_hyperparameters('optimizer_name', 'learning_rate')

    def forward(self, x):
        return self.model(x)
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        return model_output

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
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

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "train", self.train_metrics)
        state = {"loss": loss, "preds": preds}
        # if self.train_metrics is not None:
        #     mets = self.train_metrics(preds, y)
        #     state.update(mets)
        return state                  
    
    def on_train_epoch_end(self) -> None:
        if self.train_metrics is not None:
            self._epoch_end_metric_logging(self.train_metrics, 'train', print_metrics=False)

    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "val", self.val_metrics)
        state = {"val_loss": loss, "preds": preds}
        # if self.val_metrics is not None:
        #     mets = self.val_metrics(preds, y)
        #     state.update(mets)

        return state
    
    def on_validation_epoch_end(self) -> None: 
        if self.val_metrics is not None: # On epoch metric logging
           self._epoch_end_metric_logging(self.val_metrics, 'val', print_metrics=True)
    
    def test_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "test", self.test_metrics)
        state = {"test_loss": loss}
        # if self.test_metrics is not None:
        #     mets = self.test_metrics(preds, y)
        #     state.update(mets)
        return state

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, _ = batch
        pred = self(x)
        pred = self.prediction(pred)

        return pred
    
    def on_test_epoch_end(self) -> None:
        if self.test_metrics is not None: # On epoch metric logging
            self._epoch_end_metric_logging(self.test_metrics, 'test')

    def get_model(self):
        return self.model
    
    def set_criteria(self, criterion):
        self.criterion = criterion
    
    def _epoch_end_metric_logging(self, metrics, prefix, print_metrics=False):
        metric_res = metrics.compute()
        if print_metrics:
            print(f'{"="*10} {prefix} metrics {"="*10}')
        for metric_name, metric_val in metric_res.items():
            if print_metrics:
                print(f'\t{prefix}_{metric_name}: {metric_val}')
        metrics.reset()

    def configure_optimizers(self):
        return self._resolve_optimizer(self.hparams.optimizer_name)
    
    def _check_model_gradients(self):
        print(f'\n{"="*10} Model Values & Gradients {"="*10}')
        for name, param in self.model.named_parameters():
            print(f'\t{name} -- value: {param.data.item():.5f} grad: {param.grad}')

    def _resolve_optimizer(self, optimizer_name:str):
        optimizer_name = optimizer_name.lower()
        if  optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'lbfgs':
            return torch.optim.LBFGS(self.model.parameters(), lr=self.hparams.learning_rate, max_iter=20)
        
        raise NotImplementedError(f'Optimizer {self.hparams.optimizer_name} not implemented')
    


class LitSceneNet_multiclass(LitWrapperModel):

    def __init__(self, 
                geneo_num:dict,
                num_observers:int, 
                kernel_size:Tuple[int],
                hidden_dims:Tuple[int],
                num_classes:int, 
                criterion:torch.nn.Module, 
                optimizer:str, 
                learning_rate=1e-2, 
                metric_initializer=None):
    
        model = SceneNet_multiclass(geneo_num, num_observers, kernel_size, hidden_dims, num_classes)
        self.save_hyperparameters('geneo_num', 'kernel_size')
        self.logged_batch = False
        self.gradient_check = False
        super().__init__(model, criterion, optimizer, learning_rate, metric_initializer, num_classes=num_classes)

    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.model.prediction(model_output)
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y, pts_locs = batch
        out = self.model(x, pts_locs)
        loss = self.criterion(out, y)
        preds = self.prediction(out)
        
        if stage:
            on_step = stage == "train" 
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
        cvx_coeffs = self.model.get_cvx_coefficients()
        print(f'\n\n{"="*10} cvx coefficients {"="*10}')
        for name, coeff in cvx_coeffs.items():
            for i in range(coeff.shape[0]):
                if torch.any(coeff[i] < 0) or torch.any(coeff[i] > 0.5):
                    print(f'\t{name}_obs{i}:\n\t {coeff[i]}')

        self.logged_batch = False

    # def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
    #     print(f'\n{"="*10} Model Values & Gradients {"="*10}')
    #     cvx_coeffs = self.model.get_cvx_coefficients()
    #     geneo_params = self.model.get_geneo_params()
    #     for name, cvx in cvx_coeffs.items():
    #         print(f'\t{name} -- cvx: {cvx}  --grad: {cvx.grad}')
    #     # for name, param in geneo_params.items():
    #     #     print(f'\t{name} -- value: {param} --grad: {param.grad}')
    #     return super().on_before_optimizer_step(optimizer)
    
    # def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
    #     # logging batch point clouds to wandb
    #     if self.trainer.current_epoch % 10 == 0 and not self.logged_batch: 
    #         x, y, pt_locs = batch 
    #         preds = outputs["preds"]

    #         pt_locs = pt_locs[0].detach().cpu().numpy()

    #         preds = torch.squeeze(preds[0]).detach().cpu().numpy() # 1st batch sample
    #         preds = preds / np.max(preds) # to plot the different classes in different colors
    #         preds = np.column_stack((pt_locs, preds))

    #         x = torch.squeeze(x[0]).detach().cpu().numpy()
    #         x = Vox.plot_voxelgrid(x, color_mode='ranges', plot=False)
            
    #         y = torch.squeeze(y[0]).detach().cpu().numpy() # 1st batch sample
    #         y = y / np.max(y) # to plot the different classes in different colors
    #         y = np.column_stack((pt_locs, y))

    #         self._log_pointcloud_wandb(preds, x, y, prefix=f'val_{self.trainer.global_step}')
    #         self.logged_batch = True


    def get_geneo_params(self):
        return self.model.get_geneo_params()

    def get_cvx_coefficients(self):
        return self.model.get_cvx_coefficients()
