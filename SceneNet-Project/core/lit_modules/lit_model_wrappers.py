from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from core.models.SCENE_Net import SceneNet
from utils.scripts_utils import ParseKwargs




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

    def __init__(self, model:torch.nn.Module, criterion:torch.nn.Module, optimizer_name:str, learning_rate=1e-2, metric_initilizer=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        if metric_initilizer is not None:
            self.train_metrics = metric_initilizer()
            self.val_metrics = metric_initilizer()
            self.test_metrics = metric_initilizer()
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
    
        self.save_hyperparameters('optimizer_name', 'learning_rate')

    def forward(self, x):
        return self.model(x)                     

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.train_metrics is not None: # on step metric logging
            metric_res = self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int))
            for metric_name, metric_val in metric_res.items():
                self.log(f'train_{metric_name}', metric_val, on_epoch=True, prog_bar=True, logger=True)  

        self.log(f'train_loss', loss)
        return loss
    
    def training_epoch_end(self, outputs) -> None:
        if self.train_metrics is not None:
            self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.val_metrics is not None:
            self.val_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_epoch_end(self, outputs) -> None: 
        if self.val_metrics is not None: # On epoch metric logging
           self._epoch_end_metric_logging(self.val_metrics, 'val')

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.test_metrics is not None:
            self.test_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'test_loss', loss, logger=True)
        return loss
    
    def test_epoch_end(self, outputs) -> None:
        if self.test_metrics is not None: # On epoch metric logging
            self._epoch_end_metric_logging(self.test_metrics, 'test')

    def get_model(self):
        return self.model
    
    def _epoch_end_metric_logging(self, metrics, prefix):
        metric_res = metrics.compute()
        for metric_name, metric_val in metric_res.items():
            self.log(f'{prefix}_{metric_name}', metric_val, prog_bar=True, logger=True) 
        metrics.reset()

    def configure_optimizers(self):
        return self._resolve_optimizer()

    def _resolve_optimizer(self):
        opt_name = self.hparams.optimizer_name
        opt_name = opt_name.lower()
        if  opt_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif opt_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        elif opt_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.hparams.learning_rate)
        else:
            raise NotImplementedError(f'Optimizer {self.hparams.optimizer_name} not implemented')



class LitSceneNet(LitWrapperModel):

    def __init__(self, geneo_num:dict, kernel_size:Tuple[int], criterion:torch.nn.Module, optimizer:str, learning_rate=1e-2, metric_initilizer=None):
    
        model = SceneNet(geneo_num, kernel_size)
        self.save_hyperparameters('geneo_num', 'kernel_size')
        super().__init__(model, criterion, optimizer, learning_rate, metric_initilizer)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('LitSceneNet')
        # careful, the values from this dict are parsed as strings
        # use example: --geneo_num cy=1 cone=1 neg=1
        parser.add_argument('--geneo_num', type=dict, action=ParseKwargs, default= {'cy'  : 1, 
                                                                                    'cone': 1, 
                                                                                    'neg' : 1})
        parser.add_argument('--kernel_size', type=tuple, default=(9, 7, 7))
        return parent_parser
    
    def training_step(self, batch, batch_idx):
        # As this is a GENEO model, we need to pass the convex coefficients and the GENEO parameters to the loss function
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_cvx_coefficients(), self.model.get_geneo_params())

        if self.train_metrics is not None: # on step metric logging
            metric_res = self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int))
            for metric_name, metric_val in metric_res.items():
                self.log(f'train_{metric_name}', metric_val, on_epoch=True, prog_bar=True, logger=True)  

        self.log(f'train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_cvx_coefficients(), self.model.get_geneo_params())

        if self.val_metrics is not None:
            self.val_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_cvx_coefficients(), self.model.get_geneo_params())

        if self.test_metrics is not None:
            self.test_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'test_loss', loss, logger=True)
        return loss
        

