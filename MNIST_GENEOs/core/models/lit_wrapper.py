from typing import Any
import torch
import pytorch_lightning as pl
import  pytorch_lightning.callbacks as  pl_callbacks



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

    def __init__(self, model:torch.nn.Module, criterion:torch.nn.Module, optimizer_name:str, learning_rate=1e-2, metric_initializer=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        if metric_initializer is not None:
            self.train_metrics = metric_initializer()
            self.val_metrics = metric_initializer()
            self.test_metrics = metric_initializer()
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
            self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update()

        self.log(f'train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return pred, loss
    
    def training_epoch_end(self, outputs) -> None:
        if self.train_metrics is not None:
            self._epoch_end_metric_logging(self.train_metrics, 'train', print_metrics=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.val_metrics is not None:
            self.val_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'val_loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return pred, loss
    
   
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
        return pred, loss
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, _ = batch
        pred = self(x)

        return pred
    
    def test_epoch_end(self, outputs) -> None:
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
            self.log(f'{prefix}_{metric_name}', metric_val, on_epoch=True, on_step=False, logger=True) 
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
    


########################################
# Callbacks
########################################



def callback_model_checkpoint(dirpath, filename, monitor, mode, save_top_k=1, save_last=True, verbose=True, \
                                                                 every_n_epochs=1, every_n_train_steps=0, **kwargs):
    """
    Callback for model checkpointing. 

    Parameters
    ----------
    `dirpath` - str: 
        The directory where the checkpoints will be saved.
    `filename` - str: 
        The filename of the checkpoint.
    `mode` - str: 
        The mode of the monitored metric. Can be either 'min' or 'max'.
    `monitor` - str: 
        The metric to be monitored.
    `save_top_k` - int: 
        The number of top checkpoints to save.
    `save_last` - bool: 
        If True, the last checkpoint will be saved.
    `verbose` - bool: 
        If True, the callback will print information about the checkpoints.
    `every_n_epochs` - int: 
        The period of the checkpoints.
    """
    return pl_callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
        every_n_epochs=every_n_epochs,
        every_n_train_steps=every_n_train_steps,
        verbose=verbose,
        **kwargs
    )