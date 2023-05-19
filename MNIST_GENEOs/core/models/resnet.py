
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

from core.models.lit_wrapper import LitWrapperModel



def create_model(num_classes, pretrained=False):
    if pretrained:
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model



class LitResnet(pl.LightningModule):
    def __init__(self, num_classes, metric_initializer, optimizer_name='adam', learning_rate=0.05):
        super().__init__()

        self.save_hyperparameters('optimizer_name', 'learning_rate')
        self.model = create_model(num_classes)
        self.criterion = F.nll_loss

        if metric_initializer is not None:
            self.train_metrics = metric_initializer()
            self.val_metrics = metric_initializer()
            self.test_metrics = metric_initializer()
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

        return loss, preds, y
    
    def training_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "train")
        if self.train_metrics is not None:
            self.train_metrics(preds, y).update()
        return {"loss": loss}
    
    def training_epoch_end(self, outputs) -> None:
        if self.train_metrics is not None:
            self._epoch_end_metric_logging(self.train_metrics, 'train', print_metrics=False)

    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "val")
        if self.val_metrics is not None:
            self.val_metrics(preds, y).update()
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs) -> None: 
        if self.val_metrics is not None: # On epoch metric logging
           self._epoch_end_metric_logging(self.val_metrics, 'val', print_metrics=True)
    
    def test_step(self, batch, batch_idx):
        loss, preds, y = self.evaluate(batch, "test")
        if self.test_metrics is not None:
            self.test_metrics(preds, y).update()
        return {"test_loss": loss}

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
    

from torch.optim.swa_utils import AveragedModel, update_bn

class SWAResnet(LitResnet):
    """
    Stochastic Weight Averaging Resnet

    Takes a trained resnet model and performs stochastic weight averaging.

    """
    def __init__(self, trained_model, lr=0.01):
        super().__init__()

        self.save_hyperparameters("lr")
        self.model = trained_model
        self.swa_model = AveragedModel(self.model)

    def forward(self, x):
        out = self.swa_model(x)
        return F.log_softmax(out, dim=1)

    def training_epoch_end(self, training_step_outputs):
        self.swa_model.update_parameters(self.model)

    def validation_step(self, batch, batch_idx, stage=None):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        return optimizer

    def on_train_end(self):
        """
        Update batch norm statistics one last time before saving
        """
        update_bn(self.trainer.datamodule.train_dataloader(), self.swa_model, device=self.device)