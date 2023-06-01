
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy



from core.models.lit_modules.lit_wrapper import LitWrapperModel



def create_model(num_classes, pretrained=False):
    if pretrained:
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = torchvision.models.resnet18(pretrained=pretrained, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model



class LitResnet(LitWrapperModel):

    def __init__(self, 
                 num_classes: int, 
                 optimizer_name: str,
                 learning_rate=0.01, 
                 metric_initializer=None):

        model = create_model(num_classes)
        criterion = F.nll_loss
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer)


    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)
    
    def prediction(self, model_output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)
    

    

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