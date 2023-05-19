
import torch
import torch.nn as nn

from core.models.FC_Classifier import Classifier_OutLayer
from core.models.lit_wrapper import LitWrapperModel
from torchmetrics.functional import accuracy



class Conv_Feature_Extractor(nn.Module):

    def __init__(self, in_channels=1, hidden_dim=128, kernel_size=3):
        super(Conv_Feature_Extractor, self).__init__()

        self.block = self.conv_block(in_channels, hidden_dim, kernel_size)


    def conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.block(x)
        return x
    


class CNN_Classifier(nn.Module):
    

    def __init__(self, in_channels=1, hidden_dim=128, ghost_sample:torch.Tensor = None, kernel_size=3, num_classes=10):
        """
        Convolutional Neural Network Classifier

        Parameters
        ----------

        in_channels: int
            Number of input channels

        hidden_dim: int
            Number of hidden dimensions

        ghost_sample: torch.Tensor
            Ghost sample to be used to produce feature dimensions for the FC layer

        kernel_size: int
            Kernel size for the convolutional layer

        num_classes: int
            Number of classes for the output layer
        """
        super().__init__()
        self.feature_extractor = Conv_Feature_Extractor(in_channels, hidden_dim, kernel_size)
        ghost_shape = self.feature_extractor(ghost_sample).shape
        # print(ghost_shape)
        # input("Press Enter to continue...")
        self.classifier = Classifier_OutLayer(torch.prod(torch.tensor(ghost_shape[1:])), num_classes)


    def forward(self, x):
        x = self.feature_extractor(x)
        #print(x.shape) # batch_size, hidden_dim, new_height, new_width
        x = self.classifier(x)
        return x


class Lit_CNN_Classifier(LitWrapperModel):

    def __init__(self, 
                 in_channels=1, 
                 hidden_dim=128, 
                 kernel_size=3,
                 ghost_sample:torch.Tensor = None,
                 num_classes=10,
                 optimizer_name = 'adam', 
                 learning_rate=0.01, 
                 metric_initializer=None):

        model = CNN_Classifier(in_channels, hidden_dim, ghost_sample, kernel_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer)


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
        acc = accuracy(preds, y)
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
        

