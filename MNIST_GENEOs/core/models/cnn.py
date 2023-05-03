
import torch
import torch.nn as nn

import pytorch_lightning as pl
from FC_Classifier import Classifier_OutLayer
from core.models.lit_wrapper import LitWrapperModel




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
    

    def __init__(self, in_channels=1, hidden_dim=128, kernel_size=3, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = Conv_Feature_Extractor(in_channels, hidden_dim, kernel_size)
        self.classifier = Classifier_OutLayer(hidden_dim, learning_rate)


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class Lit_CNN_Classifier(LitWrapperModel):

    def __init__(self, 
                 in_channels=1, 
                 hidden_dim=128, 
                 kernel_size=3,
                 optimizer_name = 'adam', 
                 learning_rate=0.01, 
                 metric_initializer=None):

        model = CNN_Classifier(in_channels, hidden_dim, kernel_size)
        criterion = nn.CrossEntropyLoss()
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer)
    

