

from typing import List
import torch
from torch.nn import functional as F
from torch import nn
from torchmetrics.functional import accuracy


class Classifier_OutLayer(nn.Module):
    
    def __init__(self, hidden_dims:List[int], num_classes:int):
        """

        """
        super().__init__()
        self.hidden_layers = [torch.nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1]) for i in range(len(hidden_dims)-1)]
        self.bns = [torch.nn.BatchNorm1d(num_features=hidden_dims[i]) for i in range(len(hidden_dims)-1)]
        self.out_layer = torch.nn.Linear(in_features=hidden_dims[-1], out_features=num_classes)


    def forward(self, x):
        
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = self.bns[i](x)
            x = F.relu(x)

        x = self.out_layer(x)
        return x

    def _compute_loss(self, logits, y):
        return F.cross_entropy(logits, y)

    def _accuracy(self, logits, y):
        return accuracy(logits, y)