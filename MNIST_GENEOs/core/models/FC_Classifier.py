import torch
from torch.nn import functional as F
from torch import nn


class Classifier_OutLayer(nn.Module):
    
    def __init__(self, in_channels: int = 128):
        super().__init__()
        #self.l1 = torch.nn.Linear(28*28, self.hparams.in_channels)
        self.l2 = torch.nn.Linear(self.hparams.in_channels, 10)
 
    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x
 
    def _compute_loss(self, logits, y):
        return F.cross_entropy(logits, y)
 
    def _accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc
