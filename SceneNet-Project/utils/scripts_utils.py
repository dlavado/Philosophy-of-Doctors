
import argparse
import numpy as np
import torch
import random
import pytorch_lightning as pl
import sys

from torchmetrics import MetricCollection, JaccardIndex, Precision, Recall, F1Score, FBetaScore, AUROC, AveragePrecision


sys.path.insert(0, '..')

from core.criterions.dice_loss import BinaryDiceLoss, BinaryDiceLoss_BCE
from core.criterions.tversky_loss import FocalTverskyLoss, TverskyLoss
from core.criterions.w_mse import WeightedMSE
from core.criterions.geneo_loss import GENEO_Loss, GENEO_Dice_BCE, GENEO_Dice_Loss, GENEO_Tversky_Loss




class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value



def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    pl.seed_everything(hash("setting random seeds") % 2**32 - 1)


def _resolve_geneo_criterions(criterion_name):
    criterion_name = criterion_name.lower()
    if criterion_name == 'geneo':
        return GENEO_Loss
    elif criterion_name == 'geneo_dice_bce':
        return GENEO_Dice_BCE
    elif criterion_name == 'geneo_dice':
        return GENEO_Dice_Loss
    elif criterion_name == 'geneo_tversky':
        return GENEO_Tversky_Loss
    else:
        raise NotImplementedError(f'GENEO Criterion {criterion_name} not implemented')


def resolve_criterion(criterion_name):
    criterion_name = criterion_name.lower()
    if criterion_name == 'mse':
        return WeightedMSE
    elif criterion_name == 'dice':
        return BinaryDiceLoss
    elif criterion_name == 'dice_bce':
        return BinaryDiceLoss_BCE
    elif criterion_name == 'tversky':
        return TverskyLoss
    elif criterion_name == 'focal_tversky':
        return FocalTverskyLoss
    elif 'geneo' in criterion_name: 
        return _resolve_geneo_criterions(criterion_name)
    else:
        raise NotImplementedError(f'Criterion {criterion_name} not implemented')
    

def init_metrics(tau=0.65):
    return MetricCollection([
        JaccardIndex(num_classes=2, threshold=tau),
        Precision(threshold=tau),
        Recall(threshold=tau),
        F1Score(threshold=tau),
        FBetaScore(beta=0.5, threshold=tau),
        #AveragePrecision(),
        #Accuracy(threshold=tau),
        #AUROC() # Takes too much GPU memory
        #BinnedAveragePrecision(num_classes=1, thresholds=torch.linspace(0.5, 0.95, 20))
    ])
