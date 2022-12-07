import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithLogCalibration(nn.Module):
    """
    Wrapper class on uncalibrated regressor to perform probability calibration
    through Logistic Regression
    """

    def __init__(self) -> None:
        super().__init__()
