

import os
import sys
from pathlib import Path
import torch


def get_project_root() -> Path:
    #return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return Path().resolve().parent.absolute()

ROOT_PROJECT = Path().resolve().parent.absolute()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if "didi" in str(ROOT_PROJECT):
    EXT_PATH = "/media/didi/TOSHIBA EXT/"
else:
    EXT_PATH = "/home/d.lavado/" #cluster data dir


TS40K_PATH = os.path.join(EXT_PATH, 'TS40K-NEW/')

EXPERIMENTS_PATH = os.path.join(get_project_root(), 'experiments')

WEIGHT_SCHEME_PATH = os.path.join(ROOT_PROJECT, 'core/criterions/hist_estimation.pickle')
#HIST_PATH = os.path.join(SCNET_PIPELINE, "torch_geneo/hist_estimation.pickle")

def get_experiment_path(model, dataset) -> Path:
    return os.path.join(EXPERIMENTS_PATH, f"{model}_{dataset}")

def get_experiment_config_path(model, dataset) -> Path:
    return os.path.join(get_experiment_path(model, dataset), 'config.yml')


if __name__ == '__main__':
    print(ROOT_PROJECT)
    print(TS40K_PATH)
    print(WEIGHT_SCHEME_PATH)
    print(EXPERIMENTS_PATH)
    print(get_experiment_path('scnet', 'ts40k'))
    print(get_experiment_config_path('scnet', 'ts40k'))
    print(device)
