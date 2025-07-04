import os
import sys
from pathlib import Path
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_experiment_dir(model, dataset) -> Path:
    return os.path.join(get_project_root(), 'experiments', dataset, f"{model}_{dataset}")

def get_checkpoint_dir(model, dataset) -> Path:
    return os.path.join(get_experiment_dir(model, dataset), 'checkpoints')

def get_experiment_config_path(model, dataset) -> Path:
    return os.path.join(get_experiment_dir(model, dataset), 'defaults_config.yml')


ROOT_PROJECT = get_project_root()
TOSH_PATH = "/media/didi/TOSHIBA EXT/"
SSD_PATH = "/media/didi/PortableSSD/"
DOWNLOAD_PATH = "/home/didi/Downloads/data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if "didi" in str(ROOT_PROJECT): ## Local machine
    if os.path.exists(SSD_PATH):
        EXT_PATH = SSD_PATH
    elif os.path.exists(TOSH_PATH):
        EXT_PATH = TOSH_PATH
    elif os.path.exists(DOWNLOAD_PATH):
        EXT_PATH = DOWNLOAD_PATH
    else:
        EXT_PATH = "/home/didi/" ## default
else:
    EXT_PATH = "/data/d.lavado/" # cluster data dir
    
if not os.path.exists(EXT_PATH): # home cluster dir
    EXT_PATH = "/home/d.lavado/"

TS40K_PATH = os.path.join(EXT_PATH, 'TS40K-Dataset/')
TS40K_FULL_PATH = os.path.join(EXT_PATH, 'TS40K-Dataset/TS40K-FULL/')
TS40K_FULL_PREPROCESSED_PATH = os.path.join(EXT_PATH, 'TS40K-Dataset/TS40K-FULL-Preprocessed/')


NUSCENES_PATH = os.path.join(EXT_PATH, 'nuscenes/')
S3DIS_PATH = os.path.join(EXT_PATH, 's3dis/')
SEMANTIC_KITTI_PATH = os.path.join(EXT_PATH, 'semkitti/')
WAYMO_PATH = os.path.join(EXT_PATH, 'waymo/converted/')
SCANNET_PATH = os.path.join(EXT_PATH, 'scannetv2/scannet/')   