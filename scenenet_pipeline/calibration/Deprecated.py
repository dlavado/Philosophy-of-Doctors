

import math
import pprint as pp
import shutil
from typing import Any, List, Mapping, Tuple, Union
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose
from temperature_scaling import ModelWithTemperature

import os
from pathlib import Path
from datetime import datetime
import sys
ROOT_PROJECT = Path(os.path.abspath(__file__)).parents[2].resolve()


sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from torch_geneo.models.SCENE_Net import SCENE_Net
from scenenet_pipeline.torch_geneo.models.geneo_loss import GENEO_Loss
from torch_geneo.datasets.ts40kv2 import torch_TS40Kv2
from torch_geneo.datasets.torch_transforms import Voxelization, ToTensor, ToFullDense
from torch_geneo.observer_utils import *
import EDA.EDA_utils as eda

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXT_PATH = "/media/didi/TOSHIBA EXT/"
TS40K_PATH = os.path.join(EXT_PATH, 'TS40K/')


SCNET_PIPELINE = os.path.join(ROOT_PROJECT, 'scenenet_pipeline')

SAVED_SCNETS_PATH = os.path.join(SCNET_PIPELINE, 'torch_geneo/saved_scnets')
HIST_PATH = os.path.join(SCNET_PIPELINE, "torch_geneo/models")
FREQ_SAMPLES = os.path.join(SCNET_PIPELINE, "dataset/freq_samples")



if __name__ == "__main__":

    print(f"root: {ROOT_PROJECT}")


    # --- Model Definition ---
    gnet_class = SCENE_Net
    opt_class = torch.optim.RMSprop
    gnet_loss = GENEO_Loss

    MODEL_PATH = os.path.join(SAVED_SCNETS_PATH, 'models_geneo')
    model_dir_idx = -1
    MODEL_PATH = os.path.join(MODEL_PATH, sorted(os.listdir(MODEL_PATH), key=lambda date: datetime.fromisoformat(date))[model_dir_idx])

    MODEL_PATH = os.path.join(MODEL_PATH, 'gnet.pt')
    assert os.path.exists(MODEL_PATH)

    print(f"Model Path: {MODEL_PATH}")

    gnet, _ = load_state_dict(MODEL_PATH, gnet_class, model_tag='FBetaScore')

    

    # --- Dataset Initialization ---
    vxg_size = (64, 64, 64)
    vox_size = (0.5, 0.5, 0.5) # only use vox_size after training or with batch_size = 1
    composed = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size, vox_size=None),
                        ToTensor(), 
                        ToFullDense(apply=(True, False))])

    #ts40k_train = torch_TS40Kv2(dataset_path=TS40K_PATH, split=parser.ts40k_split, transform=composed)
    ts40k_val = torch_TS40Kv2(dataset_path=TS40K_PATH, split='test', transform=composed)
    ts40k_test = torch_TS40Kv2(dataset_path=TS40K_PATH, split='train', transform=composed)

    ts40k_val_loader = DataLoader(ts40k_val, batch_size=8, shuffle=True, num_workers=4)


    # --- Temperature Scaling ---

    scaled_model = ModelWithTemperature(gnet)
    scaled_model.set_temperature(ts40k_val_loader)


