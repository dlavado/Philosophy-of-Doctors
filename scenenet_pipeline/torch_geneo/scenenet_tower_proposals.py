
import argparse
from datetime import datetime
import os
from random import sample
import sys
from pathlib import Path

import math
import pprint as pp
from typing import Any, List, Mapping, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchviz import make_dot
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from PIL import Image

from datasets.ts40kv2 import xyz_ToFullDense, torch_TS40Kv2, ToTensor, Voxelization
from models.SCENE_Net import SCENE_Net
from models.geneo_loss import GENEO_Loss
from observer_utils import *

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from EDA import EDA_utils as eda
from VoxGENEO import Voxelization as Vox
import argparse
import labelec_parser as lbl


# --------------- Project Constant Initialization -------------------

ROOT_PROJECT = str(Path(os.getcwd()).parent.absolute())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SAVE_DIR = "/media/didi/TOSHIBA EXT/coord128_dataset/"

# DATA_DIR = os.path.join(ROOT_PROJECT, 'dataset/torch_dataset')
# DATA_DIR = "/media/didi/TOSHIBA EXT/TS40K/"

MODEL_DIR = os.path.join(ROOT_PROJECT, 'models_geneo')
PICKLE_PATH = os.path.join(ROOT_PROJECT, "torch_geneo/models")

CSV_DIR = os.path.join(ROOT_PROJECT, "labelec_csvs")
LAS_DIR = "/media/didi/TOSHIBA EXT/Labelec_LAS/"


TAU = 0.7
ts40k_splits = ['samples', 'train', 'val', 'test']


# ------------------- Ancillary Functions ----------------------



def arg_parser():
    parser = argparse.ArgumentParser(description="Process observer arguments")

    # Visualization hotkey
    parser.add_argument("--vis", action="store_true")

    # Model Loading
    parser.add_argument("--load_model", type=str, default=-1)  # 'latest' or path or index
    parser.add_argument("--model_tag", type=str, default='FBetaScore') # tag of the model to load

    # Data Config
    parser.add_argument("--ts40k_split", type=str, choices=ts40k_splits, default='samples')

    # Voxelization Setting
    parser.add_argument('--vox_size', type=float, action='append', nargs="+", default=None)           # size of voxels; overrides --vxg_size
    parser.add_argument('--vxg_size', type=int, action='append', nargs="+", default=None)     # size of the voxel-grid

    # SCENE-Net Hyperparameters
    parser.add_argument('-k', '--k_size', type=int, action='append', nargs="+", default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('-a', '--alpha', type=float, default=5)
    parser.add_argument('-e', '--epsilon', type=float, default=1e-1)


    # DBSCAN hyperparameters:
    parser.add_argument('--min_dist', type=float, default=3) # min distance between points to form a tower cluster
    parser.add_argument('--min_points', type=int, default=6) # min num of points to form a tower cluster


    return parser.parse_args()



def run_observer(model_path, tau=TAU, tag='latest'):

    # --- Load Best Model ---
    gnet, chkp = load_state_dict(model_path, gnet_class, tag, kernel_size=kernel_size)

    print("\n\nLoading Data...")
    ts40k_loader = DataLoader(ts40k, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    geneo_loss = gnet_loss(torch.tensor([]), hist_path=PICKLE_PATH, alpha=ALPHA, rho=RHO, epsilon=EPSILON)

    test_metrics = init_metrics(tau) 
    test_loss = 0
 
    # --- Test Loop ---
    print(f"\n\nBegin testing with tau={tau:.3f} ...")
    for batch in tqdm(ts40k_loader, desc=f"testing..."):
        loss, _ = process_batch(gnet, batch, geneo_loss, None, test_metrics, requires_grad=False)
        test_loss += loss

        # if parser.vis and vis:
        #     visualize_batch(batch[0], gt, pred, tau=tau)
        #     input("\n\nPress Enter to continue...")

    test_loss = test_loss /  len(ts40k_loader)
    test_res = test_metrics.compute()
    print(f"\ttest_loss = {test_loss:.3f};")
    for met in test_res:
        print(f"\t{met} = {test_res[met]:.3f};")

    return test_res, test_loss


def examine_samples(model_path, tau=None, tag='loss'):

    gnet, _ = load_state_dict(model_path, gnet_class, tag, kernel_size=kernel_size)


    print(f"\n\nExamining samples with model {model_path.split('/')[-2]}")
    print(f"\t with tau={tau} trying to optimize {tag}...")

    ts40k_loader = DataLoader(ts40k, batch_size=1, shuffle=True, num_workers=4)
    geneo_loss = gnet_loss(torch.tensor([]), hist_path=PICKLE_PATH, alpha=ALPHA, rho=RHO, epsilon=EPSILON)
    
    test_metrics = init_metrics(tau) 
    test_loss = 0
    sample_distances = None
 
    # --- Test Loop ---
    for i in range(3, len(ts40k)):
        print(f"Examining TS40K Sample {i}...")

        xyz, dens, labels = ts40k[i]
        batch = dens[None], labels[None] # for batch dimension
    
        loss, pred = process_batch(gnet, batch, geneo_loss, None, test_metrics, requires_grad=False)
        test_loss += loss

        # print(dens.shape)
        # print(labels.shape)
        # print(pred.shape)


        test_res = test_metrics.compute()
        pre = test_res['Precision']
        rec = test_res['Recall']
    
        np.set_printoptions(precision=4)
        if parser.vis:
            print(f"Precision = {pre}")
            print(f"Recall = {rec}")
            vox, gt = transform_batch(batch)
            visualize_batch(vox, gt, pred, tau=tau)

        if device != 'cpu':        
            pred = pred.cpu()
        min_dist = parser.min_dist
        min_points = parser.min_points

        distances = compute_euc_dists(torch.cat((xyz[0], Vox.prob_to_label(pred[0], tau))), #decapsulate prediction
                                        torch.cat((xyz[0], labels)),
                                        min_dist, min_points, visual=parser.vis)
        if parser.vis:
            
            print("\n\nXY Coordinate Predictions:\nGround Truth\t\t\tPrediction\t\t\tEuclidean Dist")
            for row in distances:
                print(f"{row[0]};\t{row[1]};\t{row[2]:.4f};")

            input("\n\nPress Enter to continue...")
        else:
            if sample_distances is None:
                sample_distances = np.array([row[-1] for row in distances])
                print(sample_distances)
            else:
                sample_distances = np.concatenate((sample_distances, np.array([row[-1] for row in distances])))

        test_metrics.reset()
        print(f"Mean Euclidean Distance:{np.mean(sample_distances)}")

    print(f"\n\nMean Euclidean Distance:{np.mean(sample_distances)}")

    test_loss = test_loss /  len(ts40k_loader)
    test_res = test_metrics.compute()
    print(f"\ttest_loss = {test_loss:.3f};")
    for met in test_res:
        print(f"\t{met} = {test_res[met]:.3f};")


def process_samples(model_path, tau=None, tag='loss'):
    """
    Same general behavior as examine_samples, but with no performance measure.

    We assume that can only be one supporting tower per sample following Labelec's instructions
    """

    gnet, _ = load_state_dict(model_path, gnet_class, tag, kernel_size=kernel_size)

    if parser.vis:
        print(f"\n\nExamining samples with model {model_path.split('/')[-2]}")
        print(f"\t with tau={tau} trying to optimize {tag}...")

    # ts40k_loader = DataLoader(ts40k, batch_size=1, shuffle=True, num_workers=4)
    geneo_loss = gnet_loss(torch.tensor([]), hist_path=PICKLE_PATH, alpha=ALPHA, rho=RHO, epsilon=EPSILON)
    
    centroids = None
    centroids_file = os.path.join(CSV_DIR, 'centroids.csv')
 
    # --- Test Loop ---
    for i in tqdm(range(0, len(ts40k)), desc='Processing data...'):

        if parser.vis:
            print(f"Examining TS40K Sample {i}...")

        xyz, dens, labels = ts40k[i]
        batch = dens[None], labels[None] # include batch dimension
    
        _, pred = process_batch(gnet, batch, geneo_loss, None, None, requires_grad=False)

        # print(dens.shape)
        # print(labels.shape)
        # print(pred.shape)
      
        np.set_printoptions(precision=4)

        # if parser.vis:
        #     vox, _ = transform_batch(batch)
        #     visualize_batch(vox, None, pred, tau=tau)
        if device == 'cpu':
            pred = pred.cpu()
        min_dist = parser.min_dist
        min_points = parser.min_points

        tower_centroids = get_tower_proposals(xyz[0], Vox.prob_to_label(dens, 0.01),  # [0] to decapsulate batch dim
                                              Vox.prob_to_label(pred[0], tau),        # [0] to decapsulate batch dim
                                              min_dist=min_dist, min_points=min_points, 
                                              visual=False)


        if len(tower_centroids) == 0:
            continue
       
        # if there is more than one tower proposal, we run the sample in high resolution and keep the intersection of centroid predictions
        elif len(tower_centroids) > 1:
            if parser.vis:
                print(f"\nMultiple centroids proposals; Switching to High Resolution mode...")

            prior_centroids = tower_centroids

            xyz, dens, labels = ts40k_highres[i]
            batch = dens[None], labels[None] # include batch dimension

            _, pred = process_batch(gnet, batch, geneo_loss, None, None, requires_grad=False)
            
            if device == 'cpu':
                pred = pred.cpu()

            tower_centroids = get_tower_proposals(xyz[0], Vox.prob_to_label(dens, 0.01),  # [0] to decapsulate batch dim
                                                  Vox.prob_to_label(pred[0], tau),        # [0] to decapsulate batch dim
                                                  min_dist=min_dist, min_points=10, 
                                                  visual=False)

            
            if len(tower_centroids) > 0:
                mask = np.zeros(len(tower_centroids), dtype=np.bool)
                for i, c in enumerate(tower_centroids):
                    c_full = np.full_like(prior_centroids, fill_value=c)
                    mask[i] = np.any(eda.euclidean_distance(c_full, prior_centroids, axis=1) <= min_dist/2)
                    print(eda.euclidean_distance(c_full, prior_centroids, axis=1))

                tower_centroids = tower_centroids[mask]

            
        if len(tower_centroids) > 0:
            if centroids is None: # this prevents dimensions mismatches
                centroids = tower_centroids
            else:
                centroids = np.concatenate((centroids, tower_centroids), axis=0)

            # Saves centroids in a CSV file
            np.savetxt(centroids_file, centroids, delimiter=',', fmt='%f')


        if parser.vis:
            plot_centroids(xyz[0], dens, tower_centroids) 

            print(tower_centroids)
            input("Next?")

    # centroids = np.loadtxt(centroids_file)
    # centroids = np.unique(centroids, axis=0)
    # np.savetxt(centroids_file, centroids, delimiter=',', fmt='%f')            


    print(f"\n\n Processing Complete! \n\n")


if __name__ == "__main__":

    parser = arg_parser()

    # ----------- Model Path Resolution -----------

    try:
        idx = int(parser.load_model)
    except ValueError:
        # not an int
        idx = -1  if parser.load_model == 'latest' else None

    if idx is not None:
        model_dir = sorted(os.listdir(MODEL_DIR), key=lambda date: datetime.fromisoformat(date))[idx]
    else:
        model_dir = parser.load_model

    META_MODEL_PATH = os.path.join(MODEL_DIR, model_dir)
    
    MODEL_PATH = os.path.join(META_MODEL_PATH, "gnet.pt")

    assert os.path.exists(MODEL_PATH), f"Directory {model_dir} does not contain a model checkpoint"

    # ----------- Dataset Initialization -----------
    
    data_path = os.path.join(LAS_DIR, 'Processed_PCD')


    parser.vox_size = parser.vox_size[0] if parser.vox_size is not None else None

    parser.vxg_size = parser.vxg_size[0] if parser.vxg_size is not None else (64, 64, 64) # default
    
    composed = Compose([Voxelization(vox_size=parser.vox_size, vxg_size=parser.vxg_size), 
                        ToTensor(), 
                        xyz_ToFullDense()])

    ts40k = torch_TS40Kv2(dataset_path=data_path, split=parser.ts40k_split, transform=composed)


    composed_highres = Compose([Voxelization(vox_size=(0.5, 0.5, 0.5)), 
                        ToTensor(), 
                        xyz_ToFullDense()])

    ts40k_highres = torch_TS40Kv2(dataset_path=data_path, split=parser.ts40k_split, transform=composed_highres)

    # --- Model Definition ---
    gnet_class = SCENE_Net
    gnet_loss = GENEO_Loss
    state_dict = torch.load(MODEL_PATH)

    # --- SCENE-Net Hyperparameters ---
    kernel_size = state_dict['model_props']['kernel_size'] if parser.k_size is None else tuple(*parser.k_size)
    
    BATCH_SIZE = parser.batch_size
    ALPHA = parser.alpha     # importance factor of the weighting scheme applied during dense loss
    RHO = 5                  # scaling factor of cvx_loss
    EPSILON = parser.epsilon # min value of dense_loss


    pp.pprint(state_dict['model_props'])


    ###############################################################
    #                   Examining Samples                         #
    ###############################################################

    with torch.no_grad():
        examine_tag = input("Insert metric/value for threshold: ")
        try:
            # its a value
            tau = float(examine_tag)
        except ValueError:
            # its a metric
            if examine_tag == '':
                tau = 0.7 # default
            else:
                tau = state_dict['model_props']['best_tau'][examine_tag] if examine_tag != 'latest' else TAU
        
        # test_metrics, loss = run_observer(MODEL_PATH, tau, tag=parser.model_tag)


        process_samples(MODEL_PATH, tau=tau, tag=parser.model_tag)

