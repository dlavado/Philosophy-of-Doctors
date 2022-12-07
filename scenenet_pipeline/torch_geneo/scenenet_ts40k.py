
# %%
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

from datasets.ts40kv2 import torch_TS40Kv2
from datasets.torch_transforms import Voxelization, ToTensor, ToFullDense
from torch_geneo.models.SCENE_Net import SCENE_Net, SCENE_Net_Class
from models.geneo_loss import PICKLE_PATH, GENEO_Loss, GENEO_Loss_Class
from observer_utils import *

import sys
from pathlib import Path


sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from Calibration.plot_calibration import ConfidenceHistogram, ReliabilityDiagram
from EDA import EDA_utils as eda
from VoxGENEO import Voxelization as Vox
import argparse

ROOT_PROJECT = Path(os.path.abspath(__file__)).parents[2].resolve()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXT_PATH = "/media/didi/TOSHIBA EXT/"
TS40K_PATH = os.path.join(EXT_PATH, 'TS40K/')


SCNET_PIPELINE = os.path.join(ROOT_PROJECT, 'scenenet_pipeline')

SAVED_SCNETS_PATH = os.path.join(SCNET_PIPELINE, 'torch_geneo/saved_scnets')
PICKLE_PATH = os.path.join(SCNET_PIPELINE, "torch_geneo/models")
FREQ_SAMPLES = os.path.join(SCNET_PIPELINE, "dataset/freq_samples")

TAU=0.65


ts40k_splits = ['samples', 'train', 'val', 'test']


def arg_parser():
    parser = argparse.ArgumentParser(description="Process observer arguments")

    # Visualization hotkey
    parser.add_argument("--vis", action="store_true")

    
    parser.add_argument("--val", action="store_true") # perform validation
    parser.add_argument("--no_train", action="store_true") # dont train the model - requires load_model flag
    parser.add_argument("--bce_loss", action='store_true') # use BCELoss

    # Model Loading
    parser.add_argument("--tuning", action='store_true') # Inits a new model with the model in --load_model
    parser.add_argument("--load_model", type=str, default=None) # 'latest' or path or index
    parser.add_argument("--model_tag", type=str, default='loss') # tag of the model to load
    parser.add_argument("--model_date", type=str, default='today')

    # Training Config
    parser.add_argument("--ts40k_split", type=str, choices=ts40k_splits, default='samples')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)

    # GENEO-Net definition
    parser.add_argument('-a', '--alpha', type=float, default=1)
    parser.add_argument('-e', '--epsilon', type=float, default=1e-1)
    parser.add_argument('-k', '--k_size', type=int, action='append', nargs="+", default=None)
    parser.add_argument('--cy', type=int, default=1)
    parser.add_argument('--cone', type=int, default=1)
    parser.add_argument('--neg', type=int, default=1)

    return parser.parse_args()

# def transform_batch(batch):
#     vxg, gt = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
#     batch = (vxg[:, :, -1], gt[:, :, -1])
#     return batch

def transform_batch(batch):
    return batch[0].to(device), batch[1].to(device)


def process_batch(gnet:SCENE_Net, batch, geneo_loss:GENEO_Loss, opt : Union[torch.optim.Optimizer, None], metrics:Union[MetricCollection, None], requires_grad=True):
    batch = transform_batch(batch)
    loss, pred = forward(gnet, batch, geneo_loss, opt, requires_grad)
    if metrics is not None:
        # gt = Vox.prob_to_label(batch[1].to(device), gt_tau).to(torch.int)
        gt = batch[1].to(device)
        # if torch.any((gt > 0) & (gt < 1)): # it has tower probabilities
        #     gt = Vox.prob_to_label(gt, gt_tau)
        metrics(torch.flatten(pred), torch.flatten(gt).to(torch.int))
    return loss, pred


def training_loop(gnet:SCENE_Net, train_loader, val_loader, geneo_loss, opt:torch.optim.Optimizer, tau=TAU):

    # ---- Model Checkpoint ---
    if parser.load_model is not None:
        gnet, chkp = load_state_dict(MODEL_PATH, gnet_class, parser.model_tag, kernel_size=None)

        opt = opt_class(gnet.parameters(), lr=LR)
        #opt.load_state_dict(chkp['optimizer_state_dict'])
        state_dict['models'][parser.model_tag] = chkp

        for k, v in chkp.items():
            if not isinstance(v, dict):
                print(f"{k}: {v}")

        if parser.tuning:
            chkp_epoch = 0
        else: # else we are resuming training
            chkp_epoch = chkp['epoch']
    else:
        chkp_epoch = 0


    params = gnet.get_dict_parameters()
    gnet_params = pd.DataFrame({'epoch': np.zeros(len(params)), 'name': params.keys(), 'value': params.values()})
    print("\n\nInitial GENEO parameters:")    
    print(gnet_params)

    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.001)
    # early_metric = 'FBetaScore'
    # early_stopping = EarlyStopping(tolerance=25, metric=early_metric)


    train_metrics = init_metrics(tau)
    if parser.val:
        val_metrics = init_metrics(tau)

    # `plot_metrics` saves the train/val metric results and loss for each epoch for later plotting
    init = torch.zeros((NUM_EPOCHS, 1 + parser.val), device=device)
    plot_metrics = dict([(met, torch.clone(init)) for met in train_metrics])
    plot_metrics['loss'] = init

    # `best_train_metrics` saves the best training metrics to update the run's state_dict
    best_train_metrics = dict([(met, 0) for met in train_metrics])
    best_train_metrics['loss'] = 1e10

    # --- Training Loop ---
    print(f"\n\nBegin training...\n{NUM_EPOCHS} epochs with {NUM_SAMPLES} samples")
    for epoch in range(chkp_epoch, NUM_EPOCHS):
        train_loss = torch.tensor(0.0, device=device)
        if parser.val:
            val_loss = torch.tensor(0.0, device=device)        
        for batch in tqdm(train_loader, desc=f"train"): 
            loss, _ = process_batch(gnet, batch, geneo_loss, opt, train_metrics)
            train_loss += loss
        
        if parser.val:
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"val  "):
                    loss, _ = process_batch(gnet, batch, geneo_loss, opt, val_metrics, requires_grad=False)
                    val_loss += loss
        
        #scheduler.step()
        
        # --- Calculate avg loss / metrics of train and validation loops ---
        train_res = train_metrics.compute()  
        train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} / {NUM_EPOCHS}:\n")
        print(f"\t(threshold = {tau})")
        print(f"\ttrain_loss = {train_loss:.5f};")
        for met in train_res: 
            print(f"\t{met} = {train_res[met]};")

        if parser.val:
            val_res = val_metrics.compute()
            val_loss = val_loss / len(val_loader)
            print(f"\tval_loss = {val_loss:.5f};")
            for met in val_res:
                print(f"\t{met} = {val_res[met]:.5f};")

        # --- Save Best Model ---

        for metric in train_res:
            if train_res[metric] >=  best_train_metrics[metric]:
                state_dict['models'][metric] = {
                    'loss' : train_loss,
                    'epoch': epoch,
                    'tau' : tau,
                    **train_res,
                    'model_state_dict': gnet.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }

                best_train_metrics[metric] = train_res[metric]

        state_dict['models']['latest'] = {
                'loss' : train_loss,
                'epoch': epoch,
                'tau' : tau,
                **train_res,
                'model_state_dict': gnet.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }

        if train_loss <= best_train_metrics['loss']:
            state_dict['models']['loss'] = {
                'loss' : train_loss,
                'epoch': epoch,
                'tau' : tau,
                **train_res,
                'model_state_dict': gnet.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }

            best_train_metrics['loss'] = train_loss
            

        # for name, param in gnet.named_parameters():
        #     print(f"{name} = {param.item():.4f}")
        # print("\n")

        for name, param in gnet.get_cvx_coefficients().items():
            print(f"{name} = {param.item():.4f}")
        print("\n")
        # --- Update GENEO Parameter Tracker ---
        params = gnet.get_dict_parameters()
        epoch_params = pd.DataFrame({'epoch': np.full(len(gnet.get_dict_parameters()), epoch+1), 'name': params.keys(), 'value': params.values()})
        gnet_params = pd.concat([gnet_params, epoch_params], ignore_index = True, axis = 0)

        # --- Update Loss/Metric Tracker
        if parser.val:
            plot_metrics['loss'][epoch] = torch.tensor([train_loss, val_loss])
            for metric in train_metrics:
                plot_metrics[metric][epoch] = torch.tensor([train_res[metric], val_res[metric]])
        else:
            plot_metrics['loss'][epoch] = train_loss[None, ...]
            for metric in train_metrics:
                plot_metrics[metric][epoch] = train_res[metric][None, ...]

        if torch.sum(plot_metrics['FBetaScore']) == 0 and epoch >= 50:
            break # early stop if metrics remain null after 50 epochs
        
        # early stopping
        # early_stopping(train_res[early_metric])
        # if early_stopping.early_stop:
        #     print("We are at epoch:", epoch)
        #     break
        
        train_metrics.reset()
        if parser.val:
            val_metrics.reset()
        
        if chkp_epoch % 10 == 0:
            torch.save(state_dict, MODEL_PATH)

    return plot_metrics, gnet_params, best_train_metrics
    

def train_observer(tau=TAU):

    print("Loading TS40K dataset...")
    ts40k_train_loader = DataLoader(ts40k_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    ts40k_val_loader = DataLoader(ts40k_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


    if parser.load_model is None:

        # --- Model Initialization ---
        geneos = {'cy': parser.cy,
                  'cone':parser.cone,
                  'neg':parser.neg
               }
               
        state_dict['model_props']['geneos_used'] = geneos
        gnet = gnet_class(geneo_num=geneos, kernel_size=kernel_size).to(device)
        opt = opt_class(gnet.parameters(), lr=LR)
    else:
        # both loaded from state_dict
        gnet = None
        opt = None

    # --- Loss and Optimizer ---
    torch.autograd.set_detect_anomaly(True)
    geneo_loss = gnet_loss(torch.tensor([]), hist_path=PICKLE_PATH, alpha=ALPHA, rho=RHO, epsilon=EPSILON)
    

    # --- Train GENEO Net ---
    plot_metrics, gnet_params, best_metrics = training_loop(gnet, ts40k_train_loader, ts40k_val_loader, geneo_loss, opt, tau)

 
    # --- Plot GENEO-kernel's parameters and Train/Val loss ---
    csv_name = os.path.join(META_MODEL_PATH, "GENEO_Net_parameters.csv")
    gnet_params.to_csv(csv_name, index=False)

    for metric in plot_metrics:
        if device != 'cpu':
            metric_data = plot_metrics[metric].detach().cpu().numpy()
        else:
            metric_data = plot_metrics[metric].detach().numpy()
        plot_metric(metric_data, META_MODEL_PATH,  f"GNet {metric}", legend=[f'train_{metric}', f'val_{metric}'])

    plot_geneo_params(csv_name, META_MODEL_PATH)

    torch.save(state_dict, MODEL_PATH)



def testing_observer(model_path, tau=TAU, tag='latest'):

    # --- Load Best Model ---
    gnet, chkp = load_state_dict(model_path, gnet_class, tag, kernel_size=kernel_size)
    print(f"Tag '{tag}' Loss = {chkp['loss']}")

    print("Load Testing Data...")
    ts40k_test_loader = DataLoader(ts40k_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    geneo_loss = gnet_loss(torch.tensor([]), hist_path=PICKLE_PATH, alpha=ALPHA, rho=RHO, epsilon=EPSILON)

    test_metrics = init_metrics(tau) 
    test_loss = 0
 
    # --- Test Loop ---
    print(f"\n\nBegin testing with tau={tau:.3f} ...")
    for batch in tqdm(ts40k_test_loader, desc=f"testing..."):
        loss, _ = process_batch(gnet, batch, geneo_loss, None, test_metrics, requires_grad=False)
        test_loss += loss

        # if parser.vis and vis:
        #     visualize_batch(batch[0], gt, pred, tau=tau)
        #     input("\n\nPress Enter to continue...")

    test_loss = test_loss /  len(ts40k_test_loader)
    test_res = test_metrics.compute()
    print(f"\ttest_loss = {test_loss:.3f};")
    for met in test_res:
        print(f"\t{met} = {test_res[met]:.3f};")

    return test_res, test_loss


def visualize_thresholding(model_path, taus):
    """
    Visualizes various samples for different threshold values
    """

    # --- Load Best Model ---
    gnet, _ = load_state_dict(model_path, gnet_class, parser.model_tag, kernel_size=kernel_size)

    print("Load Testing Data...")
    ts40k_test_loader = DataLoader(ts40k_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    geneo_loss = gnet_loss(torch.tensor([]), hist_path=PICKLE_PATH, alpha=ALPHA, rho=RHO, epsilon=EPSILON)

    for batch in tqdm(ts40k_test_loader, desc=f"..."):
        loss, pred  = process_batch(gnet, batch, geneo_loss, None, None, requires_grad=False)
        vox, gt = batch

        for tau in taus:
            print(f"\n\n\nGENEO Net Prediction with threshold {tau:.3f}")
            idx = np.random.randint(0, BATCH_SIZE, size=1)[0]
            visualize_batch(vox, gt, pred, idx, tau)


def thresholding(model_dir, state_dict, taus_size=10):

    model_path = os.path.join(model_dir, 'gnet.pt')

    roc = np.empty((taus_size, 5))

    best_loss = 1e10
    best_tau_loss = 0
    taus = torch.linspace(0.0, 0.95, taus_size)
    for i, tau in enumerate(taus):
        print(f"\n\n\nTesting for threshold = {tau}")
        with torch.no_grad():
            test_metrics, loss = testing_observer(model_path, tau, tag=testing_model_tag)
        pre = test_metrics['Precision']
        rec= test_metrics['Recall']
        f1 = test_metrics['F1Score']
        fbeta = test_metrics['FBetaScore']
        roc[i] = [pre, rec, f1, fbeta, tau]

        if loss <= best_loss:
            best_loss =  loss
            best_tau_loss = tau

    np.save(open('thresholding.npy', 'wb'), roc)
    input("Stop?")
    
    # --- Plot ROC ---
    plt.close()
    fig = plt.figure()
    ax = plt.subplot(111)
    rem_axis(ax, ['top', 'right'])  
    colours = cm.jet(np.linspace(0, 1, taus_size))
    for row, c in zip(roc, colours):
        #x, y = (row[:2] - min_val) / (max_val - min_val)
        x, y = row[:2]
        ax.scatter(x, y, label=f"{np.round(row[-1], decimals=3)}", color=c)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim([0, np.max(roc[:, :2])])
    ax.set_ylim([0, np.max(roc[:, :2])])
    plt.title(f"Precision-Recall Curve")
    plt.savefig(os.path.join(model_dir, f"ROC.png"))

    plt.close()
    fig = plt.figure()
    ax = plt.subplot(111)
    rem_axis(ax, ['top', 'right'])  
    plt.plot(roc[:, -1], roc[:, 2])
    xmax = roc[:, -1][np.argmax(roc[:, 2])]
    ymax = np.max(roc[:, 2])
    label= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    plt.annotate(label, # this is the text
                (xmax,ymax), # these are the coordinates parser.add_argument("--val", action="store_true")to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
    plt.xlabel('Threshold')
    plt.ylabel('F1_Score')
    plt.title(f"F1_Score per threshold")
    plt.savefig(os.path.join(model_dir, f"F1_Score_threshold.png"))

    plt.close()
    fig = plt.figure()
    ax = plt.subplot(111)
    rem_axis(ax, ['top', 'right'])  
    plt.plot(roc[:, -1], roc[:, 3])
    xmax = roc[:, -1][np.argmax(roc[:, 3])]
    ymax = np.max(roc[:, 3])
    label= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    plt.annotate(label, # this is the text
                (xmax,ymax), # these are the coordinates parser.add_argument("--val", action="store_true")to position the label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
    plt.xlabel('Threshold')
    plt.ylabel('FBetaScore')
    plt.title(f"FBetaScore(beta = {0.5}) per threshold")
    plt.savefig(os.path.join(model_dir, f"FBetaScore_threshold.png"))

    if parser.vis:
        visualize_thresholding(model_path, taus)

    print(f"\n\nBest tau = {best_tau_loss}")
    max_metrics = np.argmax(roc, axis=0)

    if not parser.vis:
        state_dict['model_props']["best_tau"] = {'loss' : best_tau_loss,
                                                'Precision' : roc[max_metrics[0], -1],
                                                'Recall': roc[max_metrics[1], -1],
                                                'F1Score' : roc[max_metrics[2], -1],
                                                'FBetaScore':roc[max_metrics[3], -1]
                                                }


def examine_samples(model_path, data_path, tau=None, tag='loss'):

    gnet, _ = load_state_dict(model_path, gnet_class, tag, kernel_size=kernel_size)

    ts40k = torch_TS40Kv2(dataset_path=data_path, split='val', transform=composed)

    st_dict = torch.load(model_path)
    if tau is None:
        tau = st_dict['model_props']['best_tau'][tag]

    print(f"\n\nExamining samples with model {model_path.split('/')[-2]}")
    print(f"\t with tau={tau} trying to optimize {tag}...")

    ts40k_loader = DataLoader(ts40k, batch_size=1, shuffle=True, num_workers=4)
    geneo_loss = GENEO_Loss(torch.tensor([]), hist_path=PICKLE_PATH, alpha=ALPHA, rho=RHO, epsilon=EPSILON)
    
    test_metrics = init_metrics(tau) 
    test_loss = 0

    gts = None
    preds = None
 
    # --- Test Loop ---
    for i in tqdm(range(0, 100)):
        # print(f"Examining TS40K Sample {i}...")
        pts, labels = ts40k[i]
        batch = pts[None], labels[None]
        loss, pred = process_batch(gnet, batch, geneo_loss, None, test_metrics, requires_grad=False)
        test_loss += loss

        test_res = test_metrics.compute()
        pre = test_res['Precision']
        rec = test_res['Recall']

        if preds is None and gts is None:
            preds = torch.flatten(pred)
            gts = torch.flatten(labels)
        else:
            preds = torch.concat((preds, torch.flatten(pred)))
            gts = torch.concat((gts, torch.flatten(labels)))

        if False:
            print(f"Precision = {pre}")
            print(f"Recall = {rec}")
            vox, gt = transform_batch(batch)
            visualize_batch(vox, gt, pred, tau=tau)
            input("\n\nPress Enter to continue...")

        test_metrics.reset()

    preds = preds.reshape(-1, 1).cpu().numpy()
    gts = gts.cpu().numpy()

    aux = np.concatenate((preds, gts[:,None]), axis=1)
    # aux = aux[aux[:, 0] > 0] # disregard zero predictions
    # aux = aux[aux[:, 1] > 0] # disregard zero targets
    preds, gts = aux[:, 0], aux[:, 1]
    preds = preds[:, None]
    print(preds.shape)

    hist = ConfidenceHistogram()
    hist = hist.plot(preds, gts, n_bins=20, logits=False)
    hist.show()
    dia = ReliabilityDiagram()
    dia = dia.plot(preds, gts, n_bins=20, logits=False)
    dia.show()

    input("?")

    test_loss = test_loss /  len(ts40k_loader)
    test_res = test_metrics.compute()
    print(f"\ttest_loss = {test_loss:.3f};")
    for met in test_res:
        print(f"\t{met} = {test_res[met]:.3f};")



def update_testing_results(models_root):
     for root, _, files in os.walk(models_root):

        if 'gnet.pt' in files: # root contains a gnet model
            model_path = os.path.join(root, 'gnet.pt')
            state_dict = torch.load(model_path)

            if 'model_props' not in state_dict.keys() or 'test_results' not in state_dict['model_props'].keys():
                continue
            print(f"\n\n\nUpdating {root}...")
            with torch.no_grad():
                taus_size=20
                thresholding(root, state_dict, taus_size=taus_size)
                best_tau = state_dict['model_props'].get('best_tau', TAU)
                if isinstance(best_tau, dict):
                    tau = best_tau['loss']
                else:
                    tau = best_tau
                test_res, test_loss = testing_observer(model_path, tau=tau, tag=testing_model_tag)

                if not parser.vis:
                    state_dict['model_props']['test_results'] = {'loss' : test_loss,
                                                                'tau': tau,
                                                                **test_res}

                torch.save(state_dict, model_path)




# %%
if __name__ == "__main__":
    
    print(f"root: {ROOT_PROJECT}")
    parser = arg_parser()

    torch.cuda.empty_cache()

    if parser.model_date == 'saved':
        META_TODAY = os.path.join(SAVED_SCNETS_PATH, 'models_geneo')
    elif parser.model_date == 'today':
        META_TODAY = os.path.join(SAVED_SCNETS_PATH, str(datetime.now().date()))
    else:
        date = parser.model_date.split('-') if '-' in parser.model_date else parser.model_date.split('/')
        date = list(map(int, date))
        META_TODAY = os.path.join(SAVED_SCNETS_PATH, str(datetime(*date).date()))

    print(META_TODAY)

    if parser.load_model is None:
        if not os.path.exists(META_TODAY):
            os.mkdir(META_TODAY)
        META_MODEL_PATH = os.path.join(META_TODAY, str(datetime.now()))
        os.mkdir(META_MODEL_PATH)
    else:
        try:
            idx = int(parser.load_model)
        except ValueError:
            # not an int
            idx = -1  if parser.load_model == 'latest' else None

        if idx is not None:
            model_dir = sorted(os.listdir(META_TODAY), key=lambda date: datetime.fromisoformat(date))[idx]
        else:
            model_dir = parser.load_model

        META_MODEL_PATH = os.path.join(META_TODAY, model_dir)
        
        assert os.path.exists(os.path.join(META_MODEL_PATH, "gnet.pt")), f"Directory {model_dir} does not contain a model checkpoint"

    
    MODEL_PATH = os.path.join(META_MODEL_PATH, "gnet.pt")
    
    if parser.tuning:
        """
        We transfer the `gnet.pt` to the current directory and perform regular training
        with that model as initialization.
        """
        if parser.load_model is None:
            ValueError("Provide a valid model path to perform fine-tuning")

        print(f"Initializing Fine-tuning of model {META_MODEL_PATH}")
        aux_date_path = os.path.join(SAVED_SCNETS_PATH, str(datetime.now().date())) # corresponds to META_TODAY
        if not os.path.exists(aux_date_path):
            os.mkdir(aux_date_path)
        aux_meta_model_path = os.path.join(aux_date_path, str(datetime.now()))
        os.mkdir(aux_meta_model_path)
        shutil.copy2(MODEL_PATH, aux_meta_model_path) # copy gnet to current directory
        META_TODAY = aux_date_path
        META_MODEL_PATH = aux_meta_model_path
        MODEL_PATH = os.path.join(META_MODEL_PATH, "gnet.pt")
    

    # --- Dataset Initialization ---
    vxg_size = (64, 64, 64)
    vox_size = (0.5, 0.5, 0.5) # only use vox_size after training or with batch_size = 1
    composed = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size, vox_size=None),
                        ToTensor(), 
                        ToFullDense(apply=(True, False))])

    ts40k_train = torch_TS40Kv2(dataset_path=TS40K_PATH, split=parser.ts40k_split, transform=composed)
    
    ts40k_val = torch_TS40Kv2(dataset_path=TS40K_PATH, split='val', transform=composed)

    ts40k_test = torch_TS40Kv2(dataset_path=TS40K_PATH, split='test', transform=composed)


    # --- HyperParameters ---
    NUM_EPOCHS = parser.epochs
    BATCH_SIZE = parser.batch_size
    NUM_SAMPLES = len(ts40k_train)
    NUM_ITER = math.ceil(NUM_SAMPLES / BATCH_SIZE)
    LR = parser.lr # learning rate
    #TAU = 0.65 # default metric threshold 
    ALPHA = parser.alpha #importance factor of the weighting scheme applied during dense loss
    RHO = 5 # scaling factor of cvx_loss
    EPSILON = parser.epsilon # min value of dense_loss
    gt_tau = 0.5 # default ground-truth threshold 

    kernel_size = (9, 6 ,6) if parser.k_size is None else tuple(*parser.k_size)
    

    # --- Model Definition ---
    gnet_class = SCENE_Net
    opt_class = torch.optim.RMSprop

    if parser.bce_loss:
        gnet_loss = GENEO_Loss_Class
    else:
        gnet_loss = GENEO_Loss


    # --- Define State Dict ---
    if parser.load_model is None:
        state_dict = {
            'model_props' : {'geneos_used' : {'cy': parser.cy,
                                              'cone':parser.cone,
                                              'neg':parser.neg
                                            },
                            'GENEO-Net Class' : gnet_class,
                            'GENEO_Net Loss' : gnet_loss,
                            'opt_class' : opt_class,
                            'kernel_size': kernel_size,
                            'epsilon': EPSILON,
                            'alpha' : ALPHA,
                            'rho' : RHO,
                            'test_results' : {}
                            },
            
            'run_config' : {'batch_size' : BATCH_SIZE,
                            'num_epochs' : NUM_EPOCHS,
                            'num_samples' : NUM_SAMPLES,
                            'learning_rate': LR
                            },

            'models' : {'loss':     {'loss' : 1e10,
                                    },
                        'latest':   {'loss' : 1e10,
                                    },
                    },

        }
    else:

        state_dict = torch.load(MODEL_PATH)

        if parser.tuning and not parser.vis:
            state_dict['run_config'] = {'batch_size' : BATCH_SIZE,
                                        'num_epochs' : NUM_EPOCHS,
                                        'num_samples' : NUM_SAMPLES,
                                        'learning_rate': LR
                                        },

            state_dict['model_props']['kernel_size'] = kernel_size
            state_dict['model_props']['epsilon'] = EPSILON
            state_dict['model_props']['alpha'] = ALPHA
            state_dict['model_props']['rho'] = RHO
            state_dict['model_props']['test_results'] = {}              
            

        

    pp.pprint(state_dict['model_props'])
    # print()
    # input("Stop?")
    
    # update_testing_results(META_DIR)
    # find_best_gnet(META_DIR)
    # input("Stop?")


    temperature_scaling(MODEL_PATH)
    input("Continue?")


    ###############################################################
    #                   Examining Samples                         #
    ###############################################################

    if parser.vis:
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
                    tau = None
                #examine_tag = examine_tag.strip()
            
            #test_metrics, loss = testing_observer(MODEL_PATH, tau, tag=parser.model_tag)

            examine_samples(MODEL_PATH, TS40K_PATH, tau=tau, tag=parser.model_tag)

    ###############################################################
    #                   Training GENEO-NET                        #
    ###############################################################

    if not parser.no_train:
        train_observer()

        # with torch.no_grad():
        #     #plot_gt_pred_corr(MODEL_PATH)


    ###############################################################
    #                       Thresholding                          #
    ###############################################################
    with torch.no_grad():
        taus_size=20
        testing_model_tag = parser.model_tag
        thresholding(META_MODEL_PATH, state_dict, taus_size=taus_size)
        # best_tau = state_dict['model_props'].get('best_tau', TAU)

        # if isinstance(best_tau, dict):
        #     tau = best_tau['loss']
        # else:
        #     tau = best_tau

        for metric in ['loss', 'F1Score', 'FBetaScore', 'Precision', 'latest']:
            
            tau = state_dict['model_props']['best_tau'][metric] if metric != 'latest' else TAU

            print(metric, tau)

            test_res, test_loss = testing_observer(MODEL_PATH, tau=tau, tag=metric)

            if not parser.vis:
                state_dict['model_props']['test_results'][metric] = {'loss' : test_loss,
                                                                        'tau': tau,
                                                                        **test_res}

        if parser.vis:
            taus = torch.linspace(0.5, 0.95, taus_size)
            visualize_thresholding(MODEL_PATH, taus)
        else:
            torch.save(state_dict, MODEL_PATH)

    chkp = torch.load(MODEL_PATH)
    pp.pprint(chkp['models'].keys())
    pp.pprint(chkp['model_props'])

    ###############################################################
    #                   Saving Run Config                         #
    ###############################################################

    text_file = open(os.path.join(META_MODEL_PATH, 'model_props.txt'), "w")

    text_file.write(pp.pformat(chkp['models'].keys()))
    text_file.write('\n\n\n' + '#'*100+'\n'+'#'*100 + '\n\n\n')
    if parser.no_train:
        text_file.write(f"FINE TUNED FROM MODEL: {parser.load_model} ; {parser.model_date} ; {parser.model_tag} ;")
    else:
        text_file.write(pp.pformat(chkp['run_config']))
    text_file.write('\n\n\n' + '#'*100+'\n'+'#'*100 + '\n\n\n')
    text_file.write(pp.pformat(chkp['model_props']))

    text_file.close()


    # %%

    # META_DIR = os.path.join(ROOT_PROJECT, 'metadata_geneo')

    # META_TODAY = os.path.join(ROOT_PROJECT, 'metadata_geneo', '2022-06-15')

    # for meta in os.listdir(META_TODAY):
    #     if os.path.exists(os.path.join(META_TODAY, meta,"GENEO_Net_parameters.csv")):
    #         plot_geneo_params(os.path.join(META_TODAY, meta,"GENEO_Net_parameters.csv"), os.path.join(META_TODAY, meta))

    # META_MODEL_PATH = os.path.join(META_DIR, sorted(os.listdir(META_DIR))[1])

    # meta_model_imgs = [os.path.join(META_MODEL_PATH, f) for f in os.listdir(META_MODEL_PATH) if '.png' in f]
    # print(meta_model_imgs)
    # merge_imgs('training_info', meta_model_imgs)
# %%

# %%
