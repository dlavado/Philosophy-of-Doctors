

import torch
import torch.nn as nn
import wandb
import lightning as pl
import numpy as np
import ast
from tqdm import tqdm

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')


import scripts.constants as constants
import utils.pcd_processing as eda
from core.datasets.partnet import CATEGORY_CLASS_MAP
from utils.scripts_utils import main_arg_parser, resolve_criterion

from scripts.main import init_partnet, init_ts40k
from core.criterions.geneo_loss import GENEO_Loss
from core.lit_modules.lit_model_wrappers import LitSceneNet_multiclass


def init_criterion(model, **kwargs):
    criterion_params = {
        'tversky_alpha': wandb.config.tversky_alpha,
        'tversky_beta': wandb.config.tversky_beta,
        'tversky_smooth': wandb.config.tversky_smooth,
        'focal_gamma': wandb.config.focal_gamma,
    }


    criterion = torch.nn.CrossEntropyLoss(ignore_index=wandb.config.ignore_index) # default criterion; idx zero is noise

    
    if wandb.config.geneo_criterion:
        criterion = GENEO_Loss(criterion, 
                                model.get_geneo_params(),
                                model.get_cvx_coefficients(),
                                convex_weight=wandb.config.convex_weight,
                            )
        

    criterion_class = resolve_criterion(wandb.config.criterion)

    criterion = criterion_class(criterion, **criterion_params)

    model.criterion = criterion # assign criterion to model


def download_best_models(dataset, metric_name='JaccardIndex'):

    api = wandb.Api()
    runs = api.runs(f"{wandb.api.viewer()['entity']}/{project_name}")
    

    metric = 'test_' + 'Multiclass' + metric_name + '_epoch'
    metric_name_options = [metric, metric.replace('test', 'val'), metric[:-6], metric[:-6].replace('test', 'val')]
    print(metric_name_options)


    if dataset == 'partnet':

        best_runs = {name.lower(): 0.0 for name in CATEGORY_CLASS_MAP}

        for run in runs:
            config = run.config
            keys = list(run.summary.keys())
            # keys = sorted([key for key in keys if metric_name in key])
            # print(keys)
            if len(keys) == 0:
                continue # if the summary is empty, skip this run
            if not np.any([metric_name in key for key in keys]):
                continue # if the metric is not in the summary, skip this run

    
            metric = [key for key in keys if key in metric_name_options]
            if len(metric) == 0:
                continue
            metric_val = run.summary[metric[0]]
            
            keep_object = str(config["keep_objects"]).lower()

            if "[" in keep_object or "]" in keep_object:
                keep_object = ast.literal_eval(keep_object)
                if len(keep_object) > 1: # if more than one object, skip
                    continue
                else:
                    keep_object = str(keep_object[0]).lower()

            if keep_object in best_runs:

                if isinstance(best_runs[keep_object], dict):
                    if float(metric_val) > best_runs[keep_object][metric_name]:
                        best_runs[keep_object] = {
                            metric_name: metric_val,
                            "run_id": run.id,
                            "run_name": run.name,
                        }
                else:
                    best_runs[keep_object] = {
                        metric_name: metric_val,
                        "run_id": run.id,
                        "run_name": run.name,
                    }

        print(best_runs)
        
        for object in best_runs:
            # get run in run_id
            download_dir = os.path.join(experiment_path, 'best_model', f"{object}")
            print(best_runs[object])
            run = api.run(f"{wandb.api.viewer()['entity']}/{project_name}/{best_runs[object]['run_id']}")
            for file in run.files():
                if metric_name in file.name and file.name.endswith(".ckpt"):
                    model_path = os.path.abspath(file.download(download_dir, replace=True).name)
                    break

            best_runs[object]["model_path"] = model_path
    
        return best_runs
    
    elif dataset == 'ts40k':

        best_run = {
            metric_name: 0,
            "run_id": None,
            "run_name": None,
        }

        for run in runs:

            config = run.config
            keys = list(run.summary.keys())
            # keys = sorted([key for key in keys if metric_name in key])
            # print(keys)
            if len(keys) == 0:
                continue # if the summary is empty, skip this run
            if not np.any([metric_name in key for key in keys]):
                continue # if the metric is not in the summary, skip this run

    
            metric = [key for key in keys if key in metric_name_options]
            if len(metric) == 0:
                continue
            metric_val = run.summary[metric[0]]

            if float(metric_val) > best_run[metric_name]:
                
                best_run = {
                    metric_name: metric_val,
                    "run_id": run.id,
                    "run_name": run.name,
                    "model_path": None,
                }

        # get run in run_id
        download_dir = os.path.join(experiment_path, 'best_model')
        run = api.run(f"{wandb.api.viewer()['entity']}/{project_name}/{best_run['run_id']}")
        for file in run.files():
            if metric_name in file.name and file.name.endswith(".ckpt"):
                model_path = os.path.abspath(file.download(download_dir, replace=True).name)
                break

        best_run["model_path"] = model_path

        return best_run
    


def visualize_predictions(pt_locs:torch.Tensor, gt:torch.Tensor, pred:torch.Tensor):
    # bring args to cpu
    pt_locs = pt_locs.cpu().numpy()
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    print(pt_locs.shape, gt.shape, pred.shape)
    pcd = eda.np_to_ply(pt_locs)
    print(pcd)
    eda.color_pointcloud(pcd, gt)  # Color the point cloud with the ground truth
    eda.visualize_ply([pcd])
    eda.color_pointcloud(pcd, pred)      # Color the point cloud with the prediction
    eda.visualize_ply([pcd])
        



def predict(model_path, data_module):
    # Load the PyTorch model
    state = torch.load(model_path, map_location=device)
    print(state.keys())
    hparams = state["hyper_parameters"]
    print(hparams)
    arch_hparams = LitSceneNet_multiclass.get_model_architecture_hyperparameters()
    for key in arch_hparams:
        if key not in hparams:
            print(f"key {key} not in hparams; config: {wandb.config[key]}")
            hparams[key] = ast.literal_eval(str(wandb.config[key]))
    print("\n\n\n")
    print(hparams)

    if main_parser.dataset == 'partnet':
        hparams["num_classes"] = CATEGORY_CLASS_MAP[object.upper()[0] + object[1:]]

    # ovewirte some hparams
    hparams["num_observers"] = 10
    model = LitSceneNet_multiclass(**hparams, optimizer='adam', criterion=None)
    init_criterion(model)
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    # model = LitSceneNet_multiclass.load_from_checkpoint(model_path, **hparams, optimizer='adam', criterion=nn.CrossEntropyLoss())
    model.eval()
 
    # Use the provided data module to create an inference data loader
    data_module.setup(stage="test")
    dataloader = data_module.test_dataloader()

    # Make predictions using the loaded model
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting..."):
            batch = [b.to(device) for b in batch]
            voxel, gt, pt_loc = batch  # Modify this to extract the inputs from your dataset
            outputs = model(voxel, pt_loc)
            pred = model.prediction(outputs)

            for i in range(gt.shape[0]): # iterate over batch
                visualize_predictions(pt_loc[i], gt[i], pred[i])

                ans = input("Continue?")
                if ans == "n" or ans == "N" or ans == "no" or ans == "No":
                    break



def evaluate(dataset, object=None, visualize=False):

    if dataset == 'partnet':
        data_module = init_partnet(constants.PARTNET_PREPROCESSED_PATH, True)
        object = object.lower()
        model_dict = best_model[object]
        model_path = model_dict["model_path"]

    elif dataset == 'ts40k':
        data_module = init_ts40k(constants.TS40K_PREPROCESSED_PATH, True)
        model_path = best_model["model_path"]
    
    predict(model_path, data_module)

    
    



if __name__ == '__main__':
    import os

    entity = "dlavado"
    main_parser = main_arg_parser().parse_args()
    project_name = f"SceneNet_Multiclass_{main_parser.dataset}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment_path = constants.get_experiment_path('scenenet', main_parser.dataset)

    os.environ["WANDB_DIR"] = os.path.abspath(os.path.join(experiment_path, 'wandb'))

    config = os.path.join(experiment_path, 'defaults_config.yml')
    wandb.init(
        project=project_name, 
        dir = experiment_path,
        config=config,
        mode='disabled', 
    )

    object = 'lamp'

    # best_model = download_best_models(main_parser.dataset)


    if main_parser.dataset == 'ts40k':

        best_model = {'model_path': os.path.join(experiment_path, 'best_model', f"{object}" , 'checkpoints',  'MulticlassJaccardIndex.ckpt')}
    else:
        best_model = {object: {'model_path': os.path.join(experiment_path, 'best_model', f"{object}" , 'checkpoints',  'MulticlassJaccardIndex.ckpt')}}

    print(best_model)

    # EVALUATE MODEL
    evaluate(main_parser.dataset, object=object, visualize=True)




