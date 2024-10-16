
from datetime import datetime
from pprint import pprint
from typing import List, Mapping
import warnings
import numpy as np
import sys
import os
import yaml
import ast

# Vanilla PyTorch
import torch
from torchvision.transforms import Compose


# PyTorch Lightning
import pytorch_lightning as pl
import  pytorch_lightning.callbacks as  pl_callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

import constants

import utils.pcd_processing as eda
import utils.scripts_utils as su

from core.admm_utils.admm_constraints import Arrow_Constraint, Constraint, Convexity_Constraint, Cylinder_Constraint, Nonnegativity_Constraint
from core.criterions.admm_loss import ADMM_Loss
from core.criterions.constrained_loss import Constrained_Loss


import core.lit_modules.lit_callbacks as lit_callbacks
import core.lit_modules.lit_model_wrappers as lit_models
from core.lit_modules.lit_data_wrappers import LitTS40K

from core.datasets.torch_transforms import Voxelization, ToTensor, ToFullDense


"""
TODO: add CLI that overrides default_config.yml
TODO: differiantiate between fine-tuning and resume-training
"""

"""
DONE: add 3D point cloud visualizer for testing
DONE: add .onnx model export
DONE: investigate gradient accumulation and the gradient from tversky_loss
DONE: SCENE NET parameter watch from histograms to line plots
DONE: update sweeps and ensure that sweeps are working
DONE: code the GENEO layer differently to try and get a better propragation of the gradients
"""





def init_callbacks(ckpt_dir):
    # Call back definition
    callbacks = []
    model_ckpts: List[pl_callbacks.ModelCheckpoint] = []

    ckpt_metrics = [str(met) for met in su.init_metrics()]

    for metric in ckpt_metrics:
        model_ckpts.append(
            lit_callbacks.callback_model_checkpoint(
                dirpath=ckpt_dir,
                filename=f"{metric}",
                monitor=f"train_{metric}",
                mode="max",
                save_top_k=2,
                save_last=False,
                every_n_epochs=wandb.config.checkpoint_every_n_epochs,
                every_n_train_steps=wandb.config.checkpoint_every_n_steps,
                verbose=False,
            )
        )


    model_ckpts.append( # train loss checkpoint
        lit_callbacks.callback_model_checkpoint(
            dirpath=ckpt_dir, #None for default logger dir
            filename=f"train_loss",
            monitor=f"train_loss",
            mode="min",
            every_n_epochs=wandb.config.checkpoint_every_n_epochs,
            every_n_train_steps=wandb.config.checkpoint_every_n_steps,
            verbose=False,
        )
    )

    callbacks.extend(model_ckpts)

    early_stop_callback = EarlyStopping(monitor=wandb.config.early_stop_metric, 
                                        min_delta=0.00, 
                                        patience=30, 
                                        verbose=False, 
                                        mode="max")

    callbacks.append(early_stop_callback)

    return callbacks


def init_model(model_wrapper_class, criterion, ckpt_path):

    if wandb.config.resume_from_checkpoint:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        
        print(f"Resuming from checkpoint {ckpt_path}")
        model = model_wrapper_class.load_from_checkpoint(ckpt_path,
                                                        criterion=criterion,
                                                        optimizer=wandb.config.optimizer,
                                                        learning_rate=wandb.config.learning_rate,
                                                        metric_initilizer=su.init_metrics)
    else:
        # Model definition
        geneo_config = {
            'cy'   : wandb.config.cylinder_geneo,
            'cone' : wandb.config.arrow_geneo,
            'neg'  : wandb.config.neg_sphere_geneo, 
        }

        model = model_wrapper_class(geneo_config,
                                    ast.literal_eval(wandb.config.kernel_size),
                                    criterion,
                                    wandb.config.optimizer,
                                    wandb.config.learning_rate,
                                    su.init_metrics)
        
    return model


def init_data(data_path):
    vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
    vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1
    composed = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size, vox_size=vox_size),
                        ToTensor(), 
                        ToFullDense(apply=(True, True))])

    data_module = LitTS40K(data_path,
                           wandb.config.batch_size,
                           composed,
                           wandb.config.num_workers,
                           wandb.config.val_split,
                           wandb.config.test_split)
    
    return data_module


def set_up_admm_criterion(base_criterion, model:lit_models.LitSceneNet_ADMM, constraints:Mapping[str, Constraint]):

    return ADMM_Loss(base_criterion, 
                     constraints, 
                     model.get_model_parameters(), 
                     wandb.config.admm_rho,
                     wandb.config.admm_rho_update_factor,
                     wandb.config.admm_rho_max,
                     wandb.config.admm_stepsize,
                     wandb.config.admm_stepsize_update_factor
            )


def set_up_penalty_criterion(base_criterion, constraints):

    return Constrained_Loss(base_criterion, constraints)



def main():
    # ------------------------
    # 1 INIT CALLBACKS
    # ------------------------

    if not wandb.config.resume_from_checkpoint:
        ckpt_dir = os.path.join(wandb.run.dir, "checkpoints") 
    else:
        ckpt_dir = wandb.config.checkpoint_dir

    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')

    callbacks = init_callbacks(ckpt_dir)

    # ------------------------
    # 2 INIT MODEL
    # ------------------------
    if wandb.config.convergence_mode.lower() == 'admm':
        model = init_model(lit_models.LitSceneNet_ADMM, None, ckpt_path)
    elif wandb.config.convergence_mode.lower() == 'penalty':
        model = init_model(lit_models.LitSceneNet_Penalty, None, ckpt_path)
    elif wandb.config.convergence_mode.lower() == 'auglag':
        model = init_model(lit_models.LitSceneNet_AugLag, None, ckpt_path)
    else:
        raise ValueError(f"Convergence mode {wandb.config.convergence_mode} not recognized.")
   
    # ------------------------
    # 3 INIT TRAINING CRITERION
    # ------------------------


    weighting_scheme_path = wandb.config.weighting_scheme_path

    if not os.path.exists(weighting_scheme_path): # adds full path to the weighting scheme
        weighting_scheme_path = os.path.join(ROOT_PROJECT, weighting_scheme_path)
        wandb.config.update({"weighting_scheme_path": weighting_scheme_path}, allow_val_change=True)
    if not os.path.exists(weighting_scheme_path): # if the path still does not exist, use default scheme
        print(f"Weighting scheme path {weighting_scheme_path} does not exist. Using default scheme.")
        weighting_scheme_path = WEIGHT_SCHEME_PATH
        wandb.config.update({"weighting_scheme_path": weighting_scheme_path}, allow_val_change=True)
    if not os.path.exists(weighting_scheme_path): # if the path still does not exist
        ValueError(f"Weighting scheme path {weighting_scheme_path} does not exist.")


    criterion_params = {
        'weighting_scheme_path' : weighting_scheme_path,
        'weight_alpha': wandb.config.weight_alpha,
        'weight_epsilon': wandb.config.weight_epsilon,
        'mse_weight': wandb.config.mse_weight,
        'convex_weight': wandb.config.convex_weight,
        'tversky_alpha': wandb.config.tversky_alpha,
        'tversky_beta': wandb.config.tversky_beta,
        'tversky_smooth': wandb.config.tversky_smooth,
        'focal_gamma': wandb.config.focal_gamma,
    }

    criterion_class = su.resolve_criterion(wandb.config.criterion)
    base_criterion = criterion_class(**criterion_params)


    constraints = {
        'arrow'         : Arrow_Constraint(),
        'cylinder'      : Cylinder_Constraint(kernel_height=ast.literal_eval(wandb.config.kernel_size)[1]),
        'convexity'     : Convexity_Constraint(),
        'nonnegativity' : Nonnegativity_Constraint(),
    }


    if wandb.config.convergence_mode.lower() == 'admm':
        criterion = set_up_admm_criterion(base_criterion, model, constraints)
        model.criterion = criterion # dynamically set up model criterion
    elif wandb.config.convergence_mode.lower() == 'penalty':
        criterion = set_up_penalty_criterion(base_criterion, constraints)
        model.criterion = criterion # dynamically set up model criterion
    elif not wandb.config.convergence_mode.lower() == 'auglag':
        raise ValueError(f"Convergence mode {wandb.config.convergence_mode} not recognized.")
    
    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------

    data_path = wandb.config.data_path

    if not os.path.exists(wandb.config.data_path):
        data_path = TS40K_PATH
        wandb.config.update({'data_path': data_path}, allow_val_change=True) # override data path

    data_module = init_data(data_path)
    
    # ------------------------
    # 5 INIT TRAINER
    # ------------------------

    # WandbLogger
    wandb_logger = WandbLogger(project=f"{project_name}",
                               log_model=True, 
                               name=wandb.run.name, 
                               config=wandb.config)
    
    wandb_logger.watch(model, log="all", log_freq=100)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        detect_anomaly=True,
        max_epochs=wandb.config.max_epochs,
        gpus=wandb.config.gpus,
        #fast_dev_run = wandb.config.fast_dev_run,
        auto_lr_find=wandb.config.auto_lr_find,
        auto_scale_batch_size=wandb.config.auto_scale_batch_size,
        profiler=wandb.config.profiler,
        enable_model_summary=True,
        accumulate_grad_batches = ast.literal_eval(wandb.config.accumulate_grad_batches),
        #resume_from_checkpoint=ckpt_path
    )

    # if wandb.config.auto_lr_find or wandb.config.auto_scale_batch_size:
    #     trainer.tune(model, data_module) # auto_lr_find and auto_scale_batch_size
    #     print(f"Learning rate in use is: {model.hparams.learning_rate}")


    trainer.fit(model, data_module)

    print(f"{'='*20} Model ckpt scores {'='*20}")

    for ckpt in trainer.callbacks:
        if isinstance(ckpt, pl_callbacks.ModelCheckpoint):
            print(f"{ckpt.monitor} checkpoint : score {ckpt.best_model_score}")

    wandb_logger.experiment.unwatch(model)

    # ------------------------
    # 6 TEST
    # ------------------------

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} does not exist. Using last checkpoint.")
        ckpt_path = None

    if wandb.config.save_onnx:
        print("Saving ONNX model...")
        onnx_file_path = os.path.join(ckpt_dir, f"{project_name}.onnx")
        input_sample = next(iter(data_module.test_dataloader()))
        model.to_onnx(onnx_file_path, input_sample, export_params=True)
        wandb_logger.log({"onnx_model": wandb.File(onnx_file_path)})

    trainer.test(model, 
                 datamodule=data_module,
                 ckpt_path=ckpt_path) # use the last checkpoint
    

if __name__ == '__main__':
    import pathlib
    #from constants import ROOT_PROJECT, TS40K_PATH, WEIGHT_SCHEME_PATH
    ROOT_PROJECT = pathlib.Path(__file__).resolve().parent.parent
    
    if "didi" in str(ROOT_PROJECT):
        EXT_PATH = "/media/didi/TOSHIBA EXT/"
    else:
        EXT_PATH = "/home/d.lavado/" #cluster data dir


    TS40K_PATH = os.path.join(EXT_PATH, 'TS40K-NEW/')
    WEIGHT_SCHEME_PATH = os.path.join(ROOT_PROJECT, 'core/criterions/hist_estimation.pickle')
    EXPERIMENTS_PATH = os.path.join(ROOT_PROJECT, 'experiments')

    # --------------------------------
    su.fix_randomness()
    warnings.filterwarnings("ignore")

    main_parser = su.main_arg_parser().parse_args()

    method = main_parser.method
    model_name = 'admm'
    dataset_name = 'ts40k'
    project_name = f"{model_name}_{dataset_name}"

    run_name = f"{project_name}_{method}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    #config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = os.path.join(EXPERIMENTS_PATH, f"{model_name}_{dataset_name}")
    # print(pathlib.Path(__file__).resolve().parent.parent)
    # print(f"Experiment path: {experiment_path}")
    # print(ROOT_PROJECT)
    
    # input("Press Enter to continue...")


    os.environ["WANDB_DIR"] = os.path.abspath(experiment_path)

    # raw_config = yaml.safe_load(open(config_path))
    # pprint(raw_config)

    print(f"\n\n{'='*50}")
    print("Entering main method...") 


    if main_parser.wandb_sweep: 
        #sweep mode
        print("wandb sweep.")
        wandb.init(project = project_name, 
                dir = experiment_path,
                name = run_name,
        )
    else:
        # default mode
        sweep_config = os.path.join(experiment_path, 'admm_config.yml')
        print(f"Loading config from {sweep_config}")

        print("wandb init.")

        wandb.init(project=project_name, 
                dir = experiment_path,
                name = run_name,
                config=sweep_config,
                #mode='disabled'
        )

        #pprint(wandb.config)

    main()
    

    



