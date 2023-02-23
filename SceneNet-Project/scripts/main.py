
from datetime import datetime
from pprint import pprint
from typing import List
import warnings
import numpy as np
import sys
import os
import yaml
import ast

# 🍦 Vanilla PyTorch
import torch
from torchvision.transforms import Compose


# ⚡ PyTorch Lightning
import pytorch_lightning as pl
import  pytorch_lightning.callbacks as  pl_callbacks

# 🏋️‍♀️ Weights & Biases
import wandb
# ⚡ 🤝 🏋️‍♀️
from pytorch_lightning.loggers import WandbLogger

# 📚 Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from constants import ROOT_PROJECT, TS40K_PATH, WEIGHT_SCHEME_PATH, get_experiment_config_path, get_experiment_path

import utils.pcd_processing as eda
import utils.scripts_utils as su

import core.lit_modules.lit_callbacks as lit_callbacks
import core.lit_modules.lit_model_wrappers as lit_models
from core.lit_modules.lit_data_wrappers import LitTS40K

from core.datasets.torch_transforms import Voxelization, ToTensor, ToFullDense


"""
TODO: add CLI that overrides default_config.yml
TODO: differiantiate between fine-tuning and resume-training
TODO: SCENE NET parameter watch from histograms to line plots
TODO: update sweeps and ensure that sweeps are working
TODO: code the GENEO layer differently to try and get a better propragation of the gradients
"""

"""
DONE: add 3D point cloud visualizer for testing
DONE: add .onnx model export
DONE: investigate gradient accumulation and the gradient from tversky_loss
"""


if __name__ == '__main__':

    # --------------------------------
    su.fix_randomness()
    warnings.filterwarnings("ignore")

    model_name = 'scenenet'
    dataset_name = 'ts40k'
    project_name = f"{model_name}_{dataset_name}"

    config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = get_experiment_path(model_name, dataset_name)

    # raw_config = yaml.safe_load(open(config_path))
    # pprint(raw_config)

    print(f"\n\n{'='*50}")
    print("Entering main method...") 
    #wandb.init(config=config_path) # pass config here if using CLI
    wandb.init(project=project_name, 
               dir = experiment_path,
               name = f'{project_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
               config=os.path.join(experiment_path, 'defaults_config.yml'),
               #mode='disabled'
    )

    print("wandb init.")

    pprint(wandb.config)

    # ------------------------
    # 1 INIT CALLBACKS
    # ------------------------

    # Call back definition
    callbacks = []

    model_ckpts: List[pl_callbacks.ModelCheckpoint] = []
    if not wandb.config.resume_from_checkpoint:
        ckpt_dir = os.path.join(wandb.run.dir, "checkpoints") 
    else:
        ckpt_dir = wandb.config.checkpoint_dir

   
    ckpt_metrics = [str(met) for met in su.init_metrics()]

    for metric in ckpt_metrics:
        model_ckpts.append(
            lit_callbacks.callback_model_checkpoint(
                dirpath=ckpt_dir,
                filename=f"{metric}",
                monitor=f"train_{metric}",
                mode="max",
                save_top_k=1,
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

    # ------------------------
    # 2 INIT TRAINING CRITERION
    # ------------------------

    if not os.path.exists(wandb.config.weighting_scheme_path):
        wandb.config.update({"weighting_scheme_path": os.path.join(ROOT_PROJECT, wandb.config.weighting_scheme_path)}, allow_val_change=True)
    if not os.path.exists(wandb.config.weighting_scheme_path):
        print(f"Weighting scheme path {wandb.config.weighting_scheme_path} does not exist. Using default scheme.")
    else:
        wandb.config.update({"weighting_scheme_path": WEIGHT_SCHEME_PATH}, allow_val_change=True)

    criterion_params = {
        'weighting_scheme_path' : wandb.config.weighting_scheme_path,
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
    criterion = criterion_class(**criterion_params)

    # ------------------------
    # 3 INIT MODEL
    # ------------------------

    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')

    if wandb.config.resume_from_checkpoint:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        
        print(f"Resuming from checkpoint {ckpt_path}")
        model = lit_models.LitSceneNet.load_from_checkpoint(ckpt_path,
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

        model = lit_models.LitSceneNet(geneo_config,
                                       ast.literal_eval(wandb.config.kernel_size),
                                       criterion,
                                       wandb.config.optimizer,
                                       wandb.config.learning_rate,
                                       su.init_metrics)

    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------

    if not os.path.exists(wandb.config.data_path):
        wandb.config.update({'data_path': TS40K_PATH}, allow_val_change=True) # override data path

    vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
    vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1
    composed = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size, vox_size=vox_size),
                        ToTensor(), 
                        ToFullDense(apply=(True, True))])

    data_module = LitTS40K(wandb.config.data_path,
                           wandb.config.batch_size,
                           composed,
                           wandb.config.num_workers,
                           wandb.config.val_split,
                           wandb.config.test_split)
    
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
        max_epochs=wandb.config.max_epochs,
        gpus=wandb.config.gpus,
        #fast_dev_run = wandb.config.fast_dev_run,
        auto_lr_find=wandb.config.auto_lr_find,
        auto_scale_batch_size=wandb.config.auto_scale_batch_size,
        profiler=wandb.config.profiler,
        enable_model_summary=True,
        accumulate_grad_batches = wandb.config.accumulate_grad_batches
        #resume_from_checkpoint=ckpt_path
    )

    #if wandb.config.auto_lr_find or wandb.config.auto_scale_batch_size:
        #trainer.tune(model, data_module) # auto_lr_find and auto_scale_batch_size

    trainer.fit(model, data_module)

    print(f"{'='*20} Model ckpt scores {'='*20}")
    for ckpt in model_ckpts:
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
    



