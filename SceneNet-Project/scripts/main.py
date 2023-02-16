
from datetime import datetime
from pprint import pprint
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




if __name__ == '__main__':

    # --------------------------------
    su.fix_randomness()
    warnings.filterwarnings("ignore")

    model_name = 'scenenet'
    dataset_name = 'ts40k'

    config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = get_experiment_path(model_name, dataset_name)

    # raw_config = yaml.safe_load(open(config_path))
    # pprint(raw_config)

    print(f"\n\n{'='*50}")
    print("Entering main method...") 
    #wandb.init(config=config_path) # pass config here if using CLI
    wandb.init(project="scenenet", 
               dir = experiment_path,
               name = f'{model_name}_{dataset_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
               config=os.path.join(experiment_path, 'defaults_config.yml'))
    print("wandb init.")

    pprint(wandb.config)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

    # Call back definition
    callbacks = []

    model_ckpts = []
    metric_names = [str(met) for met in su.init_metrics()]

    for metric in metric_names:
        model_ckpts.append(
            lit_callbacks.callback_model_checkpoint(
                dirpath=None,
                filename=f"{metric}_ckpt",
                monitor=f"train_{metric}",
                mode="max",
                save_top_k=1,
                save_last=True,
                every_n_epochs=1,
                verbose=True,
            )
        )

    model_ckpts.append( # train loss checkpoint
        lit_callbacks.callback_model_checkpoint(
            dirpath=None, #default logger dir
            filename=f"train_loss_ckpt",
            monitor=f"train_loss",
            mode="min"
        )
    )

    callbacks.extend(model_ckpts)

    if not os.path.exists(wandb.config.weighting_scheme_path):
        wandb.config.update({"weighting_scheme_path": os.path.join(ROOT_PROJECT, wandb.config.weighting_scheme_path)}, allow_val_change=True)
    if not os.path.exists(wandb.config.weighting_scheme_path):
        print(f"Weighting scheme path {wandb.config.weighting_scheme_path} does not exist. Using default scheme.")
    else:
        wandb.config.update({"weighting_scheme_path": WEIGHT_SCHEME_PATH}, allow_val_change=True)

    # resolving criterion
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
    # 2 INIT DATA MODULE
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
    # 3 INIT TRAINER
    # ------------------------

    # WandbLogger
    wandb_logger = WandbLogger(project="scenenet", name=wandb.run.name, config=wandb.config)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=wandb.config.max_epochs,
        gpus=wandb.config.gpus,
        #fast_dev_run = wandb.config.fast_dev_run,
        auto_lr_find=wandb.config.auto_lr_find,
        auto_scale_batch_size=wandb.config.auto_scale_batch_size,
        profiler=wandb.config.profiler,
        enable_model_summary=True
        #accumulate_grad_batches,
        #resume_from_checkpoint,
    )

    trainer.fit(model, data_module)

    trainer.test(model, 
                 datamodule=data_module,
                 ckpt_path=None) # use the last checkpoint
    



