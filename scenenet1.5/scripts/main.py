
from datetime import datetime
from pprint import pprint
from typing import List
import warnings
import numpy as np
import sys
import os
import yaml
import ast

# Vanilla PyTorch
import torch
from torchvision.transforms import Compose
import torch.nn.functional as F


# PyTorch Lightning
import pytorch_lightning as pl
import  pytorch_lightning.callbacks as  pl_callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import BatchSizeFinder
from lightning.pytorch.tuner import Tuner

# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from constants import *

import utils.scripts_utils as su
import utils.pcd_processing as eda

import core.lit_modules.lit_callbacks as lit_callbacks
import core.lit_modules.lit_model_wrappers as lit_models
from core.lit_modules.lit_data_wrappers import LitTS40K, LitTS40K_Preprocessed
from core.datasets.partnet import LitPartNetDataset


from core.criterions.geneo_loss import GENEO_Loss


from core.datasets.torch_transforms import Dict_to_Tuple, EDP_Labels, Farthest_Point_Sampling, Normalize_Labels, ToTensor, Voxelization_withPCD

class EvalBatchSizeFinder(BatchSizeFinder):
    def __init__(self, mode, *args, **kwargs):
        super().__init__(mode, *args, **kwargs)

    def on_test_start(self, trainer, pl_module):
        self.scale_batch_size(trainer, pl_module)


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
                monitor=f"val_{metric}",
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
            filename=f"val_loss",
            monitor=f"val_loss",
            mode="min",
            every_n_epochs=wandb.config.checkpoint_every_n_epochs,
            every_n_train_steps=wandb.config.checkpoint_every_n_steps,
            verbose=False,
        )
    )

    callbacks.extend(model_ckpts)

    # batch_finder = EvalBatchSizeFinder(mode='power')
    # callbacks.append(batch_finder)

    # early_stop_callback = EarlyStopping(monitor=wandb.config.early_stop_metric, 
    #                                     min_delta=0.00, 
    #                                     patience=10, 
    #                                     verbose=False, 
    #                                     mode="max")

    # callbacks.append(early_stop_callback)

    return callbacks


def init_model(criterion, ckpt_path):

    if wandb.config.resume_from_checkpoint:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
        
        print(f"Resuming from checkpoint {ckpt_path}")
        model = lit_models.LitSceneNet_multiclass.load_from_checkpoint(ckpt_path,
                                                                criterion=criterion,
                                                                optimizer=wandb.config.optimizer,
                                                                learning_rate=wandb.config.learning_rate,
                                                                metric_initilizer=su.init_metrics
                                                            )
    else:
        # Model definition
        geneo_config = {
            'cy'   : wandb.config.cylinder_geneo,
            'arrow': wandb.config.arrow_geneo,
            'neg'  : wandb.config.neg_sphere_geneo,
            'disk' : wandb.config.disk_geneo,
            'cone' : wandb.config.cone_geneo,
            'ellip': wandb.config.ellipsoid_geneo, 
        }

        hidden_dims = ast.literal_eval(wandb.config.hidden_dims)

        model = lit_models.LitSceneNet_multiclass(geneo_config,
                                        wandb.config.num_observers,
                                        ast.literal_eval(wandb.config.kernel_size),
                                        hidden_dims,
                                        wandb.config.num_classes,
                                        criterion,
                                        wandb.config.optimizer,
                                        wandb.config.learning_rate,
                                        su.init_metrics
                                    )
        
    return model


def init_ts40k(data_path, preprocessed=False):

    if preprocessed:
        return LitTS40K_Preprocessed(data_path,
                                    wandb.config.batch_size,
                                    wandb.config.num_workers,
                                    wandb.config.val_split,
                                    wandb.config.test_split,
                                )

    vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
    vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1

    # keep_labels = list(eda.DICT_EDP_LABELS.keys())
    # semantic_labels = [eda.DICT_NEW_LABELS[label] for label in keep_labels]
    # assert len(torch.unique(torch.tensor(semantic_labels))) == wandb.config.num_classes
    composed = Compose([
                        ToTensor(),

                        # EDP_Labels(),

                        Farthest_Point_Sampling(wandb.config.fps_points),
        
                        Voxelization_withPCD(keep_labels='all', 
                                             vxg_size=vxg_size, 
                                             vox_size=vox_size
                                            ),

                        EDP_Labels(),
                        # Normalize_Labels()
                    ])
    

    data_module = LitTS40K(data_path,
                           wandb.config.batch_size,
                           composed,
                           wandb.config.num_workers,
                           wandb.config.val_split,
                           wandb.config.test_split,
                           min_points= wandb.config.min_points
                        )
    
    return data_module


def init_partnet(data_path):

    vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
    vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1
    composed = Compose([
                        Dict_to_Tuple(omit=['category']),

                        # ToTensor(),

                        # Farthest_Point_Sampling(wandb.config.fps_points), # ParNet data already has FPS
        
                        Voxelization_withPCD(keep_labels='all', 
                                             vxg_size=vxg_size, vox_size=vox_size
                                            ),
                        # Normalize_Labels()
                    ])

    return LitPartNetDataset(data_path,
                            wandb.config.coarse_level,
                            wandb.config.batch_size,
                            transform=composed,
                            keep_objects=ast.literal_eval(wandb.config.keep_objects),
                            num_workers=wandb.config.num_workers,
                        )


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

    # TODO: add resume training
    # criterion will be dynamically assigned; GENEO criterion require model parameters
    model = init_model(None, ckpt_path)

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

    criterion_params = {
        'weighting_scheme_path' : weighting_scheme_path,
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
        

    criterion_class = su.resolve_criterion(wandb.config.criterion)

    criterion = criterion_class(criterion, **criterion_params)

    model.criterion = criterion # assign criterion to model

    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------

    data_path = wandb.config.data_path

    if not os.path.exists(wandb.config.data_path):
        if dataset_name == 'ts40k':
            data_path = TS40K_PATH
            if wandb.config.preprocessed:
                data_path = TS40K_PREPROCESSED_PATH
        elif dataset_name == 'partnet':
            data_path = PARTNET_PATH
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")
        wandb.config.update({'data_path': data_path}, allow_val_change=True) # override data path

    if dataset_name == 'ts40k':
        data_module = init_ts40k(data_path, wandb.config.preprocessed)
    
    elif dataset_name == 'partnet':
        data_module = init_partnet(data_path)
    
    print(f"\n=== Data Module {dataset_name.upper()} initialized. ===\n")
    print(f"{data_module}")
    print(data_path)   
    
    # ------------------------
    # 5 INIT TRAINER
    # ------------------------

    # WandbLogger
    wandb_logger = WandbLogger(project=f"{project_name}",
                               log_model=True, 
                               name=wandb.run.name, 
                               config=wandb.config
                            )
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        detect_anomaly=True,
        #
        max_epochs=wandb.config.max_epochs,
        accelerator=wandb.config.accelerator,
        devices=wandb.config.devices,
        num_nodes=wandb.config.num_nodes,
        strategy=wandb.config.strategy,
        #fast_dev_run = wandb.config.fast_dev_run,
        profiler=wandb.config.profiler if wandb.config.profiler else None,
        precision=wandb.config.precision,
        enable_model_summary=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accumulate_grad_batches = wandb.config.accumulate_grad_batches,
        # resume_from_checkpoint=ckpt_path
    )

    # if wandb.config.auto_lr_find or wandb.config.auto_scale_batch_size:
    # #   trainer.tune(model, data_module) # auto_lr_find and auto_scale_batch_size
    #     tuner = Tuner(trainer)
    #     tuner.scale_batch_size(model, datamodule=data_module, mode="power")

    trainer.fit(model, data_module)

    print(f"{'='*20} Model ckpt scores {'='*20}")

    for ckpt in trainer.callbacks:
        if isinstance(ckpt, pl_callbacks.ModelCheckpoint):
            print(f"{ckpt.monitor} checkpoint : score {ckpt.best_model_score}")

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

    test_results = trainer.test(model, 
                            datamodule=data_module,
                            ckpt_path=ckpt_path)
    
    test_results = trainer.test(model,
                                datamodule=data_module,
                                ckpt_path='best')

if __name__ == '__main__':

    # --------------------------------
    su.fix_randomness()
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('medium')
    
    # is cuda available
    print(f"{'='*50} CUDA available: {torch.cuda.is_available()} {'='*50}")
    # get device specs
    print(f"{'='*3}> Device specs: {torch.cuda.get_device_properties(0)}")

    main_parser = su.main_arg_parser().parse_args()
    
    model_name = 'scenenet'
    dataset_name = main_parser.dataset
    project_name = f"SceneNet_Multiclass_{dataset_name}"

    config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = get_experiment_path(model_name, dataset_name)

    os.environ["WANDB_DIR"] = os.path.abspath(os.path.join(experiment_path, 'wandb'))

    # raw_config = yaml.safe_load(open(config_path))
    # pprint(raw_config)

    print(f"\n\n{'='*50}")
    print("Entering main method...") 


    if main_parser.wandb_sweep: 
        #sweep mode
        print("wandb sweep.")
        wandb.init(project=project_name, 
                dir = experiment_path,
                name = f'{project_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        )
    else:
        # default mode
        sweep_config = os.path.join(experiment_path, 'defaults_config.yml')

        print("wandb init.")

        wandb.init(project=project_name, 
                dir = experiment_path,
                name = f'{project_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                config=sweep_config,
                mode=main_parser.wandb_mode,
        )

        #pprint(wandb.config)

    main()
    

    




