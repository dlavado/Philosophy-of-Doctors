
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

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger


# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from utils.constants import *
import utils.scripts_utils as su
import utils.pointcloud_processing as eda


from core.datasets.torch_transforms import Farthest_Point_Sampling
import core.lit_modules.lit_callbacks as lit_callbacks
import core.lit_modules.lit_model_wrappers as lit_models
from core.lit_modules.lit_ts40k import LitTS40K_FULL
from core.criterions.geneo_loss import GENEO_Loss
from core.datasets.torch_transforms import EDP_Labels, Farthest_Point_Sampling, ToTensor, Voxelization_withPCD


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


#####################################################################
# INIT MODELS
#####################################################################

def init_scenenet(criterion, ckpt_path):

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
        num_classes = wandb.config.num_classes

        model = lit_models.LitSceneNet_multiclass(geneo_num=geneo_config,
                                                num_observers=wandb.config.num_observers,
                                                extra_feature_dim=wandb.config.num_data_channels,
                                                kernel_size=ast.literal_eval(wandb.config.kernel_size),
                                                hidden_dims=hidden_dims,
                                                num_classes=num_classes,
                                                classifier='conv',
                                                num_points=wandb.config.fps_points,
                                                criterion=criterion,
                                                optimizer=wandb.config.optimizer,
                                                learning_rate=wandb.config.learning_rate,
                                                metric_initializer=su.init_metrics,
                                    )
        
    return model


#####################################################################
# INIT DATASETS
#####################################################################

def init_ts40k(data_path, preprocessed=False):

    if wandb.config.model == 'scenenet':
        vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
        vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1

        composed = Compose([
                            ToTensor(),

                            Farthest_Point_Sampling(wandb.config.fps_points),
            
                            Voxelization_withPCD(keep_labels='all', 
                                                vxg_size=vxg_size, 
                                                vox_size=vox_size
                                                ),
                            EDP_Labels(),
                        ])
        
    else:
         composed = Compose([
                            ToTensor(),

                            Farthest_Point_Sampling(wandb.config.fps_points),
            
                            EDP_Labels(),
                        ])

    

    data_module = LitTS40K_FULL(
                           data_path,
                           wandb.config.batch_size,
                           sample_types='all',
                           task='sem_seg',
                           transform=composed,
                           num_workers=wandb.config.num_workers,
                           val_split=wandb.config.val_split,
                           min_points= wandb.config.min_points,
                           load_into_memory=wandb.config.load_into_memory,
                        )
    
    return data_module



def main():
    # ------------------------
    # 0 INIT CALLBACKS
    # ------------------------

    if not wandb.config.resume_from_checkpoint:
        ckpt_dir = os.path.join(wandb.run.dir, "checkpoints") 
    else:
        ckpt_dir = wandb.config.checkpoint_dir

    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')

    callbacks = init_callbacks(ckpt_dir)


    # ------------------------
    # 1 INIT BASE CRITERION
    # ------------------------
    criterion = torch.nn.CrossEntropyLoss(ignore_index=wandb.config.ignore_index) # default criterion; idx zero is noise


    # ------------------------
    # 2 INIT MODEL
    # ------------------------

    # TODO: add resume training
    if wandb.config.model == 'scenenet':
        # criterion will be dynamically assigned; GENEO criterion require model parameters
        model = init_scenenet(None, ckpt_path)
    else:
        raise ValueError(f"Model {wandb.config.model} not supported.")


    # ------------------------
    # 3 INIT GENEO TRAINING CRITERION
    # ------------------------

    criterion_params = {}

    if 'tversky' in wandb.config.criterion.lower():
        criterion_params = {
            'tversky_alpha': wandb.config.tversky_alpha,
            'tversky_beta': wandb.config.tversky_beta,
            'tversky_smooth': wandb.config.tversky_smooth,
            'focal_gamma': wandb.config.focal_gamma,
        }

    if 'focal' in wandb.config.criterion.lower():
        criterion_params['focal_gamma'] = wandb.config.focal_gamma
    
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

    dataset_name = wandb.config.dataset
           
    if dataset_name == 'ts40k':
        data_path = TS40K_PATH
        if wandb.config.preprocessed:
            data_path = TS40K_PREPROCESSED_PATH
        data_module = init_ts40k(data_path, wandb.config.preprocessed)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    wandb.config.update({'data_path': data_path}, allow_val_change=True) # override data path

    
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
        detect_anomaly=False,
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
    
    model_name = main_parser.model
    dataset_name = main_parser.dataset
    project_name = f"{model_name}_{dataset_name}"

    # config_path = get_experiment_config_path(model_name, dataset_name)
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
    

    




