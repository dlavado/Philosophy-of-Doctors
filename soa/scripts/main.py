
from datetime import datetime
from math import ceil
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
from pytorch_lightning.callbacks import BatchSizeFinder

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger


# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

import utils.constants as C
import utils.my_utils as su
import utils.pointcloud_processing as eda

import core.lit_modules.lit_callbacks as lit_callbacks
from core.lit_modules.lit_ts40k import LitTS40K_FULL, LitTS40K_FULL_Preprocessed
import core.datasets.torch_transforms as tt

#####################################################################
# PARSER
#####################################################################

def replace_variables(string):
    """
    Replace variables marked with '$' in a string with their corresponding values from the local scope.

    Args:
    - string: Input string containing variables marked with '$'

    Returns:
    - Updated string with replaced variables
    """
    import re

    pattern = r'\${(\w+)}'
    matches = re.finditer(pattern, string)

    for match in matches:
        variable = match.group(1)
        value = locals().get(variable)
        if value is None:
            value = globals().get(variable)

        if value is not None:
            string = string.replace(match.group(), str(value))
        else:
            raise ValueError(f"Variable '{variable}' not found.")

    return string


#####################################################################
# INIT CALLBACKS
#####################################################################

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
                every_n_epochs=1, #wandb.config.checkpoint_every_n_epochs,
                every_n_train_steps=0, #wandb.config.checkpoint_every_n_steps,
                verbose=True,
            )
        )

    model_ckpts.append( # train loss checkpoint
        lit_callbacks.callback_model_checkpoint(
            dirpath=ckpt_dir, #None for default logger dir
            filename=f"val_loss",
            monitor=f"val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1, # wandb.config.checkpoint_every_n_epochs,
            every_n_train_steps=0, # wandb.config.checkpoint_every_n_steps,
            verbose=True,
        )
    )

    callbacks.extend(model_ckpts)

    if wandb.config.auto_scale_batch_size:
        batch_finder = BatchSizeFinder(mode='power')
        callbacks.append(batch_finder)

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


def init_pointnet(model_name='pointnet', criterion=None):
    from core.lit_modules.lit_pointnet import LitPointNet
    
    if 'gibli' in model_name:
        gibli_params = {   'gib_dict': wandb.config.gib_dict,
                            'num_observers': wandb.config.num_observers,
                            'kernel_reach': wandb.config.kernel_reach,
                            'neighbor_size': wandb.config.neighbor_size,
                    }
        
    else:
        gibli_params = {} 

    # Model definition
    model = LitPointNet(model=model_name,
                        criterion=criterion, # criterion is defined in the model
                        optimizer_name=wandb.config.optimizer,
                        num_classes=wandb.config.num_classes,
                        num_channels=wandb.config.num_data_channels,
                        learning_rate=wandb.config.learning_rate,
                        ignore_index=wandb.config.ignore_index,
                        metric_initializer=su.init_metrics,
                        gibli_params=gibli_params
                    )
    return model


def init_kpconv(criterion, model_version='kpconv'):
    from core.lit_modules.lit_kpconv import LitKPConv
    
    if 'gibli' in model_version:
         gibli_params = {   
                        'gib_dict': wandb.config.gib_dict,
                        'num_observers': wandb.config.num_observers,
                        'kernel_reach': wandb.config.kernel_reach,
                        'neighbor_size': wandb.config.neighbor_size,
                    }
    else:
        gibli_params = {} 

    # Model definition
    model = LitKPConv(criterion=criterion,
                      optimizer_name=wandb.config.optimizer,
                      model_version=model_version,
                      num_stages=wandb.config.num_stages,
                      voxel_size=wandb.config.kpconv_voxel_size,
                      kpconv_radius=wandb.config.kpconv_radius,
                      kpconv_sigma=wandb.config.kpconv_sigma,
                      neighbor_limits=ast.literal_eval(wandb.config.neighbor_limits),
                      init_dim=wandb.config.init_dim,
                      num_classes=wandb.config.num_classes,
                      input_dim=wandb.config.num_data_channels,
                      learning_rate=wandb.config.learning_rate,
                      ignore_index=wandb.config.ignore_index,
                      metric_initializer=su.init_metrics,
                      gibli_params=gibli_params
                    )
    return model


def init_randlanet(criterion):
    from core.lit_modules.lit_randlanet import LitRandLANet

    # Model definition
    model = LitRandLANet(criterion=criterion,
                         optimizer_name=wandb.config.optimizer,
                         in_channels=wandb.config.num_data_channels,
                         num_classes=wandb.config.num_classes,
                         num_neighbors=wandb.config.num_neighbors,
                         decimation=wandb.config.decimation,
                         learning_rate=wandb.config.learning_rate,
                         
                         metric_initializer=su.init_metrics,
    )

    return model

def init_point_transformer(criterion, model_version):
    from core.lit_modules.lit_point_transformer import Lit_PointTransformer
    
    if 'gibli' in model_version:
        gibli_params = {   'gib_dict': wandb.config.gib_dict,
                            'num_observers': wandb.config.num_observers,
                            'kernel_reach': wandb.config.kernel_reach,
                            'neighbor_size': wandb.config.neighbor_size,
                    }
    else:
        gibli_params = {}

    # Model definition
    model = Lit_PointTransformer(criterion=criterion,
                                 in_channels=wandb.config.num_data_channels,
                                 num_classes=wandb.config.num_classes,
                                 version=model_version,
                                 optimizer_name=wandb.config.optimizer,
                                 learning_rate=wandb.config.learning_rate,
                                 ignore_index=wandb.config.ignore_index,
                                 metric_initializer=su.init_metrics,
                                 #### gib parameters
                                 gib_params=gibli_params
                            )

    return model


def init_gibli_sota(sota_model_name:str, criterion):
    from core.lit_modules.lit_gibli_sota import Lit_GIBLiSOTA
    
    # GIBLiLayer parameters
    # len(num_observers) == len(neighbor_size) and it defines the number of neighborhoods that each GIBLiLayer considers
    sota_kwargs = {
        'gib_dict': wandb.config.gib_dict,
        'num_observers': wandb.config.num_observers, # int or List[int]
        'kernel_reach': wandb.config.kernel_reach,
        'neighbor_size': wandb.config.neighbor_size, # int or List[int]
        'feat_enc_channels': 16, # num channels to encode the features of query points
    }
    
    # SOTA parameters
    if sota_model_name == 'ptv1':
        sota_kwargs.update({
            "shared_channels": 1,
            "num_neighbors": wandb.config.sota_neighbors,
        })
    elif sota_model_name == 'ptv2':
        sota_kwargs.update({
            "depth": wandb.config.depth,
            "neighbours": wandb.config.sota_neighbors
        })
    # no modification to `ptv3`
    elif sota_model_name == 'kpconv':
        sota_kwargs.update({
            'kernel_size': 3 if wandb.config.sota_kernel_size < 1 else int(wandb.config.sota_kernel_size), # this must be an int
            'radius': wandb.config.kernel_reach,
            'dimension': wandb.config.num_data_channels,
            'kpconv_neighbors': wandb.config.sota_neighbors,
        })
    # no modification to `pointnet`
    elif sota_model_name == 'pointnet2':
        sota_kwargs.update({
            'radius': wandb.config.kernel_reach,
            'nsample': wandb.config.sota_neighbors,
            'mlp': [64, 64],
        })
            
    return Lit_GIBLiSOTA(in_channels=wandb.config.num_data_channels,
                        num_classes=wandb.config.num_classes,
                        model_name=sota_model_name,
                        num_levels=wandb.config.num_levels,
                        grid_size=wandb.config.grid_size,
                        embed_channels=wandb.config.embed_channels,
                        out_channels=wandb.config.out_channels,
                        depth=wandb.config.depth,
                        sota_kwargs=sota_kwargs,
                        sota_update_kwargs={},
                        criterion=criterion,
                        optimizer_name=wandb.config.optimizer,
                        ignore_index=wandb.config.ignore_index,
                        learning_rate=wandb.config.learning_rate,
                        metric_initializer=su.init_metrics,
                    )
    
#####################################################################
# INIT DATASETS
#####################################################################
# fd654c61852c40948c264d606c81f59a9dddcc67

def init_ts40k(data_path, preprocessed=False):

    sample_types = 'all'
    
    if preprocessed:
        transform = []

        if wandb.config.add_normals:
            transform.append(tt.Add_Normal_Vector())

        transform = Compose(transform)
    
        return LitTS40K_FULL_Preprocessed(
                        data_path,
                        wandb.config.batch_size,
                        sample_types=sample_types,
                        transform=transform,
                        transform_test=transform,
                        num_workers=wandb.config.num_workers,
                        val_split=wandb.config.val_split,
                        load_into_memory=wandb.config.load_into_memory,
                        use_full_test_set=False
                    )

    composed = Compose([
                        tt.Normalize_PCD(),
                        tt.Farthest_Point_Sampling(wandb.config.fps_points),
                        tt.To(torch.float32),
                    ])
    
    data_module = LitTS40K_FULL(
                           data_path,
                           wandb.config.batch_size,
                           sample_types=sample_types,
                           task='sem_seg',
                           transform=composed,
                           transform_test=None,
                           num_workers=wandb.config.num_workers,
                           val_split=wandb.config.val_split,
                           load_into_memory=wandb.config.load_into_memory,
                        )
    return data_module


def init_semantickitti(data_path):
    from core.lit_modules.lit_semkitti import LitSemanticKITTI

    data_module = LitSemanticKITTI(
                        data_path,
                        batch_size=wandb.config.batch_size,
                        num_workers=wandb.config.num_workers,
                        ignore_index=wandb.config.ignore_index,
                    )
    return data_module


def init_nuscenes(data_path):
    from core.lit_modules.lit_nuscenes import LitNuScenes

    data_module = LitNuScenes(
                        data_path,
                        batch_size=wandb.config.batch_size,
                        num_workers=wandb.config.num_workers,
                        ignore_index=wandb.config.ignore_index,
                    )
    return data_module


def init_scannet(data_path):
    from core.lit_modules.lit_scannet import LitScanNet

    data_module = LitScanNet(
                        data_path,
                        batch_size=wandb.config.batch_size,
                        num_workers=wandb.config.num_workers,
                        ignore_index=wandb.config.ignore_index,
                    )
    return data_module


def init_s3dis(data_path):
    from core.lit_modules.lit_s3dis import LitS3DIS

    data_module = LitS3DIS(
                        data_path,
                        batch_size=wandb.config.batch_size,
                        num_workers=wandb.config.num_workers,
                        ignore_index=wandb.config.ignore_index,
                    )
    return data_module

def init_waymo(data_path):
    from core.lit_modules.lit_waymo import LitWaymo

    data_module = LitWaymo(
                        data_path,
                        batch_size=wandb.config.batch_size,
                        num_workers=wandb.config.num_workers,
                        ignore_index=wandb.config.ignore_index,
                    )
    return data_module

#####################################################################
# INIT MODELS
#####################################################################

def init_model(model_name, criterion) -> pl.LightningModule:
    if 'gibli_sota' in model_name:
        return init_gibli_sota(wandb.config.model_hash, criterion)
    elif 'pointnet' in model_name:
        return init_pointnet(model_name, criterion)
    elif 'kpconv' in model_name:
        return init_kpconv(criterion, model_name)
    elif 'randlanet' in model_name:
        return init_randlanet(criterion)
    elif 'pt_transformer' in model_name:
        return init_point_transformer(criterion, wandb.config.model_version)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    

def resume_from_checkpoint(ckpt_path, model:pl.LightningModule, class_weights=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
    
    checkpoint = torch.load(ckpt_path)
    print(f"Loading model from checkpoint {ckpt_path}...\n\n")
    # if wandb.config.class_weights:
    #     checkpoint['state_dict']['criterion.weight'] = class_weights
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model loaded from checkpoint {ckpt_path}")
    
    current_epoch = checkpoint['epoch']
    
    
    # print(f"Resuming from checkpoint {ckpt_path}")
    # model = model_class.load_from_checkpoint(ckpt_path,
    #                                    criterion=criterion,
    #                                    optimizer=wandb.config.optimizer,
    #                                    learning_rate=wandb.config.learning_rate,
    #                                    metric_initilizer=su.init_metrics
    #                                 )
    return model, current_epoch


def init_criterion(class_weights=None):
    from core.criterions.seg_losses import SegLossWrapper
    from core.criterions.joint_loss import JointLoss
    
    print("Loss function: ", wandb.config.criterion)
    print(f"{'='*5}> Class weights: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=wandb.config.ignore_index,
                                          weight=class_weights,
                                          label_smoothing=0.1,
                                        )
    
    
    seg_losses = wandb.config.segmentation_losses
    
    if seg_losses is None:
        return criterion

    for loss_name in seg_losses:
        loss = SegLossWrapper(loss_name, ignore_index=wandb.config.ignore_index)
        seg_losses[loss_name] = (loss, seg_losses[loss_name]) # store the loss function and its weight in a tuple
    
    seg_losses['cross_entropy'] = (criterion, 1.0) # add cross entropy loss to the list of losses
    
    print(f"{'='*5}> Segmentation losses:\n {seg_losses}")
    
    criterion = JointLoss(*[loss[0] for loss in seg_losses.values()], weights=[loss[1] for loss in seg_losses.values()])
    
    return criterion

def main():
    # ------------------------
    # 0 INIT CALLBACKS
    # ------------------------

    # if wandb is disbabled, use local directory
    if main_parser.wandb_mode != 'disabled' and not wandb.config.resume_from_checkpoint:
        # if wandb is enabled, use wandb directory
        ckpt_dir = os.path.join(wandb.run.dir, "checkpoints") 
    else:
        ckpt_dir = wandb.config.checkpoint_dir
        
        if ckpt_dir == 'latest': # get latest experiment dir
            ckpt_dir = os.path.join(experiment_path, 'wandb', 'latest-run', 'files', 'checkpoints')
            
    
    ckpt_dir = replace_variables(ckpt_dir)
    callbacks = init_callbacks(ckpt_dir)  
    
    # --- Resume Experiment ---
    resume_ckpt_path = None 
    if main_parser.resumable:
        # find experiment to resume
        wandb_runs_path = os.path.join(experiment_path, "wandb")
        existing_exps = [
            os.path.join(wandb_runs_path, exp) 
            for exp in os.listdir(wandb_runs_path) 
                if os.path.isdir(os.path.join(wandb_runs_path, exp)) and  # Ensure it's a run directory
                   os.path.isdir(os.path.join(wandb_runs_path, exp, 'files', 'checkpoints')) and  # Checkpoints directory exists
                   os.path.isfile(os.path.join(wandb_runs_path, exp, 'files', 'config.yaml'))  # Config file exists
        ]
        
        for exp in sorted(existing_exps, key=os.path.getmtime, reverse=True):
            cfg_path = os.path.join(exp, 'files', 'config.yaml')
            pre_cfg = yaml.safe_load(open(cfg_path))
            exp_hash = pre_cfg['model_hash']['value']
            if exp_hash == wandb.config.model_hash:
                print(f"{'='*5}> Resuming from experiment: {exp}")
                resume_ckpt_path = os.path.join(os.path.join(exp, 'files', 'checkpoints'), 'last.ckpt')
                break
                
        if resume_ckpt_path is None: # if no experiment was found
            print(f"{'='*5}> No experiment found to resume. Starting new experiment.")
                    

    # ------------------------
    # 1 INIT BASE CRITERION
    # ------------------------
    if wandb.config.class_weights:
        alpha, epsilon = 3, 0.1
        if wandb.config.dataset == 'ts40k':
            class_densities = torch.tensor([0.0702, 0.3287, 0.4226, 0.1495, 0.0046, 0.0244], dtype=torch.float32)
        else:
            class_densities = torch.ones(wandb.config.num_classes, dtype=torch.float32)
        class_weights = torch.max(1 - alpha*class_densities, torch.full_like(class_densities, epsilon))
        # class_weights = 1 / class_densities
        if wandb.config.ignore_index > -1:
            class_weights[wandb.config.ignore_index] = 0.0 # ignore noise class
        class_weights = class_weights / class_weights.mean() # normalize with the mean to not affect the lr
    else:
        class_weights = None
        
    criterion = init_criterion(class_weights)

    # ------------------------
    # 2 INIT MODEL
    # ------------------------

    model = init_model(wandb.config.model, criterion)
    # torchinfo.summary(model, input_size=(wandb.config.batch_size, 1, 64, 64, 64))
    
    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')
    
    if resume_ckpt_path:
        model, ckpt_epoch = resume_from_checkpoint(resume_ckpt_path, model, class_weights)
    elif wandb.config.resume_from_checkpoint:
        ckpt_path = replace_variables(ckpt_path)
        model, _ = resume_from_checkpoint(ckpt_path, model, class_weights)
        ckpt_epoch = 0
    else:
        ckpt_epoch = 0

    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------

    dataset_name = wandb.config.dataset       
    if dataset_name == 'ts40k':
        data_path = C.TS40K_FULL_PREPROCESSED_PATH  if wandb.config.preprocessed else C.TS40K_FULL_PATH
        data_module = init_ts40k(data_path, wandb.config.preprocessed)
    elif dataset_name == 'semantickitti':
        data_module = init_semantickitti(C.SEMANTIC_KITTI_PATH)
    elif dataset_name == 'nuscenes':
        data_module = init_nuscenes(C.NUSCENES_PATH)
    elif dataset_name == 'scannet':
        data_module = init_scannet(C.SCANNET_PATH)
    elif dataset_name == 's3dis':
        data_module = init_s3dis(C.S3DIS_PATH)
    elif dataset_name == 'waymo':
        data_module = init_waymo(C.WAYMO_PATH)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    

    print(f"\n=== Data Module {dataset_name.upper()} initialized. ===\n")
    
    # ------------------------
    # 5 INIT TRAINER
    # ------------------------

    # WandbLogger
    wandb_logger = WandbLogger(project=f"{project_name}",
                               log_model=True, 
                               name=wandb.run.name, 
                               config=wandb.config
                            )
    
    wandb_logger.watch(model, log='all', log_freq=100, log_graph=False)
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        detect_anomaly=False,
        max_epochs=wandb.config.max_epochs - ckpt_epoch,
        accelerator=wandb.config.accelerator,
        devices='auto',#wandb.config.devices,
        num_nodes=wandb.config.num_nodes,
        strategy=wandb.config.strategy,
        profiler=wandb.config.profiler if wandb.config.profiler else None,
        precision=wandb.config.precision,
        enable_model_summary=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        # overfit_batches=0.1, # overfit on 10 batches
        accumulate_grad_batches = wandb.config.accumulate_grad_batches,
    )

    if not prediction_mode:
        trainer.fit(model, data_module)

        print(f"{'='*20} Model ckpt scores {'='*20}")

        for ckpt in trainer.callbacks:
            if isinstance(ckpt, pl_callbacks.ModelCheckpoint):
                checkpoint_path = ckpt.best_model_path
        
                if checkpoint_path: 
                    print(f"{ckpt.monitor} checkpoint : score {ckpt.best_model_score}")

                    # Initialize artifact to log checkpoint
                    artifact = wandb.Artifact(name=f"{model_name}_{ckpt.monitor}", type="model")
                    artifact.add_file(checkpoint_path, name=f"{model_name}_{ckpt.monitor}.ckpt") 
                    wandb.log_artifact(artifact)
                    print(f"Checkpoint logged to Wandb: {checkpoint_path}")
            
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
                                ckpt_path='best' if not prediction_mode else None,
                            )




if __name__ == '__main__':
    # --------------------------------
    su.fix_randomness()
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(True)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"{'='*50} CUDA available: {torch.cuda.is_available()} {'='*50}")
    print(f"{'='*3}> Device specs: {torch.cuda.get_device_properties(0)}")
    # --------------------------------


    main_parser = su.main_arg_parser().parse_args()
    
    model_name = main_parser.model
    dataset_name = main_parser.dataset
    job_id = main_parser.job_id
    project_name = f"GIBLi-SOA"

    prediction_mode = main_parser.predict

    # config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = C.get_experiment_dir(model_name, dataset_name)
    
    os.environ["WANDB_DIR"] = os.path.abspath(os.path.join(experiment_path))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    print(f"\n\n{'='*100}")
    
    run_name = f'{model_name}_{dataset_name}_{job_id}_{datetime.now().strftime("%Y/%m/%d_%H:%M:%S")}'

    if main_parser.wandb_sweep: 
        #sweep mode
        print("wandb sweep.")
        wandb.init(project=project_name, 
                dir = experiment_path,
                name = run_name,
        )
    else:
        # default mode
        if main_parser.arch is None:
            sweep_config = os.path.join(experiment_path, 'defaults_config.yml')
        else:
            sweep_config = os.path.join(experiment_path, f'{main_parser.arch}_config.yml')

        print("wandb init.")

        run = wandb.init(project=project_name, 
                dir = experiment_path,
                name = run_name,
                config=sweep_config,
                mode=main_parser.wandb_mode,
        )


    # if wandb.config.add_normals:
    #     wandb.config.update({'num_data_channels': wandb.config.num_data_channels + 3}, allow_val_change=True) # override data path

    if main_parser.wandb_mode == 'disabled':
        ckpt_dir = C.get_checkpoint_dir(model_name, dataset_name)
        wandb.config.update({'checkpoint_dir': ckpt_dir}, allow_val_change=True)
      
    # print(f"wandb.config.num_data_channels: {wandb.config.num_data_channels}")
    main()
    

    




