
from datetime import datetime
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
from torchinfo import summary


# PyTorch Lightning
import pytorch_lightning as pl
import  pytorch_lightning.callbacks as  pl_callbacks
from pytorch_lightning.callbacks import BatchSizeFinder

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler


# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

import utils.constants as C
import utils.my_utils as su
# import utils.pointcloud_processing as eda

import core.lit_modules.lit_callbacks as lit_callbacks
from core.lit_modules.lit_gibli import LitGIBLi
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
                every_n_epochs=wandb.config.checkpoint_every_n_epochs,
                every_n_train_steps=wandb.config.checkpoint_every_n_steps,
                verbose=True,
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
            verbose=True,
        )
    )

    callbacks.extend(model_ckpts)

    if wandb.config.auto_scale_batch_size:
        batch_finder = BatchSizeFinder(mode='binsearch')
        callbacks.append(batch_finder)

    # early_stop_callback = EarlyStopping(monitor=wandb.config.early_stop_metric, 
    #                                     min_delta=0.00, 
    #                                     patience=10, 
    #                                     verbose=False, 
    #                                     mode="max")

    # callbacks.append(early_stop_callback)

    return callbacks




def init_profiler(profiler:str, log_filename):
    if profiler == 'simple':
        profiler = SimpleProfiler(dirpath=os.getcwd(), filename=log_filename)       
    elif profiler == 'advanced':
        profiler = AdvancedProfiler(dirpath=os.getcwd(),filename=log_filename)     
    elif profiler == 'pytorch':
        profiler = PyTorchProfiler(dirpath=os.getcwd(), filename=log_filename)
    else:
        profiler = None
    return profiler
    
    
#####################################################################
# INIT MODELS
#####################################################################

def init_gibli(criterion, pyramid_builder=None) -> pl.LightningModule:
    
    gig_dict = {
        'cy': wandb.config.cylinder,
        'ellip': wandb.config.ellipsoid,
        'disk': wandb.config.disk,
        'cone': wandb.config.cone
    }
    
    return LitGIBLi(
        in_channels=wandb.config.in_channels,
        num_classes=wandb.config.num_classes,
        num_levels=wandb.config.num_levels,
        out_gib_channels=wandb.config.out_gib_channels,
        num_observers=wandb.config.num_observers,
        kernel_size=wandb.config.kernel_size,
        gib_dict=gig_dict,
        skip_connections=wandb.config.skip_connections,
        pyramid_builder = pyramid_builder,
        criterion=criterion,
        optimizer_name=wandb.config.optimizer,
        learning_rate=wandb.config.learning_rate,
        metric_initializer=su.init_metrics,
    )

#####################################################################
# INIT DATASETS
#####################################################################
# fd654c61852c40948c264d606c81f59a9dddcc67


def init_pyramid_builder():
    from core.models.giblinet.GIBLi_utils import BuildGraphPyramid
    
    neigh_strategy = wandb.config.neighborhood_strategy
    
    if neigh_strategy == 'knn':
        neighborhood_kwargs = {}
        neighborhood_update_kwargs = {}
    # elif neigh_strategy == 'dbscan': #TODO: fix DBSCAN
    #     neighborhood_kwargs = {'eps': wandb.config.dbscan_eps, 'min_points': wandb.config.dbscan_min_points}
    #     neighborhood_update_kwargs = {'eps': wandb.config.dbscan_eps_update, 'min_points': wandb.config.dbscan_min_points_update}
    elif neigh_strategy == 'radius_ball':
        neighborhood_kwargs = {'radius': wandb.config.radius_ball_radius}
        neighborhood_update_kwargs = {'radius': wandb.config.radius_ball_radius_update}
    else:
        raise ValueError(f"Neighborhood strategy {neigh_strategy} not supported.")
    
    if wandb.config.graph_strategy=="grid":
        voxel_size = torch.tensor(ast.literal_eval(wandb.config.voxel_size))
    else:
        voxel_size = None
    
    neighborhood_size = ast.literal_eval(wandb.config.neighborhood_size) 
    num_levels =  wandb.config.num_levels
    if isinstance(neighborhood_size, int):
        # if the neighborhood size is an integer, then increase its size by a factor of 2 for each level
        neighborhood_size = [neighborhood_size * (2**i) for i in range(num_levels)]
        
    sampling_factor = wandb.config.graph_pooling_factor
    if wandb.config.graph_strategy == 'fps':
        sampling_factor = 1 / sampling_factor

    return BuildGraphPyramid(num_layers=num_levels,
                            graph_strategy=wandb.config.graph_strategy,
                            sampling_factor=sampling_factor,
                            num_neighbors=neighborhood_size,
                            neighborhood_strategy=neigh_strategy,
                            neighborhood_kwargs=neighborhood_kwargs,
                            neighborhood_kwarg_update=neighborhood_update_kwargs,
                            voxel_size = voxel_size
                        )


def init_ts40k(data_path, preprocessed=False, pyramid_builder=None):
    sample_types = 'all'
    if preprocessed:
        
        transform = []
        # transform = [
        #         tt.Remove_Label(1), # GROUND
        #         tt.Repeat_Points(10_000), # repeat points to 10k so that we can batchfy the data
        # ]

        if wandb.config.add_normals:
            # transform.append(tt.Add_Normal_Vector())
            data_path = C.TS40K_FULL_PREPROCESSED_NORMALS_PATH

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
                        use_full_test_set=False,
                        _pyramid_builder=pyramid_builder
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

#####################################################################
# INIT MODELS
#####################################################################

def init_model(model_name, criterion, pyramid_builder=None) -> pl.LightningModule:
    if model_name == 'gibli':
        return init_gibli(criterion, pyramid_builder)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    

def resume_from_checkpoint(ckpt_path, model:pl.LightningModule, class_weights=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")
    
    checkpoint = torch.load(ckpt_path)
    # print(f"{checkpoint.keys()}")
    print(f"Loading model from checkpoint {ckpt_path}...\n\n")
    if wandb.config.class_weights and 'pointnet' not in ckpt_path.lower() and 'scenenet' not in ckpt_path.lower():
        checkpoint['state_dict']['criterion.weight'] = class_weights
        
    checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'montecarlo' not in k}
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model loaded from checkpoint {ckpt_path}")
    
    # model_class = model.__class__
    
    # print(f"Resuming from checkpoint {ckpt_path}")
    # model = model_class.load_from_checkpoint(ckpt_path,
    #                                    criterion=criterion,
    #                                    optimizer=wandb.config.optimizer,
    #                                    learning_rate=wandb.config.learning_rate,
    #                                    metric_initilizer=su.init_metrics
    #                                 )
    return model


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

    ckpt_dir = replace_variables(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')

    callbacks = init_callbacks(ckpt_dir)

    # ------------------------
    # 1 INIT BASE CRITERION
    # ------------------------
    if wandb.config.class_weights:
        alpha, epsilon = 4, 0.1
        if wandb.config.dataset == 'labelec':
            class_densities = torch.tensor([0.0541, 0.0006, 0.3098, 0.6208, 0.0061, 0.0085], dtype=torch.float32)
        elif wandb.config.dataset == 'ts40k':
            class_densities = torch.tensor([0.0702, 0.3287, 0.4226, 0.1495, 0.0046, 0.0244], dtype=torch.float32)
        class_weights = torch.max(1 - alpha*class_densities, torch.full_like(class_densities, epsilon))
        # class_weights = 1 / class_densities
        if wandb.config.ignore_index > -1:
            class_weights[wandb.config.ignore_index] = 0.0 #ignore noise class
        # class_weights = class_weights / class_weights.mean()
    else:
        class_weights = None
    
        
    criterion = init_criterion(class_weights)

    # ------------------------
    # 2 INIT MODEL
    # ------------------------
    
    ### PYRAMID BUILDER
    pyramid_builder = init_pyramid_builder()

    # val_MulticlassJaccardIndex: tensor([0.4354, 0.6744, 0.4973, 0.3322, 0.3667, 0.8061], device='cuda:0'); mean: 0.5187
    model = init_model(wandb.config.model, criterion, pyramid_builder)
    # model = torch.compile(model, mode="reduce-overhead")
    input_size = (wandb.config.batch_size, wandb.config.num_points, wandb.config.in_channels)
    print(f"{'='*30} Model initialized {'='*30}")
    summary(model, input_size=input_size)
    
    if wandb.config.resume_from_checkpoint:
        ckpt_path = replace_variables(ckpt_path)
        model = resume_from_checkpoint(ckpt_path, model, class_weights)

    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------

    dataset_name = wandb.config.dataset       
    if dataset_name == 'ts40k':
        if wandb.config.preprocessed:
            data_path = C.TS40K_FULL_PREPROCESSED_PATH
        else:
            data_path = C.TS40K_FULL_PATH
        data_module = init_ts40k(data_path, wandb.config.preprocessed, pyramid_builder=pyramid_builder)
    # elif dataset_name == 'labelec':
    #     if wandb.config.preprocessed:
    #         data_path  = C.LABELEC_RGB_PREPROCESSED
    #     else:
    #         data_path = C.LABELEC_RGB_DIR
    #     data_module = init_labelec(data_path, wandb.config.preprocessed)
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
    """
    TODO:
    DONE: 1. Put MC Points in a higher layer shared by all levels;
    2. Put angles and rotations in the GIB Layer to only call the trig funs and rotations once;
    DONE: 3. Make Precision 16 viable;
    4. Use KeOps on the GIB Components;
    """
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        detect_anomaly=False,
        max_epochs=wandb.config.max_epochs,
        accelerator=wandb.config.accelerator,
        devices='auto',#wandb.config.devices,
        num_nodes=wandb.config.num_nodes,
        strategy=wandb.config.strategy,
        profiler= init_profiler(wandb.config.profiler, 'profiler'),
        precision=wandb.config.precision,
        enable_model_summary=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        # gradient_clip_val=1.0,
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
        onnx_file_path = os.path.join(ckpt_dir, f"{run_name}.onnx")
        input_sample = next(iter(data_module.train_dataloader()))
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
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cudnn.benchmark = True # enable cudnn benchmark mode for faster training in fixed input size
    # --------------------------------


    # is cuda available
    print(f"{'='*50} CUDA available: {torch.cuda.is_available()} {'='*50}")
    # get device specs
    print(f"{'='*3}> Device specs: {torch.cuda.get_device_properties(0)}")

    main_parser = su.main_arg_parser().parse_args()
    
    model_name = main_parser.model
    dataset_name = main_parser.dataset
    project_name = "GIBLi-Net"
    prediction_mode = main_parser.predict

    # config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = C.get_experiment_dir(model_name, dataset_name)

    os.environ["WANDB_DIR"] = os.path.abspath(os.path.join(experiment_path, 'wandb'))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # raw_config = yaml.safe_load(open(config_path))
    # pprint(raw_config)

    print(f"\n\n{'='*50}")
    print("Entering main method...") 
    
    run_name = f"{project_name}_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if main_parser.wandb_sweep: 
        #sweep mode
        print("wandb sweep.")
        wandb.init(project=project_name, 
                dir = experiment_path,
                name = run_name,
        )
    else:
        # default mode
        sweep_config = os.path.join(experiment_path, 'defaults_config.yml')

        print("wandb init.")

        wandb.init(project=project_name, 
                dir = experiment_path,
                name = run_name,
                config=sweep_config,
                mode=main_parser.wandb_mode,
        )


    if wandb.config.add_normals:
        wandb.config.update({'in_channels': wandb.config.in_channels + 3}, allow_val_change=True) # override data path
        print(f"Added Normals to Dataset; in_channels: {wandb.config.in_channels}")

    if main_parser.wandb_mode == 'disabled':
        ckpt_dir = C.get_checkpoint_dir(model_name, dataset_name)
        wandb.config.update({'checkpoint_dir': ckpt_dir}, allow_val_change=True)
      
    # print(f"wandb.config.num_data_channels: {wandb.config.num_data_channels}")
    main()
    
