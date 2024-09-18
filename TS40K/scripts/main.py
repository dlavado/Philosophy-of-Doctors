
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

from utils.constants import *
import utils.my_utils as su
import utils.pointcloud_processing as eda


import core.lit_modules.lit_callbacks as lit_callbacks
from core.lit_modules.lit_scenenet import LitSceneNet_multiclass
from core.lit_modules.lit_ts40k import LitTS40K_FULL, LitTS40K_FULL_Preprocessed
from core.lit_modules.lit_labelec import LitLabelec
from core.criterions.geneo_loss import GENEO_Loss
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

def init_scenenet(criterion):

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

    model = LitSceneNet_multiclass(geneo_num=geneo_config,
                                    num_observers=ast.literal_eval(wandb.config.num_observers),
                                    kernel_size=ast.literal_eval(wandb.config.kernel_size),
                                    hidden_dims=hidden_dims,
                                    num_classes=num_classes,
                                    ignore_index=wandb.config.ignore_index,
                                    criterion=criterion,
                                    optimizer=wandb.config.optimizer,
                                    learning_rate=wandb.config.learning_rate,
                                    metric_initializer=su.init_metrics,
                                )
        
    return model


def init_GENEO_loss(model, base_criterion=None):
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


    if base_criterion is None:
        criterion_class = su.resolve_criterion(wandb.config.criterion)
        base_criterion = criterion_class(criterion, **criterion_params)
    
    if wandb.config.geneo_criterion:
        criterion = GENEO_Loss(base_criterion, 
                                model.get_geneo_params(),
                                model.get_cvx_coefficients(),
                                convex_weight=wandb.config.convex_weight,
                            )  
    else:
        criterion = base_criterion

    model.criterion = criterion # assign criterion to model
    

def init_pointnet(model_name='pointnet'):
    from core.lit_modules.lit_pointnet import LitPointNet

    # Model definition
    model = LitPointNet(model=model_name,
                        criterion=None, # criterion is defined in the model
                        optimizer_name=wandb.config.optimizer,
                        num_classes=wandb.config.num_classes,
                        num_channels=wandb.config.num_data_channels,
                        learning_rate=wandb.config.learning_rate,
                        metric_initializer=su.init_metrics,
                    )
    return model


def init_kpconv(criterion):
    from core.lit_modules.lit_kpconv import LitKPConv

    # Model definition
    model = LitKPConv(criterion=criterion,
                      optimizer_name=wandb.config.optimizer,
                      num_stages=wandb.config.num_stages,
                      voxel_size=wandb.config.voxel_size,
                      kpconv_radius=wandb.config.kpconv_radius,
                      kpconv_sigma=wandb.config.kpconv_sigma,
                      neighbor_limits=ast.literal_eval(wandb.config.neighbor_limits),
                      init_dim=wandb.config.init_dim,
                      num_classes=wandb.config.num_classes,
                      input_dim=wandb.config.num_data_channels,
                      learning_rate=wandb.config.learning_rate,
                      metric_initializer=su.init_metrics,
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

def init_point_transformer(criterion, model_version='v3'):
    from core.lit_modules.lit_point_transformer import Lit_PointTransformer
   

    # Model definition
    model = Lit_PointTransformer(criterion=criterion,
                                   in_channels=wandb.config.num_data_channels,
                                   num_classes=wandb.config.num_classes,
                                   version=model_version,
                                   optimizer_name=wandb.config.optimizer,
                                   learning_rate=wandb.config.learning_rate,
                                   metric_initializer=su.init_metrics,
                            )

    return model


def init_ensemble_model(criterion):
    from core.lit_modules.lit_ensemble import Lit_EnsembleModel
    model_names:list[str] = ast.literal_eval(wandb.config.ensemble_models) # list[str]
    model_names = [model_name.strip().lower() for model_name in model_names]

    models = []

    pretrained_dir = wandb.config.pretrained_model_dir
    pretrained_dir = replace_variables(pretrained_dir)

    for model_name in model_names:
        pretrained_path = None

        if model_name == 'kpconv':
            model = init_kpconv(criterion)
        elif 'pointnet' in model_name:
            ptnet_name = model_name.replace('++', '2')
            model = init_pointnet(ptnet_name)
        elif 'ptv' in model_name:
            model_version = model_name.split('pt')[-1]
            model = init_point_transformer(criterion, model_version)

        for ckpt_file in os.listdir(pretrained_dir):
            if model_name in ckpt_file.lower() and ckpt_file.endswith('.ckpt'):
                pretrained_path = os.path.join(pretrained_dir, ckpt_file)
        
        if pretrained_path is None:
            raise FileNotFoundError(f"Pretrained model for {model_name} not found.")
        else:
            model = resume_from_checkpoint(pretrained_path, model, criterion.weight)
        
        models.append(model)

    model = Lit_EnsembleModel(models=models,
                              criterion=criterion,
                              num_classes=wandb.config.num_classes,
                              use_small_net=wandb.config.use_small_net,
                              full_train=wandb.config.full_train,
                              optimizer_name=wandb.config.optimizer,
                              learning_rate=wandb.config.learning_rate,
                              ignore_index=wandb.config.ignore_index,
                              metric_initializer=su.init_metrics,
                            )
    
    return model   

#####################################################################
# INIT DATASETS
#####################################################################
# fd654c61852c40948c264d606c81f59a9dddcc67

def init_ts40k(data_path, preprocessed=False):

    sample_types = 'all'
    
    if preprocessed:
        if 'scenenet' in wandb.config.model or wandb.config.model == 'cnn':
            vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
            vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1
            transform = Compose([
                            tt.Voxelization_withPCD(keep_labels='all', vxg_size=vxg_size, vox_size=vox_size)
                        ])
        else:
            transform = Compose([
                    tt.Merge_Label({eda.LOW_VEGETATION: eda.MEDIUM_VEGETAION}),
            ])

        if wandb.config.add_normals:
            transform.transforms.append(tt.Add_Normal_Vector())

        return LitTS40K_FULL_Preprocessed(
                        data_path,
                        wandb.config.batch_size,
                        sample_types=sample_types,
                        transform=transform,
                        transform_test=transform,
                        num_workers=wandb.config.num_workers,
                        val_split=wandb.config.val_split,
                        load_into_memory=wandb.config.load_into_memory,
                        use_full_test_set=True
                    )

    if wandb.config.model == 'scenenet' or wandb.config.model == 'unet':
        vxg_size = ast.literal_eval(wandb.config.voxel_grid_size) # default is 64^3
        vox_size = ast.literal_eval(wandb.config.voxel_size) # only use vox_size after training or with batch_size = 1

        voxel_method = tt.Voxelization_withPCD if wandb.config.model == 'scenenet' else tt.Voxelization

        composed = Compose([
                            tt.Farthest_Point_Sampling(wandb.config.fps_points),
                            voxel_method(keep_labels='all', vxg_size=vxg_size, vox_size=vox_size)
                        ])
    else:
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


def init_labelec(data_path):

    transform = Compose([
        tt.EDP_Labels(),
        tt.Merge_Label({eda.LOW_VEGETATION: eda.MEDIUM_VEGETAION}),
        tt.Normalize_PCD([0, 10]),
        # tt.Add_Normal_Vector(),
    ])

    if wandb.config.add_normals:
        transform.transforms.append(tt.Add_Normal_Vector())


    data_module = LitLabelec(
        data_path,
        save_chunks=False,
        transform=transform,
        test_transform=transform,
        load_into_memory=wandb.config.load_into_memory,
        batch_size=wandb.config.batch_size,
        val_split=wandb.config.val_split,
        num_workers=wandb.config.num_workers,
    )

    return data_module

 

#####################################################################
# INIT MODELS
#####################################################################

def init_model(model_name, criterion) -> pl.LightningModule:
    if model_name == 'scenenet':
        # test_MulticlassJaccardIndex: tensor([0.0000, 0.6459, 0.3951, 0.3087, 0.0633, 0.7802], device='cuda:0'); mean: 0.4387
        return init_scenenet(criterion)
    elif 'pointnet' in model_name:
        return init_pointnet(model_name)
    elif 'kpconv' in model_name:
        return init_kpconv(criterion)
    elif 'randlanet' in model_name:
        return init_randlanet(criterion)
    elif 'pt_transformer' in model_name:
        return init_point_transformer(criterion, wandb.config.model_version)
    elif 'ensemble' in model_name:
        return init_ensemble_model(criterion)
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
    
    print("Loss function: ", wandb.config.criterion)
    print(f"{'='*5}> Class weights: {class_weights}")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=wandb.config.ignore_index,
                                          weight=class_weights) # default criterion; idx zero is noise
    return criterion



def main():
    # ------------------------
    # 0 INIT CALLBACKS
    # ------------------------

    if not wandb.config.resume_from_checkpoint:
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
        alpha, epsilon = 3, 0.1
        class_densities = torch.tensor([0.0702, 0.3287, 0.4226, 0.1495, 0.0046, 0.0244], dtype=torch.float32)
        class_weights = torch.max(1 - alpha*class_densities, torch.full_like(class_densities, epsilon))
        # class_weights = 1 / class_densities
        class_weights[0] = 0.0 # ignore noise class
        # class_weights = class_weights / class_weights.mean()
    else:
        class_weights = None

    criterion = init_criterion(class_weights)

    # ------------------------
    # 2 INIT MODEL
    # ------------------------

    model = init_model(wandb.config.model, criterion)
    # torchinfo.summary(model, input_size=(wandb.config.batch_size, 1, 64, 64, 64))
    
    if wandb.config.resume_from_checkpoint:
        ckpt_path = replace_variables(ckpt_path)
        model = resume_from_checkpoint(ckpt_path, model, class_weights)
    

    if wandb.config.get('geneo_criterion', False):
        init_GENEO_loss(model, base_criterion=criterion)

    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------

    dataset_name = wandb.config.dataset       
    if dataset_name == 'ts40k':
        data_path = TS40K_FULL_PATH
        if wandb.config.preprocessed:
            data_path = TS40K_FULL_PREPROCESSED_PATH
            if idis_mode:
                data_path = TS40K_FULL_PREPROCESSED_IDIS_PATH
            elif smote_mode:
                data_path = TS40K_FULL_PREPROCESSED_SMOTE_PATH
        data_module = init_ts40k(data_path, wandb.config.preprocessed)
    elif dataset_name == 'labelec':
        data_path = LABELEC_RGB_DIR
        data_module = init_labelec(data_path)
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
        max_epochs=wandb.config.max_epochs,
        accelerator=wandb.config.accelerator,
        devices=wandb.config.devices,
        num_nodes=wandb.config.num_nodes,
        strategy=wandb.config.strategy,
        profiler=wandb.config.profiler if wandb.config.profiler else None,
        precision=wandb.config.precision,
        enable_model_summary=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accumulate_grad_batches = wandb.config.accumulate_grad_batches,
    )

    if not prediction_mode:
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
                                ckpt_path='best' if not prediction_mode else None,
                            )
    
    # test_preds = model.test_preds

    # test_preds = torch.cat(test_preds, dim=0)
    # torch.save(test_preds, os.path.join(ckpt_dir, 'test_preds.pt'))
    input("Press Enter to continue...")
    from tqdm import tqdm
    batch_size = wandb.config.batch_size
    model_name = wandb.config.model
    if model_name == 'pt_transformer':
        model_name = f"pt_transformer_{wandb.config.model_version}"

    if torch.no_grad():
        model = model.to(device)
        for stage in ['fit', 'test']:
            data_module.setup(stage=stage)

            dataset = data_module.fit_ds if stage == 'fit' else data_module.test_ds
            dataloader = data_module._fit_dataloader() if stage == 'fit' else data_module.test_dataloader()

            for i, batch in tqdm(enumerate(dataloader), desc=f"Predicting {stage} data..."):
                x, _ = batch
                batch = (batch[0].to(device), batch[1].to(device))
                preds = model.evaluate(batch, stage=stage, metric=None, prog_bar=False, logger=False)[1]

                preds = preds.reshape(x.shape[0], x.shape[1])

                assert x.shape[:2] == preds.shape
                # get dataset index
                for j in range(x.shape[0]):
                    idx = batch_size * i + j

                    dataset_x, _ = dataset[idx]
                    assert dataset_x.squeeze().shape == x[j].squeeze().shape

                    file_path = dataset._get_file_path(idx)
                    file_name =     (file_path) 
                    preds_file_path = file_path.replace('TS40K-FULL', 'TS40K-FULL-Preds')
                    preds_file_path = preds_file_path.replace(file_name, f'{file_name.split(".")[0]}/{model_name}.pt')

                    if torch.randint(0, 10, (1,)).item() >= 9:
                        print(f"{preds[j].shape}")
                        print(f"Saving predictions to {preds_file_path}")

                    os.makedirs(os.path.dirname(preds_file_path), exist_ok=True)
                    torch.save(preds[j], preds_file_path)


        


if __name__ == '__main__':
    # --------------------------------
    su.fix_randomness()
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
    # --------------------------------


    # is cuda available
    print(f"{'='*50} CUDA available: {torch.cuda.is_available()} {'='*50}")
    # get device specs
    print(f"{'='*3}> Device specs: {torch.cuda.get_device_properties(0)}")

    main_parser = su.main_arg_parser().parse_args()
    
    model_name = main_parser.model
    dataset_name = main_parser.dataset
    project_name = f"TS40K_SoA"

    prediction_mode = main_parser.predict
    idis_mode = main_parser.idis
    smote_mode = main_parser.smote

    # config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = get_experiment_path(model_name, dataset_name)

    os.environ["WANDB_DIR"] = os.path.abspath(os.path.join(experiment_path, 'wandb'))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # raw_config = yaml.safe_load(open(config_path))
    # pprint(raw_config)

    print(f"\n\n{'='*50}")
    print("Entering main method...") 

    if main_parser.wandb_sweep: 
        #sweep mode
        print("wandb sweep.")
        wandb.init(project=project_name, 
                dir = experiment_path,
                name = f'{project_name}_{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        )
    else:
        # default mode
        sweep_config = os.path.join(experiment_path, 'defaults_config.yml')

        print("wandb init.")

        wandb.init(project=project_name, 
                dir = experiment_path,
                name = f'{model_name}_{dataset_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                config=sweep_config,
                mode=main_parser.wandb_mode,
        )  

    if wandb.config.add_normals:
        wandb.config.update({'num_data_channels': wandb.config.num_data_channels + 3}, allow_val_change=True) # override data path
      
    # print(f"wandb.config.num_data_channels: {wandb.config.num_data_channels}")
    main()
    

    




