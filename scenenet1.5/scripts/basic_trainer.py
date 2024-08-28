


import torch
from torchvision.transforms import Compose

import ast
from tqdm import tqdm

import sys
# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')


import pointcept.datasets.transform as pc_trans
from pointcept.datasets.scannet import Collect, Compress
from core.datasets.torch_transforms import Farthest_Point_Sampling


import core.lit_modules.lit_model_wrappers as lit_models
from core.lit_modules.lit_data_wrappers import LitTS40K, LitTS40K_Preprocessed
from core.datasets.partnet import CATEGORY_CLASS_MAP, LitPartNet_Preprocessed, LitPartNetDataset


from core.datasets.torch_transforms import Dict_to_Tuple, EDP_Labels, Farthest_Point_Sampling, ToTensor, Voxelization_withPCD

from pointcept.models.sparse_unet.spconv_unet_v1m1_base import SpUNetBase
from core.lit_modules.lit_CAC import Lit_Context_Aware_Classifier
from core.lit_modules.lit_sparse_unet import Lit_Sparse_UNet
from core.lit_modules.lit_PTv2 import Lit_PointTransformerV2




#####################################################################
# INIT MODELS
#####################################################################

def init_scenenet(criterion, config):
    geneo_config = {
        'cy': config["cylinder_geneo"]["value"],
        'arrow': config["arrow_geneo"]["value"],
        'neg': config["neg_sphere_geneo"]["value"],
        'disk': config["disk_geneo"]["value"],
        'cone': config["cone_geneo"]["value"],
        'ellip': config["ellipsoid_geneo"]["value"],
    }

    hidden_dims = ast.literal_eval(config["hidden_dims"]["value"])

    if dataset_name == 'partnet':
        num_classes = partnet_num_classes_resolution(config)
    else:
        num_classes = config["num_classes"]["value"]

    model = lit_models.LitSceneNet_multiclass(
        geneo_num=geneo_config,
        num_observers=config["num_observers"]["value"],
        extra_feature_dim=config["num_data_channels"]["value"],
        kernel_size=ast.literal_eval(config["kernel_size"]["value"]),
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        classifier='conv',
        num_points=config["fps_points"]["value"],
        criterion=criterion,
        optimizer=config["optimizer"]["value"],
        learning_rate=config["learning_rate"]["value"],
        metric_initializer=su.init_metrics,
    )

    return model


def init_CAC(criterion, cpkt_path, config):
    backbone = SpUNetBase(
        in_channels=config["num_data_channels"]["value"],
        num_classes=config["backbone_channels"]["value"],
        channels=ast.literal_eval(config["hidden_dims"]["value"])
    )

    if dataset_name == 'partnet':
        num_classes = partnet_num_classes_resolution(config)
    else:
        num_classes = config["num_classes"]["value"]

    model = Lit_Context_Aware_Classifier(
        backbone=backbone,
        num_classes=num_classes,
        backcbone_out_channels=config["backbone_channels"]["value"],
        criterion=criterion,
        optimizer_name=config["optimizer"]["value"],
        learning_rate=config["learning_rate"]["value"],
        metric_initializer=su.init_metrics,
    )

    return model


def init_SPCONV(criterion, ckpt_path, config):
    if dataset_name == 'partnet':
        num_classes = partnet_num_classes_resolution(config)
    else:
        num_classes = config["num_classes"]["value"]

    model = Lit_Sparse_UNet(
        criterion=criterion,
        optimizer_name=config["optimizer"]["value"],
        in_channels=config["num_data_channels"]["value"],
        num_classes=num_classes,
        channels=ast.literal_eval(config["hidden_dims"]["value"]),
        learning_rate=config["learning_rate"]["value"],
        metric_initializer=su.init_metrics,
    )

    return model

def init_PointTransformerV2(criterion, ckpt_path, config):
    if dataset_name == 'partnet':
        num_classes = partnet_num_classes_resolution(config)
    else:
        num_classes = config["num_classes"]["value"]

    model = Lit_PointTransformerV2(
        criterion=criterion,
        in_channels=config["in_channels"]["value"],
        num_classes=num_classes,
        version='v1',
        optimizer_name=config["optimizer"]["value"],
        learning_rate=config["learning_rate"]["value"],
        metric_initializer=su.init_metrics,
    )

    return model


def init_ts40k(data_path, preprocessed, config):
    if preprocessed:
        return LitTS40K_Preprocessed(data_path,
                                     config["batch_size"]["value"],
                                     config["num_workers"]["value"],
                                     config["val_split"]["value"],
                                     config["test_split"]["value"],
                                    )

    if config["model"]["value"] == 'scenenet':
        vxg_size = ast.literal_eval(config["voxel_grid_size"]["value"])  # default is 64^3
        vox_size = ast.literal_eval(config["voxel_size"]["value"])  # only use vox_size after training or with batch_size = 1

        composed = Compose([
            ToTensor(),
            Farthest_Point_Sampling(config["fps_points"]["value"]),
            Voxelization_withPCD(keep_labels='all',
                                vxg_size=vxg_size,
                                vox_size=vox_size
                            ),
            EDP_Labels(),
        ])
    else:
        composed = Compose([
            ToTensor(),
            Farthest_Point_Sampling(config["fps_points"]["value"]),
            EDP_Labels(),
        ])

    data_module = LitTS40K(data_path,
                        config["batch_size"]["value"],
                        composed,
                        config["num_workers"]["value"],
                        config["val_split"]["value"],
                        config["test_split"]["value"],
                        min_points=config["min_points"]["value"],
                        load_into_memory=config["load_into_memory"]["value"],
                    )

    return data_module



def init_partnet(data_path, preprocessed, config):
    if preprocessed:
        if config["keep_objects"]["value"] == "None" or config["keep_objects"]["value"] == "all":
            keep_objects = "all"
        else:
            keep_objects = ast.literal_eval(config["keep_objects"]["value"])
        return LitPartNet_Preprocessed(data_path,
                                       batch_size=config["batch_size"]["value"],
                                       keep_objects=keep_objects,
                                       num_workers=config["num_workers"]["value"],
                                    )

    vxg_size = ast.literal_eval(config["voxel_grid_size"]["value"])
    vox_size = ast.literal_eval(config["voxel_size"]["value"])

    if config["model"]["value"] == 'scenenet':
        composed = Compose([
            Dict_to_Tuple(omit=['category']),
            ToTensor(),
            Voxelization_withPCD(keep_labels='all',
                                vxg_size=vxg_size,
                                vox_size=vox_size,
                            ),
        ])
    else:
        composed = Compose([
            Dict_to_Tuple(omit=['category']),
            ToTensor(),
        ])

    return LitPartNetDataset(data_path,
                            config["coarse_level"]["value"],
                            config["batch_size"]["value"],
                            transform=composed,
                            keep_objects=ast.literal_eval(config["keep_objects"]["value"]),
                            num_workers=config["num_workers"]["value"],
                        )

def partnet_num_classes_resolution(config):
    if config["keep_objects"]["value"] == "None" or config["keep_objects"]["value"] == "all":
        num_classes = max(CATEGORY_CLASS_MAP.values())  # all classes
    else:
        num_classes = max([CATEGORY_CLASS_MAP[obj] for obj in ast.literal_eval(config["keep_objects"]["value"])])
    return num_classes







def validate(model: torch.nn.Module, val_dataloader, val_metrics, print_metrics):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(consts.device), labels.to(consts.device)

            loss, preds, _ = model.evaluate((inputs, labels), stage="val", metric=val_metrics, prog_bar=True, logger=True)

            if val_metrics:
                for metric_name, metric_val in val_metrics.items():
                    met = metric_val(preds.reshape(-1), labels.reshape(-1))
                    if isinstance(met, torch.Tensor):
                        met = met.mean()
    
    if print_metrics:
        print(f"{val_dataloader} metrics:")
        for metric_name, metric_val in train_metrics.items():
                print(f"\t{metric_name}: {metric_val}")
            
    model.train()  # Set the model back to training mode



def train(model:torch.nn.Module, optimizer):
    model.train()

    for epoch in tqdm(range(num_epochs), desc="training..."):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            inputs, labels = inputs.to(consts.device), labels.to(consts.device)

            optimizer.zero_grad()

            loss, preds, _ = model.evaluate((inputs, labels), stage="train", metric=train_metrics, prog_bar=True, logger=True) 

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if train_metrics:
                for metric_name, metric_val in train_metrics.items():
                    met = metric_val(preds.reshape(-1), labels.reshape(-1))
                    if isinstance(met, torch.Tensor):
                        met = met.mean()


        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}")

        if train_metrics:
            for metric_name, metric_val in train_metrics.items():
                print(f"\t{metric_name}: {metric_val}")

        
        validate(model, val_dataloader, val_metrics, True)    
        print("") # \n

    print("Training finished.")



if __name__ == '__main__':
    import scripts.constants as consts
    import utils.scripts_utils as su
    import warnings
    import yaml

    su.fix_randomness()
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('medium')
    
    # is cuda available
    print(f"{'='*50} CUDA available: {torch.cuda.is_available()} {'='*50}")
    # get device specs
    print(f"{'='*3}> Device specs: {torch.cuda.get_device_properties(0)}")

    main_parser = su.main_arg_parser().parse_args()
    
    model_name = main_parser.model.lower()
    dataset_name = main_parser.dataset.lower()
    project_name = f"SceneNet_Multiclass_{dataset_name}"


    CONFIG_PATH = consts.get_experiment_config_path(model_name, dataset_name)

    # Load configuration from a YAML file
    with open(CONFIG_PATH, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        config = dict(**config)


    # get hyperparameters
    batch_size = config['batch_size']['value']
    learning_rate = config['learning_rate']['value']
    num_epochs = config['max_epochs']['value']


    # ------------------------
    # 1 INIT BASE CRITERION
    # ------------------------
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config['ignore_index']['value']) # default criterion; idx zero is noise


    train_metrics = su.init_metrics(num_classes=config['num_classes']['value'], ignore_index=config['ignore_index']['value']).to(consts.device)
    val_metrics = su.init_metrics(num_classes=config['num_classes']['value'], ignore_index=config['ignore_index']['value']).to(consts.device)


    # ------------------------
    # 2 INIT MODEL
    # ------------------------

    # TODO: add resume training
    if model_name == 'scenenet':
        # criterion will be dynamically assigned; GENEO criterion require model parameters
        model = init_scenenet(criterion, None, config)
    elif model_name == 'cac':
        model = init_CAC(criterion, None, config)
    elif model_name == 'spconv':
        model = init_SPCONV(criterion, None, config)
    elif model_name == 'ptv2':
        model = init_PointTransformerV2(criterion, None, config)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    model = model.to(consts.device)
    print(f"\n=== Model {model_name.upper()} initialized. ===\n")    

    # ------------------------
    # RESOLVE OPTIM
    # ------------------------

    optim = su.resolve_optimizer(config['optimizer']['value'], model, learning_rate=learning_rate)
    
    # ------------------------
    # 4 INIT DATA MODULE
    # ------------------------
           
    if dataset_name == 'ts40k':
        data_path = consts.TS40K_PATH
        if config['preprocessed']['value']:
            data_path = consts.TS40K_PREPROCESSED_PATH
        data_module = init_ts40k(data_path, config['preprocessed']['value'], config)
    # elif dataset_name == 'partnet':
    #     data_path = consts.PARTNET_PATH
    #     if config.preprocessed.value:
    #         data_path = consts.PARTNET_PREPROCESSED_PATH
    #     data_module = init_partnet(data_path, config.preprocessed.value)
    # elif dataset_name == 'scannet':
    #     data_path = consts.SCANNET_PATH
    #     if config.preprocessed.value:
    #         data_path = consts.SCANNET_PREPROCESSED_PATH
    #     data_module = init_scannet(data_path, config.preprocessed.value)
    # elif dataset_name == 's3dis':
    #     data_path = consts.S3DIS_PATH
    #     if wanfig.preprocessed.value:
    #         data_path = consts.S3DIS_PREPROCESSED_PATH
    #     data_module = init_s3dis(data_path, config.preprocessed.value)
    # elif dataset_name == 'kitti':
    #     data_path = consts.KITTI_PATH
    #     if wanfig.preprocessed.value:
    #         data_path = consts.KITTI_PREPROCESSED_PATH
    #     data_module = init_kitti(data_path, config.preprocessed.value)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    
    print(f"\n=== Data Module {dataset_name.upper()} initialized. ===\n")

    data_module.setup('fit')
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()


    #------------------
    # INIT TRAINER
    #------------------


    train(model, optim)


    # testing
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    validate(model, test_dataloader, val_metrics, True)









