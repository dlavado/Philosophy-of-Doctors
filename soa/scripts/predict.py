import torch
import wandb
import sys
import pytorch_lightning as pl

# Our code
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
import utils.my_utils as su
import utils.constants as consts
from scripts import main as m
import utils.pointcloud_processing as eda


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



def predict(model:pl.LightningModule, data_module:pl.LightningDataModule):

    test_loader = data_module.test_dataloader()

    metrics = su.init_metrics(
        num_classes=wandb.config.num_classes,
        ignore_index=wandb.config.ignore_index,
    ).to(consts.device)

    for i, batch in enumerate(test_loader):
        # # batch to device
        # if i < 29:
        #     continue
        
        for key in batch.keys():
            batch[key] = batch[key].to(consts.device)

        loss, pred, y = model.evaluate(batch, stage='test', metric=metrics)

        # skip samples with less whan 100 tower points, as they are not useful for evaluation
        uq, counts = torch.unique(y, return_counts=True)
        if 4 not in uq or counts[uq == 4] < 100:
            print(torch.unique(y, return_counts=True)[1])
            continue
      
        pred = pred.reshape(y.shape) # reshape to match y

        xyz = batch['coord'][..., :3] # get xyz

        print(f"batch {i}; sample 0")
        print(f"Cross Entropy loss: {loss.item()}")
        print(f"{xyz.shape=} {pred.shape=} {y.shape=}")
        # if loss.item() > 0.5:
        #     continue

        # print metrics:
        print(f"{'='*50} METRICS {'='*50}")
        for key, value in metrics.items():
            print(f"{key}: {value.compute()}")

        jac_index = metrics['MulticlassJaccardIndex'].compute()
        mean_jac_index = jac_index.mean().item()
        print(f"Mean Jaccard Index: {mean_jac_index}")
    
        metrics.reset()

        # if loss.item() > 0.5 or loss.item() < 0.2: # visualize the interesting cases

        if xyz.ndim == 3: # batched
            xyz = xyz[0]
            y = y[0]
            pred = pred[0]
    
        xyz = xyz.squeeze().cpu().numpy()
        pynt = eda.np_to_ply(xyz)

        ### ground truth
        y = y.reshape(-1).cpu().numpy()
        eda.color_pointcloud(pynt, y, use_preset_colors=True)
        eda.visualize_ply([pynt], window_name="Ground Truth")

        ### prediction
        pred = pred.reshape(-1).cpu().numpy()
        eda.color_pointcloud(pynt, pred, use_preset_colors=True)
        eda.visualize_ply([pynt], window_name="Prediction")
    




def main():
    # ------------------------
    # 0 INIT CKPT PATH
    # ------------------------

    ckpt_dir = wandb.config.checkpoint_dir

    ckpt_path = os.path.join(ckpt_dir, wandb.config.resume_checkpoint_name + '.ckpt')
    ckpt_path = replace_variables(ckpt_path)
    print(f"Checkpoint path: {ckpt_path}")
    

    # ------------------------
    # 1 INIT BASE CRITERION
    # ------------------------
    class_densities = torch.tensor([0.0702, 0.3287, 0.4226, 0.1495, 0.0046, 0.0244], dtype=torch.float32)
    class_weights = 1 / class_densities
    class_weights[0] = 0.0 # ignore noise class
    class_weights = class_weights / class_weights.mean()
    criterion = m.init_criterion(class_weights=class_weights)

    # ------------------------
    # 2 INIT MODEL
    # ------------------------
    model = m.init_model(wandb.config.model, criterion)
    model, _ = m.resume_from_checkpoint(ckpt_path, model, class_weights)
    model = model.to(consts.device)
    model.eval()

    # ------------------------
    # 3 INIT DATA MODULE
    # ------------------------

    data_module = m.init_dataset(wandb.config.dataset)

    print(f"\n=== Data Module {wandb.config.dataset.upper()} initialized. ===\n")
    print(f"{data_module}")
    

    ####### if Pytorch Lightning Trainer is not called, setup() needs to be called manually
    data_module.setup('fit')
    data_module.setup('test')
    

    # ------------------------
    # 4 PREDICT
    # ------------------------
    predict(model, data_module)

if __name__ == "__main__":
    # --------------------------------
    import warnings
    from datetime import datetime
    import os

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
    project_name = f"TS40K_SoA"

    # config_path = get_experiment_config_path(model_name, dataset_name)
    experiment_path = consts.get_experiment_dir(model_name, dataset_name)

    os.environ["WANDB_DIR"] = os.path.abspath(os.path.join(experiment_path, 'wandb'))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # idk man

    if main_parser.arch in ['ptv1', 'ptv2', 'ptv3', 'kpconv', 'pointnet', 'pointnet2']:
        sweep_config = os.path.join(experiment_path, f'{main_parser.arch}_config.yml')
    else:
        sweep_config = os.path.join(experiment_path, 'defaults_config.yml')


    print("wandb init.")

    wandb.init(project=project_name, 
            dir = experiment_path,
            name = f'{project_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config=sweep_config,
            mode='disabled',
    ) 
    
    if wandb.config.add_normals:
        wandb.config.update({'num_data_channels': wandb.config.num_data_channels + 3}, allow_val_change=True) # override data path

    main()       
