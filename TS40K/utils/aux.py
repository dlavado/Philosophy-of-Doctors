
# %%

# import torch

# # Paths to the prediction files
# TEST_PRED_FILES = {
#     'ptv1'  : '/home/didi/VSCode/Philosophy-of-Doctors/TS40K/experiments/pt_ts40k/wandb/ptv1_resume/files/checkpoints/test_preds.pt',
#     'ptv2'  : '/home/didi/VSCode/Philosophy-of-Doctors/TS40K/experiments/pt_ts40k/wandb/ptv2_resume/test_preds.pt',
#     'kpconv': '/home/didi/VSCode/Philosophy-of-Doctors/TS40K/experiments/kpconv_ts40k/checkpoints/test_preds.pt',
#     'pointnet2' : '/home/didi/VSCode/Philosophy-of-Doctors/TS40K/experiments/pointnet_ts40k/pointnet++/test_preds_p2.pt',
#     'pointnet' : '/home/didi/VSCode/Philosophy-of-Doctors/TS40K/experiments/pointnet_ts40k/pointnet/test_preds.pt'
# }

# TEST_Y_FILE = '/home/didi/VSCode/Philosophy-of-Doctors/TS40K/utils/test_y.pt'

# # Load test predictions
# test_preds = {}
# for model_name, test_pred_file in TEST_PRED_FILES.items():
#     test_preds[model_name] = torch.load(test_pred_file).unsqueeze(1)  # Shape = (num_points, 1)

# # Load the true labels
# test_y = torch.load(TEST_Y_FILE).reshape(-1)
# print(f"test y shape {test_y.shape}")

# # Print the shape of each model's predictions
# for model_name, preds in test_preds.items():
#     print(f"{model_name} preds shape = {preds.shape}")

# # Concatenate predictions along the model dimension (dim=1)
# preds = torch.concatenate([test_preds[model_name] for model_name in test_preds], dim=1)  # Shape = (num_points, num_models)

# print(f"Concatenated preds shape = {preds.shape}")

# # Calculate agreement per point
# mode_classes, counts = torch.mode(preds, dim=1)
# agreement_scores = (preds == mode_classes.unsqueeze(1)).float().mean(dim=1) # Shape = (num_points,)

# print(f"Agreement scores shape = {agreement_scores.shape}")
# print(f"mean: {agreement_scores.mean().item()}, std: {agreement_scores.std().item()}")
# print(f"min: {agreement_scores.min().item()}, max: {agreement_scores.max().item()}")

# # %%

# # Calculate agreement score for each class
# classes = torch.unique(test_y)
# class_agreement_scores = {}

# for cls in classes:
#     # Find indices where the true label is the current class
#     class_indices = test_y == cls

#     # Compute the mean agreement score for the current class
#     class_agreement_scores[cls.item()] = agreement_scores[class_indices].mean().item()

# # Output agreement scores for each class
# for cls, score in class_agreement_scores.items():
#     print(f"Class {cls}: Agreement score = {score}")


# agreement_threshold = 0.7

# # Calculate the number of points where the agreement score is above the threshold
# num_agreed_points = (agreement_scores >= agreement_threshold).sum().item()
# print(f"Number of agreed points above the threshold: {num_agreed_points}; Percentage: {num_agreed_points / len(agreement_scores) * 100:.2f}%")


# %%
import sys
sys.path.append('../')
import utils.constants as C
from core.datasets.TS40K import TS40K_FULL_Preprocessed
import os
import utils.pointcloud_processing as eda
import torch
import numpy as np

split = 'fit'
sample_type = ['2_towers']

ts40k = TS40K_FULL_Preprocessed(C.TS40K_FULL_PREPROCESSED_PATH, 
                                split=split,
                                sample_types=sample_type, 
                                load_into_memory=True)

PREDS_DIR = '/media/didi/PortableSSD/TS40K-Dataset/TS40K-FULL-Preds-Preprocessed/'

for idx in torch.randperm(len(ts40k)):
    # Get the point cloud and the true label
    pc, y_true = ts40k[idx]
    # pc = pc.to(C.device) # Shape = (num_points, 3)
    # y = y.to(C.device)   # Shape = (num_points,)

    sample_file_path = ts40k._get_file_path(idx)
    sample_name = os.path.basename(sample_file_path) 
    # print(f"Sample file name = {sample_file_name}")

    sample_type = sample_file_path.split('/')[-3]

    sample_preds_dir = os.path.join(PREDS_DIR, f"{sample_type}/{split}/{sample_name.split('.')[0]}/")
    print(f"Sample preds dir = {sample_preds_dir}")

    model_preds = None
    for model_name in os.listdir(sample_preds_dir):
        if model_name == 'pointnet2.pt':
            continue
        model_pred = torch.load(os.path.join(sample_preds_dir, model_name)).unsqueeze(1)  # Shape = (num_points, 1)

        if model_preds is None:
            model_preds = model_pred
        else:
            model_preds = torch.cat([model_preds, model_pred], dim=1)
    
    print(f"Model preds shape = {model_preds.shape}; Model preds device = {model_preds.device}")

    model_preds = model_preds.cpu()

    # Calculate agreement per point
    mode_classes, counts = torch.mode(model_preds, dim=1)

    print(f"Mode classes shape = {mode_classes.shape}")
    
    agreement_scores = (model_preds == mode_classes.unsqueeze(1)).float().mean(dim=1) # Shape = (num_points,)
    majority_class = mode_classes.squeeze().cpu() # Shape = (num_points,)

    print(f"Agreement scores shape = {agreement_scores.shape}")

    # feats = torch.cat([agreement_scores.unsqueeze(1), majority_class.unsqueeze(1)], dim=1) # Shape = (num_points, 2)

    # torch.save(feats, os.path.join(sample_preds_dir, 'agreement_scores.pt')) # Save the agreement scores


    agreement_threshold  = 0.7

    agreement_mask = agreement_scores >= agreement_threshold
    agreement_mask[y_true == 4] = True # Include the points where the true label is 4 (Tower)
    
    new_pc = pc[agreement_mask]
    y = y_true[agreement_mask]
    majority_class = majority_class[agreement_mask]

    # Replace the points where the majority class is 2, 1, or 5 with the majority class
    mask = (majority_class == 2) | (majority_class == 1) | (majority_class == 5)
    new_y = torch.where(mask, majority_class, y)
    # When the true label is Noise (class 0), keep it
    y_mask = (y == 0) | (y == 4) | (y == 3)
    new_y = torch.where(y_mask, y, new_y)

    # new_y = torch.where(majority_class == 5, majority_class, new_y)

    print(f"Agreed points shape = {new_pc.shape}")
    print(f"Avg agreement score = {agreement_scores[agreement_mask].mean().item()}")

    if agreement_scores[agreement_mask].mean().item() > 0.9:

        eda.plot_pointcloud(pc.cpu().numpy(), y_true.cpu().numpy(), window_name=f"Sample {idx} - True Labels", use_preset_colors=True)

        eda.plot_pointcloud(new_pc.cpu().numpy(), new_y.cpu().numpy(), window_name=f"Sample {idx} - Majority Class", use_preset_colors=True)
        



   
# %%
