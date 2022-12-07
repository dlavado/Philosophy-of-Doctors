
 # %%
from pathlib import Path
import sys
from typing import Union

sys.path.insert(0, '..')
from pyntcloud import PyntCloud
import EDA.EDA_utils as eda
import numpy as np
import pandas as pd
import EDA.EDA_utils as eda
import laspy as lp
import torch
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import webcolors

ROOT_PROJECT = str(Path(os.getcwd()).parent.absolute())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#ROOT_PROJECT = "/home/didi/VSCode/lidar_thesis"

DATA_DIR = os.path.join(ROOT_PROJECT, "/dataset")

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c/255 - requested_colour[0]) ** 2
        gd = (g_c/255 - requested_colour[1]) ** 2
        bd = (b_c/255 - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def plot_voxelgrid(grid:Union[np.ndarray, torch.Tensor], color_mode='density', title='VoxelGrid', visual=False):
    """
    Plots voxel-grid.\\
    Color of the voxels depends on the values of each voxel;

    Parameters
    ----------
    `grid` - np.3Darray:
        voxel-grid with len(grid.shape) == 3

    `color_mode` - str:
        How to color the voxel-grid;
        color_mode \in ['density', 'ranges']

            `density` mode - colors the points as [-1, 0[ - blue; ~0 white; ]0, 1] - red

            `ranges` mode - selects colors for specific ranges of values according to the 'jet' colormap
    
    `title` - str:
        Title for the visualization window of the voxelgrid

    """

    if isinstance(grid, torch.Tensor):
        grid = grid.numpy()

    z, x, y = grid.nonzero()

    xyz = np.empty((len(z), 4))
    idx = 0
    for i, j, k in zip(x, y, z):
        xyz[idx] = [int(i), int(j), int(k), grid[k, i, j]]
        idx += 1

    if len(xyz) == 0:
        return
    
    uq_classes = np.unique(xyz[:, -1])
    class_colors = np.empty((len(uq_classes), 3))
    if color_mode == 'density': #colored according to 'coolwarm' scheme
        
        for i, c in enumerate(uq_classes):
        # [-1, 0[ - blue; ~0 white; ]0, 1] - red
            if c < 0:
                class_colors[i] = [1+c, 1+c, 1]
            else:
                class_colors[i] = [1, 1-c, 1-c]
    
    # meant to be used only when `grid` contains probabilistic values
    elif color_mode == 'ranges': #colored according to the ranges of values in `grid`
        import matplotlib.cm as cm
        r = 10
        lin = np.linspace(0, 1, r)
        # color for each range
        color_ranges = cm.jet(lin) # shape = (r, 4); 4 = (r, g, b, a)
        color_ranges[0] = [1, 1, 1, 1] # [0, 0.111] -> force color white 

        # idx in `color_ranges` for each `uq_classes`
        color_idxs = np.argmin(np.abs(np.expand_dims(uq_classes, -1) - lin), axis=-1) # len == len(uq_classes)

        for i, c in enumerate(uq_classes): 
            class_colors[i] = color_ranges[color_idxs[i]][:-1]

        if visual:
            print('Ranges Colors:')
            for i in range(r-1):
                print(f"[{lin[i]:.3f}, {lin[i+1]:.3f}[ : {get_colour_name(color_ranges[i][:-1])[1]}")

    else:
        ValueError(f"color_mode must be in ['coolwarm', 'ranges']; got {color_mode}")


    pcd = eda.np_to_ply(xyz[:, :-1])
    eda.color_pointcloud(pcd, xyz[:, -1], class_color=class_colors)

    eda.visualize_ply([pcd], window_name=title)


def hist_on_voxel(xyz, voxelgrid_dims =(64, 64, 64), voxel_dims=None):
    """
    Voxelizes the point cloud xyz and applies
    a histogram function on each voxel as a density function,

    Parameters
    ----------
    xyz - numpy array:
        Point cloud in (N, 3) format;

    voxegrid_dims - tuple int:
        Dimensions of the voxel grid to be applied to the point cloud

    voxel_dims - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;
    
    
    Returns
    -------
    data - ndarray with voxel_dims shape    
        voxelized data with histogram density functions
    """
    
    crop_tower_ply = eda.np_to_ply(xyz)
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=voxelgrid_dims, voxel_dims=voxel_dims, plot=False)
    grid = pynt.structures[id]
    grid_shape = grid.x_y_z

    data = np.zeros((grid_shape[-1], grid_shape[0], grid_shape[1]))

    voxs = pd.DataFrame({"z": grid.voxel_z, "x": grid.voxel_x, "y": grid.voxel_y, 
                        "points": np.ones_like(grid.voxel_x)})
    groups = voxs.groupby(["z", "x", "y"]).count()

    for i, hist in groups.iterrows():
        data[i] = hist

    _, data  = eda.normalize_xyz(data)

    return data

def centroid_hist_on_voxel(xyz:np.ndarray, voxelgrid_dims =(64, 64, 64), voxel_dims=None) -> np.ndarray:
    """
    Voxelizes the point cloud xyz and calculates the 
    centroid of each voxel along with its point density.\\

    Parameters
    ----------
    `xyz` - numpy array:
        Point cloud in (N, 3) format;

    `voxegrid_dims` - tuple int:
        Dimensions of the voxel grid to be applied to the point cloud

    `voxel_dims` - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;

    
    Returns
    -------
    `data` - np.ndarray with shape (4, `voxelgrid_dims`):\\
        channel 0 - mean x coordinate\\
        channel 1 - mean y coordinate\\
        channel 2 - mean z coordinate\\
        channel 3 - point density

    """
    
    crop_tower_ply = eda.np_to_ply(xyz)
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=voxelgrid_dims, voxel_dims=voxel_dims, plot=False)
    grid = pynt.structures[id]
    grid_shape = grid.x_y_z

    data = np.zeros((4, grid_shape[-1], grid_shape[0], grid_shape[1]), dtype=np.float)

    voxs = pd.DataFrame({"vz": grid.voxel_z, "vx": grid.voxel_x, "vy": grid.voxel_y, 
                        "x": xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2], 
                        "points": np.ones_like(grid.voxel_x, dtype=np.float)
                        })
    groups = voxs.groupby(["vz", "vx", "vy"]).agg({
                                                    'x' : 'mean',
                                                    'y' : 'mean',
                                                    'z' : 'mean',
                                                    'points' : 'count'
                                                    })

    for (z, x, y), centroid in groups.iterrows():
        data[:, z, x, y] = np.array(centroid).astype(np.float)

    data[3] = (data[3] - np.min(data[3])) / (np.max(data[3]) - np.min(data[3]))

    return data

def classes_on_voxel(xyz, labels, voxel_dims=(64, 64, 64)):
    """
    Voxelizes the point cloud xyz and computes the ground-truth
    for each voxel according to the labels of the containing points.
    \n
    Parameters
    ----------
    xyz - numpy array:
        point cloud in (N, 3) format.
    labels - 1d numpy array:
        point labels in (1, N) format.
    voxel_dims - tupple int:
        dimensions of the voxel_grid to be applied to the point cloud

    Returns
    -------
        data - 3d numpy array with shape == voxel_dims:
            voxelized point cloud.
    """

    crop_tower_ply = eda.np_to_ply(xyz)
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=voxel_dims, plot=False)
    grid = pynt.structures[id]
    grid_shape = grid.x_y_z

    data = np.zeros((grid_shape[-1], grid_shape[0], grid_shape[1]))

    voxs = pd.DataFrame({"label" : labels, 
                        "z": grid.voxel_z, "x": grid.voxel_x, "y": grid.voxel_y})

    groups = voxs.groupby(["z", "x", "y"]).max()
    for i, hist in groups.iterrows():
        data[i] = hist

    return data


def reg_on_voxel(xyz, labels, tower_label, voxelgrid_dims=(64, 64, 64), voxel_dims=None):
    """
    Voxelizes the point cloud xyz and computes a regression ground-truth
    for each voxel demonstrating the percentage of tower_points each voxel contains.
    \n
    Parameters
    ----------
    `xyz` - numpy array:
        point cloud in (N, 3) format.

    `labels` - 1d numpy array:
        point labels in (1, N) format.

    `tower_label` - int or list:
        labels that identify towers in the dataset

    `voxelgrid_dims` - tuple int:
        dimensions of the voxel_grid to be applied to the point cloud
    
    `voxel_dims` - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;

    Returns
    -------
        data - 3d numpy array with shape == voxel_dims:
            voxelized point cloud.
    """

    crop_tower_ply = eda.np_to_ply(xyz)
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=voxelgrid_dims, voxel_dims=voxel_dims, plot=False)
    grid = pynt.structures[id]
    grid_shape = grid.x_y_z

    data = np.zeros((grid_shape[-1], grid_shape[0], grid_shape[1]))

    voxs = pd.DataFrame({"label" : labels, 
                        "z": grid.voxel_z, "x": grid.voxel_x, "y": grid.voxel_y})

    # tower_points = voxs.loc[voxs["label"] == tower_label].groupby(["z", "x", "y"]).count()
    # tower_points = np.array(tower_points["label"])
    def count_towers(x):
        group = np.array(x)
        keep = np.array(tower_label).reshape(-1)
        count = np.isin(group, keep)
        return np.sum(count)

    voxs['tower'] = voxs['label'].copy()

    totals = voxs.groupby(["z", "x", "y"]).agg({'label': 'count', 'tower': count_towers})
    # print(totals)
    assert totals.shape[1] == 2 # the total_count and the tower_count

    for zxy, row in totals.iterrows():
        data[zxy] = row["tower"] / row["label"]

    return data


def centroid_reg_on_voxel(xyz:np.ndarray, labels:np.ndarray, tower_label:int, voxelgrid_dims=(64, 64, 64), voxel_dims=None) -> np.ndarray:
    """
    Voxelizes the point cloud xyz and computes a regression ground-truth
    for each voxel demonstrating the percentage of tower_points each voxel contains
    along with the gt's centroid
    \n
    Parameters
    ----------
    `xyz` - numpy array:
        point cloud in (N, 3) format.

    `labels` - 1d numpy array:
        point labels in (N,) format.

    `voxelgrid_dims` - tuple int:
        dimensions of the voxel_grid to be applied to the point cloud
    
    `voxel_dims` - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;

    Returns
    -------
    `data` - np.ndarray with shape (4, `voxelgrid_dims`):\\
        channel 0 - mean x coordinate\\
        channel 1 - mean y coordinate\\
        channel 2 - mean z coordinate\\
        channel 3 - ground truth tower point density
    """

    crop_tower_ply = eda.np_to_ply(xyz)
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=voxelgrid_dims, voxel_dims=voxel_dims, plot=False)
    grid = pynt.structures[id]
    grid_shape = grid.x_y_z

    data = np.zeros((4, grid_shape[-1], grid_shape[0], grid_shape[1]))

    voxs = pd.DataFrame({"vz": grid.voxel_z, "vx": grid.voxel_x, "vy": grid.voxel_y, 
                         "x": xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2], 
                         "label" : labels.astype(np.float),
                         "tower" : labels.astype(np.float)})

    def count_towers(x):
        group = np.array(x)
        keep = np.array(tower_label).reshape(-1)
        count = np.isin(group, keep)
        return np.sum(count)

    totals = voxs.groupby(["vz", "vx", "vy"]).agg({
                                                'x' : 'mean',
                                                'y' : 'mean',
                                                'z' : 'mean',
                                                'label': 'count', 
                                                'tower': count_towers
                                                })

    for (z, x, y), row in totals.iterrows():
        #row = np.array(row)
        data[:, z, x, y] = np.array([row['x'], row['y'], row['z'], row["tower"] / row["label"]])

    return data


def prob_to_label(voxelgrid:Union[torch.Tensor, np.ndarray], tau:float) -> Union[torch.Tensor, np.ndarray]:
    """
    Transforms `voxelgrid` from a probability values to binary label according to a `tau` threshold. 

    Parameters
    ----------
    `voxelgrid` - numpy 3Darray | torch.tensor:
        Voxelized point cloud
    
    `tau` - float \in ]0, 1[:
        classification threshold
    """

    if isinstance(voxelgrid, torch.Tensor):
        # if voxelgrid.requires_grad:
        #     voxelgrid.data = (voxelgrid >= tau).to(voxelgrid.dtype).data
        # return voxelgrid
        return (voxelgrid >= tau).to(voxelgrid.dtype)
    else:
        return (voxelgrid >= tau).astype(voxelgrid.dtype)


def vxg_to_xyz(vxg:torch.Tensor) -> np.ndarray:
    """
    Converts voxel-grid (with centroid) to a raw point cloud.\\
    The selected voxels to represent the raw point cloud have label == 1.0\n

    Parameters
    ----------
    `vxg` - torch.Tensor:
        voxel-grid to be transformed with shape (4, voxelgrid_dims)\\
        4 channels are required - (x, y, z, label)"

    Returns
    -------
    `points` - np.ndarray:
        (N, 3) numpy array that encodes the raw pcd.
    """

    if vxg.shape[0] != 4:
        ValueError(f"Voxel-grid with incorrect format; 4 channels are required - (x, y, z, label)")

    #vxg[-1] is the label 
    points = vxg[:-1, vxg[-1] == 1.0].T

    return points.cpu().detach().numpy()


def visualize_pred_vs_gt(pred:np.ndarray, gt:np.ndarray, plot=True):
    """
    Visualize a classifier's prediction against the ground truth
    """

    #vxg_diff = (pred - gt) # 1 -> FP; -1 -> FN

    if len(pred) == 0 or len(gt) == 0:
        return
   
    vxg_common = (pred*gt + 1.5*pred + 0.5*gt) / 2 # 1-> TP;  0.75 -> FP;  0.25 -> FN;  0 -> TN

    y_true, y_pred = gt.flatten(), pred.flatten()

    plot_voxelgrid(np.squeeze(vxg_common), color_mode='ranges', title='Prediction vs. Ground Truth') 

    if plot:
        print(f"Jaccard Score: {jaccard_score(y_true, y_pred)}")
        print(f"Precision: {precision_score(y_true, y_pred)}")
        print(f"Recall: {recall_score(y_true, y_pred)}")
        print(f"F1 score: {f1_score(y_true, y_pred)}")
        # if len(pred[pred > 0]) <= 1:
        #     print(f"Prediction is Empty!")
        # else:
        #     print(f"Plotting Class Prediction...")
        #     plot_voxelgrid(pred, title='GENEO-Net Prediction')
        # if len(gt[gt > 0]) <= 1:
        #     print("GT is empty!")
        # else:
        #     print(f"Plotting Ground Truth...")
        #     plot_voxelgrid(gt, title='Ground Truth')
        # plot_voxelgrid(vxg_diff, title='pred vs. gt | FP & FN')



# %%
if __name__ == "__main__":

    
    tower_files = eda.get_tower_files(['/home/didi/VSCode/lidar_thesis/Data_sample'], False)

    pcd_xyz, classes = eda.las_to_numpy(lp.read(tower_files[0]))

    pcd_tower, _ = eda.select_object(pcd_xyz, classes, [eda.POWER_LINE_SUPPORT_TOWER])
    towers = eda.extract_towers(pcd_tower, visual=False)
    crop_tower_xyz, crop_tower_classes = eda.crop_tower_radius(pcd_xyz, classes, towers[0])
  

    # %%
    crop_tower_ply = eda.np_to_ply(crop_tower_xyz)
    eda.color_pointcloud(crop_tower_ply, crop_tower_classes)
    eda.visualize_ply([crop_tower_ply])

    # %%

    downsample_xyz, downsample_classes = eda.downsampling_relative_height(crop_tower_xyz, crop_tower_classes)
    down_tower_ply = eda.np_to_ply(downsample_xyz)
    eda.color_pointcloud(down_tower_ply, downsample_classes)
    eda.visualize_ply([down_tower_ply])

    # %%

    print(f"Crop tower point size = {crop_tower_xyz.shape}\nDown sample tower point size = {downsample_xyz.shape}")
    print(f"Downsample percentage: {downsample_xyz.shape[0] / crop_tower_xyz.shape[0]}")
    # %%
    down_tower_xyz, down_classes = eda.downsampling(crop_tower_ply, crop_tower_classes, samp_per=0.8)
    down_tower_ply = eda.np_to_ply(down_tower_xyz)
    #down_tower_ply, _ = eda.select_object(down_tower_xyz, down_classes, [eda.POWER_LINE_SUPPORT_TOWER])

    eda.color_pointcloud(down_tower_ply, down_classes)
    eda.visualize_ply([down_tower_ply])

    # %%
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=(64, 64, 64), plot=False)
    grid = pynt.structures[id]

    # %%
    vox = reg_on_voxel(crop_tower_xyz, crop_tower_classes, eda.POWER_LINE_SUPPORT_TOWER, (64, 64, 64))
    plot_voxelgrid(vox, color_mode='ranges')

    # %%
    vox = hist_on_voxel(crop_tower_xyz, (64, 64, 64))
    plot_voxelgrid(vox)
    plot_voxelgrid(vox, color_mode='ranges')



    
    
# %%


