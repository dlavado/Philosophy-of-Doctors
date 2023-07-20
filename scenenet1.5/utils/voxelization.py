
 # %%
from pathlib import Path
import sys
from typing import Tuple, Union

sys.path.insert(0, '..')
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd
import utils.pcd_processing as eda
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


def plot_voxelgrid(grid:Union[np.ndarray, torch.Tensor], color_mode='density', title='VoxelGrid', visual=False, plot=True, **kwargs):
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

    `visual` - bool:
        If True, it shows a legend for the point cloud visualization

    `plot` - bool:
        If True, it plots the voxelgrid; if False, it returns the voxelgrid colored accordingly.


    Returns
    -------
    if plot is True:
        None
    else:
        colored pointcloud in (x, y, z, r, g, b) format

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
    
    # meant to be used only when `grid` contains values \in [0, 1]
    elif color_mode == 'ranges': #colored according to the ranges of values in `grid`
        import matplotlib.cm as cm
        r = 10 if 'r' not in kwargs else kwargs['r']
        step = (1 / r) / 2
        lin = np.linspace(0, 1, r) 
        # color for each range
        color_ranges = cm.jet(lin) # shape = (r, 4); 4 = (r, g, b, a)
        color_ranges[0] = [1, 1, 1, 0] # [0, 0.111] -> force color white 

        xyz = xyz[xyz[:, -1] > lin[1]] # voxels with the color white are eliminated for a better visual
        #xyz = np.delete(xyz, np.arange(0, len(xyz))[xyz[:, -1] < lin[1]], axis=0) # voxels with the color white are eliminated for a better visual
        uq_classes = np.unique(xyz[:, -1])
    
        # idx in `color_ranges` for each `uq_classes`
        color_idxs = np.argmin(np.abs(np.expand_dims(uq_classes, -1) - lin - step), axis=-1) # len == len(uq_classes)

        class_colors = np.empty((len(uq_classes), 3))
        for i, c in enumerate(uq_classes): 
            class_colors[i] = color_ranges[color_idxs[i]][:-1]

        if visual:
            print('Ranges Colors:')
            for i in range(r-1):
                print(f"[{lin[i]:.3f}, {lin[i+1]:.3f}[ : {get_colour_name(color_ranges[i][:-1])[1]}")

    else:
        ValueError(f"color_mode must be in ['coolwarm', 'ranges']; got {color_mode}")

   
    pcd = eda.np_to_ply(xyz[:, :-1])
    xyz_colors = eda.color_pointcloud(pcd, xyz[:, -1], class_color=class_colors)

    if not plot:
        return np.concatenate((xyz[:, :-1], xyz_colors*255), axis=1) # 255 to convert color to int8

    eda.visualize_ply([pcd], window_name=title)


def plot_quantile_uncertainty(vxg:Union[np.ndarray, torch.Tensor], legend=False):

    assert len(vxg.shape) == 4, print(f"Inadmissible shape: {vxg.shape}") #vxg should have shape: Q, VXG_Z, VXG_X, VXG_Y
    assert vxg.shape[0] >= 2 # the plot requires at least two quantile predictions

    # `grid` holds the difference between quantiles to plot.
    # The higher the difference, higher the uncertainty.
    grid = vxg[-1] - vxg[0] # -1 is the 90th quantile and 0 is the 10th Quantile
    plot_voxelgrid(grid, color_mode='ranges', title='Aletoric Uncertainty through Quantiles', visual=legend) 




###############################################################
#                  Voxelization Functions                     #
###############################################################


def voxelize_sample(xyz, labels, keep_labels, voxelgrid_dims=(64, 64, 64), voxel_dims=None):
    """
    Voxelizes the point cloud xyz and applies a histogram function on each voxel as a density function.

    Parameters
    ----------
    `xyz` - numpy array:
        point cloud in (N, 3) format.

    `labels` - 1d numpy array:
        point labels in (1, N) format.

    `keep_labels` - int or list:
        labels to be kept in the voxelization process.

    `voxegrid_dims` - tuple int:
        Dimensions of the voxel grid to be applied to the point cloud

    `voxel_dims` - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;
    
    Returns
    -------
    `in` - np.ndarray with voxel_dims shape    
        voxelized data with histogram density functions
    
    `gt` - np.ndarray with voxel_dims shape
        voxelized labels with histogram density functions
    
    `point_locations` - np.ndarray withg shape (N, 3)
        point locations in the voxel grid
    """
    to_tensor = False
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.numpy()
        to_tensor = True
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    crop_tower_ply = eda.np_to_ply(xyz)
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=voxelgrid_dims, voxel_dims=voxel_dims, plot=False)
    grid = pynt.structures[id]
    grid_shape = grid.x_y_z

    inp = np.zeros((grid_shape[-1], grid_shape[0], grid_shape[1]))
    
    gt = np.copy(inp)

    voxs = pd.DataFrame(data = {
                            "z": grid.voxel_z, 
                            "x": grid.voxel_x, 
                            "y": grid.voxel_y,
                            "points": np.ones_like(grid.voxel_x), 
                            "labels": labels
                           }
                        )
    
    groups = voxs.groupby(['z', 'x', 'y'])

    point_locations = np.column_stack((grid.voxel_z, grid.voxel_x, grid.voxel_y))

    def voxel_label(x):
        group = np.array(x)
        keep = group[np.isin(group, keep_labels)]
        if len(keep) == 0:
            return 0.0
        label, count = np.unique(keep, return_counts=True)
        label = label[np.argmax(count)] # performs a mode operation
        return label

    aggs = groups.agg({'labels': voxel_label, 'points': 'count'})

    for zxy, row in aggs.iterrows():
        inp[zxy] = 1.0 if row['points'] > 0 else 0.0

        gt[zxy] = eda.DICT_NEW_LABELS[row['labels']] # convert EDP labels to semantic labels

    if to_tensor:
        inp = torch.from_numpy(inp).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        point_locations = torch.from_numpy(point_locations)
    
    return inp, gt, point_locations


def voxelize_input_pcd(xyz, labels, keep_labels=None, voxelgrid_dims=(64, 64, 64), voxel_dims=None):
    """
    Voxelizes the point cloud xyz and applies a histogram function on each voxel as a density function.

    Parameters
    ----------
    `xyz` - numpy array:
        point cloud in (N, 3) format.

    `labels` - 1d numpy array:
        point labels in (1, N) format.

    `keep_labels` - int or list:
        labels to be kept in the voxelization process.

    `voxegrid_dims` - tuple int:
        Dimensions of the voxel grid to be applied to the point cloud

    `voxel_dims` - tuple int:
        Dimensions of the voxels that compose the voxel_grid that will encase the point cloud
        if voxel_dims is not None, it overrides voxelgrid_dims;
    
    Returns
    -------
    `in` - np.ndarray with voxel_dims shape    
        voxelized data with histogram density functions
    
    `gt` - np.ndarray with shape (1, N) and semantic labels
    
    `point_locations` - np.ndarray withg shape (N, 3)
        point locations in the voxel grid
    """
    to_tensor = False
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.numpy()
        to_tensor = True
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    crop_tower_ply = eda.np_to_ply(xyz)
    pynt, id = eda.voxelize_ply(crop_tower_ply, voxelgrid_dims=voxelgrid_dims, voxel_dims=voxel_dims, plot=False)
    grid = pynt.structures[id]
    grid_shape = grid.x_y_z

    inp = np.zeros((grid_shape[-1], grid_shape[0], grid_shape[1]))

    voxs = pd.DataFrame(data = {
                            "z": grid.voxel_z, 
                            "x": grid.voxel_x, 
                            "y": grid.voxel_y,
                            "points": np.ones_like(grid.voxel_x), 
                           }
                        )
    
    groups = voxs.groupby(['z', 'x', 'y'])

    point_locations = np.column_stack((grid.voxel_z, grid.voxel_x, grid.voxel_y))

    aggs = groups.agg({'points': 'count'})

    for zxy, row in aggs.iterrows():
        inp[zxy] = 1.0 if row['points'] > 0 else 0.0

    if keep_labels is None or keep_labels == 'all':
        keep_labels = np.unique(labels)

    def change_label(x):
        return eda.DICT_NEW_LABELS[x] if x in keep_labels else 0
    
    gt = np.vectorize(change_label)(labels) # convert EDP labels to semantic labels, shape = (1, N)

    if to_tensor:
        inp = torch.from_numpy(inp).unsqueeze(0)
        gt = torch.from_numpy(gt).to(torch.long)
        point_locations = torch.from_numpy(point_locations)
    
    return inp, gt, point_locations

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
    
    groups = voxs.groupby(["z", "x", "y"])

    xyz_voxel_coord = groups.keys()

    counts = groups.count()

    for i, hist in counts.iterrows():
        data[i] = hist

    _, data  = eda.normalize_xyz(data)

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




def vxg_to_xyz(vxg:torch.Tensor, origin = None, voxel_size = None) -> None:
    """
    Converts voxel-grid to a raw point cloud.\\
    The selected voxels to represent the raw point cloud have label == 1.0\n

    Parameters
    ----------
    `vxg` - torch.Tensor:
        voxel-grid to be transformed with shape (64, 64, 64) for instance

    `origin` - np.ndarray:
        (3,) numpy array that encodes the origin of the voxel-grid

    `voxel_size` - np.ndarray:
        (3,) numpy array that encodes the voxel size of the voxel-grid

    Returns
    -------
    `points` - np.ndarray:
        (N, 4) numpy array that encodes the raw pcd.
    """
    # point_cloud_np = np.asarray([voxel_volume.origin + pt.grid_index*voxel_volume.voxel_size for pt in voxel_volume.get_voxels()])

    shape = vxg.shape
    origin = np.array([0, 0, 0]) if origin is None else origin
    voxel_size = np.array([1, 1, 1]) if voxel_size is None else voxel_size
    grid_indexes = np.indices(shape).reshape(3, -1).T

    points = origin + grid_indexes * voxel_size

    labels = np.array([vxg[tuple(index)] for index in grid_indexes])

    return np.concatenate((points, labels.reshape(-1, 1)), axis=1)



def voxel_to_pointcloud(voxelgrid:np.ndarray, point_locations:np.ndarray):
    """
    Converts a voxelgrid to a pointcloud given the point locations inside the voxelgrid.
    
    """
    voxel_values = np.array([voxelgrid[tuple(point)] for point in point_locations])

    return np.concatenate((point_locations, voxel_values.reshape(-1, 1)), axis=1)


def torch_voxel_to_pointcloud(voxelgrid:torch.Tensor, point_locations:torch.Tensor):

    voxel_values = torch.tensor([voxelgrid[tuple(point)] for point in point_locations])

    return torch.cat((point_locations, voxel_values.reshape(-1, 1)), dim=1)


def vox_to_pts(vox:torch.Tensor, pt_loc:torch.Tensor, include_locs:bool=False) -> torch.Tensor:
    """

    Transforms a voxelgrid tensor into a pointcloud tensor

    Parameters
    ----------

    `vox`: torch.Tensor
        Voxelgrid tensor of shape (batch, 1, z, x, y)
    
    `pt_loc`: torch.Tensor
        Pointcloud locations tensor of shape (batch, P, 3) where P is the max number of points in the input batch (padded with zeros)

    `include_locs`: bool
        If True, the output tensor will include the point locations in the voxelgrid. 
        If False, the output tensor will only include the point values in the voxelgrid.

    Returns
    -------
    `pt`: torch.Tensor
        Pointcloud tensor of shape (batch, P, 3+C | C) where P is the max number of points in the input batch (padded with zeros)

    """

    batch_size = vox.shape[0]

    #Gather the pointcloud values from the voxelgrid tensor
    pt_cloud = []
    for i in range(batch_size):
        pt_cloud.append(vox[i, :, pt_loc[i, :, 0], pt_loc[i, :, 1], pt_loc[i, :, 2]]) # (C, P, 1)
    
    pt_cloud = torch.stack(pt_cloud, dim=0) # (batch, C, P)
   
    pt_cloud = pt_cloud.permute(0, 2, 1) # (batch, P, C)

    if include_locs:
        # Concatenate pt_loc and pt_cloud tensors
        return torch.cat((pt_loc, pt_cloud), dim=2)  # (batch, P, 3+C)

    return pt_cloud



if __name__ == "__main__":

    dataset_path = '/media/didi/TOSHIBA EXT/TS40K-NEW/fit'

    npy_files:np.ndarray = np.array([file for file in os.listdir(dataset_path)
                        if os.path.isfile(os.path.join(dataset_path, file)) and '.npy' in file])
    
    npy_path = os.path.join(dataset_path, npy_files[np.random.randint(0, len(npy_files))])
        
    npy = np.load(npy_path)
    sample = (npy[:, 0:-1], npy[:, -1])

    vox_input, vox_gt, pt_loc = voxelize_sample(sample[0], sample[1], [eda.POWER_LINE_SUPPORT_TOWER, eda.MAIN_POWER_LINE, eda.MEDIUM_VEGETAION, eda.LOW_VEGETATION])

    plot_voxelgrid(vox_input, color_mode='ranges', title='Input')
    vox_gt = vox_gt / np.max(vox_gt)
    print(np.unique(vox_gt))
    print(pt_loc.shape)
    plot_voxelgrid(vox_gt, color_mode='ranges', title='Target')
    
    #VOXEL TO POINT CLOUD OPERATION
    batched_inp = torch.tensor(vox_gt).unsqueeze(0).unsqueeze(0).repeat(4, 2, 1, 1, 1)
    batched_pt_loc = torch.tensor(pt_loc).unsqueeze(0).repeat(4, 1, 1)
    pcd = vox_to_pts(batched_inp, batched_pt_loc, include_locs=True)
    pcd = pcd[0].numpy()
    pcd = pcd[pcd[:, -1] > 0]
    print(np.unique(pcd[:, -1]))
    # pcd = voxel_to_pointcloud(sample_on_voxel[0], sample_on_voxel[2])
    ply = eda.np_to_ply(pcd[:, 0:3])
    eda.color_pointcloud(ply, pcd[:, -2])
    eda.visualize_ply([ply])

 
    input("Finished Plot...")









    tower_files = eda.get_tower_files(['/media/didi/TOSHIBA EXT/Labelec_LAS'], True)

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


