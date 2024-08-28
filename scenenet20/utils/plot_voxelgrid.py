

from pathlib import Path
import sys
from typing import Union

sys.path.insert(0, '..')
import numpy as np
import utils.pointcloud_processing as eda
import torch
import os
import webcolors


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
        grid = grid.cpu().numpy()

    z, x, y = grid.nonzero()

    xyz = np.empty((len(z), 4))
    idx = 0
    for i, j, k in zip(x, y, z):
        xyz[idx] = [int(i), int(j), int(k), grid[k, i, j]]
        idx += 1

    if len(xyz) == 0:
        return
    
    uq_classes = np.unique(xyz[:, -1])
    # print(f"Unique classes: {uq_classes}")
    class_colors = np.empty((len(uq_classes), 3))

    if color_mode == 'density': # colored according to 'coolwarm' scheme
        for i, c in enumerate(uq_classes):
        # [-1, 0[ - blue; ~0 white; ]0, 1] - red
            if c < 0:
                class_colors[i] = [1+c, 1+c, 1]
            else:
                class_colors[i] = [1, 1-c, 1-c]
    # meant to be used only when `grid` contains values \in [0, 1]
    elif color_mode == 'ranges': #colored according to the ranges of values in `grid`
        import matplotlib.cm as cm
        r = 10
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
    
    # class_colors = class_colors * (class_colors > 0) # remove negative color values

    colors = np.array([class_colors[np.where(uq_classes == c)[0][0]] for c in xyz[:, -1]])
    pcd = eda.np_to_ply(xyz[:, :-1])
    xyz_colors = eda.color_pointcloud(pcd, xyz[:, -1], colors=colors)

    if not plot:
        return np.concatenate((xyz[:, :-1], xyz_colors*255), axis=1) # 255 to convert color to int8

    eda.visualize_ply([pcd], window_name=title) # visualize the point cloud



