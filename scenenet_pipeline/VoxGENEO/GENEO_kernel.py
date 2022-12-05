


import sys
sys.path.insert(0, '..')
from EDA import EDA_utils as eda
from abc import abstractmethod
from scipy.ndimage import convolve
import numpy as np
import time
import torch

class GENEO_kernel():
    """
    Initialization class for GENEO kernels.

    Kernels are built on top of the convolution operation as 3D arrays.

    * kernel shape in (z, x, y)
    """

    def __init__(self, name, kernel_size):
        self.name = name
        self.kernel_size = kernel_size
        self.kernel = self.compute_kernel()


    @abstractmethod
    def compute_kernel(self, plot=False):
        """
        Returns a 3D GENEO kernel in numpy format
        """
        return

    def convolution(self, xyz, plot=True):
        """
        Convolves the kernel with the xyz data passed as argument.

        plot ? Visualize convolution output
        """
        start = time.time()
        conv = convolve(xyz, self.kernel, mode='constant', cval=0.0)
        end = time.time() - start

        if plot:
            print(f"elapsed time for convolution: {end}")
            self.plot_voxelgrid(conv)

        return conv

    @abstractmethod
    def score_kernel(self, xyz, voxel_labels, conv_function, gt_function):
    
        """
        Scores the kernel when convolving xyz and compared against the ground truth.
        The result of the convolution is taken as a GENEO prediction, with a voxel 
        being labelled according to conv_function.
        It will output precision, recall and visualization of the confusion matrix.

        Parameters
        ----------
        xyz - 3d numpy array:
        point cloud data in numpy format

        voxel_labels - 3d numpy array:
        ground truth

        conv_function - function:
        transforms the convolution output into a GENEO prediction

        gt_function - function:
        transforms the labels in voxel_labels into an appropriate GT
        """
        return 

    @staticmethod
    def plot_voxelgrid(grid):
        """
        Plots voxel-grid.\\
        Color of the voxels depends on the values
        of each voxel, i.e., values close to 1 (high values)
        are red, and low values are blue.\\
        Values below a threshold values are not shown.
        """
        grid = np.array(grid)
        z, x, y = grid.nonzero()

        xyz = np.empty((len(z), 4))
        idx = 0
        for i, j, k in zip(x, y, z):
            xyz[idx] = [int(i), int(j), int(k), grid[k, i, j]]
            idx += 1
        
        uq_classes = np.unique(xyz[:, -1])
        class_colors = np.empty((len(uq_classes), 3))
        # for i, c in enumerate(uq_classes):
        # [0, 0.5] - blue; [0.5, 1] - red; c.c. - white
        #     if c <= 0:
        #         class_colors[i] = [1, 1, 1] # white
        #     else:
        #         class_colors[i] = [c, 0, 1-c] # high values are red; low values are blue;

        # for i, c in enumerate(uq_classes):
        # [-1, -0.5[ - blue; [-0.5, 0.5] - purple; ]0.5, 1] - red
        #     redness = (c + 1) / 2
        #     class_colors[i] = [redness, 0, 1-redness] # high values are red; low values are blue;

        for i, c in enumerate(uq_classes):
        # [-1, 0[ - blue; ~0 white; ]0, 1] - red
            if c < 0:
                class_colors[i] = [1+c, 1+c, 1]
            else:
                class_colors[i] = [1, 1-c, 1-c]


        pcd = eda.np_to_ply(xyz[:, :-1])
        eda.color_pointcloud(pcd, xyz[:, -1], class_color=class_colors)

        eda.visualize_ply([pcd])

    def visualize_kernel(self):
        self.plot_voxelgrid(self.kernel)

    @staticmethod
    def mandatory_parameters():
        return []

