
# %% 
import sys

sys.path.insert(0, '..')
from VoxGENEO.GENEO_kernel import GENEO_kernel
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from VoxGENEO.Voxelization import hist_on_voxel, classes_on_voxel, plot_voxelgrid
from sklearn.metrics import precision_score, recall_score, f1_score

import time
from EDA import EDA_utils as eda
import laspy as lp
import itertools


class cylinder_kernel(GENEO_kernel):

    def __init__(self, name, kernel_size, **kwargs):

        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cylinder in the kernel.")


        self.radius = np.float(kwargs['radius'])

        self.sigma = 1
        if kwargs.get('sigma') is not None:
            self.sigma = np.float(kwargs['sigma'])           

        self.epsilon = 0.1
        if kwargs.get('epsilon') is not None:
            self.epsilon = np.float(kwargs['epsilon'])

        self.tau = 0.7
        if kwargs.get('tau') is not None:
            self.tau = np.float(kwargs['tau'])
        
        super().__init__(name, kernel_size, **kwargs)   


    def compute_kernel(self, plot=False):
        """
        Returns a 3darray numpy with size kernel_size that demonstrates a cylinder.\n

        Parameters
        ----------
        radius - float:
        radius of the cylinder's base; radius <= kernel_size[1];
        
        epsilon - float:
        threshold below and beyond the radius to consider in the cylinder definition.
        x^2 + y^2 <= (radius + epsilon)^2 && x^2 + y^2 >= (radius - epsilon)^2;

        sigma - float:
        sigma variable for the gaussian distribution when assigning weights to the kernel;

        plot - bool:
        plot ? show kernel;

        Returns
        -------
        3darray numpy with the cylinder kernel.   
        """

        kernel = np.zeros(self.kernel_size)

        #floor_idxs holds all the idx combinations of the 'floor' of the kernel 
        floor_idxs = list(itertools.product(list(range(self.kernel_size[1])), list(range(self.kernel_size[2]))))
        floor_idxs = np.array([*floor_idxs]) # (x*y, 2) 

        center = ((self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2)
        
        # circle will hold idx values inside the floor_idx that correspond to a circle
        # note that there is not a direct bijection between these two, we just find points that fit 
        # the given circle definition and store them for later comparisson with floor_idxs
        circle = list()
        for i in np.arange(0, self.kernel_size[1], self.epsilon):
            for j in np.arange(0, self.kernel_size[2], self.epsilon):
                if (i - center[0])**2  + (j - center[1])**2 <= (self.radius + self.epsilon)**2 and\
                (i - center[0])**2  + (j - center[1])**2 >= (self.radius - self.epsilon)**2:
                            circle.append((i, j))
        circle = np.array([*circle]) #(N, 2)
        assert circle.shape[-1] == floor_idxs.shape[-1] == 2

        neg_sum = 0
        neg_count = 0
        for idx in floor_idxs:
            eucs = [eda.euclidean_distance(idx, circ) for circ in circle]
            # the min of the eucs corresponds to the euc_dist between the point idx
            # and the closest point in the circle, which will act as mean
            floor_value = np.exp(-0.5* (np.min(eucs)/self.sigma)**2 )
            if (idx[0] - center[0])**2  + (idx[1] - center[1])**2 <= (self.radius + self.epsilon)**2:
                kernel[:, idx[0], idx[1]] = np.full(self.kernel_size[0], floor_value)
                neg_sum += floor_value*self.kernel_size[0]
            else:
                kernel[:, idx[0], idx[1]] = np.full(self.kernel_size[0], -1)
                neg_count += self.kernel_size[0]

        neg_value = neg_sum / neg_count
        print(f"{neg_sum} / {neg_count} = {neg_value}")
        
        kernel = np.where(kernel == -1, -neg_value, kernel)
        print(kernel)
        assert np.isclose(np.sum(kernel), 0), np.sum(kernel)

        if plot:
            plot_voxelgrid(kernel)
        return kernel
        
    def keep_label(x, label):
        return 1 if x == label else 0

    def binary_class(x, tau):
        return 1 if x >= tau else 0
    
    def score_kernel(self, xyz, voxel_labels, conv_function=binary_class, gt_function=keep_label):
        gt = np.vectorize(gt_function)
        bin = np.vectorize(conv_function)
        voxel_labels = gt(voxel_labels, eda.POWER_LINE_SUPPORT_TOWER)
        conv = bin(self.convolution(xyz, plot=False), self.tau)
        plot_voxelgrid(voxel_labels)
        plot_voxelgrid(conv)

        assert len(np.unique(conv)) == 2
        assert conv.shape == voxel_labels.shape

        voxel_loss = (conv + voxel_labels) / 2
        voxel_loss2 = (conv - voxel_labels)

        non_empty_voxels = len(voxel_loss.nonzero()[0])
        print(f"Loss: {np.sum(np.abs(voxel_loss2)) / non_empty_voxels}")
        y_true = voxel_labels.flatten()
        y_pred = conv.flatten()
        print(f"Precision: {precision_score(y_true, y_pred)}")
        print(f"Recall: {recall_score(y_true, y_pred)}")
        print(f"F1 score: {f1_score(y_true, y_pred)}")
        self.plot_voxelgrid(voxel_loss) #shows things in common
        self.plot_voxelgrid(voxel_loss2) #shows where they disagree

    def mandatory_parameters():
        return ['radius']

# %%
if __name__ == "__main__":

    tower_files = eda.get_tower_files(False)

    pcd_xyz, classes = eda.las_to_numpy(lp.read(tower_files[0]))

    pcd_tower, _ = eda.select_object(pcd_xyz, classes, [eda.POWER_LINE_SUPPORT_TOWER])
    towers = eda.extract_towers(pcd_tower, visual=False)
    crop_tower_xyz, crop_tower_classes = eda.crop_tower_radius(pcd_xyz, classes, towers[0])

    xyz = crop_tower_xyz
    down_xyz, down_classes = eda.downsampling(eda.np_to_ply(xyz), crop_tower_classes, samp_per=0.3)
    eda.visualize_ply([eda.np_to_ply(down_xyz)])
    # %%
    vox_shape = (64, 64, 64)
    cylinder = cylinder_kernel('cy', (9, 6, 6), radius=2) # kernel shape in (z, x, y)
    cylinder.visualize_kernel()
    # %%
    start = time.time()
    data = hist_on_voxel(down_xyz, vox_shape) # if no xyz is passed, it uses towers[0]
    end = time.time() - start
    print(f"elapsed time voxelization: {end}")

    voxel_labels = classes_on_voxel(down_xyz, down_classes, vox_shape)
    # %%
    print(f" {data[data.nonzero()]} of {len(data)}^3" )
    plot_voxelgrid(data)
    plot_voxelgrid(voxel_labels)
    
    # %%
    cylinder.convolution(data)

    # %%
    cylinder.score_kernel(data, voxel_labels)



# %%
