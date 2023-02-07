
# %% 
import sys


sys.path.insert(0, '..')
from pipelines.datasets.ts40k import torch_TS40K

from VoxGENEO.GENEO_kernel import GENEO_kernel
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from VoxGENEO.Voxelization import hist_on_voxel, classes_on_voxel, plot_voxelgrid
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import time
from EDA import EDA_utils as eda
import laspy as lp
import itertools

class cone_kernel(GENEO_kernel):
    """
    GENEO kernel that encodes a cone on top of a cylinder.\n

    The cylinder's radius and cone's apex location are required to
    compute it

    Required
    --------

    radius - float \in ]0, kernel_size[1]] :
    radius of the cylinder's base;

    apex - float \in [0, kernel_size[0]]:
    cone's apex point relative to the height of the kernel; 

    Optional
    --------
    epsilon - float \in [0, 1]:
    threshold below and beyond the radius to consider in the pattern's definition.
    x^2 + y^2 <= (radius + epsilon)^2 && x^2 + y^2 >= (radius - epsilon)^2;

    sigma - float:
    sigma variable for the gaussian distribution when assigning weights to the kernel;

    tau - float \in [0, 1]:
    Detection threshold for GENEO prediction score

    """

    def __init__(self, name, kernel_size, **kwargs):
       
        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cylinder in the kernel.")

        if kwargs.get('apex') is None:
            raise KeyError("Provide a height for the cone.")

        self.radius = np.float(kwargs['radius'])
        self.apex = np.int(kwargs['apex'])

        self.sigma = 1
        if kwargs.get('sigma') is not None:
            self.sigma = np.float(kwargs['sigma'])           

        self.epsilon = 0.1
        if kwargs.get('epsilon') is not None:
            self.epsilon = np.float(kwargs['epsilon'])
        
        self.tau = 0.7
        if kwargs.get('tau') is not None:
            self.tau = np.float(kwargs['tau'])
        
        super().__init__(name, kernel_size)    


    def compute_kernel(self, plot=False):
        """
        Returns a 3darray numpy with size kernel_size that demonstrates am inverted cone and a cylinder.\n

        Parameters
        ----------
        plot - bool:
        plot ? show kernel;

        Returns
        -------
        3darray numpy with the cone kernel.   
        """

        kernel = np.zeros(self.kernel_size)

        #floor_idxs holds all the idx combinations of the 'floor' of the kernel 
        floor_idxs = list(itertools.product(list(range(self.kernel_size[1])), list(range(self.kernel_size[2]))))
        floor_idxs = np.array([*floor_idxs]) # (x*y, 2) 

        center = ((self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2)
        
        # circle will hold idx values inside the floor_idx that correspond to a circle
        # note that there is not a direct bijection between these two, we just find points that fit 
        # the given circle definition and store them for later comparison with floor_idxs
        circle = list()
        for i in np.arange(0, self.kernel_size[1], self.epsilon):
            for j in np.arange(0, self.kernel_size[2], self.epsilon):
                if (i - center[0])**2  + (j - center[1])**2 <= (self.radius + self.epsilon)**2 and\
                    (i - center[0])**2  + (j - center[1])**2 >= (self.radius - self.epsilon)**2:
                            circle.append((i, j)) # cylinder's base points 
                            
        cone_radius = np.linspace(0, self.kernel_size[1] -1, self.apex)
        cone = list()
        for a in np.arange(self.apex):
            z = self.kernel_size[0] - 1 - a 
            z_list = list()
            # for each height from kernel_size[0] to apex, we are going to save the corresponding circle points
            for i in np.arange(0, self.kernel_size[1], self.epsilon):
                for j in np.arange(0, self.kernel_size[2], self.epsilon):
                    if (i - center[0])**2  + (j - center[1])**2 <= (cone_radius[a] + self.epsilon)**2:
                                z_list.append((i, j)) # cone's points
            cone.append(z_list) 
            
        circle = np.array([*circle]) #(N, 2)

        neg_sum = 0
        cylinder_height = self.kernel_size[0] - self.apex
        for idx in floor_idxs:
            eucs = [eda.euclidean_distance(idx, circ) for circ in circle]
            # the min of the eucs corresponds to the euc_dist between the point idx
            # and the closest point in the circle, which will act as mean
            weight = np.exp(-0.5* (np.min(eucs)/self.sigma)**2 )
            if (idx[0] - center[0])**2  + (idx[1] - center[1])**2 <= (self.radius + self.epsilon)**2:
                kernel[0:cylinder_height, idx[0], idx[1]] = np.full(cylinder_height, weight)
                neg_sum += weight*cylinder_height

        for a in np.arange(self.apex):
            z = self.kernel_size[0] - 1 - a
            z_list  = np.array([*cone[a]])
            for idx in floor_idxs:
                eucs = [eda.euclidean_distance(idx, c) for c in z_list]
                # the min of the eucs corresponds to the euc_dist between the point idx
                # and the closest point in the circle, which will act as mean
                weight = np.exp(-0.5* (np.min(eucs)/self.sigma)**2 )
            
                if (idx[0] - center[0])**2  + (idx[1] - center[1])**2 <= (cone_radius[a] + self.epsilon)**2:
                    kernel[z, idx[0], idx[1]] = weight
                    neg_sum += weight

        neg_count = len(np.where(kernel == 0)[0])
        neg_value = - neg_sum / neg_count
        print(f"{neg_sum} / {neg_count} = {-1*neg_value}")
        
        kernel = np.where(kernel == 0, neg_value, kernel)
        assert np.isclose(np.sum(kernel), 0), np.sum(kernel)

        if plot:
            plot_voxelgrid(kernel)
        return kernel


    def keep_label(x, label, label2):
        return 1 if x >= label and x <= label2 else 0

    def binary_class(x, tau):
        return 1 if x >= tau else 0

    def score_kernel(self, xyz, voxel_labels, conv_function=binary_class, gt_function=keep_label):
        gt = np.vectorize(gt_function)
        bin = np.vectorize(conv_function)
        voxel_labels = gt(voxel_labels, eda.POWER_LINE_SUPPORT_TOWER, eda.OTHER_POWER_LINE)
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
        return ['radius', 'apex']


# %%
if __name__ == "__main__":

    tower_files = eda.get_tower_files(False)

    pcd_xyz, classes = eda.las_to_numpy(lp.read(tower_files[0]))

    pcd_tower, _ = eda.select_object(pcd_xyz, classes, [eda.POWER_LINE_SUPPORT_TOWER])
    towers = eda.extract_towers(pcd_tower, visual=False)
    crop_tower_xyz, crop_tower_classes = eda.crop_tower_radius(pcd_xyz, classes, towers[0])
    # %%
    xyz = crop_tower_xyz
    down_xyz, down_classes = eda.downsampling(eda.np_to_ply(xyz), crop_tower_classes, samp_per=0.3)
    eda.visualize_ply([eda.np_to_ply(down_xyz)])

    # %% 
    ROOT_PROJECT = "/home/didi/VSCode/lidar_thesis"

    DATA_SAMPLE_DIR = ROOT_PROJECT + "/Data_sample"
    SAVE_DIR = ROOT_PROJECT + "/dataset/torch_dataset"
    print(DATA_SAMPLE_DIR)

    #build_data_samples([DATA_SAMPLE_DIR], SAVE_DIR)
    ts40k = torch_TS40K(dataset_path=SAVE_DIR)

    vox, vox_gt = ts40k[2]
    print(vox.shape)
    plot_voxelgrid(vox.numpy()[0])
    plot_voxelgrid(vox_gt.numpy()[0])
    # %%
    kernel_size = (9, 6, 6)
    cone = cone_kernel('cone', kernel_size, radius=1, apex=4, tau=0.5)
    cone.visualize_kernel()

    cone.convolution(vox.numpy()[0])
    # %% 
    vox_shape = (64, 64, 64)
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
    conv = cone.convolution(data)
    cone.score_kernel(data, voxel_labels)
  


# %%
