
# %% 
import sys

sys.path.insert(0, '..')
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

class neg_sphere_kernel(GENEO_kernel):
    """
    GENEO kernel that encodes a negative sphere.\n

    Required
    --------

    radius - float \in ]0, kernel_size[1]] :
    radius of the cylinder's base;

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

        kernel = np.zeros(self.kernel_size)

        #idxs holds all the idx combinations of the kernel 
        idxs = list(itertools.product(list(range(self.kernel_size[0])), list(range(self.kernel_size[1])), list(range(self.kernel_size[2]))))
        #idxs = np.array([*idxs]) # (z*x*y, 3) 

        # center
        c_z, c_x, c_y = ((self.kernel_size[0]-1)/2, (self.kernel_size[1]-1)/2, (self.kernel_size[2]-1)/2)
        
        neg_sum = 0
        for z, x, y in idxs:
            if (z - c_z)**2  + (x - c_x)**2 + (y - c_y)**2 <= (self.radius + self.epsilon)**2:
                kernel[z, x, y] = -1
                neg_sum += 1
    
        neg_count = len(np.where(kernel == 0)[0])
        neg_value = neg_sum / neg_count
        print(f"{neg_sum} / {neg_count} = {neg_value}")
        
        kernel = np.where(kernel == 0, neg_value/4, kernel)
        #assert np.isclose(np.sum(kernel), 0), np.sum(kernel)

        if plot:
            plot_voxelgrid(kernel)
        return kernel


    def keep_label(x, label, label2):
        return 1 if x >= label and x <= label2 else 0

    def binary_class(x, tau):
        return 1 if x >= tau else 0

    def score_kernel(self, xyz, voxel_labels, conv_function=binary_class, gt_function=keep_label):
        """
        TODO
        """

    def mandatory_parameters():
        return ['radius']


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
    kernel_size = (6, 6, 6)
    sphere = neg_sphere_kernel('cone', kernel_size, radius=3)
    sphere.visualize_kernel()
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
    conv = sphere.convolution(data)

    sphere.score_kernel(data, voxel_labels)
    
    # %%
    def binary_class(x, tau):
        return 1 if x >= tau else 0
    f = np.vectorize(binary_class)
    conv2 = f(conv, 0.5)
    plot_voxelgrid(conv2)

# %%
