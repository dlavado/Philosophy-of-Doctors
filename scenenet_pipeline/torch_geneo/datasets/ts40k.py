# %%
import math
import random
import shutil
from time import sleep
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np
import laspy as lp
import gc
import torch.nn.functional as F

import sys

from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from EDA import EDA_utils as eda

from VoxGENEO import Voxelization as Vox

import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%

def build_data_samples(data_dirs:List[str], save_dir=os.getcwd(), tower_radius=True, data_split:dict={"train": 0.7, "val": 0.2, "test": 0.1}):
    """

    Builds a dataset of voxelized point clouds in npy format according to 
    the las files in data_dirs and saves it in save_dir.

    Parameters
    ----------

    `data_dirs` - str list : 
        list of directores with las files to be converted

    `save_dir` - str:
        directory where to save the npy files.

    `tower_radius` - bool:
        type of sample we want to build -> tower_radius or two_towers

    `data_split` - dict {"str": float} or int:
        if data_split is dict:
            split of the dataset into sub-folders; keys correspond to the folders and the values to the sample split
        elif data_split is int:
            data_split == 0 for no dataset split 
    """
    
    samples_dir = os.path.join(save_dir, 'samples')

    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    counter = len(os.listdir(samples_dir)) 

    read_files = []
    pik_name = 'read_files.pickle'
    pik_path = os.path.join(save_dir, pik_name)
    if os.path.exists(pik_path):
        read_files = eda.load_pickle(pik_path)

    for dir in data_dirs:
        os.chdir(dir)

        for las_file in os.listdir("."):
            filename = os.getcwd() + "/" + las_file

            if ".las" in filename:
                las = lp.read(filename)
            else:
                continue

            if filename in read_files:
                continue

            print(f"\n\n\nReading...{filename}")

            xyz, classes = eda.las_to_numpy(las)

            if np.any(classes == eda.POWER_LINE_SUPPORT_TOWER):
                if tower_radius:
                    t_samples = eda.crop_tower_samples(xyz, classes)
                else:
                    t_samples = eda.crop_two_towers_samples(xyz, classes)
            else:
                t_samples = []
                continue
            #f_samples = eda.crop_ground_samples(xyz, classes)
            f_samples = []
            file_samples = f_samples + t_samples
            print(f"file samples: {len(file_samples)} (tower, no_tower): ({len(t_samples)}, {len(f_samples)})")

            voxelgrid_shape = (64, 64, 64)
            for sample in file_samples:
                try:
               
                    down_xyz, down_class = eda.downsampling_relative_height(sample[:, :3], sample[:, 3], sampling_per=0.8)

                    # tower -> 1 ; no_tower -> 0
                    voxeled_xyz = Vox.centroid_hist_on_voxel(down_xyz, voxelgrid_dims=voxelgrid_shape)
                    voxeled_gt = Vox.centroid_reg_on_voxel(down_xyz, down_class, eda.POWER_LINE_SUPPORT_TOWER, voxelgrid_dims=voxelgrid_shape)
                    # voxeled_xyz = Vox.hist_on_voxel(down_xyz, voxelgrid_dims=voxelgrid_shape)
                    # voxeled_gt = Vox.reg_on_voxel(down_xyz, down_class, eda.POWER_LINE_SUPPORT_TOWER, voxelgrid_dims=voxelgrid_shape)

                    assert voxeled_xyz.shape == voxeled_gt.shape
                    print(voxeled_xyz.shape)

                    npy = np.array([voxeled_xyz, voxeled_gt])
                    
                
                    npy_name = f'{save_dir}/samples/sample_{counter}.npy'
                    print(npy_name)

                    with open(npy_name, 'wb') as f:
                        np.save(f, npy)  # sample, (x, y, z, bin)
                        counter += 1
                except:
                    print(f"Problem occurred while reading {filename}\n\n")
                    continue
                del npy
                del voxeled_gt
                del voxeled_xyz
                del down_class
                del down_xyz
                
                gc.collect()
            read_files.append(filename)
            eda.save_pickle(read_files, pik_path)
            del xyz
            del classes
            del las
            del t_samples
            gc.collect()

    if data_split == 0:
        return

    samples = os.listdir(samples_dir)

    random.shuffle(samples)

    assert sum(list(data_split.values())) <= 1, "data splits should not surpass 1"

    split_sum = 0
    sample_size = len(samples)
    print(f"Number of total samples: {sample_size}")

    for key, val in data_split.items():
        split = samples[int(split_sum*sample_size):math.ceil((split_sum+val)*sample_size)]
        split_sum += val
        print(f"Samples in {key}: {len(split)}")
        for sample in split:
            dir = f'{save_dir}/{key}/'
            if not os.path.exists(dir):
                os.makedirs(dir)

            shutil.copy2(sample, dir)
            # with open(f'{dir}/sample_{counter}.npy', 'wb') as f:
            #     np.save(f, npy)  # sample, (x, y, z, class, bin, label)
            #     counter += 1

class ToTensor:

    def __call__(self, sample):
        pts, labels = sample
        return torch.from_numpy(pts.astype(np.float)), torch.from_numpy(labels.astype(np.float))

class AddPad:

    def __init__(self, pad:Tuple[int]):
        """
        `pad` is a tuple of ints that contains the pad sizes for each dimension in each direction.\n
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4]) 
        """
        self.p3d = pad

    def __call__(self, sample):
        pts, labels = sample
        return F.pad(pts, self.p3d, 'constant', 0), F.pad(labels, self.p3d, 'constant', 0)


class Voxelization:

    def __init__(self, vox_size:Tuple[int]=None, vxg_size:Tuple[int]=None) -> None:
        """
        Voxelizes raw LiDAR 3D point points in `numpy` (N, 3) format 
        according to the provided discretization

        Parameters
        ----------
        `vox_size` - Tuple of 3 Ints:
            Size of the voxels to dicretize the point clouds
        `vxg_size` - Tuple of 3 Ints:
            Size of the voxelgrid used to discretize the point clouds

        One of the two parameters need to be provided, `vox_size` takes priority

        Returns
        -------
        A Voxelized 3D point cloud in Density/Regression mode
        """
        
        if vox_size is None and vxg_size is None:
            ValueError("Voxel size or Voxelgrid size must be provided")


        self.vox_size = vox_size
        self.vxg_size = vxg_size


    def __call__(self, sample:np.ndarray):
        
        pts, labels = sample

        voxeled_xyz = Vox.centroid_hist_on_voxel(pts, voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size)
        voxeled_gt = Vox.centroid_reg_on_voxel(pts, labels, eda.POWER_LINE_SUPPORT_TOWER, voxel_dims=self.vox_size, voxelgrid_dims=self.vxg_size)

        return np.stack((voxeled_xyz, voxeled_gt))

class ToFullDense:
    """
    Transforms a Regression Dataset into a Belief Dataset.

    Essentially, any voxel that has tower points is given a belief of 1,
    in order to maximze the towers' geometry.
    For the input, the density is notmalized to 1, so empty voxels have a value
    of 0 and 1 otherwise.

    It requires a discretization of raw LiDAR Point Clouds in Torch format.
    """

    def __call__(self, sample:torch.Tensor):
        pts, labels = sample

        if len(pts.shape) == 4: # no coords
            return (pts > 0).to(pts), (labels > 0).to(labels) #full dense
            #return (pts > 0).to(pts), labels # gt still with probabilities
        else:
            pts[:, -1] = (pts[:, -1]  > 0).to(pts[:, -1])
            labels[:, -1] = (labels[:, -1]  > 0).to(labels[:, -1])
            return pts, labels
        


class torch_TS40K(Dataset):

    def __init__(self, dataset_path, split='samples', transform=ToTensor()):
        """
        Initializes the TS40K dataset

        Parameters
        ----------
        `dataset_path` - str:
            path to the directory with the voxelized point clouds crops in npy format

        `split` - str:
            split of the dataset to access 
            split \in [samples, train, val, test] 
        
        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds
        """
        # loading data

        
        self.transform = transform
        self.split = split
        self.dataset_path = os.path.join(dataset_path, split)

        self.npy_files = np.array([file for file in os.listdir(self.dataset_path)
                        if os.path.isfile(os.path.join(self.dataset_path, file)) and '.npy' in file])

    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.split == 'samples':
            print(f"Sample idx = {idx}")

        npy_path = os.path.join(self.dataset_path, self.npy_files[idx])

        try:
            npy = np.load(npy_path)
        except:
            print(f"{npy_path} probably corrupted")
            npy_path = os.path.join(self.dataset_path, self.npy_files[random.randint(0, len(self))])
            npy = np.load(npy_path)

        sample = (npy[None, 0], npy[None, 1])

        if self.transform:
            sample = self.transform(sample)

        return sample

# %%
def main():
    
    ROOT_PROJECT = "/home/didi/VSCode/lidar_thesis"

    DATA_SAMPLE_DIR = os.path.join(ROOT_PROJECT, "Data_sample")
    EXT_DIR = "/media/didi/TOSHIBA EXT/LIDAR/"
    SAVE_DIR = os.path.join(ROOT_PROJECT, "dataset/torch_dataset")

    DATA_COORD_DIR = os.path.join(ROOT_PROJECT, "dataset/coord_dataset")

    #build_data_samples([EXT_DIR, DATA_SAMPLE_DIR], SAVE_DIR)

    composed = Compose([ToTensor(), AddPad((3, 3, 3, 3, 3, 3))])
    
    ts40k = torch_TS40K(dataset_path=DATA_COORD_DIR, split='val')

    vox, vox_gt = ts40k[0]
    print(vox.shape, vox_gt.shape)

    # Hyper parameters
    NUM_EPOCHS = 5
    NUM_SAMPLES = len(ts40k)
    BATCH_SIZE = 2
    NUM_ITER = math.ceil(NUM_SAMPLES / BATCH_SIZE)

    ts40k_loader = DataLoader(ts40k, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for epoch in range(NUM_EPOCHS):
        for i, (vox, vox_gt) in tqdm(enumerate(ts40k_loader), desc=f"epoch {epoch+1}", ): 
            #print(f"epoch {epoch} / {NUM_EPOCHS}; step {i+1} / {NUM_ITER}; input: {vox.shape}; labels: {vox_gt.shape}")
            print(vox.shape)
            print(vox[:, -1].shape)
            vox_data = vox[0][-1]
            print(vox_data.shape)
            gt_label = vox_gt[0][-1]
            print(gt_label.shape)

            Vox.plot_voxelgrid(vox_data)
            Vox.plot_voxelgrid(gt_label)
            xyz_gt = Vox.vxg_to_xyz(vox_gt[0])
            print(f"xyz centroid = {eda.xyz_centroid(xyz_gt)}")
            
            print(f"euc dist = {eda.euclidean_distance(xyz_gt, xyz_gt)}")
            pcd_gt = eda.np_to_ply(xyz_gt)
            eda.visualize_ply([pcd_gt])


if __name__ == "__main__":
    main()

# %%
