# %%
import math
from pathlib import Path
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
import torch

import sys

from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from utils import pcd_processing as eda
from core.datasets.torch_transforms import EDP_Labels, Farthest_Point_Sampling, ToFullDense, Voxelization, ToTensor, Voxelization_withPCD

from utils import voxelization as Vox
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def edp_labels(labels:np.ndarray) -> np.ndarray:
        #cast each label to its corresponding EDP label
        labels = np.array([eda.DICT_NEW_LABELS[label] if label >= 0 else label for label in labels]).reshape(labels.shape)
        return labels

def build_data_samples(data_dirs:List[str], save_dir=os.getcwd(), tower_radius=True, data_split:dict={"fit": 0.6, "test": 0.4}):
    """

    Builds a dataset of voxelized point clouds in npy format according to 
    the las files in data_dirs and saves it in save_dir.

    Parameters
    ----------

    `data_dirs` - str list : 
        list of directories with las files to be converted

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

    print(f"\n\nBuilding dataset in {save_dir}...")
    
    for folder in data_split.keys():
        if not os.path.exists(os.path.join(save_dir, folder)):
            os.makedirs(os.path.join(save_dir, folder))

    # process all files to `fit_path` and then retreive test samples
    fit_path = os.path.join(save_dir, 'fit')
    counter = len(os.listdir(fit_path)) 

    read_files = []
    pik_name = 'read_files.pickle'
    pik_path = os.path.join(save_dir, pik_name)
    if os.path.exists(pik_path):
        read_files = eda.load_pickle(pik_path)

    for dir in data_dirs:
        os.chdir(dir)

        for las_file in os.listdir("."):
            filename = os.getcwd() + "/" + las_file

            if filename in read_files:
                continue

            if ".las" in filename:
                try:
                    las = lp.read(filename)
                except Exception as e:
                    print(f"Problem occurred while reading {filename}\n\n")
                    continue
            else:
                continue

            print(f"\n\n\nReading...{filename}")

            xyz, classes = eda.las_to_numpy(las)

            if np.any(classes == eda.POWER_LINE_SUPPORT_TOWER):
                if tower_radius:
                    t_samples = eda.crop_tower_samples(xyz, classes, radius=20)
                else:
                    t_samples = eda.crop_two_towers_samples(xyz, classes)
            else:
                continue

            f_samples = []
            f_samples = eda.crop_ground_samples(xyz, classes)
            
            file_samples = f_samples + t_samples
            print(f"file samples: {len(file_samples)} (tower, no_tower): ({len(t_samples)}, {len(f_samples)})")

            for sample in file_samples:
                uq_classes = np.unique(edp_labels(sample[:, -1])) 
                # quality check of the point cloud
                if sample.shape[0] < 500:
                    print(f"Sample has less than 500 points...\nSkipping...")
                    continue
                elif np.sum(uq_classes > 0) <= 2:
                    print(f"Sample has less than 2 semantic classes...\nSkipping...")
                    continue
                else:
                    print(f"Sample has {sample.shape[0]} points and {uq_classes} classes")

                try:             
                    npy_name = f'{fit_path}/sample_{counter}.npy'
                    print(npy_name)

                    with open(npy_name, 'wb') as f:
                        np.save(f, sample)  # sample: x, y, z, class
                        counter += 1
                except:
                    print(f"Problem occurred while reading {filename}\n\n")
                    continue
                
                gc.collect()

            del las
            del t_samples
            gc.collect()
            read_files.append(filename)
            eda.save_pickle(read_files, pik_path)
          

    if data_split == 0:
        return # all files in fit_path

    samples = os.listdir(fit_path)
    random.shuffle(samples)

    assert sum(list(data_split.values())) <= 1, "data splits should not surpass 1"

    split_sum = 0
    sample_size = len(samples)
    print(f"Number of total samples: {sample_size}")

    for folder, split in data_split.items():
        if folder == 'fit':
            split_sum += split
            continue

        dir = os.path.join(save_dir, folder)
        if not os.path.exists(dir):
            os.makedirs(dir)

        samples_split = samples[int(split_sum*sample_size):math.ceil((split_sum+split)*sample_size)]
        split_sum += split
        print(f"Samples in {folder}: {len(samples_split)}")

        for sample in samples_split:
            shutil.move(os.path.join(fit_path, sample), dir)




def save_preprocessed_data(data_dir, save_dir, fps_points, vxg_size, vox_size):
    """
    
    """

    data_split = ['fit', 'test']

    ts40k_transform = Compose([
                        ToTensor(),

                        Farthest_Point_Sampling(fps_points),
        
                        Voxelization_withPCD(keep_labels='all', 
                                             vxg_size=vxg_size, 
                                             vox_size=vox_size
                                            ),
                        EDP_Labels(),
                    ])

    for folder in data_split:

        folder_path = os.path.join(save_dir, folder)

        dm = TS40K(data_dir, split=folder, transform=ts40k_transform, min_points=None)

        os.makedirs(folder_path, exist_ok=True)

        # get folder count
        folder_count = len(os.listdir(folder_path))

        for i in tqdm(range(folder_count, len(dm)), desc=f"Saving Preprocessed {folder} samples..."):
            x, y, pt_locs = dm[i]
            sample = [x, y, pt_locs] 
            # torch save
            sample_path = os.path.join(folder_path, f"sample_{i}.pt")
            torch.save(sample, sample_path)


class TS40K(Dataset):

    def __init__(self, dataset_path, split='fit', transform=None, min_points=None, load_into_memory=False) -> None:
        """
        Initializes the TS40K dataset

        Parameters
        ----------
        `dataset_path` - str:
            path to the directory with the voxelized point clouds crops in npy format

        `split` - str:
            split of the dataset to access 
            split \in [fit, test] 
            
        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds

        `min_points` - int:
            minimum number of points in the point cloud

        `load_into_memory` - bool:
            if True, loads the entire dataset into memory
        """
        super().__init__()

        self.transform = transform
        self.split = split

        self.dataset_path = os.path.join(dataset_path, split)

        self.data_files:np.ndarray = np.array([file for file in os.listdir(self.dataset_path)
                        if os.path.isfile(os.path.join(self.dataset_path, file)) and ('.npy' in file or '.pt' in file)])
        
        if min_points:
            self.data_files = np.array([file for file in self.data_files if np.load(os.path.join(self.dataset_path, file)).shape[0] >= min_points])
    
        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True

    def __len__(self):
        return len(self.data_files)

    def __str__(self) -> str:
        return f"TS40K {self.split} Dataset with {len(self)} samples"
    
    def _load_data_into_memory(self):
        
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    def set_transform(self, new_transform):
        self.transform = new_transform
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if self.load_into_memory:
            return self.data[idx]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, self.data_files[idx])

        try:
            npy = np.load(npy_path)
        except:
            print(f"Unreadable file: {npy_path}")
        
        sample = (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 


        if self.transform:
            sample = self.transform(sample)
            #print(f"Transformed sample: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")
            return sample
        
        return sample
    

class TS40K_Preprocessed(TS40K):

    def __init__(self, dataset_path, split='fit', load_into_memory=False) -> None:
        super().__init__(dataset_path, split, None, None, load_into_memory)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if self.load_into_memory:
            return self.data[idx]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        pt_path = os.path.join(self.dataset_path, self.data_files[idx])
        try:
            pt = torch.load(pt_path)
        except:
            print(f"Unreadable file: {pt_path}")
        
        sample = (pt[0], pt[1], pt[2])
        return sample



                


            
            

def main():
    
    ROOT_PROJECT = constants.ROOT_PROJECT

    #EXT_DIR = "/media/didi/TOSHIBA EXT/LIDAR/"
    EXT_DIR = constants.EXT_PATH
    TS40K_DIR = os.path.join(EXT_DIR, "TS40K-Dataset")
    #EXT_DIR = "/media/didi/TOSHIBA EXT"
    SAVE_PATH = os.path.join(TS40K_DIR, "TS40K-NEW")


    save_preprocessed_data(constants.TS40K_PATH, constants.TS40K_PREPROCESSED_PATH, fps_points=10000, vxg_size=(64, 64, 64), vox_size=None)

    # build_data_samples([os.path.join(TS40K_DIR, 'LIDAR'), os.path.join(TS40K_DIR, 'Labelec_LAS')], SAVE_PATH, data_split={'fit': 0.8, 'test': 0.2})

    input("Press Enter to continue...")

    #composed = Compose([ToTensor(), AddPad((3, 3, 3, 3, 3, 3))]) 
    vxg_size  = (64, 64, 64)
    vox_size = (0.5, 0.5, 0.5) #only use vox_size after training or with batch_size = 1
    composed =  None
    
    ts40k = TS40K(dataset_path=EXT_DIR, split='fit', transform=composed)
    
    small_samples = []

    for i in range(len(ts40k)):
        pts, gts = ts40k[i]
        if pts.shape[1] < 500:
            small_samples.append(i)

    # del small samples
    for i in reversed(small_samples):
        npy_path = os.path.join(ts40k.dataset_path, ts40k.data_files[i])
        os.remove(npy_path)

    print(len(ts40k))

    print(ts40k[0])
    vox, vox_gt = ts40k[0]
    print(vox.shape, vox_gt.shape)

    Vox.plot_voxelgrid(torch.squeeze(vox))
    Vox.plot_voxelgrid(torch.squeeze(vox_gt))

    # Hyper parameters
    NUM_EPOCHS = 5
    NUM_SAMPLES = len(ts40k)
    BATCH_SIZE = 1
    NUM_ITER = math.ceil(NUM_SAMPLES / BATCH_SIZE)

    ts40k_loader = DataLoader(ts40k, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for epoch in range(NUM_EPOCHS):
        for i, (xyz, vox, vox_gt) in tqdm(enumerate(ts40k_loader), desc=f"epoch {epoch+1}", ): 
            #print(f"epoch {epoch} / {NUM_EPOCHS}; step {i+1} / {NUM_ITER}; input: {vox.shape}; labels: {vox_gt.shape}")
            print(xyz.shape)
            print(vox.shape)
            print(vox_gt.shape)

            xyz, vox, vox_gt = xyz[0][0], vox[0][0], vox_gt[0][0]  #1st sample of batch; decapsulate

            Vox.plot_voxelgrid(vox)
            Vox.plot_voxelgrid(vox_gt)
            xyz_gt = Vox.vxg_to_xyz(torch.cat((xyz, vox_gt[None]), axis=0))
            pcd_gt = eda.np_to_ply(xyz_gt)
            eda.visualize_ply([pcd_gt])

            print(f"xyz centroid = {eda.xyz_centroid(xyz_gt)}")
            print(f"euc dist = {eda.euclidean_distance(xyz_gt, xyz_gt)}")


if __name__ == "__main__":

    from scripts import constants
    
    main()

# %%
