# %%
import math
from pathlib import Path
import random
import shutil
from typing import List, Tuple
import torch
import torch.nn as nn
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
from datasets.TS40K.torch_transforms import ToFullDense, Voxelization, ToTensor

from VoxGENEO import Voxelization as Vox

import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TS40KDataset(Dataset):

    def __init__(self, dataset_path, split='samples', transform=ToTensor()):
        """
        Initializes the TS40K dataset

        Parameters
        ----------
        `dataset_path` - str:
            path to the directory with the point clouds crops in npy format

        `split` - str:
            split of the dataset to access 
            split \in [samples, train, val, test] 


        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds
        """
        # loading data
        
        self.transform = transform
        self.split = split
        self.data_split = {
            'samples' : [0.0, 1.0],   # 100%
            'train' :   [0.0, 0.5],   # 50%
            'val' :     [0.5, 0.7],   # 20%
            'test' :    [0.7, 1.0]    # 30%
        }
        

        self.dataset_path = os.path.join(dataset_path, 'samples')

        self.npy_files:np.ndarray = np.array([file for file in os.listdir(self.dataset_path)
                        if os.path.isfile(os.path.join(self.dataset_path, file)) and '.npy' in file])
        
        # seed = 123
        # np.random.seed(seed)
        # np.random.shuffle(self.npy_files)

        beg, end = self.data_split[split]
        self.npy_files = self.npy_files[math.floor(beg*self.npy_files.size): math.floor(end*self.npy_files.size)]


    def __len__(self):
        return len(self.npy_files)

    def __str__(self) -> str:
        return f"TS40Kv2 {self.split} Dataset with {len(self)} samples"

    def set_transform(self, new_transform):
        self.transform = new_transform
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, self.npy_files[idx])

        try:
            npy = np.load(npy_path)
        except:
            print(f"{npy_path} probably corrupted")
            npy_path = os.path.join(self.dataset_path, self.npy_files[random.randint(0, len(self))])
            npy = np.load(npy_path)

        sample = (npy[:, 0:-1], npy[:, -1]) # xyz-coord (N, 3); label (N,) 

        try:
            if self.transform:
                sample = self.transform(sample)
            else:
                sample = (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 
        
        except:
            print(f"Corrupted or Empty Sample: {npy_path}")

            if self.transform:
                sample = (np.zeros((100, 3)), np.zeros(100))
                sample = self.transform(sample)
            else:
                sample =  (np.zeros((1, 100, 3)), np.zeros((1, 100))) # batch dim

        return sample

    
    def get_item_no_transform(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, self.npy_files[idx])

        if self.split == 'val':
            print(npy_path)

        try:
            npy = np.load(npy_path)

            return (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 
        except:
            print(f"Corrupted or Empty Sample")

            return np.zeros((1, 100, 3)), np.zeros((1, 100))

    def get_item_from_path(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, f"sample_{idx}.npy")

        npy = np.load(npy_path)

        return (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 



class TS40KDataset_Unsupervised(TS40KDataset):

    def __init__(self, dataset_path, split='samples', vxg_size=(128,)*3, downsample=0.8, transform=None):
        """
        Initializes the TS40K dataset for unsupervised learning.

        Parameters
        ----------

        `dataset_path` - str:
            path to the directory with the point clouds crops in npy format

        `split` - str:
            split of the dataset to access 
            split \in [samples, train, val, test] 

        `vxg_size` - tuple:
            size of the voxel grid to use for the voxelization

        `downsample` - float:
            downsampling factor to apply to the point clouds
        """

        super().__init__(dataset_path, split=split, transform=transform)

        self.gt_transform = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size), 
                                     ToTensor(), 
                                     ToFullDense(apply=(True, True))])
        
        # Downsample the input voxel grid to a lower resolution
        self.downsample = downsample
        input_vxg_size = tuple([int(vxg_size[i] * downsample) for i in range(len(vxg_size))]) if downsample is not None else vxg_size
        
        self.input_transform = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=input_vxg_size), 
                                     ToTensor(), 
                                     ToFullDense(apply=(True, True))])
        

        # Upsampling layer to upsample the input voxel grid to the desired resolution
        self.upsampling_layer = nn.Upsample(size=vxg_size, mode='trilinear', align_corners=False)
        

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, self.npy_files[idx])

        try:
            npy = np.load(npy_path)
        except:
            print(f"Error loading sample: {npy_path}")
            return
        
    
        sample = (npy[:, 0:-1], npy[:, -1]) # xyz-coord (N, 3); label (N,) 

        # The transform functions are programmed for supervised learning
        gt, _ = self.gt_transform(sample)  # gt (1, *vxg_size)
        input_vxg, _ = self.input_transform(sample) # input_vxg (1, *input_vxg_size)

        # print(gt.shape, input_vxg.shape)

        if self.downsample is not None:
            input_vxg = self.upsampling_layer(input_vxg[None]).squeeze(dim=0) # input_vxg (1, *vxg_size)

        sample = (input_vxg, gt)

        if self.transform:
            return self.transform(sample)

        return sample

     

def main():
    
    ROOT_PROJECT = Path(os.path.abspath(__file__)).parents[3].resolve()

    DATA_SAMPLE_DIR = os.path.join(ROOT_PROJECT, "Data_sample")
    #EXT_DIR = "/media/didi/TOSHIBA EXT/LIDAR/"
    EXT_DIR = "/media/didi/TOSHIBA EXT/TS40K/"

    #build_data_samples([EXT_DIR, DATA_SAMPLE_DIR], SAVE_DIR)

    #composed = Compose([ToTensor(), AddPad((3, 3, 3, 3, 3, 3))]) 
    vxg_size  = (64, 64, 64)
    vox_size = (0.5, 0.5, 0.5) #only use vox_size after training or with batch_size = 1
    composed = Compose([Voxelization([eda.POWER_LINE_SUPPORT_TOWER], vxg_size=vxg_size), 
                        ToTensor(), 
                        ToFullDense(apply=(True, False))])
    
    #ts40k = TS40KDataset(dataset_path=EXT_DIR, split='val', transform=composed)
    ts40k = TS40KDataset_Unsupervised(dataset_path=EXT_DIR, split='val', transform=None)

    print(len(ts40k))

    print(ts40k[0])
    vox, vox_gt = ts40k[0]
    print(vox.shape, vox_gt.shape)

    Vox.plot_voxelgrid(torch.squeeze(vox))
    Vox.plot_voxelgrid(torch.squeeze(vox_gt))

    input("Press Enter to continue...")

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
    main()

# %%
