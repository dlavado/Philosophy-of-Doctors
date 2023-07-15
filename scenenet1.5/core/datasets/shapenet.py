



from typing import Dict, List
from pytorch3d.datasets import ShapeNetCore
from torchvision.transforms import Compose
import torch
from torch.utils.data import Dataset
from torch import nn as nn

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

import sys
sys.path.append("..")
sys.path.append("../..")

from core.datasets.shapenet_transforms import *


class ShapeNet(ShapeNetCore):


    def __init__(self, data_dir, synsets=None, version: int = 1, load_textures: bool = True, texture_resolution: int = 4, transform=None) -> None:
        super().__init__(data_dir, synsets, version, load_textures, texture_resolution)

        self.transform = transform

    def __getitem__(self, idx: int):

        if self.transform is not None:
            return self.transform(super().__getitem__(idx))
        
        return super().__getitem__(idx)
    



class LitShapeNet(pl.LightningDataModule):


    def __init__(self, data_dir, batch_size, transform, num_workers=8, val_split =0.2, test_split=0.1) -> None:
        
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        if transform is None:
            self.transforms = Compose([ # TODO: This should be in main script to allow hyperparameter configuration
                MeshToPCD(10000, True),
                # Farthest_Point_Sampling(1024),
                Normalize_PCD()
            ])
        else:
            self.transforms = transform

        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers


    def setup(self, stage:str=None):

        shapenet = ShapeNet(self.data_dir, version=2, transform=self.transforms)

        self.shapenet_trainval, self.shapenet_test = random_split(shapenet, [len(shapenet) - int(len(shapenet) * self.test_split), int(len(shapenet) * self.test_split)])
        
        if stage == 'fit':
            self.shapenet_train, self.shapenet_val = random_split(self.shapenet_trainval, [len(self.shapenet_trainval) - int(len(self.shapenet_trainval) * self.val_split), int(len(self.shapenet_trainval) * self.val_split)])
            del self.shapenet_trainval
        if stage == 'test' or stage == 'predict':
            pass
    
    def train_dataloader(self):
        return DataLoader(self.shapenet_train, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.shapenet_val, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.shapenet_test, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.shapenet_test, batch_size=self.batch_size, num_workers=self.num_workers)
    



if __name__ == '__main__':
    from pytorch3d.datasets import ShapeNetCore
    import h5py
    import os
    import numpy as np
    import sys
    sys.path.insert(0, '..')
    sys.path.insert(1, '../..')
    import utils.pcd_processing as eda

    SHAPE_NET_CORE_PATH = "/media/didi/TOSHIBA EXT/ShapeNetCore"

    PART_NET_PATH = "/media/didi/TOSHIBA EXT/sem_seg_h5"

    
    categories = list(os.listdir(PART_NET_PATH))
    categories = [os.path.join(PART_NET_PATH, cat) for cat in categories]

    for cat_path in categories:
        
        print("\n")
        print(cat_path)

        for h5_file in os.listdir(cat_path):
            if '.h5' not in h5_file:
                continue    

            h5_path = os.path.join(cat_path, h5_file)

            with h5py.File(h5_path, 'r') as h5:
                
                print(h5.keys())
                print(h5["data_num"])
                print(h5["data"].shape)
                print(h5["label_seg"].shape)

                # input("Press Enter to continue...")

                # pcd = eda.np_to_ply(h5["data"][0])
                # eda.color_pointcloud(pcd, h5["label_seg"][0])
                # eda.visualize_ply([pcd])



    composed = Compose([
        MeshToPCD(10000, True),
        Farthest_Point_Sampling(1024),
        Normalize_PCD()
    ])


    shapenet_dataset  = LitShapeNet(SHAPE_NET_CORE_PATH, 1, None)

    shapenet_dataset.setup('fit')

    for batch in shapenet_dataset.train_dataloader():

        x, y = batch

