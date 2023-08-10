
from typing import Dict, List
from torchvision.transforms import Compose
import torch
from torch.utils.data import Dataset
from torch import nn as nn
import numpy as np
import os
import h5py

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
sys.path.append("../..")

from scripts.constants import PARTNET_PATH



class PartNetDataset(Dataset):


    def __init__(self, 
                 data_dir:str,
                 coarse_level:int=1,
                 keep_objects:List[str] = None,
                 transform:Compose = None,
                 stage:str = 'train'
                 ) -> None:

        super(PartNetDataset, self).__init__()


        self.partnet_dir = data_dir
        self.transform = transform

        categories = os.listdir(self.partnet_dir)
        # Filter out categories that are not at the coarse level specified
        categories = [cat_folder for cat_folder in categories if f"-{coarse_level}" in cat_folder] 

        if keep_objects:
            keep_objects = [obj.lower() for obj in keep_objects]
            categories = [cat_folder for cat_folder in categories if cat_folder.split("-")[0].lower() in keep_objects]

        self.h5_obj_train = []
        self.h5_obj_val = []
        self.h5_obj_test = []

        for cat_folder in categories:
            object_category = cat_folder.split("-")[0]
            cat_dir = os.path.join(self.partnet_dir, cat_folder)
            h5_files = os.listdir(cat_dir)
            h5_files = [h5_file for h5_file in h5_files if ".h5" in h5_file]

            for h5_file in h5_files:
                h5_path = os.path.join(cat_dir, h5_file)
                cat_objects = h5py.File(h5_path, 'r')
                samples = (object_category, cat_objects)
                if "train" in h5_file and stage == 'train':
                    self.h5_obj_train.append(samples)
                elif "val" in h5_file and stage == 'val':
                    self.h5_obj_val.append(samples)
                elif "test" in h5_file and stage == 'test':
                    self.h5_obj_test.append(samples)
                
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.stage = stage
        self._setup(stage)
        del self.h5_obj_train
        del self.h5_obj_val
        del self.h5_obj_test

    def __len__(self):
        if self.stage == 'train':
            return len(self.train_dataset)
        elif self.stage == 'val':
            return len(self.val_dataset)
        elif self.stage == 'test':
            return len(self.test_dataset)
                
    def _setup(self, stage:str=None):
        if stage == 'test':
            self._test_setup()

        elif stage == 'train':
            self._train_setup()
        
        elif stage == 'val':
            self._val_setup()
            

    def _train_setup(self):

        for category, h5_obj in self.h5_obj_train:
            
            # obj shape = (B, N, 3) ; seg_labels shape = (B, N)
            obj = np.array(h5_obj['data'])
            seg_labels = np.array(h5_obj['label_seg'])

            # assert obj.shape[0] == seg_labels.shape[0]   
            
            samples = [(category, obj[i], seg_labels[i]) for i in range(obj.shape[0])]        

            if self.train_dataset is None:
                self.train_dataset = samples
            else:
                self.train_dataset += samples
            

    def _val_setup(self):

        for category, h5_obj in self.h5_obj_val:

             # obj shape = (B, N, 3) ; seg_labels shape = (B, N)
            obj, seg_labels = h5_obj['data'], h5_obj['label_seg']

            # assert obj.shape[0] == seg_labels.shape[0]
            samples = [(category, obj[i], seg_labels[i]) for i in range(obj.shape[0])]

            if self.val_dataset is None:
                self.val_dataset = samples
            else:
                self.val_dataset += samples

    def _test_setup(self):

        for category, h5_obj in self.h5_obj_test:
             # obj shape = (B, N, 3) ; seg_labels shape = (B, N)
            obj, seg_labels = h5_obj['data'], h5_obj['label_seg']

            # assert obj.shape[0] == seg_labels.shape[0]

            samples = [(category, obj[i], seg_labels[i]) for i in range(obj.shape[0])]

            if self.test_dataset is None:
                self.test_dataset = samples
            else:
                self.test_dataset += samples
   

    def __getitem__(self, idx:int) -> Dict[str, torch.Tensor]:

        if self.stage == 'train':
            category, obj, seg_labels = self.train_dataset[idx]
        elif self.stage == 'val':
            category, obj, seg_labels = self.val_dataset[idx]
        elif self.stage == 'test':
            category, obj, seg_labels = self.test_dataset[idx]
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

        sample = {
            'category': category,
            'obj': torch.from_numpy(obj),
            'seg_labels': torch.from_numpy(seg_labels)
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


class LitPartNetDataset(pl.LightningDataModule):


    def __init__(self, data_dir:str, coarse_level:int, batch_size, transform=None, keep_objects=None, num_workers=8) -> None:
        
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transforms = transform

        self.keep_objects = keep_objects

        self.num_workers = num_workers
        self.coarse_level = coarse_level


    def setup(self, stage:str=None):

        if stage == 'fit' or stage is None:
            self.train_dataset = PartNetDataset(data_dir=self.data_dir, coarse_level=self.coarse_level, keep_objects=self.keep_objects, transform=self.transforms, stage='train')
            self.val_dataset = PartNetDataset(data_dir=self.data_dir, coarse_level=self.coarse_level, keep_objects=self.keep_objects, transform=self.transforms, stage='val')
        elif stage == 'test' or stage  == 'predict':
            self.test_dataset = PartNetDataset(data_dir=self.data_dir, coarse_level=self.coarse_level, keep_objects=self.keep_objects, transform=self.transforms, stage='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)




if __name__ == '__main__':
    import h5py
    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    sys.path.insert(0, '..')
    sys.path.insert(1, '../..')
    from utils import pcd_processing as eda

    coarse = 2

    # partnet = PartNetDataset(data_dir=PARTNET_PATH, coarse_level=coarse, keep_objects=['chair'], stage='train')

    # a = np.empty((0,))
    # for i in range(len(partnet)):
    #     sample = partnet[i]
    #     a = np.concatenate((a, sample['seg_labels'].numpy().flatten()))

    # # print class densities
    # print(np.unique(a, return_counts=True))

    # # #plot an histogram with class densities
    # # plt.hist(a, bins=50)
    # # plt.title(f"Class distribution in PartNet coarse level {coarse}")
    # # plt.show()


    # sample = partnet[0]
    # pcd = eda.np_to_ply(sample['obj'].numpy())
    # eda.color_pointcloud(pcd, sample['seg_labels'].numpy())
    # eda.visualize_ply([pcd])

    # input("Press Enter to continue...")
    
    
    categories = list(os.listdir(PARTNET_PATH))
    categories = [cat for cat in categories if f"-{coarse}" in cat]
    categories = [os.path.join(PARTNET_PATH, cat) for cat in categories]

    for cat_path in categories:
        
        print("\n")
        print(cat_path)

        for h5_file in os.listdir(cat_path):
            if '.h5' not in h5_file:
                continue    

            h5_path = os.path.join(cat_path, h5_file)

            print(h5_path)

            with h5py.File(h5_path, 'r') as h5:
                
                print(h5.keys())
                # print(h5["data_num"])
                # print(h5["data"].shape)
                # print(h5["label_seg"].shape)
                print(np.unique(h5["label_seg"]))

                # input("Press Enter to continue...")

                # pcd = eda.np_to_ply(h5["data"][0])
                # eda.color_pointcloud(pcd, h5["label_seg"][0])
                
                # eda.visualize_ply([pcd])

