
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class Dataset_Preprocessed(Dataset):
    """
    This class is a wrapper for preprocessed datasets. It is used in the classes.

    Essentially, it loads the preprocessed data from the disk and returns it as a sample.
    This preprocess follows a specific format according to the training of SceneNet, i.e.,
    the data is saved as a tuple of (voxel, label, point locations), with shapes: (1, vxg_size), (1, N), (1, N, 3)
    vxg_size is usually set to 64^3, and N is the number of points in the point cloud.
    """

    def __init__(self, dataset_path, split='fit', load_into_memory=False) -> None:

        self.dataset_path = os.path.join(dataset_path, split)
        self.data_files = [os.path.join(self.dataset_path, sample) for sample in os.listdir(self.dataset_path) if sample.endswith('.pt')]

        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True

    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    def __getitem__(self, idx):
        # data[i]

        if self.load_into_memory:
            return self.data[idx]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        pt_path = self.data_files[idx]
        try:
            pt = torch.load(pt_path)
        except Exception as e:
            print(e)
            print(f"Unreadable file: {pt_path}")
        
        sample = [pt[0].to(torch.float64).cpu(), 
                  pt[1].to(torch.long).cpu(), 
                  pt[2].to(torch.float64).cpu()
            ] # voxel (1, 64, 64, 64); label (1, N); point locations (1, N, 3)
        
        if sample[0].ndim == 3: # if voxel is 3 dimensional, add the channel dimension
            sample[0] = sample[0].unsqueeze(0)
        
        # # print sample device
        # if str(sample[0].device )== 'cpu':
        #     sample = [sample[0].cuda(), sample[1].cuda(), sample[2].cuda()]
        return sample
    
    def __len__(self):
        return len(self.data_files)
    
    

class Lit_Dataset_Preprocessed(pl.LightningDataModule):

    def __init__(self, dataset_path, dataset_name='', load_into_memory=False, batch_size=32, num_workers=1) -> None:
        super().__init__()
        self.data_path = dataset_path
        self.dataset_name = dataset_name
        self.load_into_memory = load_into_memory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == 'train' or stage == 'fit':
            self.train_ds = Dataset_Preprocessed(
                self.data_path, split='train', load_into_memory=self.load_into_memory
            )
        
        if stage == 'val' or stage == 'fit':
            self.val_ds = Dataset_Preprocessed(
                self.data_path, split='val', load_into_memory=self.load_into_memory
            )

        if stage == 'test' or stage == 'predict':
            self.test_ds = Dataset_Preprocessed(
                self.data_path, split='test', load_into_memory=False
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)


    def __str__(self) -> str:
        return super().__str__() + f"\nDataset: {self.dataset_name}"