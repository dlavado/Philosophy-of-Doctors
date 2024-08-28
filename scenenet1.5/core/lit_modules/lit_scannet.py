



from typing import Any
import lightning as pl
from torch.utils.data import DataLoader
import os

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from pointcept.datasets.scannet import ScanNetDataset, ScanNet200Dataset
from core.lit_modules.preprocessed_wrapper import Dataset_Preprocessed
import pointcept.datasets.transform as pc_trans
from pointcept.datasets.scannet import Collect, Compress
from core.datasets.torch_transforms import Farthest_Point_Sampling, Voxelization_withPCD, ToDevice
import os
from tqdm import tqdm
from torchvision.transforms import Compose
import concurrent.futures as futures


# def process_sample(sample, i, folder_path):
#     print(f"Processing sample {i}: {type(sample)}, saving to {folder_path}")
#     transform = [
#         pc_trans.NormalizeColor(),
#         pc_trans.NormalizeCoord(),
#         Collect(['coord', 'normal', 'color', 'segment']),
#         pc_trans.ToTensor(),
#         Compress(),
#         Farthest_Point_Sampling(50000),
#         Voxelization_withPCD('all', None, (64, 64, 64))
#     ]
#     sample = Compose(transform)(sample)
#     x, y, pt_locs = sample
#     sample = [x, y, pt_locs]
#     print(x.shape, y.shape, pt_locs.shape)
#     sample_path = os.path.join(folder_path, f"sample_{i}.pt")
#     torch.save(sample, sample_path)
#     print(f"Saved sample {i} to {sample_path}")


# def save_preprocessed_data(data_dir, save_dir, vxg_size, vox_size):
#     data_split = ['val']

#     for folder in data_split:
#         folder_path = os.path.join(save_dir, folder)
#         os.makedirs(folder_path, exist_ok=True)

#         dm = ScanNetDataset(split=folder, data_root=data_dir, transform=None, test_mode=False, test_cfg=None, loop=1)

#         # Get the total number of samples to process
#         num_samples = len(dm)
#         num_processes = 16

#         for run in range(int(num_samples/num_processes)): # Run the loop for the number of times required to process all the samples
#             # running num_sample samples at a time
#             folder_count = len(os.listdir(folder_path))
#             print(f"Processing samples {folder_count} to {min(folder_count + num_processes, num_samples)}")
#             # Set the number of parallel processes to run
#             with futures.ProcessPoolExecutor() as executor:
#                 samples = [dm[i] for i in range(folder_count, min(folder_count + num_processes, num_samples))]
#                 print(f"Processing {len(samples)} samples")
#                 results = executor.map(process_sample, samples, range(folder_count, num_samples), [folder_path] * num_samples)
#                 for result in results:
#                     print(result)

def save_preprocessed_data(data_dir, save_dir, vxg_size, vox_size):
    """
    Saves preprocessed data to save_dir
    """
    import pointcept.datasets.transform as pc_trans
    from pointcept.datasets.scannet import Collect, Compress
    from core.datasets.torch_transforms import Farthest_Point_Sampling, Voxelization_withPCD
    import os
    from tqdm import tqdm
    
    data_split = ['train']


    transform = [
        pc_trans.NormalizeColor(),
        pc_trans.NormalizeCoord(),
        Collect(['coord', 'normal', 'color', 'segment']),
        pc_trans.ToTensor(),
        Compress(),
        # ToDevice(),
        Farthest_Point_Sampling(50000),
        Voxelization_withPCD('all', vox_size, vxg_size)
    ]

    num_samples = 16

    dm = Lit_ScanNetDataset(data_dir, batch_size=num_samples, num_workers=1, transform=transform)

    for folder in data_split:

        folder_path = os.path.join(save_dir, folder)

        dm.setup(folder)
        if folder == 'train':
            data_loader = dm.train_dataloader()
        elif folder == "val":
            data_loader = dm.val_dataloader()
        elif folder == "test":
            data_loader = dm.test_dataloader()

        os.makedirs(folder_path, exist_ok=True)

        # get folder count
        folder_count = len(os.listdir(folder_path))
        counter = folder_count
        for b, batch in tqdm(enumerate(data_loader), desc=f"Saving Preprocessed {folder} samples..."):
            if b*len(batch[0]) < folder_count:
                continue # Skip the batches that have already been processed
            print(f"Commencing batch: [{counter*len(batch[0])}, {(counter+1)*len(batch[0])}]")
            x, y, pt_locs = batch
            for i in range(len(batch[0])):
                print(f"Processing sample {i}: {type(batch)}, saving to {folder_path}")
                sample = [x[i], y[i], pt_locs[i]] 
                sample_path = os.path.join(folder_path, f"sample_{counter}.pt")
                torch.save(sample, sample_path)
                counter += 1



class Lit_ScanNetDataset(pl.LightningDataModule):


    def __init__(self, data_path, 
                transform=None,
                lr_file=None,
                la_file=None,
                ignore_index=-1,
                test_cfg=None,
                cache=False,
                loop=1,
                scannet_200=False,
                load_into_memory=False,
                batch_size=32,
                num_workers=1) -> None:
        
        super().__init__()

        self.data_path = data_path
        self.transform = transform
        self.lr_file = lr_file
        self.la_file = la_file
        self.ignore_index = ignore_index
        self.test_cfg = test_cfg
        self.cache = cache
        self.loop = loop
        self.load_into_memory = load_into_memory
        self.dataset_class = ScanNet200Dataset if scannet_200 else ScanNetDataset
        self.save_hyperparameters()

    
    def setup(self, stage: str) -> None:
        if stage == 'train' or stage == 'fit':

            self.train_ds = self.dataset_class(
                split='train',
                data_root=self.data_path,
                transform=self.transform,
                lr_file=self.lr_file,
                la_file=self.la_file,
                ignore_index=self.ignore_index,
                test_mode=False,
                test_cfg=None,
                cache=self.cache,
                load_into_memory=self.load_into_memory,
                loop=self.loop
            )

        if stage == 'val' or stage == 'fit':

            self.val_ds = self.dataset_class(
                split='val',
                data_root=self.data_path,
                transform=self.transform,
                lr_file=self.lr_file,
                la_file=self.la_file,
                ignore_index=self.ignore_index,
                test_mode=False,
                test_cfg=None,
                cache=self.cache,
                load_into_memory=self.load_into_memory,
                loop=self.loop
            )

        if stage == 'test' or stage == 'predict':
            self.test_ds = self.dataset_class(
                split='val',
                data_root=self.data_path,
                transform=self.transform,
                lr_file=self.lr_file,
                la_file=self.la_file,
                ignore_index=self.ignore_index,
                test_mode=True,
                test_cfg=self.test_cfg,
                cache=self.cache,
                loop=self.loop
            )


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
       



    

if __name__ == '__main__':
    import torch
    from scripts.constants import SCANNET_PATH
    import pointcept.datasets.transform as pc_trans
    from pointcept.datasets.scannet import Collect, Compress
    from core.datasets.torch_transforms import Farthest_Point_Sampling, Voxelization_withPCD
    import utils.pcd_processing as eda



    # save_preprocessed_data(SCANNET_PATH, os.path.join(SCANNET_PATH, 'preprocessed'), (64, 64, 64), None)
    # input("Continue?")





    transform = [
        pc_trans.NormalizeColor(),
        pc_trans.NormalizeCoord(),
        Collect(['coord', 'normal', 'color', 'segment']),
        pc_trans.ToTensor(),
        Compress(),
        Farthest_Point_Sampling(102400),
        # Voxelization_withPCD('all', None, (64, 64, 64))
    ]

    data_module = Lit_ScanNetDataset(SCANNET_PATH, batch_size=1, num_workers=1, transform=None)

    data_module.setup(stage="train")

    dataloader = data_module.train_dataloader()

    for batch in dataloader:

        print(batch.keys())

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            if key == 'segment':
                print(torch.unique(value))
            
        # x, y = batch

        # print(x.shape, y.shape)
        # pcd = eda.np_to_ply(x.numpy()[0][:, :3])
        # eda.color_pointcloud(pcd, None, x.numpy()[0][:, 6:9])  # Color the point cloud with the ground truth
        # # eda.color_pointcloud(pcd, y.numpy()[0])  # Color the point cloud with the ground truth
        # eda.visualize_ply([pcd])

        input("Continue?")
            

    

    