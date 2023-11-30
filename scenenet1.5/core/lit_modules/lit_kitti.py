





import pytorch_lightning as pl
from torch.utils.data import DataLoader


import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from pointcept.datasets.semantic_kitti import SemanticKITTIDataset
from core.lit_modules.preprocessed_wrapper import Dataset_Preprocessed


import pointcept.datasets.transform as pc_trans
from pointcept.datasets.scannet import Collect, Compress
from core.datasets.torch_transforms import Farthest_Point_Sampling, Voxelization_withPCD, ToDevice
import os
from torchvision.transforms import Compose
from tqdm import tqdm
import torch
import concurrent.futures as futures

# def process_sample(sample, i, folder_path):
#     print(f"Processing sample {i}: {type(sample)}, saving to {folder_path}")
#     transform = [
#         pc_trans.NormalizeCoord(),
#         Collect(['coord', 'strength', 'segment']),
#         pc_trans.ToTensor(),
#         Compress(),
#         ToDevice(),
#         Farthest_Point_Sampling(50000),
#         Voxelization_withPCD('all', (0.01, 0.01, 0.01), None)
#     ]
#     sample = Compose(transform)(sample)
#     x, y, pt_locs = sample
#     sample = [x, y, pt_locs]
#     print(x.shape, y.shape, pt_locs.shape)
#     sample_path = os.path.join(folder_path, f"sample_{i}.pt")
#     torch.save(sample, sample_path)
#     print(f"Saved sample {i} to {sample_path}")


# def save_preprocessed_data(data_dir, save_dir, vxg_size, vox_size):
#     data_split = ['train', 'val', 'test']


#     for folder in data_split:
#         folder_path = os.path.join(save_dir, folder)
#         os.makedirs(folder_path, exist_ok=True)

#         dm = SemanticKITTIDataset(split=folder, data_root=data_dir, transform=None, test_mode=False, test_cfg=None, loop=1)

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
    
    data_split = ['val']


    transform = [
        pc_trans.NormalizeCoord(),
        Collect(['coord', 'strength', 'segment']),
        pc_trans.ToTensor(),
        Compress(),
        ToDevice(),
        Farthest_Point_Sampling(50000),
        Voxelization_withPCD('all', (0.01, 0.01, 0.01), None)
    ]

    num_samples = 1

    dm = Lit_KITTI(data_dir, batch_size=num_samples, num_workers=0, transform=transform)

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
        counter = 0
        for batch in tqdm(data_loader, desc=f"Saving {folder} data..."):
            if counter*len(batch[0]) <= folder_count:
                print(f"Skipping batch {counter}")
                counter += len(batch[0])
            else:
                print(f"Commencing batch: [{counter*len(batch[0])}, {(counter+1)*len(batch[0])}]")
                x, y, pt_locs = batch
                for i in range(len(batch[0])):
                    print(f"Processing sample {counter}: {type(batch[0])}, saving to {folder_path}")
                    print(f"Vox shape: {x[i].shape}, label shape: {y[i].shape}, pt_locs shape: {pt_locs[i].shape}")
                    sample = [x[i], y[i], pt_locs[i]] 
                    sample_path = os.path.join(folder_path, f"sample_{counter}.pt")
                    torch.save(sample, sample_path)
                    counter += 1

        


class Lit_KITTI(pl.LightningDataModule):


    def __init__(self, 
                data_dir, 
                learning_map=None,
                transform=None,
                batch_size=1, 
                num_workers=1, 
                load_into_memory=False,
                **kwargs
            )-> None:
        
        super().__init__()
        self.data_dir = data_dir
        self.learning_map = learning_map
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.load_into_memory = load_into_memory
        self.save_hyperparameters()


    def setup(self, stage: str) -> None:
        
        if stage == 'train' or stage == 'fit':
            self.train_ds = SemanticKITTIDataset(
                split='train',
                data_root=self.data_dir,
                learning_map=self.learning_map,
                transform=self.transform,
                test_mode=False,
                test_cfg=None,
                load_into_memory=self.load_into_memory,
                loop=1,
            )

        if stage == 'val' or stage == 'fit':
            self.val_ds = SemanticKITTIDataset(
                split='val',
                data_root=self.data_dir,
                learning_map=self.learning_map,
                transform=self.transform,
                test_mode=False,
                test_cfg=None,
                load_into_memory=self.load_into_memory,
                loop=1,
            )

        if stage == 'test' or stage == 'predict':
            self.test_ds = SemanticKITTIDataset(
                split='test',
                data_root=self.data_dir,
                learning_map=self.learning_map,
                transform=self.transform,
                test_mode=True,
                test_cfg=None,
                loop=1,
            )
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return self.test_dataloader()
    
    


if __name__ == "__main__":
    import torch
    from scripts.constants import KITTI_PATH
    import pointcept.datasets.transform as pc_trans
    from pointcept.datasets.scannet import Collect, Compress
    from core.datasets.torch_transforms import Farthest_Point_Sampling, Voxelization_withPCD
    import utils.pcd_processing as eda
    import pyaml
    import os


    save_preprocessed_data(KITTI_PATH, os.path.join(KITTI_PATH, 'preprocessed'), None, (0.01, 0.01, 0.01))
    input(F"Continue?")

    transform = [
        pc_trans.NormalizeCoord(),
        Collect(['coord','strength', 'segment']),
        pc_trans.ToTensor(),
        Compress(),
        Farthest_Point_Sampling(50000),
        Voxelization_withPCD('all', (0.01, 0.05, 0.01), None)
    ]

    data_module = Lit_KITTI(KITTI_PATH, batch_size=1, num_workers=10, transform=transform)

    data_module.setup(stage="test")

    dataloader = data_module.test_dataloader()

    for batch in dataloader:
        x, y, pt_locs = batch
        print(x.shape, y.shape, pt_locs.shape)
        print(torch.unique(y))


        pcd = eda.np_to_ply(pt_locs.numpy()[0][:, :3])
        eda.color_pointcloud(pcd, y.numpy()[0])
        # eda.color_pointcloud(pcd, None, x.numpy()[0][:, 6:9])  # Color the point cloud the color channel
        eda.visualize_ply([pcd])

        # print(batch.keys())

        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"{key}: {value.shape}")

        input("Continue?")
