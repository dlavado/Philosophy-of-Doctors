


import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from pointcept.datasets.s3dis import S3DISDataset
from core.lit_modules.preprocessed_wrapper import Dataset_Preprocessed


AREA_LIST = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]

AREA_SPLIT = {
    "train": AREA_LIST[:4],
    "val": AREA_LIST[4:5],
    "test": AREA_LIST[5:],
}



def save_preprocessed_data(data_dir, save_dir, vxg_size, vox_size):
    """
    Saves preprocessed data to save_dir
    """
    import pointcept.datasets.transform as pc_trans
    from pointcept.datasets.scannet import Collect, Compress
    from core.datasets.torch_transforms import Farthest_Point_Sampling, Voxelization_withPCD, ToDevice
    import os
    from tqdm import tqdm
    
    data_split = ['test']


    transform = [
        pc_trans.NormalizeColor(),
        pc_trans.NormalizeCoord(),
        Collect(['coord', 'normal', 'color', 'segment']),
        pc_trans.ToTensor(),
        Compress(),
        ToDevice(),
        Farthest_Point_Sampling(50000),
        Voxelization_withPCD('all', vox_size, vxg_size)
    ]

    num_samples = 16

    dm = Lit_S3DISDataset(data_dir, batch_size=num_samples, num_workers=0, transform=transform)

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
                    sample = [x[i], y[i], pt_locs[i]] 
                    sample_path = os.path.join(folder_path, f"sample_{counter}.pt")
                    torch.save(sample, sample_path)
                    counter += 1



class Lit_S3DISDataset(pl.LightningDataModule):

    def __init__(self,
                data_root="data/s3dis",
                transform=None,
                test_cfg=None,
                cache=False,
                loop=1,
                load_into_memory=False,
                batch_size=32,
                num_workers=1
            ) -> None:
        super().__init__()

        self.data_root = data_root
        self.transform = transform
        # self.test_mode = test_mode
        self.test_cfg = test_cfg
        self.cache = cache
        self.loop = loop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_into_memory = load_into_memory
        self.save_hyperparameters()


    def setup(self, stage: str) -> None:
        if stage == 'train' or stage == 'fit':
            self.train_ds = S3DISDataset(
                split=AREA_SPLIT["train"],
                data_root=self.data_root,
                transform=self.transform,
                test_mode=False,
                test_cfg=None,
                cache=False,
                load_into_memory=self.load_into_memory,
                loop=1,
            )

        if stage == 'val' or stage == 'fit':
            self.val_ds = S3DISDataset(
                split=AREA_SPLIT["val"],
                data_root=self.data_root,
                transform=self.transform,
                test_mode=False,
                test_cfg=None,
                cache=False,
                load_into_memory=self.load_into_memory,
                loop=1,
            )

        if stage == 'test' or stage == 'predict':
            self.test_ds = S3DISDataset(
                split=AREA_SPLIT["test"],
                data_root=self.data_root,
                transform=self.transform,
                test_mode=True,
                test_cfg=self.test_cfg,
                cache=False,
                loop=1,
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    


if __name__ == "__main__":
    import torch
    import os
    from scripts.constants import S3DIS_PATH, S3DIS_PREPROCESSED_PATH
    import pointcept.datasets.transform as pc_trans
    from pointcept.datasets.scannet import Collect, Compress
    from core.datasets.torch_transforms import Farthest_Point_Sampling, Voxelization_withPCD
    import utils.pcd_processing as eda

    # save_preprocessed_data(S3DIS_PATH, S3DIS_PREPROCESSED_PATH, (64, 64, 64), None)

    transform = [
        pc_trans.NormalizeColor(),
        pc_trans.NormalizeCoord(),
        Collect(['coord', 'normal', 'color', 'segment']),
        pc_trans.ToTensor(),
        Compress(),
        Farthest_Point_Sampling(50000),
        Voxelization_withPCD('all', None, (64, 64, 64))
    ]

    data_module = Lit_S3DISDataset(S3DIS_PATH, batch_size=1, num_workers=1, transform=transform)

    data_module.setup(stage="test")

    dataloader = data_module.test_dataloader()

    for batch in dataloader:
        x, y, pt_locs = batch
        print(x.shape, y.shape, pt_locs.shape)
        print(torch.unique(y))

        pcd = eda.np_to_ply(x.numpy()[0][:, :3])
        eda.color_pointcloud(pcd, y.numpy()[0])
        # eda.color_pointcloud(pcd, None, x.numpy()[0][:, 6:9])  # Color the point cloud the color channel
        eda.visualize_ply([pcd])

        # print(batch.keys())

        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"{key}: {value.shape}")

        input("Continue?")


