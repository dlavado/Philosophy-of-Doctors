

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys
sys.path.append('../..')
from core.datasets.nuscenes import NuScenesDataset
from core.datasets.utils import point_collate_fn
from core.models.pointcept.pointcept.utils.config import Config


class LitNuScenes(pl.LightningDataModule):
    
    def __init__(self, 
                 data_dir,
                 batch_size=12,
                 transform=None,
                 loop=1,
                 ignore_index=-1,
                 num_workers=8, 
            ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.loop = loop
        self.num_workers = num_workers
        
        
        self.train_config=dict(
            # type=dataset_type,
            # split=["train", "val"],
            data_root=self.data_dir,
            transform=[
                # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
                dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='x', p=0.5),
                # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='y', p=0.5),
                dict(type="RandomScale", scale=[0.9, 1.1]),
                # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                dict(type="RandomFlip", p=0.5),
                dict(type="RandomJitter", sigma=0.005, clip=0.02),
                # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                # dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
                #      keys=("coord", "strength", "segment"), return_grid_coord=True),
                # dict(type="SphereCrop", point_max=1000000, mode="random"),
                # dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "segment"),
                    feat_keys=("coord", "strength"),
                ),
            ] if self.transform is None else self.transform,
            test_mode=False,
            ignore_index=ignore_index,
        )
        
        self.test_config=dict(
            # type=dataset_type,
            split="test",
            data_root=self.data_dir,
            transform=[],
            test_mode=True,
            test_cfg=dict(
                voxelize=None,
                crop=None,
                post_transform=[
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "index"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                aug_transform=[
                    [dict(type="RandomScale", scale=[0.9, 0.9])],
                    [dict(type="RandomScale", scale=[0.95, 0.95])],
                    [dict(type="RandomScale", scale=[1, 1])],
                    [dict(type="RandomScale", scale=[1.05, 1.05])],
                    [dict(type="RandomScale", scale=[1.1, 1.1])],
                    [
                        dict(type="RandomScale", scale=[0.9, 0.9]),
                        dict(type="RandomFlip", p=1),
                    ],
                    [
                        dict(type="RandomScale", scale=[0.95, 0.95]),
                        dict(type="RandomFlip", p=1),
                    ],
                    [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                    [
                        dict(type="RandomScale", scale=[1.05, 1.05]),
                        dict(type="RandomFlip", p=1),
                    ],
                    [
                        dict(type="RandomScale", scale=[1.1, 1.1]),
                        dict(type="RandomFlip", p=1),
                    ],
                ],
            ),
            ignore_index=ignore_index,
        )
        
        self.train_config = Config(self.train_config)
        self.test_config = Config(self.test_config)
        
              
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_ds = NuScenesDataset(**self.train_config, split="train")
            self.val_ds = NuScenesDataset(**self.train_config, split="val")
        if stage == 'test':
            self.test_ds = NuScenesDataset(**self.test_config)
            
        if stage == 'predict':
            self.predict_ds = NuScenesDataset(**self.test_config)
            
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, collate_fn=point_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=point_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=point_collate_fn)
    
    
    
    
    
if __name__ == '__main__':
    import os
    from utils import constants as C
    
    lit_nuscenes = LitNuScenes(data_dir=os.path.join(C.get_project_root(), 'data/nuscenes'), 
                               batch_size=12, 
                               transform=None, 
                               loop=1, 
                               ignore_index=-1, 
                               num_workers=1, 
                            )
    
    
    lit_nuscenes.setup('fit')
    
    train_dl = lit_nuscenes.train_dataloader()
    
    for batch in train_dl:
        print(type(batch))
        print(batch.keys())
        for k, v in batch.items():
            print(k, v.shape)
        break
        
        