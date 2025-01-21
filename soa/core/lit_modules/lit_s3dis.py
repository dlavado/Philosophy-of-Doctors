
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys
sys.path.append('../..')
from core.datasets.s3dis import S3DISDataset
from core.datasets.utils import point_collate_fn
from core.models.pointcept.pointcept.utils.config import Config


class LitS3DIS(pl.LightningDataModule):
    
    def __init__(self,
                 data_root,
                 batch_size=12,
                 transform=None,
                 loop=1,
                 ignore_index=-1,
                 num_workers=8
                ):
    
        super().__init__()
        
        self.data_root = data_root
        self.batch_size = batch_size
        self.transform = transform
        self.loop = loop
        self.num_workers = num_workers
        
        self.data_config = dict(
            num_classes=13,
            ignore_index=ignore_index,
            names=[
                "ceiling",
                "floor",
                "wall",
                "beam",
                "column",
                "window",
                "door",
                "table",
                "chair",
                "sofa",
                "bookcase",
                "board",
                "clutter",
            ],
            train=dict(
                # type=dataset_type,
                split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
                data_root=data_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                    # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.04,
                        hash_type="fnv",
                        mode="train",
                        keys=("coord", "color", "segment"),
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=80000, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    # dict(type="ShufflePoint"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment"),
                        feat_keys=["coord", "color"],
                    ),
                ] if self.transform is None else self.transform,
                test_mode=False,
            ),
            val=dict(
                # type=dataset_type,
                split="Area_5",
                data_root=data_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="Copy",
                        keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
                    ),
                    dict(
                        type="GridSample",
                        grid_size=0.04,
                        hash_type="fnv",
                        mode="train",
                        keys=("coord", "color", "segment"),
                        return_grid_coord=True,
                    ),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment"),
                        offset_keys_dict=dict(offset="coord"),
                        feat_keys=["coord", "color"],
                    ),
                ],
                test_mode=False,
            ),
            test=dict(
                # type=dataset_type,
                split="Area_5",
                data_root=data_root,
                transform=[dict(type="CenterShift", apply_z=True), dict(type="NormalizeColor")],
                test_mode=True,
                test_cfg=dict(
                    voxelize=dict(
                        type="GridSample",
                        grid_size=0.04,
                        hash_type="fnv",
                        mode="test",
                        keys=("coord", "color"),
                        return_grid_coord=True,
                    ),
                    crop=None,
                    post_transform=[
                        dict(type="CenterShift", apply_z=False),
                        dict(type="ToTensor"),
                        dict(
                            type="Collect",
                            keys=("coord", "grid_coord", "index"),
                            feat_keys=("coord", "color"),
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
            ),
        )
        
        
        self.data_config = Config(self.data_config)
        
        
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = S3DISDataset(**self.data_config['train'])
            self.val_dataset = S3DISDataset(**self.data_config['val'])
        if stage == 'test' or stage is None:
            self.test_dataset = S3DISDataset(**self.data_config['test'])
            
            
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=point_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=point_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=point_collate_fn)
    
    
    
if __name__ == '__main__':
    import os
    from utils import constants as C
    
    data_module = LitS3DIS(data_root=os.path.join(C.ROOT_PROJECT, 'data', 's3dis'), 
                           batch_size=12)
    
    data_module.setup()
    
    train_dl = data_module.train_dataloader()
    
    for batch in train_dl:
        print(type(batch))
        print(batch.keys())
        for k, v in batch.items():
            print(k, v.shape)
        break
    