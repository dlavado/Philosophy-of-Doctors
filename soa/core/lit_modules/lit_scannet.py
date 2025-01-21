

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys
sys.path.append('../..')
from core.datasets.scannet import ScanNetDataset
from core.datasets.utils import point_collate_fn
from core.models.pointcept.pointcept.utils.config import Config


class LitScanNet(pl.LightningDataModule):
    
    def __init__(
            self,
            data_root,
            batch_size=12,
            transform=None,
            loop=1,
            num_workers=8,
            ignore_index=-1,
        ):
        
        
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.loop = loop
        self.num_workers = num_workers
        
        self.data_config = Config(
                dict(
                    num_classes=20,
                    ignore_index=ignore_index,
                    names=[
                        "wall",
                        "floor",
                        "cabinet",
                        "bed",
                        "chair",
                        "sofa",
                        "table",
                        "door",
                        "window",
                        "bookshelf",
                        "picture",
                        "counter",
                        "desk",
                        "curtain",
                        "refridgerator",
                        "shower curtain",
                        "toilet",
                        "sink",
                        "bathtub",
                        "otherfurniture",
                    ],
                    train=dict(
                        # type=dataset_type,
                        # split=["train", "val"],
                        data_root=data_root,
                        transform=[
                            dict(type="CenterShift", apply_z=True),
                            dict(
                                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
                            ),
                            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                            dict(type="RandomScale", scale=[0.9, 1.1]),
                            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                            dict(type="RandomFlip", p=0.5),
                            dict(type="RandomJitter", sigma=0.005, clip=0.02),
                            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                            dict(type="ChromaticJitter", p=0.95, std=0.05),
                            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                            dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="train",
                                return_min_coord=True,
                            ),
                            dict(type="SphereCrop", point_max=100000, mode="random"),
                            dict(type="CenterShift", apply_z=False),
                            dict(type="NormalizeColor"),
                            dict(type="ShufflePoint"),
                            dict(type="ToTensor"),
                            dict(
                                type="Collect",
                                keys=("coord", "segment"),
                                feat_keys=("coord", "color", "normal"),
                            ),
                        ] if transform is None else transform,
                        test_mode=False,
                    ),
                    test=dict(
                        #type=dataset_type,
                        split="test",
                        data_root=data_root,
                        transform=[
                            dict(type="CenterShift", apply_z=True),
                            dict(type="NormalizeColor"),
                        ],
                        test_mode=True,
                        test_cfg=dict(
                            voxelize=dict(
                                type="GridSample",
                                grid_size=0.02,
                                hash_type="fnv",
                                mode="test",
                                keys=("coord", "color", "normal"),
                            ),
                            crop=None,
                            post_transform=[
                                dict(type="CenterShift", apply_z=False),
                                dict(type="ToTensor"),
                                dict(
                                    type="Collect",
                                    keys=("coord", "index"),
                                    feat_keys=("coord", "color", "normal"),
                                ),
                            ],
                            aug_transform=[
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[0],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    )
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[1 / 2],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    )
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[1],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    )
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[3 / 2],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    )
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[0],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    ),
                                    dict(type="RandomScale", scale=[0.95, 0.95]),
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[1 / 2],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    ),
                                    dict(type="RandomScale", scale=[0.95, 0.95]),
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[1],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    ),
                                    dict(type="RandomScale", scale=[0.95, 0.95]),
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[3 / 2],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    ),
                                    dict(type="RandomScale", scale=[0.95, 0.95]),
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[0],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    ),
                                    dict(type="RandomScale", scale=[1.05, 1.05]),
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[1 / 2],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    ),
                                    dict(type="RandomScale", scale=[1.05, 1.05]),
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[1],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    ),
                                    dict(type="RandomScale", scale=[1.05, 1.05]),
                                ],
                                [
                                    dict(
                                        type="RandomRotateTargetAngle",
                                        angle=[3 / 2],
                                        axis="z",
                                        center=[0, 0, 0],
                                        p=1,
                                    ),
                                    dict(type="RandomScale", scale=[1.05, 1.05]),
                                ],
                                [dict(type="RandomFlip", p=1)],
                            ],
                        ),
                    ),
                )
        )
        
        
        
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ScanNetDataset(**self.data_config['train'], split='train')
            self.val_dataset = ScanNetDataset(**self.data_config['train'], split='val')
        if stage == 'test' or stage is None:
            self.test_dataset = ScanNetDataset(**self.data_config['test'])
            
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=point_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=point_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=point_collate_fn)
    
    
    
if __name__ == '__main__':
    import os
    from utils import constants as C
    
    lit_scannet = LitScanNet(data_root=os.path.join(C.ROOT_PROJECT, 'data', 'scannet'),
                             batch_size=2
                            )
    
    
    lit_scannet.setup()
    
    train_dl = lit_scannet.train_dataloader()
    
    for batch in train_dl:
        print(type(batch))
        print(batch.keys())
        for k, v in batch.items():
            print(k, v.shape)
        break
        
                            
                    
    
    