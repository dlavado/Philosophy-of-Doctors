

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys
sys.path.append('../..')
from core.datasets.semantic_kitti import SemanticKITTIDataset
from core.datasets.utils import point_collate_fn
from core.models.pointcept.pointcept.utils.config import Config



class LitSemanticKITTI(pl.LightningDataModule):
    
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
        self.transform = transform
        self.loop = loop
        self.num_workers = num_workers
        self.ignore_index = ignore_index
        
        
        self.data_config = Config(
                dict(
                    num_classes=19,
                    ignore_index=ignore_index,
                    names=[
                        "car",
                        "bicycle",
                        "motorcycle",
                        "truck",
                        "other-vehicle",
                        "person",
                        "bicyclist",
                        "motorcyclist",
                        "road",
                        "parking",
                        "sidewalk",
                        "other-ground",
                        "building",
                        "fence",
                        "vegetation",
                        "trunk",
                        "terrain",
                        "pole",
                        "traffic-sign",
                    ],
                    train=dict(
                        # type=dataset_type,
                        # split=["train", "val"],
                        data_root=data_root,
                        transform=[
                            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                            dict(type="RandomScale", scale=[0.9, 1.1]),
                            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                            dict(type="RandomFlip", p=0.5),
                            dict(type="RandomJitter", sigma=0.005, clip=0.02),
                            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                            dict(
                                type="GridSample",
                                grid_size=0.05,
                                hash_type="fnv",
                                mode="train",
                                keys=("coord", "strength", "segment"),
                                return_grid_coord=True,
                            ),
                            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
                            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
                            dict(type="SphereCrop", point_max=120000, mode="random"),
                            # dict(type="CenterShift", apply_z=False),
                            dict(type="ToTensor"),
                            dict(
                                type="Collect",
                                keys=("coord", "grid_coord", "segment"),
                                feat_keys=("coord", "strength"),
                            ),
                        ] if transform is None else transform,
                        test_mode=False,
                        ignore_index=ignore_index,
                    ),
                    test=dict(
                        # type=dataset_type,
                        split="test",
                        data_root=data_root,
                        transform=[],
                        test_mode=True,
                        test_cfg=dict(
                            voxelize=dict(
                                type="GridSample",
                                grid_size=0.05,
                                hash_type="fnv",
                                mode="test",
                                return_grid_coord=True,
                                keys=("coord", "strength"),
                            ),
                            crop=None,
                            post_transform=[
                                dict(
                                    type="PointClip",
                                    point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                                ),
                                dict(type="ToTensor"),
                                dict(
                                    type="Collect",
                                    keys=("coord", "grid_coord", "index"),
                                    feat_keys=("coord", "strength"),
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
                            ],
                        ),
                        ignore_index=ignore_index,
                    ),
                )
        )
        
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SemanticKITTIDataset(
                            **self.data_config.train,
                            split='train',
                        )
            
            self.val_dataset = SemanticKITTIDataset(
                            **self.data_config.train,
                            split='val',
                        )
            
        if stage == 'test' or stage is None:
            self.test_dataset = SemanticKITTIDataset(
                            **self.data_config.test
                        )     
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=point_collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=point_collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=point_collate_fn)
    
    
    
if __name__ == '__main__':
    
    import os
    from utils import constants as C
    
    
    lit_semkitti = LitSemanticKITTI(
                    data_root=os.path.join(C.ROOT_PROJECT, 'data', 'semantic_kitti'),
                    batch_size=12,
                )
    
    lit_semkitti.setup()
    
    train_dl = lit_semkitti.train_dataloader()
    
    for batch in train_dl:
        print(type(batch))
        print(batch.keys())
        for k, v in batch.items():
            print(k, v.shape)
        break