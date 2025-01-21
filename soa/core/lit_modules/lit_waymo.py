
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys
sys.path.append('../..')
from core.datasets.waymo import WaymoDataset
from core.datasets.utils import point_collate_fn
from core.models.pointcept.pointcept.utils.config import Config


class LitWaymo(pl.LightningDataModule):
    
    def __init__(self,
                 data_root,
                 split="train", 
                 batch_size=12,
                 transform=None,
                 loop=1,
                 ignore_index=-1,
                 num_workers=8, 
            ):
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.batch_size = batch_size
        self.transform = transform
        self.test_mode = split == "test"
        self.loop = loop
        self.num_workers = num_workers
        
        names = [
            "Car",
            "Truck",
            "Bus",
            # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
            "Other Vehicle",
            "Motorcyclist",
            "Bicyclist",
            "Pedestrian",
            "Sign",
            "Traffic Light",
            # Lamp post, traffic sign pole etc.
            "Pole",
            # Construction cone/pole.
            "Construction Cone",
            "Bicycle",
            "Motorcycle",
            "Building",
            # Bushes, tree branches, tall grasses, flowers etc.
            "Vegetation",
            "Tree Trunk",
            # Curb on the edge of roads. This does not include road boundaries if there’s no curb.
            "Curb",
            # Surface a vehicle could drive on. This includes the driveway connecting
            # parking lot and road over a section of sidewalk.
            "Road",
            # Marking on the road that’s specifically for defining lanes such as
            # single/double white/yellow lines.
            "Lane Marker",
            # Marking on the road other than lane markers, bumps, cateyes, railtracks etc.
            "Other Ground",
            # Most horizontal surface that’s not drivable, e.g. grassy hill, pedestrian walkway stairs etc.
            "Walkable",
            # Nicely paved walkable surface when pedestrians most likely to walk on.
            "Sidewalk",
        ]
        self.data_config = Config(
                    dict(
                num_classes=22,
                ignore_index=ignore_index,
                names=names,
                train=dict(
                    #type=dataset_type,
                    split="training",
                    data_root=data_root,
                    transform=[
                        # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                        # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                        dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                        # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                        # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                        dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
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
                        # dict(type="SphereCrop", point_max=1000000, mode="random"),
                        # dict(type="CenterShift", apply_z=False),
                        dict(type="ToTensor"),
                        dict(
                            type="Collect",
                            keys=("coord", "grid_coord", "segment"),
                            feat_keys=("coord", "strength"),
                        ),
                    ],
                    test_mode=False,
                    ignore_index=ignore_index,
                ),
                val=dict(
                    # type=dataset_type,
                    split="validation",
                    data_root=data_root,
                    transform=[
                        dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
                        dict(
                            type="GridSample",
                            grid_size=0.05,
                            hash_type="fnv",
                            mode="train",
                            keys=("coord", "strength", "segment"),
                            return_grid_coord=True,
                        ),
                        dict(type="ToTensor"),
                        dict(
                            type="Collect",
                            keys=("coord", "grid_coord", "segment"),
                            feat_keys=("coord", "strength"),
                        ),
                    ],
                    test_mode=False,
                    ignore_index=ignore_index,
                ),
                test=dict(
                    # type=dataset_type,
                    split="validation",
                    data_root=data_root,
                    transform=[
                        dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
                    ],
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
        
        if stage == "fit" or stage is None:
            self.train_dataset = WaymoDataset(**self.data_config.train)
            self.val_dataset = WaymoDataset(**self.data_config.val)
            
        if stage == "test" or stage is None:
            self.test_dataset = WaymoDataset(**self.data_config.test)
            
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          collate_fn=point_collate_fn)
        
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          collate_fn=point_collate_fn)
        
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers, 
                          collate_fn=point_collate_fn)
        
        
        
        
if __name__ == '__main__':
    
    import os
    from utils import constants as C
    
    
    lit_waymo = LitWaymo(data_root=os.path.join(C.get_project_root(), 'data/waymo'), 
                         split="train", 
                         batch_size=4, 
                         transform=None, 
                         loop=1, 
                         ignore_index=-1, 
                         num_workers=1, 
                        )
    
    lit_waymo.setup('fit')
    
    train_dl = lit_waymo.train_dataloader()
    for batch in train_dl:
        print(type(batch))
        print(batch.keys())
        for k, v in batch.items():
            print(k, v.shape)
        break
        
    