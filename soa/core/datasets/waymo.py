"""
Waymo dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob

import sys
sys.path.append("../..")

from core.datasets.defaults import DefaultDataset


class WaymoDataset(DefaultDataset):
    def __init__(
        self,
        split="training",
        data_root="data/waymo",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            self.split = [self.split]

        data_list = []
        for split in self.split:
            data_list += glob.glob(
                os.path.join(self.data_root, split, "*", "velodyne", "*.bin")
            )
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = np.tanh(scan[:, -1].reshape([-1, 1]))

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = (
                    np.fromfile(a, dtype=np.int32).reshape(-1, 2)[:, 1] - 1
                )  # ignore_index 0 -> -1
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name
    
    
    
if __name__ == '__main__':
    
    dataset_type = "WaymoDataset"
    # data_root = "data/waymo"
    data_root = "/media/didi/PortableSSD/waymo/converted"
    ignore_index = -1
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

    data = dict(
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
    
    
    from core.models.pointcept.pointcept.utils.config import Config
    
    data_config = Config(data)
    
    
    waymo = WaymoDataset(
        **data_config.train
    )
    
    
    sample_dict = waymo.get_data(0)
    
    print(sample_dict.keys())
    
    for key in sample_dict.keys():
        print(key, sample_dict[key].shape, sample_dict[key].dtype)
        
        if key == 'segment':
            print(np.unique(sample_dict[key]))

