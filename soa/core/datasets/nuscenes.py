"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from collections.abc import Sequence
import pickle

from core.datasets.defaults import DefaultDataset


class NuScenesDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="data/nuscenes",
        sweeps=10,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
    ):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])

            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map
    
    
    
if __name__ == '__main__':
    
    dataset_type = "NuScenesDataset"
    # data_root = "data/nuscenes"
    data_root = "/media/didi/PortableSSD/nuscenes"
    ignore_index = -1
    names = [
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    ]

    data = dict(
        num_classes=16,
        ignore_index=ignore_index,
        names=names,
        train=dict(
            # type=dataset_type,
            split=["train", "val"],
            data_root=data_root,
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
            ],
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
        ),
    )



    #### config
    
    # from core.models.pointcept.pointcept.engines.defaults import default_config_parser
    from core.models.pointcept.pointcept.utils.config import Config
    
    data_config = Config(data)
    
    
    nuscenes = NuScenesDataset(
       **data_config.train
    )
    
    
    sample_dict = nuscenes.get_data(0)
    
    print(sample_dict.keys())
    
    for key in sample_dict.keys():
        print(key, sample_dict[key].shape)
        
        if key == 'segment':
            print(np.unique(sample_dict[key]))
            
            
            
    nuscenes_test = NuScenesDataset(
         **data_config.test
    )
    
    sample_dict = nuscenes_test.get_data(0)
    
    print(sample_dict.keys())
    
    for key in sample_dict.keys():
        print(key, sample_dict[key].shape, sample_dict[key].dtype)
        
        if key == 'segment':
            print(np.unique(sample_dict[key]))
    
    