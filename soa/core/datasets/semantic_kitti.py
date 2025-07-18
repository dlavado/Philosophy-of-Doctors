"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np

from core.datasets.defaults import DefaultDataset

class SemanticKITTIDataset(DefaultDataset):
    def __init__(
        self,
        split="train",
        data_root="data/semantic_kitti",
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
    ):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        # split2seq = dict(
        #     train=[0, 1, 2, 3, 4, 5, 6, 7],
        #     val=[8],
        #     test=[9, 10],
        # )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
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

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv
    
    
    
if __name__ == '__main__':
    
    
    dataset_type = "SemanticKITTIDataset"
    # data_root = "data/semantic_kitti"
    data_root = "/media/didi/PortableSSD/semkitti"
    ignore_index = -1
    names = [
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
    ]

    data = dict(
        num_classes=19,
        ignore_index=ignore_index,
        names=names,
        train=dict(
            # type=dataset_type,
            split=["train", "val"],
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
    
    
    
    from core.models.pointcept.pointcept.utils.config import Config
    
    data_config = Config(data)
    
    kitti = SemanticKITTIDataset(
        **data_config.train
    )
    
    sample_dict = kitti.get_data(0)
    
    print(sample_dict.keys())
    
    for key in sample_dict.keys():
        print(key, sample_dict[key].shape)
        
        if key == 'segment':
            print(np.unique(sample_dict[key]))
            
    kitti_test = SemanticKITTIDataset(
        **data_config.test
    )
    
    sample_dict = kitti_test.get_data(0)
    
    print(sample_dict.keys())
    
    for key in sample_dict.keys():
        print(key, sample_dict[key].shape)
        
        if key == 'segment':
            print(np.unique(sample_dict[key]))
    
