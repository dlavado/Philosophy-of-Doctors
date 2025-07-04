"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from core.datasets.transform import Compose, TRANSFORMS


VALID_CLASS_IDS_20 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)

CLASS_LABELS_20 = (
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
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


class ScanNetDataset(Dataset):
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        transform=None,
        lr_file=None,
        la_file=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(ScanNetDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        if lr_file:
            self.data_list = [
                os.path.join(data_root, "train", name + ".pth")
                for name in np.loadtxt(lr_file, dtype=str)
            ]
        else:
            self.data_list = self.get_data_list()
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index

    # def get_data_list(self):
    #     if isinstance(self.split, str):
    #         data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
    #     elif isinstance(self.split, Sequence):
    #         data_list = []
    #         for split in self.split:
    #             data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
    #     else:
    #         raise NotImplementedError
    #     return data_list
    
    
    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = [os.path.join(self.data_root, self.split, name) for name in os.listdir(os.path.join(self.data_root, self.split))]
            # data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                # data_list += glob.glob(os.path.join(self.data_root, split, "*.pth")) 
                data_list += [os.path.join(self.data_root, split, name) for name in os.listdir(os.path.join(self.data_root, split))]
        else:
            raise NotImplementedError
        
        return data_list
    
    def get_sample(self, sample_dir):
        data_dict = {}
        for file in os.listdir(sample_dir):
            file_path = os.path.join(sample_dir, file)
            file_name = file.split('.')[0]
            data_dict[file_name] = np.load(file_path)  
            
        data_dict['scene_id'] = int(sample_dir.split('/')[-1].split('_')[0].replace('scene', ''))
        
        return data_dict
        

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            # data = torch.load(data_path)
            data = self.get_sample(data_path)
        else:
            raise ValueError("Cache not implemented")
        # else:
        #     data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
        #         "."
        #     )[0]
        #     cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
        #     data = shared_dict(cache_name)
        
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "segment20" in data.keys():
            segment = data["segment20"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance" in data.keys():
            instance = data["instance"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(segment).astype(np.bool)
            mask[sampled_index] = False
            segment[mask] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


    
if __name__ == "__main__":

    # dataset settings
    dataset_type = "ScanNetDataset"
    data_root = "data/scannet"
    data_root = "/media/didi/PortableSSD/scannetv2/scannet"

    data = dict(
        num_classes=20,
        ignore_index=-1,
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
            split=["train", "val"],
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
            ],
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


    from core.models.pointcept.pointcept.utils.config import Config
    
    data_config = Config(data)
    
    
    scannet = ScanNetDataset(
        **data_config.train
    )
    
    sample_dict = scannet.get_data(0)
    
    print(sample_dict.keys())
    
    for key in sample_dict.keys():
        if key == 'scene_id':
            print(key, sample_dict[key])
        else:
            print(key, sample_dict[key].shape)
        
        if key == 'segment':
            print(np.unique(sample_dict[key]))
            
            
    scannet_test = ScanNetDataset(
            **data_config.test
    )
    
    sample_dict = scannet_test.get_data(0)
    
    print(sample_dict.keys())
    
    for key in sample_dict.keys():
        print(key, sample_dict[key].shape)
        
        if key == 'segment':
            print(np.unique(sample_dict[key]))
    
    
    
    