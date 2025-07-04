"""
S3DIS Dataset

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


class S3DISDataset(Dataset):
    def __init__(
        self,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root="data/s3dis",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(S3DISDataset, self).__init__()
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
            print(self.test_cfg.keys())

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()

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
            data_list = [os.path.join(self.data_root, self.split, room) for room in os.listdir(os.path.join(self.data_root, self.split))]
            # data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += [os.path.join(self.data_root, split, room) for room in os.listdir(os.path.join(self.data_root, split))] 
                #glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        
        return data_list
    
    def get_sample(self, sample_dir):
        data_dict = {}
        for file in os.listdir(sample_dir):
            file_path = os.path.join(sample_dir, file)
            file_name = file.split('.')[0]
            data_dict[file_name] = np.load(file_path)  
            
        data_dict['scene_id'] = sample_dir.split('/')[-1]
        
        return data_dict

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            # data = torch.load(data_path)
            data = self.get_sample(data_path)
        # else:
        #     data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
        #         "."
        #     )[0]
        #     cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
        #     data = shared_dict(cache_name)
        else:
            raise NotImplementedError
        
        name = (
            os.path.basename(self.data_list[idx % len(self.data_list)])
            .split("_")[0]
            .replace("R", " r")
        )
        coord = data["coord"]
        color = data["color"]
        scene_id = data["scene_id"]
        
        if "segment" in data.keys():
            segment = data["segment"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance" in data.keys():
            instance = data["instance"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            name=name,
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if "normal" in data.keys():
            data_dict["normal"] = data["normal"]
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
    
    
    
    
if __name__ == '__main__':
    
    dataset_type = "S3DISDataset"
    data_root = "data/s3dis"
    data_root = "/media/didi/PortableSSD/s3dis"

    data = dict(
        num_classes=13,
        ignore_index=-1,
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
            ],
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
    
     
    from core.models.pointcept.pointcept.utils.config import Config
    
    data_config = Config(data)
    
    
    s3dis = S3DISDataset(**data_config.train)
    
    sample_dict = s3dis.get_data(0)
    
    print(sample_dict.keys())
    
    for key in sample_dict.keys():
        if key == 'scene_id' or key == 'name':
            print(key, sample_dict[key])
        else:
            print(key, sample_dict[key].shape)
        
        if key == 'segment':
            print(np.unique(sample_dict[key]))
                
    
    s3dis_test = S3DISDataset(**data_config.test)
    
    sample_dict = s3dis_test.get_data(0)
    
    print(sample_dict.keys())
    
    for key in sample_dict.keys():
        if key == 'scene_id' or key == 'name':
            print(key, sample_dict[key])
        else:
            print(key, sample_dict[key].shape)
        
        if key == 'segment':
            print(np.unique(sample_dict[key]))
            
    
    
    
