"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import TRANSFORMS
from torchvision.transforms import Compose
from tqdm import tqdm
import torch


import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
import pyaml
from scripts.constants import KITTI_PATH


@DATASETS.register_module()
class SemanticKITTIDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/semantic_kitti",
        learning_map=None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        load_into_memory=False,
        loop=1,
    ):
        super(SemanticKITTIDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.learning_map = learning_map
        if self.learning_map is None:
            self.learning_map = pyaml.yaml.load(open(os.path.join(data_root, "semantic-kitti.yaml"), "r"), Loader=pyaml.yaml.Loader)
            self.learning_map = self.learning_map["learning_map"]

        # self.split2seq = dict(
        #     train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        #     val=[8],
        #     test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        # )
        self.split2seq = dict(
            train=[0, 1, 2, 3, 4],
            val=[5],
            test=[6],
        )
        self.transform = Compose(transform) if transform else None
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if isinstance(self.split, str):
            seq_list = self.split2seq[split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += self.split2seq[split]
        else:
            raise NotImplementedError

        self.data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            self.data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True


    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    def prepare_train_data(self, idx):
        name = self.get_data_name(idx)
        # load data
        data_idx = idx % len(self.data_list)
        with open(self.data_list[data_idx], "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = (
            self.data_list[data_idx]
            .replace("velodyne", "labels")
            .replace(".bin", ".label")
        )
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        segment = np.vectorize(self.learning_map.__getitem__)(segment & 0xFFFF).astype(
            np.int64
        )
        data_dict = dict(coord=coord, strength=strength, segment=segment)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    def __getitem__(self, idx):
        if self.load_into_memory:
            return self.data[idx]
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

