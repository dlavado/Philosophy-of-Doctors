
import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import torch


from tqdm import tqdm
import os


class TS40K(Dataset):

    def __init__(self, dataset_path, split='fit', transform=None, min_points=None, load_into_memory=True) -> None:
        """
        Initializes the TS40K dataset

        Parameters
        ----------
        `dataset_path` - str:
            path to the directory with the voxelized point clouds crops in npy format

        `split` - str:
            split of the dataset to access 
            split \in [fit, test] 
            
        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds

        `min_points` - int:
            minimum number of points in the point cloud

        `load_into_memory` - bool:
            if True, loads the entire dataset into memory
        """
        super().__init__()

        self.transform = transform
        self.split = split

        self.dataset_path = os.path.join(dataset_path, split)

        self.data_files:np.ndarray = np.array([file for file in os.listdir(self.dataset_path)
                        if os.path.isfile(os.path.join(self.dataset_path, file)) and ('.npy' in file or '.pt' in file)])
        
        if min_points:
            self.data_files = np.array([file for file in self.data_files if np.load(os.path.join(self.dataset_path, file)).shape[0] >= min_points])
    
        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True

    def __len__(self):
        return len(self.data_files)

    def __str__(self) -> str:
        return f"TS40K {self.split} Dataset with {len(self)} samples"
    
    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    def set_transform(self, new_transform):
        self.transform = new_transform
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if self.load_into_memory:
            return self.data[idx]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_path = os.path.join(self.dataset_path, self.data_files[idx])

        try:
            npy = np.load(npy_path)
        except:
            print(f"Unreadable file: {npy_path}")
        
        sample = (npy[None, :, 0:-1], npy[None, :, -1]) # xyz-coord (1, N, 3); label (1, N) 


        if self.transform:
            sample = self.transform(sample)
            #print(f"Transformed sample: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")
            return sample
        
        return sample
    

class TS40K_FULL(Dataset):

    def __init__(self, dataset_path, split='fit', sample_types='all', task="sem_seg", transform=None, load_into_memory=True) -> None:
        """
        Initializes the TS40K dataset.

        There are different types of samples in the dataset:
        - tower_radius: samples with a single tower in the center
        - 2_towers: samples with two towers and a power line between them
        - no_tower: samples with no towers

        There are also two types of tasks available:
        - sem_seg: semantic segmentation
        - obj_det: object detection, which inlcudes objects of the classes: `power_line` and `tower` 

        Parameters
        ----------

        `dataset_path` - str:
            path to the directory with the TS40K dataset

        `split` - str:
            split of the dataset to access 
            split \in [fit, test]

        `sample_types` - str list:
            list of the types of samples to be used
            sample_types \in ['tower_radius', '2_towers', 'no_tower']
            or sample_types == 'all' for all types

        `task` - str:
            task to perform
            task \in ['sem_seg', 'obj_det']

        `transform` - (None, torch.Transform) :
            transformation to apply to the point clouds

        `load_into_memory` - bool:
            if True, loads the entire dataset into memory

        """
        super().__init__()

        
        if task not in ['sem_seg', 'obj_det']:
            raise ValueError(f"Task {task} not supported. Task should be one of ['sem_seg', 'obj_det']")

        if sample_types != 'all' and not isinstance(sample_types, list):
            raise ValueError(f"sample_types should be a list of strings or 'all'")

    
        self.transform = transform
        self.split = split
        self.task = task

        self.dataset_path = dataset_path

        if sample_types == 'all':
            sample_types = ['tower_radius', '2_towers', 'no_tower']

        self.data_files = []

        for type in sample_types:
            type_path = os.path.join(self.dataset_path, type, split)
            self.data_files += [os.path.join(type_path, file) for file in os.listdir(type_path)
                                 if os.path.isfile(os.path.join(type_path, file)) and ('.npy' in file or '.pt' in file)]

        self.data_files = np.array(self.data_files)
        
        # if min_points:
        #     # if min_points is not None, filter out samples with less than min_points
        #     self.data_files = np.array([file for file in self.data_files if torch.load(os.path.join(self.dataset_path, file))['input_pcd'].shape[0] >= min_points])
    
        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True
            
    def __len__(self):
        return len(self.data_files)

    def __str__(self) -> str:
        return f"TS40K FULL {self.split} Dataset with {len(self)} samples"
    
    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    
    def _bboxes_to_tensor(self, bboxes:list[dict]):
        """
        Converts a list of bounding boxes to a tensor of shape (O, 8) with the following format:
        [x, y, z, dx, dy, dz, heading_angle, class_label]

        Parameters
        ----------
        `bboxes` - list[dict]:
            list of bounding boxes with the following keys:
            ['class_label', 'position', 'dimensions', 'rotation']

        Returns
        -------
        `boxes` - torch.Tensor:
            tensor with shape (O, 8) with the bounding boxes
        """
        if len(bboxes) == 0:
            return torch.tensor([])
        
        boxes = torch.zeros((len(bboxes), 8))

        for i, bbox in enumerate(bboxes):
            pos   = torch.tensor([bbox['position']['x'], bbox['position']['y'], bbox['position']['z']])
            dims  = torch.tensor([bbox['dimensions']['width'], bbox['dimensions']['height'], bbox['dimensions']['length']])
            angle = torch.tensor([bbox['rotation']])
            label = torch.tensor([bbox['class_label']])

            cat = torch.cat((pos, dims, angle, label), axis=0) # [x, y, z, dx, dy, dz, heading_angle, class_label], shape (8,)
            boxes[i] = cat 
        

        return boxes


    def _get_dict(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_path = self.data_files[idx]

        try:
            if sample_path.endswith('.pt'):
                sample_dict = torch.load(sample_path)
            elif sample_path.endswith('.npy'):
                sample_dict = np.load(sample_path, allow_pickle=True).item()
            else:
                raise ValueError(f"File {sample_path} is not a .pt or .npy file")
        
        except Exception as e:
            print(f"Unreadable file: {sample_path}")
            print(e)
            return self[random.randint(0, len(self)-1)]
        
        return sample_dict
    
    def _get_file_path(self, idx):
        return os.path.join(self.dataset_path, self.data_files[idx])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The format of the sample files is as follows:
        sample_dict = {
            'type' :            sample_type, # tower_radius, 2_towers, no_tower
            'input_pcd' :       input, # torch.tensor  with shape (N, 3)
            'semantic_labels' : labels[None], # torch.tensor with shape (N, 1)
            'obj_boxes':        obj_boxes # list of dicts with keys: ['class_label', 'position', 'dimensions', 'rotation']
        }        
        """
        # data[i]

        if self.load_into_memory:
            return self.data[idx]

        sample_dict = self._get_dict(idx)
        
        if isinstance(sample_dict, dict):
            x = sample_dict['input_pcd']
            if self.task == "sem_seg":
                y = sample_dict['semantic_labels']
                y = torch.squeeze(y) # reshape to (N,)
                sample = (x, y) # xyz-coord (N, 3); label (N,)
            else:
                y = sample_dict['obj_boxes']
                if not isinstance(y, torch.Tensor):
                    y = self._bboxes_to_tensor(y)
                sample = (x, y) # xyz-coord (N, 3); label (O, 8)

            if self.transform:
                # send sample to gpu
                sample = self.transform(sample)
                #print(f"Transformed sample: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")
        
        else: # data was preprocessed as saved as a tuple
            x, y = sample_dict
            sample = (x.squeeze(), y.squeeze())
        
        return sample


class TS40K_FULL_Preprocessed(Dataset):
    """
    The preprocesssed data follows the following transformations:
    - Normalize_PCD() : normalize the point cloud to have mean 0 and std 1
    - Farthest_Point_Sampling(fps_points) : sample fps_points from the point cloud with a total of 10K points
    - To(torch.float32) : cast the point cloud to float32

    This results in a datasets with similar structure to the original TS40K_FULL dataset, but with the preprocessed data.

    The targets are specific to the sem_seg task, for others, different preprocessing should be applied.
    """

    def __init__(self, dataset_path:str, split='fit', sample_types='all', transform=None, load_into_memory=True, use_full_test_set=False) -> None:
        super().__init__()

        if sample_types != 'all' and not isinstance(sample_types, list):
            raise ValueError(f"sample_types should be a list of strings or 'all'")
        
        self.dataset_path = dataset_path
        self.transform = transform

        if split == 'test' and use_full_test_set:
            self.dataset_path = self.dataset_path.replace('-Preprocessed', '')
            self.ts40k_full = TS40K_FULL(self.dataset_path, split='test', sample_types=sample_types, task='sem_seg', transform=transform, load_into_memory=load_into_memory)
        else:
            self.ts40k_full = None

        if sample_types == 'all':
            sample_types = ['tower_radius', '2_towers', 'no_tower']

        self.data_files = []

        for type in sample_types:
            type_path = os.path.join(self.dataset_path, type, split)
            self.data_files += [os.path.join(type_path, file) for file in os.listdir(type_path)
                                 if os.path.isfile(os.path.join(type_path, file)) and ('.npy' in file or '.pt' in file)]

        self.data_files = np.array(self.data_files)


        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True


    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))


    def __len__(self):
        return len(self.data_files)
    
    def _get_file_path(self, idx) -> str:
        return os.path.join(self.dataset_path, self.data_files[idx])


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # data[i]

        if self.ts40k_full:
            return self.ts40k_full[idx]

        if self.load_into_memory:
            return self.data[idx]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        pt_path = self.data_files[idx]
        
        try:
            pt = torch.load(pt_path)  # xyz-coord (1, N, 3); label (1, N)
        except:
            print(f"Unreadable file: {pt_path}")

        if self.transform:
            pt = self.transform(pt)
            #print(f"Transformed sample: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")

        return pt
       