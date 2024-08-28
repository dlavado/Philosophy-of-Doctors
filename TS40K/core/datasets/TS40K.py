
import math
import random
import shutil
from time import sleep
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import numpy as np
import laspy as lp
import gc
import torch.nn.functional as F
import torch

import sys

from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from utils import pointcloud_processing as eda
from utils import voxelization as Vox
from core.datasets.torch_transforms import EDP_Labels, Farthest_Point_Sampling, ToTensor, Voxelization_withPCD

import os

def edp_labels(labels:Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        #cast each label to its corresponding EDP label
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        
        labels = torch.tensor([eda.DICT_NEW_LABELS[label.item()] if label.item() >= 0 else label.item() for label in labels.squeeze()]).reshape(labels.shape)
        return labels


def build_data_samples(data_dirs:List[str], save_dir=os.getcwd(), sem_labels=True, fps=None, sample_types='all', data_split:dict={"fit": 0.7, "test": 0.3}):
    """

    Builds a dataset of voxelized point clouds in npy format according to 
    the las files in data_dirs and saves it in `save_dir`.

    The dataset will have the following structure:
    /save_dir
    | /tower_radius
    |     | - /fit
    |     | - /test
    | /2_towers
    |     | - /fit
    |     | - /test
    | /no_tower
    |     | - /fit
    |     | - /test

    In each 

    Parameters
    ----------

    `data_dirs` - str list : 
        list of directories with las files to be converted

    `save_dir` - str:
        directory where to save the npy files.

    `sem_labels` - bool:
        if True, semantic labels are casted to the new EDP labels

    `fps` - int:
        number of points to sample with Farthest Point Sampling
        if None, FPS is not applied

    `sample_types` - str list:
        list of the types of samples to be saved
        sample_types \in ['tower_radius', '2_towers', 'no_tower']
        or sample_types == 'all' for all types


    `data_split` - dict {"str": float} or int:
        if data_split is dict:
            split of the dataset into sub-folders; keys correspond to the folders and the values to the sample split
        elif data_split is int:
            data_split == 0 for no dataset split 
    """

    print(f"\n\nBuilding dataset in {save_dir}...")

    if sample_types is None or sample_types == 'all':
        sample_types = ['tower_radius', '2_towers', 'no_tower']

    # Build Directories
    # /save_dir
    # | /tower_radius
    # |     | - /fit
    # |     | - /test
    # | /2_towers
    # |     | - /fit
    # |     | - /test
    # | /no_tower
    # |     | - /fit
    # |     | - /test
    for type in sample_types:
        cwd = os.path.join(save_dir, type)
        for folder in data_split.keys():
            if not os.path.exists(os.path.join(cwd, folder)):
                os.makedirs(os.path.join(cwd, folder))

    # to resume processing from the last read .las file
    read_files = []
    pik_name = 'read_files.pickle'
    pik_path = os.path.join(save_dir, pik_name)
    if os.path.exists(pik_path):
        read_files = eda.load_pickle(pik_path)
 
    for cwd in data_dirs:
        print(f"\n\n\nReading files in {cwd}...")

        for las_file in os.listdir(cwd):
            filename = os.path.join(cwd, las_file)

            if filename in read_files:
                print(f"File {filename} already read...\n\n")
                continue

            if ".las" in filename:
                try:
                    las = lp.read(filename)
                except Exception as e:
                    print(f"Problem occurred while reading {filename}\n\n")
                    continue
            else:
                print(f"File {filename} is not a .las file...\n\n")
                continue

            print(f"\n\n\nReading...{filename}")

            xyz, classes = eda.las_to_numpy(las)


            t_samples = []
            tt_samples = [] 
            f_samples = []
            if np.any(classes == eda.POWER_LINE_SUPPORT_TOWER):
                if 'tower_radius' in sample_types:
                    t_samples = eda.crop_tower_samples(xyz, classes, radius=30)
                if '2_towers' in sample_types:
                    tt_samples = eda.crop_two_towers_samples(xyz, classes)    

            if 'no_tower' in sample_types:
                f_samples = eda.crop_ground_samples(xyz, classes)     

            xyz, classes = None, None
            del xyz, classes
            gc.collect()

            if len(t_samples) == 0 and len(tt_samples) == 0 and len(f_samples) == 0:
                print(f"No samples in {filename}\n\n")
                continue

            if len(sample_types) < 3:
                sample_types = ['tower_radius', '2_towers', 'no_tower'] # to avoid errors, some lists will be empty so no samples will be saved
                
            
            for sample_type, samples in zip(sample_types, [t_samples, tt_samples, f_samples]):

                fit_path = os.path.join(save_dir, sample_type, 'fit')
                counter = len(os.listdir(fit_path))
                print(f"\n\nNumber of samples in {sample_type}: {len(samples)}")

                if fps is not None:
                    fps_sampler = Farthest_Point_Sampling(fps)

                for sample in samples:
                    
                    sample = torch.from_numpy(sample)
                    if fps is not None:
                        input, labels = fps_sampler(sample)
                    else:
                        input, labels = sample[:, :-1], sample[:, -1]

                    if sem_labels:
                        labels = edp_labels(labels)

                    uq_classes = torch.unique(labels) 
                    # quality check of the point cloud
                    if input.shape[0] < 500:
                        print(f"Sample has less than 500 points...\nSkipping...")
                        continue
                    elif torch.sum(uq_classes > 0) < 2:
                        print(f"Sample has less than 2 semantic classes...\nSkipping...")
                        continue
                    else:
                        print(f"Sample has {input.shape[0]} points and {uq_classes} classes")

                    # get bounding boxes
                    obj_boxes = []
                    # for obj_label, obj_name in eda.DICT_OBJ_DET_LABELS.items():
                    #     if obj_label in labels:
                    #         print(f"Getting bounding boxes for {obj_name}...")
                    #         obj_boxes += eda.extract_bounding_boxes(input.numpy(), labels.numpy(), obj_label, eps=10, min_samples=50)

                    sample_dict = {
                        'type' :            sample_type, # tower_radius, 2_towers, no_tower
                        'input_pcd' :       input, # torch.tensor  with shape (N, 3)
                        'semantic_labels' : labels, # torch.tensor with shape (N,)
                        'obj_boxes':        obj_boxes # list of dicts with keys: ['class_label', 'position', 'dimensions', 'rotation']
                    }

                    # print the sample info
                    print("------- Sample Info -------")
                    print(f"\tSample type: {sample_type}")
                    print(f"\tSample input shape: {input.shape}")
                    print(f"\tSample semantic labels shape: {labels.shape}")
                    print(f"\tSample number of 3D bounding boxes: {len(obj_boxes)}")
                    if len(obj_boxes) > 0:
                        print(f"\tSample random box: {obj_boxes[random.randint(0, len(obj_boxes)-1)]}")
                    print("---------------------------")

                    try:  
                        sample_path = os.path.join(fit_path, f"sample_{counter}.pt")           
                        torch.save(sample_dict, sample_path)
                        print(f"Saving... {sample_path}")
                        counter += 1
                    except Exception as e:
                        print(e)
                        print(f"Problem occurred while saving {filename}\n\n")
                        continue


                    # free up as much memory as possible
                    sample_dict, sample, input, labels, uq_classes, obj_boxes, sample_path = None, None, None, None, None, None, None
                    del sample_dict, sample, input, labels, uq_classes, obj_boxes, sample_path
                    gc.collect()
                    torch.cuda.empty_cache()
                    sleep(0.1)

            del las
            del t_samples
            del tt_samples
            del f_samples
            gc.collect()
            read_files.append(filename)
            eda.save_pickle(read_files, pik_path)
          

    if data_split == 0:
        return # all files in fit_path
    
    # split data
    print(f"\n\nSplitting data in {save_dir}...")
    
    for sample_type in sample_types:

        fit_path = os.path.join(save_dir, sample_type, 'fit')
        print(f"Splitting {fit_path}...")

        samples = os.listdir(fit_path)
        random.shuffle(samples)

        assert sum(list(data_split.values())) <= 1, "data splits should not surpass 1"

        split_sum = 0
        sample_size = len(samples)
        print(f"Number of total samples: {sample_size}")

        for folder, split in data_split.items():
            if folder == 'fit':
                split_sum += split
                continue

            cwd = os.path.join(save_dir, sample_type, folder)
            if not os.path.exists(cwd):
                os.makedirs(cwd)

            samples_split = samples[int(split_sum*sample_size):math.ceil((split_sum+split)*sample_size)]
            split_sum += split
            print(f"Samples in {folder}: {len(samples_split)}")

            for sample in samples_split:
                shutil.move(os.path.join(fit_path, sample), cwd)


def build_bounding_boxes(dataset_dir:str, objects_detect:list[str]):
    """
    Takes the samples built in `build_data_samples`, extracts the bounding boxes in the samples according to the objects to detect
    and adds this information to the samples.
    """
    # process each sample at a time to avoid memory issues
    for sample_type in os.listdir(dataset_dir):
        sample_type_path = os.path.join(dataset_dir, sample_type)
        for split in os.listdir(sample_type_path):
            split_path = os.path.join(sample_type_path, split)
            for sample in tqdm(os.listdir(split_path), desc=f"Processing {sample_type} {split} samples..."):
                sample_path = os.path.join(split_path, sample)
                sample_dict = torch.load(sample_path)

                # only process sample if it has not been processed before
                if isinstance(sample_dict['obj_boxes'], list) and len(sample_dict['obj_boxes']) > 0:
                    print(f"Sample {sample} already processed...")
                    print(sample_dict['obj_boxes'])
                    continue

                input = sample_dict['input_pcd']
                labels = torch.squeeze(sample_dict['semantic_labels'])
                sample_dict['semantic_labels'] = labels
                

                obj_boxes = []
                for obj_label, obj_name in eda.DICT_OBJ_DET_LABELS.items():
                    print(f"Getting bounding boxes for {obj_name}...")
                    obj_boxes += eda.extract_bounding_boxes(input.numpy(), labels.numpy(), obj_label, eps=8, min_samples=300)

                sample_dict['obj_boxes'] = obj_boxes

                # print the sample info
                if random.randint(0, 10) >= 8:
                    print("------- Sample Info -------")
                    print(f"\tSample type: {sample_type}")
                    print(f"\tSample input shape: {input.shape}")
                    print(f"\tSample semantic labels shape: {labels.shape}")
                    print(f"\tSample number of 3D bounding boxes: {len(obj_boxes)}")
                    if len(obj_boxes) > 0:
                        print(f"\tSample random box: {obj_boxes[random.randint(0, len(obj_boxes)-1)]}")
                    print("---------------------------")

                torch.save(sample_dict, sample_path)

                # clean as much memory as possible
                sample_dict, input, labels, obj_boxes = None, None, None, None
                del sample_dict, input, labels, obj_boxes
                gc.collect()



def save_preprocessed_data(data_dir, save_dir, fps_points, vxg_size, vox_size):
    """
    
    """

    data_split = ['fit', 'test']

    ts40k_transform = Compose([
                        ToTensor(),

                        Farthest_Point_Sampling(fps_points),
        
                        Voxelization_withPCD(keep_labels='all', 
                                             vxg_size=vxg_size, 
                                             vox_size=vox_size
                                            ),
                        EDP_Labels(),
                    ])

    for folder in data_split:

        folder_path = os.path.join(save_dir, folder)

        dm = TS40K(data_dir, split=folder, transform=ts40k_transform, min_points=None)

        os.makedirs(folder_path, exist_ok=True)

        # get folder count
        folder_count = len(os.listdir(folder_path))

        for i in tqdm(range(folder_count, len(dm)), desc=f"Saving Preprocessed {folder} samples..."):
            x, y, pt_locs = dm[i]
            sample = [x, y, pt_locs] 
            # torch save
            sample_path = os.path.join(folder_path, f"sample_{i}.pt")
            torch.save(sample, sample_path)



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

    def __init__(self, dataset_path, split='fit', sample_types='all', task="sem_seg", transform=None, min_points=None, load_into_memory=True) -> None:
        """
        Initializes the TS40K dataset.

        There are different types of samples in the dataset:
        - tower_radius: samples with a single tower in the center
        - 2_towers: samples with two towers and a power line between them
        - no_tower: samples with no towers

        There are also two types of tasks available:
        - sem_seg: semantic segmentation
        - obj_det: object detection, which inlcudes objects of the classes: `vegetation`, `power_line`, `obstacle` and `tower` 

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

        `min_points` - int:
            minimum number of points in the point cloud

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
        
        if min_points:
            self.data_files = np.array([file for file in self.data_files if np.load(os.path.join(self.dataset_path, file)).shape[0] >= min_points])
    
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
        if len(bboxes) == 0:
            return torch.tensor([])
        
        boxes = torch.zeros((len(bboxes), 7))

        for i, bbox in enumerate(bboxes):
            pos   = torch.tensor([bbox['position']['x'], bbox['position']['y'], bbox['position']['z']])
            dims  = torch.tensor([bbox['dimensions']['width'], bbox['dimensions']['height'], bbox['dimensions']['length']])
            angle = torch.tensor([bbox['rotation']])

            cat = torch.cat((pos, dims, angle), axis=0)
            boxes[i] = cat
        

        return boxes


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

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_path = self.data_files[idx]

        if sample_path.endswith('.pt'):
            sample_dict = torch.load(sample_path)
        elif sample_path.endswith('.npy'):
            sample_dict = np.load(sample_path, allow_pickle=True).item()
        else:
            raise ValueError(f"File {sample_path} is not a .pt or .npy file")
        
        x = sample_dict['input_pcd']
        if self.task == "sem_seg":
            y = sample_dict['semantic_labels']
        else:
            y = sample_dict['obj_boxes']
            y = self._bboxes_to_tensor(y)

        sample = (x, y) # xyz-coord (N, 3); label (N, 1)

        if self.transform:
            sample = self.transform(sample)
            #print(f"Transformed sample: {sample[0].shape}, {sample[1].shape}, {sample[2].shape}")
            return sample
        
        return sample


# class TS40K_Preprocessed(TS40K):

#     def __init__(self, dataset_path, split='fit', load_into_memory=False) -> None:
#         super().__init__(dataset_path, split, None, None, load_into_memory)

#     def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
#         # data[i]

#         if self.load_into_memory:
#             return self.data[idx]

#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         pt_path = os.path.join(self.dataset_path, self.data_files[idx])
#         try:
#             pt = torch.load(pt_path)
#         except:
#             print(f"Unreadable file: {pt_path}")
        
#         sample = (pt[0], pt[1], pt[2]) # voxel (1, 64, 64, 64); label (1, N); point locations (1, N, 3)
#         return sample
       
            

def main():
    
    TS40K_DIR = os.path.join(constants.TOSH_PATH, "TS40K-Dataset")
    
    
    
    #SAVE_PATH = os.path.join(TS40K_DIR, "TS40K-NEW")
    # save_preprocessed_data(constants.TS40K_PATH, TS40K_PREPROCESSED_PATH, fps_points=10000, vxg_size=(64, 64, 64), vox_size=None)

    # build_data_samples([os.path.join(TS40K_DIR, "LIDAR"), os.path.join(TS40K_DIR, "Labelec_LAS")], 
    #                    save_dir = os.path.join(TS40K_DIR, "TS40K-FULL"), 
    #                    sem_labels=True, 
    #                    fps=None, 
    #                    sample_types='all', 
    #                    data_split={"fit": 0.8, "test": 0.2})
    
    build_bounding_boxes(os.path.join(TS40K_DIR, "TS40K-FULL"), 
                         objects_detect=eda.DICT_OBJ_DET_LABELS.values())

    input("Press Enter to continue...")

    #composed = Compose([ToTensor(), AddPad((3, 3, 3, 3, 3, 3))]) 
    vxg_size  = (64, 64, 64)
    vox_size = (0.5, 0.5, 0.5) #only use vox_size after training or with batch_size = 1
    composed =  None
    
    ts40k = TS40K(dataset_path=None, split='fit', transform=composed)
    
    small_samples = []

    for i in range(len(ts40k)):
        pts, gts = ts40k[i]
        if pts.shape[1] < 500:
            small_samples.append(i)

    # del small samples
    for i in reversed(small_samples):
        npy_path = os.path.join(ts40k.dataset_path, ts40k.data_files[i])
        os.remove(npy_path)

    print(len(ts40k))

    print(ts40k[0])
    vox, vox_gt = ts40k[0]
    print(vox.shape, vox_gt.shape)

    Vox.plot_voxelgrid(torch.squeeze(vox))
    Vox.plot_voxelgrid(torch.squeeze(vox_gt))

    # Hyper parameters
    NUM_EPOCHS = 5
    NUM_SAMPLES = len(ts40k)
    BATCH_SIZE = 1
    NUM_ITER = math.ceil(NUM_SAMPLES / BATCH_SIZE)

    ts40k_loader = DataLoader(ts40k, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    for epoch in range(NUM_EPOCHS):
        for i, (xyz, vox, vox_gt) in tqdm(enumerate(ts40k_loader), desc=f"epoch {epoch+1}", ): 
            #print(f"epoch {epoch} / {NUM_EPOCHS}; step {i+1} / {NUM_ITER}; input: {vox.shape}; labels: {vox_gt.shape}")
            print(xyz.shape)
            print(vox.shape)
            print(vox_gt.shape)

            xyz, vox, vox_gt = xyz[0][0], vox[0][0], vox_gt[0][0]  #1st sample of batch; decapsulate

            Vox.plot_voxelgrid(vox)
            Vox.plot_voxelgrid(vox_gt)
            xyz_gt = Vox.vxg_to_xyz(torch.cat((xyz, vox_gt[None]), axis=0))
            pcd_gt = eda.np_to_ply(xyz_gt)
            eda.visualize_ply([pcd_gt])

            print(f"xyz centroid = {eda.xyz_centroid(xyz_gt)}")
            print(f"euc dist = {eda.euclidean_distance(xyz_gt, xyz_gt)}")


if __name__ == "__main__":

    from utils import constants
    
    main()

# %%
