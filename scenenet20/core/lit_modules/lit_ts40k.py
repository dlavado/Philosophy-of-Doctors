

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch

from core.datasets.TS40K import build_data_samples, TS40K_FULL, TS40K_FULL_Preprocessed

from core.models.giblinet.conversions import list_tensor_to_batch


class LitTS40K_FULL(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for TS40K dataset.

    Parameters
    ----------

    `data_dir` - str :
        directory where the dataset is stored

    `batch_size` - int :
        batch size to use for routines

    `transform` - (None, torch.Transform) :
        transformation to apply to the point clouds

    `num_workers` - int :
        number of workers to use for data loading

    `val_split` - float :
        fraction of the training data to use for validation

    `test_split` - float :
        In the case of building the dataset from raw data, fraction of the data to use for testing

    """

    def __init__(self, 
                 data_dir, 
                 batch_size,
                 sample_types='all', 
                 task="sem_seg",
                 transform=None, 
                 transform_test=None,
                 num_workers=8, 
                 val_split=0.1, 
                 min_points=None, 
                 load_into_memory=False
            ):
        super().__init__()
        self.data_dir = data_dir
        self.sample_types = sample_types
        self.task = task
        self.transform = transform
        self.transform_test = transform_test
        self.min_points = min_points
        self.load_into_memory = load_into_memory
        self.save_hyperparameters()

    def _build_dataset(self, raw_data_dir): #Somewhat equivalent to `prepare_data` hook of LitDataModule
        build_data_samples(raw_data_dir, self.data_dir, sem_labels=True, fps=None, sample_types=self.sample_types, data_split={"fit": 0.8, "test": 0.2})

    def setup(self, stage:str=None):
        # build dataset
        if stage == 'fit':
            self.fit_ds = TS40K_FULL(self.data_dir, split="fit", task=self.task, sample_types=self.sample_types, transform=self.transform, load_into_memory=self.load_into_memory)
            self.train_ds, self.val_ds = random_split(self.fit_ds, [1 - self.hparams.val_split, self.hparams.val_split])
        if stage == 'test':
            self.test_ds = TS40K_FULL(self.data_dir, split="test", task=self.task, sample_types=self.sample_types, transform=self.transform_test, load_into_memory=self.load_into_memory)

        if stage == 'predict':
            self.predict_ds = TS40K_FULL(self.data_dir, split="test", task=self.task, sample_types=self.sample_types, transform=self.transform_test, load_into_memory=self.load_into_memory)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    
    def _fit_dataloader(self):
        return DataLoader(self.fit_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, shuffle=False)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TS40K")
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--val_split", type=float, default=0.1)
        parser.add_argument("--test_split", type=float, default=0.4)
        return parent_parser
    




def collate_fn_pyramids(batch):
    # batch is a dict with  'pointcloud', 'sem_labels', 'graph_pyramid'
    # it returns the same dict but with the batch dim and point dim collapsed, i.e, (B*N, ...)
    # plus, it returns the lengths of each sample in the batch
    
    x = [b["pointcloud"] for b in batch]
    y = [b["sem_labels"] for b in batch]
    graph_pyramids = [b["graph_pyramid"] for b in batch]  
    
    
    batched_graph_pyramids = {
        'points_list': [],
        'neighbors_idxs_list': [],
        'subsampling_idxs_list': [],
        'upsampling_idxs_list': []
    }
    
    for key in batched_graph_pyramids.keys(): # for each type of pyramid
        for i in range(len(graph_pyramids[0][key])): # for each level of the pyramid
            b_t = [gp[key][i] for gp in graph_pyramids]
            b_t = list_tensor_to_batch(b_t, pad_value=-1)
            batched_graph_pyramids[key].append(b_t)
            
    return {
        'pointcloud': torch.stack(x, dim=0),
        'sem_labels': torch.stack(y, dim=0),
        'graph_pyramid': batched_graph_pyramids,
    }


class LitTS40K_FULL_Preprocessed(LitTS40K_FULL):

    def __init__(self, 
                 data_dir, 
                 batch_size, 
                 sample_types='all', 
                 transform=None, 
                 transform_test=None, 
                 num_workers=8, 
                 val_split=0.1, 
                 load_into_memory=False, 
                 use_full_test_set=False,
                 _pyramid_builder=None):

        super().__init__(data_dir, batch_size, sample_types, None, transform, transform_test, num_workers, val_split, None, load_into_memory)
        
        self.fit_split = 0.01 # split of the data to use, for testing purposes

        self.use_full_test_set = use_full_test_set
        self.pyramid_builder = _pyramid_builder
        self.collate_fn = None if self.pyramid_builder is None else collate_fn_pyramids

    def setup(self, stage:str=None):
        # build dataset
        if stage == 'fit':
            self.fit_ds = TS40K_FULL_Preprocessed(self.data_dir, split="fit", sample_types=self.sample_types, transform=self.transform, load_into_memory=self.load_into_memory)
            
            # self.fit_ds.data_files = self.fit_ds.data_files[:int(len(self.fit_ds) * self.fit_split)]
            
            if self.pyramid_builder is not None:
                self.fit_ds._build_pyramid(self.pyramid_builder)
            
            self.train_ds, self.val_ds = random_split(self.fit_ds, [1 - self.hparams.val_split, self.hparams.val_split])
            
        if stage == 'test':
            self.test_ds = TS40K_FULL_Preprocessed(self.data_dir, split="test", sample_types=self.sample_types, transform=self.transform_test, load_into_memory=self.load_into_memory, use_full_test_set=self.use_full_test_set)
            
            if self.pyramid_builder is not None:
                self.test_ds._build_pyramid(self.pyramid_builder)
                
        if stage == 'predict':
            self.predict_ds = TS40K_FULL_Preprocessed(self.data_dir, split="test", sample_types=self.sample_types, transform=self.transform_test, load_into_memory=self.load_into_memory)

            if self.pyramid_builder is not None:
                self.predict_ds._build_pyramid(self.pyramid_builder)
                
                
    def train_dataloader(self):
        
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True, collate_fn=self.collate_fn)
         

    # def test_dataloader(self):
    #     if self.use_full_test_set:
    #         batch_size = 1 # point clouds do not have the same number of points; thus they cannot be stacked.
    #     else:
    #         batch_size = self.hparams.batch_size

    #     return DataLoader(self.test_ds, batch_size=batch_size, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=collate_fn_pyramids)

