

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch

import sys
sys.path.append('../..')
from core.datasets.TS40K import TS40K_FULL, TS40K_FULL_Preprocessed
from core.models.giblinet.conversions import list_tensor_to_batch

from torch.utils.data.dataloader import default_collate
from collections.abc import Sequence, Mapping


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
    



def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def collate_offset(batch):
    """
    `batch` is an iterable with with batch elements.
    """
    assert isinstance(
        batch[0], Mapping
    )
    return collate_fn(batch)
    


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
                ):

        super().__init__(data_dir, batch_size, sample_types, None, transform, transform_test, num_workers, val_split, None, load_into_memory)
        
        self.fit_split = 0.01 # split of the data to use, for testing purposes

        self.use_full_test_set = use_full_test_set
        self.collate_fn = collate_offset

    def setup(self, stage:str=None):
        # build dataset
        if stage == 'fit':
            self.fit_ds = TS40K_FULL_Preprocessed(self.data_dir, split="fit", sample_types=self.sample_types, transform=self.transform, load_into_memory=self.load_into_memory)
            
            # self.fit_ds.data_files = self.fit_ds.data_files[:int(len(self.fit_ds) * self.fit_split)]
            
            self.train_ds, self.val_ds = random_split(self.fit_ds, [1 - self.hparams.val_split, self.hparams.val_split])
            
        if stage == 'test':
            self.test_ds = TS40K_FULL_Preprocessed(self.data_dir, split="test", sample_types=self.sample_types, transform=self.transform_test, load_into_memory=self.load_into_memory, use_full_test_set=self.use_full_test_set)
                
        if stage == 'predict':
            self.predict_ds = TS40K_FULL_Preprocessed(self.data_dir, split="test", sample_types=self.sample_types, transform=self.transform_test, load_into_memory=self.load_into_memory)
                
                
    def train_dataloader(self): 
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True, collate_fn=self.collate_fn)
         

    def test_dataloader(self):

        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=self.collate_fn)
    
    
    
if __name__ == '__main__':
    import utils.constants as C 
    

    
    
    ts40k = LitTS40K_FULL_Preprocessed(
        data_dir=C.TS40K_FULL_PREPROCESSED_PATH,
        batch_size=2,
        sample_types='all',
        transform=None,
        transform_test=None,
        num_workers=8,
        val_split=0.1,
        load_into_memory=False,
        use_full_test_set=False
    )
    
    
    ts40k.setup('fit')
    
    train_dl = ts40k.train_dataloader()
    
    for i, batch in enumerate(train_dl):
        print(f"{batch['coord'].shape=}, {batch['offset'].shape=}, {batch['segment'].shape=}")   
        print(f"{batch['offset']=}")
        break
