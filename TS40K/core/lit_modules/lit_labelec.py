

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from core.datasets.Labelec import Labelec_Dataset as Labelec


class LitLabelec(pl.LightningDataModule):

    def __init__(self, 
                 las_data_dir, 
                 save_chunks=False, 
                 chunk_size=10_000_000, 
                 bins=5, 
                 transform=None, 
                 test_transform=None,
                 load_into_memory=False,
                 batch_size=1,
                 val_split=0.2,
                 num_workers=8   
            ):
        
        super().__init__()
        self.las_data_dir = las_data_dir
        self.transform = transform
        self.test_transform = test_transform        
        self.load_into_memory = load_into_memory
        if save_chunks:
            self._build_dataset(chunk_size, bins)
        
        self.save_hyperparameters()

    def _build_dataset(self, chunk_size, bins): #Somewhat equivalent to `prepare_data` hook of LitDataModule
        Labelec(las_data_dir=self.las_data_dir, split='fit', save_chunks=True, chunk_size=chunk_size, bins=bins, transform=self.transform, load_into_memory=self.load_into_memory) 
        Labelec(las_data_dir=self.las_data_dir, split='test', save_chunks=True, chunk_size=chunk_size, bins=bins, transform=self.test_transform, load_into_memory=self.load_into_memory)           
   
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.fit_ds = Labelec(las_data_dir=self.las_data_dir, split='fit', save_chunks=False, transform=self.transform, load_into_memory=self.load_into_memory)
            self.train_dataset, self.val_dataset = random_split(self.fit_ds, [1 - self.hparams.val_split, self.hparams.val_split])
        
        elif stage == 'test':
            self.test_ds = Labelec(las_data_dir=self.las_data_dir, split='test', save_chunks=False, transform=self.test_transform, load_into_memory=self.load_into_memory)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
    


if __name__ == '__main__':
    from utils import constants as consts

    lit_labelec = LitLabelec(las_data_dir=consts.LABELEC_RGB_DIR, 
                             save_chunks=False, 
                             chunk_size=10_000_000, 
                             bins=5, 
                             transform=None, 
                             test_transform=None, 
                             load_into_memory=False, 
                             batch_size=16, 
                             val_split=0.2, 
                             num_workers=8
                        )
    
    lit_labelec.setup('fit')

    train_loader = lit_labelec.train_dataloader()

    for batch in train_loader:
        x, y = batch

        print(x.shape, y.shape)
    

