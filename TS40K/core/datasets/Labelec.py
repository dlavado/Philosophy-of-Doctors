import torch
from torch.utils.data import Dataset
import laspy as lp
import os
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
import utils.pointcloud_processing as eda

def convert_las_to_laz(las_file_paths:list[str]):
    import subprocess
    new_file_paths = []
    for las_file_path in las_file_paths:
        laz_file_path = las_file_path.replace(".las", ".laz")
        
        # Run the LASzip command to convert .las to .laz
        subprocess.run(["laszip", "-i", las_file_path, "-o", laz_file_path], check=True)
        
        # Check if the .laz file was created successfully
        if os.path.exists(laz_file_path):
            # print(f"{laz_file_path} converted succesfully!")
            new_file_paths.append(laz_file_path)
            os.remove(las_file_path)
            # print(f"Removed {las_file_path}")
        else:
            print(f"Failed to convert {las_file_path}")
            input("Continue?")

    return new_file_paths


def spatial_chunk_iterator(las: lp.LasReader, chunk_size: int, bins: int = 5):
    chunk_per_bin = chunk_size // bins

    for points in las.chunk_iterator(chunk_size):
        x_coords, y_coords, z_coords = points.x, points.y, points.z
        
        # Calculate the continuity of points along X and Y:
        # sort by coord; calc consecutive diff; more contiguous axes will have less var in consecutive diff;
        x_var = np.var(np.diff(np.sort(x_coords)))
        y_var = np.var(np.diff(np.sort(y_coords)))

        # Choose the axis with the lower variance => more continuity
        if x_var < y_var:
            sorted_indices = np.lexsort((z_coords, y_coords, x_coords))
        else:
            sorted_indices = np.lexsort((z_coords, x_coords, y_coords))
        sorted_points = points[sorted_indices]

        for i in range(0, len(sorted_points), chunk_per_bin):
            offset = min(i + chunk_per_bin, len(sorted_points))
            yield sorted_points[i:offset] # yileds las objects


class Labelec_Dataset(Dataset):
    """
    Labelec Dataset class for PyTorch

    Parameters
    ----------

    `las_data_dir` - str :
        directory where the dataset is stored in LAS format

    `split` - str :
        split of the dataset to use. Options: 'fit', 'test'

    `save_chunks` - bool :
        whether to save the chunks of the las files; if False, the chunks are assumed to be saved already

    `chunk_size` - int :
        size of the chunks to process

    `bins` - int :
        number of bins to divide the chunk into; chunk_size // bins is the size of sample

    `transform` - (None, torch.Transform) :
        transformation to apply to the point clouds

    `load_into_memory` - bool :
        whether to load the entire dataset into memory
    """
    def __init__(self,
                las_data_dir,
                split = 'fit',
                save_chunks = False,
                chunk_size = 10_000_000,
                bins = 5,
                transform = None,
                load_into_memory = False,
            ) -> None:
        super().__init__()

        las_dir = os.path.join(las_data_dir, split)
        self.chunk_dir = os.path.join(las_dir, 'chunks/')
        self.split = split
        self.transform = transform

        if save_chunks:
            assert isinstance(bins, (float, int)) and isinstance(chunk_size, (float, int)) and bins <= chunk_size

            las_file_paths = np.array([
                    os.path.join(las_dir, file_name) for file_name in os.listdir(las_dir) if file_name.endswith('.las') or file_name.endswith('.laz')
                ])
            
            self.chunk_file_paths = []
            
            for las_file in las_file_paths:
                chunk_file_paths = self._save_chunks(las_file, chunk_size, bins, to_laz=True)
                self.chunk_file_paths += chunk_file_paths
        else:
            self.chunk_file_paths = [
                os.path.join(self.chunk_dir, file_name) for file_name in os.listdir(self.chunk_dir)
            ]

        self.chunk_file_paths = np.array(self.chunk_file_paths)

        self.load_into_memory = False
        if load_into_memory:
            self._load_data_into_memory()
            self.load_into_memory = True

    def __len__(self):
        return len(self.chunk_file_paths)

    def __str__(self) -> str:
        return f"Labelec {self.split} Dataset in {self.chunk_dir} with {len(self)} samples"
    
    def _load_data_into_memory(self):
        self.data = []
        for i in tqdm(range(len(self)), desc="Loading data into memory..."):
            self.data.append(self.__getitem__(i))

    def __getitem__(self, index):
        if self.load_into_memory:
            return self.data[index]
        

        # upload las file
        chunk_las_file = self.chunk_file_paths[index]
        with lp.open(chunk_las_file) as chunk_las:
            chunk_las = chunk_las.read()
            chunk = eda.las_to_numpy(chunk_las, include_feats=['classification', 'rgb']) #xyz;rgb;label

        chunk = torch.from_numpy(chunk)

        sample = (chunk[:, :-1], chunk[:, -1])

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def _save_chunks(self, las_file_path: str, chunk_size: int, bins: int = 5, to_laz=True) -> list[str]:
        file_paths = []
        file_counter = 0

        with lp.open(las_file_path, mode='r') as las_open:
            # header for the chunk las files
            header = lp.LasHeader(point_format=las_open.header.point_format, version=las_open.header.version)

            for _, las_chunk in tqdm(enumerate(spatial_chunk_iterator(las_open, chunk_size, bins)), desc=f"Saving Chunks of {las_file_path}..."):
                
                # if file_counter < len(list(os.listdir(self.chunk_dir))): # skip already saved chunks
                #     file_counter += 1
                #     continue

                file_name = f"chunk_{file_counter}.laz" if to_laz else f"chunk_{file_counter}.las"
                file_path = os.path.join(self.chunk_dir, file_name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with lp.open(file_path, header=header, mode="w") as las_write:
                    buffer_chunk = lp.PackedPointRecord.zeros(point_count=len(las_chunk), point_format=header.point_format)
                    buffer_chunk.copy_fields_from(las_chunk)
                    las_write.write_points(buffer_chunk)

                file_counter += 1
                file_paths.append(file_path)

        return file_paths
    


if __name__ == '__main__':
    from utils import constants as consts

    LABELEC_DIR = consts.LABELEC_RGB_DIR

    labelec = Labelec_Dataset(
        las_data_dir=LABELEC_DIR,
        split='test',
        save_chunks=False,
        chunk_size=20_000_000,
        bins=20,
        transform=None,
        load_into_memory=False
    )


    x, y = labelec[0]

    print(x.shape, y.shape)

    rgb = x[:, 3:]
    print(x[:10, :])
    print(y[:10])
