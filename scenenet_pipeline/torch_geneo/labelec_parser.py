import argparse
from typing import List
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import re

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from EDA import EDA_utils as eda

ROOT_PROJECT = str(Path(os.getcwd()).parent.absolute())
CSV_DIR = os.path.join(ROOT_PROJECT, "labelec_csvs")
LAS_DIR = "/media/didi/TOSHIBA EXT/Labelec_LAS/"

SAVE_PATH = os.path.join(LAS_DIR, 'Processed_PCD')


def arg_parser():
    parser = argparse.ArgumentParser(description="Process Arguments")

    
    #radius used to cut around the given locations
    parser.add_argument('-r', '--radius', type=float, default=20)

    #CSV file with the relevant locations
    parser.add_argument('-i', "--input", type=str, default='input.csv') 

    return parser.parse_args()


def get_csv(file_path, filename):
    file_path = os.path.join(file_path, filename)
    return pd.read_csv(file_path)

def parse_input_csv(input_coords, merge = False):
    """
    Parses Labelec's input.
    All input .csv files should follow the format of `input5.csv`.

    Parameters
    ----------
    `input_coords` - pandas Dataframe:
        original .csv file in in pandas format

    `merge` - bool:
        merges the coords of each set in the input into a single numpy array
    
    Returns
    -------
    list of (N,3) numpy arrays with coordinates.
    Each list corresponds to a set of coordinates in the input .csv file
    """
    coords = input_coords['WKT'][0]
    coords = coords.split("((")
    coords = coords[1].split("))")[0]
    coords = "(" + coords + ")"
    # print(coords)
    list_coords = []
    index = 0
    while index < len(coords):
        index = coords.find('(', index)
        if index == -1:
            break
        aux_index = coords.find(')', index)
        list_coords.append(coords[index+1:aux_index].split(","))
        index = aux_index
    #print(list_coords[0])

    np_coords = []

    for l in range(len(list_coords)):
        np_list = np.empty((len(list_coords[l]), 3))
        for i in range(len(list_coords[l])):
            np_list[i] = np.array(list_coords[l][i].split(" "))
        np_coords.append(np_list)


    if merge:
        return merge_coords(np_coords)
    return np_coords


def parse_output_csv(output_coords, merge=False):

    list_coords = []
    output_coords = output_coords['WKT']

    for i in range(len(output_coords)):
        coords = output_coords[i]
        coords = coords.split("(")[1].split(")")[0]
        coords = coords.split(",")
        #print(coords)
        np_coords = np.empty((len(coords), 2))
        for j in range(len(coords)):
            np_coords[j] = np.array(coords[j].split(" "))
        np_coords = np.concatenate((np_coords, np.zeros((len(coords), 1)) ), axis=1)
        list_coords.append(np_coords)
    
    if merge:
        return merge_coords(list_coords)

    return list_coords


def merge_coords(coords:List[np.ndarray]) -> np.ndarray:
    
    merge = None

    for npc in coords:
        if merge is None:
            merge = npc
        else:
            merge = np.concatenate((merge, npc), axis = 0)

    return merge


def get_coords_in_pcd(xyz:np.ndarray, coords:np.ndarray) -> np.ndarray:
    """
    Returns the set of `coords` that intersect with the volume of `xyz`.\\

    Parameters
    ----------

    `xyz` - numpy.ndarray:
        Point cloud data without classes : (N, 3) format.

    `coords` - numpy-ndarray:
        coordinate candidates in (M, 3) format.

    Returns
    -------
        numpy array with coordinates

    """

    xyz_min, xyz_max = np.min(xyz, axis=0), np.max(xyz, axis=0)

    #print(xyz_min, xyz_max)

    min_mask = (coords[:, :-1][None] >= xyz_min[:-1]).all(axis=-1) # disregard z coord

    max_mask = (coords[:, :-1][None] <= xyz_max[:-1]).all(axis=-1) # disregard z coord

    mask = min_mask & max_mask

    return coords[mask[0]], mask[0] #decapsulates mask and applies it coords


def save_samples(save_path, samples):
    save_path = save_path + '/samples/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        counter = 0
    else:
        counter = len(os.listdir(save_path))

    print(f"PCD file content counter: {counter}")

    for npy in samples:
        npy_name = os.path.join(save_path, f'sample_{counter}.npy')

        with open(npy_name, 'wb') as f:
            np.save(f, npy)     # py: N, 4 (x, y, z, label)
            counter += 1


if __name__ == "__main__":

    parser = arg_parser()

    """
    Columns:
    Index(['WKT', 'Name', 'descriptio', 'timestamp', 'begin', 'end', 'altitudeMo',
    'tessellate', 'extrude', 'visibility', 'drawOrder', 'icon', 'snippet'],
    dtype='object')
    """
    in_coords = get_csv(CSV_DIR, parser.input)
    in_coords = parse_input_csv(in_coords, merge=True)
    print(in_coords.shape)

    # out_coords = get_csv(CSV_DIR, 'output3.csv')
    # out_coords = parse_output_csv(out_coords, merge=True)
    # print(out_coords.shape)


    for i, las in enumerate(os.listdir(LAS_DIR)):
        # this processes one LAS file at a time for resource efficiency


        xyz_files, class_files = eda.get_las_from_dir([LAS_DIR], file_idxs=[i], to_numpy=True)

        xyz, classes = eda.merge_pcds(xyz_files, class_files)

        if xyz is None: # the LAS file is empty
            print(f"Empty LAS File {i}")
            continue
        #print(xyz.shape, classes.shape)

        coords, mask = get_coords_in_pcd(xyz, in_coords)
        print(f"Coordinates in LAS file {i}: {coords.shape}")

        if len(coords) == 0: # there are no relevant coordinates in this LAS file
            continue

        locs = eda.crop_at_locations(xyz, coords, radius = parser.radius, classes=classes)

        print(locs[0].shape)

        #print(len(locs))
        assert len(locs) == len(coords)

        in_coords = in_coords[np.bitwise_not(mask)] # LAS file intersect, so we remove the processed coords

        print(f"Samples Left: {in_coords.shape}")

        #print(locs[0].shape)
        #eda.visualize_ply([eda.np_to_ply(locs[0])], window_name='location')

        # for loc in locs:
        #     eda.visualize_ply([eda.np_to_ply(loc)], window_name='location')

        save_samples(SAVE_PATH, locs)

        if len(in_coords) == 0:
            print("Done!")
            break
        else:
            print("")
   



    