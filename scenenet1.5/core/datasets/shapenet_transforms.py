



from typing import Union
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesAtlas
from torch_transforms import Normalize_PCD, Farthest_Point_Sampling

import torch


class MeshToPCD:
    """
    Transform a mesh to a point cloud.

    Torch transform for ShapeNet dataset.

    Parameters
    ----------

    num_samples: int
        Number of points to sample from the mesh
    use_textures: bool
        Whether to use textures or not
    """

    def __init__(self, num_samples, use_textures=True) -> None:

        self.num_samples = num_samples
        self.use_textures = use_textures


    def __call__(self, mesh_dict) -> torch.Tensor:

        verts, faces = mesh_dict["verts"], mesh_dict["faces"]

        if self.use_textures:
            textures = mesh_dict["textures"]
            mesh = Meshes(verts=[verts], faces=[faces], textures=TexturesAtlas(textures[None, ...]))

            pcd, textures = sample_points_from_meshes(mesh, num_samples=self.num_samples, return_textures=True)

            print(pcd.shape, textures.shape)

            return torch.cat([pcd, textures], dim=-1) # shape = (num_samples, 6)
        else:
            mesh = Meshes(verts=[verts], faces=[faces])

            pcd = sample_points_from_meshes(mesh, num_samples=self.num_samples)

            return pcd



if __name__ == '__main__':
    from pytorch3d.datasets import ShapeNetCore
    import h5py
    import os
    import numpy as np
    import sys
    sys.path.insert(0, '..')
    sys.path.insert(1, '../..')
    import utils.pcd_processing as eda

    SHAPE_NET_CORE_PATH = "/media/didi/TOSHIBA EXT/ShapeNetCore"

    PART_NET_PATH = "/media/didi/TOSHIBA EXT/sem_seg_h5"

    
    categories = list(os.listdir(PART_NET_PATH))
    categories = [os.path.join(PART_NET_PATH, cat) for cat in categories]

    for cat_path in categories:

        for h5_file in os.listdir(cat_path):
            if '.h5' not in h5_file:
                continue    

            h5_path = os.path.join(cat_path, h5_file)

            with h5py.File(h5_path, 'r') as h5:

                print(h5["data"].shape)
                print(h5["label_seg"])

                pcd = eda.np_to_ply(h5["data"][0])
                eda.color_pointcloud(pcd, h5["label_seg"][0])
                eda.visualize_ply([pcd])

                input("Press Enter to continue...")


    shape = ShapeNetCore(SHAPE_NET_CORE_PATH, version=2, load_textures=True)

    for sample in shape:
        id = sample["model_id"]

                

    mesh_dict = shape[6]

    print(mesh_dict["label"])

    pcd = MeshToPCD(10000, True)(mesh_dict)

    pcd = Farthest_Point_Sampling(1024)(pcd)

    pcd = Normalize_PCD()(pcd)






    


