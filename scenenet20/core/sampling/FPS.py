
import torch
from typing import Union, Tuple, Optional


import sys
sys.path.append('../../')

import core.models.giblinet.conversions as conversions
import pointops as pops

class Farthest_Point_Sampling:

    def __init__(self, num_points: int) -> None:
        self.num_points = num_points


    # def __call__(self, x:torch.Tensor) -> torch.Tensor:
    #     """
    #     Parameters
    #     ----------
    #     x - torch.Tensor
    #         input tensor of shape ([B], N, 3 + F)

    #     Returns
    #     -------
    #     q_points - torch.Tensor
    #         query point coords of shape ([B], num_points, 3)
    #     """
    #     import torch_fpsample
    #     # from torch_cluster import fps

    #     pointcloud = x[..., :3]

    #     if pointcloud.ndim == 3: # if the input is a batch of point clouds
    #         B, _, F = pointcloud.shape
    #         batch_vector = torch.arange(pointcloud.shape[0], device=pointcloud.device).repeat_interleave(pointcloud.shape[1])
    #         pointcloud = pointcloud.reshape(-1, pointcloud.shape[-1]) # shape = (B*N, 3 + F)
    #         self.num_points = self.num_points * B
    #     else:
    #         B = None
    #         batch_vector = None

    #     start_idx = torch.randint(0, pointcloud.shape[0], (1,)).item()
    #     fps_indices = torch_fpsample.sample(pointcloud.cpu(), self.num_points, start_idx=start_idx)[1].to(pointcloud.device)
    #     q_points = x[fps_indices]

    #     if B is not None:
    #         q_points = q_points.reshape(B, -1, q_points.shape[-1]).contiguous()
    #         # fps_indices = fps_indices.view(B, -1)

    #     return q_points
    
    
    
    def __call__(self, x:torch.Tensor, get_idxs=False) -> torch.Tensor:
        """
        Parameters
        ----------
        x - torch.Tensor
            input tensor of shape ([B], N, 3 + F)

        Returns
        -------
        q_points - torch.Tensor
            query point coords of shape ([B], num_points, 3)
        """

        pointcloud = x[..., :3]

        if pointcloud.ndim < 3:
            pointcloud = pointcloud.unsqueeze(0) # create fake batch dimension
            
        offset = conversions.get_offset_vector(pointcloud).contiguous()
        B, N, F = pointcloud.shape
        pointcloud = pointcloud.view(-1, 3).to(torch.float32).contiguous() # shape: (B*N, 3)
        
        fps_offset = torch.arange(self.num_points, (B + 1)*self.num_points, self.num_points, device=pointcloud.device).contiguous()
        
        fps_idxs = pops.farthest_point_sampling(pointcloud, offset, fps_offset)
        # print(f"{fps_offset.shape=} {fps_idxs.shape=}")
        fps_pcd = pointcloud[fps_idxs]
        # print(f"{fps_pcd.shape=}")
        pointcloud = conversions.build_batch_tensor(fps_pcd, fps_offset)
        # print(f"{pointcloud.shape=}")
        
        if get_idxs:
            return pointcloud, fps_idxs
        
        return pointcloud



if __name__ == '__main__':
    
    from utils import constants
    from core.datasets.TS40K import TS40K_FULL_Preprocessed
    from torch.utils.data import DataLoader
    import utils.pointcloud_processing as eda
    
    ts40k = TS40K_FULL_Preprocessed(
        constants.TS40K_FULL_PREPROCESSED_PATH, 
        split='fit', 
        sample_types=['tower_radius', '2_towers'], 
        transform=None, 
        load_into_memory=False
    )
    
    dt = DataLoader(ts40k, batch_size=16, shuffle=True)
    
    for batch in dt:
        xyz, y = batch
        xyz, y = xyz.cuda(), y.cuda() 
        
        print(f"{xyz.shape=} {y.shape=}")
        
        fps = Farthest_Point_Sampling(5000)
        q_points, fps_idxs = fps(xyz)
        q_y = y.reshape(-1)[fps_idxs]
        fps_offset = torch.arange(fps.num_points, (dt.batch_size + 1)*fps.num_points, fps.num_points, device=y.device)
        q_y = conversions.build_batch_tensor(q_y, fps_offset)
        
        print(f"{q_points.shape=} {q_y.shape=}") 
        
        # select the first sample
        xyz = xyz[0].cpu()
        y = y[0].cpu()
        q_points = q_points[0].cpu()
        q_y = q_y[0].cpu()
        
                
        # visualize original sample
        labels = y.reshape(-1).numpy()
        pcd = xyz.squeeze().numpy()
        print(f"{labels.shape=} {pcd.shape=}")
        pynt = eda.np_to_ply(pcd)
        eda.color_pointcloud(pynt, labels, use_preset_colors=True)
        eda.visualize_ply([pynt]) 
        
        # visualize fps sample
        labels = q_y.reshape(-1).numpy()
        pcd = q_points.squeeze().numpy()
        print(f"{labels.shape=} {pcd.shape=}")
        pynt = eda.np_to_ply(pcd)
        eda.color_pointcloud(pynt, labels, use_preset_colors=True)
        eda.visualize_ply([pynt]) 
     
        
         
