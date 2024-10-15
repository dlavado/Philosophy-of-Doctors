

import torch

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
sys.path.insert(3, '../../../..')
from core.models.GENEONets.geneos.GIB_Stub import GIB_Stub, GIB_PARAMS


class Disk(GIB_Stub):

    def __init__(self, kernel_reach:float, **kwargs):
        """
        GIB that encodes a disk.
        
        Parameters
        ----------
        `radius` - float:
            radius of the disk; radius <= kernel_reach;

        `width` - float:
            width of the disk; width <= kernel_reach;

        `intensity` - float:
            variance for the gaussian function when assigning weights to the kernel;
        """

        super().__init__(kernel_reach, angles=kwargs.get('angles', None))

        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the disk in the kernel.")
        
        if kwargs.get('width') is None:
            raise KeyError("Provide a width for the disk in the kernel.")

        self.radius = kwargs['radius']
        self.width = kwargs['width']

        self.intensity = kwargs.get('intensity', 1)


    def mandatory_parameters():
        return ['radius', 'width']
    
    def gib_parameters():
        return Disk.mandatory_parameters() + ['intensity']
    
    def gib_random_config(kernel_reach):
        rand_config = GIB_Stub.gib_random_config(kernel_reach)

        disk_params = {
             
            'radius' : torch.rand(1)[0] * kernel_reach, # float \in ]0, kernel_reach]
            'width' : torch.rand(1)[0] * kernel_reach,  # float \in ]0, kernel_reach]
            'intensity' : torch.rand(1)
        }

        rand_config[GIB_PARAMS].update(disk_params)

        return rand_config
        


    def gaussian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function of the Disk GIB for the input tensor.

        Parameters
        ----------
        `x` - torch.Tensor:
            Tensor of shape (K, 2) representing the input tensor.

        Returns
        -------
        `gaussian` - torch.Tensor:
            Tensor of shape (K, 1) representing the gaussian function of the input tensor.
        """
        x_norm = torch.linalg.norm(x, dim=-1) # Kx1
        return self.intensity * torch.exp((x_norm**2) * (-1 / (2*(self.radius + self.epsilon)**2))) # Kx1
    

    def compute_integral(self) -> torch.Tensor:
        mc_weights = self.gaussian(self.montecarlo_points[:, :2]).squeeze()
        in_width_mask = torch.abs(self.montecarlo_points[:, 2]) <= self.width
        mc_weights = mc_weights * in_width_mask
        # self._plot_integral(mc_weights)
        return torch.sum(mc_weights)    


    # def forward(self, points: torch.Tensor, q_points: torch.Tensor, supports_idxs: torch.Tensor) -> torch.Tensor:
        
    #     q_output = torch.zeros(len(query_idxs), dtype=points.dtype, device=points.device)

    #     for i, center in enumerate(q_points):
    #         # retrieve the query point and its support points
    #         # center = points[q] # 1x3
    #         support_points = points[supports_idxs[i]] #Kx3
    #         # center the support points
    #         s_centered = support_points - center
    #         # rotate the support points
    #         # s_centered = s_centered.to(self.device)
    #         s_centered = self.rotate(s_centered)
    #         # compute the weights of the support points
    #         weights = self.gaussian(s_centered[:, :2]).squeeze()
    #         # zero the weights of all points outside the disk's width
    #         in_width_mask = torch.abs(s_centered[:, 2]) <= self.width
    #         weights = weights * in_width_mask

    #         weights = self.sum_zero(weights)
    #         q_output[i] = torch.sum(weights) 

    #     return q_output


    def forward(self, points: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor) -> torch.Tensor:
        """
        Generalized version that computes the output of the Disk GIB on the query points
        given the support points, for either batched or unbatched data.
        
        Parameters
        ----------
        `points` - torch.Tensor:
            Tensor of shape ([B], N, 3), representing the point cloud.

        `q_points` - torch.Tensor:
            Tensor of shape ([B], M, 3), representing the query points; M <= N.

        `supports_idxs` - torch.Tensor[int]:
            Tensor of shape ([B], M, K), representing the indices of the support points for each query point; K <= N.

        Returns
        -------
        `q_outputs` - torch.Tensor:
            Tensor of shape ([B], M), representing the output of the Disk GIB on the query points.
        """

        if points.dim() == 2:
            # If unbatched, add a batch dimension
            points = points.unsqueeze(0)
            q_points = q_points.unsqueeze(0)
            support_idxs = support_idxs.unsqueeze(0)
            batched = False
        else:
            batched = True

        # Gather support points: (B, M, K) -> (B, M, K, 3)
        support_points = self._retrieve_support_points(points, support_idxs)
        valid_mask = (support_idxs != -1) # Mask out invalid indices with -1; shape (B, M, K)

        # Center support points: (B, M, K, 3) - (B, M, 1, 3)
        s_centered = support_points - q_points.unsqueeze(2) # (B, M, K, 3)
        s_centered = self.rotate(s_centered)

        # Compute GIB weights; (B, M, K, 2) -> (B, M, K)
        weights = self.gaussian(s_centered[..., :2])
        in_width_mask = torch.abs(s_centered[..., 2]) <= self.width
        weights = weights * in_width_mask.float() * valid_mask.float()
        
        weights = self.sum_zero(weights) # (B, M, K)
        q_output = torch.sum(weights, dim=-1) # (B, M)

        if not batched:
            q_output = q_output.squeeze(0)

        return q_output


if __name__ == '__main__':
    from core.neighboring.radius_ball import k_radius_ball
    from core.neighboring.knn import torch_knn
    from core.pooling.fps_pooling import fps_sampling
    
    # generate some points, query points, and neighbors. For the neighbors, I want to test two scenarios: 
    # 1) where the neighbors are at radius distance from the query points
    # 2) where the neighbors are very distance fromt the query points, at least 2*radius distance
    points = torch.rand((3, 100_000, 3), device='cuda')
    q_points = fps_sampling(points, num_points=1_000)
    print(f"{q_points.shape=}")
    num_neighbors = 8
    neighbors_idxs = k_radius_ball(q_points, points, 0.2, num_neighbors, loop=True)
    _, neighbors_idxs = torch_knn(q_points, points, num_neighbors)


    print(points.shape)
    print(neighbors_idxs.shape)
    print(q_points.shape)

    disk = Disk(0.2, radius=0.05, width=0.2)

    disk_weights = disk.forward(points, q_points, neighbors_idxs)
    print(disk_weights.shape)
    print(disk_weights)

    # plot q_points + kernel
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    q_points = q_points[0]
    q_points = q_points.cpu().numpy()
    disk_weights = disk_weights[0]
    disk_weights = disk_weights.cpu().detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=disk_weights, cmap='magma')

    plt.show()  

    
    

