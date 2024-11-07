

import torch

from core.models.giblinet.geneos.GIB_Stub import GIB_Stub, GIBCollection, GIB_PARAMS


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
             
            'radius' : torch.rand(1)[0] * kernel_reach + 0.01, # float \in ]0, kernel_reach]
            'width' : torch.rand(1)[0] * kernel_reach + 0.01,  # float \in ]0, kernel_reach]
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
        # print(f"{weights.shape=} {weights.device=}")
        # print(f"{s_centered.shape=} {s_centered.device=}")
        # in_width_mask = torch.abs(s_centered[..., 2]) <= self.width
        weights = weights * torch.relu(self.width - torch.abs(s_centered[..., 2])) # Zero out weights outside the disk's width
        # weights = weights * (s_centered[..., 2] <= self.width).float() # Zero out weights outside the disk's width
        weights = weights *  valid_mask.float()
        
        weights = self.sum_zero(weights) # (B, M, K)
        q_output = torch.sum(weights, dim=-1) # (B, M)

        if not batched:
            q_output = q_output.squeeze(0)

        return q_output
    
    
    
class DiskCollection(GIBCollection):
    
    
    def __init__(self, kernel_reach:float, num_gibs, **kwargs):
        """
        Collection of Ellipsoid GIBs.
        
        Parameters
        ----------
        `kernel_reach` - float:
            Maximum reach of the kernel.
        
        `num_gibs` - int:
            Number of GIBs in the collection.
        
        Required Kwargs
        --------------
        `radius` - float:
            radius of the disk; radius <= kernel_reach;

        `width` - float:
            width of the disk; width <= kernel_reach;
        """
        
        super().__init__(kernel_reach, num_gibs=num_gibs, angles=kwargs.get('angles', None), intensity=kwargs.get('intensity', 1))
        
        self.radius = kwargs.get('radius', None)
        self.width = kwargs.get('width', None)
        
        if self.radius is None:
            raise KeyError("Provide a radius for the disk in the kernel.")
        
        if self.width is None:
            raise KeyError("Provide a width for the disk in the kernel.")
        
    def mandatory_parameters():
        return ['radius', 'width']
    
    def gib_parameters():
        return DiskCollection.mandatory_parameters() + ['intensity']
    
    def gib_random_config(num_gibs, kernel_reach):
        rand_config = GIBCollection.gib_random_config(num_gibs, kernel_reach)

        disk_params = {
            'radius' : torch.rand((num_gibs, 1)) * kernel_reach + 0.01, # float \in ]0, kernel_reach]
            'width' : torch.rand((num_gibs, 1)) * kernel_reach + 0.01,  # float \in ]0, kernel_reach]
        }

        rand_config[GIB_PARAMS].update(disk_params)

        return rand_config
        
    
    def gaussian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function of the Disk GIB for the input tensor.

        Parameters
        ----------
        `x` - torch.Tensor:
            Tensor of shape (..., G, K, 2) representing the input tensor. 
            Where G is the number of GIBs, and K is the number of neighbors and their dimensions.

        Returns
        -------
        `gaussian` - torch.Tensor:
            Tensor of shape (..., G, K) representing the gaussian function of the input tensor.
        """
        x_norm = torch.linalg.norm(x, dim=-1) # shape (..., G, K)
        return self.intensity * torch.exp((x_norm**2) * (-1 / (2*(self.radius + self.epsilon)**2))) # Kx1
    
    def compute_integral(self) -> torch.Tensor:
        mc_weights = self._compute_gib_weights(self.montecarlo_points)
        # print(f"{mc_weights.shape=}")
        # for g in range(self.num_gibs):
        #     self._plot_integral(mc_weights[g])
        return torch.sum(mc_weights, dim=-1) # (G, K)
    
    
    def _compute_gib_weights(self, s_centered: torch.Tensor) -> torch.Tensor:
        """
        Computes the weights of the support points for each GIB in the collection.

        Parameters
        ----------
        `s_centered` - torch.Tensor:
            Tensor of shape (..., K, 3), representing the centered support points.

        Returns
        -------
        `weights` - torch.Tensor:
            Tensor of shape (..., G, K), representing the weights of the support points for each GIB.
        """
        weights = self.gaussian(s_centered[..., :2])
        weights = weights * torch.relu(self.width - torch.abs(s_centered[..., 2]))   
        return weights
    
    
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
            Tensor of shape ([B], M, G), representing the output of the Disk GIB on the query points.
        """

        ##### prep for GIB computation #####
        s_centered, valid_mask, batched = self._prep_support_vectors(points, q_points, support_idxs)
        
        # Compute GIB weights; (B, M, K, 2) -> (B, M, K)
        weights = self._compute_gib_weights(s_centered)
        
        ### Post Processing ###
        q_output = self._validate_and_sum(weights, valid_mask) # (B, M, G)

        if not batched:
            q_output = q_output.squeeze(0)

        return q_output


if __name__ == '__main__':
    from core.neighboring.radius_ball import keops_radius_search
    from core.neighboring.knn import torch_knn
    from core.pooling.fps_pooling import fps_sampling
    
    # generate some points, query points, and neighbors. For the neighbors, I want to test two scenarios: 
    # 1) where the neighbors are at radius distance from the query points
    # 2) where the neighbors are very distance fromt the query points, at least 2*radius distance
    points = torch.rand((3, 100_000, 3))
    q_points = fps_sampling(points, num_points=1_000)
    print(f"{q_points.shape=}")
    num_neighbors = 16
    # neighbors_idxs = keops_radius_search(q_points, points, 0.2, num_neighbors, loop=True)
    _, neighbors_idxs = torch_knn(q_points, q_points, num_neighbors)


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
    # q_points = q_points[0]
    # q_points = q_points.cpu().numpy()
    # disk_weights = disk_weights[0]
    # disk_weights = disk_weights.cpu().detach().numpy()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=disk_weights, cmap='magma')

    # plt.show()  
    
    num_gibs = 2
    
    radius = torch.tensor([0.5]).repeat(num_gibs, 1)
    width = torch.tensor([0.1]).repeat(num_gibs, 1)
    radius[1] = 0.1
    width[1] = 0.8
    
    
    disk_collection = DiskCollection(0.2, num_gibs, radius=radius, width=width)
    
    disk_weights = disk_collection.forward(points, q_points, neighbors_idxs)
    
    print(disk_weights.shape)
    
    q_points = q_points.cpu().numpy()
    q_points = q_points[0]
    disk_weights = disk_weights.cpu().detach().numpy()
    disk_weights = disk_weights[0]
    disk_weights = disk_weights[:, 0]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=disk_weights, cmap='magma')
    plt.show()
    

    
    

