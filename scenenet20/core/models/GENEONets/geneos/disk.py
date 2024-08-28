

import torch

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
sys.path.insert(3, '../../../..')
from core.models.GENEONets.geneos.GIB_Stub import GIB_Stub


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
    
    def geneo_parameters():
        return Disk.mandatory_parameters() + ['intensity']
    
    def geneo_random_config(kernel_reach):
        rand_config = GIB_Stub.geneo_random_config(kernel_reach)

        disk_params = {
            'radius' : kernel_reach / torch.randint(1, kernel_reach*2, (1,))[0],
            'width' : kernel_reach / torch.randint(1, kernel_reach*2, (1,))[0],
            'intensity' : torch.rand(1)
        }

        rand_config['geneo_params'] = disk_params

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
        x_norm = torch.linalg.norm(x, dim=1, keepdim=True) # Kx1
        return self.intensity * torch.exp((x_norm**2) * (-1 / (2*(self.radius + self.epsilon)**2))) # Kx1
    

    def compute_integral(self) -> torch.Tensor:
        mc_weights = self.gaussian(self.montecarlo_points[:, :2]).squeeze()
        in_width_mask = torch.abs(self.montecarlo_points[:, 2]) <= self.width
        mc_weights = mc_weights * in_width_mask
        # print(mc_weights.shape)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x = self.montecarlo_points.cpu().detach().numpy()
        # ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=mc_weights.cpu().detach().numpy(), cmap='magma')
        # plt.show()  
        return torch.sum(mc_weights)    


    def forward(self, points: torch.Tensor, query_idxs: torch.Tensor, supports_idxs: torch.Tensor) -> torch.Tensor:
        
        q_output = torch.zeros(len(query_idxs), dtype=points.dtype, device=points.device)

        for i, q in enumerate(query_idxs):
            center = points[q] # 1x3
            support_points = points[supports_idxs[i]] #Kx3
            s_centered = support_points - center
            weights = self.gaussian(s_centered[:, :2]).squeeze()
            # zero the weights of all points outside the disk's width
            in_width_mask = torch.abs(s_centered[:, 2]) <= self.width
            weights = weights * in_width_mask

            weights = self.sum_zero(weights)
            q_output[i] = torch.sum(weights) 

        return q_output

if __name__ == '__main__':
    from core.neighboring.radius_ball import k_radius_ball
    from core.neighboring.knn import torch_knn
    from core.pooling.farthest_point import farthest_point_pooling
    
    # generate some points, query points, and neighbors. For the neighbors, I want to test two scenarios: 
    # 1) where the neighbors are at radius distance from the query points
    # 2) where the neighbors are very distance fromt the query points, at least 2*radius distance
    points = torch.rand((100_000, 3))
    query_idxs = farthest_point_pooling(points, 20)
    q_points = points[query_idxs]
    num_neighbors = 20
    # neighbors = k_radius_ball(q_points, points, 0.2, 10, loop=True)
    _, neighbors_idxs = torch_knn(q_points, q_points, num_neighbors)

    print(points.shape)
    print(neighbors_idxs.shape)
    print(query_idxs.shape)

    disk = Disk(0.2, radius=0.1, width=0.01)

    disk_weights = disk.forward(points, query_idxs, neighbors_idxs)
    print(disk_weights.shape)
    print(disk_weights)

    # plot q_points + kernel
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=disk_weights.cpu().detach().numpy(), cmap='magma')

    plt.show()  

    
    

