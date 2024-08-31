
import torch
# import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
sys.path.insert(3, '../../../..')
from core.models.GENEONets.geneos.GIB_Stub import GIB_Stub, GIB_PARAMS, NON_TRAINABLE



class Cylinder(GIB_Stub):

    def __init__(self, kernel_reach, **kwargs):
        """
        GIB that encodes a cylinder.

        Parameters
        ----------
        `radius` - float:
            radius of the cylinder's base; radius <= kernel_size[1];

        `intensity` - float:
            variance for the gaussian function when assigning weights to the kernel;

        """

        super().__init__(kernel_reach, angles=kwargs.get('angles', None))

        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cylinder in the kernel.")

        self.radius = kwargs['radius']

        self.intensity = kwargs.get('intensity', 1)

    def mandatory_parameters():
        return ['radius']
    
    def gib_parameters():
        return Cylinder.mandatory_parameters() + ['intensity']

    def gib_random_config(kernel_reach):
        rand_config = GIB_Stub.gib_random_config(kernel_reach)

        geneo_params = {
            'radius' : torch.randint(1, 10, (1,))[0]/100, # float \in [0.1, 1]
            'intensity' : torch.randint(0, 10, (1,))[0]/5 # float \in [0, 2]
        }   
        rand_config[GIB_PARAMS].update(geneo_params)

        return rand_config


    def gaussian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function of the Cylinder GIB for the input tensor.

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
        """
        Computes an integral approximation of the gaussian function within the kernel_reach.

        Returns
        -------
        `integral` - torch.Tensor:
            Tensor of shape (1,) representing the integral of the gaussian function within the kernel reach.
        """
        # calculate the integral of the gaussian function in a `self.kernel_reach` ball radius
        gaussian_x = self.gaussian(self.montecarlo_points[:, :2]) # seeing as the cylinder is in 3D, we only consider the first two dimensions
        # print(f"{gaussian_x.shape=}")
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_inside[:, 0], x_inside[:, 1], x_inside[:, 2], c=gaussian_x, cmap='magma')
        # plt.show()  
        integral = torch.sum(gaussian_x)
        return integral
    

    def forward(self, points:torch.Tensor, query_idxs:torch.Tensor, supports_idxs:torch.Tensor) -> torch.Tensor:

        q_output = torch.zeros(len(query_idxs), dtype=points.dtype, device=points.device)
        for i, q in enumerate(query_idxs):
            center = points[q] # 1x3
            support = supports_idxs[i]
            support = support[support != -1]
            support_points = points[support] #Kx3
            # center the support points
            s_centered = support_points - center
            # rotate the support points
            # s_centered = s_centered.to(self.device)
            s_centered = self.rotate(s_centered)
            weights = self.gaussian(s_centered[:, :2]) # Kx1; seeing as the cylinder is in 3D, we only consider the first two dimensions
            weights = self.sum_zero(weights)
            q_output[i] = torch.sum(weights)

        return q_output


if __name__ == "__main__":
    from core.neighboring.radius_ball import k_radius_ball
    from core.neighboring.knn import torch_knn
    from core.pooling.farthest_point import farthest_point_pooling
    
    # generate some points, query points, and neighbors. For the neighbors, I want to test two scenarios: 
    # 1) where the neighbors are at radius distance from the query points
    # 2) where the neighbors are very distance fromt the query points, at least 2*radius distance
    points = torch.rand((100_000, 3), devicde='cuda')
    query_idxs = farthest_point_pooling(points, 20)
    q_points = points[query_idxs]
    num_neighbors = 20
    # neighbors = k_radius_ball(q_points, points, 0.2, 10, loop=True)
    _, neighbors_idxs = torch_knn(q_points, q_points, num_neighbors)


    print(points.shape)
    print(neighbors_idxs.shape)
    print(query_idxs.shape)

    cylinder = Cylinder(kernel_reach=1, radius=0.1)

    cy_weights = cylinder.forward(points, query_idxs, neighbors_idxs)
    print(cy_weights.shape)
    print(cy_weights)

    # plot q_points + kernel
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=cy_weights, cmap='magma')

    plt.show()    




