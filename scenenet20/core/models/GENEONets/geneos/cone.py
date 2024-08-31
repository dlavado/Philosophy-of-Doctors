
import torch
import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
sys.path.insert(3, '../../../..')
from core.models.GENEONets.geneos.GIB_Stub import GIB_Stub, GIB_PARAMS, NON_TRAINABLE


class Cone(GIB_Stub):

    def __init__(self, kernel_reach, **kwargs):
        """
        GIB that encodes a cone.\n

        Required
        --------
        `radius` - float \in  ]0, kernel_size[1]]:
        cone's base radius

        `inc` - float \in ]0, 1[
        cone's inclination
        
        `apex` - int \in [0, kernel_size[0]-1]
            cone's height
        """

        super().__init__(kernel_reach, angles=kwargs.get('angles', None))  

        if kwargs.get('apex') is None:
            raise KeyError("Provide a height for the cone.")

        if kwargs.get('inc') is None:
            raise KeyError("Provide an inclination for the cone.")
        
        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cone.")

        self.apex = kwargs['apex']#.to(self.device)
        # self.apex = self._to_parameter(self.apex)
        self.cone_inc = kwargs['inc']#.to(self.device)
        # self.cone_inc = self._to_parameter(self.cone_inc)
        # self.cone_inc = self._to_tensor(self.cone_inc)
        self.cone_radius = kwargs['radius']#.to(self.device)
        # self.cone_radius = self._to_parameter(self.cone_radius)

        self.intensity = kwargs.get('intensity', 1)#.to(self.device)
            
    
    def mandatory_parameters():
        return ['apex', 'radius','inc', 'cylinder_radius']

    def gib_parameters():
        return Cone.mandatory_parameters() + ['intensity']

    
    def gib_random_config(kernel_reach):
        rand_config = GIB_Stub.gib_random_config(kernel_reach)

        geneo_params = {
            'radius' : kernel_reach / torch.randint(1, kernel_reach*2, (1,))[0],
            'inc' : torch.rand(1,)[0], #float \in [0, 1]
            'apex': torch.randint(0, kernel_reach-1, (1,))[0],
            'intensity' : torch.randint(5, 10, (1,))[0] / 5 #float \in [1, 2]
        }   
        
        rand_config[GIB_PARAMS].update(geneo_params)
        rand_config[NON_TRAINABLE] = ['apex']

        return rand_config


    def gaussian(self, x:torch.Tensor, rad=None) -> torch.Tensor:
        rad = self.cone_radius if rad is None else rad

        x_c_norm = torch.linalg.norm(x, dim=1)
        gauss_dist = x_c_norm**2
        return self.intensity * torch.exp((gauss_dist**2) * (-1 / (2*(rad + self.epsilon)**2)))
    
    def compute_integral(self) -> torch.Tensor:
        """
        Computes an integral approximation of the gaussian function within the kernel_reach.

        Returns
        -------
        `integral` - torch.Tensor:
            Tensor of shape (1,) representing the integral of the gaussian function within the kernel reach.
        """
        # import matplotlib.pyplot as plt
        # calculate the integral of the gaussian function in a `self.kernel_reach` ball radius
        cone_inc = torch.clamp(self.cone_inc, 0, 0.499) # tan is not defined for 90 degrees
        mc_height = self.apex - self.montecarlo_points[:, 2]
        radius = self.cone_radius*mc_height*torch.tan(cone_inc*torch.pi) # cone's radius at the height of the support point
        gaussian_x = self.gaussian(self.montecarlo_points[:, :2], rad=radius)
        # print(f"{gaussian_x.shape=}")
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x_inside = self.montecarlo_points[mask_inside]
        # mask_high = gaussian_x > 0.1
        # x_inside = x_inside[mask_high]
        # x_inside = x_inside.cpu().detach().numpy()
        # ax.scatter(x_inside[:, 0], x_inside[:, 1], x_inside[:, 2], c=gaussian_x[mask_high].cpu().detach().numpy(), cmap='magma')
        # plt.show()  
        integral = torch.sum(gaussian_x)
        return integral

    
    def forward(self, points:torch.Tensor, query_idxs:torch.Tensor, supports_idxs:torch.Tensor) -> torch.Tensor:
     
        cone_inc = torch.clamp(self.cone_inc, 0, 0.499) # tan is not defined for 90 degrees

        q_output = torch.zeros(len(query_idxs), dtype=points.dtype, device=points.device)
        for i, q in enumerate(query_idxs):
            center = points[q] # 1x3
            support_points = points[supports_idxs[i]] #Kx3
            # center the support points
            s_centered = support_points - center
            # rotate the support points
            # s_centered = s_centered.to(self.device)
            s_centered = self.rotate(s_centered)
            s_height = self.apex - support_points[:, 2]
            radius = self.cone_radius*s_height*torch.tan(cone_inc*torch.pi) #S; cone's radius at the height of the support point
            weights = self.gaussian(s_centered[:, :2], rad=radius) # Kx1;
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
    points = torch.rand((100_000, 3))
    query_idxs = farthest_point_pooling(points, 20)
    q_points = points[query_idxs]
    num_neighbors = 20
    # neighbors = k_radius_ball(q_points, points, 0.2, 10, loop=True)
    _, neighbors_idxs = torch_knn(q_points, points, num_neighbors)

    print(points.shape)
    print(neighbors_idxs.shape)
    print(query_idxs.shape)

    cone = Cone(kernel_reach=1, cone_radius=0.1, cone_inc=0.1, apex=1, intensity=0.1)

    cone_weights = cone.forward(points, query_idxs, neighbors_idxs)
    print(cone_weights.shape)
    print(cone_weights)

    # plot q_points + kernel
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=cone_weights, cmap='magma')

    plt.show()  
