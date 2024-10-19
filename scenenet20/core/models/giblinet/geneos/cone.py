
import torch
import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
sys.path.insert(3, '../../../..')
from .GIB_Stub import GIB_Stub, GIB_PARAMS, NON_TRAINABLE, to_parameter, to_tensor


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
        self.inc = kwargs['inc']#.to(self.device)
        # self.cone_inc = self._to_parameter(self.cone_inc)
        # self.cone_inc = self._to_tensor(self.cone_inc)
        if isinstance(self.inc, float):
            self.inc = to_tensor(self.inc)
        self.radius = kwargs['radius']#.to(self.device)

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
        rad = self.radius if rad is None else rad

        x_c_norm = torch.linalg.norm(x, dim=-1)

        return self.intensity * torch.exp((x_c_norm**2) * (-1 / (2*(rad + self.epsilon)**2)))
    
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
        cone_inc = torch.clamp(self.inc, 0, 0.499) # tan is not defined for 90 degrees
        mc_height = self.apex - self.montecarlo_points[..., 2]
        radius = self.radius*mc_height*torch.tan(cone_inc*torch.pi) # cone's radius at the height of the support point
        gaussian_x = self.gaussian(self.montecarlo_points[..., :2], rad=radius)
        # self._plot_integral(gaussian_x)
        integral = torch.sum(gaussian_x)
        return integral

    
    # def forward(self, points:torch.Tensor, q_points:torch.Tensor, supports_idxs:torch.Tensor) -> torch.Tensor:
     
    #     cone_inc = torch.clamp(self.cone_inc, 0, 0.499) # tan is not defined for 90 degrees

    #     q_output = torch.zeros(len(q_points), dtype=points.dtype, device=points.device)
    #     for i, center in enumerate(q_points):
    #         # center = points[q] # 1x3
    #         support_points = points[supports_idxs[i]] #Kx3
    #         # center the support points
    #         s_centered = support_points - center
    #         # rotate the support points
    #         # s_centered = s_centered.to(self.device)
    #         s_centered = self.rotate(s_centered)
    #         s_height = self.apex - support_points[:, 2]
    #         radius = self.cone_radius*s_height*torch.tan(cone_inc*torch.pi) #S; cone's radius at the height of the support point
    #         weights = self.gaussian(s_centered[:, :2], rad=radius) # Kx1;
    #         weights = self.sum_zero(weights)
    #         q_output[i] = torch.sum(weights)

    #     return q_output
    
    def forward(self, points:torch.Tensor, q_points:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:
        """
        Generalized version that computes the output of the Cone GIB on the query points
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
            Tensor of shape ([B], M), representing the output of the Cone GIB on the query points.
        """
        # Check if data is batched (points has shape (B, N, 3) or (N, 3))
        if points.dim() == 2:
            # If unbatched, add a batch dimension
            points = points.unsqueeze(0)
            q_points = q_points.unsqueeze(0)
            support_idxs = support_idxs.unsqueeze(0)
            batched = False
        else:
            batched = True

        cone_inc = torch.clamp(self.inc, 0, 0.499) # tan is not defined for 90 degrees

        # Gather support points: (B, M, K) -> (B, M, K, 3)
        support_points = self._retrieve_support_points(points, support_idxs)
        valid_mask = (support_idxs != -1) # Mask out invalid indices with -1; shape (B, M, K)

        # Center support points: (B, M, K, 3) - (B, M, 1, 3)
        s_centered = support_points - q_points.unsqueeze(2) # (B, M, K, 3)
        s_centered = self.rotate(s_centered)

        # Compute GIB weights; (B, M, K, 3) -> (B, M, K)
        s_height = self.apex - support_points[..., 2] # (B, M, K)
        radius = self.radius*s_height*torch.tan(cone_inc*torch.pi) #S; cone's radius at the height of the support point
        weights = self.gaussian(s_centered[..., :2], rad=radius) # Kx1;

        weights = weights * valid_mask.float()
        weights = self.sum_zero(weights)
        q_output = torch.sum(weights, dim=-1)

        if not batched:
            q_output = q_output.squeeze(0)

        return q_output


if __name__ == "__main__":
    from core.neighboring.radius_ball import k_radius_ball
    from core.neighboring.knn import torch_knn
    from core.pooling.fps_pooling import fps_sampling
    
    # generate some points, query points, and neighbors. For the neighbors, I want to test two scenarios: 
    # 1) where the neighbors are at radius distance from the query points
    # 2) where the neighbors are very distance fromt the query points, at least 2*radius distance
    points = torch.rand((3, 100_000, 3), device='cuda')
    q_points = fps_sampling(points, num_points=1_000)
    print(f"{q_points.shape=}")
    num_neighbors = 16
    neighbors_idxs = k_radius_ball(q_points, points, 0.2, num_neighbors, loop=True)
    # _, neighbors_idxs = torch_knn(q_points, q_points, num_neighbors)


    print(points.shape)
    print(neighbors_idxs.shape)
    print(q_points.shape)

    cone = Cone(kernel_reach=0.3, radius=0.5, inc=0.1, apex=0, intensity=1.0)

    cone_weights = cone.forward(points, q_points, neighbors_idxs)
    print(cone_weights.shape)
    print(cone_weights)

    # plot q_points + kernel
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    q_points = q_points.cpu().numpy()
    q_points = q_points[0]
    cone_weights = cone_weights.cpu().numpy()
    cone_weights = cone_weights[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=cone_weights, cmap='magma')

    plt.show()  
