
import torch
from core.models.giblinet.geneos.GIB_Stub import GIB_Stub, GIBCollection, GIB_PARAMS, NON_TRAINABLE, to_parameter, to_tensor, gaussian_2d, _prep_support_vectors, _retrieve_support_points

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
        """

        super().__init__(kernel_reach, angles=kwargs.get('angles', None))  

        if kwargs.get('inc') is None:
            raise KeyError("Provide an inclination for the cone.")
        
        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cone.")

        # self.apex = kwargs['apex']#.to(self.device)
        self.inc = kwargs['inc']#.to(self.device)
        if isinstance(self.inc, float):
            self.inc = to_tensor(self.inc)
        self.radius = kwargs['radius']#.to(self.device)

        self.intensity = kwargs.get('intensity', 1)#.to(self.device)
            
    
    def mandatory_parameters():
        return ['radius','inc', 'cylinder_radius']

    def gib_parameters():
        return Cone.mandatory_parameters() + ['intensity']

    
    def gib_random_config(kernel_reach):
        rand_config = GIB_Stub.gib_random_config(kernel_reach)

        geneo_params = {
            'radius' : torch.rand(1)[0] * kernel_reach + 0.01, # float \in [0.01, kernel_reach]
            'inc' : torch.rand(1,)[0], #float \in [0, 1]
            'intensity' : torch.randint(5, 10, (1,))[0] / 5 #float \in [1, 2]
        }   
        
        rand_config[GIB_PARAMS].update(geneo_params)

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
        mc_height = self.montecarlo_points[..., 2]
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
        support_points = _retrieve_support_points(points, support_idxs)
        valid_mask = (support_idxs != -1) # Mask out invalid indices with -1; shape (B, M, K)

        # Center support points: (B, M, K, 3) - (B, M, 1, 3)
        s_centered = support_points - q_points.unsqueeze(2) # (B, M, K, 3)
        s_centered = self.rotate(s_centered)

        # Compute GIB weights; (B, M, K, 3) -> (B, M, K)
        s_height = support_points[..., 2] # (B, M, K)
        radius = self.radius*s_height*torch.tan(cone_inc*torch.pi) #S; cone's radius at the height of the support point
        weights = self.gaussian(s_centered[..., :2], rad=radius) # Kx1;

        weights = weights * valid_mask.float()
        weights = self.sum_zero(weights)
        q_output = torch.sum(weights, dim=-1)

        if not batched:
            q_output = q_output.squeeze(0)

        return q_output
    
    
    
class ConeCollection(GIBCollection):
    
    def __init__(self, kernel_reach:float, num_gibs, **kwargs):
        """
        Collection of Cone GIBs.
        
        Parameters
        ----------
        
        `kernel_reach` - float:
            reach of the kernel
            
        `num_gibs` - int:
            number of GIBs to generate
            
        Required
        --------
        `radius` - float \in  ]0, kernel_size[1]]:
            cone's base radius

        `inc` - float \in ]0, 1[
            cone's inclination
        """
        
        
        super().__init__(kernel_reach, num_gibs=num_gibs, angles=kwargs.get('angles', None), intensity=kwargs.get('intensity', 1))
        
        self.radius = kwargs.get('radius', None)
        self.inc = kwargs.get('inc', None)
        
        if self.radius is None:
            raise KeyError("Provide a radius for the cone.")
        
        if self.inc is None:
            raise KeyError("Provide an inclination for the cone.")
            
    
    def mandatory_parameters():
        return ['radius', 'inc']

    def gib_parameters():
        return ConeCollection.mandatory_parameters() + ['intensity']

    
    def gib_random_config(num_gibs, kernel_reach):
        rand_config = GIBCollection.gib_random_config(num_gibs, kernel_reach)

        geneo_params = {
            'radius' : torch.rand((num_gibs, 1)) * kernel_reach + 0.01, # float \in [0.01, kernel_reach]
            'inc' : torch.rand((num_gibs, 1)) / 2, #float \in [0, 0.5]
        }
        
        rand_config[GIB_PARAMS].update(geneo_params)

        return rand_config

    
    def gaussian(self, x:torch.Tensor, rad:torch.Tensor) -> torch.Tensor:
        # x_norm = torch.linalg.norm(x, dim=-1)
        # print(f"{x_norm.shape=} {rad.shape=}")
        # return self.intensity * torch.exp(torch.linalg.norm(x, dim=-1) * (-1 / (2*(rad + self.epsilon)**2)))
        return self.intensity * gaussian_2d(x, rad + self.epsilon)

    
    
    def _compute_gib_weights(self, s_centered: torch.Tensor) -> torch.Tensor:
        """
        Computes the weights of the GIBs for the given support points.
        
        Parameters
        ----------
        `s_centered` - torch.Tensor:
            Tensor of shape ([B], M, G, K, 3), representing the centered support points for each query point and GIB.
        
        Returns
        -------
        `weights` - torch.Tensor:
            Tensor of shape ([B], M, G, K), representing the weights of the GIBs for each query point.
        """
        cone_inc = torch.fmod(self.inc, 0.499) # cone inclination is in range [0, 0.499]
        cone_inc.mul_(torch.pi)
        
        #                                   s_height pointing up;             the radius of the cone at the height of the support point
        cone_inc = torch.relu(self.radius * torch.relu(-s_centered[..., 2]) * cone_inc) # (G,)
        
        # Compute weights with Gaussian
        weights = self.gaussian(s_centered[..., :2], rad=cone_inc) 
        return weights
        
    
    def forward(self, points:torch.Tensor, q_points:torch.Tensor, support_idxs:torch.Tensor) -> torch.Tensor:
        """
        Generalized version that computes the output of the Cone GIB on the query points
        given the support points
        
        
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
            Tensor of shape ([B], M, G), representing the output of the Cone GIB on the query points.
        """
        ##### prep for GIB computation #####
        s_centered, valid_mask, batched = _prep_support_vectors(points, q_points, support_idxs)
        s_centered = s_centered.unsqueeze(2).expand(-1, -1, self.num_gibs, -1, -1)
        montecarlo_points = torch.rand((int(10_000), 3), device=s_centered.device) * 2 * self.kernel_reach - self.kernel_reach # \in [-kernel_reach, kernel_reach]
        montecarlo_points = montecarlo_points[torch.norm(montecarlo_points, dim=-1) <= self.kernel_reach]
        q_output = self._prepped_forward(s_centered, valid_mask, montecarlo_points)
        
        return q_output
    
    
    
    
class HollowConeCollection(ConeCollection):
    
    def __init__(self, kernel_reach: float, num_gibs, **kwargs):
        """
        A collection of Hollow Cone GIBs.

        Parameters
        ----------
        `radius` - torch.Tensor:
            Tensor of shape (num_gibs, 1) containing the radius of the cone's base for each GIB.

        `intensity` - torch.Tensor:
            Tensor of shape (num_gibs, 1) representing scalar intensities for each cone GIB.

        `thickness` - torch.Tensor:
            Tensor of shape (num_gibs, 1) containing the thickness of the cone's base for each GIB.
        """

        super().__init__(kernel_reach, num_gibs=num_gibs, **kwargs)

        if kwargs.get('thickness') is None:
            raise KeyError("Provide a thickness for the hollow cone in the kernel.")

        self.thickness = kwargs['thickness']

    def mandatory_parameters():
        return ['radius', 'inc', 'thickness']

    def gib_parameters():
        return HollowConeCollection.mandatory_parameters() + ['intensity']

    def gib_random_config(num_gibs, kernel_reach):
        rand_config = GIBCollection.gib_random_config(num_gibs, kernel_reach)

        geneo_params = {
            'radius': torch.rand((num_gibs, 1)) * kernel_reach + 0.01,  # float \in [0.01, kernel_reach]
            'inc': torch.rand((num_gibs, 1)) / 2,  # float \in [0, 0.5]
            'thickness': torch.rand((num_gibs, 1)) * kernel_reach + 0.01,  # float \in [0.01, kernel_reach]
        }
        rand_config[GIB_PARAMS].update(geneo_params)

        return rand_config

    def gaussian(self, x: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function of the Hollow Cone GIB for the input tensor.

        Parameters
        ----------
        `x` - torch.Tensor:
            Tensor of shape (..., G, K, 2) representing the input tensor.
            Where G is the number of GIBs, and K is the number of neighbors and their dimensions.

        `rad` - torch.Tensor:
            Tensor of shape (..., G, K) representing the radius at the height of the support points.

        Returns
        -------
        `gaussian` - torch.Tensor:
            Tensor of shape (..., G, K) representing the gaussian function of the input tensor.
        """
        outer_gaussian = self.intensity * gaussian_2d(x, rad + self.epsilon)
        inner_gaussian = self.intensity * gaussian_2d(x, rad - self.thickness + self.epsilon)
        return outer_gaussian - inner_gaussian
    
    
    

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    sys.path.insert(1, '../..')
    sys.path.insert(2, '../../..')
    sys.path.insert(3, '../../../..')
    from core.models.giblinet.neighboring.radius_ball import keops_radius_search
    from core.models.giblinet.neighboring.knn import torch_knn
    from core.models.giblinet.pooling.fps_pooling import fps_sampling
    
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

    # cone = Cone(kernel_reach=0.3, radius=0.5, inc=0.1, apex=0, intensity=1.0)

    # cone_weights = cone.forward(points, q_points, neighbors_idxs)
    # print(cone_weights.shape)
    # print(cone_weights)

    # plot q_points + kernel
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # q_points = q_points.cpu().numpy()
    # q_points = q_points[0]
    # cone_weights = cone_weights.cpu().numpy()
    # cone_weights = cone_weights[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=cone_weights, cmap='magma')

    # plt.show()  
    
    
    ################ Test Cone Collection ################
    
    
    num_gibs = 2
    radius = torch.tensor([0.5]).repeat(num_gibs, 1)
    inc = torch.tensor([0.1]).repeat(num_gibs, 1)
    print(f"{radius.shape=} {inc.shape=}")
    angles = torch.tensor([0.0, 0.0, -180.0]).repeat(num_gibs, 1)
    cone_collection = ConeCollection(kernel_reach=0.3, num_gibs=num_gibs, radius=radius, inc=inc, angles=angles)
    
    cone_weights = cone_collection.forward(points, q_points, neighbors_idxs)
    print(cone_weights.shape)
    
    # plot q_points + kernel
    q_points = q_points.cpu().numpy()
    q_points = q_points[0]
    cone_weights = cone_weights.cpu().numpy()
    cone_weights = cone_weights[0]
    cone_weights = cone_weights[:, 0] # get the weights of the first cone
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=cone_weights, cmap='magma')
    plt.show()
