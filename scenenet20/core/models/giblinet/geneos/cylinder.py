
import torch
from core.models.giblinet.geneos.GIB_Stub import GIB_Stub, GIBCollection, GIB_PARAMS, NON_TRAINABLE



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
            'radius' : torch.rand(1)[0] * kernel_reach + 0.01, # float \in [0.1, 1]
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
            Tensor of shape (K,) representing the gaussian function of the input tensor.
        """
        x_norm = torch.linalg.norm(x, dim=-1) # Kx1
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
        # self._plot_integral(gaussian_x)  
        integral = torch.sum(gaussian_x)
        return integral

    def forward(self, points: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor) -> torch.Tensor:
        """
        Generalized version that computes the output of the Cylinder GIB on the query points
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
            Tensor of shape ([B], M), representing the output of the Cylinder GIB on the query points.
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

        # Gather support points: (B, M, K) -> (B, M, K, 3)
        support_points = self._retrieve_support_points(points, support_idxs)
        valid_mask = (support_idxs != -1) # Mask out invalid indices with -1; shape (B, M, K)

        # Center support points: (B, M, K, 3) - (B, M, 1, 3)
        s_centered = support_points - q_points.unsqueeze(2) # (B, M, K, 3)
        s_centered = self.rotate(s_centered)

        # Compute GIB weights; (B, M, K, 2) -> (B, M, K)
        weights = self.gaussian(s_centered[..., :2])
        # print(f"{weights.shape=}")
        weights = weights * valid_mask.float()

        weights = self.sum_zero(weights) # (B, M, K)
        q_output = torch.sum(weights, dim=-1) # (B, M)

        if not batched:
            q_output = q_output.squeeze(0)  # Shape becomes (M)

        return q_output
    
    

class CylinderCollection(GIBCollection):
    
    def __init__(self, kernel_reach:float, num_gibs, **kwargs):
        """
        A collection of Cylinder GIBs.

        Parameters
        ----------
        `radius` - torch.Tensor:
            tensor of shape (num_gibs, 1) containing the radius of the cylinder's base for each GIB;

        `intensity` - torch.Tensor:
            tensor of shape (num_gibs, 1) representing scalar intensities for each cylinder GIB;
        """

        super().__init__(kernel_reach, num_gibs=num_gibs, angles=kwargs.get('angles', None), intensity=kwargs.get('intensity', 1))
        
        if kwargs.get('radius') is None:
            raise KeyError("Provide a radius for the cylinder in the kernel.")

        self.radius = kwargs['radius']
        
    def mandatory_parameters():
        return ['radius']
    
    def gib_parameters():
        return CylinderCollection.mandatory_parameters() + ['intensity']

    def gib_random_config(num_gibs, kernel_reach):
        rand_config = GIBCollection.gib_random_config(num_gibs, kernel_reach)

        geneo_params = {
            'radius' : torch.rand((num_gibs, 1)) * kernel_reach + 0.01, # float \in [0.1, 1]
        }   
        rand_config[GIB_PARAMS].update(geneo_params)

        return rand_config
    
    
    def gaussian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function of the Cylinder GIB for the input tensor.

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
        # x_norm = torch.linalg.norm(x, dim=-1) # shape (..., G, K)
        return self.intensity * torch.exp((torch.linalg.norm(x, dim=-1)**2) * (-1 / (2*(self.radius + self.epsilon)**2))) # Kx1
    
    
    def _compute_gib_weights(self, s_centered: torch.Tensor) -> torch.Tensor:
        """
        Computes the weights of the Cylinder GIBs for the input tensor.

        Parameters
        ----------
        `s_centered` - torch.Tensor:
            Tensor of shape (..., G, K, 3) representing the centered support points for each GIB.

        Returns
        -------
        `weights` - torch.Tensor:
            Tensor of shape (..., G, K) representing the weights of the Cylinder GIBs for the input tensor.
        """
        return self.gaussian(s_centered[..., :2])
    

    def forward(self, points: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor) -> torch.Tensor:
        """
        Generalized version that computes the output of a Cylinder GIB Collection on the query points
        given the support points.
        
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
            Tensor of shape ([B], M, G), representing the output of the Cylinder GIB Collection on the query points.
            where G is the number of GIBs in the collection.
        """
        ##### prep for GIB computation #####
        s_centered, valid_mask, batched = GIBCollection._prep_support_vectors(points, q_points, support_idxs)
        s_centered = s_centered.unsqueeze(2).expand(-1, -1, self.num_gibs, -1, -1)
        montecarlo_points = torch.rand((int(10_000), 3), device=s_centered.device) * 2 * self.kernel_reach - self.kernel_reach # \in [-kernel_reach, kernel_reach]
        montecarlo_points = montecarlo_points[torch.norm(montecarlo_points, dim=-1) <= self.kernel_reach]
        q_output = self._prepped_forward(s_centered, valid_mask, batched, montecarlo_points)
        
        return q_output


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    sys.path.insert(1, '../..')
    sys.path.insert(2, '../../..')
    sys.path.insert(3, '../../../..')
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

    # cylinder = Cylinder(kernel_reach=1, radius=0.2)

    # cy_weights = cylinder.forward(points, q_points, neighbors_idxs)
    # print(cy_weights.shape)
    # print(cy_weights)

    # plot q_points + kernel
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # q_points = q_points.cpu().numpy()
    # q_points = q_points[0]
    # cy_weights = cy_weights.cpu().numpy()
    # cy_weights = cy_weights[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=cy_weights, cmap='magma')

    # plt.show()  
    
    
    
    ###############################################################
    # Test Cylinder Collection
    
    num_gibs = 1
    
    cy_rand_config = CylinderCollection.gib_random_config(num_gibs, kernel_reach=1)
    cylinders = CylinderCollection(kernel_reach=1, num_gibs=num_gibs, **cy_rand_config[GIB_PARAMS])  
    
    
    cy_weights = cylinders.forward(points, q_points, neighbors_idxs) # shape (B, Q, G)
    print(cy_weights.shape)
    
    print(cylinders.radius)
    
    # plot q_points + kernel
    q_points = q_points.cpu().numpy()
    q_points = q_points[0]
    cy_weights = cy_weights.cpu().numpy()
    cy_weights = cy_weights[0] # shape (M, G)
    cy_weights = cy_weights[:, 0] # select the first cylinder
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=cy_weights, cmap='magma')
    plt.show()




