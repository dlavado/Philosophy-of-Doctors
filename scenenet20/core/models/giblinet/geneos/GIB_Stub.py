


from abc import abstractmethod
import torch


GIB_PARAMS = "gib_params"
NON_TRAINABLE = "non_trainable"
KERNEL_REACH = "kernel_reach"
NUM_GIBS = "num_gibs"


###############################################################
#                          GIB Utils                          #
###############################################################

def to_parameter(value, requires_grad=True):
    """
    Converts the input value to a torch.nn.Parameter.
    """
    t = to_tensor(value)
    return torch.nn.Parameter(t, requires_grad=requires_grad)    
    
def to_tensor(value):
    """
    Converts the input value to a torch.Tensor.
    """
    import numpy as np
    if isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, int) or isinstance(value, float):
        return torch.tensor(value, dtype=torch.float)
    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    elif isinstance(value, list) or isinstance(value, tuple):
        return torch.tensor(value, dtype=torch.float)
 
    raise ValueError("Input value must be a torch.Tensor")


@torch.jit.script
def gaussian_2d(x:torch.Tensor, sigma:torch.Tensor) -> torch.Tensor:
    """
    Computes a two-dimensional gaussian function.

    Parameters
    ----------
    `x` - torch.Tensor:
        Tensor of shape (..., G, K, 2) representing the input tensor. 
        Where G is the number of GIBs, and K is the number of neighbors and their dimensions.
        
    `sigma` - torch.Tensor:
        Tensor of shape (..., G, 1) representing the standard deviation of the gaussian function for each GIB.

    Returns
    -------
    `gaussian` - torch.Tensor:
        Tensor of shape (..., G, K) representing the gaussian function of the input tensor.
    """
    return torch.exp(torch.linalg.norm(x, dim=-1) * (-1 / (2*sigma**2)))


@torch.jit.script
def _retrieve_support_points(points: torch.Tensor, support_idxs: torch.Tensor) -> torch.Tensor:
    """
    Retrieves the support points from the input tensor given the support indices.

    Parameters
    ----------

    `points` - torch.Tensor:
        Tensor of shape (B, N, 3) representing the point cloud.

    `supports_idxs` - torch.Tensor:
        Tensor of shape (B, M, K) representing the indices of the support points for each query point.

    Returns
    -------
    `support_points` - torch.Tensor:
        Tensor of shape (B, M, K, 3) representing the support points.
    """
    B, M, K = support_idxs.shape
    F = points.shape[-1]

    support_idxs = support_idxs.reshape(B, -1) # (B, M*K)
    # illegal indices are -1, we need to mask them out
    support_idxs = torch.where(support_idxs == -1, torch.zeros_like(support_idxs), support_idxs)
    # Gather the points (B, M*K, 3); Reshape back to (B, M, K, 3)
    return torch.gather(points, 1, support_idxs.unsqueeze(-1).expand(-1, -1, F)).reshape(B, M, K, F).to(torch.float16)

@torch.jit.script
def _prep_support_vectors(points: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor):
        
    # if points.ndimension() == 2:
    #     # If unbatched, add a batch dimension
    #     points = points.unsqueeze(0)
    #     q_points = q_points.unsqueeze(0)
    #     support_idxs = support_idxs.unsqueeze(0)
    #     batched = False
    # else:
    #     batched = True

    # Gather support points: (B, M, K) -> (B, M, K, 3)
    s_centered = _retrieve_support_points(points, support_idxs)
    # Center the support points around the query points
    s_centered = s_centered - q_points.unsqueeze(2) # (B, M, K, 3)
    
    valid_mask = (support_idxs != -1) # Mask out invalid indices with -1; shape (B, M, K)

    return s_centered.contiguous(), valid_mask, True

###############################################################
#                           GIB Stub                          #
###############################################################

class GIB_Stub(torch.nn.Module):
    """
    Abstract class for Geometric Inductive Bias operators.
    """

    def __init__(self, kernel_reach:float, angles=None, **kwargs):
        """
        Initializes the GIB kernel.

        Parameters
        ----------

        `kernel_reach` - int:
            The kernel's neighborhood reach in Geometric space.
        """
        super(GIB_Stub, self).__init__()

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sign = 1 if torch.any(torch.rand(1) > 0.5) else -1 # random sign for the kernel

        self.angles:torch.Tensor = angles if angles is not None else None
        self.kernel_reach = kernel_reach
        
        # variables to compute the integral of the GIB function within the kernel reach
        self.n_samples = 1e4
        self.ndims = 3
        self.montecarlo_points = torch.rand((int(self.n_samples), self.ndims)) * 2 * self.kernel_reach - self.kernel_reach
        # self.montecarlo_points = self.montecarlo_points.cuda()
        mask_inside = torch.linalg.norm(self.montecarlo_points, dim=1) <= self.kernel_reach
        self.montecarlo_points = self.montecarlo_points[mask_inside]

        self.epsilon = 1e-8 # small value to avoid division by zero
        self.intensity = 1  # intensity of the gaussian function
        

    @abstractmethod
    def compute_integral(self) -> torch.Tensor:
        """
        Computes an integral approximation of the gaussian function within the kernel_reach.
       
        Returns
        -------
        `integral` - torch.Tensor:
            Tensor of shape (1,) representing the integral of the gaussian function within the kernel reach.
        """

    @abstractmethod
    def gaussian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function for the given input tensor.
        """
    

    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor by subtracting the integral of the resulting gaussian function within the kernel reach.

        Parameters
        ----------
        `tensor` - torch.Tensor:
            Tensor of shape (N,) representing the values of the kernel; this tensor directly results from the gaussian function.
            
        Returns
        -------
        `tensor` - torch.Tensor:
            Tensor of shape (N,) representing the normalized values of the kernel.
        """
        self.montecarlo_points = self.montecarlo_points.to(tensor.device)
        integral = self.compute_integral().to(tensor.device)
        return tensor - integral / self.n_samples
    

    @abstractmethod
    def forward(self, points:torch.Tensor, q_points:torch.Tensor, supports_idxs:torch.Tensor) -> torch.Tensor:
        """
        Computes a GIB on the query points given the support points.

        Parameters
        ----------
        `points` - torch.Tensor:
            Tensor of shape (N, 3) representing the point cloud.

        `q_points` - torch.Tensor:
            Tensor of shape (M, 3) representing the query points.

        `supports_idxs` - torch.Tensor[int]:
            Tensor of shape (M, K) representing the indices of the support points for each query point. With K <= N.

        Returns
        -------
        `q_output` - torch.Tensor:
            Tensor of shape (M,) representing the output of the GIB on the query points.    
        """
        

    @staticmethod
    def mandatory_parameters():
        return []

    @staticmethod
    def gib_parameters():
        return []

    @staticmethod
    def gib_random_config(kernel_reach:int):
        """
        Returns a random GENEO configuration
        """
        config = {
            KERNEL_REACH: kernel_reach   
        }
        gib_params = {
            'intensity' : torch.randint(5, 10, (1,))[0]/5, # float \in [0, 1]
            'angles'    : torch.zeros(3)
        }

        for param in GIB_Stub.gib_parameters():
            gib_params[param] = torch.randint(0, 10, (1,))[0]/5 # float \in [0, 2]

        config[GIB_PARAMS] = gib_params
        config[NON_TRAINABLE] = []

        return config
    

    def _retrieve_support_points(self, points: torch.Tensor, supports_idxs: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the support points from the input tensor given the support indices.

        Parameters
        ----------

        `points` - torch.Tensor:
            Tensor of shape (B, N, 3) representing the point cloud.

        `supports_idxs` - torch.Tensor:
            Tensor of shape (B, M, K) representing the indices of the support points for each query point.

        Returns
        -------
        `support_points` - torch.Tensor:
            Tensor of shape (B, M, K, 3) representing the support points.
        """
        B, M, K = supports_idxs.shape
        F = points.shape[-1]

        flat_supports_idxs = supports_idxs.reshape(B, -1) # (B, M*K)

        # Gather the points (B, M*K, 3)
        # the expanded tensor lets us gather the points at the indices in flat_supports_idxs
        gathered_support_points = torch.gather(points, 1, flat_supports_idxs.unsqueeze(-1).expand(-1, -1, F))

        # Reshape back to (B, M, K, 3)
        support_points = gathered_support_points.reshape(B, M, K, F)
        return support_points#.contiguous()
    

    def _plot_integral(self, gaussian_x:torch.Tensor):
        print(f"{gaussian_x.shape=}")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mc = self.montecarlo_points.detach().cpu().numpy()
        ax.scatter(mc[:, 0], mc[:, 1], mc[:, 2], c=gaussian_x.detach().cpu().numpy(), cmap='magma')
        plt.show()  
    
   
    def rotate(self, points:torch.Tensor) -> torch.Tensor:
        """
        Rotate a tensor along the x, y, and z axes by the given angles.
        
        Parameters
        ----------
        `angles` - torch.Tensor:
            Tensor of shape (3,) containing rotation angles for the x, y, and z axes.
            These are nromalized in the range [-1, 1] and represent angles_normalized = angles / pi.

        `points` - torch.Tensor:
            Tensor of shape (N, 3) representing the 3D points to rotate.
            
        Returns
        -------
        `points` - torch.Tensor:
            Tensor of shape (N, 3) containing the rotated
        """
        if self.angles is None:
            return points
        
        from core.models.giblinet.geneos.diff_rotation_transform import rotate_points
        angles = self.angles % 2 # rotations higher than 2\pi are equivalent to rotations within 2\pi
        angles = 2 - torch.relu(-angles) # convert negative angles to positive
        return rotate_points(angles, points)
    
    
    
class GIBCollection(torch.nn.Module):
    
    def __init__(self, kernel_reach:float, num_gibs, intensity=1, **kwargs):
        """
        Initializes the GIB kernel.

        Parameters
        ----------

        `kernel_reach` - int:
            The kernel's neighborhood reach in Geometric space.
            
        `num_gibs` - int:
            The number of GIBs in the collection.
            
        `intensity` - torch.Tensor:
            tensor of shape (num_gibs,) representing scalar intensities for each GIB;
            
        `angles` - torch.Tensor:
            tensor of shape (num_gibs, 3) containing rotation angles for the x, y, and z axes for each GIB;
        """
        super(GIBCollection, self).__init__()    

        self.kernel_reach = kernel_reach
        self.num_gibs = num_gibs
        
        # variables to compute the integral of the GIB function within the kernel reach
        self.epsilon = 1e-8 # small value to avoid division by zero        
        self.intensity = intensity # intensity of the gaussian function   
        

    @abstractmethod
    def gaussian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function for the given input tensor.
        """
        
    @staticmethod
    def mandatory_parameters():
        return []

    @staticmethod
    def gib_parameters():
        return []
    
    @staticmethod
    def gib_random_config(num_gibs:int, kernel_reach:int):
        """
        Returns a random GENEO configuration
        """
        config = {
            KERNEL_REACH: kernel_reach,
            NUM_GIBS : num_gibs
        }
        gib_params = {
            'intensity' : torch.randint(5, 10, (num_gibs, 1))/5, # float \in [0, 1]
            # 'angles'    : torch.randn((num_gibs, 3))
        }

        for param in GIBCollection.gib_parameters():
            gib_params[param] = torch.randint(0, 10, (num_gibs,1))/5 # float \in [0, 2]

        config[GIB_PARAMS] = gib_params
        config[NON_TRAINABLE] = []

        return config
    
    
    def _plot_integral(self, mc, mc_weights:torch.Tensor, plot_valid=False):
        print(f"{mc_weights.shape=}")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if plot_valid: # only plot points with positive weights
            valid_mask = mc_weights > self.epsilon
            mc = mc[valid_mask]
            mc_weights = mc_weights[valid_mask]
        ax.scatter(mc[:, 0], mc[:, 1], mc[:, 2], c=mc_weights.detach().cpu().numpy(), cmap='magma')
        plt.show()  
    
    
    def compute_integral(self, mc_points:torch.Tensor) -> torch.Tensor:
        """
        Computes an integral approximation of the gaussian function within the kernel_reach.
        
        Parameters
        ----------
        
        `mc_points` - torch.Tensor:
            Tensor of shape (G, N, 3) representing the montecarlo points for each GIB in the collection.

        Returns
        -------
        `integral` - torch.Tensor:
            Tensor of shape (G,) representing the integral of the gaussian function within the kernel reach for each gib in the collection;
        """
        # mc_points = mc_points[None].expand(self.num_gibs, -1, -1)
        mc_weights = self._compute_gib_weights(mc_points)
        # print(f"{mc_weights.shape=}")
        # for g in range(self.num_gibs):
        #     self._plot_integral(mc_points[g], mc_weights[g], plot_valid=False)
        return torch.sum(mc_weights, dim=-1)
    
    def sum_zero(self, tensor:torch.Tensor, mc_points:torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input tensor by subtracting the integral of the resulting gaussian functions within the kernel reach.

        Parameters
        ----------
        `tensor` - torch.Tensor:
            Tensor of shape (..., G, K) representing the values of the kernel;
            This tensor is the product of the gaussian function of the GIB Collection; 
            where G is the number of GIBs and K is the num of neighbors;
            
        Returns
        -------
        `tensor` - torch.Tensor:
            Tensor of shape (..., G, K) representing the normalized values of the kernel.
        """
        #mc_points = mc_points.unsqueeze(0).expand(self.num_gibs, -1, -1)
        integral = self.compute_integral(mc_points)
        return tensor - integral.unsqueeze(-1) / mc_points.shape[0]
    
    
    @abstractmethod
    def _compute_gib_weights(self, s_centered:torch.Tensor) -> torch.Tensor:
        """
        Computes the weights of the GIB kernel        
        """
        
        
    def _prepped_forward(self, s_centered:torch.Tensor, valid_mask:torch.Tensor, batched:bool, mc_points:torch.Tensor) -> torch.Tensor:
        
        # rotations moved to GIB Layer for faster computation
        # s_centered = self.apply_rotations(s_centered.contiguous()) # (B, M, G, K, 3)
        
        # print(f"{s_centered.shape=}")   
        # Compute GIB weights; (B, M, G, K, 3) -> (B, M, G, K)
        weights = self._compute_gib_weights(s_centered)
        # print(f"{weights.shape=}")
        
        ### Post Processing ###
        weights = self._validate_and_sum(weights, valid_mask, mc_points) # (B, M, G)

        if not batched:
            weights = weights.squeeze(0)

        return weights


    def _validate_and_sum(self, weights:torch.Tensor, valid_mask:torch.Tensor, mc_points:torch.Tensor) -> torch.Tensor:
        """
        Validates the weights and sums them up.
        """
        weights = weights * valid_mask.unsqueeze(2).expand_as(weights).to(torch.float16)
        weights = self.sum_zero(weights, mc_points) # (B, M, G, K)
        weights = torch.sum(weights, dim=-1) # (B, M, G)
        return weights
    
    
    
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    sys.path.insert(1, '../..')
    sys.path.insert(2, '../../..')
    sys.path.insert(3, '../../../..')
    from core.models.giblinet.neighboring.radius_ball import keops_radius_search
    from core.models.giblinet.neighboring.knn import torch_knn
    from core.models.giblinet.pooling.fps_pooling import fps_sampling
    from core.models.giblinet.geneos import cylinder, disk, cone, ellipsoid
    
    
    ### test rotate
    angles = torch.tensor([[0.5, 0.0, 0.0]]) # rotate around z-axis by \pi radians
    gib    = GIBCollection(kernel_reach=1.0, num_gibs=1, angles=angles)
    points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_points = gib.rotate(points)
    # print(rotated_points)
    # do an assert of the rotated points
    assert torch.allclose(rotated_points, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]), atol=1e-6)
    input("Press Enter to continue...")
    
    ##################################
    
    # generate some points, query points, and neighbors. For the neighbors, I want to test two scenarios: 
    # 1) where the neighbors are at radius distance from the query points
    # 2) where the neighbors are very distance fromt the query points, at least 2*radius distance
    points = torch.rand((3, 100_000, 3))
    q_points = fps_sampling(points, num_points=1_000)
    # print(f"{q_points.shape=}")
    num_neighbors = 16
    # neighbors_idxs = keops_radius_search(q_points, points, 0.2, num_neighbors, loop=True)
    _, neighbors_idxs = torch_knn(q_points, q_points, num_neighbors)

    # # print(points.device, q_points.device, neighbors_idxs.device)

    # gib_setup = {
    #     'cy': 2,
    #     'cone': 2,
    #     'disk': 2,
    #     'ellip': 2
    # }

    # gib_layer = GIB_Layer(gib_setup, kernel_reach=16, num_observers=2)

    # gib_weights = gib_layer(points, query_idxs, neighbors_idxs)

    # print(gib_layer.get_cvx_coefficients())
    # print(gib_weights.shape)


    # input("Press Enter to continue...")


    from core.datasets.TS40K import TS40K_FULL_Preprocessed
    from utils import constants as C
    import utils.pointcloud_processing as eda

    ts40k = TS40K_FULL_Preprocessed(
        C.TS40K_FULL_PREPROCESSED_PATH,
        split='fit',
        sample_types=['tower_radius'],
        transform=None,
        load_into_memory=False
    )

    pcd, y = ts40k[0]
    pcd = pcd.to('cuda')
    num_query_points = 10000
    num_neighbors = 10
    kernel_reach = 0.1

    # max dist between points:
    # c_dist = torch.cdist(pcd, pcd)
    # print(torch.max(c_dist), torch.min(c_dist), torch.mean(c_dist)) # tensor(1.5170, device='cuda:0') tensor(0., device='cuda:0') tensor(0.4848, device='cuda:0')
    query_idxs = fps_sampling(pcd, num_points=num_query_points)
    # support_idxs = torch_knn(pcd[query_idxs], pcd[query_idxs], num_neighbors)[1]
    support_idxs = keops_radius_search(pcd[query_idxs], pcd, kernel_reach, num_neighbors, loop=False)

    gib_setup = {
        'cy': 1,
        # 'cone': 1,
        # 'disk': 1,
        # 'ellip': 1
    }

    # gib_layer = GIB_Layer(gib_setup, kernel_reach=0.1, num_observers=1)

    # effectively retrieve the body of towers
    # gib_layer = cylinder.Cylinder(kernel_reach=kernel_reach, radius=0.02)

    # disk is a generalization of the cylinder with the width parameter;
    # small width -> can be used to detect ground / surface
    # large width -> can be used to detect towers
    # gib_layer = disk.Disk(kernel_reach=kernel_reach, radius=0.02, width=0.1)

    # cone can be used to detect arboreal structures such as medium vegetation with small apex
    # gib_layer = cone.Cone(kernel_reach=kernel_reach, radius=0.05, inc=torch.tensor([0.1], device='cuda'), apex=1, intensity=1.0)

    # Ideally and ellipsoid can be used to detect the ground and power lines by assuming different radii
    gib_layer = ellipsoid.Ellipsoid(kernel_reach=kernel_reach, radii=torch.tensor([0.1, 0.1, 0.001], device='cuda'))
    

    gib_weights = gib_layer(pcd, query_idxs, support_idxs)


    # eda.plot_pointcloud(
    #     pcd.cpu().detach().numpy(),
    #     y.cpu().detach().numpy(),
    #     use_preset_colors=True
    # )

    # print(gib_weights.shape)

    colors = eda.weights_to_colors(gib_weights.cpu().detach().numpy(), cmap='seismic')
    eda.plot_pointcloud(
        pcd[query_idxs].cpu().detach().numpy(), 
        classes=None,
        rgb=colors
    )