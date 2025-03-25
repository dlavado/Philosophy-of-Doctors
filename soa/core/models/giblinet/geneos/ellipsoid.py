
import torch
import torch.nn.functional as F
import sys
sys.path.append('../../../..')
from core.models.giblinet.geneos.GIB_Stub import GIB_Stub, GIBCollection, GIB_PARAMS, NON_TRAINABLE, gaussian_2d, _prep_support_vectors, _retrieve_support_points

class Ellipsoid(GIB_Stub):

    def __init__(self, kernel_reach, **kwargs):
        """
        GIB that encodes an ellipsoid.

        Required
        --------

        radii - torch.tensor \in ]0, kernel_reach] with shape (3,):
            ellipsoid's radii in the x, y, and z directions;

        """
        super().__init__(kernel_reach, angles=kwargs.get('angles', None))

        self.radii = kwargs.get('radii', None)
        if self.radii is None:
            raise KeyError("Provide radii for the ellipsoid.")        

        # self.radii = GIB_Stub._to_parameter(self.radii).to(self.device)
        self.intensity = kwargs.get('intensity', 1)

  
    def mandatory_parameters():
        return ['radii']
    
    def gib_parameters():
        return Ellipsoid.mandatory_parameters() + ['intensity']

    def gib_random_config(kernel_reach):
        rand_config = GIB_Stub.gib_random_config(kernel_reach)

        gib_params = {
            'radii' : torch.randn(3),
            'intensity' : 1.0,
        }
        rand_config[GIB_PARAMS].update(gib_params)

        return rand_config


    def gaussian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function of the Ellipsoid GIB for the input tensor.

        Parameters
        ----------
        `x` - torch.Tensor:
            Tensor of shape (K, 3) representing the input tensor.

        Returns
        -------
        `gaussian` - torch.Tensor:
            Tensor of shape (K,) representing the gaussian function of the input tensor.
        """
        cov_matrix = torch.diag(torch.relu(self.radii)) # 3x3, relu to avoid negative radii and ensure positive semi-definite matrix
        precision_matrix = torch.inverse(cov_matrix) # 3x3

        exponent = -0.5 * torch.sum(x * torch.matmul(x, precision_matrix), dim=-1) # K
        gauss_dist = torch.exp(exponent)

        return self.intensity * gauss_dist
    

    def compute_integral(self) -> torch.Tensor:
        mc_weights = self.gaussian(self.montecarlo_points)
        # self._plot_integral(mc_weights)
        return torch.sum(mc_weights)


    # def forward(self, points: torch.Tensor, q_points: torch.Tensor, supports_idxs: torch.Tensor) -> torch.Tensor:
    #     q_output = torch.zeros(len(q_points), dtype=points.dtype, device=points.device)
    #     for i, center in enumerate(q_points):
    #         # center = points[q]
    #         support_points = points[supports_idxs[i]]
    #         s_centered = support_points - center
    #         # s_centered = s_centered.to(self.device)
    #         s_centered = self.rotate(s_centered)
    #         weights = self.gaussian(s_centered)
    #         weights = self.sum_zero(weights)
    #         q_output[i] = torch.sum(weights)

    #     return q_output

    def forward(self, points: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor) -> torch.Tensor:
        """
        Generalized version that computes the output of the Ellipoid GIB on the query points
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
            Tensor of shape ([B], M), representing the output of the Ellipsoid GIB on the query points.
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
        support_points = _retrieve_support_points(points, support_idxs)
        valid_mask = (support_idxs != -1) # Mask out invalid indices with -1; shape (B, M, K)

        # Center support points: (B, M, K, 3) - (B, M, 1, 3)
        s_centered = support_points - q_points.unsqueeze(2) # (B, M, K, 3)
        s_centered = self.rotate(s_centered)

        # Compute GIB weights; (B, M, K, 3) -> (B, M, K)
        weights = self.gaussian(s_centered)
        weights = weights * valid_mask.float()

        weights = self.sum_zero(weights) # (B, M, K)
        q_output = torch.sum(weights, dim=-1) # (B, M)

        if not batched:
            q_output = q_output.squeeze(0)

        return q_output
    
    
    
class EllipsoidCollection(GIBCollection):
    
    
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
        `L` - torch.Tensor:
            Lower triangular matrix for the Cholesky Decomposition of the precision matrix.
            This guarantees that the precision matrix is positive definite at running time, avoids numerical issues and grants the ellipsoid more degrees of freedom.
            For a more intuitive use of the ellipsoid, provide the `radii` instead (the eigenvalues of the precision matrix).
        """
        
        super().__init__(kernel_reach, num_gibs=num_gibs, angles=kwargs.get('angles', None), intensity=kwargs.get('intensity', 1))
        
        if 'radii' not in kwargs and 'L' not in kwargs:
            raise KeyError("Provide radii or precision matrix for the ellipsoid.")
        
        self.L = kwargs.get('L', None)

        if self.L is None: # if L matrix is not provided, compute it from the radii
            self.L = torch.diag_embed(1 / kwargs.get('radii', None)**2) # (G, 3, 3)
        
        self.epsilon = self.epsilon * torch.eye(3, device=self.L.device).cuda()        
        
        
    def mandatory_parameters():
        return ['L']
    
    def gib_parameters():
        return EllipsoidCollection.mandatory_parameters() + ['intensity']

    def gib_random_config(num_gibs, kernel_reach):
        rand_config = GIBCollection.gib_random_config(num_gibs, kernel_reach)

        gib_params = {
            # Lower Trig. Matrix for the Cholesky Decomposition of the Precision Matrix
            'L' : torch.tril(torch.randn(num_gibs, 3, 3)), # (G, 3, 3)
        }
        
        rand_config[GIB_PARAMS].update(gib_params)

        return rand_config
    
    def gaussian(self, x:torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function of the Ellipsoid GIB for the input tensor.

        Parameters
        ----------
        `x` - torch.Tensor:
            Tensor of shape (..., G, K, 3) representing the input tensor, where G is the number of GIBs.

        Returns
        -------
        `gaussian` - torch.Tensor:
            Tensor of shape (..., G, K) representing the gaussian function of the input tensor.
        """
    
        """ Covariance matrix is diagonal, so we can simplify the computation:
        \Sigma = 
        [ r1^2  r1r2  r1r3 
          r2r1  r2^2  r2r3
          r3r1  r3r2  r3^2  
        ]
        """
        # this way, we guarantee that the precision matrix is positive definite
        # precision_matrix =  self.L @ self.L.transpose(-1, -2) + self.epsilon# (..., G, 3, 3)
        # gauss_dist = torch.exp(-0.5 * torch.einsum('...gki,gij,...gkj->...gk', x, precision_matrix, x)) # (..., G, K)
        # gauss_dist = torch.sum((x @ (self.L @ self.L.transpose(-1, -2) + self.epsilon)) * x, dim=-1)  # (..., G, K)
        # gauss_dist = torch.exp(-0.5 * gauss_dist)

        # return self.intensity * gauss_dist
        return self.intensity*EllipsoidCollection.gaussian_3d(x, self.L, self.epsilon)
    
    
    @torch.jit.script
    def gaussian_3d(x:torch.Tensor, L:torch.Tensor, eps:torch.Tensor) -> torch.Tensor:
        """
        Computes a three-dimensional gaussian function.

        Parameters
        ----------
        `x` - torch.Tensor:
            Tensor of shape (..., G, K, 3) representing the input tensor. 
            Where G is the number of GIBs, and K is the number of neighbors and their dimensions.
            
        `L` - torch.Tensor:
            Tensor of shape (G, 3, 3) representing the lower triangular matrix for the 
            Cholesky Decomposition of the precision matrix.
            
        `eps` - torch.Tensor:
            Tensor of shape (3, 3) representing the epsilon matrix.
            This matrix is added to the precision matrix to ensure numerical stability and positive definiteness.

        Returns
        -------
        `gaussian` - torch.Tensor:
            Tensor of shape (..., G, K) representing the gaussian function of the input tensor.
        """
        return torch.exp(-0.5 * torch.sum((x @ (L @ L.transpose(-1, -2) + eps)) * x, dim=-1))
        
    
    
    def _compute_gib_weights(self, s_centered: torch.Tensor) -> torch.Tensor:
        """
        Computes the gaussian function of the Ellipsoid GIB for the input tensor.

        Parameters
        ----------
        `s_centered` - torch.Tensor:
            Tensor of shape (..., G, K, 3) representing the centered support points for each query point.

        Returns
        -------
        `weights` - torch.Tensor:
            Tensor of shape (..., G, K) representing the gaussian function of the input tensor.
        """
        return self.gaussian(s_centered)
    
    
    def forward(self, points: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor) -> torch.Tensor:
        """
        Generalized version that computes the output of the Ellipoid GIB on the query points
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
            Tensor of shape ([B], M, G), representing the output of the Ellipsoid GIB on the query points.
        """
        
        # print("Ellipsoid Forward")
        
        ##### prep for GIB computation #####
        s_centered, valid_mask, batched = _prep_support_vectors(points, q_points, support_idxs)
        s_centered = s_centered.unsqueeze(2).expand(-1, -1, self.num_gibs, -1, -1)
        montecarlo_points = torch.rand((int(10_000), 3), device=s_centered.device) * 2 * self.kernel_reach - self.kernel_reach # \in [-kernel_reach, kernel_reach]
        montecarlo_points = montecarlo_points[torch.norm(montecarlo_points, dim=-1) <= self.kernel_reach]
        q_output = self._prepped_forward(s_centered, valid_mask, montecarlo_points)
        return q_output

    
    
        

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

    # ellip = Ellipsoid(0.2,
    #                   angles = torch.tensor([0.0, 0.0, 0.0], device=points.device),
    #                   radii  = torch.tensor([0.01, 0.5, 0.01], device=points.device),)

    # ellip_weights = ellip.forward(points, q_points, neighbors_idxs)
    # print(ellip_weights.shape)

    # plot q_points + kernel
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # q_points = q_points[0]
    # q_points = q_points.cpu().numpy()
    # ellip_weights = ellip_weights.cpu().detach().numpy()
    # ellip_weights = ellip_weights[0]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=ellip_weights, cmap='magma')

    # plt.show()    
    
    
    
    ################# EllipsoidCollection #################
    import time
    
    num_gibs = 8
    
    radii = torch.tensor([0.05, 0.05, 0.5], device=points.device)
    radii = radii.repeat(num_gibs, 1)
    print(f"{radii.shape=}")
    
    # ellip_init = EllipsoidCollection.gib_random_config(num_gibs, 0.2)
    # ellip_coll = EllipsoidCollection(kernel_reach=0.2, num_gibs=num_gibs, **ellip_init[GIB_PARAMS])
    
    ellip_coll = EllipsoidCollection(kernel_reach=0.2, num_gibs=num_gibs, radii=radii)
    
    start = time.time()
    ellip_weights = ellip_coll.forward(points, q_points, neighbors_idxs) # (B, M, G)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"Time taken for EllipsoidCollection: {time.time() - start}")
    
    print(ellip_weights.shape)
    print(ellip_coll.radii.shape)
    
    q_points = q_points.cpu().numpy()
    q_points = q_points[0]
    ellip_weights = ellip_weights.cpu().detach().numpy()
    ellip_weights = ellip_weights[0]
    ellip_weights = ellip_weights[:, 0] # take the first gib
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=ellip_weights, cmap='magma')
    plt.show()
 
