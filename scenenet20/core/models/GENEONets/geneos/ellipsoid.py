
import torch
import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
sys.path.insert(3, '../../../..')
from core.models.GENEONets.geneos.GIB_Stub import GIB_Stub, GIB_PARAMS, NON_TRAINABLE, KERNEL_REACH

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

        exponent = -0.5 * torch.sum(x * torch.matmul(x, precision_matrix), dim=1) # K
        gauss_dist = torch.exp(exponent)

        return self.intensity * gauss_dist
    

    def compute_integral(self) -> torch.Tensor:
        mc_weights = self.gaussian(self.montecarlo_points)
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
            center = points[q]
            support_points = points[supports_idxs[i]]
            s_centered = support_points - center
            # s_centered = s_centered.to(self.device)
            s_centered = self.rotate(s_centered)
            weights = self.gaussian(s_centered)
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

    ellip = Ellipsoid(0.3,
                      angles = torch.tensor([0.0, 0.0, 0.0]),
                      radii  = torch.tensor([0.01, 0.01, 0.1]))

    ellip_weights = ellip.forward(points, query_idxs, neighbors_idxs)
    print(ellip_weights.shape)
    print(ellip_weights)

    # plot q_points + kernel
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c='b')
    ax.scatter(q_points[:, 0], q_points[:, 1], q_points[:, 2], c=ellip_weights.cpu().detach().numpy(), cmap='magma')

    plt.show()    

