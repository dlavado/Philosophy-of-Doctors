# %%

import torch


import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
from core.models.geneos.GENEO_kernel_torch import GENEO_kernel

class Ellipsoid(GENEO_kernel):

    def __init__(self, name, kernel_size, plot=False, **kwargs):
        """
        GENEO kernel that encodes an ellipsoid.\n

        Required
        --------

        radii - torch.tensor \in ]0, kernel_size[1]] with shape (3,):
            ellipsoid's radii;

        Optional
        --------

        scaler - float:
            scalar to multiply the ellipsoid;

        Returns
        -------
            3D torch tensor with the ellipsoid kernel
        """

        super().__init__(name, kernel_size)  


        self.radii = kwargs.get('radii', None)
        if self.radii is None:
            raise KeyError("Provide radii for the ellipsoid.")
       
    
        self.radii = self.radii.to(self.device)
        self.scaler = kwargs.get('scaler', torch.tensor(1.0)).to(self.device)

  
    def mandatory_parameters():
        return ['radii']
    
    def geneo_parameters():
        return Ellipsoid.mandatory_parameters() + ['scaler']

    def geneo_random_config(name="ellip", kernel_size=None):
        rand_config = GENEO_kernel.geneo_random_config(name, kernel_size)

        geneo_params = {
            'radii' : torch.tensor([torch.randint(0.1, rand_config['kernel_size'][0], (1,))[0],  #int \in [0.1, kernel_size[1]] ,
                                    torch.randint(0.1, rand_config['kernel_size'][1], (1,))[0],  #int \in [0.1, kernel_size[1]] ,
                                    torch.randint(0.1, rand_config['kernel_size'][2], (1,))[0]   #int \in [0.1, kernel_size[1]] ,
                                    ], dtype=torch.float),
            'scaler': torch.randint(1, 10, (1,))[0]/10, #float \in [0, 1]
        }

        rand_config['geneo_params'] = geneo_params

        rand_config['non_trainable'] = []

        return rand_config


    def gaussian(self, x:torch.Tensor, epsilon=1e-8) -> torch.Tensor:
        shape = torch.tensor(self.kernel_size, dtype=torch.float, device=self.device, requires_grad=True)
        center = (shape - 1) / 2

        x_c = x - center # Nx3

        cov_matrix = torch.diag(torch.relu(self.radii))  # 3x3, relu to avoid negative radii and ensure positive semi-definite matrix
        precision_matrix = torch.inverse(cov_matrix) # 3x3

        exponent = -0.5 * torch.sum(x_c * torch.matmul(x_c, precision_matrix), dim=1) # Nx1
        normalization = 1 # torch.sqrt((2 * torch.pi)**3 * torch.det(cov_matrix))

        gauss_dist = torch.exp(exponent)
        return self.scaler*gauss_dist[:, None]

    def sum_zero(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor - torch.sum(tensor) / torch.prod(torch.tensor(self.kernel_size)) 

    
    def compute_kernel(self):

        idxs = torch.stack(
                    torch.meshgrid(
                                torch.arange(self.kernel_size[0], dtype=torch.float),
                                torch.arange(self.kernel_size[1], dtype=torch.float),
                                torch.arange(self.kernel_size[2], dtype=torch.float)
                            )
                ).T.reshape(-1, 3)
        idxs = idxs.to(self.device).requires_grad_(True)

        kernel = self.gaussian(idxs)
        kernel = self.sum_zero(kernel)
        kernel = kernel.view(self.kernel_size)
        print(kernel.shape)

        return kernel



if __name__ == "__main__":
    
    ellip = Ellipsoid('ellip', (9, 9, 9), radii=torch.tensor([3.0, 1.0, 1.0]), scaler=torch.tensor(1.0))

    ellip.plot_kernel(ellip.compute_kernel())


    input("Press Enter to continue...")


    # %%
    import torch
    class EllipsoidalDistribution:
        def __init__(self, mean, covariance_matrix):
            self.mean = mean
            self.covariance_matrix = covariance_matrix
            self.precision_matrix = torch.inverse(covariance_matrix)

        def pdf(self, x):
            x_centered = x - self.mean
            exponent = -0.5 * torch.sum(x_centered * torch.matmul(x_centered, self.precision_matrix), dim=1)
            normalization = torch.sqrt((2 * torch.pi)**3 * torch.det(self.covariance_matrix))
            pdf_values = torch.exp(exponent) / normalization
            return pdf_values
        
    # %%
        
    mean = torch.tensor([0.0, 0.0, 0.0])
    cov_matrix = torch.tensor([[5.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 2.0]])

    ellipsoidal_dist = EllipsoidalDistribution(mean, cov_matrix)

    input_points = torch.tensor([[0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0]])
    pdf_values = ellipsoidal_dist.pdf(input_points)
    print(pdf_values)



    # %%

    mean = torch.tensor([0.0, 0.0, 0.0])
    cov_matrix = torch.tensor([[5.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 2.0]])

    num_samples = 100000
    samples = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov_matrix).sample((num_samples,))


    # %%
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    samples_np = samples.numpy()
    ax.scatter(samples_np[:, 0], samples_np[:, 1], samples_np[:, 2], c='b', marker='o')

    eigenvalues, eigenvectors = torch.eig(cov_matrix, eigenvectors=True)
    principal_axes = eigenvectors * torch.sqrt(eigenvalues.unsqueeze(1))

    # Plot principal axes if desired
    ax.quiver(mean[0], mean[1], mean[2], principal_axes[0, :], principal_axes[1, :], principal_axes[2, :], color='r')


    # Set the same range for all axes to make it appear spherical
    range_value = 3 * torch.max(cov_matrix)
    ax.set_box_aspect([range_value, range_value, range_value])
    # ax.set_xlim(mean[0] - range_value, mean[0] + range_value)
    # ax.set_ylim(mean[1] - range_value, mean[1] + range_value)
    # ax.set_zlim(mean[2] - range_value, mean[2] + range_value)

    plt.show()





# %%
