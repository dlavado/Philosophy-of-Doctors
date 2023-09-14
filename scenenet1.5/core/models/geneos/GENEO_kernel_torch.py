


from abc import abstractmethod
import time
import torch
import torch.nn.functional as F
import json


import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from utils import voxelization as Vox


class GENEO_kernel:
    """
    Initialization class for GENEO kernels.

    Kernels are built on top of the convolution operation as 3D arrays.

    * kernel shape in (z, x, y)
    """

    def __init__(self, name, kernel_size, plot=False):
        self.name = name
        self.kernel_size = kernel_size
        self.plot = plot
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.sign = 1 if torch.any(torch.rand(1) > 0.9) else -1 # random sign for the kernel
        self.sign = 1

        self.volume = torch.prod(torch.tensor(self.kernel_size, device=self.device))


    @abstractmethod
    def compute_kernel(self) -> torch.Tensor:
        """
        Returns a 3D GENEO kernel in torch format
        """
        return

    def convolution(self, tensor:torch.Tensor, plot=True) -> torch.Tensor:
        """
        Convolves the kernel with the tensor data passed as argument.

        Attention: tensor should be in format [B, 1, data], where B is the batch number.

        plot ? Visualize convolution output
        """
        start = time.time()
        conv = F.conv3d(tensor, self.kernel.view(1, 1, *self.kernel.shape), padding='same')
        end = time.time() - start

        if plot:
            print(f"Elapsed time for convolution: {end}")
            for i in range(tensor.shape[0]): # for each sample in the batch
                Vox.plot_voxelgrid(conv[i][0].detach().cpu().numpy())

        return conv


    def plot_kernel(self, kernel):
        print(f"\n{'*'*50}")
        print(f"kernel shape = {kernel.shape}")
        print(f"kernel sum = {torch.sum(kernel)}")
        Vox.plot_voxelgrid(kernel.cpu().detach().numpy())

    @staticmethod
    def mandatory_parameters():
        return []

    @staticmethod
    def geneo_parameters():
        return []

    @staticmethod
    def geneo_config_from_json(filename):
        """
        Returns a GENEO configuration based on a json file
        """
        with open(filename) as json_file:
            return json.load(json_file)

    @staticmethod
    def geneo_smart_config():
        """
        Returns a tailored GENEO configuration
        """
        return

    @staticmethod
    def geneo_random_config(name='GENEO_rand', kernel_size=(9, 9, 9)):
        """
        Returns a random GENEO configuration
        """
        config = {
            'name' : name,
            'kernel_size': kernel_size,
            'plot': False,
        }

        geneo_params = {}

        for param in GENEO_kernel.geneo_parameters():
            geneo_params[param] = torch.randint(0, 10, (1,))[0]/5 # float \in [0, 2]

        config['geneo_params'] = geneo_params

        config['non_trainable'] = []

        return config



