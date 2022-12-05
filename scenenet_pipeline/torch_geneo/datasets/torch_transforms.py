



from typing import Tuple
import torch
import numpy as np
import torch.nn.functional as F

class ToTensor:

    def __call__(self, sample):
        sample = list(sample)
        return tuple([torch.from_numpy(s.astype(np.float)) for s in sample])
    


class ToFullDense:
    """
    Transforms a Regression Dataset into a Belief Dataset.

    Essentially, any voxel that has tower points is given a belief of 1,
    in order to maximze the towers' geometry.
    For the input, the density is notmalized to 1, so empty voxels have a value
    of 0 and 1 otherwise.

    It requires a discretization of raw LiDAR Point Clouds in Torch format.
    """

    def __call__(self, sample:torch.Tensor):
        xyz, dense, labels = sample

        return xyz, (dense > 0).to(dense), (labels > 0).to(labels) #full dense
    


class AddPad:

    def __init__(self, pad:Tuple[int]):
        """
        `pad` is a tuple of ints that contains the pad sizes for each dimension in each direction.\n
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4]) 
        """
        self.p3d = pad

    def __call__(self, sample):
        pts, labels = sample
        return F.pad(pts, self.p3d, 'constant', 0), F.pad(labels, self.p3d, 'constant', 0)
        
