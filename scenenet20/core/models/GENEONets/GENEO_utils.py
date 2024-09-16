from typing import List, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os


sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.models.GENEONets.geneos.GIB_Stub import GIB_Stub
from core.models.GENEONets.geneos import cylinder, neg_sphere, arrow, disk, cone, ellipsoid


###############################################################
#                         Args Utils                          #
###############################################################

import collections
from itertools import repeat

def _ntuple(n, name="parse"):
        def parse(x):
            if isinstance(x, collections.abc.Iterable):
                return tuple(x)
            return tuple(repeat(x, n))

        parse.__name__ = name
        return parse
_triple = _ntuple(3, "_triple")



###############################################################
#                         GENEO Layer                         #
###############################################################

class GENEO_Operator(nn.Module):

    def __init__(self, geneo_class:GIB_Stub, kernel_reach:tuple=None, **kwargs):
        super(GENEO_Operator, self).__init__()  

        self.geneo_class = geneo_class
        self.kernel_reach = kernel_reach
        
        if len(kwargs) > 0:
            self.init_from_kwargs(**kwargs)
        else:
            self.init_from_config()

    
    def init_from_config(self):

        config = self.geneo_class.geneo_random_config()

        self.geneo_params = {}
        for param in config['geneo_params']:
            if isinstance(config['geneo_params'][param], torch.Tensor):
                t_param = config['geneo_params'][param].to(dtype=torch.float)
            else:
                t_param = torch.tensor(config['geneo_params'][param], dtype=torch.float)
            t_param = nn.Parameter(t_param, requires_grad = not param in config['non_trainable'])
            self.geneo_params[param] = t_param

        self.geneo_params = nn.ParameterDict(self.geneo_params)


    def init_from_kwargs(self, **kwargs):
        self.geneo_params = {}
        for param in self.geneo_class.mandatory_parameters():
            self.geneo_params[param] = nn.Parameter(torch.tensor(kwargs[param], dtype=torch.float))

        self.geneo_params = nn.ParameterDict(self.geneo_params)

    def compute_kernel(self) -> torch.Tensor:
       """
       TODO
       """

    def forward(self, x:torch.Tensor) -> torch.Tensor:
      """
      TODO
      """



###############################################################
#                         SCENE-Nets                          #
###############################################################

class GENEO_Layer(nn.Module):

    def __init__(self, gib_dict:dict, kernel_reach:int, num_observers:int=1):
        """
        Instantiates a GENEO-Layer Module with specific GENEOs and their respective cvx coefficients.

        Parameters
        ----------
        `gib_dict` - dict:
            Mappings that contain the number of GENEOs of each kind (the key) to initialize

        `kernel_reach` - int:
            The kernel's neighborhood reach in Geometric space.

        `num_observers` - int:
            number os GENEO observers to form the output of the Module
        
        """
        super(GENEO_Layer, self).__init__()
        
        if gib_dict is None or gib_dict == {}:
            geneo_keys = ['cy', 'arrow', 'cone', 'neg', 'disk', 'ellip']
            self.gib_dict = {
                g : torch.randint(1, 64, (1,))[0] for g in geneo_keys
            }
        else:
            self.gib_dict = gib_dict

        self.num_observers = num_observers

        self.gibs:Mapping[str, GENEO_Operator] = nn.ModuleDict()

        # --- Initializing GENEOs ---
        for key in self.gib_dict:
            if key == 'cy':
                g_class = cylinder.Cylinder
            elif key == 'arrow':
                g_class = arrow.arrow
            elif key == 'neg':
                g_class = neg_sphere.negSpherev2
            elif key == 'disk':
                g_class = disk.Disk
            elif key == 'cone':
                g_class = cone.Cone
            elif g_class == 'ellip':
                g_class = ellipsoid.Ellipsoid

            for i in range(self.gib_dict[key]):
                self.gibs[f'{key}_{i}'] = GENEO_Operator(g_class, kernel_reach=kernel_reach)

        # --- Initializing Convex Coefficients ---
        self.lambdas = torch.rand((num_observers, len(self.gibs)))
        self.lambdas = torch.softmax(self.lambdas, dim=1) # make sure they sum to 1
        self.lambdas = nn.Parameter(self.lambdas, requires_grad=True)   


    def maintain_convexity(self):
        self.lambdas = torch.softmax(self.lambdas, dim=1)

    def get_cvx_coefficients(self) -> torch.Tensor:
        return self.lambdas

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        conv = self._perform_conv(x)

        conv_pred = self._observer_cvx_combination(conv)
        conv_pred = torch.relu(torch.tanh(conv_pred)) # (batch, num_observers, z, x, y)

        return conv_pred
    




##############################################################
# Replicating ConvTransposed with GENEO weights
##############################################################


class GENEO_Transposed(nn.ConvTranspose3d):


    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 output_padding=0, 
                 groups=1, 
                 bias=False, 
                 padding_mode='zeros', 
                 device=None, 
                 dtype=None) -> None:
        
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode, device, dtype)

        self.geneo_layer = GENEO_Layer(gib_dict=None, num_observers=out_channels, kernel_size=kernel_size)
        self.weight = None


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self._build_weights(repeat=input.shape[1])
        # print(f"forward->build_weights: {weight.shape}; {weight.requires_grad}; {weight.grad_fn}")
        conv_trans = F.conv_transpose3d(input, weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        # print(conv_trans.shape)
        # print(f"forward->conv_transpose3d: {conv_trans.shape}; {conv_trans.requires_grad}; {conv_trans.grad_fn}")
        conv_trans = self.geneo_layer._observer_cvx_combination(conv_trans)
        # print(f"forward->observer_cvx_combination: {conv_trans.shape}; {conv_trans.requires_grad}; {conv_trans.grad_fn}")
        # print(conv_trans.shape)
        return conv_trans
    
    def _build_weights(self, repeat=1) -> torch.Tensor:
        kernels = self.geneo_layer._build_kernels()
        # print(f"build_weights->build_kernels: {kernels.shape}; {kernels.requires_grad}; {kernels.grad_fn}")
        kernels = kernels.permute(1, 0, 2, 3, 4) # (1, num_geneos, k_z, k_x, k_y)
        # print(f"build_weights->permute: {kernels.shape}; {kernels.requires_grad}; {kernels.grad_fn}")
        kernels = kernels.repeat(repeat, 1, 1, 1, 1) # (num_channels, num_geneos, k_z, k_x, k_y)
        # print(f"build_weights->repeat: {kernels.shape}; {kernels.requires_grad}; {kernels.grad_fn}")
        # kernels = torch.nn.Parameter(kernels, requires_grad=True)
        return kernels

    def maintain_convexity(self):
        self.geneo_layer.maintain_convexity()

    def get_cvx_coefficients(self):
        return self.geneo_layer.get_cvx_coefficients()
    
    def get_geneo_params(self):
        return self.geneo_layer.get_geneo_params()
    


if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    import numpy as np

    def kernel2matrix(K:torch.Tensor) -> torch.Tensor:
        """
        Matrix transposition of a 2D kernel; used to perform the convolution operation with a matrix multiplication

        Parameters
        ----------
        K - torch.Tensor:
            2D tensor representing the kernel

        Returns
        -------
        W - torch.Tensor:
            2D tensor representing the kernel in matrix form
        """
        W_height, W_width = K.numel(), K.numel()*2 + (K.shape[0] - 1)
        line = torch.zeros(K.numel() + K.shape[0] - 1) # 1D tensor to store the kernel, with zeros to separate the rows

        offset = 0
        for i in range(K.shape[0]): # for each row
            line[offset:offset+K.shape[1]] = K[i]
            offset += K.shape[1] + 1

        W = torch.zeros((W_height, W_width))

        # print(W.shape)
        # print(line)

        offset = 0
        for i in range(W_height//2): # for each row
            # wrtite at the top left corner of the matrix
            W[i, offset : line.shape[0] + offset] = line
            # write at the bottom right of the matrix
            W[W_height - i - 1, W_width - line.shape[0] - offset : W_width - offset] = line
            # update the offset to write the next row
            offset += 1

        if W_height % 2 == 1:
            W[W_height//2, offset: line.shape[0] + offset] = line

        return W
    
    def _kernel2matrix(K: torch.Tensor) -> torch.Tensor:
        """
        Matrix transposition of a 2D kernel; used to perform the convolution operation with a matrix multiplication

        Parameters
        ----------
        K - torch.Tensor with shape (C, H, W)
            2D tensor representing the kernel; 

        Returns
        -------
        W - torch.Tensor:
            2D tensor representing the kernel in matrix form
        """

        if K.ndim == 2:
            return kernel2matrix(K)

        W_height, W_width = K[0].numel(), K[0].numel()*2 + (K.shape[1] - 1)
        W = torch.zeros((K.shape[0], W_height, W_width))

        # print(W.shape)

        for i in range(K.shape[0]):
            W[i] = kernel2matrix(K[i])

        return W
    

    def trans_conv(X, K):
        h, w = K.shape
        Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Y[i: i + h, j: j + w] += X[i, j] * K
        return Y 
       
    
    inputShape = [64, 64, 64]    
    batch_size = 2
    CIn        = 1
    COut       = 8    
    kernelSize = (9,7,7)
    pad        =  'valid' #'same' # (4,2,2)
    stride     = (2,2,2)

    def same_padding_3d(input_size, kernel_size, stride):
        """
        Calculate 'same' padding for 3D convolution.

        Args:
            input_size (tuple): Size of the input tensor (batch, channels, depth, height, width).
            kernel_size (tuple): Size of the convolutional kernel (depth, height, width).
            stride (tuple): Stride of the convolution (depth, height, width).

        Returns:
            tuple: Padding values for the 'same' padding.
        """
        input_depth, input_height, input_width = input_size[2:]
        kernel_depth, kernel_height, kernel_width = kernel_size
        stride_depth, stride_height, stride_width = stride

        pad_depth = ((input_depth - 1) * stride_depth + kernel_depth - input_depth) // 2
        pad_height = ((input_height - 1) * stride_height + kernel_height - input_height) // 2
        pad_width = ((input_width - 1) * stride_width + kernel_width - input_width) // 2

        return (pad_depth, pad_height, pad_width)

    # normal conv
    conv = nn.Conv3d(CIn, COut, kernelSize, stride, pad, bias=False).cuda()

    deconv = nn.ConvTranspose3d(COut, CIn, kernelSize, stride, pad, bias=False).cuda()
                      
    # alternativeConv
    def alternativeConv(X, K,
                        COut       = None,
                        kernelSize = (3,3,3),
                        pad        = (1,1,1),
                        stride     = (1,1,1) ):
        
        if pad == 'same':
            pad = same_padding_3d(X.shape, kernelSize, stride)
        elif pad == 'valid':
            pad = (0,0,0)

        def unfold3d(tensor, kernelSize, pad, stride): 

            B, C, _, _, _ = tensor.shape

            # Input shape: (B, C, D, H, W)
            
            tensor = F.pad(tensor,
                            (pad[2], pad[2],
                             pad[1], pad[1],
                             pad[0], pad[0]),
                            )

            tensor = (tensor
                      .unfold(2, size=kernelSize[0], step=stride[0])
                      .unfold(3, size=kernelSize[1], step=stride[1])
                      .unfold(4, size=kernelSize[2], step=stride[2])
                      .permute(0, 2, 3, 4, 1, 5, 6, 7)
                      .reshape(B, -1, C * np.prod(kernelSize))
                      .transpose(1, 2)
                     )
            
            return tensor
    
        B,_,H,W,D = X.shape
        outShape = ( (torch.tensor([H,W,D]) - torch.tensor(kernelSize) + 2 * torch.tensor(pad)) / torch.tensor(stride) ) + 1
        outShape = outShape.int()
        
        X = unfold3d(X, kernelSize, pad, stride)
  
        K = K.view(COut, -1)
        #K = torch.randn(COut, CIn, *kernelSize).cuda() 
        #K = K.view(COut, -1)
                    
        Y = torch.matmul(K, X).view(B, COut, *outShape)
        
        return Y
    
    
    X = torch.randn(batch_size, CIn, *inputShape).cuda()
    
    Y1 = conv(X)
    
    Y2 = alternativeConv(X, conv.weight, 
                         COut       = COut,
                         kernelSize = kernelSize,
                         pad        = pad,
                         stride     = stride
                         )
    
    print(Y2.shape)
    
    print(torch.all(torch.isclose(Y1, Y2)))   
    
    
    
    # X = torch.randint(1, 10, size=(1, 64, 64, 64))
    # # K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # K = torch.randint(1, 10, size=(1, 3, 3))
    # W = _kernel2matrix(K)
    # print(K)
    # print(W.shape)
    # print(W)
    
    
    # Example usage
    # X  = torch.arange(9.0).reshape(3, 3)
    # K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # W = kernel2matrix(K)
    # CONV = torch.matmul(W, X.reshape(-1)).reshape(2, 2)
    # DECONV = torch.matmul(W.T, CONV.reshape(-1)).reshape(3, 3)
    # Z = trans_conv(CONV, K)
    # print(K)
    # print(W)
    # print(CONV)
    # print(DECONV)
    # print(Z)
    # print(Z == DECONV)
    
