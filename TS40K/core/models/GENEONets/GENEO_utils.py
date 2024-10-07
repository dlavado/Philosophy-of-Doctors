from typing import List, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os


sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.models.GENEONets.geneos.GENEO_kernel_torch import GENEO_kernel
from core.models.GENEONets.geneos import cylinder, neg_sphere, arrow, disk, cone, ellipsoid



def load_state_dict(model_path, gnet_class, model_tag='loss', kernel_size=None):
    """
    Returns SCENE-Net model and model_checkpoint
    """
    # print(model_path)
    # --- Load Best Model ---
    if os.path.exists(model_path):
        run_state_dict = torch.load(model_path)
        if model_tag == 'loss' and 'best_loss' in run_state_dict['models']:
            model_tag = 'best_loss'
        if model_tag in run_state_dict['models']:
            if kernel_size is None:
                kernel_size = run_state_dict['model_props'].get('kernel_size', (9, 6, 6))
            gnet = gnet_class(run_state_dict['model_props']['geneos_used'], 
                              kernel_size=kernel_size, 
                              plot=False)
            print(f"Loading Model in {model_path}")
            model_chkp = run_state_dict['models'][model_tag]

            try:
                gnet.load_state_dict(model_chkp['model_state_dict'])
            except RuntimeError: 
                for key in list(model_chkp['model_state_dict'].keys()):
                    model_chkp['model_state_dict'][key.replace('phi', 'lambda')] = model_chkp['model_state_dict'].pop(key) 
                gnet.load_state_dict(model_chkp['model_state_dict'])
            return gnet, model_chkp
        else:
            ValueError(f"{model_tag} is not a valid key; run_state_dict contains: {run_state_dict['models'].keys()}")
    else:
        ValueError(f"Invalid model path: {model_path}")

    return None, None


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

    def __init__(self, geneo_class:GENEO_kernel, kernel_size:tuple=None):
        super(GENEO_Operator, self).__init__()  

        self.geneo_class = geneo_class
    
        if kernel_size is not None:
            self.kernel_size = kernel_size

            self.kernel_size = _triple(self.kernel_size)

        self.init_from_config()

    
    def init_from_config(self):

        config = self.geneo_class.geneo_random_config(kernel_size=self.kernel_size)

        self.name = config['name']
        self.plot = config['plot']

        self.geneo_params = {}
        for param in config['geneo_params']:
            if isinstance(config['geneo_params'][param], torch.Tensor):
                t_param = config['geneo_params'][param].to(dtype=torch.float)
            else:
                t_param = torch.tensor(config['geneo_params'][param], dtype=torch.float)
            t_param = nn.Parameter(t_param, requires_grad = not param in config['non_trainable'])
            self.geneo_params[param] = t_param

        self.geneo_params = nn.ParameterDict(self.geneo_params)


    def init_from_kwargs(self, kernel_size, **kwargs):
        self.kernel_size = kernel_size
        self.geneo_params = {}
        self.name = 'GENEO'
        self.plot = False
        for param in self.geneo_class.mandatory_parameters():
            self.geneo_params[param] = nn.Parameter(torch.tensor(kwargs[param], dtype=torch.float))

        self.geneo_params = nn.ParameterDict(self.geneo_params)

    def compute_kernel(self) -> torch.Tensor:
        geneo:GENEO_kernel = self.geneo_class(self.name, self.kernel_size, plot=self.plot, **self.geneo_params)
        kernel:torch.Tensor = geneo.compute_kernel().to(dtype=torch.float32)*geneo.sign
        return kernel.unsqueeze(0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        kernel = self.compute_kernel()  
        return F.conv3d(x, kernel.view(1, 1, *kernel.shape), padding='same')



###############################################################
#                         SCENE-Nets                          #
###############################################################

class GENEO_Layer(nn.Module):

    def __init__(self, geneo_num:dict, num_observers:int=1, kernel_size=None):
        """
        Instantiates a GENEO-Layer Module with specific GENEOs and their respective cvx coefficients.

        Parameters
        ----------
        `geneo_num` - dict:
            Mappings that contain the number of GENEOs of each kind (the key) to initialize

        `num_observers` - int:
            number os GENEO observers to form the output of the Module
        
        `kernel_size` - tuple/list:
            3 elem array with the kernel_size dimensions to discretize the GENEOs in (z, x, y) format
        """
        super(GENEO_Layer, self).__init__()
        
        if geneo_num is None:
            geneo_keys = ['cy', 'arrow', 'cone', 'neg', 'disk', 'ellip']
            self.geneo_kernel_arch = {
                g : torch.randint(1, 64, (1,))[0] for g in geneo_keys
            }
        else:
            self.geneo_kernel_arch = geneo_num

        self.num_observers = num_observers

        if kernel_size is not None:
            self.kernel_size = kernel_size
            self.kernel_size = _triple(self.kernel_size)
        # else is the default on @GENEO_kernel_torch class, which is (9, 6, 6)

        self.geneos:Mapping[str, GENEO_Operator] = nn.ModuleDict()

        # --- Initializing GENEOs ---
        for key in self.geneo_kernel_arch:
            if key == 'cy':
                g_class = cylinder.cylinderv2
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

            for i in range(self.geneo_kernel_arch[key]):
                self.geneos[f'{key}_{i}'] = GENEO_Operator(g_class, kernel_size=kernel_size)

        # --- Initializing Convex Coefficients ---
        self.lambdas = torch.rand((num_observers, len(self.geneos)))
        self.lambdas = torch.softmax(self.lambdas, dim=1) # make sure they sum to 1
        self.lambdas = nn.Parameter(self.lambdas, requires_grad=True)   


    def maintain_convexity(self):
        with torch.no_grad():
            # torch.clip(self.lambdas, 0, 1, out=self.lambdas)
            # self.lambdas = nn.Parameter(torch.relu(torch.tanh(self.lambdas)), requires_grad=True).to('cuda')
            self.lambdas[:, -1] = 1 - torch.sum(self.lambdas[:, :-1], dim=1)
            self.lambdas = torch.relu(self.lambdas)


    def get_cvx_coefficients(self):
        return nn.ParameterDict({'lambda': self.lambdas})

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_parameters(self, detach=False):
        if detach:
            return {name: torch.tensor(param.detach()) for name, param in self.named_parameters()}
        return {name: param for name, param in self.named_parameters()}

    def get_geneo_params(self):
        return nn.ParameterDict(dict([(name.replace('.', '_'), p) for name, p in self.named_parameters() if not 'lambda' in name]))

    def get_model_parameters_in_dict(self):
        ddd = {}
        for key, val in self.named_parameters(): #update theta to remove the module prefix
            key_split = key.split('.')
            parameter_name = f"{key_split[-3]}.{key_split[-1]}" if 'geneo' in key else key_split[-1]
            ddd[parameter_name] = val.data.item()
        return ddd
        #return dict([(n, param.data.item()) for n, param in self.named_parameters()])
        
    def _build_kernels(self) -> torch.Tensor:
        kernels = torch.stack([self.geneos[geneo].compute_kernel() for geneo in self.geneos]) # shape: (num_geneos, 1, k_z, k_x, k_y)
        return kernels

    def _perform_conv(self, x:torch.Tensor) -> torch.Tensor:
        kernels = self._build_kernels()
        # extend the kernel to the number of channels in x
        kernels = kernels.repeat(1, x.shape[1], 1, 1, 1) # (num_geneos, num_channels, k_z, k_x, k_y)
        conv = F.conv3d(x, kernels, padding='same') # (batch, num_geneos, z, x, y)
        return conv

    def _observer_cvx_combination(self, conv:torch.Tensor) -> torch.Tensor:
        """
        Performs the convex combination of the GENEOs convolutions to form the output of the SCENE-Net

        lambdas are the convex coeffcients with shape: (num_observers, num_geneos)

        `conv` - torch.Tensor:
            convolution output of the GENEO kernels with shape: (batch, num_geneos, z, x, y)

        returns
        -------
        cvx_comb - torch.Tensor:
            convex combination of the GENEOs convolutions with shape: (batch, num_observers, z, x, y)
        """

        cvx_comb_list = []

        for i in range(self.num_observers):
            comb = torch.sum(torch.relu(self.lambdas[i, :, None, None, None]) * conv, dim=1, keepdim=True)
            cvx_comb_list.append(comb)

        # Concatenate the computed convex combinations along the second dimension (num_observers)
        cvx_comb = torch.cat(cvx_comb_list, dim=1) # (batch, num_observers, z, x, y)

        return cvx_comb

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

        self.geneo_layer = GENEO_Layer(geneo_num=None, num_observers=out_channels, kernel_size=kernel_size)
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
    
