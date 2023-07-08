from typing import List, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os


sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.models.geneos.GENEO_kernel_torch import GENEO_kernel_torch
from core.models.geneos import cylinder, neg_sphere, arrow
import utils.voxelization as Vox



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
#                         GENEO Layer                         #
###############################################################

class GENEO_Layer(nn.Module):

    def __init__(self, geneo_class:GENEO_kernel_torch, kernel_size:tuple=None, smart=False):
        super(GENEO_Layer, self).__init__()  

        self.geneo_class = geneo_class
        self.init_from_config(smart)

        if kernel_size is not None:
            self.kernel_size = kernel_size

    
    def init_from_config(self, smart=False):

        if smart:
            config = self.geneo_class.geneo_smart_config()
            if config['plot']:
                print("JSON file GENEO Initialization...")
        else:
            config = self.geneo_class.geneo_random_config()
            if config['plot']:
                print("Random GENEO Initialization...")

        self.name = config['name']
        self.kernel_size = config['kernel_size']
        self.plot = config['plot']

        self.geneo_params = {}
        for param in config['geneo_params']:
            t_param = torch.tensor(config['geneo_params'][param], dtype=torch.float)
            t_param = nn.Parameter(t_param, requires_grad = not param in config['non_trainable'])
            self.geneo_params[param] = t_param

        self.geneo_params = nn.ParameterDict(self.geneo_params)


    def init_from_kwargs(self, kernel_size, kwargs):
        self.kernel_size = kernel_size
        self.geneo_params = {}
        self.name = 'GENEO'
        self.plot = False
        for param in self.geneo_class.mandatory_parameters():
            
            self.geneo_params[param] = nn.Parameter(torch.tensor(kwargs[param], dtype=torch.float))

        self.geneo_params = nn.ParameterDict(self.geneo_params)

    def compute_kernel(self) -> torch.Tensor:
        geneo = self.geneo_class(self.name, self.kernel_size, plot=self.plot, **self.geneo_params)
        kernel = geneo.kernel.to(dtype=torch.double)
        return kernel.view(1, *kernel.shape)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        geneo = self.geneo_class(self.name, self.kernel_size, plot=self.plot, **self.geneo_params)

        kernel = geneo.kernel.to(self.device, dtype=torch.double)
        
        return F.conv3d(x, kernel.view(1, 1, *kernel.shape), padding='same')



###############################################################
#                         SCENE-Nets                          #
###############################################################

class SCENE_Net(nn.Module):

    def __init__(self, geneo_num=None, kernel_size=None, plot=False,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Instantiates a SCENE-Net Module with specific GENEOs and their respective cvx coefficients.


        Parameters
        ----------
        `geneo_num` - dict:
            Mappings that contain the number of GENEOs of each kind (the key) to initialize
        
        `kernel_size` - tuple/list:
            3 elem array with the kernel_size dimensions to discretize the GENEOs in (z, x, y) format

        `plot` - bool:
            if True plot information about the Module; It's propagated to submodules

        `device` - str:
            device where to load the Module.
        """
        super(SCENE_Net, self).__init__()

        self.device = device

        if geneo_num is None:
            self.sizes = {'cy'  : 1, 
                        'cone': 1, 
                        'neg' : 1}
        else:
            self.sizes = geneo_num

        if kernel_size is not None:
            self.kernel_size = kernel_size
        # else is the default on GENEO_kernel_torch class

        self.geneos:Mapping[str, GENEO_Layer] = nn.ModuleDict()

        for key in self.sizes:
            if key == 'cy':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(cylinder.cylinder_kernel, kernel_size=kernel_size)

            elif key == 'cone':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(arrow.cone_kernel, kernel_size=kernel_size)

            elif key == 'neg':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(neg_sphere.neg_sphere_kernel, kernel_size=kernel_size)

        # --- Initializing Convex Coefficients ---
        num_lambdas = sum(self.sizes.values())
        lambda_init_max = 0.6 #2 * 1/num_lambdas
        lambda_init_min =  0 #-1/num_lambdas # for testing purposes
        self.lambdas = (lambda_init_max - lambda_init_min)*torch.rand(num_lambdas, device=self.device, dtype=torch.float) + lambda_init_min
        self.lambdas = [nn.Parameter(lamb) for lamb in self.lambdas]
    
        self.lambda_names = [f'lambda_{key}_{i}' for key, val in self.sizes.items() for i in range(val)]
        self.last_lambda = self.lambda_names[torch.randint(0, num_lambdas, (1,))[0]]
        if plot:
            print(f"last cvx_coeff: {self.last_lambda}")

        # Updating last lambda
        self.lambdas_dict = dict(zip(self.lambda_names, self.lambdas)) # last cvx_coeff is equal to 1 - sum(lambda_i)
        self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
        
        self.lambdas_dict = nn.ParameterDict(self.lambdas_dict)

        if plot:
            print(f"Total Number of train. params = {self.get_num_total_params()}")

    def get_geneo_nums(self):
        return self.sizes

    def get_cvx_coefficients(self):
        return self.lambdas_dict

    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_geneo_params(self):
        return nn.ParameterDict(dict([(name.replace('.', '_'), p) for name, p in self.named_parameters() if not 'lambda' in name]))

    def get_dict_parameters(self):
        return dict([(n, param.data.item()) for n, param in self.named_parameters()])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        kernels = torch.stack([self.geneos[geneo].compute_kernel() for geneo in self.geneos])
        conv = F.conv3d(x, kernels, padding='same')

        conv_pred = torch.zeros_like(x)

        for i, g_name in enumerate(self.geneos):
            if f'lambda_{g_name}' == self.last_lambda:
                conv_pred += (1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda])*conv[:, [i]]
                #recompute last_lambda's actual value
                self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
            else:
                conv_pred += self.lambdas_dict[f'lambda_{g_name}']*conv[:, [i]]

        conv_pred = torch.relu(torch.tanh(conv_pred))

        return conv_pred


class SceneNet(nn.Module):

    def __init__(self, geneo_num=None, kernel_size=None, plot=False):
        """
        Instantiates a SCENE-Net Module with specific GENEOs and their respective cvx coefficients.

        Parameters
        ----------
        `geneo_num` - dict:
            Mappings that contain the number of GENEOs of each kind (the key) to initialize
        
        `kernel_size` - tuple/list:
            3 elem array with the kernel_size dimensions to discretize the GENEOs in (z, x, y) format

        `plot` - bool:
            if True plot information about the Module; It's propagated to submodules

        `device` - str:
            device where to load the Module.
        """
        super(SceneNet, self).__init__()

        if geneo_num is None:
            self.sizes= {'cy' : 1, 
                        'cone': 1, 
                        'neg' : 1}
        else:
            self.sizes = geneo_num

        if kernel_size is not None:
            self.kernel_size = kernel_size
        # else is the default on GENEO_kernel_torch class

        self.geneos:Mapping[str, GENEO_Layer] = nn.ModuleDict()

        for key in self.sizes:
            if key == 'cy':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(cylinder.cylinderv2, kernel_size=kernel_size)

            elif key == 'cone':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(arrow.arrow, kernel_size=kernel_size)

            elif key == 'neg':
                for i in range(self.sizes[key]):
                    self.geneos[f'{key}_{i}'] = GENEO_Layer(neg_sphere.negSpherev2, kernel_size=kernel_size)

        # --- Initializing Convex Coefficients ---
        num_lambdas = sum(self.sizes.values())
        lambda_init_max = 1/num_lambdas
        lambda_init_min =  0 # for testing purposes
        self.lambdas = (lambda_init_max - lambda_init_min)*torch.rand(num_lambdas, dtype=torch.float) + lambda_init_min
        self.lambdas = [nn.Parameter(lamb) for lamb in self.lambdas]
    
        self.lambda_names = [f'lambda_{key}_{i}' for key, val in self.sizes.items() for i in range(val)]
        self.last_lambda = self.lambda_names[torch.randint(0, num_lambdas, (1,))[0]]
        if plot:
            print(f"last cvx_coeff: {self.last_lambda}")

        # Updating last lambda
        self.lambdas_dict = dict(zip(self.lambda_names, self.lambdas)) # last cvx_coeff is equal to 1 - sum(lambda_i)
        self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
        
        self.lambdas_dict = nn.ParameterDict(self.lambdas_dict)

        if plot:
            print(f"Total Number of train params = {self.get_num_total_params()}")


    def get_cvx_coefficients(self):
        return self.lambdas_dict

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

    def _perform_conv(self, x:torch.Tensor) -> torch.Tensor:
        kernels = torch.stack([self.geneos[geneo].compute_kernel() for geneo in self.geneos])
        conv = F.conv3d(x, kernels, padding='same') # (batch, num_geneos, z, x, y)
        return conv

    def _observer_cvx_combination(self, conv:torch.Tensor) -> torch.Tensor:
        conv_pred = []

        for i, g_name in enumerate(self.geneos):
            if f'lambda_{g_name}' == self.last_lambda:
                conv_pred.append((1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda])*conv[:, [i]])
                #recompute last_lambda's actual value
                self.lambdas_dict[self.last_lambda] = nn.Parameter(1 - sum(self.lambdas_dict.values()) + self.lambdas_dict[self.last_lambda], requires_grad=False)
            else:
                conv_pred.append(self.lambdas_dict[f'lambda_{g_name}']*conv[:, [i]])
        
        return torch.sum(conv_pred, dim=1, keepdim=True) # (batch, 1, z, x, y)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        conv = self._perform_conv(x)

        conv_pred = self._observer_cvx_combination(conv)
        conv_pred = torch.relu(torch.tanh(conv_pred))

        return conv_pred


class SceneNet_semseg(nn.Module):


    def __init__(self, 
                geneo_num:Mapping[str, int]=None, 
                kernel_size:tuple=None,
                hidden_dims:List[int]=None,
                num_classes:int=2,
                plot:bool=False, 
                device:str='cuda') -> None:
        
        self.scenenet = SceneNet(geneo_num, kernel_size, plot, device)

        self.convs = []
        self.bns = []
        self.num_classes = num_classes

        for i in range(len(hidden_dims)-1):
            self.convs.append(nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=1))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i+1]))
        
        self.out_conv = nn.Conv1d(hidden_dims[-1], num_classes, kernel_size=1)


    def forward(self, x:torch.Tensor, pt_loc:torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------

        `x`: torch.Tensor
            Input tensor of shape (batch, z, x, y)

        `pt_loc`: torch.Tensor
            Input tensor of shape (batch, P, 3) where P is the number of points

        Returns
        -------

        `pts`: torch.Tensor
            Output tensor of shape (batch, P, num_classes)
        """

        batch_size = x.shape[0]
        num_points = pt_loc.shape[1]

        conv = self.scenenet._perform_conv(x) # (batch, num_geneos, z, x, y)

        conv_pts = Vox.vox_to_pts(conv, pt_loc) # (batch, P, num_geneos)

        observer = self.scenenet._observer_cvx_combination(conv) # (batch, 1, z, x, y)
        observer = torch.relu(torch.tanh(observer))

        observer_pts = Vox.vox_to_pts(observer, pt_loc) # (batch, P, 1)

        x = torch.cat([pt_loc, conv_pts, observer_pts], dim=2) # (batch, P, 3+num_geneos+1)

        for conv, bn in zip(self.convs, self.bns):
            x = torch.relu(bn(conv(x)))
        
        x = self.out_conv(x) # (batch, P, num_classes)

        x = x.transpose(2,1).contiguous() # (batch, num_classes, P)
        x = F.log_softmax(x.view(-1, self.num_classes), dim=-1) # (batch*P, num_classes)
        x = x.view(batch_size, num_points, self.num_classes) # (batch, P, num_classes)

        return x
    
    def _compute_loss(self, pred:torch.Tensor, target:torch.Tensor, weights:torch.Tensor=None) -> torch.Tensor:
        """

        Parameters
        ----------

        `pred`: torch.Tensor
            Prediction tensor of shape (batch, P, num_classes)

        `target`: torch.Tensor
            Target tensor of shape (batch, P)

        `weights`: torch.Tensor
            Weights tensor of shape (batch, P)

        Returns
        -------

        `loss`: torch.Tensor
            Loss tensor of shape (batch, P)
        """

        if weights is None:
            weights = torch.ones_like(target) # equal weights

        loss = F.nll_loss(pred, target, reduction='none')
        loss = torch.sum(loss*weights, dim=1)

        return loss






def main():
    gnet = SCENE_Net()
    for name, param in gnet.named_parameters():
        print(f"{name}: {type(param)}; {param}")
    return

if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    from core.datasets.ts40k import ToFullDense, torch_TS40Kv2, ToTensor
    from torchvision.transforms import Compose
    from criterions.quant_loss import QuantileGENEOLoss
    from criterions.w_mse import HIST_PATH 
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from utils.observer_utils import forward, process_batch, init_metrics, visualize_batch
    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity

    EXT_PATH = "/media/didi/TOSHIBA EXT/"
    TS40K_PATH = os.path.join(EXT_PATH, 'TS40K/')

    MODEL_PATH = "/home/didi/VSCode/soa_scenenet/scenenet_pipeline/torch_geneo/saved_scnets/models_geneo/2022-08-04 16:29:24.530075/gnet.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnet = SceneNet_semseg(None, (9, 6, 6))

    for name, param in gnet.named_parameters():
        print(f"{name} = {param.item():.4f}")
    print("\n")

    input("?")


    composed = Compose([ToTensor(), ToFullDense()])
    ts40k = torch_TS40Kv2(dataset_path=TS40K_PATH, split='test', transform=composed)

    ts40k_loader = DataLoader(ts40k, batch_size=1, shuffle=True, num_workers=4)
    geneo_loss = QuantileGENEOLoss(torch.tensor([]), hist_path=HIST_PATH, alpha=1, rho=3, epsilon=0.1)
    tau=0.65
    test_metrics = init_metrics(tau) 
    test_loss = 0
    composed = Compose([ToTensor(), ToFullDense()])
    ts40k = torch_TS40Kv2(dataset_path=TS40K_PATH, split='test', transform=composed)

    ts40k_loader = DataLoader(ts40k, batch_size=1, shuffle=True, num_workers=4)
    geneo_loss = QuantileGENEOLoss(torch.tensor([]), hist_path=HIST_PATH, alpha=1, rho=3, epsilon=0.1)
    tau=0.65
    test_metrics = init_metrics(tau) 
    test_loss = 0
 
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            with record_function("model_inference"):
                vox, gt = ts40k[0]
                gnet(vox[None])

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))


        input("\n\nContinue?\n\n")


        for batch in tqdm(ts40k_loader, desc=f"testing..."):
            loss, pred = process_batch(gnet, batch, geneo_loss, None, test_metrics, requires_grad=False)
            test_loss += loss

            test_res = test_metrics.compute()
            pre = test_res['Precision']
            rec = test_res['Recall']

            # print(f"Precision = {pre}")
            # print(f"Recall = {rec}")
            #if pre <= 0.1 or rec <= 0.1:
            if pre >= 0.3 and rec >= 0.20:
            #if True:
                print(f"Precision = {pre}")
                print(f"Recall = {rec}")
                vox, gt = batch
                visualize_batch(vox, gt, pred, tau=tau)
                input("\n\nPress Enter to continue...")

            test_metrics.reset()

        test_loss = test_loss /  len(ts40k_loader)
        test_res = test_metrics.compute()
        print(f"\ttest_loss = {test_loss:.3f};")
        for met in test_res:
            print(f"\t{met} = {test_res[met]:.3f};")


    








