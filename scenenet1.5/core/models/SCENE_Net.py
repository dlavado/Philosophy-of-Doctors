from typing import List, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os


sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.models.geneos.GENEO_kernel_torch import GENEO_kernel
from core.models.geneos import cylinder, neg_sphere, arrow, disk, cone, ellipsoid
import utils.voxelization as Vox
from core.datasets.torch_transforms import Normalize_PCD



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

    def __init__(self, geneo_class:GENEO_kernel, kernel_size:tuple=None):
        super(GENEO_Layer, self).__init__()  

        self.geneo_class = geneo_class
    
        if kernel_size is not None:
            self.kernel_size = kernel_size

        self.init_from_config()

    
    def init_from_config(self):

        config = self.geneo_class.geneo_random_config(kernel_size=self.kernel_size)

        self.name = config['name']
        self.plot = config['plot']

        self.geneo_params = {}
        for param in config['geneo_params']:
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
        kernel:torch.Tensor = geneo.compute_kernel().to(dtype=torch.double)*geneo.sign
        return kernel.unsqueeze(0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        kernel = self.compute_kernel()  
        return F.conv3d(x, kernel.view(1, 1, *kernel.shape), padding='same')



###############################################################
#                         SCENE-Nets                          #
###############################################################

class SceneNet(nn.Module):

    def __init__(self, geneo_num:dict, num_observers:int=1, kernel_size=None):
        """
        Instantiates a SCENE-Net Module with specific GENEOs and their respective cvx coefficients.

        Parameters
        ----------
        `geneo_num` - dict:
            Mappings that contain the number of GENEOs of each kind (the key) to initialize

        `num_observers` - int:
            number os GENEO observers to form the output of the Module
        
        `kernel_size` - tuple/list:
            3 elem array with the kernel_size dimensions to discretize the GENEOs in (z, x, y) format
        """
        super(SceneNet, self).__init__()

        self.geneo_kernel_arch = geneo_num
        self.num_observers = num_observers

        if kernel_size is not None:
            self.kernel_size = kernel_size
        # else is the default on @GENEO_kernel_torch class, which is (9, 6, 6)

        self.geneos:Mapping[str, GENEO_Layer] = nn.ModuleDict()

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
                self.geneos[f'{key}_{i}'] = GENEO_Layer(g_class, kernel_size=kernel_size)

        # --- Initializing Convex Coefficients ---
        self.lambdas = torch.rand((num_observers, len(self.geneos)))
        self.lambdas = torch.softmax(self.lambdas, dim=1) # make sure they sum to 1
        self.lambdas = nn.Parameter(self.lambdas, requires_grad=True)   


    def maintain_convexity(self):
        with torch.no_grad():
            # torch.clip(self.lambdas, 0, 1, out=self.lambdas)
            # self.lambdas = nn.Parameter(torch.relu(torch.tanh(self.lambdas)), requires_grad=True).to('cuda')
            self.lambdas[:, -1] = 1 - torch.sum(self.lambdas[:, :-1], dim=1)


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

    def _perform_conv(self, x:torch.Tensor) -> torch.Tensor:
        kernels = torch.stack([self.geneos[geneo].compute_kernel() for geneo in self.geneos])
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
        conv_pred = torch.relu(torch.tanh(conv_pred))

        return conv_pred


class SceneNet_multiclass(nn.Module):
    """

    SceneNet 1.5 - 3D Convolutional Neural Network for Scene Classification based on GENEOs
    multiclass is accomplished by adding a fully connected layer after the last convolutional layer
    """


    def __init__(self, 
                geneo_num:Mapping[str, int]=None, 
                num_observers:int=1,
                kernel_size:tuple=None,
                hidden_dims:List[int]=None,
                num_classes:int=2) -> None:
        
        super(SceneNet_multiclass, self).__init__()
        
        self.scenenet = SceneNet(geneo_num, num_observers, kernel_size)

        #self.feat_dim = sum(geneo_num.values()) + 4 # +1 for the cvx combination and +3 for the point locations
        self.feat_dim = (sum(geneo_num.values()) + num_observers)*2 + 3 # +1 for the cvx combination and +3 for the point locations

        self.convs = []
        self.bns = []
        self.num_classes = num_classes

        self.normalize_pt_locs = Normalize_PCD()

        self.convs.append(nn.Conv1d(self.feat_dim, hidden_dims[0], kernel_size=1))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))

        for i in range(len(hidden_dims)-1):
            self.convs.append(nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=1))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i+1]))
        
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        
        self.out_conv = nn.Conv1d(hidden_dims[-1], num_classes, kernel_size=1)

    def get_geneo_params(self):
        return self.scenenet.get_geneo_params()

    def get_cvx_coefficients(self):
        return self.scenenet.get_cvx_coefficients()
    
    def maintain_convexity(self):
        self.scenenet.maintain_convexity()

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

        # print(f"conv = shape: {conv.shape}, req_grad: {conv.requires_grad}")
    
        conv_pts = Vox.vox_to_pts(conv, pt_loc) # (batch, P, num_geneos)

        # print(f"conv_pts = shape: {conv_pts.shape}, req_grad: {conv_pts.requires_grad}")

        observer = self.scenenet._observer_cvx_combination(conv) # (batch, 1, z, x, y)
        observer = torch.relu(torch.tanh(observer)) # class probability

        # print(f"observer = shape: {observer.shape}, req_grad: {observer.requires_grad}")

        observer_pts = Vox.vox_to_pts(observer, pt_loc) # (batch, P, num_observers)

        # print(f"observed_pts = shape: {observer_pts.shape}, req_grad: {observer_pts.requires_grad}")

        # normalize pt_locs
        pt_loc = self.normalize_pt_locs.normalize(pt_loc)

        global_feat_descriptor = torch.cat([conv_pts, observer_pts], dim=2) # (batch, P, num_geneos+ num_observers)
        # print(f"global_feat_descriptor = shape: {global_feat_descriptor.shape}, req_grad: {global_feat_descriptor.requires_grad}")

        global_feat_descriptor = global_feat_descriptor.transpose(2,1).contiguous() # (batch, num_geneos+ num_observers, P)

        global_feat_descriptor = torch.max_pool1d(global_feat_descriptor, kernel_size=num_points, padding=0, stride=1) # (batch, num_geneos+ num_observers, 1)

        # print(f"global_feat_descriptor = shape: {global_feat_descriptor.shape}, req_grad: {global_feat_descriptor.requires_grad}")

        global_feat_descriptor = global_feat_descriptor.squeeze(2).unsqueeze(1) # (batch, 1, num_geneos+ num_observers)

        x = torch.cat([pt_loc, conv_pts, observer_pts], dim=2) # (batch, P, 3 + num_geneos + num_observers)

        # print(f"x cat1 = shape: {x.shape}, req_grad: {x.requires_grad}")

        x = torch.cat([x, global_feat_descriptor.repeat(1, num_points, 1)], dim=2) # (batch, P, 3 + (num_geneos+ num_observers)*2)

        x = x.transpose(2,1).contiguous() # (batch, 3+num_geneos+1, P)

        # print(f"x transposed = shape: {x.shape}, req_grad: {x.requires_grad}")

        x = x.to(torch.float32)

        for conv, bn in zip(self.convs, self.bns):
            # print(f"x = shape: {x.shape}, req_grad: {x.requires_grad}, type: {x.dtype}")
            # print(f"conv = shape: {conv.weight.shape}, req_grad: {conv.weight.requires_grad}, type: {conv.weight.dtype}")
            x = torch.relu(bn(conv(x))) # (batch, hidden_dims[i+1], P)
        
        x = self.out_conv(x) # (batch, num_classes, P)

        # print(f"x = shape: {x.shape}, req_grad: {x.requires_grad}")

        return x
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------

        `model_output`: torch.Tensor
            Output tensor of shape (batch, num_classes, P)

        Returns
        -------

        `pts`: torch.Tensor
            Output tensor of shape (batch, P)
        """

        return torch.argmax(model_output.permute(0, 2, 1), dim=2) # (batch, P)

    
    def _compute_loss(pred:torch.Tensor, target:torch.Tensor, weights:torch.Tensor=None) -> torch.Tensor:
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
    gnet = SceneNet()
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
    gnet = SceneNet_multiclass(None, (9, 6, 6))

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


    








