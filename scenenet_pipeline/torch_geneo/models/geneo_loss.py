
# %% 
import os
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import cloudpickle

import sys
from pathlib import Path

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from torch_geneo.datasets.ts40k import torch_TS40K
from VoxGENEO import Voxelization as Vox

ROOT_PROJECT = "/home/didi/VSCode/lidar_thesis"

#ROOT_PROJECT = "/home/d.lavado/lidar"

DATA_SAMPLE_DIR = ROOT_PROJECT + "/Data_sample"
SAVE_DIR = ROOT_PROJECT + "/dataset/torch_dataset"

PICKLE_PATH = os.path.join(ROOT_PROJECT, "torch_geneo/models")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(data, filename):
    with open(filename, 'wb') as handle:
        cloudpickle.dump(data, handle)


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        data = cloudpickle.load(handle)
    return data

class GENEO_Loss(torch.nn.Module):
   

    def __init__(self, targets:torch.Tensor, hist_path=PICKLE_PATH, alpha=1, rho=1, epsilon=0.1, gamma=1) -> None:
        """
        GENEO Loss is a custom loss for GENEO_Net that takes into account data imbalance
        w.r.t. regression and punishes convex coefficient that fall out of admissible values

        Parameters
        ----------
        `alpha` - float: 
            Weighting factor that tells the model how important the rarer samples are;
            The higher the alpha, the higher the rarer samples' importance in the model.

        `rho` - float: 
            Regularizing term that punishes negative convex coefficients;

        `epsilon` - float:
            Base value for the dense loss weighting function;
            Essentially, no data point is weighted with less than epsilon. So it always has at least epsilon importance;
        """

        super(GENEO_Loss, self).__init__()
        
         
        self.targets = torch.flatten(targets)
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon
        self.gamma= gamma
        self.relu = torch.nn.ReLU()
        self.pik_name = f"{os.path.join(hist_path, 'hist_estimation.pickle')}"

        if os.path.exists(self.pik_name):
            self.freqs, self.ranges = load_pickle(self.pik_name)
        else:
            print("calculating histogram estimation...")
            self.freqs, self.ranges = self.hist_frequency_estimation(self.targets)
            save_pickle((self.freqs, self.ranges), self.pik_name)
        self.freqs = self.freqs.to(device)
        self.ranges = self.ranges.to(device)

    
    def hist_frequency_estimation(self, y:torch.Tensor, hist_len=10, plot=False):
        """
        Performs a histogram frequency estimation with y;\n
        The values of y are aggregated into hist_len ranges, then the density of each range
        is calculated and normalized.\n
        This serves as a good alternative to KDE since y is univariate.

        Parameters
        ----------
        `y` - torch.Tensor:
            estimation targets; must be one dimensional with values between 0 and 1;
        
        `hist_len` - int:
            number of ranges to use when aggregating the values of y

        `plot` - bool:
            plots the calculated historgram and shows the true count for each range

        Returns
        -------
        `hist_count` - torch.Tensor:
            tensor with the frequency of each range
        
        `hist_range` - torch.Tensor:
            the employed ranges
        """

        hist_range = torch.linspace(0, 1, hist_len, device=device) # the probabilities go from 0 to 1, with hist_len bins
        y = y.to(device)
        hist_idxs = torch.abs(torch.unsqueeze(y, -1) - hist_range).argmin(dim=-1) # calculates which bin each value of y belongs to
        hist_count = torch.bincount(hist_idxs) # counts the occurence of value in hist_idxs

        if len(hist_count) < hist_len: # this happens when not all idxs in the hist have y values near them
            # These idxs are filled with zero in this case
            count_idxs = torch.unique(hist_idxs)
            org_idxs = torch.arange(hist_len).to(device)
            present_idxs = (org_idxs[:, None] == count_idxs).any(dim=1)

            hist_count = org_idxs.apply_(lambda x : hist_count[x] if present_idxs[x] else 0)

        #min_count = torch.min(hist_count)
        #count_normed = (hist_count - min_count) / (torch.max(hist_count) - min_count)
        if plot:
            print(f"hist_count = {list(zip(hist_range.numpy(), hist_count.numpy()))}")
            sns.displot(y, bins=hist_range.numpy(), kde=True)
        return hist_count, hist_range

        
    def get_dens_target(self, y:torch.Tensor):
        """
        Returns the density of each value in y following the `hist_frequency_estimation` result.
        """

        closest_idx = torch.abs(torch.unsqueeze(y, -1) - self.ranges).argmin(dim=-1)

        for idx in range(len(self.freqs)):
            closest_idx[closest_idx == idx] = self.freqs[idx]
        # f = self.freqs.cpu()
        # target_freq = closest_idx.cpu().apply_(lambda x: f[x])
       
        target_dens = closest_idx / torch.sum(self.freqs)
        return target_dens

    def get_weight_target(self, y:torch.Tensor):
        """
        Returns the weight value for each value in y according to the performed
        `hist_frequency_estimation`
        """
        y = y.to(device)
        y_dens = self.get_dens_target(y)
        weights = torch.max(1 - self.alpha*y_dens, torch.full_like(y_dens, self.epsilon, device=device))

        return weights / torch.mean(weights)

    def cvx_loss(self, cvx_coeffs:torch.nn.ParameterDict):
        """
        Penalizes non-positive convex parameters;FBetaScore
        The last cvx coefficient is calculated in function of the previous ones: phi_n = 1 - sum_i^N-1(phi_i)

        This results from the the relaxation of the cvx restriction: sum(cvx_coeffs) == 1
        """

        #penalties = [self.relu(-phi)**2 for phi in cvx_coeffs.values()]
        #print([(p, p.requires_grad) for p in penalties])

        last_phi = [phi_name for phi_name in cvx_coeffs if not cvx_coeffs[phi_name].requires_grad][0]

        #assert sum([self.relu(-phi)**2 for phi in cvx_coeffs.values()]).requires_grad

        #assert np.isclose(sum(cvx_coeffs.values()).data.item(), 1), sum(cvx_coeffs.values()).data.item()

        # return self.rho * (sum([self.relu(-phi)**2 for phi_n, phi in cvx_coeffs.items() if phi_n != last_phi])
        #                     + self.relu(-(1 - sum(cvx_coeffs.values()) + cvx_coeffs[last_phi]))**2
        #                 )

        return self.rho * (sum([self.relu(-phi) for phi_n, phi in cvx_coeffs.items() if phi_n != last_phi])
                    + self.relu(-(1 - sum(cvx_coeffs.values()) + cvx_coeffs[last_phi]))
                )

    def positive_regularizer(self, params:torch.nn.ParameterDict):
        """
        Penalizes non positive parameters
        """
        return self.rho * sum([self.relu(-g) for g in params.values()])

    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, cvx_coeffs:torch.nn.ParameterDict, geneo_params:torch.nn.ParameterDict):

        exp_y_pred, exp_y_gt = torch.broadcast_tensors(y_pred, y_gt) ## ensures equal dims;
        weights_y_gt = self.get_weight_target(exp_y_gt)

        dense_loss = torch.mean(self.gamma * weights_y_gt * (exp_y_gt - exp_y_pred)**2) ## weight_function * squared error

        if len(cvx_coeffs) == 0:
            cvx_l = 0
        else:
            cvx_l = self.cvx_loss(cvx_coeffs)

        
        if len(geneo_params) == 0:
            geneo_l = 0
        else:
            geneo_l = self.positive_regularizer(geneo_params)
        
        return dense_loss + cvx_l + geneo_l


class GENEO_Loss_Class(GENEO_Loss):

    def __init__(self, targets: torch.Tensor, hist_path=PICKLE_PATH, alpha=1, rho=1, epsilon=0.1, gamma=1) -> None:
        super().__init__(targets, hist_path, alpha, rho, epsilon, gamma)

        self.bce = torch.nn.BCELoss()


    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, cvx_coeffs:torch.nn.ParameterDict, geneo_params:torch.nn.ParameterDict):

        exp_y_pred, exp_y_gt = torch.broadcast_tensors(y_pred, y_gt) ## ensures equal dims; probably not necessary
        
        weights_y_gt = self.get_weight_target(exp_y_gt)

        bce = torch.nn.BCELoss(weight=weights_y_gt)

        return bce(exp_y_pred, exp_y_gt) + self.cvx_loss(cvx_coeffs) + self.positive_regularizer(geneo_params)



# %%
if __name__ == '__main__':
   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
    ts40k = torch_TS40K(dataset_path=SAVE_DIR)
    # targets = None
    # for (_, y) in ts40k:
    #      if targets is None:
    #          targets = y.flatten()
    #      else:
    #          targets = torch.cat([targets, y.flatten()])

    _, targets = ts40k[2]

    
    #targets = torch.rand(1000)

    # %%
    import scipy.stats as st

    kde = st.gaussian_kde(targets.flatten())
    lin = torch.linspace(0, 1, 1000)
    plt.plot(lin, kde.pdf(lin), label="PDF")

    # %%
    print(f"targets size = {targets.shape}")
    print(f"TS40K number of samples = {len(ts40k)}")

    loss = GENEO_Loss(targets, alpha=2, epsilon=0.001)
    # %%
    # y = torch.flatten(targets)
    # freq, range = loss.hist_density_estimation(y, plot=True)
    freq = loss.freqs
    w = loss.get_weight_target(loss.ranges)
    min_dens = torch.min(freq)
    dens = (freq - min_dens) / (torch.max(freq) - min_dens)
    dens = freq / torch.sum(freq)
    float_formatter = "{:.8f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print(f" range\t frequency\t1/density\t weight")
    print(np.array([*zip(loss.ranges.cpu().numpy(), freq.cpu().numpy(), 1/dens.cpu().numpy(), w.cpu().numpy())]))
    plt.show()
  
    plt.plot(loss.ranges.cpu().numpy(), dens.cpu().numpy(), label='y density')
    plt.plot(loss.ranges.cpu().numpy(), w.cpu().numpy(), label='f_w')
    plt.legend()
    plt.show()
    #sns.displot(w, bins=range, kde=True)



    

        



# %%
