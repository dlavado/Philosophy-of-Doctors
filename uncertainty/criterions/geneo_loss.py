
# %% 
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cloudpickle

from criterions.dice_loss import BinaryDiceLoss, BinaryDiceLoss_BCE
from criterions.tversky_loss import FocalTverskyLoss
from criterions.w_mse import WeightedMSE



def save_pickle(data, filename):
    with open(filename, 'wb') as handle:
        cloudpickle.dump(data, handle)


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        data = cloudpickle.load(handle)
    return data


class GENEO_Loss(WeightedMSE):
    """
    GENEO Loss is a custom loss for SCENE-Net that takes into account data imbalance
    w.r.t. regression and punishes convex coefficient that fall out of admissible values
    """

    def __init__(self, targets: torch.Tensor, hist_path=None, alpha=1, rho=1, epsilon=0.1, gamma=1) -> None:
       
        super().__init__(targets, hist_path, alpha, rho, epsilon, gamma)

    def cvx_loss(self, cvx_coeffs:torch.nn.ParameterDict):
        """
        Penalizes non-positive convex parameters;
        The last cvx coefficient is calculated in function of the previous ones: phi_n = 1 - sum_i^N-1(phi_i)

        This results from the the relaxation of the cvx restriction: sum(cvx_coeffs) == 1
        """

        #penalties = [self.relu(-phi)**2 for phi in cvx_coeffs.values()]
        #print([(p, p.requires_grad) for p in penalties])

        if len(cvx_coeffs) == 0:
            return 0

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
        if len(params) == 0:
            return 0

        return self.rho * sum([self.relu(-g) for g in params.values()])

    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, cvx_coeffs:torch.nn.ParameterDict, geneo_params:torch.nn.ParameterDict):

        dense_criterion = super().forward(y_pred, y_gt)
        
        cvx_penalty = self.cvx_loss(cvx_coeffs)
        
        non_positive_penalty = self.positive_regularizer(geneo_params)
        
        return dense_criterion + cvx_penalty + non_positive_penalty


# class GENEO_Loss_BCE(GENEO_Loss):

#     def __init__(self, targets: torch.Tensor, hist_path, alpha=1, rho=1, epsilon=0.1, gamma=1) -> None:
#         super().__init__(targets, hist_path, alpha, rho, epsilon, gamma)

#         self.bce = torch.nn.BCELoss()


#     def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, cvx_coeffs:torch.nn.ParameterDict, geneo_params:torch.nn.ParameterDict):

#         exp_y_pred, exp_y_gt = torch.broadcast_tensors(y_pred, y_gt) ## ensures equal dims; probably not necessary
#         weights_y_gt = self.get_weight_target(exp_y_gt)
#         bce = torch.nn.BCELoss(weight=weights_y_gt)

#         return bce(exp_y_pred, exp_y_gt) + self.cvx_loss(cvx_coeffs) + self.positive_regularizer(geneo_params)



class GENEO_Dice_BCE(GENEO_Loss):

    def __init__(self, targets: torch.Tensor, hist_path, alpha=1, rho=1, epsilon=0.1, gamma=1, reduction='mean') -> None:
        
        super().__init__(targets, hist_path, alpha, rho, epsilon, gamma)

        self.dice_bce = BinaryDiceLoss_BCE(targets, hist_path, alpha, rho, epsilon, gamma, reduction=reduction)


    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, cvx_coeffs:torch.nn.ParameterDict, geneo_params:torch.nn.ParameterDict):


        return self.gamma*self.dice_bce(y_pred, y_gt) + self.cvx_loss(cvx_coeffs) + self.positive_regularizer(geneo_params)


class GENEO_Dice_Loss(GENEO_Loss):

    def __init__(self, targets: torch.Tensor, hist_path, alpha=1, rho=1, epsilon=0.1, gamma=1) -> None:
        
        super().__init__(targets, hist_path, alpha, rho, epsilon, gamma)

        self.dice = BinaryDiceLoss()


    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, cvx_coeffs:torch.nn.ParameterDict, geneo_params:torch.nn.ParameterDict):

        dense_criterion = WeightedMSE.forward(self, y_pred, y_gt)

        return dense_criterion + self.dice(y_pred, y_gt) + self.cvx_loss(cvx_coeffs) + self.positive_regularizer(geneo_params)

class GENEO_Tversky_Loss(GENEO_Loss):

    def __init__(self, targets: torch.Tensor, hist_path, alpha=1, rho=1, epsilon=0.1, gamma=1, tversky_alpha=0.5, tversky_beta=1) -> None:
        
        super().__init__(targets, hist_path, alpha, rho, epsilon, gamma)

        self.tversky = FocalTverskyLoss()
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta


    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, cvx_coeffs:torch.nn.ParameterDict, geneo_params:torch.nn.ParameterDict):

        dense_criterion = WeightedMSE.forward(self, y_pred, y_gt)

        tversky_crit = self.tversky(y_pred, y_gt, alpha=self.tversky_alpha, beta=self.tversky_beta)

        return dense_criterion + tversky_crit + self.cvx_loss(cvx_coeffs) + self.positive_regularizer(geneo_params)



# %%
if __name__ == '__main__':
   
    from datasets.ts40k import torch_TS40Kv2
    EXT_PATH = "/media/didi/TOSHIBA EXT/"
    TS40K_PATH = os.path.join(EXT_PATH, 'TS40K/')

    ts40k = torch_TS40Kv2(dataset_path=TS40K_PATH)
    # targets = None
    # for (_, y) in ts40k:
    #      if targets is None:
    #          targets = y.flatten()
    #      else:
    #          targets = torch.cat([targets, y.flatten()])

    #_, targets = ts40k[2]

    targets = []

    
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
