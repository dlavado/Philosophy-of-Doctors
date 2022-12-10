
# %% 
import os
from pathlib import Path
import torch
import seaborn as sns
import cloudpickle

def save_pickle(data, filename):
    with open(filename, 'wb') as handle:
        cloudpickle.dump(data, handle)


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        data = cloudpickle.load(handle)
    return data


ROOT_PROJECT = Path(os.path.abspath(__file__)).parents[3].resolve()
HIST_PATH = os.path.join(ROOT_PROJECT, 'scenenet_pipeline/torch_geneo/criterions/hist_estimation.pickle')


class WeightedMSE(torch.nn.Module):

    def __init__(self, targets:torch.Tensor, hist_path=HIST_PATH, alpha=1, rho=1, epsilon=0.1, gamma=1) -> None:
        """

        Weighted MSE criterion.
        If no weights exist (i.e., if hist_path is not valid), then they will be built from `targets` through inverse density estimation.
        Else, the weights will be the ones previously defined in `hist_path`

        Parameters
        ----------

        `targets` - torch.tensor:
            Target values to build weighted MSE

        `hist_path` - Path:
            If existent, previously computed weights

        `alpha` - float: 
            Weighting factor that tells the model how important the rarer samples are;
            The higher the alpha, the higher the rarer samples' importance in the model.

        `rho` - float: 
            Regularizing term that punishes negative convex coefficients;

        `epsilon` - float:
            Base value for the dense loss weighting function;
            Essentially, no data point is weighted with less than epsilon. So it always has at least epsilon importance;
        """

        super(WeightedMSE, self).__init__()
         
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon
        self.gamma= gamma
        self.relu = torch.nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if hist_path is not None and os.path.exists(hist_path):
            self.pik_name = hist_path
            self.freqs, self.ranges = load_pickle(self.pik_name)
        else:
            print("calculating histogram estimation...")
            self.freqs, self.ranges = self.hist_frequency_estimation(torch.flatten(targets))
            save_pickle((self.freqs, self.ranges), f"{os.path.join('.', 'hist_estimation.pickle')}")
        self.freqs = self.freqs.to(self.device)
        self.ranges = self.ranges.to(self.device)

    
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

        hist_range = torch.linspace(0, 1, hist_len, device=self.device) # the probabilities go from 0 to 1, with hist_len bins
        y = y.to(self.device)
        hist_idxs = torch.abs(torch.unsqueeze(y, -1) - hist_range).argmin(dim=-1) # calculates which bin each value of y belongs to
        hist_count = torch.bincount(hist_idxs) # counts the occurence of value in hist_idxs

        if len(hist_count) < hist_len: # this happens when not all idxs in the hist have y values near them
            # These idxs are filled with zero in this case
            count_idxs = torch.unique(hist_idxs)
            org_idxs = torch.arange(hist_len).to(self.device)
            present_idxs = (org_idxs[:, None] == count_idxs).any(dim=1)

            hist_count = org_idxs.apply_(lambda x : hist_count[x] if present_idxs[x] else 0)

        # min_count = torch.min(hist_count)
        # count_normed = (hist_count - min_count) / (torch.max(hist_count) - min_count)
        if plot:
            print(f"hist_count = {list(zip(hist_range.numpy(), hist_count.numpy()))}")
            sns.displot(y, bins=hist_range.numpy(), kde=True)
        return hist_count, hist_range

    def get_dens_target(self, y:torch.Tensor, calc_weights = False):
        """
        Returns the density of each value in y following the `hist_frequency_estimation` result.
        """

        if calc_weights:
            self.freqs, self.ranges = self.hist_frequency_estimation(y)

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
        y = y.to(self.device)
        y_dens = self.get_dens_target(y)
        weights = torch.max(1 - self.alpha*y_dens, torch.full_like(y_dens, self.epsilon, device=self.device))

        return weights / torch.mean(weights)

    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor):
        exp_y_pred, exp_y_gt = torch.broadcast_tensors(y_pred, y_gt) ## ensures equal dims
        weights_y_gt = self.get_weight_target(exp_y_gt)

        return torch.sum(self.gamma * weights_y_gt * (exp_y_gt - exp_y_pred)**2) ## weight_function * squared error