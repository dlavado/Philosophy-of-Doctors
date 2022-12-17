from typing import Tuple, Union
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from scenenet_pipeline.calibration.plot_calibration import ConfidenceHistogram, ReliabilityDiagram

from scenenet_pipeline.calibration.temperature_scaling import _ECELoss
from scenenet_pipeline.torch_geneo.criterions.w_mse import HIST_PATH, WeightedMSE
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError

class ModelWithLogCalibration(nn.Module):
    """
    Wrapper class on uncalibrated regressor to perform probability calibration
    through Logistic Regression
    """

    def __init__(self, model:nn.Module, out_features) -> None:
        super(ModelWithLogCalibration, self).__init__()

        self.model = model
        self.out_f = out_features
        self.linear = torch.nn.Linear(out_features, out_features)

    def forward(self, x):
        with torch.no_grad():
            pred = self.model(x)

        pred = torch.flatten(pred, start_dim=1) # cuz linear model need flattened data; dim 0 is the batch dim

        # kinda jank I know
        pred, idxs = torch.sort(pred, dim=1, descending=True)
        pred, idxs  = pred[:, :self.out_f], idxs[:, :self.out_f] # we select the highest probabilities to regress

        outputs = torch.sigmoid(self.linear(pred.to(torch.float)))
        return outputs, idxs


    def init_metrics(self, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        return MetricCollection([
                    MeanAbsoluteError(),
                    MeanSquaredError()
        ]).to(device)

def transform_forward(batch, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    return batch[0].to(device), batch[1].to(device)


def calibrate(model:nn.Module, in_feat, val_loader, epochs, 
              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        log_model = ModelWithLogCalibration(model.to(device), in_feat).to(device)
        mse_criterion = WeightedMSE(torch.tensor([]), hist_path=HIST_PATH).to(device)
        ece_criterion = _ECELoss().to(device)

        optimizer = optim.RMSprop(log_model.linear.parameters())

        for e in tqdm(range(epochs), desc=f"Calibrating..."):
            ece = 0
            mse = 0

            for input, target in val_loader:
                
                # forward
                input, target = transform_forward((input, target))
                pred, idxs = log_model(input)

                target = torch.flatten(target, start_dim=1)
                target = torch.concat([target[i, idxs[i, :in_feat]].reshape(1, -1) for i in range(target.shape[0])], dim=0)

                mse_loss = mse_criterion(pred, target) 

                # --- Backward pass ---
                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()

                target = target.to(device)
                ece_loss = ece_criterion(torch.flatten(pred).reshape(-1, 1), torch.flatten(target))
                ece += ece_loss
                mse += mse_loss

            print(f"\n\nEpoch {e} / {epochs}:")
            print(f"\t\tWeighted MSE Loss: {mse / len(val_loader)}")
            print(f"\t\tECE Loss: {ece.item() / len(val_loader)}")

        
        # Calibration Diagrams
        pred = torch.flatten(pred).reshape(-1, 1).detach().cpu().numpy()
        target = torch.flatten(target).detach().cpu().numpy()

        hist = ConfidenceHistogram()
        hist = hist.plot(pred, target, n_bins=20, logits=False, title='ConfidenceHistogram')
        hist.savefig('ConfidenceHistogram.png')
        dia = ReliabilityDiagram()
        dia = dia.plot(pred, target, n_bins=20, logits=False, title='ReliabilityDiagram')
        dia.savefig('ReliabilityDiagram.png')






        




