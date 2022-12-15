from typing import Tuple, Union
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from scenenet_pipeline.calibration.plot_calibration import ConfidenceHistogram, ReliabilityDiagram

from scenenet_pipeline.calibration.temperature_scaling import _ECELoss
from scenenet_pipeline.torch_geneo.criterions.w_mse import WeightedMSE
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
        print(out_features)
        self.linear = torch.nn.Linear(out_features, out_features)

    def forward(self, x):
        with torch.no_grad():
            pred = self.model(x)

        pred = torch.flatten(pred, dim=1) # cuz linear model need flattened data; dim 0 is the batch dim

        # kinda jank I know
        pred, idxs = torch.sort(pred, dim=1, descending=True)
        pred, idxs  = pred[:, :self.out_f], idxs[:, self.out_f] # we select the highest probabilities to regress
        # pred = pred[torch.randperm(pred.shape[1])] #shuffle prediction

        outputs = torch.sigmoid(self.linear(pred))
        return outputs, idxs


    def init_metrics(self, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        return MetricCollection([
                    MeanAbsoluteError(),
                    MeanSquaredError()
        ]).to(device)


def transform_forward(batch, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    return batch[0].to(device), batch[1].to(device)

def transform_metrics(pred:torch.Tensor, target:torch.Tensor):
    return torch.flatten(pred), torch.flatten(target)


def process_batch(model, batch, geneo_loss, opt, metrics, requires_grad=True):
    batch = transform_forward(batch)
    loss, pred = forward(model, batch, geneo_loss, opt, requires_grad)
    if metrics is not None:
        metrics(*transform_metrics(pred, batch[1]))
    return loss, pred

def forward(model:torch.nn.Module, batch, criterion:torch.nn.Module, opt: Union[torch.optim.Optimizer, None], requires_grad=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a forward pass of `model` with data `batch`, loss `criterion` and optimizer `opt`.

    if `requires_grad`, then it computes the backwards pass through the network.

    Returns
    -------
    `loss` - float:
        loss value computed with `criterion`

    `pred` - torch.tensor:
        gnet's prediction

    """
   
    input, target = batch
    
    # --- Forward pass ---
    #start = time.time()
    pred, idxs = model(input)
    #end = time.time()
    #print(f"Prediction inference time: {end - start}")

    loss = criterion(pred, target[:, idxs]) 

    # --- Backward pass ---
    if requires_grad:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss, pred


def calibrate(model:nn.Module, in_feat, val_loader, epochs, 
              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        
        log_model = ModelWithLogCalibration(model.to(device), in_feat)
        mse_criterion = WeightedMSE(None).to(device)
        ece_criterion = _ECELoss().to(device)

        optimizer = optim.RMSprop(log_model.linear.parameters())

        metrics = log_model.init_metrics()
        ece = 0
        mse = 0
        for e in tqdm(range(epochs), desc=f"Calibrating..."):

            for input, target in val_loader:
                    
                mse_loss, pred = process_batch(model, (input, target), mse_criterion, optimizer, metrics)
                
                target = target.to(device)
                ece_loss = ece_criterion(torch.flatten(pred).reshape(-1, 1), torch.flatten(target))
                ece += ece_loss
                mse += mse_loss

            cal_res = metrics.compute()
            print(f"\n\nEpoch {e} / {epochs}:")
            print(f"\t\tWeighted MSE Loss: {mse / len(val_loader)}")
            print(f"\t\tECE Loss: {ece.item() / len(val_loader)}")
            for met in cal_res: 
                print(f"\t\t{met} = {cal_res[met]};")

        
        # Calibration Diagrams
        pred = torch.flatten(pred).reshape(-1, 1).cpu().numpy()
        target = torch.flatten(target).cpu().numpy()

        hist = ConfidenceHistogram()
        hist = hist.plot(pred, target, n_bins=20, logits=False, title='ConfidenceHistogram')
        hist.savefig('ConfidenceHistogram.png')
        dia = ReliabilityDiagram()
        dia = dia.plot(pred, target, n_bins=20, logits=False, title='ReliabilityDiagram')
        dia.savefig('ReliabilityDiagram.png')






        




