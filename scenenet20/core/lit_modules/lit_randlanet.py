

import torch
from torch.nn.modules import Module
from core.lit_modules.lit_model_wrappers import LitWrapperModel
from core.models.randlanet import RandLANet



class LitRandLANet(LitWrapperModel):
    

    def __init__(self, 
                 criterion: Module, 
                 optimizer_name: str, 
                 in_channels, num_classes, num_neighbors=16, decimation=4,
                 learning_rate=0.01, 
                 metric_initializer=None, 
                 **kwargs):


        model = RandLANet(
            in_channels, num_classes, num_neighbors=num_neighbors, decimation=decimation
        )

        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, **kwargs)

        self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=0)
        self.val_metrics = metric_initializer(num_classes=num_classes, ignore_index=0)
        self.test_metrics = metric_initializer(num_classes=num_classes, ignore_index=0)

        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self(x) # out shape: (B, N, C)

        # print(x.shape, trans_feat.shape)
        # print(out.shape, y.shape)

        # out (B, N, C) to (B*N, C) ; y (B, N) to (B*N)
        out = out.reshape(-1, out.shape[-1])
        y = y.reshape(-1).to(torch.long)

        loss = self.criterion(out, y)

        # print(f"\nLoss: {loss}\n")

        preds = self.prediction(out)

        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y)
                    if isinstance(met, torch.Tensor):
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        return loss, preds, y 
    

    def _epoch_end_metric_logging(self, metrics, prefix, print_metrics=False):
        metric_res = metrics.compute()
        if print_metrics:
            print(f'{"="*10} {prefix} metrics {"="*10}')
        for metric_name, metric_val in metric_res.items():
            if print_metrics:
                # if metric is per class
                if isinstance(metric_val, torch.Tensor) and metric_val.ndim > 0: 
                    print(f'\t{prefix}_{metric_name}: {metric_val}; mean: {metric_val[1:].mean():.4f}')
                else:
                    print(f'\t{prefix}_{metric_name}: {metric_val:.4f}')

        metrics.reset()