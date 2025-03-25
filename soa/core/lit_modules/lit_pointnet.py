


from typing import Tuple
import torch
from core.lit_modules.lit_model_wrappers import LitWrapperModel

import core.models.pointnet.models.pointnet2_sem_seg as pointnet2
import core.models.pointnet.models.pointnet_sem_seg as pointnet



class LitPointNet(LitWrapperModel):
    # https://github.com/yanx27/Pointnet_Pointnet2_pytorch


    def __init__(self, 
                 model:str,
                 criterion:torch.nn.Module,
                 optimizer_name: str, 
                 num_classes = 10,
                 num_channels=3,
                 learning_rate=0.01, 
                 ignore_index=-1,
                 metric_initializer=None, 
                 **kwargs):
        
        if 'pointnet2' in model:
            if criterion is None:
                criterion = pointnet2.get_loss()
            
            if 'pre' in model:
                model = pointnet2.get_pre_gibli_model(in_channels=3 + num_channels, num_classes=num_classes, **kwargs['gibli_params'])
            elif 'gibli' in model:
                model = pointnet2.get_gibli_model(num_classes, num_channels=3 + num_channels, gibli_params=kwargs['gibli_params'])
            else:
                model = pointnet2.get_model(num_classes, num_channels=num_channels)
        elif 'pointnet' in model:
            if criterion is None:
                criterion = pointnet.get_loss()
                
            if 'pre' in model:
                model = pointnet.get_pre_gibli_model(in_channels=3 + num_channels, num_classes=num_classes, **kwargs['gibli_params'])
            elif 'gibli' in model:
                model = pointnet.get_gibli_model(num_classes, num_channels=3 + num_channels, gibli_params=kwargs['gibli_params'])
            else:
                model = pointnet.get_model(num_classes, num_channels=num_channels)
        else:
            raise ValueError(f"Unknown model {model}")
        
        super().__init__(model, criterion, optimizer_name, learning_rate, None)

        self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
        self.val_metrics = metric_initializer(num_classes=num_classes)
        self.test_metrics = metric_initializer(num_classes=num_classes)

        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)
    
    def forward_model_output(self, x:torch.Tensor) -> torch.Tensor:
        # run the model and return the model output in (B, N, C) format
        return self.model(x)[0]
    
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)
    
    
    def process_batch(self, batch):
        from core.models.giblinet.conversions import build_batch_tensor
        if isinstance(batch, dict):
            x = batch["coord"]
            y = batch["segment"]
            offset = batch["offset"]
            feat = batch["feat"]
        else:
            x, y = batch
            
        x = torch.cat([x, feat], dim=-1)
        x = build_batch_tensor(x, offset)[0]
        y = build_batch_tensor(y, offset)[0]    
        return x, y
    

    def evaluate(self, data_dict, stage=None, metric=None, prog_bar=True, logger=True):
        
        x, y = self.process_batch(data_dict)
        # print(x.shape, y.shape)
        out, trans_feat = self(x)

        # print(x.shape, trans_feat.shape)
        # print(out.shape, y.shape)

        # out (B, N, C) to (B*N, C) ; y (B, N) to (B*N)
        out = out.reshape(-1, out.shape[-1])
        y = y.reshape(-1).to(torch.long)

    
        # loss = self.criterion(out, y, trans_feat)
        loss = self.criterion(out, y)

        # print(f"\nLoss: {loss}\n")

        preds = self.prediction(out)

        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y)
                    if met.numel() > 1: 
                        if stage == 'val':   
                            for i, m in enumerate(met.tolist()):
                                self.log(f"class_{i}_{metric_name}", m, on_epoch=True, on_step=False, prog_bar=False, logger=logger)
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=logger)

        return loss, preds, y 
    

    def _epoch_end_metric_logging(self, metrics, prefix, print_metrics=False):
        metric_res = metrics.compute()
        if print_metrics:
            print(f'{"="*10} {prefix} metrics {"="*10}')
        for metric_name, metric_val in metric_res.items():
            if print_metrics:
                # if metric is per class
                if isinstance(metric_val, torch.Tensor) and metric_val.ndim > 0: 
                    print(f'\t{prefix}_{metric_name}: {metric_val}; mean: {metric_val.mean():.4f}')
                else:
                    print(f'\t{prefix}_{metric_name}: {metric_val:.4f}')

        metrics.reset()
    

    
