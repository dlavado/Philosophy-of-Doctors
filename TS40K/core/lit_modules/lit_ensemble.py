

import torch
import torch.nn as nn
from torch.nn.modules import Module
from core.lit_modules.lit_model_wrappers import LitWrapperModel

from typing import List

########### coined from PTV2 ############

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError
############################################


class Lit_EnsembleModel(LitWrapperModel):

    def __init__(self, 
                 models:List[torch.nn.Module],
                 criterion: Module,
                 num_classes,
                 use_small_net:bool=True,
                 full_train:bool=False, 
                 optimizer_name: str='adam', 
                 learning_rate=0.01,
                 ignore_index=-1, 
                 metric_initializer=None, 
                 **kwargs):
        
        super().__init__(None, criterion, optimizer_name, learning_rate, metric_initializer, **kwargs)
        
        self.soa = models

        if use_small_net:
            dec_channels = len(self.soa) * num_classes
            self.seg_head = (
                nn.Sequential(
                    nn.Linear(dec_channels, dec_channels),
                    PointBatchNorm(dec_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(dec_channels, num_classes),
                )
            )
        else:
            self.seg_head = None

        if not full_train:
            for model in self.soa:
                for param in model.parameters():
                    param.requires_grad = False
        
        # self.model is a list of SoA models inclunding the seg_head so that we can we can retrieve all params for optimizer
        self.model = nn.Sequential(*self.soa, self.seg_head) 

        if metric_initializer is not None:
            self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.val_metrics = metric_initializer(num_classes=num_classes)
            self.test_metrics = metric_initializer(num_classes=num_classes)



    def forward(self, x):
        # # Initialize CUDA streams for parallel execution
        # streams = [torch.cuda.Stream() for _ in range(len(self.soa))]
        # logits = [None] * len(self.soa)

        # # Run each model asynchronously in its own stream
        # for i, pretrained in enumerate(self.soa):
        #     with torch.cuda.stream(streams[i]):
        #         logits[i] = pretrained.forward_model_output(x)  # Each model returns its logits

        # # Wait for all streams to complete
        # torch.cuda.synchronize()
        
        logits = [pretrained.forward_model_output(x) for pretrained in self.soa]  # Each model returns its logits
        # for i, logit in enumerate(logits):
        #     print(f"Model {i} logits shape: {logit.shape}")

        combined_logits = torch.stack(logits, dim=2) # shape = (B, num_points, num_models, num_classes)
        
        if self.seg_head:
            batch_size, num_points, num_models, num_classes = combined_logits.shape
            combined_logits = combined_logits.view(batch_size*num_points, num_models*num_classes) # shape = (B*num_points, num_models, num_classes)
            output = self.seg_head(combined_logits) # shape = (B*num_points, num_classes)
            output = output.view(batch_size, num_points, num_classes) # shape = (B, num_points, num_classes)
        else: # just average the logits
            output = torch.mean(combined_logits, dim=-2) # shape = (B, num_points, num_classes)
        
        return output # shape = (B, num_points, num_classes)

    def prediction(self, model_output):
        return torch.argmax(model_output, dim=-1)
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch # x shape = (batch_size, num_points, in_channels), y shape = (batch_size, num_points)
        y = y.to(torch.long)

        out = self(x)

        out = out.reshape(-1, out.shape[-1])
        loss = self.criterion(out, y.reshape(-1))
        preds = self.prediction(out)
        
        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y.reshape(-1))
                    if isinstance(met, torch.Tensor):
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        return loss, preds, y  
    

    
