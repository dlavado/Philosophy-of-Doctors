
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pointcept.models.point_transformer_v2.point_transformer_v2m2_base as ptv2
import pointcept.models.point_transformer.point_transformer_seg as pts
import torch
import torch.nn as nn
from core.lit_modules.lit_model_wrappers import LitWrapperModel


class Lit_PointTransformerV2(LitWrapperModel):


    def __init__(self,
                criterion, 
                in_channels,
                num_classes,
                version:str='v2',
                optimizer_name: str='adam', 
                learning_rate=0.01,
                metric_initializer=None, 
                **kwargs):
    

        if version == 'v2':
            model = ptv2.PointTransformerV2(in_channels=in_channels, num_classes=num_classes)
        else:
            model = pts.PointTransformerSeg50(in_channels=in_channels, num_classes=num_classes)

        super().__init__(model, criterion, optimizer_name, learning_rate, None, num_classes=num_classes, **kwargs)

        ig_index = 0
        
        if metric_initializer is not None:
            self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ig_index)
            self.val_metrics = metric_initializer(num_classes=num_classes, ignore_index=ig_index)
            self.test_metrics = metric_initializer(num_classes=num_classes, ignore_index=ig_index)



    def forward(self, x):
        return self.model(x)
    
    def prediction(self, model_output):
        # model_output shape = (batch_size, num_classes, num_points)
        return torch.argmax(model_output, dim=-1)
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch # x shape = (batch_size, num_points, in_channels), y shape = (batch_size, num_points)
        x = x.to(torch.float16)
        out = self(x) # out shape = (batch_size*num_points, num_classes)

        print(f"out shape = {out.shape}")
        print(out)
    
        loss = self.criterion(out, y.reshape(-1))
        preds = self.prediction(out) # preds shape = (batch_size*num_points)

        print(f"preds shape = {preds.shape}")
        print(preds)
        
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
    
    

    def on_train_batch_end(self, batch, batch_idx, dataloader_idx):
        torch.cuda.empty_cache()
        # release memory
        return super().on_train_batch_end(batch, batch_idx, dataloader_idx)