





import torch
from core.lit_modules.lit_model_wrappers import LitWrapperModel
from pointcept.models.sparse_unet.spconv_unet_v1m1_base import SpUNetBase


class Lit_Sparse_UNet(LitWrapperModel):


    def __init__(self, 
                criterion, 
                optimizer_name: str, 
                in_channels:int,
                num_classes:int,
                base_channels: int = 32,
                channels = (32, 64, 128, 256, 256, 128, 96, 96),
                layers = (2, 3, 4, 6, 2, 2, 2, 2),
                cls_mode: bool = False,
                learning_rate=0.01, 
                metric_initializer=None, 
                **kwargs):
        

        model = SpUNetBase(in_channels=in_channels,
                            num_classes=num_classes,
                            base_channels=base_channels,
                            channels=channels,
                            layers=layers,
                            cls_mode=cls_mode)
        
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, num_classes=num_classes, **kwargs)


    def forward(self, x):
        return self.model(x)
    
    def prediction(self, model_output):
        # model_output shape = (batch_size, num_classes, num_points)
        return torch.argmax(model_output, dim=-1)
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        out = self(x)
        
        loss = self.criterion(out, y)
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