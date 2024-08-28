



import torch
import torch.nn as nn
from core.lit_modules.lit_model_wrappers import LitWrapperModel
from pointcept.models.context_aware_classifier.context_aware_classifier_v1m1_base import CACSegmentor




class Lit_Context_Aware_Classifier(LitWrapperModel):

    def __init__(self, 
                 backbone:nn.Module,
                 num_classes:int,
                 backcbone_out_channels:int,
                 criterion, 
                 optimizer_name: str, 
                 learning_rate=0.01, 
                 metric_initializer=None, 
                 **kwargs):
        
        
        model = CACSegmentor(num_classes=num_classes,
                             backbone=backbone,
                             backbone_out_channels=backcbone_out_channels
                            )

        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, num_classes=num_classes)


    def forward(self, x, stage):
        return self.model(x, stage=stage)
    
    def prediction(self, model_output) -> torch.Tensor:
        return torch.argmax(model_output["seg_logits"], dim=-1)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        pred = self(batch, stage="test")
        pred = self.prediction(pred)
        return pred

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch
        print(x.shape, y.shape)
        forward_dict = self(batch, stage=stage)

        loss = forward_dict["loss"]
        preds = self.prediction(forward_dict)
        
        if stage:
            on_step = stage == "train" 
            for key in forward_dict:
                if "loss" in key:
                    self.log(f"{stage}_{key}", forward_dict[key], on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, batch[1])
                    if isinstance(met, torch.Tensor):
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)

        return loss, preds, batch[1]  



    