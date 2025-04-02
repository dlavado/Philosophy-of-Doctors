
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import core.models.pointcept.pointcept.models.point_transformer_v2.point_transformer_v2m2_base as ptv2
import core.models.pointcept.pointcept.models.point_transformer.point_transformer_seg as pts
import core.models.pointcept.pointcept.models.point_transformer_v3.point_transformer_v3m1_base as ptv3
from core.models.pointcept.pointcept.models.utils import offset2batch, batch2offset
import torch
import torch.nn as nn
from core.lit_modules.lit_model_wrappers import LitWrapperModel

from core.models.giblinet.GIBLi_SOTA import GIBLiNetPTV1, GIBLiNetPTV2, GIBLiNetPTV3


class Lit_PointTransformer(LitWrapperModel):


    def __init__(self,
                criterion, 
                in_channels,
                num_classes,
                version:str='v2',
                optimizer_name: str='adam', 
                learning_rate=0.01,
                metric_initializer=None, 
                ignore_index=-1,
                **kwargs):
        
        version = version.lower()
        
        if 'gibli' in version:
            if 'v1' in version: # gibli v1
                model = pts.GIBLiPointTransformerSeg50(in_channels=in_channels, 
                                                       num_classes=num_classes, 
                                                       **kwargs['gib_params'])
            elif 'v2' in version: # gibli v2
                model = ptv2.GIBLiPointTransformerV2(in_channels=in_channels,
                                                     num_classes=num_classes,
                                                     grid_sizes=(0.05, 0.10, 0.20, 0.40),
                                                     **kwargs['gib_params']) 
            elif 'v3' in version: # gibli v3
                model = ptv3.GIBLiPointTransformerV3(in_channels=in_channels, num_classes=num_classes,
                                                      order=["z", "z-trans", "hilbert", "hilbert-trans"], enable_flash=False,
                                                      **kwargs['gib_params'])
            else:
                ValueError("Invalid version")
        else:
            if version == 'v2':
                model = ptv2.PointTransformerV2(in_channels=in_channels, 
                                                num_classes=num_classes,
                                                # patch_embed_channels=256,
                                                # patch_embed_groups=8,
                                                # patch_embed_depth=2,
                                            )
            elif version == 'v3':
                model = ptv3.PointTransformerV3(in_channels=in_channels, num_classes=num_classes,
                                                order=["z", "z-trans", "hilbert", "hilbert-trans"], enable_flash=False)     
            elif version == 'v1':
                model = pts.PointTransformerSeg50(in_channels=in_channels, num_classes=num_classes)
            else:
                ValueError("Invalid version")

        self.version = version

        super().__init__(model, criterion, optimizer_name, learning_rate, None, num_classes=num_classes, **kwargs)
 
        if metric_initializer is not None:
            self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.val_metrics = metric_initializer(num_classes=num_classes)
            self.test_metrics = metric_initializer(num_classes=num_classes)



    def forward(self, data_dict:dict) -> torch.Tensor:
        x_dict = self.process_input_tensor(data_dict)
        # for k, v in x_dict.items():
        #     print(f"{k} shape = {v.shape}")
        return self.model(x_dict) # out shape = (batch_size*num_points, num_classes)
    
    def prediction(self, model_output):
        # model_output shape = (batch_size*num_points, classes)
        return torch.argmax(model_output, dim=-1)
    
    def process_input_tensor(self, data_dict:dict) -> dict:
        data_dict['batch'] = offset2batch(data_dict['offset'])
        data_dict['grid_size'] = torch.tensor([0.02, 0.02, 0.02], device=data_dict['coord'].device)
        return data_dict

    def evaluate(self, data_dict, stage=None, metric=None, prog_bar=True, logger=True):

        out = self(data_dict) # out shape = (batch_size*num_points, num_classes)
        y = data_dict['segment'].to(torch.long)

        # print(f"out shape = {out.shape}, y shape = {y.shape}")

        if torch.isnan(out).any():
            ValueError("out has nan")
    
        loss = self.criterion(out, y)
        preds = self.prediction(out) # preds shape = (batch_size*num_points)
        
        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    metric_val = metric_val
                    met = metric_val(preds, y.reshape(-1))
                    if met.numel() > 1: 
                        if stage == 'val':   
                            for i, m in enumerate(met.tolist()):
                                self.log(f"class_{i}_{metric_name}", m, on_epoch=True, on_step=False, prog_bar=False, logger=logger)
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=logger)

        return loss, preds, y
    

    # def on_train_batch_end(self, batch, batch_idx, dataloader_idx):
    #     torch.cuda.empty_cache()
    #     # release memory
    #     return super().on_train_batch_end(batch, batch_idx, dataloader_idx)
    
    
    def on_after_backward(self):
        for name, param in self.model.named_parameters():
            # if 'gib_params' in name and 'disk' in name:
            if param.grad is not None:
                print(f"{name},  grad.norm={param.grad.norm().item():.2e}")
            else:
                print(f"{name} has .grad to None")
        
        # print(f"Memory after backward: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
        # print(torch.cuda.memory_summary())
        return