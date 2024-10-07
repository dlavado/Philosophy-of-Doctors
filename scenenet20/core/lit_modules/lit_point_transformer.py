
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import core.models.pointcept.pointcept.models.point_transformer_v2.point_transformer_v2m2_base as ptv2
import core.models.pointcept.pointcept.models.point_transformer.point_transformer_seg as pts
import core.models.pointcept.pointcept.models.point_transformer_v3.point_transformer_v3m1_base as ptv3
from core.models.pointcept.pointcept.models.utils import offset2batch, batch2offset
import torch
import torch.nn as nn
from core.lit_modules.lit_model_wrappers import LitWrapperModel


class Lit_PointTransformer(LitWrapperModel):


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
        elif version == 'v3':
            model = ptv3.PointTransformerV3(in_channels=in_channels, num_classes=num_classes,
                                            order=["z", "z-trans", "hilbert", "hilbert-trans"], enable_flash=False)
            
        else:
            model = pts.PointTransformerSeg50(in_channels=in_channels, num_classes=num_classes)

        self.version = version

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
    

    def process_input_tensor(self, inpt:torch.Tensor):
        """
        `inpt` - torch.Tensor with shape (B, N, 3 + F) 
            where B is batch size, N is number of points, F is feature dimension and 3 is for x, y, z coordinates

        Returns
        -------
        `coords` - torch.Tensor with shape (B*N, 3)

        `feat` - torch.Tensor with shape (B*N, F)

        `batch` - torch.Tensor with shape (B*N, 1)
        """

        coords = inpt[:, :, :3].contiguous()
        coords = coords.view(-1, 3)
        if inpt.shape[-1] > 3:
            feat = inpt[:, :, 3:].contiguous()
            # print(f"feat shape = {feat.shape}")
            feat = feat.view(-1, feat.shape[-1])
        else:
            # feat = torch.zeros((coords.shape[0], 1), device=coords.device)
            feat = coords

        # print(f"coords shape = {coords.shape}")
        # print(f"feat shape = {feat.shape}")

        batch = torch.cat([torch.full((inpt.shape[1],), i, device=inpt.device) for i in range(inpt.shape[0])], dim=0)

        return {"coord": coords.to(torch.float32),
                "feat": feat.to(torch.float32),
                "batch": batch,
                "offset": batch2offset(batch),
                "grid_size": torch.tensor([0.1, 0.1, 0.1], device=inpt.device)
            }
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch # x shape = (batch_size, num_points, in_channels), y shape = (batch_size, num_points)
        y = y.to(torch.long)

        out = self(self.process_input_tensor(x)) # out shape = (batch_size*num_points, num_classes)
    
        loss = self.criterion(out, y.reshape(-1))
        preds = self.prediction(out) # preds shape = (batch_size*num_points)

        # print(f"preds shape = {preds.shape}")
        # print(preds)
        
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
    


from core.models.GENEONets.GIBLi import SceneNet_PreBackbone

class Lit_PointTransformer_wSCENENet(Lit_PointTransformer):
    
    def __init__(self, 
                 criterion, 
                 in_channels, 
                 num_classes,
                 version='v2',
                 geneo_num = None,
                 num_observers = [1],
                 kernel_size = (9, 7, 7), 
                 **kwargs):
        
        scenenet = SceneNet_PreBackbone(geneo_num, num_observers, kernel_size)
        
        super().__init__(criterion, scenenet.feat_dim, num_classes, version=version, **kwargs)

        self.scenenet = scenenet


    def forward(self, x):
        return self.model(x)
    
    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y, pt_loc = batch # x shape = (batch_size, num_points, in_channels), y shape = (batch_size, num_points)
        y = y.to(torch.long)

        out = self(self.process_input_tensor(self.scenenet(x, pt_loc))) # out shape = (batch_size*num_points, num_classes)
    
        loss = self.criterion(out, y.reshape(-1))
        preds = self.prediction(out) # preds shape = (batch_size*num_points)

        # print(f"preds shape = {preds.shape}")
        # print(preds)
        
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