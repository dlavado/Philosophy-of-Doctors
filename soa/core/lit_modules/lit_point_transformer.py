
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
                ignore_index=-1,
                **kwargs):
        
        
        if 'pre' in version:
            
            if 'v3' in version:
                model = ptv3.PreGIBLiPointTransformerV3(in_channels=in_channels, 
                                                        num_classes=num_classes, 
                                                        enable_flash=False,
                                                        order=["z", "z-trans", "hilbert", "hilbert-trans"],
                                                        giblinet_params=kwargs['gib_params'])
                
                
            elif 'v2' in version:
                model = ptv2.PreGIBLiPointTransformerV2(in_channels=in_channels, 
                                                        num_classes=num_classes,
                                                        grid_sizes=(0.05, 0.10, 0.20, 0.40),
                                                        giblinet_params=kwargs['gib_params'])
                
            else:
                model = pts.PreGIBLiPointTransformerSeg50(in_channels=in_channels, 
                                                        num_classes=num_classes, 
                                                        giblinet_params=kwargs['gib_params'])
        
        elif 'gibli' in version:
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
            else:
                model = pts.PointTransformerSeg50(in_channels=in_channels, num_classes=num_classes)

        self.version = version

        super().__init__(model, criterion, optimizer_name, learning_rate, None, num_classes=num_classes, **kwargs)

        
        if metric_initializer is not None:
            self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.val_metrics = metric_initializer(num_classes=num_classes)
            self.test_metrics = metric_initializer(num_classes=num_classes)



    def forward(self, x):
        x_dict = self.process_input_tensor(x)
        # for k, v in x_dict.items():
        #     print(f"{k} shape = {v.shape}")
        return self.model(x_dict)
    
    def prediction(self, model_output):
        # model_output shape = (batch_size*num_points, classes)
        return torch.argmax(model_output, dim=-1)
    
    def forward_model_output(self, x:torch.Tensor) -> torch.Tensor:
        # run the model and return the model output in (B, N, C) format
        inpt_dict = self.process_input_tensor(x)

        out = self.model(inpt_dict) # out shape = (batch_size*num_points, num_classes)

        batch = inpt_dict["batch"] # batch shape = (batch_size*num_points, 1)
        batch_size = torch.max(batch).item() + 1
        
        out = out.view(batch_size, -1, out.shape[-1])
        return out
    

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

        coords = inpt[..., :3].contiguous()
        coords = coords.view(-1, 3)
        if inpt.shape[-1] > 3:
            # feat = inpt[:, :, 3:].contiguous()
            feat = inpt.contiguous() # coords included in the features
            # print(f"feat shape = {feat.shape}")
            feat = feat.view(-1, feat.shape[-1])
        else:
            feat = coords

        # print(f"coords shape = {coords.shape}")
        # print(f"feat shape = {feat.shape}")   

        # batch e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, ...] with shape (B*N, 1)
        batch = torch.cat([torch.full((inpt.shape[1],), i, device=inpt.device) for i in range(inpt.shape[0])], dim=0)
        # batch = torch.arange(inpt.shape[0], device=inpt.device).repeat_interleave(inpt.shape[1])

        return {"coord": coords.to(torch.float32),
                "feat": feat.to(torch.float32),
                "batch": batch,
                "offset": batch2offset(batch),
                "grid_size": torch.tensor([0.05, 0.05, 0.05], device=inpt.device)
            }
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch # x shape = (batch_size, num_points, in_channels), y shape = (batch_size, num_points)
        y = y.to(torch.long)

        # print(f"x shape = {x.shape}")
        # print(f"y shape = {y.shape}")

        out = self(x) # out shape = (batch_size*num_points, num_classes)

        # print(f"out shape = {out.shape}, y shape = {y.shape}")

        # print(torch.max(out), torch.min(out))

        # check if out has nan
        if torch.isnan(out).any():
            ValueError("out has nan")
    
        loss = self.criterion(out, y.reshape(-1))
        preds = self.prediction(out) # preds shape = (batch_size*num_points)
        
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