



import torch
from core.lit_modules.lit_model_wrappers import LitWrapperModel


from core.models.kpconv.examples.scene_segmentation.model import KPFCNN, GIBLi_KPFCNN, PreGIBLi_KPFCNN
from core.models.kpconv.easy_kpconv.ops.calibrate_neighbors import calibrate_neighbors_pack_mode
from core.models.kpconv.easy_kpconv.ops.conversion import batch_to_pack, pack_to_batch



class LitKPConv(LitWrapperModel):
    # https://github.com/qinzheng93/Easy-KPConv

    def __init__(self,
                 criterion: torch.nn.Module, 
                 optimizer_name: str, 
                 model_version: str="kpconv",
                 num_stages=5,
                 voxel_size=0.04,
                 kpconv_radius=2.5,
                 kpconv_sigma=2.0,
                 neighbor_limits=[24, 40, 34, 35, 34],
                 input_dim=5,
                 init_dim=64,
                 kernel_size=15,
                 num_classes=10,
                 learning_rate=0.001, 
                 ignore_index=-1,
                 metric_initializer=None, 
                 **kwargs
            ):
        
        if neighbor_limits is None:
            neighbor_limits = calibrate_neighbors_pack_mode() #TODO
        
        if 'pre' in model_version:
            kpconv_params = {
                "num_stages": num_stages,
                "voxel_size": voxel_size,
                "kpconv_radius": kpconv_radius,
                "kpconv_sigma": kpconv_sigma,
                "neighbor_limits": neighbor_limits,
                # "input_dim": input_dim,
                "init_dim": init_dim,
                "kernel_size": kernel_size,
                # "num_classes": num_classes,
            }
            model = PreGIBLi_KPFCNN(
                        num_classes,
                        input_dim,
                        kpconv_params,
                        kwargs['gibli_params']    
                    )
            
        elif model_version == "kpconv":

            model = KPFCNN(
                        num_stages,
                        voxel_size,
                        kpconv_radius,
                        kpconv_sigma,
                        neighbor_limits,
                        input_dim,
                        init_dim,   
                        kernel_size,
                        num_classes,
                    )
        elif model_version == "kpconv-gibli":
            model = GIBLi_KPFCNN(
                        num_stages,
                        voxel_size,
                        kpconv_radius,
                        kpconv_sigma,
                        neighbor_limits,
                        input_dim,
                        init_dim,   
                        kernel_size,
                        num_classes,
                        **kwargs['gibli_params']
                    )
        else:
            raise ValueError(f"Unknown model version: {model_version}")
            
        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, **kwargs)


        self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
        self.val_metrics = metric_initializer(num_classes=num_classes)
        self.test_metrics = metric_initializer(num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x:torch.Tensor):
        points, lengths = batch_to_pack(x)
        if points.shape[-1] > 3:
            points, feats = points[:, :3], points
        else:
            feats = points

        return self.model(points, feats, lengths)
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=-1)
    
    def forward_model_output(self, x:torch.Tensor) -> torch.Tensor:
        # runs the model and returns the model output in (B, N, C) format
        points, lengths = batch_to_pack(x)
        if x.shape[-1] > 3:
            points, feats = points[:, :3], points
        else:
            feats = points

        model_dict = self.model(points, feats, lengths)

        return pack_to_batch(model_dict["scores"], lengths)[0]
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch

        out_dict = self(x)

        scores = out_dict["scores"]
        # flat targets
        y = y.reshape(-1).to(torch.long)
        
        # print("scores: ", scores.shape, "targets: ", y.shape)
        # print(torch.max(scores), torch.min(scores))

        loss = self.criterion(scores, y)
        preds = self.prediction(scores)

        # print(f"{loss=}")
        # print(f"{preds.shape=}")
        # print(f"{torch.unique(y)=}")
        # print(f"{torch.unique(preds)=}")

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

