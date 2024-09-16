



import torch
from torch.nn.modules import Module
from core.lit_modules.lit_model_wrappers import LitWrapperModel


from core.models.kpconv.examples.scene_segmentation.model import KPFCNN
from core.models.kpconv.easy_kpconv.ops.calibrate_neighbors import calibrate_neighbors_pack_mode
from core.models.kpconv.easy_kpconv.ops.conversion import batch_to_pack, pack_to_batch



class LitKPConv(LitWrapperModel):
    # https://github.com/qinzheng93/Easy-KPConv

    def __init__(self, 
                 criterion: torch.nn.Module, 
                 optimizer_name: str, 
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
                 metric_initializer=None, 
                 **kwargs
            ):
        
        if neighbor_limits is None:
            neighbor_limits = calibrate_neighbors_pack_mode() #TODO

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

        super().__init__(model, criterion, optimizer_name, learning_rate, metric_initializer, **kwargs)


        self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=0)
        self.val_metrics = metric_initializer(num_classes=num_classes, ignore_index=0)
        self.test_metrics = metric_initializer(num_classes=num_classes)

        self.save_hyperparameters()



    def forward(self, x:torch.Tensor):
        points, lengths = batch_to_pack(x)
        if x.shape[-1] > 3:
            feats = points[:, 3:]
        else:
            feats = torch.ones_like(points[:, :2])

        return self.model(points, feats, lengths)
    
    def prediction(self, model_output:torch.Tensor) -> torch.Tensor:
        return torch.argmax(model_output, dim=1)
    
    def forward_model_output(self, x:torch.Tensor) -> torch.Tensor:
        # runs the model and returns the model output in (B, N, C) format
        points, lengths = batch_to_pack(x)
        if x.shape[-1] > 3:
            feats = points[:, 3:]
        else:
            feats = torch.ones_like(points[:, :2])

        model_dict = self.model(points, feats, lengths)

        return pack_to_batch(model_dict["scores"], lengths)[0]
    

    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y = batch

        out_dict = self(x)

        scores = out_dict["scores"]
        # flat targets
        y = y.reshape(-1).to(torch.long)
        # print("scores: ", scores.shape, "targets: ", batch_to_pack(y)[0].shape)

        loss = self.criterion(scores, y)
        preds = self.prediction(scores)

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


from core.models.GENEONets.SCENE_Net import SceneNet_PreBackbone

class LitKPConv_wSCENENet(LitKPConv):
    
    def __init__(self, 
                 criterion: Module, 
                 optimizer_name: str, 
                 num_stages=5, 
                 voxel_size=0.04, 
                 kpconv_radius=2.5, 
                 kpconv_sigma=2, 
                 neighbor_limits=[24, 40, 34, 35, 34], 
                 input_dim=5, 
                 init_dim=64, 
                 kernel_size=15, 
                 num_classes=10, 
                 learning_rate=0.001,
                 geneo_num=None,
                 num_observers=[1], 
                 metric_initializer=None, 
                 **kwargs):
        
        scenenet = SceneNet_PreBackbone(geneo_num, num_observers, (9, 7, 7))

        super().__init__(criterion, optimizer_name, num_stages, voxel_size, kpconv_radius, kpconv_sigma, neighbor_limits, scenenet.feat_dim, init_dim, kernel_size, num_classes, learning_rate, metric_initializer, **kwargs)

        self.scenenet = scenenet


    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        x, y, pt_locs = batch

        x = self.scenenet(x, pt_locs) # x shape = (batch_size, num_points, feat_dim)

        points, lengths = batch_to_pack(x) 
        points, feats = points[:, :3], points[:, 3:] # shape = (N, 3), (N, feat_dim - 3)

        # print(points.shape, feats.shape, lengths.shape)
        
        out_dict = self(points, feats, lengths)

        # print(out_dict.keys())

        scores = out_dict["scores"]
        # flat targets
        y = y.reshape(-1).to(torch.long)
        # print("scores: ", scores.shape, "targets: ", batch_to_pack(y)[0].shape)

        loss = self.criterion(scores, y)
        preds = self.prediction(scores)

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