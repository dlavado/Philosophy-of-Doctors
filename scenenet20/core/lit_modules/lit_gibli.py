
import torch
import sys

from torch.optim.optimizer import Optimizer
sys.path.insert(0, '..')
sys.path.insert(1, '../..')


from core.lit_modules.lit_model_wrappers import LitWrapperModel
from core.models.giblinet.GIBLi import GIBLiNet
from core.criterions.elastic_net_reg import ElasticNetRegularization




class LitGIBLi(LitWrapperModel):
    """
    Pytorch Lightning Model Wrapper for GIBLi models.
    """
    
    def __init__(self, 
                ###### GIBLi parameters ######
                in_channels, 
                num_classes, 
                num_levels, 
                out_gib_channels, 
                num_observers, 
                kernel_size, 
                gib_dict, 
                skip_connections,
                pyramid_builder,
                ###### LitWrapperModel parameters ######
                criterion, 
                optimizer_name:str=None, 
                learning_rate:float=1e-2, 
                metric_initializer=None, 
                ignore_index=-1,
                **kwargs
            ) -> None:
          
        
        model = GIBLiNet(in_channels, 
                    num_classes, 
                    num_levels, 
                    out_gib_channels, 
                    num_observers, 
                    kernel_size, 
                    gib_dict, 
                    skip_connections,
                    pyramid_builder
                )
        
        super().__init__(model, criterion, optimizer_name, learning_rate, None, **kwargs)
        
        if metric_initializer is not None:
            self.train_metrics = metric_initializer(num_classes=num_classes, ignore_index=ignore_index)
            self.val_metrics = metric_initializer(num_classes=num_classes)
            self.test_metrics = metric_initializer(num_classes=num_classes)
            
        self.elastic_reg = ElasticNetRegularization(alpha=0.001, l1_ratio=0.5)
            
            
    def prediction(self, model_output):
        # model_output shape = (batch_size, num_points, classes)
        return torch.argmax(model_output, dim=-1)
    
    
    def elastic_loss(self):
        return self.elastic_reg(self.model.get_cvx_coefficients())
    
    
    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        torch.autograd.set_detect_anomaly(True)

        if isinstance(batch, dict):
            x = batch["coords"]
            y = batch["sem_labels"]
            graph_pyramid = batch["graph_pyramid"]
        else:
            x, y = batch
            graph_pyramid = None
            
        # total_mem = 0
        # if graph_pyramid is not None:
        #     for _, val in graph_pyramid.items():
        #         total_mem += sum([self.tensor_memory_in_mb(t) for t in val])
                
        
        # print(f"Memory occupied by input: {self.tensor_memory_in_mb(x) + self.tensor_memory_in_mb(y)} MB")
        # print(f"Memory occupied by graph pyramid: {total_mem} MB")
        # print(f"Memory before backward: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
        
        # print(torch.cuda.memory_stats())
        
        logits = self.model(x, graph_pyramid)
        
        if torch.isnan(logits).any():
            print("Nan in logits")
            print(logits)
            print(x)
            print(y)
            print(graph_pyramid)
            raise ValueError("Nan in logits")
        
        del graph_pyramid
        torch.cuda.empty_cache()
        
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.to(torch.long).reshape(-1)
        
        # print(f"{logits.shape=}, {y.shape=}")
        
        loss = self.criterion(logits, y) + self.elastic_loss()
        
        assert not torch.isnan(loss), f"Loss is NaN df:{self.criterion(logits, y)} and elastic:{self.elastic_loss()}"
        preds = self.prediction(logits)
        
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
    
    def on_after_backward(self) -> None:
                
        # for name, param in self.model.named_parameters():
        #     if 'gib_params' in name or 'lambdas' in name:
        #         print(f"{name=},  {param=},  {param.grad=}")
        
        # print(f"Memory after backward: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
        super().on_after_backward()
    
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        
        # print(f"Memory after optimizer step: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")

        optimizer.step(closure=optimizer_closure) # update the model parameters
        
        # run logic right after the optimizer step
        with torch.no_grad():
            self.model.maintain_convexity()
            
        optimizer.zero_grad()
            

 