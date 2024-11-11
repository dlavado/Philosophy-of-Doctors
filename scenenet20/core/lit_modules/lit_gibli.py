
import torch
import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')


from core.lit_modules.lit_model_wrappers import LitWrapperModel
from core.models.giblinet.GIBLi import GIBLiNet
from core.criterions.elastic_net_reg import ElasticNetRegularization
from core.criterions.geneo_loss import GENEORegularizer


from core.neighboring.conversions import pack_to_batch, _pack_tensor_to_batch

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
        self.geneo_reg = GENEORegularizer(rho=0.01)
            
            
    def prediction(self, model_output):
        # model_output shape = (batch_size, num_points, classes)
        return torch.argmax(model_output, dim=-1)
    
    
    def elastic_loss(self):
        return self.elastic_reg(self.model.get_cvx_coefficients())
    
    def geneo_loss(self):
        return self.geneo_reg(self.model.get_gib_parameters(), self.model.get_cvx_coefficients())
    
    
    def evaluate(self, batch, stage=None, metric=None, prog_bar=True, logger=True):
        torch.autograd.set_detect_anomaly(True)

        if isinstance(batch, dict):
            x = batch["pointcloud"]
            y = batch["sem_labels"]
            lengths = batch["lengths"]
            graph_pyramid = batch["graph_pyramid"]
            pyramid_lengths = batch["lengths_graph_pyramid"]    
        else:
            x, y = batch
            graph_pyramid = None
        
        x, _ = pack_to_batch(x, lengths)  
        pyramid = {}
        for key, val in graph_pyramid.items():
            # print(f"{key=}...", end="")
            # print(f"{len(val)=}")
            # print("pack mode...")
            # for t in val:
            #     print(f"{t.shape=}")  
            pyramid[key] = [_pack_tensor_to_batch(t, l) for t, l in zip(val, pyramid_lengths[key])]
            # pyramid[key] = [pack_to_batch(t, l)[0] for t, l in zip(val, pyramid_lengths[key])]
            # print("batched mode...")
            # for t in pyramid[key]:
            #     print(f"{t.shape=}")
        
        # for key, val in graph_pyramid.items():
        #     print(f"{key=}...", end="")
        #     for t in val:
        #        print(f"{t.shape=}", end="")
        #     print()
            
        # input("Press Enter to continue...")
        
        
        logits = self.model(x, pyramid)
        
        del graph_pyramid
        torch.cuda.empty_cache()
        
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.to(torch.long).reshape(-1)
                
        # print(f"{logits.shape=}, {y.shape=}")
        elastic_loss = self.elastic_loss()
        geneo_loss = self.geneo_loss()
        data_fidelity_loss = self.criterion(logits, y)
        
        loss = data_fidelity_loss + elastic_loss + geneo_loss
        preds = self.prediction(logits)
        
        if stage:
            on_step = stage == "train" 
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_data_fidelity_loss", data_fidelity_loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_elastic_loss", elastic_loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)
            self.log(f"{stage}_geneo_loss", geneo_loss, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=logger)

            if metric:
                for metric_name, metric_val in metric.items():
                    met = metric_val(preds, y.reshape(-1))
                    if isinstance(met, torch.Tensor):
                        met = met.mean()
                    self.log(f"{stage}_{metric_name}", met, on_epoch=True, on_step=on_step, prog_bar=True, logger=True)
                    
                    
        return loss, preds, y
    
    def on_after_backward(self) -> None:
                
        # for name, param in self.model.named_parameters():
        #     if 'lambdas' in name or 'gib_params' in name:
        #         print(f"{name=},  {param=},  {param.grad=}")
        
        # print(f"Memory after backward: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")
        super().on_after_backward()
    
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        
        # print(f"Memory after optimizer step: {torch.cuda.memory_allocated() / (1024 ** 2)} MB")

        optimizer.step(closure=optimizer_closure) # update the model parameters
        
        # run logic right after the optimizer step
        # with torch.no_grad():
        #     self.model.maintain_convexity()
            
        optimizer.zero_grad()
        
        
    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        
    #     lambda_params = [name for name, _ in self.named_parameters() if 'lambda' in name]

    #     # Accumulate gradients for 'lambda' parameters for 9 steps
    #     if (batch_idx + 1) % 100 != 0:
    #         # Zero out gradients for non-lambda parameters
    #         for name, param in self.named_parameters():
    #             if name not in lambda_params and param.grad is not None:
    #                 param.grad.zero_()
    #         # Execute the optimizer closure without stepping
    #         optimizer_closure()
        
    #     else:
    #         # Perform a regular optimizer step for all parameters
    #         optimizer.step(closure=optimizer_closure)
            
    #         with torch.no_grad():
    #             self.model.maintain_convexity()

    #         # Zero out all gradients after updating
    #         optimizer.zero_grad()
            

 