from typing import Any, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


import sys
sys.path.insert(0, '..')
sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')
import utils.voxelization as Vox
from core.models.SCENE_Net import SceneNet
from core.criterions.aug_Lag_loss import Augmented_Lagrangian_Loss
from utils.scripts_utils import ParseKwargs, pointcloud_to_wandb



class LitWrapperModel(pl.LightningModule):
    """
    Generic Pytorch Lightning wrapper for Pytorch models that defines the logic for training, validation,testing and prediciton. 
    It also defines the logic for logging metrics and losses.    
    
    Parameters
    ----------

    `model` - torch.nn.Module:
        The model to be wrapped.
    
    `criterion` - torch.nn.Module:
        The loss function to be used

    `optimizer` - str:
        The Pytorch optimizer to be used for training.
        Note: str must be \in {'Adam', 'SGD', 'RMSprop'}

    `metric_initilizer` - function:
        A function that returns a TorchMetric object. The metric object must have a reset() and update() method.
        The reset() method is called at the end of each epoch and the update() method is called at the end of each step.
    """

    def __init__(self, model:torch.nn.Module, criterion:torch.nn.Module, optimizer_name:str, learning_rate=1e-2, metric_initializer=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        if metric_initializer is not None:
            self.train_metrics = metric_initializer()
            self.val_metrics = metric_initializer()
            self.test_metrics = metric_initializer()
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
    
        self.save_hyperparameters('optimizer_name', 'learning_rate')

    def forward(self, x):
        return self.model(x)                     

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.train_metrics is not None: # on step metric logging
            self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update()

        self.log(f'train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return pred, loss
    
    def training_epoch_end(self, outputs) -> None:
        if self.train_metrics is not None:
            self._epoch_end_metric_logging(self.train_metrics, 'train', print_metrics=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.val_metrics is not None:
            self.val_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'val_loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return pred, loss
    
   
    def validation_epoch_end(self, outputs) -> None: 
        if self.val_metrics is not None: # On epoch metric logging
           self._epoch_end_metric_logging(self.val_metrics, 'val')

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)

        if self.test_metrics is not None:
            self.test_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'test_loss', loss, logger=True)
        return pred, loss
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, _ = batch
        pred = self(x)

        return pred
    
    def test_epoch_end(self, outputs) -> None:
        if self.test_metrics is not None: # On epoch metric logging
            self._epoch_end_metric_logging(self.test_metrics, 'test')

    def get_model(self):
        return self.model
    
    def set_criteria(self, criterion):
        self.criterion = criterion
    
    
    def _epoch_end_metric_logging(self, metrics, prefix, print_metrics=False):
        metric_res = metrics.compute()
        if print_metrics:
            print(f'{"="*10} {prefix} metrics {"="*10}')
        for metric_name, metric_val in metric_res.items():
            if print_metrics:
                print(f'\t{prefix}_{metric_name}: {metric_val}')
            self.log(f'{prefix}_{metric_name}', metric_val, on_epoch=True, on_step=False, logger=True) 
        metrics.reset()

    def configure_optimizers(self):
        return self._resolve_optimizer(self.hparams.optimizer_name)
    
    def _check_model_gradients(self):
        print(f'\n{"="*10} Model Values & Gradients {"="*10}')
        for name, param in self.model.named_parameters():
            print(f'\t{name} -- value: {param.data.item():.5f} grad: {param.grad}')

    def _resolve_optimizer(self, optimizer_name:str):
        optimizer_name = optimizer_name.lower()
        if  optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.hparams.learning_rate)
        elif optimizer_name == 'lbfgs':
            return torch.optim.LBFGS(self.model.parameters(), lr=self.hparams.learning_rate, max_iter=20)
        
        raise NotImplementedError(f'Optimizer {self.hparams.optimizer_name} not implemented')


class LitSceneNet(LitWrapperModel):

    def __init__(self, geneo_num:dict, kernel_size:Tuple[int], criterion:torch.nn.Module, optimizer:str, learning_rate=1e-2, metric_initializer=None):
    
        model = SceneNet(geneo_num, kernel_size)
        self.save_hyperparameters('geneo_num', 'kernel_size')
        self.logged_batch = False
        self.gradient_check = False
        super().__init__(model, criterion, optimizer, learning_rate, metric_initializer)

    def get_model_parameters(self) -> nn.ParameterDict:
        return self.model.get_model_parameters()
    
    def training_step(self, batch, batch_idx):
        # As this is a GENEO model, we need to pass the convex coefficients and the GENEO parameters to the loss function
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_cvx_coefficients(), self.model.get_geneo_params())

        if self.train_metrics is not None: # on step metric logging
            self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update()

        self.log(f'train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def training_epoch_end(self, outputs) -> None:
        self.gradient_check = False
        # logging the GENEO parameters and the convex coefficients
        scenenet_params = self.model.get_model_parameters_in_dict()
        for name, param in scenenet_params.items():
            self.log(f'{name}', param, on_epoch=True, on_step=False, logger=True)
        return super().training_epoch_end(outputs)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_cvx_coefficients(), self.model.get_geneo_params())

        if self.val_metrics is not None:
            self.val_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'val_loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_cvx_coefficients(), self.model.get_geneo_params())

        if self.test_metrics is not None:
            self.test_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'test_loss', loss, logger=True)
        return loss
    

    def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:
        if not self.gradient_check:  # don't make the tf file huge
            self._check_model_gradients()
            self.gradient_check = True
        

    def _log_pointcloud_wandb(self, pcd, input=None, gt=None, prefix='run'):
        point_clouds = pointcloud_to_wandb(pcd, input, gt)
        self.logger.experiment.log({f'{prefix}_point_cloud': point_clouds})
        #self.log(f'{prefix}_point_cloud', point_clouds, logger=True)

    def on_validation_epoch_end(self) -> None:
        self.logged_batch = False

    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # logging batch point clouds to wandb
        if self.trainer.current_epoch % 10 == 0 and not self.logged_batch: 
            x, y = batch 
            pred = torch.squeeze(self(torch.unsqueeze(x[0], dim=0))).detach().cpu().numpy() # 1st batch sample
            pred = Vox.plot_voxelgrid(pred, color_mode='ranges', plot=False)
            x = torch.squeeze(x[0]).detach().cpu().numpy()
            x = Vox.plot_voxelgrid(x, color_mode='ranges', plot=False)
            y = torch.squeeze(y[0]).detach().cpu().numpy() # 1st batch sample
            y = Vox.plot_voxelgrid(y, color_mode='ranges', plot=False)
            self._log_pointcloud_wandb(pred, x, y, prefix=f'val_{self.trainer.global_step}')
            self.logged_batch = True

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('LitSceneNet')
        # careful, the values from this dict are parsed as strings
        # use example: --geneo_num cy=1 cone=1 neg=1
        parser.add_argument('--geneo_num', type=dict, action=ParseKwargs, default= {'cy'  : 1, 
                                                                                    'cone': 1, 
                                                                                    'neg' : 1})
        parser.add_argument('--kernel_size', type=tuple, default=(9, 7, 7))
        return parent_parser
        



class LitSceneNet_ADMM(LitSceneNet):

    def __init__(self, geneo_num:dict, kernel_size:Tuple[int], criterion:torch.nn.Module, optimizer:str, learning_rate=1e-2, metric_initializer=None):
        super().__init__(geneo_num, kernel_size, criterion, optimizer, learning_rate, metric_initializer)
        self.lag_mult_check = False
        #self.automatic_optimization = False


    def v3(self, batch, batch_idx):
        # As this is a GENEO model, we need to pass the convex coefficients and the GENEO parameters to the loss function
        x, y = batch
        pred = self(x)

        lbfgs, adam = self.configure_optimizers()

    
        constraint_norm = self.criterion.get_constraint_violation(self.model.get_model_parameters())

        if constraint_norm > 1e-3:
            """
            If the constraint is violated, we use the LBFGS optimizer to quickly reach the feasible space.
            Otherwise, we use Adam to optimize the objective function.
            """
            opt = lbfgs
            def closure():
                opt.zero_grad()
                loss = self.criterion(pred, y, self.model.get_model_parameters())
                self.manual_backward(loss, retain_graph=True)
                return loss
        else:
            opt = adam # Adam

        print(f"Using optimizer {str(opt)}...")
        
        if constraint_norm > 1e-3:
            opt.step(closure)
        else:
            opt.zero_grad()
            loss = self.criterion(pred, y, self.model.get_model_parameters())
            self.manual_backward(loss)
            opt.step()


        if self.train_metrics is not None: # on step metric logging
            self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update()

        self.log(f'train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def v2(self, batch, batch_idx):
        # As this is a GENEO model, we need to pass the convex coefficients and the GENEO parameters to the loss function
        lbfgs, adam = self.optimizers()

        x, y = batch
        pred = self(x)
        
        adam.zero_grad()
        data_loss = self.criterion.objective_function(pred, y)
        self.manual_backward(data_loss)
        adam.step()

        pred = self(x)

        
        constraint_loss = self.criterion.ADMM_regularizer(self.model.get_model_parameters())

        def closure():
            lbfgs.zero_grad()
            constraint_loss = self.criterion.ADMM_regularizer(self.model.get_model_parameters())
            self.manual_backward(constraint_loss, retain_graph=True)
            return constraint_loss

        lbfgs.step(closure)

        loss = data_loss + constraint_loss

        if self.train_metrics is not None: # on step metric logging
            self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update()

        self.log(f'train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    

    
    # def configure_optimizers(self):
    #     return super().configure_optimizers(), torch.optim.Adam(self.model.parameters(), lr=0.1)


    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_model_parameters())

        if self.train_metrics is not None: # on step metric logging
            self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update()

        self.log(f'train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        """
        After each training step, we update the contraint variables (i.e., \psi) and the lagrangian multipliers (i.e., \lambda)
        """

        theta_n = self.get_model_parameters()
        constraint_violation = self.criterion.get_constraint_violation(theta_n)
        # print(f"{'='*10} Theta_n {'='*10}")
        # for name, param in theta_n.items():
        #     print(f"\t{name}: {param}")

        #print(f"theta_n: {type(theta_n)}\n {theta_n}")

        with torch.no_grad():
            self.criterion.update_theta_k(theta_n)
            self.criterion.update_psi()
            self.criterion.update_lag_multipliers()
           
            if self.criterion.best_constraint_norm < constraint_violation:
                # if the constraint violation is increasing, we increase the penalty
                self.criterion.update_penalty()
            else:
                # otherwise, we increase the stepsize to accelerate convergence
                self.criterion.update_stepsize()
                
            self.criterion.current_constraint_norm = constraint_violation
            self.criterion.update_best_constraint_norm(theta_n)
        
        psi_values = self.criterion.get_psi()

        # if self.criterion.get_constraint_violation(theta_n) > 1e-3:
        #     self.criterion.increase_penalty()

        if not self.lag_mult_check: # don't make the tf file huge
            print(f"{'='*10} Lag Multipliers {'='*10}")
            for name, param in self.criterion.get_lag_multipliers().items():
                print(f"\t{name}: {float(param):.3f} = penalty * ({float(theta_n[name]):.3f} - {float(psi_values[name]):.3f})")
            self.lag_mult_check = True

    # def backward(self, loss, optimizer, optimizer_idx) -> None:
    #     loss.backward(retain_graph=True)
    
    def on_validation_epoch_end(self) -> None:
        self.lag_mult_check = False
        return super().on_validation_epoch_end() 
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_model_parameters())

        if self.val_metrics is not None:
            self.val_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'val_loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_model_parameters())

        if self.test_metrics is not None:
            self.test_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'test_loss', loss, logger=True)
        return loss

    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        
        constraint_violation = self.criterion.get_constraint_violation(self.model.get_model_parameters())

        self.log(f'constraint_violation', constraint_violation, on_epoch=False, on_step=True, prog_bar=False, logger=True)

        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
    

class LitSceneNet_AugLag(LitSceneNet):

    def __init__(self, geneo_num:dict, kernel_size:Tuple[int], criterion:Augmented_Lagrangian_Loss, optimizer:str, learning_rate=1e-2, metric_initializer=None):
        super().__init__(geneo_num, kernel_size, criterion, optimizer, learning_rate, metric_initializer)
        self.lag_mult_check = False

    def training_step(self, batch, batch_idx):
        # As this is a GENEO model, we need to pass the convex coefficients and the GENEO parameters to the loss function
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_model_parameters())

        if self.train_metrics is not None: # on step metric logging
            self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update()

        self.log(f'train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    
    def on_validation_epoch_end(self) -> None:
        self.lag_mult_check = False
        return super().on_validation_epoch_end() 
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_model_parameters())

        if self.val_metrics is not None:
            self.val_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'val_loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_model_parameters())

        if self.test_metrics is not None:
            self.test_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'test_loss', loss, logger=True)
        return loss
    

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        """
        After each training step, we update the contraint variables (i.e., \psi) and the lagrangian multipliers (i.e., \lambda)
        """
        #print('Update Lagrangian Multipliers and \psi...')

        theta_n = self.model.get_model_parameters()

        with torch.no_grad():
            if self.criterion.has_constraint_norm_decreased(theta_n): 
                # then update the lagrangian multipliers
                # This entails that the constraints are better satisfied by the current model parameters
                self.criterion.update_lag_multipliers(theta_n)
                try:
                    self.criterion.update_psi(theta_n)
                except:
                    pass
                self.criterion.update_best_constraint_norm(theta_n) # update the best constraint norm since it has decreased
                # the penalty is  maintained since it is sufficient to lead to a decrease in the constraint norm
            else:
                # the penalty is increased since the constraint norm has not decreased
                self.criterion.increase_penalty()
                # The lagrangian multipliers are maintained since it the constraints are not better satisfied by the current model parameters  


        if not self.lag_mult_check: # don't make the tf file huge
            print(f"Penalty: {self.criterion.get_penalty()}")
            print(f"{'='*10} Lag Multipliers {'='*10}")
            for name, param in self.criterion.get_lag_multipliers().items():
                print(f"\t{name}: {param}")
            self.lag_mult_check = True
        return None
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        
        constraint_violation = self.criterion.get_constraint_violation(self.model.get_model_parameters())

        self.log(f'constraint_violation', constraint_violation, on_epoch=False, on_step=True, prog_bar=False, logger=True)

        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    # def backward(self, loss, optimizer, optimizer_idx) -> None:
    #     loss.backward(retain_graph=True)
    



class LitSceneNet_Penalty(LitSceneNet):

    def __init__(self, geneo_num:dict, kernel_size:Tuple[int], criterion:torch.nn.Module, optimizer:str, learning_rate=1e-2, metric_initializer=None):
        super().__init__(geneo_num, kernel_size, criterion, optimizer, learning_rate, metric_initializer)


    def training_step(self, batch, batch_idx):
        # As this is a GENEO model, we need to pass the convex coefficients and the GENEO parameters to the loss function
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_model_parameters())

        if self.train_metrics is not None: # on step metric logging
            self.train_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update()

        self.log(f'train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_model_parameters())

        if self.val_metrics is not None:
            self.val_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'val_loss', loss, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y, self.model.get_model_parameters())

        if self.test_metrics is not None:
            self.test_metrics(torch.flatten(pred), torch.flatten(y).to(torch.int)).update() 

        self.log(f'test_loss', loss, logger=True)
        return loss
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        
        constraint_violation = self.criterion.get_constraint_violation(self.model.get_model_parameters())

        self.log(f'constraint_violation', constraint_violation, on_epoch=False, on_step=True, prog_bar=False, logger=True)

        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)