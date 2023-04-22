# %% 
import os
from typing import Mapping
import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')
from core.admm_utils.admm_constraints import Constraint



class Constrained_Loss(nn.Module):

    def __init__(self,
                 objective_function,
                 constraints:Mapping[str, Constraint]=None) -> None:
        
        """
        Constrained Loss.
        The objective function represents the primal problem, i.e., the data fidelity term.
        The constraints represent penalty terms that are added to the objective function and enforce 
        that the model parameters remain within the feasible set.


        Parameters
        ----------
        `objective_function` - torch.nn.Module:
            The objective function to be minimized. Models data fidelity.

        `constraints` - Dict[str, Constraint] or List[Constraint]:

        """
        super().__init__()

        self.objective_function = objective_function

        self.constraints = constraints

    
    def forward(self, y_pred, y_gt, model_params:nn.ParameterDict=None) -> torch.Tensor:

        data_fidelity = self.objective_function(y_pred, y_gt)

        if self.constraints is not None and model_params is not None:
            #penalty = self._compute_constraint_norm(model_params, ord=1)
            penalty = self._compute_constraint_sum(model_params)
            
            return data_fidelity + penalty + self.parameters_norm(model_params, ord=2)
        else:
            return data_fidelity
        
    
    def _compute_eval_constraint(self, theta_n:nn.ParameterDict) -> Mapping[str, float]:
        """
        Computes the constraint violation w.r.t. theta_n.
        """

        eval_constraint = {key: torch.tensor(0.0, requires_grad=False, device='cuda:0') for key in theta_n}

        if isinstance(self.constraints, list):
            for constraint in self.constraints:
                for key, eval in constraint.evaluate_constraint(theta_n).items():
                    eval_constraint[key] += eval
        else:
            for constraint in self.constraints.values():
                for key, eval in constraint.evaluate_constraint(theta_n).items():
                    eval_constraint[key] += eval

        return eval_constraint
    
    def get_constraint_violation(self, theta_n:nn.ParameterDict):
        """
        i.e., || `constraint_function`(\.theta) ||_2^2
        """

        pows = [torch.pow(eval, 2) for eval in self._compute_eval_constraint(theta_n).values()]
        return sum(pows)
        #return (self.penalty_factor/2) * self.compute_constraint_norm(theta_n, ord=2) ** 2
    
    def _compute_constraint_sum(self, theta_n:nn.ParameterDict) -> float:
        """
        Computes the constraint violation w.r.t. theta_n.
        """
        return sum(self._compute_eval_constraint(theta_n).values())
    
    def parameters_norm(self, theta_n:nn.ParameterDict, ord=2) -> float:
        """
        Computes the norm of the parameters.
        """
        return torch.linalg.norm(torch.tensor(list(theta_n.values())), ord=ord)
        
    def _compute_constraint_norm(self, theta_n:nn.ParameterDict, ord=2):
        """
        Computes the norm of the constraint violation w.r.t. theta_n.
        """
        #print(list(self._compute_eval_constraint(theta_n).values()))
        return torch.linalg.norm(torch.tensor(list(self._compute_eval_constraint(theta_n).values())), ord=ord) 
