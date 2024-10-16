

from typing import Iterable, Mapping, Union
import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.admm_utils.admm_constraints import Constraint




class Augmented_Lagrangian_Loss(nn.Module):


    def __init__(self, objective_function:nn.Module, 
                 constraints:Mapping[str, Constraint], 
                 theta_0:nn.ParameterDict, 
                 best_constraint_norm = None, 
                 Lag_initializer = Union[None , Mapping[str, torch.Tensor]], 
                 init_penalty = 1.0, 
                 penalty_update_factor = 1.1,
                 max_penalty = 10) -> None:
        """
        Augmented Lagrangian Loss.
        This loss is utilized to solve the following optimization problem:\n

        \t\t\t \minimize_{\.theta} `objective_function`(\. theta) + `constraint_function`(\.theta)

        Augmented Lagrangian Loss is of the following form:

        Repeat until convergence:

            1. \.theta^{k+1} = argmin_{\.theta} `objective_function`(\.theta) + \.lambda^k * `constraint_function`(\.theta) + \.rho/2 ||`constraint_function`(\.theta)||^2

            2. \.lambda^{k+1} = \.lambda^k + \.rho * `constraint_function`(\.theta^{k+1})

            3. \.rho^{k+1} = \.rho * `penalty_update_factor`

        Parameters
        ----------

        `objective_function` - torch.nn.Module:
            The objective function to be minimized. Models data fidelity.

        `constraints` - Dict[Str, Constraint]:
            A dictionary of Constraint objects. Each constraint object contains a constraint of the optimization problem.

        `theta_0` - nn.ParameterDict:
            The initial values of the optimization variables (model parameters).

        `best_constraint_norm` - float:
            The initial value of the constraint norm. If None, the initial value is computed from the initial values of the optimization variables.

        `Lag_initializer` - Dict[Str, torch.Tensor]:
            The initial values of the Lagrangian multipliers. If None, the initial values are set to 0.

        `init_penalty` - float:
            (a.k.a., \.rho.)
            The initial penalty factor used to enforce the constraints. The higher the penalty, the stiffer the opt problem becomes.

        `penalty_update_factor` - float:
            The factor by which the penalty is updated at each iteration.

        `max_penalty` - float:
            The maximum penalty value.
        """

        super().__init__()

        self.objective_function = objective_function

        self.constraints = constraints

        self.max_penalty = max_penalty
        self.penalty_factor = init_penalty
        self.penalty_update_factor = penalty_update_factor

        self.constraint_norm = self.compute_constraint_norm(theta_0, ord=2)
        if best_constraint_norm is None:
            self.best_constraint_norm = self.constraint_norm
        else:
            self.best_constraint_norm = best_constraint_norm

        # init Lag multiplier values:
        if Lag_initializer is None: # if no Lag initializer is provided, initialize Lag multipliers to 0
            self.lag_multipliers = {key: torch.tensor(0.0, requires_grad=False, device='cuda:0') for key in theta_0}
        else:
            self.lag_multipliers = Lag_initializer


    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, theta_n:nn.ParameterDict):
        """
        Computes the forward pass of the Augmented Lagrangian loss.
        The consists in an update of the main optimization variables \.theta (that is, step 1. in the AugLag algorithm).~

        Parameters
        ----------

        `y_pred` - torch.Tensor:
            The predicted output of the model.

        `y_gt` - torch.Tensor:
            The ground truth output.

        `theta_n` - nn.ParameterDict:
            The current values of the optimization variables.
        """

        # print(f"theta_n: {np.array(list(theta_n.values()))}")
        # input("Press Enter to continue...")

        return self.objective_function(y_pred, y_gt) + self.Lagrangian_regularizer(theta_n) + self.aug_Lagrangian_regularizer(theta_n)

    
    def aug_Lagrangian_regularizer(self, theta_n:nn.ParameterDict):
        """
        Computes the augmented Lagrangian regularizer of the ADMM loss.

        i.e., || `constraint_function`(\.theta) ||_2^2
        """

        pows = [torch.pow(eval, 2) for eval in self.compute_eval_constraint(theta_n).values()]
        return (self.penalty_factor/2) * sum(pows)
        #return (self.penalty_factor/2) * self.compute_constraint_norm(theta_n, ord=2) ** 2

    
    def get_constraint_violation(self, theta_n:nn.ParameterDict):
        """
        i.e., || `constraint_function`(\.theta) ||_2^2
        """
        pows = [torch.pow(eval, 2) for eval in self.compute_eval_constraint(theta_n).values()]
        return sum(pows)
    

    def Lagrangian_regularizer(self, theta_n:nn.ParameterDict):
        return sum([self.lag_multipliers[key]*eval for key, eval in self.compute_eval_constraint(theta_n).items()])


    def compute_eval_constraint(self, theta_n:nn.ParameterDict) -> Mapping[str, float]:
        """
        Computes the constraint violation w.r.t. theta_n.
        """

        eval_constraint = {key: torch.tensor(0.0, requires_grad=False, device='cuda:0') for key in theta_n}
        for constraint in self.constraints.values():
            for key, eval in constraint.evaluate_constraint(theta_n).items():
                eval_constraint[key] += eval

        return eval_constraint

    
    def compute_constraint_norm(self, theta_n:nn.ParameterDict, ord=2):
        """
        Computes the norm of the constraint violation w.r.t. theta_n.
        """
        #print(list(self._compute_eval_constraint(theta_n).values()))
        return torch.linalg.norm(torch.tensor(list(self.compute_eval_constraint(theta_n).values())), ord=ord) 

    def has_constraint_norm_decreased(self, theta_n:nn.ParameterDict):
        """
        Checks if the constraint norm has decreased.
        """
        return self.compute_constraint_norm(theta_n) <= self.best_constraint_norm
    
    def update_lag_multipliers(self, theta_n:nn.ParameterDict):
        """
        Updates the Lagrangian multipliers.
        """
        eval = self.compute_eval_constraint(theta_n)
        for key in self.lag_multipliers:
            self.lag_multipliers[key] = self.lag_multipliers[key] + self.penalty_factor * eval[key]

    def update_best_constraint_norm(self, theta_n:nn.ParameterDict):
        """
        Updates the best constraint norm.
        """
        self.best_constraint_norm = min(self.best_constraint_norm, self.compute_constraint_norm(theta_n))

    def increase_penalty(self):
        """
        Updates the penalty factor.
        """
        self.penalty_factor = min(self.penalty_factor * self.penalty_update_factor, self.max_penalty)

    def get_penalty(self):
        return self.penalty_factor

    def get_lag_multipliers(self):
        return self.lag_multipliers
    
    def get_best_constraint_norm(self):
        return self.best_constraint_norm
    

class Stochastic_ADMM_Loss(Augmented_Lagrangian_Loss):

    def __init__(self, objective_function: nn.Module, 
                 constraints: Mapping[str, Constraint], 
                 theta_0: nn.ParameterDict, 
                 best_constraint_norm=None, 
                 Lag_initializer=Union[None, Mapping[str, torch.Tensor]], 
                 init_penalty=1, 
                 penalty_update_factor=1.1, 
                 max_penalty=10) -> None:
        
        super().__init__(objective_function, constraints, theta_0, best_constraint_norm, 
                         Lag_initializer, init_penalty, penalty_update_factor, max_penalty)
        
        self.psi, self.lag_multipliers = {}, {}
        for key, theta_p in theta_0.items():
            # \psi are the primal optimization variables where the constraints are enforced.
            self.psi[key] = torch.tensor(theta_p.clone(), requires_grad=False, device='cuda:0')
            # Lagrangian multipliers of the optimization variables. a.k.a., \lambda.
            # Intuitively, these can be thought as an offset between \.theta and \psi.
            self.lag_multipliers[key] = torch.tensor(0.0, requires_grad=False, device='cuda:0')

    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, theta_n:nn.ParameterDict):

        return self.objective_function(y_pred, y_gt) + self.ADMM_regularizer(theta_n)


    def ADMM_regularizer(self, theta_n:nn.ParameterDict):
        pows = [torch.pow(theta_n[key] - self.psi[key] + self.lag_multipliers[key], 2) for key in theta_n] # tradtional ADMM formulation

        return (self.penalty_factor/2) * sum(pows) # L2 norm
    

    def update_psi(self, theta_k_plus_1:nn.ParameterDict):
        """

        Updates the optimization variables \psi (that is, step 2. in the ADMM algorithm).

        Since the constraints we are considering are linear, this problem has a closed-form solution.
        Specifically, the solution is given by:

            \psi_{k+1} = \.theta_{k+1} + \.lambda_{k+1} if `constraints`(\.theta_{k+1}) = 0 else project onto the feasible set.
        """
        #print(f"psi: {np.array(list(self.psi.values()))}")
        updated_keys = []
        for constraint in self.constraints.values():
            constraint_eval = constraint.evaluate_constraint(self.psi)
            updated_psi = constraint.update_psi(self.psi, theta_k_plus_1, self.lag_multipliers)
            # print(f"{'='*10} contraint: {constraint.constraint_name} {'='*10}")
            # print(f"constraint_eval:\n {constraint_eval}")
            # print(f"updated_psi:\n {updated_psi}")

            for key in updated_psi:
                if key not in updated_keys: # a parameter can only be updated once
                    self.psi[key] = updated_psi[key]

                    if constraint_eval[key] > 0: # constraint is violated, then the psi is updated
                        updated_keys.append(key)
                        #print(f"constraint violated: {constraint.constraint_name} at key {key} with value {constraint_eval[key]}")
            # print(f"updated keys: {updated_keys} at constraint {constraint.constraint_name}")

        # print(f"new psi:\n {np.array(list(self.psi.values()), dtype=np.float32)}")


    def update_lag_multipliers(self, theta_k_plus_1:nn.ParameterDict):
        """
        Updates the Lagrangian multipliers \.lambda (that is, step 3. in the ADMM algorithm).
        """

        for key in self.lag_multipliers:
            # Lagrangian multipliers are updated by adding the offset between \.theta and \psi. These should be non-negative.
            self.lag_multipliers[key] = self.lag_multipliers[key] + self.penalty_factor * (theta_k_plus_1[key] - self.psi[key])
            #self.lag_multipliers[key] = self.penalty_factor * (theta_k_plus_1[key] - self.psi[key])
            #print(f"{self.lag_multipliers[key]} = {self.lag_multipliers[key]} + {self.penalty_factor} * ({theta_k_plus_1[key]} - {self.psi[key]})")
        
        #print(np.array(list(self.lag_multipliers.values())))

