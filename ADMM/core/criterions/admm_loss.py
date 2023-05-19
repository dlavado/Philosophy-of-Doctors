

from typing import Iterable, Mapping
import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '..')
sys.path.insert(1, '../..')

from core.admm_utils.admm_constraints import Constraint



class ADMM_Loss(nn.Module):

    def __init__(self, objective_function:nn.Module, 
                 constraints:Mapping[str, Constraint], 
                 theta_0:nn.ParameterDict, 
                 init_penalty = 1.0, 
                 penalty_update_factor = 1.1,
                 max_penalty = 10,
                 stepsize = 10,
                 stepsize_update_factor = 0.5) -> None:
        """

        Alternating Direction Method of Multipliers (ADMM) Loss.
        This loss is utilized to solve the following optimization problem:\n
  

           \minimize_{\.theta} `objective_function`(\.theta) + `constraint_function`(\.theta)

        = \minimize_{\.theta, \psi} `objective_function`(\.theta) + `constraint_function`(\psi)
                s.t. \psi = \.theta    

        In this setting, the ADMM algorithm is of the following form:

        Repeat until convergence:

            1. \.theta^{k+1} = argmin_{\.theta} `objective_function`(\.theta) + \.rho/2 ||\.theta - \psi^k + \.lambda^k||^2

            2. \psi^{k+1} = argmin_{\psi} `constraint_function`(\psi) + \.rho/2 ||\.theta^{k +1} - \psi + \.lambda^k||^2

            3. \.lambda^{k+1} = \.lambda^k + \.rho(\.theta^{k+1} - \psi^{k+1})

        Parameters
        ----------

        `objective_function` - torch.nn.Module:
            The objective function to be minimized. Models data fidelity.

        `constraints` - List[Constraint]:
            A list of Constraint objects. Each constraint object contains a constraint os the optimization problem.

        `theta_0` - nn.ParameterDict:
            The initial values of the optimization variables (model parameters).

        `penalty_factor` - float:
            (a.k.a., \.rho.)
            The penalty factor used to enforce the \.theta = \psi constraint. The higher the penalty, the stiffer the opt problem becomes.

        """
        super().__init__()

        self.objective_function = objective_function

        self.constraints:Mapping[str, Constraint] = constraints

        self.psi, self.lag_multipliers = {}, {} # \psi and \lambda
        self.theta_k = {} # \.theta^{k}
        for key, theta_p in theta_0.items():
            # \psi are the primal optimization variables where the constraints are enforced.
            self.psi[key] = torch.tensor(theta_p.clone(), requires_grad=False, device='cuda:0')
            # Lagrangian multipliers of the optimization variables. a.k.a., \lambda.
            # Intuitively, these can be thought as an offset between \.theta and \psi.
            self.lag_multipliers[key] = torch.tensor(0.0, requires_grad=False, device='cuda:0')

            self.theta_k[key] = torch.tensor(theta_p.clone(), requires_grad=False, device='cuda:0')

    
        self.current_constraint_norm  = self.get_constraint_violation(theta_0)
        self.best_constraint_norm = self.current_constraint_norm

        self.penalty_factor = init_penalty  # a.k.a., \.rho
        self.max_penalty = max_penalty
        self.stepsize = stepsize
        self.penalty_update_factor = penalty_update_factor
        self.stepsize_update_factor = stepsize_update_factor


    def forward(self, y_pred:torch.Tensor, y_gt:torch.Tensor, theta_n:nn.ParameterDict):
        """
        Computes the forward pass of the ADMM loss.
        The consists in an update of the main optimization variables \.theta (that is, step 1. in the ADMM algorithm).~

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
        
        return self.objective_function(y_pred, y_gt) + self.ADMM_regularizer(theta_n) + self.Stochastic_ADMM_regularizer(theta_n)
        #return self.objective_function(y_pred, y_gt) + self.Lagrangian_regularizer(theta_n) + self.aug_Lagrangian_regularizer(theta_n)
        # self.objective_function(y_pred, y_gt)
        # return self.aug_Lagrangian_regularizer(theta_n) + self.Lagrangian_regularizer(theta_n)

    def Stochastic_ADMM_regularizer(self, theta_n:nn.ParameterDict):

        diff = [theta_n[key] - self.theta_k[key] for key in theta_n]

        return torch.linalg.norm(torch.tensor(diff), ord=2) / (2*self.stepsize)
    
    
    def ADMM_regularizer(self, theta_n:nn.ParameterDict):
        pows = [torch.pow(theta_n[key] - self.psi[key] + self.lag_multipliers[key], 2) for key in theta_n] # tradtional ADMM formulation

        return (self.penalty_factor/2) * sum(pows) # L2 norm

    
    def aug_Lagrangian_regularizer(self, theta_n:nn.ParameterDict):
        """
        Computes the augmented Lagrangian regularizer of the ADMM loss.
        """
        # print(f"psi: {np.array(list(self.psi.values()))}")
        # print(f"lag_mult: {np.array(list(self.lag_multipliers.values()))}")
        # input("Press Enter to continue...")

        #print(f"theta_n: {theta_n}")
        
        #pows = [torch.pow(theta_n[key] - self.psi[key] + self.lag_multipliers[key], 2) for key in theta_n] # tradtional ADMM formulation
        pows = [torch.pow(theta_param - self.psi[key], 2) for key, theta_param in theta_n.items()]
        # print([t.grad for t in theta_n.values()])
        # print(f"pows: {pows}")
        # print(f"pows sum: {sum(pows)}")
        # input("Press Enter to continue...")
        return (self.penalty_factor/2) * sum(pows) # L2 norm
        #return self.penalty_factor/2 * torch.linalg.norm(pows, dim=-1, ord=2)

    def Lagrangian_regularizer(self, theta_n:nn.ParameterDict):
        return sum([torch.abs(self.lag_multipliers[key] * (theta_param - self.psi[key])) for key, theta_param in theta_n.items()]) # L1 norm
    
    
    def _compute_eval_constraint(self, theta_n:nn.ParameterDict) -> Mapping[str, float]:
        """
        Computes the constraint violation w.r.t. theta_n.
        """

        eval_constraint = {key: torch.tensor(0.0, requires_grad=False, device='cuda:0') for key in theta_n}
        for constraint in self.constraints.values():
            for key, eval in constraint.evaluate_constraint(theta_n).items():
                eval_constraint[key] += eval

        return eval_constraint
    

    def get_constraint_violation(self, theta_n:nn.ParameterDict, power=2):
        """
        i.e., || `constraint_function`(\.theta) ||_2^2
        """
        pows = [torch.pow(eval, power) for eval in self._compute_eval_constraint(theta_n).values()]
        return sum(pows)

    def update_psi(self):
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
            updated_psi = constraint.update_psi(self.psi, self.theta_k, self.lag_multipliers)
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


    def update_lag_multipliers(self):
        """
        Updates the Lagrangian multipliers \.lambda (that is, step 3. in the ADMM algorithm).
        """

        for key in self.lag_multipliers:
            # Lagrangian multipliers are updated by adding the offset between \.theta and \psi. These should be non-negative.
            self.lag_multipliers[key] = self.lag_multipliers[key] + self.penalty_factor * (self.theta_k[key] - self.psi[key])
            #self.lag_multipliers[key] = self.penalty_factor * (theta_k_plus_1[key] - self.psi[key])
            #print(f"{self.lag_multipliers[key]} = {self.lag_multipliers[key]} + {self.penalty_factor} * ({theta_k_plus_1[key]} - {self.psi[key]})")
        
        #print(np.array(list(self.lag_multipliers.values())))

    def update_stepsize(self):
        self.stepsize = self.stepsize * self.stepsize_update_factor
        
    def update_penalty(self):
        self.penalty_factor = min(self.penalty_factor * self.penalty_update_factor, self.max_penalty)

    def get_lag_multipliers(self):
        return self.lag_multipliers
    

    def get_best_constraint_norm(self):
        return self.best_constraint_norm
    
    def update_best_constraint_norm(self, theta_n:nn.ParameterDict):
        """
        Updates the best constraint norm
        """
        self.best_constraint_norm = min(self.best_constraint_norm, self.get_constraint_violation(theta_n))



    def _clone_theta(self, theta:nn.ParameterDict):
        return {key: theta[key].clone() for key in theta}
    
    def update_theta_k(self, theta:nn.ParameterDict):
        self.theta_k = self._clone_theta(theta)
    
    def get_psi(self):
        return self.psi

