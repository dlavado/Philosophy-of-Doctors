

import faulthandler
from typing import Iterator, Mapping, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys


class Constraint(nn.Module):

    def __init__(self, weight:float, constraint_name:str) -> None:

        super().__init__()
        self.constraint_name = constraint_name
        self.weight = weight


    def constraint_function(self, model_parameters:Iterator[Tuple[str, nn.Parameter]]) -> float:
        """
        The constraint function.
        """
        raise NotImplementedError
    
    def evaluate_constraint(self, model_parameters:Iterator[Tuple[str, nn.Parameter]]) -> float:
        """
        Evaluates the constraint function.

        If the constraint defined by `constraint function` is satisfied, then its evaluation is 0. 
        Else, it is a positive number (follows the flag logic in `_flag_constraint()`)
        """
        return self.constraint_function(model_parameters)
    

    def _constraint_on_params(self, model_parameters: Iterator[Tuple[str, nn.Parameter]]) -> dict:
        """
        Evaluates the constraint function on the model parameters.
        """
        raise NotImplementedError
    
    def update_psi(self, psi:dict, 
                         theta: dict, 
                         lag_multiplier: dict) -> dict:
        """
        Updates the \psi values of the ADMM algothm.
        """
        raise NotImplementedError
    
    def _flag_constraint(self, x):
        """
        If the constraint is not satisfied, i.e., if Constraint < 0, then we flag it with the relu function. 
        This also enforces the non-negativity constraint.
        """
        return torch.relu(-x)
    
    def __str__(self) -> str:
        return self.constraint_name
    

class Lp_Constraint(Constraint):

    def __init__(self, weight:float, p:int) -> None:
        """
        Parameters
        ----------
        `weight` - float:
            The weight of the constraint in the overall loss function.

        `p` - int:
            The p-norm to be used for the constraint.
        """
        super().__init__(weight, "Lp Constraint")

        self.p = p
    
    def constraint_function(self, model_parameters:Iterator[Tuple[str, nn.Parameter]]) -> float:
        """
        The constraint function.
        """

        lp_regularizer = 0
        for p_name, p_value in model_parameters:
            lp_regularizer += torch.norm(p_value, p=self.p)
        return lp_regularizer
    

    def evaluate_constraint(self, model_parameters:Iterator[Tuple[str, nn.Parameter]]) -> float:
        return self.weight * self.constraint_function(model_parameters)
    
    def _constraint_on_params(self, model_parameters: Iterator[Tuple[str, nn.Parameter]]) -> dict:
        return {p_name: torch.norm(p_value, p=self.p) for p_name, p_value in model_parameters} # the constraint is always satisfied
    
    def update_psi(self, psi: dict, 
                         theta: dict, 
                         lag_multiplier: dict) -> dict:
        """
        Updates the \psi values of the ADMM algothm.
        """

        eval = {}
        for p_name, p_value in psi.items():
                # if the constraint is satisfied, then the update is just the sum of the lagrangian multiplier and the primal variable
                # else, the psi value is 0
                val = torch.zeros_like(p_value)
                satisfy = theta[p_name] + lag_multiplier[p_name] # constraint satisfied
                val[p_value == 0] = satisfy[p_value == 0]
                eval[p_name] = val

        return eval
    

class Orthogonality_Constraint(Constraint):


    def __init__(self, weight:float, ieneo=False) -> None:
        """
        Orthogonal constraint for the kernels of a Conv2d layer.
        
        It promotes diversity among the kernels of a Conv2d layer by penalizing the lack distance between them.

        Specifically, it computes the distance between each pair of kernels and penalizes the distance if it is less than a `threshold`.

        """
        super().__init__(weight, "Orthogonality")

        self.ieneo = ieneo # if True, then the constraint is evaluated on the kernels resulting from the ieneo layer

    def conv_orth_dist(self, kernel, stride = 1):
        """
        Deprecated - uses np, no auto-diff

        Computes the orthogonal distance between the kernels of a Conv2d layer.

        Code stolen from: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Orthogonal_Convolutional_Neural_Networks_CVPR_2020_paper.pdf
        :)
        """
        [o_c, i_c, w, h] = kernel.shape
        assert (w == h),"Do not support rectangular kernel"
        #half = np.floor(w/2)
        assert stride<w,"Please use matrix orthgonality instead"
        new_s = stride*(w-1) + w#np.int(2*(half+np.floor(half/stride))+1)
        temp = torch.eye(new_s*new_s*i_c).reshape((new_s*new_s*i_c, i_c, new_s,new_s)).cuda()
        out = (faulthandler.conv2d(temp, kernel, stride=stride)).reshape((new_s*new_s*i_c, -1))
        Vmat = out[np.floor(new_s**2/2).astype(int)::new_s**2, :]
        temp= np.zeros((i_c, i_c*new_s**2))
        for i in range(temp.shape[0]):temp[i,np.floor(new_s**2/2).astype(int)+new_s**2*i]=1
        return torch.norm( Vmat@torch.t(out) - torch.from_numpy(temp).float().cuda() )
        
    def deconv_orth_dist(self, kernel:torch.Tensor, stride = 2, padding = 1):
        """
        Computes the orthogonal distance between the kernels of a Conv2d layer.

        Code stolen from: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Orthogonal_Convolutional_Neural_Networks_CVPR_2020_paper.pdf
        :)
        """
        [o_c, i_c, w, h] = kernel.shape
        
        output = torch.conv2d(kernel, kernel, stride=stride, padding=padding).cuda()
        target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
        ct = int(np.floor(output.shape[-1]/2))
        target[:,:,ct,ct] = torch.eye(o_c).cuda()
        return torch.norm( output - target )
        
    def orth_dist(self, mat:torch.Tensor, stride=None):
        """
        Computes the orthogonal distance between MLP weights.

        Code stolen from: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Orthogonal_Convolutional_Neural_Networks_CVPR_2020_paper.pdf
        :)
        """
        mat = mat.reshape( (mat.shape[0], -1) ).cuda()
        if mat.shape[0] < mat.shape[1]:
            mat = mat.permute(1,0)
        return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).cuda())

        
    def constraint_function(self, weight:torch.Tensor, conv=True) -> float:

        if conv:
            # if the weight is a convolutional layer, then we need to compute the distance between the conv kernels
            return self.deconv_orth_dist(weight)
        else:
            return self.orth_dist(weight)

        regularization_loss = 0.0

        # if weight.dim() <= 1:
        #     return regularization_loss
        # # Get the weight tensor of the convolutional layer
        # weight = weight.view(weight.size(0), -1)
        # # Calculate the Gram matrix
        # gram_matrix = torch.matmul(weight, weight.t())  
        # # Calculate the Frobenius norm of the Gram matrix
        # frobenius_norm = torch.norm(gram_matrix, p='fro'  
        # regularization_loss += frobenius_norm
                
        # return regularization_loss
    
    def evaluate_constraint(self, model_parameters: Iterator[Tuple[str, nn.Parameter]]) -> torch.Tensor:

        orthogonality_penalty = 0.0

        for p_name, p_value in model_parameters:
            if 'weight' in p_name:
                orthogonality_penalty += self.constraint_function(p_value, 'conv' in p_name)
        return self.weight * orthogonality_penalty
    
    def _constraint_on_params(self, model_parameters: Iterator[Tuple[str, nn.Parameter]]) -> dict:

        const = {}
        for p_name, p_value in model_parameters:
            if 'weight' in p_name and 'conv' in p_name:
                const[p_name] = self.constraint_function(p_value)/p_value.numel()
            else:
                const[p_name] = torch.tensor(0.0, device=p_value.device)
        return const
    
    def update_psi(self, psi:dict, 
                         theta: dict, 
                         lag_multiplier: dict) -> dict:
        """
        Updates the \psi values of the ADMM algothm.
        """
        # previous \psi is not used in this case, kept for consistency

        # the constraint is always satisfied, so the update is just the sum of the lagrangian multiplier and the primal variable

        eval = {}

        for p_name, p_value in psi.items():
            if 'weight' in p_name:
                # if conv weights, apply the constraint
                eval[p_name] = torch.full_like(theta[p_name], self.constraint_function(p_value, 'conv' in p_name)/p_value.numel())
            # else:
            #     # if not conv weights, the constraint is always satisfied
            #     eval[p_name] = theta[p_name] + lag_multiplier[p_name]

        return eval
    


class Nonnegativity_Constraint(Constraint):
    
        def __init__(self, weight:float, param_name = None) -> None:
            super().__init__(weight, 'nonnegativity')

            # In case we want to apply the constraint to a specific parameter
            self.param_name = param_name


        def constraint_function(self, model_parameters: Iterator[Tuple[str, nn.Parameter]]) -> float:

            nonneg_penalty = 0.0

            if self.param_name is None:
                for _, p_value in model_parameters:
                    nonneg_penalty += self._flag_constraint(p_value).sum()
            else:
                for p_name, p_value in model_parameters:
                    if self.param_name in p_name:
                        nonneg_penalty += self._flag_constraint(p_value).sum()
            
            return nonneg_penalty
                
            
        def evaluate_constraint(self, model_parameters:Iterator[Tuple[str, nn.Parameter]]) -> float:
            """
            Evaluates the constraint function.
            """
            return self.weight * self.constraint_function(model_parameters)
        
        
        def _constraint_on_params(self, model_parameters: Iterator[Tuple[str, nn.Parameter]]) -> dict:
            return {p_name: self._flag_constraint(p_value) for p_name, p_value in model_parameters}                    

        def update_psi(self, psi:dict, 
                        theta:dict, 
                        lag_multiplier: dict) -> dict:
            """
            Updates the \psi values of the ADMM algothm.

            Nonnegativity constraint is applied to all parameters in \psi

            Returns
            -------
            psi : nn.ParameterDict
                The updated \psi values.
            """
            
            eval = self._constraint_on_params(psi.items())

            psi_n_plus_1 = {}
            
            for key in eval:
                if eval[key] == 0: # If the constraint is satisfied, then we update the \psi value.
                    psi_n_plus_1[key] = torch.tensor(theta[key] + lag_multiplier[key], device=psi[key].device, dtype=psi[key].dtype)
                    #print(f'psi[{key}] = {psi[key]} = {theta[key]} + {lag_multiplier[key]}')
                else: 
                    # if the constraint is not satisfied, that means that the \psi value is non-negative, 
                    # thus we project it to the feasible set, i.e., to 0.
                    psi_n_plus_1[key] = torch.tensor(0.0, device=psi[key].device, dtype=psi[key].dtype)

            return psi_n_plus_1


class Convexity_Constraint(Constraint):

    def __init__(self, weight:float, cvx_coeff_name) -> None:
        super().__init__(weight, 'convexity')

        self.cvx_coeff_name = cvx_coeff_name

    def constraint_function(self, model_parameters:Iterator[Tuple[str, nn.Parameter]]) -> float:

        cvx_penalty = torch.tensor(0.0, device='cuda:0')

        for p_name, p_value in model_parameters:
            if self.cvx_coeff_name in p_name:
                cvx_penalty = (1 - torch.sum(p_value, dim=p_value.dim()-1)).abs()

        return cvx_penalty.sum()


    def evaluate_constraint(self, model_parameters:Iterator[Tuple[str, nn.Parameter]]) -> float:
        """
        Evaluates the constraint function.
        """

        return self.weight * self.constraint_function(model_parameters)
    

    def _constraint_on_params(self, model_parameters: Iterator[Tuple[str, nn.Parameter]]) -> dict:

        for p_name, p_value in model_parameters:
            if self.cvx_coeff_name in p_name:
                # soft_cvx represents the intended values for the convex coeffcients while respecting their magnitude.
                soft_cvx = (1 - torch.sum(p_value, dim=p_value.dim()-1)).abs()
                
                return {p_name: soft_cvx} # The difference between the intended values and the actual values is the penalty.
        

    def update_psi(self, psi:dict, theta:dict, lag_multiplier: dict) -> dict:
        """
        Updates the \psi values of the ADMM algothm.

        Convexity constraint is applied to all parameters in \psi

        Returns
        -------
        psi : nn.ParameterDict
            The updated \psi values
        """
        
        eval = self._constraint_on_params(psi.items())

        psi_n_plus_1 = {}
        
        for key in eval:
            if eval[key] == 0: # If the constraint is satisfied, then we update the \psi value.
                psi_n_plus_1[key] = torch.tensor(theta[key] + lag_multiplier[key], device=psi[key].device, dtype=psi[key].dtype)
                #print(f'psi[{key}] = {psi[key]} = {theta[key]} + {lag_multiplier[key]}')
            else: 
                # if the constraint is not satisfied, that means that the \psi value is non-negative, 
                # thus we project it to the feasible set, i.e., to 0.
                psi_n_plus_1[key] = torch.zeros_like(psi[key], device=psi[key].device, dtype=psi[key].dtype)

        return psi_n_plus_1
    


class Convex_Square_Constraint(Constraint):

    def __init__(self, weight:float, cvx_coeff_name) -> None:
        super().__init__(weight, 'convex_square')

        self.cvx_coeff_name = cvx_coeff_name

    def constraint_function(self, model_parameters:Iterator[Tuple[str, nn.Parameter]]) -> float:

        
        for p_name, p_value in model_parameters:
            if self.cvx_coeff_name in p_name:
                cvx_penalty = 1 - torch.sum(torch.pow(p_value, 2), dim=p_value.dim()-1)

        return cvx_penalty.sum()


    def evaluate_constraint(self, model_parameters:Iterator[Tuple[str, nn.Parameter]]) -> float:
        """
        Evaluates the constraint function.
        """

        return self.weight * self.constraint_function(model_parameters)
    

    def _constraint_on_params(self, model_parameters: Iterator[Tuple[str, nn.Parameter]]) -> dict:

        for p_name, p_value in model_parameters:
            if self.cvx_coeff_name in p_name:
                # soft_cvx represents the intended values for the convex coeffcients while respecting their magnitude.
                soft_cvx = torch.softmax(torch.pow(p_value, 2), dim=p_value.dim()-1)
                
                return {p_name: p_value - soft_cvx} # The difference between the intended values and the actual values is the penalty.
        

    def update_psi(self, psi:dict, theta:dict, lag_multiplier: dict):
        """
        Updates the \psi values of the ADMM algothm.

        Convexity constraint is applied to all parameters in \psi

        Returns
        -------
        psi : nn.ParameterDict
            The updated \psi values
        """
        
        return self._constraint_on_params(psi.items())


    


    
