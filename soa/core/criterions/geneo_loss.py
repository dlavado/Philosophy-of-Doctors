
import torch

class GENEORegularizer(torch.nn.Module):
    """
    Loss function for GENEO-based models
    """

    def __init__(self, rho=0.1) -> None:
        """
        GENEO Loss is a custom loss GENEO-based models:
            - convexity penalty: penalizes non-convex coefficients (they should be positive and sum to 1)
            - non-positivity penalty: penalizes non-positive coefficients. 
                (This one is not general for all GENEO-based models, but in geometric inductive biases, parameters such as the radius of a sphere do not have a physical meaning if they are negative)

        Parameters
        ----------
        `rho`: float
            The weight of the convexity and non-positivity penalties
        """

        super(GENEORegularizer, self).__init__()
        
        self.rho = rho
       

    def cvx_regularizer(self, cvx_coeffs:list[torch.Tensor]):
        """
        Penalizes non-positive convex parameters;
        
        Each tensor in cvx_coeffs is a tensor of shape (G, O) where G is the number of GENEOs and O is the number of Observers.
        We want to maintain convexity for each observer
        """
        cum = 0
        for cvx in cvx_coeffs:
           cum += torch.sum(torch.relu(-cvx)) + torch.abs(torch.sum(1 - torch.sum(cvx, dim=0)))
        return cum
       
    
     
    def positive_regularizer(self, geneo_params:list[torch.Tensor]):
        """
        Penalizes non-positive parameters
        """
        cum = 0
        for geneo in geneo_params:
            cum += torch.sum(torch.relu(-geneo))
        return cum
        
        
    def forward(self, geneo_params:list[torch.Tensor], cvx_params:list[torch.Tensor]):
        """
        Compute the GENEO loss
        
        Parameters
        ----------
        
        `geneo_params`: list[torch.Tensor]
            List of tensors containing the parameters of different GENEOs, each tensor has variable shape
            
        `cvx_params`: list[torch.Tensor]
            List of tensors containing the convex coefficients of a GENEO Layer, each tensor has shape (G, O)
        """

        cvx_reg = self.cvx_regularizer(cvx_params)
        pos_reg = self.positive_regularizer(geneo_params)

        return self.rho * (cvx_reg + pos_reg)
    