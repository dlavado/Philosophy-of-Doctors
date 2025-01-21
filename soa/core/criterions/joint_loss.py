import torch


class JointLoss(torch.nn.Module):
    def __init__(self, *losses, weights=None):
        super(JointLoss, self).__init__()
        self.losses = torch.nn.ModuleList(losses)
        if weights is None:
            self.weights = [1.0] * len(losses)
        else:
            self.weights = weights
        

    def forward(self, outputs, targets):
        total_loss = 0.0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(outputs, targets)
        return total_loss

# Example usage:
# criterion = JointLoss(nn.CrossEntropyLoss(), nn.MSELoss(), weights=[0.7, 0.3])
# loss = criterion(outputs, targets)