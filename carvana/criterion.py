import torch

from torch import nn

class diceLoss(nn.Module):
    def __init__(self, size_average = True):
        self.size_average = size_average
        super(diceLoss, self).__init__()

    def forward(self, inputs, targets):
        size = targets.size(0)
        inputs = inputs.view(size, -1)
        targets = targets.view(size, -1)

        intersection = inputs * targets
        res = 2. * (intersection.sum(1) + 1) \
            / (inputs.sum(1) + targets.sum(1) + 1)

        # Something is wrong here with the signs, check in the near future!
        return (1 - res.sum() / size) if self.size_average else -res
            
