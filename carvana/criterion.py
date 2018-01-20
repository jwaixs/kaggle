import torch

from torch import nn

class diceLoss(nn.Module):
    def __init__(self, size_average = True, invert_loss = True):
        '''
        parameters:
            size_average: normalize loss between 0 and 1. (default: True)
            invert_loss: invert loss output. (default: True)
        '''
        super(diceLoss, self).__init__()
        self.size_average = size_average
        self.invert_loss = invert_loss

    def forward(self, inputs, targets):
        size = targets.size(0)
        inputs = inputs.view(size, -1)
        targets = targets.view(size, -1)

        intersection = inputs * targets
        res = 2. * (intersection.sum(1) + 1) \
            / (inputs.sum(1) + targets.sum(1) + 1)

        if self.invert_loss:
            return (1 - 0.5 * res.sum() / size) if self.size_average else -res.sum()
        else:
            return 0.5 * res.sum() / size if self.size_average else res.sum()

class BCELossLogits2d(nn.Module):
    def __init__(self, weight = None, size_average = True):
        super(BCELossLogits2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, inputs, targets):
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(inputs_flat, targets_flat)
