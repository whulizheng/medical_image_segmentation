from torch import nn
class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection = 2.0 * (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        if inputs.sum() == 0 and targets.sum() == 0:
            return 1.
        return intersection / union
        