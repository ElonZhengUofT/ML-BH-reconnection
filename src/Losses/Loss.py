from torch import nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, labels):
        raise NotImplementedError