from torchvision.ops import sigmoid_focal_loss
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.85):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, labels):
        return sigmoid_focal_loss(outputs, labels, alpha=self.alpha,
                                  gamma=self.gamma, reduction="mean")