from torchvision.ops import sigmoid_focal_loss
from torch import nn

class FocalMSELoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.85, f_weight=0.5):
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.f_weight = f_weight

    def forward(self, outputs, labels):
        return (sigmoid_focal_loss(outputs, labels, alpha=self.alpha,
                                  gamma=self.gamma, reduction="mean") * self.f_weight +
                nn.MSELoss(outputs, labels)) * (1 - self.f_weight)