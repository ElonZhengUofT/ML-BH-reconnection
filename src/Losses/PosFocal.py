from torchvision.ops import sigmoid_focal_loss
from torch import nn

from .PosFocusLoss import PosFocusLoss

class PosFocal(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.85):
        super(PosFocal, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, labels):
        focal_loss = sigmoid_focal_loss(outputs, labels, alpha=self.alpha,
                                        gamma=self.gamma, reduction="mean")
        pos_focus_loss = PosFocusLoss(reduction='mean')
        pos_focus_loss_value = pos_focus_loss(outputs, labels)

        return focal_loss * pos_focus_loss_value
