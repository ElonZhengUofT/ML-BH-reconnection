from torchvision.ops import sigmoid_focal_loss
from torch import nn
from Loss import Loss

class FocalMSELoss(Loss):
    def __init__(self, gamma=1.5, alpha=0.85, f_weight=0.5):
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.f_weight = f_weight

    def forward(self, outputs, labels):
        mse_loss = nn.MSELoss()  # 创建一个MSELoss实例
        loss_value = mse_loss(outputs, labels) * (1 - self.f_weight)  # 计算MSE损失
        loss_value += sigmoid_focal_loss(outputs, labels, alpha=self.alpha,
                                         gamma=self.gamma, reduction="mean") * self.f_weight
        return loss_value