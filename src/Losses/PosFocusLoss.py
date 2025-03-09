import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from Loss import Loss

class PosFocusLoss(Loss):
    """
    """
    def __init__(self, reduction='mean'):
        super(PosFocusLoss, self).__init__()
        self.reduction = reduction
        # self.dists is a 2D list of tensors, where self.dists[b][c] is
        # the distance map of the c-th class of the b-th sample
        self.dists = None

    def forward(self, outputs, labels):
        B, C, H, W = labels.shape
        loss = 0

        if self.dists is None:
            self.dists = []
            for b in range(B):
                channel_dists = []
                for c in range(C):
                    label = labels[b, c].detach().cpu().numpy()
                    dist_map = distance_transform_edt(1 - label)
                    dist_tensor = torch.tensor(dist_map, device=labels.device)
                    channel_dists.append(dist_tensor)
                self.dists.append(channel_dists)

        for b in range(B):
            for c in range(C):
                pred = outputs[b, c]
                loss_local = torch.sum(pred * self.dists[b][c])
                loss += loss_local
        if self.reduction == 'mean':
            loss /= (B * C * H * W)

        return loss


if __name__ == '__main__':
    # 测试 PosFocusLoss
    loss_fn = PosFocusLoss()

    B, C, H, W = 1, 1, 32, 32

    labels = torch.zeros(B, C, H, W)
    labels[0, 0, 3, 3] = 1
    labels[0, 0, 10, 10] = 1
    labels[0, 0, 20, 20] = 1
    labels[0, 0, 25, 25] = 1

    outputs = torch.rand(B, C, H, W)

    loss = loss_fn(outputs, labels)
    print(loss.item())

    # visualize the label output and distance map
    import matplotlib.pyplot as plt
    plt.subplot(131)
    plt.imshow(labels[0, 0].detach().cpu().numpy())
    plt.subplot(132)
    plt.imshow(outputs[0, 0].detach().cpu().numpy())
    plt.subplot(133)
    plt.imshow(loss_fn.dists[0][0].detach().cpu().numpy())
    plt.show()

    loss = loss_fn(labels, labels)
    print(loss.item())


