import torch
import torch.nn as nn

class HausdorffLoss(nn.Module):
    def __init__(self):
        super(HausdorffLoss, self).__init__()

    def forward(self, pred_points, gt_points):
        """
        :param pred_points: Tensor of shape (N, M, 2)  # 预测点集 (batch, num_pred, 2)
        :param gt_points: Tensor of shape (N, K, 2)    # 真实点集 (batch, num_gt, 2)
        :return: Hausdorff Loss
        """
        # 计算 pairwise 欧几里得距离 (N, M, K)
        d_matrix = torch.cdist(pred_points, gt_points, p=2)

        # 每个 GT 点到最近预测点的距离 (N, K)
        d_s_t = torch.min(d_matrix, dim=1)[0]
        # 每个预测点到最近 GT 点的距离 (N, M)
        d_t_s = torch.min(d_matrix, dim=2)[0]

        # 平均化
        loss = d_s_t.mean() + d_t_s.mean()
        return loss


class SoftHausdorffLoss(HausdorffLoss):
    def __init__(self, alpha=5.0):
        super(SoftHausdorffLoss, self).__init__()
        self.alpha = alpha  # 控制 softmin 平滑度

    def forward(self, pred_points, gt_points):
        """
        :param pred_points: Tensor of shape (N, M, 2)  # 预测点集
        :param gt_points: Tensor of shape (N, K, 2)    # 真实点集
        :return: Soft Hausdorff Loss
        """
        d_matrix = torch.cdist(pred_points, gt_points, p=2)  # 计算 pairwise 欧几里得距离

        # Softmin 替代硬最小值
        softmin_s_t = -torch.logsumexp(-self.alpha * d_matrix, dim=1)
        softmin_t_s = -torch.logsumexp(-self.alpha * d_matrix, dim=2)

        loss = softmin_s_t.mean() + softmin_t_s.mean()
        return loss
