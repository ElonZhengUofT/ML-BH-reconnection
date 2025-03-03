import torch
import numpy as np


def iou_score(prediction, target):
    """
    计算交并比（IoU），支持torch tensor和numpy数组。

    参数:
        prediction: 模型预测结果（二值张量或数组）
        target: 真实标签（二值张量或数组）

    返回:
        交并比值
    """
    # 若输入为torch tensor，则使用torch操作
    if torch.is_tensor(target) and torch.is_tensor(prediction):
        # 计算预测和目标的交集与并集
        inter = torch.logical_and(target, prediction)
        union = torch.logical_or(target, prediction)
        # 求和得到交集与并集的像素总数，注意将torch.sum(union)转换为python数值
        iou_value = torch.sum(inter) / torch.sum(union).item()
    else:
        # 否则当输入为numpy数组时
        inter = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_value = np.sum(inter) / np.sum(union)
    return iou_value


def f_beta(precision, recall, beta):
    """
    计算f_beta分数，用于综合衡量precision和recall。

    参数:
        precision: 精确率
        recall: 召回率
        beta: beta值，控制precision与recall的权重

    返回:
        f_beta分数
    """
    # 计算分子：(1 + beta^2) * precision * recall
    numerator = (1 + beta ** 2) * precision * recall
    # 计算分母：beta^2 * precision + recall
    denominator = beta ** 2 * precision + recall
    # 使用np.divide避免分母为0的情况，返回与denominator形状一致的数组
    return np.divide(numerator, denominator, out=np.zeros_like(denominator),
                     where=(denominator != 0))


def pick_best_threshold_by_f_beta(precision, recall, thresholds, beta):
    """
    从多个阈值中挑选出f_beta分数最大的那个对应的阈值。

    参数:
        precision: 各阈值下的精确率数组
        recall: 各阈值下的召回率数组
        thresholds: 阈值数组
        beta: beta值，用于f_beta分数计算

    返回:
        三个值: 最大的f_beta分数、对应的索引、以及最佳阈值
    """
    # 根据给定的precision、recall和beta计算f_beta分数
    f_scores = f_beta(precision, recall, beta)
    # 找出f_beta分数中的最大值及其索引
    max_f_score = np.max(f_scores)
    max_f_index = np.argmax(f_scores)
    # 取得与最大f_beta分数对应的阈值
    max_f_thresh = thresholds[max_f_index]
    return max_f_score, max_f_index, max_f_thresh


def normalize(name, feature, norms):
    """
    根据特征名称对特征进行归一化处理。

    参数:
        name: 特征名称（用于判断选择哪种归一化方式）
        feature: 要归一化的特征数组
        norms: 包含各向量归一化基准的字典，键有 'E'、'B'、'v'

    返回:
        归一化后的特征数组
    """
    # 判断特征名称的前缀选择对应的归一化基准
    if name.startswith('E'):
        max_val = np.max(norms['E'])
    elif name.startswith('B'):
        max_val = np.max(norms['B'])
    else:
        max_val = np.max(feature)
    # 返回归一化结果
    return feature / max_val


def standardize(feature):
    """
    标准化特征数据，使其均值为0，标准差为1。

    参数:
        feature: 输入的特征数组

    返回:
        标准化后的特征数组
    """
    avg = np.mean(feature)  # 计算均值
    std = np.std(feature)  # 计算标准差
    return (feature - avg) / std


def euclidian(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def split_data(files, file_fraction, data_splits):
    """
    按照指定比例拆分文件列表为训练、验证和测试集。

    参数:
        files: 文件路径列表
        file_fraction: 参与拆分的文件比例（例如0.8表示只使用80%的文件）
        data_splits: 包含三个元素的元组或列表，分别表示训练、验证、测试集的比例（总和应为1）

    返回:
        三个列表，分别为训练文件、验证文件和测试文件
    """
    # 根据给定的比例确定参与拆分的文件总数
    num_files = file_fraction * len(files)
    train_split, val_split, test_split = data_splits
    # 计算各数据集的边界索引
    train_index = int(train_split * num_files)
    val_index = train_index + int(val_split * num_files)
    test_index = val_index + int(test_split * num_files)
    # 使用切片操作划分文件列表
    train_files = files[:train_index]
    val_files = files[train_index:val_index]
    test_files = files[val_index:test_index]
    return train_files, val_files, test_files

