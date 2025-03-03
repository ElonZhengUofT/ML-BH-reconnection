import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from src.utils import normalize, standardize, euclidian


class NPZDataset(Dataset):
    """
    自定义NPZ数据集。该类实现的功能与原代码一致，
    但使用了不同的变量命名和代码结构。
    """

    def __init__(self, npz_paths, feature_list, use_normalize, use_standardize,
                 binary_mode):
        """
        :param npz_paths: NPZ文件的路径列表
        :param feature_list: 需要加载的特征名称列表
        :param use_normalize: 是否对数据进行归一化处理
        :param use_standardize: 是否对数据进行标准化处理
        :param binary_mode: 是否以二值方式输出标签
        """
        self.npz_paths = npz_paths
        self.feature_list = feature_list
        self.use_normalize = use_normalize
        self.use_standardize = use_standardize
        self.binary_mode = binary_mode

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, index):
        # 加载指定索引处的NPZ文件
        sample = np.load(self.npz_paths[index])

        # 若启用归一化，则计算各向量的欧几里得模长
        if self.use_normalize:
            norm_E = euclidian(sample['Ex'], sample['Ey'], sample['Ez'])
            norm_B = euclidian(sample['Bx'], sample['By'], sample['Bz'])
            norm_dict = {
                'E': norm_E,
                'B': norm_B,
            }

        # 处理每个指定特征
        processed_features = {}
        for feat in self.feature_list:
            feat_data = sample[feat].copy()
            if self.use_normalize:
                feat_data = normalize(feat,feat_data,norm_dict)
            elif self.use_standardize:
                feat_data = standardize(feat_data)
            processed_features[feat] = feat_data

        # 将各特征堆叠成输入张量X（第一维为特征通道）
        X = np.stack([processed_features[feat] for feat in self.feature_list],
                     axis=0)

        # 根据二值模式处理标签输出
        if self.binary_mode:
            y = sample['labeled_domain'][np.newaxis, :, :]
        else:
            original_label = sample['labeled_domain']
            inverse_label = np.where(original_label, 1, 0)
            y = np.stack((original_label, inverse_label))

        return {
            'X': torch.tensor(X, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'fname': Path(self.npz_paths[index]).stem
        }
