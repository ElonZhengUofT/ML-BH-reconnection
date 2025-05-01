from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.utils import normalize, standardize, euclidian
from src.utils.TwoDGaussian import gaussianize_image
from scipy.ndimage import sobel

def compute_grad_b(b1, b2, b3):
    # 对每个分量求梯度
    grad_sq = 0
    for b in [b1, b2, b3]:
        dx = sobel(b, axis=1, mode='constant')  # x方向（列）
        dy = sobel(b, axis=0, mode='constant')  # y方向（行）
        grad_sq += dx**2 + dy**2
    grad_mag = np.sqrt(grad_sq + 1e-8)
    return grad_mag.astype(np.float32)

class NPZDataset(Dataset):
    """
    自定义NPZ数据集。该类实现的功能与原代码一致，
    但使用了不同的变量命名和代码结构。
    """

    def __init__(self, npz_paths, feature_list, use_normalize, use_standardize,
                 binary_mode, gaussian_noise, grad_b_on=True):
        """
        :param npz_paths: NPZ文件的路径列表
        :param feature_list: 需要加载的特征名称列表
        :param use_normalize: 是否对数据进行归一化处理
        :param use_standardize: 是否对数据进行标准化处理
        :param binary_mode: 是否以二值方式输出标签
        """
        self.npz_paths = npz_paths
        self._in_feature_list = feature_list.copy()
        self.use_normalize = use_normalize
        self.use_standardize = use_standardize
        self.binary_mode = binary_mode
        self.gaussian_noise = gaussian_noise
        self.grad_b_on = grad_b_on
        self.feature_list = feature_list.copy()

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, index):
        # 加载指定索引处的NPZ文件
        sample = np.load(self.npz_paths[index])

        # 若启用归一化，则计算各向量的欧几里得模长
        if self.use_normalize:
            norm_E = euclidian(sample['e1'], sample['e2'], sample['e3'])
            norm_B = euclidian(sample['b1'], sample['b2'], sample['b3'])
            norm_dict = {
                'e': norm_E,
                'b': norm_B,
            }


        # 处理每个指定特征
        processed_features = {}
        for feat in self._in_feature_list:
            feat_data = sample[feat].copy()
            if self.use_normalize:
                feat_data = normalize(feat,feat_data,norm_dict)
            elif self.use_standardize:
                feat_data = standardize(feat_data)
            #elif self.use_bias:
            #     if feat.startswith('b'):
            #         feat_data = feat_data + np.mean(feat_data)
            processed_features[feat] = feat_data
            if feat == 'b3' and self.grad_b_on:
                grad_b = compute_grad_b(
                    processed_features['b1'],
                    processed_features['b2'],
                    processed_features['b3']
                )
                grad_b = (grad_b - np.mean(grad_b)) / np.std(grad_b)
                grad_b_reciprocal = 1 / (grad_b + 1e-8)
                processed_features['gradB'] = grad_b_reciprocal

        # 将各特征堆叠成输入张量X（第一维为特征通道）
        if 'gradB' in processed_features.keys():
            X = np.stack([processed_features[feat] for feat in
                          self._in_feature_list + ['gradB']],
                         axis=0)
            print(f"gradB is added")
            self.feature_list.append('gradB')
        else:
            X = np.stack([processed_features[feat] for feat in self._in_feature_list],
                         axis=0)

        # 根据二值模式处理标签输出
        if self.binary_mode:
            if self.gaussian_noise:
                y_pre = sample['labels'].astype(np.float32)
                # 交换x和y轴
                y_pre = np.swapaxes(y_pre, 0, 1)
                # y轴对称翻转
                y_pre = np.flip(y_pre, axis=0)
                y = 0.9 * gaussianize_image(y_pre,sigma=25)[np.newaxis, :, :] + 0.1 * gaussianize_image(y_pre,sigma=5)[np.newaxis, :, :]
                # scale the blurred image to [0, 1]
                y = y / np.max(y)
            else:
                y_pre = sample['labels']
                y_pre = np.swapaxes(y_pre, 0, 1)
                y_pre = np.flip(y_pre, axis=0).copy()
                y = y_pre[np.newaxis, :, :]
            label_pre = sample['labels']
            label_pre = np.swapaxes(label_pre, 0, 1)
            label_pre = np.flip(label_pre, axis=0).copy()
            label = label_pre[np.newaxis, :, :]

        else:
            original_label = sample['labels']
            inverse_label = np.where(original_label, 1, 0)
            y = np.stack((original_label, inverse_label))

        return {
            'X': torch.tensor(X, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'fname': Path(self.npz_paths[index]).stem
        }

if __name__ == "__main__":
    npz_paths = ["/Users/zhengshizhao/PycharmProjects/ML-BH-reconnection/smaller_data/data_time_0032_smaller.npz"]
    feature_list = ['e1', 'e2', 'e3', 'b1', 'b2', 'b3', 'p', 'rho']
    use_normalize = False
    use_standardize = False
    binary_mode = True
    gaussian_noise = False
    sample_npz = NPZDataset(npz_paths, feature_list, use_normalize, use_standardize, binary_mode, gaussian_noise)
    # 在图像中央设置一个目标点
    sample = sample_npz[0]['y'][0].numpy()

    where_positive = np.where(sample == 1)
    print(f"Positive labels: {len(where_positive[0])}")

    import matplotlib.pyplot as plt
    plt.imshow(sample, cmap='gray')
    plt.show()

    gaussian_noise = True
    sample_npz = NPZDataset(npz_paths, feature_list, use_normalize, use_standardize, binary_mode, gaussian_noise)
    # 在图像中央设置一个目标点
    sample = sample_npz[0]['y'][0].numpy()

    where_positive = np.where(sample > 0)
    print(f"Positive labels: {len(where_positive[0])}")

    import matplotlib.pyplot as plt
    plt.imshow(sample, cmap='gray')
    plt.show()

    X = sample_npz[0]['X']
    print(X.shape)

