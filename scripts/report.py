#!/usr/bin/env python
import os
import sys
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import gif
from sklearn import metrics
import src.utils as utils
from src.utils import iou_score, pick_best_threshold


################################################################################
# 绘制重连点图像
################################################################################
def report_reconnection_points(file):
    """
    根据输入的NPZ文件，绘制各向异性图并标记重连点。

    参数:
      file: NPZ文件路径，文件中包含xmin、xmax、zmin、zmax、anisotropy以及labeled_domain等数据

    返回:
      earth_center_x: 根据x坐标数组计算出的地球中心x位置索引
      data['xmin'], data['xmax'], data['zmin'], data['zmax']: 图像的坐标范围
    """
    # 加载数据文件
    data = np.load(file)

    # 创建画布，并设置图像大小
    fig = plt.figure(figsize=(10, 6))

    # 根据anisotropy数组的尺寸构造x和z方向的坐标
    xx = np.linspace(data['xmin'], data['xmax'], data['anisotropy'].shape[1])
    zz = np.linspace(data['zmin'], data['zmax'], data['anisotropy'].shape[0])

    # 找出标记区域的非零索引
    labeled_indices = data['labeled_domain'].nonzero()
    labeled_z = zz[labeled_indices[0]]
    labeled_x = xx[labeled_indices[1]]

    # 添加子图，并显示anisotropy数据
    ax = fig.add_subplot()
    c = ax.imshow(data['anisotropy'],
                  extent=[data['xmin'], data['xmax'], data['zmin'],
                          data['zmax']])

    # 在图上标记出重连点（以红色叉号表示）
    ax.scatter(labeled_x, labeled_z, marker='x', color='red')
    ax.set_title('Pseudocolor-Anisotropy with reconnection points', fontsize=16)
    ax.set_xlabel('x/Re', fontsize=12)
    ax.set_ylabel('z/Re', fontsize=12)
    fig.colorbar(c, ax=ax)

    # 保存图像，并关闭画布释放资源
    fig.savefig('reconnection_points.png', bbox_inches='tight')
    plt.close(fig)

    # 计算x方向最接近0的索引，视为地球中心x坐标
    earth_center_x = np.argmin(np.abs(xx))

    return earth_center_x, data['xmin'], data['xmax'], data['zmin'], data[
        'zmax']


################################################################################
# 绘制预测与真实值的对比图
################################################################################
def report_comparison(preds, truth, file, epoch):
    """
    绘制预测结果与真实标签的对比图，并保存为文件。

    参数:
      preds: 模型预测结果数组
      truth: 真实标签数组
      file: 保存图像的文件路径
      epoch: 当前训练轮次（用于图像标题）
    """
    # 创建画布并设置尺寸
    fig = plt.figure(figsize=(12, 8))

    # 分为上下两个子图，上图显示预测结果，下图显示真实标签
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # 展示预测结果
    c1 = ax1.imshow(preds)
    fig.colorbar(c1, ax=ax1)
    ax1.set_title(f'Preds, epoch {epoch}')

    # 展示真实标签
    c2 = ax2.imshow(truth)
    fig.colorbar(c2, ax=ax2)
    ax2.set_title('Truth')

    # 保存图像到指定文件
    plt.savefig(file)
    plt.close(fig)


################################################################################
# 生成几何级数序列（用于gif动画帧选择）
################################################################################
def generate_geom_seq(num_epochs):
    """
    生成一个几何级数序列，其终值不超过num_epochs，用于选择动画帧。

    参数:
      num_epochs: 总训练轮数

    返回:
      序列列表
    """
    seq = [1]
    i = 1
    step = 1
    # 不断累加step直到达到或超过num_epochs
    while True:
        if i % 10 == 0:
            step += 1
        seq.append(seq[-1] + step)
        if seq[-1] >= num_epochs:
            break
        i += 1
    if seq[-1] > num_epochs:
        seq.pop()
    return seq


################################################################################
# 绘制gif动画中的单帧
################################################################################
@gif.frame
def report_gif_frame(preds, truth, epoch, xmin, xmax, zmin, zmax):
    """
    绘制一帧gif动画，显示预测结果，并标记图像坐标范围。

    参数:
      preds: 预测结果数组
      truth: 真实标签数组（用于提取标记位置）
      epoch: 当前训练轮次
      xmin, xmax, zmin, zmax: 图像的坐标范围
    """
    fig = plt.figure(figsize=(5, 3), dpi=100)

    # 构造x和z坐标轴
    xx = np.linspace(xmin, xmax, truth.shape[1])
    zz = np.linspace(zmin, zmax, truth.shape[0])

    # 找出truth中非零的点，作为标记位置（如果需要可用于后续展示）
    labeled_indices = truth.nonzero()
    labeled_z = zz[labeled_indices[0]]
    labeled_x = xx[labeled_indices[1]]

    ax = fig.add_subplot()
    c = ax.imshow(preds, extent=[xmin, xmax, zmin, zmax])
    ax.set_title(f'Epoch {epoch}')
    ax.set_xlabel('x/Re')
    ax.set_ylabel('z/Re')
    plt.tight_layout()
    # 返回当前帧
    return fig


################################################################################
# 绘制训练和验证损失曲线
################################################################################
def report_loss(train_losses, val_losses, lr_history, outdir):
    """
    绘制训练与验证损失曲线，并在学习率变化处添加垂直参考线。

    参数:
      train_losses: 训练损失列表
      val_losses: 验证损失列表
      lr_history: 学习率变化记录（字典，key为epoch）
      outdir: 输出保存路径
    """
    # 从第4个epoch开始绘制
    x = range(4, len(train_losses) + 1)

    # 设置y轴使用科学计数法显示
    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    plt.gca().yaxis.set_major_
