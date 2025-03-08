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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import *
from matplotlib.colors import LogNorm



################################################################################
# 绘制重连点图像
################################################################################
def report_reconnection_points(file):
    """
    根据输入的NPZ文件，绘制各向异性图并标记重连点。
    """
    # 加载数据文件
    data = np.load(file)

    # 创建画布，并设置图像大小
    fig = plt.figure(figsize=(12,4))

    L = 0.3
    N_grid = 5000
    x = np.linspace(-L / 0.6, L / 0.6, N_grid)
    y = np.linspace(-L / 3.0, L / 3.0, N_grid)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 找出标记区域的非零索引
    labeled_indices = data['labels'].nonzero()
    labeled_x = x[labeled_indices[0]]
    labeled_y = y[labeled_indices[1]]

    x_min, x_max = -L / 0.6, L / 0.6
    y_min, y_max = -L / 3.0, L / 3.0

    b1 = data['b1']
    b2 = data['b2']
    b3 = data['b3']

    B_magnitude = np.sqrt(b1 ** 2 + b2 ** 2 + b3 ** 2)
    B_magnitude[B_magnitude <= 1e-7] = 1e-7

    # 添加子图，并显示anisotropy数据
    ax = fig.add_subplot()

    c = ax.imshow(
        B_magnitude,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',  # 保证网格 y[0] 在下
        norm=LogNorm(vmin=1e-1, vmax=1e1),  # 你可根据数据分布调节
        cmap='inferno',  # 或者 'viridis', 'plasma' 等
        aspect='auto'  # 或 'equal'
    )

    # 在图上标记出重连点（以红色叉号表示）
    ax.scatter(labeled_x, labeled_y, marker='o', color='red')
    ax.set_title('Pseudocolor-Anisotropy with reconnection points', fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    fig.colorbar(c, ax=ax)

    # 保存图像，并关闭画布释放资源
    fig.savefig('reconnection_points.png', bbox_inches='tight')
    plt.close(fig)

    # 计算x方向最接近0的索引，视为地球中心x坐标
    earth_center_x = np.argmin(np.abs(x))

    return earth_center_x, x_min, x_max, y_min, y_max


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

    # 展示预测结果, mark groud truth reconnection points with a color won't cover the prediction
    labeled_indices = truth.nonzero()
    labeled_x = labeled_indices[0]
    labeled_y = labeled_indices[1]
    c1 = ax1.imshow(preds)
    ax1.scatter(labeled_y, labeled_x, marker='o', color='red')
    fig.colorbar(c1, ax=ax1)
    ax1.set_title(f'Preds, epoch {epoch}')

    # 展示真实标签, mark the reconnection points with red circles
    labeled_indices = truth.nonzero()
    labeled_x = labeled_indices[0]
    labeled_y = labeled_indices[1]
    c2 = ax2.imshow(truth)
    ax2.scatter(labeled_y, labeled_x, marker='o', color='red')
    ax2.set_title('Truth')
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
    plt.gca().yaxis.set_major_formatter(fmt)

    # 绘制训练和验证损失曲线
    plt.plot(x, train_losses[3:], label='Train Loss')
    plt.plot(x, val_losses[3:], label='Validation Loss')

    if lr_history:
        ymin, ymax = plt.gca().get_ylim()
        plt.vlines(lr_history[1:], ymin, ymax, linestyles='dashed', colors='gray')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'loss_curve.png'))
    plt.close()

################################################################################
# 绘制ROC曲线
################################################################################
def report_roc(preds, truth, outdir):
    """
    绘制接收者工作特征（ROC）曲线，并保存图像。

    参数:
      preds: 模型预测结果（展平后）
      truth: 真实标签（展平后）
      outdir: 保存图像的目录
    """
    fpr, tpr, _ = metrics.roc_curve(truth, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:0.2f}')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(os.path.join(outdir, 'roc_curve.png'))
    plt.close()


################################################################################
# 绘制Precision-Recall曲线，并标记出F1和F2最佳阈值点
################################################################################
def report_precision_recall(precision, recall, max_f1_score, max_f1_index,
                            max_f1_thresh, max_f2_score, max_f2_index,
                            max_f2_thresh, outdir):
    """
    绘制Precision-Recall曲线，同时标出F1和F2最佳分数对应的点及其阈值。

    参数:
      precision: 精确率数组
      recall: 召回率数组
      max_f1_score: F1最佳分数
      max_f1_index: F1最佳分数对应的索引
      max_f1_thresh: F1最佳分数对应的阈值
      max_f2_score: F2最佳分数
      max_f2_index: F2最佳分数对应的索引
      max_f2_thresh: F2最佳分数对应的阈值
      outdir: 输出目录
    """
    plt.title('Precision Recall')
    plt.plot(recall, precision, marker='.', markersize=2, label='U-Net')
    plt.plot(recall[max_f1_index], precision[max_f1_index], marker='.',
             color='tab:green', markersize=12,
             label=f'Max F1 = {max_f1_score:.4f}\nThreshold = {max_f1_thresh:.4f}')
    plt.plot(recall[max_f2_index], precision[max_f2_index], marker='^',
             color='tab:red', markersize=7,
             label=f'Max F2 = {max_f2_score:.4f}\nThreshold = {max_f2_thresh:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', direction='in')
    plt.savefig(os.path.join(outdir, 'precision_recall.png'))
    plt.close()


################################################################################
# 绘制不同阈值下的Precision和Recall曲线
################################################################################
def report_thresholds(precision, recall, thresholds, max_f1_thresh, max_f2_thresh,
                      outdir):
    """
    绘制阈值与Precision/Recall的关系曲线，并标出F1和F2最佳阈值位置。

    参数:
      precision: 精确率数组
      recall: 召回率数组
      thresholds: 阈值数组
      max_f1_thresh: F1最佳阈值
      max_f2_thresh: F2最佳阈值
      outdir: 输出目录
    """
    plt.title('Precision & Recall with Different Thresholds')
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.axvline(max_f1_thresh, ymin=0.04, ymax=0.96, ls='--', c='gray',
                label='Max F1 Threshold')
    plt.axvline(max_f2_thresh, ymin=0.04, ymax=0.96, ls='-.', c='black',
                label='Max F2 Threshold')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', direction='in')
    plt.savefig(os.path.join(outdir, 'thresholds.png'))
    plt.close()


################################################################################
# 绘制混淆矩阵
################################################################################
def report_confusion_matrix(binary_preds, truth, score, outdir):
    """
    根据二值预测和真实标签计算混淆矩阵，并将结果以对数尺度显示后保存。

    参数:
      binary_preds: 二值化后的预测结果
      truth: 真实标签
      score: 指标名称（用于文件命名，如'f1'或'f2'）
      outdir: 输出目录
    """
    plt.title('Confusion Matrix')
    cm = metrics.confusion_matrix(truth, binary_preds).T
    plt.imshow(cm, norm=mpl.colors.LogNorm())
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    # 在每个矩阵单元中添加数字标注
    for i in [0, 1]:
        for j in [0, 1]:
            text_color = 'yellow' if cm[i, j] < np.max(cm) / 2 else 'black'
            plt.annotate(cm[i, j], (i, j), va='center', ha='center',
                         color=text_color)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.colorbar()
    plt.savefig(os.path.join(outdir, f'confusion_matrix_{score}.png'))
    plt.close()


################################################################################
# 评估分类器性能指标
################################################################################
def evaluate_classifier(preds, truth):
    """
    计算混淆矩阵、准确率、精确率、召回率、特异度、AUC、IoU等分类指标。

    参数:
      preds: 二值预测结果（展平后）
      truth: 真实标签（展平后）

    返回:
      包含各项指标的字典
    """
    tn, fp, fn, tp = metrics.confusion_matrix(truth, preds).ravel()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    auc_roc = metrics.roc_auc_score(y_score=preds, y_true=truth)
    iou = utils.iou_score(preds, truth)

    return {
        'True Positive': int(tp),
        'True Negative': int(tn),
        'False Positive': int(fp),
        'False Negative': int(fn),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall/Sensitivity': recall,
        'Specificity': tn / (fp + tn),
        'AUC ROC': auc_roc,
        'IoU': iou
    }


################################################################################
# 主程序入口
################################################################################
if __name__ == '__main__':
    # 设置命令行参数解析
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--dir', required=True, type=str,
                            help="结果文件所在目录")
    arg_parser.add_argument('-g', '--gif', action='store_true',
                            help="是否生成gif动画")
    arg_parser.add_argument('-m', '--modeldir', help="模型文件目录")
    # f 表示固定阈值（若需要）
    arg_parser.add_argument('-f')
    args = arg_parser.parse_args()

    ############################################################################
    # 绘制各向异性图及重连点
    ############################################################################
    # 这里使用一个示例数据文件 'sample/data/3600.npz'
    earth_center_x, xmin, xmax, zmin, zmax = report_reconnection_points('smaller_data/data_time_0032_smaller.npz')


    ############################################################################
    # 读取元数据
    ############################################################################
    if args.modeldir:
        # 若提供了模型目录，则从该目录中读取元数据
        metadata_path = os.path.join(args.modeldir, 'metadata.json')
    else:
        metadata_path = os.path.join(args.dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    ############################################################################
    # 绘制损失曲线（仅在未指定模型目录时）
    ############################################################################
    if not args.modeldir:
        if 'lr_history' in metadata.keys():
            lr_history = [int(epoch) for epoch in metadata['lr_history'].keys()]
        else:
            lr_history = None
        report_loss(metadata['train_losses'], metadata['val_losses'], lr_history,
                  args.dir)

        ############################################################################
        # 若指定生成gif动画，则按几何序列生成验证结果动画帧
        ############################################################################

        if args.gif:
            frames = []
            fname = Path(metadata['val_files'][0]).stem
            sequence = generate_geom_seq(metadata['last_epoch'])
            for i in sequence:
                # 加载对应epoch的预测结果与真实标签
                data = np.load(
                    os.path.join(args.dir, 'val', str(i), f'{fname}.npz'))
                preds, truth = data['outputs'], data['labels']
                frame = report_gif_frame(preds, truth, i, xmin, xmax, zmin, zmax)
                frames.append(frame)
            gif.save(frames, os.path.join(args.dir, 'epochs.gif'), duration=100)

    ############################################################################
    # 处理测试集预测结果
    ############################################################################
    test_list = glob(os.path.join(args.dir, 'test', '*.npz'))
    num_test_files = len(test_list)
    all_preds = np.zeros(
        (num_test_files, metadata['args']['height'], metadata['args']['width']))
    all_truth = np.zeros(
        (num_test_files, metadata['args']['height'], metadata['args']['width']))
    # 遍历所有测试文件，收集预测与真实结果，并保存对比图
    for i, test_file in enumerate(test_list):
        data = np.load(test_file)
        preds, truth = data['outputs'], data['labels']
        print(f"shape of preds: {preds.shape}")
        print(f"shape of truth: {truth.shape}")
        if args.modeldir:
            test_plot_file = os.path.join(args.dir, 'test',
                                          Path(test_file).stem)
            plot_comparison(preds, truth, test_plot_file,
                            metadata['best_epoch'])
        all_preds[i] = preds
        all_truth[i] = truth

    # 根据地球中心划分夜侧和日侧数据
    nightside_preds = all_preds[:, :, :earth_center_x]
    nightside_truth = all_truth[:, :, :earth_center_x]
    dayside_preds = all_preds[:, :, earth_center_x:]
    dayside_truth = all_truth[:, :, earth_center_x:]

    ############################################################################
    # 分别对整体、夜侧和日侧计算Precision-Recall及混淆矩阵
    ############################################################################
    f1 = {}
    f2 = {}
    for preds, truth, side in [
        (all_preds.ravel(), all_truth.ravel(), 'both_sides'),
        (nightside_preds.ravel(), nightside_truth.ravel(), 'nightside'),
        (dayside_preds.ravel(), dayside_truth.ravel(), 'dayside')
    ]:
        # if there is no positive in truth, skip, and set f1 and f2 to 0
        if len(np.where(truth == 1)[0]) == 0:
            continue
        side_dir = os.path.join(args.dir, side)
        os.makedirs(side_dir, exist_ok=True)

        # find where is positive in truth
        where_positive = np.where(truth == 1)
        print(f"where_positive is {where_positive}")
        print(f"truth is {truth}")
        print(f"truth has {np.unique(truth)}")
        # 计算Precision-Recall曲线
        precision, recall, thresholds = metrics.precision_recall_curve(truth,
                                                                       preds, pos_label=1)
        d = {'precision': precision, 'recall': recall}
        np.savez(os.path.join(side_dir, 'precision_recall.npz'), **d)

        # 根据F1得分确定最佳阈值
        max_f1_score, max_f1_index, max_f1_thresh = pick_best_threshold_by_f_beta(
            precision, recall, thresholds, 1)
        f1[side] = {'score': max_f1_score, 'threshold': max_f1_thresh}
        binary_preds = np.where(preds < max_f1_thresh, 0, 1)
        report_confusion_matrix(binary_preds, truth, 'f1', side_dir)

        # 根据F2得分确定最佳阈值
        max_f2_score, max_f2_index, max_f2_thresh = pick_best_threshold_by_f_beta(
            precision, recall, thresholds, 2)
        f2[side] = {'score': max_f2_score, 'threshold': max_f2_thresh}
        binary_preds = np.where(preds < max_f2_thresh, 0, 1)
        report_confusion_matrix(binary_preds, truth, 'f2', side_dir)

        # 绘制Precision-Recall曲线及阈值曲线
        report_precision_recall(precision, recall, max_f1_score, max_f1_index,
                                max_f1_thresh,
                                max_f2_score, max_f2_index, max_f2_thresh,
                                side_dir)
        report_thresholds(precision, recall, thresholds, max_f1_thresh,
                          max_f2_thresh, side_dir)

    if 'dayside' in f1:
        binary_preds_f1_nightside = np.where(
            nightside_preds < f1['nightside']['threshold'], 0, 1)
        binary_preds_f1_dayside = np.where(
            dayside_preds < f1['dayside']['threshold'], 0, 1)
        all_binary_preds = np.concatenate(
            (binary_preds_f1_nightside, binary_preds_f1_dayside), axis=2)
    else:
        print(
            "dayside f1 threshold not available, using nightside only for f1.")
        all_binary_preds = np.where(
            nightside_preds < f1['nightside']['threshold'], 0, 1)

    report_confusion_matrix(all_binary_preds.ravel(), all_truth.ravel(),
                            'f1', args.dir)

    if 'dayside' in f2:
        binary_preds_f2_nightside = np.where(
            nightside_preds < f2['nightside']['threshold'], 0, 1)
        binary_preds_f2_dayside = np.where(
            dayside_preds < f2['dayside']['threshold'], 0, 1)
        all_binary_preds = np.concatenate(
            (binary_preds_f2_nightside, binary_preds_f2_dayside), axis=2)
    else:
        print(
            "dayside f2 threshold not available, using nightside only for f2.")
        all_binary_preds = np.where(
            nightside_preds < f2['nightside']['threshold'], 0, 1)

    report_confusion_matrix(all_binary_preds.ravel(), all_truth.ravel(),
                            'f2', args.dir)

    ############################################################################
    # 绘制ROC曲线，并输出分类指标
    ############################################################################
    report_roc(all_preds.ravel(), all_truth.ravel(), args.dir)
    metrics_dict = evaluate_classifier(all_binary_preds.ravel(),
                                       all_truth.ravel())
    metrics_dict['F1'] = f1
    metrics_dict['F2'] = f2
    print(json.dumps(metrics_dict, indent=2))

    # 将计算得到的指标保存到metrics.json文件中
    with open(os.path.join(args.dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)