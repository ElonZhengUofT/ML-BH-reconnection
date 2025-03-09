#!/usr/bin/env python
import os
import json
import torch
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import sys
import os

# 导入自定义模块和第三方包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import *
from report import report_comparison
from ptflops import get_model_complexity_info
import netron
import torch.onnx
from torchvision.ops import sigmoid_focal_loss


def train(model, train_loader, device, criterion, optimizer, scheduler,
          early_stopping, val_loader, epochs, lr, binary, outdir):
    """
    模型训练函数，遍历多个epoch进行训练和验证，自动保存最优模型，并根据验证损失调整学习率和进行早停。

    参数:
      Model: 待训练的模型
      train_loader: 训练数据加载器
      device: 运行设备（'cpu' 或 'cuda'）
      criterion: 损失函数
      optimizer: 优化器
      scheduler: 学习率调度器
      early_stopping: 早停回调函数
      val_loader: 验证数据加载器
      epochs: 总训练轮数
      lr: 初始学习率
      binary: 是否为二分类任务
      outdir: 输出文件夹路径

    返回:
      best_model, best_epoch, epoch, lr_history, train_losses, val_losses
    """
    train_losses = []
    val_losses = []
    lr_change_epoch = 0
    lr_history = {lr_change_epoch: lr}
    best_val_loss = np.inf  # 初始化最佳验证损失为无穷大

    # 开始训练每个epoch
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_count = 0
        print('Epoch:', epoch)

        # 使用 tqdm 展示训练进度
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            # 从当前批次中提取输入和标签
            inputs = data['X'].to(device)
            labels = data['y'].to(device)

            optimizer.zero_grad()  # 清空梯度

            # 前向传播计算输出
            outputs = model(
                inputs)  # 输出形状 (batch_size, n_classes, img_cols, img_rows)

            # 计算仅在非地球区域的损失
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            loss_value = loss.item()
            total_loss += loss_value
            count = labels.size(0)
            total_count += count

            # 通过tqdm实时显示当前批次损失
            tqdm.write(f"Loss: {loss_value / count:.7f}")

        avg_train_loss = total_loss / total_count
        print(f'Training loss: {avg_train_loss}')
        train_losses.append(avg_train_loss)

        # 为当前epoch的验证结果创建输出目录
        val_dir = os.path.join(outdir, 'val', str(epoch))
        os.makedirs(val_dir, exist_ok=True)

        # 调用evaluate函数计算验证损失
        val_loss = evaluate(model, val_loader, device, criterion, val_dir,
                            epoch, binary, mode='val')
        val_losses.append(val_loss)
        print('Validation loss:', val_loss)

        # 如果当前验证损失更低，则保存模型和优化器状态
        if val_loss < best_val_loss:
            model_path = os.path.join(outdir, 'unet_best_epoch.pt')
            opt_path = os.path.join(outdir, 'optimizer_best_epoch.pt')
            if args.gpus:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), opt_path)
            best_val_loss = val_loss
            best_model = model
            best_epoch = epoch

        # 调整学习率
        scheduler.step(val_loss)
        # 执行早停检查
        early_stopping(val_loss)

        # 检查是否学习率有下降，若下降则恢复最优模型权重
        [last_lr] = scheduler._last_lr
        if last_lr < lr_history[lr_change_epoch]:
            lr_change_epoch = epoch
            lr_history[lr_change_epoch] = last_lr
            print('Restoring best Model weights.')
            if args.gpus:
                model.module.load_state_dict(
                    torch.load(os.path.join(outdir, 'unet_best_epoch.pt'),
                               map_location=device))
            else:
                model.load_state_dict(
                    torch.load(os.path.join(outdir, 'unet_best_epoch.pt'),
                               map_location=device))
            optimizer.load_state_dict(
                torch.load(os.path.join(outdir, 'optimizer_best_epoch.pt'),
                           map_location='cpu'))

        # 当早停触发时退出训练循环
        if early_stopping.should_stop:
            print("Early stopping triggered, breaking training loop.")
            break

    return best_model, best_epoch, epoch, lr_history, train_losses, val_losses


def evaluate(model, data_loader, device, criterion, outdir, epoch, binary,
             mode):
    """
    模型评估函数，用于在验证或测试阶段计算模型损失和保存预测结果。

    参数:
      Model: 待评估的模型
      data_loader: 数据加载器（验证或测试数据）
      device: 运行设备（'cpu' 或 'cuda'）
      criterion: 损失函数
      outdir: 输出结果的文件夹路径
      epoch: 当前轮次（用于保存文件时标记）
      binary: 是否为二分类任务
      mode: 模式标记，'val' 或 'test'

    返回:
      平均损失值
    """
    model.eval()  # 设定模型为评估模式

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for i, data in tqdm(enumerate(data_loader),
                            desc=f"Evaluating epoch {epoch}"):
            inputs = data['X'].to(device)
            labels = data['y'].to(device)
            fname = data['fname']

            outputs = model(inputs)
            loss = criterion(outputs, labels).item()
            total_loss += loss

            batch_size = labels.size(0)
            total_count += batch_size
            width = labels.size(2)
            height = labels.size(3)

            # 根据任务类型计算正确预测数
            if binary:
                threshold_outputs = torch.where(outputs > 0.5, 1, 0)
                correct = (threshold_outputs == labels[:, 0]).sum().item()
            else:
                _, outputs = outputs.max(1)  # 选择概率最大的类别
                correct = (outputs == labels[:, 0]).sum().item()

            # 对测试模式或者验证集第一个批次，保存预测与真实对比图
            if mode == 'test' or i == 0:
                num_plots = 1 if mode == 'val' else batch_size
                for n in range(num_plots):
                    preds_np = outputs[n].detach().cpu().numpy().squeeze()
                    truth_np = labels[n, 0].detach().cpu().numpy().squeeze()
                    plot_file = os.path.join(outdir, f'{fname[n]}.png')
                    report_comparison(preds=preds_np, truth=truth_np,
                                    file=plot_file, epoch=epoch)

                    # 将预测结果和真实标签保存为npz文件
                    results = {'outputs': preds_np, 'labels': truth_np}
                    np.savez(os.path.join(outdir, f'{fname[n]}.npz'), **results)

            # 在进度条中显示当前批次的损失和准确率
            tqdm.write(
                f"Loss: {loss / batch_size:.7f}, Accuracy: {correct / (batch_size * width * height):.7f}")

    avg_loss = total_loss / total_count
    return avg_loss


def visualize_model(model):
    """
    将模型导出为ONNX格式，并使用Netron进行可视化。
    """
    x = torch.randn(1, 6, 344, 620, requires_grad=True)
    modelData = "./demo.pth"
    torch.onnx.export(
        model,
        x,
        modelData,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    netron.start(modelData)


if __name__ == '__main__':
    # 设置PYTHONPATH环境变量
    os.environ["PYTHONPATH"] = "/content/ML-BH-reconnection"

    # 解析命令行参数
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--indir', required=True, type=str)
    arg_parser.add_argument('-o', '--outdir', required=True, type=str)
    arg_parser.add_argument('-f', '--file-fraction', default=1.0, type=float)
    arg_parser.add_argument('-d', '--data-splits', default=[0.8, 0.1, 0.1],
                            nargs='+', type=float)
    arg_parser.add_argument('-e', '--epochs', default=10, type=int)
    arg_parser.add_argument('-b', '--batch-size', default=2, type=int)
    arg_parser.add_argument('-l', '--learning-rate', default=1.e-5, type=float)
    arg_parser.add_argument('-c', '--num-classes', default=1, type=int)
    arg_parser.add_argument('-k', '--kernel-size', default=3, type=int)
    arg_parser.add_argument('-y', '--height', default=5000, type=int)
    arg_parser.add_argument('-x', '--width', default=625, type=int)
    arg_parser.add_argument('-n', '--normalize', action='store_true')
    arg_parser.add_argument('-s', '--standardize', action='store_true')
    arg_parser.add_argument('-g', '--gpus', nargs='+',
                            help='GPUs to run on in the form 0 1 etc.')
    arg_parser.add_argument('-w', '--num-workers', default=0, type=int)
    # arg_parser.add_argument('--rho', action='store_true')
    arg_parser.add_argument('--model', default='ViTUnet', type=str)
    arg_parser.add_argument('--loss', default='focal', type=str)
    args = arg_parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.outdir, exist_ok=True)

    # 获取指定目录下所有NPZ文件，并拆分为训练、验证和测试集
    script_dir = os.path.dirname(
        os.path.abspath(__file__))  # 获取 train.py 所在的 scripts 目录
    project_root = os.path.abspath(os.path.join(script_dir, ".."))  # 回到项目根目录
    data_dir = os.path.join(project_root, args.indir)
    print(f"Checking input directory: {data_dir}")
    files = glob(os.path.join(data_dir, "*.npz"))
    print(f'Found {len(files)} files in {args.indir}')
    train_files, val_files, test_files = split_data(files, args.file_fraction,
                                                    args.data_splits)
    print(len(train_files), 'train files:', train_files)
    print(len(val_files), 'val files:', val_files)
    print(len(test_files), 'test files:', test_files)

    # 定义所需特征列表
    features = ['b1', 'b2', 'b3', 'e1', 'e2', 'e3', 'rho', 'p']
    print(len(features), 'features:', features)

    binary = args.num_classes == 1

    # 初始化训练、验证和测试数据集及其加载器
    train_dataset = NPZDataset(train_files, features, args.normalize,
                               args.standardize, binary)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               drop_last=True,
                                               num_workers=args.num_workers)

    val_dataset = NPZDataset(val_files, features, args.normalize,
                             args.standardize, binary)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             drop_last=False,
                                             num_workers=args.num_workers)

    test_dataset = NPZDataset(test_files, features, args.normalize,
                              args.standardize, binary)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              drop_last=False,
                                              num_workers=args.num_workers)

################################################################################
    # Choose the model based on the argument
################################################################################
    unet = ViTUNet(
        down_chs=(8, 64, 128),
        up_chs=(128, 64),
        num_class=args.num_classes,
        retain_dim=True,
        out_sz=(args.height, args.width),
        kernel_size=args.kernel_size
    )

    if args.model == 'UNet':
        unet = UNet(
            down_chs=(8, 64, 128),
            up_chs=(128, 64),
            num_class=args.num_classes,
            retain_dim=True,
            out_sz=(args.height, args.width),
            kernel_size=args.kernel_size
        )

    if args.model == 'ViTUnet':
        unet = ViTUNet(
            down_chs=(8, 64, 128),
            up_chs=(128, 64),
            num_class=args.num_classes,
            retain_dim=True,
            out_sz=(args.height, args.width),
            kernel_size=args.kernel_size
        )

    # visualize_model(unet)

    print("Third Checkpoint")

    # 计算模型复杂度（MACs）和参数数量，并打印出来
    macs, params = get_model_complexity_info(
        unet, (len(features), args.height, args.width),
        as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print("Fourth Checkpoint")

    # 设置设备和多GPU并行
    if args.gpus:
        assert torch.cuda.is_available(), "CUDA is not available but GPUs were specified."
        device = torch.device(f'cuda:{args.gpus[0]}')
        print('gpus:', args.gpus)
        unet = torch.nn.parallel.DataParallel(unet,
                                              device_ids=[int(gpu) for gpu in
                                                          args.gpus])
    else:
        device = 'cpu'
    print('device:', device)
    unet.to(device)

################################################################################
    # Choose the loss function based on the argument
################################################################################
    #     if args.num_classes == 1:
    #         criterion = FocalLoss(gamma=1.5, alpha=0.85)
    #     else:
    #         criterion = torch.nn.CrossEntropyLoss()

    if args.loss == 'focal':
        criterion = FocalLoss(gamma=1.5, alpha=0.85)
    elif args.loss == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'focall2':
        criterion = FocalMSELoss(gamma=1.5, alpha=0.85)
    elif args.loss == 'focall2+':
        criterion = FocalMSELoss(gamma=1.5, alpha=0.85, f_weight=0.5,
                                composition_method="sum")
    elif args.loss == 'posfocus':
        criterion = PosFocusLoss()

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate,
                                 weight_decay=1.e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, threshold=1.e-5,
        verbose=True
    )
    early_stopping = EarlyStopping(patience=10, min_delta=0)
    # early_stopping = EarlyStopping(patience=200, min_delta=0, verbose=True)

    print('Starting training...')
    best_model, best_epoch, last_epoch, lr_history, train_losses, val_losses = train(
        unet, train_loader, device, criterion, optimizer, scheduler,
        early_stopping,
        val_loader, args.epochs, args.learning_rate, binary, args.outdir
    )
    print('Finished training!')

    print('Evaluating best Model from epoch', best_epoch)
    test_dir = os.path.join(args.outdir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    test_loss = evaluate(
        best_model, test_loader, device, criterion,
        test_dir, best_epoch, binary, mode='test'
    )
    print('Test loss:', test_loss)

    # 将所有元数据写入metadata.json文件
    metadata = {
        'args': vars(args),
        'features': features,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'last_epoch': last_epoch,
        'best_epoch': best_epoch,
        'lr_history': lr_history,
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
    }
    metadata_path = os.path.join(args.outdir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
