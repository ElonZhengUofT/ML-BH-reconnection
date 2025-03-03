import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from model import UNet
from dataset import NpzDataset
from utilities import split_data, plot_results
from optimizers import EarlyStopping
from ptflops import get_model_complexity_info


def train_model(
        model, train_loader, device, loss_fn, optimizer, lr_scheduler,
        early_stop, val_loader, num_epochs, initial_lr, binary_class, output_dir
    ):
    """
    Trains a UNet model with the given parameters.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training set.
        device (torch.device): The device (CPU/GPU) to train on.
        loss_fn (torch.nn.Module): Loss function used during training.
        optimizer (torch.optim.Optimizer): Optimizer for gradient descent.
        lr_scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
        early_stop (EarlyStopping): Early stopping mechanism.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation set.
        num_epochs (int): Number of training epochs.
        initial_lr (float): Initial learning rate.
        binary_class (bool): Whether it's a binary classification problem.
        output_dir (str): Directory to save training results.

    Returns:
        tuple[int, List[float], List[float]]: Best epoch, training loss history, validation loss history.
    """

    train_loss_history = []
    val_loss_history = []
    best_epoch = 0
    best_val_loss = np.inf

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        total_samples = 0

        with tqdm(train_loader, unit='batch') as tepoch:
            for batch in tepoch:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, type=str)
    parser.add_argument('-o', '--output_dir', required=True, type=str)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-l', '--learning_rate', default=1e-5, type=float)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-s', '--standardize', action='store_true')
    parser.add_argument('-g', '--gpu', default=None, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = glob(os.path.join(args.input_dir, '*.npz'))
