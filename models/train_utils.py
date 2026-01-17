"""
Training utilities for NCF model
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
from typing import Dict, Tuple


class Metrics:
    """Calculate evaluation metrics"""

    @staticmethod
    def rmse(predictions, targets):
        """Root Mean Square Error"""
        return torch.sqrt(torch.mean((predictions - targets) ** 2))

    @staticmethod
    def mae(predictions, targets):
        """Mean Absolute Error"""
        return torch.mean(torch.abs(predictions - targets))

    @staticmethod
    def mse(predictions, targets):
        """Mean Square Error"""
        return torch.mean((predictions - targets) ** 2)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch

    Returns:
        avg_loss, avg_rmse, avg_mae
    """
    model.train()

    total_loss = 0
    total_rmse = 0
    total_mae = 0
    n_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        user_ids = batch['user_id'].to(device)
        movie_ids = batch['movie_id'].to(device)
        ratings = batch['rating'].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids)

        # Calculate loss
        loss = criterion(predictions, ratings)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Update weights
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            rmse = Metrics.rmse(predictions, ratings)
            mae = Metrics.mae(predictions, ratings)

        total_loss += loss.item()
        total_rmse += rmse.item()
        total_mae += mae.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rmse': f'{rmse.item():.4f}',
            'mae': f'{mae.item():.4f}'
        })

    avg_loss = total_loss / n_batches
    avg_rmse = total_rmse / n_batches
    avg_mae = total_mae / n_batches

    return avg_loss, avg_rmse, avg_mae


def validate(model, val_loader, criterion, device, epoch=None):
    """
    Validate model

    Returns:
        avg_loss, avg_rmse, avg_mae
    """
    model.eval()

    total_loss = 0
    total_rmse = 0
    total_mae = 0
    n_batches = len(val_loader)

    desc = f"Epoch {epoch} [Val]" if epoch else "Validation"
    pbar = tqdm(val_loader, desc=desc)

    with torch.no_grad():
        for batch in pbar:
            user_ids = batch['user_id'].to(device)
            movie_ids = batch['movie_id'].to(device)
            ratings = batch['rating'].to(device)

            # Forward pass
            predictions = model(user_ids, movie_ids)

            # Calculate metrics
            loss = criterion(predictions, ratings)
            rmse = Metrics.rmse(predictions, ratings)
            mae = Metrics.mae(predictions, ratings)

            total_loss += loss.item()
            total_rmse += rmse.item()
            total_mae += mae.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rmse': f'{rmse.item():.4f}',
                'mae': f'{mae.item():.4f}'
            })

    avg_loss = total_loss / n_batches
    avg_rmse = total_rmse / n_batches
    avg_mae = total_mae / n_batches

    return avg_loss, avg_rmse, avg_mae


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']

    print(f"  ✓ Checkpoint loaded from epoch {epoch}")
    return epoch, metrics