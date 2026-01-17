"""
Training script for NCF model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from datetime import datetime

from models.ncf import NCF
from models.dataset import create_dataloaders
from models.train_utils import (
    train_epoch, validate, EarlyStopping, save_checkpoint
)
from src.config import OUTPUTS_DIR


def train_ncf(
        embedding_dim: int = 128,
        hidden_layers: list = [256, 128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 4096,
        n_epochs: int = 20,
        patience: int = 5,
        device: str = 'cuda'
):
    """
    Train NCF model

    Args:
        embedding_dim: Embedding dimension
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size
        n_epochs: Number of epochs
        patience: Early stopping patience
        device: Device to use ('cuda' or 'cpu')
    """

    print("\n" + "=" * 70)
    print(" " * 20 + "TRAINING NCF MODEL")
    print("=" * 70)

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Device: {device}")

    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")

    # Create dataloaders
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = NCF(
        n_users=metadata['n_users'],
        n_movies=metadata['n_movies'],
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate
    ).to(device)

    print(f"\n📊 Model Architecture:")
    print(f"  Users: {metadata['n_users']:,}")
    print(f"  Movies: {metadata['n_movies']:,}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"\n  Total parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.count_parameters() * 4 / (1024 ** 2):.2f} MB")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001, mode='min')

    # Training history
    history = {
        'train_loss': [], 'train_rmse': [], 'train_mae': [],
        'val_loss': [], 'val_rmse': [], 'val_mae': [],
        'learning_rates': []
    }

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUTS_DIR / 'ncf' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Output directory: {output_dir}")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_rmse = float('inf')
    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_rmse, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_rmse, val_mae = validate(
            model, val_loader, criterion, device, epoch
        )

        # Update learning rate
        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_rmse'].append(train_rmse)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['learning_rates'].append(current_lr)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\n📊 Epoch {epoch}/{n_epochs} Summary ({epoch_time:.1f}s):")
        print(f"  Train - Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_checkpoint(
                model, optimizer, epoch,
                {'val_rmse': val_rmse, 'val_mae': val_mae},
                output_dir / 'best_model.pt'
            )
            print(f"  🎯 New best validation RMSE: {val_rmse:.4f}")

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            {'val_rmse': val_rmse, 'val_mae': val_mae},
            output_dir / 'latest_checkpoint.pt'
        )

        # Early stopping
        if early_stopping(val_rmse):
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            break

        print("-" * 60)

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"\n⏱️  Total training time: {total_time / 60:.1f} minutes")
    print(f"🎯 Best validation RMSE: {best_val_rmse:.4f}")

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n💾 Saved training history to {output_dir / 'history.json'}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_rmse, test_mae = validate(model, test_loader, criterion, device)

    print(f"\n📊 Test Set Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")

    # Save final results
    results = {
        'model': 'NCF',
        'embedding_dim': embedding_dim,
        'hidden_layers': hidden_layers,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'n_epochs': epoch,
        'best_val_rmse': best_val_rmse,
        'test_loss': test_loss,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'training_time_minutes': total_time / 60,
        'timestamp': timestamp
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved results to {output_dir / 'results.json'}")
    print("=" * 60)

    return model, history, results


if __name__ == "__main__":
    # Train with default hyperparameters
    model, history, results = train_ncf(
        embedding_dim=128,
        hidden_layers=[256, 128, 64],
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=4096,
        n_epochs=20,
        patience=5,
        device='cuda'
    )

    print("\n✅ Training completed successfully!")