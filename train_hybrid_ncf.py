"""
Training script for Hybrid NCF model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from datetime import datetime

from models.hybrid_ncf import HybridNCF
from models.hybrid_dataset import create_hybrid_dataloaders
from models.train_utils import (
    train_epoch, validate, EarlyStopping, save_checkpoint, Metrics
)
from src.config import OUTPUTS_DIR


def train_hybrid_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch with content features"""
    from tqdm import tqdm

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
        user_features = batch['user_features'].to(device)
        movie_features = batch['movie_features'].to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids, user_features, movie_features)

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


def validate_hybrid(model, val_loader, criterion, device, epoch=None):
    """Validate model with content features"""
    from tqdm import tqdm

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
            user_features = batch['user_features'].to(device)
            movie_features = batch['movie_features'].to(device)

            # Forward pass
            predictions = model(user_ids, movie_ids, user_features, movie_features)

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


def train_hybrid_ncf(
        embedding_dim_gmf: int = 64,
        embedding_dim_mlp: int = 64,
        mlp_layers: list = [512, 256, 128],
        content_dim: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 4096,
        n_epochs: int = 25,
        patience: int = 7,
        device: str = 'cuda',
        use_content_features: bool = True
):
    """
    Train Hybrid NCF model
    """

    print("\n" + "=" * 70)
    print(" " * 15 + "TRAINING HYBRID NCF MODEL")
    print("=" * 70)

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Device: {device}")

    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")

    # Create dataloaders
    train_loader, val_loader, test_loader, metadata = create_hybrid_dataloaders(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    # Create model
    print("\n" + "=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = HybridNCF(
        n_users=metadata['n_users'],
        n_movies=metadata['n_movies'],
        user_feature_dim=metadata['user_feature_dim'],
        movie_feature_dim=metadata['movie_feature_dim'],
        embedding_dim_gmf=embedding_dim_gmf,
        embedding_dim_mlp=embedding_dim_mlp,
        mlp_layers=mlp_layers,
        content_dim=content_dim,
        dropout_rate=dropout_rate,
        use_content_features=use_content_features
    ).to(device)

    print(f"\n📊 Model Architecture:")
    print(f"  Users: {metadata['n_users']:,}")
    print(f"  Movies: {metadata['n_movies']:,}")
    print(f"  GMF embedding dim: {embedding_dim_gmf}")
    print(f"  MLP embedding dim: {embedding_dim_mlp}")
    print(f"  MLP layers: {mlp_layers}")
    print(f"  Content dim: {content_dim}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Use content features: {use_content_features}")
    print(f"\n  Total parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.count_parameters() * 4 / (1024 ** 2):.2f} MB")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False
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
    output_dir = OUTPUTS_DIR / 'hybrid_ncf' / timestamp
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
        train_loss, train_rmse, train_mae = train_hybrid_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_rmse, val_mae = validate_hybrid(
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
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_rmse, test_mae = validate_hybrid(model, test_loader, criterion, device)

    print(f"\n📊 Test Set Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")

    # Compare with baseline NCF
    baseline_rmse = 0.9016  # From Model 1
    improvement = ((baseline_rmse - test_rmse) / baseline_rmse) * 100

    print(f"\n📈 Improvement over baseline NCF:")
    print(f"  Baseline RMSE: {baseline_rmse:.4f}")
    print(f"  Hybrid RMSE:   {test_rmse:.4f}")
    print(f"  Improvement:   {improvement:.2f}%")

    # Save final results
    results = {
        'model': 'Hybrid_NCF',
        'embedding_dim_gmf': embedding_dim_gmf,
        'embedding_dim_mlp': embedding_dim_mlp,
        'mlp_layers': mlp_layers,
        'content_dim': content_dim,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'n_epochs': epoch,
        'use_content_features': use_content_features,
        'best_val_rmse': best_val_rmse,
        'test_loss': test_loss,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'baseline_rmse': baseline_rmse,
        'improvement_percent': improvement,
        'training_time_minutes': total_time / 60,
        'timestamp': timestamp
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved results to {output_dir / 'results.json'}")
    print("=" * 60)

    return model, history, results


if __name__ == "__main__":
    # Train with optimized hyperparameters
    model, history, results = train_hybrid_ncf(
        embedding_dim_gmf=64,
        embedding_dim_mlp=64,
        mlp_layers=[512, 256, 128],
        content_dim=64,
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=4096,
        n_epochs=25,
        patience=7,
        device='cuda',
        use_content_features=True
    )

    print("\n✅ Hybrid NCF training completed successfully!")