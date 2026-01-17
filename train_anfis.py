"""
Training script for ANFIS model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np

from models.anfis import SimplifiedANFIS
from models.anfis_features import ANFISFeatureExtractor
from models.dataset import create_dataloaders
from models.train_utils import Metrics
from src.config import OUTPUTS_DIR


def train_anfis_epoch(model, train_loader, feature_extractor, criterion, optimizer, device, epoch):
    """Train ANFIS for one epoch"""
    model.train()

    total_loss = 0
    total_rmse = 0
    total_mae = 0
    n_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        user_ids = batch['user_id'].cpu().numpy()
        movie_ids = batch['movie_id'].cpu().numpy()
        ratings = batch['rating'].to(device)

        # Extract ANFIS features
        features = feature_extractor.extract_features(user_ids, movie_ids)
        features = torch.FloatTensor(features).to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(features)

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

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'rmse': f'{rmse.item():.4f}',
            'mae': f'{mae.item():.4f}'
        })

    return total_loss / n_batches, total_rmse / n_batches, total_mae / n_batches


def validate_anfis(model, val_loader, feature_extractor, criterion, device, epoch=None):
    """Validate ANFIS model"""
    model.eval()

    total_loss = 0
    total_rmse = 0
    total_mae = 0
    n_batches = len(val_loader)

    desc = f"Epoch {epoch} [Val]" if epoch else "Validation"
    pbar = tqdm(val_loader, desc=desc)

    with torch.no_grad():
        for batch in pbar:
            user_ids = batch['user_id'].cpu().numpy()
            movie_ids = batch['movie_id'].cpu().numpy()
            ratings = batch['rating'].to(device)

            # Extract ANFIS features
            features = feature_extractor.extract_features(user_ids, movie_ids)
            features = torch.FloatTensor(features).to(device)

            # Forward pass
            predictions = model(features)

            # Calculate metrics
            loss = criterion(predictions, ratings)
            rmse = Metrics.rmse(predictions, ratings)
            mae = Metrics.mae(predictions, ratings)

            total_loss += loss.item()
            total_rmse += rmse.item()
            total_mae += mae.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rmse': f'{rmse.item():.4f}',
                'mae': f'{mae.item():.4f}'
            })

    return total_loss / n_batches, total_rmse / n_batches, total_mae / n_batches


def train_anfis(
        n_rules: int = 8,
        n_mfs: int = 3,
        learning_rate: float = 0.01,
        batch_size: int = 2048,
        n_epochs: int = 50,
        device: str = 'cuda'
):
    """
    Train ANFIS model
    """

    print("\n" + "=" * 70)
    print(" " * 20 + "TRAINING ANFIS MODEL")
    print("=" * 70)

    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Device: {device}")

    # Create feature extractor
    print("\n" + "=" * 60)
    print("CREATING FEATURE EXTRACTOR")
    print("=" * 60)
    feature_extractor = ANFISFeatureExtractor()

    # Create dataloaders
    print("\n" + "=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    # Create model
    print("\n" + "=" * 60)
    print("CREATING ANFIS MODEL")
    print("=" * 60)

    model = SimplifiedANFIS(
        n_inputs=8,  # Changed from 3 to 8
        n_rules=n_rules,
        n_mfs=n_mfs
    ).to(device)

    print(f"\n📊 Model Architecture:")
    print(f"  Input features: 8")  # Changed
    print(f"    1. User activity")
    print(f"    2. Movie popularity")
    print(f"    3. User average rating")
    print(f"    4. Movie average rating")
    print(f"    5. User rating std")
    print(f"    6. User-genre affinity")
    print(f"    7. User deviation from mean")
    print(f"    8. Movie deviation from mean")
    print(f"  Fuzzy rules: {n_rules}")
    print(f"  Membership functions per input: {n_mfs}")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.count_parameters() * 4 / 1024:.2f} KB")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    # Training history
    history = {
        'train_loss': [], 'train_rmse': [], 'train_mae': [],
        'val_loss': [], 'val_rmse': [], 'val_mae': [],
        'learning_rates': []
    }

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUTS_DIR / 'anfis' / timestamp
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
        train_loss, train_rmse, train_mae = train_anfis_epoch(
            model, train_loader, feature_extractor, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_rmse, val_mae = validate_anfis(
            model, val_loader, feature_extractor, criterion, device, epoch
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse
            }, output_dir / 'best_model.pt')
            print(f"  🎯 New best validation RMSE: {val_rmse:.4f}")

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

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_rmse, test_mae = validate_anfis(
        model, test_loader, feature_extractor, criterion, device
    )

    print(f"\n📊 Test Set Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")

    # Compare with other models
    baseline_rmse = 0.9016
    hybrid_rmse = 0.8920
    pso_rmse = 0.8721

    print(f"\n📈 Comparison with other models:")
    print(f"  Baseline NCF:  {baseline_rmse:.4f}")
    print(f"  Hybrid NCF:    {hybrid_rmse:.4f}")
    print(f"  PSO Optimized: {pso_rmse:.4f}")
    print(f"  ANFIS:         {test_rmse:.4f}")

    # Extract learned rules
    print("\n" + "=" * 60)
    print("LEARNED FUZZY RULES")
    print("=" * 60)

    rules = model.get_rules_summary()
    for rule in rules[:5]:  # Show first 5 rules
        print(f"\nRule {rule['rule_id'] + 1}:")
        print(f"  {rule['formula']}")

    # Save results
    results = {
            'model': 'Enhanced_ANFIS',  # Changed
            'n_inputs': 8,  # Added this line
            'n_rules': n_rules,
            'n_mfs': n_mfs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'best_val_rmse': float(best_val_rmse),  # Convert to float
            'test_loss': float(test_loss),  # Convert to float
            'test_rmse': float(test_rmse),  # Convert to float
            'test_mae': float(test_mae),  # Convert to float
            'baseline_rmse': float(baseline_rmse),
            'hybrid_rmse': float(hybrid_rmse),
            'pso_rmse': float(pso_rmse),
            'training_time_minutes': float(total_time / 60),
            'learned_rules': rules,
            'timestamp': timestamp
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved results to {output_dir / 'results.json'}")
    print("=" * 60)

    return model, history, results


if __name__ == "__main__":
    # Train Enhanced ANFIS
    model, history, results = train_anfis(
        n_rules=16,           # Increased from 8
        n_mfs=3,
        learning_rate=0.001,  # Reduced from 0.01
        batch_size=2048,
        n_epochs=50,
        device='cuda'
    )
    print("\n✅ ANFIS training completed successfully!")