"""
PSO-based hyperparameter optimization and final model training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from datetime import datetime

from models.pso_optimizer import PSOOptimizer
from models.hybrid_ncf import HybridNCF
from models.hybrid_dataset import create_hybrid_dataloaders
from models.train_utils import Metrics
from src.config import OUTPUTS_DIR, METADATA_FILE
import pickle


def train_final_model(config: dict, output_dir: Path, n_epochs: int = 25):
    """
    Train final model with PSO-optimized hyperparameters

    Args:
        config: Optimized hyperparameters
        output_dir: Output directory
        n_epochs: Number of training epochs
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "TRAINING FINAL PSO-OPTIMIZED MODEL")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🎮 Device: {device}")

    # Load metadata
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)

    # Create dataloaders with optimized batch size
    train_loader, val_loader, test_loader, metadata = create_hybrid_dataloaders(
        batch_size=config['batch_size'],
        num_workers=4
    )

    # Create model with optimized hyperparameters
    print("\n📊 Creating PSO-optimized model...")
    model = HybridNCF(
        n_users=metadata['n_users'],
        n_movies=metadata['n_movies'],
        user_feature_dim=metadata['user_feature_dim'],
        movie_feature_dim=metadata['movie_feature_dim'],
        embedding_dim_gmf=config['embedding_dim'],
        embedding_dim_mlp=config['embedding_dim'],
        mlp_layers=config['mlp_layers'],
        dropout_rate=config['dropout_rate'],
        use_content_features=True
    ).to(device)

    print(f"\n  Total parameters: {model.count_parameters():,}")
    print(f"  Configuration:")
    for key, value in config.items():
        print(f"    {key}: {value}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    from tqdm import tqdm

    best_val_rmse = float('inf')
    history = {'train_rmse': [], 'val_rmse': [], 'learning_rates': []}

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_rmse_total = 0
        n_train_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            user_ids = batch['user_id'].to(device)
            movie_ids = batch['movie_id'].to(device)
            ratings = batch['rating'].to(device)
            user_features = batch['user_features'].to(device)
            movie_features = batch['movie_features'].to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, movie_ids, user_features, movie_features)
            loss = criterion(predictions, ratings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                rmse = Metrics.rmse(predictions, ratings)
                train_rmse_total += rmse.item()
                n_train_batches += 1

        train_rmse = train_rmse_total / n_train_batches

        # Validate
        model.eval()
        val_rmse_total = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                user_ids = batch['user_id'].to(device)
                movie_ids = batch['movie_id'].to(device)
                ratings = batch['rating'].to(device)
                user_features = batch['user_features'].to(device)
                movie_features = batch['movie_features'].to(device)

                predictions = model(user_ids, movie_ids, user_features, movie_features)
                rmse = Metrics.rmse(predictions, ratings)
                val_rmse_total += rmse.item()
                n_val_batches += 1

        val_rmse = val_rmse_total / n_val_batches

        # Update scheduler
        scheduler.step(val_rmse)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['learning_rates'].append(current_lr)

        print(f"\n📊 Epoch {epoch}/{n_epochs}:")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'config': config
            }, output_dir / 'pso_best_model.pt')
            print(f"  🎯 New best! Saved checkpoint")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(output_dir / 'pso_best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_rmse_total = 0
    test_mae_total = 0
    n_test_batches = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            user_ids = batch['user_id'].to(device)
            movie_ids = batch['movie_id'].to(device)
            ratings = batch['rating'].to(device)
            user_features = batch['user_features'].to(device)
            movie_features = batch['movie_features'].to(device)

            predictions = model(user_ids, movie_ids, user_features, movie_features)
            rmse = Metrics.rmse(predictions, ratings)
            mae = Metrics.mae(predictions, ratings)
            test_rmse_total += rmse.item()
            test_mae_total += mae.item()
            n_test_batches += 1

    test_rmse = test_rmse_total / n_test_batches
    test_mae = test_mae_total / n_test_batches

    print(f"\n📊 Test Results:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")

    # Compare with baselines
    baseline_ncf = 0.9016
    hybrid_ncf = 0.8920

    print(f"\n📈 Improvements:")
    print(f"  vs Baseline NCF: {((baseline_ncf - test_rmse) / baseline_ncf * 100):.2f}%")
    print(f"  vs Hybrid NCF: {((hybrid_ncf - test_rmse) / hybrid_ncf * 100):.2f}%")

    # Save results
    results = {
        'model': 'PSO_Optimized_Hybrid_NCF',
        'config': config,
        'best_val_rmse': best_val_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'baseline_ncf_rmse': baseline_ncf,
        'hybrid_ncf_rmse': hybrid_ncf,
        'improvement_vs_baseline': ((baseline_ncf - test_rmse) / baseline_ncf * 100),
        'improvement_vs_hybrid': ((hybrid_ncf - test_rmse) / hybrid_ncf * 100)
    }

    with open(output_dir / 'pso_final_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'pso_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n💾 Results saved to {output_dir}")

    return model, results


def main():
    """Main PSO optimization and training"""

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUTS_DIR / 'pso' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"💾 Output directory: {output_dir}")

    # Run PSO optimization
    print("\n🔍 Starting PSO hyperparameter search...")

    pso = PSOOptimizer(
        n_particles=10,  # Number of configurations to try per iteration. Change from 10 to 20
        n_iterations=8,  # Number of optimization iterations. Changed from 8 to 15
        w=0.7,  # Inertia weight
        c1=1.5,  # Cognitive coefficient
        c2=1.5,  # Social coefficient
        device='cuda'
    )

    start_time = time.time()
    best_config, best_fitness = pso.optimize(output_dir)
    pso_time = time.time() - start_time

    print(f"\n⏱️  PSO optimization completed in {pso_time / 60:.1f} minutes")

    # Train final model with best configuration
    print("\n🚀 Training final model with optimized hyperparameters...")

    start_time = time.time()
    model, results = train_final_model(best_config, output_dir, n_epochs=25)
    training_time = time.time() - start_time

    print(f"\n⏱️  Final training completed in {training_time / 60:.1f} minutes")
    print(f"⏱️  Total time: {(pso_time + training_time) / 60:.1f} minutes")

    print("\n✅ PSO optimization and training completed successfully!")


if __name__ == "__main__":
    main()