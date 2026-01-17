"""
Particle Swarm Optimization for Neural Network Hyperparameter Tuning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import copy

from models.hybrid_ncf import HybridNCF
from models.hybrid_dataset import create_hybrid_dataloaders
from models.train_utils import Metrics


class Particle:
    """Represents a single particle (hyperparameter configuration)"""

    def __init__(self, bounds: Dict):
        """
        Initialize particle with random position

        Args:
            bounds: Dictionary of parameter bounds
        """
        self.bounds = bounds
        self.position = self._initialize_position()
        self.velocity = self._initialize_velocity()
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')

    def _initialize_position(self) -> Dict:
        """Initialize random position within bounds"""
        position = {}
        for param, (choices, param_type) in self.bounds.items():
            if param_type == 'categorical':
                position[param] = np.random.choice(choices)
            elif param_type == 'continuous':
                position[param] = np.random.uniform(choices[0], choices[1])
            elif param_type == 'discrete':
                position[param] = np.random.choice(choices)
        return position

    def _initialize_velocity(self) -> Dict:
        """Initialize random velocity"""
        velocity = {}
        for param, (choices, param_type) in self.bounds.items():
            if param_type == 'continuous':
                range_size = choices[1] - choices[0]
                velocity[param] = np.random.uniform(-range_size * 0.1, range_size * 0.1)
            else:
                velocity[param] = 0
        return velocity

    def update_velocity(self, global_best_position: Dict, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """
        Update particle velocity

        Args:
            global_best_position: Best position found by swarm
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
        """
        for param in self.velocity.keys():
            r1 = np.random.random()
            r2 = np.random.random()

            cognitive = c1 * r1 * (self.best_position[param] - self.position[param])
            social = c2 * r2 * (global_best_position[param] - self.position[param])

            self.velocity[param] = w * self.velocity[param] + cognitive + social

    def update_position(self):
        """Update particle position based on velocity"""
        for param, (choices, param_type) in self.bounds.items():
            if param_type == 'continuous':
                self.position[param] += self.velocity[param]
                # Clip to bounds
                self.position[param] = np.clip(self.position[param], choices[0], choices[1])
            elif param_type == 'discrete':
                # For discrete, velocity influences probability of change
                if abs(self.velocity.get(param, 0)) > 0.5:
                    self.position[param] = np.random.choice(choices)
            elif param_type == 'categorical':
                # For categorical, occasionally explore new options
                if np.random.random() < 0.2:
                    self.position[param] = np.random.choice(choices)


class PSOOptimizer:
    """Particle Swarm Optimizer for hyperparameter tuning"""

    def __init__(
            self,
            n_particles: int = 10,
            n_iterations: int = 8,
            w: float = 0.7,
            c1: float = 1.5,
            c2: float = 1.5,
            device: str = 'cuda'
    ):
        """
        Args:
            n_particles: Number of particles in swarm
            n_iterations: Number of iterations (generations)
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            device: Training device
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Define hyperparameter search space
        self.bounds = {
            'embedding_dim': ([32, 64, 128], 'discrete'),
            'mlp_layers_config': ([0, 1, 2], 'categorical'),  # Index into predefined configs
            'dropout_rate': ([0.1, 0.5], 'continuous'),
            'learning_rate': ([0.0001, 0.01], 'continuous'),
            'batch_size': ([2048, 4096, 8192], 'discrete')
        }

        # Predefined MLP layer configurations
        self.mlp_configs = [
            [256, 128, 64],
            [512, 256, 128],
            [256, 128]
        ]

        self.particles = [Particle(self.bounds) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.history = []

    def _decode_particle(self, particle: Particle) -> Dict:
        """Convert particle position to model hyperparameters"""
        config = particle.position.copy()

        # Convert MLP config index to actual config
        mlp_idx = int(config['mlp_layers_config'])
        config['mlp_layers'] = self.mlp_configs[mlp_idx]
        del config['mlp_layers_config']

        # Convert to appropriate types
        config['embedding_dim'] = int(config['embedding_dim'])
        config['batch_size'] = int(config['batch_size'])
        config['dropout_rate'] = float(config['dropout_rate'])
        config['learning_rate'] = float(config['learning_rate'])

        return config

    def _evaluate_particle(
            self,
            particle: Particle,
            metadata: Dict,
            train_loader,
            val_loader,
            quick_eval: bool = True
    ) -> float:
        """
        Evaluate a particle by training a model with its hyperparameters

        Args:
            particle: Particle to evaluate
            metadata: Dataset metadata
            train_loader: Training data loader
            val_loader: Validation data loader
            quick_eval: If True, train for fewer epochs (faster)

        Returns:
            Validation RMSE (fitness score)
        """
        config = self._decode_particle(particle)

        # Create model
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
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

        # Quick evaluation: 5 epochs, Full evaluation: 15 epochs
        n_epochs = 5 if quick_eval else 15

        best_val_rmse = float('inf')

        for epoch in range(n_epochs):
            # Train
            model.train()
            for batch in train_loader:
                user_ids = batch['user_id'].to(self.device)
                movie_ids = batch['movie_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                user_features = batch['user_features'].to(self.device)
                movie_features = batch['movie_features'].to(self.device)

                optimizer.zero_grad()
                predictions = model(user_ids, movie_ids, user_features, movie_features)
                loss = criterion(predictions, ratings)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            # Validate
            model.eval()
            val_rmse_total = 0
            n_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    user_ids = batch['user_id'].to(self.device)
                    movie_ids = batch['movie_id'].to(self.device)
                    ratings = batch['rating'].to(self.device)
                    user_features = batch['user_features'].to(self.device)
                    movie_features = batch['movie_features'].to(self.device)

                    predictions = model(user_ids, movie_ids, user_features, movie_features)
                    rmse = Metrics.rmse(predictions, ratings)
                    val_rmse_total += rmse.item()
                    n_batches += 1

            val_rmse = val_rmse_total / n_batches
            best_val_rmse = min(best_val_rmse, val_rmse)

        # Clean up
        del model
        torch.cuda.empty_cache()

        return best_val_rmse

    def optimize(self, output_dir: Path):
        """
        Run PSO optimization

        Args:
            output_dir: Directory to save results
        """
        print("\n" + "=" * 70)
        print(" " * 15 + "PSO HYPERPARAMETER OPTIMIZATION")
        print("=" * 70)

        print(f"\n🎮 Device: {self.device}")
        print(f"📊 Swarm configuration:")
        print(f"  Particles: {self.n_particles}")
        print(f"  Iterations: {self.n_iterations}")
        print(f"  Inertia (w): {self.w}")
        print(f"  Cognitive (c1): {self.c1}")
        print(f"  Social (c2): {self.c2}")

        print(f"\n🔍 Search space:")
        print(f"  Embedding dim: {[32, 64, 128]}")
        print(f"  MLP configs: {len(self.mlp_configs)} options")
        print(f"  Dropout rate: [0.1, 0.5]")
        print(f"  Learning rate: [0.0001, 0.01]")
        print(f"  Batch size: [2048, 4096, 8192]")

        # Load data once (we'll create loaders per particle batch size)
        print("\n📊 Loading metadata...")
        from models.hybrid_dataset import create_hybrid_dataloaders
        from src.config import METADATA_FILE
        import pickle

        with open(METADATA_FILE, 'rb') as f:
            metadata = pickle.load(f)

        # Update metadata for actual feature dimensions
        _, _, _, metadata = create_hybrid_dataloaders(batch_size=4096)

        print("\n" + "=" * 70)
        print("STARTING PSO OPTIMIZATION")
        print("=" * 70)

        for iteration in range(self.n_iterations):
            print(f"\n{'=' * 70}")
            print(f"ITERATION {iteration + 1}/{self.n_iterations}")
            print(f"{'=' * 70}")

            # Evaluate all particles
            for i, particle in enumerate(tqdm(self.particles, desc=f"Evaluating particles")):
                config = self._decode_particle(particle)

                # Create data loaders with particle's batch size
                train_loader, val_loader, _, _ = create_hybrid_dataloaders(
                    batch_size=config['batch_size'],
                    num_workers=2  # Reduce workers to save memory
                )

                # Evaluate particle
                fitness = self._evaluate_particle(
                    particle,
                    metadata,
                    train_loader,
                    val_loader,
                    quick_eval=True
                )

                particle.fitness = fitness

                # Update particle best
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    print(f"\n🎯 New global best! RMSE: {fitness:.4f}")
                    print(f"   Config: {self._decode_particle(particle)}")

            # Record iteration history
            iteration_stats = {
                'iteration': iteration + 1,
                'global_best_fitness': self.global_best_fitness,
                'global_best_config': self._decode_particle(
                    type('Particle', (), {'position': self.global_best_position, 'bounds': self.bounds})()),
                'avg_fitness': np.mean([p.fitness for p in self.particles]),
                'best_fitness': min([p.fitness for p in self.particles]),
                'worst_fitness': max([p.fitness for p in self.particles])
            }
            self.history.append(iteration_stats)

            print(f"\n📊 Iteration {iteration + 1} Summary:")
            print(f"  Global best RMSE: {self.global_best_fitness:.4f}")
            print(f"  Average RMSE: {iteration_stats['avg_fitness']:.4f}")
            print(f"  Best this iteration: {iteration_stats['best_fitness']:.4f}")
            print(f"  Worst this iteration: {iteration_stats['worst_fitness']:.4f}")

            # Update particles
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()

            # Save progress
            with open(output_dir / 'pso_history.json', 'w') as f:
                json.dump(self.history, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print("PSO OPTIMIZATION COMPLETED")
        print("=" * 70)

        print(f"\n🎯 Best configuration found:")
        best_config = self._decode_particle(
            type('Particle', (), {'position': self.global_best_position, 'bounds': self.bounds})()
        )
        for key, value in best_config.items():
            print(f"  {key}: {value}")

        print(f"\n📊 Best validation RMSE: {self.global_best_fitness:.4f}")

        return best_config, self.global_best_fitness