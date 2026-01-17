"""
Enhanced Dataset that includes content features
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple

from src.config import TRAIN_FILE, VAL_FILE, TEST_FILE, METADATA_FILE, FEATURES_DIR


class HybridMovieLensDataset(Dataset):
    """
    Dataset with content features for Hybrid NCF
    """

    def __init__(self, data_path: Path, user_features_df: pd.DataFrame, movie_features_df: pd.DataFrame):
        """
        Args:
            data_path: Path to .pt file (train.pt, val.pt, or test.pt)
            user_features_df: DataFrame with user content features
            movie_features_df: DataFrame with movie content features
        """
        self.data = torch.load(data_path, weights_only=False)

        self.user_ids = self.data['user_ids']
        self.movie_ids = self.data['movie_ids']
        self.ratings = self.data['ratings']

        # Create ID to index mappings
        # The features dataframes are indexed by their actual user_id/movie_id
        # But we need to map these to row indices for tensor indexing

        # Sort by ID and reset index to ensure alignment
        user_features_df = user_features_df.sort_values('user_id').reset_index(drop=True)
        movie_features_df = movie_features_df.sort_values('movie_id').reset_index(drop=True)

        # Create mapping dictionaries: ID -> row index
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_features_df['user_id'].values)}
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_features_df['movie_id'].values)}

        # Select ONLY numeric columns for features
        user_numeric_cols = user_features_df.select_dtypes(include=[np.number]).columns.tolist()
        user_numeric_cols = [col for col in user_numeric_cols if col not in ['user_id']]

        movie_numeric_cols = movie_features_df.select_dtypes(include=[np.number]).columns.tolist()
        movie_numeric_cols = [col for col in movie_numeric_cols if col not in ['movie_id']]

        # Create feature tensors from numeric columns only
        user_feature_array = user_features_df[user_numeric_cols].values.astype(np.float32)
        movie_feature_array = movie_features_df[movie_numeric_cols].values.astype(np.float32)

        # Replace any NaN or inf values with 0
        user_feature_array = np.nan_to_num(user_feature_array, nan=0.0, posinf=0.0, neginf=0.0)
        movie_feature_array = np.nan_to_num(movie_feature_array, nan=0.0, posinf=0.0, neginf=0.0)

        self.user_features = torch.FloatTensor(user_feature_array)
        self.movie_features = torch.FloatTensor(movie_feature_array)

        print(f"✓ Loaded dataset from {data_path.name}")
        print(f"  Samples: {len(self):,}")
        print(f"  User features: {self.user_features.shape} (from {len(user_numeric_cols)} numeric columns)")
        print(f"  Movie features: {self.movie_features.shape} (from {len(movie_numeric_cols)} numeric columns)")

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx].item()
        movie_id = self.movie_ids[idx].item()

        # Map IDs to feature indices
        user_feature_idx = self.user_id_to_idx[user_id]
        movie_feature_idx = self.movie_id_to_idx[movie_id]

        return {
            'user_id': self.user_ids[idx],
            'movie_id': self.movie_ids[idx],
            'rating': self.ratings[idx],
            'user_features': self.user_features[user_feature_idx],
            'movie_features': self.movie_features[movie_feature_idx]
        }


def create_hybrid_dataloaders(
    batch_size: int = 4096,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create dataloaders with content features

    Returns:
        train_loader, val_loader, test_loader, metadata
    """

    print("\n" + "="*60)
    print("CREATING HYBRID DATALOADERS")
    print("="*60)

    # Load feature dataframes
    print("\n📊 Loading feature dataframes...")
    user_features_df = pd.read_parquet(FEATURES_DIR / "user_features.parquet")
    movie_features_df = pd.read_parquet(FEATURES_DIR / "movie_features.parquet")

    print(f"  User features: {user_features_df.shape}")
    print(f"  Movie features: {movie_features_df.shape}")

    # Create datasets
    train_dataset = HybridMovieLensDataset(TRAIN_FILE, user_features_df, movie_features_df)
    val_dataset = HybridMovieLensDataset(VAL_FILE, user_features_df, movie_features_df)
    test_dataset = HybridMovieLensDataset(TEST_FILE, user_features_df, movie_features_df)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Load metadata and update with actual feature dimensions
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)

    # Update metadata with actual feature dimensions from the dataset
    metadata['user_feature_dim'] = train_dataset.user_features.shape[1]
    metadata['movie_feature_dim'] = train_dataset.movie_features.shape[1]

    print(f"\n✓ Dataloaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")
    print(f"  Test batches: {len(test_loader):,}")
    print(f"  Workers: {num_workers}")
    print(f"  User feature dim: {metadata['user_feature_dim']}")
    print(f"  Movie feature dim: {metadata['movie_feature_dim']}")
    print("="*60)

    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    # Test dataset loading
    print("Testing hybrid dataset loading...")
    train_loader, val_loader, test_loader, metadata = create_hybrid_dataloaders(batch_size=2048)

    print(f"\n📊 Metadata:")
    print(f"  Users: {metadata['n_users']:,}")
    print(f"  Movies: {metadata['n_movies']:,}")
    print(f"  User feature dim: {metadata['user_feature_dim']}")
    print(f"  Movie feature dim: {metadata['movie_feature_dim']}")

    # Test batch loading
    batch = next(iter(train_loader))
    print(f"\n📦 Sample batch:")
    print(f"  User IDs: {batch['user_id'].shape}")
    print(f"  Movie IDs: {batch['movie_id'].shape}")
    print(f"  Ratings: {batch['rating'].shape}")
    print(f"  User features: {batch['user_features'].shape}")
    print(f"  Movie features: {batch['movie_features'].shape}")
    print(f"  Rating range: [{batch['rating'].min():.1f}, {batch['rating'].max():.1f}]")

    print("\n✓ Hybrid dataset test passed!")