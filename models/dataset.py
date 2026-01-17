"""
PyTorch Dataset for MovieLens 1M
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from typing import Dict, Tuple

from src.config import TRAIN_FILE, VAL_FILE, TEST_FILE, METADATA_FILE


class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for MovieLens ratings

    Loads preprocessed data and provides samples for training
    """

    def __init__(self, data_path: Path):
        """
        Args:
            data_path: Path to .pt file (train.pt, val.pt, or test.pt)
        """
        self.data = torch.load(data_path)

        self.user_ids = self.data['user_ids']
        self.movie_ids = self.data['movie_ids']
        self.ratings = self.data['ratings']

        print(f"✓ Loaded dataset from {data_path.name}")
        print(f"  Samples: {len(self):,}")

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'movie_id': self.movie_ids[idx],
            'rating': self.ratings[idx]
        }


def load_metadata() -> Dict:
    """Load dataset metadata"""
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    return metadata


def create_dataloaders(
        batch_size: int = 4096,
        num_workers: int = 4,
        pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders

    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader, test_loader, metadata
    """

    print("\n" + "=" * 60)
    print("CREATING DATALOADERS")
    print("=" * 60)

    # Load datasets
    train_dataset = MovieLensDataset(TRAIN_FILE)
    val_dataset = MovieLensDataset(VAL_FILE)
    test_dataset = MovieLensDataset(TEST_FILE)

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

    # Load metadata
    metadata = load_metadata()

    print(f"\n✓ Dataloaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")
    print(f"  Test batches: {len(test_loader):,}")
    print(f"  Workers: {num_workers}")
    print("=" * 60)

    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    train_loader, val_loader, test_loader, metadata = create_dataloaders(batch_size=2048)

    print(f"\n📊 Metadata:")
    print(f"  Users: {metadata['n_users']:,}")
    print(f"  Movies: {metadata['n_movies']:,}")
    print(f"  Ratings: {metadata['n_ratings']:,}")

    # Test batch loading
    batch = next(iter(train_loader))
    print(f"\n📦 Sample batch:")
    print(f"  User IDs shape: {batch['user_id'].shape}")
    print(f"  Movie IDs shape: {batch['movie_id'].shape}")
    print(f"  Ratings shape: {batch['rating'].shape}")
    print(f"  Rating range: [{batch['rating'].min():.1f}, {batch['rating'].max():.1f}]")