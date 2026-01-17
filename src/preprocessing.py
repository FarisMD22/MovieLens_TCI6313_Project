"""
Main preprocessing pipeline for MovieLens 1M dataset
"""
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

from src.config import *
from src.data_loader import MovieLensDataLoader, print_data_summary
from src.feature_engineering import FeatureEngineer


class MovieLensPreprocessor:
    """Complete preprocessing pipeline"""

    def __init__(self):
        self.loader = MovieLensDataLoader()
        self.engineer = FeatureEngineer()

        self.ratings_df = None
        self.movies_df = None
        self.users_df = None

        self.user_features = None
        self.movie_features = None
        self.interactions = None

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.metadata = {}

    def load_data(self):
        """Load all raw data"""
        self.ratings_df, self.movies_df, self.users_df = self.loader.load_all()
        print_data_summary(self.ratings_df, self.movies_df, self.users_df)

    def engineer_features(self):
        """Create all features"""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        # Create user features
        self.user_features = self.engineer.create_user_features(
            self.users_df,
            self.ratings_df
        )

        # Create movie features
        self.movie_features = self.engineer.create_movie_features(
            self.movies_df,
            self.ratings_df
        )

        # Create interaction features
        self.interactions, self.user_features = self.engineer.create_interaction_features(
            self.ratings_df,
            self.user_features,
            self.movie_features
        )

        # Normalize features
        self.user_features, self.movie_features = self.engineer.normalize_features(
            self.user_features,
            self.movie_features
        )

        print("\n✓ Feature engineering completed!")
        print(f"  User features: {self.user_features.shape}")
        print(f"  Movie features: {self.movie_features.shape}")
        print(f"  Interactions: {self.interactions.shape}")

    def create_train_val_test_split(self):
        """Split data into train/validation/test sets"""
        print("\n" + "=" * 60)
        print("CREATING TRAIN/VAL/TEST SPLITS")
        print("=" * 60)

        # Set random seed
        np.random.seed(RANDOM_SEED)

        # First split: train+val vs test
        train_val_df, self.test_df = train_test_split(
            self.interactions,
            test_size=TEST_RATIO,
            random_state=RANDOM_SEED,
            stratify=None  # Random split
        )

        # Second split: train vs val
        val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        self.train_df, self.val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=RANDOM_SEED,
            stratify=None
        )

        print(f"\n✓ Data split completed:")
        print(f"  Train set: {len(self.train_df):,} ratings ({len(self.train_df) / len(self.interactions) * 100:.1f}%)")
        print(f"  Val set:   {len(self.val_df):,} ratings ({len(self.val_df) / len(self.interactions) * 100:.1f}%)")
        print(f"  Test set:  {len(self.test_df):,} ratings ({len(self.test_df) / len(self.interactions) * 100:.1f}%)")

        # Verify no overlap
        assert len(set(self.train_df.index) & set(self.val_df.index)) == 0
        assert len(set(self.train_df.index) & set(self.test_df.index)) == 0
        assert len(set(self.val_df.index) & set(self.test_df.index)) == 0
        print("  ✓ No overlap between splits verified")

    def create_metadata(self) -> Dict:
        """Create metadata dictionary"""
        self.metadata = {
            'n_users': int(self.users_df['user_id'].max()) + 1,
            'n_movies': int(self.movies_df['movie_id'].max()) + 1,
            'n_ratings': len(self.ratings_df),
            'n_train': len(self.train_df),
            'n_val': len(self.val_df),
            'n_test': len(self.test_df),
            'rating_min': int(self.ratings_df['rating'].min()),
            'rating_max': int(self.ratings_df['rating'].max()),
            'user_feature_dim': len([col for col in self.user_features.columns if col != 'user_id']),
            'movie_feature_dim': len([col for col in self.movie_features.columns if col != 'movie_id']),
            'genre_list': GENRE_LIST,
            'random_seed': RANDOM_SEED
        }

        return self.metadata

    def save_processed_data(self):
        """Save all processed data"""
        print("\n" + "=" * 60)
        print("SAVING PROCESSED DATA")
        print("=" * 60)

        # Save train/val/test splits as PyTorch tensors
        print("\n💾 Saving train/val/test splits...")

        for name, df in [('train', self.train_df), ('val', self.val_df), ('test', self.test_df)]:
            data_dict = {
                'user_ids': torch.LongTensor(df['user_id'].values),
                'movie_ids': torch.LongTensor(df['movie_id'].values),
                'ratings': torch.FloatTensor(df['rating'].values),
                'timestamps': torch.LongTensor(df['timestamp'].values)
            }

            filepath = PROCESSED_DATA_DIR / f"{name}.pt"
            torch.save(data_dict, filepath)
            print(f"  ✓ Saved {name}.pt ({len(df):,} samples)")

        # Save feature dataframes
        print("\n💾 Saving feature dataframes...")
        self.user_features.to_parquet(FEATURES_DIR / "user_features.parquet")
        self.movie_features.to_parquet(FEATURES_DIR / "movie_features.parquet")
        print(f"  ✓ Saved user_features.parquet")
        print(f"  ✓ Saved movie_features.parquet")

        # Save metadata
        print("\n💾 Saving metadata...")
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"  ✓ Saved metadata.pkl")

        # Save statistics as JSON
        print("\n💾 Saving statistics...")
        stats = self.loader.get_statistics()
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"  ✓ Saved statistics.json")

        # Save scalers for future use
        scaler_data = {
            'user_scaler': self.engineer.user_scaler,
            'movie_scaler': self.engineer.movie_scaler,
            'mlb_genres': self.engineer.mlb_genres
        }
        with open(FEATURES_DIR / "scalers.pkl", 'wb') as f:
            pickle.dump(scaler_data, f)
        print(f"  ✓ Saved scalers.pkl")

        print("\n" + "=" * 60)
        print("✓ ALL DATA SAVED SUCCESSFULLY!")
        print("=" * 60)

    def run_full_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "=" * 70)
        print(" " * 15 + "MOVIELENS 1M PREPROCESSING PIPELINE")
        print("=" * 70)

        # Step 1: Load data
        self.load_data()

        # Step 2: Engineer features
        self.engineer_features()

        # Step 3: Create splits
        self.create_train_val_test_split()

        # Step 4: Create metadata
        self.create_metadata()

        # Step 5: Save everything
        self.save_processed_data()

        print("\n" + "=" * 70)
        print("🎉 PREPROCESSING COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 70)

        self.print_final_summary()

    def print_final_summary(self):
        """Print final preprocessing summary"""
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)

        print(f"\n📊 Dataset:")
        print(f"  Total users: {self.metadata['n_users']:,}")
        print(f"  Total movies: {self.metadata['n_movies']:,}")
        print(f"  Total ratings: {self.metadata['n_ratings']:,}")

        print(f"\n📦 Splits:")
        print(f"  Train: {self.metadata['n_train']:,}")
        print(f"  Validation: {self.metadata['n_val']:,}")
        print(f"  Test: {self.metadata['n_test']:,}")

        print(f"\n🔧 Features:")
        print(f"  User feature dimensions: {self.metadata['user_feature_dim']}")
        print(f"  Movie feature dimensions: {self.metadata['movie_feature_dim']}")

        print(f"\n💾 Output files:")
        print(f"  {PROCESSED_DATA_DIR}/train.pt")
        print(f"  {PROCESSED_DATA_DIR}/val.pt")
        print(f"  {PROCESSED_DATA_DIR}/test.pt")
        print(f"  {FEATURES_DIR}/user_features.parquet")
        print(f"  {FEATURES_DIR}/movie_features.parquet")
        print(f"  {PROCESSED_DATA_DIR}/metadata.pkl")
        print(f"  {PROCESSED_DATA_DIR}/statistics.json")

        print("\n✓ Ready for model training!")
        print("=" * 60)


if __name__ == "__main__":
    preprocessor = MovieLensPreprocessor()
    preprocessor.run_full_pipeline()