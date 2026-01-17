"""
Data loading utilities for MovieLens 1M dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

from src.config import *


class MovieLensDataLoader:
    """Load and parse MovieLens 1M dataset"""

    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None

    def load_ratings(self) -> pd.DataFrame:
        """
        Load ratings.dat file
        Format: UserID::MovieID::Rating::Timestamp
        """
        print("\n📊 Loading ratings data...")

        if not RATINGS_FILE.exists():
            raise FileNotFoundError(
                f"Ratings file not found at {RATINGS_FILE}\n"
                f"Please download MovieLens 1M dataset and place ratings.dat in {RAW_DATA_DIR}"
            )

        # Read with proper delimiter
        self.ratings_df = pd.read_csv(
            RATINGS_FILE,
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={
                'user_id': np.int32,
                'movie_id': np.int32,
                'rating': np.int8,
                'timestamp': np.int64
            }
        )

        # Convert timestamp to datetime
        self.ratings_df['datetime'] = pd.to_datetime(
            self.ratings_df['timestamp'],
            unit='s'
        )

        print(f"  ✓ Loaded {len(self.ratings_df):,} ratings")
        print(f"  ✓ Date range: {self.ratings_df['datetime'].min()} to {self.ratings_df['datetime'].max()}")

        return self.ratings_df

    def load_movies(self) -> pd.DataFrame:
        """
        Load movies.dat file
        Format: MovieID::Title::Genres
        """
        print("\n🎬 Loading movies data...")

        if not MOVIES_FILE.exists():
            raise FileNotFoundError(
                f"Movies file not found at {MOVIES_FILE}\n"
                f"Please download MovieLens 1M dataset and place movies.dat in {RAW_DATA_DIR}"
            )

        # Read with proper delimiter
        self.movies_df = pd.read_csv(
            MOVIES_FILE,
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres'],
            dtype={'movie_id': np.int32},
            encoding='latin-1'  # Handle special characters
        )

        # Extract year from title
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)

        # Clean title (remove year)
        self.movies_df['title_clean'] = self.movies_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)

        print(f"  ✓ Loaded {len(self.movies_df):,} movies")
        print(f"  ✓ Year range: {self.movies_df['year'].min():.0f} to {self.movies_df['year'].max():.0f}")

        return self.movies_df

    def load_users(self) -> pd.DataFrame:
        """
        Load users.dat file
        Format: UserID::Gender::Age::Occupation::Zip-code
        """
        print("\n👥 Loading users data...")

        if not USERS_FILE.exists():
            raise FileNotFoundError(
                f"Users file not found at {USERS_FILE}\n"
                f"Please download MovieLens 1M dataset and place users.dat in {RAW_DATA_DIR}"
            )

        # Read with proper delimiter
        self.users_df = pd.read_csv(
            USERS_FILE,
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zipcode'],
            dtype={
                'user_id': np.int32,
                'gender': 'category',
                'age': np.int8,
                'occupation': np.int8
            }
        )

        # Map age groups
        self.users_df['age_group'] = self.users_df['age'].map(AGE_MAPPING)

        # Map occupations
        self.users_df['occupation_name'] = self.users_df['occupation'].map(OCCUPATION_MAPPING)

        # Extract state from zipcode (first 2-3 digits for US)
        self.users_df['state'] = self.users_df['zipcode'].str[:2]

        print(f"  ✓ Loaded {len(self.users_df):,} users")
        print(f"  ✓ Gender distribution: {self.users_df['gender'].value_counts().to_dict()}")

        return self.users_df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three datasets"""
        print("=" * 60)
        print("LOADING MOVIELENS 1M DATASET")
        print("=" * 60)

        ratings = self.load_ratings()
        movies = self.load_movies()
        users = self.load_users()

        print("\n" + "=" * 60)
        print("✓ All data loaded successfully!")
        print("=" * 60)

        return ratings, movies, users

    def get_statistics(self) -> Dict:
        """Generate dataset statistics"""
        if any(df is None for df in [self.ratings_df, self.movies_df, self.users_df]):
            raise ValueError("Please load data first using load_all()")

        stats = {
            'n_ratings': len(self.ratings_df),
            'n_users': self.users_df['user_id'].nunique(),
            'n_movies': self.movies_df['movie_id'].nunique(),
            'rating_range': (self.ratings_df['rating'].min(), self.ratings_df['rating'].max()),
            'avg_rating': self.ratings_df['rating'].mean(),
            'sparsity': 1 - (len(self.ratings_df) / (
                        self.users_df['user_id'].nunique() * self.movies_df['movie_id'].nunique())),
            'ratings_per_user': self.ratings_df.groupby('user_id').size().describe().to_dict(),
            'ratings_per_movie': self.ratings_df.groupby('movie_id').size().describe().to_dict(),
            'gender_distribution': self.users_df['gender'].value_counts().to_dict(),
            'age_distribution': self.users_df['age_group'].value_counts().to_dict()
        }

        return stats


def print_data_summary(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame):
    """Print comprehensive data summary"""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print("\n📊 RATINGS:")
    print(f"  Total ratings: {len(ratings_df):,}")
    print(f"  Rating range: {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")
    print(f"  Average rating: {ratings_df['rating'].mean():.2f}")
    print(f"  Rating distribution:")
    for rating in sorted(ratings_df['rating'].unique()):
        count = (ratings_df['rating'] == rating).sum()
        pct = count / len(ratings_df) * 100
        print(f"    {rating} stars: {count:>7,} ({pct:>5.2f}%)")

    print("\n🎬 MOVIES:")
    print(f"  Total movies: {len(movies_df):,}")
    print(f"  Unique genres: {len(GENRE_LIST)}")
    print(f"  Year range: {movies_df['year'].min():.0f} - {movies_df['year'].max():.0f}")

    print("\n👥 USERS:")
    print(f"  Total users: {len(users_df):,}")
    print(f"  Gender distribution:")
    for gender, count in users_df['gender'].value_counts().items():
        pct = count / len(users_df) * 100
        print(f"    {gender}: {count:>5,} ({pct:>5.2f}%)")

    print("\n📈 SPARSITY:")
    n_possible = len(users_df) * len(movies_df)
    sparsity = 1 - (len(ratings_df) / n_possible)
    density = 1 - sparsity
    print(f"  Possible ratings: {n_possible:,}")
    print(f"  Actual ratings: {len(ratings_df):,}")
    print(f"  Density: {density * 100:.2f}%")
    print(f"  Sparsity: {sparsity * 100:.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test data loading
    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_all()
    print_data_summary(ratings, movies, users)

    # Get statistics
    stats = loader.get_statistics()
    print("\n📊 Statistics computed successfully!")