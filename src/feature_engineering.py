"""
Feature engineering for MovieLens 1M dataset
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')

from src.config import GENRE_LIST


class FeatureEngineer:
    """Create features from raw MovieLens data"""

    def __init__(self):
        self.user_scaler = StandardScaler()
        self.movie_scaler = StandardScaler()
        self.mlb_genres = MultiLabelBinarizer()

    def create_user_features(self, users_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive user features

        Features:
        - Demographics: gender, age, occupation
        - Activity: rating count, mean, std
        - Genre preferences: average rating per genre
        - Temporal: days since first/last rating
        """
        print("\n🔧 Engineering user features...")

        user_features = users_df.copy()

        # === Demographic Features ===
        # Gender: Binary encoding
        user_features['gender_encoded'] = (user_features['gender'] == 'M').astype(int)

        # Age: One-hot encoding
        age_dummies = pd.get_dummies(user_features['age'], prefix='age')
        user_features = pd.concat([user_features, age_dummies], axis=1)

        # Occupation: One-hot encoding
        occupation_dummies = pd.get_dummies(user_features['occupation'], prefix='occupation')
        user_features = pd.concat([user_features, occupation_dummies], axis=1)

        # === Activity Features ===
        user_stats = ratings_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'timestamp': ['min', 'max']
        }).reset_index()
        user_stats.columns = ['user_id', 'rating_count', 'rating_mean', 'rating_std', 'first_rating_time',
                              'last_rating_time']

        # Fill NaN std with 0 (users with only 1 rating)
        user_stats['rating_std'].fillna(0, inplace=True)

        # Temporal features
        user_stats['days_active'] = (user_stats['last_rating_time'] - user_stats['first_rating_time']) / (24 * 3600)
        user_stats['avg_ratings_per_day'] = user_stats['rating_count'] / (user_stats['days_active'] + 1)

        # Merge with user features
        user_features = user_features.merge(user_stats, on='user_id', how='left')

        print(f"  ✓ Created {len(user_features.columns)} user feature columns")

        return user_features

    def create_movie_features(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive movie features

        Features:
        - Content: genres (multi-hot), year
        - Popularity: rating count, mean, std
        - Derived: genre count, age of movie
        """
        print("\n🔧 Engineering movie features...")

        movie_features = movies_df.copy()

        # === Genre Features ===
        # Split genres and create multi-hot encoding
        movie_features['genres_list'] = movie_features['genres'].str.split('|')

        # Multi-hot encoding for genres
        genres_encoded = self.mlb_genres.fit_transform(movie_features['genres_list'])
        genre_df = pd.DataFrame(
            genres_encoded,
            columns=[f'genre_{g}' for g in self.mlb_genres.classes_],
            index=movie_features.index
        )
        movie_features = pd.concat([movie_features, genre_df], axis=1)

        # Genre count
        movie_features['genre_count'] = movie_features['genres_list'].apply(len)

        # === Temporal Features ===
        # Movie age (assuming current year is 2000 for consistency)
        reference_year = ratings_df['datetime'].max().year
        movie_features['movie_age'] = reference_year - movie_features['year']
        movie_features['movie_age'].fillna(movie_features['movie_age'].median(), inplace=True)

        # === Popularity Features ===
        movie_stats = ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean', 'std']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'rating_count', 'rating_mean', 'rating_std']
        movie_stats['rating_std'].fillna(0, inplace=True)

        # Merge with movie features
        movie_features = movie_features.merge(movie_stats, on='movie_id', how='left')

        # Fill missing values (movies with no ratings)
        movie_features['rating_count'].fillna(0, inplace=True)
        movie_features['rating_mean'].fillna(movie_features['rating_mean'].mean(), inplace=True)
        movie_features['rating_std'].fillna(0, inplace=True)

        print(f"  ✓ Created {len(movie_features.columns)} movie feature columns")

        return movie_features

    def create_interaction_features(
            self,
            ratings_df: pd.DataFrame,
            user_features: pd.DataFrame,
            movie_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create user-movie interaction features

        Features:
        - User-genre affinity
        - Demographic-movie popularity
        - Temporal features from timestamp
        """
        print("\n🔧 Engineering interaction features...")

        interactions = ratings_df.copy()

        # === Temporal Features ===
        interactions['year'] = interactions['datetime'].dt.year
        interactions['month'] = interactions['datetime'].dt.month
        interactions['day_of_week'] = interactions['datetime'].dt.dayofweek
        interactions['hour'] = interactions['datetime'].dt.hour

        # === User-Genre Affinity ===
        # Calculate average rating per user per genre
        print("  Computing user-genre affinities...")

        # Merge to get genres for each rating
        ratings_with_genres = interactions.merge(
            movie_features[['movie_id', 'genres_list']],
            on='movie_id',
            how='left'
        )

        # Explode genres to get one row per user-genre-rating
        ratings_exploded = ratings_with_genres.explode('genres_list')

        # Calculate user-genre affinity
        user_genre_affinity = ratings_exploded.groupby(['user_id', 'genres_list'])['rating'].mean().reset_index()
        user_genre_affinity.columns = ['user_id', 'genre', 'genre_affinity']

        # Pivot to wide format
        user_genre_pivot = user_genre_affinity.pivot(
            index='user_id',
            columns='genre',
            values='genre_affinity'
        ).reset_index()
        user_genre_pivot.columns = ['user_id'] + [f'affinity_{col}' for col in user_genre_pivot.columns[1:]]

        # Fill missing affinities with global mean
        for col in user_genre_pivot.columns[1:]:
            user_genre_pivot[col].fillna(user_genre_pivot[col].mean(), inplace=True)

        # Merge back to user features
        user_features_enhanced = user_features.merge(user_genre_pivot, on='user_id', how='left')

        print(f"  ✓ Created {len(interactions.columns)} interaction feature columns")

        return interactions, user_features_enhanced

    def normalize_features(
            self,
            user_features: pd.DataFrame,
            movie_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize continuous features
        """
        print("\n🔧 Normalizing features...")

        # User continuous features to normalize
        user_continuous = ['rating_count', 'rating_mean', 'rating_std',
                           'days_active', 'avg_ratings_per_day']

        # Movie continuous features to normalize
        movie_continuous = ['year', 'movie_age', 'rating_count',
                            'rating_mean', 'rating_std', 'genre_count']

        # Normalize user features
        user_features_norm = user_features.copy()
        if all(col in user_features_norm.columns for col in user_continuous):
            user_features_norm[user_continuous] = self.user_scaler.fit_transform(
                user_features_norm[user_continuous]
            )

        # Normalize movie features
        movie_features_norm = movie_features.copy()
        if all(col in movie_features_norm.columns for col in movie_continuous):
            movie_features_norm[movie_continuous] = self.movie_scaler.fit_transform(
                movie_features_norm[movie_continuous]
            )

        print(f"  ✓ Normalized features")

        return user_features_norm, movie_features_norm


if __name__ == "__main__":
    # Test feature engineering
    from src.data_loader import MovieLensDataLoader

    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_all()

    engineer = FeatureEngineer()
    user_feats = engineer.create_user_features(users, ratings)
    movie_feats = engineer.create_movie_features(movies, ratings)

    print(f"\n✓ User features shape: {user_feats.shape}")
    print(f"✓ Movie features shape: {movie_feats.shape}")