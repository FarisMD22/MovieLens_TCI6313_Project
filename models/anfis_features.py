"""
Enhanced ANFIS Feature Extractor
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR


class ANFISFeatureExtractor:
    """Extract enhanced features for ANFIS model"""

    def __init__(self):
        """Initialize feature extractor"""

        print("\n" + "=" * 60)
        print("CREATING ENHANCED ANFIS FEATURE EXTRACTOR")
        print("=" * 60)

        # Load data (same as before)
        ratings_df = pd.read_csv(
            RAW_DATA_DIR / 'ratings.dat',
            sep='::',
            engine='python',
            header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )

        movies_df = pd.read_csv(
            RAW_DATA_DIR / 'movies.dat',
            sep='::',
            engine='python',
            header=None,
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )

        print(f"  ✓ Loaded {len(ratings_df)} ratings")
        print(f"  ✓ Loaded {len(movies_df)} movies")

        # 1. Compute user activity levels (ORIGINAL)
        user_counts = ratings_df.groupby('user_id').size()
        max_count = user_counts.max()
        self.user_activity = (user_counts / max_count).to_dict()

        # 2. Compute movie popularity (ORIGINAL)
        movie_counts = ratings_df.groupby('movie_id').size()
        max_movie_count = movie_counts.max()
        self.movie_popularity = (movie_counts / max_movie_count).to_dict()

        # 3. NEW: User average rating (user bias)
        self.user_avg_rating = ratings_df.groupby('user_id')['rating'].mean().to_dict()

        # 4. NEW: Movie average rating (movie quality)
        self.movie_avg_rating = ratings_df.groupby('movie_id')['rating'].mean().to_dict()

        # 5. NEW: User rating variance (how critical is user?)
        self.user_rating_std = ratings_df.groupby('user_id')['rating'].std().fillna(1.0).to_dict()

        # 6. NEW: Movie rating variance (how polarizing is movie?)
        self.movie_rating_std = ratings_df.groupby('movie_id')['rating'].std().fillna(1.0).to_dict()

        # 7. Compute user-genre preferences (ORIGINAL - slightly modified)
        print("  Computing user-genre preferences...")
        self.user_genre_affinity = {}
        merged = ratings_df.merge(movies_df[['movie_id', 'genres']], on='movie_id')

        for user_id in ratings_df['user_id'].unique():
            user_ratings = merged[merged['user_id'] == user_id]
            genre_ratings = {}

            for _, row in user_ratings.iterrows():
                genres = row['genres'].split('|')
                for genre in genres:
                    if genre not in genre_ratings:
                        genre_ratings[genre] = []
                    genre_ratings[genre].append(row['rating'])

            # Store average rating per genre (not normalized here)
            self.user_genre_affinity[user_id] = {
                genre: np.mean(ratings) for genre, ratings in genre_ratings.items()
            }

        # 8. Store movie genres (ORIGINAL)
        self.movie_genres = movies_df.set_index('movie_id')['genres'].to_dict()

        # NEW: Global statistics for normalization
        self.global_mean = ratings_df['rating'].mean()
        self.global_std = ratings_df['rating'].std()

        print(f"  ✓ User activity levels: {len(self.user_activity)} users")
        print(f"  ✓ Movie popularity: {len(self.movie_popularity)} movies")
        print(f"  ✓ User rating statistics computed")
        print(f"  ✓ Movie rating statistics computed")
        print(f"  ✓ User-genre preferences: {len(self.user_genre_affinity)} users")
        print(f"  ✓ Global mean rating: {self.global_mean:.2f}")
        print(f"  ✓ Global std rating: {self.global_std:.2f}")
        print("✓ Enhanced ANFIS Feature Extractor initialized")

    def extract_features(self, user_ids, movie_ids):
        """
        Extract 8 enhanced features for ANFIS

        Features:
        1. User activity (normalized) - ORIGINAL
        2. Movie popularity (normalized) - ORIGINAL
        3. User average rating (normalized) - NEW
        4. Movie average rating (normalized) - NEW
        5. User rating std (normalized) - NEW
        6. User-genre affinity for this movie - ORIGINAL (enhanced)
        7. User deviation from global mean - NEW
        8. Movie deviation from global mean - NEW

        Args:
            user_ids: Array of user IDs
            movie_ids: Array of movie IDs

        Returns:
            numpy array of shape (batch_size, 8)
        """
        features = []

        for uid, mid in zip(user_ids, movie_ids):
            # Feature 1: User activity level (ORIGINAL)
            activity = self.user_activity.get(uid, 0.5)

            # Feature 2: Movie popularity (ORIGINAL)
            popularity = self.movie_popularity.get(mid, 0.5)

            # Feature 3: User average rating (NEW)
            user_avg = self.user_avg_rating.get(uid, self.global_mean)
            user_avg_norm = (user_avg - 1.0) / 4.0  # Scale from [1,5] to [0,1]

            # Feature 4: Movie average rating (NEW)
            movie_avg = self.movie_avg_rating.get(mid, self.global_mean)
            movie_avg_norm = (movie_avg - 1.0) / 4.0  # Scale from [1,5] to [0,1]

            # Feature 5: User rating variance (NEW)
            user_std = self.user_rating_std.get(uid, 1.0)
            user_std_norm = min(user_std / 2.0, 1.0)  # Normalize, cap at 1

            # Feature 6: User-genre affinity (ORIGINAL - enhanced)
            genre_affinity = 0.5
            if uid in self.user_genre_affinity and mid in self.movie_genres:
                movie_genres = self.movie_genres[mid].split('|')
                genre_ratings = []
                for genre in movie_genres:
                    if genre in self.user_genre_affinity[uid]:
                        genre_ratings.append(self.user_genre_affinity[uid][genre])
                if genre_ratings:
                    # Normalize to [0, 1]
                    genre_affinity = (np.mean(genre_ratings) - 1.0) / 4.0

            # Feature 7: User deviation from global mean (NEW)
            user_deviation = (user_avg - self.global_mean) / self.global_std
            user_deviation_norm = (user_deviation + 3) / 6  # Assume ±3 std, normalize to [0,1]
            user_deviation_norm = np.clip(user_deviation_norm, 0, 1)

            # Feature 8: Movie deviation from global mean (NEW)
            movie_deviation = (movie_avg - self.global_mean) / self.global_std
            movie_deviation_norm = (movie_deviation + 3) / 6  # Assume ±3 std, normalize to [0,1]
            movie_deviation_norm = np.clip(movie_deviation_norm, 0, 1)

            features.append([
                activity,              # 1
                popularity,            # 2
                user_avg_norm,         # 3
                movie_avg_norm,        # 4
                user_std_norm,         # 5
                genre_affinity,        # 6
                user_deviation_norm,   # 7
                movie_deviation_norm   # 8
            ])

        return np.array(features, dtype=np.float32)