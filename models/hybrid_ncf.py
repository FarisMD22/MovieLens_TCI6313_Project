"""
Deep Hybrid Neural Collaborative Filtering (Hybrid NCF)
Combines GMF + MLP + Content Features for improved recommendations
"""

import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path


class HybridNCF(nn.Module):
    """
    Deep Hybrid NCF with multiple paths:
    1. GMF (Generalized Matrix Factorization) - multiplicative interaction
    2. MLP (Multi-Layer Perceptron) - deep learning
    3. Content features - genres, demographics
    """

    def __init__(
            self,
            n_users: int,
            n_movies: int,
            user_feature_dim: int = 62,
            movie_feature_dim: int = 29,
            embedding_dim_gmf: int = 64,
            embedding_dim_mlp: int = 64,
            mlp_layers: list = [512, 256, 128],
            content_dim: int = 64,
            dropout_rate: float = 0.3,
            use_content_features: bool = True
    ):
        """
        Args:
            n_users: Number of users
            n_movies: Number of movies
            user_feature_dim: User feature dimensions
            movie_feature_dim: Movie feature dimensions
            embedding_dim_gmf: GMF embedding dimension
            embedding_dim_mlp: MLP embedding dimension
            mlp_layers: MLP hidden layer sizes
            content_dim: Content feature processing dimension
            dropout_rate: Dropout rate
            use_content_features: Whether to use content features
        """
        super(HybridNCF, self).__init__()

        self.n_users = n_users
        self.n_movies = n_movies
        self.use_content_features = use_content_features

        # ===== GMF Path =====
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim_gmf)
        self.movie_embedding_gmf = nn.Embedding(n_movies, embedding_dim_gmf)

        # ===== MLP Path =====
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim_mlp)
        self.movie_embedding_mlp = nn.Embedding(n_movies, embedding_dim_mlp)

        # MLP layers
        mlp_input_dim = embedding_dim_mlp * 2
        mlp_modules = []

        for hidden_size in mlp_layers:
            mlp_modules.append(nn.Linear(mlp_input_dim, hidden_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.BatchNorm1d(hidden_size))
            mlp_modules.append(nn.Dropout(dropout_rate))
            mlp_input_dim = hidden_size

        self.mlp = nn.Sequential(*mlp_modules)

        # ===== Content Features Path (Optional) =====
        if use_content_features:
            # User content processing
            self.user_content_processor = nn.Sequential(
                nn.Linear(user_feature_dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(128, content_dim),
                nn.ReLU()
            )

            # Movie content processing
            self.movie_content_processor = nn.Sequential(
                nn.Linear(movie_feature_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(64, content_dim),
                nn.ReLU()
            )

            fusion_input_dim = embedding_dim_gmf + mlp_layers[-1] + (content_dim * 2)
        else:
            fusion_input_dim = embedding_dim_gmf + mlp_layers[-1]

        # ===== Fusion Layer =====
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        # Embeddings
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.movie_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.movie_embedding_mlp.weight, std=0.01)

        # Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, user_ids, movie_ids, user_features=None, movie_features=None):
        """
        Forward pass

        Args:
            user_ids: User IDs (batch_size,)
            movie_ids: Movie IDs (batch_size,)
            user_features: User content features (batch_size, user_feature_dim)
            movie_features: Movie content features (batch_size, movie_feature_dim)

        Returns:
            Predicted ratings (batch_size,)
        """
        # ===== GMF Path =====
        user_embed_gmf = self.user_embedding_gmf(user_ids)
        movie_embed_gmf = self.movie_embedding_gmf(movie_ids)
        gmf_output = user_embed_gmf * movie_embed_gmf  # Element-wise product

        # ===== MLP Path =====
        user_embed_mlp = self.user_embedding_mlp(user_ids)
        movie_embed_mlp = self.movie_embedding_mlp(movie_ids)
        mlp_input = torch.cat([user_embed_mlp, movie_embed_mlp], dim=1)
        mlp_output = self.mlp(mlp_input)

        # ===== Content Features Path =====
        if self.use_content_features and user_features is not None and movie_features is not None:
            user_content = self.user_content_processor(user_features)
            movie_content = self.movie_content_processor(movie_features)

            # Concatenate all paths
            fusion_input = torch.cat([gmf_output, mlp_output, user_content, movie_content], dim=1)
        else:
            # Concatenate GMF and MLP only
            fusion_input = torch.cat([gmf_output, mlp_output], dim=1)

        # ===== Fusion =====
        output = self.fusion(fusion_input).squeeze()

        # Clip to rating range [1, 5]
        output = torch.clamp(output, min=1.0, max=5.0)

        return output

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing Hybrid NCF model...")

    n_users = 6041
    n_movies = 3953
    batch_size = 32

    # Create model
    model = HybridNCF(
        n_users=n_users,
        n_movies=n_movies,
        user_feature_dim=62,
        movie_feature_dim=29,
        embedding_dim_gmf=64,
        embedding_dim_mlp=64,
        mlp_layers=[512, 256, 128],
        content_dim=64,
        use_content_features=True
    )

    print(f"\n📊 Model Info:")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.count_parameters() * 4 / (1024 ** 2):.2f} MB")

    # Test forward pass
    user_ids = torch.randint(0, n_users, (batch_size,))
    movie_ids = torch.randint(0, n_movies, (batch_size,))
    user_features = torch.randn(batch_size, 62)
    movie_features = torch.randn(batch_size, 29)

    output = model(user_ids, movie_ids, user_features, movie_features)

    print(f"\n📦 Forward pass test:")
    print(f"  Input shape: ({batch_size},)")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
    print("\n✓ Model test passed!")