"""
Neural Collaborative Filtering (NCF) Model
Based on He et al. (2017) - Neural Collaborative Filtering
"""

import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Neural Collaborative Filtering Model

    Architecture:
    - User and movie embeddings
    - Concatenation
    - Multi-layer perceptron
    - Rating prediction
    """

    def __init__(
            self,
            n_users: int,
            n_movies: int,
            embedding_dim: int = 128,
            hidden_layers: list = [256, 128, 64],
            dropout_rate: float = 0.3
    ):
        """
        Args:
            n_users: Number of users
            n_movies: Number of movies
            embedding_dim: Dimension of embeddings
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(NCF, self).__init__()

        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_dim = embedding_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        # MLP layers
        layers = []
        input_size = embedding_dim * 2

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        self.mlp = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(hidden_layers[-1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, user_ids, movie_ids):
        """
        Forward pass

        Args:
            user_ids: User IDs (batch_size,)
            movie_ids: Movie IDs (batch_size,)

        Returns:
            Predicted ratings (batch_size,)
        """
        # Get embeddings
        user_embed = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        movie_embed = self.movie_embedding(movie_ids)  # (batch_size, embedding_dim)

        # Concatenate embeddings
        x = torch.cat([user_embed, movie_embed], dim=1)  # (batch_size, embedding_dim * 2)

        # MLP
        x = self.mlp(x)  # (batch_size, hidden_layers[-1])

        # Output (rating prediction)
        output = self.output(x).squeeze()  # (batch_size,)

        # Clip to rating range [1, 5]
        output = torch.clamp(output, min=1.0, max=5.0)

        return output

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing NCF model...")

    # Model parameters
    n_users = 6041
    n_movies = 3953
    batch_size = 32

    # Create model
    model = NCF(n_users, n_movies, embedding_dim=128, hidden_layers=[256, 128, 64])

    print(f"\n📊 Model Info:")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Model size: {model.count_parameters() * 4 / (1024 ** 2):.2f} MB")

    # Test forward pass
    user_ids = torch.randint(0, n_users, (batch_size,))
    movie_ids = torch.randint(0, n_movies, (batch_size,))

    output = model(user_ids, movie_ids)

    print(f"\n📦 Forward pass test:")
    print(f"  Input shape: ({batch_size},)")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
    print("\n✓ Model test passed!")