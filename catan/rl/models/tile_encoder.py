"""
Tile encoder module.

This is a small wrapper around a TransformerEncoder that:
  - Takes per-tile feature vectors of shape (B, N_tiles, tile_in_dim)
  - Applies a stack of self-attention layers so tiles can "see" each other
  - Projects each tile down to ``proj_tile_dim``
  - Flattens all tiles into a single vector per batch element
"""

from typing import Optional

import torch
import torch.nn as nn

from catan.rl.models.utils import init_weights


class TileEncoder(nn.Module):
    def __init__(
        self,
        tile_in_dim: int,
        tile_model_dim: int,
        tile_model_num_heads: int = 4,
        tile_encoder_num_layers: int = 2,
        proj_tile_dim: int = 25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Project raw per-tile features into the model dimension.
        self.proj_in = nn.Sequential(
            init_weights(nn.Linear(tile_in_dim, tile_model_dim)),
            nn.LayerNorm(tile_model_dim),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tile_model_dim,
            nhead=tile_model_num_heads,
            dim_feedforward=tile_model_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=tile_encoder_num_layers,
        )

        # Per-tile projection down to a compact representation.
        self.proj_out = nn.Sequential(
            init_weights(nn.Linear(tile_model_dim, proj_tile_dim)),
            nn.LayerNorm(proj_tile_dim),
            nn.ReLU(),
        )

    def forward(self, tile_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tile_features: (B, N_tiles, tile_in_dim)

        Returns:
            (B, N_tiles * proj_tile_dim) flattened per-tile representations.
        """
        x = self.proj_in(tile_features)   # (B, N, model_dim)
        x = self.encoder(x)               # (B, N, model_dim)
        x = self.proj_out(x)              # (B, N, proj_tile_dim)
        B, N, D = x.shape
        return x.reshape(B, N * D)

