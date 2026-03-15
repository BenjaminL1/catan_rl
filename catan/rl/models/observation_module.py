"""
Charlesworth-style observation encoder operating on dict observations.

Environment now returns a dict:
  - tile_representations: (B, 19, 78)
  - current_player_main: (B, 166)
  - current_player_played_dev: list/Tensor of dev card IDs
  - current_player_hidden_dev: list/Tensor of dev card IDs
  - next_player_main: (B, 173)
  - next_player_played_dev: list/Tensor of dev card IDs
  - proposed_trade, current_resources, ... (can be wired in later)
"""

from typing import Dict

import torch
import torch.nn as nn

from catan.rl.models.tile_encoder import TileEncoder
from catan.rl.models.player_modules import CurrentPlayerModule, OtherPlayersModule
from catan.rl.models.multi_headed_attention import MultiHeadedAttention
from catan.rl.models.utils import init_weights


class ObservationModule(nn.Module):
    """Top-level Charlesworth-style observation encoder."""

    def __init__(
        self,
        tile_in_dim: int = 79,
        tile_model_dim: int = 128,
        curr_player_main_in_dim: int = 166,
        other_player_main_in_dim: int = 173,
        dev_card_embed_dim: int = 64,
        dev_card_model_dim: int = 64,
        obs_output_dim: int = 512,
        tile_model_num_heads: int = 4,
        proj_dev_card_dim: int = 25,
        dev_card_model_num_heads: int = 4,
        tile_encoder_num_layers: int = 2,
        proj_tile_dim: int = 25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_output_dim = obs_output_dim

        # Per-tile encoder.
        self.tile_encoder = TileEncoder(
            tile_in_dim=tile_in_dim,
            tile_model_dim=tile_model_dim,
            tile_model_num_heads=tile_model_num_heads,
            tile_encoder_num_layers=tile_encoder_num_layers,
            proj_tile_dim=proj_tile_dim,
            dropout=dropout,
        )

        # Dev card embedding + attention (shared across players)
        self.dev_card_embedding = nn.Embedding(6, dev_card_embed_dim)
        self.hidden_card_mha = MultiHeadedAttention(dev_card_model_dim, dev_card_model_num_heads)
        self.played_card_mha = MultiHeadedAttention(dev_card_model_dim, dev_card_model_num_heads)

        # Player modules
        self.current_player_module = CurrentPlayerModule(
            main_input_dim=curr_player_main_in_dim,
            dev_card_embed_dim=dev_card_embed_dim,
            dev_card_model_dim=dev_card_model_dim,
            proj_dev_card_dim=proj_dev_card_dim,
        )
        self.other_players_module = OtherPlayersModule(
            main_input_dim=other_player_main_in_dim,
            dev_card_embed_dim=dev_card_embed_dim,
            dev_card_model_dim=dev_card_model_dim,
            proj_dev_card_dim=proj_dev_card_dim,
        )

        # Final fusion: tiles (19 * proj_tile_dim) + current(128) + next(128)
        fusion_in_dim = 19 * proj_tile_dim + 2 * 128
        self.final_layer = init_weights(nn.Linear(fusion_in_dim, obs_output_dim))
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(obs_output_dim)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_dict: dict batch from env:
              - 'tile_representations': (B, 19, 78)
              - 'current_player_main': (B, 166)
              - 'current_player_hidden_dev': list/Tensor
              - 'current_player_played_dev': list/Tensor
              - 'next_player_main': (B, 173)
              - 'next_player_played_dev': list/Tensor
        """
        tile_encodings = self.tile_encoder(obs_dict["tile_representations"])  # (B, 19*proj_tile_dim)

        current_player_output = self.current_player_module(
            obs_dict["current_player_main"],
            obs_dict["current_player_hidden_dev"],
            obs_dict["current_player_played_dev"],
            self.dev_card_embedding,
            self.hidden_card_mha,
            self.played_card_mha,
        )

        other_player_output = self.other_players_module(
            obs_dict["next_player_main"],
            obs_dict["next_player_played_dev"],
            self.dev_card_embedding,
            self.played_card_mha,
        )

        final_input = torch.cat(
            (tile_encodings, current_player_output, other_player_output), dim=-1
        )
        return self.relu(self.norm(self.final_layer(final_input)))


