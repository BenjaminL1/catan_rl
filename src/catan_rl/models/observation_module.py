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

import torch
import torch.nn as nn

from catan_rl.models.multi_headed_attention import MultiHeadedAttention
from catan_rl.models.player_modules import CurrentPlayerModule, OtherPlayersModule
from catan_rl.models.tile_encoder import TileEncoder
from catan_rl.models.utils import init_weights


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
        use_devcard_mha: bool = True,
        max_dev_seq: int = 16,
        dev_card_vocab_excl_pad: int = 5,
        # Phase 2 model upgrades (all opt-in to preserve back-compat).
        use_axial_pos_emb: bool = False,
        axial_pos_dim: int = 24,
        transformer_dropout: float | None = None,
        transformer_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_output_dim = obs_output_dim
        self.use_devcard_mha = bool(use_devcard_mha)

        # Phase 2.2: ``transformer_dropout`` overrides ``dropout`` for the
        # encoder layers when explicitly set, so we can opt in to
        # transformer-internal dropout without disturbing other sites.
        encoder_dropout = transformer_dropout if transformer_dropout is not None else dropout

        # Per-tile encoder.
        self.tile_encoder = TileEncoder(
            tile_in_dim=tile_in_dim,
            tile_model_dim=tile_model_dim,
            tile_model_num_heads=tile_model_num_heads,
            tile_encoder_num_layers=tile_encoder_num_layers,
            proj_tile_dim=proj_tile_dim,
            dropout=encoder_dropout,
            activation=transformer_activation,
            use_axial_pos_emb=use_axial_pos_emb,
            axial_pos_dim=axial_pos_dim,
        )

        # Phase 1.4: when count-encoding, the embedding+MHA pipeline is
        # entirely unused. We construct it only in legacy mode so that the
        # parameter count under the new flag genuinely drops by ~30k.
        if self.use_devcard_mha:
            self.dev_card_embedding = nn.Embedding(6, dev_card_embed_dim)
            self.hidden_card_mha = MultiHeadedAttention(
                dev_card_model_dim, dev_card_model_num_heads
            )
            self.played_card_mha = MultiHeadedAttention(
                dev_card_model_dim, dev_card_model_num_heads
            )
        else:
            self.dev_card_embedding = None
            self.hidden_card_mha = None
            self.played_card_mha = None

        # Player modules
        self.current_player_module = CurrentPlayerModule(
            main_input_dim=curr_player_main_in_dim,
            dev_card_embed_dim=dev_card_embed_dim,
            dev_card_model_dim=dev_card_model_dim,
            proj_dev_card_dim=proj_dev_card_dim,
            use_count_encoding=not self.use_devcard_mha,
            dev_card_vocab_excl_pad=dev_card_vocab_excl_pad,
            max_dev_seq=max_dev_seq,
        )
        self.other_players_module = OtherPlayersModule(
            main_input_dim=other_player_main_in_dim,
            dev_card_embed_dim=dev_card_embed_dim,
            dev_card_model_dim=dev_card_model_dim,
            proj_dev_card_dim=proj_dev_card_dim,
            use_count_encoding=not self.use_devcard_mha,
            dev_card_vocab_excl_pad=dev_card_vocab_excl_pad,
            max_dev_seq=max_dev_seq,
        )

        # Final fusion: tiles (19 * proj_tile_dim) + current(128) + next(128)
        fusion_in_dim = 19 * proj_tile_dim + 2 * 128
        self.final_layer = init_weights(nn.Linear(fusion_in_dim, obs_output_dim))
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(obs_output_dim)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
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
        tile_encodings = self.tile_encoder(
            obs_dict["tile_representations"]
        )  # (B, 19*proj_tile_dim)

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
