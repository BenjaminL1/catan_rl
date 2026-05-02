"""Tile encoder: per-tile features → transformer self-attention → flattened repr.

The 19 hex tiles get projected into ``tile_model_dim``, passed through a
TransformerEncoder so tiles can "see" each other, then projected down to
``proj_tile_dim`` and concatenated into a single flat vector.

Phase 2 enhancements (all opt-in via flags):

  - **2.1 Axial positional embedding.** Without positional info the
    transformer is permutation-equivariant over tiles, which is wrong:
    Catan's board has fixed spatial structure. We add a learned 2D
    embedding indexed by hex axial coordinates ``(q, r)`` to each tile's
    feature vector before the projection layer.
  - **2.2 Pre-norm + GELU + dropout.** Modern Transformer recipe. The
    encoder layer was already pre-norm; we now expose dropout (default
    0.05 in Phase 2) and switch the FFN activation to GELU. Both are
    drop-in replacements that help small-batch training stability.

Defaults preserve back-compat: ``use_axial_pos_emb=False``,
``transformer_dropout=0.0``, ``transformer_activation='relu'``.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn as nn

from catan_rl.models.utils import init_weights


@functools.lru_cache(maxsize=1)
def _hex_axial_coords() -> np.ndarray:
    """Axial coordinates ``(q, r)`` for the 19 hex tiles, in tile-index order.

    Cached (one ``catanBoard()`` construction) and shifted into a non-negative
    range so they can index a 1D ``nn.Embedding`` directly: ``q ∈ [-2..2]``
    becomes ``q + 2 ∈ [0..4]``, same for ``r``.
    """
    from catan_rl.engine.board import catanBoard

    board = catanBoard()
    coords = np.zeros((19, 2), dtype=np.int64)
    for i in range(19):
        tile = board.hexTileDict[i]
        coords[i, 0] = tile.q + 2  # shift to non-negative
        coords[i, 1] = tile.r + 2
    return coords


_AXIAL_RANGE = 5  # q, r each ∈ [0..4] after shifting; 5 distinct values


class TileEncoder(nn.Module):
    """Per-tile transformer encoder.

    Args:
        tile_in_dim: Feature width per tile (Phase 0/1: 79).
        tile_model_dim: Width of the transformer's d_model.
        tile_model_num_heads: Number of attention heads.
        tile_encoder_num_layers: Number of TransformerEncoder layers.
        proj_tile_dim: Output width per tile after the final projection.
        dropout: Encoder-internal dropout (Phase 2.2 default 0.05 in
            ``configs/phase2_full.yaml``; 0.0 elsewhere).
        activation: Transformer FFN activation (Phase 2.2 default ``'gelu'``
            in ``configs/phase2_full.yaml``; ``'relu'`` elsewhere).
        use_axial_pos_emb: Phase 2.1 — add a learned 2D embedding indexed by
            ``(q, r)`` axial coords before the input projection. Default
            False to preserve back-compat with Phase 0/1 checkpoints.
        axial_pos_dim: Width of the axial positional embedding (split half
            for q, half for r).
    """

    def __init__(
        self,
        tile_in_dim: int,
        tile_model_dim: int,
        tile_model_num_heads: int = 4,
        tile_encoder_num_layers: int = 2,
        proj_tile_dim: int = 25,
        dropout: float = 0.0,
        activation: str = "relu",
        use_axial_pos_emb: bool = False,
        axial_pos_dim: int = 24,
    ) -> None:
        super().__init__()
        self.use_axial_pos_emb = bool(use_axial_pos_emb)
        self._axial_pos_dim = int(axial_pos_dim)

        # Phase 2.1: register axial positional embedding. Each axis (q, r)
        # gets half the budget; sum produces the per-tile pos vector. Indices
        # are static (board topology is fixed), so we register the tile→(q,r)
        # lookup as a buffer for trivial device handoff.
        if self.use_axial_pos_emb:
            if axial_pos_dim % 2 != 0:
                raise ValueError(f"axial_pos_dim must be even, got {axial_pos_dim}")
            half = axial_pos_dim // 2
            self.q_emb = nn.Embedding(_AXIAL_RANGE, half)
            self.r_emb = nn.Embedding(_AXIAL_RANGE, half)
            # Small-gain init so the new dims start near zero and don't
            # overpower the input features for the first few updates.
            nn.init.normal_(self.q_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.r_emb.weight, mean=0.0, std=0.02)
            coords = _hex_axial_coords()
            self.register_buffer("_q_idx", torch.from_numpy(coords[:, 0]).long())
            self.register_buffer("_r_idx", torch.from_numpy(coords[:, 1]).long())
            effective_in_dim = tile_in_dim + axial_pos_dim
        else:
            effective_in_dim = tile_in_dim

        # Project raw per-tile features (with optional pos-emb) into the
        # model dimension.
        self.proj_in = nn.Sequential(
            init_weights(nn.Linear(effective_in_dim, tile_model_dim)),
            nn.LayerNorm(tile_model_dim),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tile_model_dim,
            nhead=tile_model_num_heads,
            dim_feedforward=tile_model_dim * 2,
            dropout=dropout,
            activation=activation,
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
            tile_features: ``(B, N_tiles, tile_in_dim)``

        Returns:
            ``(B, N_tiles * proj_tile_dim)`` flattened per-tile representations.
        """
        if self.use_axial_pos_emb:
            B = tile_features.shape[0]
            # (N_tiles, half) → (1, N_tiles, half) → (B, N_tiles, half)
            q = self.q_emb(self._q_idx).unsqueeze(0).expand(B, -1, -1)
            r = self.r_emb(self._r_idx).unsqueeze(0).expand(B, -1, -1)
            pos = torch.cat([q, r], dim=-1)
            tile_features = torch.cat([tile_features, pos], dim=-1)

        x = self.proj_in(tile_features)  # (B, N, model_dim)
        x = self.encoder(x)  # (B, N, model_dim)
        x = self.proj_out(x)  # (B, N, proj_tile_dim)
        B, N, D = x.shape
        return x.reshape(B, N * D)
