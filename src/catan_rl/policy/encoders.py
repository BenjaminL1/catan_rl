"""Obs-side encoder modules for the v2 policy network.

Components:
  * :class:`TileEncoder` — 2-layer transformer over the 19 hex tiles with a
    learned axial positional embedding (q, r). Pre-norm GeLU recipe.
  * :class:`GraphEncoder` — lightweight tripartite GNN over hex / vertex /
    edge nodes. 2 rounds of mean-pool message passing.
  * :class:`CountDevEncoder` — bincount-style dev card encoder; replaces v1's
    MHA + sum-pool. Permutation-invariant by construction.
  * :class:`PlayerEncoder` — 2-layer MLP for the per-player scalar feature
    vector (54 dim for the agent, 61 dim for the opponent).

All modules use module-level constants from :mod:`catan_rl.policy.obs_schema`.
"""

from __future__ import annotations

import torch
from torch import nn

from catan_rl.policy.obs_schema import (
    CURR_PLAYER_DIM,
    N_DEV_TYPES,
    N_EDGES,
    N_TILES,
    N_VERTICES,
    NEXT_PLAYER_DIM,
    TILE_DIM,
)

# ---------------------------------------------------------------------------
# TileEncoder
# ---------------------------------------------------------------------------


class _AxialPositionalEmbedding(nn.Module):
    """Learned 2D positional embedding indexed by axial (q, r) coordinates.

    The Catan board has a fixed canonical layout (19 tiles arranged as a
    hexagonal cluster). We index tiles by their axial coordinates ``(q, r)``
    and learn a separate embedding for each axis, then concatenate. Init
    std=0.02 so the embedding starts as a small perturbation on the raw
    tile features (the encoder is approximately permutation-equivariant
    at the start of training).

    The actual ``(q, r)`` table is wired in by ``CatanPolicy`` via
    :meth:`set_axial_indices` — Step 2 uses placeholder zeros so the
    network is testable in isolation; Step 3 plumbs the real values
    derived from the board geometry.
    """

    # Declared for mypy: register_buffer attributes are otherwise typed
    # ``Tensor | Module`` via nn.Module.__getattr__ (mypy 2.x strictness).
    q_idx: torch.Tensor
    r_idx: torch.Tensor

    def __init__(self, q_range: int = 5, r_range: int = 5, dim: int = 24) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"axial dim must be even, got {dim}")
        half = dim // 2
        self.q_emb = nn.Embedding(q_range, half)
        self.r_emb = nn.Embedding(r_range, half)
        nn.init.normal_(self.q_emb.weight, std=0.02)
        nn.init.normal_(self.r_emb.weight, std=0.02)
        # Placeholder indices: index 0 for all tiles. Step-3 will overwrite.
        self.register_buffer("q_idx", torch.zeros(N_TILES, dtype=torch.long))
        self.register_buffer("r_idx", torch.zeros(N_TILES, dtype=torch.long))

    def set_axial_indices(self, q_idx: torch.Tensor, r_idx: torch.Tensor) -> None:
        """Wire in the real (q, r) coordinates derived from the board geometry."""
        if q_idx.shape != (N_TILES,) or r_idx.shape != (N_TILES,):
            raise ValueError(f"axial indices must be ({N_TILES},)")
        self.q_idx.copy_(q_idx)
        self.r_idx.copy_(r_idx)

    def forward(self) -> torch.Tensor:
        """Return (N_TILES, dim) — broadcast over the batch by the caller."""
        return torch.cat([self.q_emb(self.q_idx), self.r_emb(self.r_idx)], dim=-1)


class TileEncoder(nn.Module):
    """Per-tile feature -> per-tile context vector via transformer + pos emb.

    Output is (B, N_TILES, out_dim). Flattening to (B, N_TILES * out_dim)
    is left to the fusion stage so :class:`GraphEncoder` can also consume
    the per-tile vectors directly.
    """

    def __init__(
        self,
        in_dim: int = TILE_DIM,
        d_model: int = 96,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_ff: int = 192,
        # 0.0 (not 0.05): the training loop never toggles .eval()/.train(), so a
        # non-zero transformer dropout makes the rollout old_log_prob and the SGD
        # new_log_prob use DIFFERENT dropout masks -> the PPO ratio != 1 at epoch 0
        # before any gradient step, inflating clip_frac + approx_kl (the early-stop
        # gate). No-param layer -> existing checkpoints load unchanged.
        dropout: float = 0.0,
        axial_dim: int = 24,
        out_dim: int = 25,
    ) -> None:
        super().__init__()
        self.pos_emb = _AxialPositionalEmbedding(dim=axial_dim)
        self.input_proj = nn.Linear(in_dim + axial_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, out_dim)
        self.out_dim = out_dim

    def forward(self, tile_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tile_feats: (B, N_TILES, in_dim)
        Returns:
            (B, N_TILES, out_dim)
        """
        if tile_feats.dim() != 3 or tile_feats.shape[1] != N_TILES:
            raise ValueError(f"expected (B, {N_TILES}, _), got {tuple(tile_feats.shape)}")
        pos = self.pos_emb()  # (N_TILES, axial_dim)
        pos = pos.unsqueeze(0).expand(tile_feats.size(0), -1, -1)
        x = torch.cat([tile_feats, pos], dim=-1)
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# GraphEncoder (tripartite hex / vertex / edge)
# ---------------------------------------------------------------------------


class GraphEncoder(nn.Module):
    """Tripartite GNN over the hex / vertex / edge graph.

    Each node type has its own initial-projection MLP and aggregation
    bias. Two rounds of mean-pool message passing exchange information
    between the three layers. Output is a single pooled vector per game
    (mean over all hex nodes after the final round).

    Adjacency tables are registered as buffers — :class:`CatanPolicy`
    wires the real values in at construction time; Step 2 uses zero
    placeholders so the network is testable on synthetic data.
    """

    # Declared for mypy: register_buffer attributes are otherwise typed
    # ``Tensor | Module`` via nn.Module.__getattr__ (mypy 2.x strictness).
    hex_to_vertex: torch.Tensor
    vertex_to_hex: torch.Tensor
    vertex_to_hex_mask: torch.Tensor
    edge_to_vertex: torch.Tensor
    vertex_to_edge: torch.Tensor
    vertex_to_edge_mask: torch.Tensor

    def __init__(
        self,
        hex_in_dim: int = 19,
        vertex_in_dim: int = 16,
        edge_in_dim: int = 16,
        hidden_dim: int = 64,
        n_rounds: int = 2,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_rounds = n_rounds
        # Exposed so the pointer readouts (D1) can size their per-node input to
        # the GNN's per-node state width without hardcoding it.
        self.hidden_dim = hidden_dim
        self.hex_proj = nn.Linear(hex_in_dim, hidden_dim)
        self.vertex_proj = nn.Linear(vertex_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        # Per-round message-passing weights.
        self.hex_update = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_rounds)]
        )
        self.vertex_update = nn.ModuleList(
            [nn.Linear(hidden_dim * 3, hidden_dim) for _ in range(n_rounds)]
        )
        self.edge_update = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_rounds)]
        )
        self.act = nn.GELU()
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

        # Adjacency placeholders. Masks default to 1.0 (everything is a valid
        # neighbor) so gradient flows through the placeholder graph at
        # construction time — the real 0/1 pattern is plumbed in by
        # :meth:`set_adjacency` once the board geometry is known.
        # (N_TILES, 6) — each hex has 6 corner vertices.
        self.register_buffer("hex_to_vertex", torch.zeros(N_TILES, 6, dtype=torch.long))
        # (N_VERTICES, 3) — each vertex touches up to 3 hexes; padding marked by mask=0.
        self.register_buffer("vertex_to_hex", torch.zeros(N_VERTICES, 3, dtype=torch.long))
        self.register_buffer("vertex_to_hex_mask", torch.ones(N_VERTICES, 3, dtype=torch.float32))
        # (N_EDGES, 2) — each edge connects 2 vertices.
        self.register_buffer("edge_to_vertex", torch.zeros(N_EDGES, 2, dtype=torch.long))
        # (N_VERTICES, 3) — each vertex touches up to 3 edges.
        self.register_buffer("vertex_to_edge", torch.zeros(N_VERTICES, 3, dtype=torch.long))
        self.register_buffer("vertex_to_edge_mask", torch.ones(N_VERTICES, 3, dtype=torch.float32))

    def set_adjacency(
        self,
        hex_to_vertex: torch.Tensor,
        vertex_to_hex: torch.Tensor,
        vertex_to_hex_mask: torch.Tensor,
        edge_to_vertex: torch.Tensor,
        vertex_to_edge: torch.Tensor,
        vertex_to_edge_mask: torch.Tensor,
    ) -> None:
        self.hex_to_vertex.copy_(hex_to_vertex)
        self.vertex_to_hex.copy_(vertex_to_hex)
        self.vertex_to_hex_mask.copy_(vertex_to_hex_mask)
        self.edge_to_vertex.copy_(edge_to_vertex)
        self.vertex_to_edge.copy_(vertex_to_edge)
        self.vertex_to_edge_mask.copy_(vertex_to_edge_mask)

    @staticmethod
    def _gather_mean(
        nodes: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            nodes: (B, N_src, D)
            idx: (N_dst, K)
            mask: (N_dst, K) — 1.0 for valid, 0.0 for padded; if ``None`` all valid.
        Returns:
            (B, N_dst, D) — mean over K source neighbors, ignoring masked slots.
        """
        b = nodes.size(0)  # noqa: F841
        # gathered: (B, N_dst, K, D)
        gathered = nodes[:, idx]
        if mask is None:
            return gathered.mean(dim=2)
        m = mask.unsqueeze(0).unsqueeze(-1)  # (1, N_dst, K, 1)
        denom = m.sum(dim=2).clamp(min=1.0)
        return (gathered * m).sum(dim=2) / denom

    def forward(
        self, hex_feats: torch.Tensor, vertex_feats: torch.Tensor, edge_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hex_feats: (B, N_TILES, hex_in_dim)
            vertex_feats: (B, N_VERTICES, vertex_in_dim)
            edge_feats: (B, N_EDGES, edge_in_dim)
        Returns:
            ``(pooled, v, e, h)`` where ``pooled`` is (B, out_dim) — the
            mean-pool over hex nodes after MP, then projected (the legacy
            trunk contribution) — and ``v`` (B, N_VERTICES, hidden_dim),
            ``e`` (B, N_EDGES, hidden_dim), ``h`` (B, N_TILES, hidden_dim)
            are the post-final-round per-node states the pointer readouts
            (D1) consume. The pooled path and topology are byte-unchanged;
            the per-node tensors were already computed and simply discarded
            at the mean-pool before this fork.
        """
        h = self.act(self.hex_proj(hex_feats))
        v = self.act(self.vertex_proj(vertex_feats))
        e = self.act(self.edge_proj(edge_feats))

        for r in range(self.n_rounds):
            # Round order is chosen so the round-r updates ALL feed into the
            # final round-r ``h`` (which is what we pool over). Edges go
            # first so vertices can read the freshly-updated edges; vertices
            # go second so hexes can read the freshly-updated vertices.

            # Edges aggregate from their 2 endpoint vertices.
            msg_e_from_v = self._gather_mean(v, self.edge_to_vertex)
            e = self.norm_e(e + self.act(self.edge_update[r](torch.cat([e, msg_e_from_v], dim=-1))))

            # Vertices aggregate from their hexes + updated edges.
            msg_v_from_h = self._gather_mean(h, self.vertex_to_hex, self.vertex_to_hex_mask)
            msg_v_from_e = self._gather_mean(e, self.vertex_to_edge, self.vertex_to_edge_mask)
            v = self.norm_v(
                v
                + self.act(
                    self.vertex_update[r](torch.cat([v, msg_v_from_h, msg_v_from_e], dim=-1))
                )
            )

            # Hexes aggregate from their 6 corner vertices.
            msg_h_from_v = self._gather_mean(v, self.hex_to_vertex)
            h = self.norm_h(h + self.act(self.hex_update[r](torch.cat([h, msg_h_from_v], dim=-1))))

        pooled = h.mean(dim=1)
        return self.output_proj(pooled), v, e, h


# ---------------------------------------------------------------------------
# Dev card encoder (count-based)
# ---------------------------------------------------------------------------


class CountDevEncoder(nn.Module):
    """Bincount + 2-layer MLP — replaces v1's MHA + sum-pool over a padded sequence.

    Input is a (B, N_DEV_TYPES) tensor of integer counts (already produced
    by the obs encoder via :func:`numpy.bincount` over the card list);
    output is (B, out_dim). Permutation invariance is by construction.
    """

    def __init__(self, hidden_dim: int = 32, out_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_DEV_TYPES, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        if counts.dim() != 2 or counts.shape[-1] != N_DEV_TYPES:
            raise ValueError(f"expected (B, {N_DEV_TYPES}), got {tuple(counts.shape)}")
        return self.net(counts.float())


# ---------------------------------------------------------------------------
# Player encoder
# ---------------------------------------------------------------------------


class PlayerEncoder(nn.Module):
    """2-layer MLP for the per-player scalar feature vector."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"expected last dim={self.in_dim}, got {x.shape[-1]}")
        return self.net(x)


def curr_player_encoder() -> PlayerEncoder:
    """Convenience constructor for the agent's player module."""
    return PlayerEncoder(in_dim=CURR_PLAYER_DIM, out_dim=128)


def opp_player_encoder() -> PlayerEncoder:
    """Convenience constructor for the opponent's player module."""
    return PlayerEncoder(in_dim=NEXT_PLAYER_DIM, out_dim=128)
