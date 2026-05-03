"""Phase 2.3 — tripartite GNN over hex / vertex / edge graph.

Catan's board has a natural graph structure that the tile transformer
in `tile_encoder.py` doesn't capture: hex tiles share corner *vertices*,
and roads run along *edges* connecting two vertices. Settlements and
cities live on vertices; roads on edges. The full game state lives on
all three node types, but the legacy obs encodes only per-tile features.

This module adds an explicit GNN that:

  - **Initializes node features** from existing tile features
    (no env-side schema change required):
      - hex_node[h]  = projection of tile_features[h]
      - vertex_node[v] = mean-pool over hex_node[h] for each adjacent hex
      - edge_node[e]   = mean-pool over the two vertex_node endpoints
  - **Runs ``n_rounds=2`` rounds of message passing**:
      - hex receives messages from its 6 corner vertices
      - vertex receives messages from its 1–3 adjacent hexes plus its 2–3
        adjacent edges
      - edge receives messages from its 2 endpoint vertices
    Each round is a `Linear → LayerNorm → ReLU` over the concatenation of
    "self features" and "aggregated neighbor features."
  - **Pools** by computing the mean of each node type, concatenates the
    three pooled vectors, and projects to ``out_dim``.

Adjacency is precomputed once at module construction from a single
``catanBoard()`` instance and stored as registered buffers so the
encoder is purely-tensor at forward time.

**[1v1] Why this works in 1v1:**
  The board topology is identical regardless of player count. The GNN's
  inductive bias (3 node types, fixed adjacency) is correct for 4-player
  too — but the rest of the pipeline (action heads, reward shaping,
  belief targets) is 1v1-only. Don't graduate this module to 4-player
  without auditing those.

**Param footprint:** at hidden=64, n_rounds=2, ~30k parameters.
Negligible relative to the policy encoder; the inductive bias is the
expensive part of the value, not the parameter count.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn as nn

from catan_rl.models.utils import init_weights

N_HEXES = 19
N_VERTICES = 54
N_EDGES = 72


def _canonical_edge_key(v1: tuple, v2: tuple) -> tuple:
    """Order-invariant edge key — matches ``CatanEnv._edge_key``."""
    s1, s2 = str(v1), str(v2)
    return (s1, s2) if s1 < s2 else (s2, s1)


@functools.lru_cache(maxsize=1)
def _board_adjacency_tables() -> dict[str, np.ndarray]:
    """Precompute the four adjacency tables we need for message passing.

    Returns a dict with:
      - ``hex_to_vertex``: ``(19, 6)`` int64. Vertex index of each corner of
        each hex; canonical hex-corner ordering matches engine ``getCornerCoordinates``.
      - ``vertex_to_hex``: ``(54, 3)`` int64 with -1 padding. Up to 3
        adjacent hex indices per vertex (corner vertices have only 1 or 2).
      - ``edge_to_vertex``: ``(72, 2)`` int64. The two endpoint vertex
        indices of each edge.
      - ``vertex_to_edge``: ``(54, 3)`` int64 with -1 padding. Up to 3
        edges incident to each vertex.

    Cached because the board topology is fixed; building it costs one
    ``catanBoard()`` construction.
    """
    from catan_rl.engine.board import catanBoard

    board = catanBoard()

    # hex → vertex (each hex has 6 corner vertices in canonical order).
    # ``get_corners(flat)`` is the engine API for hex corner pixel coords.
    hex_to_vertex = np.full((N_HEXES, 6), -1, dtype=np.int64)
    for h_idx in range(N_HEXES):
        tile = board.hexTileDict[h_idx]
        corners = tile.get_corners(board.flat)
        for c_idx, corner_px in enumerate(corners):
            v_idx = board.boardGraph[corner_px].vertex_index
            hex_to_vertex[h_idx, c_idx] = v_idx

    # vertex → hex (use the cached adjacent_hex_indices on each vertex).
    vertex_to_hex = np.full((N_VERTICES, 3), -1, dtype=np.int64)
    for v_px, v_obj in board.boardGraph.items():
        v_idx = v_obj.vertex_index
        adj = list(v_obj.adjacent_hex_indices)
        for slot in range(min(3, len(adj))):
            vertex_to_hex[v_idx, slot] = adj[slot]

    # Edge index map (matching env's deterministic ordering).
    seen: set[tuple] = set()
    edges: list[tuple] = []
    for v_px, v_obj in board.boardGraph.items():
        for nb_px in v_obj.neighbors:
            key = _canonical_edge_key(v_px, nb_px)
            if key not in seen:
                seen.add(key)
                edges.append((v_px, nb_px))
    if len(edges) != N_EDGES:
        raise RuntimeError(f"Expected {N_EDGES} edges from board topology, found {len(edges)}")
    edge_to_vertex = np.zeros((N_EDGES, 2), dtype=np.int64)
    vertex_to_edge_list: list[list[int]] = [[] for _ in range(N_VERTICES)]
    for e_idx, (v1_px, v2_px) in enumerate(edges):
        v1_idx = board.boardGraph[v1_px].vertex_index
        v2_idx = board.boardGraph[v2_px].vertex_index
        edge_to_vertex[e_idx, 0] = v1_idx
        edge_to_vertex[e_idx, 1] = v2_idx
        vertex_to_edge_list[v1_idx].append(e_idx)
        vertex_to_edge_list[v2_idx].append(e_idx)

    vertex_to_edge = np.full((N_VERTICES, 3), -1, dtype=np.int64)
    for v_idx, e_list in enumerate(vertex_to_edge_list):
        for slot in range(min(3, len(e_list))):
            vertex_to_edge[v_idx, slot] = e_list[slot]

    return {
        "hex_to_vertex": hex_to_vertex,
        "vertex_to_hex": vertex_to_hex,
        "edge_to_vertex": edge_to_vertex,
        "vertex_to_edge": vertex_to_edge,
    }


def _gather_neighbors(
    node_features: torch.Tensor, adj: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool neighbor features over a padded adjacency table.

    Args:
        node_features: ``(B, N_src, D)``.
        adj: ``(N_dst, K)`` int64. Index into ``node_features`` axis 1, with
            -1 marking padded slots.
        valid_mask: ``(N_dst, K)`` bool. True where ``adj`` is a real index.

    Returns:
        ``(B, N_dst, D)`` mean-pooled neighbor features. Empty rows
        (no valid neighbors) get zeros.
    """
    B, _, D = node_features.shape
    safe_adj = adj.clamp(min=0)  # -1 → 0; mask zeros out the contribution
    # gather: (B, N_dst, K, D)
    expanded = node_features[:, safe_adj, :]  # (B, N_dst, K, D)
    mask = valid_mask.float().unsqueeze(0).unsqueeze(-1)  # (1, N_dst, K, 1)
    summed = (expanded * mask).sum(dim=2)  # (B, N_dst, D)
    counts = mask.sum(dim=2).clamp(min=1.0)  # (1, N_dst, 1)
    return summed / counts


class GraphEncoder(nn.Module):
    """Tripartite (hex / vertex / edge) message-passing encoder.

    Args:
        tile_in_dim: per-tile input feature width (matches obs schema).
        hidden_dim: shared message-passing width across all node types.
        n_rounds: number of MP rounds. 2 rounds let information travel
            hex → vertex → edge → vertex → hex, which is enough to cover
            the diameter of "two settlements share a hex" / "two roads
            share a vertex" interactions.
        out_dim: width of the pooled+projected output.
    """

    def __init__(
        self,
        tile_in_dim: int,
        hidden_dim: int = 64,
        n_rounds: int = 2,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_rounds = int(n_rounds)

        # Adjacency tables — precompute once and register as buffers so
        # they ride along to whichever device the module is moved to.
        adj = _board_adjacency_tables()
        self.register_buffer("hex_to_vertex", torch.from_numpy(adj["hex_to_vertex"]))
        self.register_buffer("vertex_to_hex", torch.from_numpy(adj["vertex_to_hex"]))
        self.register_buffer("edge_to_vertex", torch.from_numpy(adj["edge_to_vertex"]))
        self.register_buffer("vertex_to_edge", torch.from_numpy(adj["vertex_to_edge"]))
        # Validity masks are the negation of -1 sentinels.
        self.register_buffer("_h2v_mask", (self.hex_to_vertex >= 0))
        self.register_buffer("_v2h_mask", (self.vertex_to_hex >= 0))
        self.register_buffer("_e2v_mask", (self.edge_to_vertex >= 0))
        self.register_buffer("_v2e_mask", (self.vertex_to_edge >= 0))

        # Per-node-type input projections from per-hex tile features.
        # Vertex/edge initial features are derived from hex features via
        # mean-pooling, then projected — see ``forward``.
        self.hex_in_proj = nn.Sequential(
            init_weights(nn.Linear(tile_in_dim, self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.vertex_in_proj = nn.Sequential(
            init_weights(nn.Linear(tile_in_dim, self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.edge_in_proj = nn.Sequential(
            init_weights(nn.Linear(tile_in_dim, self.hidden_dim)),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )

        # One MLP per (node_type, round) for updates.  Each takes
        # ``cat(self_feature, aggregated_neighbor_feature)`` → hidden.
        # For vertex updates we aggregate from BOTH hex neighbors AND
        # edge neighbors, so the input width is hidden + 2 * hidden.
        def _round_mlp(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                init_weights(nn.Linear(in_dim, self.hidden_dim)),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
            )

        self.hex_updates = nn.ModuleList(
            [_round_mlp(2 * self.hidden_dim) for _ in range(self.n_rounds)]
        )
        self.vertex_updates = nn.ModuleList(
            [_round_mlp(3 * self.hidden_dim) for _ in range(self.n_rounds)]
        )
        self.edge_updates = nn.ModuleList(
            [_round_mlp(2 * self.hidden_dim) for _ in range(self.n_rounds)]
        )

        # Final readout: mean-pool each node type, concat, project.
        self.readout = nn.Sequential(
            init_weights(nn.Linear(3 * self.hidden_dim, int(out_dim))),
            nn.LayerNorm(int(out_dim)),
            nn.ReLU(),
        )

    def forward(self, tile_features: torch.Tensor) -> torch.Tensor:
        """Run message passing and pool to a fixed-size representation.

        Args:
            tile_features: ``(B, 19, tile_in_dim)``.

        Returns:
            ``(B, out_dim)`` pooled graph representation.
        """
        B = tile_features.shape[0]

        # Initial node features.
        hex_h = self.hex_in_proj(tile_features)  # (B, 19, hidden)

        # Vertex initial = mean-pool over adjacent hex features (raw),
        # then project. We project *after* pooling so the projection sees
        # the summarized neighborhood, not raw single-tile features.
        vertex_raw = _gather_neighbors(tile_features, self.vertex_to_hex, self._v2h_mask)
        vertex_h = self.vertex_in_proj(vertex_raw)  # (B, 54, hidden)

        # Edge initial = mean-pool over the two endpoint vertices' raw
        # tile-aggregated features, then project. Equivalent to taking
        # the average tile feature seen by either endpoint.
        edge_raw = _gather_neighbors(vertex_raw, self.edge_to_vertex, self._e2v_mask)
        edge_h = self.edge_in_proj(edge_raw)  # (B, 72, hidden)

        # Message passing rounds.
        for r in range(self.n_rounds):
            # Aggregate neighbor features for each node type before any
            # update — this ensures all updates in a round see the SAME
            # round-r snapshot, not a partially-updated state.
            v_from_h = _gather_neighbors(hex_h, self.vertex_to_hex, self._v2h_mask)
            v_from_e = _gather_neighbors(edge_h, self.vertex_to_edge, self._v2e_mask)
            h_from_v = _gather_neighbors(vertex_h, self.hex_to_vertex, self._h2v_mask)
            e_from_v = _gather_neighbors(vertex_h, self.edge_to_vertex, self._e2v_mask)

            hex_h = self.hex_updates[r](torch.cat([hex_h, h_from_v], dim=-1))
            vertex_h = self.vertex_updates[r](torch.cat([vertex_h, v_from_h, v_from_e], dim=-1))
            edge_h = self.edge_updates[r](torch.cat([edge_h, e_from_v], dim=-1))

        # Mean-pool each node type and project. Sum-pool would amplify
        # nodes-with-many-neighbors; mean keeps the readout invariant to
        # the number of nodes (a property we want for transfer between
        # game states with different occupancy).
        hex_pool = hex_h.mean(dim=1)  # (B, hidden)
        vertex_pool = vertex_h.mean(dim=1)
        edge_pool = edge_h.mean(dim=1)
        pooled = torch.cat([hex_pool, vertex_pool, edge_pool], dim=-1)
        return self.readout(pooled)
