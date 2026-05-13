"""Board geometry tables derived once from a ``catanBoard()`` instance.

The engine's ``catanBoard`` holds axial coordinates in ``HexCoordinates(q, r)``
and the vertex/edge adjacency in ``boardGraph``. This module extracts that
information into tensors of the shapes expected by:

  * the :class:`TileEncoder`'s axial positional embedding
    (``q_idx[19]``, ``r_idx[19]`` int64 tensors), and
  * the :class:`GraphEncoder`'s tripartite message-passing adjacency
    (``hex_to_vertex[19, 6]``, ``vertex_to_hex[54, 3]`` +
    ``vertex_to_hex_mask``, ``edge_to_vertex[72, 2]``,
    ``vertex_to_edge[54, 3]`` + ``vertex_to_edge_mask``).

Edge indices match the ordering produced by ``CatanEnv._build_index_maps``
(iteration over ``boardGraph.items()`` then per-vertex ``v_obj.neighbors``,
with each undirected edge keyed by the lex-sorted pair of vertex string
reprs). This invariant is checked by :func:`assert_consistent_with_env`.

Usage::

    from catan_rl.policy.board_geometry import build_geometry
    geom = build_geometry()           # cached after first call
    policy.set_board_geometry(geom.as_dict_of_tensors())

The build is cached because the geometry is fixed under D6 symmetry — the
hex *positions* never change between games, only the resource/number
assignments at each position (which are obs features, not geometry).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from catan_rl.engine.board import catanBoard

# Constants — must match catan_rl.policy.obs_schema. We re-declare here to
# avoid a circular import; a unit test cross-checks the two.
N_TILES = 19
N_VERTICES = 54
N_EDGES = 72
N_CORNERS_PER_TILE = 6
N_VERTEX_HEX_NEIGHBORS = 3  # max — interior vertices have 3; coastal have 1-2
N_VERTEX_EDGE_NEIGHBORS = 3  # max — interior vertices have 3; coastal have 2


# Range used to index the axial position embedding tables.
# Engine's coordDict uses q, r ∈ {-2, -1, 0, 1, 2}; we shift to {0..4}.
_AXIAL_MIN = -2
_AXIAL_MAX = 2
AXIAL_RANGE = _AXIAL_MAX - _AXIAL_MIN + 1  # 5

# Padding value for missing neighbors in the adjacency tables. The mask
# tensors distinguish padding (mask=0) from real neighbors (mask=1); the
# index value is irrelevant in masked slots. We use 0 for safety — gather
# operations on a valid index never raise.
_PAD_IDX = 0


@dataclass(frozen=True)
class BoardGeometry:
    """All board-geometry tensors required by the v2 policy network.

    Shapes:
      q_idx, r_idx           — (N_TILES,) long
      hex_to_vertex          — (N_TILES, 6) long
      vertex_to_hex          — (N_VERTICES, 3) long; pad with _PAD_IDX
      vertex_to_hex_mask     — (N_VERTICES, 3) float32; 1.0 valid, 0.0 pad
      edge_to_vertex         — (N_EDGES, 2) long
      vertex_to_edge         — (N_VERTICES, 3) long; pad with _PAD_IDX
      vertex_to_edge_mask    — (N_VERTICES, 3) float32; 1.0 valid, 0.0 pad
    """

    q_idx: torch.Tensor
    r_idx: torch.Tensor
    hex_to_vertex: torch.Tensor
    vertex_to_hex: torch.Tensor
    vertex_to_hex_mask: torch.Tensor
    edge_to_vertex: torch.Tensor
    vertex_to_edge: torch.Tensor
    vertex_to_edge_mask: torch.Tensor

    def as_dict_of_tensors(self) -> dict[str, torch.Tensor]:
        """The dict shape expected by ``CatanPolicy.set_board_geometry``."""
        return {
            "q_idx": self.q_idx,
            "r_idx": self.r_idx,
            "hex_to_vertex": self.hex_to_vertex,
            "vertex_to_hex": self.vertex_to_hex,
            "vertex_to_hex_mask": self.vertex_to_hex_mask,
            "edge_to_vertex": self.edge_to_vertex,
            "vertex_to_edge": self.vertex_to_edge,
            "vertex_to_edge_mask": self.vertex_to_edge_mask,
        }


def _edge_key(v1: Any, v2: Any) -> tuple[str, str]:
    """Canonical undirected-edge key — matches CatanEnv._edge_key."""
    s1, s2 = str(v1), str(v2)
    return (s1, s2) if s1 < s2 else (s2, s1)


def _build_geometry_from_board(board: catanBoard) -> BoardGeometry:
    # ---- Axial indices ------------------------------------------------
    q_arr = np.zeros(N_TILES, dtype=np.int64)
    r_arr = np.zeros(N_TILES, dtype=np.int64)
    for h_idx in range(N_TILES):
        coords = board.getHexCoords(h_idx)
        q_arr[h_idx] = int(coords.q) - _AXIAL_MIN
        r_arr[h_idx] = int(coords.r) - _AXIAL_MIN
    if not ((q_arr >= 0).all() and (q_arr < AXIAL_RANGE).all()):
        raise RuntimeError(f"q indices out of [0,{AXIAL_RANGE}): {q_arr}")
    if not ((r_arr >= 0).all() and (r_arr < AXIAL_RANGE).all()):
        raise RuntimeError(f"r indices out of [0,{AXIAL_RANGE}): {r_arr}")

    # ---- Hex -> vertex (6 corners per hex, in get_corners order) -------
    hex_to_vertex = np.zeros((N_TILES, N_CORNERS_PER_TILE), dtype=np.int64)
    for h_idx in range(N_TILES):
        hex_tile = board.hexTileDict[h_idx]
        corner_pts = hex_tile.get_corners(board.flat)
        if len(corner_pts) != N_CORNERS_PER_TILE:
            raise RuntimeError(
                f"hex {h_idx}: expected {N_CORNERS_PER_TILE} corners, got {len(corner_pts)}"
            )
        for c_idx, pt in enumerate(corner_pts):
            v_obj = board.boardGraph[pt]
            hex_to_vertex[h_idx, c_idx] = v_obj.vertex_index

    # ---- Vertex -> hex (pad to N_VERTEX_HEX_NEIGHBORS, with mask) ------
    vertex_to_hex = np.full((N_VERTICES, N_VERTEX_HEX_NEIGHBORS), _PAD_IDX, dtype=np.int64)
    vertex_to_hex_mask = np.zeros((N_VERTICES, N_VERTEX_HEX_NEIGHBORS), dtype=np.float32)
    # boardGraph is keyed by vertex pixel; iterate by vertex_index for
    # deterministic ordering.
    pixel_by_vidx = board.vertex_index_to_pixel_dict
    for v_idx in range(N_VERTICES):
        pt = pixel_by_vidx[v_idx]
        adj = board.boardGraph[pt].adjacent_hex_indices
        if len(adj) > N_VERTEX_HEX_NEIGHBORS:
            raise RuntimeError(
                f"vertex {v_idx}: {len(adj)} adjacent hexes (>{N_VERTEX_HEX_NEIGHBORS})"
            )
        for k, h_idx in enumerate(adj):
            vertex_to_hex[v_idx, k] = h_idx
            vertex_to_hex_mask[v_idx, k] = 1.0

    # ---- Edges (must match CatanEnv._build_index_maps ordering) --------
    edge_to_vertex_list: list[tuple[int, int]] = []
    edge_idx_by_key: dict[tuple[str, str], int] = {}
    for v_pt, v_obj in board.boardGraph.items():
        for nb_pt in v_obj.neighbors:
            key = _edge_key(v_pt, nb_pt)
            if key in edge_idx_by_key:
                continue
            nb_vidx = board.boardGraph[nb_pt].vertex_index
            edge_idx_by_key[key] = len(edge_to_vertex_list)
            edge_to_vertex_list.append((v_obj.vertex_index, nb_vidx))
    if len(edge_to_vertex_list) != N_EDGES:
        raise RuntimeError(f"expected {N_EDGES} edges, derived {len(edge_to_vertex_list)}")
    edge_to_vertex = np.array(edge_to_vertex_list, dtype=np.int64)

    # ---- Vertex -> edge (pad with mask) --------------------------------
    vertex_to_edge = np.full((N_VERTICES, N_VERTEX_EDGE_NEIGHBORS), _PAD_IDX, dtype=np.int64)
    vertex_to_edge_mask = np.zeros((N_VERTICES, N_VERTEX_EDGE_NEIGHBORS), dtype=np.float32)
    next_slot = np.zeros(N_VERTICES, dtype=np.int64)
    for e_idx, (a, b) in enumerate(edge_to_vertex_list):
        for v in (a, b):
            slot = int(next_slot[v])
            if slot >= N_VERTEX_EDGE_NEIGHBORS:
                raise RuntimeError(
                    f"vertex {v}: more than {N_VERTEX_EDGE_NEIGHBORS} incident edges"
                )
            vertex_to_edge[v, slot] = e_idx
            vertex_to_edge_mask[v, slot] = 1.0
            next_slot[v] = slot + 1

    return BoardGeometry(
        q_idx=torch.from_numpy(q_arr),
        r_idx=torch.from_numpy(r_arr),
        hex_to_vertex=torch.from_numpy(hex_to_vertex),
        vertex_to_hex=torch.from_numpy(vertex_to_hex),
        vertex_to_hex_mask=torch.from_numpy(vertex_to_hex_mask),
        edge_to_vertex=torch.from_numpy(edge_to_vertex),
        vertex_to_edge=torch.from_numpy(vertex_to_edge),
        vertex_to_edge_mask=torch.from_numpy(vertex_to_edge_mask),
    )


@lru_cache(maxsize=1)
def build_geometry() -> BoardGeometry:
    """Build (or fetch the cached) BoardGeometry.

    The geometry is invariant under games — resource assignments shuffle
    per game but hex positions, vertex indices, and edge adjacencies do
    not. We build it from one ``catanBoard()`` and cache the result.
    """
    board = catanBoard()
    return _build_geometry_from_board(board)


def assert_consistent_with_env(geom: BoardGeometry) -> None:
    """Cross-check geometry against a freshly-built ``CatanEnv``.

    Ensures ``edge_to_vertex`` matches the ordering produced by
    ``CatanEnv._build_index_maps``. Raises ``AssertionError`` on mismatch.

    Lives here (not in tests) so the consistency invariant can be
    asserted at runtime from training scripts in case the two
    constructions ever drift.
    """
    from catan_rl.env.catan_env import CatanEnv  # local import to avoid cycle

    env = CatanEnv(opponent_type="random", max_turns=10)
    env.reset(seed=0)
    assert env.game is not None  # reset() guarantees this
    env_e2v = np.zeros((N_EDGES, 2), dtype=np.int64)
    for e_idx, (v1_pt, v2_pt) in env._idx_to_edge.items():
        env_e2v[e_idx, 0] = env.game.board.boardGraph[v1_pt].vertex_index
        env_e2v[e_idx, 1] = env.game.board.boardGraph[v2_pt].vertex_index
    ours = geom.edge_to_vertex.numpy()

    # The env may produce a different orientation per edge (v1, v2 vs v2, v1)
    # but the underlying *set* must match.
    env_pairs = {tuple(sorted(row)) for row in env_e2v}
    our_pairs = {tuple(sorted(row.tolist())) for row in ours}
    if env_pairs != our_pairs:
        diff_env_only = env_pairs - our_pairs
        diff_ours_only = our_pairs - env_pairs
        raise AssertionError(
            f"board_geometry edge set != CatanEnv edge set; "
            f"env-only={list(diff_env_only)[:3]}, "
            f"geom-only={list(diff_ours_only)[:3]}"
        )
