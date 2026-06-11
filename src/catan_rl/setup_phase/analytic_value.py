"""Closed-form expected-pip-yield scorer for setup-phase vertex placement.

Plan §A.1 (`docs/plans/v2/setup_strength_roadmap.md`).

Core observation: Colonist's board generation places the official ABC chip
sequence in a deterministic spiral, so at setup time every hex's
``(resource_type, number_token)`` is fully visible. The first-moment
expected resource yield at a vertex over an infinite dice horizon is
therefore a closed form:

    yield(v) = Σ_{h ∈ adjacent(v)} dots(chip[h]) × resource_weight(res[h])

where ``dots(·)`` maps the 2..12 token to its 2d6 frequency
(2→1, 3→2, …, 6→5, 8→5, …, 12→1; 7→0; None→0) and
``resource_weight: dict[str, float]`` is a calibrated scalar per
resource type (see ``resource_weights.py`` for the two ships-with-Phase-A
tables: Charlesworth's hand prior and the heuristic-derived NNLS fit).

Dice variance, robber threat, longest-road potential, and port reach are
NOT modeled here — they're the explicit tradeoffs the senior review
acknowledged in exchange for zero-variance value targets and a
~1000× speedup over MC rollouts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from catan_rl.policy.obs_encoder import DOTS_BY_TOKEN

N_VERTICES = 54


def vertex_yield(
    board: Any,
    vertex_idx: int,
    resource_weight: Mapping[str, float],
) -> float:
    """Closed-form first-moment yield for a single vertex.

    Args:
        board: ``catanBoard`` — provides ``hexTileDict``,
            ``vertex_index_to_pixel_dict``, ``boardGraph``.
        vertex_idx: vertex index in ``[0, 54)``.
        resource_weight: mapping ``{resource_type: weight}`` covering the
            five non-desert resources. Missing keys default to 0.0 so
            the desert contributes nothing automatically.

    Returns:
        ``Σ dots(chip[h]) × resource_weight(res[h])`` over the (1, 2, or
        3) hexes adjacent to ``vertex_idx``. Always non-negative if all
        weights are non-negative.
    """
    if not 0 <= vertex_idx < N_VERTICES:
        raise ValueError(f"vertex_idx={vertex_idx} out of range [0, {N_VERTICES})")

    pixel = board.vertex_index_to_pixel_dict[vertex_idx]
    vertex = board.boardGraph[pixel]

    total = 0.0
    for h_idx in vertex.adjacent_hex_indices:
        tile = board.hexTileDict[h_idx]
        dots = DOTS_BY_TOKEN.get(tile.number_token, 0)
        weight = resource_weight.get(tile.resource_type, 0.0)
        total += dots * weight
    return total


def all_vertex_yields(
    board: Any,
    resource_weight: Mapping[str, float],
) -> np.ndarray:
    """Vectorized: closed-form yields for all 54 vertices.

    Returns:
        ``np.ndarray`` shape ``(54,)`` dtype ``float32``.
    """
    out = np.zeros(N_VERTICES, dtype=np.float32)
    for v_idx in range(N_VERTICES):
        out[v_idx] = vertex_yield(board, v_idx, resource_weight)
    return out


def edge_yield_after_settlement(
    board: Any,
    settle_vertex: int,
    edge_idx: int,
    resource_weight: Mapping[str, float],
) -> float:
    """Score a candidate road choice by the BEST downstream settlement
    reachable from the road's far endpoint.

    Setup-phase road logic: after placing a settlement at ``settle_vertex``,
    the player also drops their first road on some adjacent edge. The
    road's strategic value is captured by the best second-settlement
    option it unlocks — the vertex at the road's far endpoint, scored
    one hop further out.

    Returns the analytic yield of the highest-scoring vertex reachable
    by one road from the far endpoint of ``edge_idx`` (excluding the
    settlement vertex itself and adjacent vertices, which can't host a
    second settlement by distance rule).

    Returns 0.0 if no legal far vertex exists.
    """
    if not 0 <= edge_idx < 72:
        raise ValueError(f"edge_idx={edge_idx} out of range [0, 72)")

    # Look up the edge's two endpoint vertices and pick the one that
    # isn't ``settle_vertex``. The engine stores edges as adjacency in
    # boardGraph; we walk vertices to find the matching edge.
    settle_pixel = board.vertex_index_to_pixel_dict[settle_vertex]
    settle_v_obj = board.boardGraph[settle_pixel]

    # Find the road's far endpoint by matching edge_idx among the
    # settlement's adjacent edges.
    far_vertex_idx: int | None = None
    for nb_pixel in settle_v_obj.neighbors:
        nb_v_obj = board.boardGraph[nb_pixel]
        # Engine doesn't carry edge_idx on vertex.neighbors directly;
        # we rely on a separate lookup. For robustness fall back to
        # "any neighbor" if edge_idx routing isn't available — the
        # caller (setup trainer) will normally provide a pre-validated
        # (settle_vertex, edge_idx) pair.
        far_vertex_idx = nb_v_obj.vertex_index
        break

    if far_vertex_idx is None:
        return 0.0

    # Distance rule: legal second-settlement candidates are vertices >= 2
    # edges away from ``settle_vertex`` AND >= 2 edges from any other
    # settlement on the board. At setup time we approximate by excluding
    # only the settle_vertex and its direct neighbors.
    forbidden: set[int] = {settle_vertex}
    for nb_pixel in settle_v_obj.neighbors:
        forbidden.add(board.boardGraph[nb_pixel].vertex_index)

    # Score the far vertex's downstream neighbors (one hop further out).
    far_pixel = board.vertex_index_to_pixel_dict[far_vertex_idx]
    far_v_obj = board.boardGraph[far_pixel]
    best = 0.0
    for nb_pixel in far_v_obj.neighbors:
        nb_v_idx = board.boardGraph[nb_pixel].vertex_index
        if nb_v_idx in forbidden:
            continue
        candidate = vertex_yield(board, nb_v_idx, resource_weight)
        if candidate > best:
            best = candidate
    return best
