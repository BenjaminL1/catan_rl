"""Unit tests for board_geometry.

Pins the contract between the engine's coordinate / adjacency exposure and
the tensors the policy network's encoders consume.
"""

from __future__ import annotations

import torch

from catan_rl.policy.board_geometry import (
    AXIAL_RANGE,
    BoardGeometry,
    assert_consistent_with_env,
    build_geometry,
)
from catan_rl.policy.obs_schema import (
    N_EDGES,
    N_TILES,
    N_VERTICES,
)


def test_build_geometry_returns_correct_shapes() -> None:
    geom = build_geometry()
    assert isinstance(geom, BoardGeometry)
    assert geom.q_idx.shape == (N_TILES,)
    assert geom.r_idx.shape == (N_TILES,)
    assert geom.hex_to_vertex.shape == (N_TILES, 6)
    assert geom.vertex_to_hex.shape == (N_VERTICES, 3)
    assert geom.vertex_to_hex_mask.shape == (N_VERTICES, 3)
    assert geom.edge_to_vertex.shape == (N_EDGES, 2)
    assert geom.vertex_to_edge.shape == (N_VERTICES, 3)
    assert geom.vertex_to_edge_mask.shape == (N_VERTICES, 3)


def test_axial_indices_within_range() -> None:
    geom = build_geometry()
    assert (geom.q_idx >= 0).all() and (geom.q_idx < AXIAL_RANGE).all()
    assert (geom.r_idx >= 0).all() and (geom.r_idx < AXIAL_RANGE).all()


def test_axial_indices_match_engine_for_known_hexes() -> None:
    """Spot-check known hex axial coords against engine's getHexCoords.

    Engine uses q, r ∈ {-2..2}. We shift to [0..4]. The known hexes:
      hex 0 (center)      -> (q=0, r=0)  -> idx (2, 2)
      hex 1 (top)         -> (q=0, r=-1) -> idx (2, 1)
      hex 9 (corner)      -> (q=2, r=-2) -> idx (4, 0)
      hex 15 (other corner) -> (q=-2, r=2) -> idx (0, 4)
    """
    geom = build_geometry()
    assert int(geom.q_idx[0]) == 2 and int(geom.r_idx[0]) == 2, "hex 0 = center"
    assert int(geom.q_idx[1]) == 2 and int(geom.r_idx[1]) == 1, "hex 1 = top-inner"
    assert int(geom.q_idx[9]) == 4 and int(geom.r_idx[9]) == 0, "hex 9 corner"
    assert int(geom.q_idx[15]) == 0 and int(geom.r_idx[15]) == 4, "hex 15 corner"


def test_hex_to_vertex_indices_in_bounds() -> None:
    geom = build_geometry()
    assert (geom.hex_to_vertex >= 0).all()
    assert (geom.hex_to_vertex < N_VERTICES).all()


def test_hex_to_vertex_no_duplicates_per_hex() -> None:
    """Each hex has 6 distinct corners."""
    geom = build_geometry()
    for h in range(N_TILES):
        corners = geom.hex_to_vertex[h].tolist()
        assert len(set(corners)) == 6, f"hex {h} has duplicate corners: {corners}"


def test_hex_to_vertex_and_vertex_to_hex_are_consistent() -> None:
    """Every (h, v) edge in hex_to_vertex must also appear in vertex_to_hex."""
    geom = build_geometry()
    hv = {(h, int(v)) for h in range(N_TILES) for v in geom.hex_to_vertex[h]}
    vh = set()
    for v in range(N_VERTICES):
        mask = geom.vertex_to_hex_mask[v]
        for k in range(3):
            if mask[k] > 0.5:
                vh.add((int(geom.vertex_to_hex[v, k]), v))
    assert hv == vh, f"hex<->vertex inconsistency: |hv\\vh|={len(hv - vh)}, |vh\\hv|={len(vh - hv)}"


def test_edge_to_vertex_degree_two() -> None:
    geom = build_geometry()
    for e in range(N_EDGES):
        v1, v2 = int(geom.edge_to_vertex[e, 0]), int(geom.edge_to_vertex[e, 1])
        assert v1 != v2, f"edge {e} is self-loop ({v1},{v2})"
        assert 0 <= v1 < N_VERTICES
        assert 0 <= v2 < N_VERTICES


def test_edges_are_undirected_and_unique() -> None:
    geom = build_geometry()
    pairs = {tuple(sorted(row.tolist())) for row in geom.edge_to_vertex}
    assert len(pairs) == N_EDGES, (
        f"edges contain duplicates: {N_EDGES - len(pairs)} duplicate pairs"
    )


def test_vertex_to_edge_consistent_with_edge_to_vertex() -> None:
    geom = build_geometry()
    # Reconstruct vertex->edge from edge->vertex and compare.
    from collections import defaultdict

    expected: dict[int, set[int]] = defaultdict(set)
    for e in range(N_EDGES):
        a, b = int(geom.edge_to_vertex[e, 0]), int(geom.edge_to_vertex[e, 1])
        expected[a].add(e)
        expected[b].add(e)

    for v in range(N_VERTICES):
        recorded: set[int] = set()
        for k in range(3):
            if geom.vertex_to_edge_mask[v, k] > 0.5:
                recorded.add(int(geom.vertex_to_edge[v, k]))
        assert recorded == expected[v], (
            f"vertex {v}: vertex_to_edge has {sorted(recorded)}, "
            f"edge_to_vertex implies {sorted(expected[v])}"
        )


def test_vertex_degrees_in_2_or_3() -> None:
    """Catan board: every vertex has 2 (coast) or 3 (interior) incident edges."""
    geom = build_geometry()
    for v in range(N_VERTICES):
        deg = int(geom.vertex_to_edge_mask[v].sum().item())
        assert deg in (2, 3), f"vertex {v} has unexpected edge degree {deg}"


def test_vertex_hex_degrees_in_1_2_3() -> None:
    """Catan board: every vertex touches 1 (corner), 2 (coast) or 3 (interior) hexes."""
    geom = build_geometry()
    for v in range(N_VERTICES):
        deg = int(geom.vertex_to_hex_mask[v].sum().item())
        assert deg in (1, 2, 3), f"vertex {v} has unexpected hex degree {deg}"


def test_geometry_is_cached() -> None:
    """build_geometry caches; consecutive calls return the same object."""
    g1 = build_geometry()
    g2 = build_geometry()
    assert g1 is g2


def test_consistent_with_env() -> None:
    """The runtime consistency check passes for the canonical board."""
    geom = build_geometry()
    assert_consistent_with_env(geom)


def test_geometry_works_with_policy_set_board_geometry() -> None:
    """Smoke-test: feeding the geometry into the policy doesn't blow up."""
    from catan_rl.policy import CatanPolicy

    policy = CatanPolicy()
    geom = build_geometry()
    policy.set_board_geometry(geom.as_dict_of_tensors())

    # After plumbing, q_idx and r_idx on the policy should match the geometry.
    assert torch.equal(policy.tile_encoder.pos_emb.q_idx, geom.q_idx)
    assert torch.equal(policy.tile_encoder.pos_emb.r_idx, geom.r_idx)
    assert torch.equal(policy.graph_encoder.edge_to_vertex, geom.edge_to_vertex)
