"""Validation tests for ``catan_engine.BoardStatic``.

Per the Q2 decision (resolved in
``docs/plans/rust_engine_migration.md``), Rust edge ordering does
NOT need to match Python's `dict.items()` insertion order — a one-
shot ``scripts/migrate_checkpoint.py`` permutation patch handles the
checkpoint compatibility.

These tests therefore verify INVARIANTS of the Rust board (correct
multisets, correct topology counts, determinism under seed), not
byte-equality against the Python board. The Python ↔ Rust topology
parity (e.g., "every hex's 6 corners resolve to a vertex") is
covered by the cargo tests in ``crates/catan_engine/src/board.rs``.
"""

from __future__ import annotations

from collections import Counter

import pytest

catan_engine = pytest.importorskip("catan_engine")


@pytest.fixture
def board() -> dict:
    return catan_engine.BoardStatic(42).board_static()


def test_has_19_hexes_54_vertices_72_edges_9_ports(board: dict) -> None:
    assert len(board["hexes"]) == 19
    assert len(board["vertices"]) == 54
    assert len(board["edges"]) == 72
    assert len(board["ports"]) == 9


def test_hex_resource_multiset(board: dict) -> None:
    """Standard 1v1 Colonist distribution: 1 desert, 3 ore, 3 brick,
    4 wheat, 4 wood, 4 sheep."""
    counts: Counter[str] = Counter(h["resource"] for h in board["hexes"])
    assert counts == {
        "DESERT": 1,
        "ORE": 3,
        "BRICK": 3,
        "WHEAT": 4,
        "WOOD": 4,
        "SHEEP": 4,
    }


def test_hex_number_token_multiset(board: dict) -> None:
    """Non-desert hexes carry chip tokens from the official ABC
    spiral sequence (see board.rs SPIRAL_CHIP_SEQUENCE)."""
    desert_count = sum(1 for h in board["hexes"] if h["resource"] == "DESERT")
    assert desert_count == 1
    tokens = sorted(h["number_token"] for h in board["hexes"] if h["number_token"] is not None)
    expected = sorted([5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11])
    assert tokens == expected


def test_desert_has_initial_robber(board: dict) -> None:
    desert = [h for h in board["hexes"] if h["resource"] == "DESERT"]
    assert len(desert) == 1
    assert desert[0]["has_robber_initial"] is True
    # All others must not.
    non_desert_with_robber = [
        h for h in board["hexes"] if h["resource"] != "DESERT" and h["has_robber_initial"]
    ]
    assert non_desert_with_robber == []


def test_hex_axial_coords_match_canonical_table(board: dict) -> None:
    """The hex_idx → (q, r) mapping must match Python's
    catanBoard.getHexCoords (board.py:202-225). Pinned here so a
    future change to the table is caught by the unit test."""
    expected = {
        0: (0, 0),
        1: (0, -1),
        2: (1, -1),
        3: (1, 0),
        4: (0, 1),
        5: (-1, 1),
        6: (-1, 0),
        7: (0, -2),
        8: (1, -2),
        9: (2, -2),
        10: (2, -1),
        11: (2, 0),
        12: (1, 1),
        13: (0, 2),
        14: (-1, 2),
        15: (-2, 2),
        16: (-2, 1),
        17: (-2, 0),
        18: (-1, -1),
    }
    for h in board["hexes"]:
        assert (h["q"], h["r"]) == expected[h["hex_idx"]], f"hex {h['hex_idx']} coord mismatch"


def test_vertex_adjacency_lists_are_non_empty(board: dict) -> None:
    for v in board["vertices"]:
        n = len(v["adjacent_hex_indices"])
        assert 1 <= n <= 3
        # Hex indices must be in [0, 19).
        for h in v["adjacent_hex_indices"]:
            assert 0 <= h < 19


def test_edge_vertex_indices_are_in_range(board: dict) -> None:
    for e in board["edges"]:
        assert 0 <= e["v1_idx"] < 54
        assert 0 <= e["v2_idx"] < 54
        assert e["v1_idx"] != e["v2_idx"]


def test_edge_indices_are_dense_0_to_71(board: dict) -> None:
    idxs = sorted(e["edge_idx"] for e in board["edges"])
    assert idxs == list(range(72))


def test_vertex_indices_are_dense_0_to_53(board: dict) -> None:
    idxs = sorted(v["vertex_idx"] for v in board["vertices"])
    assert idxs == list(range(54))


def test_port_type_distribution(board: dict) -> None:
    """5 specific 2:1 ports (one per resource) + 4 generic 3:1 ports."""
    ratios = Counter(p["ratio"] for p in board["ports"])
    assert ratios == {"2:1": 5, "3:1": 4}
    specific_resources = sorted(p["resource"] for p in board["ports"] if p["ratio"] == "2:1")
    assert specific_resources == sorted(["WOOD", "BRICK", "WHEAT", "ORE", "SHEEP"])
    # 3:1 ports have resource=None.
    for p in board["ports"]:
        if p["ratio"] == "3:1":
            assert p["resource"] is None


def test_port_vertex_pairs_in_range(board: dict) -> None:
    for p in board["ports"]:
        v1, v2 = p["vertex_idx_pair"]
        assert 0 <= v1 < 54
        assert 0 <= v2 < 54
        assert v1 != v2


def test_determinism_under_same_seed() -> None:
    a = catan_engine.BoardStatic(123).board_static()
    b = catan_engine.BoardStatic(123).board_static()
    # Hexes byte-equal.
    for ha, hb in zip(a["hexes"], b["hexes"], strict=True):
        assert ha == hb
    # Ports byte-equal.
    for pa, pb in zip(a["ports"], b["ports"], strict=True):
        assert pa == pb


def test_different_seeds_diverge() -> None:
    a = catan_engine.BoardStatic(1).board_static()
    b = catan_engine.BoardStatic(2).board_static()
    # Resource sequences should differ.
    a_resources = [h["resource"] for h in a["hexes"]]
    b_resources = [h["resource"] for h in b["hexes"]]
    assert a_resources != b_resources or a["ports"] != b["ports"]
