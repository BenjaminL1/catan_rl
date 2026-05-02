"""Unit tests for the precomputed D6 symmetry tables (Phase 1.5).

These guard the *correctness* of the permutation tables that the
augmentation pipeline depends on. A wrong table would silently corrupt
training data — these tests catch that before any rollout collects bad
samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.augmentation.symmetry_tables import (
    D6_GROUP_SIZE,
    D6_IDENTITY,
    D6_INVERSE,
    corner_perm,
    edge_perm,
    is_reflection,
    rotation_steps,
    tile_perm,
    within_tile_corner_perm,
    within_tile_edge_perm,
)

# ── Identity element ────────────────────────────────────────────────────────


def test_identity_is_zero() -> None:
    """The identity element of D6 is encoded as ``0``."""
    assert D6_IDENTITY == 0
    assert rotation_steps(0) == 0
    assert not is_reflection(0)


def test_identity_perms_are_arange() -> None:
    """All permutations under the identity are ``[0, 1, 2, ...]``."""
    np.testing.assert_array_equal(tile_perm(D6_IDENTITY), np.arange(19))
    np.testing.assert_array_equal(corner_perm(D6_IDENTITY), np.arange(54))
    np.testing.assert_array_equal(edge_perm(D6_IDENTITY), np.arange(72))
    np.testing.assert_array_equal(within_tile_corner_perm(D6_IDENTITY), np.arange(6))
    np.testing.assert_array_equal(within_tile_edge_perm(D6_IDENTITY), np.arange(6))


# ── Geometric structure ─────────────────────────────────────────────────────


def test_center_tile_is_fixed_under_every_d6_element() -> None:
    """Hex 0 sits at axial (0, 0) — invariant under every D6 element."""
    for g in range(D6_GROUP_SIZE):
        assert tile_perm(g)[0] == 0, f"g={g} moves the center tile"


def test_inner_ring_cycles_under_60_rotation() -> None:
    """One step of 60° CCW cycles tiles 1→2→3→4→5→6→1 (the inner ring)."""
    # ``tile_perm(g)[i]`` = where old tile ``i`` lands. Under 60° CCW it
    # advances by one position around the ring.
    perm = tile_perm(1)
    assert list(perm[1:7]) == [2, 3, 4, 5, 6, 1]


def test_outer_ring_cycles_correctly() -> None:
    """The 12 outer-ring tiles split into two orbits of size 6 under D6 — no
    tiles outside the inner/outer ring exist on a 19-hex Catan board."""
    perm = tile_perm(1)  # 60° CCW
    # All outer-ring tiles 7..18 stay in 7..18 under rotation.
    assert all(7 <= int(perm[i]) <= 18 for i in range(7, 19))
    # And cycle with period dividing 6 (full ring).
    p6 = perm[perm[perm[perm[perm[perm[np.arange(19)]]]]]]
    np.testing.assert_array_equal(p6, np.arange(19))


# ── Bijection check ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,fn,size",
    [
        ("tile", tile_perm, 19),
        ("corner", corner_perm, 54),
        ("edge", edge_perm, 72),
        ("within_corner", within_tile_corner_perm, 6),
        ("within_edge", within_tile_edge_perm, 6),
    ],
)
def test_perms_are_bijections(name: str, fn, size: int) -> None:
    """Every D6 element produces a valid permutation of the index set."""
    for g in range(D6_GROUP_SIZE):
        p = fn(g)
        assert sorted(p.tolist()) == list(range(size)), f"{name} g={g} not a bijection"


# ── Group structure ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,fn",
    [
        ("tile", tile_perm),
        ("corner", corner_perm),
        ("edge", edge_perm),
    ],
)
def test_inverse_round_trip(name: str, fn) -> None:
    """``perm(g)[perm(g_inv)]`` is the identity for every element."""
    for g in range(D6_GROUP_SIZE):
        g_inv = D6_INVERSE(g)
        composed = fn(g)[fn(g_inv)]
        np.testing.assert_array_equal(
            composed,
            np.arange(len(fn(g))),
            err_msg=f"{name}: g={g}, inv={g_inv}",
        )


def test_d6_has_twelve_distinct_elements() -> None:
    """Corner permutations are unique across the 12 group elements."""
    seen = set()
    for g in range(D6_GROUP_SIZE):
        seen.add(tuple(corner_perm(g).tolist()))
    assert len(seen) == D6_GROUP_SIZE


# ── Edge permutation consistency ────────────────────────────────────────────


def test_edge_permutation_matches_corner_permutation() -> None:
    """Every edge connects two corners. Under any g, the edge slot the
    new pair falls in must match what ``corner_perm`` says."""
    # Reconstruct edges from the same temporary board the tables used.
    from catan_rl.engine.board import catanBoard

    board = catanBoard()
    edge_set = set()
    for v_obj in board.boardGraph.values():
        a = v_obj.vertex_index
        for nb_px in v_obj.neighbors:
            b = board.boardGraph[nb_px].vertex_index
            edge_set.add((min(a, b), max(a, b)))
    edges = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    for g in range(D6_GROUP_SIZE):
        cp = corner_perm(g)
        ep = edge_perm(g)
        for old_idx, (a, b) in enumerate(edges):
            na, nb = int(cp[a]), int(cp[b])
            expected = edge_to_idx[(min(na, nb), max(na, nb))]
            assert ep[old_idx] == expected, (
                f"g={g}: edge {old_idx} → {ep[old_idx]} but expected {expected}"
            )
