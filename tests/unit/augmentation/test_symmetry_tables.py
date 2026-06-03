"""Unit tests for D6 symmetry tables.

The tables are computed from board topology; bugs here are silent (model
trains on inconsistent labels with no error message) so the tests are
group-theoretic and exhaustive rather than spot-check.
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

# ---------------------------------------------------------------------------
# Range / identity properties
# ---------------------------------------------------------------------------


def test_group_size_is_twelve() -> None:
    assert D6_GROUP_SIZE == 12


def test_identity_is_zero() -> None:
    assert D6_IDENTITY == 0


@pytest.mark.parametrize("perm_fn", [tile_perm, corner_perm, edge_perm])
def test_identity_is_identity_for_board_perms(perm_fn) -> None:
    """g = 0 (identity) must leave every slot in place for all three board perms."""
    p = perm_fn(D6_IDENTITY)
    assert np.array_equal(p, np.arange(p.shape[0]))


@pytest.mark.parametrize("perm_fn", [within_tile_corner_perm, within_tile_edge_perm])
def test_identity_is_identity_for_within_tile_perms(perm_fn) -> None:
    p = perm_fn(D6_IDENTITY)
    assert np.array_equal(p, np.arange(6))


def test_shapes() -> None:
    """Tile/corner/edge perms have the right sizes for every group element."""
    for g in range(D6_GROUP_SIZE):
        assert tile_perm(g).shape == (19,)
        assert corner_perm(g).shape == (54,)
        assert edge_perm(g).shape == (72,)
        assert within_tile_corner_perm(g).shape == (6,)
        assert within_tile_edge_perm(g).shape == (6,)


def test_out_of_range_raises() -> None:
    for fn in (tile_perm, corner_perm, edge_perm, within_tile_corner_perm, within_tile_edge_perm):
        with pytest.raises(ValueError):
            fn(-1)
        with pytest.raises(ValueError):
            fn(D6_GROUP_SIZE)
    with pytest.raises(ValueError):
        rotation_steps(-1)
    with pytest.raises(ValueError):
        is_reflection(D6_GROUP_SIZE)


# ---------------------------------------------------------------------------
# Permutation discipline — every output is a valid permutation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("perm_fn,size", [(tile_perm, 19), (corner_perm, 54), (edge_perm, 72)])
def test_perms_are_valid_permutations(perm_fn, size) -> None:
    for g in range(D6_GROUP_SIZE):
        p = perm_fn(g)
        assert sorted(p.tolist()) == list(range(size)), (
            f"{perm_fn.__name__}({g}) is not a permutation of [0..{size}): {p}"
        )


@pytest.mark.parametrize("perm_fn", [within_tile_corner_perm, within_tile_edge_perm])
def test_within_tile_perms_are_valid(perm_fn) -> None:
    for g in range(D6_GROUP_SIZE):
        p = perm_fn(g)
        assert sorted(p.tolist()) == list(range(6))


# ---------------------------------------------------------------------------
# Group structure — inverses + composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("perm_fn", [tile_perm, corner_perm, edge_perm])
def test_inverse_undoes_perm(perm_fn) -> None:
    """For every g, applying g then D6_INVERSE(g) must yield identity."""
    for g in range(D6_GROUP_SIZE):
        p = perm_fn(g)
        p_inv = perm_fn(D6_INVERSE(g))
        composed = p_inv[p]
        assert np.array_equal(composed, np.arange(p.shape[0])), (
            f"D6_INVERSE({g}) does not undo g for {perm_fn.__name__}"
        )


def test_inverse_table_is_involutive() -> None:
    """Applying D6_INVERSE twice returns the original element."""
    for g in range(D6_GROUP_SIZE):
        assert D6_INVERSE(D6_INVERSE(g)) == g


def test_rotation_inverse_is_negation_mod_six() -> None:
    """Pure rotations: inverse(rot_k) == rot_(-k mod 6)."""
    for k in range(6):
        expected = (6 - k) % 6
        assert D6_INVERSE(k) == expected, f"D6_INVERSE({k}) = {D6_INVERSE(k)}, expected {expected}"


def test_pure_reflection_is_self_inverse() -> None:
    """g = 6 is reflection-with-no-rotation; should be its own inverse."""
    assert D6_INVERSE(6) == 6


@pytest.mark.parametrize("perm_fn", [tile_perm, corner_perm, edge_perm])
def test_six_rotations_compose_to_identity(perm_fn) -> None:
    """Applying rot_1 six times must return identity (the order-6 rotation)."""
    p = perm_fn(1)  # 60° rotation
    composed = p
    for _ in range(5):
        composed = p[composed]
    assert np.array_equal(composed, np.arange(p.shape[0]))


@pytest.mark.parametrize("perm_fn", [tile_perm, corner_perm, edge_perm])
def test_reflection_applied_twice_is_identity(perm_fn) -> None:
    """The pure reflection element (g=6) has order 2."""
    p = perm_fn(6)
    composed = p[p]
    assert np.array_equal(composed, np.arange(p.shape[0]))


# ---------------------------------------------------------------------------
# Cross-perm consistency — corner perm and edge perm agree on every edge
# ---------------------------------------------------------------------------


def test_cayley_convention() -> None:
    """Pin the Cayley composition rule the tables actually follow.

    Empirically verified 2026-06-03 (post reflection-axis fix): the
    tables satisfy the textbook D6 algebra ``s·r^i ∘ r^j = s·r^(i-j) mod 6``.

    This pin exists so that future code which composes D6 elements
    algebraically (inverse computation, optimization) does not silently
    diverge from the table-derived composition.
    """
    # Permutation composition: ``(p_left[p_right])[x] = p_left[p_right[x]]``
    # i.e. apply ``right`` first, then ``left``. So ``p_left ∘ p_right``
    # in the function-composition sense corresponds to fancy-indexing
    # ``p_left[p_right]``.
    for i in range(6):
        for j in range(6):
            s_ri = 6 + i  # reflection composed with r^i (left operand)
            r_j = j  # rotation r^j (right operand, applied first)
            p_left = corner_perm(s_ri)
            p_right = corner_perm(r_j)
            composed = p_left[p_right]
            expected_idx = 6 + ((i - j) % 6)
            assert (composed == corner_perm(expected_idx)).all(), (
                f"Cayley s·r^{i} ∘ r^{j}: expected s·r^{(i - j) % 6} "
                f"(textbook D6), got perm differing from element {expected_idx}"
            )


def test_engine_topology_d6_symmetric_under_corner_and_tile_perm() -> None:
    """Pin the topology relation ``A_{corner_perm(g)[v]} = tile_perm(g)[A_v]``
    for all 12 D6 elements and all 54 vertices.

    Pre-2026-06-03 this relation failed for every reflection (g=6..11)
    because ``tile_perm`` used axial-coordinate reflection while
    ``corner_perm`` used pixel-space ``x → -x`` — two different geometric
    axes (150° vs 90°). The PPO main-loop aug survived only because both
    self-play sides saw the same wrong transform; BC labels would have
    been silently corrupted. See ``memory/project_d6_reflection_bug.md``.

    Any regression to the reflection axis must fail this test. If a
    future ``port_perm`` is added, it must also satisfy a port-specific
    analog of this same topology relation.
    """
    from catan_rl.engine.board import catanBoard

    board = catanBoard()
    adjacency = {}
    for v_idx in range(54):
        px = board.vertex_index_to_pixel_dict[v_idx]
        adjacency[v_idx] = frozenset(board.boardGraph[px].adjacent_hex_indices)

    for g in range(D6_GROUP_SIZE):
        p = tile_perm(g)
        c_p = corner_perm(g)
        for v in range(54):
            lhs = adjacency[int(c_p[v])]
            rhs = frozenset(int(p[h]) for h in adjacency[v])
            assert lhs == rhs, (
                f"g={g} v={v}: A_{{corner_perm[v]={int(c_p[v])}}}={sorted(lhs)} "
                f"!= tile_perm(A_v)={sorted(rhs)}"
            )


def test_edge_perm_consistent_with_corner_perm() -> None:
    """For every edge (a, b), under D6 element g, edge_perm(g) must map to
    the edge whose endpoints are (corner_perm(g)[a], corner_perm(g)[b])."""
    from catan_rl.augmentation.symmetry_tables import _board_data

    data = _board_data()
    edges = data["edges"]
    edge_to_idx = data["edge_to_idx"]

    for g in range(D6_GROUP_SIZE):
        c_p = corner_perm(g)
        e_p = edge_perm(g)
        for old_idx, (a, b) in enumerate(edges):
            na, nb = int(c_p[a]), int(c_p[b])
            expected_new_idx = edge_to_idx[(min(na, nb), max(na, nb))]
            assert e_p[old_idx] == expected_new_idx, (
                f"D6 element {g}: edge {old_idx} {(a, b)} corner-perm maps to "
                f"endpoints {(na, nb)}, expected edge idx {expected_new_idx}, "
                f"got {e_p[old_idx]}"
            )


# ---------------------------------------------------------------------------
# Center-tile invariance
# ---------------------------------------------------------------------------


def test_center_tile_is_a_fixed_point() -> None:
    """The centre hex (index 0) must map to itself under every D6 element."""
    for g in range(D6_GROUP_SIZE):
        assert tile_perm(g)[0] == 0, f"D6 element {g}: centre tile moved to {tile_perm(g)[0]}"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_tables_are_deterministic_across_calls() -> None:
    """Cached tables don't drift between calls."""
    for g in range(D6_GROUP_SIZE):
        assert np.array_equal(tile_perm(g), tile_perm(g))
        assert np.array_equal(corner_perm(g), corner_perm(g))
        assert np.array_equal(edge_perm(g), edge_perm(g))


def test_returned_arrays_are_copies() -> None:
    """Mutating a returned array must not corrupt the cached table."""
    arr = tile_perm(3)
    original = arr.copy()
    arr[0] = -999
    # Cached array unchanged
    assert np.array_equal(tile_perm(3), original)
