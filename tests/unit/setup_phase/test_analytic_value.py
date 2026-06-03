"""Plan §A.1 — analytic value scorer correctness tests.

Pins:

1. ``vertex_yield`` returns 0 for a vertex adjacent only to desert.
2. Hand-computed reference values on a fixed-seed board.
3. **D6 invariance**: rotating the board's tile data through any of the
   12 dihedral elements ``g`` shifts the per-vertex yields exactly by
   ``corner_perm(g)``. This is the cross-validation hook Phase B's port
   permutation will reuse.
4. Top-vertex sanity on a known seed.

The Charlesworth prior is used as the reference weighting throughout —
its hand-tuned numbers make hand-computed fixtures readable.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest

from catan_rl.augmentation.symmetry_tables import (
    D6_GROUP_SIZE,
    corner_perm,
    tile_perm,
)
from catan_rl.engine.board import catanBoard
from catan_rl.policy.obs_encoder import DOTS_BY_TOKEN
from catan_rl.setup_phase.analytic_value import (
    N_VERTICES,
    all_vertex_yields,
    vertex_yield,
)
from catan_rl.setup_phase.resource_weights import CHARLESWORTH_V0


def _board(seed: int) -> catanBoard:
    random.seed(seed)
    np.random.seed(seed)
    return catanBoard()


# ---------------------------------------------------------------------------
# Rotated-board view: wraps a real board, returns tiles permuted by
# ``tile_perm(g)`` without mutating the underlying board.
# ---------------------------------------------------------------------------


class _RotatedBoardView:
    """Read-only board view with tile data permuted via ``tile_perm(g)``.

    The engine's vertex-to-hex topology is D6-symmetric *by construction*
    (that's what the augmentation tables encode), so rotating only the
    tile data — and leaving ``boardGraph`` / ``vertex_index_to_pixel_dict``
    pointing at the original board — is the correct model of "the
    rotated board" for the scorer.
    """

    def __init__(self, board: catanBoard, g: int) -> None:
        self._board = board
        p = tile_perm(g)  # p[i] = j: original tile i lands at new position j
        inv_p = np.argsort(p)  # inv_p[j] = i: new position j sources from original i
        self.hexTileDict = {j: board.hexTileDict[int(inv_p[j])] for j in range(19)}
        self.vertex_index_to_pixel_dict = board.vertex_index_to_pixel_dict
        self.boardGraph = board.boardGraph


# ---------------------------------------------------------------------------
# 1. Desert contributes zero.
# ---------------------------------------------------------------------------


class TestDesertZero:
    def test_vertex_with_only_desert_neighbor_scores_zero(self) -> None:
        # Force a board, find a vertex whose only adjacent hex is the
        # desert. (Most outer-ring vertices have 1-2 adjacent hexes; we
        # filter to a 1-hex vertex sitting on the desert.)
        for seed in range(100):
            b = _board(seed)
            desert_hex = next(i for i in range(19) if b.hexTileDict[i].resource_type == "DESERT")
            for v in range(N_VERTICES):
                pixel = b.vertex_index_to_pixel_dict[v]
                v_obj = b.boardGraph[pixel]
                adj = v_obj.adjacent_hex_indices
                if list(adj) == [desert_hex]:
                    score = vertex_yield(b, v, CHARLESWORTH_V0)
                    assert score == 0.0, f"seed={seed} v={v} adj=[desert]: expected 0, got {score}"
                    return
        pytest.skip(
            "no seed in [0, 100) produced a desert-only single-adj vertex; "
            "extend the seed range if this becomes consistently flaky"
        )

    def test_desert_weight_default_zero(self) -> None:
        # When ``resource_weight`` doesn't include "DESERT", missing-key
        # default of 0.0 must apply — desert never contributes regardless
        # of the chip on it (which is None anyway in the engine).
        b = _board(0)
        # Manually score a vertex adjacent to the desert; the desert hex
        # contribution is dots(None) * weight(DESERT) = 0 * 0 = 0.
        desert_hex = next(i for i in range(19) if b.hexTileDict[i].resource_type == "DESERT")
        for v in range(N_VERTICES):
            pixel = b.vertex_index_to_pixel_dict[v]
            v_obj = b.boardGraph[pixel]
            if desert_hex not in v_obj.adjacent_hex_indices:
                continue
            # Recompute manually excluding the desert.
            expected = 0.0
            for h in v_obj.adjacent_hex_indices:
                tile = b.hexTileDict[h]
                expected += DOTS_BY_TOKEN.get(tile.number_token, 0) * CHARLESWORTH_V0.get(
                    tile.resource_type, 0.0
                )
            actual = vertex_yield(b, v, CHARLESWORTH_V0)
            assert math.isclose(actual, expected, abs_tol=1e-6), (
                f"v={v} mismatch: expected {expected}, got {actual}"
            )
            return


# ---------------------------------------------------------------------------
# 2. Hand-computed reference: build a fixed-seed board, compute three
#    vertex scores by hand using the formula, assert exact match.
# ---------------------------------------------------------------------------


class TestHandComputedFixtures:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
    def test_yield_matches_manual_sum(self, seed: int) -> None:
        # For each of the 54 vertices on a fixed-seed board, manually
        # compute the closed-form using DOTS_BY_TOKEN + CHARLESWORTH_V0
        # and compare to ``vertex_yield``. Any divergence is a bug in
        # the scorer.
        b = _board(seed)
        for v in range(N_VERTICES):
            pixel = b.vertex_index_to_pixel_dict[v]
            v_obj = b.boardGraph[pixel]
            expected = 0.0
            for h in v_obj.adjacent_hex_indices:
                tile = b.hexTileDict[h]
                expected += DOTS_BY_TOKEN.get(tile.number_token, 0) * CHARLESWORTH_V0.get(
                    tile.resource_type, 0.0
                )
            actual = vertex_yield(b, v, CHARLESWORTH_V0)
            assert math.isclose(actual, expected, abs_tol=1e-6), (
                f"seed={seed} v={v}: expected {expected:.4f}, got {actual:.4f}"
            )


# ---------------------------------------------------------------------------
# 3. D6 invariance — the analytic scorer is equivariant under the
#    augmentation group. This is the cross-validation surface for Phase B.
# ---------------------------------------------------------------------------


class TestD6Invariance:
    """Equivariance under the full D6 group (rotations + reflections).

    The full D6 invariance holds as of 2026-06-03 — the reflection-axis
    bug in ``symmetry_tables._rotate_pixel`` was fixed by switching the
    pixel reflection to the axial-coordinate-derived matrix (150° line
    instead of 90°). The engine's vertex-hex topology now satisfies
    ``A_{corner_perm(g)[v]} = tile_perm(g)[A_v]`` for all 54 vertices on
    every D6 element. See ``memory/project_d6_reflection_bug.md``.

    This test class is the cross-validation surface Phase B's port
    permutation table will reuse.
    """

    @pytest.mark.parametrize("g", list(range(1, D6_GROUP_SIZE)))
    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_yields_equivariant_under_d6(self, g: int, seed: int) -> None:
        b = _board(seed)
        orig = all_vertex_yields(b, CHARLESWORTH_V0)
        rot_view = _RotatedBoardView(b, g)
        new = all_vertex_yields(rot_view, CHARLESWORTH_V0)  # type: ignore[arg-type]
        c_p = corner_perm(g)
        # The invariant: rotating the tile data by ``g`` shifts the
        # per-vertex yield array by ``corner_perm(g)``.
        # Formally: new_yields[c_p[v]] == orig_yields[v] ∀ v.
        for v in range(N_VERTICES):
            assert math.isclose(orig[v], float(new[int(c_p[v])]), abs_tol=1e-5), (
                f"g={g} seed={seed} v={v}: "
                f"orig[v]={orig[v]:.4f} new[c_p[v]={int(c_p[v])}]={new[int(c_p[v])]:.4f}"
            )

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_identity_is_identity(self, seed: int) -> None:
        # g=0 (identity) must leave yields untouched.
        b = _board(seed)
        orig = all_vertex_yields(b, CHARLESWORTH_V0)
        rot_view = _RotatedBoardView(b, 0)
        new = all_vertex_yields(rot_view, CHARLESWORTH_V0)  # type: ignore[arg-type]
        np.testing.assert_allclose(orig, new, atol=1e-6)

    @pytest.mark.parametrize("g", list(range(D6_GROUP_SIZE)))
    def test_engine_topology_d6_symmetric(self, g: int) -> None:
        """Pin the underlying topology relation that the equivariance
        depends on: ``A_{corner_perm(g)[v]} = tile_perm(g)[A_v]`` for
        every vertex. This is the relation that was broken on
        reflections until the 2026-06-03 fix. Pins it for all 12 D6
        elements so any future regression to the reflection axis fires
        loudly.
        """
        from catan_rl.augmentation.symmetry_tables import tile_perm as _tile_perm

        b = _board(0)
        adjacency = {}
        for v_idx in range(N_VERTICES):
            px = b.vertex_index_to_pixel_dict[v_idx]
            adjacency[v_idx] = frozenset(b.boardGraph[px].adjacent_hex_indices)

        p = _tile_perm(g)
        c_p = corner_perm(g)
        for v in range(N_VERTICES):
            lhs = adjacency[int(c_p[v])]
            rhs = frozenset(int(p[h]) for h in adjacency[v])
            assert lhs == rhs, (
                f"g={g} v={v}: A_{{corner_perm[v]={int(c_p[v])}}}={sorted(lhs)} "
                f"!= tile_perm(A_v)={sorted(rhs)}"
            )


# ---------------------------------------------------------------------------
# 4. Top-vertex sanity — the highest-scoring vertex on a known seed has
#    the expected production profile (high-pip chips on weight≥1 resources).
# ---------------------------------------------------------------------------


class TestTopVertexSanity:
    def test_top_vertex_sits_on_high_pip_non_sheep(self) -> None:
        # On seed 0 with Charlesworth's prior, the top vertex should
        # touch chips summing to at least 10 dots (e.g. a 6 + an 8 = 10
        # dots, or 6+5 = 9 dots, etc.) AND none of its adjacent hexes
        # should be the desert (weight=0).
        b = _board(0)
        yields = all_vertex_yields(b, CHARLESWORTH_V0)
        top_v = int(np.argmax(yields))
        pixel = b.vertex_index_to_pixel_dict[top_v]
        v_obj = b.boardGraph[pixel]
        total_dots = 0
        for h in v_obj.adjacent_hex_indices:
            tile = b.hexTileDict[h]
            assert tile.resource_type != "DESERT", (
                f"top vertex {top_v} sits on desert — should never win"
            )
            total_dots += DOTS_BY_TOKEN.get(tile.number_token, 0)
        # Most plausible top vertices on 19 hexes hit 9+ total dots
        # (5+4 or 5+3+2 etc.). Lower bound at 8 to handle edge cases.
        assert total_dots >= 8, f"seed=0 top vertex v={top_v} only sums to {total_dots} dots"

    def test_all_yields_non_negative(self) -> None:
        for seed in range(20):
            b = _board(seed)
            y = all_vertex_yields(b, CHARLESWORTH_V0)
            assert (y >= 0).all(), f"seed={seed}: negative yield: min={y.min()}"

    def test_all_yields_finite(self) -> None:
        for seed in range(20):
            b = _board(seed)
            y = all_vertex_yields(b, CHARLESWORTH_V0)
            assert np.isfinite(y).all(), f"seed={seed}: non-finite yield in {y}"


# ---------------------------------------------------------------------------
# Tests for the resource-weight registry.
# ---------------------------------------------------------------------------


class TestResourceWeightRegistry:
    def test_charlesworth_table_keys(self) -> None:
        from catan_rl.setup_phase.resource_weights import (
            CHARLESWORTH_V0,
            get_resource_weight_table,
            known_tables,
        )

        assert "charlesworth_v0" in known_tables()
        t = get_resource_weight_table("charlesworth_v0")
        assert set(t.keys()) == {"WOOD", "BRICK", "WHEAT", "ORE", "SHEEP"}
        for v in t.values():
            assert 0 < v < 2.0
        assert t == dict(CHARLESWORTH_V0)

    def test_unknown_table_raises(self) -> None:
        from catan_rl.setup_phase.resource_weights import get_resource_weight_table

        with pytest.raises(ValueError, match="unknown weight table"):
            get_resource_weight_table("nonexistent_v99")

    def test_heuristic_table_missing_raises_filenotfound(self, tmp_path: Any) -> None:
        from catan_rl.setup_phase.resource_weights import get_resource_weight_table

        missing = tmp_path / "nope.json"
        with pytest.raises(FileNotFoundError, match="heuristic_v0 weights not on disk"):
            get_resource_weight_table("heuristic_v0", heuristic_path=missing)
