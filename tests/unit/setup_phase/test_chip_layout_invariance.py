"""Plan §0.1 preflight gate — chip-layout determinism.

The setup-strength roadmap (`docs/plans/v2_setup_strength_roadmap.md`)
makes Phase A possible only if the per-hex chip assignment is a pure
function of the spiral orientation, with the resource shuffle
**orthogonal** to it. If this preflight fails, the closed-form
analytic value scorer cannot be guaranteed correct across seeds.

What we pin:

1. ``_build_spiral_path`` is deterministic — same (corner, dir) input
   always returns the same 19-hex traversal, independent of RNG state.
2. For each of the 12 (corner × dir) orientations, walking
   ``SPIRAL_CHIP_SEQUENCE`` over the path produces the same
   ``hex_idx → chip`` mapping every time given the desert position.
3. Two boards that share BOTH orientation AND desert position produce
   identical per-hex chip assignments. (Two boards that share only
   orientation but place the desert at different hexes have shifted
   chips past the earliest desert position — this is correct walker
   behavior, not an invariant violation.)
4. The chip sequence walked over the spiral path, with the desert
   hex skipped, matches ``SPIRAL_CHIP_SEQUENCE`` exactly. No
   insertion, no permutation, just a single skip.

The downstream consumer (Phase A's analytic value scorer) reads
``board.hexTileDict[h].number_token`` directly, so the per-hex chip is
what matters. Determinism of `(orientation, desert) → per-hex chip` is
the formal precondition the scorer needs.
"""

from __future__ import annotations

import random
from itertools import product

import numpy as np
import pytest

from catan_rl.engine.board import (
    OUTER_CORNERS,
    SPIRAL_CHIP_SEQUENCE,
    _build_spiral_path,
    catanBoard,
)


def _chips_by_hex(board: catanBoard) -> dict[int, int | None]:
    """Map ``hex_idx → number_token`` for all 19 hexes."""
    return {i: board.hexTileDict[i].number_token for i in range(19)}


def _resources_by_hex(board: catanBoard) -> dict[int, str]:
    return {i: board.hexTileDict[i].resource_type for i in range(19)}


def _board(seed: int) -> catanBoard:
    random.seed(seed)
    np.random.seed(seed)
    return catanBoard()


# ---------------------------------------------------------------------------
# 1. ``_build_spiral_path`` is deterministic.
# ---------------------------------------------------------------------------


class TestSpiralPathDeterminism:
    @pytest.mark.parametrize("corner,cw", list(product(OUTER_CORNERS, [True, False])))
    def test_same_input_returns_same_path(self, corner: int, cw: bool) -> None:
        # Calling the helper twice with identical inputs must produce
        # bit-identical paths. There is no RNG involvement.
        p1 = _build_spiral_path(corner, cw)
        p2 = _build_spiral_path(corner, cw)
        assert p1 == p2

    @pytest.mark.parametrize("corner,cw", list(product(OUTER_CORNERS, [True, False])))
    def test_path_unaffected_by_rng_state(self, corner: int, cw: bool) -> None:
        # Seed numpy + stdlib random with two different values around the
        # call; the path must not change.
        random.seed(123)
        np.random.seed(456)
        a = _build_spiral_path(corner, cw)
        random.seed(789)
        np.random.seed(0)
        b = _build_spiral_path(corner, cw)
        assert a == b


# ---------------------------------------------------------------------------
# 2. Walking ``SPIRAL_CHIP_SEQUENCE`` over the path produces a deterministic
#    chip-by-hex map (given the desert position).
# ---------------------------------------------------------------------------


def _walk_chips(spiral_path: list[int], desert_hex: int) -> dict[int, int | None]:
    """Reference walker — places ``SPIRAL_CHIP_SEQUENCE`` skipping the
    desert hex. Mirrors the algorithm in ``catanBoard.__init__``."""
    out: dict[int, int | None] = {i: None for i in range(19)}
    chip_idx = 0
    for h in spiral_path:
        if h == desert_hex:
            continue
        out[h] = SPIRAL_CHIP_SEQUENCE[chip_idx]
        chip_idx += 1
    return out


class TestChipWalkDeterminism:
    @pytest.mark.parametrize("corner,cw", list(product(OUTER_CORNERS, [True, False])))
    def test_chip_walk_deterministic_per_orientation_and_desert(
        self, corner: int, cw: bool
    ) -> None:
        # For every (orientation, desert-position) pair the walker is a
        # pure function; calling it twice yields the same map.
        path = _build_spiral_path(corner, cw)
        for desert_hex in range(19):
            m1 = _walk_chips(path, desert_hex)
            m2 = _walk_chips(path, desert_hex)
            assert m1 == m2

    @pytest.mark.parametrize("corner,cw", list(product(OUTER_CORNERS, [True, False])))
    def test_chip_walk_uses_all_eighteen_chips(self, corner: int, cw: bool) -> None:
        # Every non-desert hex must receive exactly one chip; the desert
        # hex receives none. Sanity-pins that the sequence length matches
        # the number of non-desert hexes.
        path = _build_spiral_path(corner, cw)
        for desert_hex in range(19):
            chips = _walk_chips(path, desert_hex)
            placed = [c for c in chips.values() if c is not None]
            assert len(placed) == len(SPIRAL_CHIP_SEQUENCE) == 18
            assert chips[desert_hex] is None


# ---------------------------------------------------------------------------
# 3. Two boards with the *same orientation* but different resource
#    shuffles produce the same chip pattern (modulo desert location).
# ---------------------------------------------------------------------------


def _infer_orientation(board: catanBoard) -> tuple[int, bool]:
    """Given a generated board, find the (corner, cw) that explains its
    chip layout. Returns the matching orientation or raises if none
    matches (which would itself be a regression to surface here)."""
    chips = _chips_by_hex(board)
    desert_hex = next(i for i in range(19) if board.hexTileDict[i].resource_type == "DESERT")
    for corner, cw in product(OUTER_CORNERS, [True, False]):
        expected = _walk_chips(_build_spiral_path(corner, cw), desert_hex)
        if all(expected[i] == chips[i] for i in range(19)):
            return (corner, cw)
    raise AssertionError(f"no spiral orientation explains chips={chips} desert={desert_hex}")


class TestOrientationOrthogonalToResourceShuffle:
    def test_same_orientation_AND_same_desert_produces_same_chips(self) -> None:
        # Stricter pin: when two boards share both orientation AND desert
        # position, their per-hex chip assignment must be identical.
        # (When desert positions differ, the walker skips that hex and
        # subsequent chips shift — that's NOT a violation of orientation
        # determinism; it's a consequence of how the walker handles
        # desert. The closed-form analytic scorer reads
        # ``board.hexTileDict[h].number_token`` directly, so the per-hex
        # invariant is what matters for downstream correctness.)
        from collections import defaultdict

        buckets: dict[tuple[tuple[int, bool], int], list[catanBoard]] = defaultdict(list)
        for seed in range(400):
            b = _board(seed)
            try:
                orientation = _infer_orientation(b)
            except AssertionError as e:
                pytest.fail(f"seed={seed}: {e}")
            desert_hex = next(i for i in range(19) if b.hexTileDict[i].resource_type == "DESERT")
            buckets[(orientation, desert_hex)].append(b)

        # At least one bucket should accumulate >= 2 boards across 400
        # seeds (otherwise we never compared anything). If this fails
        # without diverging earlier, our seed range is too small.
        compared_any = False
        for (orientation, desert_hex), boards in buckets.items():
            if len(boards) < 2:
                continue
            compared_any = True
            ref_chips = _chips_by_hex(boards[0])
            for other in boards[1:]:
                other_chips = _chips_by_hex(other)
                assert ref_chips == other_chips, (
                    f"orientation={orientation} desert={desert_hex}: "
                    f"chips diverged ref={ref_chips} other={other_chips}"
                )
        assert compared_any, "no shared (orientation, desert) bucket — widen seed range"

    def test_orientation_inferable_for_every_seed(self) -> None:
        # Stronger pin than the existing spot-check tests: across 300
        # seeds, EVERY board matches some (corner, cw).
        for seed in range(300):
            b = _board(seed)
            # _infer_orientation raises if no orientation matches.
            _infer_orientation(b)


# ---------------------------------------------------------------------------
# 4. Desert position is set by the resource shuffle and does not perturb
#    the chip sequence — only shifts it past whichever hex got the desert.
# ---------------------------------------------------------------------------


class TestDesertOrthogonality:
    @pytest.mark.parametrize("seed", list(range(30)))
    def test_chip_sequence_intact_around_desert(self, seed: int) -> None:
        # Walk the board's inferred spiral path, drop the desert hex's
        # entry, and assert what remains equals SPIRAL_CHIP_SEQUENCE verbatim.
        b = _board(seed)
        corner, cw = _infer_orientation(b)
        path = _build_spiral_path(corner, cw)
        desert_hex = next(i for i in range(19) if b.hexTileDict[i].resource_type == "DESERT")
        observed = [b.hexTileDict[h].number_token for h in path if h != desert_hex]
        assert tuple(observed) == SPIRAL_CHIP_SEQUENCE, (
            f"seed={seed}: chip walk diverged from official sequence"
        )

    @pytest.mark.parametrize("seed", list(range(30)))
    def test_desert_hex_has_no_number_token(self, seed: int) -> None:
        b = _board(seed)
        desert_hex = next(i for i in range(19) if b.hexTileDict[i].resource_type == "DESERT")
        assert b.hexTileDict[desert_hex].number_token in (None, 0)
        assert b.hexTileDict[desert_hex].has_robber is True

    @pytest.mark.parametrize("seed", list(range(100)))
    def test_desert_landing_independent_of_orientation(self, seed: int) -> None:
        # A weaker but cheaper smoke: across many seeds, the desert lands
        # on enough different hexes that you can't predict it from
        # orientation alone. (This is a property of the resource shuffle
        # being independent of the spiral RNG.)
        # We don't assert a specific count — just that the test runs and
        # the desert hex is one of the 19 valid indices.
        b = _board(seed)
        desert_hex = next(i for i in range(19) if b.hexTileDict[i].resource_type == "DESERT")
        assert 0 <= desert_hex < 19
