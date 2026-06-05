"""Tests for the official ABC spiral number placement (Colonist.io algorithm).

Per audit (2026-06-02): Colonist uses the official Catan ABC chip sequence
walked in spiral order around the board, with the spiral's
starting outer corner + direction chosen randomly. Resources are
shuffled independently.

Sequence: ``5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11``

Spiral path (canonical, start at hex 7 going CW outer):
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,   (outer ring)
    1, 2, 3, 4, 5, 6,                              (inner ring)
    0                                              (center)

These tests pin:
1. The chip sequence is used verbatim (in some rotation/mirror) on every board.
2. All 12 orientations (6 outer corners × 2 directions) are reachable.
3. Resources remain a uniform random shuffle independent of the spiral.
4. Desert is skipped (no chip placed there).
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from catan_rl.engine.board import (
    OUTER_CORNERS,
    SPIRAL_CHIP_SEQUENCE,
    _build_spiral_path,
    catanBoard,
)


def _board(seed: int) -> catanBoard:
    random.seed(seed)
    np.random.seed(seed)
    return catanBoard()


class TestSpiralPathConstruction:
    """The helper that builds a 19-hex spiral order for a given orientation."""

    def test_canonical_spiral_start_hex_7_cw(self) -> None:
        path = _build_spiral_path(start_corner=7, clockwise=True)
        assert path == [
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            1,
            2,
            3,
            4,
            5,
            6,
            0,
        ]

    def test_canonical_spiral_start_hex_7_ccw(self) -> None:
        path = _build_spiral_path(start_corner=7, clockwise=False)
        assert path == [
            7,
            18,
            17,
            16,
            15,
            14,
            13,
            12,
            11,
            10,
            9,
            8,
            1,
            6,
            5,
            4,
            3,
            2,
            0,
        ]

    def test_rotation_60_cw_starts_at_hex_9(self) -> None:
        # 60° CW from canonical: corner shifts from 7 → 9, inner shifts by 1.
        path = _build_spiral_path(start_corner=9, clockwise=True)
        assert path[0] == 9
        assert path[:12] == [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 7, 8]
        assert path[12:18] == [2, 3, 4, 5, 6, 1]
        assert path[18] == 0

    def test_rotation_120_cw_starts_at_hex_11(self) -> None:
        path = _build_spiral_path(start_corner=11, clockwise=True)
        assert path[:12] == [11, 12, 13, 14, 15, 16, 17, 18, 7, 8, 9, 10]
        assert path[12:18] == [3, 4, 5, 6, 1, 2]
        assert path[18] == 0

    def test_all_six_corner_starts_valid(self) -> None:
        for corner in OUTER_CORNERS:
            for cw in (True, False):
                path = _build_spiral_path(corner, cw)
                # 19 unique entries covering all engine hex indices.
                assert sorted(path) == list(range(19))
                assert path[0] == corner
                assert path[-1] == 0  # center is always last

    def test_invalid_start_corner_raises(self) -> None:
        # 8 is an outer between-corner hex, not a corner.
        with pytest.raises(ValueError):
            _build_spiral_path(start_corner=8, clockwise=True)
        # 1 is an inner hex.
        with pytest.raises(ValueError):
            _build_spiral_path(start_corner=1, clockwise=True)
        # 0 is the center.
        with pytest.raises(ValueError):
            _build_spiral_path(start_corner=0, clockwise=True)


class TestSpiralChipPlacement:
    """Every board's chips match the official ABC sequence in spiral order."""

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_chip_sequence_is_official(self, seed: int) -> None:
        board = _board(seed)
        # Reconstruct: walk every spiral orientation; one must match.
        observed = {i: board.hexTileDict[i].number_token for i in range(19)}
        found_match = False
        for corner in OUTER_CORNERS:
            for cw in (True, False):
                path = _build_spiral_path(corner, cw)
                # Apply the sequence skipping the desert.
                expected: dict[int, int | None] = {i: None for i in range(19)}
                chip_idx = 0
                for h in path:
                    if board.hexTileDict[h].resource_type == "DESERT":
                        continue
                    expected[h] = SPIRAL_CHIP_SEQUENCE[chip_idx]
                    chip_idx += 1
                if all(expected[i] == observed[i] for i in range(19)):
                    found_match = True
                    break
            if found_match:
                break
        assert found_match, f"seed={seed}: board does not match any spiral orientation"

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_token_distribution_matches_standard(self, seed: int) -> None:
        board = _board(seed)
        tokens = [
            board.hexTileDict[i].number_token
            for i in range(19)
            if board.hexTileDict[i].resource_type != "DESERT"
        ]
        counts = Counter(tokens)
        for num in (2, 12):
            assert counts[num] == 1, f"seed={seed}: expected one {num}, got {counts[num]}"
        for num in (3, 4, 5, 6, 8, 9, 10, 11):
            assert counts[num] == 2, f"seed={seed}: expected two {num}s, got {counts[num]}"

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_desert_has_no_token(self, seed: int) -> None:
        board = _board(seed)
        for i in range(19):
            tile = board.hexTileDict[i]
            if tile.resource_type == "DESERT":
                assert tile.number_token in (None, 0)
                assert tile.has_robber is True

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_exactly_one_desert(self, seed: int) -> None:
        board = _board(seed)
        n_desert = sum(1 for i in range(19) if board.hexTileDict[i].resource_type == "DESERT")
        assert n_desert == 1


class TestSpiralAdjacencyImpliedRules:
    """Rules guaranteed by spiral construction (no rejection sampling needed)."""

    @pytest.mark.parametrize("seed", list(range(30)))
    def test_no_6_or_8_adjacent_to_6_or_8(self, seed: int) -> None:
        from tests.unit.engine.test_board_generation import _HEX_NEIGHBORS

        board = _board(seed)
        tokens = {i: board.hexTileDict[i].number_token for i in range(19)}
        for h_idx, t in tokens.items():
            if t not in (6, 8):
                continue
            for nb in _HEX_NEIGHBORS[h_idx]:
                assert tokens[nb] not in (6, 8), (
                    f"seed={seed}: hex {h_idx}({t}) adj hex {nb}({tokens[nb]})"
                )

    @pytest.mark.parametrize("seed", list(range(30)))
    def test_no_same_number_adjacent(self, seed: int) -> None:
        from tests.unit.engine.test_board_generation import _HEX_NEIGHBORS

        board = _board(seed)
        tokens = {i: board.hexTileDict[i].number_token for i in range(19)}
        for h_idx, t in tokens.items():
            if t is None or t == 0:
                continue
            for nb in _HEX_NEIGHBORS[h_idx]:
                assert tokens[nb] != t, f"seed={seed}: hex {h_idx} adj hex {nb} both have {t}"

    @pytest.mark.parametrize("seed", list(range(30)))
    def test_no_2_or_12_adjacent_to_each_other(self, seed: int) -> None:
        # Colonist consequence: extreme low-pip tokens never adjacent.
        from tests.unit.engine.test_board_generation import _HEX_NEIGHBORS

        board = _board(seed)
        tokens = {i: board.hexTileDict[i].number_token for i in range(19)}
        for h_idx, t in tokens.items():
            if t not in (2, 12):
                continue
            for nb in _HEX_NEIGHBORS[h_idx]:
                assert tokens[nb] not in (2, 12), (
                    f"seed={seed}: hex {h_idx}({t}) adj hex {nb}({tokens[nb]})"
                )


class TestOrientationDiversity:
    """All 12 (6 corners × 2 directions) orientations should be reachable."""

    def test_many_seeds_cover_multiple_orientations(self) -> None:
        # The orientation can be inferred by checking which (start, dir)
        # explains each seed's chip layout. Across 200 seeds we expect at
        # least 8 of the 12 orientations to appear (uniform sampling would
        # give ~17 per orientation).
        observed_orientations: set[tuple[int, bool]] = set()
        for seed in range(200):
            board = _board(seed)
            chips = {i: board.hexTileDict[i].number_token for i in range(19)}
            for corner in OUTER_CORNERS:
                for cw in (True, False):
                    path = _build_spiral_path(corner, cw)
                    expected = {i: None for i in range(19)}
                    chip_idx = 0
                    for h in path:
                        if board.hexTileDict[h].resource_type == "DESERT":
                            continue
                        expected[h] = SPIRAL_CHIP_SEQUENCE[chip_idx]
                        chip_idx += 1
                    if all(expected[i] == chips[i] for i in range(19)):
                        observed_orientations.add((corner, cw))
                        break
                else:
                    continue
                break
        # Expect at least 8 of 12 orientations to show up.
        assert len(observed_orientations) >= 8, (
            f"only {len(observed_orientations)} orientations observed: "
            f"{sorted(observed_orientations)}"
        )


class TestResourceShuffleIndependence:
    """Resources shouldn't correlate with the spiral — they're independent."""

    def test_desert_can_land_anywhere(self) -> None:
        """Across many seeds, the desert should land on many different hexes."""
        desert_locations: set[int] = set()
        for seed in range(200):
            board = _board(seed)
            for i in range(19):
                if board.hexTileDict[i].resource_type == "DESERT":
                    desert_locations.add(i)
                    break
        # In 200 seeds we should see the desert on at least 8 different hexes.
        assert len(desert_locations) >= 8, (
            f"desert only landed on {len(desert_locations)} hexes — too rigid"
        )

    def test_resource_count_invariant(self) -> None:
        random.seed(0)
        np.random.seed(0)
        board = catanBoard()
        types = [board.hexTileDict[i].resource_type for i in range(19)]
        counts = Counter(types)
        assert counts["DESERT"] == 1
        assert counts["ORE"] == 3
        assert counts["BRICK"] == 3
        assert counts["WHEAT"] == 4
        assert counts["WOOD"] == 4
        assert counts["SHEEP"] == 4
