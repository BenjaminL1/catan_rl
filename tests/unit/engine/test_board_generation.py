"""Tests for board generation invariants (Colonist.io 1v1 ruleset).

The 1v1 ruleset on Colonist.io uses balanced map generation:
1. No two 6/8 tiles adjacent (standard Catan rule).
2. **No two same-number tiles adjacent** (Colonist.io 1v1 rule — top-
   level players cited this; pre-existing engine did not enforce it).

These tests pin both invariants across many seeded boards.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from catan_rl.engine.board import catanBoard

# Hex adjacency for the standard 19-hex board (axial → flat-top indexing
# used by the engine). Same as ``catanBoard.checkHexNeighbors``.
_HEX_NEIGHBORS: dict[int, list[int]] = {
    0: [1, 2, 3, 4, 5, 6],
    1: [0, 2, 6, 7, 8, 18],
    2: [0, 1, 3, 8, 9, 10],
    3: [0, 2, 4, 10, 11, 12],
    4: [0, 3, 5, 12, 13, 14],
    5: [0, 4, 6, 14, 15, 16],
    6: [0, 1, 5, 16, 17, 18],
    7: [1, 8, 18],
    8: [1, 2, 7, 9],
    9: [2, 8, 10],
    10: [2, 3, 9, 11],
    11: [3, 10, 12],
    12: [3, 4, 11, 13],
    13: [4, 12, 14],
    14: [4, 5, 13, 15],
    15: [5, 14, 16],
    16: [5, 6, 15, 17],
    17: [6, 16, 18],
    18: [1, 6, 7, 17],
}


def _board_tokens(seed: int) -> dict[int, int | None]:
    """Build a board with the given seed, return {hex_idx → number_token}."""
    random.seed(seed)
    np.random.seed(seed)
    board = catanBoard()
    return {i: board.hexTileDict[i].number_token for i in range(19)}


class TestNoAdjacentSixOrEight:
    """The pre-existing rule that survives the new constraint."""

    @pytest.mark.parametrize("seed", list(range(50)))
    def test_no_6_adjacent_to_6_or_8(self, seed: int) -> None:
        tokens = _board_tokens(seed)
        for h_idx, t in tokens.items():
            if t not in (6, 8):
                continue
            for nb in _HEX_NEIGHBORS[h_idx]:
                if tokens[nb] in (6, 8):
                    pytest.fail(
                        f"seed={seed}: hex {h_idx} ({t}) adjacent to hex "
                        f"{nb} ({tokens[nb]}) — 6/8 rule violated"
                    )


class TestNoAdjacentSameNumber:
    """The new constraint: no two equal number tokens adjacent."""

    @pytest.mark.parametrize("seed", list(range(50)))
    def test_no_same_number_adjacent(self, seed: int) -> None:
        tokens = _board_tokens(seed)
        for h_idx, t in tokens.items():
            if t is None or t == 0:  # desert has no token
                continue
            for nb in _HEX_NEIGHBORS[h_idx]:
                if tokens[nb] == t:
                    pytest.fail(
                        f"seed={seed}: hex {h_idx} ({t}) adjacent to hex "
                        f"{nb} ({tokens[nb]}) — same-number rule violated"
                    )


class TestNumberTokenDistribution:
    """Sanity: the distribution itself doesn't change."""

    def test_token_counts_match_standard(self) -> None:
        random.seed(0)
        np.random.seed(0)
        board = catanBoard()
        tokens = [board.hexTileDict[i].number_token for i in range(19)]
        # Standard Catan distribution: one each of 2 and 12, two each of
        # 3 4 5 6 8 9 10 11, and one desert with token=None or 0.
        from collections import Counter

        counts = Counter(t for t in tokens if t is not None and t != 0)
        assert counts.get(2, 0) == 1
        assert counts.get(12, 0) == 1
        for n in (3, 4, 5, 6, 8, 9, 10, 11):
            assert counts.get(n, 0) == 2, f"number {n}: expected 2, got {counts.get(n, 0)}"


class TestGenerationConverges:
    """The balanced-map rejection loop must converge within reasonable retries."""

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_board_constructs_under_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        # Should complete without raising.
        catanBoard()
