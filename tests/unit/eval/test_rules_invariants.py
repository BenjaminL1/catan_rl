"""Tests for `eval/rules_invariants.py`.

Runs each check against a real ``catanGame`` instance (via the env)
and against synthetic mocks where the check needs to fire.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from catan_rl.eval.rules_invariants import (
    RulesInvariantViolation,
    check_friendly_robber,
    check_max_points_15,
    check_no_p2p_trade,
    check_stacked_dice,
    check_terminal_state,
    check_two_players,
    run_all_invariants,
)

# ---------------------------------------------------------------------------
# Real env fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_game():
    """Construct a fresh ``catanGame`` via the env path."""
    from catan_rl.env.catan_env import CatanEnv

    random.seed(0)
    np.random.seed(0)
    env = CatanEnv(opponent_type="random", max_turns=50)
    env.reset(seed=0)
    yield env.game
    env.close()


# ---------------------------------------------------------------------------
# Individual checks pass on a real fresh game
# ---------------------------------------------------------------------------


class TestRealGamePasses:
    def test_max_points_passes(self, fresh_game) -> None:
        check_max_points_15(fresh_game)

    def test_two_players_passes(self, fresh_game) -> None:
        check_two_players(fresh_game)

    def test_friendly_robber_passes(self, fresh_game) -> None:
        check_friendly_robber(fresh_game)

    def test_no_p2p_trade_passes(self, fresh_game) -> None:
        check_no_p2p_trade(fresh_game)

    def test_stacked_dice_passes(self, fresh_game) -> None:
        check_stacked_dice(fresh_game)

    def test_run_all_returns_empty(self, fresh_game) -> None:
        violations = run_all_invariants(fresh_game, truncated=True)
        assert violations == [], f"fresh game should have no rules violations; got {violations}"


# ---------------------------------------------------------------------------
# Individual checks fail on intentionally broken mocks
# ---------------------------------------------------------------------------


class TestBrokenGameRaises:
    def test_max_points_10_rejected(self) -> None:
        broken = SimpleNamespace(maxPoints=10)
        with pytest.raises(RulesInvariantViolation, match="maxPoints"):
            check_max_points_15(broken)

    def test_num_players_4_rejected(self) -> None:
        broken = SimpleNamespace(
            numPlayers=4,
            playerQueue=SimpleNamespace(queue=[None, None, None, None]),
        )
        with pytest.raises(RulesInvariantViolation, match="numPlayers"):
            check_two_players(broken)

    def test_friendly_robber_missing_method_rejected(self) -> None:
        # Board missing get_robber_spots → violation.
        broken = SimpleNamespace(board=SimpleNamespace())
        with pytest.raises(RulesInvariantViolation, match="Friendly Robber"):
            check_friendly_robber(broken)

    def test_friendly_robber_filter_leak_detected(self) -> None:
        # Construct a synthetic game where a <3-VP player owns vertex
        # 'V' adjacent to hex 7, but ``get_robber_spots`` includes hex
        # 7 — a regression that silently dropped the friendly-robber
        # filter. The behavioural probe must raise.
        protected = SimpleNamespace(
            name="P0",
            victoryPoints=2,
            devCards={"VP": 0},
            buildGraph={"SETTLEMENTS": ["V"], "CITIES": []},
        )
        vertex_obj = SimpleNamespace(adjacent_hex_indices=[7])
        board = SimpleNamespace(
            # Buggy: returns hex 7 even though P0 sits on it with 2 VP.
            get_robber_spots=lambda: {7: object(), 3: object()},
            boardGraph={"V": vertex_obj},
        )
        broken = SimpleNamespace(
            board=board,
            playerQueue=SimpleNamespace(queue=[protected]),
        )
        with pytest.raises(RulesInvariantViolation, match="Friendly Robber filter leak"):
            check_friendly_robber(broken)

    def test_friendly_robber_filter_correct_passes(self) -> None:
        # Same synthetic shape, but the filter correctly excludes hex 7.
        protected = SimpleNamespace(
            name="P0",
            victoryPoints=2,
            devCards={"VP": 0},
            buildGraph={"SETTLEMENTS": ["V"], "CITIES": []},
        )
        vertex_obj = SimpleNamespace(adjacent_hex_indices=[7])
        board = SimpleNamespace(
            get_robber_spots=lambda: {3: object(), 11: object()},
            boardGraph={"V": vertex_obj},
        )
        ok = SimpleNamespace(
            board=board,
            playerQueue=SimpleNamespace(queue=[protected]),
        )
        check_friendly_robber(ok)  # no raise

    def test_p2p_trade_event_rejected(self) -> None:
        broken = SimpleNamespace(
            broadcast=SimpleNamespace(events=[{"type": "P2P_TRADE", "from": "p1", "to": "p2"}])
        )
        with pytest.raises(RulesInvariantViolation, match="P2P"):
            check_no_p2p_trade(broken)

    def test_player_trade_event_rejected(self) -> None:
        broken = SimpleNamespace(broadcast=SimpleNamespace(events=[{"type": "PLAYER_TRADE_OFFER"}]))
        with pytest.raises(RulesInvariantViolation, match="P2P"):
            check_no_p2p_trade(broken)

    def test_stacked_dice_absent_rejected(self) -> None:
        broken = SimpleNamespace(dice=MagicMock(spec=list))
        with pytest.raises(RulesInvariantViolation, match="StackedDice"):
            check_stacked_dice(broken)

    def test_truncated_negative_vp_rejected(self) -> None:
        broken = SimpleNamespace(
            playerQueue=SimpleNamespace(
                queue=[
                    SimpleNamespace(name="A", victoryPoints=-1),
                    SimpleNamespace(name="B", victoryPoints=3),
                ]
            )
        )
        with pytest.raises(RulesInvariantViolation, match="negative-VP"):
            check_terminal_state(broken, truncated=True)

    def test_terminated_no_winner_rejected(self) -> None:
        broken = SimpleNamespace(
            playerQueue=SimpleNamespace(
                queue=[
                    SimpleNamespace(name="A", victoryPoints=10),
                    SimpleNamespace(name="B", victoryPoints=11),
                ]
            )
        )
        with pytest.raises(RulesInvariantViolation, match="no player reached 15"):
            check_terminal_state(broken, truncated=False)


# ---------------------------------------------------------------------------
# Aggregator collects all violations
# ---------------------------------------------------------------------------


class TestAggregator:
    def test_aggregates_multiple_violations(self) -> None:
        # Build a game with multiple violations and verify they all
        # surface (not short-circuit on the first).
        broken = SimpleNamespace(
            maxPoints=10,
            numPlayers=4,
            playerQueue=SimpleNamespace(
                queue=[
                    SimpleNamespace(name="A", victoryPoints=10),
                    SimpleNamespace(name="B", victoryPoints=11),
                    SimpleNamespace(name="C", victoryPoints=5),
                    SimpleNamespace(name="D", victoryPoints=2),
                ]
            ),
            board=SimpleNamespace(get_robber_spots=lambda: list(range(19))),
            broadcast=SimpleNamespace(events=[{"type": "P2P_TRADE"}]),
            dice=None,
        )
        violations = run_all_invariants(broken, truncated=False)
        # Expect at least 4 violations (maxPoints, numPlayers,
        # terminal, p2p, stacked_dice — 5 total but some may overlap).
        assert len(violations) >= 4
        msg = "\n".join(violations)
        assert "maxPoints" in msg
        assert "numPlayers" in msg
        assert "P2P" in msg
        assert "StackedDice" in msg

    def test_empty_event_log_no_violation(self) -> None:
        # If the broadcast has no ``events`` attribute, the check
        # silently skips rather than crashing.
        broken = SimpleNamespace(broadcast=SimpleNamespace())
        check_no_p2p_trade(broken)  # no raise
