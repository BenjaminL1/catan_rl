"""Phase 2d smoke — recorder-specific regression tests.

Matchup-matrix coverage moved to ``tests/integration/test_record_matchups.py``
(Phase 4). This file keeps regression tests that are tied to specific
Phase 2d code paths (actor attribution, heuristic-as-agent raise,
winner/truncation XOR) and that aren't duplicated by the parametrized
matrix sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from catan_rl.replay import record_game, save_replay
from catan_rl.replay.player_factory import PlayerSpec as RecorderPlayerSpec


def test_record_game_winner_or_truncation(tmp_path: Path) -> None:
    """The replay metadata MUST decide: either a winner is named OR
    the game truncated cleanly (winner=None, winner_seat=None)."""
    replay = record_game(
        RecorderPlayerSpec(kind="random", ckpt_path=None),
        RecorderPlayerSpec(kind="random", ckpt_path=None),
        seed=7,
        max_turns=60,
    )
    # XOR: either both winner fields are set, or neither.
    has_winner = replay.metadata.winner is not None
    has_seat = replay.metadata.winner_seat is not None
    assert has_winner == has_seat

    out_path = tmp_path / "r.json"
    save_replay(replay, out_path)
    assert out_path.exists()


def test_main_phase_actor_attribution_is_split() -> None:
    """Reviewer-flagged HIGH-1 + MED-2 regression: events from the opp's
    turn (inside the same env.step where the agent ends their turn)
    must be attributed to the opp ReplayStep, not folded into the
    agent's. We check this end-to-end by recording a short
    (random, heuristic) game and asserting:

    * BOTH actors appear in main-phase ReplaySteps.
    * The total main-phase actor count is roughly balanced (each
      actor should appear in at least 25% of main steps over a
      short game).
    """
    replay = record_game(
        RecorderPlayerSpec(kind="random", ckpt_path=None),
        RecorderPlayerSpec(kind="heuristic", ckpt_path=None),
        seed=11,
        max_turns=40,
    )
    main_actors = [s.actor for s in replay.steps if s.kind in ("main", "terminal")]
    # Both actors appear.
    assert "player_a" in main_actors
    assert "player_b" in main_actors
    # Both at least 25% — sanity check that one isn't being absorbed
    # into the other.
    n = len(main_actors)
    if n >= 4:
        assert main_actors.count("player_a") >= n // 4
        assert main_actors.count("player_b") >= n // 4


def test_record_game_heuristic_as_agent_raises() -> None:
    """Tracks the Phase 2d-known limitation: heuristic-as-agent matchups
    raise :class:`NotImplementedError` until the heuristic-to-action
    translator lands."""
    with pytest.raises(NotImplementedError, match="heuristic"):
        record_game(
            RecorderPlayerSpec(kind="heuristic", ckpt_path=None),
            RecorderPlayerSpec(kind="random", ckpt_path=None),
            seed=1,
            max_turns=20,
        )

    with pytest.raises(NotImplementedError, match="heuristic"):
        record_game(
            RecorderPlayerSpec(kind="heuristic", ckpt_path=None),
            RecorderPlayerSpec(kind="heuristic", ckpt_path=None),
            seed=1,
            max_turns=20,
        )
