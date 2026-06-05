"""Phase 2d smoke test — record_game produces a valid Replay end-to-end.

Coverage matrix (Phase 2d's 6 supported matchups; Phase 4 will extend):

* ``(random, random)`` seat=0
* ``(random, heuristic)`` seat=0 (heuristic in opp slot)

The two policy-required matchups are gated on a checkpoint that may
not be present in CI; they're covered by a separate Phase 4 smoke
suite. The two heuristic-as-agent matchups raise NotImplementedError
in Phase 2d and are asserted in :mod:`tests.unit.replay.test_record_cli`.

The test loads the produced JSON via :func:`load_replay` to confirm
the schema contract is satisfied end-to-end — round-trip is the
load-bearing assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from catan_rl.replay import load_replay, record_game, save_replay
from catan_rl.replay.player_factory import PlayerSpec as RecorderPlayerSpec


@pytest.mark.parametrize(
    "kind_a,kind_b",
    [
        ("random", "random"),
        ("random", "heuristic"),
    ],
)
def test_record_game_supported_matchups(kind_a: str, kind_b: str, tmp_path: Path) -> None:
    spec_a = RecorderPlayerSpec(kind=kind_a, ckpt_path=None)  # type: ignore[arg-type]
    spec_b = RecorderPlayerSpec(kind=kind_b, ckpt_path=None)  # type: ignore[arg-type]

    replay = record_game(
        spec_a,
        spec_b,
        seed=42,
        max_turns=80,  # short cap so the test finishes quickly even on truncation
    )

    # ----- structural assertions -----
    assert replay.schema_version >= 1
    assert replay.metadata.player_a.kind == kind_a
    assert replay.metadata.player_b.kind == kind_b
    assert replay.metadata.seed == 42
    assert replay.metadata.partial is False
    assert replay.metadata.total_steps == len(replay.steps)

    # Setup phase MUST emit exactly 4 setup steps regardless of seat.
    setup_steps = [s for s in replay.steps if s.kind == "setup"]
    assert len(setup_steps) == 4

    # Snake-draft actor sequence: seat 0 → [a, b, b, a]; seat 1 →
    # [b, a, a, b]. The recorder uses agent_seat=0 for (random, *)
    # matchups (per ``_resolve_seat_and_opp``).
    actors = [s.actor for s in setup_steps]
    if replay.metadata.player_a.seat_index == 0:
        assert actors == ["player_a", "player_b", "player_b", "player_a"]
    else:
        assert actors == ["player_b", "player_a", "player_a", "player_b"]

    # Each setup step has exactly 2 sub-actions: settle + road.
    for step in setup_steps:
        assert len(step.actions) == 2
        assert step.actions[0].kind == "BuildSettlement"
        assert step.actions[1].kind == "BuildRoad"

    # Main phase has at least 1 step (the game runs SOME turns).
    main_steps = [s for s in replay.steps if s.kind == "main"]
    assert len(main_steps) >= 1

    # ----- round-trip through save/load -----
    out_path = tmp_path / "replay.json"
    save_replay(replay, out_path)
    loaded = load_replay(out_path, strict=True)
    assert loaded.metadata.total_steps == replay.metadata.total_steps
    assert loaded.schema_version == replay.schema_version
    assert len(loaded.steps) == len(replay.steps)


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
