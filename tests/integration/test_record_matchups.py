"""Phase 4 — parametrized smoke over the full 8-matchup matrix.

Coverage:

* 6 supported matchups: ``(random, random)``, ``(random, heuristic)``,
  ``(random, policy)``, ``(heuristic, policy)``, ``(policy, random)``,
  ``(policy, heuristic)``.
* 2 heuristic-as-agent matchups: ``(heuristic, random)`` and
  ``(heuristic, heuristic)`` — assert ``NotImplementedError``. The
  recorder will gain heuristic-as-agent support in a follow-up phase;
  removing these xfail-style asserts is the gating step.
* 1 unsupported: ``(policy, policy)`` — assert ``NotImplementedError``.
  Deferred to Phase 8's snapshot-opponent path.

Policy matchups gate on ``runs/train/sanity_phase10_20260603_231643/checkpoints/ckpt_000000099.pt``.
If the checkpoint is missing (CI without the training artifact),
those parametrize cases are ``pytest.skip``-ed at fixture build time.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from catan_rl.replay import load_replay, record_game, save_replay
from catan_rl.replay.player_factory import PlayerSpec as RecorderPlayerSpec

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SANITY_CKPT = (
    _REPO_ROOT / "runs/train/sanity_phase10_20260603_231643/checkpoints/ckpt_000000099.pt"
)


def _ckpt_or_skip() -> str:
    if not _SANITY_CKPT.exists():
        pytest.skip(
            "sanity_phase10 checkpoint missing (not committed to repo); "
            "policy matchups are skipped in CI. Expected at: "
            f"{_SANITY_CKPT.relative_to(_REPO_ROOT)}"
        )
    return str(_SANITY_CKPT)


@pytest.mark.parametrize(
    "kind_a,kind_b,uses_ckpt",
    [
        ("random", "random", False),
        ("random", "heuristic", False),
        ("random", "policy", True),
        ("heuristic", "policy", True),
        ("policy", "random", True),
        ("policy", "heuristic", True),
    ],
)
@pytest.mark.parametrize("seed", [42, 7, 11])
def test_supported_matchups_produce_valid_replay(
    kind_a: str, kind_b: str, uses_ckpt: bool, seed: int, tmp_path: Path
) -> None:
    ckpt_a = _ckpt_or_skip() if kind_a == "policy" else None
    ckpt_b = _ckpt_or_skip() if kind_b == "policy" else None
    spec_a = RecorderPlayerSpec(kind=kind_a, ckpt_path=ckpt_a)  # type: ignore[arg-type]
    spec_b = RecorderPlayerSpec(kind=kind_b, ckpt_path=ckpt_b)  # type: ignore[arg-type]

    replay = record_game(
        spec_a,
        spec_b,
        seed=seed,
        max_turns=60,  # short cap — smoke tests are speed-sensitive
    )

    # Core schema invariants.
    assert replay.schema_version >= 1
    assert replay.metadata.player_a.kind == kind_a
    assert replay.metadata.player_b.kind == kind_b
    assert replay.metadata.partial is False
    assert replay.metadata.total_steps == len(replay.steps)

    # Exactly 4 setup steps in snake-draft order.
    setup_steps = [s for s in replay.steps if s.kind == "setup"]
    assert len(setup_steps) == 4
    actors = [s.actor for s in setup_steps]
    assert actors in (
        ["player_a", "player_b", "player_b", "player_a"],  # agent_seat=0
        ["player_b", "player_a", "player_a", "player_b"],  # agent_seat=1
    )

    # Each setup step has exactly 2 sub-actions: settle then road.
    for step in setup_steps:
        assert len(step.actions) == 2
        assert step.actions[0].kind == "BuildSettlement"
        assert step.actions[1].kind == "BuildRoad"

    # Main phase has at least one step (game runs SOME turns even on
    # truncation).
    main_steps = [s for s in replay.steps if s.kind in ("main", "terminal")]
    assert len(main_steps) >= 1
    # Per-actor partitioning: both actors must appear, AND neither side
    # should be silently dropped. With max_turns=60 the empirical
    # floor is ~150 steps; >=20 covers truncation-on-first-turn edge
    # cases. The balance check catches "one actor's events folded
    # into the other's ReplayStep" regressions (reviewer HIGH-1
    # from Phase 2d, also asserted in test_record_smoke.py).
    if len(main_steps) >= 20:
        counts: Counter[str] = Counter(s.actor for s in main_steps)
        assert set(counts) == {"player_a", "player_b"}
        assert min(counts.values()) >= len(main_steps) // 5

    # Winner/seat XOR + final_vp consistency.
    has_winner = replay.metadata.winner is not None
    has_seat = replay.metadata.winner_seat is not None
    assert has_winner == has_seat
    a_vp, b_vp = replay.metadata.final_vp
    if has_winner:
        winner_vp = a_vp if replay.metadata.winner_seat == 0 else b_vp
        assert winner_vp >= 15  # 1v1 Colonist VP cap

    # JSON round-trip.
    out = tmp_path / "r.json"
    save_replay(replay, out)
    loaded = load_replay(out, strict=True)
    assert loaded.metadata.total_steps == replay.metadata.total_steps
    assert len(loaded.steps) == len(replay.steps)
    assert loaded.metadata.player_a.kind == kind_a
    assert loaded.metadata.player_b.kind == kind_b


def test_recorder_resolves_winner_at_high_turn_cap(tmp_path: Path) -> None:
    """Drive a long-enough game that a winner emerges, so the
    recorder's terminal-step + winner_seat + final_vp resolution path
    is actually exercised. At max_turns=60 the smoke truncates;
    raising the cap to 1000 with a random vs random matchup at
    seed=3 reliably terminates on a 15-VP win. We pick this seed
    after smoke-testing several seeds offline.

    Asserts:
    * Terminal step's kind == 'terminal'.
    * winner is not None.
    * winner_seat in {0, 1} and matches the named winner via
      ``Metadata.player_{a,b}.seat_index``.
    * final_vp[winner_seat] >= 15 (1v1 Colonist cap).
    """
    replay = record_game(
        RecorderPlayerSpec(kind="random", ckpt_path=None),
        RecorderPlayerSpec(kind="random", ckpt_path=None),
        seed=3,
        max_turns=1000,
    )
    # If this assertion ever fires, the seed needs re-rolling. A
    # deterministic-termination test is the right approach here since
    # we want to actually exercise the winner code path.
    assert replay.metadata.winner is not None, (
        "no winner at seed=3 / max_turns=1000 — re-roll the seed "
        "(or, if the engine changed StackedDice or grant logic, the "
        "deterministic seed family needs refreshing)."
    )
    assert replay.metadata.winner_seat in (0, 1)
    a_vp, b_vp = replay.metadata.final_vp
    winner_vp = a_vp if replay.metadata.winner_seat == 0 else b_vp
    assert winner_vp >= 15
    # The last ReplayStep must be marked terminal.
    assert replay.steps[-1].kind == "terminal"
    # JSON round-trip preserves the winner.
    out = tmp_path / "win.json"
    save_replay(replay, out)
    loaded = load_replay(out, strict=True)
    assert loaded.metadata.winner == replay.metadata.winner
    assert loaded.metadata.winner_seat == replay.metadata.winner_seat
    assert loaded.metadata.final_vp == replay.metadata.final_vp


@pytest.mark.parametrize(
    "kind_a,kind_b",
    [
        ("heuristic", "random"),
        ("heuristic", "heuristic"),
    ],
)
def test_heuristic_as_agent_matchups_raise(kind_a: str, kind_b: str) -> None:
    # Phase TBD: when heuristic-as-agent lands, DELETE this test
    # (not modify it). The recorder's _resolve_seat_and_opp is the
    # source of the raise; remove that branch + this assertion in
    # the same PR.
    with pytest.raises(NotImplementedError, match="heuristic"):
        record_game(
            RecorderPlayerSpec(kind=kind_a, ckpt_path=None),  # type: ignore[arg-type]
            RecorderPlayerSpec(kind=kind_b, ckpt_path=None),  # type: ignore[arg-type]
            seed=1,
            max_turns=10,
        )


def test_policy_vs_policy_raises() -> None:
    # Phase 8 (snapshot-opponent) will lift this restriction; DELETE
    # this test (not modify it) in the PR that adds the snapshot-opp
    # env path.
    ckpt = _ckpt_or_skip()
    with pytest.raises(NotImplementedError, match=r"policy.*policy"):
        record_game(
            RecorderPlayerSpec(kind="policy", ckpt_path=ckpt),
            RecorderPlayerSpec(kind="policy", ckpt_path=ckpt),
            seed=1,
            max_turns=10,
        )
