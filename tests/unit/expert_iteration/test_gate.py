"""Pilot gate logic (T009): PASS iff Wilson LB>0.50; search-free; forgetting guard.

Monkeypatches the eval (the loader + full-game eval are covered by the smoke /
test_distill / the harness's own tests) to pin the gate's decision logic cheaply.
"""

from __future__ import annotations

import types

import catan_rl.eval.harness as H
import catan_rl.replay.player_factory as PF
from catan_rl.eval.harness import EvalMatchupResult
from catan_rl.eval.wilson import wilson_interval


def _res(wins: int, n: int) -> EvalMatchupResult:
    return EvalMatchupResult(
        opponent_type="snapshot",
        games=(),
        wins=wins,
        n=n,
        ci=wilson_interval(wins=wins, n=n),
        opponent_ref="v6",
    )


def _fake_harness(heur_wr: float):  # type: ignore[no-untyped-def]
    class _FakeHarness:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, champ: object) -> object:
            return types.SimpleNamespace(
                results=[types.SimpleNamespace(wr=heur_wr, rules_violations=())]
            )

    return _FakeHarness


def _patch(  # type: ignore[no-untyped-def]
    monkeypatch, eval_fn, heur_wr: float = 0.9, champ_epoch: str = "R0"
) -> None:
    monkeypatch.setattr(PF, "build_actor", lambda spec, **k: types.SimpleNamespace(policy=object()))
    # The gate reads the CHAMPION's epoch off its own checkpoint (D8); the
    # stub keeps these decision-logic tests off disk.
    monkeypatch.setattr(H, "checkpoint_ruleset", lambda path: champ_epoch)
    monkeypatch.setattr(H, "evaluate_policy_vs_policy", eval_fn)
    monkeypatch.setattr(H, "EvalHarness", _fake_harness(heur_wr))


def test_gate_fails_when_quick_lb_below_half(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[int] = []

    def fake(champ, opp, *, n_games, seed, device, ruleset):  # type: ignore[no-untyped-def]
        calls.append(n_games)
        return _res(n_games // 2, n_games)  # WR ~ 0.50

    _patch(monkeypatch, fake)
    from catan_rl.expert_iteration.gate import run_gate

    verdict = run_gate("d.pt", "v6.pt", n_quick=200, n_confirm=500)
    assert verdict["passed"] is False
    assert "<= 0.50" in verdict["failure_mode"]
    assert calls == [200]  # confirm NOT run when quick fails


def test_gate_passes_when_distilled_beats_v6(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake(champ, opp, *, n_games, seed, device, ruleset):  # type: ignore[no-untyped-def]
        return _res(int(n_games * 0.62), n_games)  # WR ~ 0.62 -> LB > 0.50

    _patch(monkeypatch, fake)
    from catan_rl.expert_iteration.gate import run_gate

    verdict = run_gate("d.pt", "v6.pt", n_quick=200, n_confirm=500)
    assert verdict["passed"] is True
    assert verdict["failure_mode"] is None
    assert verdict["wr_vs_heuristic"] == 0.9  # forgetting guard reported (>= floor)


def test_gate_fails_on_catastrophic_forgetting(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Beats v6 (WR 0.62) but COLLAPSED vs the heuristic (WR 0.40 < floor) -> FAIL.
    def fake(champ, opp, *, n_games, seed, device, ruleset):  # type: ignore[no-untyped-def]
        return _res(int(n_games * 0.62), n_games)

    _patch(monkeypatch, fake, heur_wr=0.40)
    from catan_rl.expert_iteration.gate import run_gate

    verdict = run_gate("d.pt", "v6.pt", n_quick=200, n_confirm=500)
    assert verdict["passed"] is False
    assert "forgetting" in verdict["failure_mode"]


def test_gate_confirm_uses_disjoint_seed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seeds: list[int] = []

    def fake(champ, opp, *, n_games, seed, device, ruleset):  # type: ignore[no-untyped-def]
        seeds.append(seed)
        return _res(int(n_games * 0.62), n_games)

    _patch(monkeypatch, fake)
    from catan_rl.expert_iteration.gate import run_gate

    run_gate("d.pt", "v6.pt", n_quick=200, n_confirm=500, seed=0)
    assert len(seeds) == 2 and seeds[0] != seeds[1]  # confirm games disjoint from quick


def test_gate_seats_the_champion_under_its_own_epoch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """D8: the distilled seat plays the epoch ITS checkpoint was trained under.

    Left to the ``evaluate_policy_vs_policy`` default it would silently take
    ``R0`` — an R1 champion judged under rules it was not trained for, or a
    refusal on a legitimate same-epoch gate.
    """
    epochs: list[str] = []

    def fake(champ, opp, *, n_games, seed, device, ruleset):  # type: ignore[no-untyped-def]
        epochs.append(ruleset)
        return _res(int(n_games * 0.62), n_games)

    _patch(monkeypatch, fake, champ_epoch="R1")
    from catan_rl.expert_iteration.gate import run_gate

    run_gate("d.pt", "v6.pt", n_quick=200, n_confirm=500)
    assert epochs == ["R1", "R1"]  # quick + confirm both seated under R1
