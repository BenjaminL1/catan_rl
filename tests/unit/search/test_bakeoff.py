"""Bake-off gate LOGIC (T015, contract C6).

The expensive loader + full-game path (``evaluate_search_vs_policy`` ->
``build_actor`` -> games) is exercised by the actual gate run (T016) and by
``test_eval_search`` / the integration smoke; here we monkeypatch the eval to
canned results and pin the gate's decision logic: pass/fail thresholds,
confirm-only-on-quick-pass, disjoint confirm seed, and the verdict-dict shape.
"""

from __future__ import annotations

from catan_rl.eval.harness import EvalMatchupResult
from catan_rl.eval.wilson import wilson_interval
from catan_rl.search import bakeoff


def _result(wins: int, n: int) -> EvalMatchupResult:
    return EvalMatchupResult(
        opponent_type="snapshot",
        games=(),
        wins=wins,
        n=n,
        ci=wilson_interval(wins=wins, n=n),
        opponent_ref="x",
    )


def test_fails_when_quick_lb_below_half_and_skips_confirm(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[int] = []

    def fake(cfg, search_ckpt, opp_ckpt, *, n_games, seed, **kw):  # type: ignore[no-untyped-def]
        calls.append(n_games)
        return _result(wins=n_games // 2, n=n_games)  # WR ~ 0.50 -> LB <= 0.50

    monkeypatch.setattr(bakeoff, "evaluate_search_vs_policy", fake)
    verdict = bakeoff.run_bakeoff("ckpt.pt", n_quick=200, n_confirm=500)
    assert verdict["passed"] is False
    assert "<= 0.50" in verdict["failure_mode"]
    assert calls == [200]  # confirm must NOT run when quick fails


def test_passes_when_both_clear(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake(cfg, search_ckpt, opp_ckpt, *, n_games, seed, **kw):  # type: ignore[no-untyped-def]
        return _result(wins=int(n_games * 0.62), n=n_games)  # ~0.62 -> LB > 0.50

    monkeypatch.setattr(bakeoff, "evaluate_search_vs_policy", fake)
    verdict = bakeoff.run_bakeoff("ckpt.pt", n_quick=200, n_confirm=500)
    assert verdict["passed"] is True
    assert verdict["failure_mode"] is None
    assert "wr_quick" in verdict and "wr_confirm" in verdict
    assert verdict["n_quick"] == 200 and verdict["n_confirm"] == 500


def test_fails_when_confirm_regresses(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake(cfg, search_ckpt, opp_ckpt, *, n_games, seed, **kw):  # type: ignore[no-untyped-def]
        wr = 0.62 if n_games == 200 else 0.50  # quick clears, confirm regresses
        return _result(wins=round(n_games * wr), n=n_games)

    monkeypatch.setattr(bakeoff, "evaluate_search_vs_policy", fake)
    verdict = bakeoff.run_bakeoff("ckpt.pt", n_quick=200, n_confirm=500)
    assert verdict["passed"] is False
    assert "confirm" in verdict["failure_mode"]


def test_confirm_uses_disjoint_seed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seeds: list[int] = []

    def fake(cfg, search_ckpt, opp_ckpt, *, n_games, seed, **kw):  # type: ignore[no-untyped-def]
        seeds.append(seed)
        return _result(wins=int(n_games * 0.62), n=n_games)

    monkeypatch.setattr(bakeoff, "evaluate_search_vs_policy", fake)
    bakeoff.run_bakeoff("ckpt.pt", seed=0, n_quick=200, n_confirm=500)
    assert len(seeds) == 2
    assert seeds[0] != seeds[1]  # confirm games disjoint from quick
