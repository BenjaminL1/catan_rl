"""Unit tests for the transitive-ruler additions to scripts/elo_ladder.py.

Pins the load-bearing pure functions (no eval/training): the BT prob helper, the
non-transitivity residual (≈0 on a transitive set, large on a planted A>B>C>A cycle),
bootstrap determinism, and that the 3-clause promotion check REJECTS a lateral
counter (beats v8 head-to-head but flat vs v6/v7 — the ckpt_524 failure mode).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "elo_ladder.py"
_spec = importlib.util.spec_from_file_location("elo_ladder_module", _SCRIPT)
assert _spec is not None and _spec.loader is not None
elo = importlib.util.module_from_spec(_spec)
sys.modules["elo_ladder_module"] = elo
_spec.loader.exec_module(elo)


def test_bt_prob_symmetry() -> None:
    assert abs(elo.bt_prob(600.0, 600.0) - 0.5) < 1e-9
    for ra, rb in [(700.0, 500.0), (500.0, 900.0), (1100.0, 1000.0)]:
        assert abs(elo.bt_prob(ra, rb) + elo.bt_prob(rb, ra) - 1.0) < 1e-9
    # higher rating => higher win prob
    assert elo.bt_prob(700.0, 500.0) > 0.5


def test_residual_near_zero_on_transitive_set() -> None:
    # Ratings imply WRs; generate matches AT those exact predicted WRs -> residual ~ 0.
    ratings = {"heuristic": 500.0, "B": 600.0, "C": 700.0}
    n = 1000
    matches = []
    for a, b in [("B", "heuristic"), ("C", "heuristic"), ("C", "B")]:
        wins = round(elo.bt_prob(ratings[a], ratings[b]) * n)
        matches.append((a, b, wins, n))
    r, _detail, ev = elo.nontransitivity_residual(matches, ratings)
    assert r < 0.01, f"transitive set should have ~0 residual, got {r}"
    assert ev > 0.9


def test_residual_large_on_planted_cycle() -> None:
    # A>B>C>A rock-paper-scissors: each beats the next 0.75. No transitive Elo can fit
    # this, so the residual must be large and at least one pair must exceed 0.10.
    names = ["heuristic", "A", "B", "C"]
    n = 1000
    matches = [
        ("A", "B", 750, n),
        ("B", "C", 750, n),
        ("C", "A", 750, n),
        ("A", "heuristic", 600, n),
        ("B", "heuristic", 600, n),
        ("C", "heuristic", 600, n),
    ]
    ratings = elo.fit_elo(matches, names)
    r, detail, _ev = elo.nontransitivity_residual(matches, ratings)
    assert r > 0.05, f"planted cycle should flag a large residual, got {r}"
    assert any(abs(float(d["resid"])) > 0.10 for d in detail)


def test_bootstrap_determinism() -> None:
    names = ["heuristic", "B", "C"]
    matches = [("B", "heuristic", 620, 1000), ("C", "heuristic", 700, 1000), ("C", "B", 560, 1000)]
    a = elo.bootstrap_elo_ci(matches, names, n_boot=50, rng_seed=7, candidate="C", baseline="B")
    b = elo.bootstrap_elo_ci(matches, names, n_boot=50, rng_seed=7, candidate="C", baseline="B")
    assert a["elo_ci"] == b["elo_ci"]
    assert a["delta_ci"] == b["delta_ci"]


def test_promotion_check_rejects_lateral_counter() -> None:
    # ckpt_524-like: beats v8 head-to-head (0.62) but FLAT vs v7/v6 (same as v8 ~0.66)
    # -> not globally stronger -> the gate must reject (clause 1 delta CI straddles 0).
    names = ["heuristic", "v6_u1399", "v7_u399", "v8_u243", "cand"]
    n = 1000
    matches = [
        ("v6_u1399", "heuristic", 900, n),
        ("v7_u399", "heuristic", 920, n),
        ("v8_u243", "heuristic", 940, n),
        ("cand", "heuristic", 935, n),
        ("v7_u399", "v6_u1399", 600, n),
        ("v8_u243", "v7_u399", 668, n),
        ("cand", "v7_u399", 662, n),  # cand ≈ v8 vs v7 (flat — no global gain)
        ("cand", "v8_u243", 620, n),  # cand beats v8 head-to-head (the counter)
        ("v8_u243", "v6_u1399", 700, n),
        ("cand", "v6_u1399", 695, n),  # cand ≈ v8 vs v6
    ]
    ratings = elo.fit_elo(matches, names)
    boot = elo.bootstrap_elo_ci(matches, names, n_boot=200, candidate="cand", baseline="v8_u243")
    gate = elo.promotion_check(matches, names, ratings, boot, candidate="cand", baseline="v8_u243")
    assert gate["passed"] is False, "a lateral v8-counter must NOT pass the global promotion gate"
