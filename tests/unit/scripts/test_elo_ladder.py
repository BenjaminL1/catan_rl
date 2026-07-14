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

_GATE = Path(__file__).resolve().parents[3] / "scripts" / "ladder_gate.py"
_gspec = importlib.util.spec_from_file_location("ladder_gate_module", _GATE)
assert _gspec is not None and _gspec.loader is not None
gate_mod = importlib.util.module_from_spec(_gspec)
sys.modules["ladder_gate_module"] = gate_mod
_gspec.loader.exec_module(gate_mod)


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
    # Audit finding: on these REAL ckpt_524 numbers BT ranks cand ABOVE v8 and the
    # joint-Elo delta is strictly positive — so the gate must rest on the HARDENED
    # clause-1 (min pairwise-Elo delta vs the un-gamed anchors), which is ~-5 here.
    assert ratings["cand"] > ratings["v8_u243"], (
        "sanity: BT does rank the counter above v8 (the trap)"
    )
    boot = elo.bootstrap_elo_ci(matches, names, n_boot=200, candidate="cand", baseline="v8_u243")
    gate = elo.promotion_check(matches, names, ratings, boot, candidate="cand", baseline="v8_u243")
    assert gate["clause1_global_gain_vs_anchors"] is False, (
        "hardened clause-1 must reject a counter that's flat vs the un-gamed anchors"
    )
    assert gate["passed"] is False, "a lateral v8-counter must NOT pass the global promotion gate"


def test_fit_name_universe_covers_matches_absent_from_cached_ladder() -> None:
    # Regression (the v10-vs-v9 crash): the cached ladder predates a rung the candidate
    # is gated against — v9_chain_u424 was added to RUNGS AFTER elo_ladder_transitive.json
    # was last built, so it is not in the cached ratings keys. Deriving the fit's name
    # universe from the cached file ALONE KeyErrors in fit_elo the moment a merged matchup
    # references v9 (or the candidate). The universe must be the UNION of cached names +
    # every participant in every matchup being fit.
    cached_names = ["heuristic", "v6_u1399", "v7_u399"]
    merged = [
        ("v6_u1399", "heuristic", 900, 1000),
        ("v7_u399", "heuristic", 920, 1000),
        ("v7_u399", "v6_u1399", 600, 1000),
        ("v9_chain_u424", "v6_u1399", 451, 600),  # baseline absent from the cached ladder
        ("v9_chain_u424", "v7_u399", 397, 600),
        ("v10_cand", "v9_chain_u424", 335, 600),  # candidate matchup references the absent name
        ("v10_cand", "v6_u1399", 452, 600),
        ("v10_cand", "v7_u399", 432, 600),
        ("v10_cand", "heuristic", 578, 600),
    ]
    names = gate_mod.fit_name_universe(cached_names, merged)
    assert "v9_chain_u424" in names and "v10_cand" in names
    # Would raise KeyError('v9_chain_u424') before the fix; must fit cleanly now.
    ratings = elo.fit_elo(merged, names)
    assert set(ratings) == set(names)
    assert ratings["heuristic"] == 500.0  # pin held


def test_promotion_check_accepts_global_gain() -> None:
    # A GENUINE global step: cand beats v6/v7 by MORE than v8 does (not just v8 h2h).
    names = ["heuristic", "v6_u1399", "v7_u399", "v8_u243", "cand"]
    n = 1000
    matches = [
        ("v6_u1399", "heuristic", 900, n),
        ("v7_u399", "heuristic", 920, n),
        ("v8_u243", "heuristic", 940, n),
        ("cand", "heuristic", 955, n),
        ("v7_u399", "v6_u1399", 600, n),
        ("v8_u243", "v7_u399", 668, n),
        ("v8_u243", "v6_u1399", 700, n),
        ("cand", "v7_u399", 730, n),  # beats v7 by MORE than v8 (0.668)
        ("cand", "v6_u1399", 760, n),  # beats v6 by MORE than v8 (0.700)
        ("cand", "v8_u243", 600, n),  # and beats v8 head-to-head
    ]
    ratings = elo.fit_elo(matches, names)
    boot = elo.bootstrap_elo_ci(matches, names, n_boot=200, candidate="cand", baseline="v8_u243")
    gate = elo.promotion_check(matches, names, ratings, boot, candidate="cand", baseline="v8_u243")
    assert gate["clause1_global_gain_vs_anchors"] is True, (
        "a genuine global gain (beats every un-gamed anchor by more than v8) must pass clause-1"
    )
