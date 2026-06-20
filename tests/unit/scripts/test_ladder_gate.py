"""Unit tests for scripts/ladder_gate.py — the parametric fold-back gate.

Pins the pure merge logic (cached ladder + candidate matchups + high-n overrides) and an
end-to-end gate check on synthetic data: a genuine global gain FOLDS IN, a lateral counter
(beats the baseline head-to-head but flat vs the un-gamed anchors) does NOT.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_GATE = Path(__file__).resolve().parents[3] / "scripts" / "ladder_gate.py"
_spec = importlib.util.spec_from_file_location("ladder_gate_module", _GATE)
assert _spec is not None and _spec.loader is not None
lg = importlib.util.module_from_spec(_spec)
sys.modules["ladder_gate_module"] = lg
_spec.loader.exec_module(lg)
elo = lg.elo  # the wholesale-reused elo_ladder module


def test_build_merged_adds_candidate_and_preserves_ladder() -> None:
    ladder = [
        {"a": "v8_u243", "b": "v7_u399", "wins_a": 1243, "n": 2000},
        {"a": "v8_u243", "b": "v6_u1399", "wins_a": 1419, "n": 2000},
    ]
    cand = [("expl", "v8_u243", 1100, 2000), ("expl", "v7_u399", 1300, 2000)]
    merged = lg.build_merged_matches(ladder, cand)
    # all ladder pairs preserved (as tuples) + both candidate pairs appended
    assert ("v8_u243", "v7_u399", 1243, 2000) in merged
    assert ("expl", "v8_u243", 1100, 2000) in merged
    assert len(merged) == 4


def test_build_merged_applies_overrides() -> None:
    ladder = [{"a": "v8_u243", "b": "v6_u1399", "wins_a": 1300, "n": 2000}]  # n=600-era value
    override = {frozenset(("v8_u243", "v6_u1399")): ("v8_u243", "v6_u1399", 1419, 2000)}
    merged = lg.build_merged_matches(ladder, [], overrides=override)
    assert merged == [("v8_u243", "v6_u1399", 1419, 2000)]  # replaced, not duplicated


def test_kind_and_path_engine_vs_policy() -> None:
    assert lg._kind_and_path("heuristic") == ("heuristic", None)
    assert lg._kind_and_path("random") == ("random", None)
    kind, _path = lg._kind_and_path("v8_u243")
    assert kind == "policy"


def test_gate_folds_in_global_gain_and_rejects_lateral_counter() -> None:
    # Synthetic ladder: heuristic(pin) < v6 < v7 < v8, all transitive.
    n = 1000
    base_ladder = [
        {"a": "v6_u1399", "b": "heuristic", "wins_a": 900, "n": n},
        {"a": "v7_u399", "b": "heuristic", "wins_a": 920, "n": n},
        {"a": "v8_u243", "b": "heuristic", "wins_a": 940, "n": n},
        {"a": "v7_u399", "b": "v6_u1399", "wins_a": 600, "n": n},
        {"a": "v8_u243", "b": "v7_u399", "wins_a": 660, "n": n},
        {"a": "v8_u243", "b": "v6_u1399", "wins_a": 700, "n": n},
    ]
    names0 = ["heuristic", "v6_u1399", "v7_u399", "v8_u243"]
    anchors = ("v6_u1399", "v7_u399")

    def run_gate(cand_matches: list[tuple[str, str, int, int]]) -> dict:
        names = [*names0, "cand"]
        merged = lg.build_merged_matches(base_ladder, cand_matches)
        ratings = elo.fit_elo(merged, names)
        boot = elo.bootstrap_elo_ci(merged, names, n_boot=200, candidate="cand", baseline="v8_u243")
        return elo.promotion_check(
            merged, names, ratings, boot, candidate="cand", baseline="v8_u243", anchors=anchors
        )

    # GLOBAL gain: beats v6/v7 by MORE than v8 does, and beats v8 head-to-head.
    win = run_gate(
        [
            ("cand", "heuristic", 955, n),
            ("cand", "v7_u399", 730, n),
            ("cand", "v6_u1399", 760, n),
            ("cand", "v8_u243", 600, n),
        ]
    )
    assert win["clause1_global_gain_vs_anchors"] is True
    assert win["passed"] is True

    # LATERAL counter: beats v8 head-to-head but FLAT vs the un-gamed anchors (≈ v8's WRs).
    lat = run_gate(
        [
            ("cand", "heuristic", 935, n),
            ("cand", "v7_u399", 662, n),  # ≈ v8's 0.66 vs v7
            ("cand", "v6_u1399", 695, n),  # ≈ v8's 0.70 vs v6
            ("cand", "v8_u243", 620, n),  # beats v8 head-to-head
        ]
    )
    assert lat["clause1_global_gain_vs_anchors"] is False
    assert lat["passed"] is False


def test_validate_inputs_rejects_bad_candidate(tmp_path: Path) -> None:
    led = tmp_path / "ladder.json"
    led.write_text("{}")
    with pytest.raises(SystemExit, match="candidate-ckpt"):
        lg.validate_inputs(str(tmp_path / "nope.pt"), ["v8_u243"], str(led), lg._DEFAULT_REVERIFY)


def test_validate_inputs_rejects_unknown_opponent(tmp_path: Path) -> None:
    ckpt = tmp_path / "cand.pt"
    ckpt.write_text("x")
    led = tmp_path / "ladder.json"
    led.write_text("{}")
    with pytest.raises(SystemExit, match="unknown opponent"):
        lg.validate_inputs(str(ckpt), ["v8_invalid"], str(led), lg._DEFAULT_REVERIFY)


def test_validate_inputs_rejects_missing_ladder_json(tmp_path: Path) -> None:
    ckpt = tmp_path / "cand.pt"
    ckpt.write_text("x")
    absent = str(tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="ladder-json"):
        lg.validate_inputs(str(ckpt), ["v8_u243"], absent, lg._DEFAULT_REVERIFY)


def test_validate_inputs_accepts_valid(tmp_path: Path) -> None:
    ckpt = tmp_path / "cand.pt"
    ckpt.write_text("x")
    led = tmp_path / "ladder.json"
    led.write_text("{}")
    # heuristic is engine-driven (no path) and a real rung name -> must pass
    lg.validate_inputs(str(ckpt), ["v8_u243", "heuristic"], str(led), lg._DEFAULT_REVERIFY)
