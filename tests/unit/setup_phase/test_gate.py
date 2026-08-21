"""D4 paired per-position gate evaluator (acceptance criterion 6).

Fixture-scale: the real read happens at >=150 fresh picks. What is pinned here
is the ARITHMETIC and the refusals.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.store import load_scenarios
from catan_rl.setup_phase.fit import fit_scorer
from catan_rl.setup_phase.gate import (
    CLEAR_TOP1_BAR,
    KILL_BAR_PICKS,
    GateError,
    evaluate_gate,
    fresh_exam_picks,
    paired_binary_difference,
    paired_mean_difference,
    reveal_arm,
    scorer_version_of,
    session_metadata,
)
from catan_rl.setup_phase.scorer import PickGrade


def _flat_grades(n: int, *, log_prob: float = -5.0, agree: bool = False) -> list[PickGrade]:
    """A stand-in u500 baseline: constant log-prob, never top-1 or top-3.

    Constant so the paired arithmetic is hand-checkable, and DELIBERATELY bad so
    a scorer that is merely well-formed still produces a positive delta — the
    tests below pin the plumbing and the refusals, never a quality claim.
    """
    return [
        PickGrade(log_prob=log_prob, top1=-1, agree=agree, in_top3=agree, margin=0.0, rank=99)
        for _ in range(n)
    ]


def _legal_pair(scenario, *, nth: int = 0) -> tuple[int, int]:
    corners = np.flatnonzero(scenario.legal_settlement_corners)
    settlement = int(corners[min(nth, len(corners) - 1)])
    edges = np.flatnonzero(scenario.compute_legal_road_edges(settlement))
    return settlement, int(edges[0])


class TestPairedArithmetic:
    def test_hand_computed_paired_difference(self) -> None:
        # scorer right on 3/4, baseline right on 1/4; per-pick differences are
        # [1, 1, 0, 0] -> mean 0.5, sd 0.5773..., se 0.2886..., z 1.95996.
        result = paired_binary_difference([True, True, True, False], [False, False, True, False])
        assert result.n == 4
        assert result.rate_a == 0.75
        assert result.rate_b == 0.25
        assert result.delta == pytest.approx(0.5)
        # sd(ddof=1) = sqrt(4*0.25/3) = 1/sqrt(3); se = sd / sqrt(4).
        se = (1.0 / np.sqrt(3)) / np.sqrt(4)
        assert result.ci_lower == pytest.approx(0.5 - 1.959964 * se, rel=1e-5)
        assert result.ci_upper == pytest.approx(0.5 + 1.959964 * se, rel=1e-5)

    def test_identical_graders_give_a_zero_width_interval_at_zero(self) -> None:
        result = paired_binary_difference([True, False, True], [True, False, True])
        assert result.delta == 0.0
        assert result.ci_lower == 0.0 and result.ci_upper == 0.0

    def test_continuous_paired_difference_is_the_same_arithmetic(self) -> None:
        # log-probabilities, not 0/1: differences [1.0, -1.0, 2.0] -> mean 2/3.
        result = paired_mean_difference([-1.0, -3.0, -1.0], [-2.0, -2.0, -3.0])
        assert result.delta == pytest.approx(2.0 / 3.0)
        assert result.ci_lower < result.delta < result.ci_upper

    def test_a_non_finite_score_is_refused_not_averaged(self) -> None:
        with pytest.raises(GateError, match="non-finite"):
            paired_mean_difference([-1.0, float("-inf")], [-1.0, -1.0])

    def test_mismatched_lengths_and_tiny_n_raise(self) -> None:
        with pytest.raises(GateError, match="same length"):
            paired_binary_difference([True], [True, False])
        with pytest.raises(GateError, match="at least 2"):
            paired_binary_difference([True], [False])


@pytest.fixture()
def exam(tmp_path: Path):
    """A reveal session and a no-reveal session over the same scorer version."""
    seed_session = LabelingSession(data_dir=tmp_path, labeler_id="ben", session_seed=5)
    seed_session.start()
    for _ in range(4):
        scenario = seed_session.current_scenario()
        assert scenario is not None
        pick = _legal_pair(scenario)
        seed_session.submit(settlement_vertex=pick[0], road_edge=pick[1])
    seed_session.quit()
    scorer = fit_scorer(
        load_scenarios(tmp_path / "scenarios.jsonl"), version="v1", seed=0, iters=30
    ).scorer

    for mode, seed in (("reveal", 21), ("no_reveal", 22)):
        session = LabelingSession(
            data_dir=tmp_path,
            labeler_id="ben",
            session_seed=seed,
            reveal_mode=mode,
            scorer_version="v1",
        )
        session.start()
        for idx in range(4):
            scenario = session.current_scenario()
            assert scenario is not None
            pick = _legal_pair(scenario)
            fields = (
                {
                    "scorer_version": "v1",
                    "scorer_top1": pick[0],
                    "scorer_rank_of_pick": 1,
                    "agree": True,
                    "reveal_mode": "reveal",
                }
                if mode == "reveal"
                else None
            )
            session.submit(
                settlement_vertex=pick[0],
                road_edge=pick[1],
                scorer_fields=fields,
                # Alternate the owner's D3 clarity tag so BOTH v2 bars (top-1
                # on ``clear``, top-3 on ``close``) have picks under them.
                pick_clarity="clear" if idx % 2 == 0 else "close",
            )
        session.quit()

    rows = load_scenarios(tmp_path / "scenarios.jsonl")
    return tmp_path, rows, scorer


class TestFreshPickSelection:
    def test_pre_scorer_labels_are_not_exam_picks(self, exam) -> None:
        tmp_path, rows, _scorer = exam
        meta = session_metadata(tmp_path)
        picks = fresh_exam_picks(rows, meta)
        # The seed session predates any scorer: it carries the DEFAULT reveal
        # mode but no scorer_version, so it must not enter the exam.
        assert len(picks) == 8
        assert all(reveal_arm(p, meta) in ("reveal", "no_reveal") for p in picks)
        assert all(scorer_version_of(p, meta) == "v1" for p in picks)

    def test_no_reveal_rows_are_found_by_manifest_join(self, exam) -> None:
        tmp_path, rows, _scorer = exam
        meta = session_metadata(tmp_path)
        no_reveal = [p for p in fresh_exam_picks(rows, meta) if reveal_arm(p, meta) == "no_reveal"]
        assert len(no_reveal) == 4
        # ...even though the ROWS carry no reveal fields at all. The version
        # comes back from the manifest join, so D6 grading still works.
        raw = json.loads((tmp_path / "scenarios.jsonl").read_text().splitlines()[-1])
        assert "reveal_mode" not in raw and "scorer_version" not in raw
        assert all(p["scorer_version"] == "v1" for p in no_reveal)


class TestGate:
    def test_report_shape_and_kill_bar(self, exam) -> None:
        tmp_path, rows, scorer = exam
        meta = session_metadata(tmp_path)
        n = len(fresh_exam_picks(rows, meta))
        report = evaluate_gate(
            rows,
            scorers_by_version={"v1": scorer},
            baseline_grades=_flat_grades(n),
            session_meta=meta,
            min_fresh_picks=4,
        )
        assert report["primary_metric"] == "paired_mean_log_probability_of_owner_pick"
        assert report["n_fresh_picks"] == n
        assert set(report["by_position"]) == {"1", "2", "3", "4"}
        assert set(report["arms"]) == {"reveal", "no_reveal"}
        assert report["kill_bar"]["bar"] == KILL_BAR_PICKS
        assert report["kill_bar"]["metric"] == "picks_2_4_paired_log_probability"
        assert report["kill_bar"]["reached"] is False
        assert report["anchoring_control"]["fraction_no_reveal"] == 0.5
        assert report["anchoring_control"]["satisfied"] is True
        assert report["anchoring_control"]["arms_divergent"] is False
        assert report["gate_subset"] == "all_fresh_picks"
        assert report["n_gate_picks"] == n
        assert set(report["clarity"]) == {"clear", "close"}
        assert report["clarity"]["clear"]["bar"] == CLEAR_TOP1_BAR
        assert report["clarity"]["close"]["metric"] == "owner_pick_in_scorer_top3"
        assert set(report["calibration"]) >= {"clear", "close", "note"}
        assert set(report["relational_weights"]["v1"]) == {
            "opponent_new_resources",
            "opponent_best_margin",
            "adjacency_block",
            "scarcity_starve",
        }
        assert json.dumps(report)  # the report is JSON-serialisable

    def test_primary_metric_is_log_probability_not_agreement(self, exam) -> None:
        """A grader that is never top-1 can still beat one on log-prob, and the
        verdict reads the log-prob clause — that IS the amendment."""
        tmp_path, rows, scorer = exam
        meta = session_metadata(tmp_path)
        n = len(fresh_exam_picks(rows, meta))
        report = evaluate_gate(
            rows,
            scorers_by_version={"v1": scorer},
            baseline_grades=_flat_grades(n, log_prob=-50.0),
            session_meta=meta,
            min_fresh_picks=4,
        )
        # The scorer's masked softmax never puts less than e^-50 on a legal
        # vertex, so the paired LOG-PROB difference is decisively positive...
        assert report["overall"]["paired"]["delta"] > 0.0
        assert report["pass_clauses"]["overall_log_prob_ci_lower_gt_0"] is True
        # ...and the superseded top-1 number is reported alongside, not used.
        assert report["overall"]["agreement"] is not None

    def test_below_the_minimum_it_cannot_pass(self, exam) -> None:
        tmp_path, rows, scorer = exam
        meta = session_metadata(tmp_path)
        n = len(fresh_exam_picks(rows, meta))
        report = evaluate_gate(
            rows,
            scorers_by_version={"v1": scorer},
            baseline_grades=_flat_grades(n),
            session_meta=meta,
        )
        assert report["enough_picks"] is False
        assert report["passes"] is False

    def test_an_unsatisfied_anchoring_control_forces_a_fail(self, exam) -> None:
        """D3/D4 are a MUST on both sides: a gate read on 100% reveal-arm picks
        is the precise situation where the reveals may have trained the owner."""
        tmp_path, rows, scorer = exam
        meta = session_metadata(tmp_path)
        reveal_only = [r for r in rows if reveal_arm(r, meta) != "no_reveal"]
        picks = fresh_exam_picks(reveal_only, meta)
        report = evaluate_gate(
            reveal_only,
            scorers_by_version={"v1": scorer},
            baseline_grades=_flat_grades(len(picks), log_prob=-50.0),
            session_meta=meta,
            min_fresh_picks=1,
        )
        assert report["anchoring_control"]["fraction_no_reveal"] == 0.0
        assert report["anchoring_control"]["satisfied"] is False
        assert report["pass_clauses"]["anchoring_control"] is False
        # ...even though every other clause is green.
        assert report["pass_clauses"]["overall_log_prob_ci_lower_gt_0"] is True
        assert report["passes"] is False

    def test_the_clear_bar_fails_closed_with_no_clear_picks(self, exam) -> None:
        tmp_path, rows, scorer = exam
        meta = session_metadata(tmp_path)
        untagged = [dict(r, pick_clarity="close") for r in rows]
        n = len(fresh_exam_picks(untagged, meta))
        report = evaluate_gate(
            untagged,
            scorers_by_version={"v1": scorer},
            baseline_grades=_flat_grades(n, log_prob=-50.0),
            session_meta=meta,
            min_fresh_picks=1,
        )
        assert report["clarity"]["clear"]["scorer"]["n"] == 0
        assert report["clarity"]["clear"]["satisfied"] is False
        assert report["passes"] is False
        # UNMEASURED, not below_bar: the remedy is "tag some picks clear", not
        # "improve the scorer", and a bare False cannot say which.
        assert report["clarity"]["clear"]["status"] == "unmeasured"
        assert report["pass_clauses"]["clear_top1_bar_status"] == "unmeasured"

    def test_a_measured_miss_is_distinguished_from_an_unmeasured_bar(self, exam) -> None:
        tmp_path, rows, scorer = exam
        meta = session_metadata(tmp_path)
        # Every pick tagged ``clear`` — the bar is now measured, and whichever
        # side of it the scorer lands on, the status must not read "unmeasured".
        tagged = [dict(r, pick_clarity="clear") for r in rows]
        n = len(fresh_exam_picks(tagged, meta))
        report = evaluate_gate(
            tagged,
            scorers_by_version={"v1": scorer},
            baseline_grades=_flat_grades(n, log_prob=-50.0),
            session_meta=meta,
            min_fresh_picks=1,
        )
        clear = report["clarity"]["clear"]
        assert clear["scorer"]["n"] == n
        assert clear["status"] in ("satisfied", "below_bar")
        assert clear["satisfied"] is (clear["status"] == "satisfied")

    def test_unknown_scorer_version_raises(self, exam) -> None:
        tmp_path, rows, scorer = exam
        with pytest.raises(GateError, match="scorer_version"):
            evaluate_gate(
                rows,
                scorers_by_version={"v9": scorer},
                baseline_grades=_flat_grades(8),
                session_meta=session_metadata(tmp_path),
                min_fresh_picks=4,
            )

    def test_needs_exactly_one_baseline_source(self, exam) -> None:
        tmp_path, rows, scorer = exam
        with pytest.raises(GateError, match="exactly one"):
            evaluate_gate(
                rows,
                scorers_by_version={"v1": scorer},
                session_meta=session_metadata(tmp_path),
                min_fresh_picks=4,
            )

    def test_baseline_length_mismatch_raises(self, exam) -> None:
        tmp_path, rows, scorer = exam
        with pytest.raises(GateError, match="baseline grades have"):
            evaluate_gate(
                rows,
                scorers_by_version={"v1": scorer},
                baseline_grades=_flat_grades(2),
                session_meta=session_metadata(tmp_path),
                min_fresh_picks=4,
            )
