"""D4 paired per-position gate evaluator (acceptance criterion 6).

Fixture-scale: the real read happens at >=150 fresh picks. What is pinned here
is the ARITHMETIC and the refusals.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catan_rl.eval.wilson import wilson_interval
from catan_rl.labeling.session import LabelingSession
from catan_rl.labeling.store import load_scenarios
from catan_rl.setup_phase.fit import fit_scorer
from catan_rl.setup_phase.gate import (
    CLEAR_TOP1_BAR,
    CLEAR_TOP1_CI_FLOOR,
    KILL_BAR_PICKS,
    MIN_CLEAR_PICKS,
    GateError,
    clarity_report,
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


PICKS_PER_ARM = 12
"""Boards x 4 per arm. Deliberately > ``MIN_CLEAR_PICKS`` so the ``clear``
strictness bar is MEASURED in the fixture — a fixture below that threshold can
only ever exercise the "unmeasured" branch, which is how a bar stops being
tested."""


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
        for idx in range(PICKS_PER_ARM):
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
        assert len(picks) == 2 * PICKS_PER_ARM
        assert all(reveal_arm(p, meta) in ("reveal", "no_reveal") for p in picks)
        assert all(scorer_version_of(p, meta) == "v1" for p in picks)

    def test_no_reveal_rows_are_found_by_manifest_join(self, exam) -> None:
        tmp_path, rows, _scorer = exam
        meta = session_metadata(tmp_path)
        no_reveal = [p for p in fresh_exam_picks(rows, meta) if reveal_arm(p, meta) == "no_reveal"]
        assert len(no_reveal) == PICKS_PER_ARM
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
        # feature_version v3 merged ``adjacency_block`` into the margin, so
        # the published denial weight is one identified number, not two
        # collinear halves.
        assert set(report["relational_weights"]["v1"]) == {
            "opponent_new_resources",
            "opponent_best_margin",
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
        # Reported at TOP LEVEL, not as a pass clause: it is a reason string,
        # and ``pass_clauses`` must stay all-boolean.
        assert report["clear_top1_bar_status"] == "unmeasured"
        assert "clear_top1_bar_status" not in report["pass_clauses"]

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

    def test_the_pass_clauses_are_exactly_the_pre_registered_ones(self, exam) -> None:
        """D4 v2 names four PASS clauses. Picks-2-4 is NOT one of them.

        The amended spec lists one primary metric (overall paired
        log-probability, CI lower bound > 0 on >=150 fresh picks), the ``clear``
        strictness bar, and D3's >=20% no-reveal control; the picks-2-4
        comparison is the KILL bar's metric. Carrying it as a PASS clause makes
        the gate stricter than the one that was pre-registered, which is the
        exact drift pre-registration exists to prevent — so the clause SET is
        pinned, not just the individual clauses.

        The set is also pinned as ALL-BOOLEAN. The clear bar's three-valued
        reason string used to live in here; ``"unmeasured"`` is truthy, so a
        future reader summarising the verdict as ``all(pass_clauses.values())``
        would have read a failed-closed bar as a satisfied one.
        """
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
        assert set(report["pass_clauses"]) == {
            "enough_picks",
            "anchoring_control",
            "overall_log_prob_ci_lower_gt_0",
            "clear_top1_bar",
        }
        assert all(isinstance(v, bool) for v in report["pass_clauses"].values())
        assert report["passes"] is all(report["pass_clauses"].values())
        # The reason string is still reported — one level up, where a truthy
        # non-boolean cannot be mistaken for a satisfied clause.
        assert report["clear_top1_bar_status"] in (
            "satisfied",
            "below_bar",
            "unmeasured",
        )
        # The picks-2-4 delta is still REPORTED — under the kill bar, where it
        # belongs — so nothing was lost by dropping the clause.
        assert report["kill_bar"]["metric"] == "picks_2_4_paired_log_probability"
        assert isinstance(report["kill_bar"]["delta_gt_0"], bool)
        assert report["picks_2_4"]["paired"] is not None

    def test_the_kill_bar_counter_and_metric_share_one_denominator(self, exam) -> None:
        """The mixed-denominator bug: counting all fresh picks while grading the
        gate subset would fire the kill bar off picks the metric never saw."""
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
        kill = report["kill_bar"]
        assert kill["subset"] == report["gate_subset"]
        assert kill["cumulative_fresh_picks"] == report["n_gate_picks"]
        assert kill["n_all_fresh_picks"] == report["n_fresh_picks"]
        # Both n's are reported explicitly, and the picks-2-4 n is the subset
        # the metric is actually computed over.
        assert kill["n_picks_2_4"] == report["picks_2_4"]["n"]
        assert kill["n_picks_2_4"] < kill["cumulative_fresh_picks"]
        assert kill["reached"] is (kill["cumulative_fresh_picks"] >= KILL_BAR_PICKS)

    def test_divergent_arms_fall_back_to_the_no_reveal_picks_alone(self, exam) -> None:
        """D3: "only no-reveal picks count for the gate until understood".

        Disjoint per-arm CIs mean the reveals may simply be training the owner,
        so the verdict is recomputed on the control arm rather than annotated.
        The baseline is rigged per ARM — decisively beaten in the reveal arm,
        decisively beating in the no-reveal arm — which is what makes the two
        arms' paired intervals disjoint.
        """
        tmp_path, rows, scorer = exam
        meta = session_metadata(tmp_path)
        picks = fresh_exam_picks(rows, meta)
        baseline = [
            PickGrade(
                log_prob=(-40.0 if reveal_arm(p, meta) == "reveal" else -0.001),
                top1=-1,
                agree=False,
                in_top3=False,
                margin=0.0,
                rank=99,
            )
            for p in picks
        ]
        report = evaluate_gate(
            rows,
            scorers_by_version={"v1": scorer},
            baseline_grades=baseline,
            session_meta=meta,
            min_fresh_picks=1,
        )
        assert report["anchoring_control"]["arms_divergent"] is True
        assert report["gate_subset"] == "no_reveal_only"
        assert report["anchoring_control"]["gate_subset"] == "no_reveal_only"
        assert report["n_gate_picks"] == report["arms"]["no_reveal"]["n"] == PICKS_PER_ARM
        assert report["n_fresh_picks"] == 2 * PICKS_PER_ARM
        # ...and the verdict is now read on the control arm, where the rigged
        # baseline WINS, so the reveal arm's blowout cannot carry the gate.
        assert report["overall"]["paired"]["delta"] < 0.0
        assert report["pass_clauses"]["overall_log_prob_ci_lower_gt_0"] is False
        assert report["passes"] is False

    def test_unknown_scorer_version_raises(self, exam) -> None:
        tmp_path, rows, scorer = exam
        with pytest.raises(GateError, match="scorer_version"):
            evaluate_gate(
                rows,
                scorers_by_version={"v9": scorer},
                baseline_grades=_flat_grades(2 * PICKS_PER_ARM),
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


class TestClearStrictnessBar:
    """The one PASS clause whose arithmetic is not obvious from the report.

    Estimator ratified 2026-08-21: the subset rate must clear the bar on its
    POINT ESTIMATE (>= 0.70) *and* on its Wilson LOWER BOUND (>= 0.50), at
    >= ``MIN_CLEAR_PICKS`` picks. The four tests below are one fixture per
    corner of that statement — pass, fail-on-the-lower-bound,
    fail-on-the-point-estimate, and unmeasured — so no single clause can be
    deleted without a red test.
    """

    @staticmethod
    def _picks(n_clear: int) -> list[dict]:
        return [{"scenario_id": f"s{i}", "pick_clarity": "clear"} for i in range(n_clear)]

    @staticmethod
    def _grades(n: int, n_agree: int) -> list[PickGrade]:
        return [
            PickGrade(
                log_prob=-1.0,
                top1=0,
                agree=i < n_agree,
                in_top3=True,
                margin=0.1,
                rank=1 if i < n_agree else 2,
            )
            for i in range(n)
        ]

    def test_a_high_point_estimate_still_fails_on_the_confidence_floor(self) -> None:
        """8 of 10: point 0.80 (over the bar), lower bound 0.4902 (under 0.50).

        The arithmetic, checkable by hand from the Wilson formula at
        alpha=0.05: ``wilson_interval(wins=8, n=10)`` = point 0.8000,
        [0.4902, 0.9433]. So the point-estimate clause passes and the
        confidence clause does not — and the bar as a whole fails. This is the
        case the two-clause estimator exists to catch: a rate that looks like a
        pass on ten picks while still overlapping a coin-flip scorer.
        """
        ci = wilson_interval(wins=8, n=10)
        assert ci.point == pytest.approx(0.80)
        assert ci.lower == pytest.approx(0.4902, abs=5e-5)
        assert ci.point >= CLEAR_TOP1_BAR  # clause 1 passes...
        assert ci.lower < CLEAR_TOP1_CI_FLOOR  # ...clause 2 does not

        report = clarity_report(
            self._picks(10), self._grades(10, 8), self._grades(10, 0), alpha=0.05
        )
        clear = report["clear"]
        assert clear["bar_read_on"] == "point_estimate_and_wilson_lower_bound"
        assert clear["bar"] == CLEAR_TOP1_BAR
        assert clear["ci_floor"] == CLEAR_TOP1_CI_FLOOR
        assert clear["scorer"]["rate"] == pytest.approx(ci.point)
        assert clear["status"] == "below_bar"
        assert clear["satisfied"] is False

    def test_a_tight_confidence_interval_still_fails_on_the_point_estimate(self) -> None:
        """65 of 100: lower bound 0.5525 (over the floor), point 0.65 (under
        the bar).

        The mirror image of the 8-of-10 case, and the reason the point estimate
        cannot be dropped in favour of "a lower bound over 0.50": a scorer can
        be measured precisely and still be measured BELOW 70%. Wilson at
        alpha=0.05 gives [0.5525, 0.7364] here.
        """
        ci = wilson_interval(wins=65, n=100)
        assert ci.point == pytest.approx(0.65)
        assert ci.lower == pytest.approx(0.5525, abs=5e-5)
        assert ci.lower >= CLEAR_TOP1_CI_FLOOR  # clause 2 passes...
        assert ci.point < CLEAR_TOP1_BAR  # ...clause 1 does not

        report = clarity_report(
            self._picks(100), self._grades(100, 65), self._grades(100, 0), alpha=0.05
        )
        clear = report["clear"]
        assert clear["scorer"]["rate"] == pytest.approx(ci.point)
        assert clear["scorer"]["ci_lower"] == pytest.approx(ci.lower)
        assert clear["status"] == "below_bar"
        assert clear["satisfied"] is False

    def test_a_unanimous_measured_subset_clears_it(self) -> None:
        """10 of 10: point 1.0000, Wilson [0.7225, 1.0]. Both clauses pass.

        Worth pinning as the PASS fixture precisely because it is the tightest
        one available at ``MIN_CLEAR_PICKS``: under the superseded
        lower-bound-at-0.70 reading this was the ONLY passing record at n=10,
        clearing by 0.0225 — which is what made that reading a bar on 100%
        agreement rather than on 70%.
        """
        ci = wilson_interval(wins=10, n=10)
        assert ci.point == pytest.approx(1.0)
        assert ci.lower == pytest.approx(0.7225, abs=5e-5)
        report = clarity_report(
            self._picks(10), self._grades(10, 10), self._grades(10, 0), alpha=0.05
        )
        assert report["clear"]["status"] == "satisfied"
        assert report["clear"]["satisfied"] is True

    def test_below_the_minimum_it_is_unmeasured_and_never_satisfied(self) -> None:
        """A perfect record on too few picks is "not yet measurable", not a pass.

        9 of 9 would clear BOTH ratified clauses on their arithmetic alone —
        point 1.0000 and a Wilson lower bound of 0.7009 — so nothing but the
        subset-size floor stops it. That is the floor's whole job: at n < 10 one
        pick moves the point estimate by more than 10 percentage points, and a
        clean small subset is evidence of very little.
        """
        n = MIN_CLEAR_PICKS - 1
        report = clarity_report(self._picks(n), self._grades(n, n), self._grades(n, 0), alpha=0.05)
        clear = report["clear"]
        assert clear["scorer"]["n"] == n
        assert clear["scorer"]["rate"] == 1.0
        ci = wilson_interval(wins=n, n=n)
        assert ci.point >= CLEAR_TOP1_BAR and ci.lower >= CLEAR_TOP1_CI_FLOOR
        assert clear["status"] == "unmeasured"
        assert clear["satisfied"] is False
        assert clear["min_picks"] == MIN_CLEAR_PICKS == 10
