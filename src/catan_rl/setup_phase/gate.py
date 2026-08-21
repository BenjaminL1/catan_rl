"""The D4 pre-registered forward exam **v2**: tie-aware, paired, per-position.

The exam compares the fitted scorer against the champion (``ptr_v1_u500``) on
the IDENTICAL fresh blind-first picks, over the IDENTICAL legal-vertex mask.

**Why v2 grades distributions, not top-1 agreement.** D0 measured the owner's
own top-1 self-agreement at ~35% (n=20, Wilson [18, 57], the ``free_replay``
estimator of :mod:`catan_rl.labeling.consistency` — reproduce it with
``report_label_consistency.py --estimator free_replay``) — a labeler-noise
ceiling the pilot scorer already sits at. Exact agreement therefore cannot
discriminate: on picks 2-4 the owner's policy has ~3 effective near-ties, and a
grader that answers a genuine 3-way tie with ~1/3 each is RIGHT, while exact
agreement scores it as two-thirds wrong. The amended spec (owner-ratified
2026-08-21) makes the primary metric the **paired mean log-probability of the
owner's pick** — a proper scoring rule, under which honesty about ties is the
winning strategy and confident wrongness is punished.

Design decisions carried straight from the spec:

* **Paired, not two independent rates.** Both graders answer the same picks, so
  the unit of analysis is the per-pick difference. An unpaired comparison throws
  away the variance reduction and needs roughly four times the corpus.
* **Per draft position, always.** u500 already matches the owner ~55% on pick 1
  and 0-18% on picks 2-4. An aggregate number is dominated by pick 1 — the one
  position where there is nothing to learn — so PASS additionally requires the
  scorer to beat u500 on the picks-2-4 subset.
* **Strictness where the owner says so.** Picks the owner tagged ``clear`` are
  not ties, so there the scorer's top-1 MUST match, at a >=70% bar. Picks tagged
  ``close`` are reported by top-3 containment.
* **Agreement is measured against the scorer version LIVE AT LABEL TIME** (D6).
  Refitting on blind-first labels is legitimate; grading an old pick with a
  scorer fitted after it was made is not.
* **The D3 anchoring control GATES.** >=20% of the fresh picks must come from
  ``--no-reveal`` sessions, and if the two arms diverge the gate falls back to
  the no-reveal picks alone. A gate read on 100% reveal-arm picks is the exact
  situation where the reveals may simply have trained the owner.

The **kill bar** is a counter, not a suggestion: at 300 cumulative fresh picks
with the picks-2-4 paired LOG-PROB still not exceeding u500's, the report says
DEAD and the program re-plans.

This module is a driver: it does the arithmetic and the bookkeeping. It runs at
fixture scale in tests; the real read happens at >=150 fresh picks.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from catan_rl.eval.wilson import normal_ppf, wilson_interval
from catan_rl.labeling.store import (
    PICK_CLARITIES,
    PICK_CLARITY_CLEAR,
    PICK_CLARITY_CLOSE,
    PICK_CLARITY_FIELD,
    REVEAL_MODE_NO_REVEAL,
    REVEAL_MODE_REVEAL,
)
from catan_rl.setup_phase.fit import settlement_grades
from catan_rl.setup_phase.scorer import PickGrade, SetupScorer

MIN_FRESH_PICKS: int = 150
"""D4's gate threshold. Below it the gate REFUSES to report a verdict."""

KILL_BAR_PICKS: int = 300
"""D4's kill bar. At or past this many cumulative fresh picks with the
picks-2-4 paired log-prob still not exceeding u500's, the theory-feature
approach is declared dead."""

MIN_NO_REVEAL_FRACTION: float = 0.2
"""D3's anchoring control: at least this fraction of fresh exam picks must come
from ``--no-reveal`` sessions, or the reveals may simply be training the owner
and the gate is reading its own tail."""

CLEAR_TOP1_BAR: float = 0.70
"""D4 v2's strictness bar on picks the owner tagged ``clear``. "Revisable only
upward" — a number that moves down is the bar following the result."""

MIN_CLEAR_PICKS: int = 2
"""How many ``clear``-tagged picks the strictness bar needs before it is
MEASURED rather than merely unsatisfied.

Below this the bar reports ``status = "unmeasured"`` and the gate still fails
closed. That is deliberate but it is also a trap worth naming: the owner tags
via a submit key, so an exam in which nothing was tagged ``clear`` can never
pass no matter how decisively the primary metric clears its bar. The
``clear_top1_bar_status`` reason exists so that NO-GO reads "tag some picks"
instead of "the scorer is bad". The submit keys are bound accordingly — ``S``,
the reflexive one, writes ``close``; asserting ``clear`` costs a deliberate
``B`` — so the ``clear`` subset stays the owner's statement rather than their
habit."""

CLEAR_BAR_SATISFIED: str = "satisfied"
CLEAR_BAR_BELOW_BAR: str = "below_bar"
CLEAR_BAR_UNMEASURED: str = "unmeasured"


class GateError(ValueError):
    """Raised when the gate cannot be computed from the inputs given."""


@dataclass(frozen=True)
class PairedDifference:
    """Mean paired difference of two graders scored on the same picks."""

    n: int
    rate_a: float
    rate_b: float
    delta: float
    ci_lower: float
    ci_upper: float
    alpha: float


def paired_mean_difference(
    a: Sequence[float], b: Sequence[float], *, alpha: float = 0.05
) -> PairedDifference:
    """CI on ``mean(a) - mean(b)`` for two graders scored on the same items.

    Normal CI on the per-item differences (Bessel-corrected SD), which is the
    same arithmetic ``bc.gates.paired_wr_non_inferiority`` uses for paired WR.
    Works for the log-probability primary metric and for the 0/1 secondary
    rates alike — the paired difference of bounded scores is asymptotically
    normal either way, and one implementation is one place to be wrong.
    """
    if len(a) != len(b):
        raise GateError(f"paired inputs must be the same length, got {len(a)} and {len(b)}")
    n = len(a)
    if n < 2:
        raise GateError(f"need at least 2 paired picks, got {n}")
    xs = np.asarray([float(x) for x in a], dtype=np.float64)
    ys = np.asarray([float(y) for y in b], dtype=np.float64)
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise GateError(
            "paired inputs contain a non-finite score: a grader put zero mass on "
            "the owner's pick, which a mean cannot absorb. Grade in log space."
        )
    d = xs - ys
    mean = float(d.mean())
    se = float(d.std(ddof=1)) / float(np.sqrt(n))
    z = normal_ppf(1.0 - alpha / 2.0)
    return PairedDifference(
        n=n,
        rate_a=float(xs.mean()),
        rate_b=float(ys.mean()),
        delta=mean,
        ci_lower=mean - z * se,
        ci_upper=mean + z * se,
        alpha=alpha,
    )


def paired_binary_difference(
    a: Sequence[bool], b: Sequence[bool], *, alpha: float = 0.05
) -> PairedDifference:
    """:func:`paired_mean_difference` over two boolean graders."""
    return paired_mean_difference([float(x) for x in a], [float(y) for y in b], alpha=alpha)


SessionMeta = Mapping[str, Mapping[str, Any]]


def session_metadata(data_dir: Path) -> dict[str, dict[str, Any]]:
    """``session_id -> {"reveal_mode", "scorer_version"}`` from the manifests.

    A ``--no-reveal`` row deliberately carries NO scorer fields of its own
    (acceptance criterion 4) — not even the version stamp — so BOTH the arm and
    the scorer version live at label time are recovered by joining on
    ``session_id``. That is exactly why the manifest records them for both arms.
    """
    out: dict[str, dict[str, Any]] = {}
    sessions = Path(data_dir) / "sessions"
    if not sessions.is_dir():
        return out
    for manifest in sorted(sessions.glob("*/manifest.json")):
        payload = json.loads(manifest.read_text())
        out[str(payload["session_id"])] = {
            "reveal_mode": payload.get("reveal_mode"),
            "scorer_version": payload.get("scorer_version"),
        }
    return out


def reveal_arm(row: Mapping[str, Any], session_meta: SessionMeta) -> str | None:
    """Which D3 arm a row belongs to, or ``None`` if it predates the split."""
    mode = row.get("reveal_mode")
    if mode is not None:
        return str(mode)
    meta = session_meta.get(str(row["session_id"]), {})
    mode = meta.get("reveal_mode")
    return None if mode is None else str(mode)


def scorer_version_of(row: Mapping[str, Any], session_meta: SessionMeta) -> str | None:
    """The scorer version that was LIVE when ``row`` was labeled (D6)."""
    version = row.get("scorer_version")
    if version is not None:
        return str(version)
    meta = session_meta.get(str(row["session_id"]), {})
    version = meta.get("scorer_version")
    return None if version is None else str(version)


def pick_clarity_of(row: Mapping[str, Any]) -> str:
    """The owner's clarity tag, defaulting to ``close`` for untagged rows."""
    value = row.get(PICK_CLARITY_FIELD)
    if value is None:
        return PICK_CLARITY_CLOSE
    text = str(value)
    if text not in PICK_CLARITIES:
        raise GateError(
            f"row {row.get('scenario_id')!r} carries pick_clarity {text!r}, which is "
            f"not one of {PICK_CLARITIES}"
        )
    return text


def fresh_exam_picks(
    rows: Iterable[Mapping[str, Any]], session_meta: SessionMeta | None = None
) -> list[dict[str, Any]]:
    """The blind-first picks created AFTER a scorer shipped.

    A pick counts as fresh iff it is not a replay, belongs to a known D3 arm,
    AND has a resolvable ``scorer_version``. The version requirement is what
    keeps a pre-scorer v3 session out: those rows were made with no scorer in
    the world, so they cannot test one — and a session created before any fit
    still carries the default ``reveal`` mode.

    The resolved version is written back onto the returned copy, so downstream
    grading never has to re-do the manifest join.
    """
    meta = dict(session_meta or {})
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("replay_of") is not None:
            continue
        if reveal_arm(row, meta) not in (REVEAL_MODE_REVEAL, REVEAL_MODE_NO_REVEAL):
            continue
        version = scorer_version_of(row, meta)
        if version is None:
            continue
        copy = dict(row)
        copy["scorer_version"] = version
        out.append(copy)
    return out


# ---------------------------------------------------------------------------
# Subset reporting
# ---------------------------------------------------------------------------
def _paired_block(
    scorer: Sequence[PickGrade], baseline: Sequence[PickGrade], *, alpha: float
) -> dict[str, Any] | None:
    if len(scorer) < 2:
        return None
    return {
        "log_prob": asdict(
            paired_mean_difference(
                [g.log_prob for g in scorer], [g.log_prob for g in baseline], alpha=alpha
            )
        ),
        "agreement": asdict(
            paired_binary_difference(
                [g.agree for g in scorer], [g.agree for g in baseline], alpha=alpha
            )
        ),
    }


def _subset_report(
    label: str,
    scorer: Sequence[PickGrade],
    baseline: Sequence[PickGrade],
    *,
    alpha: float,
) -> dict[str, Any]:
    """One subset's paired block.

    ``paired`` is the PRIMARY (log-probability) difference; ``agreement`` is the
    superseded top-1 metric, kept alongside it for continuity with the
    pre-amendment reports and for the calibration read — never for the verdict.
    """
    block = _paired_block(scorer, baseline, alpha=alpha)
    return {
        "label": label,
        "n": len(scorer),
        "paired": None if block is None else block["log_prob"],
        "agreement": None if block is None else block["agreement"],
    }


def _rate(flags: Sequence[bool], *, alpha: float) -> dict[str, Any]:
    n = len(flags)
    if n == 0:
        return {"n": 0, "n_hits": 0, "rate": None, "ci_lower": None, "ci_upper": None}
    hits = int(sum(1 for f in flags if f))
    ci = wilson_interval(wins=hits, n=n, alpha=alpha)
    return {
        "n": n,
        "n_hits": hits,
        "rate": ci.point,
        "ci_lower": ci.lower,
        "ci_upper": ci.upper,
    }


def _clarity_report(
    picks: Sequence[Mapping[str, Any]],
    scorer: Sequence[PickGrade],
    baseline: Sequence[PickGrade],
    *,
    alpha: float,
) -> dict[str, Any]:
    """D4 v2's clarity-conditioned bars: top-1 on ``clear``, top-3 on ``close``."""
    tags = [pick_clarity_of(p) for p in picks]
    clear = [i for i, t in enumerate(tags) if t == PICK_CLARITY_CLEAR]
    close = [i for i, t in enumerate(tags) if t == PICK_CLARITY_CLOSE]
    clear_rate = _rate([scorer[i].agree for i in clear], alpha=alpha)
    close_rate = _rate([scorer[i].in_top3 for i in close], alpha=alpha)
    measured = bool(clear_rate["n"] >= MIN_CLEAR_PICKS and clear_rate["rate"] is not None)
    satisfied = bool(measured and clear_rate["rate"] >= CLEAR_TOP1_BAR)
    # Fails CLOSED on an empty subset — a bar with no picks under it has not
    # been cleared. But "unmeasured" and "measured and missed" are DIFFERENT
    # failures with different remedies (tag some picks ``clear`` vs improve the
    # scorer), and a bare ``satisfied: False`` cannot tell them apart. The
    # status names which one it is, so a gate report can be acted on.
    status = (
        CLEAR_BAR_SATISFIED
        if satisfied
        else (CLEAR_BAR_BELOW_BAR if measured else CLEAR_BAR_UNMEASURED)
    )
    return {
        PICK_CLARITY_CLEAR: {
            "metric": "scorer_top1_equals_owner",
            "scorer": clear_rate,
            "baseline": _rate([baseline[i].agree for i in clear], alpha=alpha),
            "bar": CLEAR_TOP1_BAR,
            "min_picks": MIN_CLEAR_PICKS,
            "status": status,
            "satisfied": satisfied,
        },
        PICK_CLARITY_CLOSE: {
            "metric": "owner_pick_in_scorer_top3",
            "scorer": close_rate,
            "baseline": _rate([baseline[i].in_top3 for i in close], alpha=alpha),
        },
    }


def _calibration_report(
    picks: Sequence[Mapping[str, Any]], scorer: Sequence[PickGrade]
) -> dict[str, Any]:
    """Scorer confidence (top-1 probability margin) against the owner's tags.

    The two failure modes this maps are named, not inferred after the fact:
    confident where the owner says ``close`` (the scorer is inventing a
    distinction the owner does not make), and unsure where the owner says
    ``clear`` (a feature the owner is using is missing from the design matrix).
    """
    tags = [pick_clarity_of(p) for p in picks]
    out: dict[str, Any] = {}
    for tag in PICK_CLARITIES:
        margins = [scorer[i].margin for i, t in enumerate(tags) if t == tag]
        out[tag] = {
            "n": len(margins),
            "mean_top1_margin": float(np.mean(margins)) if margins else None,
            "median_top1_margin": float(np.median(margins)) if margins else None,
        }
    return out | {
        "note": (
            "top-1 minus top-2 PROBABILITY under the scorer's masked softmax. "
            "High margin on 'close' picks or low margin on 'clear' picks maps a "
            "missing feature; it is a diagnostic, never part of the verdict."
        )
    }


def _cis_disjoint(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    if a is None or b is None:
        return False
    return bool(a["ci_lower"] > b["ci_upper"] or b["ci_lower"] > a["ci_upper"])


def evaluate_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    scorers_by_version: Mapping[str, SetupScorer],
    baseline_grades: Sequence[PickGrade] | None = None,
    baseline_ckpt: Path | None = None,
    session_meta: SessionMeta | None = None,
    alpha: float = 0.05,
    min_fresh_picks: int = MIN_FRESH_PICKS,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the D4 v2 paired per-position exam over the fresh picks in ``rows``.

    ``scorers_by_version`` maps a ``scorer_version`` stamp to the fitted scorer
    that was live when those picks were made. A pick whose stamp is missing from
    the map RAISES — silently grading it with whatever scorer happens to be
    loaded is the coupling D6 exists to prevent.

    Exactly one of ``baseline_grades`` (already-computed u500 per-pick grades,
    used by tests with a stub policy) or ``baseline_ckpt`` (a real checkpoint,
    scored through ``bc.finetune.setup_pick_grades``) must be given. The
    checkpoint path is imported lazily so this module stays importable without
    torch.
    """
    meta = dict(session_meta or {})
    picks = fresh_exam_picks(rows, meta)
    if not picks:
        raise GateError("no fresh exam picks: the corpus has no reveal-mode labels yet")

    missing = sorted({str(p.get("scorer_version")) for p in picks} - set(scorers_by_version))
    if missing:
        raise GateError(
            f"fresh picks carry scorer_version(s) {missing} with no matching fitted "
            f"scorer. Agreement must be measured against the version live at label "
            f"time (D6); grading them with another scorer would be a different number."
        )

    graded: dict[int, PickGrade] = {}
    for version in sorted({str(p["scorer_version"]) for p in picks}):
        idx = [i for i, p in enumerate(picks) if str(p["scorer_version"]) == version]
        subset = [picks[i] for i in idx]
        grades = settlement_grades(scorers_by_version[version], subset)
        for i, grade in zip(idx, grades, strict=True):
            graded[i] = grade
    scorer_grades = [graded[i] for i in range(len(picks))]

    if (baseline_grades is None) == (baseline_ckpt is None):
        raise GateError("pass exactly one of baseline_grades or baseline_ckpt")
    if baseline_grades is None:
        from catan_rl.bc.finetune import setup_pick_grades

        assert baseline_ckpt is not None
        baseline = list(setup_pick_grades(Path(baseline_ckpt), picks, device=device))
    else:
        baseline = list(baseline_grades)
    if len(baseline) != len(picks):
        raise GateError(f"baseline grades have {len(baseline)} entries for {len(picks)} picks")

    def sub(
        indices: Sequence[int],
    ) -> tuple[list[Mapping[str, Any]], list[PickGrade], list[PickGrade]]:
        return (
            [picks[i] for i in indices],
            [scorer_grades[i] for i in indices],
            [baseline[i] for i in indices],
        )

    def where(pred: Any) -> list[int]:
        return [i for i, p in enumerate(picks) if pred(p)]

    all_idx = list(range(len(picks)))
    arm_idx = {
        arm: where(lambda p, arm=arm: reveal_arm(p, meta) == arm)
        for arm in (REVEAL_MODE_REVEAL, REVEAL_MODE_NO_REVEAL)
    }
    arms = {arm: _subset_report(arm, *sub(idx)[1:], alpha=alpha) for arm, idx in arm_idx.items()}

    n = len(picks)
    n_no_reveal = len(arm_idx[REVEAL_MODE_NO_REVEAL])
    no_reveal_fraction = n_no_reveal / n if n else 0.0
    control_satisfied = no_reveal_fraction >= MIN_NO_REVEAL_FRACTION
    divergent = _cis_disjoint(
        arms[REVEAL_MODE_REVEAL]["paired"], arms[REVEAL_MODE_NO_REVEAL]["paired"]
    )
    # D3: "only no-reveal picks count for the gate until understood." Divergence
    # between the arms is the signal that the reveals may be training the owner,
    # so the verdict falls back to the control arm rather than being annotated.
    gate_idx = arm_idx[REVEAL_MODE_NO_REVEAL] if divergent else all_idx
    gate_subset = "no_reveal_only" if divergent else "all_fresh_picks"

    g_picks, g_scorer, g_baseline = sub(gate_idx)
    overall = _subset_report("overall", g_scorer, g_baseline, alpha=alpha)
    p24_idx = [i for i in gate_idx if int(picks[i]["draft_position"]) >= 2]
    p24 = _subset_report("picks_2_4", *sub(p24_idx)[1:], alpha=alpha)
    by_position = {
        str(pos): _subset_report(
            f"position_{pos}",
            *sub([i for i in gate_idx if int(picks[i]["draft_position"]) == pos])[1:],
            alpha=alpha,
        )
        for pos in (1, 2, 3, 4)
    }
    clarity = _clarity_report(g_picks, g_scorer, g_baseline, alpha=alpha)
    calibration = _calibration_report(g_picks, g_scorer)

    overall_paired = overall["paired"]
    p24_paired = p24["paired"]
    scorer_beats_u500_on_2_4 = bool(p24_paired is not None and p24_paired["delta"] > 0.0)
    clear_bar_satisfied = bool(clarity[PICK_CLARITY_CLEAR]["satisfied"])
    clear_bar_status = str(clarity[PICK_CLARITY_CLEAR]["status"])
    enough = len(gate_idx) >= min_fresh_picks
    passes = bool(
        enough
        and control_satisfied
        and overall_paired is not None
        and overall_paired["ci_lower"] > 0.0
        and scorer_beats_u500_on_2_4
        and clear_bar_satisfied
    )

    return {
        "primary_metric": "paired_mean_log_probability_of_owner_pick",
        "n_fresh_picks": n,
        "n_gate_picks": len(gate_idx),
        "gate_subset": gate_subset,
        "min_fresh_picks": min_fresh_picks,
        "enough_picks": enough,
        "alpha": alpha,
        "overall": overall,
        "picks_2_4": p24,
        "by_position": by_position,
        "arms": arms,
        "clarity": clarity,
        "calibration": calibration,
        "anchoring_control": {
            "n_no_reveal": n_no_reveal,
            "fraction_no_reveal": no_reveal_fraction,
            "required_fraction": MIN_NO_REVEAL_FRACTION,
            "satisfied": control_satisfied,
            "arms_divergent": divergent,
            "gate_subset": gate_subset,
        },
        "relational_weights": {
            version: {
                name: float(w)
                for name, w in zip(
                    scorers_by_version[version].settlement.feature_names,
                    scorers_by_version[version].settlement.weights,
                    strict=True,
                )
                if name
                in (
                    "opponent_new_resources",
                    "opponent_best_margin",
                    "adjacency_block",
                    "scarcity_starve",
                )
            }
            for version in sorted({str(p["scorer_version"]) for p in picks})
        },
        "kill_bar": {
            "metric": "picks_2_4_paired_log_probability",
            "cumulative_fresh_picks": n,
            "bar": KILL_BAR_PICKS,
            "reached": n >= KILL_BAR_PICKS,
            "dead": bool(n >= KILL_BAR_PICKS and not scorer_beats_u500_on_2_4),
        },
        "scorer_wilson": asdict(_wilson_dict(sum(g.agree for g in g_scorer), len(g_scorer), alpha)),
        "passes": passes,
        "pass_clauses": {
            "enough_picks": enough,
            "anchoring_control": control_satisfied,
            "overall_log_prob_ci_lower_gt_0": bool(
                overall_paired is not None and overall_paired["ci_lower"] > 0.0
            ),
            "picks_2_4_log_prob_delta_gt_0": scorer_beats_u500_on_2_4,
            "clear_top1_bar": clear_bar_satisfied,
            # Named separately so a NO-GO on this clause says which remedy it
            # wants: ``unmeasured`` means the exam holds fewer than
            # ``MIN_CLEAR_PICKS`` picks the owner tagged ``clear`` (tag some, or
            # the strictness clause has nothing to bite on); ``below_bar`` means
            # the scorer was measured on them and missed.
            "clear_top1_bar_status": clear_bar_status,
        },
    }


@dataclass(frozen=True)
class _WilsonRow:
    n: int
    n_agree: int
    rate: float
    ci_lower: float
    ci_upper: float


def _wilson_dict(n_agree: int, n: int, alpha: float) -> _WilsonRow:
    if n <= 0:
        return _WilsonRow(n=0, n_agree=0, rate=0.0, ci_lower=0.0, ci_upper=0.0)
    ci = wilson_interval(wins=n_agree, n=n, alpha=alpha)
    return _WilsonRow(n=n, n_agree=n_agree, rate=ci.point, ci_lower=ci.lower, ci_upper=ci.upper)
