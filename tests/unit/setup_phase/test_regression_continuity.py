"""Acceptance criterion 3 — regression continuity against the 2026-08-15 pilot.

Two claims, both SOFT and both reported rather than asserted tightly:

(a) refitting the pilot's 10 features on the pilot's 168/44 split reproduces the
    pilot's 34.1% held-out / 45.2% train to within a tolerance; and
(b) the FULL-feature fit on the same split lands inside the Wilson CI of 34.1%
    at n=44.

Neither can be bit-exact: the pilot's script lived in a scratchpad that no
longer exists (``scripts/dev/fit_scorer_pilot.py`` is a documented
RECONSTRUCTION, not the artifact). And one pick moves the held-out number by
2.3pp at n=44, which is why the spent 44-pick set never gates anything again.

**Two layers, because the owner's corpus is untracked.**

* :class:`TestContinuityArithmetic` runs ALWAYS, on a committed SYNTHETIC store
  (``tests/fixtures/setup_labels/``: 32 generated rows over 8 boards, no owner
  data). It cannot say anything about the pilot — a different corpus proves
  nothing about a fit on another one — but it does pin the machinery the real
  claim is made with: that the split is read from the manifest and not
  re-derived, that a pilot-subset fit and a full-feature fit both run, and that
  the Wilson-CI containment check is a real comparison. Before this existed, a
  green gate exercised NONE of that, because a git worktree carries no untracked
  files and every test in the module skipped.
* :class:`TestOwnerCorpus` runs only where ``data/labels/**`` and the pilot
  shard manifest exist — i.e. the owner's checkout, never CI and never a
  ``/dev-loop`` worktree. Run it deliberately::

      make ac3-continuity CATAN_LABELS_DIR=/path/to/checkout

  which points the two paths at a real checkout and runs this module.

**RESULT, run out-of-band against the owner's 292-row corpus (2026-08-21, under
FEATURE_VERSION v2)** — recorded here because the gate cannot record it:

* pilot-10-feature refit on the manifest's 168/44 split: held-out **0.3636**
  (pilot 0.341, tol 0.06), train **0.4583** (pilot 0.452). Both inside
  tolerance, and both UNCHANGED by the v2 feature fixes — the pilot's ten
  columns do not include ``expansion_value`` or ``opponent_best_margin``.
* full-feature fit, same split: held-out **0.3409** (v1 read: 0.3182), inside
  the pilot's Wilson CI **[0.2188, 0.4886]** at n=44.
* road head vs the ``opens_best_vertex_value`` null, now OUT OF SAMPLE (5-fold,
  grouped by ``game_seed``): fitted NLL **0.9707** vs null **0.9704** —
  ``beaten_by_fit`` is **False**, and top-1 is a dead heat at **0.5476** each.
  In sample the same fit reads 0.9344 vs 0.9524 (a "win"), which is the point:
  a 3-feature model nesting a 1-feature special case can hardly lose in sample.
  D1 says the "point at the expansion target" rule is "the null hypothesis it
  must beat"; on this corpus it does not beat it. Reported, not buried.

Re-run and update these figures whenever the corpus grows; a stale number here
is worse than none, because it reads as a verification that did not happen.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from catan_rl.eval.wilson import wilson_interval
from catan_rl.setup_phase.fit import fit_scorer, top1_agreement
from catan_rl.setup_phase.scorer_features import PILOT_FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]

_LABELS_ROOT = Path(os.environ.get("CATAN_LABELS_DIR", REPO_ROOT))
LABELS = _LABELS_ROOT / "data" / "labels" / "setup" / "v1" / "scenarios.jsonl"
MANIFEST = _LABELS_ROOT / "data" / "bc" / "human_openings" / "v1" / "manifest.json"

SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "setup_labels"

PILOT_HELD_OUT = 0.341
PILOT_TRAIN = 0.452
TOLERANCE = 0.06


def _pilot_split(labels: Path, manifest: Path) -> tuple[list[dict], list[dict]]:
    if str(REPO_ROOT / "scripts" / "dev") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))
    from fit_scorer_pilot import pilot_split

    return pilot_split(labels, manifest)


class TestContinuityArithmetic:
    """The machinery of acceptance criterion 3, on committed synthetic data.

    Runs everywhere. It makes no claim about the pilot — the corpus is not the
    pilot's — and every assertion here is about mechanism, not about a number.
    """

    @pytest.fixture(scope="class")
    def split(self) -> tuple[list[dict], list[dict]]:
        return _pilot_split(SYNTHETIC / "scenarios.jsonl", SYNTHETIC / "manifest.json")

    def test_the_split_comes_from_the_manifest_not_from_held_out_split(
        self, split: tuple[list[dict], list[dict]]
    ) -> None:
        """``held_out_split`` is seed-count-relative and would silently change
        the moment another board is labeled; the manifest's pinned seed list is
        what makes a re-run comparable to the run before it."""
        import json

        train, held_out = split
        manifest = json.loads((SYNTHETIC / "manifest.json").read_text())
        assert len(train) == manifest["n_scenarios"] == 24
        assert len(held_out) == manifest["n_held_out_scenarios"] == 8
        held_seeds = set(manifest["held_out_game_seeds"])
        assert {int(r["game_seed"]) for r in held_out} == held_seeds
        assert not {int(r["game_seed"]) for r in train} & held_seeds

    def test_the_pilot_subset_and_full_fits_both_run_and_are_comparable(
        self, split: tuple[list[dict], list[dict]]
    ) -> None:
        train, held_out = split
        subset = list(PILOT_FEATURE_NAMES)
        pilot = fit_scorer(
            train, version="acc3-synth-pilot", seed=0, iters=200, settlement_feature_subset=subset
        )
        full = fit_scorer(train, version="acc3-synth-full", seed=0, iters=200)
        assert pilot.scorer.settlement.weights.shape == (10,)
        assert full.scorer.settlement.weights.shape == (len(full.scorer.settlement.feature_names),)

        pilot_held = top1_agreement(pilot.scorer, held_out, settlement_feature_subset=subset)
        full_held = top1_agreement(full.scorer, held_out)
        # Both graders answer the SAME held-out picks — that is what makes the
        # two held-out rates comparable at all.
        assert len(pilot_held) == len(full_held) == len(held_out)
        assert all(isinstance(a, bool) for a in pilot_held + full_held)

    def test_the_wilson_containment_check_is_a_real_comparison(
        self, split: tuple[list[dict], list[dict]]
    ) -> None:
        """The criterion-3 arithmetic itself: a rate inside the CI passes, one
        outside it fails. Pinned on the synthetic split so a broken CI check
        cannot slip through on a corpus CI never sees."""
        _train, held_out = split
        ci = wilson_interval(wins=round(PILOT_HELD_OUT * len(held_out)), n=len(held_out))
        assert ci.lower < PILOT_HELD_OUT < ci.upper  # the pilot rate is in its own CI
        assert not (ci.lower <= 0.0 <= ci.upper)  # ...and a total miss is not
        assert not (ci.lower <= 1.0 <= ci.upper)  # ...nor a perfect score
        # The interval narrows as n grows, which is why the module docstring
        # calls the spent 44-pick read a soft floor rather than a gate.
        n_big = 4 * len(held_out)
        narrower = wilson_interval(wins=round(PILOT_HELD_OUT * n_big), n=n_big)
        assert (narrower.upper - narrower.lower) < (ci.upper - ci.lower)


@pytest.mark.skipif(
    not (LABELS.exists() and MANIFEST.exists()),
    reason=(
        "owner label corpus / pilot shard manifest not present (both untracked). "
        "Run `make ac3-continuity CATAN_LABELS_DIR=/path/to/checkout`."
    ),
)
class TestOwnerCorpus:
    """The real acceptance-criterion-3 claim. Owner-run; see the module docstring."""

    @pytest.fixture(scope="class")
    def split(self) -> tuple[list[dict], list[dict]]:
        return _pilot_split(LABELS, MANIFEST)

    def test_pilot_split_is_read_from_the_manifest_not_re_derived(
        self, split: tuple[list[dict], list[dict]]
    ) -> None:
        train, held_out = split
        # 168 / 44 are the manifest's own numbers.
        assert len(train) == 168
        assert len(held_out) == 44

    def test_pilot_features_reproduce_the_pilot_within_tolerance(
        self, split: tuple[list[dict], list[dict]]
    ) -> None:
        train, held_out = split
        subset = list(PILOT_FEATURE_NAMES)
        result = fit_scorer(train, version="acc3-pilot", seed=0, settlement_feature_subset=subset)
        held = float(
            np.mean(top1_agreement(result.scorer, held_out, settlement_feature_subset=subset))
        )
        train_agreement = float(result.metrics["settlement"]["agreement"])
        assert abs(held - PILOT_HELD_OUT) < TOLERANCE, f"held-out {held} vs {PILOT_HELD_OUT}"
        assert abs(train_agreement - PILOT_TRAIN) < TOLERANCE

    def test_full_features_land_inside_the_wilson_ci_of_the_pilot(
        self, split: tuple[list[dict], list[dict]]
    ) -> None:
        train, held_out = split
        result = fit_scorer(train, version="acc3-full", seed=0)
        held = float(np.mean(top1_agreement(result.scorer, held_out)))
        ci = wilson_interval(wins=round(PILOT_HELD_OUT * len(held_out)), n=len(held_out))
        assert ci.lower <= held <= ci.upper, (
            f"full-feature held-out {held:.3f} outside the pilot's CI "
            f"[{ci.lower:.3f}, {ci.upper:.3f}] at n={len(held_out)}"
        )
