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

The label corpus is UNTRACKED, so these skip when it is absent — a hermetic
fixture would be a different corpus and would prove nothing about the pilot.

**THEY SKIP IN CI AND IN THE /dev-loop WORKTREE.** A git worktree carries no
untracked files, so ``data/labels/**`` and the pilot shard manifest are both
absent there and all three tests skip. A green gate therefore says NOTHING about
acceptance criterion 3. Point ``data/labels/setup/v1`` and
``data/bc/human_openings/v1`` at the owner's checkout to actually run them.

**RESULT, run out-of-band against the owner's 292-row corpus (2026-08-21, all
three PASS)** — recorded here because the gate cannot record it:

* pilot-10-feature refit on the manifest's 168/44 split: held-out **0.3636**
  (pilot 0.341, tol 0.06), train **0.4583** (pilot 0.452). Both inside
  tolerance.
* full-feature fit, same split: held-out **0.3182**, inside the pilot's Wilson
  CI **[0.2188, 0.4886]** at n=44.
* road head **0.5536** against the ``opens_best_vertex_value`` null at
  **0.5476** — a real but very thin margin, reported and not asserted, exactly
  as the "road model is FIT, not asserted" decision requires.

Re-run and update these figures whenever the corpus grows; a stale number here
is worse than none, because it reads as a verification that did not happen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catan_rl.eval.wilson import wilson_interval
from catan_rl.setup_phase.fit import fit_scorer, top1_agreement
from catan_rl.setup_phase.scorer_features import PILOT_FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS = REPO_ROOT / "data" / "labels" / "setup" / "v1" / "scenarios.jsonl"
MANIFEST = REPO_ROOT / "data" / "bc" / "human_openings" / "v1" / "manifest.json"

PILOT_HELD_OUT = 0.341
PILOT_TRAIN = 0.452
TOLERANCE = 0.06

pytestmark = pytest.mark.skipif(
    not (LABELS.exists() and MANIFEST.exists()),
    reason="label corpus / pilot shard manifest not present (both are untracked)",
)


@pytest.fixture(scope="module")
def split():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))
    from fit_scorer_pilot import pilot_split

    return pilot_split(LABELS, MANIFEST)


def test_pilot_split_is_read_from_the_manifest_not_re_derived(split) -> None:
    train, held_out = split
    # 168 / 44 are the manifest's own numbers. ``held_out_split`` would
    # re-derive a DIFFERENT split the moment the owner labels another board,
    # which is exactly why the manifest is the source here.
    assert len(train) == 168
    assert len(held_out) == 44


def test_pilot_features_reproduce_the_pilot_within_tolerance(split) -> None:
    train, held_out = split
    subset = list(PILOT_FEATURE_NAMES)
    result = fit_scorer(train, version="acc3-pilot", seed=0, settlement_feature_subset=subset)
    held = float(np.mean(top1_agreement(result.scorer, held_out, settlement_feature_subset=subset)))
    train_agreement = float(result.metrics["settlement"]["agreement"])
    assert abs(held - PILOT_HELD_OUT) < TOLERANCE, f"held-out {held} vs {PILOT_HELD_OUT}"
    assert abs(train_agreement - PILOT_TRAIN) < TOLERANCE


def test_full_features_land_inside_the_wilson_ci_of_the_pilot(split) -> None:
    train, held_out = split
    result = fit_scorer(train, version="acc3-full", seed=0)
    held = float(np.mean(top1_agreement(result.scorer, held_out)))
    ci = wilson_interval(wins=round(PILOT_HELD_OUT * len(held_out)), n=len(held_out))
    assert ci.lower <= held <= ci.upper, (
        f"full-feature held-out {held:.3f} outside the pilot's CI "
        f"[{ci.lower:.3f}, {ci.upper:.3f}] at n={len(held_out)}"
    )
