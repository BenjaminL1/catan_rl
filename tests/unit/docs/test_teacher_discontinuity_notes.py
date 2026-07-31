"""Pins for the F-D teacher-discontinuity caveat.

F-D's hard constraint is "annotate the WR-vs-heuristic gates, DO NOT change the
threshold values". `docs/plans/v2/design.md` already has a pin
(``tests/unit/scripts/test_play_vs_model.py``, inherited from
``play-vs-model-recorder`` D1), but ``step4_ppo.md`` and
``setup_strength_roadmap.md`` had none — a later edit could silently move a
0.90 to 0.80 and nothing would notice.

Both halves are pinned here:

* the note is PRESENT in each annotated document, and
* each document still carries its GATE LITERALS — the specific numbers F-D
  annotated but was forbidden to retune.

The literal pin replaces an earlier file-level insert-only rule. See
``_PINNED_THRESHOLDS`` for why that rule was both too weak (it self-disarmed on
merge) and too strong (it froze three roadmap docs against any rewrite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]

#: Documents that carry the D2 teacher-discontinuity caveat.
_ANNOTATED = (
    "docs/plans/v2/design.md",
    "docs/plans/v2/step4_ppo.md",
    "docs/plans/v2/setup_strength_roadmap.md",
)

_MARKER = "Teacher discontinuity"


@pytest.mark.parametrize("rel", _ANNOTATED)
def test_document_carries_the_teacher_discontinuity_note(rel: str) -> None:
    text = (_REPO / rel).read_text(encoding="utf-8")
    assert _MARKER in text, f"{rel} lost the D2 teacher-discontinuity caveat"
    # The caveat is worthless if it does not say the bars were NOT moved.
    assert "UNCHANGED" in text, f"{rel}'s caveat no longer states the bars are unchanged"


#: The gate literals F-D annotated but must NOT retune, per document.
#:
#: Pinned by VALUE rather than by file-level insert-only (the previous rule).
#: Two reasons the diff-based pin was wrong: it self-disarms the moment the
#: branch merges — both ``HEAD`` and ``origin/main...HEAD`` go empty on a clean
#: tree, so it protects nothing on main — and it froze three roadmap documents
#: against ANY line rewrite on ANY branch, which would have blocked the
#: owner-approved empirical re-baselining without deleting this test.
#:
#: This pin is deliberately defeasible: re-baselining is EXPECTED to change
#: these numbers. Updating the literals here is the intended way to record that
#: a bar moved on purpose. What it prevents is a bar moving SILENTLY.
_PINNED_THRESHOLDS: dict[str, tuple[str, ...]] = {
    "docs/plans/v2/step4_ppo.md": ("0.90", "0.70"),
    "docs/plans/v2/design.md": ("0.95", "0.70", "0.40"),
    "docs/plans/v2/setup_strength_roadmap.md": ("2pp",),
}


@pytest.mark.parametrize("rel", _ANNOTATED)
def test_annotated_document_still_carries_its_gate_literals(rel: str) -> None:
    """F-D annotates; it must not RETUNE.

    Reads the file directly, so it keeps working after merge — unlike a diff
    against ``origin/main``, which goes empty on a clean tree.
    """
    text = (_REPO / rel).read_text(encoding="utf-8")
    missing = [lit for lit in _PINNED_THRESHOLDS[rel] if lit not in text]
    assert missing == [], (
        f"{rel} no longer contains gate literal(s) {missing}. If a bar was "
        f"re-baselined ON PURPOSE, update _PINNED_THRESHOLDS in the same commit "
        f"and say so in the doc; that is the intended path. This pin exists to "
        f"stop a threshold moving silently, not to freeze it forever."
    )
