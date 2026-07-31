"""Pins for the F-D teacher-discontinuity caveat.

F-D's hard constraint is "annotate the WR-vs-heuristic gates, DO NOT change the
threshold values". `docs/plans/v2/design.md` already has a pin
(``tests/unit/scripts/test_play_vs_model.py``, inherited from
``play-vs-model-recorder`` D1), but ``step4_ppo.md`` and
``setup_strength_roadmap.md`` had none — a later edit could silently move a
0.90 to 0.80 and nothing would notice.

Both halves are pinned here:

* the note is PRESENT in each annotated document, and
* each document is INSERT-ONLY on this branch — no existing line was rewritten
  or deleted, which is what "do not change the threshold values" reduces to at
  the file level (a threshold edit necessarily removes a line).
"""

from __future__ import annotations

import subprocess
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


def _diff_lines(rel: str, rev: str) -> list[str]:
    return subprocess.run(
        ["git", "diff", "-U0", rev, "--", rel],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()


@pytest.mark.parametrize("rel", _ANNOTATED)
def test_document_carries_the_teacher_discontinuity_note(rel: str) -> None:
    text = (_REPO / rel).read_text(encoding="utf-8")
    assert _MARKER in text, f"{rel} lost the D2 teacher-discontinuity caveat"
    # The caveat is worthless if it does not say the bars were NOT moved.
    assert "UNCHANGED" in text, f"{rel}'s caveat no longer states the bars are unchanged"


@pytest.mark.parametrize("rel", _ANNOTATED)
def test_annotated_document_is_insert_only(rel: str) -> None:
    """F-D annotates; it must not RETUNE. A changed threshold shows up as a
    removed line, so insert-only is the pin."""
    for rev in ("HEAD", "origin/main...HEAD"):
        diff = _diff_lines(rel, rev)
        added = {ln[1:] for ln in diff if ln.startswith("+") and not ln.startswith("+++")}
        # A pure insertion at EOF against a file whose last line lacked a
        # trailing newline re-emits that line as a -/+ pair. Not a rewrite.
        removed = [
            ln
            for ln in diff
            if ln.startswith("-") and not ln.startswith("---") and ln[1:] not in added
        ]
        assert removed == [], f"{rel} had lines REWRITTEN (a threshold retune?): {removed}"
