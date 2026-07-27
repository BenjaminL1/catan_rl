"""Pins for the ``scripts/play_vs_model.py`` RENAME (D1).

The old script is gone with no shim, nothing outside the spec still names it,
and the locked design doc still reserves the new path (and is NOT edited to
match the build).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "play_vs_model.py"
#: Assembled at runtime ON PURPOSE. Spelling the old name literally would make
#: this file its own last surviving reference the moment it is tracked, and the
#: grep pin below would fail on itself.
_OLD_STEM = "play_vs_" + "v8"


class TestRenamePin:
    def test_new_path_exists_and_old_one_does_not(self) -> None:
        assert _SCRIPT.is_file()
        assert not (_REPO / "scripts" / f"{_OLD_STEM}.py").exists(), "no deprecation shim"

    def test_no_old_name_reference_survives(self) -> None:
        # ``--untracked`` so a not-yet-staged file cannot hide an offender (and
        # so this pin does not flip red the moment the change is committed);
        # git still skips venvs / runs/ via .gitignore.
        out = subprocess.run(
            ["git", "grep", "--untracked", "-l", _OLD_STEM],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        # The spec itself documents the rename and is allowed to name the old file.
        offenders = [p for p in out if not p.startswith(".claude/veriloop/specs/")]
        assert offenders == [], f"stale {_OLD_STEM} references: {offenders}"

    def test_locked_design_doc_still_reserves_the_path(self) -> None:
        rel = "docs/plans/v2/design.md"
        design = (_REPO / rel).read_text(encoding="utf-8")
        assert "play_vs_model.py" in design
        # D1 forbids editing the ratified design to match the build, so pin
        # that the file is UNMODIFIED (not merely that it mentions the path).
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", rel],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        assert dirty == "", f"{rel} was modified: {dirty}"
        committed = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD", "--", rel],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert committed.stdout.strip() == "", f"{rel} was edited on this branch"
