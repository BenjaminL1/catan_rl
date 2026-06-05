"""The viewer must NOT transitively import torch / gymnasium / engine
/ env / policy. Otherwise the recorder's "viewer is a thin dep-light
companion" contract is silently broken whenever a future PR adds a
type hint that crosses the boundary.

We assert this with a subprocess import + ``sys.modules`` check —
``grep`` would miss conditional imports inside the module bodies.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Modules that the viewer MUST NOT touch.
_BANNED_PREFIXES = (
    "torch",
    "gymnasium",
    "catan_rl.engine",
    "catan_rl.env",
    "catan_rl.policy",
    "catan_rl.algorithms",
    "catan_rl.selfplay",
    "catan_rl.bc",
    "catan_rl.checkpoint",
    "catan_rl.agents",
)


@pytest.mark.parametrize(
    "import_target",
    [
        "catan_rl.replay.viewer",
        "catan_rl.replay.viewer.event_loop",
    ],
)
def test_viewer_imports_do_not_pull_heavy_deps(import_target: str) -> None:
    """Import the viewer module in a clean subprocess and assert that
    no banned module ended up in ``sys.modules``.

    A subprocess is required because pytest itself has already
    imported torch / engine / etc. by the time this test runs in the
    parent process — the in-process ``sys.modules`` check would be
    meaningless."""
    src = _REPO_ROOT / "src"
    check_lines = [
        f"banned_prefixes = {_BANNED_PREFIXES!r}",
        f"import sys; sys.path.insert(0, {str(src)!r})",
        f"import {import_target}",
        (
            "leaked = [m for m in sys.modules if any("
            "m == p or m.startswith(p + '.') for p in banned_prefixes)]"
        ),
        "assert not leaked, ('leaked: ' + ', '.join(sorted(leaked)))",
    ]
    script = "; ".join(check_lines)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={"SDL_VIDEODRIVER": "dummy", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        f"viewer import leaked banned modules.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
