"""Pygame replay viewer CLI.

Usage::

    python scripts/replay_viewer.py path/to/replay.json
    python scripts/replay_viewer.py path/to/replay.json --window-size 1600x900

Loads the JSON replay and opens a pygame window. ESC or window-close
button exit cleanly. See :mod:`catan_rl.replay.viewer.event_loop` for
the rendering hook.

This script imports ONLY pygame + the replay package (NO torch,
gymnasium, engine, env, or policy). The transitive-import contract
is asserted in ``tests/unit/replay/test_viewer_import_isolation.py``.
"""

from __future__ import annotations

# Back-compat shim. Canonical entry points are the console script
# ``catan-rl-replay`` (installed by ``pip install -e .``) and
# ``python -m catan_rl.replay.viewer.event_loop``. The
# ``sys.path.insert(REPO_ROOT/'src')`` shim was dropped in the
# maturin sole-backend cutover — ``catan_rl`` is importable from
# the install path now.
from catan_rl.replay.viewer.event_loop import main

if __name__ == "__main__":
    raise SystemExit(main())
