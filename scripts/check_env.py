#!/usr/bin/env python3
"""Fail if the importable ``catan_rl`` is a DIFFERENT checkout than this one.

Why this exists
---------------
The dev-loop runs ``pip install -e .`` inside its own git worktree. Against a
shared interpreter (conda base, say) that rewrites the global ``catan_rl.pth``
to point at the worktree. Two things then happen, both observed:

* a later ``python scripts/foo.py`` from another checkout silently imports the
  OTHER tree's code — measurements describe code you are not looking at;
* deleting that worktree leaves ``import catan_rl`` broken outright.

Why HERE and not in the package or a test fixture
-------------------------------------------------
* Not in ``catan_rl/__init__.py``: when a stale tree is what gets imported, the
  guard would live in the code being *replaced*. The stale tree does not have
  it, so nothing fires — verified.
* Not in ``tests/conftest.py``: pytest is immune. ``pythonpath = ["src"]`` in
  ``pyproject.toml`` inserts at ``sys.path[0]`` relative to the ini file, so it
  always beats the ``.pth`` — a conftest assertion can never fail, verified by
  pointing the ``.pth`` at another tree and watching the suite pass.

The vulnerable surface is plain scripts and bare subprocesses: 31 of 41
``scripts/*.py`` resolve ``catan_rl`` purely through the ``.pth``.

Uses ``find_spec`` rather than importing, so a broken or foreign tree is
reported instead of executed.
"""

from __future__ import annotations

import importlib.util
import os
import sys


def main() -> int:
    spec = importlib.util.find_spec("catan_rl")
    origin = spec.origin if spec is not None else None
    got = os.path.realpath(os.path.dirname(origin)) if origin else None
    want = os.path.realpath(os.path.join(os.getcwd(), "src", "catan_rl"))

    if got == want:
        return 0

    if got is None:
        detail = (
            "catan_rl is not importable at all. The editable install probably "
            "points at a git worktree that has since been deleted."
        )
    else:
        detail = f"catan_rl resolves to\n    {got}\nbut this checkout is\n    {want}"

    print(
        f"ENV MISMATCH: {detail}\n"
        "  A `pip install -e .` from another tree (usually a dev-loop worktree) "
        "has hijacked the shared environment.\n"
        "  Any measurement run outside pytest may describe the WRONG code.\n"
        f"  Fix: {sys.executable} -m pip install -e . --no-deps",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
