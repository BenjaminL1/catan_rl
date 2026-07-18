#!/usr/bin/env python3
"""Thin shim. Canonical entry point is
``python -m catan_rl.cli.reclaim_disk``. The body lives in
:mod:`catan_rl.cli.reclaim_disk`.

Report + reclaim disk from a checkpoint tree: dedup byte-identical files and
re-save fat anchors as policy-only slim checkpoints. Dry-run by default; pass
``--execute`` to act.

Usage::

    python scripts/reclaim_disk.py runs/            # dry-run report
    python scripts/reclaim_disk.py runs/ --execute  # dedup + slim anchors
"""

from catan_rl.cli.reclaim_disk import main

if __name__ == "__main__":
    raise SystemExit(main())
