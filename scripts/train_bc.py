#!/usr/bin/env python3
"""Back-compat shim. Canonical entry points are the console script
``catan-rl-bc-train`` and ``python -m catan_rl.cli.train_bc``.
The full body moved to :mod:`catan_rl.cli.train_bc` in the maturin
sole-backend cutover.

NOTE: :func:`catan_rl.cli.train_bc.main` returns ``None`` (not an
int). ``SystemExit(None)`` exits with code 0 — the shim preserves
the legacy return semantics.
"""

from catan_rl.cli.train_bc import main

if __name__ == "__main__":
    raise SystemExit(main())
