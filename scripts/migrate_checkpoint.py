#!/usr/bin/env python3
"""Back-compat shim. Canonical entry points are the console script
``catan-rl-migrate-ckpt`` and ``python -m catan_rl.cli.migrate_checkpoint``.
The full body moved to :mod:`catan_rl.cli.migrate_checkpoint` in the
maturin sole-backend cutover.
"""

from catan_rl.cli.migrate_checkpoint import main

if __name__ == "__main__":
    raise SystemExit(main())
