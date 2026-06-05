#!/usr/bin/env python3
"""Back-compat shim. Canonical entry points are the console script
``catan-rl-bc-generate`` and ``python -m catan_rl.cli.generate_bc_dataset``.
The full body moved to :mod:`catan_rl.cli.generate_bc_dataset` in
the maturin sole-backend cutover.
"""

from catan_rl.cli.generate_bc_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
