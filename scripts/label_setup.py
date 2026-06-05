#!/usr/bin/env python3
"""Back-compat shim. Canonical entry points are the console script
``catan-rl-label-setup`` and ``python -m catan_rl.cli.label_setup``.
The full body moved to :mod:`catan_rl.cli.label_setup` in the
maturin sole-backend cutover.
"""

from catan_rl.cli.label_setup import main

if __name__ == "__main__":
    raise SystemExit(main())
