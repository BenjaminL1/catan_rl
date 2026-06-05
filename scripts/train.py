#!/usr/bin/env python3
"""Back-compat shim. Canonical entry points are the console script
``catan-rl-train`` (installed by ``pip install -e .``) and
``python -m catan_rl.cli.train``.

The full body moved to :mod:`catan_rl.cli.train` in the maturin
sole-backend cutover so ``[project.scripts]`` can wire it onto PATH.
"""

from catan_rl.cli.train import main

if __name__ == "__main__":
    raise SystemExit(main())
