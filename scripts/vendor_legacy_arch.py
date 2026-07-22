"""One-time (idempotent) generator for the vendored v11-era policy arch.

Copies ``src/catan_rl/policy/{obs_schema,encoders,heads,network,obs_encoder}.py``
from the last pre-fork commit (:data:`catan_rl.eval.legacy_arch._provenance.VENDOR_COMMIT`)
into ``src/catan_rl/eval/legacy_arch/`` as faithful copies, rewriting only the
three intra-package import lines. Re-run after nothing — the source commit is
frozen — but the script is idempotent, so running it is a safe way to prove the
files on disk match the pinned commit.

Usage: ``python scripts/vendor_legacy_arch.py [--check]``
  --check : do not write; exit non-zero if any file differs from what would be
            generated (mirrors the provenance test; handy in a pre-commit hook).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from catan_rl.eval.legacy_arch import _provenance as prov

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEST = _REPO_ROOT / "src" / "catan_rl" / "eval" / "legacy_arch"


def _git_show(module: str) -> str:
    path = f"{prov.SRC_PREFIX}/{module}.py"
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", f"{prov.VENDOR_COMMIT}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only; do not write")
    args = parser.parse_args()

    drift = False
    for module in prov.VENDORED_MODULES:
        rendered = prov.render_from_original(module, _git_show(module))
        target = _DEST / f"{module}.py"
        if args.check:
            current = target.read_text() if target.exists() else ""
            if current != rendered:
                print(f"DRIFT: {target.relative_to(_REPO_ROOT)} differs from pinned source")
                drift = True
            else:
                print(f"ok:    {target.relative_to(_REPO_ROOT)}")
        else:
            target.write_text(rendered)
            print(f"wrote: {target.relative_to(_REPO_ROOT)}")

    if args.check and drift:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
