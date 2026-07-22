"""Faithful-copy provenance rules for the vendored v11-era (legacy) policy arch.

The five generated modules in this package (``obs_schema``, ``encoders``,
``heads``, ``network``, ``obs_encoder``) are FAITHFUL COPIES of
``src/catan_rl/policy/*.py`` as of :data:`VENDOR_COMMIT` — the last commit
before the pointer-arch fork (``9692a79``). Only three intra-package import
LINES are rewritten (``catan_rl.policy.{obs_schema,encoders,heads}`` ->
``catan_rl.eval.legacy_arch.*``); every other import (engine, board_geometry,
hand_tracker) still targets the current, fork-unchanged module — which is sound
because the engine + geometry are byte-identical across the fork.

This module is PURE (no IO, no git): it holds the constants + the deterministic
``render_from_original`` transform. The one-time generator
(``scripts/vendor_legacy_arch.py``) and the provenance test
(``tests/unit/eval/test_legacy_arch_provenance.py``) each fetch the original
source via ``git show`` and feed it here — so the rewrite logic has a single
source of truth and the vendored files are provably a faithful copy.
"""

from __future__ import annotations

#: Full SHA of the last pre-fork commit (== ``9692a79~1``). The vendored arch is
#: a copy of ``src/catan_rl/policy/*.py`` at this commit.
VENDOR_COMMIT = "501b3d2bfcb87b0014c22b06cd67c17c013b989c"
VENDOR_COMMIT_SHORT = "501b3d2"

#: Repo-relative directory the originals were copied from.
SRC_PREFIX = "src/catan_rl/policy"

#: Modules vendored, in dependency order (obs_schema first — everything imports
#: it; network last — it imports encoders + heads).
VENDORED_MODULES: tuple[str, ...] = (
    "obs_schema",
    "encoders",
    "heads",
    "network",
    "obs_encoder",
)

#: Line-prefix rewrites: policy sibling -> legacy_arch sibling. ONLY the three
#: modules the fork mutated are redirected; imports of unchanged modules
#: (``catan_rl.engine.board``, ``catan_rl.env.hand_tracker``,
#: ``catan_rl.policy.board_geometry``) are left pointing at the live code.
IMPORT_REWRITES: dict[str, str] = {
    "from catan_rl.policy.obs_schema import": "from catan_rl.eval.legacy_arch.obs_schema import",
    "from catan_rl.policy.encoders import": "from catan_rl.eval.legacy_arch.encoders import",
    "from catan_rl.policy.heads import": "from catan_rl.eval.legacy_arch.heads import",
}


def header(module: str) -> str:
    """Return the generated-file banner prepended to every vendored module."""
    return (
        f"# GENERATED — faithful copy of {SRC_PREFIX}/{module}.py @ {VENDOR_COMMIT_SHORT}\n"
        f"# (the last commit before the pointer-arch fork). DO NOT EDIT BY HAND.\n"
        f"# Regenerate: python scripts/vendor_legacy_arch.py\n"
        f"# See the catan_rl.eval.legacy_arch package + _provenance.py for rationale.\n"
    )


def rewrite_imports(source: str) -> str:
    """Apply :data:`IMPORT_REWRITES` to matching import lines (line-prefix match).

    Only the leading module path is swapped; the imported names + any
    parenthesised continuation are preserved verbatim.
    """
    out: list[str] = []
    for line in source.splitlines(keepends=True):
        for old, new in IMPORT_REWRITES.items():
            if line.startswith(old):
                line = new + line[len(old) :]
                break
        out.append(line)
    return "".join(out)


def render_from_original(module: str, original_source: str) -> str:
    """Deterministically render the vendored file body from the original source.

    ``header(module) + rewrite_imports(original_source)``. Both the generator and
    the provenance test call THIS, so a vendored file on disk is faithful iff it
    equals ``render_from_original(module, git_show(VENDOR_COMMIT, ...))``.
    """
    return header(module) + rewrite_imports(original_source)
