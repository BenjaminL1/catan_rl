"""Engine-parity guard for the cross-architecture eval.

The in-process cross-arch head-to-head (:mod:`catan_rl.eval.cross_arch`) rests on
ONE premise: the game engine (+ board geometry) is byte-identical to the tree the
vendored v11-era encoder (:mod:`catan_rl.eval.legacy_arch`) is validated against
— originally the pre-fork tree, plus any deliberate change that has since been
re-validated and re-pinned (see the re-pin log below). If the live engine ever
diverges from it, the legacy encoder reads a game it was not written for and the
head-to-head is silently invalid — so this guard REFUSES to run in that case
(the third mandatory correctness guard for the harness).

The pinned ids are git TREE / BLOB object ids of the engine dir + board_geometry
file, hashed from the **WORKING TREE** (not from HEAD, and not from the old
pre-fork commit): a git object id is content-derived, so the id of an edit is the
same before and after it is committed. That matters — it is what lets a
deliberate engine change be re-validated and re-pinned *in the branch that makes
it*, instead of the guard's answer changing at commit time. It also works in
shallow CI clones, which need only HEAD (to seed a scratch index).

If the engine is EVER changed deliberately (which Constitution/CLAUDE.md require
be flagged), this guard will fire until the vendored arch is re-validated against
the new engine and these pins are updated — that tripwire is the point.

Re-pin log (append one line per deliberate engine change):

* 2026-07-28 — window size ``1000x800`` -> ``1200x900`` (``engine/board.py``, the
  playtest-HUD overflow fix). Re-validated for cross-arch use: the vendored
  legacy encoder consumes board pixels only as DICT KEYS / index lookups
  (``obs_encoder.py:262,404,425`` and ``policy/board_geometry.py:138``), never as
  feature VALUES, and the resize is a pure integer translation of the lattice
  that renumbers nothing (pinned by ``tests/unit/engine/test_topology_stability``).
  Engine tree ``261098d190c8`` -> ``3388b69026cb``; board_geometry blob unchanged.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

_LOG = logging.getLogger("catan_rl.eval.cross_arch")

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: git object ids of the cross-arch-validated engine tree + board_geometry blob.
#: The board_geometry blob is still the fork-unchanged one (== the blob at
#: ``_provenance.VENDOR_COMMIT`` == ``9692a79~1``); the engine tree was re-pinned
#: on 2026-07-28 (see the module docstring's re-pin log). Update these ONLY after
#: a deliberate engine change has been re-validated for cross-arch use.
PINNED_ENGINE_TREE = "3388b69026cb813c4f612e62ffdfa78725dcf77b"
PINNED_BOARD_GEOMETRY_BLOB = "70813dcf76fde390ef43b249bc50c7ea57e1b0ad"

_ENGINE_PATH = "src/catan_rl/engine"
_BOARD_GEOMETRY_PATH = "src/catan_rl/policy/board_geometry.py"


class EngineParityError(RuntimeError):
    """Raised when the live engine differs from the pinned, validated tree."""


def _git(*args: str, index_file: str | None = None) -> subprocess.CompletedProcess[str]:
    env = None
    if index_file is not None:
        env = {**os.environ, "GIT_INDEX_FILE": index_file}
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def _live_object_id(path: str) -> str | None:
    """git object id of the **working-tree** content at ``path``.

    A file hashes to its blob id directly; a directory is hashed by staging it
    into a scratch index seeded from HEAD and reading the subtree id back, which
    yields exactly the id the same content gets once committed. Returns ``None``
    if git cannot answer (no git binary, no repo, exported source tree).
    """
    target = _REPO_ROOT / path
    if target.is_file():
        blob = _git("hash-object", "--", str(target))
        return blob.stdout.strip() if blob.returncode == 0 else None
    with tempfile.TemporaryDirectory() as tmp:
        index = str(Path(tmp) / "index")
        if _git("read-tree", "HEAD", index_file=index).returncode != 0:
            return None
        if _git("add", "--all", "--", path, index_file=index).returncode != 0:
            return None
        tree = _git("write-tree", index_file=index)
        if tree.returncode != 0:
            return None
        sub = _git("rev-parse", f"{tree.stdout.strip()}:{path}")
        return sub.stdout.strip() if sub.returncode == 0 else None


def assert_engine_parity(*, strict: bool = True) -> dict[str, str]:
    """Refuse (raise :class:`EngineParityError`) if the live engine + board
    geometry differ from the pinned pre-fork tree. Returns a ``{path: sha}``
    stamp for logging / display.

    Behaviour:

    * **Detected drift** — the WORKING-TREE engine-tree / board-geometry-blob
      SHA differs from the pin -> raise. This is the real safety case, and it
      covers uncommitted edits too (their object id is the same one they will
      have once committed).
    * **Cannot verify** — git binary or repo unavailable -> log a WARNING and
      proceed UNVERIFIED, so shallow clones / exported source trees still run.
      (Only HEAD is needed to seed the scratch index, so an ordinary git
      checkout — even ``--depth 1`` — resolves it.)
    * ``strict=False`` -> skip the check entirely (deliberate-engine-experiment
      / trust-me escape hatch).
    """
    if not strict:
        return {"engine": "unchecked", "board_geometry": "unchecked"}

    engine_sha = _live_object_id(_ENGINE_PATH)
    geom_sha = _live_object_id(_BOARD_GEOMETRY_PATH)
    if engine_sha is None or geom_sha is None:
        _LOG.warning(
            "cross-arch: could not verify engine parity (git/repo unavailable); "
            "proceeding UNVERIFIED — ensure the engine matches the pre-fork tree"
        )
        return {"engine": "unverified", "board_geometry": "unverified"}

    problems: list[str] = []
    if engine_sha != PINNED_ENGINE_TREE:
        problems.append(f"engine tree {engine_sha[:12]} != pinned {PINNED_ENGINE_TREE[:12]}")
    if geom_sha != PINNED_BOARD_GEOMETRY_BLOB:
        problems.append(
            f"board_geometry blob {geom_sha[:12]} != pinned {PINNED_BOARD_GEOMETRY_BLOB[:12]}"
        )
    if problems:
        raise EngineParityError(
            "ENGINE DRIFT — the cross-arch measurement is INVALID because the live "
            "engine no longer matches the pre-fork tree the v11-era encoder was "
            "written against: "
            + "; ".join(problems)
            + ". The vendored legacy encoder reads the live game state, so any "
            "engine change breaks faithful cross-version play. Re-vendor + re-pin "
            "after a deliberate, flagged engine change, or pass strict=False / "
            "--skip-engine-parity-check to override (the result is then untrustworthy)."
        )
    return {"engine": engine_sha[:12], "board_geometry": geom_sha[:12]}
