"""Engine-parity guard for the cross-architecture eval.

The in-process cross-arch head-to-head (:mod:`catan_rl.eval.cross_arch`) rests on
ONE premise: the game engine (+ board geometry) is byte-identical to the pre-fork
tree the vendored v11-era encoder (:mod:`catan_rl.eval.legacy_arch`) was written
against. If the live engine ever diverges, the legacy encoder reads a game it was
not written for and the head-to-head is silently invalid — so this guard REFUSES
to run in that case (the third mandatory correctness guard for the harness).

The pinned ids are git TREE / BLOB object ids of the engine dir + board_geometry
file at :data:`catan_rl.eval.legacy_arch._provenance.VENDOR_COMMIT` (the last
pre-fork commit); they were verified byte-identical to HEAD when the fork landed.
The check reads them off **HEAD** (not the old commit), so it works in shallow CI
clones where the pre-fork commit object is absent.

If the engine is EVER changed deliberately (which Constitution/CLAUDE.md require
be flagged), this guard will fire until the vendored arch is re-validated against
the new engine and these pins are updated — that tripwire is the point.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

_LOG = logging.getLogger("catan_rl.eval.cross_arch")

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: git object ids of the fork-unchanged engine tree + board_geometry blob (==
#: the tree at ``_provenance.VENDOR_COMMIT`` == ``9692a79~1``). Update these ONLY
#: after a deliberate engine change has been re-validated for cross-arch use.
#:
#: RE-PIN LOG. Every entry must carry the re-VALIDATION argument; "I ran the
#: guard and it passed" is circular and is not one.
#:
#: * ``261098d190c8`` -> ``08bd139cfa14`` (2026-08-02, human-path-conformance-fixes
#:   D1/D2). ``engine/player.play_devCard`` was restructured so that no
#:   cancellable choice follows a state commit. ``eval/legacy_arch/obs_encoder.py``
#:   does read exactly the counters that move (``knightsPlayed`` / ``yopPlayed`` /
#:   ``monopolyPlayed`` / ``roadBuilderPlayed``), so the tripwire firing is
#:   correct. It is nonetheless UNREACHABLE from any cross-arch measurement:
#:   ``play_devCard`` has exactly two call sites — ``engine/game.py``'s legacy
#:   pygame loop, which the RL stack never enters, and ``scripts/play_vs_model.py``
#:   (both pinned by ``tests/unit/scripts/test_play_vs_model.py``) — and
#:   ``agents/heuristic.py`` deliberately routes around it through
#:   ``heuristic_play_*``. ``cross_arch_h2h`` is bot-vs-bot through
#:   ``env/catan_env._apply_main_action``, which this change does not touch. The
#:   diff therefore moves the byte pin without changing any behaviour a
#:   cross-arch measurement can observe; the vendored arch is re-validated
#:   against the new engine by construction, not by re-running this guard.
#:   (The DRAFT ``bank-fix-slice-and-champion-bank.md`` records a re-pin
#:   ``261098d190c8 -> 3388b69026cb`` as landed 2026-07-28. It did not land: that
#:   sha appears nowhere in the tree and the live constant was still
#:   ``261098d190c8``. This log starts from the real value, not that one.)
PINNED_ENGINE_TREE = "08bd139cfa14de1922e0105f31bfb7ee5e401ef3"
PINNED_BOARD_GEOMETRY_BLOB = "70813dcf76fde390ef43b249bc50c7ea57e1b0ad"

_ENGINE_PATH = "src/catan_rl/engine"
_BOARD_GEOMETRY_PATH = "src/catan_rl/policy/board_geometry.py"


class EngineParityError(RuntimeError):
    """Raised when the live engine differs from the pinned pre-fork tree."""


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), *args],
        capture_output=True,
        text=True,
    )


def assert_engine_parity(*, strict: bool = True) -> dict[str, str]:
    """Refuse (raise :class:`EngineParityError`) if the live engine + board
    geometry differ from the pinned pre-fork tree. Returns a ``{path: sha}``
    stamp for logging / display.

    Behaviour:

    * **Detected drift** — the committed engine-tree / board-geometry-blob SHA
      differs from the pin, OR there are uncommitted changes under the guarded
      paths -> raise. This is the real safety case.
    * **Cannot verify** — git binary or repo unavailable -> log a WARNING and
      proceed UNVERIFIED, so shallow clones / exported source trees still run.
      (The pins live at HEAD, so an ordinary git checkout — even ``--depth 1`` —
      resolves them.)
    * ``strict=False`` -> skip the check entirely (deliberate-engine-experiment
      / trust-me escape hatch).
    """
    if not strict:
        return {"engine": "unchecked", "board_geometry": "unchecked"}

    live_engine = _git("rev-parse", f"HEAD:{_ENGINE_PATH}")
    live_geom = _git("rev-parse", f"HEAD:{_BOARD_GEOMETRY_PATH}")
    if live_engine.returncode != 0 or live_geom.returncode != 0:
        _LOG.warning(
            "cross-arch: could not verify engine parity (git/repo unavailable); "
            "proceeding UNVERIFIED — ensure the engine matches the pre-fork tree"
        )
        return {"engine": "unverified", "board_geometry": "unverified"}

    engine_sha = live_engine.stdout.strip()
    geom_sha = live_geom.stdout.strip()
    problems: list[str] = []
    if engine_sha != PINNED_ENGINE_TREE:
        problems.append(f"engine tree {engine_sha[:12]} != pinned {PINNED_ENGINE_TREE[:12]}")
    if geom_sha != PINNED_BOARD_GEOMETRY_BLOB:
        problems.append(
            f"board_geometry blob {geom_sha[:12]} != pinned {PINNED_BOARD_GEOMETRY_BLOB[:12]}"
        )
    for path in (_ENGINE_PATH, _BOARD_GEOMETRY_PATH):
        dirty = _git("diff", "--quiet", "HEAD", "--", path)
        if dirty.returncode == 1:
            problems.append(f"uncommitted changes under {path}")
        elif dirty.returncode not in (0, 1):
            _LOG.warning("cross-arch: could not check working-tree cleanliness of %s", path)

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
