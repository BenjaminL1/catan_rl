"""Named resource-weight tables for the analytic setup-phase scorer.

Plan §A.2 (`docs/plans/v2/setup_strength_roadmap.md`).

Two tables ship with Phase A; §A.4 acceptance gate picks the winner:

  * ``CHARLESWORTH_V0`` — published hand prior. Static constant. Ships
    immediately.
  * ``HEURISTIC_V0`` — NNLS fit against v2 heuristic-vs-heuristic end-game
    resource shares. Lives on disk as JSON, loaded on demand. Generated
    by ``scripts/calibrate_setup_resource_weights.py``.

A third future table (``CHAMPION_V0``) is deferred until a v2 PPO
champion checkpoint exists — see plan §0.2.

The registry function :func:`get_resource_weight_table` is the single
entry point; downstream consumers should never reach for the named
constants directly so we can change the storage format under them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Table 1 — Charlesworth's published prior
# ---------------------------------------------------------------------------

CHARLESWORTH_V0: Final[dict[str, float]] = {
    "WOOD": 1.0,
    "BRICK": 1.0,
    "WHEAT": 1.0,
    "ORE": 1.1,
    "SHEEP": 0.7,
}
"""Charlesworth's hand prior weighting (cited 2026-06-03). Ore is bumped
modestly to reflect its scarcity at 3 hexes; sheep is discounted as the
weakest single-resource for city building."""


# ---------------------------------------------------------------------------
# Table 2 — Heuristic-derived (loaded from JSON on demand)
# ---------------------------------------------------------------------------

_DEFAULT_HEURISTIC_PATH = Path("data/setup_phase/resource_weights_heuristic_v0.json")


def _load_heuristic_v0(path: Path | None = None) -> dict[str, float] | None:
    """Load Table 2 from disk. Returns None if not yet calibrated."""
    p = path if path is not None else _DEFAULT_HEURISTIC_PATH
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    weights = payload.get("weights", payload)
    return {str(k): float(v) for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Final[tuple[str, ...]] = ("charlesworth_v0", "heuristic_v0")


def get_resource_weight_table(
    name: str,
    *,
    heuristic_path: Path | None = None,
) -> dict[str, float]:
    """Return a fresh copy of the named resource-weight table.

    Args:
        name: one of ``"charlesworth_v0"``, ``"heuristic_v0"``.
        heuristic_path: optional override for the heuristic table's
            JSON path (testing).

    Raises:
        ValueError: unknown table name.
        FileNotFoundError: the heuristic table requested but not yet
            calibrated on disk.
    """
    if name == "charlesworth_v0":
        return dict(CHARLESWORTH_V0)
    if name == "heuristic_v0":
        weights = _load_heuristic_v0(heuristic_path)
        if weights is None:
            raise FileNotFoundError(
                "heuristic_v0 weights not on disk — run "
                "`scripts/calibrate_setup_resource_weights.py` first"
            )
        return weights
    raise ValueError(f"unknown weight table {name!r}; known: {_REGISTRY}")


def known_tables() -> tuple[str, ...]:
    """Return the tuple of registered table names."""
    return _REGISTRY
