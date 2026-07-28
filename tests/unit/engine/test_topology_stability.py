"""Vertex/edge INDEX stability under a window-size change.

``catanBoard.size`` sets the hex layout origin to ``(width/2, height/2)`` while
``edgeLength`` is independent of it, so resizing the window TRANSLATES the whole
lattice. Indices, however, are assigned by first-occurrence dedup on *rounded
pixel equality*, and the committed ``topology.json`` (plus every vertex-keyed
record in the localized human corpus, and the TS engine's baked fixture) depends
on that numbering never moving.

This is the hard gate on any change to ``catanBoard.size``: if a resize renumbers
a single vertex, the corpus is silently invalidated with no other symptom.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_EXPORT = _REPO / "scripts" / "export_topology.py"
_COMMITTED = _REPO / "src" / "catan_rl" / "human_data" / "topology.json"


def _live_topology() -> dict:
    """Re-derive the topology from a live board via the canonical exporter."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO / "src")
    env["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    out = subprocess.run(
        [sys.executable, str(_EXPORT)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    return json.loads(out)


def test_all_five_topology_tables_match_the_committed_fixture() -> None:
    live = _live_topology()
    committed = json.loads(_COMMITTED.read_text(encoding="utf-8"))
    assert set(live) == set(committed)
    for key in committed:
        assert live[key] == committed[key], f"{key} renumbered — the human corpus is invalid"


def test_the_lattice_matches_the_cv_template_up_to_translation() -> None:
    """``engine_template.json`` is shape-only: its consumer fits scale+offset per
    frame (``human_data/board_cv``), so a window resize is legal for it iff the
    live lattice differs from the template by a single constant offset."""
    from catan_rl.engine.board import catanBoard

    payload = json.loads(
        files("catan_rl.human_data").joinpath("engine_template.json").read_text(encoding="utf-8")
    )
    board = catanBoard()

    tmpl_hex = np.array([payload["hex_centers"][str(i)] for i in range(19)], float)
    tmpl_vtx = np.array([payload["vertex_px"][str(i)] for i in range(54)], float)
    live_hex = np.array(
        [[p.x, p.y] for p in (board.hexTileDict[i].to_pixel(board.flat) for i in range(19))],
        float,
    )
    vpx = board.vertex_index_to_pixel_dict
    live_vtx = np.array([[vpx[i].x, vpx[i].y] for i in range(54)], float)

    offset = live_hex[0] - tmpl_hex[0]
    assert np.allclose(live_hex - tmpl_hex, offset, atol=1e-6), "hex lattice is not a translation"
    assert np.allclose(live_vtx - tmpl_vtx, offset, atol=1e-6), "vertices are not a translation"
    # And the offset really is the window enlargement, not an accidental rescale.
    assert math.isclose(offset[0], (board.width - 1000) / 2.0, abs_tol=1e-6)
    assert math.isclose(offset[1], (board.height - 800) / 2.0, abs_tol=1e-6)
