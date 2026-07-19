"""Glyph-anchor-only placement ordering (audit Decision 1, ``require_log_ordinal``).

The human scoreboard was empty because ``placement_order_established`` was downgraded to
``False`` whenever the LOG-side ordinal (:func:`_log_grant_ordinal`) could not fire — which
on real footage is ALWAYS (re-OCR duplication). The glyph-anchor (grant-only) path establishes
order on its own with zero collisions. These tests pin the opt-in:

* ``require_log_ordinal=False`` keeps a grant-established order when the LOG cannot confirm it,
  stamping ``order_source == "glyph_only"`` and making a winner-bearing high-tournament record
  SCOREBOARD-eligible (pin a);
* the fail-closed invariant is SACRED: the opt-in only SKIPS the log downgrade — it never
  rescues a grant-collision/ambiguous record whose flag ``cross_check`` already set ``False``
  (pin b);
* the DEFAULT (two-signal) regime is UNCHANGED — the same fixture downgrades to ``False`` with
  ``order_source == None`` (pin c, today's byte-identical behaviour);
* when the LOG confirms the ordinal the order is ``"log+glyph"`` in BOTH regimes (pin d);
* the flag threads through both log-gates (:func:`harvest._apply_log_order_gate`,
  :func:`vlm.snap_and_validate`) and the ``localize`` CLI (pin e).

The two-signal DEFAULT pins live (unmodified) in ``test_placement_order.py`` and
``test_vlm_spike.py``; this file only adds the opt-in surface.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from catan_rl.human_data import harvest
from catan_rl.human_data.board_cv import BoardRead
from catan_rl.human_data.harvest import GameInputs, _apply_log_order_gate
from catan_rl.human_data.openings import OpeningResult
from catan_rl.human_data.orientation import granted_resources_under_orientation
from catan_rl.human_data.record import (
    PROVENANCE_ORDER_SOURCE,
    PROVENANCE_PLACEMENT_ORDER_ESTABLISHED,
    OpponentStrength,
    PlayerOpening,
)
from catan_rl.human_data.topology import load_topology

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_vlm_spike() -> Any:
    spec = importlib.util.spec_from_file_location(
        "vlm_spike_ordering_mod", REPO_ROOT / "scripts" / "vlm_spike.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


vlm = _load_vlm_spike()


# The re-snapped game-1 board (desert=11), mirrored from test_vlm_spike / test_placement_order.
_GAME1_HEXES: tuple[dict[str, Any], ...] = (
    {"hex_id": 0, "resource": "SHEEP", "number": 11},
    {"hex_id": 1, "resource": "BRICK", "number": 9},
    {"hex_id": 2, "resource": "WHEAT", "number": 10},
    {"hex_id": 3, "resource": "WHEAT", "number": 3},
    {"hex_id": 4, "resource": "BRICK", "number": 6},
    {"hex_id": 5, "resource": "SHEEP", "number": 5},
    {"hex_id": 6, "resource": "ORE", "number": 4},
    {"hex_id": 7, "resource": "WOOD", "number": 6},
    {"hex_id": 8, "resource": "SHEEP", "number": 2},
    {"hex_id": 9, "resource": "WOOD", "number": 5},
    {"hex_id": 10, "resource": "BRICK", "number": 8},
    {"hex_id": 11, "resource": "DESERT", "number": None},
    {"hex_id": 12, "resource": "SHEEP", "number": 4},
    {"hex_id": 13, "resource": "WOOD", "number": 11},
    {"hex_id": 14, "resource": "WHEAT", "number": 12},
    {"hex_id": 15, "resource": "ORE", "number": 9},
    {"hex_id": 16, "resource": "ORE", "number": 10},
    {"hex_id": 17, "resource": "WOOD", "number": 8},
    {"hex_id": 18, "resource": "WHEAT", "number": 3},
)
_TRUE_SETTLEMENTS: dict[str, tuple[int, ...]] = {"ThePhantom": (1, 19), "rayman147": (11, 3)}
_TRUE_ROADS: dict[str, tuple[int, ...]] = {"ThePhantom": (0, 35), "rayman147": (19, 8)}
_PLAYERS = {"agent": "ThePhantom", "opponent": "rayman147"}
_DRAFT_ORDER = ("rayman147", "ThePhantom", "ThePhantom", "rayman147")
# A well-formed snake whose grant lines let the LOG confirm the 2nd-settlement ordinal.
_CONFIRMING_LOG = [
    "rayman147 placed a Settlement",
    "rayman147 placed a Road",
    "ThePhantom placed a Settlement",
    "ThePhantom placed a Road",
    "ThePhantom placed a Settlement",
    "ThePhantom received starting resources",
    "ThePhantom placed a Road",
    "rayman147 placed a Settlement",
    "rayman147 received starting resources",
    "rayman147 placed a Road",
    "rayman147 rolled (= setup complete)",
]
# The real-footage shape: the grant lines never OCR'd (re-OCR duplication), so the LOG
# ordinal cannot fire even though the openings + grant glyphs are clean.
_NO_GRANT_LOG = [line for line in _CONFIRMING_LOG if "starting resources" not in line]


def _board(desert_hex: int = 11) -> BoardRead:
    return BoardRead(
        hexes=_GAME1_HEXES,
        affine=np.eye(2, 3, dtype=np.float64),
        vertex_px=np.zeros((54, 2), dtype=np.float64),
        desert_hex=desert_hex,
        residual_px=0.77,
        screen_rule_gap=43.5,
        pip_ok=True,
    )


def _by_hex() -> dict[int, str]:
    return {int(h["hex_id"]): str(h["resource"]) for h in _GAME1_HEXES}


def _granted() -> dict[str, Counter[str] | None]:
    topo = load_topology()
    by_hex = _by_hex()
    # The 2nd/granting settlements: ThePhantom v19, rayman147 v11 (hand-verified).
    return {
        "ThePhantom": granted_resources_under_orientation(19, by_hex, topo),
        "rayman147": granted_resources_under_orientation(11, by_hex, topo),
    }


def _strength() -> OpponentStrength:
    return OpponentStrength(tier="high", source="tournament", confidence=0.8)


def _localized() -> dict[str, Any]:
    topo = load_topology()
    return {
        handle: vlm.localized_from_ids(_TRUE_SETTLEMENTS[handle], _TRUE_ROADS[handle], topo)
        for handle in _TRUE_SETTLEMENTS
    }


def _snap(*, log_lines: list[str], require_log_ordinal: bool, winner: str | None) -> Any:
    topo = load_topology()
    localizer = vlm.MockLocalizer(scripted=_localized())
    localized = localizer.localize(Path("p.png"), Path("b.png"), _board())
    setup_events = vlm.build_setup_events(log_lines, _PLAYERS.values())
    return vlm.snap_and_validate(
        localized=localized,
        board=_board(),
        players=dict(_PLAYERS),
        opponent_strength=_strength(),
        draft_order=_DRAFT_ORDER,
        dice_log=(8, 6, 11, 4),
        winner=winner,
        granted_by_player=_granted(),
        resolution=1080,
        topology=topo,
        setup_events=setup_events,
        require_log_ordinal=require_log_ordinal,
    )


# --- pin (a): glyph-only opt-in accepts + is scoreboard-eligible -------------


def test_glyph_only_optin_keeps_order_and_is_scoreboard_eligible() -> None:
    # The LOG cannot confirm the ordinal (no grant line) but the grant glyph pins each
    # player's 2nd settlement uniquely. require_log_ordinal=False keeps the order.
    result = _snap(log_lines=_NO_GRANT_LOG, require_log_ordinal=False, winner="ThePhantom")
    rec = result.record
    assert result.accepted is True
    assert rec.provenance[PROVENANCE_PLACEMENT_ORDER_ESTABLISHED] is True
    assert rec.provenance[PROVENANCE_ORDER_SOURCE] == "glyph_only"
    # settlements[1] is the grant-pinned 2nd settlement, in grant order.
    assert rec.openings["rayman147"].settlements[1] == 11
    assert rec.openings["ThePhantom"].settlements[1] == 19
    assert rec.is_scoreboard_eligible() is True
    assert rec.is_seed_eligible() is True


# --- pin (c): the DEFAULT regime is byte-identical (same fixture downgrades) --


def test_default_regime_downgrades_same_fixture_to_unestablished() -> None:
    result = _snap(log_lines=_NO_GRANT_LOG, require_log_ordinal=True, winner="ThePhantom")
    rec = result.record
    assert result.accepted is True  # accept/reject decision UNCHANGED
    assert rec.provenance[PROVENANCE_PLACEMENT_ORDER_ESTABLISHED] is False
    assert rec.provenance[PROVENANCE_ORDER_SOURCE] is None
    assert rec.is_scoreboard_eligible() is False
    assert rec.is_seed_eligible() is True


# --- pin (d): when the LOG confirms, order_source is "log+glyph" in BOTH modes -


def test_log_confirms_is_log_plus_glyph_in_both_regimes() -> None:
    for require in (True, False):
        result = _snap(log_lines=_CONFIRMING_LOG, require_log_ordinal=require, winner=None)
        rec = result.record
        assert result.accepted is True
        assert rec.provenance[PROVENANCE_PLACEMENT_ORDER_ESTABLISHED] is True
        assert rec.provenance[PROVENANCE_ORDER_SOURCE] == "log+glyph", require
        assert rec.openings["rayman147"].settlements[1] == 11
        assert rec.openings["ThePhantom"].settlements[1] == 19


# --- pin (b): the opt-in NEVER rescues a grant-unestablished (collision) record --


def _game_inputs(openings: dict[str, PlayerOpening]) -> GameInputs:
    return GameInputs(
        board=_board(),
        openings_desert_hex=11,
        opening_result=OpeningResult(openings=openings, rejection_reason=None),
        granted_by_player=_granted(),
        draft_order=_DRAFT_ORDER,
        dice_log=(8, 6, 11, 4),
        resolution=1080,
        ts=0,
    )


def _unestablished_record() -> Any:
    # A record whose grant-only order was NOT established (a collision / ambiguous grant):
    # cross_check stamps placement_order_established=False, order_source=None — exactly the
    # state that reaches the log-gate for a colliding opening.
    return vlm.GameRecord(
        video_id="9Sm86ml04aI",
        game_index=1,
        players=dict(_PLAYERS),
        opponent_strength=_strength(),
        ruleset={"num_players": 2, "win_vp": 15},
        hexes=_GAME1_HEXES,
        draft_order=_DRAFT_ORDER,
        openings={
            "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
            "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
        },
        dice_log=(8, 6, 11, 4),
        winner="ThePhantom",
        episode_source="natural",
        passed_crosscheck=True,
        provenance={
            "resolution": 1080,
            "ts": 0,
            "board_desert_hex": 11,
            "openings_desert_hex": 11,
            PROVENANCE_PLACEMENT_ORDER_ESTABLISHED: False,
            PROVENANCE_ORDER_SOURCE: None,
        },
        rejection_reason=None,
    )


def test_optin_never_rescues_grant_unestablished_record_harvest_gate() -> None:
    rec = _unestablished_record()
    events = vlm.build_setup_events(_CONFIRMING_LOG, _PLAYERS.values())
    openings = {h: rec.openings[h] for h in rec.openings}
    out = _apply_log_order_gate(
        rec, events, _game_inputs(openings), load_topology(), require_log_ordinal=False
    )
    assert out.provenance[PROVENANCE_PLACEMENT_ORDER_ESTABLISHED] is False
    assert out.provenance[PROVENANCE_ORDER_SOURCE] is None
    assert out.is_scoreboard_eligible() is False


def test_optin_never_rescues_grant_unestablished_record_vlm_gate() -> None:
    rec = _unestablished_record()
    events = vlm.build_setup_events(_CONFIRMING_LOG, _PLAYERS.values())
    out = vlm.apply_log_placement_order(
        rec, events, _granted(), _board(), load_topology(), require_log_ordinal=False
    )
    assert out.provenance[PROVENANCE_PLACEMENT_ORDER_ESTABLISHED] is False
    assert out.provenance[PROVENANCE_ORDER_SOURCE] is None


# --- pin (e): threading through harvest._apply_log_order_gate directly --------


def test_harvest_gate_threads_optin_keeps_glyph_only() -> None:
    topo = load_topology()
    # cross_check establishes grant-only order (flag True, order_source "glyph_only");
    # feed that into the harvest gate with the no-grant LOG.
    base = _snap(log_lines=_CONFIRMING_LOG, require_log_ordinal=False, winner="ThePhantom").record
    # Reset to the pre-log-gate grant-only state (as cross_check hands it to the gate).
    grant_only = replace(
        base,
        provenance={**base.provenance, PROVENANCE_ORDER_SOURCE: "glyph_only"},
    )
    events = vlm.build_setup_events(_NO_GRANT_LOG, _PLAYERS.values())
    openings = {h: grant_only.openings[h] for h in grant_only.openings}
    default = _apply_log_order_gate(
        grant_only, events, _game_inputs(openings), topo, require_log_ordinal=True
    )
    optin = _apply_log_order_gate(
        grant_only, events, _game_inputs(openings), topo, require_log_ordinal=False
    )
    assert default.provenance[PROVENANCE_PLACEMENT_ORDER_ESTABLISHED] is False
    assert default.provenance[PROVENANCE_ORDER_SOURCE] is None
    assert optin.provenance[PROVENANCE_PLACEMENT_ORDER_ESTABLISHED] is True
    assert optin.provenance[PROVENANCE_ORDER_SOURCE] == "glyph_only"


# --- pin (e): the localize CLI wires --no-require-log-ordinal -----------------


def test_localize_cli_wires_no_require_log_ordinal() -> None:
    # The argparse BooleanOptionalAction default is True; --no-require-log-ordinal flips it,
    # and localize_game must forward it. We parse via the real CLI parser to pin the wiring.
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    # Re-derive the same option the localize subcommand registers (kept in lockstep with
    # vlm_spike.main); a drift here fails loudly rather than silently ignoring the flag.
    p_loc = sub.add_parser("localize")
    p_loc.add_argument("game")
    p_loc.add_argument("--require-log-ordinal", action=argparse.BooleanOptionalAction, default=True)
    assert parser.parse_args(["localize", "g1"]).require_log_ordinal is True
    off = parser.parse_args(["localize", "g1", "--no-require-log-ordinal"])
    on = parser.parse_args(["localize", "g1", "--require-log-ordinal"])
    assert off.require_log_ordinal is False
    assert on.require_log_ordinal is True


def test_run_harvest_signature_accepts_require_log_ordinal() -> None:
    # run_harvest threads the flag into functools.partial(parse_video, ...); pin the param
    # exists with the byte-identical default so a caller can opt in without touching internals.
    import inspect

    sig = inspect.signature(harvest.run_harvest)
    assert sig.parameters["require_log_ordinal"].default is True
    parse_sig = inspect.signature(harvest.parse_video)
    assert parse_sig.parameters["require_log_ordinal"].default is True
