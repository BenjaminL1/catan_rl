#!/usr/bin/env python3
"""VLM opening-localization decision spike (human-corpus program).

BACKGROUND. The classical-CV opening detector
(:func:`catan_rl.human_data.openings.detect_openings_result`, HSV blob detection)
accepted **0 / 31** real games in the Tier-5 re-run: the wall is opening
piece/road detection on real footage (``settlement_ambiguous`` /
``settlement_blob_shortfall`` / ``road_unresolved`` across colours). HYPOTHESIS: a
VLM (vision-language model) can localize settlements/roads from the SAME frames the
pipeline already extracts, where blob detection fails.

THE HYBRID (this harness — perception is the ONLY thing delegated to the VLM):

1. FRAME EXTRACTION reuse — the post-setup frame (8 opening pieces down, robber on
   the desert, pre-first-roll) + the empty-baseline frame are produced by the
   EXISTING :mod:`catan_rl.human_data.harvest` frame-routing / :func:`read_board`
   path. This harness saves those two PNGs so a vision agent can Read them.
2. VLM PERCEPTION (localize) — a :class:`Localizer` localizes each settlement +
   road and describes its TILE ADJACENCY: for a settlement, the set of hex ids its
   corner touches; for a road, the touching-hex set of EACH of its two endpoint
   vertices. The VLM emits **no engine IDs, no order, no flip decision** — only
   perception.
3. DETERMINISTIC SNAP — :func:`snap_localized_openings` snaps each adjacency
   description to the exact engine vertex id (0..53) / edge id (0..71) using the
   committed board topology (:func:`catan_rl.human_data.topology.load_topology` —
   ``vertex_adjacent_hexes`` / ``edge_vertices``). A settlement's hex-set uniquely
   identifies its vertex; a road is the edge between its two snapped endpoints.
   FAIL-CLOSED: an adjacency that matches 0 or >1 vertex is a TYPED reject
   (``ambiguous_snap``), never a guess.
4. ORDER FROM THE LOG — placement order (which settlement is the 2nd /
   resource-granting one, ``settlements[1]``) comes from the log setup-event
   sequence + the granted-card multiset via
   :func:`catan_rl.human_data.orientation.establish_placement_order`, NOT from the
   VLM (the record.py placement-order contract).
5. EXISTING FAIL-CLOSED VALIDATORS — the snapped openings are fused through the
   EXISTING accept/reject gate :func:`catan_rl.human_data.validate.cross_check`,
   which runs the :class:`~catan_rl.human_data.record.GameRecord` invariants
   (exactly 2 settlements + 2 roads/player, distinctness, standard multisets, the
   provenance orientation-binding) AND the NON-OPTIONAL joint-flip glyph firewall
   (:func:`~catan_rl.human_data.orientation.assert_glyph_anchor`). A record is
   emitted on success or a typed rejection on failure.

THE VLM PROXY. In this spike the "VLM" is an **Opus vision subagent** that Reads
the frame PNGs — a faithful frontier-VLM proxy for the question "can a VLM do this
localization?". **This is NOT Gemini / any pinned API**; production would pin a
specific vision API (Gemini / Claude) and have it emit the ``localized/*.json`` the
:class:`FileLocalizer` consumes. The harness is API-agnostic: the localizer is a
pluggable :class:`Localizer` protocol; the deterministic snap / order / validate
math is identical no matter which model produced the adjacency.

GROUND TRUTH. ``tests/fixtures/human_data/game1_openings.json`` (a hand-verified
game, its OWN video — not one of the Tier-5 five) is the hard accuracy anchor; the
:class:`MockLocalizer` proves the snap+validate math by feeding game-1's TRUE
adjacency. The Tier-5 videos are the YIELD test (no ground truth — count
complete-valid openings via the fail-closed validators).

CPU only; never imports ``gui/`` or the training path. ``cv2`` / ``imageio`` are
imported lazily inside the real-run frame path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from catan_rl.human_data.board_cv import BoardRead
from catan_rl.human_data.logparse import LogEvent
from catan_rl.human_data.openings import OpeningResult
from catan_rl.human_data.orientation import establish_placement_order
from catan_rl.human_data.record import (
    PROVENANCE_ORDER_SOURCE,
    PROVENANCE_PLACEMENT_ORDER_ESTABLISHED,
    GameRecord,
    OpponentStrength,
    PlayerOpening,
)
from catan_rl.human_data.topology import Topology, load_topology
from catan_rl.human_data.validate import CrossCheckResult, cross_check

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# (b) Localizer protocol + the two pluggable implementations                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LocalizedSettlement:
    """A VLM-localized settlement, described PURELY by tile adjacency: the set of
    engine hex ids the settlement's corner touches. No engine vertex id — the snap
    recovers that deterministically."""

    hexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LocalizedRoad:
    """A VLM-localized road, described by the tile-adjacency of EACH of its two
    endpoint vertices (``endpoint_a`` / ``endpoint_b`` = the hex ids each endpoint
    corner touches). The snap recovers the two vertices then the edge between them.
    Order of the two endpoints is irrelevant (an edge is unordered)."""

    endpoint_a: tuple[int, ...]
    endpoint_b: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LocalizedPlayer:
    """One player's VLM-localized opening pieces (order-blind — the VLM does not
    order; the log does)."""

    settlements: tuple[LocalizedSettlement, ...]
    roads: tuple[LocalizedRoad, ...]


class Localizer(Protocol):
    """Perception-only contract. Given the two frame PNG paths + the board CV read
    (so the localizer knows the tile layout), return the per-handle localized
    opening pieces described by tile adjacency. Implementations do PERCEPTION ONLY:
    no engine IDs, no placement order, no flip decision."""

    def localize(
        self,
        post_setup_png: Path,
        empty_baseline_png: Path,
        board: BoardRead,
    ) -> dict[str, LocalizedPlayer]: ...


@dataclass(frozen=True, slots=True)
class MockLocalizer:
    """A scripted :class:`Localizer` for deterministic tests. Returns a fixed
    ``{handle: LocalizedPlayer}`` regardless of the frames — used to feed game-1's
    TRUE adjacency and prove the snap+validate math, and to feed ambiguous
    adjacency and prove the fail-closed reject."""

    scripted: dict[str, LocalizedPlayer]

    def localize(
        self,
        post_setup_png: Path,
        empty_baseline_png: Path,
        board: BoardRead,
    ) -> dict[str, LocalizedPlayer]:
        return dict(self.scripted)


@dataclass(frozen=True, slots=True)
class FileLocalizer:
    """A :class:`Localizer` that reads the per-game adjacency JSON the VLM phase
    writes to ``data/human/vlm_spike/localized/<video>__g<idx>.json``. Schema::

        {
          "players": {
            "<handle>": {
              "settlements": [ {"hexes": [2, 9, 10]}, ... ],
              "roads": [ {"endpoints": [[10, 2, 3], [2, 9, 10]]}, ... ]
            },
            ...
          }
        }

    This is the seam a production Gemini/Claude call would fill: the vision model
    emits this JSON, the harness snaps + validates it deterministically.
    """

    path: Path

    def localize(
        self,
        post_setup_png: Path,
        empty_baseline_png: Path,
        board: BoardRead,
    ) -> dict[str, LocalizedPlayer]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return parse_localized_payload(payload)


def parse_localized_payload(payload: Mapping[str, Any]) -> dict[str, LocalizedPlayer]:
    """Parse the FileLocalizer JSON body into ``{handle: LocalizedPlayer}``."""
    out: dict[str, LocalizedPlayer] = {}
    for handle, pieces in payload["players"].items():
        settlements = tuple(
            LocalizedSettlement(hexes=tuple(int(h) for h in s["hexes"]))
            for s in pieces.get("settlements", [])
        )
        roads = tuple(
            LocalizedRoad(
                endpoint_a=tuple(int(h) for h in r["endpoints"][0]),
                endpoint_b=tuple(int(h) for h in r["endpoints"][1]),
            )
            for r in pieces.get("roads", [])
        )
        out[handle] = LocalizedPlayer(settlements=settlements, roads=roads)
    return out


def localized_from_ids(
    settlement_vertices: Iterable[int],
    road_edges: Iterable[int],
    topology: Topology,
) -> LocalizedPlayer:
    """Build the adjacency description a PERFECT localizer would emit for a set of
    engine vertex/edge ids — the tile-adjacency of each settlement corner and each
    road endpoint. Used by the MockLocalizer (feed game-1's true ids) and to render
    a ground-truth ``localized/*.json`` for the FileLocalizer path."""
    settlements = tuple(
        LocalizedSettlement(hexes=tuple(topology.vertex_adjacent_hexes[v]))
        for v in settlement_vertices
    )
    roads: list[LocalizedRoad] = []
    for edge in road_edges:
        a, b = topology.edge_vertices[edge]
        roads.append(
            LocalizedRoad(
                endpoint_a=tuple(topology.vertex_adjacent_hexes[a]),
                endpoint_b=tuple(topology.vertex_adjacent_hexes[b]),
            )
        )
    return LocalizedPlayer(settlements=settlements, roads=tuple(roads))


def localized_player_to_payload(player: LocalizedPlayer) -> dict[str, Any]:
    """Inverse of :func:`parse_localized_payload` for one player (for writing a
    ground-truth ``localized/*.json``)."""
    return {
        "settlements": [{"hexes": list(s.hexes)} for s in player.settlements],
        "roads": [{"endpoints": [list(r.endpoint_a), list(r.endpoint_b)]} for r in player.roads],
    }


# --------------------------------------------------------------------------- #
# (c) SNAP: deterministic adjacency -> engine id, fail-closed                  #
# --------------------------------------------------------------------------- #


def snap_settlement_vertex(hexes: Iterable[int], topology: Topology) -> int | None:
    """The unique engine vertex whose adjacent-hex set equals ``hexes``, or ``None``
    when 0 or >1 vertices match (fail-closed ``ambiguous_snap``).

    A board corner is the vertex adjacent to EXACTLY a given set of hexes. Interior
    corners touch 3 hexes and are unique; the 6 outer board-tip vertices touch a
    single hex and share that one-hex set with a sibling (so a one-hex adjacency is
    inherently ambiguous and correctly rejected — never guessed)."""
    target = frozenset(int(h) for h in hexes)
    matches = [
        v
        for v in range(len(topology.vertex_adjacent_hexes))
        if frozenset(topology.vertex_adjacent_hexes[v]) == target
    ]
    return matches[0] if len(matches) == 1 else None


def snap_road_edge(road: LocalizedRoad, topology: Topology) -> int | None:
    """The unique engine edge between ``road``'s two snapped endpoint vertices, or
    ``None`` (fail-closed ``ambiguous_snap``) when either endpoint is ambiguous or
    no single edge connects them."""
    va = snap_settlement_vertex(road.endpoint_a, topology)
    vb = snap_settlement_vertex(road.endpoint_b, topology)
    if va is None or vb is None or va == vb:
        return None
    pair = {va, vb}
    edges = [e for e, (a, b) in enumerate(topology.edge_vertices) if {a, b} == pair]
    return edges[0] if len(edges) == 1 else None


def snap_localized_openings(
    localized: Mapping[str, LocalizedPlayer], topology: Topology
) -> OpeningResult:
    """Snap every player's localized adjacency to an order-blind
    :class:`~catan_rl.human_data.openings.OpeningResult`. Fail-closed: the FIRST
    piece that snaps ambiguously yields a typed ``ambiguous_snap:{kind}:{handle}``
    rejection (mirroring the openings CV's typed-rejection contract) — the whole
    game is rejected, never a partially-guessed opening."""
    openings: dict[str, PlayerOpening] = {}
    for handle, player in localized.items():
        settlements: list[int] = []
        for s in player.settlements:
            vertex = snap_settlement_vertex(s.hexes, topology)
            if vertex is None:
                return OpeningResult(None, f"ambiguous_snap:settlement:{handle}")
            settlements.append(vertex)
        roads: list[int] = []
        for r in player.roads:
            edge = snap_road_edge(r, topology)
            if edge is None:
                return OpeningResult(None, f"ambiguous_snap:road:{handle}")
            roads.append(edge)
        openings[handle] = PlayerOpening(settlements=tuple(settlements), roads=tuple(roads))
    return OpeningResult(openings, None)


# --------------------------------------------------------------------------- #
# (d) ORDER: from the LOG setup-event sequence, NOT the localizer               #
# --------------------------------------------------------------------------- #


# Canonical human-readable phrase per setup event kind — the inverse of the
# substring grammar :func:`build_setup_events` parses (mirrors the game-1 fixture's
# ``log_setup_sequence`` form). The real-video ``prepare_frames_from_video`` emits
# ``log_setup_sequence`` through this so the grant (``starting_resources``) line is
# carried and re-parses correctly for placement-order establishment.
_SETUP_PHRASE: dict[str, str] = {
    "setup_settlement": "placed a Settlement",
    "setup_road": "placed a Road",
    "starting_resources": "received starting resources",
}


def build_setup_events(sequence: Sequence[str], handles: Iterable[str]) -> list[LogEvent]:
    """Turn a human-readable log setup sequence (the game-1 fixture's
    ``log_setup_sequence`` form — ``"<handle> placed a Settlement / Road"`` /
    ``"<handle> received starting resources"``) into the ordered
    :class:`~catan_rl.human_data.logparse.LogEvent` stream
    :func:`~catan_rl.human_data.orientation.establish_placement_order` consumes. The
    ORDER lives here in the log, never in the localizer."""
    handle_set = list(handles)
    events: list[LogEvent] = []
    for line in sequence:
        low = line.lower()
        actor = next((h for h in handle_set if h.lower() in low), None)
        if "placed a settlement" in low:
            kind = "setup_settlement"
        elif "placed a road" in low:
            kind = "setup_road"
        elif "received starting resources" in low or "starting resources" in low:
            kind = "starting_resources"
        else:
            continue  # a non-setup line (e.g. the "rolled" terminator) — skip
        events.append(LogEvent(kind=kind, actor=actor, text=line))  # type: ignore[arg-type]
    return events


def apply_log_placement_order(
    record: GameRecord,
    setup_events: Sequence[LogEvent],
    granted_by_player: Mapping[str, Counter[str] | None],
    board: BoardRead,
    topology: Topology,
    *,
    require_log_ordinal: bool = True,
) -> GameRecord:
    """Confirm / downgrade an accepted record's placement-order flag against the LOG
    (step6 §3.1). ``cross_check`` stamps ``placement_order_established`` from the
    grant-only (``"glyph_only"``) signal; this layers the LOG-side ordinal (the grant
    must follow each player's 2nd settlement) via
    :func:`~catan_rl.human_data.orientation.establish_placement_order`. Reuses the
    canonical establisher so the order is the LOG's, never the localizer's.

    ``require_log_ordinal`` (audit Decision 1, DEFAULT ``True``) mirrors
    :func:`catan_rl.human_data.harvest._apply_log_order_gate`: when the log confirms
    the ordinal the order stands (openings re-ordered to the LOG+grant order,
    ``order_source = "log+glyph"``); when it cannot, the DEFAULT regime downgrades the
    flag to ``False`` (``order_source`` reset to ``None``) while the opt-in
    (``require_log_ordinal=False``) keeps the grant-only (``"glyph_only"``)
    establishment. The fail-closed early return (flag already not ``True`` from a
    grant-collision/ambiguous ``cross_check``) is regime-independent — the opt-in
    never rescues an unestablished record."""
    from dataclasses import replace

    if record.provenance.get(PROVENANCE_PLACEMENT_ORDER_ESTABLISHED) is not True:
        return record
    readable = {p: g for p, g in granted_by_player.items() if g is not None}
    board_resource_by_hex = {int(h["hex_id"]): str(h["resource"]) for h in board.hexes}
    ordered, log_established = establish_placement_order(
        list(setup_events), record.openings, readable, board_resource_by_hex, topology
    )
    if log_established:
        # ``ordered`` is the LOG+grant placement order (settlements[1] == granting).
        return replace(
            record,
            openings=ordered,
            provenance={**record.provenance, PROVENANCE_ORDER_SOURCE: "log+glyph"},
        )
    if not require_log_ordinal:
        # Opt-in (audit Decision 1): keep the grant-only ("glyph_only") establishment
        # (cross_check already grant-ordered the openings + stamped order_source).
        return record
    return replace(
        record,
        provenance={
            **record.provenance,
            PROVENANCE_PLACEMENT_ORDER_ESTABLISHED: False,
            PROVENANCE_ORDER_SOURCE: None,
        },
    )


# --------------------------------------------------------------------------- #
# (e) VALIDATE: reuse the existing fail-closed accept/reject gate               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SpikeResult:
    """The outcome of :func:`snap_and_validate`: the :class:`CrossCheckResult` from
    the existing gate + the (possibly log-reordered) record. ``accepted`` and
    ``rejection_reason`` are convenience mirrors."""

    cross_check_result: CrossCheckResult
    record: GameRecord

    @property
    def accepted(self) -> bool:
        return self.cross_check_result.accepted

    @property
    def rejection_reason(self) -> str | None:
        return self.record.rejection_reason


def snap_and_validate(
    *,
    localized: Mapping[str, LocalizedPlayer],
    board: BoardRead,
    players: dict[str, str],
    opponent_strength: OpponentStrength,
    draft_order: tuple[str, ...],
    dice_log: tuple[int, ...],
    winner: str | None,
    granted_by_player: dict[str, Counter[str] | None],
    resolution: int,
    topology: Topology,
    setup_events: Sequence[LogEvent] | None = None,
    openings_desert_hex: int | None = None,
    ts: int = 0,
    dice_values_readable: bool = True,
    video_id: str = "vlm_spike",
    game_index: int = 1,
    require_log_ordinal: bool = True,
) -> SpikeResult:
    """End-to-end: SNAP the localized adjacency -> ORDER from the log -> VALIDATE
    through the existing fail-closed gate. The VLM supplied ONLY the perception; the
    engine IDs, placement order, and joint-flip safety are all deterministic here.

    Returns a :class:`SpikeResult` carrying an accepted :class:`GameRecord` or a
    typed rejection (the snap's ``ambiguous_snap`` reason, or any downstream
    invariant / glyph-anchor rejection from :func:`cross_check`).

    ``require_log_ordinal`` (audit Decision 1, DEFAULT ``True``) is passed to
    :func:`apply_log_placement_order`: ``False`` opts into glyph-anchor-only ordering
    (the grant-glyph adjacency alone keeps order established when the log ordinal is
    unavailable; grant collisions/ambiguity still fail closed)."""
    opening_result = snap_localized_openings(localized, topology)
    desert = openings_desert_hex if openings_desert_hex is not None else board.desert_hex
    result = cross_check(
        video_id=video_id,
        game_index=game_index,
        players=dict(players),
        opponent_strength=opponent_strength,
        board=board,
        openings_desert_hex=desert,
        opening_result=opening_result,
        draft_order=draft_order,
        dice_log=dice_log,
        dice_values_readable=dice_values_readable,
        winner=winner,
        resolution=resolution,
        residual_px=board.residual_px,
        topology=topology,
        granted_by_player=granted_by_player,
        ts=ts,
    )
    record = result.record
    if result.accepted and setup_events is not None:
        record = apply_log_placement_order(
            record,
            setup_events,
            granted_by_player,
            board,
            topology,
            require_log_ordinal=require_log_ordinal,
        )
    return SpikeResult(cross_check_result=result, record=record)


# --------------------------------------------------------------------------- #
# (f) SCORE: exact vertex/edge match vs a ground-truth openings dict            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OpeningScore:
    """Per-corpus opening accuracy vs ground truth (order-blind set match)."""

    settlements_total: int
    settlements_correct: int
    roads_total: int
    roads_correct: int
    per_player_exact: dict[str, bool]

    @property
    def all_exact(self) -> bool:
        return (
            self.settlements_correct == self.settlements_total
            and self.roads_correct == self.roads_total
            and all(self.per_player_exact.values())
        )


def score_openings(
    got: Mapping[str, PlayerOpening],
    truth: Mapping[str, PlayerOpening],
) -> OpeningScore:
    """Exact per-player settlement/road SET match of ``got`` vs ``truth`` (order is
    scored separately by the placement-order flag, so this is order-blind — the
    unordered set of the 2 settlements + 2 roads per player)."""
    s_total = s_ok = r_total = r_ok = 0
    per_player: dict[str, bool] = {}
    for handle, truth_open in truth.items():
        got_open = got.get(handle)
        t_setts, t_roads = set(truth_open.settlements), set(truth_open.roads)
        g_setts = set(got_open.settlements) if got_open else set()
        g_roads = set(got_open.roads) if got_open else set()
        s_total += len(t_setts)
        r_total += len(t_roads)
        s_ok += len(t_setts & g_setts)
        r_ok += len(t_roads & g_roads)
        per_player[handle] = t_setts == g_setts and t_roads == g_roads
    return OpeningScore(
        settlements_total=s_total,
        settlements_correct=s_ok,
        roads_total=r_total,
        roads_correct=r_ok,
        per_player_exact=per_player,
    )


def openings_from_dict(payload: Mapping[str, Any]) -> dict[str, PlayerOpening]:
    """Load a ground-truth ``openings`` block (``{handle: {settlements, roads}}``,
    the game-1 fixture shape) into ``{handle: PlayerOpening}``."""
    return {
        handle: PlayerOpening(
            settlements=tuple(int(v) for v in block["settlements"]),
            roads=tuple(int(e) for e in block["roads"]),
        )
        for handle, block in payload.items()
    }


# --------------------------------------------------------------------------- #
# (a) FRAME EXTRACTION reuse + CLI                                             #
# --------------------------------------------------------------------------- #

_FIXTURE_DIR = REPO / "tests" / "fixtures" / "human_data"
_FRAMES_ROOT = REPO / "data" / "human" / "vlm_spike" / "frames"
_LOCALIZED_ROOT = REPO / "data" / "human" / "vlm_spike" / "localized"
_RECORDS_ROOT = REPO / "data" / "human" / "vlm_spike" / "records"
# Ground-truth openings live in a SEPARATE tree the localize/VLM phase is never
# pointed at — NOT co-located in the ``frames/<game>/meta.json`` the localizer
# reads for the tile layout. Keeping the answer out of the localizer's legitimate
# input is what stops a VLM from snooping the correct openings while learning the
# board (only the ``score`` command reads this tree).
_TRUTH_ROOT = REPO / "data" / "human" / "vlm_spike" / "truth"


def _game_key(video: str, game_index: int) -> str:
    return f"{video}__g{game_index}"


def prepare_frames_from_fixture(out_root: Path = _FRAMES_ROOT) -> Path:
    """Stage the committed GAME-1 frames (the hard accuracy anchor's own video) as a
    spike game dir — the offline, network-free ``prepare-frames`` path. Copies the
    committed post-setup + empty-baseline PNGs and writes a ``meta.json`` carrying
    the board CV read + the fixture's handles / draft order / setup sequence /
    granted resources so ``localize`` can run without the video. The ground-truth
    openings are written to a SEPARATE ``truth/<game>.json`` (NOT ``meta.json``) so
    the localize/VLM phase — which reads ``meta.json`` for the tile layout — can
    never snoop the answer; only ``score`` reads the truth tree."""
    import shutil

    import imageio.v3 as iio
    import numpy as np

    from catan_rl.human_data.board_cv import read_board

    fixture = json.loads((_FIXTURE_DIR / "game1_openings.json").read_text(encoding="utf-8"))
    game_dir = out_root / _game_key("game1", 1)
    game_dir.mkdir(parents=True, exist_ok=True)
    post = game_dir / "post_setup.png"
    baseline = game_dir / "empty_baseline.png"
    shutil.copyfile(_FIXTURE_DIR / "game1_postsetup_t247.png", post)
    shutil.copyfile(_FIXTURE_DIR / "game1_empty_baseline_t105.png", baseline)

    frame = np.asarray(iio.imread(post)[..., :3], np.uint8)
    board = read_board(frame)
    if board is None:  # pragma: no cover - the committed frame reads cleanly
        raise RuntimeError("read_board rejected the committed game-1 post-setup frame")

    meta = {
        "video": "game1",
        "game_index": 1,
        "source": "committed fixture (game1_openings.json) — offline anchor",
        "players": {"agent": "ThePhantom", "opponent": "rayman147"},
        "draft_order": fixture["draft_order"],
        "log_setup_sequence": fixture["log_setup_sequence"],
        "granted_resources": fixture["granted_resources"],
        "player_colors": fixture["player_colors"],
        "winner": fixture["winner"],
        "board_desert_hex": board.desert_hex,
        "board_residual_px": board.residual_px,
        "board_hexes": list(board.hexes),
        "resolution": int(frame.shape[0]),
        "post_setup_png": str(post),
        "empty_baseline_png": str(baseline),
    }
    (game_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Ground truth to its OWN tree, deliberately outside the frames dir the VLM reads.
    truth_path = _TRUTH_ROOT / f"{_game_key('game1', 1)}.json"
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(json.dumps(fixture["openings"], indent=2), encoding="utf-8")
    return game_dir


def _pin_ocr_threads() -> None:
    """Pin torch (easyocr's backend) to ONE intra-op thread when the parallel sweep
    driver requests it via ``CATAN_OCR_THREADS=1`` (measured: 6 threads buys only
    1.33x, so N single-thread worker PROCESSES scale far better). Prints a startup
    line so the sweep log can VERIFY the pin took effect inside this OCR process.

    No-op (and no heavy torch import) unless the env flag is set, so the normal
    single-process path is byte-identical.
    """
    if os.environ.get("CATAN_OCR_THREADS") != "1":
        return
    import cv2
    import torch

    torch.set_num_threads(1)
    try:
        # Interop pin must run BEFORE any parallel torch work or it raises
        # RuntimeError. Safe here: main() calls this first, and torch is first
        # imported two lines up — but guard anyway so a future import shuffle
        # degrades to a logged warning, never a crashed worker.
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        print(f"[ocr-threads] WARNING: set_num_interop_threads(1) rejected: {exc}", flush=True)
    cv2.setNumThreads(1)
    print(
        f"[ocr-threads] CATAN_OCR_THREADS=1 -> torch.set_num_threads(1); "
        f"torch.get_num_threads()={torch.get_num_threads()} "
        f"torch.get_num_interop_threads()={torch.get_num_interop_threads()} "
        f"cv2.getNumThreads()={cv2.getNumThreads()}",
        flush=True,
    )


class _FileSemaphore:
    """Cross-PROCESS counting semaphore over a lock directory (N slot files).

    A :class:`threading.Semaphore` cannot cap concurrency across the sweep's worker
    SUBPROCESSES — each video is its own ``python vlm_spike.py`` process. This gate is
    passed to :func:`harvest._ingest_two_pass` as its ``download_gate``, which holds it
    ONLY around ``download_video`` (the network phase), so the CPU/OCR phase still fans
    out to full worker width while at most ``n`` yt-dlp downloads run at once (YouTube
    throttling defence, mirroring ``batch.py``'s ``download_gate`` idea).

    A slot leaked by a worker that crashed mid-download is reclaimed via a pid-liveness
    check, so the sweep can never deadlock on a stale lock.
    """

    def __init__(self, lock_dir: Path, n: int, poll_s: float = 0.25) -> None:
        self._dir = lock_dir
        self._n = max(1, n)
        self._poll = poll_s
        self._held: Path | None = None
        lock_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> _FileSemaphore:
        while True:
            for i in range(self._n):
                slot = self._dir / f"slot{i}.lock"
                try:
                    fd = os.open(slot, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    self._reclaim_if_stale(slot)
                    continue
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._held = slot
                return self
            time.sleep(self._poll)

    def _reclaim_if_stale(self, slot: Path) -> None:
        try:
            pid = int(slot.read_text().strip() or "0")
        except (OSError, ValueError):
            return
        if pid <= 0 or not _pid_alive(pid):
            slot.unlink(missing_ok=True)

    def __exit__(self, *exc: object) -> None:
        if self._held is not None:
            self._held.unlink(missing_ok=True)
            self._held = None


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process (``os.kill(pid, 0)`` liveness probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _download_gate_from_env() -> _FileSemaphore | None:
    """Build the cross-process download gate from the sweep driver's env, or ``None``.

    ``None`` (env unset) keeps the standalone single-process behaviour unchanged.
    """
    lock_dir = os.environ.get("CATAN_DOWNLOAD_LOCK_DIR")
    if not lock_dir:
        return None
    n = int(os.environ.get("CATAN_NET_CONCURRENCY", "2"))
    return _FileSemaphore(Path(lock_dir), n)


def prepare_frames_from_video(
    video: str, out_root: Path = _FRAMES_ROOT
) -> list[Path]:  # pragma: no cover - real-run network path (yt-dlp/ffmpeg)
    """Extract the post-setup + empty-baseline frames for every game window of a
    real video, REUSING the existing harvest frame-routing (download-then-delete
    ingest, board stability, HUD colour binding, segment/grant routing). Saves the
    two PNGs + a ``meta.json`` per game under ``frames/<video>__g<idx>/``.

    Not unit-tested (it needs the network + heavy OCR); it composes the EXISTING
    :mod:`catan_rl.human_data.harvest` functions rather than reinventing extraction.
    """
    import imageio.v3 as iio

    from catan_rl.human_data import harvest
    from catan_rl.human_data.segment import segment_games

    topology = load_topology()
    # Streaming ingest (Phase A OCR w/o pixel retention → Phase B metadata routing →
    # Phase C targeted re-decode): the sparse pass finds the grant lines, a 1 s dense pass
    # re-samples those windows so the >=2-frame grant consensus is not starved. Holds
    # < 1.5 GB/video (was ~7.5 GB) so the corpus sweep runs beside v11 training. SAME path
    # as the harvest driver (`parse_video`), so the two cannot diverge.
    ctx = harvest._ingest_route_and_materialize(
        video, download_gate=_download_gate_from_env(), work_dir=None
    )
    segments = segment_games(ctx.events, list(ctx.handles))
    game_dirs: list[Path] = []
    game_index = 0
    for seg_idx, segment in enumerate(segments):
        if not harvest.ruleset_ok(segment):
            continue
        game_index += 1
        if seg_idx >= len(ctx.game_frames) or ctx.game_frames[seg_idx] is None:
            continue
        gf = ctx.game_frames[seg_idx]
        assert gf is not None
        if gf.post_setup_frame is None:
            # No honest 8-pieces-down frame exists in this window (the game reached a
            # victory but its builds were never sampled — harvest's typed
            # POST_SETUP_UNRESOLVED reject). Emitting the window's END-GAME frame here
            # is exactly the fail-open this pipeline no longer does: skip the game.
            print(f"  {video} g{game_index}: post-setup frame unresolved — skipped")
            continue
        # SAME helper the harvest driver uses (setup_frames, then this game's own
        # in-window frames) so the two paths cannot diverge.
        board = harvest._stable_board_for_game(gf)
        game_dir = out_root / _game_key(video, game_index)
        game_dir.mkdir(parents=True, exist_ok=True)
        post = game_dir / "post_setup.png"
        baseline = game_dir / "empty_baseline.png"
        iio.imwrite(post, gf.post_setup_frame.frame)
        iio.imwrite(baseline, gf.empty_baseline)
        # Consensus grant read — REUSE the harvest multi-frame grant path so the LOG
        # placement order can be established downstream (localize_game +
        # establish_placement_order need the granting player's card multiset). Only
        # the readable (non-None) grants are serialised; an unreadable grant is simply
        # absent, and localize_game treats an absent handle as an unread grant.
        # Record WHY a grant failed, not just that it did — a `glyph_unreadable` reject
        # then explains itself (which gate starved: no line found / line but no glyph
        # boxes / only 1 readable / the reads DISAGREED) instead of needing a bespoke
        # probe per video. The opponent's grant is the one that keeps failing.
        granted_resources: dict[str, Any] = {}
        grant_diag: dict[str, Any] = {}
        for handle in ctx.handles:
            d: dict[str, Any] = {}
            grant = harvest._consensus_grant(handle, gf.grant_frames, ctx.handles, diag=d)
            if grant is not None:
                granted_resources[handle] = dict(grant)
                # Record HOW a non-unanimous grant was accepted (dominant_read /
                # subset_collapse + the collapse events) so a rescue is verifiable
                # from the meta alone; plain-unanimity accepts stay diag-less.
                if "accepted_by" in d:
                    grant_diag[handle] = d
            else:
                grant_diag[handle] = d
        meta: dict[str, Any] = {
            "video": video,
            "game_index": game_index,
            "source": "real video (harvest ingest+routing)",
            "players": dict(ctx.players),
            "player_colors": dict(ctx.player_colors),
            "draft_order": list(harvest._draft_order(segment.events, ctx.handles)),
            "dice_log": list(harvest._dice_log(segment.events)),
            # Colonist renders the rolled dice as FACE GLYPHS, so on real footage the
            # roll VALUES never OCR and dice_log is empty even though rolls happened.
            # Record that honestly (an empty log + this False is an "unread", not a
            # fabricated luck series, and is what lets a completed game still validate).
            "dice_values_readable": harvest._dice_values_readable(
                segment.events, harvest._dice_log(segment.events)
            ),
            "granted_resources": granted_resources,
            "grant_diag": grant_diag,
            "winner": segment.winner,
            "log_setup_sequence": [
                f"{e.actor} {_SETUP_PHRASE[e.kind]}"
                for e in segment.events
                if e.kind in _SETUP_PHRASE
            ],
            "board_desert_hex": None if board is None else board.desert_hex,
            "board_residual_px": None if board is None else board.residual_px,
            "board_hexes": None if board is None else list(board.hexes),
            "resolution": int(gf.post_setup_frame.native_resolution),
            "post_setup_png": str(post),
            "empty_baseline_png": str(baseline),
        }
        (game_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        game_dirs.append(game_dir)
    return game_dirs


def _board_from_meta(meta: Mapping[str, Any]) -> BoardRead:
    """Reconstruct the BoardRead the validators need from a ``meta.json`` (only the
    fields ``cross_check`` reads — hexes / desert / residual / pip_ok)."""
    import numpy as np

    return BoardRead(
        hexes=tuple(dict(h) for h in meta["board_hexes"]),
        affine=np.eye(2, 3, dtype=np.float64),
        vertex_px=np.zeros((54, 2), dtype=np.float64),
        desert_hex=int(meta["board_desert_hex"]),
        residual_px=float(meta["board_residual_px"]),
        screen_rule_gap=float("inf"),
        pip_ok=True,
    )


def _unlocalizable_reason(loc_path: Path) -> str | None:
    """The typed fail-closed reason for a game the VLM did NOT localize: ``None`` when
    a real localization JSON is present. An ABSENT file -> ``unlocalizable:no_frame``
    (the VLM produced nothing); a file with ``{"unlocalizable": "<why>"}`` ->
    ``unlocalizable:<why>`` (the VLM inspected the frame and declined — e.g. the router
    handed back an end-game / non-setup frame)."""
    if not loc_path.exists():
        return "unlocalizable:no_frame"
    payload = json.loads(loc_path.read_text(encoding="utf-8"))
    if "unlocalizable" in payload:
        return f"unlocalizable:{payload['unlocalizable']}"
    return None


def _rejected_spike(
    meta: Mapping[str, Any],
    board: BoardRead,
    players: dict[str, str],
    topology: Topology,
    reason: str,
) -> SpikeResult:
    """Build a fail-closed rejected :class:`SpikeResult` for a game with no localizable
    opening, routing an ``OpeningResult(None, reason)`` through the EXISTING gate so the
    reject is a real :class:`CrossCheckResult` (identical machinery to the CV path)."""
    strength = OpponentStrength(tier="high", source="tournament", confidence=0.8)
    result = cross_check(
        video_id=str(meta.get("video", "vlm_spike")),
        game_index=int(meta.get("game_index", 1)),
        players=dict(players),
        opponent_strength=strength,
        board=board,
        openings_desert_hex=int(meta["board_desert_hex"]),
        opening_result=OpeningResult(None, reason),
        draft_order=tuple(meta["draft_order"]),
        dice_log=tuple(int(d) for d in meta.get("dice_log", [])),
        dice_values_readable=bool(meta.get("dice_values_readable", True)),
        winner=meta.get("winner") if meta.get("winner") in players.values() else None,
        resolution=int(meta["resolution"]),
        residual_px=board.residual_px,
        topology=topology,
        granted_by_player={h: None for h in players.values()},
        ts=0,
    )
    return SpikeResult(cross_check_result=result, record=result.record)


def localize_game(
    game_dir: Path,
    *,
    localizer: Localizer | None = None,
    require_log_ordinal: bool = True,
) -> SpikeResult:
    """Run the localize -> snap -> order -> validate pipeline for one prepared game
    dir. ``localizer`` defaults to the :class:`FileLocalizer` reading
    ``localized/<game_key>.json`` (the VLM's output). Uses the game dir's
    ``meta.json`` for the board + log + strength context.

    ``require_log_ordinal`` (audit Decision 1, DEFAULT ``True``) is threaded straight
    into :func:`snap_and_validate`: ``False`` opts into glyph-anchor-only placement
    ordering (the log ordinal is not required to keep an established order)."""
    meta = json.loads((game_dir / "meta.json").read_text(encoding="utf-8"))
    topology = load_topology()
    board = _board_from_meta(meta)
    players = dict(meta["players"])
    handles = tuple(players.values())
    if localizer is None:
        loc_path = _LOCALIZED_ROOT / f"{game_dir.name}.json"
        # A game the VLM DECLINED to localize (frame carries no clean opening — e.g.
        # the frame-router handed back an end-game / non-setup frame) is a fail-closed
        # TYPED reject, never a guess. Two forms both reject: the localized JSON is
        # absent, or it carries an explicit ``{"unlocalizable": "<reason>"}`` marker.
        reason = _unlocalizable_reason(loc_path)
        if reason is not None:
            return _rejected_spike(meta, board, players, topology, reason)
        localizer = FileLocalizer(loc_path)
    localized = localizer.localize(
        Path(meta["post_setup_png"]), Path(meta["empty_baseline_png"]), board
    )
    granted: dict[str, Counter[str] | None] = {
        handle: (
            Counter(meta["granted_resources"][handle])
            if handle in meta.get("granted_resources", {})
            else None
        )
        for handle in handles
    }
    setup_events = build_setup_events(meta.get("log_setup_sequence", []), handles)
    winner = meta.get("winner")
    dice_log = tuple(int(d) for d in meta.get("dice_log", []))
    # A tournament-source high strength keeps a real winner scoreboard-eligible; the
    # spike only counts complete-valid openings, so the exact tier is immaterial.
    strength = OpponentStrength(tier="high", source="tournament", confidence=0.8)
    return snap_and_validate(
        localized=localized,
        board=board,
        players=players,
        opponent_strength=strength,
        draft_order=tuple(meta["draft_order"]),
        dice_log=dice_log,
        dice_values_readable=bool(meta.get("dice_values_readable", True)),
        winner=winner if winner in players.values() else None,
        granted_by_player=granted,
        resolution=int(meta["resolution"]),
        topology=topology,
        setup_events=setup_events,
        openings_desert_hex=int(meta["board_desert_hex"]),
        video_id=str(meta.get("video", "vlm_spike")),
        game_index=int(meta.get("game_index", 1)),
        require_log_ordinal=require_log_ordinal,
    )


def _cmd_prepare_frames(args: argparse.Namespace) -> int:
    if args.fixture == "game1":
        game_dir = prepare_frames_from_fixture()
        print(f"prepared {game_dir}")
        return 0
    if args.video:
        dirs = prepare_frames_from_video(args.video)
        for d in dirs:
            print(f"prepared {d}")
        return 0
    print("prepare-frames: pass --fixture game1 or --video <id>", file=sys.stderr)
    return 2


def _cmd_localize(args: argparse.Namespace) -> int:
    game_dir = _FRAMES_ROOT / args.game
    if not game_dir.exists():
        print(f"no prepared game dir {game_dir}", file=sys.stderr)
        return 2
    result = localize_game(game_dir, require_log_ordinal=args.require_log_ordinal)
    _RECORDS_ROOT.mkdir(parents=True, exist_ok=True)
    out = _RECORDS_ROOT / f"{args.game}.json"
    payload: dict[str, Any] = {
        "game": args.game,
        "accepted": result.accepted,
        "rejection_reason": result.rejection_reason,
        "record": result.record.to_dict(),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    verdict = "ACCEPTED" if result.accepted else f"REJECTED ({result.rejection_reason})"
    print(f"{args.game}: {verdict} -> {out}")
    return 0


_REJECT_CATEGORIES = ("ambiguous_snap", "unreadable", "flip", "hud", "board")


def categorize_rejection(reason: str | None) -> str:
    """Fold a typed ``rejection_reason`` into one of the five spike buckets
    (``ambiguous_snap`` / ``unreadable`` / ``flip`` / ``hud`` / ``board``). The VLM
    perception failure is ``ambiguous_snap`` (a piece that snapped to 0 / >1 vertex);
    everything else is a downstream deterministic-gate reject the classical pipeline
    would raise identically."""
    r = reason or ""
    if r.startswith("ambiguous_snap"):
        return "ambiguous_snap"
    if r.startswith("unlocalizable"):
        # The frame-router handed back a frame with no clean opening (end-game / splash
        # / non-setup) — a board/frame-level failure UPSTREAM of piece perception, which
        # a VLM front-end cannot recover. Bucketed with the board failures.
        return "board"
    if "glyph_unreadable" in r:
        return "unreadable"
    if "orientation_joint_flip_glyph_mismatch" in r:
        return "flip"
    if any(
        tok in r
        for tok in (
            "resolution_below",
            "affine_residual",
            "orientation_mismatch_desert",
            "dice",
            "pip",
        )
    ):
        return "board"
    # record-contract / winner-handle / provenance-binding / anything else -> the HUD
    # / record-binding bucket (the fail-closed invariant gate, not the VLM's job).
    return "hud"


def wilson_ci(accepted: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% score interval for ``accepted / total`` (``z`` = 1.96). Returns
    ``(0.0, 0.0)`` for an empty sample."""
    if total == 0:
        return (0.0, 0.0)
    phat = accepted / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = (z * ((phat * (1 - phat) + z * z / (4 * total)) / total) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _cmd_batch_score(args: argparse.Namespace) -> int:
    """Snap + validate EVERY localized game and report the yield: accepted / seen,
    Wilson-95 CI, and the typed-rejection breakdown. Accepted non-ground-truth games
    (no ``truth/<key>.json``) are appended to ``provisional_openings.jsonl`` FLAGGED
    for user hand-verification — never treated as verified."""
    localized_dir = _LOCALIZED_ROOT
    keys = sorted(
        {p.stem.removesuffix("") for p in localized_dir.glob("*.json")}
        | {p.name for p in _FRAMES_ROOT.iterdir() if (p / "meta.json").exists()}
    )
    # YIELD is measured over the REAL-VIDEO population only; the hand-verified fixture
    # anchor (game1, the only game with a committed ``truth/*.json``) is the accuracy
    # anchor, NOT a yield sample, so it is excluded from accepted/seen.
    seen = 0
    accepted = 0
    breakdown: Counter[str] = Counter()
    provisional_lines: list[str] = []
    per_game: list[dict[str, Any]] = []
    for key in keys:
        game_dir = _FRAMES_ROOT / key
        if not (game_dir / "meta.json").exists():
            continue
        result = localize_game(game_dir)
        has_truth = (_TRUTH_ROOT / f"{key}.json").exists()
        if not has_truth:
            seen += 1
        row: dict[str, Any] = {
            "game": key,
            "accepted": result.accepted,
            "rejection_reason": result.rejection_reason,
            "category": None if result.accepted else categorize_rejection(result.rejection_reason),
            "ground_truth": has_truth,
        }
        per_game.append(row)
        if has_truth:
            continue  # the accuracy anchor — scored by `score`, not part of the yield
        if result.accepted:
            accepted += 1
            provisional_lines.append(
                json.dumps(
                    {
                        "game": key,
                        "flagged": "PROVISIONAL — NOT hand-verified",
                        "openings": {
                            h: {"settlements": list(o.settlements), "roads": list(o.roads)}
                            for h, o in result.record.openings.items()
                        },
                        "placement_order_established": result.record.provenance.get(
                            "placement_order_established"
                        ),
                        "winner": result.record.winner,
                    }
                )
            )
        else:
            breakdown[categorize_rejection(result.rejection_reason)] += 1

    lo, hi = wilson_ci(accepted, seen)
    prov_path = REPO / "data" / "human" / "vlm_spike" / "provisional_openings.jsonl"
    prov_body = "\n".join(provisional_lines) + ("\n" if provisional_lines else "")
    prov_path.write_text(prov_body, encoding="utf-8")
    summary = {
        "seen": seen,
        "accepted": accepted,
        "yield": (accepted / seen) if seen else 0.0,
        "wilson95": [lo, hi],
        "rejection_breakdown": {c: breakdown.get(c, 0) for c in _REJECT_CATEGORIES},
        "per_game": per_game,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    game_dir = _FRAMES_ROOT / args.game
    truth_path = _TRUTH_ROOT / f"{args.game}.json"
    if not truth_path.exists():
        print(
            f"{args.game}: no ground-truth openings (truth/{args.game}.json absent) — "
            "nothing to score (real videos have no hand-verified answer)",
            file=sys.stderr,
        )
        return 2
    truth = openings_from_dict(json.loads(truth_path.read_text(encoding="utf-8")))
    result = localize_game(game_dir)
    if not result.accepted:
        print(f"{args.game}: REJECTED ({result.rejection_reason}) — no openings to score")
        return 1
    score = score_openings(result.record.openings, truth)
    print(
        f"{args.game}: settlements {score.settlements_correct}/{score.settlements_total} "
        f"roads {score.roads_correct}/{score.roads_total} "
        f"all_exact={score.all_exact} per_player={score.per_player_exact}"
    )
    return 0 if score.all_exact else 1


def main(argv: Sequence[str] | None = None) -> int:
    _pin_ocr_threads()  # pin torch to 1 thread INSIDE this OCR process when asked
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare-frames", help="extract/stage the two frames per game")
    p_prep.add_argument("--fixture", choices=["game1"], help="stage the committed game-1 anchor")
    p_prep.add_argument("--video", help="real video id (reuses harvest ingest+routing)")
    p_prep.set_defaults(func=_cmd_prepare_frames)

    p_loc = sub.add_parser("localize", help="snap+order+validate via the FileLocalizer")
    p_loc.add_argument("game", help="prepared game key, e.g. game1__g1")
    p_loc.add_argument(
        "--require-log-ordinal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "require the LOG setup-event ordinal (the grant follows each player's 2nd "
            "settlement) to keep placement order established (DEFAULT). "
            "--no-require-log-ordinal opts into glyph-anchor-only ordering (audit "
            "Decision 1): the grant-glyph adjacency alone keeps order established when "
            "the log ordinal is unavailable (re-OCR duplication on real footage); "
            "grant collisions/ambiguity still fail closed."
        ),
    )
    p_loc.set_defaults(func=_cmd_localize)

    p_score = sub.add_parser("score", help="exact vertex/edge match vs ground truth")
    p_score.add_argument("game", help="prepared game key, e.g. game1__g1")
    p_score.set_defaults(func=_cmd_score)

    p_batch = sub.add_parser(
        "batch-score", help="yield over every localized game (accepted/seen + Wilson CI)"
    )
    p_batch.set_defaults(func=_cmd_batch_score)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
