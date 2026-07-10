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
import sys
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
) -> GameRecord:
    """Confirm / downgrade an accepted record's placement-order flag against the LOG
    (step6 §3.1). ``cross_check`` stamps ``placement_order_established`` from the
    grant-only signal; this adds the LOG-side ordinal (the grant must follow each
    player's 2nd settlement) via
    :func:`~catan_rl.human_data.orientation.establish_placement_order`. Reuses the
    canonical establisher so the order is the LOG's, never the localizer's."""
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
        return replace(record, openings=ordered)
    return replace(
        record,
        provenance={**record.provenance, PROVENANCE_PLACEMENT_ORDER_ESTABLISHED: False},
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
) -> SpikeResult:
    """End-to-end: SNAP the localized adjacency -> ORDER from the log -> VALIDATE
    through the existing fail-closed gate. The VLM supplied ONLY the perception; the
    engine IDs, placement order, and joint-flip safety are all deterministic here.

    Returns a :class:`SpikeResult` carrying an accepted :class:`GameRecord` or a
    typed rejection (the snap's ``ambiguous_snap`` reason, or any downstream
    invariant / glyph-anchor rejection from :func:`cross_check`)."""
    opening_result = snap_localized_openings(localized, topology)
    desert = openings_desert_hex if openings_desert_hex is not None else board.desert_hex
    result = cross_check(
        video_id="vlm_spike",
        game_index=1,
        players=dict(players),
        opponent_strength=opponent_strength,
        board=board,
        openings_desert_hex=desert,
        opening_result=opening_result,
        draft_order=draft_order,
        dice_log=dice_log,
        winner=winner,
        resolution=resolution,
        residual_px=board.residual_px,
        topology=topology,
        granted_by_player=granted_by_player,
        ts=ts,
    )
    record = result.record
    if result.accepted and setup_events is not None:
        record = apply_log_placement_order(record, setup_events, granted_by_player, board, topology)
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


def _game_key(video: str, game_index: int) -> str:
    return f"{video}__g{game_index}"


def prepare_frames_from_fixture(out_root: Path = _FRAMES_ROOT) -> Path:
    """Stage the committed GAME-1 frames (the hard accuracy anchor's own video) as a
    spike game dir — the offline, network-free ``prepare-frames`` path. Copies the
    committed post-setup + empty-baseline PNGs and writes a ``meta.json`` carrying
    the board CV read + the fixture's handles / draft order / setup sequence /
    granted resources / true openings so ``localize`` and ``score`` can run without
    the video."""
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
        "openings_truth": fixture["openings"],
        "board_desert_hex": board.desert_hex,
        "board_residual_px": board.residual_px,
        "board_hexes": list(board.hexes),
        "resolution": int(frame.shape[0]),
        "post_setup_png": str(post),
        "empty_baseline_png": str(baseline),
    }
    (game_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return game_dir


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
    from catan_rl.human_data.board_cv import read_board_stable
    from catan_rl.human_data.segment import segment_games

    topology = load_topology()
    frames = harvest._ingest(video, download_gate=None, work_dir=None)
    ctx = harvest._extract_context(video, frames)
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
        board = read_board_stable([f.frame for f in gf.setup_frames])
        game_dir = out_root / _game_key(video, game_index)
        game_dir.mkdir(parents=True, exist_ok=True)
        post = game_dir / "post_setup.png"
        baseline = game_dir / "empty_baseline.png"
        iio.imwrite(post, gf.post_setup_frame.frame)
        iio.imwrite(baseline, gf.empty_baseline)
        meta: dict[str, Any] = {
            "video": video,
            "game_index": game_index,
            "source": "real video (harvest ingest+routing)",
            "players": dict(ctx.players),
            "player_colors": dict(ctx.player_colors),
            "draft_order": list(harvest._draft_order(segment.events, ctx.handles)),
            "dice_log": list(harvest._dice_log(segment.events)),
            "winner": segment.winner,
            "log_setup_sequence": [
                f"{e.actor} {e.kind}" for e in segment.events if e.kind.startswith("setup_")
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


def localize_game(
    game_dir: Path,
    *,
    localizer: Localizer | None = None,
) -> SpikeResult:
    """Run the localize -> snap -> order -> validate pipeline for one prepared game
    dir. ``localizer`` defaults to the :class:`FileLocalizer` reading
    ``localized/<game_key>.json`` (the VLM's output). Uses the game dir's
    ``meta.json`` for the board + log + strength context."""
    meta = json.loads((game_dir / "meta.json").read_text(encoding="utf-8"))
    topology = load_topology()
    board = _board_from_meta(meta)
    players = dict(meta["players"])
    handles = tuple(players.values())
    if localizer is None:
        localizer = FileLocalizer(_LOCALIZED_ROOT / f"{game_dir.name}.json")
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
        winner=winner if winner in players.values() else None,
        granted_by_player=granted,
        resolution=int(meta["resolution"]),
        topology=topology,
        setup_events=setup_events,
        openings_desert_hex=int(meta["board_desert_hex"]),
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
    result = localize_game(game_dir)
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


def _cmd_score(args: argparse.Namespace) -> int:
    game_dir = _FRAMES_ROOT / args.game
    meta = json.loads((game_dir / "meta.json").read_text(encoding="utf-8"))
    truth = openings_from_dict(meta["openings_truth"])
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
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare-frames", help="extract/stage the two frames per game")
    p_prep.add_argument("--fixture", choices=["game1"], help="stage the committed game-1 anchor")
    p_prep.add_argument("--video", help="real video id (reuses harvest ingest+routing)")
    p_prep.set_defaults(func=_cmd_prepare_frames)

    p_loc = sub.add_parser("localize", help="snap+order+validate via the FileLocalizer")
    p_loc.add_argument("game", help="prepared game key, e.g. game1__g1")
    p_loc.set_defaults(func=_cmd_localize)

    p_score = sub.add_parser("score", help="exact vertex/edge match vs ground truth")
    p_score.add_argument("game", help="prepared game key, e.g. game1__g1")
    p_score.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
