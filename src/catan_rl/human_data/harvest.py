"""End-to-end HARVEST driver for the human-data video-parsing pipeline (§4 / step6).

This is the missing e2e piece that turns ThePhantom's YouTube corpus into the
:class:`~catan_rl.human_data.record.GameRecord` JSONL dataset. It wires the full
per-game chain the individual modules implement:

    ingest (download-then-delete, two-pass frame sampling)
      -> logparse (events + winner)         [Stage 1]
      -> segment  (game boundaries + ruleset filter + manifest strength tier)
      -> board_cv (orientation-locked, cross-frame-stable board)   [Stage 2]
      -> openings (piece detect + snap + HUD colours)
      -> PLACEMENT-ORDER establishment from the log setup sequence (step6 §3.1)
      -> glyph anchor (multi-frame CONSENSUS grant read)
      -> validate/cross-check (typed accept / reject)
      -> GameRecord
      -> batch ledger (resumable atomic appends; corpus.jsonl + rejected.jsonl)

Design — the driver owns the ORCHESTRATION GLUE; the per-video CV/OCR reads are
INJECTED (exactly as :func:`catan_rl.human_data.batch.run_batch` injects its
``parse_fn`` — "the real download + ffmpeg + OCR pipeline is out of that slice's
scope"). Three module-level stage functions are the seam, so the game-loop glue is
unit-testable with mocked stages (they are looked up on this module at call time,
so a test monkeypatches ``harvest._ingest`` / ``harvest._extract_context`` /
``harvest._read_game_inputs``):

- :func:`_ingest` — download → two-pass sample → in-memory frames (delete on exit).
- :func:`_extract_context` — Stage-1 tail: player handles, per-game player→colour
  binding + seat order, the parsed event stream, and per-game frame routing.
- :func:`_read_game_inputs` — Stage-2 CV: the cross-frame-stable board, the HUD-keyed
  openings, the multi-frame CONSENSUS grant read, and the per-game draft order +
  dice log. Every CV failure is surfaced as a typed reject
  (:class:`GameInputs` carrying an :class:`OpeningResult` rejection reason over a
  legal placeholder board) so the §5.6 rejection-bias audit gets a loadable row —
  never a silent drop.

**Fail-closed semantics are preserved.** The whole run is HARD-GATED on a validated
glyph classifier by :func:`run_batch` (which calls
:func:`~catan_rl.human_data.orientation.assert_scale_up_orientation_gates` once per
run with the fingerprint-gated
:func:`~catan_rl.human_data.glyph_anchor.load_glyph_validation` result the CLI
supplies); "the joint-flip anchor actually ran for both players" is an explicit
precondition of acceptance inside
:func:`~catan_rl.human_data.validate.cross_check`; an unreadable grant is a typed
reject (:data:`~catan_rl.human_data.validate.GLYPH_UNREADABLE_REASON`); and a game
whose LOG placement order cannot be established is FLAGGED order-unestablished
(EVAL-excluded, still seed-eligible) per the step6 §3.1 record contract, never
promoted into the scoreboard.

CPU-only; never imports ``gui/`` or the training path.
"""

from __future__ import annotations

import functools
import json
import re
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from catan_rl.human_data.batch import (
    BatchResult,
    VideoParseError,
    _anchor_telemetry,
    run_batch,
)
from catan_rl.human_data.board_cv import BoardRead, read_board_stable
from catan_rl.human_data.glyph_anchor import (
    consensus_granted_glyphs,
    detect_glyph_boxes,
)
from catan_rl.human_data.ingest import DecodedFrame, ingest_video
from catan_rl.human_data.logparse import LogEvent, crop_log, ocr_log_crop, parse_log
from catan_rl.human_data.openings import OpeningResult, detect_openings_result, read_hud_seat_colors
from catan_rl.human_data.orientation import MIN_RESOLUTION, establish_placement_order
from catan_rl.human_data.record import (
    PROVENANCE_PLACEMENT_ORDER_ESTABLISHED,
    GameRecord,
    OpponentStrength,
)
from catan_rl.human_data.segment import (
    load_strength_manifest,
    ruleset_ok,
    segment_games,
    segment_opponent_strength,
)
from catan_rl.human_data.topology import Topology, load_topology
from catan_rl.human_data.validate import cross_check

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


#: Typed reject reason for a game whose Stage-2 board CV never produced a
#: cross-frame-stable board — there are no archetype features to bucket, so the
#: §5.6 audit row loads over a legal placeholder board carrying this reason.
BOARD_UNREADABLE_REASON = "board_unreadable"

#: Typed reject reason for a game window that segmented but whose sampled frames
#: could not be routed to it (the frame router produced fewer game groups than the
#: event stream produced windows). A loadable audit row, never a silent drop.
FRAMES_UNROUTED_REASON = "frames_unrouted"

#: Roll-value extractor: ``"ThePhantom rolled a 8"`` → ``8``. The rolled total is
#: the only per-game dice-luck covariate (brief §5.4); a roll line whose number
#: does not OCR is skipped rather than fabricated.
_ROLL_RE = re.compile(r"rolled\D*(\d+)")

#: A legal standard 19-tile board (desert at hex 11) used as the placeholder board
#: for a Stage-2 reject that has no readable board of its own. Its only job is to
#: make the §5.6 audit row LOAD; the record is ``passed_crosscheck=False`` and never
#: seed/scoreboard-eligible, so it is never consumed as archetype data. Mirrors the
#: placeholder in :mod:`catan_rl.human_data.validate`.
_PLACEHOLDER_HEXES: tuple[dict[str, Any], ...] = (
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
_PLACEHOLDER_DESERT_HEX = 11


# --- per-game / per-video parse artefacts ------------------------------------


@dataclass(frozen=True, slots=True)
class GameFrames:
    """The sampled frames routed to ONE game (its slot is index-aligned with ``segment_games``).

    ``setup_frames`` are the ≥2 setup-window frames the board-stability gate agrees
    across; ``post_setup_frame`` is the order-blind 8-pieces-down frame the openings
    CV reads; ``empty_baseline`` the no-pieces board (the green-tile-subtraction
    source, §5.13); ``grant_frames`` every frame the ``"received starting
    resources"`` line is visible on (the multi-frame CONSENSUS grant read).
    """

    setup_frames: tuple[DecodedFrame, ...]
    post_setup_frame: DecodedFrame
    empty_baseline: npt.NDArray[np.uint8]
    grant_frames: tuple[DecodedFrame, ...]


@dataclass(frozen=True, slots=True)
class VideoContext:
    """Stage-1-tail facts one video contributes, shared across its games.

    ``players`` is the ``{"agent": handle, "opponent": handle}`` binding the record
    contract requires (ThePhantom is always the POV/agent seat); ``handles`` the two
    raw player handles; ``player_colors`` the per-game ``{handle: colour}`` HUD
    binding (§5.14); ``seat_order`` the handle order top→bottom the HUD assignment
    check pairs with; ``events`` the whole-video parsed :class:`LogEvent` stream;
    ``game_frames`` the per-game frame routing (index-aligned with
    :func:`~catan_rl.human_data.segment.segment_games` — one entry per segment,
    ``None`` for a segment that gathered too few frames to gate board stability).
    """

    players: dict[str, str]
    handles: tuple[str, str]
    player_colors: dict[str, str]
    seat_order: tuple[str, ...]
    events: tuple[LogEvent, ...]
    game_frames: tuple[GameFrames | None, ...]


@dataclass(frozen=True, slots=True)
class GameInputs:
    """The Stage-2 CV artefacts feeding :func:`~catan_rl.human_data.validate.cross_check`.

    On any Stage-2 CV failure the reader returns a placeholder ``board`` + an
    ``opening_result`` carrying the typed reject reason (see :func:`_reject_inputs`),
    so the game-loop glue calls ``cross_check`` uniformly and every game — accepted
    or rejected — emits a loadable §5.6 audit row.
    """

    board: BoardRead
    openings_desert_hex: int
    opening_result: OpeningResult
    granted_by_player: dict[str, Counter[str] | None]
    draft_order: tuple[str, ...]
    dice_log: tuple[int, ...]
    resolution: int
    ts: int


# --- run-level telemetry -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class HarvestTelemetry:
    """Corpus-state telemetry for a harvest run (printed + written per run).

    Derived by scanning the on-disk ``corpus.jsonl`` + ``rejected.jsonl`` after the
    batch, so it reflects the FULL audit-relevant corpus (resume-consistent — a
    resumed run sees the prior run's rows too). The anchor / grant-coverage counters
    are reason-derived per record (:func:`catan_rl.human_data.batch._anchor_telemetry`),
    exactly as :class:`~catan_rl.human_data.batch.BatchResult` derives them for a
    single run.

    - ``games_seen`` — accepted + rejected records on disk (one per game window that
      reached ``cross_check``; ruleset-filtered non-game windows never get here).
    - ``accepted`` / ``rejected`` — the corpus / rejected split.
    - ``rejected_by_reason`` — ``{rejection_reason: count}`` over ``rejected.jsonl``.
    - ``anchor_ran`` / ``anchor_unreadable`` / ``anchor_mismatch`` — how often the
      joint-flip glyph anchor executed / could-not-run / fired.
    - ``order_unestablished`` — ACCEPTED records whose LOG placement order was not
      established (EVAL-excluded, still seed-eligible; step6 §3.1).
    - ``grant_read_coverage`` — ``anchor_ran / games_seen`` (0.0 when empty).
    - ``videos_processed`` / ``videos_skipped`` / ``videos_failed`` — from the batch.
    """

    games_seen: int = 0
    accepted: int = 0
    rejected: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    anchor_ran: int = 0
    anchor_unreadable: int = 0
    anchor_mismatch: int = 0
    order_unestablished: int = 0
    grant_read_coverage: float = 0.0
    videos_processed: int = 0
    videos_skipped: int = 0
    videos_failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "games_seen": self.games_seen,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejected_by_reason": dict(self.rejected_by_reason),
            "anchor_ran": self.anchor_ran,
            "anchor_unreadable": self.anchor_unreadable,
            "anchor_mismatch": self.anchor_mismatch,
            "order_unestablished": self.order_unestablished,
            "grant_read_coverage": self.grant_read_coverage,
            "videos_processed": self.videos_processed,
            "videos_skipped": self.videos_skipped,
            "videos_failed": self.videos_failed,
        }

    def render(self) -> str:
        """A compact multi-line human summary (printed by the CLI)."""
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.rejected_by_reason.items())) or "—"
        return (
            f"[harvest] videos: processed={self.videos_processed} "
            f"skipped={self.videos_skipped} failed={self.videos_failed}\n"
            f"[harvest] games_seen={self.games_seen} accepted={self.accepted} "
            f"rejected={self.rejected} order_unestablished={self.order_unestablished}\n"
            f"[harvest] anchor: ran={self.anchor_ran} unreadable={self.anchor_unreadable} "
            f"mismatch={self.anchor_mismatch} grant_read_coverage={self.grant_read_coverage:.3f}\n"
            f"[harvest] rejected_by_reason: {reasons}"
        )


# --- Stage-2 pure helpers (draft order + dice log from the log stream) --------


def _draft_order(events: Sequence[LogEvent], handles: tuple[str, str]) -> tuple[str, ...]:
    """The length-4 snake draft order ``(a, b, b, a)`` for the two handles.

    ``a`` is the FIRST player to place a setup settlement (snake 1→2→2→1 places
    ``a`` first and last); ``b`` is the other handle. When no setup-settlement line
    resolved (a missing / unsampled draft), falls back to sorted handles — the
    resulting record is order-unestablished (the log-side ordinal cannot fire) and
    so EVAL-excluded regardless, and this only keeps the record CONTRACT-valid
    (``draft_order`` must be a snake) so the §5.6 audit row still loads.
    """
    first: str | None = None
    for event in events:
        if event.kind == "setup_settlement" and event.actor in handles:
            first = event.actor
            break
    if first is None:
        first = min(handles)
    other = handles[1] if first == handles[0] else handles[0]
    return (first, other, other, first)


def _dice_log(events: Sequence[LogEvent]) -> tuple[int, ...]:
    """The ordered per-game dice totals read from the ``"rolled"`` log lines (§5.4).

    A roll line whose total does not OCR to a legal 2..12 is skipped rather than
    fabricated; the sampled multi-frame stream may re-OCR a roll across frames, so
    this is a plausible over-count, never a fabricated luck series.
    """
    rolls: list[int] = []
    for event in events:
        if event.kind != "roll":
            continue
        match = _ROLL_RE.search(event.text.lower())
        if match is None:
            continue
        value = int(match.group(1))
        if 2 <= value <= 12:
            rolls.append(value)
    return tuple(rolls)


def _placeholder_board() -> BoardRead:
    """A legal placeholder :class:`BoardRead` for a board-less Stage-2 reject."""
    import numpy as np

    return BoardRead(
        hexes=_PLACEHOLDER_HEXES,
        affine=np.eye(2, 3, dtype=np.float64),
        vertex_px=np.zeros((54, 2), dtype=np.float64),
        desert_hex=_PLACEHOLDER_DESERT_HEX,
        residual_px=0.0,
        screen_rule_gap=float("inf"),
        pip_ok=True,
    )


def _reject_inputs(events: Sequence[LogEvent], handles: tuple[str, str], reason: str) -> GameInputs:
    """A :class:`GameInputs` that forces a typed Stage-2 reject through ``cross_check``.

    The placeholder board passes the coarse resolution/residual/pip pre-screen, then
    ``cross_check`` check-4 carries ``reason`` through from the ``openings=None``
    :class:`OpeningResult` — a loadable §5.6 audit row with the true CV cause.
    """
    return GameInputs(
        board=_placeholder_board(),
        openings_desert_hex=_PLACEHOLDER_DESERT_HEX,
        opening_result=OpeningResult(openings=None, rejection_reason=reason),
        granted_by_player={},
        draft_order=_draft_order(events, handles),
        dice_log=_dice_log(events),
        resolution=MIN_RESOLUTION,
        ts=0,
    )


# --- injected stages (real defaults; monkeypatched wholesale in unit tests) ---


def _ingest(
    video_id: str,
    *,
    download_gate: threading.BoundedSemaphore | None = None,
    work_dir: Path | None = None,
) -> list[DecodedFrame]:
    """Download → two-pass sample → in-memory frames for one video (delete on exit).

    Wraps :func:`~catan_rl.human_data.ingest.ingest_video` (download-then-delete, no
    PNG accumulation). The video duration bounds the sparse pass; it is probed once
    via yt-dlp metadata (:func:`_probe_duration_s`). The ``download_gate`` (when
    supplied by :func:`run_batch`) caps concurrent downloads — held around ONLY the
    network phase so the CPU/OCR phase fans out to the full worker width (§5.11).
    """
    duration_s = _probe_duration_s(video_id)
    gate = download_gate if download_gate is not None else _NULL_GATE
    with gate:
        frames = list(ingest_video(video_id, duration_s=duration_s, work_dir=work_dir))
    if not frames:
        raise VideoParseError(f"ingest produced no frames for {video_id!r}")
    return frames


def _probe_duration_s(video_id: str) -> float:
    """Video duration in seconds from yt-dlp metadata (no full download).

    ``yt-dlp --no-download --print duration`` prints the stream duration; a value
    that does not parse is a recoverable per-video failure (marks the video failed,
    retried on resume).
    """
    import subprocess

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        out = subprocess.run(
            ["yt-dlp", "--no-download", "--print", "duration", url],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - network
        raise VideoParseError(f"duration probe failed for {video_id!r}: {exc}") from exc
    token = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    try:
        return float(token)
    except ValueError as exc:  # pragma: no cover - malformed yt-dlp output
        raise VideoParseError(f"duration probe returned {token!r} for {video_id!r}") from exc


def _extract_context(video_id: str, frames: list[DecodedFrame]) -> VideoContext:
    """Stage-1 tail: player handles, HUD colour binding, event stream, frame routing.

    BEST-EFFORT and SAFE under a wrong guess: every downstream firewall
    (HUD-assignment check §5.14, board-stability §5.2, the glyph anchor) REJECTS a
    mislabelled game, so a wrong handle/colour/routing guess produces a typed reject,
    never a confidently-wrong ACCEPT. Handles are discovered from the leading tokens
    of event-shaped OCR lines; the per-game player→colour binding is read from the
    HUD seat rings; frames are routed to games by the ``game_reset`` markers in
    source order (index-aligned with :func:`~catan_rl.human_data.segment.segment_games`).
    """  # pragma: no cover - real-run CV path (exercised on hardware, mocked in CI)
    per_frame_lines = [ocr_log_crop(crop_log(f.frame)) for f in frames]
    all_lines = [line for lines in per_frame_lines for line in lines]
    handles = _discover_handles(all_lines)
    events = parse_log(all_lines, handles).events
    seat_colours = read_hud_seat_colors(frames[len(frames) // 2].frame) if frames else ()
    player_colors, seat_order = _bind_colours(handles, seat_colours)
    game_frames = _route_frames_to_games(frames, per_frame_lines, handles)
    return VideoContext(
        players=_agent_binding(handles),
        handles=handles,
        player_colors=player_colors,
        seat_order=seat_order,
        events=events,
        game_frames=game_frames,
    )


def _read_game_inputs(
    video_id: str,
    segment_index: int,
    segment_events: Sequence[LogEvent],
    ctx: VideoContext,
    topology: Topology,
) -> GameInputs:
    """Stage-2 CV for one game: stable board + HUD-keyed openings + consensus grants.

    ``segment_index`` is the game's position in the FULL ``segment_games`` list —
    the same index ``ctx.game_frames`` is aligned to — so the routed frames belong to
    THIS game (never a ruleset-dropped neighbour's). Composes the exported CV
    functions over the game's routed frames (:class:`GameFrames`). Any CV failure —
    unrouted / too-few frames included — returns a typed reject via
    :func:`_reject_inputs` so the glue's single ``cross_check`` call still emits a
    loadable audit row.
    """  # pragma: no cover - real-run CV path (exercised on hardware, mocked in CI)
    if segment_index >= len(ctx.game_frames):
        return _reject_inputs(segment_events, ctx.handles, FRAMES_UNROUTED_REASON)
    gf = ctx.game_frames[segment_index]
    if gf is None:  # segment gathered < 2 frames — too few to gate board stability
        return _reject_inputs(segment_events, ctx.handles, FRAMES_UNROUTED_REASON)
    board = read_board_stable([f.frame for f in gf.setup_frames])
    if board is None:
        return _reject_inputs(segment_events, ctx.handles, BOARD_UNREADABLE_REASON)
    opening_result = detect_openings_result(
        gf.post_setup_frame.frame,
        gf.empty_baseline,
        board,
        player_colors=ctx.player_colors,
        seat_order=list(ctx.seat_order),
    )
    granted: dict[str, Counter[str] | None] = {}
    for handle in ctx.handles:
        granted[handle] = _consensus_grant(handle, gf.grant_frames)
    return GameInputs(
        board=board,
        openings_desert_hex=board.desert_hex,
        opening_result=opening_result,
        granted_by_player=granted,
        draft_order=_draft_order(segment_events, ctx.handles),
        dice_log=_dice_log(segment_events),
        resolution=gf.post_setup_frame.native_resolution,
        ts=int(gf.post_setup_frame.ts),
    )


# --- game-loop glue (the tested core) ----------------------------------------


def parse_video(
    video_id: str,
    *,
    manifest: dict[str, Any],
    topology: Topology,
    download_gate: threading.BoundedSemaphore | None = None,
    work_dir: Path | None = None,
) -> list[GameRecord]:
    """Parse one video into its per-game :class:`GameRecord` rows (the ``parse_fn``).

    The game-loop glue :func:`run_batch` injects. Runs ingest → context → segment,
    then for EACH ruleset-passing game window:

    1. reads the Stage-2 CV inputs (:func:`_read_game_inputs`);
    2. fuses them through :func:`~catan_rl.human_data.validate.cross_check` (the
       accept/reject gate that also runs the NON-OPTIONAL joint-flip glyph anchor);
    3. for an ACCEPTED game, applies the LOG-side placement-order gate on top of
       ``cross_check``'s grant-only order: if
       :func:`~catan_rl.human_data.orientation.establish_placement_order` cannot
       confirm the grant follows each player's 2nd settlement, the record is
       DOWNGRADED to order-unestablished
       (``provenance[placement_order_established] = False``) — EVAL-excluded, still
       seed-eligible (step6 §3.1), never a silent scoreboard promotion.

    A ruleset-failing window (not exactly two acting seats — a mis-segmented / non-1v1
    window) is NOT a game and is dropped from the game list before CV. Every window
    that reaches ``cross_check`` emits a record (accepted or typed-rejected), so no
    game is silently lost.
    """
    strength = segment_opponent_strength(manifest, video_id)
    if strength is None:
        raise VideoParseError(
            f"{video_id!r} has no manifest opponent strength (should be harvest-gated out)"
        )

    frames = _ingest(video_id, download_gate=download_gate, work_dir=work_dir)
    ctx = _extract_context(video_id, frames)
    segments = segment_games(ctx.events, list(ctx.handles))

    records: list[GameRecord] = []
    game_index = 0
    for seg_idx, segment in enumerate(segments):
        if not ruleset_ok(segment):
            continue  # not a 1v1 game window — pre-filtered, not a record
        game_index += 1
        # ``seg_idx`` (raw position in the FULL segment list) — NOT ``game_index``
        # (the post-ruleset-filter ordinal) — indexes the frame routing: both
        # ``segments`` and ``ctx.game_frames`` are aligned to ``segment_games``, so a
        # ruleset-dropped window ahead of this one must NOT shift the frame lookup
        # (that would weld another game's frames on — a cross-game mismatch no
        # firewall can catch). ``game_index`` remains the record's real-game ordinal.
        gi = _read_game_inputs(video_id, seg_idx, segment.events, ctx, topology)
        result = cross_check(
            video_id=video_id,
            game_index=game_index,
            players=dict(ctx.players),
            opponent_strength=_video_strength(strength),
            board=gi.board,
            openings_desert_hex=gi.openings_desert_hex,
            opening_result=gi.opening_result,
            draft_order=gi.draft_order,
            dice_log=gi.dice_log,
            winner=segment.winner,
            resolution=gi.resolution,
            residual_px=gi.board.residual_px,
            topology=topology,
            granted_by_player=gi.granted_by_player,
            ts=gi.ts,
        )
        record = result.record
        if result.accepted:
            record = _apply_log_order_gate(record, segment.events, gi, topology)
        records.append(record)
    return records


def _apply_log_order_gate(
    record: GameRecord,
    segment_events: Sequence[LogEvent],
    gi: GameInputs,
    topology: Topology,
) -> GameRecord:
    """Downgrade an accepted record to order-unestablished when the LOG cannot confirm
    the placement order (step6 §3.1 — the harvest-side gate on top of ``cross_check``).

    ``cross_check`` stamps ``placement_order_established`` from the grant-only
    (VERTEX-side) signal because it carries no setup-event stream; the harvest driver
    additionally requires the LOG-side ordinal (the grant follows each player's 2nd
    settlement) via :func:`~catan_rl.human_data.orientation.establish_placement_order`.
    When the record is currently ESTABLISHED but the log cannot confirm it, the flag
    is flipped to ``False`` — EVAL-excluded, still seed-eligible. When it is already
    unestablished, nothing changes.
    """
    if record.provenance.get(PROVENANCE_PLACEMENT_ORDER_ESTABLISHED) is not True:
        return record
    openings = gi.opening_result.openings
    if openings is None:  # unreachable on the accept path (cross_check guarantees it)
        return record
    readable = {p: g for p, g in gi.granted_by_player.items() if g is not None}
    board_resource_by_hex = {int(h["hex_id"]): str(h["resource"]) for h in gi.board.hexes}
    _ordered, log_established = establish_placement_order(
        list(segment_events), openings, readable, board_resource_by_hex, topology
    )
    if log_established:
        return record
    return replace(
        record,
        provenance={**record.provenance, PROVENANCE_PLACEMENT_ORDER_ESTABLISHED: False},
    )


def _video_strength(strength: OpponentStrength) -> OpponentStrength:
    """The per-record opponent strength (video-level; identical for every game)."""
    return strength


# --- run orchestration + telemetry -------------------------------------------


def run_harvest(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    video_ids: Sequence[str] | None = None,
    max_workers: int = 4,
    net_concurrency: int = 2,
    work_dir: str | Path | None = None,
    glyph_validation: Any = None,
    now_fn: Any = time.time,
) -> tuple[BatchResult, HarvestTelemetry]:
    """Run the e2e harvest and return ``(BatchResult, HarvestTelemetry)``.

    Binds :func:`parse_video` to the loaded manifest + topology and hands it to
    :func:`~catan_rl.human_data.batch.run_batch` (which owns the resumable ledger,
    atomic corpus/rejected appends, the parallel-per-video pool, and the once-per-run
    glyph-classifier HARD GATE — an absent/failed/unbound ``glyph_validation`` raises
    :class:`~catan_rl.human_data.orientation.GlyphClassifierNotValidated` before any
    video is parsed). Telemetry is then computed from the on-disk corpus.
    """
    manifest = load_strength_manifest(manifest_path)
    topology = load_topology()
    parse_fn = functools.partial(
        parse_video,
        manifest=manifest,
        topology=topology,
        work_dir=Path(work_dir) if work_dir is not None else None,
    )
    result = run_batch(
        manifest_path=manifest_path,
        out_dir=out_dir,
        parse_fn=parse_fn,
        max_workers=max_workers,
        net_concurrency=net_concurrency,
        video_ids=video_ids,
        now_fn=now_fn,
        glyph_validation=glyph_validation,
    )
    telemetry = compute_corpus_telemetry(out_dir, result)
    return result, telemetry


def _read_records(path: Path) -> list[GameRecord]:
    """Load a JSONL corpus file, tolerating a torn/partial trailing line."""
    if not path.exists():
        return []
    out: list[GameRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(GameRecord.from_json_line(line))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue  # torn trailing line / demoted partial (batch._append_line)
    return out


def compute_corpus_telemetry(out_dir: str | Path, result: BatchResult) -> HarvestTelemetry:
    """Compute :class:`HarvestTelemetry` from the on-disk corpus + the batch summary.

    Scans ``corpus.jsonl`` + ``rejected.jsonl`` so the counters reflect the full
    audit-relevant corpus (resume-consistent). Anchor / grant-coverage counters are
    reason-derived per record, identically to
    :class:`~catan_rl.human_data.batch.BatchResult`.
    """
    out_dir = Path(out_dir)
    accepted = _read_records(out_dir / "corpus.jsonl")
    rejected = _read_records(out_dir / "rejected.jsonl")

    by_reason: Counter[str] = Counter(r.rejection_reason or "unspecified" for r in rejected)
    order_unestablished = sum(
        1 for r in accepted if r.provenance.get(PROVENANCE_PLACEMENT_ORDER_ESTABLISHED) is not True
    )
    anchor_ran = anchor_unreadable = anchor_mismatch = 0
    for record in (*accepted, *rejected):
        ran, unreadable, mismatch = _anchor_telemetry(record)
        anchor_ran += int(ran)
        anchor_unreadable += int(unreadable)
        anchor_mismatch += int(mismatch)
    games_seen = len(accepted) + len(rejected)
    return HarvestTelemetry(
        games_seen=games_seen,
        accepted=len(accepted),
        rejected=len(rejected),
        rejected_by_reason=dict(by_reason),
        anchor_ran=anchor_ran,
        anchor_unreadable=anchor_unreadable,
        anchor_mismatch=anchor_mismatch,
        order_unestablished=order_unestablished,
        grant_read_coverage=(anchor_ran / games_seen) if games_seen else 0.0,
        videos_processed=result.videos_processed,
        videos_skipped=result.videos_skipped,
        videos_failed=result.videos_failed,
    )


# --- real-run CV helpers (lazy; exercised on hardware, mocked wholesale in CI) -
#
# These compose easyocr / cv2 / yt-dlp exactly like logparse.py / board_cv.py /
# ingest.py. They run only under the default stages on real video; the unit suite
# monkeypatches the three stage functions above, so none of this executes in CI.

#: A no-op context manager standing in for an absent download gate (a legacy /
#: single-worker call path), so :func:`_ingest` can ``with`` a gate unconditionally.
_NULL_GATE = threading.BoundedSemaphore(1_000_000)

#: Minimum event-shaped lines a leading token must lead to count as a player handle
#: (a chat/log-noise token appears far less often than a real seat's action lines).
_MIN_HANDLE_LINES = 3


def _discover_handles(lines: list[str]) -> tuple[str, str]:  # pragma: no cover - real-run path
    """The two player handles: the two most frequent leading tokens of action lines.

    Colonist action lines lead with the acting seat's handle
    (``"<handle> placed a Settlement"``, ``"<handle> rolled a 8"``). The two seats
    dominate the leading-token histogram over a whole video; a chat/system token
    trails far behind. Raises :class:`VideoParseError` when two dominant handles do
    not emerge (an unreadable log — recoverable, retried on resume).
    """
    action = re.compile(
        r"\b(placed a|rolled|received starting resources|built|bought|moved the robber)\b"
    )
    counts: Counter[str] = Counter()
    for line in lines:
        low = line.strip().lower()
        if not action.search(low):
            continue
        tokens = re.findall(r"[\w.\-]+", line.strip())
        if tokens:
            counts[tokens[0]] += 1
    ranked = [tok for tok, n in counts.most_common() if n >= _MIN_HANDLE_LINES]
    if len(ranked) < 2:
        raise VideoParseError("could not discover two player handles from the log")
    return ranked[0], ranked[1]


def _agent_binding(handles: tuple[str, str]) -> dict[str, str]:  # pragma: no cover - real-run path
    """Bind ``{"agent": ThePhantom-seat, "opponent": other}`` (POV is always ThePhantom)."""
    agent = next((h for h in handles if "phantom" in h.lower()), handles[0])
    opponent = handles[1] if agent == handles[0] else handles[0]
    return {"agent": agent, "opponent": opponent}


def _bind_colours(
    handles: tuple[str, str], seat_colours: tuple[str, ...]
) -> tuple[dict[str, str], tuple[str, ...]]:  # pragma: no cover - real-run path
    """Pair handles to HUD seat colours (top→bottom). A wrong pairing is caught by the
    §5.14 HUD-assignment firewall (the game rejects), never a confidently-wrong accept.
    """
    if len(seat_colours) != 2 or len(set(seat_colours)) != 2:
        raise VideoParseError("HUD did not yield two distinct seat colours")
    seat_order = handles  # top→bottom, paired positionally with the HUD colours
    player_colors = {seat_order[0]: seat_colours[0], seat_order[1]: seat_colours[1]}
    return player_colors, seat_order


def _route_frames_to_games(
    frames: list[DecodedFrame], per_frame_lines: list[list[str]], handles: tuple[str, str]
) -> tuple[GameFrames | None, ...]:  # pragma: no cover - real-run path
    """Route sampled frames to games using the SAME boundaries ``segment_games`` computes.

    The frame router MUST agree with the event-side segmentation it is later indexed
    against (:func:`parse_video` walks ``segment_games`` and looks each game's frames
    up by its segment index). A naive per-frame reset regex RE-IMPLEMENTS — and can
    DISAGREE with — the reset-hygiene rules :func:`~catan_rl.human_data.segment.segment_games`
    embodies (consecutive-reset runs, a lingering re-OCR'd reset, the byte-identical
    merge decision): a router split where the segmenter merges (or vice versa) welds
    one game's frames onto another — a cross-game mismatch no downstream firewall can
    catch. So the boundaries come from ``segment_games`` itself.

    :func:`~catan_rl.human_data.logparse.parse_log` is per-line stateless, so parsing
    each frame's lines and concatenating reproduces the exact event stream
    :func:`_extract_context` builds; ``segment_games`` windows are contiguous and cover
    the stream except the dropped pre-first-reset prefix, so each segment's event-index
    range is reconstructed from the window lengths alone — never a re-derived reset scan.
    Each frame is assigned to the segment its events overlap most (ties → the later
    segment); a frame that only re-OCRs pre-first-reset noise is dropped.

    Returns a tuple index-aligned with ``segment_games(events, handles)`` (one slot per
    segment). A segment that gathered < 2 frames — too few to gate board stability — is
    ``None`` (its game reads out as ``frames_unrouted``, a typed reject), never dropped,
    so the segment index stays aligned. Within a game the first frames are the setup
    window (board-stability source), the first is the empty baseline, the post-setup
    read is the latest frame still showing the 8-pieces-down opening board — the last
    frame BEFORE the game's first main-game build, NOT the game's last (end-game) frame
    (:func:`_post_setup_frame`) — and every frame carrying a grant line feeds the
    consensus grant read.
    """
    per_frame_events = [parse_log(lines, handles).events for lines in per_frame_lines]
    all_events = tuple(e for evs in per_frame_events for e in evs)
    segments = segment_games(all_events, list(handles))
    n_seg = len(segments)
    if n_seg == 0:
        return ()

    # segment_games windows are contiguous and cover all_events[start0:], dropping only
    # the pre-first-reset prefix, so the (lo, hi) event-index range of each segment is
    # reconstructed from the window LENGTHS — the boundaries are segment_games', not a
    # second reset scan.
    seg_lens = [len(s.events) for s in segments]
    start0 = len(all_events) - sum(seg_lens)
    ranges: list[tuple[int, int]] = []
    cursor = start0
    for length in seg_lens:
        ranges.append((cursor, cursor + length))
        cursor += length

    buckets: list[list[DecodedFrame]] = [[] for _ in range(n_seg)]
    global_hi_by_frame: dict[int, int] = {}
    offset = 0
    last_seg = 0
    for frame, evs in zip(frames, per_frame_events, strict=True):
        lo, hi = offset, offset + len(evs)
        offset = hi
        seg = _dominant_segment(lo, hi, ranges)
        if seg is None:
            if lo < start0:
                continue  # only pre-first-reset noise — dropped
            seg = last_seg  # a zero-event frame inside the corpus — carry forward
        last_seg = seg
        buckets[seg].append(frame)
        global_hi_by_frame[id(frame)] = hi  # last all_events index this frame's OCR covers

    grant = re.compile(r"received starting resources")
    line_by_frame = dict(zip((id(f) for f in frames), per_frame_lines, strict=True))
    routed: list[GameFrames | None] = []
    for seg_idx, bucket in enumerate(buckets):
        if len(bucket) < 2:
            routed.append(None)  # too few frames to gate board stability (frames_unrouted)
            continue
        grant_frames = tuple(
            f for f in bucket if any(grant.search(line.lower()) for line in line_by_frame[id(f)])
        )
        routed.append(
            GameFrames(
                setup_frames=tuple(bucket[: max(2, len(bucket) // 2)]),
                post_setup_frame=_post_setup_frame(
                    bucket, ranges[seg_idx], all_events, global_hi_by_frame
                ),
                empty_baseline=bucket[0].frame,
                grant_frames=grant_frames,
            )
        )
    return tuple(routed)


#: The main-game piece-placing events. The FIRST of these inside a game window marks
#: the end of the stable 8-pieces-down opening board — after it the board no longer
#: shows exactly the setup pieces the openings CV demands.
_MAIN_GAME_BUILD_KINDS = frozenset({"built_settlement", "built_city", "built_road"})


def _post_setup_frame(
    bucket: list[DecodedFrame],
    seg_range: tuple[int, int],
    all_events: tuple[LogEvent, ...],
    global_hi_by_frame: dict[int, int],
) -> DecodedFrame:
    """The 8-pieces-down post-setup frame for a game — NOT the game's last frame.

    The openings detector demands exactly the opening 8 setup pieces (4 settlements +
    4 roads). ``bucket[-1]`` is the game's LAST (end-game) frame — a board cluttered
    with every main-game piece — which the detector cannot read. The opening board is
    stable from the final setup placement until the game's FIRST main-game build
    (``built_settlement`` / ``built_city`` / ``built_road``), so the latest frame whose
    whole OCR window precedes that first build still shows exactly the 8 setup pieces.
    Falls back to the first bucket frame when the first build already lands inside the
    opening frame's window, and to the last frame when the game logs no build at all (a
    cutoff that never left the setup board — the whole window is still 8-pieces-down).
    """
    rlo, rhi = seg_range
    boundary: int | None = None
    for i in range(rlo, rhi):
        if all_events[i].kind in _MAIN_GAME_BUILD_KINDS:
            boundary = i
            break
    if boundary is None:
        return bucket[-1]
    pre_build = [f for f in bucket if global_hi_by_frame[id(f)] <= boundary]
    return pre_build[-1] if pre_build else bucket[0]


def _dominant_segment(lo: int, hi: int, ranges: list[tuple[int, int]]) -> int | None:
    """The segment whose event-index window overlaps ``[lo, hi)`` most (ties → later).

    ``None`` when the frame overlaps no segment (its events fall entirely in the dropped
    pre-first-reset prefix, or it contributed no events at all).
    """
    best: int | None = None
    best_overlap = 0
    for k, (rlo, rhi) in enumerate(ranges):
        overlap = min(hi, rhi) - max(lo, rlo)
        if overlap > 0 and overlap >= best_overlap:
            best = k
            best_overlap = overlap
    return best


def _consensus_grant(
    handle: str, grant_frames: tuple[DecodedFrame, ...]
) -> Counter[str] | None:  # pragma: no cover - real-run path
    """Multi-frame CONSENSUS grant read for one player across the grant-line frames.

    Re-detects the grant-icon boxes per frame (:func:`detect_glyph_boxes`) over the
    ``"received starting resources"`` line's ~30s on-screen lifetime and feeds every
    readable ``(log_crop, boxes)`` pair to :func:`consensus_granted_glyphs`, which
    requires ≥2 frames to agree byte-identical (fail-closed, brief §5.2). ``None``
    when the boxes cannot be localised or the reads disagree — the game then falls
    out as glyph-unreadable rather than feeding the firewall a fabricated multiset.
    """
    frames_boxes: list[tuple[npt.NDArray[np.uint8], list[tuple[int, int, int, int]]]] = []
    for frame in grant_frames:
        crop = crop_log(frame.frame)
        line_box, text_boxes = _grant_line_boxes(crop, handle)
        if line_box is None:
            continue
        boxes = detect_glyph_boxes(crop, line_box, text_boxes)
        if boxes:
            frames_boxes.append((crop, boxes))
    if len(frames_boxes) < 2:
        return None
    return consensus_granted_glyphs(frames_boxes)


def _grant_line_boxes(
    crop_rgb: npt.NDArray[np.uint8], handle: str
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:  # pragma: no cover
    """OCR the log crop with bounding boxes; return the grant line's box + all text boxes.

    Uses easyocr's box-returning read (the same lazy reader logparse/board_cv use) to
    locate ``handle``'s ``"received starting resources"`` line and every text box the
    grant-icon detector must avoid.
    """
    import easyocr  # lazy: never imported on the headless CI path

    reader = easyocr.Reader(["en"], gpu=False)
    results: list[tuple[list[list[int]], str, float]] = reader.readtext(crop_rgb)
    text_boxes: list[tuple[int, int, int, int]] = []
    line_box: tuple[int, int, int, int] | None = None
    for poly, text, _conf in results:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        box = (min(xs), min(ys), max(xs), max(ys))
        text_boxes.append(box)
        low = text.lower()
        if "received starting resources" in low and handle.lower() in low:
            line_box = box
    return line_box, text_boxes
