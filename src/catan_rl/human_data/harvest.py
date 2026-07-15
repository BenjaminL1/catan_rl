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

- :func:`_ingest_two_pass` — download → SPARSE sample → OCR → DENSE re-sample around
  the grant lines → in-memory frames + their log lines (video deleted on exit). This is
  the seam :func:`parse_video` uses. The dense pass is what keeps the fail-closed grant
  consensus (>= 2 agreeing readable frames) from starving: at the 4 s sparse cadence a
  player's grant line can be caught in as few as 2 frames, and one unreadable frame there
  threw away 6/6 games of a video whose opening frames were all clean. (:func:`_ingest`
  — the single-pass original — is kept for callers that only need frames.)
- :func:`_extract_context` — Stage-1 tail: player handles, per-game player→colour
  binding + seat order, the parsed event stream, and per-game frame routing. Accepts the
  already-OCR'd log lines from the two-pass ingest (the log-crop OCR is the dominant cost
  of the harvest, so it is never paid twice).
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
import hashlib
import json
import re
import tempfile
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
from catan_rl.human_data.ffmpeg import resolve_ffmpeg, resolve_ffprobe
from catan_rl.human_data.glyph_anchor import (
    GRANT_RE,
    MIN_GRANT_CONSENSUS_FRAMES,
    classify_granted_glyphs,
    consensus_granted_glyphs,
    detect_glyph_boxes,
)
from catan_rl.human_data.ingest import (
    DecodedFrame,
    TimeWindow,
    build_sampling_schedule,
    decode_frames_at,
    download_video,
    ingest_video,
)
from catan_rl.human_data.logparse import (
    LogEvent,
    _resolve_actor,
    crop_log,
    ocr_log_crop,
    parse_log,
)
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

#: Typed reject reason for a game whose stable 8-pieces-down opening frame cannot be
#: identified: the window logs NO build (so no opening/main-game boundary exists) yet
#: demonstrably progressed past setup (it carries rolls and/or a victory). Returning
#: the window's LAST frame there would hand the openings CV an END-GAME board (the
#: "Well Played!" stats overlay) — a fail-OPEN that silently produced garbage. Reject
#: typed instead.
POST_SETUP_UNRESOLVED_REASON = "post_setup_frame_unresolved"

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
    CV reads, or ``None`` when no such frame exists in the window (a typed
    :data:`POST_SETUP_UNRESOLVED_REASON` reject — NEVER an end-game fallback);
    ``empty_baseline`` the no-pieces board (the green-tile-subtraction source,
    §5.13); ``grant_frames`` every frame the ``"received starting resources"`` line
    is visible on (the multi-frame CONSENSUS grant read).
    """

    setup_frames: tuple[DecodedFrame, ...]
    post_setup_frame: DecodedFrame | None
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
    #: False when the game HAS roll events but none of their values OCR'd (Colonist
    #: renders the dice as face glyphs) — an honest "unread", never a fabricated luck
    #: series. See :func:`_dice_values_readable`. Defaults True so fixture/test paths
    #: (whose logs carry textual roll values) keep the strict parse-failure guard.
    dice_values_readable: bool = True


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

    ``a`` is the FIRST player to place a setup piece (snake 1→2→2→1 places ``a``
    first and last); ``b`` is the other handle. When no setup-placement line
    resolved (a missing / unsampled draft), falls back to sorted handles — the
    resulting record is order-unestablished (the log-side ordinal cannot fire) and
    so EVAL-excluded regardless, and this only keeps the record CONTRACT-valid
    (``draft_order`` must be a snake) so the §5.6 audit row still loads.

    ``setup_placed_any`` (the noun-less kind real footage actually produces — the
    piece is an ICON) counts as a setup placement here: the FIRST placement of a
    snake draft is a settlement either way, so the first-placer identity is the same
    signal. This is robust to the log panel's re-OCR duplication, which repeats lines
    but never reorders the FIRST occurrence.
    """
    first: str | None = None
    for event in events:
        if event.kind in ("setup_settlement", "setup_placed_any") and event.actor in handles:
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


def _dice_values_readable(events: Sequence[LogEvent], dice_log: tuple[int, ...]) -> bool:
    """Whether this game's roll VALUES were readable at all (§5.4 luck covariate).

    Colonist renders the rolled dice as FACE GLYPHS: the line OCRs as
    ``"ThePhantom rolled"`` with no number, so :func:`_dice_log` recovers nothing on
    real footage even though the ``roll`` EVENTS parse fine. Distinguishing the two
    cases matters, because :meth:`GameRecord.validate` rejects an empty ``dice_log``
    on a completed game as a parse failure:

    * roll events exist but NO value parsed ⇒ the values are icon-rendered and simply
      unread — an honest ``False`` (empty ``dice_log`` permitted, luck covariate
      unavailable), NOT a fabricated luck series;
    * no roll events at all ⇒ nothing was there to read, so this stays ``True`` and a
      winner-set game with an empty log still trips the parse-failure guard.
    """
    if dice_log:
        return True
    return not any(event.kind == "roll" for event in events)


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
    dice_log = _dice_log(events)
    return GameInputs(
        board=_placeholder_board(),
        openings_desert_hex=_PLACEHOLDER_DESERT_HEX,
        opening_result=OpeningResult(openings=None, rejection_reason=reason),
        granted_by_player={},
        draft_order=_draft_order(events, handles),
        dice_log=dice_log,
        dice_values_readable=_dice_values_readable(events, dice_log),
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


#: The grant line's human-readable phrase — the ONLY log text whose sampling density gates
#: the glyph anchor (and therefore whether a game is usable at all). Kept for documentation;
#: all MATCHING goes through :data:`~catan_rl.human_data.glyph_anchor.GRANT_RE` (FIX B), the
#: OCR-tolerant pattern the anchor's own detector uses, so a mangled "received" (which
#: logparse still parses as a ``starting_resources`` event) can no longer produce a game
#: with zero grant frames — the ``grant_frames=0`` signature of ``KvH76fJI4f0 g2``.
_GRANT_PHRASE = "received starting resources"

#: Seconds of dense (1 s) sampling to add on EACH side of a sparse frame that showed a
#: grant line. The line lives ~30 s on screen but the sparse pass can clip it to as few
#: as 2 frames (measured), and one unreadable frame there starves the >=2-frame
#: consensus. +/-10 s brackets the whole line lifetime for ~10-15% more OCR per video.
_GRANT_DENSE_PAD_S = 10.0


def _grant_dense_windows(
    grant_ts: Sequence[float],
    duration_s: float,
    pad_s: float = _GRANT_DENSE_PAD_S,
) -> list[TimeWindow]:
    """Dense-sampling windows bracketing every sparse frame that showed a grant line.

    **Why this exists.** The glyph anchor (the ONLY joint-flip defence) needs each
    player's starting-resource multiset, and :func:`consensus_granted_glyphs` requires
    **>= 2 readable frames that agree** — deliberately fail-closed. At the deployed 4 s
    sparse cadence a player's grant line can be caught in as few as **2** frames, so a
    single unreadable one (glyph boxes not localisable in that instant) leaves 1 < 2 and
    the consensus returns ``None`` -> ``glyph_unreadable`` -> the game is thrown away
    even though its opening frame is perfect. Measured on ``9Sm86ml04aI``: 6/6 games had
    a clean opening frame and 0/6 survived, because the opponent's grant landed on
    exactly 2 sparse frames and one of them read 0 glyph boxes. Re-sampling that same
    window at 1 s gives 5 readable frames, unanimous.

    The two-pass schedule already supports this
    (:func:`~catan_rl.human_data.ingest.build_sampling_schedule` takes ``dense_windows``);
    harvest simply never supplied any, so the dense pass had never run. The fix is MORE
    FRAMES, never a looser gate.

    Windows are padded by ``pad_s`` on each side, clamped to ``[0, duration_s]``, and
    OVERLAPPING windows are merged (a game's grant burst spans several sparse frames a
    few seconds apart, and re-decoding the same range twice is wasted OCR). Returns ``[]``
    when no grant was seen — nothing to rescue, so no dense pass and no wasted work.
    """
    if not grant_ts:
        return []
    spans = sorted(
        (max(0.0, ts - pad_s), min(duration_s, ts + pad_s)) for ts in grant_ts if ts >= 0.0
    )
    merged: list[list[float]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [TimeWindow(start_s=lo, end_s=hi) for lo, hi in merged if hi > lo]


def _cached_ocr_log(
    frame: npt.NDArray[np.uint8],
    cache: dict[bytes, list[str]],
    stats: Counter[str],
) -> list[str]:
    """OCR one frame's log crop, memoised on the CROP's pixel bytes (per video).

    The HUD / dice areas OUTSIDE the log crop change on essentially every frame, so
    the crop is taken FIRST and only ITS bytes are hashed — hashing the whole frame
    would defeat the cache. easyocr on identical pixels is deterministic, so returning
    the cached lines is behaviour-preserving by construction; a COPY is returned (never
    the cached list itself) so a caller mutating the result cannot corrupt the cache.

    ``cache`` is a dict LOCAL to one video's ingest (never shared across videos, or two
    videos' identical opening frames would cross-contaminate). ``stats`` accumulates
    ``hits`` / ``misses`` for the per-video ``[ocr-cache]`` telemetry line. The dense
    pass re-decodes bounded windows the sparse pass already touched, so identical crops
    recur within a single video and the hit rate is real, not incidental.
    """
    crop = crop_log(frame)
    key = hashlib.sha1(crop.tobytes()).digest()
    cached = cache.get(key)
    if cached is not None:
        stats["hits"] += 1
        return list(cached)
    lines = ocr_log_crop(crop)
    cache[key] = lines
    stats["misses"] += 1
    return list(lines)


def _ingest_two_pass(
    video_id: str,
    *,
    download_gate: threading.BoundedSemaphore | None = None,
    work_dir: Path | None = None,
) -> tuple[list[DecodedFrame], list[list[str]]]:
    """Download once -> sparse pass -> OCR -> dense pass around the grants -> delete.

    The real two-pass ingest the brief specifies (§5.10) and :func:`_ingest` never ran:
    the sparse pass DISCOVERS where the grant lines are, then the dense (1 s) pass
    re-samples only those bounded windows (:func:`_grant_dense_windows`) so the
    fail-closed grant consensus has frames to work with.

    Returns the frames **and their log lines**, ts-sorted and index-aligned. Each frame
    is OCR'd EXACTLY ONCE here and the lines are threaded onward
    (:func:`_extract_context` accepts them), because the log-crop OCR is the dominant
    cost of the whole harvest and re-reading it would more than pay back the dense pass.

    The downloaded video is deleted on EVERY path (``finally``), matching
    :func:`~catan_rl.human_data.ingest.ingest_video`'s download-then-delete contract; the
    ``download_gate`` is held around the NETWORK phase only, so the CPU/OCR phase still
    fans out to the full worker width (§5.11).
    """  # pragma: no cover - real-run path (network + easyocr; unit-tested via its parts)
    duration_s = _probe_duration_s(video_id)
    ffmpeg_bin = resolve_ffmpeg()
    ffprobe_bin = resolve_ffprobe()
    gate = download_gate if download_gate is not None else _NULL_GATE

    tmp = tempfile.mkdtemp(prefix=f"phantom_{video_id}_", dir=str(work_dir) if work_dir else None)
    tmp_path = Path(tmp)
    try:
        with gate:
            video_path = download_video(video_id, tmp_path)

        # Per-video crop-hash OCR cache: the dense pass re-decodes windows the sparse
        # pass already touched, so identical log crops recur; OCR of identical pixels is
        # deterministic (behaviour-preserving). Cache is LOCAL to this call — never
        # crosses videos.
        ocr_cache: dict[bytes, list[str]] = {}
        ocr_stats: Counter[str] = Counter()
        sparse = build_sampling_schedule(duration_s)
        frames = list(decode_frames_at(video_path, sparse, ffmpeg=ffmpeg_bin, ffprobe=ffprobe_bin))
        lines = [_cached_ocr_log(f.frame, ocr_cache, ocr_stats) for f in frames]

        grant_ts = [
            f.ts
            for f, ls in zip(frames, lines, strict=True)
            if any(GRANT_RE.search(ln.lower()) for ln in ls)
        ]
        windows = _grant_dense_windows(grant_ts, duration_s)
        if windows:
            seen = {round(f.ts, 3) for f in frames}
            dense = [
                s
                for s in build_sampling_schedule(duration_s, windows)
                if s.pass_name == "dense" and round(s.ts, 3) not in seen
            ]
            if dense:
                extra = list(
                    decode_frames_at(video_path, dense, ffmpeg=ffmpeg_bin, ffprobe=ffprobe_bin)
                )
                frames.extend(extra)
                lines.extend(_cached_ocr_log(f.frame, ocr_cache, ocr_stats) for f in extra)
        _hits, _misses = ocr_stats["hits"], ocr_stats["misses"]
        _total = _hits + _misses
        _rate = (100.0 * _hits / _total) if _total else 0.0
        print(
            f"[ocr-cache] video={video_id} hits={_hits} misses={_misses} rate={_rate:.1f}%",
            flush=True,
        )
    finally:
        for child in tmp_path.glob("*"):
            child.unlink(missing_ok=True)
        tmp_path.rmdir()

    if not frames:
        raise VideoParseError(f"ingest produced no frames for {video_id!r}")
    order = sorted(range(len(frames)), key=lambda i: frames[i].ts)
    return [frames[i] for i in order], [lines[i] for i in order]


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


def _extract_context(
    video_id: str,
    frames: list[DecodedFrame],
    per_frame_lines: list[list[str]] | None = None,
) -> VideoContext:
    """Stage-1 tail: player handles, HUD colour binding, event stream, frame routing.

    BEST-EFFORT and SAFE under a wrong guess: every downstream firewall
    (HUD-assignment check §5.14, board-stability §5.2, the glyph anchor) REJECTS a
    mislabelled game, so a wrong handle/colour/routing guess produces a typed reject,
    never a confidently-wrong ACCEPT. Handles are discovered from the leading tokens
    of event-shaped OCR lines; the per-game player→colour binding is read from the
    HUD seat rings; frames are routed to games by the ``game_reset`` markers in
    source order (index-aligned with :func:`~catan_rl.human_data.segment.segment_games`).

    ``per_frame_lines`` (index-aligned with ``frames``) lets the caller supply log lines
    it has ALREADY OCR'd — :func:`_ingest_two_pass` must OCR the sparse frames to find
    the grant lines its dense pass brackets, and the log-crop OCR is the dominant cost of
    the whole harvest, so re-reading it here would more than cancel out the dense pass.
    ``None`` keeps the standalone behaviour (OCR the frames here).
    """  # pragma: no cover - real-run CV path (exercised on hardware, mocked in CI)
    if per_frame_lines is None:
        per_frame_lines = [ocr_log_crop(crop_log(f.frame)) for f in frames]
    all_lines = [line for lines in per_frame_lines for line in lines]
    handles = _discover_handles(all_lines)
    players = _agent_binding(handles)
    events = parse_log(all_lines, handles).events
    seat_colours = _video_seat_colours(frames)
    player_colors, seat_order = _bind_colours(players, seat_colours)
    game_frames = _route_frames_to_games(frames, per_frame_lines, handles)
    return VideoContext(
        players=players,
        handles=handles,
        player_colors=player_colors,
        seat_order=seat_order,
        events=events,
        game_frames=game_frames,
    )


#: How many of a game's own in-window frames to feed the FALLBACK board read. Each
#: `read_board` costs a token detect + RANSAC + 18 number OCRs, and the dense pass can
#: leave ~100 grant frames, so the pool is capped: >=2 agreeing reads is all the
#: stability rule needs, and a handful gives it room to find them.
_BOARD_FALLBACK_POOL = 5


def _stable_board_for_game(gf: GameFrames) -> BoardRead | None:
    """The cross-frame-stable board for ONE game, with an in-window fallback pool.

    The board is normally read from ``setup_frames`` (= ``bucket[:len//2]``). For any game
    after a video's FIRST, those earliest frames are routinely the PREVIOUS game's end
    screen / lobby, so the read fails even though the game's own frames are perfectly
    readable — MEASURED: ``0EtcbG16kHA g1`` and ``9Sm86ml04aI g3`` both had their
    post-setup AND baseline frames reading cleanly, yet both were thrown away as
    ``board_unreadable``. That was 2 of the 6 board losses in the 8-video sweep.

    So: try ``setup_frames`` first (today's behaviour, byte-identical when it works), and
    only on failure retry over frames GUARANTEED to be inside this game's window — the
    post-setup frame (bounded by this game's own first build) and the grant frames (they
    carry this game's grant lines).

    ``empty_baseline`` is deliberately NOT in the pool: it is ``bucket[0]``, which can be
    the previous game's frame, and two agreeing frames from the WRONG game would be a
    silent cross-game board weld — the one failure no downstream firewall can catch.

    :func:`read_board_stable`'s rule is untouched (>= 2 accepted frames, ALL byte-identical
    or reject), so the fail-closed §5.2 stability guarantee is fully preserved; this only
    changes WHICH frames it is offered.
    """
    board = read_board_stable([f.frame for f in gf.setup_frames])
    if board is not None:
        return board

    pool: list[Any] = []
    if gf.post_setup_frame is not None:
        pool.append(gf.post_setup_frame.frame)
    pool.extend(f.frame for f in gf.grant_frames[:_BOARD_FALLBACK_POOL])
    if len(pool) < 2:
        return None
    return read_board_stable(pool[:_BOARD_FALLBACK_POOL])


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
    if gf.post_setup_frame is None:  # no honest 8-pieces-down frame exists in the window
        return _reject_inputs(segment_events, ctx.handles, POST_SETUP_UNRESOLVED_REASON)
    board = _stable_board_for_game(gf)
    if board is None:
        return _reject_inputs(segment_events, ctx.handles, BOARD_UNREADABLE_REASON)
    opening_result = detect_openings_result(
        gf.post_setup_frame.frame,
        gf.empty_baseline,
        board,
        player_colors=ctx.player_colors,
        seat_order=list(ctx.seat_order),
        pov_handle=ctx.players["agent"],
    )
    granted: dict[str, Counter[str] | None] = {}
    for handle in ctx.handles:
        granted[handle] = _consensus_grant(handle, gf.grant_frames, ctx.handles)
    dice_log = _dice_log(segment_events)
    return GameInputs(
        board=board,
        openings_desert_hex=board.desert_hex,
        opening_result=opening_result,
        granted_by_player=granted,
        draft_order=_draft_order(segment_events, ctx.handles),
        dice_log=dice_log,
        dice_values_readable=_dice_values_readable(segment_events, dice_log),
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

    # Two-pass: the sparse pass discovers the grant lines, then a 1 s dense pass
    # re-samples ONLY those windows so the fail-closed grant consensus (>= 2 agreeing
    # readable frames) is not starved — the defect that threw away 6/6 games of
    # 9Sm86ml04aI despite every one having a clean opening frame.
    frames, per_frame_lines = _ingest_two_pass(
        video_id, download_gate=download_gate, work_dir=work_dir
    )
    ctx = _extract_context(video_id, frames, per_frame_lines)
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
            dice_values_readable=gi.dice_values_readable,
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


#: How many frames the per-video HUD colour vote samples across the whole video.
#: The two seat rings are a fixed overlay, so a couple dozen spread samples is
#: ample; the cap keeps the extra ``read_hud_seat_colors`` (cv2) calls cheap.
_HUD_VOTE_SAMPLES = 24


def _majority_two_colour(reads: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    """The most common exactly-two-DISTINCT-colour HUD read across ``reads``.

    A ThePhantom compilation stitches back-to-back games whose OPPONENT colour
    varies (ThePhantom self-seats a fixed colour throughout); only a game whose
    opponent is in :data:`~catan_rl.human_data.openings.PALETTE` yields a
    two-distinct-colour HUD read, so a single arbitrary frame (the old per-video
    middle-frame read) fails the whole video whenever that frame lands on a
    palette-unsupported game — even when a supported game exists elsewhere in the
    compilation. Voting the two-distinct reads across the video recovers the
    supported game's binding (a one-colour read — the unsupported-opponent games —
    casts no vote). Returns ``()`` when no frame yields two distinct colours.
    """
    votes: Counter[tuple[str, ...]] = Counter()
    for read in reads:
        if len(set(read)) == 2:
            votes[tuple(read)] += 1
    if not votes:
        return ()
    return votes.most_common(1)[0][0]


def _video_seat_colours(
    frames: Sequence[DecodedFrame],
) -> tuple[str, ...]:  # pragma: no cover - real-run CV path (mocked in CI)
    """The video's seat-colour pair by majority vote over spread HUD reads (§5.14).

    Reads the HUD seat rings (:func:`~catan_rl.human_data.openings.read_hud_seat_colors`)
    on up to :data:`_HUD_VOTE_SAMPLES` frames spread across the video and returns the
    most common two-distinct-colour read (:func:`_majority_two_colour`). Robust to a
    compilation whose sampled middle frame lands on a palette-unsupported game, and to
    a lone mis-read frame — unlike the single-frame read it replaces.
    """
    if not frames:
        return ()
    step = max(1, len(frames) // _HUD_VOTE_SAMPLES)
    reads = [read_hud_seat_colors(f.frame) for f in frames[::step]]
    return _majority_two_colour(reads)


def _bind_colours(
    players: dict[str, str], seat_colours: tuple[str, ...]
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Derive the handle→colour binding from the AUTHORITATIVE POV seat anchor.

    ``read_hud_seat_colors`` returns the two seat colours top→bottom. The POV/agent
    (ThePhantom) is ALWAYS the BOTTOM self-seat and the opponent the TOP seat (spike
    ``blockers/VERDICT.md``: "POV / 'You' = ThePhantom (bottom-right self-seat)"), so
    the binding is DERIVED from that structural invariant — top→opponent,
    bottom→agent — NOT from the earlier ``seat_order = handles`` guess. Log-line
    frequency (how ``handles`` is ordered) has nothing to do with HUD seat position:
    that was an unvalidated assumption the §5.14 firewall — re-reading the very HUD
    the binding was built from — could never test (a mis-seated handle passed
    silently). The §5.14 assignment check re-anchors on the POV IDENTITY against the
    post-setup frame (a DIFFERENT read), so it now has real discriminating power.
    """
    if len(seat_colours) != 2 or len(set(seat_colours)) != 2:
        raise VideoParseError("HUD did not yield two distinct seat colours")
    agent, opponent = players["agent"], players["opponent"]
    seat_order = (opponent, agent)  # top → bottom: opponent seat above the POV self-seat
    player_colors = {opponent: seat_colours[0], agent: seat_colours[1]}
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

    line_by_frame = dict(zip((id(f) for f in frames), per_frame_lines, strict=True))
    routed: list[GameFrames | None] = []
    for seg_idx, bucket in enumerate(buckets):
        if len(bucket) < 2:
            routed.append(None)  # too few frames to gate board stability (frames_unrouted)
            continue
        grant_frames = tuple(
            f for f in bucket if any(GRANT_RE.search(line.lower()) for line in line_by_frame[id(f)])
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
_MAIN_GAME_BUILD_KINDS = frozenset(
    # ``built_any`` is the noun-less kind real footage actually produces (Colonist
    # renders the built piece as an ICON, so "built a Settlement" never OCRs as text).
    # Without it NO build is ever seen on video, the boundary below is always None,
    # and the old ``bucket[-1]`` fallback handed the openings CV the END-GAME frame.
    {"built_settlement", "built_city", "built_road", "built_any"}
)

#: Events proving a game's board CANNOT still be 8-pieces-down at the window's end.
#: A game that reached a 15-VP victory (or a resign after a real game) necessarily
#: built — so a window carrying one of these but logging NO build did not "never leave
#: setup"; its builds were simply never sampled, and its last frame is an END-GAME
#: board. A bare ``roll`` is deliberately NOT here: a game can roll and then be cut off
#: without ever building, and that window IS still a legitimate 8-pieces-down read.
_PAST_SETUP_KINDS = frozenset({"victory", "resign"})


def _post_setup_frame(
    bucket: list[DecodedFrame],
    seg_range: tuple[int, int],
    all_events: tuple[LogEvent, ...],
    global_hi_by_frame: dict[int, int],
) -> DecodedFrame | None:
    """The 8-pieces-down post-setup frame for a game — NOT the game's last frame.

    The openings detector demands exactly the opening 8 setup pieces (4 settlements +
    4 roads). ``bucket[-1]`` is the game's LAST (end-game) frame — a board cluttered
    with every main-game piece — which the detector cannot read. The opening board is
    stable from the final setup placement until the game's FIRST main-game build
    (:data:`_MAIN_GAME_BUILD_KINDS`), so the latest frame whose whole OCR window
    precedes that first build still shows exactly the 8 setup pieces.

    When the window logs **no build at all** there are two very different cases, and
    conflating them is a fail-OPEN bug (it is what made every real-video game read out
    as the "Well Played!" end-game stats overlay):

    * the game never left setup (a cutoff mid-draft) — no roll, no victory — so the
      whole window IS still 8-pieces-down and ``bucket[-1]`` is correct; or
    * the game demonstrably played on (it carries rolls and/or a victory) but its
      builds were invisible to the log — the board is NOT 8-pieces-down at the end,
      and there is no frame we can honestly nominate. Return ``None`` → the game reads
      out as a typed :data:`POST_SETUP_UNRESOLVED_REASON` reject.

    When a build exists but no sampled frame's OCR window fully precedes it, falls back
    to the window's FIRST frame (the earliest, least-built board) — a best-effort the
    openings CV independently validates (it demands exactly 8 pieces and typed-rejects
    otherwise), so this cannot fail open.
    """
    rlo, rhi = seg_range
    boundary: int | None = None
    for i in range(rlo, rhi):
        if all_events[i].kind in _MAIN_GAME_BUILD_KINDS:
            boundary = i
            break
    if boundary is None:
        played_on = any(all_events[i].kind in _PAST_SETUP_KINDS for i in range(rlo, rhi))
        if played_on:
            return None  # builds unreadable — the last frame is NOT the opening board
        return bucket[-1]  # true setup-only cutoff: the window never left the draft
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


#: A DOMINANT grant read may carry the consensus when strict unanimity is not reached but
#: the dissent is plainly OCR NOISE rather than genuine ambiguity.
#:
#: :func:`~catan_rl.human_data.glyph_anchor.consensus_granted_glyphs` requires FULL
#: agreement. That is incoherent once many frames are sampled: it accepts a 2-of-2 read yet
#: rejects 47-of-48, which is far stronger evidence. Wiring the dense sampling pass raised
#: the per-grant frame count from ~2-7 to ~48-99 and turned it into a real yield killer
#: (5 of 14 games in the 8-video sweep died on disagreeing reads).
#:
#: MEASURED (``0EtcbG16kHA`` g1, from the grant diagnostics):
#:   * ``LevyChevy``  48 reads -> 47 x {BRICK:2,SHEEP:1} vs 1 x (+ORE)  = 47-vs-1 -> NOISE
#:   * ``ThePhantom``  9 reads ->  6 x {BRICK,WHEAT,WOOD} vs 3 x (+ORE) =  6-vs-3 -> AMBIGUOUS
#:
#: A dominant read therefore needs BOTH a real sample AND overwhelming agreement: 47/48 =
#: 0.979 accepts; 6/9 = 0.667 keeps failing closed (a one-card WHEAT/ORE confusion at that
#: rate is not evidence). This lives HERE, not in ``glyph_anchor``, on purpose: that module
#: is SHA-256 fingerprinted and hard-gated on a committed validation artifact
#: (:func:`~catan_rl.human_data.glyph_anchor.glyph_classifier_fingerprint`), and this rule
#: does not change a single pixel decision — it only decides how many agreeing
#: CLASSIFICATIONS are enough. The classifier, and its validation, stay untouched. The
#: winning multiset still flows through the UNCHANGED settlement-matching anchor, so the
#: joint-flip firewall is never bypassed.
DOMINANT_READ_MIN_READS = 5
DOMINANT_READ_MIN_FRAC = 0.9


def _dominant_grant_read(
    frames_boxes: list[tuple[npt.NDArray[np.uint8], list[tuple[int, int, int, int]]]],
    palette: Any = None,
) -> Counter[str] | None:
    """The modal grant multiset when it overwhelmingly dominates; else ``None``.

    The fallback for the (correct, but scale-blind) unanimity rule — see
    :data:`DOMINANT_READ_MIN_READS`. Fails closed on genuine bimodality.
    """
    reads = [
        r
        for crop, boxes in frames_boxes
        if (
            r := (
                classify_granted_glyphs(crop, boxes)
                if palette is None
                else classify_granted_glyphs(crop, boxes, palette)
            )
        )
        is not None
    ]
    if len(reads) < DOMINANT_READ_MIN_READS:
        return None
    tally = Counter(tuple(sorted(r.items())) for r in reads)
    top, top_n = tally.most_common(1)[0]
    if top_n / len(reads) < DOMINANT_READ_MIN_FRAC:
        return None  # genuine bimodality — never tie-break a coin flip
    return Counter(dict(top))


# --- FIX A: guarded subset-collapse of grant reads (dropped-icon partials) ------------
# When the minority frames read a strict SUBSET of the modal multiset (one card-icon
# missed in that instant), that is NOT a competing hypothesis — it is the same grant with
# a dropped box. Collapse those subset frames into the modal SUPERSET, then re-test the
# collapsed tally with the same fail-closed rules. Three MANDATORY guards keep this from
# ever fabricating a grant (measured basis: 9Sm86ml04aI g5 = 13x{ORE,SHEEP,WOOD} vs
# 2x{ORE,SHEEP}). The accepted multiset still flows through the UNCHANGED settlement-
# matching anchor (orientation.assert_glyph_anchor), so the joint-D6-flip firewall is
# never bypassed — this only decides WHICH multiset the anchor tests.
#: A setup grant is the 2nd settlement's adjacent resources (<=3 hexes), so a read with
#: >3 cards is definitionally a detector error (guard 3) — never a vote or collapse target.
MAX_GRANT_CARDS = 3
#: Guard 1: a subset collapses into a superset only if the superset has BOTH >= this many
#: supporting frames AND >= SUBSET_COLLAPSE_MIN_RATIO x the subset's support. A bare 2-vs-1
#: (n=3) never collapses — this deliberately drops 9Sm86ml04aI g3 (only 3 reads).
SUBSET_COLLAPSE_MIN_SUPERSET = 5
SUBSET_COLLAPSE_MIN_RATIO = 2


def _is_strict_superset(sup: dict[str, int], sub: dict[str, int]) -> bool:
    """True iff multiset ``sup`` strictly contains multiset ``sub`` (``sub`` is a proper
    subset: every card count in ``sub`` is <= ``sup`` and the two are not equal)."""
    return sup != sub and all(cnt <= sup.get(res, 0) for res, cnt in sub.items())


def _grant_reads_tally(
    frames_boxes: list[tuple[npt.NDArray[np.uint8], list[tuple[int, int, int, int]]]],
    palette: Any = None,
) -> Counter[tuple[tuple[str, int], ...]]:
    """Tally the readable grant multisets across frames, AFTER guard 3 (drop any read with
    > :data:`MAX_GRANT_CARDS` cards — a 4+-box read is a detector error and must never be a
    consensus vote or a collapse target). Keyed by the sorted (res, count) tuple."""
    reads = [
        r
        for crop, boxes in frames_boxes
        if (
            r := (
                classify_granted_glyphs(crop, boxes)
                if palette is None
                else classify_granted_glyphs(crop, boxes, palette)
            )
        )
        is not None
        and sum(r.values()) <= MAX_GRANT_CARDS
    ]
    return Counter(tuple(sorted(r.items())) for r in reads)


def _subset_collapse_tally(
    tally: Counter[tuple[tuple[str, int], ...]],
) -> tuple[Counter[tuple[tuple[str, int], ...]], list[dict[str, Any]]]:
    """Collapse each strict-subset read into its UNIQUE maximal-support strict superset.

    Guards 1 (:data:`SUBSET_COLLAPSE_MIN_SUPERSET` floor AND
    :data:`SUBSET_COLLAPSE_MIN_RATIO` ratio) and 2 (a subset with two equal-support
    maximal supersets does NOT collapse — fail closed on the tie). Single pass over the
    tally sorted by descending support; a read acts as a collapse TARGET only with its
    ORIGINAL (pre-collapse) support, so no chains/cascades form and the pass terminates in
    one sweep. Returns ``(collapsed_tally, events)`` where each event is
    ``{"subset", "into", "n"}``; total support is preserved (every read is counted once).
    """
    orig = dict(tally)
    keys = [k for k, _ in tally.most_common()]  # descending original support
    collapsed: Counter[tuple[tuple[str, int], ...]] = Counter()
    events: list[dict[str, Any]] = []
    for sub_key in keys:
        sub = dict(sub_key)
        sub_n = orig[sub_key]
        supersets = [
            (k, orig[k]) for k in keys if k != sub_key and _is_strict_superset(dict(k), sub)
        ]
        if supersets:
            max_n = max(n for _, n in supersets)
            top = [k for k, n in supersets if n == max_n]
            if (
                len(top) == 1
                and max_n >= SUBSET_COLLAPSE_MIN_SUPERSET
                and max_n >= SUBSET_COLLAPSE_MIN_RATIO * sub_n
            ):
                collapsed[top[0]] += sub_n
                events.append({"subset": dict(sub_key), "into": dict(top[0]), "n": sub_n})
                continue
        collapsed[sub_key] += sub_n
    return collapsed, events


def _collapse_grant_read(
    frames_boxes: list[tuple[npt.NDArray[np.uint8], list[tuple[int, int, int, int]]]],
    palette: Any = None,
    diag: dict[str, Any] | None = None,
) -> Counter[str] | None:
    """Guarded subset-collapse fallback: rescue a modal grant whose minority frames are a
    dropped-icon SUBSET of it. Runs ONLY after the unchanged unanimity clause returned
    ``None``. Precedence: guard 3 (drop >3-card reads) -> subset-collapse (guards 1+2) ->
    re-test the collapsed tally with unanimity (>= :data:`MIN_GRANT_CONSENSUS_FRAMES`) OR
    the existing overwhelming-dominance rule (>= :data:`DOMINANT_READ_MIN_READS` and
    >= :data:`DOMINANT_READ_MIN_FRAC`). Genuine bimodality (no subset relation) still fails
    closed. Records ``diag["accepted_by"]`` / ``diag["collapsed"]`` on accept.

    Residual risk (panel-acknowledged, unchanged from today's exposure — not new risk):
    (a) a TRUE 2-card grant whose frames also produced a spurious 3-card superset read
    could collapse the wrong way; it is backstopped by the UNCHANGED settlement-matching
    anchor (``orientation.py`` checks BOTH settlements) at the pre-existing p~=2/28~=0.07
    multiset-collision rate; (b) at that same collision rate a colliding wrong superset
    could mis-pin the granting vertex (order-establishment corruption) — the same
    exposure any accepted-but-colliding multiset already carries."""
    tally = _grant_reads_tally(frames_boxes, palette)
    if not tally:
        return None
    collapsed, events = _subset_collapse_tally(tally)
    total = sum(collapsed.values())
    distinct = list(collapsed)
    accepted: Counter[str] | None = None
    accepted_by: str | None = None
    if len(distinct) == 1 and collapsed[distinct[0]] >= MIN_GRANT_CONSENSUS_FRAMES:
        accepted = Counter(dict(distinct[0]))
        accepted_by = "subset_collapse" if events else "consensus"
    else:
        top, top_n = collapsed.most_common(1)[0]
        if total >= DOMINANT_READ_MIN_READS and top_n / total >= DOMINANT_READ_MIN_FRAC:
            accepted = Counter(dict(top))
            accepted_by = "subset_collapse" if events else "dominant_read"
    if accepted is not None and diag is not None:
        diag["accepted_by"] = accepted_by
        if events:
            diag["collapsed"] = events
    return accepted


def _drop_oversize_reads(
    frames_boxes: list[tuple[npt.NDArray[np.uint8], list[tuple[int, int, int, int]]]],
    palette: Any = None,
) -> list[tuple[npt.NDArray[np.uint8], list[tuple[int, int, int, int]]]]:
    """Guard 3 at the frame level: drop frames whose grant read exceeds
    :data:`MAX_GRANT_CARDS` cards (a 4+-box detector error must never be a consensus vote).
    Unreadable (``None``) frames are kept — :func:`consensus_granted_glyphs` drops them."""
    out = []
    for crop, boxes in frames_boxes:
        r = (
            classify_granted_glyphs(crop, boxes)
            if palette is None
            else classify_granted_glyphs(crop, boxes, palette)
        )
        if r is None or sum(r.values()) <= MAX_GRANT_CARDS:
            out.append((crop, boxes))
    return out


def _consensus_grant(
    handle: str,
    grant_frames: tuple[DecodedFrame, ...],
    handles: Sequence[str] = (),
    diag: dict[str, Any] | None = None,
) -> Counter[str] | None:  # pragma: no cover - real-run path
    """Multi-frame CONSENSUS grant read for one player across the grant-line frames.

    Re-detects the grant-icon boxes per frame (:func:`detect_glyph_boxes`) over the
    ``"received starting resources"`` line's ~30s on-screen lifetime and feeds every
    readable ``(log_crop, boxes)`` pair to :func:`consensus_granted_glyphs`, which
    requires ≥2 frames to agree byte-identical (fail-closed, brief §5.2). ``None``
    when the boxes cannot be localised or the reads disagree — the game then falls
    out as glyph-unreadable rather than feeding the firewall a fabricated multiset.

    ``handles`` (both of the game's handles) is forwarded to :func:`_grant_line_boxes`
    so the grant line's actor is resolved FUZZILY — an OCR-mangled handle otherwise
    matches no line and silently kills every game of the video (see that function).
    """
    frames_boxes: list[tuple[npt.NDArray[np.uint8], list[tuple[int, int, int, int]]]] = []
    n_line_found = 0
    for frame in grant_frames:
        crop = crop_log(frame.frame)
        line_box, text_boxes = _grant_line_boxes(crop, handle, handles)
        if line_box is None:
            continue
        n_line_found += 1
        boxes = detect_glyph_boxes(crop, line_box, text_boxes)
        if boxes:
            frames_boxes.append((crop, boxes))

    if diag is not None:
        # Which of the three gates starved, so a `glyph_unreadable` reject explains
        # ITSELF instead of needing a bespoke probe. (Two mis-diagnoses this session
        # came from theorising about this path instead of measuring it.)
        diag.update(
            grant_frames=len(grant_frames),
            line_found=n_line_found,
            with_glyph_boxes=len(frames_boxes),
        )

    if len(frames_boxes) < 2:
        if diag is not None:
            diag["fail"] = (
                "no_grant_line_in_any_frame"
                if n_line_found == 0
                else ("line_found_but_no_glyph_boxes" if not frames_boxes else "only_1_readable")
            )
        return None

    # Guard 3 (FIX A): a 4+-box read is a detector error — never a consensus vote.
    frames_g3 = _drop_oversize_reads(frames_boxes)
    result = consensus_granted_glyphs(frames_g3)
    if result is None:
        # Unanimity failed. Fall to the guarded subset-collapse rescue (dropped-icon
        # partials) which, on its collapsed tally, also applies the overwhelming-dominance
        # rule. Genuine bimodality (no subset relation) still fails closed. See
        # _collapse_grant_read + DOMINANT_READ_MIN_READS.
        result = _collapse_grant_read(frames_g3, diag=diag)
    if diag is not None and result is None:
        # >=2 frames had boxes yet consensus still refused: the reads DISAGREED (the
        # unanimity rule, fail-closed by design) or too few classified.
        reads = [
            r
            for crop, boxes in frames_boxes
            if (r := classify_granted_glyphs(crop, boxes)) is not None
        ]
        diag["classified"] = len(reads)
        # COUNTS, not just the distinct set. 45-vs-3 is OCR noise around ONE true
        # multiset; 25-vs-23 is genuine bimodality that MUST keep failing closed. The
        # consensus rule cannot be chosen without knowing which — and the current
        # STRICT-UNANIMITY rule is incoherent at scale: it accepts a 2-of-2 read while
        # rejecting 45-of-48, which is far stronger evidence. (The dense pass raised the
        # frame count and exposed this.)
        tally = Counter(tuple(sorted(r.items())) for r in reads)
        diag["read_counts"] = [{"read": dict(k), "frames": n} for k, n in tally.most_common()]
        diag["fail"] = "reads_disagree" if len(tally) > 1 else "too_few_classified"
    return result


#: Cached easyocr reader for the box-returning grant-line OCR (built once — a fresh
#: reader per grant frame per player dominated the real-run wall clock).
_BOXES_READER: Any = None


def _boxes_reader() -> Any:  # pragma: no cover - real-run path (easyocr, mocked in CI)
    """Lazily build + cache a CPU easyocr reader for :func:`_grant_line_boxes`."""
    global _BOXES_READER
    if _BOXES_READER is None:
        import easyocr  # lazy: never imported on the headless CI path

        _BOXES_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _BOXES_READER


def _grant_line_boxes(
    crop_rgb: npt.NDArray[np.uint8],
    handle: str,
    handles: Sequence[str] = (),
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:  # pragma: no cover
    """OCR the log crop with bounding boxes; return the grant line's box + all text boxes.

    Uses easyocr's box-returning read (the same lazy reader logparse/board_cv use) to
    locate ``handle``'s ``"received starting resources"`` line and every text box the
    grant-icon detector must avoid. The reader is built once and cached
    (:func:`_boxes_reader`) — the model load is ~seconds, and this runs once per
    grant frame per player, so a fresh reader per call dominated the harvest wall
    clock.

    **The handle is matched FUZZILY, not by substring.** OCR mangles handles mid-word
    (``"rayman147"`` reads as ``"raymani47"`` / ``"rayman|47"`` — the exact case
    :mod:`~catan_rl.human_data.logparse` documents), and an ``in``-substring test on
    the OCR text then MISSES the line entirely: no line box ⇒ no glyph boxes ⇒ the
    grant consensus returns ``None`` ⇒ the game rejects ``glyph_unreadable``. Because
    the glyph anchor needs BOTH players' grants, one mangled handle silently killed
    EVERY game of a video (measured: 6/6 games of ``9Sm86ml04aI``, whose opponent is
    ``rayman147``, had clean opening frames yet zero readable grants). So the actor is
    resolved with the module's existing fuzzy leading-token resolver
    (:func:`~catan_rl.human_data.logparse._resolve_actor`, bigram argmax over the two
    KNOWN handles — order-independent, and it will not mis-bind one player onto the
    other). ``handles`` should carry both of the game's handles so the argmax can
    discriminate; it falls back to ``(handle,)`` for callers that pass only one.
    """
    known = tuple(handles) or (handle,)
    results: list[tuple[list[list[int]], str, float]] = _boxes_reader().readtext(crop_rgb)
    text_boxes: list[tuple[int, int, int, int]] = []
    line_box: tuple[int, int, int, int] | None = None
    for poly, text, _conf in results:
        xs = [int(p[0]) for p in poly]
        ys = [int(p[1]) for p in poly]
        box = (min(xs), min(ys), max(xs), max(ys))
        text_boxes.append(box)
        if GRANT_RE.search(text.lower()) and _resolve_actor(text, known) == handle:
            line_box = box
    return line_box, text_boxes
