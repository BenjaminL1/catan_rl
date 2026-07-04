"""Game-boundary detection + the 1v1 ruleset filter + manifest-driven opponent
strength (the Stage-1 ``segment`` slice, build brief §4).

A single ThePhantom video contains **MANY back-to-back 1v1 games** (build brief
§5.3). This module slices the corpus-wide, ordered :class:`LogEvent` stream
produced by :func:`catan_rl.human_data.logparse.parse_log` into per-game
:class:`GameSegment` windows, applies the 1v1 ruleset filter, and reads the
per-game opponent strength from the committed strength manifest.

Correctness constraints this module is built around (build brief §5):

- **Segment on the reset marker** (§5.3): the ``"Happy settling! … List of
  commands: /help"`` new-game placeholder (parsed as a ``"game_reset"``
  :class:`LogEvent`) starts a game; the game runs until the next reset (or the
  end of the stream). easyocr splits the two reset lines, so a run of
  consecutive resets with no gameplay between them is ONE game start, not an
  empty game. Events before the first reset are pre-corpus noise and are dropped
  (we never fabricate a game start).

- **Winner is the victory LOG line ONLY, per window** (§5.1): each segment's
  winner is read from a ``"victory"`` event **inside that window** — it never
  leaks across a boundary. A window with no resolvable victory line ends by
  resign (``"resign"`` event) or by cutoff (stream ended mid-game); both leave
  ``winner=None`` (the game is scoreboard-ineligible but its board + openings may
  still be seeds).

- **1v1 ruleset filter** (§1): a game is ruleset-legal iff it shows **exactly two
  distinct actors** (both 1v1 seats acted). A third distinct actor is a
  mis-segmented / non-1v1 window; a single actor is a broken parse. There is **no
  player-to-player-trade event kind** in the log grammar (only ``bank_trade``),
  so "exactly 2 distinct actors" IS the no-P2P / 2-player invariant — a P2P trade
  would require a third-party actor or an unsupported event and cannot pass.
  ``is_scoreboard_terminal`` additionally requires a 15-VP victory line.

- **Opponent strength is read from the committed manifest BY ``video_id``**
  (§5.5): :func:`segment_opponent_strength` looks the video up in
  ``data/human/strength_manifest.json`` and derives the :class:`OpponentStrength`
  via :func:`catan_rl.human_data.record.derive_opponent_strength` — never a
  rank-OCR guess and never the old known-high-rank-window placeholder. An
  ``excluded`` or manifest-absent video yields ``None`` (NO record).

CPU-only. Never imports ``gui/`` or the training path (build brief §6). Pure
value logic over the already-parsed event stream; the only I/O is reading the
committed manifest JSON.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from catan_rl.human_data.logparse import LogEvent
from catan_rl.human_data.record import OpponentStrength, derive_opponent_strength

#: How a game window ended. ``"victory"`` = a resolvable victory LOG line inside
#: the window (the ONLY scoreboard-terminal cause, §5.1); ``"resign"`` = a
#: ``"<player> has left the game"`` line; ``"cutoff"`` = neither was seen (the
#: stream ended mid-game or the game was not sampled to its end). Only
#: ``"victory"`` yields a winner.
GameEndCause = Literal["victory", "resign", "cutoff"]


@dataclass(frozen=True, slots=True)
class GameSegment:
    """One back-to-back game sliced out of a video's corpus-wide event stream.

    ``events`` is the ordered slice for this game (from its opening reset up to,
    but not including, the next reset). ``winner`` is the victory-line winner
    **inside this window** (§5.1) or ``None`` on resign / cutoff. ``ended_by``
    records which of the three end-causes closed the window. ``actors`` is the set
    of distinct resolved actor handles seen in the window (the ruleset filter's
    unit).

    Frozen + ``slots`` so segments are hashable and cheap.
    """

    events: tuple[LogEvent, ...]
    winner: str | None
    ended_by: GameEndCause
    actors: frozenset[str]

    def is_scoreboard_terminal(self) -> bool:
        """Whether this game reached a 15-VP victory (the only scoreboard-terminal
        cause, §5.1). A resign / cutoff game is kept (a possible seed) but is not
        scoreboard-terminal — deriving a winner from it fabricates an outcome.
        """
        return self.ended_by == "victory" and self.winner is not None


def segment_games(
    events: Sequence[LogEvent],
    handles: Sequence[str],
) -> list[GameSegment]:
    """Slice a corpus-wide event stream into per-game :class:`GameSegment` windows.

    ``events`` is the full ordered :class:`LogEvent` stream for one video (from
    :func:`catan_rl.human_data.logparse.parse_log` over every sampled frame's OCR,
    in source order). ``handles`` are the two known player handles for the video
    (used only to validate the count; actor resolution already happened in
    ``parse_log``).

    Boundary rule (§5.3): a ``"game_reset"`` event opens a game; the game runs
    until the next ``"game_reset"`` (or the end of the stream). A run of
    consecutive resets with no intervening non-reset event is ONE game start
    (easyocr splits the "Happy settling" / "/help" reset across lines). Events
    before the first reset are dropped — we never fabricate a game start from a
    mid-stream sample.

    Each window's ``winner`` / ``ended_by`` is computed from the events **inside
    that window only** (§5.1 firewall): the first resolvable ``"victory"`` event's
    actor is the winner (``ended_by="victory"``); else a ``"resign"`` event closes
    it with no winner (``ended_by="resign"``); else the window ended by cutoff
    (``ended_by="cutoff"``, no winner).

    Returns the games in source order. An empty stream, or a stream with no reset
    marker, yields ``[]`` (no bounded game).
    """
    if len(handles) != 2 or len(set(handles)) != 2:
        raise ValueError(f"expected exactly two distinct player handles, got {handles!r}")

    # Collect the index of every event that opens a game: a "game_reset" that is
    # NOT immediately preceded (ignoring only other resets) by a reset — i.e. the
    # FIRST reset of a consecutive run. A run of back-to-back resets is one start.
    starts: list[int] = []
    prev_was_reset = False
    for i, event in enumerate(events):
        is_reset = event.kind == "game_reset"
        if is_reset and not prev_was_reset:
            starts.append(i)
        prev_was_reset = is_reset

    if not starts:
        return []

    segments: list[GameSegment] = []
    for s_idx, start in enumerate(starts):
        end = starts[s_idx + 1] if s_idx + 1 < len(starts) else len(events)
        window = tuple(events[start:end])
        winner, ended_by = _resolve_window_outcome(window)
        actors = frozenset(e.actor for e in window if e.actor is not None)
        segments.append(GameSegment(events=window, winner=winner, ended_by=ended_by, actors=actors))
    return segments


def _resolve_window_outcome(window: Sequence[LogEvent]) -> tuple[str | None, GameEndCause]:
    """Compute ``(winner, ended_by)`` from a single game window (§5.1).

    A victory line with a resolved actor wins (first one latches); else a resign
    line closes with no winner; else the window ended by cutoff. Never infers a
    winner from a non-victory line.
    """
    for event in window:
        if event.kind == "victory" and event.actor is not None:
            return event.actor, "victory"
    for event in window:
        if event.kind == "resign":
            return None, "resign"
    return None, "cutoff"


def ruleset_ok(segment: GameSegment) -> bool:
    """Whether a game window satisfies the 1v1 Colonist ruleset filter (§1).

    Legal iff the window shows **exactly two distinct actors** — both 1v1 seats
    acted. A third distinct actor is a mis-segmented or non-1v1 window; a single
    actor is a broken parse (one seat's lines were dropped). There is no
    player-to-player-trade event kind in the log grammar, so this two-actor check
    IS the no-P2P / 2-player invariant (a P2P trade cannot appear without an
    unsupported event or a third actor).

    Note this is a NECESSARY structural gate, not the full scoreboard predicate:
    :meth:`GameSegment.is_scoreboard_terminal` additionally requires a 15-VP
    victory line, and the downstream record's opponent-strength / cross-check
    gates apply on top.
    """
    return len(segment.actors) == 2


# --- opponent strength from the committed manifest (§5.5) --------------------


def load_strength_manifest(path: str | Path) -> dict[str, Any]:
    """Load the committed strength manifest JSON (``data/human/strength_manifest.json``).

    Returns the parsed top-level object (``{"videos": [...], ...}``). The manifest
    is the source of truth for opponent strength (built by
    ``scripts/build_strength_manifest.py``); this slice only reads it.
    """
    text = Path(path).read_text(encoding="utf-8")
    payload: dict[str, Any] = json.loads(text)
    return payload


def manifest_entry(manifest: dict[str, Any], video_id: str) -> dict[str, Any] | None:
    """Return the manifest row for ``video_id`` (or ``None`` if absent)."""
    for row in manifest.get("videos", []):
        if row.get("video_id") == video_id:
            return dict(row)
    return None


def segment_opponent_strength(manifest: dict[str, Any], video_id: str) -> OpponentStrength | None:
    """Derive a video's :class:`OpponentStrength` from the manifest, by ``video_id``.

    THE opponent-strength source of truth for the segment/record layer (§5.5):
    looks the video up in the committed manifest and runs
    :func:`catan_rl.human_data.record.derive_opponent_strength`, which gates on the
    manifest ``strength`` field (``high`` → a high record with ``source`` mapped
    ``ranked_rank``→``rank_badge`` / ``tournament``→``tournament``; ``unknown`` →
    ``tier="unknown"``; ``excluded`` → ``None`` = NO record). A video ABSENT from
    the manifest also yields ``None`` — never a fabricated tier and never the old
    known-high-rank-window placeholder.

    ``high`` feeds the scoreboard; ``high`` + ``unknown`` feed the seed corpus;
    ``excluded`` (and manifest-absent) are dropped.
    """
    entry = manifest_entry(manifest, video_id)
    if entry is None:
        return None
    return derive_opponent_strength(entry)
