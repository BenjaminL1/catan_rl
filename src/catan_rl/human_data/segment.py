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

#: How a game window ended. ``"victory"`` = exactly one resolvable victory LOG
#: line inside the window (the ONLY scoreboard-terminal cause, §5.1); ``"resign"``
#: = a ``"<player> has left the game"`` line; ``"cutoff"`` = none of the above was
#: seen (the stream ended mid-game or the game was not sampled to its end);
#: ``"weld"`` = the window contains **≥2 DISTINCT own-game winners**, i.e. two
#: back-to-back games merged into one window because the intervening reset marker
#: was OCR-corrupted below the reset regex (``"Happy setting"`` etc.). Note it is
#: DISTINCT winners, not raw victory events: a single game's terminal line re-OCR'd
#: by ≥2 sampled frames (the same winner twice) is ONE victory, not a weld. A weld
#: is
#: definitionally ambiguous (which board/openings pair with which winner?), so it
#: yields **no winner** and is NOT scoreboard-terminal — latching the first
#: victory would be the §5.1 confidently-wrong outcome. Only ``"victory"`` yields
#: a winner.
GameEndCause = Literal["victory", "resign", "cutoff", "weld"]

#: Event kinds that count as substantive in-game play. Used both by the boundary
#: hygiene rule (a lingering reset re-OCR'd across frames is only a "same game"
#: duplicate while no new gameplay separates it) and by the stale-victory gate in
#: :func:`_own_game_victories` (a game_reset / unknown / chat / victory / resign
#: line does NOT establish that the window's own game has begun). A victory
#: preceded by none of these in-window is a stale carry-over from the previous game
#: (§5.1) and must not latch a winner.
_GAMEPLAY_KINDS: frozenset[str] = frozenset(
    {
        "setup_settlement",
        "setup_road",
        "starting_resources",
        "roll",
        "got_resources",
        "built_settlement",
        "built_city",
        "built_road",
        "bought_dev",
        "used_dev",
        "moved_robber",
        "stole",
        "bank_trade",
    }
)

#: Event kinds that unambiguously end a game (a 15-VP victory or a resign). Once
#: one is seen since the current window's start, the NEXT reset unambiguously opens
#: a fresh game regardless of whether its OCR text repeats the previous reset — the
#: only case where an identical-text reset is NOT a lingering scrolling-panel
#: duplicate.
_TERMINAL_KINDS: frozenset[str] = frozenset({"victory", "resign"})

#: Setup-phase event kinds (the snake-draft placement burst). A fresh burst of
#: these AFTER an own-game victory is a structural weld signal (a new game welded
#: into the window without an intervening reset — see :func:`_has_post_victory_setup`).
_SETUP_KINDS: frozenset[str] = frozenset({"setup_settlement", "setup_road", "starting_resources"})


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

        A window is scoreboard-terminal iff it ``ended_by == "victory"`` (which
        ``_resolve_window_outcome`` only sets when the window holds **exactly one
        DISTINCT** own-game winner) AND has a resolved ``winner``. The
        single-distinct-winner requirement is re-asserted here directly on
        ``events`` as a belt-and-braces weld guard: a window that holds ≥2
        **distinct** own-game winners is a weld (two games merged by a corrupted
        reset marker) and must never pair one game's board/openings with a winner
        drawn from a two-winner stream (the §5.1 confidently-wrong failure).
        ``_resolve_window_outcome`` already maps such a window to
        ``ended_by="weld"`` / ``winner=None``, so this is a defensive double-check,
        not the sole gate.

        Crucially the guard counts **distinct** winners, not raw victory events: a
        single legitimate game whose final victory line is re-OCR'd by ≥2
        consecutive sampled frames (the most common real terminal shape — the win
        modal lingers on the scrolling panel for several seconds; brief §5.10) is
        ONE victory for ONE winner, not a weld. Only ≥2 DISTINCT winners are two
        games merged.

        The distinct-winner count has a blind spot: two back-to-back games BOTH
        won by the same player, welded with NO intervening reset (game N's reset
        marker OCR-corrupted below the reset regex), yield ``{ThePhantom}`` — one
        distinct winner — yet span two games. ThePhantom wins the large majority of
        his games, so consecutive SAME-winner games are the COMMON weld shape, and
        the distinct-winner guard is weakest exactly where welds are most frequent.
        So a **second, winner-independent structural guard** is asserted here too:
        a fresh setup burst AFTER an own-game victory (:func:`_has_post_victory_setup`)
        is a new game's draft welded into the window — never scoreboard-terminal.
        """
        return (
            self.ended_by == "victory"
            and self.winner is not None
            and len(set(_own_game_victories(self.events))) == 1
            and not _has_post_victory_setup(self.events)
        )


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

    The parse path emits an OVERLAPPING multi-frame stream, not a de-duplicated
    one (§5.10 mandates Pass-A sampling at 1 frame / 3-5s; the Colonist log is a
    SCROLLING panel, so consecutive sampled frames re-OCR the recent tail). So a
    lingering line — the "Happy settling" reset or a "won the game" victory — is
    re-OCR'd across frames and re-appears later in the flattened stream. Two
    invariants defend against that WITHOUT globally de-duplicating the stream
    (which would wrongly collapse two different games' legitimately-identical lines
    like ``"ThePhantom rolled"``):

    - **Boundary hygiene (§5.3):** a ``"game_reset"`` opens a NEW game only if it is
      the first reset, OR a terminal event (victory / resign) was seen since this
      window's start (the game demonstrably ended, so even a verbatim-repeated reset
      text is a fresh game), OR its text is new to this window AND new gameplay
      separated it from the opener. A consecutive reset run (easyocr splitting
      "Happy settling" / "/help") and a same-text reset re-shown across frames while
      the game is still unfolding both collapse onto the SAME start. All per-window
      boundary state is cleared at each real start, so a later game's
      legitimately-identical reset / gameplay text is never conflated with an
      earlier game's.

    - **Own-game victory (§5.1):** see :func:`_resolve_window_outcome` — a victory
      only latches / counts if preceded in-window by this game's own gameplay, so a
      stale victory carried past the boundary neither wins the next game nor trips
      the weld guard.

    Events before the first reset are dropped — we never fabricate a game start
    from a mid-stream sample.

    Each window's ``winner`` / ``ended_by`` is computed from the events **inside
    that window only** (§5.1 firewall): the sole own-game ``"victory"`` event's
    actor is the winner (``ended_by="victory"``); a window with ≥2 own-game victory
    events is a weld and yields no winner (``ended_by="weld"``); else a ``"resign"``
    event closes it with no winner (``ended_by="resign"``); else the window ended
    by cutoff (``ended_by="cutoff"``, no winner).

    Returns the games in source order. An empty stream, or a stream with no reset
    marker, yields ``[]`` (no bounded game).
    """
    if len(handles) != 2 or len(set(handles)) != 2:
        raise ValueError(f"expected exactly two distinct player handles, got {handles!r}")

    # Decide, for each "game_reset", whether it OPENS a new game or is a lingering
    # re-OCR of the current window's reset (the scrolling-panel artifact §5.10). A
    # reset opens a new game iff ANY of:
    #   * it is the very first reset in the stream; OR
    #   * a terminal event (victory / resign) was seen since this window's start —
    #     the game demonstrably ended, so the next reset is a fresh game even when
    #     its OCR text repeats the previous reset verbatim; OR
    #   * this reset's text has NOT already appeared in the current window AND new
    #     gameplay has appeared since the last reset — a genuinely different reset
    #     line separated from the opener by real play.
    # A BYTE-IDENTICAL reset with NO terminal seen is DELIBERATELY NOT split (design
    # decision — favour merge-then-discard over a guessed cut). Such a reset is
    # ambiguous: it is EITHER a lingering re-OCR of this game's own opener (mid-game,
    # once both seats have rolled) OR a genuine back-to-back game N+1 whose terminal
    # for game N was missed. The event stream alone cannot tell these apart, and a
    # wrong SPLIT fabricates a winnerless seed + a hollow terminal (§5.1
    # confidently-wrong), whereas a wrong MERGE yields a 2-game window that fails the
    # record firewall (>4 setup placements) and is safely DROPPED. So we never open
    # on the identical-reset signal; genuine back-to-back games with a missed terminal
    # merge into one window that downstream rejects. (A previous `>=2 rollers`
    # separator was removed: both seats roll in the FIRST round of every game, so it
    # false-split a single game on any spurious mid-game reset re-OCR.)
    # Everything else (a consecutive reset run — easyocr splitting "Happy settling"
    # / "/help"; a same-text reset re-shown across frames while the game is still
    # unfolding, even interleaved with dribbled-in setup lines) collapses onto the
    # SAME start. All per-window state is cleared at each real start, so a LATER
    # game's legitimately-identical reset / gameplay text is never conflated with an
    # earlier game's.
    starts: list[int] = []
    reset_texts_in_window: set[str] = set()
    terminal_since_start = False
    new_gameplay_since_reset = False
    for i, event in enumerate(events):
        if event.kind == "game_reset":
            opens = (
                not starts
                or terminal_since_start
                or (event.text not in reset_texts_in_window and new_gameplay_since_reset)
            )
            if opens:
                starts.append(i)
                reset_texts_in_window = {event.text}
                terminal_since_start = False
            else:
                reset_texts_in_window.add(event.text)
            new_gameplay_since_reset = False
        else:
            if event.kind in _TERMINAL_KINDS:
                terminal_since_start = True
            elif event.kind in _GAMEPLAY_KINDS:
                new_gameplay_since_reset = True

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


def _own_game_victories(window: Sequence[LogEvent]) -> list[str]:
    """Return the resolved winners of this window's OWN-game victory events (§5.1).

    A ``"victory"`` event counts as this game's own only if it is preceded
    IN-WINDOW by at least one substantive gameplay event (:data:`_GAMEPLAY_KINDS`)
    and carries a resolved actor. A victory as the first substantive event after
    the opening reset is a stale carry-over of the PREVIOUS game's win (the
    scrolling panel had not scrolled it off yet) lingering past the boundary in the
    flattened multi-frame stream — it is NOT this game's win and is excluded here,
    so it neither latches a winner nor trips the weld guard.
    """
    winners: list[str] = []
    seen_gameplay = False
    for event in window:
        if event.kind in _GAMEPLAY_KINDS:
            seen_gameplay = True
        elif event.kind == "victory" and event.actor is not None and seen_gameplay:
            winners.append(event.actor)
    return winners


def _has_post_victory_setup(window: Sequence[LogEvent]) -> bool:
    """Whether a fresh setup burst begins AFTER an own-game victory in the window.

    A winner-INDEPENDENT structural weld signal (finding: same-winner weld). When
    two back-to-back games weld with NO intervening reset (game N's reset marker
    OCR-corrupted below the reset regex), the window holds two full drafts and two
    victory lines. If both games were won by the SAME player, the distinct-winner
    guard sees one winner and passes the weld as a clean victory — its blind spot,
    and the COMMON weld shape (ThePhantom wins the large majority of his games).

    A snake-draft placement (:data:`_SETUP_KINDS`) occurring AFTER an own-game
    victory (a victory preceded in-window by this game's own gameplay, so a stale
    carry-over of the previous game's win does not count) can only be a NEW game's
    opening draft welded into the window — no legal 1v1 game drafts again after a
    15-VP win. This catches the same-winner weld the distinct-winner count
    structurally cannot. Note it keys on setup-AFTER-victory, not a raw setup-phase
    count: a single game whose opening draft is re-OCR'd across sampled frames (the
    scrolling-panel artifact §5.10) shows two setup runs with NO victory between
    them, so it is correctly NOT flagged here.
    """
    seen_gameplay = False
    seen_own_victory = False
    for event in window:
        if event.kind in _GAMEPLAY_KINDS:
            seen_gameplay = True
        if event.kind == "victory" and event.actor is not None and seen_gameplay:
            seen_own_victory = True
        elif event.kind in _SETUP_KINDS and seen_own_victory:
            return True
    return False


def _resolve_window_outcome(window: Sequence[LogEvent]) -> tuple[str | None, GameEndCause]:
    """Compute ``(winner, ended_by)`` from a single game window (§5.1).

    Counts only this window's OWN-game victories (:func:`_own_game_victories` —
    stale carry-over victories from the previous game are excluded), then reduces
    them to the set of **distinct winners**. Then:

    - **A fresh setup burst AFTER an own-game victory → weld**
      (:func:`_has_post_victory_setup`). A winner-INDEPENDENT guard that catches the
      same-winner weld the distinct-winner count structurally cannot: two games BOTH
      won by the same player, welded with no intervening reset, hold one distinct
      winner but a second draft after the first game's win. Checked FIRST so this
      shape is flagged before the distinct-winner branch mis-reads it as a clean
      victory.
    - **≥2 DISTINCT own-game winners → weld.** The window is two back-to-back games
      merged into one because the intervening reset marker was OCR-corrupted below
      the reset regex (``"Happy setting"`` etc.). It is definitionally ambiguous
      (which board/openings pair with which winner), so it yields no winner and
      ``ended_by="weld"`` — never the first victory latched (that is the §5.1
      confidently-wrong outcome the brief forbids).
    - **exactly 1 distinct winner → victory** with that winner. This is the common
      real case even when the terminal victory line is re-OCR'd across ≥2 sampled
      frames (the win modal lingers on the scrolling panel; brief §5.10) — the same
      winner recorded twice (with or without re-OCR'd tail gameplay between the
      copies) is ONE game's win, NOT a weld. Counting distinct winners (not raw
      victory events) is what stops that most-common shape from being dropped from
      the scoreboard (winner mis-set to ``None``).
    - **0 own-game victories →** a resign line closes with no winner
      (``ended_by="resign"``); else the window ended by cutoff. Never infers a
      winner from a non-victory line.
    """
    winners = _own_game_victories(window)
    distinct_winners = set(winners)
    if _has_post_victory_setup(window):
        return None, "weld"
    if len(distinct_winners) >= 2:
        return None, "weld"
    if len(distinct_winners) == 1:
        return winners[0], "victory"
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
