"""E2E harvest-driver tests (the missing Stage-3 driver, build brief §4 / step6 §3.1).

``harvest.parse_video`` is the per-video game-loop glue :func:`run_batch` injects:
it runs ingest → context → segment, then per game window fuses the Stage-2 CV
inputs through :func:`~catan_rl.human_data.validate.cross_check` and applies the
LOG-side placement-order gate on top. The heavy CV/OCR reads are the injected seam
(``_ingest`` / ``_extract_context`` / ``_read_game_inputs``, monkeypatched here), so
these tests exercise the REAL decision logic — ``cross_check`` and
``establish_placement_order`` run unmocked over controlled artefacts:

- the accept path (→ corpus, order established, scoreboard-eligible);
- each typed reject path (desert-weld / anchor-unreadable / board-unreadable);
- order-unestablished flagging (accepted, EVAL-excluded, still seed-eligible);
- the ruleset filter dropping a non-1v1 window;
- run-level telemetry + a resumable second run.

The real CV/OCR default stages are NOT exercised (no easyocr/yt-dlp/ffmpeg on CI);
they are mocked wholesale, exactly as ``test_batch`` injects its ``parse_fn``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import catan_rl.human_data.harvest as harvest
from catan_rl.human_data.batch import BatchResult
from catan_rl.human_data.board_cv import BoardRead
from catan_rl.human_data.glyph_anchor import GlyphValidation, glyph_classifier_fingerprint
from catan_rl.human_data.harvest import (
    GameInputs,
    VideoContext,
    compute_corpus_telemetry,
    parse_video,
    run_harvest,
)
from catan_rl.human_data.logparse import LogEvent
from catan_rl.human_data.openings import OpeningResult
from catan_rl.human_data.orientation import granted_resources_under_orientation
from catan_rl.human_data.record import (
    PROVENANCE_PLACEMENT_ORDER_ESTABLISHED,
    GameRecord,
    PlayerOpening,
)
from catan_rl.human_data.topology import load_topology
from catan_rl.human_data.validate import GLYPH_UNREADABLE_REASON

_TOPO = load_topology()

#: A PASS validation bound to the current classifier (run_batch is HARD-GATED on it).
_VALIDATION = GlyphValidation(
    passed=True,
    n_frames=24,
    n_correct=24,
    accuracy=1.0,
    reason=None,
    classifier_fingerprint=glyph_classifier_fingerprint(),
)

# --- a legal standard game-1 board + openings (mirrors test_batch/test_validate) ---

_HEXES: tuple[dict[str, Any], ...] = (
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
_OPENINGS: dict[str, PlayerOpening] = {
    "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
    "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
}
_HANDLES = ("ThePhantom", "rayman147")
_PLAYERS = {"agent": "ThePhantom", "opponent": "rayman147"}


def _board_read() -> BoardRead:
    return BoardRead(
        hexes=_HEXES,
        affine=np.eye(2, 3, dtype=np.float64),
        vertex_px=np.zeros((54, 2), dtype=np.float64),
        desert_hex=11,
        residual_px=0.77,
        screen_rule_gap=43.5,
        pip_ok=True,
    )


def _grants(kind: str = "valid") -> dict[str, Counter[str] | None]:
    by_hex = {int(h["hex_id"]): str(h["resource"]) for h in _HEXES}
    grants: dict[str, Counter[str] | None] = {
        "ThePhantom": granted_resources_under_orientation(19, by_hex, _TOPO),
        "rayman147": granted_resources_under_orientation(3, by_hex, _TOPO),
    }
    if kind == "unreadable":
        grants["ThePhantom"] = None
    return grants


def _game_inputs(kind: str = "accept") -> GameInputs:
    """The Stage-2 CV inputs for one game, per scenario.

    ``accept`` — clean board + openings + grants; ``desert`` — a board/openings
    desert weld (orientation firewall reject); ``anchor_unreadable`` — one grant
    ``None`` (the joint-flip anchor cannot run); ``board_unreadable`` — no board
    (openings ``None`` carries the reason through ``cross_check``).
    """
    board = _board_read()
    opening_result = OpeningResult(openings=dict(_OPENINGS), rejection_reason=None)
    grants = _grants("unreadable" if kind == "anchor_unreadable" else "valid")
    openings_desert = 11
    if kind == "desert":
        openings_desert = 17  # board.desert_hex is 11 -> orientation firewall reject
    if kind == "board_unreadable":
        opening_result = OpeningResult(openings=None, rejection_reason="board_unreadable")
    return GameInputs(
        board=board,
        openings_desert_hex=openings_desert,
        opening_result=opening_result,
        granted_by_player=grants,
        draft_order=("ThePhantom", "rayman147", "rayman147", "ThePhantom"),
        dice_log=(8, 6),
        resolution=1080,
        ts=247,
    )


def _setup_stream(established: bool) -> list[LogEvent]:
    """Snake-draft setup lines (1→2→2→1); ``established`` grants after each 2nd."""
    ev = [
        LogEvent("game_reset", None, "happy settling"),
        LogEvent("setup_settlement", "ThePhantom", "thephantom placed a settlement"),
        LogEvent("setup_road", "ThePhantom", "thephantom placed a road"),
        LogEvent("setup_settlement", "rayman147", "rayman147 placed a settlement"),
        LogEvent("setup_road", "rayman147", "rayman147 placed a road"),
        LogEvent("setup_settlement", "rayman147", "rayman147 placed a settlement"),
        LogEvent("starting_resources", "rayman147", "rayman147 received starting resources"),
        LogEvent("setup_settlement", "ThePhantom", "thephantom placed a settlement"),
        LogEvent("starting_resources", "ThePhantom", "thephantom received starting resources"),
    ]
    if not established:
        # Drop ThePhantom's grant line so the log-side ordinal cannot fire (order
        # unestablished), while the grant-side (glyph) signal still resolves.
        ev = [e for e in ev if not (e.kind == "starting_resources" and e.actor == "ThePhantom")]
    return ev


def _events(winner: str | None = "ThePhantom", *, established: bool = True) -> tuple[LogEvent, ...]:
    ev = _setup_stream(established)
    ev.append(LogEvent("roll", "ThePhantom", "thephantom rolled a 8"))
    ev.append(LogEvent("roll", "rayman147", "rayman147 rolled a 6"))
    if winner is not None:
        ev.append(LogEvent("victory", winner, f"{winner.lower()} won the game"))
    return tuple(ev)


def _context(events: tuple[LogEvent, ...]) -> VideoContext:
    return VideoContext(
        players=dict(_PLAYERS),
        handles=_HANDLES,
        player_colors={"ThePhantom": "GREEN", "rayman147": "BLACK"},
        seat_order=_HANDLES,
        events=events,
        game_frames=(),
    )


_MANIFEST_OBJ: dict[str, Any] = {
    "schema_version": 1,
    "videos": [
        {"video_id": "vidHIGH00000", "strength": "high", "source": "tournament", "evidence": {}},
        {"video_id": "vidHIGH00001", "strength": "high", "source": "tournament", "evidence": {}},
    ],
}


def _mock_stages(
    monkeypatch: pytest.MonkeyPatch,
    events: tuple[LogEvent, ...],
    game_inputs: GameInputs,
) -> None:
    monkeypatch.setattr(harvest, "_ingest_two_pass", lambda vid, **kw: ([object()], [[]]))
    monkeypatch.setattr(
        harvest, "_extract_context", lambda vid, frames, lines=None: _context(events)
    )
    monkeypatch.setattr(harvest, "_read_game_inputs", lambda *a, **kw: game_inputs)


def _parse(
    monkeypatch: pytest.MonkeyPatch,
    events: tuple[LogEvent, ...],
    game_inputs: GameInputs,
    video_id: str = "vidHIGH00000",
) -> list[GameRecord]:
    _mock_stages(monkeypatch, events, game_inputs)
    return parse_video(video_id, manifest=_MANIFEST_OBJ, topology=_TOPO)


# --- accept path -------------------------------------------------------------


def test_accept_path_scoreboard_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _parse(monkeypatch, _events(established=True), _game_inputs("accept"))
    assert len(records) == 1
    rec = records[0]
    assert rec.passed_crosscheck
    assert rec.rejection_reason is None
    assert rec.provenance.get(PROVENANCE_PLACEMENT_ORDER_ESTABLISHED) is True
    assert rec.winner == "ThePhantom"
    assert rec.is_scoreboard_eligible()
    assert rec.is_seed_eligible()


# --- order-unestablished flagging (step6 §3.1) -------------------------------


def test_order_unestablished_flagged_not_scoreboard(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _parse(monkeypatch, _events(established=False), _game_inputs("accept"))
    assert len(records) == 1
    rec = records[0]
    # Still ACCEPTED (the CV all agreed), but the log could not confirm the
    # placement order -> flagged unestablished: EVAL-excluded, still seed-eligible.
    assert rec.passed_crosscheck
    assert rec.provenance.get(PROVENANCE_PLACEMENT_ORDER_ESTABLISHED) is False
    assert not rec.is_scoreboard_eligible()
    assert rec.is_seed_eligible()


# --- typed reject paths ------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "reason_prefix"),
    [
        ("desert", "orientation_mismatch_desert_hex"),
        ("anchor_unreadable", GLYPH_UNREADABLE_REASON),
        ("board_unreadable", "board_unreadable"),
    ],
)
def test_typed_reject_paths(monkeypatch: pytest.MonkeyPatch, kind: str, reason_prefix: str) -> None:
    records = _parse(monkeypatch, _events(), _game_inputs(kind))
    assert len(records) == 1
    rec = records[0]
    assert not rec.passed_crosscheck
    assert rec.rejection_reason is not None
    assert rec.rejection_reason.startswith(reason_prefix)
    assert not rec.is_scoreboard_eligible()
    assert not rec.is_seed_eligible()


def test_anchor_unreadable_never_accepts_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # The joint-flip firewall is NON-OPTIONAL: an unreadable grant is a typed reject,
    # never a silent accept that skips the anchor.
    records = _parse(monkeypatch, _events(), _game_inputs("anchor_unreadable"))
    assert records[0].rejection_reason == GLYPH_UNREADABLE_REASON


# --- ruleset filter drops a non-1v1 window -----------------------------------


def test_ruleset_filter_drops_single_actor_window(monkeypatch: pytest.MonkeyPatch) -> None:
    solo = (
        LogEvent("game_reset", None, "happy settling"),
        LogEvent("setup_settlement", "ThePhantom", "thephantom placed a settlement"),
        LogEvent("roll", "ThePhantom", "thephantom rolled a 8"),
        LogEvent("victory", "ThePhantom", "thephantom won the game"),
    )
    called: list[int] = []

    def _guard(*a: Any, **kw: Any) -> GameInputs:
        called.append(1)
        return _game_inputs("accept")

    monkeypatch.setattr(harvest, "_ingest_two_pass", lambda vid, **kw: ([object()], [[]]))
    monkeypatch.setattr(harvest, "_extract_context", lambda vid, frames, lines=None: _context(solo))
    monkeypatch.setattr(harvest, "_read_game_inputs", _guard)
    records = parse_video("vidHIGH00000", manifest=_MANIFEST_OBJ, topology=_TOPO)
    assert records == []
    assert called == []  # a non-game window never reaches Stage-2 CV


# --- frame routing is indexed by SEGMENT position, not the ruleset ordinal ----


def test_frame_lookup_uses_segment_index_not_ruleset_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A ruleset-failing window (single actor) precedes the real 1v1 game, so the
    # post-ruleset ordinal (1) and the raw segment position (1) diverge from the
    # frame-bucket index: the OLD code passed the ordinal, indexing ctx.game_frames
    # at 0 — the DROPPED game's frames — a cross-game weld no firewall can catch.
    # The frame lookup must use the segment position instead.
    dropped = (
        LogEvent("game_reset", None, "happy settling"),
        LogEvent("setup_settlement", "ThePhantom", "thephantom placed a settlement"),
        LogEvent("roll", "ThePhantom", "thephantom rolled a 8"),
        LogEvent("victory", "ThePhantom", "thephantom won the game"),
    )
    events = dropped + _events(established=True)
    seen_index: list[int] = []

    def _capture(
        video_id: str, segment_index: int, segment_events: Any, ctx: Any, topology: Any
    ) -> GameInputs:
        seen_index.append(segment_index)
        return _game_inputs("accept")

    monkeypatch.setattr(harvest, "_ingest_two_pass", lambda vid, **kw: ([object()], [[]]))
    monkeypatch.setattr(
        harvest, "_extract_context", lambda vid, frames, lines=None: _context(events)
    )
    monkeypatch.setattr(harvest, "_read_game_inputs", _capture)

    records = parse_video("vidHIGH00000", manifest=_MANIFEST_OBJ, topology=_TOPO)
    assert seen_index == [1]  # segment position 1, NOT the ruleset ordinal's 0
    assert len(records) == 1
    assert records[0].game_index == 1  # the record keeps the 1-based real-game ordinal


def test_post_setup_frame_picks_pre_first_build_not_end_game() -> None:
    # A game whose event window spans setup -> main-game builds. bucket[-1] is the
    # END-GAME frame (board full of pieces); the openings CV needs the 8-pieces-down
    # opening board, so _post_setup_frame returns the latest frame whose whole OCR
    # window precedes the game's FIRST main-game build.
    all_events = (
        LogEvent("setup_settlement", "a", ""),  # 0
        LogEvent("setup_road", "a", ""),  # 1
        LogEvent("starting_resources", "a", ""),  # 2  (setup ends here; 8 pieces down)
        LogEvent("roll", "a", ""),  # 3
        LogEvent("built_road", "a", ""),  # 4  <- first main-game build (boundary)
        LogEvent("built_settlement", "a", ""),  # 5
        LogEvent("victory", "a", ""),  # 6
    )
    frames = [object() for _ in range(4)]
    # frame OCR windows (global_hi): f0 covers [0,2), f1 [2,4) (still pre-build),
    # f2 [4,6) (includes the build line), f3 [6,7) (end-game).
    hi = {id(frames[0]): 2, id(frames[1]): 4, id(frames[2]): 6, id(frames[3]): 7}
    got = harvest._post_setup_frame(frames, (0, 7), all_events, hi)
    assert got is frames[1]  # latest frame fully before the first build, NOT frames[-1]


def test_post_setup_frame_no_build_falls_back_to_last() -> None:
    # A cutoff game that never left the setup board (no main-game build) — the whole
    # window is 8-pieces-down, so the last frame is a fine post-setup read.
    all_events = (
        LogEvent("setup_settlement", "a", ""),
        LogEvent("starting_resources", "a", ""),
        LogEvent("roll", "a", ""),
    )
    frames = [object(), object()]
    hi = {id(frames[0]): 2, id(frames[1]): 3}
    assert harvest._post_setup_frame(frames, (0, 3), all_events, hi) is frames[-1]


def test_post_setup_frame_build_in_opening_window_falls_back_to_first() -> None:
    # The first build already lands inside the earliest frame's OCR window — no frame
    # cleanly shows the opening board, so fall back to the first (openings then rejects).
    all_events = (
        LogEvent("setup_settlement", "a", ""),
        LogEvent("built_settlement", "a", ""),  # boundary at index 1
    )
    frames = [object(), object()]
    hi = {id(frames[0]): 2, id(frames[1]): 2}  # both windows straddle the boundary
    assert harvest._post_setup_frame(frames, (0, 2), all_events, hi) is frames[0]


# --- NOUN-LESS (icon) footage: the real-video frame-routing path ---------------
#
# Colonist renders placed/built pieces as ICONS, so real footage only ever yields
# the noun-less kinds. Before this was handled, NO build was ever seen on video, the
# boundary was always None, and _post_setup_frame returned bucket[-1] — the END-GAME
# "Well Played!" stats overlay. That fail-OPEN is what drove real-video yield to 0.


def test_post_setup_frame_bounds_on_noun_less_built_any() -> None:
    # The real-footage shape: setup_placed_any x8 then built_any (the icon-rendered
    # build). built_any MUST bound the opening window exactly like built_settlement.
    all_events = (
        LogEvent("setup_placed_any", "a", ""),  # 0
        LogEvent("setup_placed_any", "a", ""),  # 1
        LogEvent("starting_resources", "a", ""),  # 2  (8 pieces down)
        LogEvent("roll", "a", ""),  # 3
        LogEvent("built_any", "a", ""),  # 4  <- first build (boundary)
        LogEvent("victory", "a", ""),  # 5
    )
    frames = [object() for _ in range(4)]
    hi = {id(frames[0]): 2, id(frames[1]): 4, id(frames[2]): 5, id(frames[3]): 6}
    got = harvest._post_setup_frame(frames, (0, 6), all_events, hi)
    assert got is frames[1]  # the pre-build opening frame, NOT the end-game frames[-1]


def test_post_setup_frame_victory_without_build_rejects_instead_of_end_game_frame() -> None:
    # THE BUG THIS FIXES: a game that reached VICTORY but whose build lines were never
    # sampled. It cannot still be 8-pieces-down (you cannot reach 15 VP without
    # building), so the last frame is an END-GAME board. Return None (typed reject),
    # never bucket[-1].
    all_events = (
        LogEvent("setup_placed_any", "a", ""),
        LogEvent("starting_resources", "a", ""),
        LogEvent("roll", "a", ""),
        LogEvent("victory", "a", ""),
    )
    frames = [object(), object()]
    hi = {id(frames[0]): 2, id(frames[1]): 4}
    assert harvest._post_setup_frame(frames, (0, 4), all_events, hi) is None


def test_post_setup_frame_roll_without_build_still_reads_last_frame() -> None:
    # A bare roll does NOT prove the board left setup (a game can roll then be cut off
    # without building), so this window is still a legitimate 8-pieces-down read.
    all_events = (
        LogEvent("setup_placed_any", "a", ""),
        LogEvent("starting_resources", "a", ""),
        LogEvent("roll", "a", ""),
    )
    frames = [object(), object()]
    hi = {id(frames[0]): 2, id(frames[1]): 3}
    assert harvest._post_setup_frame(frames, (0, 3), all_events, hi) is frames[-1]


def test_draft_order_first_placer_from_noun_less_setup_events() -> None:
    # Real footage yields only setup_placed_any; the FIRST placement of a snake draft
    # is a settlement either way, so the first-placer identity still resolves. Robust
    # to the log panel's re-OCR duplication (which repeats lines, never reorders the
    # first occurrence).
    events = [
        LogEvent("roll", "b", ""),
        LogEvent("setup_placed_any", "b", ""),  # b places first
        LogEvent("setup_placed_any", "b", ""),  # duplicate re-OCR / the road
        LogEvent("setup_placed_any", "a", ""),
    ]
    assert harvest._draft_order(events, ("a", "b")) == ("b", "a", "a", "b")


def test_draft_order_falls_back_to_sorted_handles_with_no_placements() -> None:
    events = [LogEvent("roll", "a", ""), LogEvent("got_resources", "b", "")]
    assert harvest._draft_order(events, ("a", "b")) == ("a", "b", "b", "a")


# --- grant-line handle matching must be FUZZY (OCR mangles handles) -----------
#
# The grant line is located by OCR text. OCR mangles handles mid-word
# ("rayman147" -> "raymani47" / "rayman|47"), so an exact `handle in text` test MISSES
# the line: no line box -> no glyph boxes -> the grant consensus returns None -> the
# game rejects `glyph_unreadable`. The glyph anchor needs BOTH players' grants, so ONE
# mangled handle silently killed EVERY game of a video (measured: 6/6 games of
# 9Sm86ml04aI, opponent `rayman147`, had clean opening frames yet zero readable grants).


@pytest.mark.parametrize(
    "ocr_text",
    [
        "rayman147 received starting resources",  # clean
        "raymani47 received starting resources",  # OCR mangles the digit boundary
        "rayman|47 received starting resources",  # OCR mangles it differently
    ],
)
def test_grant_line_actor_resolves_through_ocr_mangling(ocr_text: str) -> None:
    from catan_rl.human_data.logparse import _resolve_actor

    handles = ("ThePhantom", "rayman147")
    # The line must bind to rayman147 even when the handle is mangled — this is the
    # resolution _grant_line_boxes now uses in place of a brittle substring test.
    assert _resolve_actor(ocr_text, handles) == "rayman147"


def test_grant_line_actor_does_not_misbind_the_other_player() -> None:
    from catan_rl.human_data.logparse import _resolve_actor

    handles = ("ThePhantom", "rayman147")
    # Fuzziness must not bleed one player's grant onto the other: the argmax over the
    # two KNOWN handles keeps ThePhantom's line bound to ThePhantom.
    assert _resolve_actor("ThePhantom received starting resources", handles) == "ThePhantom"


def test_dominant_segment_assigns_by_max_overlap_ties_to_later() -> None:
    # segment_games windows are contiguous; a frame is routed to the segment its
    # events overlap most (ties -> the later segment), so the router agrees with the
    # segmenter it is indexed against (the SHOULD-FIX: reuse segment_games boundaries).
    ranges = [(2, 5), (5, 9)]
    assert harvest._dominant_segment(0, 2, ranges) is None  # pre-first-reset prefix -> dropped
    assert harvest._dominant_segment(2, 4, ranges) == 0
    assert harvest._dominant_segment(2, 6, ranges) == 0  # 3 in seg0 vs 1 in seg1 -> seg0
    assert harvest._dominant_segment(4, 6, ranges) == 1  # 1 vs 1 tie -> later segment
    assert harvest._dominant_segment(6, 9, ranges) == 1
    assert harvest._dominant_segment(5, 5, ranges) is None  # zero-event frame -> carry-forward


# --- pure helpers ------------------------------------------------------------


def test_draft_order_from_first_setup_settlement() -> None:
    assert harvest._draft_order(_events(), _HANDLES) == (
        "ThePhantom",
        "rayman147",
        "rayman147",
        "ThePhantom",
    )


def test_draft_order_falls_back_to_sorted_handles_when_no_setup() -> None:
    assert harvest._draft_order((), _HANDLES) == (
        "ThePhantom",
        "rayman147",
        "rayman147",
        "ThePhantom",
    )  # min("ThePhantom","rayman147") == "ThePhantom" (uppercase sorts first)


def test_bind_colours_derives_assignment_from_pov_anchor_not_log_order() -> None:
    # §5.14 circularity fix: the handle→colour binding is DERIVED from the
    # authoritative POV seat anchor (agent=bottom self-seat, opponent=top seat),
    # NOT from the log-frequency ``handles`` order. The HUD reads top→bottom
    # GREEN/BLACK; game-1 truth is rayman147 (top/green) / ThePhantom (bottom/black).
    players = {"agent": "ThePhantom", "opponent": "rayman147"}
    player_colors, seat_order = harvest._bind_colours(players, ("GREEN", "BLACK"))
    assert seat_order == ("rayman147", "ThePhantom")  # top → bottom (opponent above POV)
    assert player_colors == {"rayman147": "GREEN", "ThePhantom": "BLACK"}
    # The POV/agent is bound to the BOTTOM seat colour regardless of log order.
    assert player_colors[players["agent"]] == "BLACK"


def test_bind_colours_rejects_non_distinct_hud() -> None:
    with pytest.raises(harvest.VideoParseError, match="two distinct seat colours"):
        harvest._bind_colours({"agent": "a", "opponent": "b"}, ("GREEN", "GREEN"))


def test_majority_two_colour_recovers_supported_game_in_compilation() -> None:
    # A compilation: half the sampled frames are a palette-supported GREEN/BLACK game
    # (two-distinct reads) and half a palette-unsupported opponent (only BLACK reads,
    # one distinct). The single-frame middle read could land on a one-colour frame and
    # fail the whole video; the vote recovers the supported game's pair.
    reads = [
        ("BLACK",),
        ("BLACK",),
        ("GREEN", "BLACK"),
        ("BLACK", "GREEN"),
        ("BLACK",),
        ("GREEN", "BLACK"),
    ]
    assert harvest._majority_two_colour(reads) == ("GREEN", "BLACK")


def test_majority_two_colour_empty_when_no_two_distinct_read() -> None:
    # Every game's opponent is palette-unsupported -> only one-colour reads -> no vote.
    assert harvest._majority_two_colour([("BLACK",), (), ("BLACK",)]) == ()


def test_dice_log_extracts_valid_rolls_only() -> None:
    events = (
        LogEvent("roll", "ThePhantom", "thephantom rolled a 8"),
        LogEvent("roll", "rayman147", "rayman147 rolled a 12"),
        LogEvent("roll", "ThePhantom", "thephantom rolled a 13"),  # out of range -> skip
        LogEvent("roll", "rayman147", "rayman147 rolled a"),  # no number -> skip
        LogEvent("got_resources", "ThePhantom", "thephantom got wood"),  # not a roll
    )
    assert harvest._dice_log(events) == (8, 12)


# --- telemetry ---------------------------------------------------------------


def _write_jsonl(path: Path, records: list[GameRecord]) -> None:
    path.write_text("\n".join(r.to_json_line() for r in records) + "\n", encoding="utf-8")


def test_compute_corpus_telemetry_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    accepted = _parse(monkeypatch, _events(established=True), _game_inputs("accept"))
    unestablished = _parse(monkeypatch, _events(established=False), _game_inputs("accept"))
    desert = _parse(monkeypatch, _events(), _game_inputs("desert"))
    unreadable = _parse(monkeypatch, _events(), _game_inputs("anchor_unreadable"))

    _write_jsonl(tmp_path / "corpus.jsonl", accepted + unestablished)
    _write_jsonl(tmp_path / "rejected.jsonl", desert + unreadable)

    telem = compute_corpus_telemetry(
        tmp_path, BatchResult(videos_processed=2, videos_skipped=1, videos_failed=3)
    )
    assert telem.games_seen == 4
    assert telem.accepted == 2
    assert telem.rejected == 2
    assert telem.order_unestablished == 1
    assert telem.rejected == 2
    reasons = telem.rejected_by_reason
    assert reasons.get(GLYPH_UNREADABLE_REASON) == 1
    desert = sum(v for k, v in reasons.items() if k.startswith("orientation_mismatch_desert_hex"))
    assert desert == 1
    # The unreadable game is anchor-can't-run; the desert reject rejects on the
    # provenance desert-binding BEFORE the anchor line, so it is not anchor_unreadable.
    assert telem.anchor_unreadable == 1
    assert telem.grant_read_coverage == pytest.approx(telem.anchor_ran / 4)
    assert telem.videos_processed == 2
    assert telem.videos_skipped == 1
    assert telem.videos_failed == 3
    assert "rejected_by_reason" in telem.to_dict()
    assert "grant_read_coverage" in telem.render()


# --- run_harvest end-to-end + resume -----------------------------------------


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "strength_manifest.json"
    path.write_text(json.dumps(_MANIFEST_OBJ), encoding="utf-8")
    return path


def test_run_harvest_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_stages(monkeypatch, _events(established=True), _game_inputs("accept"))
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "out"

    result, telem = run_harvest(
        manifest_path=manifest,
        out_dir=out,
        video_ids=["vidHIGH00000"],
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
    assert result.videos_processed == 1
    assert telem.accepted == 1
    assert telem.games_seen == 1
    assert telem.order_unestablished == 0
    corpus = (out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(corpus) == 1


def test_run_harvest_resume_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_stages(monkeypatch, _events(established=True), _game_inputs("accept"))
    manifest = _write_manifest(tmp_path)
    out = tmp_path / "out"
    ids = ["vidHIGH00000", "vidHIGH00001"]

    first, _t1 = run_harvest(
        manifest_path=manifest,
        out_dir=out,
        video_ids=ids,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
    assert first.videos_processed == 2

    second, telem = run_harvest(
        manifest_path=manifest,
        out_dir=out,
        video_ids=ids,
        max_workers=1,
        glyph_validation=_VALIDATION,
    )
    # Fully complete on disk -> the resume processes nothing and duplicates no rows.
    assert second.videos_processed == 0
    assert second.videos_skipped == 2
    corpus = (out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(corpus) == 2  # no duplicate on resume
    assert telem.accepted == 2
    assert telem.games_seen == 2


# --- DENSE SAMPLING PASS around the grant lines --------------------------------
#
# The grant consensus needs >= 2 readable frames that AGREE. At the deployed 4 s
# sparse cadence a player's grant line can be sampled in EXACTLY 2 frames, and one
# unreadable frame there (glyph_boxes=0) drops it to 1 -> consensus None ->
# `glyph_unreadable` -> the game is lost even though its opening frame is clean.
# MEASURED on 9Sm86ml04aI (6 games, clean_opening 6/6, localizable 0/6): rayman147's
# grant was sampled at t=246 (0 glyph boxes) and t=250 (readable) => 1 readable < 2.
# At 1 s over the same window: 5 readable frames, unanimous {BRICK, SHEEP, ORE}.
# The two-pass design already existed (ingest.build_sampling_schedule(dense_windows));
# harvest just never passed any. These windows are what it now passes.


def _decoded_frame(ts: float) -> Any:
    from catan_rl.human_data.ingest import DecodedFrame

    return DecodedFrame(
        ts=ts,
        pass_name="sparse",
        frame=np.zeros((1080, 1920, 3), dtype=np.uint8),
        native_resolution=1080,
    )


def test_grant_dense_windows_pads_and_clamps() -> None:
    from catan_rl.human_data.ingest import TimeWindow

    got = harvest._grant_dense_windows([5.0], duration_s=100.0, pad_s=10.0)
    # clamped at 0, not negative
    assert got == [TimeWindow(start_s=0.0, end_s=15.0)]

    got = harvest._grant_dense_windows([95.0], duration_s=100.0, pad_s=10.0)
    assert got == [TimeWindow(start_s=85.0, end_s=100.0)]  # clamped at duration


def test_grant_dense_windows_merges_overlapping() -> None:
    from catan_rl.human_data.ingest import TimeWindow

    # 246 and 250 are 4 s apart (the real sparse cadence) — with pad 10 they overlap
    # heavily and must merge into ONE window, not two decode ranges.
    got = harvest._grant_dense_windows([246.0, 250.0], duration_s=600.0, pad_s=10.0)
    assert got == [TimeWindow(start_s=236.0, end_s=260.0)]


def test_grant_dense_windows_keeps_disjoint_games_separate() -> None:
    # Two different games' grant bursts, far apart, stay as two windows.
    got = harvest._grant_dense_windows([100.0, 500.0], duration_s=600.0, pad_s=10.0)
    assert len(got) == 2
    assert (got[0].start_s, got[0].end_s) == (90.0, 110.0)
    assert (got[1].start_s, got[1].end_s) == (490.0, 510.0)


def test_grant_dense_windows_empty_when_no_grant_seen() -> None:
    # No grant line anywhere => nothing to rescue => no dense pass (no wasted OCR).
    assert harvest._grant_dense_windows([], duration_s=600.0) == []


# --- FIX B: the grant-line matcher now uses the anchor's OCR-tolerant GRANT_RE ---------
# harvest used to match the grant line by EXACT substring "received starting resources"
# at three sites; the anchor's own detector uses GRANT_RE = r"rece\w{0,4} starting
# resources" (OCR mangles "received"). A mangled line therefore routed ZERO grant frames
# (the grant_frames=0 signature of KvH76fJI4f0 g2). All three sites now alias GRANT_RE.


def test_grant_re_is_ocr_tolerant_where_the_exact_substring_missed() -> None:
    from catan_rl.human_data.glyph_anchor import GRANT_RE

    # Manglings GRANT_RE accepts (rece + <=4 word chars + " starting resources")...
    for ok in (
        "thephantom received starting resources",
        "thephantom recelved starting resources",  # 'i'->'l' OCR confusion
        "thephantom recei starting resources",  # truncated
    ):
        assert GRANT_RE.search(ok), ok
    # ...exactly the line the OLD exact-substring test dropped:
    assert harvest._GRANT_PHRASE not in "thephantom recelved starting resources"
    # non-grant lines still do not match at any site:
    assert not GRANT_RE.search("thephantom rolled a 7")
    assert not GRANT_RE.search("thephantom built a")


def test_route_frames_to_games_routes_a_mangled_grant_line() -> None:
    # Site 2 (frame routing): a frame whose grant line OCR'd as "recelved" is now selected
    # as a grant frame. With the old exact substring this bucket had grant_frames=0 and the
    # game died glyph_unreadable — the KvH76fJI4f0 g2 signature.
    handles = ("phantom", "vale")
    frames = [_decoded_frame(0.0), _decoded_frame(4.0), _decoded_frame(8.0)]
    per_frame_lines = [
        ["happy settling! list of commands: /help", "phantom placed a"],
        ["vale placed a", "phantom recelved starting resources"],  # mangled 'received'
        ["vale placed a", "phantom placed a"],
    ]
    routed = harvest._route_frames_to_games(list(frames), per_frame_lines, handles)
    assert len(routed) == 1
    assert routed[0] is not None
    assert len(routed[0].grant_frames) == 1, "mangled grant line must still route a grant frame"


def test_route_frames_to_games_no_grant_line_routes_zero() -> None:
    # Control: a bucket with NO grant line (mangled or otherwise) yields grant_frames=0.
    handles = ("phantom", "vale")
    frames = [_decoded_frame(0.0), _decoded_frame(4.0), _decoded_frame(8.0)]
    per_frame_lines = [
        ["happy settling! list of commands: /help", "phantom placed a"],
        ["vale placed a", "phantom rolled a 7"],
        ["vale placed a", "phantom placed a"],
    ]
    routed = harvest._route_frames_to_games(list(frames), per_frame_lines, handles)
    assert len(routed) == 1 and routed[0] is not None
    assert len(routed[0].grant_frames) == 0


def test_extract_context_uses_precomputed_lines_and_does_not_reocr(monkeypatch) -> None:
    # The two-pass ingest OCRs each frame ONCE and threads the lines through;
    # _extract_context must NOT run its own OCR when they are supplied (that would
    # double the dominant cost of the whole harvest).
    called = {"n": 0}

    def _boom(_crop: Any) -> list[str]:
        called["n"] += 1
        return []

    monkeypatch.setattr(harvest, "ocr_log_crop", _boom)
    monkeypatch.setattr(harvest, "_discover_handles", lambda _lines: ("a", "b"))
    monkeypatch.setattr(harvest, "_agent_binding", lambda _h: {"agent": "a", "opponent": "b"})
    monkeypatch.setattr(harvest, "_video_seat_colours", lambda _f: {})
    monkeypatch.setattr(harvest, "_bind_colours", lambda _p, _s: ({}, ()))
    monkeypatch.setattr(harvest, "_route_frames_to_games", lambda _f, _l, _h: ())

    frames = [_decoded_frame(ts=0.0), _decoded_frame(ts=4.0)]
    lines = [["a placed a"], ["a received starting resources"]]
    harvest._extract_context("vid", frames, per_frame_lines=lines)
    assert called["n"] == 0, "re-OCR'd frames whose lines were already computed"


# --- board read falls back to the game's OWN in-window frames -------------------
#
# The board is read from `setup_frames` = bucket[:len//2]. For any game after a video's
# first, those earliest frames are routinely the PREVIOUS game's end screen / lobby, so
# `read_board_stable` never sees a board even though the game's own frames read fine
# (MEASURED: 0EtcbG16kHA g1 and 9Sm86ml04aI g3 both had post_setup AND baseline reading
# cleanly, yet the game still failed `board_unreadable`).
#
# The fallback pool deliberately EXCLUDES empty_baseline (= bucket[0], which can be the
# previous game's frame — a cross-game contamination risk) and uses only frames
# guaranteed inside THIS game's window: the post-setup frame (bounded by this game's own
# first build) and the grant frames (carry this game's grant lines). The >=2-agreeing
# byte-identical stability rule is UNCHANGED, so fail-closed §5.2 semantics hold.


def test_stable_board_falls_back_to_in_window_frames(monkeypatch) -> None:
    calls: list[int] = []
    good = _board_read()

    def fake_stable(frames, **kw):
        calls.append(len(frames))
        return None if len(calls) == 1 else good  # setup_frames fail, fallback succeeds

    monkeypatch.setattr(harvest, "read_board_stable", fake_stable)
    gf = harvest.GameFrames(
        setup_frames=(_decoded_frame(0.0), _decoded_frame(4.0)),
        post_setup_frame=_decoded_frame(60.0),
        empty_baseline=np.zeros((1080, 1920, 3), dtype=np.uint8),
        grant_frames=(_decoded_frame(50.0), _decoded_frame(51.0)),
    )
    assert harvest._stable_board_for_game(gf) is good
    assert len(calls) == 2, "the fallback pool was never tried"


def test_stable_board_prefers_setup_frames_and_does_not_fall_back(monkeypatch) -> None:
    good = _board_read()
    calls: list[int] = []

    def fake_stable(frames, **kw):
        calls.append(len(frames))
        return good

    monkeypatch.setattr(harvest, "read_board_stable", fake_stable)
    gf = harvest.GameFrames(
        setup_frames=(_decoded_frame(0.0), _decoded_frame(4.0)),
        post_setup_frame=_decoded_frame(60.0),
        empty_baseline=np.zeros((1080, 1920, 3), dtype=np.uint8),
        grant_frames=(_decoded_frame(50.0),),
    )
    assert harvest._stable_board_for_game(gf) is good
    assert calls == [2], "setup_frames must stay the first (and only) try when they work"


def test_stable_board_returns_none_when_every_pool_fails(monkeypatch) -> None:
    monkeypatch.setattr(harvest, "read_board_stable", lambda frames, **kw: None)
    gf = harvest.GameFrames(
        setup_frames=(_decoded_frame(0.0), _decoded_frame(4.0)),
        post_setup_frame=_decoded_frame(60.0),
        empty_baseline=np.zeros((1080, 1920, 3), dtype=np.uint8),
        grant_frames=(_decoded_frame(50.0),),
    )
    assert harvest._stable_board_for_game(gf) is None  # still board_unreadable, fail closed


# --- crop-hash OCR cache (per-video) -----------------------------------------
# The dense pass re-decodes windows the sparse pass already touched, so identical
# log crops recur within one video; caching OCR on the crop's pixel bytes turns the
# repeats free. OCR of identical pixels is deterministic, so the cache is
# behaviour-preserving by construction — the identical-output test pins that.


def _frame_with_crop_fill(crop_val: int, outside_val: int = 0) -> Any:
    """A 1080p frame whose LOG-CROP region is filled with ``crop_val`` and whose
    area OUTSIDE the crop is ``outside_val`` (to prove the hash is crop-first)."""
    from catan_rl.human_data.logparse import LOG_CROP_FRAC

    frame = np.full((1080, 1920, 3), outside_val, dtype=np.uint8)
    x0f, _y0f, x1f, y1f = LOG_CROP_FRAC
    h, w = 1080, 1920
    x0, x1 = int(x0f * w), int(x1f * w)
    y1 = int(y1f * h)
    frame[0:y1, x0:x1] = crop_val
    return frame


def test_ocr_cache_identical_crops_ocr_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Counter[str] = Counter()

    def counting_ocr(crop: Any) -> list[str]:
        calls["n"] += 1
        return ["line"]

    monkeypatch.setattr(harvest, "ocr_log_crop", counting_ocr)
    cache: dict[bytes, list[str]] = {}
    stats: Counter[str] = Counter()
    f = _frame_with_crop_fill(5)
    a = harvest._cached_ocr_log(f, cache, stats)
    b = harvest._cached_ocr_log(f, cache, stats)
    assert calls["n"] == 1, "identical crop pixels must OCR exactly once"
    assert a == b == ["line"]
    assert stats["hits"] == 1 and stats["misses"] == 1
    # returned lists are COPIES (mutating one must not corrupt the cache/other)
    a.append("mutated")
    assert b == ["line"]
    assert harvest._cached_ocr_log(f, cache, stats) == ["line"]


def test_ocr_cache_hashes_crop_not_whole_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Counter[str] = Counter()
    monkeypatch.setattr(harvest, "ocr_log_crop", lambda crop: [str(calls.update(n=1))])
    cache: dict[bytes, list[str]] = {}
    stats: Counter[str] = Counter()
    # Same crop region, DIFFERENT pixels outside it — must still hit the cache.
    harvest._cached_ocr_log(_frame_with_crop_fill(9, outside_val=0), cache, stats)
    harvest._cached_ocr_log(_frame_with_crop_fill(9, outside_val=200), cache, stats)
    assert calls["n"] == 1, "changing pixels OUTSIDE the crop must not miss the cache"
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_ocr_cache_differing_crops_both_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Counter[str] = Counter()
    monkeypatch.setattr(harvest, "ocr_log_crop", lambda crop: [str(calls.update(n=1))])
    cache: dict[bytes, list[str]] = {}
    stats: Counter[str] = Counter()
    harvest._cached_ocr_log(_frame_with_crop_fill(1), cache, stats)
    harvest._cached_ocr_log(_frame_with_crop_fill(2), cache, stats)
    assert calls["n"] == 2, "distinct crop pixels must each OCR"
    assert stats["hits"] == 0 and stats["misses"] == 2


def test_ocr_cache_does_not_leak_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Counter[str] = Counter()
    monkeypatch.setattr(harvest, "ocr_log_crop", lambda crop: [str(calls.update(n=1))])
    f = _frame_with_crop_fill(7)
    # Two separate caches == two videos: the identical opening frame must OCR in each.
    harvest._cached_ocr_log(f, {}, Counter())
    harvest._cached_ocr_log(f, {}, Counter())
    assert calls["n"] == 2, "a fresh (per-video) cache must not reuse another video's OCR"


def test_ocr_cache_output_identical_to_uncached_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # The behaviour-preserving pin: lines from the cached path == lines from a
    # cache-disabled direct OCR, over a mix of repeated + distinct crops.
    def deterministic_ocr(crop: Any) -> list[str]:
        return [f"crop:{int(crop.reshape(-1)[0])}"]

    monkeypatch.setattr(harvest, "ocr_log_crop", deterministic_ocr)
    frames = [_frame_with_crop_fill(v) for v in (3, 3, 8, 3, 8, 1)]
    cache: dict[bytes, list[str]] = {}
    stats: Counter[str] = Counter()
    cached = [harvest._cached_ocr_log(f, cache, stats) for f in frames]
    uncached = [deterministic_ocr(harvest.crop_log(f)) for f in frames]
    assert cached == uncached
    assert stats["hits"] == 3 and stats["misses"] == 3  # 6 frames, 3 distinct crops
