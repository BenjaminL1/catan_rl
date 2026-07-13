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
    monkeypatch.setattr(harvest, "_ingest", lambda vid, **kw: [object()])
    monkeypatch.setattr(harvest, "_extract_context", lambda vid, frames: _context(events))
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

    monkeypatch.setattr(harvest, "_ingest", lambda vid, **kw: [object()])
    monkeypatch.setattr(harvest, "_extract_context", lambda vid, frames: _context(solo))
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

    monkeypatch.setattr(harvest, "_ingest", lambda vid, **kw: [object()])
    monkeypatch.setattr(harvest, "_extract_context", lambda vid, frames: _context(events))
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
