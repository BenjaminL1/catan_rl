"""Unit tests for the human-data ``segment`` slice (build brief §4, §5.1, §5.3, §5.5).

Test-first guarantees for game-boundary detection + the ruleset filter +
manifest-driven opponent strength:

- a corpus-wide (multi-game) :class:`LogEvent` stream is sliced into per-game
  windows on the ``"Happy settling"`` reset marker (build brief §5.3); each
  window carries its own winner (victory LOG line ONLY, §5.1) and end-cause;
- the 1v1 ruleset filter accepts a two-actor game and REJECTS a game with a third
  actor (violates the 2-player rule) — there is no P2P-trade event kind, so
  "exactly 2 distinct actors" is the enforced no-P2P/2-player invariant (§1);
- a game that ends by victory is scoreboard-terminal; a resign / cutoff game is
  kept (a possible seed) but is NOT scoreboard-terminal (§5.1);
- ``opponent_strength.tier`` for a segment is read from the committed strength
  manifest (``data/human/strength_manifest.json``) BY ``video_id`` — never a
  placeholder — mapping high/unknown/excluded exactly as the manifest says
  (excluded / missing → NO record; build brief §5.5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from catan_rl.human_data.logparse import LogEvent, parse_log
from catan_rl.human_data.segment import (
    GameSegment,
    load_strength_manifest,
    manifest_entry,
    ruleset_ok,
    segment_games,
    segment_opponent_strength,
)

REPO = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO / "data" / "human" / "strength_manifest.json"

_HANDLES = ("ThePhantom", "rayman147")


# A synthetic corpus-wide event log: THREE back-to-back games in one video.
#  game 1: reset → setup → rolls → ThePhantom wins (victory line)
#  game 2: reset → setup → rolls → rayman147 resigns (no winner)
#  game 3: reset → setup → rolls → (cutoff: stream ends mid-game, no victory)
_SYNTHETIC_CORPUS: list[str] = [
    # ---- pre-corpus noise before the first reset (must be dropped) ----
    "ThePhantom: gg wp",
    # ---- game 1 ----
    "Happy settling! Learn how to play in the",
    "List of commands: /help",
    "ThePhantom placed a Settlement",
    "ThePhantom placed a Road",
    "rayman147 placed a Settlement",
    "rayman147 placed a Road",
    "ThePhantom rolled",
    "rayman147 rolled",
    "ThePhantom won the gamel",  # victory line (trailing-! typo)
    # ---- game 2 ----
    "Happy settlingl Learn how to play in the",
    "rayman147 placed a Settlement",
    "ThePhantom rolled",
    "rayman147 has left the game",  # resign → winner null
    # ---- game 3 ----
    "List of commands: /help",
    "ThePhantom placed a Settlement",
    "rayman147 rolled",
    "ThePhantom rolled",  # stream ends here → cutoff, winner null
]


def _segments() -> list[GameSegment]:
    parsed = parse_log(_SYNTHETIC_CORPUS, _HANDLES)
    return segment_games(parsed.events, handles=_HANDLES)


# --- boundary detection ------------------------------------------------------


def test_segments_split_on_every_reset_marker() -> None:
    # Three reset markers (game 1's paired "Happy settling"+"/help" count as ONE
    # game start, game 2 one, game 3 one) → three games. The pre-reset chat noise
    # is dropped (not its own game).
    segs = _segments()
    assert len(segs) == 3


def test_pre_reset_noise_is_dropped() -> None:
    # Events before the first reset are not part of any game.
    segs = _segments()
    for seg in segs:
        assert all("gg wp" not in e.text for e in seg.events)


def test_paired_reset_lines_do_not_start_two_games() -> None:
    # easyocr splits "Happy settling" and "List of commands: /help" across two
    # lines; both are game_reset, but a reset immediately following another reset
    # (with no gameplay between) is the SAME game start, not an empty game.
    segs = _segments()
    # game 1 owns both reset lines but is a single segment with real gameplay.
    assert segs[0].winner == "ThePhantom"
    assert any(e.kind == "roll" for e in segs[0].events)


def test_game1_winner_from_victory_line() -> None:
    segs = _segments()
    assert segs[0].winner == "ThePhantom"
    assert segs[0].ended_by == "victory"


def test_game2_resign_has_no_winner() -> None:
    segs = _segments()
    assert segs[1].winner is None
    assert segs[1].ended_by == "resign"


def test_game3_cutoff_has_no_winner() -> None:
    segs = _segments()
    assert segs[2].winner is None
    assert segs[2].ended_by == "cutoff"


def test_winner_never_leaks_across_a_boundary() -> None:
    # §5.1 firewall at the segment level: game 2's window must not inherit game 1's
    # victory. Each game's winner is read only from victory lines INSIDE its window.
    segs = _segments()
    winners = [s.winner for s in segs]
    assert winners == ["ThePhantom", None, None]


def test_empty_stream_yields_no_segments() -> None:
    assert segment_games((), handles=_HANDLES) == []


def test_stream_with_no_reset_yields_no_segments() -> None:
    # A stream that never shows the reset marker (e.g. only mid-game frames
    # sampled) produces no bounded games — we never fabricate a game start.
    parsed = parse_log(["ThePhantom rolled", "rayman147 rolled"], _HANDLES)
    assert segment_games(parsed.events, handles=_HANDLES) == []


# --- ruleset filter: exactly 2 actors, victory terminal, no P2P --------------


def test_ruleset_ok_accepts_two_actor_victory_game() -> None:
    segs = _segments()
    assert ruleset_ok(segs[0]) is True


def test_ruleset_ok_rejects_three_actors() -> None:
    # A third distinct actor violates the 2-player 1v1 rule (a mis-segmented or
    # 4-player log). There is NO P2P-trade event kind in the grammar, so
    # "exactly 2 distinct actors" IS the no-P2P / 2-player invariant.
    two_actor = segment_games(
        parse_log(
            [
                "Happy settling! Learn how to play in the",
                "Alice placed a Settlement",
                "Bob rolled",
                "Alice won the game!",
            ],
            ("Alice", "Bob"),
        ).events,
        handles=("Alice", "Bob"),
    )[0]
    assert ruleset_ok(two_actor) is True
    # A THREE-actor window: parse_log binds only two known handles at a time, so
    # name a third player by concatenating two different two-handle parses (the
    # "Charlie" line binds to Charlie only under the ("Alice","Charlie") parse)
    # into one video's event stream.
    p_ab = parse_log(
        ["Happy settling! Learn how to play in the", "Alice rolled", "Bob rolled"],
        ("Alice", "Bob"),
    )
    p_ac = parse_log(["Charlie rolled", "Alice won the game!"], ("Alice", "Charlie"))
    seg = segment_games(p_ab.events + p_ac.events, handles=("Alice", "Bob"))[0]
    assert len({e.actor for e in seg.events if e.actor is not None}) == 3
    assert ruleset_ok(seg) is False


def test_ruleset_ok_accepts_resign_game_but_not_scoreboard_terminal() -> None:
    # A resign game is ruleset-legal (2 players) and kept as a possible seed, but
    # is NOT scoreboard-terminal (no 15-VP victory line, §5.1).
    segs = _segments()
    resign = segs[1]
    assert ruleset_ok(resign) is True
    assert resign.is_scoreboard_terminal() is False


def test_scoreboard_terminal_only_on_victory() -> None:
    segs = _segments()
    assert segs[0].is_scoreboard_terminal() is True  # victory
    assert segs[1].is_scoreboard_terminal() is False  # resign
    assert segs[2].is_scoreboard_terminal() is False  # cutoff


def test_ruleset_ok_rejects_single_actor_game() -> None:
    # A window with only ONE distinct actor is a broken parse (a 1v1 game must
    # show both seats acting) → ruleset-rejected.
    parsed = parse_log(
        [
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "ThePhantom rolled",
            "ThePhantom won the game!",
        ],
        _HANDLES,
    )
    seg = segment_games(parsed.events, handles=_HANDLES)[0]
    assert ruleset_ok(seg) is False


# --- opponent strength from the manifest (NOT a placeholder) -----------------


def test_manifest_loads_and_indexes_by_video_id() -> None:
    if not MANIFEST_PATH.is_file():
        pytest.skip("committed strength manifest not present")
    manifest = load_strength_manifest(MANIFEST_PATH)
    entry = manifest_entry(manifest, "9Sm86ml04aI")
    assert entry is not None
    assert entry["video_id"] == "9Sm86ml04aI"


def test_opponent_strength_read_from_manifest_by_video_id() -> None:
    # THE test the slice exists for: tier comes from the manifest entry keyed by
    # video_id, derived via derive_opponent_strength (high/unknown/excluded), NOT
    # a hardcoded placeholder. 9Sm86ml04aI is the committed tournament high.
    if not MANIFEST_PATH.is_file():
        pytest.skip("committed strength manifest not present")
    manifest = load_strength_manifest(MANIFEST_PATH)
    strength = segment_opponent_strength(manifest, "9Sm86ml04aI")
    assert strength is not None
    assert strength.tier == "high"
    assert strength.source == "tournament"


def test_opponent_strength_uses_a_synthetic_manifest_not_a_constant() -> None:
    # Determinism, no fixture dependency: a synthetic manifest drives the tier —
    # proving segment reads the manifest, not a constant. Cover all three strengths
    # + missing video, keyed on the manifest `strength` (NOT `source`).
    manifest = {
        "rank_high_max": 200,
        "videos": [
            {"video_id": "hi", "strength": "high", "source": "ranked_rank"},
            {"video_id": "tourn", "strength": "high", "source": "tournament"},
            {"video_id": "unk", "strength": "unknown", "source": "none"},
            {"video_id": "ex", "strength": "excluded", "source": "ranked_rank"},
        ],
    }
    hi = segment_opponent_strength(manifest, "hi")
    assert hi is not None and hi.tier == "high" and hi.source == "rank_badge"
    tourn = segment_opponent_strength(manifest, "tourn")
    assert tourn is not None and tourn.source == "tournament"
    unk = segment_opponent_strength(manifest, "unk")
    assert unk is not None and unk.tier == "unknown"
    # THE trap: rank>200 excluded video (source=ranked_rank) → NO record (None),
    # never a confidently-wrong high.
    assert segment_opponent_strength(manifest, "ex") is None
    # A video absent from the manifest → None (never a fabricated tier).
    assert segment_opponent_strength(manifest, "not-there") is None


def test_manifest_entry_returns_none_for_missing_video() -> None:
    manifest = {"videos": [{"video_id": "a", "strength": "high", "source": "tournament"}]}
    assert manifest_entry(manifest, "a") is not None
    assert manifest_entry(manifest, "zzz") is None


def test_gamesegment_is_frozen() -> None:
    seg = _segments()[0]
    assert isinstance(seg, GameSegment)
    with pytest.raises((AttributeError, Exception)):
        seg.winner = "someone"  # type: ignore[misc]


# --- weld / multi-victory guard (BLOCKER: §5.1 confidently-wrong) ------------


def test_welded_window_with_two_victories_is_not_scoreboard_terminal() -> None:
    # Two back-to-back games between the SAME two players weld when game N+1's
    # reset marker is OCR-corrupted below the reset regex (e.g. "Happy setting").
    # The window then structurally contains TWO victory events. Latching the first
    # winner and reporting scoreboard-terminal is exactly the §5.1 confidently-wrong
    # failure: it pairs game N's board+openings with a winner from a welded stream.
    # A window with >=2 victory events is definitionally a weld → NOT terminal.
    parsed = parse_log(
        [
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "ThePhantom won the game!",  # game N victory
            # game N+1's reset OCR'd as "Happy setting" (dropped 'l') → NOT a reset
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "rayman147 won the game!",  # game N+1 victory — welded into the SAME window
        ],
        _HANDLES,
    )
    segs = segment_games(parsed.events, handles=_HANDLES)
    assert len(segs) == 1
    welded = segs[0]
    n_victory = sum(1 for e in welded.events if e.kind == "victory")
    assert n_victory == 2
    # The guard: a welded (multi-victory) window is flagged and NOT scoreboard-terminal.
    assert welded.ended_by == "weld"
    assert welded.winner is None
    assert welded.is_scoreboard_terminal() is False


def test_same_winner_reocrd_across_frames_resolves_to_one_victory() -> None:
    # RED-TEAM COUNTEREXAMPLE (BLOCKER): the MOST COMMON real terminal shape is a
    # SINGLE legitimate game whose final victory line is re-OCR'd by ≥2 consecutive
    # sampled frames — the win modal lingers on the scrolling panel for several
    # seconds, so Pass-A (1 frame / 3-5s) captures the SAME "won the game" line
    # twice. That is ONE game's win, NOT a weld. Counting DISTINCT winners (not raw
    # victory events) must resolve it to a single victory with the real winner —
    # the previous ">=2 victory events => weld" guard mis-set winner=None and
    # dropped the game from the scoreboard.
    parsed = parse_log(
        [
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "ThePhantom won the game!",  # game victory
            "ThePhantom won the game!",  # SAME line re-OCR'd by the next sampled frame
        ],
        _HANDLES,
    )
    segs = segment_games(parsed.events, handles=_HANDLES)
    assert len(segs) == 1
    game = segs[0]
    n_victory = sum(1 for e in game.events if e.kind == "victory")
    assert n_victory == 2  # two raw victory EVENTS, one distinct winner
    assert game.winner == "ThePhantom"
    assert game.ended_by == "victory"
    assert game.is_scoreboard_terminal() is True


def test_same_winner_weld_without_reset_is_weld_not_scoreboard_terminal() -> None:
    # RED-TEAM (BLOCKER, finding: same-winner weld blind spot). Two back-to-back
    # games BOTH won by ThePhantom weld into one window when game N's reset marker
    # OCR-corrupts below the reset regex (no intervening reset at all). The window
    # then holds TWO full drafts + TWO "ThePhantom won the game!" lines → ONE
    # distinct winner, so the distinct-winner guard passes it as a clean victory.
    # ThePhantom wins the large majority of his games, so consecutive SAME-winner
    # games are the COMMON weld shape, not the rare one. The winner-independent
    # structural guard (a fresh setup burst AFTER an own-game victory) must flag it:
    # ended_by='weld' / winner=None / NOT scoreboard-terminal.
    parsed = parse_log(
        [
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "rayman147 rolled",
            "ThePhantom won the game!",  # game N victory
            # game N's reset marker corrupted away entirely (no reset line) → the
            # next game welds straight in with a fresh draft.
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "rayman147 rolled",
            "ThePhantom won the game!",  # game N+1, SAME winner
        ],
        _HANDLES,
    )
    segs = segment_games(parsed.events, handles=_HANDLES)
    assert len(segs) == 1
    welded = segs[0]
    # ONE distinct winner (the distinct-winner guard alone would NOT catch it)...
    assert len(set(e.actor for e in welded.events if e.kind == "victory")) == 1
    # ...but the setup-after-victory guard flags the weld.
    assert welded.ended_by == "weld"
    assert welded.winner is None
    assert welded.is_scoreboard_terminal() is False


def test_missing_terminal_then_identical_reset_merges_not_split() -> None:
    # DESIGN DECISION (user: "throw it out over guessing a cut"). Game N's terminal is
    # NOT sampled AND game N+1 opens with byte-identical reset text. This is
    # indistinguishable from a lingering mid-game re-OCR of a SINGLE game's opener, so
    # we DELIBERATELY do NOT split on it (the old `>=2 rollers` separator false-split a
    # single game on any spurious mid-game reset — the worse, confidently-wrong error).
    # The two games merge into ONE window; game N is folded in and lost (the accepted
    # cost of the safer default). Catching this merged weld's mixed setup is downstream:
    # the record firewall requires exactly 2 setup settlements per player read from the
    # post-setup FRAME, so a mixed-setup window is dropped rather than mispaired.
    parsed = parse_log(
        [
            # game N: full round of play, but NO victory/resign captured (cutoff)
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "rayman147 rolled",
            # game N+1: byte-identical reset text
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "rayman147 rolled",
            "rayman147 won the game!",
        ],
        _HANDLES,
    )
    segs = segment_games(parsed.events, handles=_HANDLES)
    # merged — we do not guess a boundary we cannot see
    assert len(segs) == 1


def test_noisy_ocr_two_games_split_despite_lingering_opener_reocr() -> None:
    # SLICE green_seg (finding 5: noisy-OCR weld). Game N plays to a cutoff (its
    # terminal was never sampled), then its opener reset LINGERS on the scrolling
    # panel and re-OCRs (byte-identical) AFTER the play — and only THEN does game
    # N+1's own reset arrive with a NOISY (byte-DIFFERENT) OCR of the reset text.
    # The lingering duplicate must NOT erase the "real play happened" memory, else
    # game N+1's genuinely-new noisy reset is welded onto game N (both games lost to
    # one setup-only fragment). Boundary detection must SPLIT into two games. This is
    # distinct from the deliberate identical-reset merge below: here game N+1's reset
    # text DIFFERS (a new line), so it is not the ambiguous lingering-opener case.
    parsed = parse_log(
        [
            # game N: full round of play, terminal NOT captured (cutoff)
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "rayman147 rolled",
            # game N's opener reset lingers + re-OCRs (byte-identical) AFTER the play
            "Happy settling! Learn how to play in the",
            # game N+1 opens with a NOISY (byte-different) OCR of the reset marker —
            # still matches the reset regex ("happy settling"), but differs in the
            # trailing bytes, so it is a genuinely-new reset line, not the opener dup
            "Happy settling! Learn how to play in thc",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "rayman147 rolled",
            "rayman147 won the game!",
        ],
        _HANDLES,
    )
    segs = segment_games(parsed.events, handles=_HANDLES)
    assert len(segs) == 2  # split, not welded into one setup-only fragment
    assert segs[0].winner is None  # game N: cutoff (terminal not sampled)
    assert segs[0].ended_by == "cutoff"
    assert segs[1].winner == "rayman147"  # game N+1: its own victory
    assert segs[1].ended_by == "victory"
    assert segs[1].is_scoreboard_terminal() is True


def test_spurious_midgame_reset_does_not_false_split_one_game() -> None:
    # The primary BLOCKER the merge-favouring design fixes: a single game whose opener
    # reset re-OCRs mid-game (after both seats have rolled) must stay ONE game — never
    # split into a winnerless seed + a hollow terminal.
    parsed = parse_log(
        [
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "rayman147 rolled",
            "Happy settling! Learn how to play in the",  # spurious mid-game re-OCR
            "ThePhantom rolled",
            "ThePhantom won the game!",
        ],
        _HANDLES,
    )
    segs = segment_games(parsed.events, handles=_HANDLES)
    assert len(segs) == 1
    assert segs[0].winner == "ThePhantom"
    assert segs[0].ended_by == "victory"


def test_same_winner_thrice_and_with_reocrd_tail_gameplay_is_one_victory() -> None:
    # The other two re-OCR variants the finding names must ALSO resolve to one
    # victory: (a) 3-frame persistence (victory x3), and (b) re-OCR'd tail gameplay
    # BETWEEN the two victory copies (so a mere consecutive-dedup would not catch
    # it — DISTINCT-winner counting is required).
    thrice = parse_log(
        [
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "ThePhantom won the game!",
            "ThePhantom won the game!",
            "ThePhantom won the game!",
        ],
        _HANDLES,
    )
    seg = segment_games(thrice.events, handles=_HANDLES)[0]
    assert seg.winner == "ThePhantom"
    assert seg.ended_by == "victory"
    assert seg.is_scoreboard_terminal() is True

    with_tail = parse_log(
        [
            "Happy settling! Learn how to play in the",
            "ThePhantom placed a Settlement",
            "rayman147 placed a Settlement",
            "ThePhantom rolled",
            "ThePhantom won the game!",
            "ThePhantom rolled",  # re-OCR'd tail gameplay between the two victory copies
            "ThePhantom won the game!",
        ],
        _HANDLES,
    )
    seg2 = segment_games(with_tail.events, handles=_HANDLES)[0]
    assert seg2.winner == "ThePhantom"
    assert seg2.ended_by == "victory"
    assert seg2.is_scoreboard_terminal() is True


# --- cross-frame dedup: scrolling-panel re-OCR must not split/fabricate ------


def test_duplicate_reset_after_intervening_gameplay_does_not_split_a_game() -> None:
    # §5.3 + §5.10: the Colonist log is a SCROLLING panel and Pass A samples at
    # 1 frame / 3-5s, so a later frame re-OCRs the recent tail — including the
    # lingering "Happy settling" reset — AFTER intervening gameplay from other
    # frames. The stale run-collapse only merges STREAM-ADJACENT resets, so this
    # duplicate reset opens a spurious second game. Dedup must collapse it: ONE
    # real game → ONE segment with the correct winner.
    stream = [
        "Happy settling! Learn how to play in the",
        "ThePhantom placed a Settlement",
        "rayman147 placed a Settlement",
        "ThePhantom rolled",
        # a later overlapping frame re-shows the lingering reset then re-OCRs
        # already-seen gameplay (the scrolling tail), NOT new content
        "Happy settling! Learn how to play in the",
        "ThePhantom placed a Settlement",
        "rayman147 placed a Settlement",
        "ThePhantom rolled",
        "rayman147 rolled",
        "ThePhantom won the game!",
    ]
    parsed = parse_log(stream, _HANDLES)
    segs = segment_games(parsed.events, handles=_HANDLES)
    assert len(segs) == 1
    assert segs[0].winner == "ThePhantom"
    assert segs[0].ended_by == "victory"
    assert segs[0].is_scoreboard_terminal() is True


def test_lingering_reset_run_collapses_to_one_start() -> None:
    # A reset line that lingers on-screen across several sparse frames appears in
    # the flattened stream interleaved with new gameplay below it:
    #   reset, placed, reset(dup), placed, reset(dup), ... → still ONE game.
    stream = [
        "Happy settling! Learn how to play in the",
        "ThePhantom placed a Settlement",
        "Happy settling! Learn how to play in the",
        "rayman147 placed a Settlement",
        "Happy settling! Learn how to play in the",
        "ThePhantom rolled",
        "rayman147 rolled",
        "ThePhantom won the game!",
    ]
    segs = segment_games(parse_log(stream, _HANDLES).events, handles=_HANDLES)
    assert len(segs) == 1
    assert segs[0].winner == "ThePhantom"


def test_stale_victory_carried_past_next_reset_does_not_win_next_game() -> None:
    # §5.1: game N's "won the game" line lingers on-screen into the frame that
    # first shows game N+1's reset, so in the flattened stream the stale victory
    # can land AFTER the next reset. Windows sliced on reset index would then
    # attribute game N's win to game N+1's window (phantom win) and demote game N
    # to cutoff (real win lost). The dedup collapses the stale copy; the
    # within-window gate ignores a victory that is the FIRST substantive event of a
    # window (a stale carry-over, not this game's own victory).
    stream = [
        # game N
        "Happy settling! Learn how to play in the",
        "ThePhantom placed a Settlement",
        "rayman147 placed a Settlement",
        "ThePhantom rolled",
        "ThePhantom won the game!",
        # game N+1 opens; the stale game-N victory re-OCRs right after its reset
        "Happy settling! Learn how to play in the",
        "ThePhantom won the game!",  # stale carry-over of game N's win
        "ThePhantom placed a Settlement",
        "rayman147 placed a Settlement",
        "rayman147 rolled",
        "rayman147 won the game!",  # game N+1's real winner
    ]
    segs = segment_games(parse_log(stream, _HANDLES).events, handles=_HANDLES)
    assert len(segs) == 2
    # game N keeps its real win
    assert segs[0].winner == "ThePhantom"
    assert segs[0].ended_by == "victory"
    # game N+1 wins for rayman147, NOT the stale ThePhantom carry-over
    assert segs[1].winner == "rayman147"
    assert segs[1].ended_by == "victory"


def test_stale_victory_as_first_window_event_is_ignored() -> None:
    # Directly exercise the within-window stale-victory gate: a victory event with
    # NO preceding gameplay in its window is a carry-over from the previous game
    # (the panel had not scrolled the win off yet) → it must not latch a winner.
    events = (
        LogEvent(kind="game_reset", actor=None, text="Happy settling"),
        LogEvent(kind="victory", actor="ThePhantom", text="ThePhantom won the game"),
        LogEvent(kind="setup_settlement", actor="rayman147", text="rayman147 placed a Settlement"),
        LogEvent(kind="roll", actor="ThePhantom", text="ThePhantom rolled"),
    )
    seg = segment_games(events, handles=_HANDLES)[0]
    assert seg.winner is None
    assert seg.ended_by == "cutoff"
