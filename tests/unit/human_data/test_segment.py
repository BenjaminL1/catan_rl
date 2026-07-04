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

from catan_rl.human_data.logparse import parse_log
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
