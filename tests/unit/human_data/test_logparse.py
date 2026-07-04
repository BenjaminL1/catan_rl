"""Unit tests for the human-data ``logparse`` slice (build brief §4, §5.1, §5.3).

Test-first guarantees for the log grammar (the pure, unit-tested core — easyocr
is never invoked here):

- the grammar classifies the REAL noisy committed ``ocr_*.txt`` fixtures,
  including the ``"Happy settlingl"`` OCR typo (trailing ``!`` → ``l``);
- the winner is read from the victory LOG line ONLY, is OCR-noise-tolerant
  (``"won the gamel"``), and is always **one of the two known handles or null**
  — never a fabricated name, never inferred from a resign / ``gg`` / ``+1 VP``
  precursor (build brief §5.1);
- the ``"Happy settling … /help"`` reset marker is flagged as a segment boundary
  (build brief §5.3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from catan_rl.human_data.logparse import (
    LOG_CROP_FRAC,
    LogEvent,
    ParsedLog,
    crop_log,
    parse_log,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "human_data"

# The two known handles of the sample video (from the HUD seat row / setup lines).
_HANDLES = ("ThePhantom", "rayman147")


def _fixture_lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines()


# --- reset marker (segment boundary) + the "Happy settlingl" typo ------------


def test_grammar_parses_the_committed_noisy_ocr_fixture() -> None:
    # The committed ocr_f1080_120.txt carries the real "Happy settlingl" typo
    # (trailing ! OCR'd as l) plus the two rayman147 setup lines.
    lines = _fixture_lines("ocr_f1080_120.txt")
    assert any("Happy settlingl" in line for line in lines), "fixture lost its typo"
    parsed = parse_log(lines, _HANDLES)
    kinds = [e.kind for e in parsed.events]
    # The "Happy settlingl" typo line still classifies as a reset boundary.
    assert "game_reset" in kinds
    # The two setup placements are recognised and attributed to rayman147.
    settlements = [e for e in parsed.events if e.kind == "setup_settlement"]
    roads = [e for e in parsed.events if e.kind == "setup_road"]
    assert len(settlements) == 1 and settlements[0].actor == "rayman147"
    assert len(roads) == 1 and roads[0].actor == "rayman147"
    # A setup-only frame carries no victory line → no winner.
    assert parsed.winner is None


def test_reset_marker_flags_a_segment_boundary_even_with_typo() -> None:
    parsed = parse_log(["Happy settlingl Learn how to play in the"], _HANDLES)
    assert [e.kind for e in parsed.events] == ["game_reset"]
    assert parsed.events[0].actor is None
    # The partner "/help" line alone also marks a reset (easyocr splits them).
    parsed2 = parse_log(["List of commands: /help"], _HANDLES)
    assert [e.kind for e in parsed2.events] == ["game_reset"]


def test_provenance_comment_lines_are_skipped() -> None:
    # The fixture header (# source=… / # --- OCR …) must not become events.
    lines = _fixture_lines("ocr_f1080_120.txt")
    assert lines[0].startswith("#")
    parsed = parse_log(lines, _HANDLES)
    assert all(not e.text.startswith("#") for e in parsed.events)


# --- winner: victory LOG line ONLY, one-of-two-handles-or-null (§5.1) --------


def test_winner_from_victory_line_tolerates_the_gamel_typo() -> None:
    # The real observed victory line in the spike scans: "ThePhantom won the
    # gamel" (trailing ! OCR'd as l, same typo family as "settlingl").
    parsed = parse_log(["ThePhantom won the gamel"], _HANDLES)
    assert parsed.winner == "ThePhantom"
    assert [e.kind for e in parsed.events] == ["victory"]


def test_winner_reads_the_other_handle_too() -> None:
    parsed = parse_log(["rayman147 won the game!"], _HANDLES)
    assert parsed.winner == "rayman147"


def test_winner_is_one_of_two_handles_or_null() -> None:
    # THE §5.1 invariant, exercised across every plausible input shape: the
    # winner is ALWAYS one of the two known handles or None — never a fabricated
    # name, never inferred from a non-victory line.
    cases: list[list[str]] = [
        [],  # empty log → null
        ["ThePhantom rolled", "rayman147 got"],  # no victory line → null
        ["ThePhantom won the gamel"],  # victory → ThePhantom
        ["rayman147 won the game!"],  # victory → rayman147
        ["rayman147 built a Settlement (+1 VP)", "ThePhantom: gg"],  # concession → null
        ["rayman147 has left the game"],  # resign → null (never "other won")
        ["Happy settlingl Learn how to play in the"],  # reset only → null
        ["SomeStranger won the game!"],  # unresolvable handle → null (fail closed)
    ]
    for lines in cases:
        parsed = parse_log(lines, _HANDLES)
        assert parsed.winner in (None, *_HANDLES), (lines, parsed.winner)


def test_winner_null_on_concession_precursor() -> None:
    # §5.1: a "built a Settlement (+1 VP)" precursor + a "gg" is NOT a victory
    # line — deriving a winner from it fabricates an outcome. Winner stays null.
    parsed = parse_log(
        ["rayman147 built a Settlement (+1 VP)", "ThePhantom: gg", "rayman147: gg"],
        _HANDLES,
    )
    assert parsed.winner is None
    assert "built_settlement" in [e.kind for e in parsed.events]


def test_winner_null_on_resign_never_infers_other_player() -> None:
    # A "<player> has left the game" line ends the game with NO winner (§5.1
    # forbids inferring the other player won).
    parsed = parse_log(["rayman147 has left the game"], _HANDLES)
    assert parsed.winner is None
    assert [e.kind for e in parsed.events] == ["resign"]


def test_winner_null_on_empty_and_gameplay_only_logs() -> None:
    assert parse_log([], _HANDLES).winner is None
    assert parse_log(["ThePhantom rolled", "rayman147 got"], _HANDLES).winner is None


def test_first_victory_line_wins_when_multiple_seen() -> None:
    # Defensive: if two victory lines are OCR'd (log scroll dup), the FIRST
    # resolvable one latches — a later line never overwrites it.
    parsed = parse_log(
        ["ThePhantom won the gamel", "rayman147 won the game!"],
        _HANDLES,
    )
    assert parsed.winner == "ThePhantom"


def test_unresolvable_victory_handle_yields_null_winner() -> None:
    # A victory line whose handle can't be bound to either known handle fails
    # closed (null), rather than fabricating an out-of-set winner.
    parsed = parse_log(["zzzzz won the game!"], _HANDLES)
    assert parsed.winner is None
    assert parsed.events[-1].kind == "victory"
    assert parsed.events[-1].actor is None


# --- handle resolution is OCR-noise-tolerant but set-bounded -----------------


def test_actor_resolves_mangled_handle_to_known_handle() -> None:
    # easyocr mangles the digits/pipe: "rayman|47" / "raymani47" must bind to
    # the canonical "rayman147", not to "ThePhantom".
    for mangled in ("rayman|47 rolled", "raymani47 built a Road"):
        parsed = parse_log([mangled], _HANDLES)
        assert parsed.events[0].actor == "rayman147", mangled


def test_actor_does_not_cross_bind_to_the_wrong_handle() -> None:
    parsed = parse_log(["ThePhantom rolled"], _HANDLES)
    assert parsed.events[0].actor == "ThePhantom"
    parsed2 = parse_log(["rayman147 rolled"], _HANDLES)
    assert parsed2.events[0].actor == "rayman147"


# --- grammar coverage over the real gameplay verbs ---------------------------


def test_grammar_classifies_gameplay_verbs() -> None:
    lines = [
        "ThePhantom rolled",
        "rayman147 got",
        "rayman147 built a Road",
        "ThePhantom built a City",
        "rayman147 built a Settlement",
        "ThePhantom bought [",
        "ThePhantom used Knight",
        "rayman147 moved Robber",
        "ThePhantom gave bank",
        "rayman147 received starting resources",
        "You stole",
    ]
    parsed = parse_log(lines, _HANDLES)
    kinds = [e.kind for e in parsed.events]
    assert kinds == [
        "roll",
        "got_resources",
        "built_road",
        "built_city",
        "built_settlement",
        "bought_dev",
        "used_dev",
        "moved_robber",
        "bank_trade",
        "starting_resources",
        "stole",
    ]


def test_built_settlement_with_vp_suffix_is_not_a_victory() -> None:
    # "built a Settlement (+1 VP)" must classify as built_settlement, never
    # victory — the (+1 VP) precursor is the exact §5.1 confounder.
    parsed = parse_log(["rayman147 built a Settlement (+1 VP)"], _HANDLES)
    assert parsed.events[0].kind == "built_settlement"
    assert parsed.winner is None


def test_unknown_line_is_kept_for_the_bias_audit() -> None:
    # An unclassifiable line is retained (kind="unknown", verbatim text) rather
    # than dropped — the §5.6 rejection/bias audit needs every line.
    parsed = parse_log(["is blocked by the Robber. No resources"], _HANDLES)
    assert parsed.events[0].kind == "unknown"
    assert parsed.events[0].text == "is blocked by the Robber. No resources"


def test_parse_log_requires_exactly_two_distinct_handles() -> None:
    with pytest.raises(ValueError, match="two distinct"):
        parse_log(["ThePhantom won the game!"], ("ThePhantom",))
    with pytest.raises(ValueError, match="two distinct"):
        parse_log(["ThePhantom won the game!"], ("ThePhantom", "ThePhantom"))


def test_parsedlog_and_logevent_are_frozen() -> None:
    parsed = parse_log(["ThePhantom won the gamel"], _HANDLES)
    assert isinstance(parsed, ParsedLog)
    event = parsed.events[0]
    assert isinstance(event, LogEvent)
    with pytest.raises((AttributeError, Exception)):
        event.actor = "someone"  # type: ignore[misc]


# --- crop geometry (pure) ----------------------------------------------------


def test_crop_log_extracts_top_right_panel() -> None:
    import numpy as np

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    crop = crop_log(frame)
    x0f, y0f, x1f, y1f = LOG_CROP_FRAC
    exp_h = int(y1f * 1080) - int(y0f * 1080)
    exp_w = int(x1f * 1920) - int(x0f * 1920)
    assert crop.shape == (exp_h, exp_w, 3)
    # It is the top-right sub-image (x starts at 0.645·W, y at the top).
    assert LOG_CROP_FRAC == (0.645, 0.0, 1.0, 0.3)
