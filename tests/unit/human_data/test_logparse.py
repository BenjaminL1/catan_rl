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


# --- chat is NOT a log event: winner never latches on chat (§5.1 firewall) ---


def test_winner_never_latches_on_chat_lines_quoting_won_the_game() -> None:
    # BLOCKER regression: Colonist chat renders in the SAME top-right crop as the
    # log. A chat line quoting "won the game" must NEVER set a winner (the whole
    # §5.1 rationale: only the exact victory LOG line may set the winner).
    for chat in (
        "ThePhantom: i won the game last time",
        "rayman147: you almost won the game",
        "ThePhantom: gg you won the game",
    ):
        parsed = parse_log([chat], _HANDLES)
        assert parsed.winner is None, chat
        # A chat line classifies as unknown with no actor, never victory.
        assert parsed.events[0].kind == "unknown"
        assert parsed.events[0].actor is None


def test_real_spike_chat_lines_stay_winnerless_and_actorless() -> None:
    # The exact chat lines captured in the spike end-screen crop (blockers/ocr):
    # 'gg', 'stop hacking please sir', 'lol still dead lost'. None may set a
    # winner or fabricate an actor.
    chat_lines = [
        "ThePhantom: gg",
        "ThePhantom: stop hacking please sir",
        "rayman147: lol still dead lost",
        "rayman147: gg",
    ]
    parsed = parse_log(chat_lines, _HANDLES)
    assert parsed.winner is None
    assert all(e.kind == "unknown" and e.actor is None for e in parsed.events)


def test_bare_won_the_game_without_leading_handle_fails_closed() -> None:
    # A non-chat line where "won the game" is not the leading predicate (leading
    # token "you", no POV handle) fails closed — winner None, actor None.
    parsed = parse_log(["you almost won the game"], _HANDLES)
    assert parsed.winner is None
    assert parsed.events[0].kind == "victory"
    assert parsed.events[0].actor is None


# --- actor is the LEADING handle, not a whole-line argmax (§14 attribution) ---


def test_actor_binds_leading_handle_on_two_handle_steal_line() -> None:
    # BLOCKER regression: Colonist renders the opponent's robber steal with full
    # names ("<stealer> stole N from <victim>"). The actor is the LEADING
    # (stealer) handle, NOT the trailing victim. A whole-line argmax mis-binds
    # the victim (both handles score 1.0; the trailing one won the >= tie).
    p1 = parse_log(["rayman147 stole 1 card from ThePhantom"], _HANDLES)
    assert p1.events[0].actor == "rayman147"
    p2 = parse_log(["ThePhantom stole 1 card from rayman147"], _HANDLES)
    assert p2.events[0].actor == "ThePhantom"
    p3 = parse_log(["ThePhantom stole 1 Wheat from rayman147"], _HANDLES)
    assert p3.events[0].actor == "ThePhantom"


def test_actor_leading_handle_generalises_to_any_pair() -> None:
    for a, b in (("Alice", "Bob"), ("foo", "bar")):
        parsed = parse_log([f"{a} stole 1 card from {b}"], (a, b))
        assert parsed.events[0].actor == a, (a, b)


def test_winner_argmaxes_over_handles_for_prefix_handle_pairs() -> None:
    # BLOCKER red-team regression: when one handle is a bigram-prefix of the
    # other ("Sam" vs "Sammy"), a first-over-threshold bind picks the SHORTER
    # handle (score 0.80 >= 0.6) before the exact-match longer one is tested,
    # attributing the win to the wrong player. The resolver must argmax over the
    # two known handles so the exact ("Sammy", score 1.0) handle wins.
    for a, b in (("Sam", "Sammy"), ("Max", "Maxi"), ("Alex", "Alexa"), ("Cat", "Catan")):
        parsed = parse_log([f"{b} won the game!"], (a, b))
        assert parsed.winner == b, (a, b)


def test_actor_resolution_is_order_independent_for_prefix_handles() -> None:
    # The same victory line must yield the same winner regardless of the
    # handle-tuple order (OCR / HUD-row order is not guaranteed). A
    # first-over-threshold bind was nondeterministic in the tuple order.
    assert parse_log(["Sammy won the game!"], ("Sam", "Sammy")).winner == "Sammy"
    assert parse_log(["Sammy won the game!"], ("Sammy", "Sam")).winner == "Sammy"


def test_prefix_handle_pair_attributes_ordinary_events_correctly() -> None:
    # The argmax fix also fixes per-player actor attribution on ordinary lines
    # (setup placements / rolls / steals feed the openings + draft order), not
    # just the winner.
    parsed = parse_log(
        [
            "Sammy placed a Settlement",
            "Sammy rolled",
            "Sammy stole 1 card from Sam",
        ],
        ("Sam", "Sammy"),
    )
    assert [e.actor for e in parsed.events] == ["Sammy", "Sammy", "Sammy"]
    # Sam's own lines still bind to Sam (the shorter handle is not shadowed).
    sam = parse_log(["Sam placed a Settlement", "Sam rolled"], ("Sam", "Sammy"))
    assert [e.actor for e in sam.events] == ["Sam", "Sam"]


# --- garbage OCR fragments must not fabricate an actor (§5.6 bias audit) ------


def test_garbage_ocr_fragments_resolve_actor_none() -> None:
    # The committed real fixture ocr_f1080_500.txt leads with pure-noise lines
    # ('aymam', 'i', 'Sluic', 'UI', 'yuu'). 'aymam' scores exactly 0.50 to
    # 'rayman147'; at threshold 0.6 it (and the others) must NOT bind a handle.
    lines = _fixture_lines("ocr_f1080_500.txt")
    assert any("aymam" in line for line in lines), "fixture lost its garbage line"
    parsed = parse_log(lines, _HANDLES)
    noise = {"aymam", "i", "sluic", "ui", "yuu", "and took"}
    for event in parsed.events:
        if event.text.lower() in noise:
            assert event.actor is None, event.text
    # The real handle lines in the same fixture still resolve correctly.
    rayman_rolls = [e for e in parsed.events if e.kind == "roll" and e.actor == "rayman147"]
    assert rayman_rolls, "real rayman147 lines should still bind"


def test_mangled_handle_still_binds_at_raised_threshold() -> None:
    # Raising the threshold to 0.6 must NOT regress the real OCR handle noise:
    # "raymani47"/"rayman|47" (≥0.75) still bind to rayman147.
    for mangled in ("rayman|47 rolled", "raymani47 built a Road"):
        parsed = parse_log([mangled], _HANDLES)
        assert parsed.events[0].actor == "rayman147", mangled


# --- POV seat logs as "You"; pov_handle maps it (§14) ------------------------


def test_pov_handle_maps_leading_you_to_pov_seat() -> None:
    # The POV seat renders its own events as "You ..." (build brief §14). With
    # pov_handle supplied, a leading "You" attributes to the POV handle.
    parsed = parse_log(["You stole"], _HANDLES, pov_handle="ThePhantom")
    assert parsed.events[0].actor == "ThePhantom"
    assert parsed.events[0].kind == "stole"


def test_pov_handle_default_none_leaves_you_unresolved() -> None:
    # Default (no pov_handle) keeps prior behaviour: a leading "You" is
    # unresolved (actor None), never guessed.
    parsed = parse_log(["You stole"], _HANDLES)
    assert parsed.events[0].actor is None
    assert parsed.events[0].kind == "stole"


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
