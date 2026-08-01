"""Conformance recorder — spec ``preroll-dev-cards-r1`` D10.

Before this slice the recorder opened EVERY turn with an unconditional
``RollDice``, so re-recording under R1 produced byte-identical fixtures: the
differential-test harness was structurally blind to the whole pre-roll change
class and the two engines could fork while green.

The window is per-call (``record_game(..., ruleset=...)``) and defaults to
``R0``, so the recorder can still produce fixtures the reference TS engine —
whose D10 half is not in this slice — is able to replay.
"""

from __future__ import annotations

from catan_rl.conformance.recorder import CONFORMANCE_SCHEMA_VERSION, record_game


def test_schema_version_is_bumped_to_2() -> None:
    # Within-turn step ORDERING changed (a dev-card step may now precede
    # RollDice), which a v1 consumer would mis-align.
    assert CONFORMANCE_SCHEMA_VERSION == 2


def test_recorded_log_stamps_the_new_schema_version() -> None:
    log = record_game(7, max_main_turns=40)
    assert log["schema_version"] == 2


def test_r0_never_emits_a_dev_card_step_before_its_roll() -> None:
    """The default epoch must still be recordable by the reference TS engine,
    which throws ``'Must roll the dice before building'`` while ``rollPending``."""
    pre_roll_kinds = {"PlayKnight", "PlayYearOfPlenty", "PlayMonopoly"}
    for seed in range(12):
        log = record_game(seed, max_main_turns=120)
        assert log["ruleset"] == "R0"
        prev_kind: str | None = None
        for step in log["steps"]:
            kind = step["action"]["kind"]
            assert not (kind == "RollDice" and prev_kind in pre_roll_kinds), (
                f"seed {seed}: R0 fixture opened a turn with {prev_kind}"
            )
            prev_kind = kind


def test_some_seed_emits_a_dev_card_step_before_its_roll() -> None:
    """At least one R1 fixture-scale game must contain a real pre-roll step,
    otherwise the harness is still blind in practice."""
    pre_roll_kinds = {"PlayKnight", "PlayYearOfPlenty", "PlayMonopoly"}
    found = False
    for seed in range(12):
        log = record_game(seed, max_main_turns=120, ruleset="R1")
        steps = log["steps"]
        prev_kind: str | None = None
        for step in steps:
            kind = step["action"]["kind"]
            if kind == "RollDice" and prev_kind in pre_roll_kinds:
                found = True
                break
            prev_kind = kind
        if found:
            break
    assert found, "no seed produced a pre-roll dev-card step before its RollDice"


def test_a_turn_never_carries_two_dev_plays_around_its_roll() -> None:
    """The recorder must obey the same one-card-per-turn rule as the env: a
    pre-roll play must not be followed by a second play in the same turn."""
    dev_kinds = {"PlayKnight", "PlayYearOfPlenty", "PlayMonopoly", "PlayRoadBuilder"}
    for seed in range(6):
        log = record_game(seed, max_main_turns=120, ruleset="R1")
        plays_this_turn = 0
        for step in log["steps"]:
            kind = step["action"]["kind"]
            if kind == "EndTurn":
                plays_this_turn = 0
                continue
            if kind in dev_kinds:
                plays_this_turn += 1
                assert plays_this_turn <= 1, f"seed {seed}: two dev cards in one turn"
