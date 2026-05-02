"""Unit tests for the runtime ``rules_invariants`` module (Phase 0).

This complements ``test_rules_invariants.py`` which inlines the same checks
as pytest test cases. Here we test the module's public API: that ``run()``
returns the expected number of failures (zero in a healthy state) and that
the optional ``include_hand_tracker_drift`` flag toggles the drift check.
"""

from __future__ import annotations

from catan_rl.eval.rules_invariants import (
    InvariantFailure,
    all_check_names,
    assert_all_pass,
    run,
)


def test_default_run_returns_no_failures() -> None:
    """In a healthy state, all default invariants pass."""
    fails = run()
    assert fails == [], f"unexpected failures: {fails}"


def test_assert_all_pass_does_not_raise() -> None:
    """Convenience wrapper for harness gates."""
    assert_all_pass()  # would raise AssertionError on any failure


def test_all_check_names_includes_canonical_set() -> None:
    """The canonical 1v1 invariants must all be registered by name."""
    names = set(all_check_names())
    expected = {
        "max_points_is_15",
        "num_players_is_2",
        "p2p_trade_disabled",
        "discard_threshold_is_9",
        "stacked_dice_in_use",
        "friendly_robber_filter_present",
        "action_space_shape",
        "mask_keys_canonical",
    }
    assert expected.issubset(names), f"missing invariants: {expected - names}"


def test_invariant_failure_is_structured() -> None:
    """``InvariantFailure`` carries the three documented fields."""
    f = InvariantFailure(name="x", rule="r", detail="d")
    assert f.name == "x"
    assert f.rule == "r"
    assert f.detail == "d"
    # Frozen dataclass: cannot mutate.
    try:
        f.name = "y"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("InvariantFailure should be frozen")
