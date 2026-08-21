"""Shared duplicate/replay policy (``catan_rl.labeling.dedup``).

The module exists so the scorer fit and the BC shard converter can read ONE
implementation of "what is a duplicate" while keeping DIFFERENT defaults, and so
that ``bc`` — which consumes ``to_shard`` — never has to import the scorer
package to find out.
"""

from __future__ import annotations

from typing import Any

import pytest

from catan_rl.labeling.dedup import (
    DUPLICATE_POLICIES,
    FIT_DUPLICATE_POLICIES,
    DuplicateLabelError,
    apply_duplicate_policy,
    duplicate_positions,
    exclude_replays,
    is_replay,
)


def _row(scenario_id: str, seed: int, position: int, *, at: str, replay_of: str | None = None):
    row: dict[str, Any] = {
        "scenario_id": scenario_id,
        "game_seed": seed,
        "draft_position": position,
        "labeled_at": at,
    }
    if replay_of is not None:
        row["replay_of"] = replay_of
    return row


@pytest.fixture()
def rows() -> list[dict[str, Any]]:
    return [
        _row("a", 1, 1, at="2026-01-01T00:00:00Z"),
        _row("b", 1, 2, at="2026-01-02T00:00:00Z"),
        # An UN-annotated re-label of (1, 1): the pre-``replay_of`` free replay.
        _row("c", 1, 1, at="2026-03-01T00:00:00Z"),
        # An annotated D0 replay of the same position.
        _row("d", 1, 1, at="2026-04-01T00:00:00Z", replay_of="a"),
    ]


def test_replays_are_identified_by_the_link_not_by_position(rows) -> None:
    assert [r["scenario_id"] for r in rows if is_replay(r)] == ["d"]
    assert [r["scenario_id"] for r in exclude_replays(rows)] == ["a", "b", "c"]


def test_duplicate_positions_ignores_annotated_replays(rows) -> None:
    # (1, 1) is labeled three times, but only twice WITHOUT a link — and the
    # linked one is not a duplicate, it is a measurement.
    assert duplicate_positions(rows) == {(1, 1): ["a", "c"]}
    assert duplicate_positions(exclude_replays(rows)) == {(1, 1): ["a", "c"]}


def test_keep_is_a_no_op(rows) -> None:
    kept, dropped = apply_duplicate_policy(rows, policy="keep")
    assert kept == rows
    assert dropped == []


def test_refuse_names_the_position_and_counts_them(rows) -> None:
    with pytest.raises(DuplicateLabelError, match="two NON-replay rows") as exc:
        apply_duplicate_policy(exclude_replays(rows), policy="refuse")
    assert "game_seed=1" in str(exc.value)
    assert "1 duplicated position(s)" in str(exc.value)


def test_first_labeled_keeps_the_earliest(rows) -> None:
    kept, dropped = apply_duplicate_policy(exclude_replays(rows), policy="first-labeled")
    assert [r["scenario_id"] for r in kept] == ["a", "b"]
    assert dropped == ["c"]


def test_the_fit_may_not_ask_to_keep_duplicates(rows) -> None:
    """D0: replays never train and a duplicated position is double-counted, so
    ``keep`` is not a fit policy even though it is the converter's default."""
    assert "keep" in DUPLICATE_POLICIES
    assert "keep" not in FIT_DUPLICATE_POLICIES
    with pytest.raises(DuplicateLabelError, match="duplicate_policy must be one of"):
        apply_duplicate_policy(rows, policy="keep", allowed=FIT_DUPLICATE_POLICIES)
