"""Duplicate / replay handling for the setup-label store.

The store can hold more than one row for a single decision point:

* a **D0 replay row**, which carries ``replay_of`` naming the original it
  re-labels (spec ``setup-scorer-and-blind-reveal``, D0); and
* an **un-annotated duplicate** — the same ``(game_seed, draft_position)``
  labeled twice with no link, which is what the owner's pre-``replay_of`` free
  replay left behind.

Two consumers read the corpus and they want DIFFERENT things from it, so the
policy lives here rather than in either of them:

* the scorer fit (``catan_rl.setup_phase.fit``) applies STRICT exclusion per D0
  — replays never train, and an un-annotated duplicate is refused unless the
  caller names a policy; while
* the BC shard converter (``catan_rl.labeling.to_shard``) keeps its historical
  behaviour and only WARNS, because changing what a shard contains is a
  fine-tune-slice decision, not something a scorer slice gets to make.

This module is deliberately in ``labeling`` and not in ``setup_phase``: ``bc``
consumes ``to_shard``, and ``bc`` must not acquire a dependency on the scorer
package to find out what a duplicate row is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

DUPLICATE_POLICY_KEEP: str = "keep"
"""Keep every row, duplicates and replays included. The converter's historical
default; callers are expected to have been warned."""

DUPLICATE_POLICY_REFUSE: str = "refuse"
"""Raise on any un-annotated duplicate decision point. The fit's default."""

DUPLICATE_POLICY_FIRST_LABELED: str = "first-labeled"
"""Keep the earliest ``labeled_at`` row for each duplicated decision point and
drop the rest. The only sanctioned way to fit over the pre-``replay_of`` free
replay, and the caller is expected to record that it did so."""

DUPLICATE_POLICIES: tuple[str, ...] = (
    DUPLICATE_POLICY_KEEP,
    DUPLICATE_POLICY_REFUSE,
    DUPLICATE_POLICY_FIRST_LABELED,
)

FIT_DUPLICATE_POLICIES: tuple[str, ...] = (
    DUPLICATE_POLICY_REFUSE,
    DUPLICATE_POLICY_FIRST_LABELED,
)
"""The subset the scorer fit accepts. ``keep`` is absent on purpose: D0 says
replay rows are excluded from fitting and a duplicated position would be
double-counted, so "keep everything" is not a fit policy."""


class DuplicateLabelError(ValueError):
    """Raised when a corpus holds a duplicate decision point under ``refuse``."""


class DuplicateLabelWarning(UserWarning):
    """Warned when duplicates are being carried through rather than resolved."""


Row = TypeVar("Row", bound=Mapping[str, Any])


def is_replay(row: Mapping[str, Any]) -> bool:
    """Whether ``row`` is a D0 replay of an earlier label."""
    return row.get("replay_of") is not None


def exclude_replays(rows: Iterable[Row]) -> list[Row]:
    """Drop D0 replay rows.

    A replay re-labels a position that is already in the corpus, so keeping both
    would emit the same ``(game_seed, draft_position)`` twice with
    CONTRADICTORY targets (a replay is only informative when the owner picks
    differently) and silently up-weight exactly the positions that happened to
    be replayed.
    """
    return [r for r in rows if not is_replay(r)]


def duplicate_positions(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[int, int], list[str]]:
    """``(game_seed, draft_position) -> scenario_ids`` for repeated positions.

    Only positions with more than one row are returned, and ``replay_of`` rows
    are not counted: those are annotated re-labels, which every consumer can
    already recognise. What this finds is the UN-annotated kind.
    """
    seen: dict[tuple[int, int], list[str]] = {}
    for row in exclude_replays(rows):
        key = (int(row["game_seed"]), int(row["draft_position"]))
        seen.setdefault(key, []).append(str(row["scenario_id"]))
    return {key: ids for key, ids in seen.items() if len(ids) > 1}


def _first_labeled_drops(rows: Sequence[Row]) -> list[str]:
    """Scenario ids to drop so one row survives per duplicated position."""
    kept: dict[tuple[int, int], Row] = {}
    dropped: list[str] = []
    for row in rows:
        key = (int(row["game_seed"]), int(row["draft_position"]))
        prior = kept.get(key)
        if prior is None:
            kept[key] = row
            continue
        earlier, later = sorted(
            (prior, row), key=lambda r: (str(r["labeled_at"]), str(r["scenario_id"]))
        )
        kept[key] = earlier
        dropped.append(str(later["scenario_id"]))
    return dropped


def apply_duplicate_policy(
    rows: Iterable[Row], *, policy: str, allowed: Sequence[str] = DUPLICATE_POLICIES
) -> tuple[list[Row], list[str]]:
    """Resolve duplicated decision points, returning ``(kept, dropped_ids)``.

    ``rows`` should already have had replays removed by the caller when the
    caller wants them removed — this function is only about UN-annotated
    duplicates, so the two decisions stay separable.

    ``allowed`` narrows the accepted policy names for a particular consumer
    (see :data:`FIT_DUPLICATE_POLICIES`).
    """
    if policy not in allowed:
        raise DuplicateLabelError(
            f"duplicate_policy must be one of {tuple(allowed)}, got {policy!r}"
        )
    ordered = list(rows)
    if policy == DUPLICATE_POLICY_KEEP:
        return ordered, []
    duplicates = duplicate_positions(ordered)
    if policy == DUPLICATE_POLICY_REFUSE:
        if duplicates:
            key, ids = next(iter(sorted(duplicates.items())))
            raise DuplicateLabelError(
                f"corpus holds two NON-replay rows for game_seed={key[0]} "
                f"draft_position={key[1]} ({', '.join(repr(i) for i in ids)}), and "
                f"{len(duplicates)} duplicated position(s) in total. A re-labeled "
                f"position must carry ``replay_of`` so the fit can exclude it and the "
                f"consistency report can pair it; fitting on both would double-count "
                f"the position. Pass duplicate_policy='first-labeled' to keep the "
                f"earliest of each pair."
            )
        return ordered, []
    dropped = set(_first_labeled_drops(ordered))
    return [r for r in ordered if str(r["scenario_id"]) not in dropped], sorted(dropped)


__all__ = [
    "DUPLICATE_POLICIES",
    "DUPLICATE_POLICY_FIRST_LABELED",
    "DUPLICATE_POLICY_KEEP",
    "DUPLICATE_POLICY_REFUSE",
    "FIT_DUPLICATE_POLICIES",
    "DuplicateLabelError",
    "DuplicateLabelWarning",
    "apply_duplicate_policy",
    "duplicate_positions",
    "exclude_replays",
    "is_replay",
]
