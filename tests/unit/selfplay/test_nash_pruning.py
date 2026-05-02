"""Tests for Phase 3.5 Nash-weighted checkpoint pruning."""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.selfplay.league import League


def _league_with(n: int) -> League:
    """League with n distinct empty state_dicts and uniform stats."""
    lg = League(maxlen=max(n, 2), add_every=4, random_weight=0.0, heuristic_weight=0.0)
    for _ in range(n):
        lg.add({})
    return lg


def test_prune_nash_drops_dominated_policy() -> None:
    """Replicator dynamics give zero mass to a strictly dominated policy.

    Setup: 3 policies. Policy 0 beats both others 80% of the time. Policy 1
    and 2 are roughly even with each other but lose to 0. Policy 0 alone
    dominates the 2-player zero-sum game; the Nash mixture concentrates on
    it, so pruning should evict 1 or 2 (whichever has lower Nash mass).
    """
    lg = _league_with(3)
    ids = list(lg._policy_ids)
    payoff = np.array(
        [
            [0.5, 0.8, 0.8],
            [0.2, 0.5, 0.5],
            [0.2, 0.5, 0.5],
        ]
    )
    evicted = lg.prune_nash(payoff, ids)
    # The strongest policy (id 0) must NOT be the one evicted.
    assert evicted != ids[0]
    # League shrinks by exactly one entry.
    assert len(lg) == 2
    assert evicted not in lg._policy_ids


def test_prune_nash_uniform_payoff_evicts_some_entry() -> None:
    """When all policies are interchangeable (50% WR all-pairs), pruning still
    succeeds — an arbitrary entry is dropped, no exceptions raised, league
    shrinks by exactly one. This guards against numerical instability with
    a degenerate payoff matrix.
    """
    lg = _league_with(4)
    ids = list(lg._policy_ids)
    payoff = np.full((4, 4), 0.5)
    evicted = lg.prune_nash(payoff, ids)
    assert evicted in ids
    assert len(lg) == 3


def test_prune_nash_rejects_size_mismatch() -> None:
    lg = _league_with(3)
    ids = list(lg._policy_ids)
    bad_payoff = np.full((2, 2), 0.5)
    with pytest.raises(ValueError, match="dim"):
        lg.prune_nash(bad_payoff, ids)


def test_prune_nash_requires_at_least_two_entries() -> None:
    lg = _league_with(2)
    ids = list(lg._policy_ids)
    payoff = np.full((1, 1), 0.5)
    with pytest.raises(ValueError, match="k=1"):
        lg.prune_nash(payoff, ids[:1])


def test_evict_by_id_clears_stats_and_window() -> None:
    """A direct ``_evict_by_id`` call removes the policy from all bookkeeping."""
    lg = _league_with(3)
    target = list(lg._policy_ids)[1]
    # Seed some stats so we can verify cleanup.
    lg.update_result(target, 1)
    assert target in lg._policy_stats
    assert target in lg._policy_window
    lg._evict_by_id(target)
    assert target not in lg._policy_ids
    assert target not in lg._policy_stats
    assert target not in lg._policy_window
    # Other entries survive.
    assert len(lg) == 2


def test_prune_nash_handles_dominant_strategy_payoff() -> None:
    """A 2-strategy game where strategy 0 dominates: Nash places ~all mass on 0."""
    lg = _league_with(2)
    ids = list(lg._policy_ids)
    # Policy 0 wins 99% vs policy 1.
    payoff = np.array([[0.5, 0.99], [0.01, 0.5]])
    evicted = lg.prune_nash(payoff, ids)
    assert evicted == ids[1]  # the dominated strategy gets evicted
