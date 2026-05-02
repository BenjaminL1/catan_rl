"""Tests for Phase 3.2 latest-policy regularization."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from catan_rl.selfplay.league import League


def _league(latest_weight: float, **kw) -> League:
    lg = League(
        maxlen=4,
        add_every=4,
        random_weight=0.0,
        heuristic_weight=0.0,
        latest_policy_weight=latest_weight,
        **kw,
    )
    for _ in range(2):
        lg.add({})
    return lg


def test_default_off_does_not_emit_current_self() -> None:
    """``latest_policy_weight=0`` (default) never returns current_self."""
    rng = np.random.default_rng(0)
    np.random.seed(0)
    lg = _league(0.0)
    types = [lg.sample()[0] for _ in range(200)]
    assert "current_self" not in types
    _ = rng  # silence unused


def test_current_self_sentinel_id_is_minus_two() -> None:
    """When emitted, current_self uses policy_id=-2, state_dict=None."""
    np.random.seed(0)
    lg = _league(1.0)  # always pick current_self
    opp_type, sd, pid = lg.sample()
    assert opp_type == "current_self"
    assert sd is None
    assert pid == -2


def test_latest_policy_weight_respects_probability() -> None:
    """Empirical proportion of current_self matches the configured weight."""
    np.random.seed(0)
    lg = _league(0.30)
    counts = Counter(lg.sample()[0] for _ in range(2000))
    frac = counts["current_self"] / sum(counts.values())
    # Tolerate Monte Carlo noise: 0.30 ± 0.05 is well within 2σ at N=2000.
    assert 0.25 < frac < 0.35, f"current_self fraction {frac:.3f} outside [0.25, 0.35]"


def test_special_weight_sum_must_be_le_one() -> None:
    """Construction rejects random+heuristic+latest > 1."""
    with pytest.raises(ValueError, match=r"must.+≤ 1"):
        League(random_weight=0.5, heuristic_weight=0.4, latest_policy_weight=0.2)


def test_update_result_skips_sentinel_id() -> None:
    """``policy_id=-2`` outcomes are not recorded against any league entry."""
    lg = _league(0.0)
    # No state changes when reporting against -2.
    before_stats = {k: list(v) for k, v in lg._policy_stats.items()}
    lg.update_result(-2, 1)
    lg.update_result(-2, 0)
    after_stats = {k: list(v) for k, v in lg._policy_stats.items()}
    assert before_stats == after_stats
