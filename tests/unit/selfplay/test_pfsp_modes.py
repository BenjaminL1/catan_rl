"""Tests for Phase 3.1 PFSP-hard / PFSP-var sampling modes."""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.selfplay.league import PFSP_MODES, League


def _make_league(mode: str = "linear", maxlen: int = 4, **kwargs) -> League:
    """League with deterministic build_policy_fn=None and no random/heuristic mass."""
    lg = League(
        maxlen=maxlen,
        add_every=4,
        random_weight=0.0,
        heuristic_weight=0.0,
        pfsp_mode=mode,
        **kwargs,
    )
    # Three identical empty state-dicts; the priorities only depend on stats.
    for _ in range(3):
        lg.add({})
    return lg


def test_invalid_pfsp_mode_raises() -> None:
    with pytest.raises(ValueError, match="pfsp_mode"):
        League(pfsp_mode="bogus")


def test_pfsp_modes_constant_listed() -> None:
    """The constant export covers exactly the documented modes."""
    assert set(PFSP_MODES) == {"linear", "hard", "var"}


def test_pfsp_linear_peaks_at_half() -> None:
    """Linear mode: WR=0.5 opponent gets the most mass."""
    lg = _make_league("linear")
    ids = list(lg._policy_ids)
    # Set per-opponent WRs: 0.0 / 0.5 / 1.0.
    for _ in range(10):
        lg.update_result(ids[0], 1)  # we always win → WR=1.0
        lg.update_result(ids[2], 0)  # we always lose → WR=0.0
    for i in range(10):
        lg.update_result(ids[1], i % 2)  # WR=0.5

    p = lg._pfsp_weights(ids)
    # Peak should be the middle (0.5) opponent.
    assert np.argmax(p) == 1


def test_pfsp_hard_peaks_at_loss() -> None:
    """Hard mode: opponent we always lose to (WR=0.0) gets the most mass."""
    lg = _make_league("hard", pfsp_p=2.0)
    ids = list(lg._policy_ids)
    for _ in range(10):
        lg.update_result(ids[0], 1)  # WR=1.0 (we dominate)
        lg.update_result(ids[2], 0)  # WR=0.0 (we lose)
    for i in range(10):
        lg.update_result(ids[1], i % 2)  # WR=0.5

    p = lg._pfsp_weights(ids)
    # Peak shifts to the WR=0.0 opponent under hard mode.
    assert np.argmax(p) == 2
    assert p[2] > p[1] > p[0]


def test_pfsp_hard_p_controls_sharpness() -> None:
    """Larger ``pfsp_p`` concentrates more mass on the lossy opponent."""
    soft = _make_league("hard", pfsp_p=1.0)
    sharp = _make_league("hard", pfsp_p=4.0)
    for lg in (soft, sharp):
        ids = list(lg._policy_ids)
        for _ in range(10):
            lg.update_result(ids[0], 1)
            lg.update_result(ids[2], 0)
        for i in range(10):
            lg.update_result(ids[1], i % 2)

    p_soft = soft._pfsp_weights(list(soft._policy_ids))
    p_sharp = sharp._pfsp_weights(list(sharp._policy_ids))
    # The lossy opponent's share grows with p.
    assert p_sharp[2] > p_soft[2]


def test_pfsp_window_drops_old_outcomes() -> None:
    """Sliding window means stale outcomes don't dominate forever."""
    lg = _make_league("hard", pfsp_window=4)
    ids = list(lg._policy_ids)
    # 10 losses, then 4 wins → window shows only recent: 4 wins → WR=1.0.
    for _ in range(10):
        lg.update_result(ids[0], 0)
    for _ in range(4):
        lg.update_result(ids[0], 1)

    # The window-based WR should be 1.0 (only recent 4 wins survive).
    assert lg._opponent_win_rate(ids[0]) == 1.0
    # But cumulative stats still report the full record.
    cum_wins, cum_games = lg._policy_stats[ids[0]]
    assert cum_wins == 4 and cum_games == 14


def test_pfsp_new_opponent_gets_neutral_prior() -> None:
    """A freshly-added opponent with no games gets WR=0.5 (neutral)."""
    lg = _make_league("hard")
    new_id = lg._policy_ids[0]
    assert lg._opponent_win_rate(new_id) == 0.5


def test_pfsp_epsilon_keeps_dominated_opponent_alive() -> None:
    """Even fully-dominated opponents (WR=1.0 in hard mode) get nonzero mass.

    Without ε, ``(1 - 1.0) ** p = 0`` and `np.random.choice` would crash on
    a zero row. ε defaults to 1e-3 — tiny but always positive.
    """
    lg = _make_league("hard")
    ids = list(lg._policy_ids)
    # Make every opponent WR=1.0.
    for _ in range(10):
        for pid in ids:
            lg.update_result(pid, 1)
    p = lg._pfsp_weights(ids)
    assert np.all(p > 0)
    assert abs(p.sum() - 1.0) < 1e-9


def test_pfsp_var_mode_smoke() -> None:
    """The variance mode produces a valid distribution at p=2."""
    lg = _make_league("var", pfsp_p=2.0)
    ids = list(lg._policy_ids)
    for _ in range(5):
        lg.update_result(ids[1], 1)
        lg.update_result(ids[1], 0)
    p = lg._pfsp_weights(ids)
    assert np.all(p > 0)
    assert abs(p.sum() - 1.0) < 1e-9


def test_pfsp_eviction_drops_window() -> None:
    """When an entry is FIFO-evicted, its window is cleaned up too."""
    lg = League(
        maxlen=2,
        add_every=4,
        random_weight=0.0,
        heuristic_weight=0.0,
        pfsp_mode="hard",
    )
    lg.add({})  # id=0
    lg.add({})  # id=1
    lg.update_result(0, 0)
    assert 0 in lg._policy_window
    lg.add({})  # id=2 evicts id=0
    assert 0 not in lg._policy_window
    assert 0 not in lg._policy_stats
