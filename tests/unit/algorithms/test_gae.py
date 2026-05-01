"""GAE tests including the Phase 0 truncation-bootstrap fix.

Phase 0 split the legacy single-``dones`` GAE into separate ``terminated`` and
``truncated`` arrays. Truncation now bootstraps with ``V(s_T)`` rather than
zeroing the value (which was the pre-Phase-0 bug); the GAE accumulator still
resets at any episode boundary.

This file tests:
  - Legacy single-dones signature (unchanged-by-Phase-0 paths).
  - New terminated/truncated signature (Phase 0).
  - Equivalence between the legacy positional path and the new keyword path.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.algorithms.common.gae import compute_gae, compute_gae_vectorized


def test_gae_pure_terminal_zero_bootstrap() -> None:
    """Single-step terminated trajectory: A_0 = r_0 - V_0 (next value zeroed)."""
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.5], dtype=np.float32)
    dones = np.array([1.0], dtype=np.float32)  # terminated
    adv, ret = compute_gae(rewards, values, dones, last_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert adv[0] == pytest.approx(1.0 - 0.5)
    assert ret[0] == pytest.approx(1.0)


def test_gae_no_done_uses_last_value() -> None:
    """Single-step non-done trajectory bootstraps with last_value."""
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.0], dtype=np.float32)
    dones = np.array([0.0], dtype=np.float32)
    adv, _ = compute_gae(rewards, values, dones, last_value=2.0, gamma=0.5, gae_lambda=0.95)
    # delta = 1.0 + 0.5 * 2.0 - 0.0 = 2.0
    assert adv[0] == pytest.approx(2.0)


def test_gae_multi_step_recursion() -> None:
    """Two-step trajectory with no done: GAE recurrence works correctly."""
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0], dtype=np.float32)
    dones = np.array([0.0, 0.0], dtype=np.float32)
    adv, _ = compute_gae(rewards, values, dones, last_value=0.0, gamma=1.0, gae_lambda=1.0)
    # delta_1 = 1 + 1*0 - 0 = 1; A_1 = 1 ; delta_0 = 1 + 1*0 - 0 = 1; A_0 = 1 + 1*1*1 = 2
    assert adv[1] == pytest.approx(1.0)
    assert adv[0] == pytest.approx(2.0)


def test_gae_returns_equal_advantages_plus_values() -> None:
    """Sanity: returns = advantages + values."""
    rng = np.random.default_rng(0)
    T = 50
    rewards = rng.normal(size=T).astype(np.float32)
    values = rng.normal(size=T).astype(np.float32)
    dones = (rng.random(size=T) < 0.05).astype(np.float32)
    adv, ret = compute_gae(rewards, values, dones, last_value=0.0)
    np.testing.assert_allclose(ret, adv + values, atol=1e-6)


def test_vectorized_matches_single_env() -> None:
    """``compute_gae_vectorized`` for n_envs=1 matches ``compute_gae``."""
    rng = np.random.default_rng(1)
    T = 32
    rewards = rng.normal(size=T).astype(np.float32)
    values = rng.normal(size=T).astype(np.float32)
    dones = (rng.random(size=T) < 0.1).astype(np.float32)
    last = 0.5
    adv_a, ret_a = compute_gae(rewards, values, dones, last_value=last)
    adv_b, ret_b = compute_gae_vectorized(
        rewards, values, dones, np.array([last], dtype=np.float32), n_envs=1
    )
    np.testing.assert_allclose(adv_a, adv_b, atol=1e-6)
    np.testing.assert_allclose(ret_a, ret_b, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: terminated/truncated split contracts
# ─────────────────────────────────────────────────────────────────────────────


def test_gae_truncation_uses_bootstrap_value() -> None:
    """Truncation bootstraps with V(s_T); the GAE accumulator still resets."""
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.5], dtype=np.float32)
    terminated = np.array([0.0], dtype=np.float32)
    truncated = np.array([1.0], dtype=np.float32)
    adv, _ = compute_gae(rewards, values, terminated, truncated, 2.0, 0.99, 0.95)
    # delta = r + γ·V_bootstrap·non_terminal − V = 1 + 0.99·2·1 − 0.5 = 2.48
    expected = 1.0 + 0.99 * 2.0 - 0.5
    assert adv[0] == pytest.approx(expected)


def test_gae_terminated_zeros_bootstrap_even_if_last_value_nonzero() -> None:
    """When terminated=1, the bootstrap is forced to 0 regardless of last_value."""
    adv, _ = compute_gae(
        np.array([1.0]),
        np.array([0.5]),
        np.array([1.0]),  # terminated
        np.array([0.0]),  # truncated
        99.0,  # last_value should be ignored
        0.99,
        0.95,
    )
    assert adv[0] == pytest.approx(0.5)


def test_gae_accumulator_resets_at_truncation_boundary() -> None:
    """Across a truncation boundary, the GAE accumulator must NOT carry over.

    Build a 3-step trajectory with truncation at t=1 (mid-trajectory).
    The advantage at t=0 must equal the single-step delta — no contribution
    from the post-truncation tail at t=2.
    """
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    terminated = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    truncated = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    adv, _ = compute_gae(rewards, values, terminated, truncated, 0.0, 1.0, 1.0)
    # t=2: delta = 1 + 0 - 0 = 1; A_2 = 1
    # t=1: truncated -> bootstrap with V(s_2)=0 (not last_value, since t<T-1).
    #      delta = 1 + 1*0*1 - 0 = 1
    #      accumulator_keep = (1-0)*(1-1) = 0  → A_1 = delta + 0 = 1
    # t=0: delta = 1 + 1*0*1 - 0 = 1
    #      accumulator_keep = (1-0)*(1-0) = 1  → A_0 = 1 + 1·1·1·A_1 = 1 + 1 = 2
    assert adv[2] == pytest.approx(1.0)
    assert adv[1] == pytest.approx(1.0)
    assert adv[0] == pytest.approx(2.0)


def test_gae_legacy_signature_still_works() -> None:
    """Legacy ``compute_gae(rewards, values, dones, last_value, ...)`` still works.

    Treats ``dones`` as terminated and ``truncated=zeros`` for back-compat.
    Verifies the exact pre-Phase-0 numerical result for a single-step terminal.
    """
    adv, _ = compute_gae(np.array([1.0]), np.array([0.5]), np.array([1.0]), 0.0, 0.99, 0.95)
    assert adv[0] == pytest.approx(0.5)


def test_gae_kwarg_signature_works() -> None:
    """Keyword-only call: terminated, truncated, last_value, gamma, gae_lambda."""
    adv, _ = compute_gae(
        np.array([1.0]),
        np.array([0.5]),
        terminated=np.array([0.0]),
        truncated=np.array([1.0]),
        last_value=2.0,
        gamma=0.99,
        gae_lambda=0.95,
    )
    expected = 1.0 + 0.99 * 2.0 - 0.5
    assert adv[0] == pytest.approx(expected)


def test_gae_vectorized_truncation_and_termination_split() -> None:
    """Vectorized path mirrors single-env behavior on terminated vs truncated."""
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5], dtype=np.float32)
    terminated = np.array([1.0, 0.0], dtype=np.float32)
    truncated = np.array([0.0, 1.0], dtype=np.float32)
    last_values = np.array([3.0, 2.0], dtype=np.float32)  # only env-1 uses bootstrap
    adv, _ = compute_gae_vectorized(
        rewards, values, terminated, truncated, last_values, 2, 0.99, 0.95
    )
    # env 0: terminated → bootstrap zeroed → delta = 1 - 0.5 = 0.5
    assert adv[0] == pytest.approx(0.5)
    # env 1: truncated → bootstrap with last_values[1]=2.0 → delta = 1 + 0.99*2 - 0.5
    assert adv[1] == pytest.approx(1.0 + 0.99 * 2.0 - 0.5)
