"""GAE tests including the truncation-bootstrap fix planned for Phase 0.

Currently the implementation collapses ``terminated`` and ``truncated`` into a
single ``dones`` array, so truncations zero the bootstrap. Phase 0 of the
roadmap fixes this.

This test file ships in two parts:
  - tests for the *current* GAE behavior (unchanged-by-this-PR sanity checks)
  - tests for the *desired post-Phase-0* behavior, marked `xfail` until the
    fix lands so they document the contract without blocking CI.
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
# Post-Phase-0 contracts (xfail until the fix lands)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(reason="Phase 0 GAE truncation fix not yet implemented", strict=False)
def test_gae_truncation_uses_bootstrap_value() -> None:
    """Truncation should bootstrap with V(s_T), not zero the value."""
    # Once Phase 0 lands, ``compute_gae`` will accept ``terminated`` and
    # ``truncated`` separately. This test documents the desired contract.
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.5], dtype=np.float32)
    terminated = np.array([0.0], dtype=np.float32)
    truncated = np.array([1.0], dtype=np.float32)
    # Expected post-fix: A_0 = r_0 + γ * V_bootstrap - V_0  (no zeroing for truncation)
    # Current buggy behavior: treats truncation as terminated → A_0 = r_0 - V_0.
    # We assert the *desired* expected here; xfail until implementation.
    from catan_rl.algorithms.common.gae import compute_gae as _gae

    try:
        adv, _ = _gae(
            rewards,
            values,
            terminated,
            truncated,  # type: ignore[arg-type]
            last_value=2.0,
            gamma=0.99,
            gae_lambda=0.95,
        )
    except TypeError:
        pytest.fail("Phase 0 fix has not yet introduced terminated/truncated split")
    expected = 1.0 + 0.99 * 2.0 - 0.5
    assert adv[0] == pytest.approx(expected)
