"""Tests for `ppo/gae.py` — vectorised GAE.

Pins:

1. **Terminated vs truncated**: terminated zeros the bootstrap value at
   the boundary; truncated keeps it. Both reset the GAE accumulator.
2. **Reference equivalence** with a hand-rolled scalar GAE reference
   on a small fixture (proves the vectorised form is correct).
3. **Boundary linearity**: the advantage at step ``t+1`` does NOT bleed
   into the advantage at step ``t`` across any episode boundary.
4. **Returns identity**: ``returns == advantages + values`` always.
5. **Normalize zero mean + unit std after rollout-norm**.
6. **Degenerate constant-advantage buffer**: normalize subtracts mean
   without dividing by zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.ppo.gae import compute_gae, normalize_advantages


def _scalar_reference_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow, scalar, one-env-at-a-time reference for cross-checking."""
    T, N = rewards.shape
    adv = np.zeros_like(rewards, dtype=np.float32)
    for n in range(N):
        gae = 0.0
        for t in reversed(range(T)):
            in_buffer_next = float(last_values[n]) if t == T - 1 else float(values[t + 1, n])
            next_v = 0.0 if terminated[t, n] else in_buffer_next
            non_terminal = 0.0 if terminated[t, n] else 1.0
            if t == T - 1:
                # No in-buffer successor → inheritance always 0.
                non_done_for_inherit = 0.0
            else:
                non_done_for_inherit = 0.0 if (terminated[t, n] or truncated[t, n]) else 1.0
            delta = float(rewards[t, n]) + gamma * next_v * non_terminal - float(values[t, n])
            gae = delta + gamma * gae_lambda * non_done_for_inherit * gae
            adv[t, n] = gae
    ret = adv + values.astype(np.float32)
    return adv, ret


# ---------------------------------------------------------------------------
# Reference equivalence
# ---------------------------------------------------------------------------


class TestReferenceEquivalence:
    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_matches_scalar_reference(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        T, N = 8, 4
        rewards = rng.standard_normal((T, N)).astype(np.float32)
        values = rng.standard_normal((T, N)).astype(np.float32)
        terminated = rng.random((T, N)) < 0.1
        truncated = rng.random((T, N)) < 0.1
        # Don't let term and trunc be both True on the same step
        truncated = np.where(terminated, False, truncated)
        last_values = rng.standard_normal(N).astype(np.float32)

        adv_vec, ret_vec = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terminated,
            truncated=truncated,
            last_values=last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        adv_ref, ret_ref = _scalar_reference_gae(
            rewards,
            values,
            terminated,
            truncated,
            last_values,
            0.99,
            0.95,
        )
        np.testing.assert_allclose(adv_vec, adv_ref, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(ret_vec, ret_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Terminated vs truncated semantics
# ---------------------------------------------------------------------------


class TestTerminatedVsTruncated:
    def test_terminated_zeros_bootstrap_inside_buffer(self) -> None:
        # Two-step buffer, env 0: step 0 terminates with V(s_1)=100, so the
        # bootstrap at the boundary should be ZERO.
        rewards = np.array([[1.0], [10.0]], dtype=np.float32)
        values = np.array([[0.5], [100.0]], dtype=np.float32)
        terminated = np.array([[True], [False]])
        truncated = np.zeros((2, 1), dtype=bool)
        last_values = np.array([0.0], dtype=np.float32)

        adv, _ = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terminated,
            truncated=truncated,
            last_values=last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        # adv[0, 0] = r[0] - V[0] = 1.0 - 0.5 = 0.5
        # (V(s_1) zeroed by terminated; no inheritance from step 1 either)
        assert adv[0, 0] == pytest.approx(0.5, abs=1e-5)

    def test_truncated_keeps_bootstrap_inside_buffer(self) -> None:
        rewards = np.array([[1.0], [10.0]], dtype=np.float32)
        values = np.array([[0.5], [100.0]], dtype=np.float32)
        terminated = np.zeros((2, 1), dtype=bool)
        truncated = np.array([[True], [False]])
        last_values = np.array([0.0], dtype=np.float32)

        adv, _ = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terminated,
            truncated=truncated,
            last_values=last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        # Truncated → bootstrap V(s_1)=100 IS kept:
        # adv[0,0] = r[0] + γ*V[1] - V[0] = 1.0 + 0.99*100 - 0.5 = 99.5
        assert adv[0, 0] == pytest.approx(99.5, abs=1e-5)

    def test_terminated_at_last_step_zeros_bootstrap(self) -> None:
        # Reviewer fix: the new API derives "last step was terminated"
        # directly from terminated[T-1, n]. Passing last_values=100 with
        # terminated[T-1]=True must zero the bootstrap regardless of
        # whatever the post-rollout obs's value is.
        rewards = np.array([[1.0]], dtype=np.float32)
        values = np.array([[0.5]], dtype=np.float32)
        terminated = np.array([[True]])  # last step was a real terminal
        truncated = np.zeros((1, 1), dtype=bool)
        last_values = np.array([100.0], dtype=np.float32)

        adv, _ = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terminated,
            truncated=truncated,
            last_values=last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        # terminated[T-1] → V(s_1) treated as 0; adv = r - V = 0.5
        # The bug the reviewer caught: previously a caller could set
        # last_terminated=False (e.g. "is the auto-reset obs terminal?")
        # even with terminated[T-1]=True, and the buffer would
        # double-bootstrap. The new API can't express that bug.
        assert adv[0, 0] == pytest.approx(0.5, abs=1e-5)

    def test_truncated_at_last_step_keeps_bootstrap(self) -> None:
        rewards = np.array([[1.0]], dtype=np.float32)
        values = np.array([[0.5]], dtype=np.float32)
        terminated = np.zeros((1, 1), dtype=bool)
        truncated = np.zeros((1, 1), dtype=bool)
        last_values = np.array([100.0], dtype=np.float32)

        adv, _ = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terminated,
            truncated=truncated,
            last_values=last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        # adv = r + γ*100 - V = 1.0 + 0.99*100 - 0.5 = 99.5
        assert adv[0, 0] == pytest.approx(99.5, abs=1e-5)


# ---------------------------------------------------------------------------
# Boundary linearity
# ---------------------------------------------------------------------------


class TestBoundaryLinearity:
    def test_adv_at_step_does_not_inherit_across_terminated(self) -> None:
        # Construct two scenarios identical except for adv[1] (the future).
        # adv[0] should be the same in both because the episode terminates
        # at step 0.
        rewards = np.array([[1.0], [50.0]], dtype=np.float32)
        values = np.array([[0.5], [10.0]], dtype=np.float32)
        terminated_A = np.array([[True], [False]])
        truncated_A = np.zeros((2, 1), dtype=bool)
        last_values = np.array([0.0], dtype=np.float32)

        adv_A, _ = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terminated_A,
            truncated=truncated_A,
            last_values=last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Now perturb only the future (rewards[1] and values[1]).
        rewards2 = np.array([[1.0], [9999.0]], dtype=np.float32)
        values2 = np.array([[0.5], [-9999.0]], dtype=np.float32)
        adv_B, _ = compute_gae(
            rewards=rewards2,
            values=values2,
            terminated=terminated_A,
            truncated=truncated_A,
            last_values=last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert adv_A[0, 0] == pytest.approx(adv_B[0, 0])
        # Step-1 advantages differ — sanity that the perturbation took effect.
        assert adv_A[1, 0] != pytest.approx(adv_B[1, 0])

    def test_adv_at_step_does_not_inherit_across_truncated(self) -> None:
        # Under truncation, the in-buffer V(s_{t+1}) bootstrap IS still
        # used (truncation doesn't zero it — that's the whole point of
        # treating term and trunc differently). What truncation blocks
        # is GAE *inheritance* — the gae_{t+1} term must not bleed into
        # gae_t. So perturbing rewards[1] (which only feeds gae[1] via
        # delta[1]) must not change adv[0] under truncated[0]=True.
        # We keep values[1] fixed so the bootstrap stays the same.
        truncated_A = np.array([[True], [False]])
        terminated_A = np.zeros((2, 1), dtype=bool)
        last_values = np.array([0.0], dtype=np.float32)

        values = np.array([[0.5], [10.0]], dtype=np.float32)
        rewards_A = np.array([[1.0], [50.0]], dtype=np.float32)
        rewards_B = np.array([[1.0], [9999.0]], dtype=np.float32)

        adv_A, _ = compute_gae(
            rewards=rewards_A,
            values=values,
            terminated=terminated_A,
            truncated=truncated_A,
            last_values=last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        adv_B, _ = compute_gae(
            rewards=rewards_B,
            values=values,
            terminated=terminated_A,
            truncated=truncated_A,
            last_values=last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        # adv[0] unchanged because GAE inheritance is blocked at the truncated step.
        assert adv_A[0, 0] == pytest.approx(adv_B[0, 0])
        # adv[1] differs because rewards[1] is in its delta.
        assert adv_A[1, 0] != pytest.approx(adv_B[1, 0])


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


class TestIdentities:
    @pytest.mark.parametrize("seed", [0, 13])
    def test_returns_equal_advantages_plus_values(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        T, N = 16, 8
        rewards = rng.standard_normal((T, N)).astype(np.float32)
        values = rng.standard_normal((T, N)).astype(np.float32)
        terminated = rng.random((T, N)) < 0.05
        truncated = (rng.random((T, N)) < 0.05) & ~terminated
        last_values = rng.standard_normal(N).astype(np.float32)

        adv, ret = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terminated,
            truncated=truncated,
            last_values=last_values,
            gamma=0.995,
            gae_lambda=0.95,
        )
        np.testing.assert_allclose(ret, adv + values, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((100, 10)).astype(np.float32) * 5.0 + 7.0
        norm = normalize_advantages(a)
        assert abs(float(norm.mean())) < 1e-5
        assert abs(float(norm.std()) - 1.0) < 1e-2

    def test_degenerate_constant_input(self) -> None:
        a = np.ones((5, 3), dtype=np.float32) * 3.14
        norm = normalize_advantages(a)
        # All values become 0 (after mean subtraction); no NaN / inf.
        np.testing.assert_allclose(norm, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidators:
    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            compute_gae(
                rewards=np.zeros((4, 2)),
                values=np.zeros((4, 2)),
                terminated=np.zeros((4, 2), dtype=bool),
                truncated=np.zeros((4, 2), dtype=bool),
                last_values=np.zeros(3),  # wrong N
                gamma=0.99,
                gae_lambda=0.95,
            )

    def test_gamma_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            compute_gae(
                rewards=np.zeros((2, 1)),
                values=np.zeros((2, 1)),
                terminated=np.zeros((2, 1), dtype=bool),
                truncated=np.zeros((2, 1), dtype=bool),
                last_values=np.zeros(1),
                gamma=1.5,
                gae_lambda=0.95,
            )

    def test_gae_lambda_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="gae_lambda"):
            compute_gae(
                rewards=np.zeros((2, 1)),
                values=np.zeros((2, 1)),
                terminated=np.zeros((2, 1), dtype=bool),
                truncated=np.zeros((2, 1), dtype=bool),
                last_values=np.zeros(1),
                gamma=0.99,
                gae_lambda=1.5,
            )
