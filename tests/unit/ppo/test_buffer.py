"""Tests for `ppo/buffer.py` — CompositeRolloutBuffer.

Pins:

1. Allocation contract: pre-allocated arrays have the right shapes.
2. add() rejects bad shapes / wrong keys / overfill.
3. compute_returns_and_advantages writes correct shape outputs.
4. Per-rollout advantage normalization yields ~zero mean / unit std.
5. minibatch_indices covers the full buffer, no overlap, no leftover.
6. get_batch returns tensors on the requested device with the right
   shape and the right values (round-tripping via flat indices).
7. Optional aux storage (belief, opp-action, opp-id) is only
   allocated when configured and properly skipped otherwise.
8. Buffer raises clearly on misuse (uninitialised batch fetch,
   double-finalise, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from catan_rl.ppo.buffer import CompositeRolloutBuffer, MaskSpec, ObsSpec

# ---------------------------------------------------------------------------
# Fixture buffer
# ---------------------------------------------------------------------------


def _make_obs_spec() -> dict[str, ObsSpec]:
    return {
        "tile_repr": ObsSpec(shape=(19, 79), dtype=np.dtype(np.float32)),
        "current_player_main": ObsSpec(shape=(54,), dtype=np.dtype(np.float32)),
        "opp_kind": ObsSpec(shape=(), dtype=np.dtype(np.int64)),
    }


def _make_mask_spec() -> dict[str, MaskSpec]:
    return {
        "type": MaskSpec(shape=(13,)),
        "corner_settlement": MaskSpec(shape=(54,)),
        "resource1_default": MaskSpec(shape=(5,)),
    }


def _make_buffer(
    n_steps: int = 4,
    n_envs: int = 2,
    belief_target_dim: int | None = None,
    opp_id_enabled: bool = False,
    opp_action_target_enabled: bool = False,
) -> CompositeRolloutBuffer:
    return CompositeRolloutBuffer(
        n_steps=n_steps,
        n_envs=n_envs,
        obs_spec=_make_obs_spec(),
        mask_spec=_make_mask_spec(),
        belief_target_dim=belief_target_dim,
        opp_id_enabled=opp_id_enabled,
        opp_action_target_enabled=opp_action_target_enabled,
    )


def _make_step_data(N: int = 2) -> dict:
    rng = np.random.default_rng(0)
    obs = {
        "tile_repr": rng.standard_normal((N, 19, 79)).astype(np.float32),
        "current_player_main": rng.standard_normal((N, 54)).astype(np.float32),
        "opp_kind": rng.integers(0, 5, (N,)).astype(np.int64),
    }
    action = rng.integers(0, 5, (N, 6)).astype(np.int64)
    per_head_log_prob = rng.standard_normal((N, 6)).astype(np.float32)
    log_prob = per_head_log_prob.sum(axis=-1)
    value = rng.standard_normal((N,)).astype(np.float32)
    reward = rng.standard_normal((N,)).astype(np.float32)
    terminated = np.zeros(N, dtype=bool)
    truncated = np.zeros(N, dtype=bool)
    masks = {
        "type": np.ones((N, 13), dtype=bool),
        "corner_settlement": np.ones((N, 54), dtype=bool),
        "resource1_default": np.ones((N, 5), dtype=bool),
    }
    return {
        "obs": obs,
        "action": action,
        "per_head_log_prob": per_head_log_prob,
        "log_prob": log_prob,
        "value": value,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "masks": masks,
    }


# ---------------------------------------------------------------------------
# Construction + allocation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_shapes_pre_allocated(self) -> None:
        b = _make_buffer(n_steps=4, n_envs=3)
        assert b.actions.shape == (4, 3, 6)
        assert b.per_head_log_probs.shape == (4, 3, 6)
        assert b.values.shape == (4, 3)
        assert b.rewards.shape == (4, 3)
        assert b.terminated.shape == (4, 3) and b.terminated.dtype == bool
        assert b.truncated.shape == (4, 3) and b.truncated.dtype == bool
        assert b.obs["tile_repr"].shape == (4, 3, 19, 79)
        assert b.obs["current_player_main"].shape == (4, 3, 54)
        assert b.obs["opp_kind"].shape == (4, 3)
        assert b.masks["type"].shape == (4, 3, 13)
        assert b.masks["corner_settlement"].shape == (4, 3, 54)
        assert b.total_transitions == 12
        assert not b.is_full
        assert not b.is_finalised

    def test_belief_target_allocated_when_dim_given(self) -> None:
        b = _make_buffer(belief_target_dim=5)
        assert b.belief_target is not None
        assert b.belief_target.shape == (4, 2, 5)

    def test_belief_target_unallocated_when_dim_none(self) -> None:
        b = _make_buffer(belief_target_dim=None)
        assert b.belief_target is None

    def test_opp_id_storage_gated(self) -> None:
        b = _make_buffer(opp_id_enabled=True)
        assert b.opp_kind is not None and b.opp_kind.shape == (4, 2)
        assert b.opp_policy_id is not None and b.opp_policy_id.shape == (4, 2)
        b2 = _make_buffer(opp_id_enabled=False)
        assert b2.opp_kind is None
        assert b2.opp_policy_id is None

    def test_negative_n_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_steps"):
            CompositeRolloutBuffer(
                n_steps=0,
                n_envs=1,
                obs_spec=_make_obs_spec(),
                mask_spec=_make_mask_spec(),
            )

    def test_empty_obs_spec_rejected(self) -> None:
        with pytest.raises(ValueError, match="obs_spec"):
            CompositeRolloutBuffer(
                n_steps=4,
                n_envs=1,
                obs_spec={},
                mask_spec=_make_mask_spec(),
            )


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


class TestAdd:
    def test_basic_add(self) -> None:
        b = _make_buffer()
        data = _make_step_data(N=2)
        b.add(**data)
        assert b._pos == 1
        # The newest slot got written.
        np.testing.assert_array_equal(b.actions[0], data["action"])
        np.testing.assert_array_equal(b.rewards[0], data["reward"])

    def test_overfill_rejected(self) -> None:
        b = _make_buffer(n_steps=2, n_envs=2)
        d = _make_step_data(N=2)
        b.add(**d)
        b.add(**d)
        with pytest.raises(RuntimeError, match="Buffer full"):
            b.add(**d)

    def test_unknown_obs_key_rejected(self) -> None:
        b = _make_buffer()
        d = _make_step_data(N=2)
        d["obs"]["mystery_key"] = np.zeros((2, 3), dtype=np.float32)
        with pytest.raises(KeyError, match="mystery_key"):
            b.add(**d)

    def test_wrong_shape_rejected(self) -> None:
        b = _make_buffer(n_steps=4, n_envs=2)
        d = _make_step_data(N=2)
        d["action"] = d["action"][:1]  # (1, 6) instead of (2, 6)
        with pytest.raises(ValueError, match="action shape"):
            b.add(**d)

    def test_belief_passed_without_alloc_raises(self) -> None:
        b = _make_buffer(belief_target_dim=None)
        d = _make_step_data(N=2)
        d["belief_target"] = np.zeros((2, 5), dtype=np.float32)
        with pytest.raises(RuntimeError, match="belief_target"):
            b.add(**d)

    def test_belief_alloc_but_missing_kwarg_raises(self) -> None:
        # Reviewer fix: belief slot allocated but caller forgets to pass
        # belief_target → buffer used to silently store zeros, which
        # would collapse the belief head when the trainer reduces against
        # all-zero targets.
        b = _make_buffer(belief_target_dim=5)
        d = _make_step_data(N=2)
        with pytest.raises(ValueError, match="belief_target"):
            b.add(**d)

    def test_opp_action_alloc_but_missing_kwarg_raises(self) -> None:
        b = _make_buffer(opp_action_target_enabled=True)
        d = _make_step_data(N=2)
        with pytest.raises(ValueError, match="opp_action_target"):
            b.add(**d)

    def test_opp_id_alloc_but_missing_kwarg_raises(self) -> None:
        b = _make_buffer(opp_id_enabled=True)
        d = _make_step_data(N=2)
        with pytest.raises(ValueError, match=r"opp_kind|opp_id"):
            b.add(**d)

    def test_belief_round_trip(self) -> None:
        b = _make_buffer(belief_target_dim=5)
        d = _make_step_data(N=2)
        d["belief_target"] = np.full((2, 5), 0.2, dtype=np.float32)
        b.add(**d)
        assert b.belief_target is not None
        np.testing.assert_array_equal(b.belief_target[0], d["belief_target"])

    def test_opp_action_requires_valid_mask(self) -> None:
        b = _make_buffer(opp_action_target_enabled=True)
        d = _make_step_data(N=2)
        d["opp_action_target"] = np.zeros(2, dtype=np.int64)
        with pytest.raises(ValueError, match="opp_action_target_valid"):
            b.add(**d)

    def test_opp_id_requires_both_keys(self) -> None:
        b = _make_buffer(opp_id_enabled=True)
        d = _make_step_data(N=2)
        d["opp_kind"] = np.zeros(2, dtype=np.int64)
        with pytest.raises(ValueError, match="opp_policy_id"):
            b.add(**d)


# ---------------------------------------------------------------------------
# GAE wiring + advantage norm
# ---------------------------------------------------------------------------


def _fill_buffer(b: CompositeRolloutBuffer) -> None:
    for _ in range(b.n_steps):
        b.add(**_make_step_data(N=b.n_envs))


class TestGAEWiring:
    def test_compute_runs_without_error(self) -> None:
        b = _make_buffer(n_steps=4, n_envs=2)
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(2, dtype=np.float32),
            gamma=0.995,
            gae_lambda=0.95,
            advantage_norm="none",
        )
        assert b.advantages is not None and b.advantages.shape == (4, 2)
        assert b.returns is not None and b.returns.shape == (4, 2)
        # PPO target identity holds when advantages are NOT normalised.
        # The value head is trained against `returns`; the policy
        # gradient may use the normalised advantages.
        np.testing.assert_allclose(b.returns, b.advantages + b.values, atol=1e-5, rtol=1e-5)

    def test_advantages_raw_preserved_under_rollout_norm(self) -> None:
        # Reviewer fix: rollout norm used to overwrite advantages with
        # the standardised version, losing the raw values. Phase 4
        # trainer needs them for per-batch renorm + diagnostics.
        rng = np.random.default_rng(0)
        b = _make_buffer(n_steps=8, n_envs=4)
        for _ in range(b.n_steps):
            d = _make_step_data(N=b.n_envs)
            d["value"] = rng.standard_normal(b.n_envs).astype(np.float32) * 5
            b.add(**d)
        b.compute_returns_and_advantages(
            last_values=np.zeros(b.n_envs, dtype=np.float32),
            gamma=0.995,
            gae_lambda=0.95,
            advantage_norm="rollout",
        )
        assert b.advantages_raw is not None
        # Raw advantages are the unnormalised GAE output:
        # returns - values == raw advantages by the PPO identity.
        np.testing.assert_allclose(b.advantages_raw, b.returns - b.values, atol=1e-5, rtol=1e-5)
        # And the normalised .advantages is zero-mean (~).
        assert abs(float(b.advantages.mean())) < 1e-3

    def test_advantages_raw_equals_advantages_when_norm_none(self) -> None:
        b = _make_buffer(n_steps=4, n_envs=2)
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(2, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
            advantage_norm="none",
        )
        assert b.advantages_raw is not None
        np.testing.assert_array_equal(b.advantages, b.advantages_raw)

    def test_returns_unchanged_when_advantage_norm_rollout(self) -> None:
        # When normalisation is on, b.advantages is mean-zeroed but
        # b.returns is the UNNORMALISED value target. Pin the contract.
        b = _make_buffer(n_steps=4, n_envs=2)
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(2, dtype=np.float32),
            gamma=0.995,
            gae_lambda=0.95,
            advantage_norm="rollout",
        )
        assert b.advantages is not None
        assert b.returns is not None
        # advantages should be roughly zero-mean.
        assert abs(float(b.advantages.mean())) < 1e-3
        # returns is NOT normalised — its mean tracks values + raw adv.
        # Specifically: returns + (-advantages.normalised) != values; the
        # identity only holds vs the unnormalised advantage we no longer
        # have access to. So just sanity-check returns has not been mean
        # zeroed.
        assert abs(float(b.returns.mean())) > 1e-6

    def test_rollout_norm_yields_zero_mean(self) -> None:
        rng = np.random.default_rng(0)
        # Bigger buffer so the mean is meaningfully measurable.
        b = _make_buffer(n_steps=64, n_envs=8)
        for _ in range(b.n_steps):
            d = _make_step_data(N=b.n_envs)
            # Spice the values + rewards so the advantages are non-degenerate.
            d["value"] = rng.standard_normal(b.n_envs).astype(np.float32) * 5
            d["reward"] = rng.standard_normal(b.n_envs).astype(np.float32) * 3
            b.add(**d)
        b.compute_returns_and_advantages(
            last_values=np.zeros(b.n_envs, dtype=np.float32),
            gamma=0.995,
            gae_lambda=0.95,
            advantage_norm="rollout",
        )
        assert b.advantages is not None
        assert abs(float(b.advantages.mean())) < 1e-3
        assert abs(float(b.advantages.std()) - 1.0) < 1e-2

    def test_batch_norm_leaves_raw_advantages(self) -> None:
        # batch-mode means "trainer normalises per minibatch later" — the
        # buffer's advantages should NOT be normalised.
        rng = np.random.default_rng(0)
        b = _make_buffer(n_steps=8, n_envs=4)
        for _ in range(b.n_steps):
            d = _make_step_data(N=b.n_envs)
            d["value"] = rng.standard_normal(b.n_envs).astype(np.float32) * 5
            b.add(**d)
        b.compute_returns_and_advantages(
            last_values=np.zeros(b.n_envs, dtype=np.float32),
            gamma=0.995,
            gae_lambda=0.95,
            advantage_norm="batch",
        )
        # No global-normalisation step → mean is unlikely to be exactly 0.
        assert b.advantages is not None
        assert abs(float(b.advantages.mean())) > 1e-6

    def test_compute_before_full_rejected(self) -> None:
        b = _make_buffer()
        with pytest.raises(RuntimeError, match="not full"):
            b.compute_returns_and_advantages(
                last_values=np.zeros(2, dtype=np.float32),
                gamma=0.99,
                gae_lambda=0.95,
            )


# ---------------------------------------------------------------------------
# Minibatch sampling
# ---------------------------------------------------------------------------


class TestMinibatchIndices:
    def test_covers_full_buffer_no_overlap(self) -> None:
        b = _make_buffer(n_steps=8, n_envs=4)  # 32 transitions
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(4, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
        )
        rng = np.random.default_rng(0)
        batches = b.minibatch_indices(batch_size=8, rng=rng)
        assert len(batches) == 4
        all_idx = np.concatenate(batches)
        assert all_idx.shape == (32,)
        assert sorted(all_idx.tolist()) == list(range(32))

    def test_uneven_batch_size_rejected(self) -> None:
        b = _make_buffer(n_steps=8, n_envs=4)  # 32 transitions
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(4, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
        )
        with pytest.raises(ValueError, match="does not divide"):
            b.minibatch_indices(batch_size=7, rng=np.random.default_rng(0))

    def test_before_finalise_rejected(self) -> None:
        b = _make_buffer()
        with pytest.raises(RuntimeError, match="not finalised"):
            b.minibatch_indices(batch_size=2, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# get_batch
# ---------------------------------------------------------------------------


class TestGetBatch:
    def test_returns_tensors_with_right_shapes(self) -> None:
        b = _make_buffer(n_steps=4, n_envs=2)  # 8 transitions
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(2, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
        )
        idx = np.arange(8)
        batch = b.get_batch(idx, device="cpu")
        assert batch["actions"].shape == (8, 6)
        assert batch["obs"]["tile_repr"].shape == (8, 19, 79)
        assert batch["obs"]["opp_kind"].shape == (8,)
        assert batch["masks"]["type"].shape == (8, 13)
        assert batch["advantages"].shape == (8,)
        assert isinstance(batch["actions"], torch.Tensor)

    def test_round_trip_via_flat_indices(self) -> None:
        b = _make_buffer(n_steps=4, n_envs=2)
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(2, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
        )
        # Specific flat indices: (t=2, n=1) → idx=5; (t=0, n=0) → 0
        idx = np.array([5, 0], dtype=np.int64)
        batch = b.get_batch(idx, device="cpu")
        # batch[k][0] should equal b.actions[2, 1]
        assert (batch["actions"][0].numpy() == b.actions[2, 1]).all()
        assert (batch["actions"][1].numpy() == b.actions[0, 0]).all()

    def test_indices_out_of_range_rejected(self) -> None:
        b = _make_buffer(n_steps=4, n_envs=2)
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(2, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
        )
        with pytest.raises(IndexError, match="out of range"):
            b.get_batch(np.array([100]), device="cpu")


# ---------------------------------------------------------------------------
# Lifecycle / state machine
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_reset_clears_finalised_state(self) -> None:
        b = _make_buffer(n_steps=4, n_envs=2)
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(2, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
        )
        assert b.is_finalised
        b.reset()
        assert not b.is_finalised
        # Can add again
        b.add(**_make_step_data(N=2))
        assert b._pos == 1

    def test_add_after_finalise_rejected(self) -> None:
        b = _make_buffer(n_steps=4, n_envs=2)
        _fill_buffer(b)
        b.compute_returns_and_advantages(
            last_values=np.zeros(2, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
        )
        # add() must not silently bring a finalised buffer back to life.
        b.reset()  # required to add again — without this, raises.
        b.add(**_make_step_data(N=2))

        # And without reset(), it raises.
        b2 = _make_buffer(n_steps=4, n_envs=2)
        _fill_buffer(b2)
        b2.compute_returns_and_advantages(
            last_values=np.zeros(2, dtype=np.float32),
            gamma=0.99,
            gae_lambda=0.95,
        )
        with pytest.raises(RuntimeError, match="finalised"):
            b2.add(**_make_step_data(N=2))
