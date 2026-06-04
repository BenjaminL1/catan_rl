"""Tests for `ppo/game_manager.py` — RolloutCollector glue.

Uses a tiny stub policy + a real SerialVecEnv. The stub policy emits
random masked actions and dummy values; this test only verifies the
collector's plumbing, not policy correctness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from catan_rl.ppo.buffer import CompositeRolloutBuffer
from catan_rl.ppo.game_manager import RolloutCollector
from catan_rl.ppo.vec_env import SerialVecEnv, mask_spec_from_env, obs_spec_from_env


class _StubPolicy(nn.Module):
    """Tiny policy stand-in.

    ``sample`` picks a uniformly random *legal* type action and
    fills the remaining 5 heads with zeros. Returns scalar value /
    log_prob / per_head_log_prob so the collector exercises the full
    storage path.
    """

    def __init__(self, n_envs: int) -> None:
        super().__init__()
        self.n_envs = n_envs
        # One trainable param so torch is happy.
        self.dummy = nn.Parameter(torch.zeros(1))
        self._rng = np.random.default_rng(0)

    def sample(
        self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        B = next(iter(obs.values())).shape[0]
        type_mask = masks["type"].cpu().numpy()
        action = np.zeros((B, 6), dtype=np.int64)
        for i in range(B):
            legal = np.flatnonzero(type_mask[i])
            if legal.size:
                action[i, 0] = int(self._rng.choice(legal))
            else:
                action[i, 0] = 3  # END_TURN fallback
        device = next(iter(obs.values())).device
        return {
            "action": torch.as_tensor(action, device=device),
            "log_prob": torch.zeros(B, device=device) + self.dummy,
            "value": torch.zeros(B, device=device),
            "per_head_log_prob": torch.zeros((B, 6), device=device),
            "entropy": torch.zeros(B, device=device) + 1.5,
        }

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        B = next(iter(obs.values())).shape[0]
        device = next(iter(obs.values())).device
        return {"value": torch.zeros(B, device=device) + self.dummy}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def vec_env_and_buffer():
    kwargs = {"opponent_type": "heuristic", "max_turns": 50}
    ve = SerialVecEnv(env_kwargs_list=[kwargs, kwargs])
    obs_spec = obs_spec_from_env(ve.envs[0])
    mask_spec = mask_spec_from_env(ve.envs[0])
    buffer = CompositeRolloutBuffer(
        n_steps=4,
        n_envs=2,
        obs_spec=obs_spec,
        mask_spec=mask_spec,
    )
    yield ve, buffer
    ve.close()


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


class TestRolloutCollector:
    def test_collect_fills_buffer(self, vec_env_and_buffer) -> None:
        ve, buf = vec_env_and_buffer
        policy = _StubPolicy(n_envs=2)
        collector = RolloutCollector(vec_env=ve, policy=policy, buffer=buf, device="cpu")
        obs, masks = ve.reset_all(seeds=[0, 1])
        obs2, masks2 = collector.collect(obs, masks)
        assert buf.is_full, "buffer should be full after collect"
        # Trailing obs returned for the next rollout to use.
        assert set(obs2.keys()) == set(obs.keys())
        assert set(masks2.keys()) == set(masks.keys())

    def test_last_values_populated(self, vec_env_and_buffer) -> None:
        ve, buf = vec_env_and_buffer
        policy = _StubPolicy(n_envs=2)
        collector = RolloutCollector(vec_env=ve, policy=policy, buffer=buf, device="cpu")
        obs, masks = ve.reset_all(seeds=[0, 1])
        collector.collect(obs, masks)
        assert collector.last_values.shape == (2,)
        assert collector.last_values.dtype == np.float32

    def test_n_env_mismatch_rejected(self) -> None:
        kwargs = {"opponent_type": "random", "max_turns": 50}
        ve = SerialVecEnv(env_kwargs_list=[kwargs, kwargs])
        obs_spec = obs_spec_from_env(ve.envs[0])
        mask_spec = mask_spec_from_env(ve.envs[0])
        # Buffer with wrong n_envs.
        bad_buf = CompositeRolloutBuffer(
            n_steps=4,
            n_envs=4,  # vec env has 2
            obs_spec=obs_spec,
            mask_spec=mask_spec,
        )
        policy = _StubPolicy(n_envs=2)
        with pytest.raises(ValueError, match="n_envs"):
            RolloutCollector(vec_env=ve, policy=policy, buffer=bad_buf, device="cpu")
        ve.close()

    def test_two_rollouts_in_sequence(self, vec_env_and_buffer) -> None:
        # The trailing (obs, masks) feeds back into the next collect.
        # Verify a second collect doesn't blow up (buffer was reset
        # internally by collect).
        ve, buf = vec_env_and_buffer
        policy = _StubPolicy(n_envs=2)
        collector = RolloutCollector(vec_env=ve, policy=policy, buffer=buf, device="cpu")
        obs, masks = ve.reset_all(seeds=[0, 1])
        obs, masks = collector.collect(obs, masks)
        # Finalise first buffer, then second collect resets it.
        buf.compute_returns_and_advantages(
            last_values=collector.last_values,
            gamma=0.99,
            gae_lambda=0.95,
        )
        # Second rollout. collect() should call buffer.reset() internally.
        obs, masks = collector.collect(obs, masks)
        assert buf.is_full
