"""Tests for Phase 1.2 per-rollout advantage normalization."""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.algorithms.common.rollout_buffer import CompositeRolloutBuffer
from catan_rl.models.utils import (
    CURR_PLAYER_DIM,
    MAX_DEV_SEQ,
    N_TILES,
    NEXT_PLAYER_DIM,
    OBS_TILE_DIM,
)


def _fake_obs() -> dict:
    return {
        "tile_representations": np.zeros((N_TILES, OBS_TILE_DIM), dtype=np.float32),
        "current_player_main": np.zeros(CURR_PLAYER_DIM, dtype=np.float32),
        "next_player_main": np.zeros(NEXT_PLAYER_DIM, dtype=np.float32),
        "current_player_hidden_dev": np.zeros(MAX_DEV_SEQ, dtype=np.int32),
        "current_player_played_dev": np.zeros(MAX_DEV_SEQ, dtype=np.int32),
        "next_player_played_dev": np.zeros(MAX_DEV_SEQ, dtype=np.int32),
    }


def _fake_masks() -> dict:
    return {
        "type": np.ones(13, dtype=bool),
        "corner_settlement": np.ones(54, dtype=bool),
        "corner_city": np.ones(54, dtype=bool),
        "edge": np.ones(72, dtype=bool),
        "tile": np.ones(19, dtype=bool),
        "resource1_trade": np.ones(5, dtype=bool),
        "resource1_discard": np.ones(5, dtype=bool),
        "resource1_default": np.ones(5, dtype=bool),
        "resource2_default": np.ones(5, dtype=bool),
    }


def _fill_buffer(buf: CompositeRolloutBuffer, rewards: list[float]) -> None:
    obs, masks = _fake_obs(), _fake_masks()
    action = np.zeros(6, dtype=np.int64)
    for r in rewards:
        buf.add(obs, action, r, False, False, value=0.0, log_prob=0.0, masks=masks)


def test_rollout_norm_makes_zero_mean_unit_std() -> None:
    """Default 'rollout' mode standardizes advantages globally over the buffer."""
    buf = CompositeRolloutBuffer(n_steps=8, n_envs=1)
    _fill_buffer(buf, rewards=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    buf.compute_returns_and_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    # advantage_norm defaults to 'rollout'; check std ~ 1, mean ~ 0.
    adv = buf.advantages[:8]
    assert abs(adv.mean()) < 1e-5
    assert abs(adv.std() - 1.0) < 1e-3


def test_batch_mode_leaves_advantages_raw() -> None:
    """'batch' mode skips global standardization (trainer normalizes per-batch)."""
    buf = CompositeRolloutBuffer(n_steps=8, n_envs=1)
    _fill_buffer(buf, rewards=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    buf.compute_returns_and_advantages(
        last_value=0.0, gamma=0.99, gae_lambda=0.95, advantage_norm="batch"
    )
    adv = buf.advantages[:8]
    # Raw (un-standardized) advantages have nonzero mean for these rewards.
    assert abs(adv.mean()) > 0.01


def test_none_mode_passes_through() -> None:
    """'none' mode is identical to 'batch' for the buffer's purposes."""
    buf = CompositeRolloutBuffer(n_steps=4, n_envs=1)
    _fill_buffer(buf, rewards=[1.0, 1.0, 1.0, 1.0])
    buf.compute_returns_and_advantages(
        last_value=0.0, gamma=1.0, gae_lambda=1.0, advantage_norm="none"
    )
    adv = buf.advantages[:4]
    # Without normalization the GAE recurrence is preserved exactly.
    np.testing.assert_allclose(adv, [4.0, 3.0, 2.0, 1.0], atol=1e-5)


def test_rollout_n_envs_2_normalizes_globally() -> None:
    """Global standardization spans both envs' interleaved transitions."""
    buf = CompositeRolloutBuffer(n_steps=8, n_envs=2)
    _fill_buffer(buf, rewards=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    last_values = np.zeros(2, dtype=np.float32)
    buf.compute_returns_and_advantages(
        last_values, gamma=0.99, gae_lambda=0.95, advantage_norm="rollout"
    )
    adv = buf.advantages[:8]
    assert abs(adv.mean()) < 1e-5
    assert abs(adv.std() - 1.0) < 1e-3


def test_invalid_mode_raises() -> None:
    """Trainer-level validation rejects unknown advantage_norm values."""
    from catan_rl.algorithms.ppo.arguments import get_config
    from catan_rl.algorithms.ppo.trainer import CatanPPO

    cfg = get_config()
    cfg.update(
        {
            "advantage_norm": "lol",
            "total_timesteps": 100,
            "n_steps": 50,
            "n_envs": 1,
            "batch_size": 16,
            "n_epochs": 1,
            "eval_games": 0,
            "eval_freq": 99999,
            "checkpoint_freq": 99999,
            "log_dir": "/tmp/cr_test_advn",
            "checkpoint_dir": "/tmp/cr_test_advn_ckpts",
        }
    )
    with pytest.raises(ValueError, match="advantage_norm"):
        CatanPPO(cfg)
