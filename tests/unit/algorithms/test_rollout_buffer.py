"""Tests for the Phase 0 rollout buffer split (terminated/truncated).

The buffer's ``add()`` now takes both flags. Old code that only knows about
``done`` reads it via the back-compat ``buffer.dones`` property which OR's
the two arrays.
"""

from __future__ import annotations

import numpy as np

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


def test_buffer_uses_canonical_obs_constants() -> None:
    """Buffer storage shapes derive from models.utils constants, not magic numbers."""
    buf = CompositeRolloutBuffer(n_steps=4, n_envs=1)
    assert buf.tile_representations.shape == (4, N_TILES, OBS_TILE_DIM)
    assert buf.current_player_main.shape == (4, CURR_PLAYER_DIM)
    assert buf.next_player_main.shape == (4, NEXT_PLAYER_DIM)


def test_buffer_split_terminated_truncated() -> None:
    """add() accepts both flags; both arrays exist; dones property OR's them."""
    buf = CompositeRolloutBuffer(n_steps=4, n_envs=1)
    obs = _fake_obs()
    masks = _fake_masks()
    action = np.zeros(6, dtype=np.int64)
    # 1: terminated only
    buf.add(
        obs, action, 1.0, terminated=True, truncated=False, value=0.5, log_prob=0.0, masks=masks
    )
    # 2: truncated only
    buf.add(
        obs, action, 1.0, terminated=False, truncated=True, value=0.5, log_prob=0.0, masks=masks
    )
    # 3: neither
    buf.add(
        obs, action, 1.0, terminated=False, truncated=False, value=0.5, log_prob=0.0, masks=masks
    )
    # 4: neither
    buf.add(
        obs, action, 1.0, terminated=False, truncated=False, value=0.5, log_prob=0.0, masks=masks
    )

    assert list(buf.terminated[:4]) == [1.0, 0.0, 0.0, 0.0]
    assert list(buf.truncated[:4]) == [0.0, 1.0, 0.0, 0.0]
    # dones property is OR(terminated, truncated)
    assert list(buf.dones[:4]) == [1.0, 1.0, 0.0, 0.0]


def test_compute_returns_uses_split_signature() -> None:
    """The buffer's GAE call passes the split arrays through to compute_gae."""
    buf = CompositeRolloutBuffer(n_steps=2, n_envs=1)
    obs = _fake_obs()
    masks = _fake_masks()
    action = np.zeros(6, dtype=np.int64)
    # Truncation at the last step: last_value should be used (not zeroed).
    buf.add(obs, action, 1.0, False, False, 0.5, 0.0, masks)
    buf.add(obs, action, 1.0, False, True, 0.5, 0.0, masks)
    # advantage_norm='none' isolates the GAE recurrence from Phase 1.2's
    # global standardization; this test is a Phase 0 contract test.
    buf.compute_returns_and_advantages(
        last_value=2.0, gamma=0.99, gae_lambda=0.95, advantage_norm="none"
    )
    # If truncation were treated as termination, A_1 = 1 - 0.5 = 0.5.
    # With Phase 0, A_1 = 1 + 0.99*2 - 0.5 = 2.48.
    assert abs(buf.advantages[1] - (1.0 + 0.99 * 2.0 - 0.5)) < 1e-5
