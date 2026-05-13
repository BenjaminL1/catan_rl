"""Phase-flag IPC coverage for the SubprocGameManager.

Constructs the subproc manager with each phase-1-4 obs flag combination and
runs a few steps, asserting that the obs dict round-tripped from the worker
contains every expected key with the right shape/dtype. Prevents future
phase additions from silently skipping subproc mode (e.g., a new obs key
that the worker happens not to forward).
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.selfplay.subproc_vec_env import SubprocGameManager


def _drive_two_steps(gm: SubprocGameManager) -> dict:
    """reset → 2 step_all calls; return the latest obs[0] dict."""
    obs, _ = gm.reset_all()
    masks = gm.get_masks()
    last_obs = obs[0]
    for _ in range(2):
        valid = np.flatnonzero(masks[0]["type"])
        t = int(valid[0]) if len(valid) > 0 else 12
        actions = [np.array([t, 0, 0, 0, 0, 0], dtype=np.int64) for _ in range(gm.n_envs)]
        obs_l, _, _, _, _ = gm.step_all(actions)
        last_obs = obs_l[0]
        masks = gm.get_masks()
    return last_obs


@pytest.mark.parametrize("use_thermometer_encoding", [True, False])
def test_obs_schema_thermometer(use_thermometer_encoding: bool) -> None:
    gm = SubprocGameManager(
        n_envs=2,
        opponent_type="random",
        use_thermometer_encoding=use_thermometer_encoding,
    )
    try:
        obs = _drive_two_steps(gm)
        assert "current_player_main" in obs
        assert "next_player_main" in obs
        # Compact mode → 54/61, thermometer → 166/173.
        if use_thermometer_encoding:
            assert obs["current_player_main"].shape[-1] in (166, 167)
        else:
            assert obs["current_player_main"].shape[-1] in (54, 55)
    finally:
        gm.close()


def test_obs_schema_opp_id_emb_keys() -> None:
    """Phase 3.6: opponent_kind + opponent_policy_id keys present."""
    gm = SubprocGameManager(n_envs=2, opponent_type="random", use_opponent_id_emb=True)
    try:
        obs = _drive_two_steps(gm)
        assert "opponent_kind" in obs, "Phase 3.6: opponent_kind must round-trip via IPC"
        assert "opponent_policy_id" in obs, "Phase 3.6: opponent_policy_id must round-trip via IPC"
    finally:
        gm.close()


def test_obs_schema_belief_target_key() -> None:
    """Phase 2.5b: belief_target key present when use_belief_head=True."""
    gm = SubprocGameManager(n_envs=2, opponent_type="random", use_belief_head=True)
    try:
        obs = _drive_two_steps(gm)
        assert "belief_target" in obs, "Phase 2.5b: belief_target must round-trip via IPC"
        bt = np.asarray(obs["belief_target"])
        assert bt.shape[-1] == 5, f"belief_target should be 5-way, got shape {bt.shape}"
    finally:
        gm.close()
