"""Integration test: env produces obs matching its observation_space.

This is the integration counterpart to the obs_encoder unit tests. It
drives a real game through the env's state machine and asserts every
observation returned by ``reset()`` and ``step()`` satisfies the
declared ``observation_space``.
"""

from __future__ import annotations

import numpy as np

from catan_rl.env.catan_env import CatanEnv


def test_reset_obs_matches_observation_space() -> None:
    env = CatanEnv(opponent_type="random", max_turns=200)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), (
        f"obs from reset() not in observation_space; keys={list(obs.keys())}"
    )


def test_step_obs_matches_observation_space_for_50_steps() -> None:
    """50 steps cover the setup → roll → main-turn → end-turn cycle plus
    at least one opponent turn, exercising all phase-flag combinations."""
    rng = np.random.default_rng(0)
    env = CatanEnv(opponent_type="random", max_turns=200)
    obs, _ = env.reset(seed=0)
    for step_i in range(50):
        masks = env.get_action_masks()
        legal_types = np.flatnonzero(masks["type"])
        t = int(rng.choice(legal_types))
        action = np.zeros(6, dtype=np.int64)
        action[0] = t
        for head_idx, key in enumerate(
            ("corner_settlement", "edge", "tile", "resource1_default", "resource2_default"),
            start=1,
        ):
            if key == "corner_settlement" and t == 1:
                key = "corner_city"
            if key == "resource1_default" and t == 11:
                key = "resource1_discard"
            elif key == "resource1_default" and t == 10:
                key = "resource1_trade"
            legal = np.flatnonzero(masks[key])
            if legal.size:
                action[head_idx] = int(rng.choice(legal))
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs), (
            f"step {step_i} (type={t}): obs out of space; keys={list(obs.keys())}"
        )
        if terminated or truncated:
            obs, _ = env.reset(seed=step_i + 1)


def test_obs_hand_tracker_matches_engine_after_many_steps() -> None:
    """Drive the env for 500 steps and verify the obs encoder's opponent
    resource counts (via the hand tracker) match the engine's ground truth."""
    from catan_rl.env.hand_tracker import RESOURCES_CW

    rng = np.random.default_rng(11)
    env = CatanEnv(opponent_type="random", max_turns=400)
    env.reset(seed=0)
    assert env._hand_tracker is not None and env.opponent_player is not None

    for _ in range(500):
        masks = env.get_action_masks()
        legal_types = np.flatnonzero(masks["type"])
        t = int(rng.choice(legal_types))
        action = np.zeros(6, dtype=np.int64)
        action[0] = t
        for head_idx, key in enumerate(
            ("corner_settlement", "edge", "tile", "resource1_default", "resource2_default"),
            start=1,
        ):
            if key == "corner_settlement" and t == 1:
                key = "corner_city"
            if key == "resource1_default" and t == 11:
                key = "resource1_discard"
            elif key == "resource1_default" and t == 10:
                key = "resource1_trade"
            legal = np.flatnonzero(masks[key])
            if legal.size:
                action[head_idx] = int(rng.choice(legal))
        _, _, terminated, truncated, _ = env.step(action)
        tracked = env._hand_tracker.get_hand(env.opponent_player.name)
        actual = {r: int(env.opponent_player.resources.get(r, 0)) for r in RESOURCES_CW}
        assert tracked == actual, f"tracker drift: {tracked} vs actual {actual}"
        if terminated or truncated:
            break
