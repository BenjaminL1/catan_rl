"""Tests for `ppo/vec_env.py` — SerialVecEnv + spec helpers."""

from __future__ import annotations

import random

import numpy as np
import pytest

from catan_rl.ppo.vec_env import SerialVecEnv, mask_spec_from_env, obs_spec_from_env

# Need both stdlib and numpy seeded for engine determinism (fix #2 from
# the v1 audit) — verify here that vec env respects it too.


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="env_kwargs_list"):
            SerialVecEnv(env_kwargs_list=[])

    def test_builds_n_envs(self) -> None:
        kwargs = {"opponent_type": "random", "max_turns": 50}
        ve = SerialVecEnv(env_kwargs_list=[kwargs, kwargs, kwargs])
        assert ve.n_envs == 3
        assert len(ve.envs) == 3
        ve.close()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestResetAll:
    def test_returns_batched_obs_and_masks(self) -> None:
        kwargs = {"opponent_type": "random", "max_turns": 50}
        ve = SerialVecEnv(env_kwargs_list=[kwargs] * 4)
        try:
            obs, masks = ve.reset_all(seeds=[0, 1, 2, 3])
            for key, arr in obs.items():
                assert arr.shape[0] == 4, f"obs[{key}] not batched: {arr.shape}"
            for arr in masks.values():
                assert arr.shape[0] == 4
                assert arr.dtype == bool
        finally:
            ve.close()

    def test_seed_length_mismatch_rejected(self) -> None:
        ve = SerialVecEnv(env_kwargs_list=[{}, {}])
        try:
            with pytest.raises(ValueError, match="seeds length"):
                ve.reset_all(seeds=[0, 1, 2])
        finally:
            ve.close()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStepAll:
    def test_step_returns_expected_shapes(self) -> None:
        ve = SerialVecEnv(env_kwargs_list=[{"opponent_type": "random", "max_turns": 50}] * 3)
        try:
            _, masks = ve.reset_all(seeds=[0, 1, 2])
            # Take legal type=0 ROLL_DICE (or whatever's masked).
            rng = np.random.default_rng(0)
            actions = np.zeros((3, 6), dtype=np.int64)
            for i in range(3):
                legal = np.flatnonzero(masks["type"][i])
                actions[i, 0] = int(rng.choice(legal))
            obs2, _, rewards, term, trunc, final_obs = ve.step_all(actions)
            assert isinstance(final_obs, dict)
            for arr in obs2.values():
                assert arr.shape[0] == 3
            assert rewards.shape == (3,)
            assert term.shape == (3,) and term.dtype == bool
            assert trunc.shape == (3,) and trunc.dtype == bool
        finally:
            ve.close()

    def test_step_action_shape_rejected(self) -> None:
        ve = SerialVecEnv(env_kwargs_list=[{}] * 2)
        try:
            ve.reset_all(seeds=[0, 1])
            with pytest.raises(ValueError, match="actions shape"):
                ve.step_all(np.zeros((1, 6), dtype=np.int64))  # wrong N
            with pytest.raises(ValueError, match="actions shape"):
                ve.step_all(np.zeros((2, 5), dtype=np.int64))  # wrong action dim
        finally:
            ve.close()

    def test_step_auto_resets_on_terminal(self) -> None:
        # Drive a short game (max_turns=10 + heuristic opponent → likely
        # terminates within a rollout). Verify that when terminated /
        # truncated fires, the returned obs is from a fresh env, NOT
        # the terminal state.
        ve = SerialVecEnv(env_kwargs_list=[{"opponent_type": "heuristic", "max_turns": 10}] * 2)
        random.seed(0)
        np.random.seed(0)
        try:
            _, masks = ve.reset_all(seeds=[0, 1])
            rng = np.random.default_rng(0)
            saw_done = False
            for _ in range(200):  # large enough to hit at least one terminal
                actions = np.zeros((2, 6), dtype=np.int64)
                for i in range(2):
                    legal = np.flatnonzero(masks["type"][i])
                    if not legal.size:
                        actions[i, 0] = 3  # END_TURN
                    else:
                        actions[i, 0] = int(rng.choice(legal))
                _, masks, _r, term, trunc, final_obs = ve.step_all(actions)
                done = term | trunc
                if done.any():
                    saw_done = True
                    # The masks for done envs should still have at
                    # least one legal type action — that means the env
                    # was auto-reset (terminal envs have no legal
                    # actions because the game ended).
                    for i in np.flatnonzero(done):
                        assert masks["type"][i].any(), (
                            f"env {i} done but mask is all-False — auto-reset did not fire"
                        )
                        # Reviewer fix: final_obs must capture the
                        # terminal obs BEFORE auto-reset so the
                        # collector can value-bootstrap correctly.
                        assert int(i) in final_obs, f"env {i} done but final_obs not populated"
                    break
            assert saw_done, "no env terminated in 200 steps; widen the test"
        finally:
            ve.close()


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_yields_same_first_obs(self) -> None:
        # Vec env with explicit seed should produce the same first-rollout
        # obs sequence on two independent constructions.
        kwargs = {"opponent_type": "heuristic", "max_turns": 50}
        ve_a = SerialVecEnv(env_kwargs_list=[kwargs, kwargs], seed=42)
        ve_b = SerialVecEnv(env_kwargs_list=[kwargs, kwargs], seed=42)
        try:
            obs_a, _ = ve_a.reset_all(seeds=[0, 1])
            obs_b, _ = ve_b.reset_all(seeds=[0, 1])
            for k in obs_a:
                np.testing.assert_array_equal(obs_a[k], obs_b[k])
        finally:
            ve_a.close()
            ve_b.close()


class TestMaskSpecGlobalRNGSafety:
    def test_mask_spec_restores_global_prng(self) -> None:
        # Reviewer fix: mask_spec_from_env used to silently mutate
        # np.random + stdlib random globals via env.reset(seed=0).
        from catan_rl.env.catan_env import CatanEnv

        np.random.seed(123)
        random.seed(123)
        np_state_before = np.random.get_state()
        rand_state_before = random.getstate()

        env = CatanEnv(opponent_type="random", max_turns=50)
        mask_spec_from_env(env)

        # Globals must be byte-identical after.
        assert np.random.get_state()[1].tolist() == np_state_before[1].tolist()
        assert random.getstate() == rand_state_before
        env.close()


class TestClose:
    def test_close_is_idempotent(self) -> None:
        ve = SerialVecEnv(env_kwargs_list=[{}])
        ve.close()
        ve.close()  # no exception

    def test_step_after_close_raises(self) -> None:
        ve = SerialVecEnv(env_kwargs_list=[{}])
        ve.reset_all(seeds=[0])
        ve.close()
        with pytest.raises(RuntimeError, match="closed"):
            ve.step_all(np.zeros((1, 6), dtype=np.int64))
        with pytest.raises(RuntimeError, match="closed"):
            ve.reset_all(seeds=[0])


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


class TestSpecs:
    def test_obs_spec_covers_every_key(self) -> None:
        from catan_rl.env.catan_env import CatanEnv

        env = CatanEnv(opponent_type="random", max_turns=50)
        spec = obs_spec_from_env(env)
        # Every key in observation_space should be present.
        for key in env.observation_space.spaces:
            assert key in spec, f"missing key {key!r}"
        env.close()

    def test_obs_spec_shapes_match_env(self) -> None:
        from catan_rl.env.catan_env import CatanEnv

        env = CatanEnv(opponent_type="random", max_turns=50)
        spec = obs_spec_from_env(env)
        obs, _ = env.reset(seed=0)
        for k, arr in obs.items():
            assert spec[k].shape == arr.shape, (
                f"key {k!r}: spec shape {spec[k].shape} != obs shape {arr.shape}"
            )
        env.close()

    def test_mask_spec_matches_get_action_masks(self) -> None:
        from catan_rl.env.catan_env import CatanEnv

        env = CatanEnv(opponent_type="random", max_turns=50)
        spec = mask_spec_from_env(env)
        masks = env.get_action_masks()
        for k, arr in masks.items():
            assert spec[k].shape == arr.shape, (
                f"mask {k!r}: spec shape {spec[k].shape} != mask shape {arr.shape}"
            )
        env.close()
