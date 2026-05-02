"""
Manages parallel environment instances for experience collection.

Runs N environments sequentially (no subprocess overhead on macOS).
Supports league-based policy opponents for Charlesworth-style self-play.
"""

import copy
from collections.abc import Callable

import numpy as np

from catan_rl.env.catan_env import CatanEnv


class GameManager:
    """Manages multiple CatanEnv instances for vectorized rollout collection.

    On M1 Mac, subprocesses have high overhead due to "spawn" (not "fork").
    So we run environments sequentially in the same process. This is simpler
    and actually faster for small numbers of environments.

    When league is provided: samples opponent from league at each reset.
    Maintains one policy instance per env for loading different state dicts.
    """

    def __init__(
        self,
        n_envs: int = 1,
        opponent_type: str = "random",  # fallback when league empty; else uses league
        max_turns: int | None = 500,
        league=None,
        build_policy_fn: Callable | None = None,
        device: str = "cpu",
    ):
        self.n_envs = n_envs
        self.league = league
        self._build_policy_fn = build_policy_fn
        self.device = device
        # Stable policy ID of the current rollout opponent (-1 = random / no league)
        self._rollout_opp_policy_id: int = -1

        # Use policy opponents when league is provided and has build fn
        use_league = league is not None and build_policy_fn is not None and len(league) > 0
        opp_type = "policy" if use_league else opponent_type

        self.envs = [
            CatanEnv(render_mode=None, opponent_type=opp_type, max_turns=max_turns)
            for _ in range(n_envs)
        ]

        # ONE shared opponent policy for all envs in a rollout (enables batched inference)
        if use_league:
            league.set_build_policy_fn(build_policy_fn)
            self.rollout_opp_policy = build_policy_fn(device=device)
            self.rollout_opp_policy.eval()
        else:
            self.rollout_opp_policy = None

    def _sample_and_prepare_opponent(self, env_idx: int = 0) -> dict:
        """Sample from league and load into the shared rollout opponent policy.

        All envs share one opponent policy per rollout for batched inference.
        Stores the sampled policy's ID in _rollout_opp_policy_id for league tracking.
        env_idx is accepted for API compatibility but ignored (shared policy).
        """
        if self.league is None or len(self.league) == 0:
            self._rollout_opp_policy_id = -1
            return {"opponent_type": "random"}
        opp_type, state_dict, policy_id = self.league.sample()
        self._rollout_opp_policy_id = policy_id
        if opp_type == "random":
            return {"opponent_type": "random"}
        if opp_type == "heuristic":
            return {"opponent_type": "heuristic"}
        self.rollout_opp_policy.load_state_dict(copy.deepcopy(state_dict))
        self.rollout_opp_policy.eval()
        return {"opponent_type": "policy"}  # env uses deferred mode; no policy object needed

    def reset_all(self) -> tuple[list[dict], list[dict]]:
        """Reset all environments and return (observations, infos)."""
        observations, infos = [], []
        for i, env in enumerate(self.envs):
            options = self._sample_and_prepare_opponent(i) if self.league else {}
            obs, info = env.reset(options=options)
            observations.append(obs)
            infos.append(info)
        return observations, infos

    def step_one(self, env_idx: int, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Step a single environment (for round-robin collection).

        Args:
            env_idx: Index of the environment to step.
            action: (6,) action array.

        Returns:
            obs: Next observation (from reset if episode ended).
            reward: Reward from this step.
            terminated: True iff the episode genuinely ended (someone won).
            truncated: True iff the episode was cut by max_turns.
            info: Info dict (terminal_stats, is_success when terminated).

        When the opponent NN turn is deferred (``info['opp_turn_pending']``),
        terminated and truncated are both False and the caller
        (``collect_rollouts``) is expected to drive the deferred turn via
        ``apply_opponent_action`` until the transition completes.
        """
        env = self.envs[env_idx]
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("opp_turn_pending"):
            # Opponent NN turn deferred — caller (collect_rollouts) handles batching
            return obs, reward, False, False, info
        done = terminated or truncated
        if done:
            info["terminal_observation"] = obs
            # Report result to league for PFSP win-rate tracking
            if self.league is not None:
                win = 1 if info.get("is_success") else 0
                self.league.update_result(self._rollout_opp_policy_id, win)
            options = self._sample_and_prepare_opponent() if self.league else {}
            obs, _ = env.reset(options=options)
        return obs, reward, terminated, truncated, info

    def step_all(self, actions: list[np.ndarray]):
        """Step all environments with the given actions.

        Args:
            actions: list of (6,) action arrays, one per env.

        Returns:
            observations: list of dict observations.
            rewards: list of floats.
            dones: list of bools (terminated | truncated).
            infos: list of info dicts.
        """
        observations, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                info["terminal_observation"] = obs
                terminal_info = info
                if self.league is not None:
                    win = 1 if info.get("is_success") else 0
                    self.league.update_result(self._opponent_policy_ids[i], win)
                options = self._sample_and_prepare_opponent(i) if self.league else {}
                obs, _ = env.reset(options=options)
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(terminal_info)
            else:
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
        return observations, rewards, dones, infos

    def get_opponent_obs_masks(self, env_idx: int):
        """Get opponent (obs, masks) for an env with a deferred NN turn."""
        return self.envs[env_idx].get_opponent_obs_masks()

    def apply_opponent_action(
        self, env_idx: int, action: np.ndarray
    ) -> tuple[bool, dict | None, float, bool, bool, dict]:
        """Apply one opponent NN action.

        Returns ``(turn_complete, obs, reward, terminated, truncated, info)``.
        When ``turn_complete`` is True and the episode ended, also resets
        the env and resamples its opponent.
        """
        env = self.envs[env_idx]
        turn_complete, obs, reward, terminated, truncated, info = env.apply_opponent_action(action)
        if not turn_complete:
            return False, None, 0.0, False, False, {}
        done = terminated or truncated
        if done:
            info["terminal_observation"] = obs
            if self.league is not None:
                win = 1 if info.get("is_success") else 0
                self.league.update_result(self._rollout_opp_policy_id, win)
            options = self._sample_and_prepare_opponent() if self.league else {}
            obs, _ = env.reset(options=options)
        return True, obs, reward, terminated, truncated, info

    def get_masks(self) -> list[dict[str, np.ndarray]]:
        """Get action masks from all environments."""
        return [env.get_action_masks() for env in self.envs]

    def set_opponent_type(self, opponent_type: str) -> None:
        """Change the opponent type for all environments."""
        for env in self.envs:
            env.opponent_type = opponent_type
