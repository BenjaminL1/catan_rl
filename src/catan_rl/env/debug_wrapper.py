"""
Debug wrapper that catches NaN observations, all-zero masks, and crashes.

Wrap the CatanEnv during development to get early warnings of bugs.
Don't use in production training (adds overhead).
"""

import gymnasium as gym
import numpy as np
import traceback
from typing import Any, Dict, Tuple


class CatanCrashDebugWrapper(gym.Wrapper):
    """Catches NaN obs, all-zero masks, and exceptions in the env."""

    def __init__(self, env: gym.Env, log_file: str = "env_debug.log"):
        super().__init__(env)
        self.log_file = log_file
        self.step_count = 0

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._check_obs(obs, f"step {self.step_count}")
            if not np.isfinite(reward):
                self._report(f"Non-finite reward {reward} at step {self.step_count}")
            return obs, reward, terminated, truncated, info
        except Exception as e:
            self._report(f"CRASH at step {self.step_count}: {e}\n{traceback.format_exc()}")
            raise

    def reset(self, **kwargs):
        self.step_count = 0
        try:
            obs, info = self.env.reset(**kwargs)
            self._check_obs(obs, "reset")
            return obs, info
        except Exception as e:
            self._report(f"CRASH in reset: {e}\n{traceback.format_exc()}")
            raise

    def get_action_masks(self):
        masks = self.env.get_action_masks()
        if not masks["type"].any():
            self._report(f"All-zero type mask at step {self.step_count}")
        return masks

    def _check_obs(self, obs, context):
        if not np.isfinite(obs).all():
            nan_idx = np.where(~np.isfinite(obs))[0]
            self._report(f"Non-finite obs at {context}, indices: {nan_idx[:10]}")

    def _report(self, msg):
        print(f"[CatanDebug] {msg}")
        try:
            with open(self.log_file, "a") as f:
                f.write(msg + "\n")
        except IOError:
            pass
