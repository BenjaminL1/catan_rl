"""
Custom rollout buffer for composite (multi-head) actions and dict observations.

Observations are now Charlesworth-style dicts. We store:
  - tile_representations: (n_steps, 19, 78)
  - current_player_main: (n_steps, 166)
  - next_player_main: (n_steps, 173)
  - dev-card sequences as padded int arrays plus lengths.
"""

from collections.abc import Generator
from typing import Any

import numpy as np
import torch

from catan_rl.algorithms.common.gae import compute_gae, compute_gae_vectorized


class CompositeRolloutBuffer:
    """Stores transitions from rollouts and produces shuffled minibatches.

    During rollout collection:
      buffer.add(obs, action, reward, done, value, log_prob, masks)

    After rollout is full:
      buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)

    During PPO update:
      for batch in buffer.get_batches(batch_size):
          # batch is a dict of tensors, ready for policy.evaluate_actions()
    """

    def __init__(
        self,
        n_steps: int,
        n_action_heads: int = 6,
        mask_shapes: dict[str, int] = None,
        device: str = "cpu",
        n_envs: int = 1,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_action_heads = n_action_heads
        self.device = device
        self.pos = 0
        self.full = False

        if mask_shapes is None:
            mask_shapes = {
                "type": 13,
                "corner_settlement": 54,
                "corner_city": 54,
                "edge": 72,
                "tile": 19,
                "resource1_trade": 5,
                "resource1_discard": 5,
                "resource1_default": 5,
                "resource2_default": 5,
            }
        self.mask_shapes = mask_shapes

        # Pre-allocate storage for dict observations
        self.tile_representations = np.zeros((n_steps, 19, 79), dtype=np.float32)
        self.current_player_main = np.zeros((n_steps, 166), dtype=np.float32)
        self.next_player_main = np.zeros((n_steps, 173), dtype=np.float32)
        # Dev cards: padded sequences of ints + lengths
        self.curr_hidden_dev = np.zeros((n_steps, 16), dtype=np.int64)
        self.curr_hidden_dev_len = np.zeros(n_steps, dtype=np.int32)
        self.curr_played_dev = np.zeros((n_steps, 16), dtype=np.int64)
        self.curr_played_dev_len = np.zeros(n_steps, dtype=np.int32)
        self.next_played_dev = np.zeros((n_steps, 16), dtype=np.int64)
        self.next_played_dev_len = np.zeros(n_steps, dtype=np.int32)
        self.actions = np.zeros((n_steps, n_action_heads), dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

        self.masks = {}
        for key, size in mask_shapes.items():
            self.masks[key] = np.zeros((n_steps, size), dtype=bool)

        self._tensors_ready = False

    def reset(self) -> None:
        self.pos = 0
        self.full = False
        self._tensors_ready = False

    def _store_dev_seq(
        self, dest_arr: np.ndarray, dest_len: np.ndarray, idx: int, seq: Any
    ) -> None:
        if isinstance(seq, (list, tuple)):
            arr = np.array(seq, dtype=np.int64)
        else:
            arr = np.asarray(seq, dtype=np.int64)
        max_len = dest_arr.shape[1]
        L = min(len(arr), max_len)
        dest_arr[idx, :L] = arr[:L]
        dest_len[idx] = L

    def add(
        self,
        obs: dict[str, Any],
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        masks: dict[str, np.ndarray],
    ) -> None:
        """Store one transition."""
        # obs is a dict from env.step
        self.tile_representations[self.pos] = obs["tile_representations"]
        self.current_player_main[self.pos] = obs["current_player_main"]
        self.next_player_main[self.pos] = obs["next_player_main"]
        self._store_dev_seq(
            self.curr_hidden_dev,
            self.curr_hidden_dev_len,
            self.pos,
            obs.get("current_player_hidden_dev", []),
        )
        self._store_dev_seq(
            self.curr_played_dev,
            self.curr_played_dev_len,
            self.pos,
            obs.get("current_player_played_dev", []),
        )
        self._store_dev_seq(
            self.next_played_dev,
            self.next_played_dev_len,
            self.pos,
            obs.get("next_player_played_dev", []),
        )
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        for key in self.masks:
            self.masks[key][self.pos] = masks[key]

        self.pos += 1
        if self.pos >= self.n_steps:
            self.full = True

    def compute_returns_and_advantages(
        self, last_value, gamma: float = 0.995, gae_lambda: float = 0.95
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: For n_envs=1, a float. For n_envs>1, np.ndarray of
                shape (n_envs,) with bootstrap value per env.
        """
        n = self.n_steps if self.full else self.pos
        if self.n_envs == 1:
            lv = (
                float(last_value)
                if not isinstance(last_value, np.ndarray)
                else float(last_value[0])
            )
            self.advantages[:n], self.returns[:n] = compute_gae(
                self.rewards[:n], self.values[:n], self.dones[:n], lv, gamma, gae_lambda
            )
        else:
            last_values = np.asarray(last_value, dtype=np.float32)
            self.advantages[:n], self.returns[:n] = compute_gae_vectorized(
                self.rewards[:n],
                self.values[:n],
                self.dones[:n],
                last_values,
                self.n_envs,
                gamma,
                gae_lambda,
            )

    def finalize(self, device: str = "cpu") -> None:
        """Pre-convert numpy arrays to tensors once before the update loop.

        from_numpy is zero-copy on CPU (shares memory). Do not write to
        numpy arrays after finalize() and before the next reset().
        """
        n = self.n_steps if self.full else self.pos
        self._t_tile_reps = torch.from_numpy(self.tile_representations[:n]).to(device)
        self._t_curr_main = torch.from_numpy(self.current_player_main[:n]).to(device)
        self._t_next_main = torch.from_numpy(self.next_player_main[:n]).to(device)
        self._t_actions = torch.from_numpy(self.actions[:n]).to(device)
        self._t_values = torch.from_numpy(self.values[:n]).to(device)
        self._t_log_probs = torch.from_numpy(self.log_probs[:n]).to(device)
        self._t_advantages = torch.from_numpy(self.advantages[:n]).to(device)
        self._t_returns = torch.from_numpy(self.returns[:n]).to(device)
        self._t_curr_hidden = torch.from_numpy(self.curr_hidden_dev[:n]).to(device)
        self._t_curr_played = torch.from_numpy(self.curr_played_dev[:n]).to(device)
        self._t_next_played = torch.from_numpy(self.next_played_dev[:n]).to(device)
        self._t_masks = {k: torch.from_numpy(self.masks[k][:n]).to(device) for k in self.masks}
        self._tensors_ready = True

    def get_batches(self, batch_size: int) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield shuffled minibatches as dicts of tensors on self.device.

        Each yielded dict contains:
          'obs':        dict of batched obs tensors
          'actions':    (B, 6)
          'old_values': (B,)
          'old_log_prob': (B,)
          'advantages': (B,)
          'returns':    (B,)
          'masks':      {key: (B, mask_dim)}   per-head bool masks
        """
        n = self.n_steps if self.full else self.pos
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            idx_t = torch.from_numpy(batch_idx)

            obs_batch = {
                "tile_representations": self._t_tile_reps[idx_t],
                "current_player_main": self._t_curr_main[idx_t],
                "next_player_main": self._t_next_main[idx_t],
                "current_player_hidden_dev": self._t_curr_hidden[idx_t],
                "current_player_played_dev": self._t_curr_played[idx_t],
                "next_player_played_dev": self._t_next_played[idx_t],
            }

            yield {
                "obs": obs_batch,
                "actions": self._t_actions[idx_t],
                "old_values": self._t_values[idx_t],
                "old_log_prob": self._t_log_probs[idx_t],
                "advantages": self._t_advantages[idx_t],
                "returns": self._t_returns[idx_t],
                "masks": {k: self._t_masks[k][idx_t] for k in self._t_masks},
            }
