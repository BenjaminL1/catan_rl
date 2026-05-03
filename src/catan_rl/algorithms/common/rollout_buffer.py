"""
Custom rollout buffer for composite (multi-head) actions and dict observations.

Observations are Charlesworth-style dicts. We store:
  - tile_representations: (n_steps, N_TILES, OBS_TILE_DIM)
  - current_player_main: (n_steps, CURR_PLAYER_DIM)
  - next_player_main: (n_steps, NEXT_PLAYER_DIM)
  - dev-card sequences as padded int arrays plus lengths.

Phase 0 splits the legacy single ``done`` flag into separate ``terminated`` and
``truncated`` arrays so GAE can correctly bootstrap on truncation rather than
zeroing the value (Schulman 2015 §3.2; the bug was conflating the two).
"""

from collections.abc import Generator
from typing import Any

import numpy as np
import torch

from catan_rl.algorithms.common.gae import compute_gae, compute_gae_vectorized
from catan_rl.models.utils import (
    CURR_PLAYER_DIM,
    MAX_DEV_SEQ,
    N_TILES,
    NEXT_PLAYER_DIM,
    OBS_TILE_DIM,
)

# Internal padded-sequence width: MAX_DEV_SEQ rounded up to a multiple of 8 for
# even tensor alignment. Anything past length-N is zero-padding regardless of
# buffer width.
_DEV_SEQ_BUFFER_WIDTH = max(MAX_DEV_SEQ, 16)


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
        curr_player_dim: int = CURR_PLAYER_DIM,
        next_player_dim: int = NEXT_PLAYER_DIM,
        store_opponent_id: bool = False,
        store_belief_target: bool = False,
    ):
        """Allocate fixed-size NumPy storage for one rollout's transitions.

        Args:
            curr_player_dim: dimension of ``current_player_main``. Defaults to
                the legacy 166 (Phase 0). With Phase 1.3 compact encoding pass
                ``CURR_PLAYER_DIM_COMPACT`` (54).
            next_player_dim: dimension of ``next_player_main``. Defaults to
                the legacy 173. With compact encoding pass 61.
        """
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.n_action_heads = n_action_heads
        self.device = device
        self.curr_player_dim = int(curr_player_dim)
        self.next_player_dim = int(next_player_dim)
        # Phase 3.6: opponent identity is stored per-step only when the
        # encoder will read it; off by default to keep the buffer footprint
        # unchanged for legacy lineages.
        self.store_opponent_id = bool(store_opponent_id)
        # Phase 2.5b: belief-head supervision target. Same opt-in pattern
        # as opponent_id; off by default.
        self.store_belief_target = bool(store_belief_target)
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

        # Pre-allocate storage for dict observations. Player-feature dims are
        # encoding-dependent (Phase 1.3); tile dim is fixed at OBS_TILE_DIM.
        self.tile_representations = np.zeros((n_steps, N_TILES, OBS_TILE_DIM), dtype=np.float32)
        self.current_player_main = np.zeros((n_steps, self.curr_player_dim), dtype=np.float32)
        self.next_player_main = np.zeros((n_steps, self.next_player_dim), dtype=np.float32)
        # Dev cards: padded sequences of ints + lengths.
        self.curr_hidden_dev = np.zeros((n_steps, _DEV_SEQ_BUFFER_WIDTH), dtype=np.int64)
        self.curr_hidden_dev_len = np.zeros(n_steps, dtype=np.int32)
        self.curr_played_dev = np.zeros((n_steps, _DEV_SEQ_BUFFER_WIDTH), dtype=np.int64)
        self.curr_played_dev_len = np.zeros(n_steps, dtype=np.int32)
        self.next_played_dev = np.zeros((n_steps, _DEV_SEQ_BUFFER_WIDTH), dtype=np.int64)
        self.next_played_dev_len = np.zeros(n_steps, dtype=np.int32)
        self.actions = np.zeros((n_steps, n_action_heads), dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        # Phase 0: separate terminated (real game-over) from truncated (max_turns).
        # GAE handles each correctly: terminated zeros the bootstrap; truncated
        # keeps it but resets the GAE accumulator at the boundary.
        self.terminated = np.zeros(n_steps, dtype=np.float32)
        self.truncated = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

        # Phase 3.6: lazily allocated when the schema is opted in.
        if self.store_opponent_id:
            self.opponent_kind = np.zeros(n_steps, dtype=np.int64)
            self.opponent_policy_id = np.zeros(n_steps, dtype=np.int64)
        else:
            self.opponent_kind = None
            self.opponent_policy_id = None
        # Phase 2.5b: belief target — float vector per step, shape (n, 5).
        if self.store_belief_target:
            self.belief_target = np.zeros((n_steps, 5), dtype=np.float32)
        else:
            self.belief_target = None

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
        terminated: bool,
        truncated: bool,
        value: float,
        log_prob: float,
        masks: dict[str, np.ndarray],
    ) -> None:
        """Store one transition.

        Args:
            obs: dict from env.step.
            action: (n_action_heads,) int64.
            reward: scalar reward for this step.
            terminated: True if the episode genuinely ended (someone hit 15 VP).
            truncated: True if the episode was cut short by max_turns.
                A step is at most one of {terminated, truncated}; both False
                means the trajectory continues.
            value: bootstrap value V(s_t) used for advantage computation.
            log_prob: log-probability of ``action`` under the behavior policy.
            masks: per-head action mask dict that was active at this step.
        """
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
        self.terminated[self.pos] = float(terminated)
        self.truncated[self.pos] = float(truncated)
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        if self.store_opponent_id:
            self.opponent_kind[self.pos] = int(obs.get("opponent_kind", 0))
            self.opponent_policy_id[self.pos] = int(obs.get("opponent_policy_id", 0))
        if self.store_belief_target:
            self.belief_target[self.pos] = obs.get(
                "belief_target", np.full(5, 0.2, dtype=np.float32)
            )
        for key in self.masks:
            self.masks[key][self.pos] = masks[key]

        self.pos += 1
        if self.pos >= self.n_steps:
            self.full = True

    @property
    def dones(self) -> np.ndarray:
        """Back-compat OR-mask for code that still reads a single ``done`` flag.

        Phase 0 callers should consume ``terminated`` / ``truncated`` directly.
        """
        return np.maximum(self.terminated, self.truncated)

    def compute_returns_and_advantages(
        self,
        last_value,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        *,
        advantage_norm: str = "rollout",
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: For n_envs=1, a float. For n_envs>1, np.ndarray of
                shape (n_envs,) with bootstrap value per env.
            gamma: Discount factor.
            gae_lambda: GAE λ parameter.
            advantage_norm: When 'rollout' (default), normalize advantages
                globally over the buffer here so per-batch standardization in
                the trainer becomes a cheap no-op. When 'batch' or 'none',
                advantages are passed through raw and the trainer handles
                whatever normalization it wants. See Phase 1.2 in the roadmap.
        """
        n = self.n_steps if self.full else self.pos
        if self.n_envs == 1:
            lv = (
                float(last_value)
                if not isinstance(last_value, np.ndarray)
                else float(last_value[0])
            )
            self.advantages[:n], self.returns[:n] = compute_gae(
                self.rewards[:n],
                self.values[:n],
                self.terminated[:n],
                self.truncated[:n],
                lv,
                gamma,
                gae_lambda,
            )
        else:
            last_values = np.asarray(last_value, dtype=np.float32)
            self.advantages[:n], self.returns[:n] = compute_gae_vectorized(
                self.rewards[:n],
                self.values[:n],
                self.terminated[:n],
                self.truncated[:n],
                last_values,
                self.n_envs,
                gamma,
                gae_lambda,
            )

        # Phase 1.2: optional global advantage normalization. Keeps per-batch
        # standardization in the trainer a no-op for 'rollout' mode while
        # preserving 'batch' / 'none' as drop-in alternatives for ablation.
        if advantage_norm == "rollout" and n > 1:
            adv = self.advantages[:n]
            mean = float(adv.mean())
            std = float(adv.std())
            self.advantages[:n] = (adv - mean) / (std + 1e-8)

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
        if self.store_opponent_id:
            self._t_opp_kind = torch.from_numpy(self.opponent_kind[:n]).to(device)
            self._t_opp_policy_id = torch.from_numpy(self.opponent_policy_id[:n]).to(device)
        else:
            self._t_opp_kind = None
            self._t_opp_policy_id = None
        if self.store_belief_target:
            self._t_belief_target = torch.from_numpy(self.belief_target[:n]).to(device)
        else:
            self._t_belief_target = None
        self._tensors_ready = True

    def get_batches(
        self,
        batch_size: int,
        *,
        symmetry_aug_prob: float = 0.0,
        symmetry_rng: np.random.Generator | None = None,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield shuffled minibatches as dicts of tensors on self.device.

        Each yielded dict contains:
          'obs':        dict of batched obs tensors
          'actions':    (B, 6)
          'old_values': (B,)
          'old_log_prob': (B,)
          'advantages': (B,)
          'returns':    (B,)
          'masks':      {key: (B, mask_dim)}   per-head bool masks

        Phase 1.5: when ``symmetry_aug_prob > 0``, each minibatch is, with
        that probability, transformed by a single non-identity D6 element
        (uniform over {1..11}). The same element applies to the entire
        minibatch — mixing multiple symmetries within one batch breaks the
        equivariance assumption, so we deliberately do not.
        """
        n = self.n_steps if self.full else self.pos
        indices = np.random.permutation(n)
        rng = symmetry_rng if symmetry_rng is not None else np.random.default_rng()

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
            if self.store_opponent_id:
                obs_batch["opponent_kind"] = self._t_opp_kind[idx_t]
                obs_batch["opponent_policy_id"] = self._t_opp_policy_id[idx_t]
            if self.store_belief_target:
                obs_batch["belief_target"] = self._t_belief_target[idx_t]
            actions = self._t_actions[idx_t]
            masks = {k: self._t_masks[k][idx_t] for k in self._t_masks}

            # Phase 1.5: symmetry augmentation.
            if symmetry_aug_prob > 0.0 and rng.random() < symmetry_aug_prob:
                # Lazy import so the augmentation package isn't pulled in by
                # rollout-only code paths or by environments that disable it.
                from catan_rl.augmentation import apply_symmetry, sample_d6_element

                g = sample_d6_element(rng, exclude_identity=True)
                obs_batch, actions, masks = apply_symmetry(obs_batch, actions, masks, g)

            yield {
                "obs": obs_batch,
                "actions": actions,
                "old_values": self._t_values[idx_t],
                "old_log_prob": self._t_log_probs[idx_t],
                "advantages": self._t_advantages[idx_t],
                "returns": self._t_returns[idx_t],
                "masks": masks,
            }
