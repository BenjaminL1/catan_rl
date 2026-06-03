"""Composite rollout buffer for PPO over the 1v1 Catan env.

Stores ``(T, N, *)`` numpy arrays for every quantity the PPO update
needs:

  * **obs** — dict of per-key arrays, one per :class:`ObsEncoder` output.
  * **actions** — ``(T, N, 6)`` integer composite action (the 6
    autoregressive heads).
  * **per_head_log_probs** — ``(T, N, 6)`` float; needed for the PPO
    ratio so we can re-compute ``ratio = exp(new_logp - old_logp)`` per
    head with relevance weighting. The joint ``log_probs`` is the
    relevance-weighted sum but stored alongside for convenience.
  * **values** — ``(T, N)`` ``V(s_t)``.
  * **rewards** — ``(T, N)``.
  * **terminated** / **truncated** — ``(T, N)`` bool; stored separately
    so :mod:`catan_rl.ppo.gae` can apply the correct bootstrap.
  * **masks** — dict of per-head action masks, one ``(T, N, K_i)``
    bool array per key.
  * (optional) **belief_target**, **opp_action_target**,
    **opp_action_target_valid**, **opp_kind**, **opp_policy_id** —
    auxiliary signals enabled by the trainer when the relevant aux
    heads are turned on. The buffer pre-allocates iff the constructor
    receives a non-None spec for the corresponding field.

Storage strategy: pre-allocate numpy arrays at construction time, write
in place via :meth:`add`, finalise into torch tensors on demand via
:meth:`get_batch`. This keeps the rollout loop allocation-free (the
audit identified Python orchestration as the bottleneck — avoiding
allocator pressure is real win on M1).

Lifecycle::

    buffer = CompositeRolloutBuffer(...)
    for _ in range(n_rollouts):
        buffer.reset()
        for step in range(n_steps):
            buffer.add(obs=..., action=..., ...)
        # one extra forward pass on the post-rollout obs:
        last_v, last_term = ...
        buffer.compute_returns_and_advantages(
            last_values=last_v, last_terminated=last_term,
            gamma=cfg.gae.gamma, gae_lambda=cfg.gae.gae_lambda,
            advantage_norm=cfg.ppo.advantage_norm,
        )
        for batch_indices in buffer.minibatch_indices(batch_size, rng):
            batch = buffer.get_batch(batch_indices, device=device)
            # ... SGD step on batch ...

Pinned by ``tests/unit/ppo/test_buffer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from catan_rl.ppo.gae import compute_gae, normalize_advantages

N_HEADS = 6
"""Number of autoregressive action heads on :class:`CatanPolicy`."""


@dataclass(frozen=True)
class ObsSpec:
    """Static shape + dtype description for a single obs-dict key.

    Constructed once at trainer startup from the env's
    ``observation_space``; the buffer uses these to pre-allocate.
    """

    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class MaskSpec:
    """Static shape description for a single action-mask key. Always
    boolean."""

    shape: tuple[int, ...]


class CompositeRolloutBuffer:
    """Pre-allocated rollout storage for the 1v1 Catan PPO loop.

    Construction is shape-checked once; per-step ``add`` calls only
    validate the step index. The buffer is mutated in place across the
    rollout and reset between rollouts via :meth:`reset`.
    """

    def __init__(
        self,
        *,
        n_steps: int,
        n_envs: int,
        obs_spec: dict[str, ObsSpec],
        mask_spec: dict[str, MaskSpec],
        belief_target_dim: int | None = None,
        opp_id_enabled: bool = False,
        opp_action_target_enabled: bool = False,
    ) -> None:
        if n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {n_steps}")
        if n_envs <= 0:
            raise ValueError(f"n_envs must be > 0, got {n_envs}")
        if not obs_spec:
            raise ValueError("obs_spec must be non-empty")
        if not mask_spec:
            raise ValueError("mask_spec must be non-empty")

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_spec = dict(obs_spec)
        self.mask_spec = dict(mask_spec)
        self.belief_target_dim = belief_target_dim
        self.opp_id_enabled = opp_id_enabled
        self.opp_action_target_enabled = opp_action_target_enabled

        TN = (n_steps, n_envs)

        # --- obs --------------------------------------------------------
        self.obs: dict[str, np.ndarray] = {
            k: np.zeros((*TN, *s.shape), dtype=s.dtype) for k, s in obs_spec.items()
        }

        # --- actions + per-head log_probs ------------------------------
        self.actions: np.ndarray = np.zeros((*TN, N_HEADS), dtype=np.int64)
        self.per_head_log_probs: np.ndarray = np.zeros((*TN, N_HEADS), dtype=np.float32)
        # Joint log_prob is the relevance-weighted sum of per-head
        # log_probs at sample time. Storing it alongside lets the PPO
        # ratio computation avoid re-deriving the relevance mask at
        # batch time.
        self.log_probs: np.ndarray = np.zeros(TN, dtype=np.float32)

        # --- values, rewards, dones ------------------------------------
        self.values: np.ndarray = np.zeros(TN, dtype=np.float32)
        self.rewards: np.ndarray = np.zeros(TN, dtype=np.float32)
        self.terminated: np.ndarray = np.zeros(TN, dtype=bool)
        self.truncated: np.ndarray = np.zeros(TN, dtype=bool)

        # --- masks (one (T,N,K_i) per key) -----------------------------
        self.masks: dict[str, np.ndarray] = {
            k: np.zeros((*TN, *m.shape), dtype=bool) for k, m in mask_spec.items()
        }

        # --- optional aux -----------------------------------------------
        if belief_target_dim is not None:
            if belief_target_dim <= 0:
                raise ValueError(f"belief_target_dim must be > 0, got {belief_target_dim}")
            self.belief_target: np.ndarray | None = np.zeros(
                (*TN, belief_target_dim), dtype=np.float32
            )
        else:
            self.belief_target = None
        if opp_action_target_enabled:
            self.opp_action_target: np.ndarray | None = np.zeros(TN, dtype=np.int64)
            self.opp_action_target_valid: np.ndarray | None = np.zeros(TN, dtype=bool)
        else:
            self.opp_action_target = None
            self.opp_action_target_valid = None
        if opp_id_enabled:
            self.opp_kind: np.ndarray | None = np.zeros(TN, dtype=np.int64)
            self.opp_policy_id: np.ndarray | None = np.zeros(TN, dtype=np.int64)
        else:
            self.opp_kind = None
            self.opp_policy_id = None

        # --- post-rollout outputs (filled by compute_returns_and_advantages) ---
        self.advantages: np.ndarray | None = None
        self.advantages_raw: np.ndarray | None = None
        self.returns: np.ndarray | None = None
        self._pos: int = 0
        self._finalised: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Rewind to position 0 for a new rollout. Does NOT zero arrays —
        all writes are full overwrites via :meth:`add`."""
        self._pos = 0
        self.advantages = None
        self.returns = None
        self._finalised = False

    def add(
        self,
        *,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        per_head_log_prob: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        masks: dict[str, np.ndarray],
        belief_target: np.ndarray | None = None,
        opp_action_target: np.ndarray | None = None,
        opp_action_target_valid: np.ndarray | None = None,
        opp_kind: np.ndarray | None = None,
        opp_policy_id: np.ndarray | None = None,
    ) -> None:
        """Write one step's worth of transitions into the buffer.

        Shape contracts (each ``(N, *)``):

        * ``obs[k]`` matches ``obs_spec[k].shape``
        * ``action`` is ``(N, 6)``
        * ``per_head_log_prob`` is ``(N, 6)``
        * ``log_prob``, ``value``, ``reward`` are ``(N,)``
        * ``terminated``, ``truncated`` are ``(N,)`` bool
        * ``masks[k]`` matches ``mask_spec[k].shape``
        * (optional) ``belief_target`` is ``(N, belief_target_dim)``
        * (optional) ``opp_action_target`` is ``(N,)`` int64;
          ``opp_action_target_valid`` is ``(N,)`` bool
        * (optional) ``opp_kind``, ``opp_policy_id`` are ``(N,)`` int64
        """
        if self._finalised:
            raise RuntimeError(
                "Buffer was finalised via compute_returns_and_advantages; "
                "call reset() before adding more transitions."
            )
        if self._pos >= self.n_steps:
            raise RuntimeError(
                f"Buffer full: tried to add() at step {self._pos} but n_steps={self.n_steps}."
            )
        t = self._pos
        N = self.n_envs

        for k, arr in obs.items():
            if k not in self.obs:
                raise KeyError(f"unknown obs key {k!r}")
            self._check_shape(f"obs[{k}]", arr.shape, (N, *self.obs_spec[k].shape))
            self.obs[k][t] = arr

        self._check_shape("action", action.shape, (N, N_HEADS))
        self._check_shape("per_head_log_prob", per_head_log_prob.shape, (N, N_HEADS))
        for name, arr in (
            ("log_prob", log_prob),
            ("value", value),
            ("reward", reward),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            self._check_shape(name, arr.shape, (N,))

        self.actions[t] = action
        self.per_head_log_probs[t] = per_head_log_prob
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.terminated[t] = terminated
        self.truncated[t] = truncated

        for k, arr in masks.items():
            if k not in self.masks:
                raise KeyError(f"unknown mask key {k!r}")
            self._check_shape(f"mask[{k}]", arr.shape, (N, *self.mask_spec[k].shape))
            self.masks[k][t] = arr

        # Aux storage contract: when a slot is allocated, the caller
        # MUST pass the corresponding kwarg on every add(). A missing
        # kwarg would silently leave the slot at zero, then the aux
        # head would be trained against a degenerate target. Pin loud.
        if self.belief_target is not None:
            if belief_target is None:
                raise ValueError(
                    "belief_target_dim was set at construction; every add() "
                    "must pass belief_target. Silent all-zero supervision "
                    "would collapse the belief head."
                )
            self._check_shape(
                "belief_target", belief_target.shape, (N, self.belief_target_dim or 0)
            )
            self.belief_target[t] = belief_target
        elif belief_target is not None:
            raise RuntimeError(
                "belief_target passed but buffer was constructed with belief_target_dim=None"
            )

        if self.opp_action_target is not None:
            if opp_action_target is None:
                raise ValueError(
                    "opp_action_target_enabled=True at construction; every "
                    "add() must pass opp_action_target (use the "
                    "opp_action_target_valid mask to indicate degenerate rows)."
                )
            if opp_action_target_valid is None:
                raise ValueError(
                    "opp_action_target_valid must be supplied alongside opp_action_target"
                )
            self._check_shape("opp_action_target", opp_action_target.shape, (N,))
            self._check_shape("opp_action_target_valid", opp_action_target_valid.shape, (N,))
            assert self.opp_action_target_valid is not None  # mypy
            self.opp_action_target[t] = opp_action_target
            self.opp_action_target_valid[t] = opp_action_target_valid
        elif opp_action_target is not None:
            raise RuntimeError(
                "opp_action_target passed but buffer was constructed "
                "with opp_action_target_enabled=False"
            )

        if self.opp_kind is not None:
            if opp_kind is None:
                raise ValueError(
                    "opp_id_enabled=True at construction; every add() must "
                    "pass opp_kind and opp_policy_id."
                )
            if opp_policy_id is None:
                raise ValueError("opp_policy_id must be supplied alongside opp_kind")
            self._check_shape("opp_kind", opp_kind.shape, (N,))
            self._check_shape("opp_policy_id", opp_policy_id.shape, (N,))
            assert self.opp_policy_id is not None  # mypy
            self.opp_kind[t] = opp_kind
            self.opp_policy_id[t] = opp_policy_id
        elif opp_kind is not None:
            raise RuntimeError(
                "opp_kind passed but buffer was constructed with opp_id_enabled=False"
            )

        self._pos += 1

    # ------------------------------------------------------------------
    # GAE / advantage normalization
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self,
        *,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
        advantage_norm: str = "rollout",
    ) -> None:
        """Run GAE over the full buffer and (optionally) normalize.

        Must be called after ``n_steps`` ``add()``s and before
        :meth:`get_batch`.

        Args:
            last_values: ``(N,)`` ``V(s_T+1)``. The caller computes via
                a fresh forward pass on the post-rollout obs. Whether
                the final in-buffer step was terminated is read from
                ``self.terminated[-1]`` directly — there's no separate
                ``last_terminated`` argument (it was removed in Phase 3
                review to avoid the "is the next obs terminal?" foot-gun).
            gamma, gae_lambda: GAE hyperparameters (typically from
                ``cfg.gae``).
            advantage_norm: ``"rollout"`` (standardise over the full
                buffer), ``"batch"`` (no-op here; the trainer normalises
                per batch), or ``"none"`` (no normalization).

        Side effects: writes ``self.advantages`` (possibly normalised),
        ``self.advantages_raw`` (always unnormalised — kept so a trainer
        can re-normalise per batch later or log raw-advantage stats),
        and ``self.returns`` (always unnormalised; this is the value
        target).
        """
        if self._pos < self.n_steps:
            raise RuntimeError(f"Buffer not full: only {self._pos}/{self.n_steps} steps added")
        if advantage_norm not in ("rollout", "batch", "none"):
            raise ValueError(
                f"advantage_norm must be one of rollout/batch/none, got {advantage_norm!r}"
            )

        adv_raw, ret = compute_gae(
            rewards=self.rewards,
            values=self.values,
            terminated=self.terminated,
            truncated=self.truncated,
            last_values=last_values.astype(np.float32),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        self.advantages_raw = adv_raw.copy()
        self.advantages = normalize_advantages(adv_raw) if advantage_norm == "rollout" else adv_raw
        self.returns = ret
        self._finalised = True

    # ------------------------------------------------------------------
    # Minibatch sampling
    # ------------------------------------------------------------------

    def minibatch_indices(self, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Return a list of flat-index batches covering the full buffer
        with random order, no overlap, no leftover."""
        if not self._finalised:
            raise RuntimeError(
                "Buffer not finalised — call compute_returns_and_advantages "
                "before minibatch_indices."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        total = self.n_steps * self.n_envs
        if total % batch_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) does not divide n_steps*n_envs "
                f"({total}). Phase 1 config validator should have caught this."
            )
        perm = rng.permutation(total)
        n_batches = total // batch_size
        return [perm[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]

    def get_batch(self, flat_indices: np.ndarray, *, device: torch.device | str) -> dict[str, Any]:
        """Return a flat-batch dict for the given indices, with all
        tensors moved to ``device``.

        Index space is ``[0, n_steps * n_envs)`` flattened row-major.
        """
        if not self._finalised:
            raise RuntimeError("Buffer not finalised — call compute_returns_and_advantages first.")
        # ``_finalised`` implies these were populated; assert so the type
        # checker can narrow.
        assert self.advantages is not None
        assert self.returns is not None
        flat_indices = np.asarray(flat_indices, dtype=np.int64)
        if flat_indices.ndim != 1:
            raise ValueError(f"flat_indices must be 1D, got {flat_indices.shape}")
        if (flat_indices < 0).any() or (flat_indices >= self.n_steps * self.n_envs).any():
            raise IndexError("flat_indices out of range [0, n_steps*n_envs)")

        t = (flat_indices // self.n_envs).astype(np.int64)
        n = (flat_indices % self.n_envs).astype(np.int64)

        def _gather(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr[t, n], device=device)

        obs = {k: _gather(v) for k, v in self.obs.items()}
        masks = {k: _gather(v) for k, v in self.masks.items()}

        out: dict[str, Any] = {
            "obs": obs,
            "masks": masks,
            "actions": _gather(self.actions),
            "log_probs": _gather(self.log_probs),
            "per_head_log_probs": _gather(self.per_head_log_probs),
            "values": _gather(self.values),
            "rewards": _gather(self.rewards),
            "terminated": _gather(self.terminated),
            "truncated": _gather(self.truncated),
            "advantages": _gather(self.advantages),
            "returns": _gather(self.returns),
        }
        if self.belief_target is not None:
            out["belief_target"] = _gather(self.belief_target)
        if self.opp_action_target is not None:
            out["opp_action_target"] = _gather(self.opp_action_target)
            assert self.opp_action_target_valid is not None
            out["opp_action_target_valid"] = _gather(self.opp_action_target_valid)
        if self.opp_kind is not None:
            out["opp_kind"] = _gather(self.opp_kind)
            assert self.opp_policy_id is not None
            out["opp_policy_id"] = _gather(self.opp_policy_id)
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def total_transitions(self) -> int:
        return self.n_steps * self.n_envs

    @property
    def is_full(self) -> bool:
        return self._pos >= self.n_steps

    @property
    def is_finalised(self) -> bool:
        return self._finalised

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_shape(name: str, got: tuple[int, ...], expected: tuple[int, ...]) -> None:
        if got != expected:
            raise ValueError(f"{name} shape mismatch: got {got}, expected {expected}")
