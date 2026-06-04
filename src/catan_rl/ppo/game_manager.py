"""Rollout collector that fills a :class:`CompositeRolloutBuffer`.

Phase 5 of the v2 training-infra build-out. Glues together:

* :class:`catan_rl.ppo.vec_env.SerialVecEnv` — N envs in lockstep.
* A policy with a ``sample(obs, masks) -> dict`` method (see
  :class:`catan_rl.policy.network.CatanPolicy`).
* :class:`catan_rl.ppo.buffer.CompositeRolloutBuffer` — pre-allocated
  storage for the rollout.

The collector is intentionally synchronous and single-threaded — the
audit's measurements showed Python orchestration dominates rollout
wall-time at the default ``n_envs=128``, but parallelism inside the
collector won't help until the env loop itself is rewritten in a
faster language. For now we ship correctness; Phase 5b can add a
subproc vec env.

Usage::

    collector = RolloutCollector(vec_env=ve, policy=p, buffer=b, device="mps")
    obs, masks = ve.reset_all(seeds=[seed + i for i in range(n_envs)])
    for update_idx in range(n_updates):
        obs, masks = collector.collect(obs, masks)  # mutates buffer in place
        b.compute_returns_and_advantages(
            last_values=collector.last_values, gamma=..., gae_lambda=..., ...
        )
        trainer.update(b, update_idx=update_idx, rng=rng)
        b.reset()
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from catan_rl.ppo.buffer import CompositeRolloutBuffer
from catan_rl.ppo.vec_env import SerialVecEnv


class RolloutCollector:
    """Synchronous rollout collector for one PPO update's worth of data.

    Lifecycle:

    1. Caller seeds the vec env once with :meth:`SerialVecEnv.reset_all`.
    2. Calls :meth:`collect` with the current ``(obs, masks)`` carried
       from the previous rollout's auto-reset trailing state. The
       collector fills ``self.buffer`` and stashes ``self.last_values``
       (the V(s_T+1) needed for GAE) — see
       :meth:`CompositeRolloutBuffer.compute_returns_and_advantages`.
    3. Returns the new ``(obs, masks)`` for the next rollout to pick up
       from. This is the canonical CleanRL/SB3 pattern — the env's
       "ongoing" state lives outside the buffer.

    Belief / opp-action storage hookup is deferred to the
    point where Phase 8/9 wires those heads; the collector is
    structured so adding them is a one-line ``buffer.add(...)`` kwarg
    threading.
    """

    def __init__(
        self,
        *,
        vec_env: SerialVecEnv,
        policy: Any,
        buffer: CompositeRolloutBuffer,
        device: torch.device | str,
    ) -> None:
        if vec_env.n_envs != buffer.n_envs:
            raise ValueError(
                f"vec_env.n_envs ({vec_env.n_envs}) != buffer.n_envs ({buffer.n_envs})"
            )
        self.vec_env = vec_env
        self.policy = policy
        self.buffer = buffer
        self.device = torch.device(device) if isinstance(device, str) else device

        # Post-rollout V(s_T+1) for GAE bootstrap. Populated by
        # :meth:`collect`. Always float32 (N,). The buffer's GAE step
        # reads ``terminated[T-1]`` directly to decide whether to zero
        # the bootstrap, so we don't store ``last_terminated`` here.
        self.last_values: np.ndarray = np.zeros(self.vec_env.n_envs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(
        self,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Fill the buffer with ``self.buffer.n_steps`` rollout steps.

        The policy is run in inference mode (no grad). After the loop a
        final forward pass on the trailing ``obs`` computes
        ``self.last_values``.

        Returns the trailing ``(obs, masks)`` for the next call to pick
        up from.
        """
        self.buffer.reset()
        last_step_final_obs: dict[int, dict[str, np.ndarray]] = {}
        for step_idx in range(self.buffer.n_steps):
            obs_t = self._to_torch(obs)
            masks_t = self._masks_to_torch(masks)
            sample_out = self.policy.sample(obs_t, masks_t)

            # Extract numpy from the policy output. ``sample`` MUST
            # return per_head_log_prob — the buffer stores them so
            # Phase 6/7 per-head diagnostics + per-head importance
            # ratios have the data they need. Reviewer-caught HIGH:
            # the prior silent-zero fallback hid contract violations.
            if "per_head_log_prob" not in sample_out:
                raise RuntimeError(
                    "policy.sample(...) must return 'per_head_log_prob'; "
                    "got keys=" + repr(sorted(sample_out.keys()))
                )
            action_np = sample_out["action"].cpu().numpy().astype(np.int64)
            log_prob_np = sample_out["log_prob"].cpu().numpy().astype(np.float32)
            value_np = sample_out["value"].cpu().numpy().astype(np.float32)
            per_head_np = sample_out["per_head_log_prob"].cpu().numpy().astype(np.float32)

            # Step the vec env. ``next_obs`` / ``next_masks`` are
            # auto-reset-aware: on terminated/truncated envs they
            # already reflect the next episode's start state.
            # ``final_obs`` is sparse — populated only for envs that
            # just terminated/truncated.
            next_obs, next_masks, rewards, terminated, truncated, final_obs = self.vec_env.step_all(
                action_np
            )

            self.buffer.add(
                obs=obs,
                action=action_np,
                per_head_log_prob=per_head_np,
                log_prob=log_prob_np,
                value=value_np,
                reward=rewards,
                terminated=terminated,
                truncated=truncated,
                masks=masks,
            )
            obs, masks = next_obs, next_masks
            # On the very last in-buffer step, the truncated terminal
            # obs is what GAE needs for the bootstrap (terminated envs
            # are zeroed regardless; truncated envs need V(s_T)).
            if step_idx == self.buffer.n_steps - 1:
                last_step_final_obs = final_obs

        # GAE bootstrap. For each env:
        #   - terminated at the last step → V is zeroed by GAE; the
        #     value passed here doesn't affect the output but for
        #     cleanliness we pass V(final_obs) anyway.
        #   - truncated at the last step → V(final_obs) is the correct
        #     bootstrap (the episode would have continued).
        #   - neither → V(post-rollout obs) is the standard bootstrap.
        # We build a per-env "bootstrap obs" by replacing each
        # terminated/truncated env's auto-reset obs with its final obs.
        bootstrap_obs = self._build_bootstrap_obs(obs, last_step_final_obs)
        bootstrap_t = self._to_torch(bootstrap_obs)
        last_out: dict[str, torch.Tensor] = self.policy(bootstrap_t)
        self.last_values = last_out["value"].cpu().numpy().astype(np.float32)
        return obs, masks

    def _build_bootstrap_obs(
        self,
        post_rollout_obs: dict[str, np.ndarray],
        final_obs: dict[int, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Splice the per-env terminal obs back into the batched obs.

        For envs that ended at the last step, the buffer needs
        ``V(terminal obs)`` for the GAE bootstrap (or 0 if terminated,
        but GAE handles that). The vec env's post-rollout obs is the
        auto-reset state of a NEW episode — wrong for the bootstrap.
        Patch the rows of those envs with their saved terminal obs.
        """
        if not final_obs:
            return post_rollout_obs
        out: dict[str, np.ndarray] = {k: v.copy() for k, v in post_rollout_obs.items()}
        for env_idx, term_obs in final_obs.items():
            for k, v in term_obs.items():
                if k in out:
                    out[k][env_idx] = v
        return out

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _to_torch(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}

    def _masks_to_torch(self, masks: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            k: torch.as_tensor(v, device=self.device, dtype=torch.bool) for k, v in masks.items()
        }
