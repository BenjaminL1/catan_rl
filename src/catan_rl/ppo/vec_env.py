"""Vectorised wrapper around N ``CatanEnv`` instances.

Phase 5 ships the **serial** variant (all envs in-process) and that
remains the only implementation. The 2026-06-06 forensic audit
confirmed no ``SubprocVecEnv`` class exists anywhere in
``src/catan_rl/``; the "subproc is a follow-up sub-phase" framing
that previously lived in this docstring stayed aspirational. See
``docs/plans/rust_engine_actual_state.md`` for the audit. The Rust
migration's Phase 6 either implements ``"subproc"`` or removes the
literal from ``RolloutConfig.vec_env_mode``.

Contract:

* :meth:`SerialVecEnv.reset_all` resets every env with a distinct
  seed and returns the batched initial obs + masks.
* :meth:`SerialVecEnv.step_all` takes an ``(N, 6)`` action array and
  returns ``(obs, masks, rewards, terminated, truncated)`` where:

    * obs is a dict of ``(N, *)`` arrays.
    * masks is a dict of ``(N, K_i)`` bool arrays.
    * rewards is ``(N,)`` float32.
    * terminated / truncated are ``(N,)`` bool.

* Auto-reset on terminal: any env whose step returned True for
  terminated or truncated is reset in place, and the returned obs /
  masks reflect the **auto-reset** state (the start of the next
  episode) — NOT the terminal obs. The buffer stores
  ``terminated[t] / truncated[t]`` for the transition that *led* to
  the terminal; that pair is what :mod:`catan_rl.ppo.gae` uses to
  decide whether to zero the bootstrap.

* :meth:`SerialVecEnv.close` shuts down all envs cleanly.

Helpers :func:`obs_spec_from_env` and :func:`mask_spec_from_env`
convert ``CatanEnv``'s spaces into the
:class:`catan_rl.ppo.buffer.ObsSpec` / :class:`MaskSpec` schema used
by the rollout buffer's pre-allocation.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from catan_rl.env.catan_env import CatanEnv
from catan_rl.policy.obs_schema import OPP_KIND_LEAGUE
from catan_rl.ppo.buffer import MaskSpec, ObsSpec
from catan_rl.selfplay.league import OPPONENT_KIND_SNAPSHOT, OpponentAssignment


class SerialVecEnv:
    """N CatanEnv instances stepped in lockstep, in-process.

    Constructor takes a list of ``env_kwargs`` dicts (one per env) so
    callers can mix opponent types or per-env config knobs. The vec env
    creates one ``CatanEnv`` per dict and threads seeds via
    ``reset_all``.

    Lifetime: caller must invoke :meth:`reset_all` before the first
    :meth:`step_all`. :meth:`close` is idempotent.
    """

    def __init__(
        self,
        env_kwargs_list: Sequence[dict[str, Any]],
        *,
        seed: int | None = None,
    ) -> None:
        if not env_kwargs_list:
            raise ValueError("env_kwargs_list must be non-empty")
        self.n_envs = len(env_kwargs_list)
        self.envs: list[CatanEnv] = [CatanEnv(**kw) for kw in env_kwargs_list]
        self._closed = False

        # Per-env opponent reset options, applied on each env's NEXT reset
        # (US3 mid-rollout swap). Empty dict = use the env's construction
        # default opponent. ``set_opponents`` populates these + injects/clears
        # the frozen snapshot opponent on each env.
        self._reset_options: list[dict[str, Any]] = [{} for _ in range(self.n_envs)]

        # Per-env RNG for auto-reset seed derivation. Owned by the vec
        # env so two runs with the same ``seed`` produce identical
        # auto-reset seed sequences regardless of what the global
        # ``np.random`` state happens to be at the moment of the
        # terminal (Modal/SkyPilot cold start, prior reset calls, etc.).
        # ``seed=None`` falls back to OS entropy via SeedSequence.
        ss = np.random.SeedSequence(seed)
        child_seqs = ss.spawn(self.n_envs)
        self._reset_rngs = [np.random.default_rng(s) for s in child_seqs]

        # Cache the obs / mask specs derived from env 0 (all envs share
        # the schema; differing only in opponent_type which doesn't
        # change shapes).
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_all(
        self, *, seeds: Sequence[int] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Reset every env. Returns ``(obs, masks)`` batched along N.

        If ``seeds`` is None, derives ``base_seed + i`` from the env's
        existing RNG state via ``np.random``. Pass an explicit list when
        reproducibility matters.
        """
        if self._closed:
            raise RuntimeError("step on a closed vec env")
        if seeds is None:
            seeds = [int(np.random.randint(0, 2**31 - 1)) for _ in range(self.n_envs)]
        if len(seeds) != self.n_envs:
            raise ValueError(f"seeds length {len(seeds)} != n_envs {self.n_envs}")

        per_env_obs: list[dict[str, np.ndarray]] = []
        per_env_masks: list[dict[str, np.ndarray]] = []
        for i, (env, seed) in enumerate(zip(self.envs, seeds, strict=True)):
            obs, _ = env.reset(seed=int(seed), options=self._reset_options[i] or None)
            per_env_obs.append(obs)
            per_env_masks.append(env.get_action_masks())
        return self._stack_obs(per_env_obs), self._stack_masks(per_env_masks)

    def set_opponents(
        self,
        assignments: Sequence[OpponentAssignment],
        *,
        snapshot_resolver: Callable[[int], Any] | None = None,
    ) -> None:
        """Set each env's opponent for its NEXT reset (US3 mid-rollout swap).

        ``assignments`` is one :class:`OpponentAssignment` per env (e.g. from
        ``League.build_env_opponent_assignments``). For a ``snapshot`` kind, the
        concrete frozen opponent is resolved via ``snapshot_resolver(snapshot_id)``
        and injected into the env; if the resolver returns ``None`` (evicted
        snapshot / no loader), the env falls back to its heuristic body (FR-011).
        Non-snapshot kinds clear any previously-injected snapshot. Current
        episodes finish under their old opponent; the swap takes effect on the
        next ``reset`` (reset_all or auto-reset).
        """
        if len(assignments) != self.n_envs:
            raise ValueError(f"assignments length {len(assignments)} != n_envs {self.n_envs}")
        for i, (env, a) in enumerate(zip(self.envs, assignments, strict=True)):
            if a.kind == OPPONENT_KIND_SNAPSHOT:
                # OpponentAssignment guarantees snapshot_id is set for this kind.
                assert a.snapshot_id is not None
                frozen = snapshot_resolver(a.snapshot_id) if snapshot_resolver else None
                env.set_snapshot_opponent(frozen)
                self._reset_options[i] = {
                    "opponent_type": "snapshot",
                    "opponent_kind": OPP_KIND_LEAGUE,
                    "opponent_policy_id": a.snapshot_id,
                }
            else:
                env.set_snapshot_opponent(None)
                self._reset_options[i] = {"opponent_type": a.kind}

    def step_all(
        self, actions: np.ndarray
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[int, dict[str, np.ndarray]],
    ]:
        """Step every env and return the auto-reset-aware batched output.

        Args:
            actions: ``(N, 6)`` int64 — one action per env per the
                :class:`CatanEnv` ``MultiDiscrete`` schema.

        Returns:
            ``(obs, masks, rewards, terminated, truncated, final_obs)``:

            * obs / masks are batched along N. For envs that
              terminated or truncated at this step, the obs / masks
              reflect the **auto-reset** state (the start of the next
              episode).
            * rewards is the immediate reward for the transition.
            * terminated / truncated belong to the transition we just
              took. They are what the buffer / GAE consume.
            * ``final_obs`` is a sparse mapping ``{env_idx: terminal_obs_dict}``
              populated only for envs that just terminated or truncated.
              The collector needs this on **truncated** rollouts so the
              GAE bootstrap reads ``V(s_T)`` of the truncated episode
              instead of ``V(initial_state_of_new_game)``. For
              terminated envs the bootstrap is zeroed by GAE anyway,
              but ``final_obs`` is still populated so the caller has a
              consistent contract.
        """
        if self._closed:
            raise RuntimeError("step on a closed vec env")
        if actions.shape != (self.n_envs, 6):
            raise ValueError(f"actions shape {actions.shape} expected ({self.n_envs}, 6)")

        per_env_obs: list[dict[str, np.ndarray]] = []
        per_env_masks: list[dict[str, np.ndarray]] = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        final_obs: dict[int, dict[str, np.ndarray]] = {}

        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, _ = env.step(actions[i])
            rewards[i] = float(reward)
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            if term or trunc:
                # Stash the terminal obs BEFORE auto-resetting so the
                # collector can value-evaluate it for the truncation
                # bootstrap. Reviewer-caught CRITICAL: previously the
                # terminal obs was overwritten by the auto-reset obs
                # and the GAE bootstrap on truncated rollouts read
                # ``V(initial_state_of_new_game)``.
                final_obs[i] = {k: np.asarray(v).copy() for k, v in obs.items()}
                # Auto-reset: derive the new episode's seed from this
                # env's own RNG (NOT process-global np.random), so two
                # runs with the same ``seed`` produce identical
                # auto-reset sequences.
                new_seed = int(self._reset_rngs[i].integers(0, 2**31 - 1))
                obs, _ = env.reset(seed=new_seed, options=self._reset_options[i] or None)
            per_env_obs.append(obs)
            per_env_masks.append(env.get_action_masks())

        return (
            self._stack_obs(per_env_obs),
            self._stack_masks(per_env_masks),
            rewards,
            terminated,
            truncated,
            final_obs,
        )

    def close(self) -> None:
        """Close every env. Safe to call repeatedly."""
        if self._closed:
            return
        for env in self.envs:
            # ``CatanEnv.close`` defaults to a no-op on the BC branch
            # but we swallow defensively — a partial close should not
            # block program shutdown.
            with contextlib.suppress(Exception):
                env.close()
        self._closed = True

    # ------------------------------------------------------------------
    # Stacking helpers
    # ------------------------------------------------------------------

    def _stack_obs(self, per_env: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Stack per-env obs into ``(N, *)`` batched dict."""
        if not per_env:
            return {}
        keys = list(per_env[0].keys())
        out: dict[str, np.ndarray] = {}
        for k in keys:
            samples = [d[k] for d in per_env]
            stacked = np.stack(samples, axis=0)
            out[k] = stacked
        return out

    def _stack_masks(self, per_env: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return self._stack_obs(per_env)  # same shape contract


# ---------------------------------------------------------------------------
# Spec helpers: env.observation_space -> ObsSpec; env.get_action_masks ->
# MaskSpec
# ---------------------------------------------------------------------------


def obs_spec_from_env(env: CatanEnv) -> dict[str, ObsSpec]:
    """Build the buffer's ``obs_spec`` dict from a single env instance.

    Discrete spaces (e.g. ``opponent_kind``) become ``ObsSpec(shape=(),
    dtype=int64)`` — the buffer stores scalar per-env values as
    ``(T, N)`` int64.
    """
    out: dict[str, ObsSpec] = {}
    space = env.observation_space
    assert isinstance(space, spaces.Dict)
    for key, sub in space.spaces.items():
        if isinstance(sub, spaces.Box):
            out[key] = ObsSpec(shape=tuple(sub.shape), dtype=np.dtype(sub.dtype))
        elif isinstance(sub, spaces.Discrete):
            out[key] = ObsSpec(shape=(), dtype=np.dtype(np.int64))
        else:
            raise TypeError(f"obs key {key!r}: unsupported space {type(sub).__name__}")
    return out


def mask_spec_from_env(env: CatanEnv) -> dict[str, MaskSpec]:
    """Build the buffer's ``mask_spec`` dict by calling
    :meth:`CatanEnv.get_action_masks` on a freshly-reset env.

    Snapshots and restores BOTH process-global PRNGs (``np.random`` and
    stdlib ``random``) around the ``env.reset(seed=0)`` call —
    :meth:`CatanEnv.reset` seeds them and would otherwise contaminate a
    caller who'd already seeded for reproducibility (reviewer-caught
    HIGH).
    """
    import random as _stdlib_random

    np_state = np.random.get_state()
    rand_state = _stdlib_random.getstate()
    try:
        env.reset(seed=0)
        masks = env.get_action_masks()
        return {k: MaskSpec(shape=tuple(v.shape)) for k, v in masks.items()}
    finally:
        np.random.set_state(np_state)
        _stdlib_random.setstate(rand_state)
