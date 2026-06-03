"""Generalized Advantage Estimation (GAE) for PPO.

Vectorized over (T, N) rollouts with separate ``terminated`` and
``truncated`` arrays — a real correctness distinction that the
naive ``done = terminated | truncated`` formulation collapses
incorrectly:

* **terminated**: the game actually ended (winner reached 15 VP, or
  the engine signalled a definitive terminal state). The bootstrap
  value at the next step is **0** because there is no future return.
* **truncated**: the episode ran past ``max_turns``. The game would
  have continued; the value-function estimate ``V(s_{t+1})`` is the
  correct bootstrap.

For both, the GAE accumulator must **reset** at the boundary — the
advantage at step ``t+1`` should not contribute to the advantage at
step ``t`` across an episode boundary.

This module is consumed by :class:`catan_rl.ppo.buffer.CompositeRolloutBuffer`
but kept separate so the math is testable without the buffer's
allocation machinery.
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized GAE.

    Args:
        rewards: ``(T, N)`` rewards at each in-buffer transition.
        values: ``(T, N)`` ``V(s_t)`` estimates.
        terminated: ``(T, N)`` bool — True iff the transition ended the
            episode with a "real" terminal (the game ended).
        truncated: ``(T, N)`` bool — True iff the transition ended the
            episode via the time-limit / ``max_turns`` truncation.
        last_values: ``(N,)`` ``V(s_T+1)`` — the value estimate at the
            state *after* the last in-buffer transition. The caller
            computes this via a fresh forward pass on the post-rollout
            obs (which is the start of the next episode under auto-
            reset envs). Whether the in-buffer last step was terminated
            is read directly from ``terminated[T-1]`` — there is no
            separate ``last_terminated`` parameter to confuse with
            "is the post-rollout obs terminal" (which is always False
            under auto-reset).
        gamma: discount factor in ``(0, 1]``.
        gae_lambda: GAE λ in ``[0, 1]``.

    Returns:
        ``(advantages, returns)`` both shaped ``(T, N)``. ``returns =
        advantages + values`` by definition (PPO value target).

    Convention pin (subtle): the GAE accumulator resets at *every*
    episode boundary — both terminated and truncated. The bootstrap of
    ``V(s_{t+1})`` distinguishes the two: ``terminated`` zeros it,
    ``truncated`` keeps it. This is the standard "PPO done flag with a
    separate timeout bit" handling that prevents truncation from
    silently corrupting the value target (a real foot-gun).

    History note: an earlier API took a separate ``last_terminated``
    arg; a Phase 3 reviewer flagged that callers could confuse it with
    "is the next obs (auto-reset) terminal" and double-bootstrap at a
    real terminal. Removed — the buffer already owns ``terminated[T-1]``.
    """
    if rewards.shape != values.shape:
        raise ValueError(f"rewards shape {rewards.shape} != values shape {values.shape}")
    if rewards.shape != terminated.shape or rewards.shape != truncated.shape:
        raise ValueError(
            f"shape mismatch: rewards={rewards.shape} terminated={terminated.shape} "
            f"truncated={truncated.shape}"
        )
    if last_values.shape != (rewards.shape[1],):
        raise ValueError(f"last_values shape {last_values.shape} expected ({rewards.shape[1]},)")
    if not 0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")
    if not 0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    T, N = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)
    last_values = last_values.astype(np.float32)

    # Walk backwards from T-1 to 0. The bootstrap value at every step
    # is unified: if terminated at step t, V(s_{t+1}) is treated as 0;
    # otherwise it's the in-buffer V(s_{t+1}) (or ``last_values`` at
    # t = T-1). Truncated keeps the bootstrap; both terminated and
    # truncated reset the GAE accumulator.
    for t in reversed(range(T)):
        is_last = t == T - 1
        in_buffer_next = last_values if is_last else values[t + 1].astype(np.float32)
        next_values = np.where(terminated[t], 0.0, in_buffer_next).astype(np.float32)
        non_terminal = (~terminated[t]).astype(np.float32)
        # Last step has no in-buffer successor → inheritance is always 0.
        # Other steps block inheritance across any episode boundary.
        non_done_for_inherit = (
            np.zeros(N, dtype=np.float32)
            if is_last
            else (~(terminated[t] | truncated[t])).astype(np.float32)
        )

        r_t = rewards[t].astype(np.float32)
        v_t = values[t].astype(np.float32)
        delta = r_t + gamma * next_values * non_terminal - v_t
        gae = delta + gamma * gae_lambda * non_done_for_inherit * gae
        advantages[t] = gae

    returns = advantages + values.astype(np.float32)
    return advantages, returns


def normalize_advantages(advantages: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Standardize advantages to zero mean / unit variance over the
    whole input array.

    Returns a NEW array. Used for the per-rollout normalization mode
    (the default) — the trainer should call this once after
    :func:`compute_gae` and before sampling minibatches.
    """
    flat = advantages.reshape(-1).astype(np.float64)  # promote for numerical stability
    mean = flat.mean()
    std = flat.std()
    if std < eps:
        # Degenerate buffer (all advantages identical). Subtract mean only.
        return (advantages - mean).astype(np.float32)
    return ((advantages - mean) / (std + eps)).astype(np.float32)
