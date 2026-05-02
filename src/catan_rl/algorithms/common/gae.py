"""GAE computation, explained variance, and small numerical helpers.

Phase 0 of the roadmap split the legacy single-``dones`` GAE into separate
``terminated`` and ``truncated`` arrays. The motivation: with ``max_turns=500``
truncations on long Catan stalemates, the old code treated truncation as
genuine termination (zeroing the bootstrap value), silently miscrediting
every truncated trajectory. The post-fix recurrence:

    next_value      = V(s_{t+1})            (or last_value at t = T-1)
    non_terminal    = 1 - terminated[t]      # zero only on real game-over
    delta_t         = r_t + γ · next_value · non_terminal − V(s_t)
    last_gae        = delta_t + γ · λ · non_terminal · (1 − truncated[t]) · last_gae

That is: terminations zero both the bootstrap and the GAE accumulator;
truncations keep the bootstrap (so the value target is correct) but reset
the accumulator (so cross-episode credit doesn't leak through the truncation
boundary).

A legacy single-``dones`` calling convention is preserved for back-compat with
older test fixtures; new code should pass ``terminated`` and ``truncated``
separately.
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    *args,
    last_value: float | None = None,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE with separate terminated/truncated.

    Two calling conventions:

    Post-Phase-0 (preferred), positional::
        compute_gae(rewards, values, terminated, truncated, last_value, gamma, gae_lambda)

    Legacy single-dones, positional::
        compute_gae(rewards, values, dones, last_value, gamma, gae_lambda)
        # Equivalent to: terminated=dones, truncated=zeros_like(dones).
        # Reproduces pre-Phase-0 (zero-on-truncation) behavior so old call
        # sites and tests still pass during the migration.

    Keyword arguments are also accepted for either form.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    # Resolve calling convention from positional args.
    # Disambiguator: in the new signature args[1] is ``truncated`` (array-like);
    # in the legacy signature args[1] is ``last_value`` (scalar).
    if terminated is not None or truncated is not None:
        if terminated is None or truncated is None:
            raise TypeError("compute_gae: terminated and truncated must be passed together")
        if last_value is None:
            raise TypeError("compute_gae: last_value is required")
        term = np.asarray(terminated, dtype=np.float32)
        trunc = np.asarray(truncated, dtype=np.float32)
        lv = float(last_value)
    elif len(args) >= 3 and not np.isscalar(args[1]):
        # New positional: terminated, truncated, last_value [, gamma, gae_lambda]
        term = np.asarray(args[0], dtype=np.float32)
        trunc = np.asarray(args[1], dtype=np.float32)
        lv = float(args[2])
        if len(args) >= 4:
            gamma = float(args[3])
        if len(args) >= 5:
            gae_lambda = float(args[4])
    elif len(args) >= 1:
        # Legacy: dones positional, last_value either positional or kwarg.
        term = np.asarray(args[0], dtype=np.float32)
        trunc = np.zeros_like(term)
        if len(args) >= 2:
            lv = float(args[1])
            if len(args) >= 3:
                gamma = float(args[2])
            if len(args) >= 4:
                gae_lambda = float(args[3])
        elif last_value is not None:
            lv = float(last_value)
        else:
            raise TypeError("compute_gae: last_value is required")
    else:
        raise TypeError(
            "compute_gae: pass either (terminated, truncated, last_value, ...) "
            "or legacy (dones, last_value, ...)"
        )

    return _compute_gae_impl(rewards, values, term, trunc, lv, gamma, gae_lambda)


def _compute_gae_impl(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        # Terminated zeroes the bootstrap (value beyond a real game-over is 0
        # by definition); truncated keeps it (V(s_T) is still a valid estimate).
        non_terminal = 1.0 - terminated[t]
        # Reset the accumulator at any episode boundary (terminated OR truncated).
        # Without this, advantage from the next episode's trajectory would leak
        # backward across the boundary, miscrediting actions across game resets.
        accumulator_keep = non_terminal * (1.0 - truncated[t])

        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * accumulator_keep * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_gae_vectorized(
    rewards: np.ndarray,
    values: np.ndarray,
    *args,
    last_values: np.ndarray | None = None,
    n_envs: int | None = None,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    terminated: np.ndarray | None = None,
    truncated: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE for round-robin interleaved transitions across n_envs.

    Two calling conventions, mirroring ``compute_gae``:

    Post-Phase-0 (preferred), positional::
        compute_gae_vectorized(rewards, values, terminated, truncated,
                               last_values, n_envs, gamma, gae_lambda)

    Legacy::
        compute_gae_vectorized(rewards, values, dones,
                               last_values, n_envs, gamma, gae_lambda)
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    # Disambiguator: in the new signature args[1] is ``truncated`` (array-like
    # of shape n_steps); in the legacy signature args[1] is ``last_values``
    # (array-like of shape n_envs). They have different first-dim lengths.
    if terminated is not None or truncated is not None:
        if terminated is None or truncated is None:
            raise TypeError(
                "compute_gae_vectorized: terminated and truncated must be passed together"
            )
        if last_values is None or n_envs is None:
            raise TypeError("compute_gae_vectorized: last_values and n_envs are required")
        term = np.asarray(terminated, dtype=np.float32)
        trunc = np.asarray(truncated, dtype=np.float32)
        lv = np.asarray(last_values, dtype=np.float32)
        ne = int(n_envs)
    elif len(args) >= 4 and len(np.asarray(args[1])) == len(rewards):
        # New positional: terminated, truncated, last_values, n_envs, [gamma], [gae_lambda]
        term = np.asarray(args[0], dtype=np.float32)
        trunc = np.asarray(args[1], dtype=np.float32)
        lv = np.asarray(args[2], dtype=np.float32)
        ne = int(args[3])
        if len(args) >= 5:
            gamma = float(args[4])
        if len(args) >= 6:
            gae_lambda = float(args[5])
    elif len(args) >= 1:
        # Legacy: dones positional. last_values and/or n_envs may be either
        # positional (from args[1:]) or keyword.
        term = np.asarray(args[0], dtype=np.float32)
        trunc = np.zeros_like(term)
        # last_values: from args[1] if present, else kwarg.
        if len(args) >= 2:
            lv = np.asarray(args[1], dtype=np.float32)
        elif last_values is not None:
            lv = np.asarray(last_values, dtype=np.float32)
        else:
            raise TypeError("compute_gae_vectorized: last_values required")
        # n_envs: from args[2] if present, else kwarg.
        if len(args) >= 3:
            ne = int(args[2])
        elif n_envs is not None:
            ne = int(n_envs)
        else:
            raise TypeError("compute_gae_vectorized: n_envs required")
        if len(args) >= 4:
            gamma = float(args[3])
        if len(args) >= 5:
            gae_lambda = float(args[4])
    else:
        raise TypeError(
            "compute_gae_vectorized: pass either (terminated, truncated, last_values, n_envs, ...)"
            " or legacy (dones, last_values, n_envs, ...)"
        )

    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    returns = np.zeros(n_steps, dtype=np.float32)

    for env_idx in range(ne):
        indices = np.arange(env_idx, n_steps, ne)
        if len(indices) == 0:
            continue
        adv, ret = _compute_gae_impl(
            rewards[indices],
            values[indices],
            term[indices],
            trunc[indices],
            float(lv[env_idx]),
            gamma,
            gae_lambda,
        )
        advantages[indices] = adv
        returns[indices] = ret

    return advantages, returns


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """How well do our value predictions explain the actual returns?

    1.0 = perfect prediction, 0.0 = no better than predicting the mean,
    negative = worse than predicting the mean.

    A key diagnostic: if explained_variance stays near 0, the value
    network isn't learning, which will hurt policy gradient estimates.
    """
    var_true = np.var(y_true)
    if var_true < 1e-8:
        return 0.0
    return 1.0 - np.var(y_true - y_pred) / var_true


def safe_mean(xs: list[float] | np.ndarray) -> float:
    """Mean that returns 0.0 for empty inputs (avoids numpy warnings)."""
    if isinstance(xs, np.ndarray):
        return float(xs.mean()) if xs.size > 0 else 0.0
    return float(np.mean(xs)) if len(xs) > 0 else 0.0
