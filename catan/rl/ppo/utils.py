"""
PPO training utilities: GAE computation, metrics, and helpers.
"""
import numpy as np
from typing import List, Tuple, Union


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (Schulman et al., 2015).

    GAE smoothly interpolates between TD(0) and Monte Carlo estimates
    of the advantage function. The lambda parameter controls this:
      - lambda=0: TD(0) — low variance but high bias
      - lambda=1: Monte Carlo — no bias but high variance
      - lambda=0.95: sweet spot used by most PPO implementations

    The algorithm walks backward through the trajectory:
      delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
      A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        # TD residual: how much better was this step than the value predicted?
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        # Accumulate: current TD error + discounted future advantage
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    # Returns = advantages + baseline values (what we want the value net to predict)
    returns = advantages + values
    return advantages, returns


def compute_gae_vectorized(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    n_envs: int,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE for round-robin interleaved transitions from multiple envs.

    Transitions are stored as: env0, env1, ..., env(n-1), env0, env1, ...
    Each env's trajectory is extracted and GAE computed separately.
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    returns = np.zeros(n_steps, dtype=np.float32)

    for env_idx in range(n_envs):
        indices = np.arange(env_idx, n_steps, n_envs)
        if len(indices) == 0:
            continue
        r = rewards[indices]
        v = values[indices]
        d = dones[indices]
        adv, ret = compute_gae(r, v, d, last_values[env_idx], gamma, gae_lambda)
        advantages[indices] = adv
        returns[indices] = ret

    return advantages, returns


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """How well do our value predictions explain the actual returns?

    1.0 = perfect prediction, 0.0 = no better than predicting the mean,
    negative = worse than predicting the mean.

    This is a key diagnostic: if explained_variance stays near 0, the value
    network isn't learning, which will hurt policy gradient estimates.
    """
    var_true = np.var(y_true)
    if var_true < 1e-8:
        return 0.0
    return 1.0 - np.var(y_true - y_pred) / var_true


def safe_mean(xs: Union[List[float], np.ndarray]) -> float:
    """Mean that returns 0.0 for empty inputs (avoids numpy warnings)."""
    if isinstance(xs, np.ndarray):
        return float(xs.mean()) if xs.size > 0 else 0.0
    return float(np.mean(xs)) if len(xs) > 0 else 0.0
