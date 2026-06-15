"""Leaf value evaluation for search (contract C1).

The policy's value head outputs a raw scalar V that exceeds [-1, 1] for ~27% of
states (it was trained with a +/-1 win signal PLUS a per-VP margin bonus). PUCT
mixes the leaf value with the prior on a common scale, so the leaf MUST be a
bounded win-probability. We map V -> P(win) with the monotone squash

    P(win) = sigmoid(a * V + b),  a = 3.22, b = -1.14

fitted on peer self-play games (Brier 0.149 vs 0.243 base-rate, ECE 0.039,
Spearman 0.69). NEVER back up raw V.

Perspective: ``CatanEnv`` folds the opponent's whole turn into the agent's
``EndTurn`` transition, so every search node is an *agent* decision and the env's
observation is always built from the agent's POV. The squashed value is therefore
the agent's win-probability directly — no per-ply sign flip (see ``mcts.py``).
For a TERMINAL state the caller uses the true 1/0 outcome, never this leaf.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy

#: Fitted squash constants (see module docstring).
VALUE_SQUASH_A = 3.22
VALUE_SQUASH_B = -1.14


def squash_value(
    v: float | torch.Tensor,
    a: float = VALUE_SQUASH_A,
    b: float = VALUE_SQUASH_B,
) -> float | torch.Tensor:
    """Map a raw value-head scalar to a bounded win-probability in (0, 1).

    Accepts a python float or a tensor; returns the same kind. Strictly
    monotone increasing in ``v``.
    """
    if isinstance(v, torch.Tensor):
        return torch.sigmoid(a * v + b)
    return 1.0 / (1.0 + math.exp(-(a * float(v) + b)))


@torch.no_grad()
def value_from_obs(
    policy: CatanPolicy,
    obs: dict[str, np.ndarray],
    *,
    device: torch.device,
    a: float = VALUE_SQUASH_A,
    b: float = VALUE_SQUASH_B,
) -> float:
    """Squashed win-probability for an already-built agent-POV ``obs``.

    The hot path: ``mcts`` captures the obs returned by ``env.step`` for each
    node, so it never has to call back into the env to evaluate a leaf.
    """
    from catan_rl.policy.obs_tensor import obs_to_torch

    obs_t = obs_to_torch(obs, device, add_batch=True)
    v_raw = policy.forward(obs_t)["value"]  # (1,) — ValueHead squeezes the last dim
    sq = squash_value(v_raw, a, b)
    assert isinstance(sq, torch.Tensor)
    return float(sq.item())


@torch.no_grad()
def leaf_value(
    policy: CatanPolicy,
    env: CatanEnv,
    *,
    device: torch.device | None = None,
    a: float = VALUE_SQUASH_A,
    b: float = VALUE_SQUASH_B,
) -> float:
    """Squashed win-probability of ``env``'s current state from the agent's POV.

    Convenience over :func:`value_from_obs` that reads the env's current
    observation. ``env._get_obs()`` is the env's canonical obs builder (the same
    one ``reset``/``step`` return); search only reads it, never mutates the env.
    """
    if device is None:
        device = next(policy.parameters()).device
    return value_from_obs(policy, env._get_obs(), device=device, a=a, b=b)
