"""PPO loss building blocks.

Pure torch functions exposed for unit testability — the trainer
orchestrates them but never inlines the math itself. Each function is a
one-equation correspondence to the standard PPO formulation:

* :func:`compute_policy_loss` — clipped surrogate objective
  ``L = -E[ min(ratio*A, clip(ratio, 1±ε) * A) ]``.
* :func:`compute_value_loss` — PPO2-style ``max(MSE_unclipped,
  MSE_clipped)`` where the clip's reference is ``old_v``, not the
  rolling mean.
* :func:`compute_kl_approximation` — three Schulman 2020 estimators:
  k1 (biased, cheap), k2 (low-variance biased), k3 (unbiased low-var,
  the recommended default).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def compute_policy_loss(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO clipped surrogate objective (negated for minimisation).

    Args:
        new_log_prob: ``(B,)`` joint log-prob under the current policy.
        old_log_prob: ``(B,)`` joint log-prob captured at rollout time.
        advantages: ``(B,)`` — typically standardised by the buffer's
            ``advantage_norm="rollout"`` mode.
        clip_range: PPO clip ε (audit default 0.2).

    Returns:
        ``(loss, stats)`` where ``loss`` is a scalar and ``stats`` is a
        dict of diagnostic scalars (``ratio_mean``, ``clip_frac``,
        ``approx_kl_for_logs``). The clip fraction is the share of
        samples for which the clip activated — TB scalar
        ``train/clip_frac`` lets the operator detect over-clipping.
    """
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    loss = -torch.min(unclipped, clipped).mean()

    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean()
        ratio_mean = ratio.mean()
        # Lightweight KL estimate solely for the per-minibatch log — the
        # KL EARLY STOP gate uses compute_kl_approximation separately.
        approx_kl_k1 = (-log_ratio).mean()

    return loss, {
        "ratio_mean": ratio_mean.detach(),
        "clip_frac": clip_frac.detach(),
        "approx_kl_for_logs": approx_kl_k1.detach(),
    }


def compute_value_loss(
    *,
    new_value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    clip_range_vf: float,
    use_value_clipping: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PPO2 value-function loss with optional clipping.

    Args:
        new_value: ``(B,)`` current ``V(s_t)``.
        old_value: ``(B,)`` ``V(s_t)`` captured at rollout time.
        returns: ``(B,)`` GAE-derived target.
        clip_range_vf: clip ε around ``old_value`` for the clipped
            branch. Phase 1 config rejects ``clip_range_vf=0`` when
            ``use_value_clipping=True`` because the clipped branch
            becomes parameter-constant and freezes the value head.
        use_value_clipping: ``True`` → ``max(MSE_unclipped, MSE_clipped)``;
            ``False`` → plain MSE.

    Returns:
        ``(loss, stats)``. Stats include ``value_loss_unclipped`` and
        (when clipping is on) ``value_loss_clipped`` — useful for
        diagnosing the clipped branch's contribution.
    """
    err_unclipped = new_value - returns
    mse_unclipped = err_unclipped.pow(2)
    stats: dict[str, torch.Tensor] = {
        "value_loss_unclipped": mse_unclipped.mean().detach(),
    }
    if use_value_clipping:
        v_clipped = old_value + torch.clamp(new_value - old_value, -clip_range_vf, clip_range_vf)
        mse_clipped = (v_clipped - returns).pow(2)
        # PPO2: take the elementwise max so the loss never benefits from
        # the clip making the value head's gradient zero.
        loss_per = torch.max(mse_unclipped, mse_clipped)
        stats["value_loss_clipped"] = mse_clipped.mean().detach()
    else:
        loss_per = mse_unclipped
    loss = 0.5 * loss_per.mean()
    return loss, stats


def compute_kl_approximation(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    estimator: Literal["k1", "k2", "k3"],
) -> torch.Tensor:
    """Approximate KL(old || new) using one of Schulman 2020's estimators.

    All three are computed on the SAME minibatch's log-prob pair (joint),
    so they're directly comparable. The trainer's early-stop gate uses
    whichever the config selects; ``k3`` is the recommended default
    (unbiased, low variance).

    * **k1** = ``-log_ratio``. Biased but very cheap; can go negative
      under sample noise which is misleading as an "approx KL".
    * **k2** = ``log_ratio.pow(2) / 2``. Biased low-variance. Always
      non-negative.
    * **k3** = ``(ratio - 1) - log_ratio``. Unbiased low-variance, the
      Schulman default. Always non-negative.
    """
    log_ratio = new_log_prob - old_log_prob
    if estimator == "k1":
        return (-log_ratio).mean()
    if estimator == "k2":
        return 0.5 * log_ratio.pow(2).mean()
    if estimator == "k3":
        ratio = log_ratio.exp()
        return ((ratio - 1.0) - log_ratio).mean()
    raise ValueError(f"unknown KL estimator {estimator!r}; use k1/k2/k3")


def compute_belief_loss(
    *,
    belief_logits: torch.Tensor,
    belief_target: torch.Tensor,
) -> torch.Tensor:
    """Soft cross-entropy of ``belief_target`` (a probability vector)
    against the head's logits.

    The target is a soft distribution over the opponent's HIDDEN dev-card
    types — not a one-hot, so we use ``-(target * log_softmax(logits)).sum(-1)``
    rather than ``F.cross_entropy``.

    Masking (the "useful, not noisy" guarantee): a transition where the
    opponent holds NO hidden dev cards carries an all-zero target (an undefined
    posterior). Those rows contribute zero gradient anyway, but we also EXCLUDE
    them from the mean so the loss magnitude — and thus the effective
    ``belief_coef`` weight — reflects only the informative transitions, instead
    of being silently diluted by the (often majority) empty-hand states. If a
    minibatch happens to have no valid rows, the loss is exactly 0.
    """
    log_q = F.log_softmax(belief_logits, dim=-1)
    per_row = -(belief_target * log_q).sum(-1)  # (B,)
    valid = belief_target.sum(-1) > 0  # opponent held >=1 hidden dev card
    denom = valid.sum().clamp(min=1.0)
    return (per_row * valid).sum() / denom


def compute_entropy_bonus(*, joint_entropy: torch.Tensor) -> torch.Tensor:
    """The minimiser's entropy bonus = ``-coef * E[H(π)]``.

    This helper just returns ``E[H(π)]``; the trainer multiplies by the
    (annealed) coef and subtracts from the total loss.
    """
    return joint_entropy.mean()
