"""BC loss — per-head relevance-weighted CE + value MSE + belief soft-CE.

Per ``v2_step3_bc.md`` §3:

    L_total = L_policy + 0.10 · L_value + 0.05 · L_belief

where:

  * ``L_policy = Σ_h L_h`` with each per-head loss averaged over the
    batch rows where that head is *relevant* for the chosen action
    type (per the relevance buffer shipped with ``CatanActionHeads``);
  * ``L_value = MSE(V(s), z_disc)`` over the whole batch;
  * ``L_belief = soft-CE(belief_logits, belief_target)`` over the
    whole batch.

Hard labels (panel D6 unanimous): no label smoothing, no soft
distributions over actions. The heuristic is near-deterministic.

This module is intentionally thin — the policy network's
:meth:`CatanPolicy.evaluate_actions` already returns ``per_head_log_prob``,
``relevance`` (which encodes the head-active-for-this-type map),
``value``, and ``belief_logits``. ``bc_loss`` just composes those into
the scalar training objective + per-head diagnostics for TensorBoard.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Order matches ``CatanActionHeads`` per-head relevance buffer columns.
_HEAD_NAMES: tuple[str, ...] = ("type", "corner", "edge", "tile", "resource1", "resource2")


def bc_loss(
    *,
    policy_out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    value_weight: float = 0.10,
    belief_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compose the BC training loss from a single policy forward pass.

    Args:
        policy_out: dict returned by ``CatanPolicy.evaluate_actions``.
            Must contain ``per_head_log_prob`` (B, 6), ``relevance``
            (B, 6), ``value`` (B,), ``belief_logits`` (B, 5).
        batch: dict from the BC loader — must contain ``belief_target``
            (B, 5) and ``z_disc`` (B,). ``obs`` / ``action`` / ``mask``
            are unused here (they were inputs to ``evaluate_actions``).
        value_weight: ``L_value`` coefficient (default 0.10 per panel
            D7 with throttling — can be set to 0 if the value-drift
            safeguard fires).
        belief_weight: ``L_belief`` coefficient (default 0.05 per the
            unanimous panel vote).

    Returns:
        Dict of scalar tensors:
          * ``total`` — the loss to back-prop on.
          * ``policy`` — sum of per-head CEs.
          * ``policy/<head>`` — per-head CE for each of the 6 heads
            (zero when the head was not relevant for any row).
          * ``value`` — MSE between predicted value and ``z_disc``.
          * ``belief`` — soft-CE between belief logits and target.
    """
    log_probs = policy_out["per_head_log_prob"]  # (B, 6)
    relevance = policy_out["relevance"]  # (B, 6) — 0/1

    per_head_losses: dict[str, torch.Tensor] = {}
    policy_total = log_probs.new_zeros(())
    for h_idx, name in enumerate(_HEAD_NAMES):
        rel = relevance[:, h_idx]
        # Negative log-prob is CE; weight rows by relevance, normalise by
        # active-count so per-head magnitudes are comparable.
        active_count = rel.sum().clamp_min(1.0)
        nll = -log_probs[:, h_idx]
        head_loss = (nll * rel).sum() / active_count
        # Zero out when no row was relevant (active_count==0 branch
        # would yield a non-zero NaN-safe value; mask explicitly).
        if rel.sum().item() == 0:
            head_loss = head_loss.new_zeros(())
        per_head_losses[f"policy/{name}"] = head_loss
        policy_total = policy_total + head_loss

    # Value MSE.
    value_pred = policy_out["value"]  # (B,)
    value_loss = F.mse_loss(value_pred, batch["z_disc"])

    # Belief soft-CE.
    belief_logits = policy_out["belief_logits"]  # (B, 5)
    log_probs_belief = torch.log_softmax(belief_logits, dim=-1)
    # Mask out empty-hand rows (all-zero target) and average over the INFORMATIVE
    # rows only — matches ppo.losses.compute_belief_loss, so belief_weight means
    # the same thing in BC and PPO. A plain .mean() would divide by the batch
    # (incl. the dominant empty-hand rows), silently shrinking the belief signal.
    per_row_belief = -(batch["belief_target"] * log_probs_belief).sum(dim=-1)
    belief_valid = batch["belief_target"].sum(dim=-1) > 0
    belief_loss = (per_row_belief * belief_valid).sum() / belief_valid.sum().clamp(min=1.0)

    total = policy_total + value_weight * value_loss + belief_weight * belief_loss

    out: dict[str, torch.Tensor] = {
        "total": total,
        "policy": policy_total,
        "value": value_loss,
        "belief": belief_loss,
    }
    out.update(per_head_losses)
    return out
