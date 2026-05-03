"""Phase 2.5c — opponent next-action-type auxiliary head (1v1-only).

Pairs with the belief head (Phase 2.5b) but predicts a different signal:
*not* the opponent's hidden cards, but their *next action type*. The
training signal comes from the actually-observed opponent action right
after the agent's step.

  - **Input:** policy encoder output ``(B, obs_output_dim)``.
  - **Output:** 13-way logits over ``N_ACTION_TYPES`` (matches the env's
    action-type space — see ``catan_rl.env.catan_env`` constants).
  - **Target:** opponent's actually-taken next action type, recorded by
    the env's deferred-opponent path. Stored per-step in the rollout
    buffer alongside a boolean ``is_league_opponent_step`` mask.
  - **Loss filter:** only trained on rollouts whose current opponent is a
    *historical* league policy (i.e. not random / heuristic / current_self
    / no-opponent). Training against the current policy itself produces a
    degenerate fixed-point dynamic where the head and the policy chase
    each other; AlphaStar's recipe is explicit about excluding it.
  - **Loss:** standard cross-entropy with ``F.cross_entropy(logits, target)``
    weighted by the validity mask. Loss weight 0.03 by default — small,
    just to nudge the encoder; PPO drives the policy.

**[1v1] Why this is 1v1-only:**
  - The opponent's "next action" is well-defined only when the agent
    knows whose turn it is. With 4 players, "opponent" branches to 3
    candidates; per-opponent factorization is required.
  - With P2P trade, the action space includes propose / accept / counter,
    which depend on more than the encoder state — the supervision becomes
    too noisy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from catan_rl.models.utils import init_weights

# Number of action types the head predicts. Must equal
# ``catan_rl.models.action_heads_module.N_ACTION_TYPES`` and the first axis of
# ``MultiDiscrete([13, 54, 72, 19, 5, 5])`` in the env. Hardcoded to keep
# this module import-free of the action-heads module.
N_ACTION_TYPES = 13


class OpponentActionHead(nn.Module):
    """Two-layer MLP → 13-way logits over opponent's next action type."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            init_weights(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            init_weights(nn.Linear(hidden_dim, N_ACTION_TYPES), gain=0.01),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Returns ``(B, N_ACTION_TYPES)`` logits."""
        return self.net(encoder_output)

    @staticmethod
    def masked_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Cross-entropy averaged only over rows where ``valid_mask`` is True.

        Args:
            logits: ``(B, N_ACTION_TYPES)``.
            target: ``(B,)`` int64 ground-truth action type indices. Values
                where ``valid_mask`` is False are ignored — they may be
                garbage (e.g. -1 or 0) and will be excluded from the loss.
            valid_mask: ``(B,)`` bool. True = include this row in the loss
                (it came from a historical league opponent step).

        Returns:
            Scalar mean loss over the valid subset, or ``None`` if the
            batch contains no valid rows. Trainer skips the loss term in
            that case rather than averaging over zero.
        """
        if valid_mask.any():
            logits_v = logits[valid_mask]
            target_v = target[valid_mask].long()
            # Defensive: clamp targets into the valid range. Out-of-range
            # values would crash F.cross_entropy and indicate a buffer bug.
            target_v = target_v.clamp(0, N_ACTION_TYPES - 1)
            return F.cross_entropy(logits_v, target_v)
        return None
