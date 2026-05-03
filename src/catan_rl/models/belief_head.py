"""Phase 2.5b — opponent hidden dev-card belief head (1v1-only).

In 1v1 Colonist.io with the perfect-information broadcast tracker, the
*only* hidden state is the type distribution of the opponent's unrevealed
dev cards. Their *count* is observable (we see the announcement when they
buy a card and when they play one), but the **type** of an unplayed card
stays hidden.

The belief head turns this into a supervised auxiliary loss. The encoder
already produces a 512-dim representation that has to encode a lot —
adding a small head that predicts the opponent's hidden type distribution
gives the encoder a dense, high-quality signal early in training, before
PPO's reward-driven gradients catch up.

  - **Input:** policy observation encoder output ``(B, obs_output_dim)``.
  - **Output:** 5-way logits over ``{KNIGHT, VP, ROADBUILDER, YOP, MONOPOLY}``.
  - **Target:** the *true* normalized count vector over those 5 types,
    computed by the env from ``opponent.devCards`` + ``opponent.newDevCards``.
    The env reveals this to the trainer-side belief target only — it never
    enters the obs schema, so policy can't cheat off it.
  - **Loss:** soft cross-entropy ``-Σ target_i · log_softmax(logits)_i``.
    With a normalized count vector this is exactly the KL divergence
    ``D_KL(target || prediction) + H(target)``; the constant ``H(target)``
    drops out of the gradient.

Why not a categorical CE over individual cards? The opponent typically
holds 0–3 hidden cards. We don't know which slot is which type; the
*multiset* is what we need to predict. Normalizing the count vector and
using soft CE recovers exactly that.

Why not Bernoulli per type? The 5 types aren't independent (their counts
sum to the known opponent hand size). A normalized softmax respects that
constraint by construction.

**[1v1] Why this is 1v1-only:**
  - With P2P trade, devCards can move between players, so the env's
    "true" answer is stale by the time the loss fires.
  - With 4 players, the supervision becomes a joint distribution over 3
    opponents. Either the head's output dim explodes (5⁴=625) or we lose
    most of the structure by per-opponent factorization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from catan_rl.models.utils import init_weights

# Number of dev-card types we predict. Must stay in sync with the env's
# DEV_CARD_ORDER ordering: KNIGHT, VP, ROADBUILDER, YOP, MONOPOLY.
N_DEV_CARD_TYPES = 5


class BeliefHead(nn.Module):
    """Two-layer MLP → 5-way logits for opponent hidden dev-card types.

    The hidden width matches the action heads' default (128) so the
    parameter footprint is comparable.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            init_weights(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            init_weights(nn.Linear(hidden_dim, N_DEV_CARD_TYPES), gain=0.01),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: ``(B, input_dim)``.

        Returns:
            ``(B, 5)`` logits, one per dev-card type.
        """
        return self.net(encoder_output)

    @staticmethod
    def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft CE between predicted logits and a target probability vector.

        Equivalent to ``D_KL(target || softmax(logits)) + H(target)`` —
        the constant entropy term drops out of the gradient, so this loss
        and full KL produce identical updates.

        Args:
            logits: ``(B, N_DEV_CARD_TYPES)``.
            target: ``(B, N_DEV_CARD_TYPES)`` non-negative, sums to 1 along
                the last dim. Rows where the opponent has zero hidden cards
                should be passed in as the uniform distribution by the
                caller (or filtered out via a mask) — soft CE is undefined
                on a degenerate all-zeros target.

        Returns:
            Scalar mean loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        # ``-Σ target_i log_softmax(logits)_i`` — per-sample, then mean.
        return -(target * log_probs).sum(dim=-1).mean()
