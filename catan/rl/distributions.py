"""
Custom probability distributions with action masking.

The core trick: set invalid action logits to -1e8 before passing to
Categorical. This makes their probability effectively zero after softmax.
All sampling/log_prob/entropy methods are inherited from PyTorch's Categorical.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MaskedCategorical(Categorical):
    """Categorical distribution that zeros out invalid actions via logit masking."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            logits: (B, N) raw scores from an action head's linear layer.
            mask:   (B, N) bool — True = valid, False = invalid.
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        self.mask = mask
        # Replace invalid action logits with a very negative number.
        # After softmax these become ~0 probability.
        masked_logits = logits.masked_fill(~mask, -1e8)
        super().__init__(logits=masked_logits)

    def entropy(self) -> torch.Tensor:
        """Per-sample entropy, only over valid actions. Shape: (B,).

        We override because the parent's entropy() includes the -1e8 terms
        which can cause numerical noise. We recompute using only valid probs.
        """
        # self.probs is computed by Categorical from the masked logits
        # Invalid actions already have ~0 probability, but we clamp to be safe
        p = self.probs
        # Only sum over valid actions (masked ones contribute ~0 anyway,
        # but the log of near-zero can produce large negative numbers)
        log_p = torch.where(self.mask, torch.log(p + 1e-10), torch.zeros_like(p))
        return -(p * log_p).sum(dim=-1)

    def mode(self) -> torch.Tensor:
        """Deterministic action = argmax of probabilities. Shape: (B,)."""
        return self.probs.argmax(dim=-1)


class CategoricalHead(nn.Module):
    """A single linear layer that outputs a MaskedCategorical distribution.

    Uses a very small initialization gain (0.01 by default) so that at the
    start of training, all valid actions have roughly equal probability.
    This encourages exploration early on.
    """

    def __init__(self, input_dim: int, output_dim: int, gain: float = 0.01):
        """
        Args:
            input_dim:  feature vector size coming from the action head MLP.
            output_dim: number of discrete choices (e.g. 12 for action type).
            gain:       orthogonal init gain. 0.01 → near-uniform initial outputs.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # Orthogonal init with small gain keeps logits near zero at init,
        # so softmax(logits) ≈ uniform over valid actions.
        nn.init.orthogonal_(self.linear.weight, gain=gain)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> MaskedCategorical:
        """
        Args:
            x:    (B, input_dim) features from the upstream MLP.
            mask: (B, output_dim) bool mask — True for valid actions.

        Returns:
            MaskedCategorical distribution you can call .sample(), .log_prob(), etc. on.
        """
        logits = self.linear(x)  # (B, output_dim)
        return MaskedCategorical(logits=logits, mask=mask)
