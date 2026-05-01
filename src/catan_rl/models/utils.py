"""
Model utility functions: weight initialization, value normalization, and module cloning.
"""

import copy

import torch
import torch.nn as nn


def get_clones(module: nn.Module, num_copies: int) -> nn.ModuleList:
    """Return a ModuleList of deep copies of the given module (used by Charlesworth-style MHA/tile encoder)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


def init_weights(module: nn.Module, gain: float = 2**0.5) -> nn.Module:
    """Apply orthogonal init to weights and zero-fill biases.

    Orthogonal initialization preserves gradient magnitudes through layers,
    which helps deep networks train stably. The gain controls the scale:
      - sqrt(2) ≈ 1.414 for layers followed by ReLU (compensates for ReLU
        killing half the distribution)
      - 1.0 for the value network's final output layer
      - 0.01 for action head outputs (keeps initial policy near-uniform)

    Args:
        module: typically nn.Linear; must have .weight and optionally .bias.
        gain:   scaling factor for the orthogonal init.

    Returns:
        The same module (allows chaining like: layer = init_weights(nn.Linear(64, 32))).
    """
    if hasattr(module, "weight") and module.weight.dim() >= 2:
        nn.init.orthogonal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class ValueFunctionNormalizer:
    """Running mean/std tracker for value function targets.

    The value network predicts *normalized* returns (centered around 0,
    scaled to std≈1). This makes learning much easier because:
    - The network doesn't need to learn the scale of returns
    - LayerNorm inside the network works better with normalized targets
    - Gradient magnitudes stay consistent as reward scale changes

    Uses Welford's online algorithm: numerically stable incremental
    computation of mean and variance without storing all past values.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.var = std**2
        self.count = 1e-4  # small epsilon to avoid division by zero at start

    @property
    def std(self) -> float:
        return max(self.var**0.5, 1e-8)

    def update(self, values: torch.Tensor) -> None:
        """Update running statistics with a new batch of values.

        Welford's algorithm: for each new value x, update count, mean, and
        variance incrementally. We batch-update using the parallel formula.
        """
        batch = values.detach().cpu().numpy().flatten()
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = len(batch)

        # Parallel Welford update: combine two sets of statistics
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert raw returns to zero-mean, unit-variance."""
        return (targets - self.mean) / max(self.std, 1e-8)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original scale."""
        return normalized * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]
