"""
Setup-phase policy network for decoupled setup training.

Architecture:
  Encoder:  flat obs (1417) → Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → 256-dim
  Corner head: 256 → 128 → ReLU → Linear → 54 logits  (for settlement placement)
  Edge head:   256 → 128 → ReLU → Linear → 72 logits  (for road placement)
  Value head:  256 → 128 → ReLU → Linear → 1

The policy uses MaskedCategorical from the main model so that invalid
placements are excluded before sampling.

During a settle step: only the corner head is active (edge head masked to all-False).
During a road step:   only the edge head is active (corner head masked to all-False).
The inactive head still runs through the network (needed for batch updates),
but its log-probability is zeroed out and does not contribute to the gradient.
"""

import torch
import torch.nn as nn

from catan_rl.models.distributions import MaskedCategorical
from catan_rl.models.utils import init_weights

N_CORNERS = 54
N_EDGES = 72


class SetupPolicy(nn.Module):
    """Two-head policy for setup-phase placement decisions.

    Parameters
    ----------
    obs_dim : int
        Dimension of the flat observation vector (default 1417).
    hidden_dim : int
        Width of the shared encoder's hidden layers.
    head_hidden : int
        Width of each action head's hidden layer.
    """

    def __init__(self, obs_dim: int = 1417, hidden_dim: int = 256, head_hidden: int = 128):
        super().__init__()

        # Shared encoder: obs → feature vector
        self.encoder = nn.Sequential(
            init_weights(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            init_weights(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Corner head (settlement placement)
        self.corner_head = _ActionHead(hidden_dim, N_CORNERS, head_hidden)

        # Edge head (road placement)
        self.edge_head = _ActionHead(hidden_dim, N_EDGES, head_hidden)

        # Value head
        self.value_head = nn.Sequential(
            init_weights(nn.Linear(hidden_dim, head_hidden)),
            nn.LayerNorm(head_hidden),
            nn.ReLU(),
            init_weights(nn.Linear(head_hidden, 1), gain=1.0),
        )

    def act(
        self,
        obs: torch.Tensor,
        corner_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        is_corner_step: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions during rollout collection.

        Parameters
        ----------
        obs : (B, obs_dim)
        corner_mask : (B, 54) bool — True for valid settlement vertices
        edge_mask   : (B, 72) bool — True for valid road edges
        is_corner_step : (B,) bool — True when current step is a settlement step
        deterministic : bool

        Returns
        -------
        actions   : (B, 2) int64 — [corner_idx, edge_idx]
        value     : (B, 1)
        log_prob  : (B,)  — sum of active head log-probs
        """
        features = self.encoder(obs)
        value = self.value_head(features)

        corner_dist = self.corner_head(features, corner_mask)
        edge_dist = self.edge_head(features, edge_mask)

        if deterministic:
            corner_action = corner_dist.mode()
            edge_action = edge_dist.mode()
        else:
            corner_action = corner_dist.sample()
            edge_action = edge_dist.sample()

        # Joint log-prob: only the active head contributes
        corner_lp = corner_dist.log_prob(corner_action)
        edge_lp = edge_dist.log_prob(edge_action)

        # is_corner_step: (B,) → float mask
        is_corner = is_corner_step.float()
        is_edge = 1.0 - is_corner
        log_prob = is_corner * corner_lp + is_edge * edge_lp

        actions = torch.stack([corner_action, edge_action], dim=-1)
        return actions, value, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        corner_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        is_corner_step: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions under the current policy (PPO update step).

        Parameters
        ----------
        obs         : (B, obs_dim)
        corner_mask : (B, 54)
        edge_mask   : (B, 72)
        is_corner_step : (B,)
        actions     : (B, 2) — [corner_idx, edge_idx]

        Returns
        -------
        value    : (B, 1)
        log_prob : (B,)
        entropy  : scalar
        """
        features = self.encoder(obs)
        value = self.value_head(features)

        corner_dist = self.corner_head(features, corner_mask)
        edge_dist = self.edge_head(features, edge_mask)

        corner_action = actions[:, 0]
        edge_action = actions[:, 1]

        corner_lp = corner_dist.log_prob(corner_action)
        edge_lp = edge_dist.log_prob(edge_action)

        is_corner = is_corner_step.float()
        is_edge = 1.0 - is_corner
        log_prob = is_corner * corner_lp + is_edge * edge_lp

        # Entropy: average over active heads (weight by step type frequency)
        corner_entropy = corner_dist.entropy()
        edge_entropy = edge_dist.entropy()
        entropy = (is_corner * corner_entropy + is_edge * edge_entropy).mean()

        return value, log_prob, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Cheap value-only forward pass."""
        features = self.encoder(obs)
        return self.value_head(features)


class _ActionHead(nn.Module):
    """Single action head: Linear → LayerNorm → ReLU → Linear → MaskedCategorical."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            init_weights(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.logit_layer = init_weights(nn.Linear(hidden_dim, output_dim), gain=0.01)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> MaskedCategorical:
        h = self.net(features)
        logits = self.logit_layer(h)
        return MaskedCategorical(logits=logits, mask=mask)
