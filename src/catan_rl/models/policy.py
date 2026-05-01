"""
Top-level policy: dict observation → actions + value estimate.

ObservationModule now expects a Charlesworth-style dict obs.
"""

import torch
import torch.nn as nn

from catan_rl.models.action_heads_module import MultiActionHeads
from catan_rl.models.observation_module import ObservationModule
from catan_rl.models.utils import init_weights


class CatanPolicy(nn.Module):
    """The agent's brain: sees the board state, decides what to do,
    and estimates how well it's doing.

    Three methods the PPO trainer calls:
      act()              — during rollout collection (playing games)
      evaluate_actions() — during PPO update (learning from past games)
      get_value()        — for GAE bootstrapping (just the value, cheaper)
    """

    def __init__(
        self,
        obs_output_dim: int = 512,
        value_hidden_dims: tuple[int, ...] = (256, 128),
        action_head_hidden_dim: int = 128,
        **obs_module_kwargs,
    ):
        super().__init__()

        # Encode observations → fixed-size vector
        self.observation_module = ObservationModule(
            obs_output_dim=obs_output_dim, **obs_module_kwargs
        )

        # Multi-head action selection
        self.action_heads = MultiActionHeads(
            obs_output_dim=obs_output_dim, hidden_dim=action_head_hidden_dim
        )

        # Value network: predicts expected return from current state.
        # Architecture: Linear→LayerNorm→ReLU for each hidden layer,
        # then a final Linear→scalar with gain=1.0 (not 0.01, because
        # we want the value output to have meaningful scale from the start).
        value_layers = []
        prev_dim = obs_output_dim
        for hdim in value_hidden_dims:
            value_layers.append(init_weights(nn.Linear(prev_dim, hdim)))
            value_layers.append(nn.LayerNorm(hdim))
            value_layers.append(nn.ReLU())
            prev_dim = hdim
        # Final output layer with gain=1.0
        final = nn.Linear(prev_dim, 1)
        init_weights(final, gain=1.0)
        value_layers.append(final)
        self.value_net = nn.Sequential(*value_layers)

    def act(
        self,
        obs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used during rollout collection (playing games to gather experience).

        Args:
            obs:           dict of batched observations from environment.
            masks:         per-head action masks from environment.
            deterministic: True for evaluation (argmax), False for training (sample).

        Returns:
            actions:  (B, 6) composite action (type, corner, edge, tile, res1, res2).
            value:    (B, 1) how good the agent thinks this state is.
            log_prob: (B,) probability of the chosen actions under current policy.
        """
        obs_out = self.observation_module(obs)
        value = self.value_net(obs_out)
        actions, log_prob, _ = self.action_heads(obs_out, masks, deterministic=deterministic)
        return actions, value, log_prob

    def evaluate_actions(
        self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used during PPO update to re-evaluate previously taken actions.

        Args:
            obs:     dict of stored observations from rollout buffer.
            masks:   per-head masks that were active when these actions were taken.
            actions: (B, 6) the composite actions that were actually taken.

        Returns:
            value:    (B, 1) current value estimate for these states.
            log_prob: (B,) log-probability of these actions under current policy.
            entropy:  scalar — entropy bonus to encourage exploration.
        """
        obs_out = self.observation_module(obs)
        value = self.value_net(obs_out)
        _, log_prob, entropy = self.action_heads(obs_out, masks, actions=actions)
        return value, log_prob, entropy

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Cheap value-only forward pass (no action heads)."""
        obs_out = self.observation_module(obs)
        return self.value_net(obs_out)
