"""Phase 4.2 — GRU recurrent value head.

The policy stays Markovian; only the **value tower** picks up a recurrent
state. Motivation in 1v1 Colonist.io: with no P2P trade, resource locks
are common and many games drift toward a `max_turns=500` truncation. A
recurrent value head can read "this position has been static for K
turns" and predict draw / truncation more accurately than a memory-less
critic.

Design (minimal, no BPTT):

  - One ``nn.GRUCell`` of width ``gru_hidden_dim`` (default 64).
  - Per-env hidden state ``h_t`` is maintained by the *trainer* during
    rollout collection. The buffer stores the input-side ``h_t`` per
    step (the value going INTO the GRU at that step) so we can replay
    the cell during PPO updates.
  - On ``terminated=True``, ``h_t`` resets to zeros (true game-over —
    the next state has no temporal continuity with this one). On
    ``truncated=True``, ``h_t`` is preserved (the trajectory is cut for
    book-keeping, not because the underlying state changed).
  - During PPO update we run ``GRUCell`` once per sample using the
    stored ``h_t``, get a fresh ``h_{t+1}``, and feed the value MLP.
    Gradient flows through one GRU step per sample but not through the
    full sequence — standard truncated-BPTT-length-1 trick when
    minibatches are shuffled non-sequentially.

The output is fused with the encoder output via concatenation:
``value_features = cat(obs_out, h_{t+1})``. This preserves the existing
value MLP shape (it now takes ``obs_output_dim + gru_hidden_dim`` instead
of ``obs_output_dim``) and lets the network gracefully ignore the
recurrent feature when it doesn't help (the old features still flow
through the residual concat).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from catan_rl.models.utils import init_weights


class RecurrentValueHead(nn.Module):
    """GRUCell + value-MLP wrapper.

    Args:
        obs_output_dim: width of the encoder output (typically 512).
        hidden_dim: GRU hidden width (default 64). Concatenated with
            ``obs_out`` before the value MLP, so the MLP sees
            ``obs_output_dim + hidden_dim`` features.
        value_hidden_dims: hidden widths for the value MLP (same shape as
            ``CatanPolicy``'s standalone value_net for parity).
    """

    def __init__(
        self,
        obs_output_dim: int,
        hidden_dim: int = 64,
        value_hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRUCell(obs_output_dim, self.hidden_dim)
        layers: list[nn.Module] = []
        prev_dim = obs_output_dim + self.hidden_dim
        for hdim in value_hidden_dims:
            layers.append(init_weights(nn.Linear(prev_dim, hdim)))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        final = nn.Linear(prev_dim, 1)
        init_weights(final, gain=1.0)
        layers.append(final)
        self.value_mlp = nn.Sequential(*layers)

    def initial_hidden(self, batch_size: int, device: str | torch.device = "cpu") -> torch.Tensor:
        """Zero hidden state — the canonical reset value."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self, obs_out: torch.Tensor, hidden_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One GRU step + value MLP.

        Args:
            obs_out: ``(B, obs_output_dim)`` encoder output.
            hidden_in: ``(B, hidden_dim)`` hidden state going into this step.

        Returns:
            ``(value, hidden_out)`` where ``value`` is ``(B, 1)`` and
            ``hidden_out`` is ``(B, hidden_dim)`` — the new state to feed
            into the next step.
        """
        hidden_out = self.gru(obs_out, hidden_in)
        features = torch.cat([obs_out, hidden_out], dim=-1)
        value = self.value_mlp(features)
        return value, hidden_out
