"""
Top-level policy: dict observation → actions + value estimate.

ObservationModule expects a Charlesworth-style dict obs.

Phase 2 additions (all opt-in, defaults preserve Phase 0/1 behavior):

  - **2.4 AdaLN-conditioned action heads.** ``action_head_film=True`` flips
    every context-using head from concat to FiLM modulation.
  - **2.5 Decoupled value tower.** ``value_head_mode='decoupled'`` builds a
    second ``ObservationModule`` exclusively for the value head, breaking
    the feature-sharing tug-of-war between policy gradient and value loss.
    ``'shared'`` (default) keeps the legacy single-encoder design.
"""

import torch
import torch.nn as nn

from catan_rl.models.action_heads_module import MultiActionHeads
from catan_rl.models.belief_head import BeliefHead
from catan_rl.models.observation_module import ObservationModule
from catan_rl.models.opponent_action_head import OpponentActionHead
from catan_rl.models.recurrent_value import RecurrentValueHead
from catan_rl.models.utils import init_weights


def _build_value_net(in_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    """Linear→LayerNorm→ReLU stack ending in a single scalar (gain=1.0)."""
    layers: list[nn.Module] = []
    prev_dim = in_dim
    for hdim in hidden_dims:
        layers.append(init_weights(nn.Linear(prev_dim, hdim)))
        layers.append(nn.LayerNorm(hdim))
        layers.append(nn.ReLU())
        prev_dim = hdim
    final = nn.Linear(prev_dim, 1)
    init_weights(final, gain=1.0)
    layers.append(final)
    return nn.Sequential(*layers)


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
        action_head_film: bool = False,
        value_head_mode: str = "shared",
        use_belief_head: bool = False,
        belief_head_hidden_dim: int = 128,
        use_opponent_action_head: bool = False,
        opponent_action_head_hidden_dim: int = 128,
        use_recurrent_value: bool = False,
        gru_hidden_dim: int = 64,
        **obs_module_kwargs,
    ):
        super().__init__()
        if value_head_mode not in ("shared", "decoupled"):
            raise ValueError(
                f"value_head_mode must be 'shared' or 'decoupled', got {value_head_mode!r}"
            )
        self.value_head_mode = value_head_mode

        # Policy-side observation encoder (also used by the value head when
        # ``value_head_mode='shared'``).
        self.observation_module = ObservationModule(
            obs_output_dim=obs_output_dim, **obs_module_kwargs
        )

        # Phase 2.5: decoupled value tower has its own encoder. Same
        # architecture as the policy encoder — separate weights only — so the
        # two heads stop fighting over a single shared trunk.
        if value_head_mode == "decoupled":
            self.value_observation_module = ObservationModule(
                obs_output_dim=obs_output_dim, **obs_module_kwargs
            )
        else:
            self.value_observation_module = None

        # Multi-head action selection (Phase 2.4 AdaLN-conditioned when on).
        self.action_heads = MultiActionHeads(
            obs_output_dim=obs_output_dim,
            hidden_dim=action_head_hidden_dim,
            film=action_head_film,
        )

        # Value tower. Phase 4.2 introduces an opt-in recurrent variant
        # (``use_recurrent_value=True``) that wraps the value MLP in a
        # GRUCell. When enabled, ``value_net`` is *not* built — all value
        # evaluations go through ``recurrent_value_head`` instead, which
        # owns its own MLP. Callers that need a value forward pass MUST
        # supply a hidden state when ``use_recurrent_value=True``.
        self.use_recurrent_value = bool(use_recurrent_value)
        self.gru_hidden_dim = int(gru_hidden_dim)
        if self.use_recurrent_value:
            self.recurrent_value_head = RecurrentValueHead(
                obs_output_dim,
                hidden_dim=self.gru_hidden_dim,
                value_hidden_dims=value_hidden_dims,
            )
            self.value_net = None  # type: ignore[assignment]
        else:
            self.recurrent_value_head = None
            self.value_net = _build_value_net(obs_output_dim, value_hidden_dims)

        # Phase 2.5b: optional opponent-belief head reading the policy
        # encoder's output. Trainer-only signal — no inference dependency.
        self.use_belief_head = bool(use_belief_head)
        self.belief_head = (
            BeliefHead(obs_output_dim, hidden_dim=belief_head_hidden_dim)
            if self.use_belief_head
            else None
        )

        # Phase 2.5c: optional opponent-action auxiliary head — predicts
        # the opponent's next action TYPE distribution (13-way). Same
        # plumbing pattern as the belief head; loss only fires on rollouts
        # where the opponent is a historical league policy (filter is
        # applied trainer-side via ``opp_action_target_valid``).
        self.use_opponent_action_head = bool(use_opponent_action_head)
        self.opponent_action_head = (
            OpponentActionHead(obs_output_dim, hidden_dim=opponent_action_head_hidden_dim)
            if self.use_opponent_action_head
            else None
        )

    def _value_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the value-head's observation encoder.

        Shared mode reuses the policy encoder's output (caller already has it).
        Decoupled mode runs a separate encoder so the value gradient never
        touches the policy trunk.
        """
        assert self.value_observation_module is not None
        return self.value_observation_module(obs)

    def act(
        self,
        obs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        deterministic: bool = False,
        value_hidden_in: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Used during rollout collection (playing games to gather experience).

        Args:
            obs:             dict of batched observations from environment.
            masks:           per-head action masks from environment.
            deterministic:   True for evaluation (argmax), False for training (sample).
            value_hidden_in: ``(B, gru_hidden_dim)`` hidden state for the
                recurrent value head. Only required when
                ``use_recurrent_value=True`` — caller (the trainer) tracks
                per-env state across rollout steps.

        Returns:
            Without recurrent value: ``(actions, value, log_prob)``.
            With recurrent value: ``(actions, value, log_prob, hidden_out)``
            where ``hidden_out`` is the new GRU state to feed back at the
            next step.
        """
        obs_out = self.observation_module(obs)
        if self.use_recurrent_value:
            if value_hidden_in is None:
                value_hidden_in = self.recurrent_value_head.initial_hidden(
                    obs_out.shape[0], device=obs_out.device
                )
            value, hidden_out = self.recurrent_value_head(obs_out, value_hidden_in)
            actions, log_prob, _ = self.action_heads(obs_out, masks, deterministic=deterministic)
            return actions, value, log_prob, hidden_out
        if self.value_head_mode == "decoupled":
            value = self.value_net(self._value_features(obs))
        else:
            value = self.value_net(obs_out)
        actions, log_prob, _ = self.action_heads(obs_out, masks, deterministic=deterministic)
        return actions, value, log_prob

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        actions: torch.Tensor,
        return_per_head: bool = False,
        return_belief_logits: bool = False,
        return_opp_action_logits: bool = False,
    ) -> tuple:
        """Used during PPO update to re-evaluate previously taken actions.

        Args:
            obs:     dict of stored observations from rollout buffer.
            masks:   per-head masks that were active when these actions were taken.
            actions: (B, 6) the composite actions that were actually taken.
            return_per_head: When True, also returns the per-head entropy /
                weight / log-prob dict from MultiActionHeads. Used by Phase 0's
                per-head entropy logging for collapse detection.
            return_belief_logits: When True (and the policy has a belief
                head), append ``(B, 5)`` belief logits to the return tuple.
                Trainer uses these against ``obs['belief_target']`` for the
                Phase 2.5b auxiliary loss.

        Returns:
            Default::
                value:    (B, 1) current value estimate for these states.
                log_prob: (B,) log-probability of these actions under current policy.
                entropy:  scalar — entropy bonus to encourage exploration.

            With ``return_per_head=True``: (value, log_prob, entropy, per_head_dict).
            With ``return_belief_logits=True``: belief_logits ``(B, 5)``
            appended as the final element. Both flags can be combined.
        """
        obs_out = self.observation_module(obs)
        if self.use_recurrent_value:
            # Replay one GRU step from the buffered ``h_t``. No BPTT past
            # this single step — gradient backs into the GRU weights but
            # not into the previous timestep's hidden state.
            value_hidden_in = obs.get("value_hidden_in")
            if value_hidden_in is None:
                value_hidden_in = self.recurrent_value_head.initial_hidden(
                    obs_out.shape[0], device=obs_out.device
                )
            value, _hidden_out = self.recurrent_value_head(obs_out, value_hidden_in)
        elif self.value_head_mode == "decoupled":
            value = self.value_net(self._value_features(obs))
        else:
            value = self.value_net(obs_out)

        belief_logits: torch.Tensor | None = None
        if return_belief_logits and self.belief_head is not None:
            belief_logits = self.belief_head(obs_out)
        opp_action_logits: torch.Tensor | None = None
        if return_opp_action_logits and self.opponent_action_head is not None:
            opp_action_logits = self.opponent_action_head(obs_out)

        # Build the return tuple incrementally so any combination of the
        # three optional fields lands in a stable order:
        #   (value, log_prob, entropy[, per_head][, belief_logits][, opp_action_logits])
        if return_per_head:
            _, log_prob, entropy, per_head = self.action_heads(
                obs_out, masks, actions=actions, return_per_head=True
            )
            extras: tuple = (per_head,)
        else:
            _, log_prob, entropy = self.action_heads(obs_out, masks, actions=actions)
            extras = ()
        if return_belief_logits:
            extras = extras + (belief_logits,)
        if return_opp_action_logits:
            extras = extras + (opp_action_logits,)
        if extras:
            return (value, log_prob, entropy) + extras
        return value, log_prob, entropy

    def get_value(
        self,
        obs: dict[str, torch.Tensor],
        value_hidden_in: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Cheap value-only forward pass (no action heads).

        Decoupled mode skips the policy encoder entirely. Recurrent mode
        runs one GRU step and additionally returns the new hidden state.
        """
        if self.use_recurrent_value:
            obs_out = self.observation_module(obs)
            if value_hidden_in is None:
                value_hidden_in = self.recurrent_value_head.initial_hidden(
                    obs_out.shape[0], device=obs_out.device
                )
            value, hidden_out = self.recurrent_value_head(obs_out, value_hidden_in)
            return value, hidden_out
        if self.value_head_mode == "decoupled":
            return self.value_net(self._value_features(obs))
        return self.value_net(self.observation_module(obs))
