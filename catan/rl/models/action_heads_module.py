"""
Multi-headed autoregressive action selection for 1v1 Catan.

6 heads, evaluated in order. Each downstream head can receive context
from upstream heads (e.g. the corner head knows whether we're building
a settlement or a city).

Action types and which sub-heads they use:
  0  Build Settlement  → corner head
  1  Build City        → corner head
  2  Build Road        → edge head
  3  End Turn          → (none)
  4  Move Robber       → tile head
  5  Buy Dev Card      → (none)
  6  Play Knight       → (none)
  7  Play YoP          → resource1 + resource2
  8  Play Monopoly     → resource1
  9  Play Road Builder → (none)
  10 Bank Trade        → resource1 (give) + resource2 (receive)
  11 Discard Resource  → resource1
  12 Roll Dice         → (none) — first action of agent's turn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from catan.rl.distributions import CategoricalHead, MaskedCategorical
from catan.rl.models.utils import init_weights

# Action head output sizes
N_ACTION_TYPES = 13
N_CORNERS = 54
N_EDGES = 72
N_TILES = 19
N_RESOURCES = 5


class ActionHead(nn.Module):
    """One action head: 2-layer MLP → CategoricalHead.

    If the head has upstream context (e.g. action type), the context vector
    is concatenated with the main observation output before the MLP.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            init_weights(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            init_weights(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.head = CategoricalHead(hidden_dim, output_dim, gain=0.01)

    def forward(self, main_input: torch.Tensor,
                context: Optional[torch.Tensor],
                mask: torch.Tensor) -> MaskedCategorical:
        """
        Args:
            main_input: (B, obs_output_dim) from observation module.
            context:    (B, context_dim) or None. Concatenated before MLP.
            mask:       (B, output_dim) bool mask.

        Returns:
            MaskedCategorical distribution over this head's choices.
        """
        if context is not None:
            x = torch.cat([main_input, context], dim=-1)
        else:
            x = main_input
        features = self.net(x)
        return self.head(features, mask)


class MultiActionHeads(nn.Module):
    """Orchestrates all 6 action heads with autoregressive dependencies.

    The key idea: we always evaluate ALL heads (even irrelevant ones),
    but we multiply each head's log-probability by a binary "relevance mask"
    based on the chosen action type. This way:
    - Irrelevant heads don't affect the policy gradient
    - We can batch-process all actions in a single forward pass during PPO updates
    """

    def __init__(self, obs_output_dim: int = 512, hidden_dim: int = 128):
        super().__init__()

        # Head 0: which action type? Always evaluated.
        self.type_head = ActionHead(obs_output_dim, N_ACTION_TYPES, hidden_dim)

        # Head 1: which vertex? Context = 2-dim one-hot (is_settlement, is_city)
        self.corner_head = ActionHead(obs_output_dim + 2, N_CORNERS, hidden_dim)

        # Head 2: which edge? No context needed.
        self.edge_head = ActionHead(obs_output_dim, N_EDGES, hidden_dim)

        # Head 3: which tile? No context needed.
        self.tile_head = ActionHead(obs_output_dim, N_TILES, hidden_dim)

        # Head 4: which resource (primary)? Context = 4-dim (YoP, Monopoly, Trade, Discard)
        self.resource1_head = ActionHead(obs_output_dim + 4, N_RESOURCES, hidden_dim)

        # Head 5: which resource (secondary)? Context = 4-dim type context + 5-dim res1 one-hot
        self.resource2_head = ActionHead(obs_output_dim + 4 + N_RESOURCES, N_RESOURCES, hidden_dim)

        # Log-prob relevance masks: for each action type, which heads contribute
        # to the joint log-probability? Shape: (N_ACTION_TYPES,) per head.
        # These are registered as buffers so they move to the correct device
        # automatically but are NOT trainable parameters.
        #                                  types: 0  1  2  3  4  5  6  7  8  9 10 11 12
        self.register_buffer('lp_mask_corner',    torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
        self.register_buffer('lp_mask_edge',      torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
        self.register_buffer('lp_mask_tile',      torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))
        self.register_buffer('lp_mask_resource1', torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0], dtype=torch.float32))
        self.register_buffer('lp_mask_resource2', torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=torch.float32))

    def forward(
        self,
        obs_output: torch.Tensor,
        masks: Dict[str, torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_output:    (B, obs_output_dim) from observation module.
            masks:         per-head bool masks from the environment.
            actions:       (B, 6) int64 — provide during PPO update to re-evaluate.
            deterministic: True → argmax actions (for evaluation).

        Returns:
            actions:  (B, 6) int64 composite action.
            log_prob: (B,) joint log-probability.
            entropy:  scalar — mean entropy across batch and heads.
        """
        B = obs_output.shape[0]
        device = obs_output.device
        evaluating = actions is not None  # True during PPO update

        # Helper: sample or use provided action
        def _get_action(dist: MaskedCategorical, head_idx: int):
            if evaluating:
                return actions[:, head_idx]
            elif deterministic:
                return dist.mode()
            else:
                return dist.sample()

        # ── Head 0: Action Type ──────────────────────────────────────────
        type_dist = self.type_head(obs_output, None, masks['type'])
        type_action = _get_action(type_dist, 0)  # (B,)
        type_lp = type_dist.log_prob(type_action)  # (B,)
        type_ent = type_dist.entropy()  # (B,)

        # ── Head 1: Corner ───────────────────────────────────────────────
        # Context: one-hot indicating settlement (type=0) vs city (type=1)
        corner_ctx = torch.zeros(B, 2, device=device)
        corner_ctx[type_action == 0, 0] = 1.0  # settlement
        corner_ctx[type_action == 1, 1] = 1.0  # city

        # Select the right mask: settlement locations or city locations
        # depending on the chosen action type.
        corner_mask = torch.where(
            (type_action == 0).unsqueeze(1),
            masks['corner_settlement'],
            masks['corner_city']
        )
        # For types that don't use corner, provide all-ones mask (head is ignored via lp_mask)
        uses_corner = (type_action == 0) | (type_action == 1)
        corner_mask = torch.where(
            uses_corner.unsqueeze(1),
            corner_mask,
            torch.ones(B, N_CORNERS, device=device, dtype=torch.bool)
        )

        corner_dist = self.corner_head(obs_output, corner_ctx, corner_mask)
        corner_action = _get_action(corner_dist, 1)
        corner_lp = corner_dist.log_prob(corner_action)
        corner_ent = corner_dist.entropy()

        # ── Head 2: Edge ─────────────────────────────────────────────────
        edge_mask = masks['edge'].clone()
        uses_edge = (type_action == 2)
        edge_mask = torch.where(uses_edge.unsqueeze(1), edge_mask,
                                torch.ones(B, N_EDGES, device=device, dtype=torch.bool))

        edge_dist = self.edge_head(obs_output, None, edge_mask)
        edge_action = _get_action(edge_dist, 2)
        edge_lp = edge_dist.log_prob(edge_action)
        edge_ent = edge_dist.entropy()

        # ── Head 3: Tile ─────────────────────────────────────────────────
        tile_mask = masks['tile'].clone()
        uses_tile = (type_action == 4)
        tile_mask = torch.where(uses_tile.unsqueeze(1), tile_mask,
                                torch.ones(B, N_TILES, device=device, dtype=torch.bool))

        tile_dist = self.tile_head(obs_output, None, tile_mask)
        tile_action = _get_action(tile_dist, 3)
        tile_lp = tile_dist.log_prob(tile_action)
        tile_ent = tile_dist.entropy()

        # ── Head 4: Resource Primary ─────────────────────────────────────
        # Context: 4-dim one-hot for which type triggered this head
        # Index mapping: YoP=0, Monopoly=1, Trade=2, Discard=3
        res1_ctx = torch.zeros(B, 4, device=device)
        res1_ctx[type_action == 7, 0] = 1.0   # YoP
        res1_ctx[type_action == 8, 1] = 1.0   # Monopoly
        res1_ctx[type_action == 10, 2] = 1.0  # Trade
        res1_ctx[type_action == 11, 3] = 1.0  # Discard

        # Select resource1 mask based on action type
        uses_res1 = (type_action == 7) | (type_action == 8) | (type_action == 10) | (type_action == 11)
        # Default: all valid (for types like YoP/Monopoly where any resource is fine)
        res1_mask = masks['resource1_default'].clone()
        # Override for trade: only resources you have enough of
        is_trade = (type_action == 10)
        if is_trade.any():
            res1_mask[is_trade] = masks['resource1_trade'][is_trade]
        # Override for discard: only resources you have > 0
        is_discard = (type_action == 11)
        if is_discard.any():
            res1_mask[is_discard] = masks['resource1_discard'][is_discard]
        # For types that don't use res1, use all-ones
        res1_mask = torch.where(uses_res1.unsqueeze(1), res1_mask,
                                torch.ones(B, N_RESOURCES, device=device, dtype=torch.bool))

        res1_dist = self.resource1_head(obs_output, res1_ctx, res1_mask)
        res1_action = _get_action(res1_dist, 4)
        res1_lp = res1_dist.log_prob(res1_action)
        res1_ent = res1_dist.entropy()

        # ── Head 5: Resource Secondary ───────────────────────────────────
        # Context: same 4-dim type context + one-hot of chosen resource1
        res1_onehot = F.one_hot(res1_action.long(), N_RESOURCES).float()  # (B, 5)
        res2_ctx = torch.cat([res1_ctx, res1_onehot], dim=-1)  # (B, 9)

        uses_res2 = (type_action == 7) | (type_action == 10)
        # Default for YoP: all resources valid (can pick same twice)
        res2_mask = masks['resource2_default'].clone()
        # For trade: exclude the resource being given
        if is_trade.any():
            trade_res2 = masks['resource2_default'][is_trade].clone()
            # Can't receive same resource you're giving
            trade_res2.scatter_(1, res1_action[is_trade].unsqueeze(1).long(), False)
            res2_mask[is_trade] = trade_res2
        # For non-res2 types, all-ones
        res2_mask = torch.where(uses_res2.unsqueeze(1), res2_mask,
                                torch.ones(B, N_RESOURCES, device=device, dtype=torch.bool))

        res2_dist = self.resource2_head(obs_output, res2_ctx, res2_mask)
        res2_action = _get_action(res2_dist, 5)
        res2_lp = res2_dist.log_prob(res2_action)
        res2_ent = res2_dist.entropy()

        # ── Assemble composite action ────────────────────────────────────
        if not evaluating:
            actions = torch.stack([
                type_action, corner_action, edge_action,
                tile_action, res1_action, res2_action
            ], dim=-1)  # (B, 6)

        # ── Joint log-probability ────────────────────────────────────────
        # Each head's log_prob is weighted by whether it's relevant for the
        # chosen action type. We index into the relevance masks with type_action.
        # .detach() prevents gradients from flowing through the mask selection.
        lp_m_corner = self.lp_mask_corner[type_action].detach()    # (B,)
        lp_m_edge = self.lp_mask_edge[type_action].detach()
        lp_m_tile = self.lp_mask_tile[type_action].detach()
        lp_m_res1 = self.lp_mask_resource1[type_action].detach()
        lp_m_res2 = self.lp_mask_resource2[type_action].detach()

        log_prob = (type_lp
                    + lp_m_corner * corner_lp
                    + lp_m_edge * edge_lp
                    + lp_m_tile * tile_lp
                    + lp_m_res1 * res1_lp
                    + lp_m_res2 * res2_lp)  # (B,)

        # ── Entropy (same weighting) ─────────────────────────────────────
        entropy_per_sample = (type_ent
                              + lp_m_corner * corner_ent
                              + lp_m_edge * edge_ent
                              + lp_m_tile * tile_ent
                              + lp_m_res1 * res1_ent
                              + lp_m_res2 * res2_ent)
        entropy = entropy_per_sample.mean()  # scalar

        return actions, log_prob, entropy
