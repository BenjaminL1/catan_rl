"""Action / value / belief heads for the v2 policy network.

Design notes:

  * Six autoregressive action heads sit on top of the fused 512-d trunk:
    ``type`` (13), ``corner`` (54), ``edge`` (72), ``tile`` (19),
    ``resource1`` (5), ``resource2`` (5).
  * The ``type``, ``edge``, ``tile`` heads have no per-action context and
    use a plain 2-layer MLP.
  * The ``corner``, ``resource1``, ``resource2`` heads use FiLM/AdaLN
    conditioning on a per-action context vector — ``(1 + γ) ⊙ LN(x) + β``
    with ``γ`` initialised to zero (so the modulation is identity at the
    start of training; safe drop-in replacement for concat-MLP).
  * Per-head "relevance" weights tell the trainer which heads contribute
    to the joint log-prob for a given action type. These live in a
    registered buffer so they're moved across devices alongside the model.
  * The value and belief heads are simple MLPs that read off the same
    512-d trunk vector. Belief head outputs 5-way logits over dev-card
    types; the trainer computes soft cross-entropy against the env's
    ground-truth distribution.
"""  # noqa: RUF002

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from catan_rl.policy.obs_schema import (
    CORNER_CONTEXT_DIM,
    HEAD_DIMS,
    N_ACTION_TYPES,
    N_DEV_TYPES,
    N_RESOURCES,
    RESOURCE2_RES1_CONTEXT_DIM,
    RESOURCE_CONTEXT_DIM,
    ActionType,
)

_LOGIT_NEG_INF = -1e9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Numerically-safe ``log_softmax`` with hard masking.

    Slots where ``mask == 0`` are pushed to ``-inf`` before the softmax so
    they contribute zero probability. Rows with no valid entries return
    a uniform log-prob (avoids ``nan`` from a degenerate softmax).
    """
    masked = logits.masked_fill(~mask, _LOGIT_NEG_INF)
    any_valid = mask.any(dim=-1, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    return torch.log_softmax(safe, dim=-1)


def masked_sample(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from the masked categorical; return ``(sample, log_prob)``."""
    log_probs = masked_log_softmax(logits, mask)
    probs = log_probs.exp()
    dist = Categorical(probs=probs)
    sample = dist.sample()
    return sample, log_probs.gather(-1, sample.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Head primitives
# ---------------------------------------------------------------------------


class _SimpleHead(nn.Module):
    """2-layer MLP head with no per-action context."""

    def __init__(self, trunk_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FiLMHead(nn.Module):
    """Head with FiLM/AdaLN context modulation.

    Architecture:
        x  -> LN -> (1 + γ) ⊙ x + β   -> GeLU -> Linear -> logits
        γ, β = MLP(context); γ-init=0 → identity at construction.
    """  # noqa: RUF002

    def __init__(
        self, trunk_dim: int, out_dim: int, context_dim: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(trunk_dim)
        self.film = nn.Linear(context_dim, 2 * trunk_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(trunk_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(context).chunk(2, dim=-1)
        modulated = (1.0 + gamma) * self.norm(x) + beta
        return self.proj(modulated)


#: Per-node state width consumed by the pointer readouts — the GNN's
#: hidden_dim (``GraphEncoder.hidden_dim``). Kept as a module constant so the
#: heads default consistently with the encoder.
POINTER_NODE_DIM = 64


class _PointerHead(nn.Module):
    """Per-node pointer readout: ``logit_i = MLP([trunk_proj, node_i])`` (D1).

    A single shared ``trunk_proj`` is broadcast over the N per-node states and
    concatenated with each node's GNN state, then an MLP scores each node. The
    output dim equals the number of nodes (54/72/19); masking is applied by the
    caller exactly as before. No per-action context (used for edge / tile).
    """

    def __init__(
        self, trunk_dim: int, node_dim: int, proj_dim: int = 64, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.trunk_proj = nn.Linear(trunk_dim, proj_dim)
        # Final projection to a per-node scalar has NO bias: a single bias would
        # add the SAME constant to every node's logit, which is softmax-invariant
        # (identically zero gradient, zero effect) — a dead parameter.
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim + node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, trunk: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """Args: trunk (B, trunk_dim), nodes (B, N, node_dim). Returns (B, N)."""
        tp = self.trunk_proj(trunk).unsqueeze(1).expand(-1, nodes.size(1), -1)
        return self.mlp(torch.cat([tp, nodes], dim=-1)).squeeze(-1)


class _CornerPointerHead(nn.Module):
    """Corner pointer readout with FiLM context modulating ``trunk_proj``.

    The FiLM context (``[settlement, city, is_setup]``) modulates the projected
    trunk vector BEFORE the per-node concat (D1), so settlement/city and
    snake-draft-setup placement each get dedicated modulation. ``γ`` init-0 →
    identity at construction.
    """  # noqa: RUF002

    def __init__(
        self,
        trunk_dim: int,
        node_dim: int,
        context_dim: int,
        proj_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.trunk_proj = nn.Linear(trunk_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.film = nn.Linear(context_dim, 2 * proj_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # Final per-node scalar projection has no bias — softmax-invariant dead
        # weight (see :class:`_PointerHead`).
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim + node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self, trunk: torch.Tensor, nodes: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        tp = self.trunk_proj(trunk)
        gamma, beta = self.film(context).chunk(2, dim=-1)
        tp = (1.0 + gamma) * self.norm(tp) + beta
        tp = tp.unsqueeze(1).expand(-1, nodes.size(1), -1)
        return self.mlp(torch.cat([tp, nodes], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# CatanActionHeads
# ---------------------------------------------------------------------------


class CatanActionHeads(nn.Module):
    """Six autoregressive action heads with masking + FiLM context."""

    # Declared for mypy: register_buffer attributes are otherwise typed
    # ``Tensor | Module`` via nn.Module.__getattr__ (mypy 2.x strictness).
    head_relevance: torch.Tensor
    resource_context_idx: torch.Tensor

    def __init__(
        self, trunk_dim: int = 512, hidden_dim: int = 128, node_dim: int = POINTER_NODE_DIM
    ) -> None:
        super().__init__()
        self.type_head = _SimpleHead(trunk_dim, N_ACTION_TYPES, hidden_dim)
        # The three LOCATION heads are pointer readouts over the GNN's per-node
        # states (D1): edge/tile take no context; corner FiLM-modulates its
        # trunk_proj with [settlement, city, is_setup].
        self.edge_head = _PointerHead(trunk_dim, node_dim, hidden_dim=hidden_dim)
        self.tile_head = _PointerHead(trunk_dim, node_dim, hidden_dim=hidden_dim)
        self.corner_head = _CornerPointerHead(
            trunk_dim, node_dim, CORNER_CONTEXT_DIM, hidden_dim=hidden_dim
        )
        self.node_dim = node_dim
        self.resource1_head = _FiLMHead(trunk_dim, N_RESOURCES, RESOURCE_CONTEXT_DIM, hidden_dim)
        self.resource2_head = _FiLMHead(
            trunk_dim,
            N_RESOURCES,
            RESOURCE_CONTEXT_DIM + RESOURCE2_RES1_CONTEXT_DIM,
            hidden_dim,
        )

        # Per-head relevance: 1 if the head's output contributes to the joint
        # log-prob for this action type, else 0.
        relevance = torch.zeros(N_ACTION_TYPES, 6, dtype=torch.float32)
        # type is always relevant.
        relevance[:, 0] = 1.0
        # corner relevant on BuildSettlement / BuildCity.
        relevance[ActionType.BUILD_SETTLEMENT, 1] = 1.0
        relevance[ActionType.BUILD_CITY, 1] = 1.0
        # edge relevant on BuildRoad.
        relevance[ActionType.BUILD_ROAD, 2] = 1.0
        # tile relevant on MoveRobber and PlayKnight (knight triggers robber).
        relevance[ActionType.MOVE_ROBBER, 3] = 1.0
        relevance[ActionType.PLAY_KNIGHT, 3] = 1.0
        # resource1 relevant on YoP / Monopoly / BankTrade / Discard.
        relevance[ActionType.PLAY_YOP, 4] = 1.0
        relevance[ActionType.PLAY_MONOPOLY, 4] = 1.0
        relevance[ActionType.BANK_TRADE, 4] = 1.0
        relevance[ActionType.DISCARD, 4] = 1.0
        # resource2 relevant on YoP / BankTrade (Monopoly and Discard take only one resource).
        relevance[ActionType.PLAY_YOP, 5] = 1.0
        relevance[ActionType.BANK_TRADE, 5] = 1.0
        self.register_buffer("head_relevance", relevance)

        # Maps each action_type -> its 4-dim resource-context one-hot index
        # (YoP=0, Mono=1, Trade=2, Discard=3). -1 for "not applicable".
        res_ctx_idx = torch.full((N_ACTION_TYPES,), -1, dtype=torch.long)
        res_ctx_idx[ActionType.PLAY_YOP] = 0
        res_ctx_idx[ActionType.PLAY_MONOPOLY] = 1
        res_ctx_idx[ActionType.BANK_TRADE] = 2
        res_ctx_idx[ActionType.DISCARD] = 3
        self.register_buffer("resource_context_idx", res_ctx_idx)

    # ------------------------------------------------------------------
    # Context constructors
    # ------------------------------------------------------------------

    def _corner_context(
        self, type_idx: torch.Tensor, is_setup: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``[settlement, city, is_setup]`` context for the corner pointer head.

        ``is_setup`` is threaded from env state (obs key) at both training and
        inference (D2); when absent (legacy callers / ablations) it defaults to
        0.0 (main-phase modulation).
        """
        is_settle = (type_idx == ActionType.BUILD_SETTLEMENT).float().unsqueeze(-1)
        is_city = (type_idx == ActionType.BUILD_CITY).float().unsqueeze(-1)
        if is_setup is None:
            setup_col = torch.zeros_like(is_settle)
        else:
            setup_col = is_setup.reshape(is_settle.shape).to(is_settle.dtype)
        return torch.cat([is_settle, is_city, setup_col], dim=-1)

    def _resource1_context(self, type_idx: torch.Tensor) -> torch.Tensor:
        """4-dim one-hot ``[YoP, Mono, Trade, Discard]``."""
        ctx = torch.zeros(type_idx.shape + (RESOURCE_CONTEXT_DIM,), device=type_idx.device)  # noqa: RUF005
        slot = self.resource_context_idx[type_idx]
        valid = slot >= 0
        if valid.any():
            ctx[valid] = torch.nn.functional.one_hot(slot[valid], RESOURCE_CONTEXT_DIM).float()
        return ctx

    def _resource2_context(self, type_idx: torch.Tensor, res1_idx: torch.Tensor) -> torch.Tensor:
        """4-dim type one-hot ++ 5-dim res1 one-hot."""
        type_ctx = self._resource1_context(type_idx)
        res1_oh = torch.nn.functional.one_hot(res1_idx, N_RESOURCES).float()
        return torch.cat([type_ctx, res1_oh], dim=-1)

    # ------------------------------------------------------------------
    # Mask selection
    # ------------------------------------------------------------------

    @staticmethod
    def _corner_mask(type_idx: torch.Tensor, masks: dict[str, torch.Tensor]) -> torch.Tensor:
        is_settle = (type_idx == ActionType.BUILD_SETTLEMENT).unsqueeze(-1)
        is_city = (type_idx == ActionType.BUILD_CITY).unsqueeze(-1)
        return torch.where(
            is_settle,
            masks["corner_settlement"],
            torch.where(is_city, masks["corner_city"], masks["corner_settlement"]),
        )

    @staticmethod
    def _resource1_mask(type_idx: torch.Tensor, masks: dict[str, torch.Tensor]) -> torch.Tensor:
        is_trade = (type_idx == ActionType.BANK_TRADE).unsqueeze(-1)
        is_discard = (type_idx == ActionType.DISCARD).unsqueeze(-1)
        default = masks["resource1_default"]
        return torch.where(
            is_trade,
            masks["resource1_trade"],
            torch.where(is_discard, masks["resource1_discard"], default),
        )

    @staticmethod
    def _resource2_mask(
        type_idx: torch.Tensor,
        res1_idx: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # BANK_TRADE: forbid r2 == r1 (the engine accepts the degenerate
        # trade silently as "pay 2-4 of r1, get 1 of r1 back" — a
        # strictly losing action the policy should never sample). YOP
        # is unaffected since picking two of the same resource is a
        # legitimate strategic choice there.
        mask = masks["resource2_default"].clone()
        is_trade = type_idx == ActionType.BANK_TRADE
        if is_trade.any():
            rows = torch.nonzero(is_trade, as_tuple=True)[0]
            mask[rows, res1_idx[rows]] = False
        return mask

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        trunk: torch.Tensor,
        masks: dict[str, torch.Tensor],
        nodes: dict[str, torch.Tensor],
        is_setup: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressive sample.

        Args:
            trunk: (B, trunk_dim).
            masks: dict of bool tensors per :data:`MASK_KEYS` schema.
            nodes: per-node GNN states — ``{"v": (B,54,D), "e": (B,72,D),
                "h": (B,19,D)}`` — for the corner/edge/tile pointer readouts.
            is_setup: (B,) / (B,1) float, threaded to the corner FiLM context.
        Returns:
            ``(action, joint_log_prob, joint_entropy, per_head_log_prob)``
            where ``action`` is (B, 6) int64, ``joint_log_prob`` /
            ``joint_entropy`` are (B,) floats, and
            ``per_head_log_prob`` is (B, 6) — the relevance-aware PPO
            ratio + Phase 5 per-head diagnostics need this stored in
            the rollout buffer.
        """
        type_logits = self.type_head(trunk)
        type_logp = masked_log_softmax(type_logits, masks["type"])
        type_dist = Categorical(probs=type_logp.exp())
        type_idx = type_dist.sample()

        corner_ctx = self._corner_context(type_idx, is_setup)
        corner_logits = self.corner_head(trunk, nodes["v"], corner_ctx)
        corner_mask = self._corner_mask(type_idx, masks)
        corner_logp = masked_log_softmax(corner_logits, corner_mask)
        corner_idx = Categorical(probs=corner_logp.exp()).sample()

        edge_logits = self.edge_head(trunk, nodes["e"])
        edge_logp = masked_log_softmax(edge_logits, masks["edge"])
        edge_idx = Categorical(probs=edge_logp.exp()).sample()

        tile_logits = self.tile_head(trunk, nodes["h"])
        tile_logp = masked_log_softmax(tile_logits, masks["tile"])
        tile_idx = Categorical(probs=tile_logp.exp()).sample()

        res1_ctx = self._resource1_context(type_idx)
        res1_logits = self.resource1_head(trunk, res1_ctx)
        res1_mask = self._resource1_mask(type_idx, masks)
        res1_logp = masked_log_softmax(res1_logits, res1_mask)
        res1_idx = Categorical(probs=res1_logp.exp()).sample()

        res2_ctx = self._resource2_context(type_idx, res1_idx)
        res2_logits = self.resource2_head(trunk, res2_ctx)
        res2_logp = masked_log_softmax(res2_logits, self._resource2_mask(type_idx, res1_idx, masks))
        res2_idx = Categorical(probs=res2_logp.exp()).sample()

        action = torch.stack([type_idx, corner_idx, edge_idx, tile_idx, res1_idx, res2_idx], dim=-1)

        # Joint log-prob = sum of per-head log-probs weighted by relevance.
        per_head_logp = torch.stack(
            [
                type_logp.gather(-1, type_idx.unsqueeze(-1)).squeeze(-1),
                corner_logp.gather(-1, corner_idx.unsqueeze(-1)).squeeze(-1),
                edge_logp.gather(-1, edge_idx.unsqueeze(-1)).squeeze(-1),
                tile_logp.gather(-1, tile_idx.unsqueeze(-1)).squeeze(-1),
                res1_logp.gather(-1, res1_idx.unsqueeze(-1)).squeeze(-1),
                res2_logp.gather(-1, res2_idx.unsqueeze(-1)).squeeze(-1),
            ],
            dim=-1,
        )
        relevance = self.head_relevance[type_idx]
        joint_logp = (per_head_logp * relevance).sum(dim=-1)

        per_head_ent = torch.stack(
            [
                -(type_logp.exp() * type_logp).sum(dim=-1),
                -(corner_logp.exp() * corner_logp).sum(dim=-1),
                -(edge_logp.exp() * edge_logp).sum(dim=-1),
                -(tile_logp.exp() * tile_logp).sum(dim=-1),
                -(res1_logp.exp() * res1_logp).sum(dim=-1),
                -(res2_logp.exp() * res2_logp).sum(dim=-1),
            ],
            dim=-1,
        )
        joint_entropy = (per_head_ent * relevance).sum(dim=-1)

        return action, joint_logp, joint_entropy, per_head_logp

    # ------------------------------------------------------------------
    # Off-policy evaluation (PPO / BC)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        trunk: torch.Tensor,
        action: torch.Tensor,
        masks: dict[str, torch.Tensor],
        nodes: dict[str, torch.Tensor],
        is_setup: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute per-head log-probs and entropies for *given* actions.

        Used by PPO (importance ratios) and by BC (per-head CE).

        Args:
            trunk: (B, trunk_dim).
            action: (B, 6) int64.
            masks: dict of bool tensors per :data:`MASK_KEYS` schema.
            nodes: per-node GNN states for the corner/edge/tile pointer heads.
            is_setup: (B,) / (B,1) float threaded to the corner FiLM context.
        Returns:
            dict with keys ``log_prob`` (B,), ``entropy`` (B,),
            ``per_head_log_prob`` (B, 6), ``per_head_entropy`` (B, 6),
            ``relevance`` (B, 6).
        """
        type_idx, corner_idx, edge_idx, tile_idx, res1_idx, res2_idx = action.unbind(dim=-1)

        type_logp = masked_log_softmax(self.type_head(trunk), masks["type"])
        corner_ctx = self._corner_context(type_idx, is_setup)
        corner_logits = self.corner_head(trunk, nodes["v"], corner_ctx)
        corner_logp = masked_log_softmax(corner_logits, self._corner_mask(type_idx, masks))
        edge_logp = masked_log_softmax(self.edge_head(trunk, nodes["e"]), masks["edge"])
        tile_logp = masked_log_softmax(self.tile_head(trunk, nodes["h"]), masks["tile"])
        res1_logits = self.resource1_head(trunk, self._resource1_context(type_idx))
        res1_logp = masked_log_softmax(res1_logits, self._resource1_mask(type_idx, masks))
        res2_logits = self.resource2_head(trunk, self._resource2_context(type_idx, res1_idx))
        res2_logp = masked_log_softmax(res2_logits, self._resource2_mask(type_idx, res1_idx, masks))

        per_head_logp = torch.stack(
            [
                type_logp.gather(-1, type_idx.unsqueeze(-1)).squeeze(-1),
                corner_logp.gather(-1, corner_idx.unsqueeze(-1)).squeeze(-1),
                edge_logp.gather(-1, edge_idx.unsqueeze(-1)).squeeze(-1),
                tile_logp.gather(-1, tile_idx.unsqueeze(-1)).squeeze(-1),
                res1_logp.gather(-1, res1_idx.unsqueeze(-1)).squeeze(-1),
                res2_logp.gather(-1, res2_idx.unsqueeze(-1)).squeeze(-1),
            ],
            dim=-1,
        )

        per_head_ent = torch.stack(
            [
                -(type_logp.exp() * type_logp).sum(dim=-1),
                -(corner_logp.exp() * corner_logp).sum(dim=-1),
                -(edge_logp.exp() * edge_logp).sum(dim=-1),
                -(tile_logp.exp() * tile_logp).sum(dim=-1),
                -(res1_logp.exp() * res1_logp).sum(dim=-1),
                -(res2_logp.exp() * res2_logp).sum(dim=-1),
            ],
            dim=-1,
        )

        relevance = self.head_relevance[type_idx]
        joint_logp = (per_head_logp * relevance).sum(dim=-1)
        joint_entropy = (per_head_ent * relevance).sum(dim=-1)

        return {
            "log_prob": joint_logp,
            "entropy": joint_entropy,
            "per_head_log_prob": per_head_logp,
            "per_head_entropy": per_head_ent,
            "relevance": relevance,
        }


# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------


class ValueHead(nn.Module):
    """Shared-encoder value head: 512 -> 256 -> 128 -> 1."""

    def __init__(self, trunk_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Auxiliary value-target head (pointer-arch fork, D4 — representation-shaping)
# ---------------------------------------------------------------------------


class AuxValueHead(nn.Module):
    """Small head off the 512-d trunk predicting the discounted outcome z_disc.

    Purpose is representation-SHAPING (D4): co-trained from update 0 with a low
    coef so the shared trunk carries outcome-predictive structure. It is
    soft-additive and does NOT gate the fork; the trainer folds its loss only
    when the coef is non-zero (coef-0 is byte-identical to no head). Distinct
    from :class:`ValueHead`, whose output feeds PPO/search; this head is a pure
    auxiliary regression target and its output is never used for control.

    Final-layer init gain=0.01 so predictions start near zero.
    """

    def __init__(self, trunk_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        final = nn.Linear(hidden_dim, 1)
        self.net = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.GELU(),
            final,
        )
        nn.init.normal_(final.weight, std=0.01)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Belief head (Phase 2.5b — opponent hidden dev-card type distribution)
# ---------------------------------------------------------------------------


class BeliefHead(nn.Module):
    """5-way logits over dev-card types; trained via soft CE on env GT.

    Final layer init gain=0.01 so the predictions start near-uniform —
    loss starts at ``log(5) ≈ 1.609`` rather than a wild value that
    swamps the policy loss in the first updates.
    """

    def __init__(self, trunk_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        # Name the final layer so its .weight/.bias are typed nn.Linear
        # (mypy 2.x types nn.Sequential.__getitem__ as bare Module).
        final = nn.Linear(hidden_dim, N_DEV_TYPES)
        self.net = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.GELU(),
            final,
        )
        nn.init.normal_(final.weight, std=0.01)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Silence "unused import" for the head-dim contract re-export.
_ = HEAD_DIMS
