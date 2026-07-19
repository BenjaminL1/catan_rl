"""Top-level v2 policy network.

Assembles the encoders (TileEncoder + GraphEncoder + Player/Dev encoders +
opponent-id embedding) into a shared 512-d trunk, then sends the trunk to
the six action heads, the value head, and the belief head.

Targeted parameter count: ~1.5M (Phase 2 of the v2 design doc). The
fusion-bottleneck width is the easiest knob if the count drifts.

Forward returns a dict so callers (PPO trainer, BC trainer, MCTS) can pull
out only the heads they need.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from catan_rl.policy.encoders import (
    CountDevEncoder,
    GraphEncoder,
    TileEncoder,
    curr_player_encoder,
    opp_player_encoder,
)
from catan_rl.policy.heads import AuxValueHead, BeliefHead, CatanActionHeads, ValueHead
from catan_rl.policy.obs_schema import (
    GLOBAL_DIM,
    N_EDGES,
    N_OPP_KINDS,
    N_OPP_POLICY_SLOTS,
    N_TILES,
    N_VERTICES,
)


class _OppIdEmbedding(nn.Module):
    """Phase 3.6 opponent-identity embedding.

    Two parallel embeddings — one over the discrete "kind" (random,
    heuristic, league, etc.) and one over the league-policy slot — are
    summed in halves and concatenated. The env may stochastically mask
    both fields to UNKNOWN to keep the policy robust to eval-time games
    where opponent identity is hidden.
    """

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"opp-id embedding dim must be even, got {dim}")
        half = dim // 2
        self.kind_emb = nn.Embedding(N_OPP_KINDS, half)
        self.policy_emb = nn.Embedding(N_OPP_POLICY_SLOTS, half)
        nn.init.normal_(self.kind_emb.weight, std=0.02)
        nn.init.normal_(self.policy_emb.weight, std=0.02)
        self.out_dim = dim

    def forward(self, kind_idx: torch.Tensor, policy_idx: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.kind_emb(kind_idx), self.policy_emb(policy_idx)], dim=-1)


class CatanPolicy(nn.Module):
    """The full v2 policy: obs dict -> (action heads, value, belief)."""

    def __init__(
        self,
        tile_out_dim: int = 25,
        graph_out_dim: int = 64,
        dev_out_dim: int = 16,
        # Panel revision 2026-05-13: 16 -> 8 (D7 compromise). 4 dim per
        # embedding half (kind + policy-slot) is enough at our league size,
        # with opp_id_mask_prob=0.40 providing additional regularization.
        opp_id_dim: int = 8,
        trunk_dim: int = 512,
        # Toggles for ablations + Step-1 compatibility with deferred features.
        use_graph_encoder: bool = True,
        use_opp_id: bool = True,
        use_belief_head: bool = True,
        use_aux_value_head: bool = True,
        # Vertex/edge feature dims for the graph encoder (matched to obs).
        vertex_in_dim: int = 16,
        edge_in_dim: int = 16,
        hex_in_dim: int = 19,
    ) -> None:
        super().__init__()
        self.use_graph_encoder = use_graph_encoder
        self.use_opp_id = use_opp_id
        self.use_belief_head = use_belief_head
        self.use_aux_value_head = use_aux_value_head

        self.tile_encoder = TileEncoder(out_dim=tile_out_dim)
        self.curr_player_enc = curr_player_encoder()
        self.opp_player_enc = opp_player_encoder()
        self.curr_dev_enc = CountDevEncoder(out_dim=dev_out_dim)
        self.opp_dev_enc = CountDevEncoder(out_dim=dev_out_dim)

        fused_dim = (
            N_TILES * tile_out_dim
            + self.curr_player_enc.out_dim
            + self.opp_player_enc.out_dim
            + 2 * dev_out_dim
            # POV-neutral global block (bank + dev-deck + reserved) concatenated
            # raw into the fusion input (D3.3).
            + GLOBAL_DIM
        )
        # Per-node state width for the pointer readouts (D1) — the GNN hidden
        # dim. Falls back to the GraphEncoder default when the graph is disabled
        # (ablation): the pointer heads then read zero node states.
        node_dim = 64
        if use_graph_encoder:
            self.graph_encoder = GraphEncoder(
                hex_in_dim=hex_in_dim,
                vertex_in_dim=vertex_in_dim,
                edge_in_dim=edge_in_dim,
                out_dim=graph_out_dim,
            )
            fused_dim += graph_out_dim
            node_dim = self.graph_encoder.hidden_dim
        else:
            self.graph_encoder = None  # type: ignore[assignment]
        self._node_dim = node_dim

        if use_opp_id:
            self.opp_id_emb = _OppIdEmbedding(dim=opp_id_dim)
            fused_dim += opp_id_dim
        else:
            self.opp_id_emb = None  # type: ignore[assignment]

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
        )

        self.action_heads = CatanActionHeads(trunk_dim=trunk_dim, node_dim=node_dim)
        self.value_head = ValueHead(trunk_dim=trunk_dim)
        self.belief_head = BeliefHead(trunk_dim=trunk_dim) if use_belief_head else None
        self.aux_value_head = AuxValueHead(trunk_dim=trunk_dim) if use_aux_value_head else None

        self.trunk_dim = trunk_dim

    # ------------------------------------------------------------------
    # Param-count introspection
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    # ------------------------------------------------------------------
    # Trunk
    # ------------------------------------------------------------------

    def _encode(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(trunk, nodes)`` where ``nodes`` are the per-node GNN states
        (``{"v","e","h"}``) the pointer readouts consume. When the graph encoder
        is disabled the node states are zeros (pointer heads degrade to a
        constant-per-node readout)."""
        tile = self.tile_encoder(obs["tile_representations"])
        # Flatten per-tile vectors -> (B, N_TILES * tile_out_dim).
        tile_flat = tile.flatten(start_dim=1)

        curr_p = self.curr_player_enc(obs["current_player_main"])
        opp_p = self.opp_player_enc(obs["next_player_main"])
        curr_dev = self.curr_dev_enc(obs["current_dev_counts"])
        opp_dev = self.opp_dev_enc(obs["next_played_dev_counts"])

        parts = [tile_flat, curr_p, opp_p, curr_dev, opp_dev]

        batch = tile_flat.size(0)
        if self.graph_encoder is not None:
            g, v, e, h = self.graph_encoder(
                obs["hex_features"], obs["vertex_features"], obs["edge_features"]
            )
            parts.append(g)
            nodes = {"v": v, "e": e, "h": h}
        else:
            zeros = torch.zeros(batch, 1, self._node_dim, device=tile_flat.device)
            nodes = {
                "v": zeros.expand(batch, N_VERTICES, self._node_dim),
                "e": zeros.expand(batch, N_EDGES, self._node_dim),
                "h": zeros.expand(batch, N_TILES, self._node_dim),
            }

        if self.opp_id_emb is not None:
            parts.append(
                self.opp_id_emb(obs["opponent_kind"].long(), obs["opponent_policy_id"].long())
            )

        # POV-neutral global block (bank + public dev-deck + reserved) is
        # appended LAST so a migration zero-pads the fusion's new input columns
        # as a pure tail-append (all legacy fusion columns keep their position).
        parts.append(obs["global_features"])

        return self.fusion(torch.cat(parts, dim=-1)), nodes

    # ------------------------------------------------------------------
    # Forward / sample
    # ------------------------------------------------------------------

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode the obs and return all head outputs.

        This path is used by ``CatanPolicy.evaluate_actions`` and tests;
        the PPO rollout path uses :meth:`sample` to get an action and its
        log-prob in one shot.
        """
        trunk, nodes = self._encode(obs)
        value = self.value_head(trunk)
        out: dict[str, torch.Tensor] = {"trunk": trunk, "value": value}
        # Stash the per-node states + is_setup so sample/evaluate_actions can
        # feed the pointer readouts without re-encoding.
        out["_node_v"] = nodes["v"]
        out["_node_e"] = nodes["e"]
        out["_node_h"] = nodes["h"]
        if "is_setup" in obs:
            out["_is_setup"] = obs["is_setup"]
        if self.belief_head is not None:
            out["belief_logits"] = self.belief_head(trunk)
        if self.aux_value_head is not None:
            out["aux_value"] = self.aux_value_head(trunk)
        return out

    @staticmethod
    def _nodes_from_out(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"v": out["_node_v"], "e": out["_node_e"], "h": out["_node_h"]}

    def sample(
        self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Sample an action from the masked policy + return value + belief."""
        out = self.forward(obs)
        action, log_prob, entropy, per_head_log_prob = self.action_heads.sample(
            out["trunk"], masks, self._nodes_from_out(out), out.get("_is_setup")
        )
        out.update(
            {
                "action": action,
                "log_prob": log_prob,
                "entropy": entropy,
                "per_head_log_prob": per_head_log_prob,
            }
        )
        return out

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out = self.forward(obs)
        head_out = self.action_heads.evaluate_actions(
            out["trunk"], action, masks, self._nodes_from_out(out), out.get("_is_setup")
        )
        out.update(head_out)
        return out

    # ------------------------------------------------------------------
    # Geometry plumbing
    # ------------------------------------------------------------------

    def set_board_geometry(self, geometry: dict[str, Any]) -> None:
        """Wire in board-dependent constants (axial coords + GNN adjacency).

        Step 2 leaves these at zero placeholders for testability. Step 3
        will call this from the env's ``reset()`` once per process to
        plumb in the real values derived from ``catanBoard``.
        """
        self.tile_encoder.pos_emb.set_axial_indices(geometry["q_idx"], geometry["r_idx"])
        if self.graph_encoder is not None:
            self.graph_encoder.set_adjacency(
                hex_to_vertex=geometry["hex_to_vertex"],
                vertex_to_hex=geometry["vertex_to_hex"],
                vertex_to_hex_mask=geometry["vertex_to_hex_mask"],
                edge_to_vertex=geometry["edge_to_vertex"],
                vertex_to_edge=geometry["vertex_to_edge"],
                vertex_to_edge_mask=geometry["vertex_to_edge_mask"],
            )
