"""E0.4 — network equivariance probe with comparator baselines.

Per ``v2_design.md`` §0 E0.4 (post-faculty-re-review), this script
measures how close the v2 ``CatanPolicy`` is to D6-equivariance and
calibrates the BC + PPO symmetry-aug probability against two
reference architectures:

  equiv_lo  — "as equivariant as our toolkit can build"
              CatanPolicy with the axial pos emb zeroed out. The
              remaining TileEncoder transformer + GNN are both
              permutation-equivariant by construction; the residual
              equivariance loss is what the policy's *static* features
              (resource one-hot, number tokens, etc.) leak.

  equiv_hi  — "completely non-equivariant"
              A vanilla MLP on the flattened concatenation of every
              obs tensor. Has no notion of spatial structure.

For each head h ∈ {type, value, belief} and each non-identity D6
element g, we compute the equivariance loss:

    E_h(g) = mean_i || h(s_i) − T_g^{-1}( h(T_g(s_i)) ) ||_1 / || h(s_i) ||_1

then average across the 11 non-identity group elements. The normalised
gap

    r_h = (E_h − E_h^lo) / (E_h^hi − E_h^lo) ∈ [0, 1]

tells us where the v2 network sits on the equivariance spectrum.

Decision rule:
  r_h ≤ 0.10 for all 3 heads        → aug-prob 0.5 (carry invariance)
  0.10 < max r_h ≤ 0.40             → aug-prob 0.5 BC, 1.0 PPO
  max r_h > 0.40                    → aug-prob 1.0; flag for follow-up

The "T_g^{-1}" piece: for head outputs that index a spatial axis (like
the type head, which doesn't index any) we don't need to un-permute
the output. Type/value/belief are all spatially invariant outputs
(scalar / 13-way / 5-way), so under a correctly-D6-equivariant
network, ``h(T_g(s)) == h(s)`` (no permutation of the output needed).

This is the simpler "invariance" check — the full equivariance check
for the spatial action heads (corner/edge/tile) is a follow-up.

Output: ``runs/preflight/e04/equivariance.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from catan_rl.augmentation import D6_GROUP_SIZE, apply_symmetry
from catan_rl.engine.game import catanGame
from catan_rl.policy import CatanPolicy
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.obs_encoder import EnvObsState, ObsEncoder

# ---------------------------------------------------------------------------
# Reference networks
# ---------------------------------------------------------------------------


class _VanillaMLP(nn.Module):
    """``equiv_hi`` baseline: flatten every obs tensor, MLP, output the
    same head shapes as CatanPolicy. No spatial structure at all.

    By construction, this will have *large* equivariance loss — a random
    permutation of the inputs feeds a random permutation of the
    Linear's input columns, producing a very different output.
    """

    def __init__(self, flat_dim: int, type_dim: int = 13, belief_dim: int = 5) -> None:
        super().__init__()
        self.type_head = nn.Linear(flat_dim, type_dim)
        self.value_head = nn.Linear(flat_dim, 1)
        self.belief_head = nn.Linear(flat_dim, belief_dim)
        for layer in (self.type_head, self.value_head, self.belief_head):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        self.flat_dim = flat_dim

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        flat = _flatten_obs(obs)
        return {
            "type_logits": self.type_head(flat),
            "value": self.value_head(flat).squeeze(-1),
            "belief_logits": self.belief_head(flat),
        }


def _flatten_obs(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate every float obs tensor into a single (B, D) batch."""
    parts: list[torch.Tensor] = []
    for key in (
        "tile_representations",
        "current_player_main",
        "next_player_main",
        "current_dev_counts",
        "next_played_dev_counts",
        "hex_features",
        "vertex_features",
        "edge_features",
    ):
        if key in obs:
            parts.append(obs[key].reshape(obs[key].shape[0], -1))
    return torch.cat(parts, dim=-1)


def _build_full_policy(seed: int) -> CatanPolicy:
    torch.manual_seed(seed)
    policy = CatanPolicy().eval()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    return policy


def _build_equiv_lo(seed: int) -> CatanPolicy:
    """Same as full v2 but with axial pos emb zeroed. The remaining
    transformer + GNN trunk is permutation-equivariant by construction.
    """
    torch.manual_seed(seed)
    policy = CatanPolicy().eval()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    # Zero out the axial pos emb weights — the embedding still gets
    # added in forward(), but with all-zero values it contributes nothing.
    with torch.no_grad():
        policy.tile_encoder.pos_emb.q_emb.weight.zero_()
        policy.tile_encoder.pos_emb.r_emb.weight.zero_()
    return policy


def _build_equiv_hi(seed: int, flat_dim: int) -> _VanillaMLP:
    torch.manual_seed(seed + 1000)
    return _VanillaMLP(flat_dim).eval()


# ---------------------------------------------------------------------------
# Obs sampling — generate a batch from real game states
# ---------------------------------------------------------------------------


def _sample_obs_batch(batch_size: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Sample obs from N freshly-initialised games (no setup played).

    For the equivariance probe we don't need the games to be
    realistic — they just need to be valid v2 obs schemas. Pre-setup
    games are uniform across the board, which keeps the equivariance
    arithmetic close to the geometric ideal.
    """
    np.random.seed(seed)
    obs_list: list[dict[str, np.ndarray]] = []
    for i in range(batch_size):
        np.random.seed(seed + i)
        game = catanGame(render_mode=None)
        players = list(game.playerQueue.queue)
        agent, opp = players[0], players[1]
        encoder = ObsEncoder(game.board)
        env_state = EnvObsState(initial_placement_phase=True)
        obs_list.append(encoder.build_obs(game, agent, opp, env_state))

    # Stack into a single batched dict.
    batched: dict[str, torch.Tensor] = {}
    for key in obs_list[0]:
        arrs = [o[key] for o in obs_list]
        stacked = np.stack(arrs)
        if stacked.dtype.kind == "f":
            batched[key] = torch.from_numpy(stacked.astype(np.float32))
        else:
            batched[key] = torch.from_numpy(stacked.astype(np.int64))
    return batched


# ---------------------------------------------------------------------------
# Equivariance loss
# ---------------------------------------------------------------------------


def _get_head_outputs(model: Any, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return {type_logits, value, belief_logits} for either a CatanPolicy
    or a _VanillaMLP. CatanPolicy.forward emits trunk+value+belief; we
    pull the action_heads' type-head logits separately to match the
    _VanillaMLP shape.
    """
    if isinstance(model, _VanillaMLP):
        return model(obs)
    out = model(obs)
    trunk = out["trunk"]
    type_logits = model.action_heads.type_head(trunk)
    return {
        "type_logits": type_logits,
        "value": out["value"],
        "belief_logits": out["belief_logits"],
    }


def _equivariance_loss(
    model: Any, obs: dict[str, torch.Tensor], head_keys: tuple[str, ...]
) -> dict[str, float]:
    """Average L1 relative equivariance loss across the 11 non-identity
    D6 elements, per head.

    The type / value / belief heads are spatially *invariant* under D6
    (they don't index a spatial axis), so the correct test is
    ``|| h(s) − h(T_g(s)) ||_1 / || h(s) ||_1``.
    """
    with torch.no_grad():
        base = _get_head_outputs(model, obs)
        # Build neutral actions + masks for apply_symmetry (it expects them).
        b = obs["tile_representations"].shape[0]
        actions = torch.zeros((b, 6), dtype=torch.int64)
        masks = {
            "type": torch.ones((b, 13), dtype=torch.bool),
            "corner_settlement": torch.ones((b, 54), dtype=torch.bool),
            "corner_city": torch.ones((b, 54), dtype=torch.bool),
            "edge": torch.ones((b, 72), dtype=torch.bool),
            "tile": torch.ones((b, 19), dtype=torch.bool),
            "resource1_trade": torch.ones((b, 5), dtype=torch.bool),
            "resource1_discard": torch.ones((b, 5), dtype=torch.bool),
            "resource1_default": torch.ones((b, 5), dtype=torch.bool),
            "resource2_default": torch.ones((b, 5), dtype=torch.bool),
        }

        per_head: dict[str, list[float]] = {h: [] for h in head_keys}
        for g in range(1, D6_GROUP_SIZE):
            new_obs, _, _ = apply_symmetry(obs, actions, masks, g)
            transformed = _get_head_outputs(model, new_obs)
            for h in head_keys:
                # Per-sample relative L1; average over batch.
                base_h = base[h]
                trans_h = transformed[h]
                num = (
                    (base_h - trans_h).abs().sum(dim=-1)
                    if base_h.dim() == 2
                    else (base_h - trans_h).abs()
                )
                den = base_h.abs().sum(dim=-1) if base_h.dim() == 2 else base_h.abs()
                ratio = num / den.clamp(min=1e-8)
                per_head[h].append(float(ratio.mean().item()))

    return {h: float(np.mean(per_head[h])) for h in head_keys}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "runs" / "preflight" / "e04" / "equivariance.json",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("[E0.4] sampling obs batch...", flush=True)
    t0 = time.time()
    obs = _sample_obs_batch(args.batch_size, seed=args.seed)
    print(f"[E0.4] obs batch built in {time.time() - t0:.1f}s", flush=True)

    print("[E0.4] building reference networks...", flush=True)
    full = _build_full_policy(seed=args.seed)
    lo = _build_equiv_lo(seed=args.seed)
    flat_dim = _flatten_obs(obs).shape[-1]
    hi = _build_equiv_hi(seed=args.seed, flat_dim=flat_dim)

    head_keys = ("type_logits", "value", "belief_logits")

    print("[E0.4] computing equivariance loss for full v2 policy...", flush=True)
    E_full = _equivariance_loss(full, obs, head_keys)
    print(f"  full: {E_full}", flush=True)

    print("[E0.4] computing equivariance loss for equiv_lo baseline...", flush=True)
    E_lo = _equivariance_loss(lo, obs, head_keys)
    print(f"  equiv_lo: {E_lo}", flush=True)

    print("[E0.4] computing equivariance loss for equiv_hi baseline...", flush=True)
    E_hi = _equivariance_loss(hi, obs, head_keys)
    print(f"  equiv_hi: {E_hi}", flush=True)

    r_h: dict[str, float] = {}
    for h in head_keys:
        denom = max(E_hi[h] - E_lo[h], 1e-8)
        r_h[h] = (E_full[h] - E_lo[h]) / denom
        r_h[h] = max(0.0, r_h[h])  # clamp slightly-negative numerical drift

    max_r = max(r_h.values())
    if max_r <= 0.10:
        decision = "AUG_PROB_0.5_BOTH_BC_AND_PPO"
    elif max_r <= 0.40:
        decision = "AUG_PROB_0.5_BC_1.0_PPO"
    else:
        decision = "AUG_PROB_1.0_THROUGHOUT_FLAG_FOR_FOLLOWUP"

    summary = {
        "metadata": {
            "batch_size": args.batch_size,
            "n_d6_elements_tested": D6_GROUP_SIZE - 1,
            "wall_clock_seconds": time.time() - t0,
        },
        "E_full": E_full,
        "E_equiv_lo": E_lo,
        "E_equiv_hi": E_hi,
        "r_h": r_h,
        "max_r_h": max_r,
        "decision_rule": decision,
    }

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[E0.4] wrote {args.out}", flush=True)
    print(f"[E0.4] r_h: {r_h}, max={max_r:.4f} -> {decision}", flush=True)


if __name__ == "__main__":
    main()
