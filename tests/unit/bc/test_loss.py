"""TDD tests for bc/loss.py.

Pin the per-head relevance-weighted CE math, the value MSE, the belief
soft-CE, and the total-loss composition with config-tunable weights.
"""

from __future__ import annotations

import numpy as np
import torch

from catan_rl.policy import CatanPolicy
from catan_rl.policy.obs_schema import (
    CURR_PLAYER_DIM,
    N_DEV_TYPES,
    N_EDGES,
    N_RESOURCES,
    N_TILES,
    N_VERTICES,
    NEXT_PLAYER_DIM,
    TILE_DIM,
    ActionType,
)


def _synth_batch(b: int, type_for_row: list[int]) -> dict:
    """Hand-built batch with controlled action types per row."""
    rng = np.random.default_rng(0)
    obs = {
        "tile_representations": torch.from_numpy(
            rng.standard_normal((b, N_TILES, TILE_DIM)).astype(np.float32)
        ),
        "current_player_main": torch.from_numpy(
            rng.standard_normal((b, CURR_PLAYER_DIM)).astype(np.float32)
        ),
        "next_player_main": torch.from_numpy(
            rng.standard_normal((b, NEXT_PLAYER_DIM)).astype(np.float32)
        ),
        "current_dev_counts": torch.from_numpy(
            rng.integers(0, 5, (b, N_DEV_TYPES)).astype(np.float32)
        ),
        "next_played_dev_counts": torch.from_numpy(
            rng.integers(0, 5, (b, N_DEV_TYPES)).astype(np.float32)
        ),
        "hex_features": torch.from_numpy(rng.standard_normal((b, N_TILES, 19)).astype(np.float32)),
        "vertex_features": torch.from_numpy(
            rng.standard_normal((b, N_VERTICES, 16)).astype(np.float32)
        ),
        "edge_features": torch.from_numpy(rng.standard_normal((b, N_EDGES, 16)).astype(np.float32)),
        "opponent_kind": torch.zeros(b, dtype=torch.int64),
        "opponent_policy_id": torch.zeros(b, dtype=torch.int64),
    }
    action = torch.zeros((b, 6), dtype=torch.int64)
    for i, t in enumerate(type_for_row):
        action[i, 0] = t
        action[i, 1] = 7  # corner
        action[i, 2] = 11  # edge
        action[i, 3] = 4  # tile
        action[i, 4] = 1  # res1
        action[i, 5] = 2  # res2
    mask = {
        "type": torch.ones((b, 13), dtype=torch.bool),
        "corner_settlement": torch.ones((b, N_VERTICES), dtype=torch.bool),
        "corner_city": torch.ones((b, N_VERTICES), dtype=torch.bool),
        "edge": torch.ones((b, N_EDGES), dtype=torch.bool),
        "tile": torch.ones((b, N_TILES), dtype=torch.bool),
        "resource1_trade": torch.ones((b, N_RESOURCES), dtype=torch.bool),
        "resource1_discard": torch.ones((b, N_RESOURCES), dtype=torch.bool),
        "resource1_default": torch.ones((b, N_RESOURCES), dtype=torch.bool),
        "resource2_default": torch.ones((b, N_RESOURCES), dtype=torch.bool),
    }
    belief = torch.full((b, N_DEV_TYPES), 1.0 / N_DEV_TYPES, dtype=torch.float32)
    z_disc = torch.zeros(b, dtype=torch.float32)
    return {
        "obs": obs,
        "action": action,
        "mask": mask,
        "belief_target": belief,
        "z_disc": z_disc,
    }


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_bc_loss_returns_expected_keys() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(8, [ActionType.END_TURN] * 8)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch, value_weight=0.1, belief_weight=0.05)
    assert {"total", "policy", "value", "belief"} <= set(loss)
    for key in ("total", "policy", "value", "belief"):
        assert torch.is_tensor(loss[key])
        assert loss[key].ndim == 0  # scalar


def test_bc_loss_per_head_keys_present() -> None:
    """Per-head policy CE components should be exposed for TB logging."""
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(8, [ActionType.BUILD_SETTLEMENT] * 8)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch)
    for head in ("type", "corner", "edge", "tile", "resource1", "resource2"):
        assert f"policy/{head}" in loss


# ---------------------------------------------------------------------------
# Relevance weighting — each head only contributes when relevant
# ---------------------------------------------------------------------------


def test_per_head_loss_only_active_when_relevant() -> None:
    """For a batch of all END_TURN actions:
    * Type head IS relevant (always) → policy/type > 0.
    * Corner/edge/tile/res1/res2 heads NOT relevant for END_TURN → 0.
    """
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(8, [ActionType.END_TURN] * 8)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch)
    assert loss["policy/type"].item() > 0
    for head in ("corner", "edge", "tile", "resource1", "resource2"):
        assert loss[f"policy/{head}"].item() == 0.0, (
            f"head {head} should be zero for END_TURN actions"
        )


def test_corner_head_relevant_for_settlement() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(8, [ActionType.BUILD_SETTLEMENT] * 8)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch)
    assert loss["policy/corner"].item() > 0
    assert loss["policy/edge"].item() == 0.0


def test_edge_head_relevant_for_road() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(8, [ActionType.BUILD_ROAD] * 8)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch)
    assert loss["policy/edge"].item() > 0
    assert loss["policy/corner"].item() == 0.0


def test_tile_head_relevant_for_robber_and_knight() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(
        4,
        [
            ActionType.MOVE_ROBBER,
            ActionType.MOVE_ROBBER,
            ActionType.PLAY_KNIGHT,
            ActionType.PLAY_KNIGHT,
        ],
    )
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch)
    assert loss["policy/tile"].item() > 0


def test_resource1_resource2_for_trade_and_yop() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(
        4, [ActionType.BANK_TRADE, ActionType.BANK_TRADE, ActionType.PLAY_YOP, ActionType.PLAY_YOP]
    )
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch)
    assert loss["policy/resource1"].item() > 0
    assert loss["policy/resource2"].item() > 0


# ---------------------------------------------------------------------------
# Total composition
# ---------------------------------------------------------------------------


def test_total_is_weighted_sum() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(8, [ActionType.BUILD_SETTLEMENT] * 8)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch, value_weight=0.10, belief_weight=0.05)
    expected = loss["policy"] + 0.10 * loss["value"] + 0.05 * loss["belief"]
    assert torch.isclose(loss["total"], expected, atol=1e-6)


def test_value_weight_zero_disables_value_term() -> None:
    """``value_weight=0`` should remove the value contribution entirely
    (used by the BC plan's drift-safeguard fallback)."""
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(8, [ActionType.END_TURN] * 8)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    a = bc_loss(policy_out=out, batch=batch, value_weight=0.10, belief_weight=0.05)
    b = bc_loss(policy_out=out, batch=batch, value_weight=0.0, belief_weight=0.05)
    assert torch.isclose(
        b["total"] - b["policy"] - 0.05 * b["belief"], torch.tensor(0.0), atol=1e-6
    )
    # The total differs by exactly 0.10 * value (a sanity check).
    assert torch.isclose(a["total"] - b["total"], 0.10 * a["value"], atol=1e-6)


# ---------------------------------------------------------------------------
# Belief soft-CE
# ---------------------------------------------------------------------------


def test_belief_loss_matches_manual_soft_ce() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(4, [ActionType.END_TURN] * 4)
    # Custom belief targets.
    batch["belief_target"] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch)
    # Manual soft-CE: -sum(target * log_softmax(logits)) averaged.
    logits = out["belief_logits"]
    log_probs = torch.log_softmax(logits, dim=-1)
    manual = -(batch["belief_target"] * log_probs).sum(dim=-1).mean()
    assert torch.isclose(loss["belief"], manual, atol=1e-5)


# ---------------------------------------------------------------------------
# Value MSE
# ---------------------------------------------------------------------------


def test_value_loss_matches_mse() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(8, [ActionType.END_TURN] * 8)
    batch["z_disc"] = torch.tensor([0.5, -0.5, 1.0, -1.0, 0.0, 0.3, -0.3, 0.7], dtype=torch.float32)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch)
    manual_mse = ((out["value"] - batch["z_disc"]) ** 2).mean()
    assert torch.isclose(loss["value"], manual_mse, atol=1e-5)


# ---------------------------------------------------------------------------
# Backward — every loss term propagates gradient
# ---------------------------------------------------------------------------


def test_backward_updates_policy_value_belief_when_all_weights_nonzero() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy()
    # Spread action types to ensure every head is relevant for at least one row.
    types = [
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_ROAD,
        ActionType.MOVE_ROBBER,
        ActionType.BANK_TRADE,
        ActionType.PLAY_YOP,
        ActionType.END_TURN,
        ActionType.BUILD_CITY,
        ActionType.BUY_DEV_CARD,
    ]
    batch = _synth_batch(8, types)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch, value_weight=0.10, belief_weight=0.05)
    loss["total"].backward()

    missing: list[str] = []
    for name, p in policy.named_parameters():
        if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0):
            missing.append(name)
    assert not missing, f"these params got no grad from bc_loss.total: {missing}"


# ---------------------------------------------------------------------------
# Sanity guards
# ---------------------------------------------------------------------------


def test_bc_loss_no_nan_or_inf() -> None:
    from catan_rl.bc.loss import bc_loss

    policy = CatanPolicy().eval()
    batch = _synth_batch(8, [ActionType.END_TURN] * 8)
    out = policy.evaluate_actions(batch["obs"], batch["action"], batch["mask"])
    loss = bc_loss(policy_out=out, batch=batch)
    for k, v in loss.items():
        assert torch.isfinite(v).all(), f"{k} contains NaN/inf"
