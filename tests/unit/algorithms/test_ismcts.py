"""Tests for Phase 4.1 ISMCTS (single-step PUCT + belief determinization)."""

from __future__ import annotations

import numpy as np
import torch

from catan_rl.algorithms.search.ismcts import (
    ISMCTS,
    ISMCTSConfig,
    _ActionNode,
    _puct_score,
    visits_to_distribution,
)
from catan_rl.models.action_heads_module import N_ACTION_TYPES


def test_action_node_default_value() -> None:
    """Unvisited nodes return 0 — PUCT init."""
    n = _ActionNode(prior=0.1)
    assert n.value == 0.0


def test_puct_score_higher_for_higher_prior() -> None:
    """At equal visit count, higher prior → higher PUCT."""
    a = _ActionNode(prior=0.1, valid=True)
    b = _ActionNode(prior=0.5, valid=True)
    assert _puct_score(b, parent_visits=10, c_puct=1.5) > _puct_score(a, 10, 1.5)


def test_puct_score_invalid_returns_neg_inf() -> None:
    """Masked-out (invalid) actions are never selected."""
    n = _ActionNode(prior=1.0, valid=False)
    assert _puct_score(n, parent_visits=10, c_puct=1.5) == -np.inf


def test_visits_to_distribution_uniform_when_zero() -> None:
    """Zero-visit input → uniform distribution (avoid divide-by-zero)."""
    out = visits_to_distribution(np.zeros(N_ACTION_TYPES))
    assert abs(out.sum() - 1.0) < 1e-9
    assert np.allclose(out, 1.0 / N_ACTION_TYPES)


def test_visits_to_distribution_proportional() -> None:
    """At ``temperature=1`` the dist is proportional to visit counts."""
    visits = np.array([0, 10, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    out = visits_to_distribution(visits, temperature=1.0)
    assert abs(out[1] - 10 / 15) < 1e-9
    assert abs(out[3] - 5 / 15) < 1e-9


def test_visits_to_distribution_argmax_at_zero_temp() -> None:
    """``temperature=0`` collapses to argmax."""
    visits = np.array([0, 10, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    out = visits_to_distribution(visits, temperature=0.0)
    assert out[1] == 1.0
    assert out.sum() == 1.0


def test_visits_to_distribution_temperature_sharpens() -> None:
    """Lower ``temperature`` puts more mass on the argmax."""
    visits = np.array([10, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    soft = visits_to_distribution(visits, temperature=2.0)
    sharp = visits_to_distribution(visits, temperature=0.5)
    assert sharp[0] > soft[0]


def test_ismcts_search_visit_count_matches_budget() -> None:
    """Total visits = ``n_sims_per_det * n_determinizations``."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy()
    p.eval()
    obs = _make_obs()
    masks = _make_masks()

    cfg = ISMCTSConfig(n_sims_per_det=8, n_determinizations=3, use_belief_determinization=False)
    visits = ISMCTS(cfg).search(p, obs, masks)
    assert visits.shape == (N_ACTION_TYPES,)
    assert visits.sum() == 8 * 3


def test_ismcts_respects_action_mask() -> None:
    """Masked-out action types must receive zero visits."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy()
    p.eval()
    obs = _make_obs()
    masks = _make_masks()
    # Allow only action types 0 and 5.
    type_mask = torch.zeros(1, N_ACTION_TYPES, dtype=torch.bool)
    type_mask[0, 0] = True
    type_mask[0, 5] = True
    masks["type"] = type_mask

    cfg = ISMCTSConfig(n_sims_per_det=10, n_determinizations=1, use_belief_determinization=False)
    visits = ISMCTS(cfg).search(p, obs, masks)
    # Every visit must be on an allowed action.
    for a in range(N_ACTION_TYPES):
        if a not in (0, 5):
            assert visits[a] == 0
    assert visits.sum() == 10


def test_ismcts_uses_belief_determinization_when_provided() -> None:
    """``belief_logits`` flow through the determinization path without crashing."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy(use_belief_head=True)
    p.eval()
    obs = _make_obs()
    masks = _make_masks()

    with torch.inference_mode():
        obs_out = p.observation_module(obs)
        belief_logits = p.belief_head(obs_out)

    cfg = ISMCTSConfig(n_sims_per_det=4, n_determinizations=2)
    visits = ISMCTS(cfg).search(p, obs, masks, belief_logits=belief_logits)
    assert visits.sum() == 4 * 2


def test_ismcts_handles_no_valid_actions() -> None:
    """All-False mask → search exits cleanly with zero visits."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy()
    p.eval()
    obs = _make_obs()
    masks = _make_masks()
    masks["type"] = torch.zeros(1, N_ACTION_TYPES, dtype=torch.bool)

    cfg = ISMCTSConfig(n_sims_per_det=4, n_determinizations=1)
    visits = ISMCTS(cfg).search(p, obs, masks)
    assert visits.sum() == 0


def test_ismcts_recurrent_value_head_ok() -> None:
    """The search supports policies with a recurrent value head."""
    from catan_rl.models.policy import CatanPolicy

    p = CatanPolicy(use_recurrent_value=True, gru_hidden_dim=64)
    p.eval()
    obs = _make_obs()
    masks = _make_masks()

    cfg = ISMCTSConfig(n_sims_per_det=4, n_determinizations=1, use_belief_determinization=False)
    visits = ISMCTS(cfg).search(p, obs, masks)
    assert visits.sum() == 4


# ── helpers ──────────────────────────────────────────────────────────────


def _make_obs(B: int = 1) -> dict[str, torch.Tensor]:
    return {
        "tile_representations": torch.randn(B, 19, 79),
        "current_player_main": torch.randn(B, 166),
        "current_player_hidden_dev": torch.zeros(B, 16, dtype=torch.long),
        "current_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
        "next_player_main": torch.randn(B, 173),
        "next_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
    }


def _make_masks(B: int = 1) -> dict[str, torch.Tensor]:
    return {
        "type": torch.ones(B, N_ACTION_TYPES, dtype=torch.bool),
        "corner_settlement": torch.ones(B, 54, dtype=torch.bool),
        "corner_city": torch.ones(B, 54, dtype=torch.bool),
        "edge": torch.ones(B, 72, dtype=torch.bool),
        "tile": torch.ones(B, 19, dtype=torch.bool),
        "resource1_default": torch.ones(B, 5, dtype=torch.bool),
        "resource1_trade": torch.ones(B, 5, dtype=torch.bool),
        "resource1_discard": torch.ones(B, 5, dtype=torch.bool),
        "resource2_default": torch.ones(B, 5, dtype=torch.bool),
    }
