"""Tests for Phase 3.6 opponent identity embedding."""

from __future__ import annotations

import pytest
import torch

from catan_rl.models.observation_module import ObservationModule


def _make_obs(B: int = 2, with_opp_id: bool = False) -> dict:
    obs = {
        "tile_representations": torch.randn(B, 19, 79),
        "current_player_main": torch.randn(B, 166),
        "current_player_hidden_dev": torch.zeros(B, 16, dtype=torch.long),
        "current_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
        "next_player_main": torch.randn(B, 173),
        "next_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
    }
    if with_opp_id:
        obs["opponent_kind"] = torch.tensor([1, 4][:B], dtype=torch.long)
        obs["opponent_policy_id"] = torch.tensor([0, 5][:B], dtype=torch.long)
    return obs


def test_opp_id_emb_off_by_default() -> None:
    """Default obs encoder has no opp_kind / opp_policy embeddings."""
    om = ObservationModule()
    assert om.use_opponent_id_emb is False
    assert om.opp_id_emb_dim == 0
    assert not hasattr(om, "opp_kind_emb") or om.opp_kind_emb is None


def test_opp_id_emb_on_creates_embeddings() -> None:
    """``use_opponent_id_emb=True`` builds the kind + policy lookup tables."""
    om = ObservationModule(
        use_opponent_id_emb=True,
        opp_id_emb_dim=16,
        n_opp_kinds=6,
        league_maxlen=100,
    )
    assert om.use_opponent_id_emb is True
    assert om.opp_kind_emb.num_embeddings == 6
    # +1 to leave a slot for the "unknown" sentinel.
    assert om.opp_policy_emb.num_embeddings == 101
    assert om.opp_kind_emb.embedding_dim == 8
    assert om.opp_policy_emb.embedding_dim == 8


def test_opp_id_emb_dim_must_be_even() -> None:
    with pytest.raises(ValueError, match="opp_id_emb_dim must be even"):
        ObservationModule(use_opponent_id_emb=True, opp_id_emb_dim=15)


def test_forward_with_opp_id_keys_succeeds() -> None:
    """The forward pass consumes ``opponent_kind`` / ``opponent_policy_id``."""
    torch.manual_seed(0)
    om = ObservationModule(use_opponent_id_emb=True, opp_id_emb_dim=16)
    obs = _make_obs(B=2, with_opp_id=True)
    out = om(obs)
    assert out.shape == (2, 512)


def test_forward_param_count_grows_with_emb() -> None:
    """Enabling the embedding adds exactly the expected param count.

    Two embeddings (n_opp_kinds × half) + (league_maxlen+1 × half) plus a
    wider ``final_layer`` (its in-features grows by ``opp_id_emb_dim``).
    """
    om_off = ObservationModule()
    om_on = ObservationModule(
        use_opponent_id_emb=True, opp_id_emb_dim=16, n_opp_kinds=6, league_maxlen=100
    )
    n_off = sum(p.numel() for p in om_off.parameters() if p.requires_grad)
    n_on = sum(p.numel() for p in om_on.parameters() if p.requires_grad)
    delta = n_on - n_off

    half = 8
    emb_kind = 6 * half  # n_opp_kinds × half
    emb_policy = (100 + 1) * half  # (league_maxlen + 1) × half
    # final_layer in-features grew by 16. The output dim is 512.
    final_layer_extra = 16 * 512
    expected = emb_kind + emb_policy + final_layer_extra
    assert delta == expected, f"Expected +{expected} params, got +{delta}"


def test_unknown_sentinel_is_in_range() -> None:
    """Policy_id == league_maxlen (the unknown sentinel) is valid."""
    om = ObservationModule(use_opponent_id_emb=True, opp_id_emb_dim=16, league_maxlen=100)
    obs = _make_obs(B=1, with_opp_id=True)
    obs["opponent_kind"] = torch.tensor([0])  # unknown
    obs["opponent_policy_id"] = torch.tensor([100])  # sentinel
    om(obs)  # should not raise
