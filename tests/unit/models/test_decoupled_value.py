"""Tests for Phase 2.5 decoupled value tower."""

from __future__ import annotations

import torch

from catan_rl.models.build_agent_model import build_agent_model
from catan_rl.models.policy import CatanPolicy


def _make_obs(B: int = 2) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    obs = {
        "tile_representations": torch.randn(B, 19, 79),
        "current_player_main": torch.randn(B, 166),
        "current_player_hidden_dev": torch.zeros(B, 16, dtype=torch.long),
        "current_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
        "next_player_main": torch.randn(B, 173),
        "next_player_played_dev": torch.zeros(B, 16, dtype=torch.long),
    }
    masks = {
        "type": torch.ones(B, 13, dtype=torch.bool),
        "corner_settlement": torch.ones(B, 54, dtype=torch.bool),
        "corner_city": torch.ones(B, 54, dtype=torch.bool),
        "edge": torch.ones(B, 72, dtype=torch.bool),
        "tile": torch.ones(B, 19, dtype=torch.bool),
        "resource1_default": torch.ones(B, 5, dtype=torch.bool),
        "resource1_trade": torch.ones(B, 5, dtype=torch.bool),
        "resource1_discard": torch.ones(B, 5, dtype=torch.bool),
        "resource2_default": torch.ones(B, 5, dtype=torch.bool),
    }
    return obs, masks


def test_value_head_mode_default_shared() -> None:
    """Default keeps the legacy shared encoder."""
    p = CatanPolicy()
    assert p.value_head_mode == "shared"
    assert p.value_observation_module is None


def test_value_head_mode_decoupled_builds_extra_encoder() -> None:
    """``decoupled`` mode adds a second ObservationModule."""
    p = CatanPolicy(value_head_mode="decoupled")
    assert p.value_head_mode == "decoupled"
    assert p.value_observation_module is not None


def test_value_head_mode_invalid_raises() -> None:
    try:
        CatanPolicy(value_head_mode="bogus")
    except ValueError as e:
        assert "value_head_mode" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid value_head_mode")


def test_decoupled_value_param_count_grows() -> None:
    """Decoupled mode roughly doubles the obs-encoder params."""
    p_shared = CatanPolicy(value_head_mode="shared")
    p_decoupled = CatanPolicy(value_head_mode="decoupled")

    n_shared = sum(p.numel() for p in p_shared.parameters() if p.requires_grad)
    n_decoupled = sum(p.numel() for p in p_decoupled.parameters() if p.requires_grad)

    delta = n_decoupled - n_shared
    obs_params = sum(p.numel() for p in p_shared.observation_module.parameters())
    # The extra encoder should account for the bulk of the param increase.
    assert delta == obs_params, (
        f"Decoupled adds {delta} params, expected exactly +{obs_params} (a "
        f"second observation_module). Drift indicates other components grew."
    )


def test_decoupled_value_forward_runs() -> None:
    """Both ``act`` and ``get_value`` must work in decoupled mode."""
    torch.manual_seed(0)
    p = CatanPolicy(value_head_mode="decoupled")
    p.eval()
    obs, masks = _make_obs(B=2)
    actions, value, log_prob = p.act(obs, masks)
    assert actions.shape == (2, 6)
    assert value.shape == (2, 1)
    assert log_prob.shape == (2,)
    v_only = p.get_value(obs)
    assert v_only.shape == (2, 1)


def test_decoupled_value_uses_separate_encoder() -> None:
    """Zeroing the policy encoder must not alter the value output."""
    torch.manual_seed(0)
    p = CatanPolicy(value_head_mode="decoupled")
    p.eval()
    obs, _ = _make_obs(B=2)

    v_before = p.get_value(obs)

    # Replace the policy encoder's final layer with zeros — value path should
    # be unaffected because the value tower goes through a different encoder.
    with torch.no_grad():
        p.observation_module.final_layer.weight.zero_()
        p.observation_module.final_layer.bias.zero_()

    v_after = p.get_value(obs)
    torch.testing.assert_close(v_before, v_after)


def test_decoupled_value_gradient_isolation() -> None:
    """Gradient from a value-only loss must not flow into the policy encoder.

    This is the core motivation for the decoupled tower: stop the value-loss
    gradient from reshaping the trunk that the policy depends on.
    """
    torch.manual_seed(0)
    p = CatanPolicy(value_head_mode="decoupled")
    obs, _ = _make_obs(B=2)
    value = p.get_value(obs)
    loss = value.pow(2).mean()
    loss.backward()

    # Policy encoder: every grad must be None or zero.
    for name, param in p.observation_module.named_parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0), (
                f"Policy encoder param {name!r} got non-zero gradient from value loss"
            )

    # Value encoder: at least one parameter must have received a gradient.
    any_value_grad = any(
        param.grad is not None and param.grad.abs().sum().item() > 0
        for param in p.value_observation_module.parameters()
    )
    assert any_value_grad


def test_build_agent_model_passes_phase2_kwargs() -> None:
    """``build_agent_model`` plumbs Phase 2 flags into the resulting policy."""
    p = build_agent_model(
        device="cpu",
        action_head_film=True,
        value_head_mode="decoupled",
        use_axial_pos_emb=True,
        transformer_dropout=0.05,
        transformer_activation="gelu",
    )
    assert p.action_heads.film is True
    assert p.value_head_mode == "decoupled"
    assert p.value_observation_module is not None
    assert p.observation_module.tile_encoder.use_axial_pos_emb is True
