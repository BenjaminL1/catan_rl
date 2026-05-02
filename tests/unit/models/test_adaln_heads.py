"""Tests for Phase 2.4 AdaLN/FiLM-conditioned action heads."""

from __future__ import annotations

import torch

from catan_rl.models.action_heads_module import (
    N_ACTION_TYPES,
    N_CORNERS,
    N_RESOURCES,
    ActionHead,
    MultiActionHeads,
)


def test_actionhead_legacy_concat_default() -> None:
    """Without ``film=True``, the head is the legacy concat MLP."""
    h = ActionHead(input_dim=512 + 2, output_dim=N_CORNERS, hidden_dim=128)
    assert h.film is False
    assert hasattr(h, "net")
    assert not hasattr(h, "film_gen")


def test_actionhead_film_construction() -> None:
    """``film=True`` builds the FiLM generator and explicit fc1/fc2 layers."""
    h = ActionHead(input_dim=512, output_dim=N_CORNERS, hidden_dim=128, context_dim=2, film=True)
    assert h.film is True
    # FiLM-specific submodules exist.
    assert hasattr(h, "fc1") and hasattr(h, "fc2")
    assert hasattr(h, "film_gen")
    # FiLM generator outputs (γ_1, β_1, γ_2, β_2).
    assert h.film_gen.out_features == 4 * 128
    # γ-init: zeros so 1+γ=1 (identity at construction).
    assert torch.all(h.film_gen.weight == 0)
    assert torch.all(h.film_gen.bias == 0)


def test_actionhead_film_zero_context_equivalence() -> None:
    """At init (γ_gen weight/bias=0), FiLM with any context = FiLM with zero context."""
    torch.manual_seed(0)
    h = ActionHead(input_dim=512, output_dim=N_CORNERS, hidden_dim=128, context_dim=2, film=True)
    h.eval()
    main = torch.randn(4, 512)
    mask = torch.ones(4, N_CORNERS, dtype=torch.bool)

    out_zero = h(main, torch.zeros(4, 2), mask)
    out_random = h(main, torch.randn(4, 2), mask)
    # At init the film_gen produces zeros regardless of input, so γ=0, β=0,
    # and 1+γ=1 — both runs produce identical logits.
    torch.testing.assert_close(out_zero.logits, out_random.logits)


def test_multiactionheads_film_routes_context() -> None:
    """When ``film=True`` is passed, context-using heads use the FiLM path."""
    mh = MultiActionHeads(obs_output_dim=512, hidden_dim=128, film=True)
    # Context-using heads → film=True
    assert mh.corner_head.film is True
    assert mh.resource1_head.film is True
    assert mh.resource2_head.film is True
    # No-context heads stay on legacy path (film flag has no effect there).
    assert mh.type_head.film is False
    assert mh.edge_head.film is False
    assert mh.tile_head.film is False


def test_multiactionheads_film_off_is_legacy() -> None:
    """``film=False`` (default) keeps every head on the legacy concat path."""
    mh = MultiActionHeads(obs_output_dim=512, hidden_dim=128)
    assert mh.film is False
    for head in (mh.corner_head, mh.resource1_head, mh.resource2_head):
        assert head.film is False


def test_multiactionheads_film_forward_smoke() -> None:
    """Full forward pass with FiLM on — same shapes as legacy."""
    torch.manual_seed(0)
    mh = MultiActionHeads(obs_output_dim=512, hidden_dim=128, film=True)
    mh.eval()
    B = 4
    obs = torch.randn(B, 512)
    masks = {
        "type": torch.ones(B, N_ACTION_TYPES, dtype=torch.bool),
        "corner_settlement": torch.ones(B, N_CORNERS, dtype=torch.bool),
        "corner_city": torch.ones(B, N_CORNERS, dtype=torch.bool),
        "edge": torch.ones(B, 72, dtype=torch.bool),
        "tile": torch.ones(B, 19, dtype=torch.bool),
        "resource1_default": torch.ones(B, N_RESOURCES, dtype=torch.bool),
        "resource1_trade": torch.ones(B, N_RESOURCES, dtype=torch.bool),
        "resource1_discard": torch.ones(B, N_RESOURCES, dtype=torch.bool),
        "resource2_default": torch.ones(B, N_RESOURCES, dtype=torch.bool),
    }
    actions, log_prob, entropy = mh(obs, masks)
    assert actions.shape == (B, 6)
    assert log_prob.shape == (B,)
    assert entropy.dim() == 0


def test_multiactionheads_film_param_count_lower() -> None:
    """FiLM eliminates the concat-input width on context-using heads.

    Concat: head input is ``512 + ctx``. FiLM: head input is ``512`` plus a
    small ``ctx → 4*hidden`` generator. With ctx=2 and hidden=128, the FiLM
    gen is 2*512=1024 params vs concat saving (2*128)=256 weights — net cost
    is roughly the same. We just check the build doesn't grow absurdly.
    """
    legacy = MultiActionHeads(obs_output_dim=512, hidden_dim=128)
    film = MultiActionHeads(obs_output_dim=512, hidden_dim=128, film=True)

    n_legacy = sum(p.numel() for p in legacy.parameters() if p.requires_grad)
    n_film = sum(p.numel() for p in film.parameters() if p.requires_grad)
    # Both within 10% of each other — FiLM is roughly param-neutral.
    ratio = n_film / n_legacy
    assert 0.9 < ratio < 1.1, f"FiLM head param count ratio {ratio:.3f} outside [0.9, 1.1]"
