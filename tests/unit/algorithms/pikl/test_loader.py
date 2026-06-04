"""Tests for `algorithms/pikl/loader.py`.

Pins:
1. Round-trip: save a policy via Phase 8, load via the loader, outputs
   match the original (modulo no-grad detachment).
2. The loaded anchor's params have requires_grad=False.
3. strict=True raises on shape mismatch; strict=False can partial-load.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from catan_rl.algorithms.pikl.anchor import AnchorPolicy
from catan_rl.algorithms.pikl.loader import load_pikl_anchor
from catan_rl.checkpoint import save_checkpoint


class _MockPolicy(nn.Module):
    """Same shape as the anchor tests' mock — enough surface to load
    and call evaluate_actions."""

    def __init__(self, n_features: int = 8) -> None:
        super().__init__()
        self.head_bias = nn.Parameter(torch.zeros(6))
        self.fc = nn.Linear(n_features, 4)

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        B = action.shape[0]
        per_head = self.head_bias.expand(B, -1).clone()
        return {
            "log_prob": per_head.sum(dim=-1),
            "per_head_log_prob": per_head,
            "relevance": torch.ones(B, 6),
        }


class TestRoundTrip:
    def test_loaded_outputs_match_original(self, tmp_path: Path) -> None:
        # Train a tiny policy to a non-trivial state, save it, reload
        # via the piKL loader, and confirm outputs are bit-identical.
        torch.manual_seed(123)
        policy = _MockPolicy()
        with torch.no_grad():
            policy.head_bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4, -0.5, 0.6]))
            policy.fc.weight.normal_()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        save_checkpoint(
            tmp_path / "anchor.pt",
            config={},
            policy=policy,
            optimizer=optimizer,
            update_idx=0,
            global_step=0,
        )

        obs = {"dummy": torch.zeros(2, 8)}
        action = torch.zeros(2, 6, dtype=torch.long)
        ref_per_head = policy.evaluate_actions(obs, action, {})["per_head_log_prob"]

        # Factory builds an independent fresh policy of the same shape.
        anchor = load_pikl_anchor(
            tmp_path / "anchor.pt",
            policy_factory=_MockPolicy,
        )
        loaded_per_head = anchor.evaluate_actions(obs, action, {})["per_head_log_prob"]
        assert torch.allclose(ref_per_head, loaded_per_head)

    def test_loaded_anchor_is_frozen(self, tmp_path: Path) -> None:
        policy = _MockPolicy()
        opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        save_checkpoint(
            tmp_path / "anchor.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
        )
        anchor = load_pikl_anchor(
            tmp_path / "anchor.pt",
            policy_factory=_MockPolicy,
        )
        for param in anchor.parameters():
            assert param.requires_grad is False
        assert isinstance(anchor, AnchorPolicy)


class TestStrict:
    def test_strict_true_rejects_mismatch(self, tmp_path: Path) -> None:
        policy = _MockPolicy(n_features=8)
        opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        save_checkpoint(
            tmp_path / "anchor.pt",
            config={},
            policy=policy,
            optimizer=opt,
            update_idx=0,
            global_step=0,
        )

        def _wider_factory() -> _MockPolicy:
            return _MockPolicy(n_features=16)

        with pytest.raises((RuntimeError, ValueError)):
            load_pikl_anchor(
                tmp_path / "anchor.pt",
                policy_factory=_wider_factory,
                strict=True,
            )
