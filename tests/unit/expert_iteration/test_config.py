"""SearchLabelConfig / DistillConfig validation (T004)."""

from __future__ import annotations

import pytest

from catan_rl.expert_iteration.config import DistillConfig, SearchLabelConfig


def test_label_config_defaults() -> None:
    cfg = SearchLabelConfig(out_dir="d")
    assert cfg.sims_per_move == 50
    assert cfg.opponent == "heuristic"
    assert cfg.n_positions == 5000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sims_per_move": 0},
        {"n_positions": 0},
        {"discount": 0.0},
        {"discount": 1.5},
        {"max_turns": 0},
        {"opponent": "bogus"},
    ],
)
def test_label_config_invalid(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SearchLabelConfig(out_dir="d", **kwargs)  # type: ignore[arg-type]


def test_label_config_policy_opponent_ok() -> None:
    cfg = SearchLabelConfig(out_dir="d", opponent="policy:/tmp/x.pt")
    assert cfg.opponent.startswith("policy:")


def test_distill_config_defaults_and_validation() -> None:
    cfg = DistillConfig(data_dir="d", out_dir="o")
    assert cfg.peak_lr == 5e-5  # low: fine-tune, not from-scratch (research D8)
    assert cfg.max_epochs == 5
    for bad in ({"peak_lr": 0.0}, {"max_epochs": 0}, {"batch_size": 0}):
        with pytest.raises(ValueError):
            DistillConfig(data_dir="d", out_dir="o", **bad)  # type: ignore[arg-type]
