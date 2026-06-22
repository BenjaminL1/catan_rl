"""SearchConfig validation (Phase 2 / T002)."""

from __future__ import annotations

import dataclasses

import pytest

from catan_rl.search.config import SearchConfig
from catan_rl.search.value import VALUE_SQUASH_A, VALUE_SQUASH_B


def test_defaults_are_valid() -> None:
    cfg = SearchConfig()
    assert cfg.sims_per_move == 100
    assert cfg.time_budget_s is None
    assert cfg.n_determinizations == 1


def test_final_move_defaults_preserve_max_visit() -> None:
    # Spec 008 FR-002: the default final-move rule is the shipped max-visit one,
    # so adding the LCB config knobs is additive + default-off.
    cfg = SearchConfig()
    assert cfg.final_move_mode == "max_visit"
    assert cfg.lcb_z == 1.96


def test_lcb_mode_is_accepted() -> None:
    cfg = SearchConfig(final_move_mode="lcb", lcb_z=0.0)
    assert cfg.final_move_mode == "lcb"
    assert cfg.lcb_z == 0.0


def test_squash_defaults_track_value_module() -> None:
    # Single source of truth: the config defaults ARE the value-module constants,
    # so there is no second copy to drift.
    cfg = SearchConfig()
    assert cfg.value_squash_a == VALUE_SQUASH_A
    assert cfg.value_squash_b == VALUE_SQUASH_B


def test_sims_and_time_budget_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SearchConfig(sims_per_move=100, time_budget_s=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        SearchConfig(sims_per_move=None, time_budget_s=None)


def test_time_budget_mode() -> None:
    cfg = SearchConfig(sims_per_move=None, time_budget_s=2.5)
    assert cfg.time_budget_s == 2.5
    assert cfg.sims_per_move is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sims_per_move": 0},
        {"sims_per_move": -5},
        {"sims_per_move": None, "time_budget_s": 0.0},
        {"sims_per_move": None, "time_budget_s": -1.0},
        {"n_determinizations": 0},
        {"c_puct": 0.0},
        {"c_puct": -1.0},
        {"pw_c": 0.0},
        {"pw_alpha": 0.0},
        {"pw_alpha": 1.5},
        {"max_depth": 0},
        {"value_squash_a": 0.0},
        {"value_squash_a": -3.22},
        {"sub_actions_per_type": 0},
        {"sub_actions_per_type": -2},
        {"root_dirichlet_alpha": 0.0},
        {"root_dirichlet_alpha": -0.3},
        {"root_dirichlet_fraction": -0.1},
        {"root_dirichlet_fraction": 1.5},
        {"fpu_mode": "max"},
        {"fpu_mode": ""},
        {"final_move_mode": "max"},
        {"final_move_mode": ""},
        {"final_move_mode": "LCB"},
        {"lcb_z": -0.1},
        {"lcb_z": -1.96},
    ],
)
def test_invalid_values_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)  # type: ignore[arg-type]


def test_frozen() -> None:
    cfg = SearchConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.c_puct = 2.0  # type: ignore[misc]
