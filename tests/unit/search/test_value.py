"""Value-squash + leaf evaluator (Phase 2 / T003, contract C1)."""

from __future__ import annotations

import math

import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.search.value import (
    VALUE_SQUASH_A,
    VALUE_SQUASH_B,
    leaf_value,
    squash_value,
    value_from_obs,
)

from .conftest import drive_to_main_phase


def test_squash_is_bounded_open_unit_interval() -> None:
    # Raw V exceeds [-1, 1] for ~27% of states — squash must still land in (0,1).
    for v in [-5.0, -1.6, -1.0, 0.0, 0.354, 1.0, 1.8, 5.0]:
        p = squash_value(v)
        assert isinstance(p, float)
        assert 0.0 < p < 1.0


def test_squash_is_strictly_monotone() -> None:
    vs = [x / 10.0 for x in range(-50, 51)]
    ps = [squash_value(v) for v in vs]
    assert all(isinstance(p, float) for p in ps)
    assert all(ps[i] < ps[i + 1] for i in range(len(ps) - 1))


def test_squash_constants_match_fit() -> None:
    # P(win | V=0) = sigmoid(b) ≈ 0.242; the 0.5 crossover is at V = -b/a ≈ 0.354.
    assert math.isclose(squash_value(0.0), 1.0 / (1.0 + math.exp(-VALUE_SQUASH_B)), rel_tol=1e-9)
    crossover = -VALUE_SQUASH_B / VALUE_SQUASH_A
    assert math.isclose(squash_value(crossover), 0.5, abs_tol=1e-9)


def test_squash_float_and_tensor_agree() -> None:
    v = 0.731
    f = squash_value(v)
    t = squash_value(torch.tensor(v))
    assert isinstance(f, float)
    assert isinstance(t, torch.Tensor)
    assert math.isclose(f, float(t.item()), rel_tol=1e-6)


def test_leaf_value_in_unit_interval(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)
    p = leaf_value(policy, env)
    assert 0.0 < p < 1.0


def test_leaf_value_uses_agent_pov_obs(policy) -> None:  # type: ignore[no-untyped-def]
    # leaf_value reads the env's canonical (agent-POV) obs; assert it equals an
    # explicit value_from_obs on that same obs. No per-ply sign flip is applied
    # because the env folds the opponent turn into EndTurn (all nodes agent-POV).
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=1)
    assert drive_to_main_phase(env)
    device = next(policy.parameters()).device
    via_leaf = leaf_value(policy, env)
    via_obs = value_from_obs(policy, env._get_obs(), device=device)
    assert math.isclose(via_leaf, via_obs, rel_tol=1e-6)
    assert 0.0 < via_leaf < 1.0
