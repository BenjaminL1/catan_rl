"""Determinized PUCT-MCTS mechanics (T007, contract C3 internals)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from catan_rl.env.catan_env import ActionType, CatanEnv
from catan_rl.search.config import SearchConfig
from catan_rl.search.mcts import MCTS, _puct_select, clone_env
from catan_rl.search.node import SearchNode, agent_outcome, build_node
from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

from .conftest import drive_to_decision


def _mcts(policy, cfg: SearchConfig) -> MCTS:  # type: ignore[no-untyped-def]
    device = next(policy.parameters()).device
    opp = FrozenSnapshotOpponent(policy, device=device, seed=cfg.seed)
    return MCTS(policy, cfg, opp, device)


def _seed(mcts: MCTS, cfg: SearchConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    mcts.opponent.reset_rng(seed=cfg.seed)  # type: ignore[union-attr]


# --- pure-mechanic unit tests (no policy needed) ---------------------------


def test_puct_prefers_higher_prior_when_all_unvisited() -> None:
    node = SearchNode(
        env=None,
        obs={},
        is_terminal=False,
        outcome=0.0,
        value=0.5,
        priors={(3, 0, 0, 0, 0, 0): 0.2, (2, 0, 1, 0, 0, 0): 0.8},
    )
    # Q=0 for both unvisited -> the U term ∝ prior decides.
    assert _puct_select(node, 1.5) == (2, 0, 1, 0, 0, 0)


def test_record_and_q_value_average() -> None:
    node = SearchNode(
        env=None,
        obs={},
        is_terminal=False,
        outcome=0.0,
        value=0.5,
        priors={(3, 0, 0, 0, 0, 0): 1.0},
    )
    a = (3, 0, 0, 0, 0, 0)
    assert node.q_value(a) == 0.0  # FPU for an unvisited edge
    node.record(a, 0.7)
    node.record(a, 0.3)
    assert node.total_N == 2
    assert math.isclose(node.q_value(a), 0.5)


def test_agent_outcome_sign() -> None:
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)
    assert env.agent_player is not None and env.opponent_player is not None
    env.agent_player.victoryPoints = 15
    env.opponent_player.victoryPoints = 5
    assert agent_outcome(env) == 1.0  # agent win
    env.agent_player.victoryPoints = 5
    env.opponent_player.victoryPoints = 15
    assert agent_outcome(env) == 0.0  # agent loss
    env.agent_player.victoryPoints = 15
    env.opponent_player.victoryPoints = 15
    assert agent_outcome(env) == 0.0  # tie is not a win (must strictly exceed)


# --- full-search tests (need the policy fixture) ---------------------------


def test_run_spends_budget_and_returns_legal_action(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_decision(env)
    cfg = SearchConfig(sims_per_move=12, seed=0)
    mcts = _mcts(policy, cfg)
    _seed(mcts, cfg)
    masks = env.get_action_masks()
    action, diag = mcts.run(env)
    assert diag["sims_run"] == 12
    assert diag["root_visits"] == 12
    assert bool(masks["type"][action[0]]), "chosen type must be legal"
    assert 0.0 <= diag["best_q"] <= 1.0


def test_backup_values_are_bounded_agent_pov_probabilities(policy) -> None:  # type: ignore[no-untyped-def]
    # Direct backup-sign guard: every backed-up Q must be a probability in [0,1].
    # A perspective-sign error or a raw-V leak would push some Q negative or > 1.
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_decision(env)
    cfg = SearchConfig(sims_per_move=16, seed=0)
    mcts = _mcts(policy, cfg)
    _seed(mcts, cfg)
    device = next(policy.parameters()).device
    root = build_node(
        policy,
        clone_env(env, mcts.opponent),
        env._get_obs(),
        terminal=False,
        device=device,
        a=cfg.value_squash_a,
        b=cfg.value_squash_b,
    )
    for _ in range(16):
        mcts._simulate(root)
    assert root.total_N == 16
    assert sum(root.child_N.values()) == 16
    for action in root.legal_actions:
        if root.child_N.get(action, 0) > 0:
            assert 0.0 <= root.q_value(action) <= 1.0


def test_forced_root_short_circuits_without_budget(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)  # setup state: only BUILD_SETTLEMENT legal -> one representative action
    cfg = SearchConfig(sims_per_move=50, seed=0)
    mcts = _mcts(policy, cfg)
    _seed(mcts, cfg)
    action, diag = mcts.run(env)
    assert diag["forced"] is True
    assert diag["sims_run"] == 0
    assert action[0] == ActionType.BUILD_SETTLEMENT


def test_run_is_deterministic_under_fixed_seed(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_decision(env)
    cfg = SearchConfig(sims_per_move=12, seed=0)
    mcts = _mcts(policy, cfg)
    _seed(mcts, cfg)
    a1, d1 = mcts.run(env)
    _seed(mcts, cfg)
    a2, d2 = mcts.run(env)
    assert tuple(a1) == tuple(a2)
    assert d1["best_visits"] == d2["best_visits"]
    assert math.isclose(d1["best_q"], d2["best_q"])


def test_search_is_lookahead_not_noise(policy) -> None:  # type: ignore[no-untyped-def]
    # The core "lookahead works" guarantee: given two equally-prior actions, one
    # leading to a terminal WIN (outcome 1.0) and one to a terminal LOSS (0.0),
    # PUCT + backup must converge on the winning action — search EXPLOITS the
    # value signal, it doesn't just add noise. A perspective-sign flip or a broken
    # backup would pick the losing action / leave the Q-values wrong.
    cfg = SearchConfig(sims_per_move=30, seed=0)
    mcts = _mcts(policy, cfg)
    win = (0, 5, 0, 0, 0, 0)
    lose = (3, 0, 0, 0, 0, 0)
    root = SearchNode(
        env=None,
        obs={},
        is_terminal=False,
        outcome=0.0,
        value=0.5,
        priors={win: 0.5, lose: 0.5},
    )
    root.children[win] = SearchNode(
        env=None, obs={}, is_terminal=True, outcome=1.0, value=0.0, priors={}
    )
    root.children[lose] = SearchNode(
        env=None, obs={}, is_terminal=True, outcome=0.0, value=0.0, priors={}
    )
    for _ in range(cfg.sims_per_move):
        mcts._simulate(root)
    assert mcts._best_action(root) == win
    assert root.child_N[win] > root.child_N.get(lose, 0)
    assert math.isclose(root.q_value(win), 1.0)
    assert root.q_value(lose) == 0.0  # 0.0 whether visited (loss) or unvisited (FPU)


def test_time_budget_mode_not_supported_in_minimal_mcts(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_decision(env)  # >1 legal action so the budget path is reached
    cfg = SearchConfig(sims_per_move=None, time_budget_s=1.0, seed=0)
    mcts = _mcts(policy, cfg)
    with pytest.raises(NotImplementedError, match="time-budget"):
        mcts.run(env)
