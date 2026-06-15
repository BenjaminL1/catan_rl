"""SearchAgent decision surface (T010, contract C3)."""

from __future__ import annotations

import random

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.search.agent import SearchAgent
from catan_rl.search.config import SearchConfig

from .conftest import drive_to_main_phase


def test_choose_action_returns_legal_6tuple(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_main_phase(env)
    agent = SearchAgent(policy, SearchConfig(sims_per_move=6, seed=0))
    masks = env.get_action_masks()
    action = agent.choose_action(env)
    assert isinstance(action, np.ndarray)
    assert action.shape == (6,)
    assert action.dtype == np.int64
    assert bool(masks["type"][action[0]])


def test_forced_move_short_circuits(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)  # setup -> forced
    agent = SearchAgent(policy, SearchConfig(sims_per_move=100, seed=0))
    agent.choose_action(env)
    assert agent.last_diagnostics["forced"] is True
    assert agent.last_diagnostics["sims_run"] == 0


def test_choose_action_does_not_mutate_env(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_main_phase(env)
    assert env.agent_player is not None and env.opponent_player is not None
    agent = SearchAgent(policy, SearchConfig(sims_per_move=6, seed=0))
    before = (
        env._setup_step,
        env.roll_pending,
        env.initial_placement_phase,
        env._turn_count,
        int(env.agent_player.victoryPoints),
        int(env.opponent_player.victoryPoints),
        dict(env.agent_player.resources),
        dict(env.opponent_player.resources),
    )
    opp_ref_before = env._snapshot_opponent
    agent.choose_action(env)
    after = (
        env._setup_step,
        env.roll_pending,
        env.initial_placement_phase,
        env._turn_count,
        int(env.agent_player.victoryPoints),
        int(env.opponent_player.victoryPoints),
        dict(env.agent_player.resources),
        dict(env.opponent_player.resources),
    )
    assert before == after
    assert env._snapshot_opponent is opp_ref_before  # opponent detach/restore is exact


def test_global_rng_is_unperturbed(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_main_phase(env)
    agent = SearchAgent(policy, SearchConfig(sims_per_move=6, seed=0))
    np_before = np.random.get_state()
    py_before = random.getstate()
    torch_before = torch.random.get_rng_state()
    agent.choose_action(env)
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert random.getstate() == py_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_identical_seed_reproduces_action_sequence(policy) -> None:  # type: ignore[no-untyped-def]
    def play(n_moves: int) -> list[tuple[int, ...]]:
        env = CatanEnv(opponent_type="heuristic")
        env.reset(seed=11)
        agent = SearchAgent(policy, SearchConfig(sims_per_move=6, seed=0))
        seq: list[tuple[int, ...]] = []
        for _ in range(n_moves):
            action = agent.choose_action(env)
            seq.append(tuple(int(x) for x in action))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return seq

    assert play(8) == play(8)
