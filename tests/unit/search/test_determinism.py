"""Determinism (T020, SC-003): same seed + budget reproduces identical actions."""

from __future__ import annotations

from catan_rl.env.catan_env import CatanEnv
from catan_rl.search.agent import SearchAgent
from catan_rl.search.config import SearchConfig

from .conftest import drive_to_decision


def _play(policy, *, seed: int, n_moves: int, sims: int, n_det: int = 1) -> list[tuple[int, ...]]:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=seed)
    agent = SearchAgent(policy, SearchConfig(sims_per_move=sims, seed=0, n_determinizations=n_det))
    seq: list[tuple[int, ...]] = []
    for _ in range(n_moves):
        action = agent.choose_action(env)
        seq.append(tuple(int(x) for x in action))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return seq


def test_same_seed_budget_reproduces_action_sequence_n_det(policy) -> None:  # type: ignore[no-untyped-def]
    # Exercises the n_determinizations>1 aggregation path through SearchAgent
    # (the FR-006 RNG snapshot/restore boundary) — complements test_agent's n_det=1.
    a = _play(policy, seed=11, n_moves=8, sims=5, n_det=2)
    b = _play(policy, seed=11, n_moves=8, sims=5, n_det=2)
    assert a == b


def test_single_state_search_is_deterministic(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=4)
    assert drive_to_decision(env)
    agent = SearchAgent(policy, SearchConfig(sims_per_move=8, seed=0))
    a1 = agent.choose_action(env)
    v1 = agent.last_diagnostics["best_visits"]
    a2 = agent.choose_action(env)
    v2 = agent.last_diagnostics["best_visits"]
    assert tuple(a1) == tuple(a2)
    assert v1 == v2
