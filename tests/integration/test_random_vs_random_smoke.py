"""Random-vs-random smoke gate for the v2 env.

v2_design.md §5 Step 1 termination gate: "random-vs-random plays 1000 games
to termination without errors." This test runs a smaller batch (50 games)
locally; the full 1000-game run lives in ``scripts/smoke_random_vs_random.py``
so the test suite stays fast.

What we assert:
  * 50/50 games reach a terminal/truncated state with no exceptions.
  * At least one game terminates (i.e. the agent or opponent reaches 15 VP)
    rather than truncating on the turn cap — guards against the env getting
    stuck rolling forever without ever progressing.
  * Reward distribution is bounded: ``|reward| <= 1.0 + (15 * vp_margin_bonus)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.env.catan_env import CatanEnv


def _sample_action_from_mask(masks: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Uniform-random action sampled subject to the v2 mask.

    Mirrors what the policy network's autoregressive heads will do: pick a
    type uniformly over its mask, then pick each per-type continuation. The
    smoke test does not care which continuations are physically meaningful
    for a given type — masks for the irrelevant heads are filled with
    fallback-friendly defaults so any uniform pick is legal.
    """
    type_legal = np.flatnonzero(masks["type"])
    assert type_legal.size > 0, "every game state must have at least one legal action"
    t = int(rng.choice(type_legal))

    corner_idx = 0
    edge_idx = 0
    tile_idx = 0
    r1 = 0
    r2 = 0

    if t == 0:  # BuildSettlement
        legal = np.flatnonzero(masks["corner_settlement"])
        if legal.size:
            corner_idx = int(rng.choice(legal))
    elif t == 1:  # BuildCity
        legal = np.flatnonzero(masks["corner_city"])
        if legal.size:
            corner_idx = int(rng.choice(legal))
    elif t == 2:  # BuildRoad
        legal = np.flatnonzero(masks["edge"])
        if legal.size:
            edge_idx = int(rng.choice(legal))
    elif t == 4:  # MoveRobber
        legal = np.flatnonzero(masks["tile"])
        if legal.size:
            tile_idx = int(rng.choice(legal))
    elif t in (7,):  # PlayYoP
        legal1 = np.flatnonzero(masks["resource1_default"])
        legal2 = np.flatnonzero(masks["resource2_default"])
        if legal1.size:
            r1 = int(rng.choice(legal1))
        if legal2.size:
            r2 = int(rng.choice(legal2))
    elif t == 8:  # PlayMonopoly
        legal = np.flatnonzero(masks["resource1_default"])
        if legal.size:
            r1 = int(rng.choice(legal))
    elif t == 10:  # BankTrade
        legal1 = np.flatnonzero(masks["resource1_trade"])
        legal2 = np.flatnonzero(masks["resource2_default"])
        if legal1.size:
            r1 = int(rng.choice(legal1))
        if legal2.size:
            r2 = int(rng.choice(legal2))
    elif t == 11:  # Discard
        legal = np.flatnonzero(masks["resource1_discard"])
        if legal.size:
            r1 = int(rng.choice(legal))

    return np.array([t, corner_idx, edge_idx, tile_idx, r1, r2], dtype=np.int64)


def _play_one_game(
    env: CatanEnv, rng: np.random.Generator, max_steps: int = 5000
) -> tuple[bool, bool, float, int]:
    _obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    total_reward = 0.0
    for step_i in range(max_steps):
        masks = env.get_action_masks()
        action = _sample_action_from_mask(masks, rng)
        _obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return terminated, truncated, total_reward, step_i + 1
    raise RuntimeError(
        f"Game did not terminate within {max_steps} env-steps; possible infinite loop."
    )


@pytest.mark.parametrize("opponent_type", ["random"])
def test_random_vs_random_smoke(opponent_type: str) -> None:
    rng = np.random.default_rng(0)
    env = CatanEnv(opponent_type=opponent_type, max_turns=400)
    n_games = 50

    terminated_count = 0
    truncated_count = 0
    reward_max = 0.0
    length_sum = 0

    for _ in range(n_games):
        terminated, truncated, reward, length = _play_one_game(env, rng)
        assert terminated or truncated
        terminated_count += int(terminated)
        truncated_count += int(truncated)
        reward_max = max(reward_max, abs(reward))
        length_sum += length

    assert terminated_count + truncated_count == n_games
    assert terminated_count > 0, (
        f"All {n_games} games truncated on turn cap — env is failing to make progress"
    )
    # |reward| <= 1.0 + 15 * 1/15 = 2.0 for the default vp_margin_bonus.
    assert reward_max <= 2.05, f"unexpected reward magnitude: {reward_max}"
    avg_len = length_sum / n_games
    assert 10 <= avg_len <= 5000, f"unreasonable avg game length: {avg_len}"
