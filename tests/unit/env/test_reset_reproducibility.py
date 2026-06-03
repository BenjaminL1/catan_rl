"""Regression test for review fix #2, 2026-06-03.

``CatanEnv.reset(seed=s)`` must reseed both the numpy global RNG AND the
stdlib ``random`` global. ``StackedDice`` (engine/dice.py) draws from
the stdlib stream — leaving it unseeded made eval gating decisions
non-reproducible (re-running "A vs B at seed=42" yielded different
outcomes).

Test strategy: call ``env.reset(seed=42)`` twice, drive the same action
sequence each time, and assert the trajectories are bit-identical.
"""

from __future__ import annotations

import numpy as np

from catan_rl.env.catan_env import CatanEnv


def _drive_n_steps(env: CatanEnv, n: int) -> list[float]:
    """Drive ``n`` random-legal-action steps; return the dice-roll sequence."""
    rolls: list[float] = []
    _, _ = env.reset(seed=42)
    rng = np.random.default_rng(123)
    for _ in range(n):
        masks = env.get_action_masks()
        legal = np.flatnonzero(masks["type"])
        if legal.size == 0:
            break
        t = int(rng.choice(legal))
        action = np.zeros(6, dtype=np.int64)
        action[0] = t
        for head_idx, key in enumerate(
            ("corner_settlement", "edge", "tile", "resource1_default", "resource2_default"),
            start=1,
        ):
            if key == "corner_settlement" and t == 1:
                key = "corner_city"
            if key == "resource1_default" and t == 11:
                key = "resource1_discard"
            elif key == "resource1_default" and t == 10:
                key = "resource1_trade"
            head_legal = np.flatnonzero(masks[key])
            if head_legal.size:
                action[head_idx] = int(rng.choice(head_legal))
        _, _, terminated, truncated, _ = env.step(action)
        rolls.append(float(env.last_dice_roll))
        if terminated or truncated:
            break
    return rolls


def test_reset_seed_reproduces_dice_roll_sequence() -> None:
    """Two independent envs seeded with the same value must produce the
    same dice-roll sequence under identical action policies.

    Pre-fix: ``random.seed`` was never called inside reset, so
    ``StackedDice`` drew from whatever state the stdlib global happened
    to be in — runs diverged within a few rolls.
    """
    env_a = CatanEnv(opponent_type="random", max_turns=200)
    env_b = CatanEnv(opponent_type="random", max_turns=200)
    rolls_a = _drive_n_steps(env_a, n=40)
    rolls_b = _drive_n_steps(env_b, n=40)
    assert rolls_a == rolls_b, (
        f"reset(seed=42) produced different dice rolls across runs:\n"
        f"  run A: {rolls_a}\n"
        f"  run B: {rolls_b}"
    )


def test_two_resets_in_same_process_match() -> None:
    """Even within a single process, two consecutive ``reset(seed=42)``
    calls on the same env must yield the same dice sequence — the env
    must restore the stdlib RNG, not merely advance it."""
    env = CatanEnv(opponent_type="random", max_turns=200)
    rolls_a = _drive_n_steps(env, n=30)
    rolls_b = _drive_n_steps(env, n=30)
    assert rolls_a == rolls_b
