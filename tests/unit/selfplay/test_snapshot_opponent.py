"""T008 — frozen snapshot opponent helper: inference-only + RNG isolation.

Pins the two invariants from the senior-RL review: the snapshot policy is
never trainable (eval / no-grad / requires_grad False), and its stochastic
sampling is reproducible AND does not perturb the learner's global RNG stream
(FR-006) — the latter is the property that keeps before/after self-play
comparisons reproducible.
"""

from __future__ import annotations

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy
from catan_rl.selfplay.snapshot_opponent import build_snapshot_opponent

_CPU = torch.device("cpu")


def _geometry() -> dict:
    return build_geometry().as_dict_of_tensors()


def _snapshot_state() -> dict:
    policy = CatanPolicy()
    policy.set_board_geometry(_geometry())
    return {k: v.clone() for k, v in policy.state_dict().items()}


def _batched_obs_masks(device: torch.device) -> tuple[dict, dict]:
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)
    obs = env._get_obs()
    masks = env.get_action_masks()
    obs_t = {k: torch.as_tensor(np.expand_dims(v, 0), device=device) for k, v in obs.items()}
    masks_t = {
        k: torch.as_tensor(np.expand_dims(v, 0), device=device, dtype=torch.bool)
        for k, v in masks.items()
    }
    return obs_t, masks_t


def test_snapshot_opponent_is_frozen_and_eval() -> None:
    opp = build_snapshot_opponent(_snapshot_state(), geometry=_geometry(), device=_CPU, seed=0)
    assert not opp.policy.training
    assert all(not p.requires_grad for p in opp.policy.parameters())


def test_sample_does_not_perturb_global_rng() -> None:
    """FR-006: the learner's rollout RNG stream is untouched by opponent draws."""
    obs_t, masks_t = _batched_obs_masks(_CPU)
    opp = build_snapshot_opponent(_snapshot_state(), geometry=_geometry(), device=_CPU, seed=123)

    torch.manual_seed(999)
    before = torch.random.get_rng_state()
    for _ in range(5):
        opp.sample(obs_t, masks_t)
    after = torch.random.get_rng_state()

    assert torch.equal(before, after)


def test_sample_is_reproducible_given_seed() -> None:
    obs_t, masks_t = _batched_obs_masks(_CPU)
    state = _snapshot_state()
    o1 = build_snapshot_opponent(state, geometry=_geometry(), device=_CPU, seed=42)
    o2 = build_snapshot_opponent(state, geometry=_geometry(), device=_CPU, seed=42)

    seq1 = [o1.sample(obs_t, masks_t).clone() for _ in range(4)]
    seq2 = [o2.sample(obs_t, masks_t).clone() for _ in range(4)]
    for a, b in zip(seq1, seq2, strict=True):
        assert torch.equal(a, b)


def test_reset_rng_restarts_the_stream() -> None:
    obs_t, masks_t = _batched_obs_masks(_CPU)
    opp = build_snapshot_opponent(_snapshot_state(), geometry=_geometry(), device=_CPU, seed=7)

    first = [opp.sample(obs_t, masks_t).clone() for _ in range(3)]
    opp.reset_rng()
    again = [opp.sample(obs_t, masks_t).clone() for _ in range(3)]
    for a, b in zip(first, again, strict=True):
        assert torch.equal(a, b)


def test_sample_returns_batched_action() -> None:
    obs_t, masks_t = _batched_obs_masks(_CPU)
    opp = build_snapshot_opponent(_snapshot_state(), geometry=_geometry(), device=_CPU, seed=0)
    action = opp.sample(obs_t, masks_t)
    # MultiDiscrete([13,54,72,19,5,5]) -> 6 head indices, batch of 1.
    assert action.shape == (1, 6)
