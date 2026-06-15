"""Shared fixtures/helpers for search unit tests.

A real (randomly-initialised) ``CatanPolicy`` is enough for STRUCTURAL tests
(priors are legal+normalised, value is bounded, search is deterministic) — none
of these assertions depend on the weights being trained.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.env.catan_env import ActionType, CatanEnv
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy


def make_policy() -> CatanPolicy:
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy.eval()
    return policy


@pytest.fixture(scope="module")
def policy() -> CatanPolicy:
    return make_policy()


def _first_true(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else 0


def legal_action(env: CatanEnv) -> np.ndarray:
    """A valid 6-action: first legal type + first legal sub-index per relevant head."""
    masks = env.get_action_masks()
    action = np.zeros(6, dtype=np.int64)
    legal_types = np.flatnonzero(masks["type"])
    if legal_types.size == 0:
        action[0] = ActionType.END_TURN
        return action
    t = int(legal_types[0])
    action[0] = t
    if t == ActionType.BUILD_SETTLEMENT:
        action[1] = _first_true(masks["corner_settlement"])
    elif t == ActionType.BUILD_CITY:
        action[1] = _first_true(masks["corner_city"])
    if t == ActionType.BUILD_ROAD:
        action[2] = _first_true(masks["edge"])
    if t in (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT):
        action[3] = _first_true(masks["tile"])
    if t in (ActionType.PLAY_YOP, ActionType.PLAY_MONOPOLY):
        action[4] = _first_true(masks["resource1_default"])
    elif t == ActionType.BANK_TRADE:
        action[4] = _first_true(masks["resource1_trade"])
    elif t == ActionType.DISCARD:
        action[4] = _first_true(masks["resource1_discard"])
    if t in (ActionType.PLAY_YOP, ActionType.BANK_TRADE):
        action[5] = _first_true(masks["resource2_default"])
    return action


def at_main_phase(env: CatanEnv) -> bool:
    return (
        not env.initial_placement_phase
        and not env.roll_pending
        and not env.discard_pending
        and not env.robber_placement_pending
        and env.road_building_roads_left == 0
    )


def drive_to_main_phase(env: CatanEnv, *, max_steps: int = 300) -> bool:
    """Step legal actions through setup + the first roll to a main-phase decision.

    Returns True if a main-phase decision was reached, False if the game ended
    first (rare at tiny seeds). The guard is checked BEFORE stepping, so the env
    is left exactly at the first agent main-phase decision point.
    """
    for _ in range(max_steps):
        if at_main_phase(env):
            return True
        _, _, terminated, truncated, _ = env.step(legal_action(env))
        if terminated or truncated:
            return False
    return False
