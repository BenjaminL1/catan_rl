"""T011 — the snapshot opponent drives the opponent seat via its policy (US1).

Acceptance (spec Scenario 1): with a frozen opponent injected, a full game
(setup + main turns) completes with no error and the opponent's actions come
from the injected policy, NOT the heuristic. The stub plays legal setup
placements then EndTurns every main turn — so a heuristic opponent (which would
build) is ruled out by the opponent never advancing past its 2 setup
settlements.
"""

from __future__ import annotations

import numpy as np
import torch

from catan_rl.env.catan_env import ActionType, CatanEnv


def _first_true(mask: object) -> int:
    nz = np.nonzero(np.asarray(mask))[0]
    return int(nz[0]) if len(nz) else 0


def _fill_action(m: dict, *, prefer_end_turn: bool) -> list[int]:
    """First-legal action for the given mask dict (numpy arrays, no batch dim)."""
    action = [0, 0, 0, 0, 0, 0]
    if prefer_end_turn and bool(m["type"][ActionType.END_TURN]):
        action[0] = ActionType.END_TURN
        return action
    atype = _first_true(m["type"])
    action[0] = atype
    if atype == ActionType.BUILD_SETTLEMENT:
        action[1] = _first_true(m["corner_settlement"])
    elif atype == ActionType.BUILD_CITY:
        action[1] = _first_true(m["corner_city"])
    elif atype == ActionType.BUILD_ROAD:
        action[2] = _first_true(m["edge"])
    elif atype == ActionType.MOVE_ROBBER:
        action[3] = _first_true(m["tile"])
    elif atype == ActionType.DISCARD:
        action[4] = _first_true(m["resource1_discard"])
    return action


class _SetupThenEndTurnStub:
    """Frozen-opponent stub: legal setup placements, EndTurn in main turns."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls = 0

    def reset_rng(self, seed: int | None = None) -> None:
        pass

    def sample(self, obs: dict, masks: dict) -> torch.Tensor:
        self.calls += 1
        m = {k: v[0].numpy() for k, v in masks.items()}
        return torch.tensor([_fill_action(m, prefer_end_turn=True)], dtype=torch.int64)


def _agent_action(env: CatanEnv) -> np.ndarray:
    return np.asarray(_fill_action(env.get_action_masks(), prefer_end_turn=True), dtype=np.int64)


def test_snapshot_opponent_drives_via_policy_not_heuristic() -> None:
    env = CatanEnv(opponent_type="snapshot")
    stub = _SetupThenEndTurnStub()
    env.set_snapshot_opponent(stub)
    assert env.has_snapshot_opponent

    env.reset(seed=0)
    done = False
    steps = 0
    while not done and steps < 3000:
        _, _, term, trunc, _ = env.step(_agent_action(env))
        done = term or trunc
        steps += 1

    assert done, "game did not terminate"
    assert stub.calls > 0, "opponent actions never came from the stub policy"
    opp = env.opponent_player
    # The stub never builds in main turns -> opponent stuck at its 2 setup
    # settlements; a heuristic opponent would have built past this.
    assert len(opp.buildGraph["SETTLEMENTS"]) == 2
    assert len(opp.buildGraph["CITIES"]) == 0


def test_no_snapshot_falls_back_to_heuristic() -> None:
    """FR-011: a snapshot env with no injected policy plays via the heuristic."""
    env = CatanEnv(opponent_type="snapshot")
    assert not env.has_snapshot_opponent
    env.reset(seed=1)
    done = False
    steps = 0
    while not done and steps < 3000:
        _, _, term, trunc, _ = env.step(_agent_action(env))
        done = term or trunc
        steps += 1
    assert done  # heuristic opponent path runs without error
