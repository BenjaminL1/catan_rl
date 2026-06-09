"""T011 — the snapshot opponent drives the opponent seat via its policy (US1).

Acceptance (spec Scenario 1): with a frozen opponent injected, a full game
(setup + main turns) completes with no error and the opponent's actions come
from the injected policy, NOT the heuristic. The stub plays legal setup
placements then EndTurns every main turn — so a heuristic opponent (which would
build) is ruled out by the opponent never advancing past its 2 setup
settlements.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from catan_rl.env.catan_env import ActionType, CatanEnv
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy
from catan_rl.selfplay.snapshot_opponent import build_snapshot_opponent


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


class _RobberSpamStub:
    """Setup: legal placement; main turns: never EndTurn (spam MoveRobber) so
    the opponent turn never terminates on its own — exercises the action cap."""

    device = torch.device("cpu")

    def reset_rng(self, seed: int | None = None) -> None:
        pass

    def sample(self, obs: dict, masks: dict) -> torch.Tensor:
        m = {k: v[0].numpy() for k, v in masks.items()}
        if bool(m["type"][ActionType.END_TURN]):  # main phase -> avoid EndTurn
            action = [ActionType.MOVE_ROBBER, 0, 0, _first_true(m["tile"]), 0, 0]
            return torch.tensor([action], dtype=torch.int64)
        return torch.tensor([_fill_action(m, prefer_end_turn=False)], dtype=torch.int64)


def test_action_cap_force_ends_runaway_opponent_turn(caplog: pytest.LogCaptureFixture) -> None:
    """T013/FR-013: a policy that never samples EndTurn is force-ended by the
    hard action cap, logged as an anomaly — no livelock."""
    env = CatanEnv(opponent_type="snapshot")
    env.set_snapshot_opponent(_RobberSpamStub())
    env._opp_turn_action_cap = 5
    env.reset(seed=0)
    with caplog.at_level(logging.WARNING):
        done = False
        steps = 0
        while not done and steps < 200:
            _, _, term, trunc, _ = env.step(_agent_action(env))
            done = term or trunc
            steps += 1
            if any("action cap" in r.getMessage() for r in caplog.records):
                break
    assert any("action cap" in r.getMessage() for r in caplog.records), "cap anomaly not logged"


def _run_snapshot_game(seed: int, state: dict, geometry: dict) -> tuple[int, int, int]:
    env = CatanEnv(opponent_type="snapshot")
    env.set_snapshot_opponent(
        build_snapshot_opponent(state, geometry=geometry, device=torch.device("cpu"), seed=7)
    )
    env.reset(seed=seed)
    done = False
    steps = 0
    while not done and steps < 3000:
        _, _, term, trunc, _ = env.step(_agent_action(env))
        done = term or trunc
        steps += 1
    return (
        int(env.opponent_player.victoryPoints),
        int(env.agent_player.victoryPoints),
        steps,
    )


def test_snapshot_opponent_game_is_deterministic() -> None:
    """T014/FR-006: same seed + same snapshot -> identical game (the opponent's
    isolated RNG + per-game reset_rng make it reproducible)."""
    geometry = build_geometry().as_dict_of_tensors()
    policy = CatanPolicy()
    policy.set_board_geometry(geometry)
    state = {k: v.clone() for k, v in policy.state_dict().items()}
    assert _run_snapshot_game(0, state, geometry) == _run_snapshot_game(0, state, geometry)


def test_opponent_robber_placement_uses_snapshot() -> None:
    """T012: the opponent's robber placement (the 7-roll resume path) is driven
    by the snapshot, not the heuristic, when one is injected."""
    calls: list[int] = []

    class _RobberStub:
        device = torch.device("cpu")

        def reset_rng(self, seed: int | None = None) -> None:
            pass

        def sample(self, obs: dict, masks: dict) -> torch.Tensor:
            calls.append(1)
            m = {k: v[0].numpy() for k, v in masks.items()}
            return torch.tensor(
                [[ActionType.MOVE_ROBBER, 0, 0, _first_true(m["tile"]), 0, 0]],
                dtype=torch.int64,
            )

    env = CatanEnv(opponent_type="snapshot")
    env.set_snapshot_opponent(_RobberStub())
    env.reset(seed=0)
    # Drive the agent through the 4 setup placements to reach main phase.
    for _ in range(4):
        env.step(_agent_action(env))
    env._opponent_move_robber()
    assert calls, "snapshot was not used for the opponent's robber placement"
