"""Tests for the engine board injection API + V8-state re-bridge (step6 §2.3/§3.2).

Covers:

* **Bridge null test (i)** — round-trip REAL env-produced post-setup states and
  assert ``v̂`` parity (|Δ| < 1e-5) between the live env obs and the re-bridged
  obs, plus engine state-hash equality. The batch is required to include at
  least one state whose opening settlement sits on a port-slot vertex (catches a
  portList weld). CI runs it at N=8; the N=50 variant is marked ``slow``.
* **Injection determinism** — the same explicit port assignment / hex layout
  rebuilds an identical board twice.
* **Default board generation byte-unchanged** — with the injectable path unused,
  ``updatePorts()`` still draws from the global PRNG and is deterministic under a
  fixed seed (the broad engine + conformance suites pin the rest).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from catan_rl.engine.board import catanBoard
from catan_rl.env.catan_env import CatanEnv
from catan_rl.human_data.engine_bridge import (
    BridgeError,
    BridgeState,
    drive_env_to_post_setup,
    engine_state_hash,
    rebuild_env,
    serialize_post_setup,
)
from catan_rl.policy.network import CatanPolicy
from catan_rl.search.value import value_from_obs

_DEVICE = torch.device("cpu")


def _board_full_state(board: catanBoard) -> dict:
    """A primitive full-board snapshot (hexes + robber + ports) for equality."""
    return {
        "hexes": [
            (
                i,
                board.hexTileDict[i].resource_type,
                board.hexTileDict[i].number_token,
                bool(board.hexTileDict[i].has_robber),
            )
            for i in range(19)
        ],
        "ports": board.get_port_assignment(),
    }


@pytest.fixture(scope="module")
def policy() -> CatanPolicy:
    torch.manual_seed(0)
    return CatanPolicy().eval()


# ---------------------------------------------------------------------------
# Injection API — determinism + validation
# ---------------------------------------------------------------------------
def test_default_board_generation_deterministic_under_seed() -> None:
    """The default (non-injected) build path draws from the global PRNG exactly
    as before — same seed ⇒ byte-identical board."""
    np.random.seed(20260705)
    random.seed(20260705)
    b1 = catanBoard()
    np.random.seed(20260705)
    random.seed(20260705)
    b2 = catanBoard()
    assert _board_full_state(b1) == _board_full_state(b2)


def test_injection_same_ports_rebuild_identical_board_twice() -> None:
    """Same explicit hex layout + port assignment ⇒ two identical boards."""
    np.random.seed(7)
    src = catanBoard()
    resources = [src.hexTileDict[i].resource_type for i in range(19)]
    numbers = [src.hexTileDict[i].number_token for i in range(19)]
    robber = next(i for i in range(19) if src.hexTileDict[i].has_robber)
    port_assignment = src.get_port_assignment()

    boards = []
    for _ in range(2):
        # Build a fresh board from a DIFFERENT global-PRNG state, then inject:
        # the result must depend only on the injected layout, not the PRNG.
        np.random.seed(999)
        b = catanBoard()
        b.inject_hex_layout(resources, numbers, robber)
        b.updatePorts(port_assignment=port_assignment)
        boards.append(_board_full_state(b))
    assert boards[0] == boards[1]
    # ... and equal to the source it was serialized from.
    assert boards[0] == _board_full_state(src)


def test_get_port_assignment_round_trips() -> None:
    np.random.seed(3)
    b = catanBoard()
    pa = b.get_port_assignment()
    # 5 specific 2:1 ports (1 pair each) + 1 generic 3:1 (4 pairs) = 6 types,
    # 18 port-slot vertices total.
    assert sum(len(v) for v in pa.values()) == 18
    b.updatePorts(port_assignment=pa)
    assert b.get_port_assignment() == pa


def test_inject_hex_layout_rejects_desert_number_mismatch() -> None:
    np.random.seed(1)
    b = catanBoard()
    good_res = [b.hexTileDict[i].resource_type for i in range(19)]
    good_num = [b.hexTileDict[i].number_token for i in range(19)]
    desert_idx = good_res.index("DESERT")

    # Desert must carry number None.
    bad_num = list(good_num)
    bad_num[desert_idx] = 8
    with pytest.raises(ValueError, match="desert"):
        b.inject_hex_layout(good_res, bad_num, desert_idx)

    # A non-desert hex must carry a number.
    non_desert = next(i for i in range(19) if good_res[i] != "DESERT")
    bad_num2 = list(good_num)
    bad_num2[non_desert] = None
    with pytest.raises(ValueError, match="must carry a number"):
        b.inject_hex_layout(good_res, bad_num2, desert_idx)

    with pytest.raises(ValueError, match="19 resources"):
        b.inject_hex_layout(good_res[:-1], good_num, desert_idx)


def test_explicit_rng_ports_do_not_touch_global_prng() -> None:
    """The ``rng=`` injectable path draws from the explicit generator, leaving
    the global PRNG untouched (reproducible without global side effects)."""
    np.random.seed(55)
    b = catanBoard()  # consumes the global PRNG as usual
    global_after = np.random.get_state()
    rng = np.random.default_rng(123)
    b.updatePorts(rng=rng)
    # Global PRNG state is unchanged by the rng= path.
    assert np.array_equal(global_after[1], np.random.get_state()[1])


# ---------------------------------------------------------------------------
# Bridge null test (i)
# ---------------------------------------------------------------------------
def _run_null_test(policy: CatanPolicy, *, n: int, base_seed: int) -> int:
    """Round-trip ``n`` real env-produced post-setup states; assert v̂ parity +
    state-hash equality per state. Returns the count of states with an opening
    settlement on a port-slot vertex (the caller asserts ≥ 1)."""
    rng = np.random.default_rng(base_seed)
    env = CatanEnv(opponent_type="random")
    port_settlement_states = 0

    for i in range(n):
        drive_env_to_post_setup(env, rng, seed=base_seed + i)
        live_obs = env._get_obs()
        live_hash = engine_state_hash(env.game)

        # Serialize → primitive dict → back (proves the round-trip is via
        # primitives, never object aliasing) → rebuild.
        state = BridgeState.from_dict(serialize_post_setup(env).to_dict())
        rebuilt = rebuild_env(state)
        reb_obs = rebuilt._get_obs()
        reb_hash = engine_state_hash(rebuilt.game)

        v_live = value_from_obs(policy, live_obs, device=_DEVICE)
        v_reb = value_from_obs(policy, reb_obs, device=_DEVICE)
        assert abs(v_live - v_reb) < 1e-5, f"state {i}: v̂ parity broke Δ={abs(v_live - v_reb)}"
        assert live_hash == reb_hash, f"state {i}: engine state hash mismatch"

        # Port-slot settlement detection (portList weld guard).
        board = env.game.board
        port_vertices: set[int] = set()
        for vs in board.get_port_assignment().values():
            port_vertices.update(vs)
        pix2v = {px: idx for idx, px in board.vertex_index_to_pixel_dict.items()}
        settle_vertices = {
            int(pix2v[vpx])
            for p in list(env.game.playerQueue.queue)
            for vpx in p.buildGraph["SETTLEMENTS"]
        }
        if settle_vertices & port_vertices:
            port_settlement_states += 1

    return port_settlement_states


def test_bridge_null_test_ci(policy: CatanPolicy) -> None:
    port_hits = _run_null_test(policy, n=8, base_seed=1000)
    assert port_hits >= 1, "null-test batch must include a port-slot opening settlement"


@pytest.mark.slow
def test_bridge_null_test_n50(policy: CatanPolicy) -> None:
    port_hits = _run_null_test(policy, n=50, base_seed=5000)
    assert port_hits >= 1


# ---------------------------------------------------------------------------
# Serialize / rebuild guards
# ---------------------------------------------------------------------------
def test_serialize_rejects_mid_setup() -> None:
    env = CatanEnv(opponent_type="random")
    env.reset(seed=42, options={"agent_seat": 0})
    # Still in the placement phase — must refuse.
    with pytest.raises(BridgeError, match="post-setup"):
        serialize_post_setup(env)


def test_bridge_state_dict_round_trips() -> None:
    rng = np.random.default_rng(11)
    env = CatanEnv(opponent_type="random")
    drive_env_to_post_setup(env, rng, seed=2024)
    state = serialize_post_setup(env)
    restored = BridgeState.from_dict(state.to_dict())
    assert restored == state


def test_rebuild_hand_tracker_parity_and_conservation() -> None:
    """A rebuilt env satisfies the finite-bank conservation invariant and the
    perfect-hand-tracker parity (both asserted inside ``rebuild_env``; this test
    pins that the happy path actually exercises them)."""
    rng = np.random.default_rng(77)
    env = CatanEnv(opponent_type="heuristic")
    drive_env_to_post_setup(env, rng, seed=3131)
    state = serialize_post_setup(env)
    rebuilt = rebuild_env(state)  # raises if any invariant fails
    # Conservation: bank + hands == 19 per engine resource.
    rebuilt.game.board.assert_conservation(list(rebuilt.game.playerQueue.queue))
    # Hands are non-trivial (both seats received a setup grant).
    total_cards = sum(sum(p.resources.values()) for p in list(rebuilt.game.playerQueue.queue))
    assert total_cards > 0
