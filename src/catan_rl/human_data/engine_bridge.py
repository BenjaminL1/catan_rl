"""Engine bridge: serialize a live post-setup game and rebuild it from primitives.

Step6 §2.3 / §3.2, PRE-CORPUS lane. Two responsibilities:

1. **Serialize** a live post-setup ``catanGame`` (as wrapped by :class:`CatanEnv`)
   into a :class:`BridgeState` — a ``GameRecord``-shaped dict of pure primitives
   (hex layout, robber, snake-ordered placements, hands, to-move seat) plus an
   **out-of-band port list** (ports are engine state only, never in the corpus obs).

2. **Rebuild** a playable :class:`CatanEnv` positioned at the identical post-setup,
   agent-to-roll state via the :mod:`catan_rl.engine.board` injection API
   (``inject_hex_layout`` + ``updatePorts(port_assignment=...)``) and a legality-
   asserting placement replay. Grants for each player's second (granting)
   settlement recirculate through the spec-009 finite bank via ``bank_draw``.

The rebuild is the production code that will later re-hydrate **human**
``GameRecord``s into env states; the V8-self-play round-trip (a REAL env-produced
state serialized, rebuilt, and checked for ``v̂``/state-hash parity) is the
correctness harness that pins it (see ``tests/unit/human_data/test_engine_bridge``).

CPU only. Imports no ``gui/``. The engine ``build_*`` methods SILENTLY no-op on an
illegal placement, so every rebuild runs post-condition assertions (piece counts,
occupancy, the distance rule, road incidence) plus ``assert_conservation``,
``run_all_invariants``, and hand-tracker parity before returning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from catan_rl.agents.heuristic import heuristicAIPlayer
from catan_rl.agents.random_ai import RandomAIPlayer
from catan_rl.engine.game import catanGame
from catan_rl.engine.player import player as PlainPlayer
from catan_rl.engine.tracker import ResourceTracker
from catan_rl.env.catan_env import CatanEnv
from catan_rl.env.hand_tracker import BroadcastHandTracker
from catan_rl.eval.rules_invariants import RulesInvariantViolation, run_all_invariants
from catan_rl.policy.obs_encoder import ObsEncoder

if TYPE_CHECKING:
    import numpy as np

#: Engine hand keys (engine ``RESOURCES`` order, NOT the RL Charlesworth order —
#: CLAUDE.md rule 6). Hands are serialized keyed by these strings.
ENGINE_RESOURCES: tuple[str, ...] = ("ORE", "BRICK", "WHEAT", "WOOD", "SHEEP")


class BridgeError(RuntimeError):
    """Raised when serialize/rebuild hits an unreconstructable state."""


@dataclass(frozen=True, slots=True)
class SeatPlacement:
    """One seat's snake-draft opening as engine integer IDs, in placement order.

    ``settlements[0]`` = first-placed; ``settlements[1]`` = second-placed (the
    resource-granting settlement — the bridge grants from it). ``roads[i]`` is
    the setup road placed immediately after ``settlements[i]``.
    """

    settlements: tuple[int, int]
    roads: tuple[int, int]


@dataclass(frozen=True, slots=True)
class BridgeState:
    """Primitive, ``GameRecord``-shaped snapshot of a post-setup 1v1 game.

    Everything is a JSON-safe primitive so a round-trip proves the injection +
    replay path (never object aliasing). Seats are queue positions: seat 0 acts
    first in the snake draft; ``agent_seat`` marks which seat is the agent (the
    obs POV). At post-setup the agent is always the to-move (rolling) seat.
    """

    hexes: tuple[dict[str, Any], ...]
    robber_hex: int
    port_assignment: dict[str, list[int]]
    placements: dict[int, SeatPlacement]
    hands: dict[int, dict[str, int]]
    agent_seat: int
    opponent_type: str
    opp_kind: int
    opp_policy_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "hexes": [dict(h) for h in self.hexes],
            "robber_hex": int(self.robber_hex),
            "port_assignment": {k: list(v) for k, v in self.port_assignment.items()},
            "placements": {
                str(seat): {"settlements": list(sp.settlements), "roads": list(sp.roads)}
                for seat, sp in self.placements.items()
            },
            "hands": {str(seat): dict(hand) for seat, hand in self.hands.items()},
            "agent_seat": int(self.agent_seat),
            "opponent_type": self.opponent_type,
            "opp_kind": int(self.opp_kind),
            "opp_policy_id": int(self.opp_policy_id),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BridgeState:
        placements = {
            int(seat): SeatPlacement(
                settlements=tuple(sp["settlements"]),
                roads=tuple(sp["roads"]),
            )
            for seat, sp in payload["placements"].items()
        }
        hands = {int(seat): dict(hand) for seat, hand in payload["hands"].items()}
        return cls(
            hexes=tuple(dict(h) for h in payload["hexes"]),
            robber_hex=int(payload["robber_hex"]),
            port_assignment={k: list(v) for k, v in payload["port_assignment"].items()},
            placements=placements,
            hands=hands,
            agent_seat=int(payload["agent_seat"]),
            opponent_type=str(payload["opponent_type"]),
            opp_kind=int(payload["opp_kind"]),
            opp_policy_id=int(payload["opp_policy_id"]),
        )


# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------
def serialize_post_setup(env: CatanEnv) -> BridgeState:
    """Serialize the env's live post-setup state to a :class:`BridgeState`.

    Precondition: ``env`` is at post-setup with the agent to roll
    (``not initial_placement_phase and roll_pending``). Raises otherwise so a
    mid-setup or mid-turn state can never masquerade as a clean opening.
    """
    game = env.game
    if game is None or env.agent_player is None or env.opponent_player is None:
        raise BridgeError("serialize_post_setup: env has no live game/players")
    if env.initial_placement_phase or not env.roll_pending:
        raise BridgeError(
            "serialize_post_setup expects a post-setup, roll-pending state "
            f"(initial_placement_phase={env.initial_placement_phase}, "
            f"roll_pending={env.roll_pending})"
        )
    board = game.board

    hexes = tuple(
        {
            "hex_id": i,
            "resource": str(board.hexTileDict[i].resource_type),
            "number": (
                int(board.hexTileDict[i].number_token)
                if board.hexTileDict[i].number_token is not None
                else None
            ),
        }
        for i in range(19)
    )
    robber_hexes = [i for i in range(19) if board.hexTileDict[i].has_robber]
    if len(robber_hexes) != 1:
        raise BridgeError(f"expected exactly one robber hex, found {robber_hexes}")

    pixel_to_vidx = {px: idx for idx, px in board.vertex_index_to_pixel_dict.items()}
    players = list(game.playerQueue.queue)
    if len(players) != 2:
        raise BridgeError(f"expected a 2-player queue, got {len(players)}")

    placements: dict[int, SeatPlacement] = {}
    hands: dict[int, dict[str, int]] = {}
    for seat, p in enumerate(players):
        settle_idx = [int(pixel_to_vidx[v]) for v in p.buildGraph["SETTLEMENTS"]]
        road_idx = [
            int(env._edge_to_idx[env._edge_key(v1, v2)]) for v1, v2 in p.buildGraph["ROADS"]
        ]
        if len(settle_idx) != 2 or len(road_idx) != 2:
            raise BridgeError(
                f"seat {seat} is not a clean 2-settlement/2-road opening: "
                f"settlements={settle_idx}, roads={road_idx}"
            )
        placements[seat] = SeatPlacement(
            settlements=(settle_idx[0], settle_idx[1]),
            roads=(road_idx[0], road_idx[1]),
        )
        hands[seat] = {r: int(p.resources.get(r, 0)) for r in ENGINE_RESOURCES}

    agent_seat = 0 if players[0] is env.agent_player else 1

    return BridgeState(
        hexes=hexes,
        robber_hex=int(robber_hexes[0]),
        port_assignment=board.get_port_assignment(),
        placements=placements,
        hands=hands,
        agent_seat=agent_seat,
        opponent_type=env.opponent_type,
        opp_kind=int(env._opp_kind),
        opp_policy_id=int(env._opp_policy_id),
    )


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------
def rebuild_env(state: BridgeState) -> CatanEnv:
    """Rebuild a :class:`CatanEnv` at the post-setup state described by ``state``.

    Uses the board injection API + placement replay, grants from each seat's
    second settlement via the spec-009 finite bank, then asserts every
    post-condition (the engine ``build_*`` silently no-ops, so legality is only
    known after the fact). Returns the env positioned at the agent's roll.
    """
    env = CatanEnv(opponent_type=state.opponent_type)

    game = catanGame(render_mode=None)
    board = game.board
    resources = [str(h["resource"]) for h in state.hexes]
    numbers = [h["number"] for h in state.hexes]
    board.inject_hex_layout(resources, numbers, state.robber_hex)
    board.updatePorts(port_assignment=state.port_assignment)

    agent = PlainPlayer("Agent", "black")
    agent.game = game
    agent.isAI = False
    if state.opponent_type in ("heuristic", "snapshot"):
        opp: Any = heuristicAIPlayer("Opponent", "darkslateblue")
        opp.updateAI()
    else:
        opp = RandomAIPlayer("Opponent", "darkslateblue")
    opp.game = game
    opp.isAI = True

    # Seat 0 acts first in the snake draft; agent_seat says which seat is agent.
    import queue as _queue

    seat0, seat1 = (agent, opp) if state.agent_seat == 0 else (opp, agent)
    game.playerQueue = _queue.Queue(2)
    game.playerQueue.put(seat0)
    game.playerQueue.put(seat1)
    game.resource_tracker = ResourceTracker([seat0.name, seat1.name])

    env.game = game
    env.agent_player = agent
    env.opponent_player = opp
    env._obs_encoder = ObsEncoder(board)
    env._hand_tracker = BroadcastHandTracker([agent.name, opp.name])
    env._hand_tracker.subscribe(game.broadcast)
    env._build_index_maps(board)
    env._agent_seat = state.agent_seat
    env._opp_kind = state.opp_kind
    env._opp_policy_id = state.opp_policy_id

    # --- placement replay (snake order: seat0.1 → seat1.1 → seat1.2(+grant) →
    #     seat0.2(+grant)), mirroring CatanEnv setup exactly. ------------------
    p0, p1 = seat0, seat1
    sp0, sp1 = state.placements[0], state.placements[1]
    _place_settlement(env, p0, sp0.settlements[0])
    _place_road(env, p0, sp0.roads[0])
    _place_settlement(env, p1, sp1.settlements[0])
    _place_road(env, p1, sp1.roads[0])
    _place_settlement(env, p1, sp1.settlements[1])
    _place_road(env, p1, sp1.roads[1])
    _grant_second_settlement(game, p1)
    _place_settlement(env, p0, sp0.settlements[1])
    _place_road(env, p0, sp0.roads[1])
    _grant_second_settlement(game, p0)

    # --- env state → post-setup, agent to roll --------------------------------
    env.initial_placement_phase = False
    game.gameSetup = False
    env._setup_step = 4
    env.roll_pending = True
    env.discard_pending = False
    env.robber_placement_pending = False
    env.road_building_roads_left = 0
    env.last_dice_roll = 0
    env._turn_count = 1
    game.broadcast.setup_complete()

    _assert_post_conditions(env, state)
    board.assert_conservation(list(game.playerQueue.queue))
    violations = run_all_invariants(game, truncated=True)
    if violations:
        raise RulesInvariantViolation("; ".join(violations))
    _assert_hand_tracker_parity(env)

    return env


def _place_settlement(env: CatanEnv, p: Any, vertex_idx: int) -> None:
    assert env.game is not None
    board = env.game.board
    px = board.vertex_index_to_pixel_dict[vertex_idx]
    before = len(p.buildGraph["SETTLEMENTS"])
    p.build_settlement(px, board, is_free=True)
    if len(p.buildGraph["SETTLEMENTS"]) != before + 1:
        raise BridgeError(
            f"settlement replay no-op'd for {p.name} at vertex {vertex_idx} "
            "(engine build_settlement rejected it silently)"
        )


def _place_road(env: CatanEnv, p: Any, edge_idx: int) -> None:
    assert env.game is not None
    board = env.game.board
    v1, v2 = env._idx_to_edge[edge_idx]
    before = len(p.buildGraph["ROADS"])
    p.build_road(v1, v2, board, is_free=True)
    if len(p.buildGraph["ROADS"]) != before + 1:
        raise BridgeError(
            f"road replay no-op'd for {p.name} at edge {edge_idx} "
            "(engine build_road rejected it silently)"
        )


def _grant_second_settlement(game: catanGame, p: Any) -> None:
    """Grant starting resources from ``p``'s second (last-placed) settlement,
    drawing from the spec-009 finite bank and broadcasting a SETUP resource
    change so the hand tracker mirrors the live env exactly."""
    if not p.buildGraph["SETTLEMENTS"]:
        return
    last_settle = p.buildGraph["SETTLEMENTS"][-1]
    for adj_hex in game.board.boardGraph[last_settle].adjacent_hex_indices:
        res_type = game.board.hexTileDict[adj_hex].resource_type
        if res_type != "DESERT":
            p.resources[res_type] += 1
            game.board.bank_draw({res_type: 1})
            game.broadcast.resource_change(p.name, {res_type: 1}, "SETUP")


# ---------------------------------------------------------------------------
# Post-condition assertions
# ---------------------------------------------------------------------------
def _assert_post_conditions(env: CatanEnv, state: BridgeState) -> None:
    assert env.game is not None
    board = env.game.board
    players = list(env.game.playerQueue.queue)
    pixel_to_vidx = {px: idx for idx, px in board.vertex_index_to_pixel_dict.items()}

    all_settlement_vidx: list[int] = []
    for seat, p in enumerate(players):
        # Piece counts.
        if len(p.buildGraph["SETTLEMENTS"]) != 2 or len(p.buildGraph["ROADS"]) != 2:
            raise BridgeError(
                f"seat {seat} ({p.name}) piece count wrong after replay: "
                f"{len(p.buildGraph['SETTLEMENTS'])} settlements / "
                f"{len(p.buildGraph['ROADS'])} roads"
            )
        settle_pixels = list(p.buildGraph["SETTLEMENTS"])
        for v_px in settle_pixels:
            # Occupancy: this vertex is owned by this player.
            if board.boardGraph[v_px].owner is not p:
                raise BridgeError(
                    f"vertex {pixel_to_vidx[v_px]} not owned by {p.name} after replay"
                )
            all_settlement_vidx.append(int(pixel_to_vidx[v_px]))
            # Distance rule: no adjacent vertex carries a building.
            for nb_px in board.boardGraph[v_px].neighbors:
                if board.boardGraph[nb_px].owner is not None:
                    raise BridgeError(
                        f"distance rule violated: vertex {pixel_to_vidx[v_px]} "
                        f"adjacent to occupied vertex {pixel_to_vidx[nb_px]}"
                    )
        # Road incidence: each road edge shares a vertex with a settlement of p.
        settle_pixel_set = set(settle_pixels)
        for v1, v2 in p.buildGraph["ROADS"]:
            if v1 not in settle_pixel_set and v2 not in settle_pixel_set:
                raise BridgeError(
                    f"road incidence violated for {p.name}: edge "
                    f"({pixel_to_vidx.get(v1)}, {pixel_to_vidx.get(v2)}) touches no "
                    "own settlement"
                )

    # All four settlements distinct (no shared/double-snapped vertex).
    if len(set(all_settlement_vidx)) != 4:
        raise BridgeError(f"settlement vertices not all distinct: {all_settlement_vidx}")

    # Hands match the serialized ground truth.
    for seat, p in enumerate(players):
        expected = state.hands[seat]
        for r in ENGINE_RESOURCES:
            if int(p.resources.get(r, 0)) != int(expected.get(r, 0)):
                raise BridgeError(
                    f"seat {seat} ({p.name}) hand mismatch after replay for {r}: "
                    f"got {p.resources.get(r, 0)}, expected {expected.get(r, 0)}"
                )


def _assert_hand_tracker_parity(env: CatanEnv) -> None:
    """The BroadcastHandTracker (perfect 1v1 opponent tracking) must agree with
    every player's true hand after the SETUP grants replayed through it."""
    assert env.game is not None and env._hand_tracker is not None
    for p in list(env.game.playerQueue.queue):
        tracked = env._hand_tracker.get_hand(p.name)
        for r in ENGINE_RESOURCES:
            if int(tracked.get(r, 0)) != int(p.resources.get(r, 0)):
                raise BridgeError(
                    f"hand-tracker parity broken for {p.name} on {r}: "
                    f"tracker={tracked.get(r, 0)} vs hand={p.resources.get(r, 0)}"
                )


# ---------------------------------------------------------------------------
# Canonical engine state hash (round-trip equality check)
# ---------------------------------------------------------------------------
def engine_state_hash(game: catanGame) -> str:
    """SHA-256 over a canonical, primitive view of the engine state.

    Covers board layout (resource/number/robber per hex), the port assignment,
    every player's placements + hand + VP, and the finite bank. Two games with
    an equal hash are indistinguishable to the RL stack. Placements are keyed by
    engine vertex/edge index (geometry is board-invariant, so indices are stable
    across a re-bridge)."""
    board = game.board
    pixel_to_vidx = {px: idx for idx, px in board.vertex_index_to_pixel_dict.items()}

    def _edge_key(v1: Any, v2: Any) -> tuple[str, str]:
        s1, s2 = str(v1), str(v2)
        return (s1, s2) if s1 < s2 else (s2, s1)

    edge_key_to_idx: dict[tuple[str, str], int] = {}
    seen: set[tuple[str, str]] = set()
    idx = 0
    for v_px, v_obj in board.boardGraph.items():
        for nb_px in v_obj.neighbors:
            key = _edge_key(v_px, nb_px)
            if key not in seen:
                seen.add(key)
                edge_key_to_idx[key] = idx
                idx += 1

    hexes = [
        {
            "id": i,
            "resource": str(board.hexTileDict[i].resource_type),
            "number": (
                int(board.hexTileDict[i].number_token)
                if board.hexTileDict[i].number_token is not None
                else None
            ),
            "robber": bool(board.hexTileDict[i].has_robber),
        }
        for i in range(19)
    ]

    players_canon = []
    for seat, p in enumerate(list(game.playerQueue.queue)):
        settlements = sorted(int(pixel_to_vidx[v]) for v in p.buildGraph["SETTLEMENTS"])
        cities = sorted(int(pixel_to_vidx[v]) for v in p.buildGraph["CITIES"])
        roads = sorted(int(edge_key_to_idx[_edge_key(v1, v2)]) for v1, v2 in p.buildGraph["ROADS"])
        players_canon.append(
            {
                "seat": seat,
                "settlements": settlements,
                "cities": cities,
                "roads": roads,
                "hand": {r: int(p.resources.get(r, 0)) for r in ENGINE_RESOURCES},
                "vp": int(p.victoryPoints),
            }
        )

    canon = {
        "hexes": hexes,
        "ports": board.get_port_assignment(),
        "players": players_canon,
        "bank": {r: int(board.resourceBank[r]) for r in sorted(board.resourceBank)},
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# State generation (REAL env-produced post-setup states)
# ---------------------------------------------------------------------------
def drive_env_to_post_setup(env: CatanEnv, rng: np.random.Generator, *, seed: int) -> None:
    """Reset ``env`` and drive the agent seat through setup with random LEGAL
    actions until it reaches the post-setup, roll-pending state.

    The opponent's setup placements are driven by the env internally. Uses the
    env's own action masks so every action is legal — the states are genuine,
    env-produced openings (v8-self-play-shaped; policy identity is irrelevant to
    the round-trip invariant)."""
    env.reset(seed=seed, options={"agent_seat": 0})
    for _ in range(8):  # 4 agent setup actions; loop-bounded guard
        if not env.initial_placement_phase:
            break
        masks = env.get_action_masks()
        action = _sample_setup_action(masks, rng)
        env.step(action)
    if env.initial_placement_phase or not env.roll_pending:
        raise BridgeError("drive_env_to_post_setup did not reach a post-setup roll-pending state")


def _sample_setup_action(masks: dict[str, np.ndarray], rng: np.random.Generator) -> list[int]:
    import numpy as np

    type_choices = np.flatnonzero(masks["type"])
    a_type = int(rng.choice(type_choices))
    corner = 0
    edge = 0
    if masks["corner_settlement"].any():
        corner = int(rng.choice(np.flatnonzero(masks["corner_settlement"])))
    if masks["edge"].any():
        edge = int(rng.choice(np.flatnonzero(masks["edge"])))
    return [a_type, corner, edge, 0, 0, 0]
