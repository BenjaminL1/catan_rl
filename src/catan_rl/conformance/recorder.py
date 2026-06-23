"""Record a full reference game to a Torevan conformance replay-log.

The replay-log JSON has the shape (all resource keys Charlesworth-ordered
``WOOD, BRICK, WHEAT, ORE, SHEEP``; all geometry as integer IDs matching
``packages/engine/src/topology.fixture.json``)::

    {
      "schema_version": 1,
      "seed": 7,                          # numpy seed used for this game
      "board": {                          # the LOADED board (never regenerated)
        "robber_hex": 4,
        "tiles": [                        # 19 entries, hex_idx order 0..18
          {"hex": 0, "resource": "WOOD", "number_token": 5,
           "has_robber": false},
          ...
        ],
        "ports": [                        # 9 entries, slot order 0..8
          {"slot": 0, "vertices": [v1, v2], "ratio": 2,
           "resource": "BRICK"},
          {"slot": 5, "vertices": [v1, v2], "ratio": 3, "resource": null},
          ...
        ]
      },
      "players": [                        # seat order: [seat0, seat1]
        {"seat": 0, "name": "Player 1"},
        {"seat": 1, "name": "Player 2"}
      ],
      "steps": [                          # ordered list of atomic actions
        {
          "actor": 0,                     # seat that acts
          "action": {"kind": "BuildSettlement", "args": {"vertex": 12}},
          "outcome": {},                  # RNG-derived results, see below
          "state_after": { ... }          # canonical reference snapshot
        },
        ...
      ]
    }

The ``action.kind`` discriminators match the TS ``GameAction`` union:
``RollDice, BuildRoad, BuildSettlement, BuildCity, MoveRobber, Discard,
EndTurn, BankTrade, BuyDevCard, PlayKnight, PlayYearOfPlenty,
PlayMonopoly, PlayRoadBuilder``. Setup placements are encoded as plain
``BuildSettlement`` / ``BuildRoad`` steps in snake order.

``outcome`` carries the RNG-derived results so the TS engine can replay
the exact line of play through its production ``applyAction`` path
(via the typed ``ReplayOutcome`` seam):

* ``{"dice_roll": int}``      — for a ``RollDice`` step.
* ``{"steal": str|null}``     — the stolen resource for a robber theft
                                (``MoveRobber`` / ``PlayKnight``); null
                                if no victim / empty hand.
* ``{"dev_card": str}``       — the drawn dev-card type for ``BuyDevCard``
                                (Torevan names: KNIGHT, VP, ROADBUILDER,
                                YEAROFPLENTY, MONOPOLY).

``state_after`` is the canonical reference snapshot used for assertion,
normalised on both sides (Charlesworth resource order; settlement/city
disjointness; dev-card naming/order; longest-road / largest-army
holders by seat; robber hex; per-player VP / resources / dev hand /
knights). See :func:`_normalise_snapshot`.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

# Headless: never open a window even if pygame is importable.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from catan_rl.engine.game import catanGame

_LOG = logging.getLogger("catan_rl.conformance")

#: On-disk schema version for the conformance replay-log.
CONFORMANCE_SCHEMA_VERSION = 1

#: Canonical (Charlesworth) resource order — matches @torevan/engine's
#: ``RESOURCES`` and the obs schema. Every resource map in the log is
#: emitted in this key order.
RESOURCES_CW: tuple[str, ...] = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")

#: Torevan dev-card order — matches @torevan/engine's ``DEV_CARD_TYPES``.
DEV_CARD_TYPES_TS: tuple[str, ...] = (
    "KNIGHT",
    "VP",
    "ROADBUILDER",
    "YEAROFPLENTY",
    "MONOPOLY",
)

#: Map the snapshot's dev-card keys (replay.schema STATE_DEV_CARD_ORDER)
#: to the Torevan engine's dev-card names.
_SNAPSHOT_DEV_TO_TS: dict[str, str] = {
    "KNIGHT": "KNIGHT",
    "VP": "VP",
    "ROAD_BUILDER": "ROADBUILDER",
    "YEAR_OF_PLENTY": "YEAROFPLENTY",
    "MONOPOLY": "MONOPOLY",
}

#: The fixed port (hex, corner1, corner2) table — copied verbatim from
#: ``catanBoard.updatePorts`` (and ``scripts/export_topology.py``) so the
#: recorder resolves the SAME 9 port slots / vertex pairs the TS
#: ``PORT_SLOTS`` fixture uses. Index == slot id.
PORT_HEX_CORNERS: list[tuple[int, int, int]] = [
    (7, 2, 3),
    (8, 1, 2),
    (10, 1, 2),
    (11, 0, 1),
    (12, 5, 0),
    (14, 0, 5),
    (15, 4, 5),
    (16, 3, 4),
    (18, 3, 4),
]


# ---------------------------------------------------------------------------
# Index maps (pixel-coord engine keys -> integer IDs).
# ---------------------------------------------------------------------------


def _edge_key(v1: object, v2: object) -> tuple[str, str]:
    """Lex-sorted pixel-string edge key — replica of CatanEnv._edge_key."""
    s1, s2 = str(v1), str(v2)
    return (s1, s2) if s1 < s2 else (s2, s1)


def _build_maps(board: Any) -> tuple[dict[Any, int], dict[tuple[str, str], int]]:
    """Build ``(vertex_pixel_to_idx, edge_key_to_idx)`` in the canonical
    ordering the TS ``topology.fixture.json`` was exported with (see
    ``scripts/export_topology.py``). These bridge the engine's
    pixel-coord keys to the integer IDs the replay-log uses."""
    pixel_to_idx: dict[Any, int] = {px: idx for idx, px in board.vertex_index_to_pixel_dict.items()}
    seen: set[tuple[str, str]] = set()
    edge_key_to_idx: dict[tuple[str, str], int] = {}
    for v_px, vobj in board.boardGraph.items():
        for nb_px in vobj.neighbors:
            key = _edge_key(v_px, nb_px)
            if key in seen:
                continue
            seen.add(key)
            edge_key_to_idx[key] = len(edge_key_to_idx)
    return pixel_to_idx, edge_key_to_idx


def _resolve_port_slots(board: Any, pixel_to_idx: dict[Any, int]) -> list[dict[str, Any]]:
    """Resolve the 9 port assignments to ``{slot, vertices, ratio,
    resource}`` records — geometry from the fixed ``PORT_HEX_CORNERS``
    table, type from each vertex's engine ``.port`` label. Mirrors
    ``board.updatePorts`` so the slot id matches TS ``PORT_SLOTS``."""

    # hex,corner -> vertex_idx, via exact-pixel match (export_topology.py).
    def corner_vertex(hex_idx: int, corner: int) -> int:
        tile = board.hexTileDict[hex_idx]
        corner_pt = tile.get_corners(board.flat)[corner]
        for v_px, v_idx in pixel_to_idx.items():
            dx = corner_pt.x - v_px.x
            dy = corner_pt.y - v_px.y
            if round((dx * dx + dy * dy) ** 0.5) == 0:
                return int(v_idx)
        raise AssertionError(f"hex {hex_idx} corner {corner} pixel unmatched")

    ports: list[dict[str, Any]] = []
    for slot, (h_idx, c1, c2) in enumerate(PORT_HEX_CORNERS):
        v1 = corner_vertex(h_idx, c1)
        v2 = corner_vertex(h_idx, c2)
        px1 = board.vertex_index_to_pixel_dict[v1]
        label = board.boardGraph[px1].port  # e.g. "2:1 BRICK" or "3:1 PORT"
        if isinstance(label, str) and label.startswith("2:1"):
            ratio = 2
            resource: str | None = label.split(" ", 1)[1] if " " in label else None
        else:
            ratio = 3
            resource = None
        ports.append(
            {
                "slot": slot,
                "vertices": [v1, v2],
                "ratio": ratio,
                "resource": resource,
            }
        )
    return ports


def _board_payload(board: Any, pixel_to_idx: dict[Any, int]) -> dict[str, Any]:
    """The LOADED board: 19 tiles (resource / number / robber) + 9 port
    assignments, in integer-ID form. The TS conformance test LOADS this
    rather than regenerating, so board-gen RNG parity is irrelevant."""
    tiles: list[dict[str, Any]] = []
    robber_hex = -1
    for hex_idx in range(19):
        tile = board.hexTileDict[hex_idx]
        has_robber = bool(getattr(tile, "has_robber", False))
        if has_robber:
            robber_hex = hex_idx
        tiles.append(
            {
                "hex": hex_idx,
                "resource": str(tile.resource_type),
                "number_token": (int(tile.number_token) if tile.number_token is not None else None),
                "has_robber": has_robber,
            }
        )
    return {
        "robber_hex": robber_hex,
        "tiles": tiles,
        "ports": _resolve_port_slots(board, pixel_to_idx),
    }


# ---------------------------------------------------------------------------
# Snapshot normalisation.
# ---------------------------------------------------------------------------


def _normalise_resources(raw: dict[str, int]) -> dict[str, int]:
    """Re-key a resource dict into canonical Charlesworth order."""
    return {r: int(raw.get(r, 0)) for r in RESOURCES_CW}


def _normalise_player(snap: dict[str, Any]) -> dict[str, Any]:
    """Normalise one player's snapshot to the canonical assertion shape:
    Charlesworth resources, Torevan-named dev hand (excluding VP, which
    lives in ``dev_cards_played``), and the cumulative ``knights_played``
    / ``vp_cards`` the TS engine tracks on ``PlayerState``."""
    hand_raw = snap.get("dev_cards_hand", {})
    played_raw = snap.get("dev_cards_played", {})
    # Dev hand: KNIGHT/ROADBUILDER/YEAROFPLENTY/MONOPOLY (VP is excluded
    # from the hand — it's counted in vp_cards below, matching the engine).
    dev_hand: dict[str, int] = {}
    for ts_name in DEV_CARD_TYPES_TS:
        if ts_name == "VP":
            continue
        # Find the snapshot key that maps to this TS name.
        snap_key = next(k for k, v in _SNAPSHOT_DEV_TO_TS.items() if v == ts_name)
        dev_hand[ts_name] = int(hand_raw.get(snap_key, 0))
    return {
        "vp": int(snap.get("vp", 0)),
        "resources": _normalise_resources(snap.get("resources", {})),
        "dev_hand": dev_hand,
        "vp_cards": int(played_raw.get("VP", 0)),
        "knights_played": int(played_raw.get("KNIGHT", 0)),
    }


def _normalise_snapshot(raw: dict[str, Any], seat_to_actor: dict[str, str]) -> dict[str, Any]:
    """Convert a ``game.snapshot_state`` dict into the canonical,
    seat-indexed assertion snapshot. Settlements/cities are already
    disjoint in the engine snapshot. Holders are mapped from actor names
    back to seat indices (or null)."""
    actor_to_seat = {"player_a": 0, "player_b": 1}

    def holder_seat(actor: str | None) -> int | None:
        if actor is None:
            return None
        return actor_to_seat.get(actor)

    players: list[dict[str, Any]] = [{}, {}]
    settlements: list[list[int]] = [[], []]
    cities: list[list[int]] = [[], []]
    roads: list[list[int]] = [[], []]
    for actor, seat in actor_to_seat.items():
        players[seat] = _normalise_player(raw["players"][actor])
        settlements[seat] = sorted(int(i) for i in raw["settlements"].get(actor, []))
        cities[seat] = sorted(int(i) for i in raw["cities"].get(actor, []))
        roads[seat] = sorted(int(i) for i in raw["roads"].get(actor, []))

    return {
        "robber_hex": int(raw["robber_hex"]),
        "settlements": settlements,
        "cities": cities,
        "roads": roads,
        "players": players,
        "longest_road_holder": holder_seat(raw.get("longest_road_holder")),
        "largest_army_holder": holder_seat(raw.get("largest_army_holder")),
    }


# ---------------------------------------------------------------------------
# Steal-outcome capture via the broadcast bus.
# ---------------------------------------------------------------------------


class _StealSink:
    """Buffers STEAL broadcast events so the driver can read the actual
    stolen resource immediately after a robber move."""

    def __init__(self) -> None:
        self.last_steal: dict[str, Any] | None = None

    def callback(self, event: dict[str, Any]) -> None:
        if event.get("type") == "STEAL":
            self.last_steal = dict(event)

    def take(self) -> dict[str, Any] | None:
        out = self.last_steal
        self.last_steal = None
        return out


# ---------------------------------------------------------------------------
# Random-legal driver.
# ---------------------------------------------------------------------------


def _snapshot(game: Any, ctx: _Ctx) -> dict[str, Any]:
    raw = game.snapshot_state(ctx.seat_to_actor, ctx.pixel_to_idx, ctx.edge_key_to_idx)
    return _normalise_snapshot(raw, ctx.seat_to_actor)


class _Ctx:
    """Bundle of the per-game index maps + actor mapping."""

    def __init__(self, game: Any) -> None:
        self.game = game
        self.pixel_to_idx, self.edge_key_to_idx = _build_maps(game.board)
        self.idx_to_pixel = {idx: px for px, idx in self.pixel_to_idx.items()}
        players = list(game.playerQueue.queue)
        self.players = players
        self.seat_to_actor = {players[0].name: "player_a", players[1].name: "player_b"}
        self.name_to_seat = {players[0].name: 0, players[1].name: 1}

    def edge_idx(self, v1: Any, v2: Any) -> int:
        return self.edge_key_to_idx[_edge_key(v1, v2)]

    def vertex_idx(self, v_px: Any) -> int:
        return int(self.pixel_to_idx[v_px])


def _hand_total(player: Any) -> int:
    return int(sum(player.resources.values()))


def record_game(seed: int, *, max_main_turns: int = 200) -> dict[str, Any]:
    """Play one full reference game under a random-legal policy and return
    the conformance replay-log as a JSON-safe dict.

    Determinism: ``record_game(seed)`` is fully reproducible. We seed
    BOTH global RNG sources the reference engine consumes:

    * ``numpy`` — the board layout AND the engine's ``np.random``
      steal-resource draw (``player.steal_resource`` /
      ``draw_devCard`` use ``np.random``).
    * the GLOBAL stdlib ``random`` module — the Rust-backed
      ``StackedDice`` seeds itself from ``random.getrandbits(64)`` at
      construction (see ``engine/dice.py``), so the dice keystream is
      pinned only when the global ``random`` is seeded BEFORE the
      ``catanGame`` is built.

    The recorder draws its own legal-move selections from an
    INDEPENDENT seeded ``random.Random(seed)`` instance — distinct from
    the global stream the dice consumes, so the two never interleave,
    yet both are pinned by ``seed``.

    Order matters: seed the global BEFORE constructing ``catanGame`` so
    the dice picks up the determined seed at construction time.
    """
    np.random.seed(seed)
    random.seed(seed)  # pins the global stream the Rust dice seeds from
    rng = random.Random(seed)  # the recorder's own legal-move selection stream

    game = catanGame(render_mode="no_human")
    ctx = _Ctx(game)
    sink = _StealSink()
    game.broadcast.subscribe(sink.callback)

    # Capture the INITIAL board (robber on the desert) before any play —
    # the TS conformance test LOADS this as the starting state; per-step
    # robber movement is tracked in each step's ``state_after.robber_hex``.
    board_payload = _board_payload(game.board, ctx.pixel_to_idx)

    steps: list[dict[str, Any]] = []

    def emit(actor_seat: int, action: dict[str, Any], outcome: dict[str, Any]) -> None:
        steps.append(
            {
                "actor": actor_seat,
                "action": action,
                "outcome": outcome,
                "state_after": _snapshot(game, ctx),
            }
        )

    # --- Setup snake (1 -> 2 -> 2 -> 1) ------------------------------------
    p0, p1 = ctx.players[0], ctx.players[1]
    snake = [
        (p0, 0, False),
        (p1, 1, False),
        (p1, 1, True),
        (p0, 0, True),
    ]
    for player, seat, grants in snake:
        game.currentPlayer = player
        # Settlement.
        legal_v = list(game.board.get_setup_settlements(player).keys())
        v_px = rng.choice(legal_v)
        player.build_settlement(v_px, game.board, is_free=True)
        if grants:
            # Second settlement grants one card per adjacent non-desert hex
            # (the 1v1 Colonist rule the env applies — mirrored by TS setup.ts).
            for adj_hex in game.board.boardGraph[v_px].adjacent_hex_indices:
                res = game.board.hexTileDict[adj_hex].resource_type
                if res != "DESERT":
                    player.resources[res] += 1
        emit(seat, {"kind": "BuildSettlement", "args": {"vertex": ctx.vertex_idx(v_px)}}, {})
        # Road incident to the just-placed settlement.
        legal_r = list(game.board.get_setup_roads(player).keys())
        r = rng.choice(legal_r)
        player.build_road(r[0], r[1], game.board, is_free=True)
        emit(seat, {"kind": "BuildRoad", "args": {"edge": ctx.edge_idx(r[0], r[1])}}, {})

    game.gameSetup = False

    # --- Main turn loop ----------------------------------------------------
    turn_seat = 0
    for _ in range(max_main_turns):
        if game.gameOver:
            break
        player = ctx.players[turn_seat]
        game.currentPlayer = player
        player.updateDevCards()
        player.devCardPlayedThisTurn = False

        # 1) Roll.
        dice_roll = int(game.dice.roll(player, game.last_player_to_roll_7))
        if dice_roll == 7:
            game.last_player_to_roll_7 = player
        emit(turn_seat, {"kind": "RollDice", "args": {}}, {"dice_roll": dice_roll})

        if dice_roll == 7:
            _resolve_seven(game, ctx, player, turn_seat, sink, rng, emit)
        else:
            _distribute(game, dice_roll)
            # Distribution mutates state silently in the reference (no env
            # action); fold it into the NEXT emitted step's state_after by
            # snapshotting now via a no-op marker is unnecessary — the TS
            # RollDice handler distributes too, so the RollDice step's
            # state_after (emitted above, BEFORE distribution) would mismatch.
            # Re-emit the post-distribution snapshot onto the RollDice step.
            steps[-1]["state_after"] = _snapshot(game, ctx)

        # 2) Free-form actions (random count) until end turn / win.
        _play_actions(game, ctx, player, turn_seat, sink, rng, emit)

        if game.gameOver:
            break

        # 3) End turn.
        emit(turn_seat, {"kind": "EndTurn", "args": {}}, {})
        turn_seat = 1 - turn_seat

    game.broadcast.unsubscribe(sink.callback)

    return {
        "schema_version": CONFORMANCE_SCHEMA_VERSION,
        "seed": int(seed),
        "board": board_payload,
        "players": [
            {"seat": 0, "name": str(ctx.players[0].name)},
            {"seat": 1, "name": str(ctx.players[1].name)},
        ],
        "steps": steps,
    }


def _distribute(game: Any, dice_roll: int) -> None:
    """Distribute production for a non-7 roll (reference
    ``update_playerResources`` non-7 branch)."""
    game.update_playerResources(dice_roll, game.currentPlayer)


def _resolve_seven(
    game: Any,
    ctx: _Ctx,
    player: Any,
    turn_seat: int,
    sink: _StealSink,
    rng: random.Random,
    emit: Any,
) -> None:
    """Handle a rolled 7: each over-9 hand discards floor(n/2), then the
    current seat moves the robber (friendly filter) + steals."""
    # Discards: each seat with hand strictly > 9.
    for seat, p in enumerate(ctx.players):
        total = _hand_total(p)
        if total > 9:
            n = total // 2
            discarded = _pick_discard(p, n, rng)
            for res, cnt in discarded.items():
                p.resources[res] -= cnt
            emit(seat, {"kind": "Discard", "args": {"resources": discarded}}, {})

    # Robber move + steal (current seat).
    _move_robber(game, ctx, player, turn_seat, sink, rng, emit, dev_knight=False)


def _pick_discard(player: Any, n: int, rng: random.Random) -> dict[str, int]:
    """Pick exactly ``n`` cards to discard from the hand (Charlesworth)."""
    pool: list[str] = []
    for res in RESOURCES_CW:
        pool += [res] * int(player.resources[res])
    rng.shuffle(pool)
    chosen = pool[:n]
    out: dict[str, int] = {}
    for res in chosen:
        out[res] = out.get(res, 0) + 1
    return out


def _move_robber(
    game: Any,
    ctx: _Ctx,
    player: Any,
    turn_seat: int,
    sink: _StealSink,
    rng: random.Random,
    emit: Any,
    *,
    dev_knight: bool,
) -> None:
    """Move the robber to a random legal hex + steal; emit MoveRobber or
    PlayKnight with the captured steal outcome."""
    spots = list(game.board.get_robber_spots().keys())
    if not spots:
        return
    hex_i = rng.choice(spots)
    robbable = game.board.get_players_to_rob(hex_i)
    # The reference robs a SINGLE chosen victim (the GUI picks one); pick a
    # random eligible opponent with a non-empty hand, else None.
    victims = [pl for pl in robbable if pl is not player and _hand_total(pl) > 0]
    victim = rng.choice(victims) if victims else None
    sink.take()  # clear any stale steal
    player.move_robber(hex_i, game.board, victim)
    steal_event = sink.take()
    steal_res = str(steal_event["resource"]) if steal_event else None
    victim_seat = ctx.name_to_seat[victim.name] if victim is not None else None
    if dev_knight:
        player.knightsPlayed += 1
        game.check_largest_army(player)
        action = {"kind": "PlayKnight", "args": {"hex": int(hex_i), "victim": victim_seat}}
    else:
        action = {"kind": "MoveRobber", "args": {"hex": int(hex_i), "victim": victim_seat}}
    emit(turn_seat, action, {"steal": steal_res})


def _play_actions(
    game: Any,
    ctx: _Ctx,
    player: Any,
    turn_seat: int,
    sink: _StealSink,
    rng: random.Random,
    emit: Any,
) -> None:
    """Take a random number of legal build / trade / dev actions this turn."""
    n_actions = rng.randint(0, 6)
    for _ in range(n_actions):
        if game.gameOver:
            return
        choices = _legal_main_choices(game, ctx, player)
        if not choices:
            return
        choice = rng.choice(choices)
        _apply_main_choice(game, ctx, player, turn_seat, sink, rng, emit, choice)
        if player.victoryPoints >= game.maxPoints:
            game.gameOver = True
            return


def _legal_main_choices(game: Any, ctx: _Ctx, player: Any) -> list[str]:
    """The kinds of main-phase actions currently affordable / legal."""
    out: list[str] = []
    res = player.resources
    if (
        res["WOOD"] > 0
        and res["BRICK"] > 0
        and player.roadsLeft > 0
        and game.board.get_potential_roads(player)
    ):
        out.append("BuildRoad")
    if (
        res["WOOD"] > 0
        and res["BRICK"] > 0
        and res["WHEAT"] > 0
        and res["SHEEP"] > 0
        and player.settlementsLeft > 0
        and game.board.get_potential_settlements(player)
    ):
        out.append("BuildSettlement")
    if (
        res["ORE"] >= 3
        and res["WHEAT"] >= 2
        and player.citiesLeft > 0
        and game.board.get_potential_cities(player)
    ):
        out.append("BuildCity")
    if (
        res["ORE"] >= 1
        and res["WHEAT"] >= 1
        and res["SHEEP"] >= 1
        and sum(game.board.devCardStack.values()) > 0
    ):
        out.append("BuyDevCard")
    # Bank trade: any resource the player can afford at its best ratio.
    if _any_bank_trade(player):
        out.append("BankTrade")
    # Playable dev cards (not bought this turn, one non-VP per turn).
    if not player.devCardPlayedThisTurn:
        if player.devCards.get("KNIGHT", 0) >= 1:
            out.append("PlayKnight")
        if player.devCards.get("YEAROFPLENTY", 0) >= 1:
            out.append("PlayYearOfPlenty")
        if player.devCards.get("MONOPOLY", 0) >= 1:
            out.append("PlayMonopoly")
        if (
            player.devCards.get("ROADBUILDER", 0) >= 1
            and player.roadsLeft > 0
            and game.board.get_potential_roads(player)
        ):
            out.append("PlayRoadBuilder")
    return out


def _best_ratio(player: Any, res: str) -> int:
    if ("2:1 " + res) in player.portList:
        return 2
    if "3:1 PORT" in player.portList:
        return 3
    return 4


def _any_bank_trade(player: Any) -> bool:
    return any(player.resources[res] >= _best_ratio(player, res) for res in RESOURCES_CW)


def _apply_main_choice(
    game: Any,
    ctx: _Ctx,
    player: Any,
    turn_seat: int,
    sink: _StealSink,
    rng: random.Random,
    emit: Any,
    choice: str,
) -> None:
    board = game.board
    if choice == "BuildRoad":
        road = rng.choice(list(board.get_potential_roads(player).keys()))
        player.build_road(road[0], road[1], board, is_free=False)
        game.check_longest_road(player)
        emit(turn_seat, {"kind": "BuildRoad", "args": {"edge": ctx.edge_idx(road[0], road[1])}}, {})
    elif choice == "BuildSettlement":
        v_px = rng.choice(list(board.get_potential_settlements(player).keys()))
        player.build_settlement(v_px, board, is_free=False)
        game.check_longest_road(player)
        emit(
            turn_seat,
            {"kind": "BuildSettlement", "args": {"vertex": ctx.vertex_idx(v_px)}},
            {},
        )
    elif choice == "BuildCity":
        v_px = rng.choice(list(board.get_potential_cities(player).keys()))
        player.build_city(v_px, board)
        emit(turn_seat, {"kind": "BuildCity", "args": {"vertex": ctx.vertex_idx(v_px)}}, {})
    elif choice == "BuyDevCard":
        before_new = list(player.newDevCards)
        before_vp_cards = int(player.devCards.get("VP", 0))
        player.draw_devCard(board)
        drawn = _detect_drawn_card(player, before_new, before_vp_cards)
        emit(turn_seat, {"kind": "BuyDevCard", "args": {}}, {"dev_card": drawn})
    elif choice == "BankTrade":
        give = rng.choice(
            [r for r in RESOURCES_CW if player.resources[r] >= _best_ratio(player, r)]
        )
        recv = rng.choice([r for r in RESOURCES_CW if r != give])
        player.trade_with_bank(give, recv)
        emit(
            turn_seat,
            {"kind": "BankTrade", "args": {"give": give, "receive": recv}},
            {},
        )
    elif choice == "PlayKnight":
        player.devCards["KNIGHT"] -= 1
        player.devCardPlayedThisTurn = True
        _move_robber(game, ctx, player, turn_seat, sink, rng, emit, dev_knight=True)
    elif choice == "PlayYearOfPlenty":
        player.devCards["YEAROFPLENTY"] -= 1
        player.devCardPlayedThisTurn = True
        first = rng.choice(RESOURCES_CW)
        second = rng.choice(RESOURCES_CW)
        player.resources[first] += 1
        player.resources[second] += 1
        emit(
            turn_seat,
            {"kind": "PlayYearOfPlenty", "args": {"first": first, "second": second}},
            {},
        )
    elif choice == "PlayMonopoly":
        player.devCards["MONOPOLY"] -= 1
        player.devCardPlayedThisTurn = True
        resource = rng.choice(RESOURCES_CW)
        opp = ctx.players[1 - turn_seat]
        taken = int(opp.resources[resource])
        opp.resources[resource] -= taken
        player.resources[resource] += taken
        emit(turn_seat, {"kind": "PlayMonopoly", "args": {"resource": resource}}, {})
    elif choice == "PlayRoadBuilder":
        player.devCards["ROADBUILDER"] -= 1
        player.devCardPlayedThisTurn = True
        emit(turn_seat, {"kind": "PlayRoadBuilder", "args": {}}, {})
        # Up to 2 free roads if legal.
        for _ in range(2):
            potential = list(board.get_potential_roads(player).keys())
            if not potential or player.roadsLeft <= 0:
                break
            road = rng.choice(potential)
            player.build_road(road[0], road[1], board, is_free=True)
            game.check_longest_road(player)
            emit(
                turn_seat,
                {"kind": "BuildRoad", "args": {"edge": ctx.edge_idx(road[0], road[1])}},
                {},
            )


def _detect_drawn_card(player: Any, before_new: list[str], before_vp_cards: int) -> str:
    """Identify the dev card a ``draw_devCard`` call drew, in Torevan
    names. A non-VP card lands in ``newDevCards``; a VP card increments
    ``devCards['VP']`` immediately."""
    new_cards = list(player.newDevCards)
    if len(new_cards) > len(before_new):
        drawn = new_cards[-1]
        return _SNAPSHOT_DEV_TO_TS.get(drawn, drawn)
    if int(player.devCards.get("VP", 0)) > before_vp_cards:
        return "VP"
    raise AssertionError("draw_devCard did not change the hand (bank empty?)")


def save_log(log: dict[str, Any], out_path: Path) -> None:
    """Write ``log`` to ``out_path`` as pretty JSON (UTF-8, trailing
    newline)."""
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
        fh.write("\n")
