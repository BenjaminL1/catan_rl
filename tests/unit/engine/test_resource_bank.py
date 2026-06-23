"""Finite per-resource bank — spec 009.

Covers the official depletion rule (``resolve_bank_production``), the bank
state + helpers on ``catanBoard``, the global conservation invariant across
the recorder and env drivers (the cross-driver guard the plan review required),
apply-time gating, and the resource-key domain.
"""

from __future__ import annotations

import pytest

from catan_rl.engine.board import catanBoard
from catan_rl.engine.game import _BANK_RESOURCES, catanGame, resolve_bank_production
from catan_rl.engine.player import player as EnginePlayer

# ---------------------------------------------------------------------------
# resolve_bank_production — the depletion rule, ported verbatim from TS
# ---------------------------------------------------------------------------


def test_depletion_noop_when_nobody_owed() -> None:
    assert resolve_bank_production(5, 0, 0) == (0, 0, 5)


def test_depletion_both_paid_when_supply_covers() -> None:
    assert resolve_bank_production(5, 2, 2) == (2, 2, 1)
    # exact-fit boundary
    assert resolve_bank_production(4, 2, 2) == (2, 2, 0)


def test_depletion_both_owed_short_neither_receives() -> None:
    # total 4 > avail 3, both owed -> NEITHER, bank unchanged.
    assert resolve_bank_production(3, 2, 2) == (0, 0, 3)
    assert resolve_bank_production(1, 1, 1) == (0, 0, 1)


def test_depletion_sole_claimant_seat0_takes_remainder() -> None:
    # only seat 0 owed, supply short -> seat 0 takes avail, bank -> 0.
    assert resolve_bank_production(2, 3, 0) == (2, 0, 0)


def test_depletion_sole_claimant_seat1_takes_remainder() -> None:
    # only seat 1 owed, supply short -> seat 1 takes avail, bank -> 0.
    # (Asymmetric vs the seat-0 case — catches a d0/d1 transposition.)
    assert resolve_bank_production(2, 0, 3) == (0, 2, 0)


def test_depletion_seat_asymmetry_is_distinguishable() -> None:
    s0 = resolve_bank_production(2, 3, 0)
    s1 = resolve_bank_production(2, 0, 3)
    assert s0 != s1  # (2,0,0) vs (0,2,0)


# ---------------------------------------------------------------------------
# Bank state + helpers on the board
# ---------------------------------------------------------------------------


def _fresh_board() -> catanBoard:
    return catanBoard()


def test_bank_initialised_to_19_each() -> None:
    board = _fresh_board()
    assert board.resourceBank == {"BRICK": 19, "ORE": 19, "SHEEP": 19, "WHEAT": 19, "WOOD": 19}
    assert sum(board.resourceBank.values()) == 95


def test_bank_recirculate_and_draw_roundtrip() -> None:
    board = _fresh_board()
    board.bank_draw({"WOOD": 3, "ORE": 1})
    assert board.resourceBank["WOOD"] == 16
    assert board.resourceBank["ORE"] == 18
    board.bank_recirculate({"WOOD": 3, "ORE": 1})
    assert board.resourceBank["WOOD"] == 19
    assert board.resourceBank["ORE"] == 19


def test_bank_draw_underflow_is_a_loud_failure() -> None:
    board = _fresh_board()
    board.resourceBank["SHEEP"] = 1
    with pytest.raises(AssertionError):
        board.bank_draw({"SHEEP": 2})


def test_bank_can_supply() -> None:
    board = _fresh_board()
    board.resourceBank["BRICK"] = 1
    assert board.bank_can_supply({"BRICK": 1})
    assert not board.bank_can_supply({"BRICK": 2})


def test_key_domain_charlesworth_delta_does_not_drop_a_resource() -> None:
    # The RL stack keys resources in Charlesworth order; the engine bank keys
    # them in engine order. Both are the SAME five string keys, so a delta
    # built in either order recirculates without a KeyError or a dropped key.
    board = _fresh_board()
    board.bank_draw({"WOOD": 2, "BRICK": 2, "WHEAT": 2, "ORE": 2, "SHEEP": 2})  # Charlesworth order
    assert all(board.resourceBank[r] == 17 for r in _BANK_RESOURCES)


# ---------------------------------------------------------------------------
# Direct depletion at the production level (update_playerResources two-pass)
# ---------------------------------------------------------------------------


def _vertices_adjacent_to_hex(game: catanGame, hex_idx: int) -> list:
    return [v for v, node in game.board.boardGraph.items() if hex_idx in node.adjacent_hex_indices]


def _pick_nondesert_hex(game: catanGame) -> int:
    for h, tile in game.board.hexTileDict.items():
        if tile.resource_type != "DESERT" and not tile.has_robber:
            return h
    raise AssertionError("no non-desert hex found")


def test_production_both_owed_short_grants_neither() -> None:
    game = catanGame(render_mode="no_human")
    players = list(game.playerQueue.queue)
    p0, p1 = players[0], players[1]
    h = _pick_nondesert_hex(game)
    res = game.board.hexTileDict[h].resource_type
    num = game.board.hexTileDict[h].number_token
    verts = _vertices_adjacent_to_hex(game, h)
    assert len(verts) >= 2
    p0.buildGraph["SETTLEMENTS"].append(verts[0])
    p1.buildGraph["SETTLEMENTS"].append(verts[1])
    # Supply too short to cover combined demand (1 + 1 = 2 > 1), both owed.
    game.board.resourceBank[res] = 1
    before0, before1 = p0.resources[res], p1.resources[res]
    game.update_playerResources(num, p0)
    assert p0.resources[res] == before0  # neither receives
    assert p1.resources[res] == before1
    assert game.board.resourceBank[res] == 1  # bank unchanged


def test_production_sole_claimant_takes_remainder() -> None:
    game = catanGame(render_mode="no_human")
    players = list(game.playerQueue.queue)
    p0 = players[0]
    h = _pick_nondesert_hex(game)
    res = game.board.hexTileDict[h].resource_type
    num = game.board.hexTileDict[h].number_token
    verts = _vertices_adjacent_to_hex(game, h)
    # Only p0 is owed; bank holds 1 but p0 (as a city) is owed 2.
    p0.buildGraph["CITIES"].append(verts[0])
    game.board.resourceBank[res] = 1
    before0 = p0.resources[res]
    game.update_playerResources(num, p0)
    assert p0.resources[res] == before0 + 1  # took the remaining 1
    assert game.board.resourceBank[res] == 0


# ---------------------------------------------------------------------------
# Apply-time gating (no exception contract -> no-op on an empty bank)
# ---------------------------------------------------------------------------


def test_bank_trade_into_empty_supply_is_a_noop() -> None:
    board = _fresh_board()
    p = EnginePlayer("P", "red")
    p.resources["WOOD"] = 4
    board.resourceBank["BRICK"] = 0  # bank cannot supply the receive side
    before = dict(p.resources)
    p.trade_with_bank("WOOD", "BRICK", board)
    assert p.resources == before  # rejected: hand unchanged


# ---------------------------------------------------------------------------
# Cross-driver conservation — the global guard (recorder path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [7, 8, 15, 1, 42, 123])
def test_conservation_holds_across_recorded_game(seed: int) -> None:
    from catan_rl.conformance.recorder import record_game

    # assert_conservation checks bank[R] + Sigma hands[R] == 19 after EVERY step,
    # driving the engine + recorder mutation paths (setup grant, production,
    # build, dev-buy, bank-trade, discard, YoP, monopoly, steal).
    record_game(seed, assert_conservation=True)
