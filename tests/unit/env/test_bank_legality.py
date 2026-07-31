"""D4 — the mask must never OFFER a bank-unsupplyable receive.

``env/masks.py`` used to set ``resource2_default[:] = True`` for BANK_TRADE with
no bank check, while ``engine/player.trade_with_bank`` early-returns on an empty
bank leaving state BYTE-IDENTICAL. A stable-argmax policy therefore re-picks the
same action forever: the eval harness loops ``while not terminated and not
truncated`` and truncation only advances at a turn boundary. That is an
unbounded loop that already existed; the fix is at the legality layer, NOT an
apply-time no-op (which is what creates the fixed point).
"""

from __future__ import annotations

import numpy as np

from catan_rl.engine.game import catanGame
from catan_rl.env.masks import compute_action_masks
from catan_rl.policy.obs_encoder import EnvObsState
from catan_rl.policy.obs_schema import RESOURCES_CW, ActionType


def _index_maps(board):
    vertex_to_idx = {px: idx for idx, px in board.vertex_index_to_pixel_dict.items()}
    seen: set[tuple[str, str]] = set()
    edge_to_idx: dict[tuple[str, str], int] = {}
    for v_pt, v_obj in board.boardGraph.items():
        for nb_pt in v_obj.neighbors:
            s1, s2 = str(v_pt), str(nb_pt)
            key = (s1, s2) if s1 < s2 else (s2, s1)
            if key not in seen:
                seen.add(key)
                edge_to_idx[key] = len(edge_to_idx)
    return vertex_to_idx, edge_to_idx


def _main_phase_masks(game, player):
    vmap, emap = _index_maps(game.board)
    return compute_action_masks(
        game, player, EnvObsState(initial_placement_phase=False), vmap, emap
    )


def _drain_bank(board, keep: set[str]) -> None:
    for r in RESOURCES_CW:
        if r not in keep:
            n = board.resourceBank[r]
            if n:
                board.bank_draw({r: n})


def test_bank_trade_receive_is_gated_on_bank_supply() -> None:
    game = catanGame(render_mode=None)
    board = game.board
    p = next(iter(game.playerQueue.queue))
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.resources["WOOD"] = 6  # 4:1 generic trade available
    _drain_bank(board, keep={"ORE"})

    masks = _main_phase_masks(game, p)
    assert masks["type"][ActionType.BANK_TRADE]
    ore = RESOURCES_CW.index("ORE")
    for i, r in enumerate(RESOURCES_CW):
        if i == ore:
            assert masks["resource2_trade"][i], "supplyable receive must be offered"
        else:
            assert not masks["resource2_trade"][i], f"unsupplyable receive offered: {r}"


def test_bank_trade_type_withheld_when_only_the_given_resource_remains() -> None:
    """The only bank-supplyable resource is the one being GIVEN, and
    ``heads._resource2_mask`` forbids r2 == r1 — so there is no legal trade at
    all and the type must not be offered."""
    game = catanGame(render_mode=None)
    board = game.board
    p = next(iter(game.playerQueue.queue))
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.resources["WOOD"] = 6
    _drain_bank(board, keep={"WOOD"})

    masks = _main_phase_masks(game, p)
    assert not masks["type"][ActionType.BANK_TRADE]
    assert not masks["resource2_trade"].any()


def test_year_of_plenty_second_pick_is_gated_on_bank_supply() -> None:
    game = catanGame(render_mode=None)
    board = game.board
    p = next(iter(game.playerQueue.queue))
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.devCards["YEAROFPLENTY"] = 1
    p.devCardPlayedThisTurn = False
    _drain_bank(board, keep={"SHEEP"})

    masks = _main_phase_masks(game, p)
    assert masks["type"][ActionType.PLAY_YOP]
    sheep = RESOURCES_CW.index("SHEEP")
    assert list(np.flatnonzero(masks["resource2_yop"])) == [sheep]


def test_year_of_plenty_first_pick_is_gated_on_bank_supply() -> None:
    """The FIRST pick draws from the bank too. ``resource1_default`` cannot
    carry the gate (Monopoly shares it and is bank-independent), so YoP reads
    its own ``resource1_yop`` key."""
    game = catanGame(render_mode=None)
    board = game.board
    p = next(iter(game.playerQueue.queue))
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.devCards["YEAROFPLENTY"] = 1
    p.devCardPlayedThisTurn = False
    _drain_bank(board, keep={"SHEEP"})

    masks = _main_phase_masks(game, p)
    sheep = RESOURCES_CW.index("SHEEP")
    assert list(np.flatnonzero(masks["resource1_yop"])) == [sheep], (
        "a first pick the bank cannot supply must not be offered"
    )


def test_year_of_plenty_doubled_pick_needs_two_in_the_bank() -> None:
    """``bank[first] >= 2`` when ``first == second``. With exactly one SHEEP
    and one ORE left, (SHEEP, SHEEP) must NOT be offered but (SHEEP, ORE)
    must be."""
    game = catanGame(render_mode=None)
    board = game.board
    p = next(iter(game.playerQueue.queue))
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.devCards["YEAROFPLENTY"] = 1
    p.devCardPlayedThisTurn = False
    _drain_bank(board, keep={"SHEEP", "ORE"})
    for r in ("SHEEP", "ORE"):
        board.bank_draw({r: board.resourceBank[r] - 1})

    masks = _main_phase_masks(game, p)
    sheep = RESOURCES_CW.index("SHEEP")
    ore = RESOURCES_CW.index("ORE")
    assert masks["type"][ActionType.PLAY_YOP]
    assert sorted(np.flatnonzero(masks["resource1_yop"])) == sorted([sheep, ore])
    # Doubling either is illegal (only one of each remains).
    assert not masks["resource2_yop_same"].any()
    # The cross pair is legal.
    assert masks["resource2_yop"][sheep] and masks["resource2_yop"][ore]


def test_year_of_plenty_withheld_when_only_one_card_of_one_resource_remains() -> None:
    """A single SHEEP is not a legal YoP at all: doubling needs 2 and there is
    no other supplyable resource to pair with. The old gate offered it."""
    game = catanGame(render_mode=None)
    board = game.board
    p = next(iter(game.playerQueue.queue))
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.devCards["YEAROFPLENTY"] = 1
    p.devCardPlayedThisTurn = False
    _drain_bank(board, keep={"SHEEP"})
    board.bank_draw({"SHEEP": board.resourceBank["SHEEP"] - 1})

    masks = _main_phase_masks(game, p)
    assert not masks["type"][ActionType.PLAY_YOP]
    assert not masks["resource1_yop"].any()


def test_monopoly_first_pick_stays_bank_blind() -> None:
    """Monopoly steals from the OPPONENT, not the bank — draining the bank must
    not narrow its pick set (the reason YoP needed a separate key)."""
    game = catanGame(render_mode=None)
    board = game.board
    p = next(iter(game.playerQueue.queue))
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.devCards["MONOPOLY"] = 1
    p.devCardPlayedThisTurn = False
    _drain_bank(board, keep=set())

    masks = _main_phase_masks(game, p)
    assert masks["type"][ActionType.PLAY_MONOPOLY]
    assert masks["resource1_default"].all()


def test_year_of_plenty_withheld_when_the_bank_is_empty() -> None:
    game = catanGame(render_mode=None)
    board = game.board
    p = next(iter(game.playerQueue.queue))
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.devCards["YEAROFPLENTY"] = 1
    p.devCardPlayedThisTurn = False
    _drain_bank(board, keep=set())

    masks = _main_phase_masks(game, p)
    assert not masks["type"][ActionType.PLAY_YOP]
    assert not masks["resource1_yop"].any()
    assert not masks["resource2_yop"].any()
    assert not masks["resource2_yop_same"].any()


def test_no_bank_trade_fixed_point_when_the_bank_cannot_supply() -> None:
    """Reproduces the livelock shape: with the receive drained, the mask must
    stop offering the trade, so a stable-argmax policy cannot re-pick a
    state-preserving action forever."""
    game = catanGame(render_mode=None)
    board = game.board
    p = next(iter(game.playerQueue.queue))
    p.resources = dict.fromkeys(RESOURCES_CW, 0)
    p.resources["WOOD"] = 8
    _drain_bank(board, keep={"ORE"})

    masks = _main_phase_masks(game, p)
    ore = RESOURCES_CW.index("ORE")
    wood = RESOURCES_CW.index("WOOD")
    # Exhaust ORE mid-stream: the engine would silently no-op from here on.
    board.bank_draw({"ORE": board.resourceBank["ORE"]})
    masks_after = _main_phase_masks(game, p)
    assert masks["resource2_trade"][ore]
    assert not masks_after["resource2_trade"].any()
    assert not masks_after["type"][ActionType.BANK_TRADE]
    assert not masks_after["resource1_trade"][wood]


# ---------------------------------------------------------------------------
# The autoregressive half of the D4 rule, at the head layer.
# ---------------------------------------------------------------------------


def test_heads_swap_in_the_doubled_yop_vector_at_the_res1_index() -> None:
    """``resource2_yop`` (bank >= 1) is the DIFFERENT-pick vector; at the index
    the agent already picked, ``heads._resource2_mask`` must read
    ``resource2_yop_same`` (bank >= 2) instead. Without the swap, a bank
    holding exactly one SHEEP would offer (SHEEP, SHEEP)."""
    import torch

    from catan_rl.policy.heads import CatanActionHeads

    n = len(RESOURCES_CW)
    sheep = RESOURCES_CW.index("SHEEP")
    ore = RESOURCES_CW.index("ORE")
    masks = {
        # bank: SHEEP=1, ORE=1, rest 0.
        "resource2_yop": torch.zeros(1, n, dtype=torch.bool),
        "resource2_yop_same": torch.zeros(1, n, dtype=torch.bool),
        "resource2_trade": torch.zeros(1, n, dtype=torch.bool),
    }
    masks["resource2_yop"][0, sheep] = True
    masks["resource2_yop"][0, ore] = True

    type_idx = torch.tensor([int(ActionType.PLAY_YOP)])
    res1_idx = torch.tensor([sheep])
    out = CatanActionHeads._resource2_mask(type_idx, res1_idx, masks)
    assert not bool(out[0, sheep]), "doubled pick offered with only one in the bank"
    assert bool(out[0, ore])

    # With two SHEEP in the bank the doubled pick becomes legal again.
    masks["resource2_yop_same"][0, sheep] = True
    out = CatanActionHeads._resource2_mask(type_idx, res1_idx, masks)
    assert bool(out[0, sheep])


def test_heads_route_the_yop_first_pick_to_its_own_key() -> None:
    import torch

    from catan_rl.policy.heads import CatanActionHeads

    n = len(RESOURCES_CW)
    masks = {
        "resource1_default": torch.ones(1, n, dtype=torch.bool),
        "resource1_yop": torch.zeros(1, n, dtype=torch.bool),
        "resource1_trade": torch.zeros(1, n, dtype=torch.bool),
        "resource1_discard": torch.zeros(1, n, dtype=torch.bool),
    }
    masks["resource1_yop"][0, RESOURCES_CW.index("ORE")] = True

    yop = CatanActionHeads._resource1_mask(torch.tensor([int(ActionType.PLAY_YOP)]), masks)
    assert yop.sum().item() == 1
    mono = CatanActionHeads._resource1_mask(torch.tensor([int(ActionType.PLAY_MONOPOLY)]), masks)
    assert bool(mono.all()), "Monopoly is bank-independent and must stay wide open"
