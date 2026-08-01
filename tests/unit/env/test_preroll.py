"""Pre-roll dev-card window — spec ``preroll-dev-cards-r1`` acceptance #2-#8.

R1 gives BOTH seats the Colonist window: one dev card (Knight / YoP / Monopoly,
never Road Builder) may be played before the dice. R0 is the shipped epoch with
no window at all, and ``tests/unit/env/test_masks.py`` still pins its
``roll_pending`` mask as exactly ``[ROLL_DICE]``.
"""

from __future__ import annotations

import numpy as np

from catan_rl.engine.game import catanGame
from catan_rl.env.catan_env import CatanEnv
from catan_rl.env.masks import compute_action_masks
from catan_rl.env.ruleset import PRE_ROLL_DEV_TYPES, RULESET_R0, RULESET_R1
from catan_rl.policy.obs_encoder import EnvObsState
from catan_rl.policy.obs_schema import ActionType

_RES_KEYS = (
    "resource1_default",
    "resource1_yop",
    "resource2_yop",
    "resource2_yop_same",
)


def _build_index_maps(board) -> tuple[dict, dict]:
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


def _masks_for(player, *, roll_pending: bool, ruleset: str, game=None):
    game = game if game is not None else catanGame(render_mode=None)
    vmap, emap = _build_index_maps(game.board)
    return compute_action_masks(
        game,
        player,
        EnvObsState(initial_placement_phase=False, roll_pending=roll_pending),
        vmap,
        emap,
        ruleset=ruleset,
    )


def _fresh_env(ruleset: str = RULESET_R1, *, seed: int = 0) -> CatanEnv:
    """An env driven through the snake draft, parked at the agent's first
    pre-roll node."""
    env = CatanEnv(opponent_type="heuristic", ruleset=ruleset)
    env.reset(seed=seed)
    while env.initial_placement_phase:
        masks = env.get_action_masks()
        atype = int(np.flatnonzero(masks["type"])[0])
        key = "corner_settlement" if atype == ActionType.BUILD_SETTLEMENT else "edge"
        idx = int(np.flatnonzero(masks[key])[0])
        action = np.array([atype, idx, idx, 0, 0, 1], dtype=np.int64)
        env.step(action)
    assert env.roll_pending is True
    return env


def _action(action_type: int, *, tile_idx: int = 0, res1: int = 0, res2: int = 1) -> np.ndarray:
    return np.array([int(action_type), 0, 0, tile_idx, res1, res2], dtype=np.int64)


# ---------------------------------------------------------------------------
# Acceptance #2 — one dev card per turn (FAILS on the un-hoisted roll handler)
# ---------------------------------------------------------------------------


def test_preroll_play_then_roll_leaves_no_second_dev_card() -> None:
    env = _fresh_env()
    agent = env.agent_player
    agent.devCards["KNIGHT"] = 2  # two in hand: only the flag can stop the second
    agent.devCardPlayedThisTurn = False

    env.step(_action(ActionType.PLAY_KNIGHT))
    assert agent.devCardPlayedThisTurn is True
    assert env.roll_pending is True  # the dice are still owed
    assert env.robber_placement_pending is True

    # Resolve the Knight's robber, then roll.
    legal_tile = int(np.flatnonzero(env.get_action_masks()["tile"])[0])
    env.step(_action(ActionType.MOVE_ROBBER, tile_idx=legal_tile))
    env.step(_action(ActionType.ROLL_DICE))

    # The roll must NOT have cleared the "already played" flag.
    assert agent.devCardPlayedThisTurn is True
    if not (env.discard_pending or env.robber_placement_pending):
        type_mask = env.get_action_masks()["type"]
        for dev_type in (
            ActionType.PLAY_KNIGHT,
            ActionType.PLAY_YOP,
            ActionType.PLAY_MONOPOLY,
            ActionType.PLAY_ROAD_BUILDER,
        ):
            assert not type_mask[dev_type], f"{dev_type!r} re-offered after a pre-roll play"


# ---------------------------------------------------------------------------
# Acceptance #3 — a card bought turn N is playable at turn N+1's pre-roll node
# ---------------------------------------------------------------------------


def test_card_bought_last_turn_is_playable_at_next_preroll() -> None:
    env = _fresh_env()
    agent = env.agent_player
    env.step(_action(ActionType.ROLL_DICE))
    while env.discard_pending or env.robber_placement_pending:
        masks = env.get_action_masks()
        key = "resource1_discard" if env.discard_pending else "tile"
        idx = int(np.flatnonzero(masks[key])[0])
        atype = ActionType.DISCARD if env.discard_pending else ActionType.MOVE_ROBBER
        env.step(_action(atype, tile_idx=idx, res1=idx))

    # Simulate "bought this turn": the card sits in newDevCards, unplayable.
    agent.newDevCards.append("KNIGHT")
    assert agent.devCards.get("KNIGHT", 0) == 0

    env.step(_action(ActionType.END_TURN))
    if env.discard_pending:  # opponent rolled a 7 and we owe cards — not this test
        return
    assert env.roll_pending is True
    assert agent.devCards.get("KNIGHT", 0) == 1, "updateDevCards did not run at the turn boundary"
    assert bool(env.get_action_masks()["type"][ActionType.PLAY_KNIGHT])


# ---------------------------------------------------------------------------
# Acceptance #4 — resource sub-masks are identical pre-roll and main-phase
# ---------------------------------------------------------------------------


def test_preroll_resource_masks_match_main_phase() -> None:
    game = catanGame(render_mode=None)
    p = next(iter(game.playerQueue.queue))
    p.devCards["YEAROFPLENTY"] = 1
    p.devCards["MONOPOLY"] = 1
    p.devCardPlayedThisTurn = False

    pre = _masks_for(p, roll_pending=True, ruleset=RULESET_R1, game=game)
    main = _masks_for(p, roll_pending=False, ruleset=RULESET_R1, game=game)
    for key in _RES_KEYS:
        assert np.array_equal(pre[key], main[key]), key
        # An all-zero row is the silent-uniform bug masked_log_softmax hides.
        assert pre[key].any(), key


# ---------------------------------------------------------------------------
# Acceptance #5 / #6 / #7 — what the pre-roll type mask does and does not offer
# ---------------------------------------------------------------------------


def test_preroll_never_offers_end_turn_or_builds_or_buy() -> None:
    game = catanGame(render_mode=None)
    p = next(iter(game.playerQueue.queue))
    for card in ("KNIGHT", "YEAROFPLENTY", "MONOPOLY", "ROADBUILDER"):
        p.devCards[card] = 1
    p.devCardPlayedThisTurn = False
    for r in p.resources:
        p.resources[r] = 8  # affluent: every build/trade would be affordable main-phase

    masks = _masks_for(p, roll_pending=True, ruleset=RULESET_R1, game=game)
    tm = masks["type"]
    assert tm[ActionType.ROLL_DICE]
    for forbidden in (
        ActionType.END_TURN,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_CITY,
        ActionType.BUILD_ROAD,
        ActionType.BUY_DEV_CARD,
        ActionType.BANK_TRADE,
        ActionType.DISCARD,
        ActionType.MOVE_ROBBER,
        ActionType.PLAY_ROAD_BUILDER,  # acceptance #7
    ):
        assert not tm[forbidden], forbidden


def test_road_builder_excluded_preroll_but_legal_main_phase() -> None:
    game = catanGame(render_mode=None)
    p = next(iter(game.playerQueue.queue))
    p.devCards["ROADBUILDER"] = 1
    p.devCardPlayedThisTurn = False

    pre = _masks_for(p, roll_pending=True, ruleset=RULESET_R1, game=game)
    main = _masks_for(p, roll_pending=False, ruleset=RULESET_R1, game=game)
    assert not pre["type"][ActionType.PLAY_ROAD_BUILDER]
    assert main["type"][ActionType.PLAY_ROAD_BUILDER]
    assert ActionType.PLAY_ROAD_BUILDER not in PRE_ROLL_DEV_TYPES


def test_preroll_with_one_knight_is_exactly_roll_plus_knight() -> None:
    game = catanGame(render_mode=None)
    p = next(iter(game.playerQueue.queue))
    p.devCards["KNIGHT"] = 1
    p.devCardPlayedThisTurn = False
    masks = _masks_for(p, roll_pending=True, ruleset=RULESET_R1, game=game)
    assert set(np.flatnonzero(masks["type"]).tolist()) == {
        int(ActionType.ROLL_DICE),
        int(ActionType.PLAY_KNIGHT),
    }


def test_r1_preroll_with_an_empty_dev_hand_is_exactly_roll_dice() -> None:
    """Acceptance #6 is a SPLIT, not a weakening: the ``roll_pending`` + empty
    dev hand ⇒ ``[ROLL_DICE]`` pin has to hold under R1 too, not only R0.

    Without it, the epoch production actually trains under has no exact-set
    assertion at a pre-roll node — the nearest R1 test asserts only ``.any()``
    and ``not END_TURN``.
    """
    game = catanGame(render_mode=None)
    p = next(iter(game.playerQueue.queue))
    for card in ("KNIGHT", "YEAROFPLENTY", "MONOPOLY", "ROADBUILDER"):
        p.devCards[card] = 0
    p.devCardPlayedThisTurn = False
    masks = _masks_for(p, roll_pending=True, ruleset=RULESET_R1, game=game)
    assert np.flatnonzero(masks["type"]).tolist() == [int(ActionType.ROLL_DICE)]


def test_pre_roll_card_names_match_the_pre_roll_action_types() -> None:
    """``PRE_ROLL_DEV_CARD_NAMES`` — the cheap "is this node forced?" test that
    lets the opponent seat skip a whole forward pass — must not drift from
    ``PRE_ROLL_DEV_TYPES``, which is what the mask actually offers."""
    from catan_rl.env.ruleset import PRE_ROLL_DEV_CARD_NAMES

    type_to_card = {
        int(ActionType.PLAY_KNIGHT): "KNIGHT",
        int(ActionType.PLAY_YOP): "YEAROFPLENTY",
        int(ActionType.PLAY_MONOPOLY): "MONOPOLY",
        int(ActionType.PLAY_ROAD_BUILDER): "ROADBUILDER",
    }
    assert {type_to_card[t] for t in PRE_ROLL_DEV_TYPES} == set(PRE_ROLL_DEV_CARD_NAMES)


def test_preroll_type_mask_is_never_empty_so_end_turn_fallback_is_unreachable() -> None:
    """``ROLL_DICE`` is set before the dev bits, so the bottom-of-function
    ``if not type_mask.any(): END_TURN`` cannot fire at a pre-roll node."""
    game = catanGame(render_mode=None)
    p = next(iter(game.playerQueue.queue))
    p.devCardPlayedThisTurn = True  # nothing playable at all
    masks = _masks_for(p, roll_pending=True, ruleset=RULESET_R1, game=game)
    assert masks["type"].any()
    assert not masks["type"][ActionType.END_TURN]


# ---------------------------------------------------------------------------
# R0 preservation
# ---------------------------------------------------------------------------


def test_r0_preroll_is_still_exactly_roll_dice() -> None:
    game = catanGame(render_mode=None)
    p = next(iter(game.playerQueue.queue))
    for card in ("KNIGHT", "YEAROFPLENTY", "MONOPOLY", "ROADBUILDER"):
        p.devCards[card] = 2
    p.devCardPlayedThisTurn = False
    masks = _masks_for(p, roll_pending=True, ruleset=RULESET_R0, game=game)
    assert np.flatnonzero(masks["type"]).tolist() == [int(ActionType.ROLL_DICE)]


def test_r0_env_no_ops_a_preroll_dev_action() -> None:
    env = _fresh_env(RULESET_R0)
    agent = env.agent_player
    agent.devCards["KNIGHT"] = 1
    agent.devCardPlayedThisTurn = False
    env.step(_action(ActionType.PLAY_KNIGHT))
    assert agent.devCards["KNIGHT"] == 1
    assert agent.devCardPlayedThisTurn is False
    assert env.roll_pending is True


# ---------------------------------------------------------------------------
# Branch ordering + seat symmetry (acceptance #8)
# ---------------------------------------------------------------------------


def test_preroll_knight_robber_resolves_before_the_roll() -> None:
    env = _fresh_env()
    agent = env.agent_player
    agent.devCards["KNIGHT"] = 1
    agent.devCardPlayedThisTurn = False
    env.step(_action(ActionType.PLAY_KNIGHT))
    assert env.roll_pending is True
    assert env.robber_placement_pending is True
    # The robber branch is tested ABOVE the roll branch in masks.py, so the
    # next decision is the robber, not the dice.
    tm = env.get_action_masks()["type"]
    assert np.flatnonzero(tm).tolist() == [int(ActionType.MOVE_ROBBER)]


def test_agent_owns_current_player_throughout_its_preroll_window() -> None:
    """The opponent seat claims ``game.currentPlayer`` BEFORE its pre-roll
    window; the agent seat must too, or a pre-roll play (and any robber
    placement it triggers) runs while ``currentPlayer`` still points at the
    opponent — ``engine/game.py``'s Karma branch and its ``DICE_ROLL`` emit
    both read it."""
    env = _fresh_env()
    assert env.game is not None
    assert env.game.currentPlayer is env.agent_player
    agent = env.agent_player
    agent.devCards["KNIGHT"] = 1
    agent.devCardPlayedThisTurn = False
    env.step(_action(ActionType.PLAY_KNIGHT))
    assert env.game.currentPlayer is env.agent_player


def test_opponent_preroll_skips_the_forward_pass_when_nothing_is_playable() -> None:
    """The pre-roll node is FORCED (``{ROLL_DICE}``) whenever the opponent holds
    no whitelisted dev card — the overwhelmingly common case. Sampling the
    snapshot there would burn an obs build + mask build + forward pass on every
    opponent turn of every rollout game."""
    env = _fresh_env()
    opp = env.opponent_player
    assert opp is not None
    calls: list[object] = []

    def _spy(state: object) -> np.ndarray:
        calls.append(state)
        # ROLL_DICE is outside the pre-roll whitelist -> a plain no-op.
        return np.array([int(ActionType.ROLL_DICE), 0, 0, 0, 0, 0], dtype=np.int64)

    env._snapshot_opponent = object()  # type: ignore[assignment]
    env._sample_snapshot_action = _spy  # type: ignore[assignment,method-assign]

    for card in ("KNIGHT", "YEAROFPLENTY", "MONOPOLY", "ROADBUILDER"):
        opp.devCards[card] = 0
    opp.devCardPlayedThisTurn = False
    env._opponent_pre_roll()
    assert calls == []

    opp.devCards["KNIGHT"] = 1
    env._opponent_pre_roll()
    assert len(calls) == 1


def test_agent_and_opponent_preroll_masks_are_identical() -> None:
    env = _fresh_env()
    agent, opp = env.agent_player, env.opponent_player
    for p in (agent, opp):
        p.devCards["KNIGHT"] = 1
        p.devCards["MONOPOLY"] = 1
        p.devCards["YEAROFPLENTY"] = 1
        p.devCardPlayedThisTurn = False
        for r in p.resources:
            p.resources[r] = 3

    agent_masks = env.get_action_masks()
    opp_masks = env._compute_masks(opp, env._opponent_env_state(roll_pending=True))
    assert set(agent_masks) == set(opp_masks)
    for key in agent_masks:
        assert np.array_equal(agent_masks[key], opp_masks[key]), key
