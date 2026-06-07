"""BC dataset generation — heuristic-vs-heuristic rollouts, captured as
(obs, action, mask, belief_target) tuples per decision.

Per ``v2_step3_bc.md`` §1, the deliverable is 30,000 games:

  * 70% canonical heuristic-vs-heuristic
  * 30% perturbed-vs-heuristic (split half ε-greedy / half weight-noised)

Both players' decisions are recorded. Forced single-option moves
(mask sum == 1) are skipped at write time per the panel's unanimous D4.
Each (obs, action, mask) tuple comes with a normalised belief target
(opponent dev-card type distribution) and a discounted terminal outcome
``z_disc = γ^(T-t) · z`` so the BC value head + belief head can train
alongside the policy.

The dataset writer shards into NPZ files of ~5,000 games each so a
worker can mmap a single shard without loading the whole 30k-game
corpus into memory. A manifest JSON records every shard's seed range,
perturbation flag, pair count, and git SHA for provenance.

This module **does not** use ``CatanEnv`` — instead it drives the engine
directly and uses :func:`catan_rl.env.masks.compute_action_masks` to
compute the mask at each decision point. Two reasons:

  1. The env's state machine is built for "agent makes one action at a
     time"; the heuristic makes multiple inline decisions per turn, and
     instrumenting those inline calls is simpler than wrapping each in
     a synthetic env.step().
  2. The env's reward bookkeeping (terminal margin, truncation) isn't
     needed here — we record only the (state, action) pairs and tag
     terminal outcomes after the game ends.

Forced-move filter: a (state, action) pair is "forced" iff the type-head
mask has exactly one legal entry — i.e., the heuristic had no choice.
Records are still emitted to the in-memory log but the loader can skip
them via the ``forced`` flag, OR they can be filtered at write time
via ``include_forced=False`` (the default).
"""  # noqa: RUF002

from __future__ import annotations

import json
import queue
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from catan_rl.agents.heuristic import heuristicAIPlayer
from catan_rl.bc.perturbed_heuristic import (
    EpsilonGreedyHeuristicAIPlayer,
    WeightNoisedHeuristicAIPlayer,
)
from catan_rl.engine.game import catanGame
from catan_rl.engine.tracker import ResourceTracker
from catan_rl.env.hand_tracker import BroadcastHandTracker
from catan_rl.env.masks import compute_action_masks
from catan_rl.policy.obs_encoder import EnvObsState, ObsEncoder
from catan_rl.policy.obs_schema import (
    DEV_CARD_ORDER,
    N_DEV_TYPES,
    RESOURCES_CW,
    ActionType,
)

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class _DecisionRecord:
    """One (state, action, mask) record captured at a heuristic's decision.

    Stored in memory during a game; converted to numpy arrays at shard-
    write time. ``z_disc`` is filled in retrospectively once the game
    ends.
    """

    obs: dict[str, np.ndarray]
    action: np.ndarray  # (6,) int64
    mask: dict[str, np.ndarray]
    belief_target: np.ndarray  # (5,) float32
    forced: bool
    phase: str  # "setup" | "main" | "robber" | "discard" | "roll"
    player_seat: int  # 0 (P1) or 1 (P2)
    step_idx: int  # decision index within the game (0-based)
    # z_disc filled after game end
    z_disc: float = 0.0


@dataclass
class _GameRecord:
    """One game's records + post-game metadata."""

    game_id: int
    perturbation: str  # "canonical" | "epsilon_greedy" | "weight_noised"
    p1_won: int = 0
    p2_won: int = 0
    truncated: bool = False
    total_turns: int = 0
    decisions: list[_DecisionRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Index maps + helpers
# ---------------------------------------------------------------------------


def _build_index_maps(board: Any) -> tuple[dict[Any, int], dict[tuple[str, str], int]]:
    """Same canonical index ordering as CatanEnv._build_index_maps.

    Must match `compute_action_masks` so the recorded action indices are
    consistent with the masks.
    """
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


def _edge_key(v1: Any, v2: Any) -> tuple[str, str]:
    s1, s2 = str(v1), str(v2)
    return (s1, s2) if s1 < s2 else (s2, s1)


def _belief_target_for(opp: Any) -> np.ndarray:
    """Opponent's true hidden dev-card type distribution (5-way, normalised).

    Matches the env's belief-target convention (§2.5b in the design):
    sums ``opponent.devCards`` (held, including VPs) + ``newDevCards``
    (just-bought) across the 5 types and normalises. If opponent has
    zero hidden cards, returns the uniform 5-way distribution (so
    soft-CE is defined).
    """
    out = np.zeros(N_DEV_TYPES, dtype=np.float32)
    for c in getattr(opp, "newDevCards", []) or []:
        if c in DEV_CARD_ORDER:
            out[DEV_CARD_ORDER.index(c)] += 1.0
    for i, name in enumerate(DEV_CARD_ORDER):
        out[i] += float(getattr(opp, "devCards", {}).get(name, 0))
    total = out.sum()
    if total <= 0:
        return np.full(N_DEV_TYPES, 1.0 / N_DEV_TYPES, dtype=np.float32)
    return out / total


# ---------------------------------------------------------------------------
# Heuristic instrumentation
# ---------------------------------------------------------------------------


@dataclass
class _RecorderContext:
    """Carries everything the patched heuristic methods need to record."""

    game: Any
    agent_player: Any
    opp_player: Any
    encoder: ObsEncoder
    hand_tracker: BroadcastHandTracker
    vertex_to_idx: dict[Any, int]
    edge_to_idx: dict[tuple[str, str], int]
    records: list[_DecisionRecord]
    seat: int
    # Phase mutates as the game state machine progresses.
    env_state: EnvObsState


def _make_action(
    action_type: int,
    corner_idx: int = 0,
    edge_idx: int = 0,
    tile_idx: int = 0,
    resource1_idx: int = 0,
    resource2_idx: int = 0,
) -> np.ndarray:
    return np.array(
        [action_type, corner_idx, edge_idx, tile_idx, resource1_idx, resource2_idx],
        dtype=np.int64,
    )


def _record_decision(
    ctx: _RecorderContext,
    action: np.ndarray,
    phase: str,
) -> None:
    """Build obs + mask AT THE MOMENT of the call and append a record.

    Skips recording when the heuristic's chosen action is not actually
    legal under the current mask. This happens when:
      * ``trade_with_bank`` is called with r1 the player has < 2/3/4 of
        (it no-ops silently inside the engine);
      * ``draw_devCard`` is called when the player lacks WHEAT/ORE/SHEEP
        or the dev deck is empty (also silent no-op).
    The heuristic *calls* the method unconditionally but the action
    isn't a real decision because it has no effect. Recording it would
    corrupt BC training — we'd be supervising on actions the network
    couldn't take under the same mask at inference time. Surfaced by
    a failing TDD test on the BC loader.
    """
    ctx.env_state.initial_placement_phase = phase == "setup"
    # During setup, infer setup_step from the action type so the mask
    # accepts the chosen action. The env's state machine alternates
    # settle/road, and the agent_player's buildGraph counts tell us
    # which sub-step the heuristic is on. (We can't read it from env
    # directly — there's no env in BC data-gen path.)
    if phase == "setup":
        act_type = int(action[0])
        if act_type == ActionType.BUILD_SETTLEMENT:
            n_existing = len(ctx.agent_player.buildGraph["SETTLEMENTS"])
            ctx.env_state.setup_step = 0 if n_existing == 0 else 2
        elif act_type == ActionType.BUILD_ROAD:
            n_existing = len(ctx.agent_player.buildGraph["ROADS"])
            ctx.env_state.setup_step = 1 if n_existing == 0 else 3
    # build_obs returns ``dict[str, np.ndarray | np.int64]`` (opp_kind /
    # opp_policy_id are int64 scalars); BC stores them as-is and
    # ``_flatten_records`` np.stacks them into arrays, so treat the dict
    # as ndarray-valued at this boundary.
    obs = cast(
        dict[str, np.ndarray],
        ctx.encoder.build_obs(
            ctx.game, ctx.agent_player, ctx.opp_player, ctx.env_state, hand_tracker=ctx.hand_tracker
        ),
    )
    mask = compute_action_masks(
        ctx.game, ctx.agent_player, ctx.env_state, ctx.vertex_to_idx, ctx.edge_to_idx
    )
    if not bool(mask["type"][int(action[0])]):
        return
    belief = _belief_target_for(ctx.opp_player)
    forced = bool(int(mask["type"].sum()) <= 1)
    step_idx = len(ctx.records)
    ctx.records.append(
        _DecisionRecord(
            obs=obs,
            action=action,
            mask=mask,
            belief_target=belief,
            forced=forced,
            phase=phase,
            player_seat=ctx.seat,
            step_idx=step_idx,
        )
    )


@contextmanager
def _instrumented_player(ctx: _RecorderContext) -> Iterator[None]:
    """Patch the player's action methods to record before executing.

    Mirrors the audit-script pattern in :mod:`scripts.audit_heuristic_distribution`
    but records the full (obs, mask, belief) triple per decision rather
    than just the action-type marginals.
    """
    p = ctx.agent_player
    orig_build_settlement = p.build_settlement
    orig_build_city = p.build_city
    orig_build_road = p.build_road
    orig_draw_dev = p.draw_devCard
    orig_trade_bank = p.trade_with_bank
    orig_move_robber = p.move_robber

    def patched_build_settlement(vCoord, board, is_free=False):
        phase = "setup" if is_free else "main"
        action = _make_action(ActionType.BUILD_SETTLEMENT, corner_idx=ctx.vertex_to_idx[vCoord])
        _record_decision(ctx, action, phase)
        return orig_build_settlement(vCoord, board, is_free=is_free)

    def patched_build_city(vCoord, board):
        action = _make_action(ActionType.BUILD_CITY, corner_idx=ctx.vertex_to_idx[vCoord])
        _record_decision(ctx, action, "main")
        return orig_build_city(vCoord, board)

    def patched_build_road(v1, v2, board, is_free=False):
        phase = "setup" if is_free else "main"
        action = _make_action(ActionType.BUILD_ROAD, edge_idx=ctx.edge_to_idx[_edge_key(v1, v2)])
        _record_decision(ctx, action, phase)
        return orig_build_road(v1, v2, board, is_free=is_free)

    def patched_draw_dev(board):
        action = _make_action(ActionType.BUY_DEV_CARD)
        _record_decision(ctx, action, "main")
        return orig_draw_dev(board)

    def patched_trade_bank(r1, r2):
        action = _make_action(
            ActionType.BANK_TRADE,
            resource1_idx=RESOURCES_CW.index(r1) if r1 in RESOURCES_CW else 0,
            resource2_idx=RESOURCES_CW.index(r2) if r2 in RESOURCES_CW else 0,
        )
        _record_decision(ctx, action, "main")
        return orig_trade_bank(r1, r2)

    def patched_move_robber(hexIndex, board, player_robbed):
        action = _make_action(ActionType.MOVE_ROBBER, tile_idx=int(hexIndex))
        _record_decision(ctx, action, "robber")
        return orig_move_robber(hexIndex, board, player_robbed)

    p.build_settlement = patched_build_settlement
    p.build_city = patched_build_city
    p.build_road = patched_build_road
    p.draw_devCard = patched_draw_dev
    p.trade_with_bank = patched_trade_bank
    p.move_robber = patched_move_robber
    try:
        yield
    finally:
        p.build_settlement = orig_build_settlement
        p.build_city = orig_build_city
        p.build_road = orig_build_road
        p.draw_devCard = orig_draw_dev
        p.trade_with_bank = orig_trade_bank
        p.move_robber = orig_move_robber


# ---------------------------------------------------------------------------
# Per-game driver
# ---------------------------------------------------------------------------


def _grant_setup_resources(game: catanGame, p: Any) -> None:
    if not p.buildGraph["SETTLEMENTS"]:
        return
    last = p.buildGraph["SETTLEMENTS"][-1]
    for adj_hex in game.board.boardGraph[last].adjacent_hex_indices:
        res = game.board.hexTileDict[adj_hex].resource_type
        if res != "DESERT":
            p.resources[res] += 1
            game.broadcast.resource_change(p.name, {res: 1}, "SETUP")


def _build_players(perturbation: str, rng: np.random.Generator) -> tuple[Any, Any]:
    """Construct two players for the given perturbation mix.

    * canonical:       heur vs heur
    * epsilon_greedy:  EpsGreedy-P1 vs heur-P2
    * weight_noised:   WeightNoise-P1 vs heur-P2
    """
    if perturbation == "canonical":
        p1 = heuristicAIPlayer("P1", "black")
        p2 = heuristicAIPlayer("P2", "darkslateblue")
    elif perturbation == "epsilon_greedy":
        p1 = EpsilonGreedyHeuristicAIPlayer("P1", "black", epsilon=0.10, top_k=3, rng=rng)
        p2 = heuristicAIPlayer("P2", "darkslateblue")
    elif perturbation == "weight_noised":
        p1 = WeightNoisedHeuristicAIPlayer("P1", "black", noise_std=0.15, rng=rng)
        p2 = heuristicAIPlayer("P2", "darkslateblue")
    else:
        raise ValueError(f"unknown perturbation: {perturbation!r}")
    p1.updateAI()
    p2.updateAI()
    return p1, p2


def play_game(
    game_id: int,
    seed: int,
    perturbation: str = "canonical",
    max_turns: int = 400,
    discount: float = 0.998,
) -> _GameRecord:
    """Play one game and return a fully-populated _GameRecord."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    game = catanGame(render_mode=None)
    board = game.board

    p1, p2 = _build_players(perturbation, rng)
    p1.game = game
    p2.game = game

    game.playerQueue = queue.Queue(2)
    game.playerQueue.put(p1)
    game.playerQueue.put(p2)
    game.resource_tracker = ResourceTracker([p1.name, p2.name])

    encoder = ObsEncoder(board)
    hand_tracker = BroadcastHandTracker([p1.name, p2.name])
    hand_tracker.subscribe(game.broadcast)

    vertex_to_idx, edge_to_idx = _build_index_maps(board)

    # Shared env_state — mutated as the game progresses. Each record
    # captures a *snapshot* of phase + flags AT THE MOMENT of decision.
    env_state = EnvObsState(initial_placement_phase=True, setup_step=0)

    record = _GameRecord(game_id=game_id, perturbation=perturbation)

    ctx_p1 = _RecorderContext(
        game=game,
        agent_player=p1,
        opp_player=p2,
        encoder=encoder,
        hand_tracker=hand_tracker,
        vertex_to_idx=vertex_to_idx,
        edge_to_idx=edge_to_idx,
        records=record.decisions,
        seat=0,
        env_state=env_state,
    )
    ctx_p2 = _RecorderContext(
        game=game,
        agent_player=p2,
        opp_player=p1,
        encoder=encoder,
        hand_tracker=hand_tracker,
        vertex_to_idx=vertex_to_idx,
        edge_to_idx=edge_to_idx,
        records=record.decisions,
        seat=1,
        env_state=env_state,
    )

    with _instrumented_player(ctx_p1), _instrumented_player(ctx_p2):
        # ---- Setup ----
        env_state.initial_placement_phase = True
        env_state.setup_step = 0
        for p in (p1, p2):
            game.currentPlayer = p
            p.initial_setup(board)  # internally calls build_settlement+build_road
        env_state.setup_step = 2
        for p in (p2, p1):
            game.currentPlayer = p
            p.initial_setup(board)
            _grant_setup_resources(game, p)
        game.gameSetup = False
        env_state.initial_placement_phase = False

        # ---- Main loop ----
        while not game.gameOver and record.total_turns < max_turns:
            for p, ctx_active in ((p1, ctx_p1), (p2, ctx_p2)):
                if game.gameOver:
                    break
                game.currentPlayer = p

                # Roll-dice decision (recorded but always forced — type
                # mask is {ROLL_DICE}). Kept for completeness; filtered
                # at write time by the include_forced=False flag.
                env_state.roll_pending = True
                env_state.discard_pending = False
                env_state.robber_placement_pending = False
                env_state.road_building_roads_left = 0
                ctx_active.agent_player = p
                ctx_active.opp_player = p2 if p is p1 else p1
                _record_decision(ctx_active, _make_action(ActionType.ROLL_DICE), "roll")
                env_state.roll_pending = False

                p.updateDevCards()
                p.devCardPlayedThisTurn = False
                dice = game.rollDice()
                env_state.last_dice_roll = int(dice)

                if dice == 7:
                    # Both players may need to discard. Discards are
                    # forced (single-resource at a time, mask of held
                    # resources only); we record them as forced and
                    # they'll be filtered out by default.
                    for pp_discard, pp_ctx in (
                        (p1, ctx_p1),
                        (p2, ctx_p2),
                    ):
                        if sum(pp_discard.resources.values()) > 9:
                            env_state.discard_pending = True
                            pp_ctx.agent_player = pp_discard
                            pp_ctx.opp_player = p2 if pp_discard is p1 else p1
                            # Heuristic chooses cards itself; we record
                            # the *aggregate* action_type=DISCARD here
                            # (forced) for schema completeness. The
                            # individual per-card emissions happen via
                            # log_discard but don't go through
                            # build_*/move_*/draw_* — they don't get a
                            # patched intercept, which means we miss
                            # those records. Acceptable: they're forced.
                            _record_decision(pp_ctx, _make_action(ActionType.DISCARD), "discard")
                            pp_discard.discardResources(game)
                    env_state.discard_pending = False
                    env_state.robber_placement_pending = True
                    p.heuristic_move_robber(board)  # patched -> records
                    env_state.robber_placement_pending = False
                else:
                    game.update_playerResources(dice, p)

                if p.victoryPoints >= game.maxPoints:
                    game.gameOver = True
                    break

                # Main turn — heuristic.move() makes 0-N inline
                # decisions, each captured via the patched methods.
                p.move(board)
                game.check_longest_road(p)
                game.check_largest_army(p)
                if p.victoryPoints >= game.maxPoints:
                    game.gameOver = True
                    break

                # Implicit END_TURN.
                _record_decision(ctx_active, _make_action(ActionType.END_TURN), "main")

            record.total_turns += 1

    record.p1_won = int(p1.victoryPoints >= game.maxPoints)
    record.p2_won = int(p2.victoryPoints >= game.maxPoints)
    record.truncated = not (record.p1_won or record.p2_won)

    # Fill discounted terminal outcome z_disc per decision, counting
    # steps from terminal *in each seat's own decision stream*. The flat
    # decision list interleaves P1 and P2; if we discount by flat index
    # the value head ends up learning the wall-clock game length (a
    # privileged future signal) rather than per-seat steps-to-terminal.
    if record.p1_won:
        z_by_seat = (1.0, -1.0)
    elif record.p2_won:
        z_by_seat = (-1.0, 1.0)
    else:
        z_by_seat = (0.0, 0.0)
    steps_to_term = [0, 0]  # per-seat: decisions remaining for this seat
    for dec in reversed(record.decisions):
        seat = dec.player_seat
        dec.z_disc = (discount ** steps_to_term[seat]) * z_by_seat[seat]
        steps_to_term[seat] += 1

    return record


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


def _flatten_records(games: list[_GameRecord], include_forced: bool) -> dict[str, np.ndarray]:
    """Flatten a list of games into per-key stacked arrays for NPZ save."""
    flat_obs: dict[str, list[np.ndarray]] = {}
    flat_actions: list[np.ndarray] = []
    flat_masks: dict[str, list[np.ndarray]] = {}
    flat_belief: list[np.ndarray] = []
    flat_z: list[float] = []
    flat_game_id: list[int] = []
    flat_step_idx: list[int] = []
    flat_seat: list[int] = []
    flat_phase: list[str] = []
    flat_forced: list[bool] = []

    for game in games:
        for d in game.decisions:
            if not include_forced and d.forced:
                continue
            for k, v in d.obs.items():
                flat_obs.setdefault(k, []).append(v)
            for k, v in d.mask.items():
                flat_masks.setdefault(k, []).append(v)
            flat_actions.append(d.action)
            flat_belief.append(d.belief_target)
            flat_z.append(d.z_disc)
            flat_game_id.append(game.game_id)
            flat_step_idx.append(d.step_idx)
            flat_seat.append(d.player_seat)
            flat_phase.append(d.phase)
            flat_forced.append(d.forced)

    out: dict[str, np.ndarray] = {}
    for k, arrs in flat_obs.items():
        out[f"obs/{k}"] = np.stack(arrs)
    for k, arrs in flat_masks.items():
        out[f"mask/{k}"] = np.stack(arrs)
    out["action"] = np.stack(flat_actions) if flat_actions else np.zeros((0, 6), np.int64)
    out["belief_target"] = (
        np.stack(flat_belief) if flat_belief else np.zeros((0, N_DEV_TYPES), np.float32)
    )
    out["z_disc"] = np.asarray(flat_z, dtype=np.float32)
    out["game_id"] = np.asarray(flat_game_id, dtype=np.int64)
    out["step_idx"] = np.asarray(flat_step_idx, dtype=np.int64)
    out["player_seat"] = np.asarray(flat_seat, dtype=np.int8)
    out["phase"] = np.asarray(flat_phase, dtype=np.dtype("U10"))
    out["forced"] = np.asarray(flat_forced, dtype=bool)
    return out


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def generate_dataset(
    out_dir: Path,
    n_games: int = 30_000,
    perturb_pct: float = 0.30,
    epsilon_greedy_share_of_perturbed: float = 0.5,
    shard_size: int = 5_000,
    seed: int = 0,
    max_turns: int = 400,
    discount: float = 0.998,
    include_forced: bool = False,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Generate the BC dataset and write sharded NPZ + manifest.

    Args:
        out_dir: directory to write shards + manifest.json into.
        n_games: total games to play. Defaults to 30k per the BC plan.
        perturb_pct: fraction of games using a perturbed P1.
        epsilon_greedy_share_of_perturbed: of perturbed games, what
            fraction use ε-greedy (the rest use weight-noise).
        shard_size: how many games per shard.
        seed: master seed; per-game seeds derive deterministically.
        max_turns: per-game turn cap.
        discount: γ used to compute the discounted terminal-outcome
            label ``z_disc`` per decision.
        include_forced: if False (default), drop forced (mask-sum=1)
            pairs at write time. The BC plan §1 D4 dropped these via
            unanimous panel vote.
        progress_every: print progress every N games.

    Returns:
        The manifest dict (also written to ``out_dir/manifest.json``).
    """  # noqa: RUF002
    if not 0 <= perturb_pct <= 1.0:
        raise ValueError(f"perturb_pct must be in [0, 1]: {perturb_pct}")
    if not 0 <= epsilon_greedy_share_of_perturbed <= 1.0:
        raise ValueError(
            f"epsilon_greedy_share must be in [0, 1]: {epsilon_greedy_share_of_perturbed}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic per-game perturbation assignment.
    rng = np.random.default_rng(seed)
    game_perturbations: list[str] = []
    for _ in range(n_games):
        if rng.random() < perturb_pct:
            if rng.random() < epsilon_greedy_share_of_perturbed:
                game_perturbations.append("epsilon_greedy")
            else:
                game_perturbations.append("weight_noised")
        else:
            game_perturbations.append("canonical")

    perturbation_counts = {
        k: game_perturbations.count(k) for k in ("canonical", "epsilon_greedy", "weight_noised")
    }

    run_id = uuid.uuid4().hex[:12]
    sha = _git_sha()
    t0 = time.time()

    shards: list[dict[str, Any]] = []
    pending_games: list[_GameRecord] = []
    next_shard_idx = 0

    def _flush_shard() -> None:
        nonlocal next_shard_idx
        if not pending_games:
            return
        shard_path = out_dir / f"shard_{next_shard_idx:04d}.npz"
        flat = _flatten_records(pending_games, include_forced=include_forced)
        # numpy stub mis-binds **dict[str, ndarray] to a positional bool
        # param of savez_compressed; the call is correct at runtime. Cast
        # through Any so it typechecks under both mypy 2.x (full numpy
        # stubs) and the pre-commit hook (numpy unresolved -> Any), which
        # would otherwise flag a `# type: ignore` here as unused.
        cast(Any, np.savez_compressed)(shard_path, **flat)
        n_pairs = int(flat["action"].shape[0])
        shards.append(
            {
                "shard": shard_path.name,
                "n_games": len(pending_games),
                "game_ids": [g.game_id for g in pending_games],
                "n_pairs": n_pairs,
                "perturbations": {
                    k: sum(1 for g in pending_games if g.perturbation == k)
                    for k in ("canonical", "epsilon_greedy", "weight_noised")
                },
                "p1_wins": sum(g.p1_won for g in pending_games),
                "p2_wins": sum(g.p2_won for g in pending_games),
                "truncated": sum(g.truncated for g in pending_games),
            }
        )
        next_shard_idx += 1
        pending_games.clear()

    total_pre_filter = 0
    total_post_filter = 0
    for i in range(n_games):
        perturb = game_perturbations[i]
        rec = play_game(
            game_id=i,
            seed=seed + i,
            perturbation=perturb,
            max_turns=max_turns,
            discount=discount,
        )
        total_pre_filter += len(rec.decisions)
        total_post_filter += sum(1 for d in rec.decisions if include_forced or not d.forced)
        pending_games.append(rec)

        if len(pending_games) >= shard_size:
            _flush_shard()

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(
                f"[bc.dataset] {i + 1}/{n_games} games done in {elapsed:.1f}s "
                f"(canonical={sum(1 for g in pending_games if g.perturbation == 'canonical')}, "
                f"perturbed_in_buffer={len(pending_games)})",
                flush=True,
            )

    _flush_shard()

    manifest = {
        "run_id": run_id,
        "git_sha": sha,
        "seed": seed,
        "n_games": n_games,
        "perturb_pct": perturb_pct,
        "epsilon_greedy_share_of_perturbed": epsilon_greedy_share_of_perturbed,
        "shard_size": shard_size,
        "max_turns": max_turns,
        "discount": discount,
        "include_forced": include_forced,
        "perturbation_counts": perturbation_counts,
        "shards": shards,
        "total_decisions_pre_filter": total_pre_filter,
        "total_decisions_post_filter": total_post_filter,
        "forced_move_drop_pct": ((total_pre_filter - total_post_filter) / max(total_pre_filter, 1)),
        "wall_clock_seconds": time.time() - t0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
