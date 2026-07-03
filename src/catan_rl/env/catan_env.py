"""CatanEnv — Gymnasium environment for 1v1 Catan.

v2 design Step 1 scope. The action space, masking, and engine-driving state
machine are ported from v1 (battle-tested through >18M PPO steps); what was
trimmed:

  * No opponent-id embedding (v1 Phase 3.6 — not in the v2 plan).
  * No belief-target obs key (v1 Phase 2.5b — re-added later if needed).
  * No hand-tracker (v1 perfect opponent hand tracker — re-add in Step 3+).
  * No thermometer-encoding flag (v1 Phase 1.3 — v2 commits to compact).
  * No policy-opponent deferred-inference plumbing (re-add in Step 4 PPO).
  * Placeholder observation: Step 1 only validates engine driving; the rich
    obs schema (TileEncoder inputs, etc.) lands in Step 2 with the network.

What is preserved verbatim:

  * MultiDiscrete([13, 54, 72, 19, 5, 5]) action space.
  * 9-key mask dict structure expected by the v2 action heads.
  * Setup state machine (snake-draft: agent settle1 -> agent road1 ->
    opponent setup1+2 -> agent settle2 + road2 + starting resources).
  * Sub-turn phases: roll_pending, discard_pending, robber_placement_pending,
    road_building_roads_left.
  * Friendly Robber (handled inside ``board.get_robber_spots``).
  * 9-card discard threshold (handled inside the engine; we just dispatch).
  * StackedDice + Karma (lives entirely inside the engine).
"""

from __future__ import annotations

import logging
import queue
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from catan_rl.agents.heuristic import heuristicAIPlayer
from catan_rl.agents.random_ai import RandomAIPlayer
from catan_rl.engine.broadcast import GameBroadcast  # re-exported for downstream
from catan_rl.engine.game import catanGame
from catan_rl.engine.player import player as PlainPlayer
from catan_rl.engine.tracker import ResourceTracker
from catan_rl.env.hand_tracker import BroadcastHandTracker
from catan_rl.policy.obs_encoder import (
    EDGE_FEATURE_DIM,
    HEX_FEATURE_DIM,
    VERTEX_FEATURE_DIM,
    EnvObsState,
    ObsEncoder,
    _dev_counts,
    hidden_belief_target,
)
from catan_rl.policy.obs_schema import (
    CURR_PLAYER_DIM,
    N_DEV_TYPES,
    N_OPP_KINDS,
    N_OPP_POLICY_SLOTS,
    NEXT_PLAYER_DIM,
    OPP_KIND_HEURISTIC,
    OPP_KIND_LEAGUE,
    OPP_KIND_RANDOM,
    OPP_KIND_SELF_LATEST,
    OPP_KIND_UNKNOWN,
    TILE_DIM,
)

if TYPE_CHECKING:
    from catan_rl.selfplay.snapshot_opponent import SnapshotOpponent

__all__ = ["CatanEnv", "RESOURCES_CW", "DEV_CARD_ORDER", "ActionType"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Resource ordering used in the action space (Charlesworth order). The
#: engine's internal ``RESOURCES`` list is alphabetical; do NOT swap them.
RESOURCES_CW: tuple[str, ...] = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")

#: Dev-card ordering used in obs dev-card sequences (0 reserved for padding).
DEV_CARD_ORDER: tuple[str, ...] = ("KNIGHT", "VP", "ROADBUILDER", "YEAROFPLENTY", "MONOPOLY")

N_VERTICES = 54
N_EDGES = 72
N_TILES = 19
N_ACTION_TYPES = 13
N_RESOURCES = 5

#: Hard cap on actions in a single snapshot-opponent main turn (livelock guard,
#: FR-013). A legal Catan turn has only a handful of distinct main actions; the
#: cap is set well above any realistic turn (build spree + multiple bank trades
#: + dev plays) so it only trips on a policy that never samples EndTurn.
_DEFAULT_OPP_TURN_ACTION_CAP = 60

# Map ``opponent_type`` string → Phase 3.6 ``opp_kind`` int. Used when a
# caller doesn't pass ``opponent_kind`` explicitly via reset options.
_OPP_TYPE_TO_KIND: dict[str, int] = {
    "random": OPP_KIND_RANDOM,
    "heuristic": OPP_KIND_HEURISTIC,
    # A frozen league snapshot (a past self) feeds the existing LEAGUE slot —
    # no embedding resize (FR-008); the concrete snapshot is identified by
    # ``opponent_policy_id`` passed via reset options (T009).
    "snapshot": OPP_KIND_LEAGUE,
}


def _kind_from_opp_type(opp_type: str) -> int:
    """Return the Phase 3.6 ``opp_kind`` integer for ``opponent_type``."""
    return _OPP_TYPE_TO_KIND.get(opp_type, OPP_KIND_UNKNOWN)


class ActionType:
    """Symbolic names for the 13 action types in the type head."""

    BUILD_SETTLEMENT = 0
    BUILD_CITY = 1
    BUILD_ROAD = 2
    END_TURN = 3
    MOVE_ROBBER = 4
    BUY_DEV_CARD = 5
    PLAY_KNIGHT = 6
    PLAY_YOP = 7
    PLAY_MONOPOLY = 8
    PLAY_ROAD_BUILDER = 9
    BANK_TRADE = 10
    DISCARD = 11
    ROLL_DICE = 12


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-turn transient state (player-local; T003)
# ---------------------------------------------------------------------------


@dataclass
class _TurnState:
    """Transient sub-turn state that a single main-phase action can set up.

    Carried *per acting player* so the shared ``_apply_main_action`` does not
    write the env's agent-centric ``self.robber_placement_pending`` /
    ``self.road_building_roads_left`` when driving the opponent — otherwise a
    snapshot opponent's knight/road-builder would clobber the agent's pending
    state. The agent's ``step`` syncs these to ``self.*`` after each action;
    the opponent turn-driver keeps its own instance.
    """

    robber_placement_pending: bool = False
    road_building_roads_left: int = 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CatanEnv(gym.Env):
    """1v1 Catan, Colonist.io ruleset, exposed as a Gymnasium environment.

    The agent always plays as player 0 (first to act). The opponent's full
    turn is run inside ``step`` whenever the agent's EndTurn fires; if the
    opponent rolls a 7 and the agent owes a discard, ``step`` returns the
    discard-pending mask instead and the next ``step`` resumes the opponent
    once the agent has finished discarding.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_type: str = "random",
        max_turns: int = 500,
        vp_margin_bonus: float = 1.0 / 15.0,
        engine_backend: str = "python",
    ) -> None:
        """
        Args:
            opponent_type: ``"random"`` or ``"heuristic"``. ``"policy"``
                support lands in Step 4 alongside the v2 PPO trainer.
            max_turns: hard cap on agent turns before truncation. v1
                empirically saw 99th-percentile game lengths around 250.
            vp_margin_bonus: per-VP terminal bonus on top of the +/-1 win
                signal. ``0.0`` reduces to a pure +/-1 outcome reward.
            engine_backend: ``"python"`` (production) or ``"rust"``
                (Phase 4 scaffolding of the Rust migration remediation
                plan). The Rust path raises ``NotImplementedError``
                with a pointer to the plan until Phase 5 / 6 land the
                adapter proxies.
        """
        super().__init__()
        if engine_backend not in {"python", "rust"}:
            raise ValueError(f"engine_backend={engine_backend!r}; supported: 'python', 'rust'")
        if engine_backend == "rust":
            raise NotImplementedError(
                "CatanEnv(engine_backend='rust') is not wired into the "
                "rollout loop. The Rust migration FROZE at Phase 4-pivot "
                "on 2026-06-06 — the catanGame attribute proxies in "
                "catan_rl.env.rust_adapter.RustCatanEnvAdapter are NOT "
                "coming. Use engine_backend='python' for training; the "
                "Rust path is available for inference / deterministic "
                "eval / MCTS via catan_engine.RustCatanEnv directly. "
                "See docs/plans/rust_engine_actual_state.md."
            )
        self.engine_backend = engine_backend
        # ``"snapshot"`` is a frozen past-self policy driving the opponent seat,
        # injected post-construction via ``set_snapshot_opponent`` (the env does
        # not own the league pool). Until one is injected, a snapshot env falls
        # back to the heuristic body (FR-011) so an empty pool never crashes.
        if opponent_type not in {"random", "heuristic", "snapshot"}:
            raise ValueError(
                f"opponent_type={opponent_type!r}; supported: 'random', 'heuristic', 'snapshot'"
            )
        self.opponent_type = opponent_type
        self._snapshot_opponent: SnapshotOpponent | None = None
        self._opp_turn_action_cap = _DEFAULT_OPP_TURN_ACTION_CAP
        self.max_turns = int(max_turns)
        self.vp_margin_bonus = float(vp_margin_bonus)

        # MultiDiscrete action: [type, corner, edge, tile, resource1, resource2].
        self.action_space = spaces.MultiDiscrete(
            [N_ACTION_TYPES, N_VERTICES, N_EDGES, N_TILES, N_RESOURCES, N_RESOURCES]
        )
        # v2 obs schema (Phase 1.5). Box bounds are loose because most
        # features are normalised but a couple are raw counts (dev cards
        # in hand can briefly hit ~7 before being played).
        self.observation_space = spaces.Dict(
            {
                "tile_representations": spaces.Box(
                    low=0.0, high=1.0, shape=(N_TILES, TILE_DIM), dtype=np.float32
                ),
                "current_player_main": spaces.Box(
                    low=0.0, high=1.0, shape=(CURR_PLAYER_DIM,), dtype=np.float32
                ),
                "next_player_main": spaces.Box(
                    low=0.0, high=1.0, shape=(NEXT_PLAYER_DIM,), dtype=np.float32
                ),
                "current_dev_counts": spaces.Box(
                    low=0.0, high=25.0, shape=(N_DEV_TYPES,), dtype=np.float32
                ),
                "next_played_dev_counts": spaces.Box(
                    low=0.0, high=25.0, shape=(N_DEV_TYPES,), dtype=np.float32
                ),
                "hex_features": spaces.Box(
                    low=0.0, high=1.0, shape=(N_TILES, HEX_FEATURE_DIM), dtype=np.float32
                ),
                "vertex_features": spaces.Box(
                    low=0.0, high=1.0, shape=(N_VERTICES, VERTEX_FEATURE_DIM), dtype=np.float32
                ),
                "edge_features": spaces.Box(
                    low=0.0, high=1.0, shape=(N_EDGES, EDGE_FEATURE_DIM), dtype=np.float32
                ),
                "opponent_kind": spaces.Discrete(N_OPP_KINDS),
                "opponent_policy_id": spaces.Discrete(N_OPP_POLICY_SLOTS),
            }
        )

        # State populated in reset().
        self.game: catanGame | None = None
        self.agent_player: PlainPlayer | None = None
        self.opponent_player: PlainPlayer | None = None
        self._vertex_to_idx: dict[Any, int] = {}
        self._idx_to_vertex: dict[int, Any] = {}
        self._edge_to_idx: dict[tuple[str, str], int] = {}
        self._idx_to_edge: dict[int, tuple[Any, Any]] = {}
        # Obs encoder + hand tracker — rebuilt per reset because the
        # resource shuffle changes per game.
        self._obs_encoder: ObsEncoder | None = None
        self._hand_tracker: BroadcastHandTracker | None = None
        # Opp-id state (Phase 3.6 plumbing — defaults to UNKNOWN; PPO
        # GameManager will pass real values via reset options later).
        self._opp_kind: int = OPP_KIND_UNKNOWN
        self._opp_policy_id: int = N_OPP_POLICY_SLOTS - 1
        self._opp_id_mask_prob: float = 0.0

        # State-machine flags (also reset in reset()).
        self.initial_placement_phase = True
        self._setup_step = 0
        self.roll_pending = False
        self.discard_pending = False
        self._cards_to_discard = 0
        self.robber_placement_pending = False
        self.road_building_roads_left = 0
        self._opp_pending_robber = False
        self._turn_count = 0
        self._game_over = False
        self.last_dice_roll = 0
        # ``agent_seat`` is the agent's position in the snake draft:
        # 0 = first (default), 1 = second. Set via reset(options=...).
        # When seat=1, the opponent makes the first setup placement before
        # the agent's first action and takes the first main turn after
        # setup completes. Used by the eval harness to symmetrise away
        # first-mover advantage.
        self._agent_seat: int = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            # StackedDice (engine/dice.py) uses the stdlib ``random``
            # module — a separate global PRNG from ``np.random``. Seed it
            # too so eval-gating decisions are reproducible across runs.
            random.seed(seed)

        options = options or {}
        opp_type = options.get("opponent_type", self.opponent_type)
        # Phase 3.6 opp-id plumbing (defaults to UNKNOWN — eval-time
        # behaviour and BC pretrain default). PPO GameManager will pass
        # real values via reset options once a league exists.
        self._opp_kind = int(options.get("opponent_kind", _kind_from_opp_type(opp_type)))
        self._opp_policy_id = int(options.get("opponent_policy_id", N_OPP_POLICY_SLOTS - 1))
        if not 0 <= self._opp_kind < N_OPP_KINDS:
            self._opp_kind = OPP_KIND_UNKNOWN
        if not 0 <= self._opp_policy_id < N_OPP_POLICY_SLOTS:
            self._opp_policy_id = N_OPP_POLICY_SLOTS - 1
        self._opp_id_mask_prob = float(options.get("opp_id_mask_prob", 0.0))
        # Snake-draft seat. 0 = agent goes first (current default
        # behaviour); 1 = opponent goes first. Anything else raises.
        agent_seat = int(options.get("agent_seat", 0))
        if agent_seat not in (0, 1):
            raise ValueError(f"agent_seat must be 0 or 1; got {agent_seat}")
        self._agent_seat = agent_seat

        self.game = catanGame(render_mode=None)
        board = self.game.board

        self.agent_player = PlainPlayer("Agent", "black")
        self.agent_player.game = self.game
        self.agent_player.isAI = False

        # "snapshot" uses a heuristic body so that, until a frozen policy is
        # injected (or if the league pool is empty), the opponent still plays
        # via the heuristic — graceful fallback (FR-011).
        if opp_type in ("heuristic", "snapshot"):
            opp = heuristicAIPlayer("Opponent", "darkslateblue")
            opp.updateAI()
        else:
            opp = RandomAIPlayer("Opponent", "darkslateblue")
        opp.game = self.game
        opp.isAI = True
        self.opponent_player = opp

        # 1v1 only. Queue order matches snake draft: position 0 acts
        # first. ``_agent_seat`` decides which player sits there.
        self.game.playerQueue = queue.Queue(2)
        if self._agent_seat == 0:
            self.game.playerQueue.put(self.agent_player)
            self.game.playerQueue.put(self.opponent_player)
        else:
            self.game.playerQueue.put(self.opponent_player)
            self.game.playerQueue.put(self.agent_player)
        self.game.resource_tracker = ResourceTracker([self.agent_player.name, opp.name])

        # Obs encoder + hand tracker. Both built after the engine so the
        # encoder's static caches match this game's board. Subscribe the
        # tracker BEFORE setup runs so all setup resource_change events
        # land in it naturally (no need to seed from player.resources).
        self._obs_encoder = ObsEncoder(board)
        self._hand_tracker = BroadcastHandTracker(
            [self.agent_player.name, self.opponent_player.name]
        )
        self._hand_tracker.subscribe(self.game.broadcast)

        self._build_index_maps(board)

        self.initial_placement_phase = True
        self._setup_step = 0
        self.roll_pending = False
        self.discard_pending = False
        self._cards_to_discard = 0
        self.robber_placement_pending = False
        self.road_building_roads_left = 0
        self._opp_pending_robber = False
        self._turn_count = 0
        self._game_over = False
        self.last_dice_roll = 0

        # Restart the snapshot opponent's action stream with a FRESH per-game
        # seed drawn from the env RNG — otherwise every game replays the same
        # seed sequence, seed-locking the opponent's sampling and collapsing
        # self-play opponent diversity. Reproducible: seeded by the env seed.
        if self._snapshot_opponent is not None:
            self._snapshot_opponent.reset_rng(seed=int(self.np_random.integers(0, 2**31 - 1)))

        # Snake draft P1→P2→P2→P1. If agent is seat 1 (P2), opponent
        # places their FIRST settle+road before the agent acts. The
        # opponent's second placement and starting resources are deferred
        # until after the agent's second setup (handled in
        # _handle_setup_step at step 3).
        if self._agent_seat == 1:
            self._opponent_setup_placement()

        return self._get_obs(), {}

    def step(
        self, action: Sequence[int] | np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.game is not None and self.agent_player is not None
        assert self.opponent_player is not None

        action_type = int(action[0])
        corner_idx = int(action[1])
        edge_idx = int(action[2])
        tile_idx = int(action[3])
        res1_idx = int(action[4])
        res2_idx = int(action[5])

        board = self.game.board
        agent = self.agent_player
        opponent = self.opponent_player

        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        # ---------------------- Setup phase ----------------------
        if self.initial_placement_phase:
            return self._handle_setup_step(action_type, corner_idx, edge_idx)

        # ---------------------- Discard ----------------------
        if self.discard_pending:
            res_name = RESOURCES_CW[res1_idx]
            if agent.resources.get(res_name, 0) > 0:
                agent.resources[res_name] -= 1
                self.game.board.bank_recirculate({res_name: 1})  # spec 009: discard -> bank
                self.game.log_discard(agent, [res_name])
                self._cards_to_discard -= 1
            if self._cards_to_discard <= 0:
                self.discard_pending = False
                if self._opp_pending_robber:
                    # Opponent rolled the 7; resume their robber + main turn.
                    self._opp_pending_robber = False
                    self._opponent_move_robber()
                    self._run_opponent_main_turn()
                    terminated, truncated = self._check_terminal()
                    if not terminated and not truncated:
                        self.roll_pending = True
                        self._turn_count += 1
                else:
                    # Agent rolled the 7; place robber next.
                    self.robber_placement_pending = True
            obs = self._get_obs()
            if terminated or truncated:
                reward = self._terminal_reward()
                info = self._terminal_info(terminated)
            return obs, reward, terminated, truncated, info

        # ---------------------- Robber placement ----------------------
        if self.robber_placement_pending:
            self._apply_robber_placement(agent, tile_idx)
            self.robber_placement_pending = False
            return self._get_obs(), 0.0, False, False, info

        # ---------------------- Roll dice ----------------------
        if self.roll_pending:
            if action_type != ActionType.ROLL_DICE:
                # Mask should prevent this; treat as no-op for safety.
                return self._get_obs(), 0.0, False, False, info
            self._do_roll_for_agent()
            return self._get_obs(), 0.0, False, False, info

        # ---------------------- Road-builder cleanup ----------------------
        if self.road_building_roads_left > 0:
            if action_type == ActionType.BUILD_ROAD:
                v1, v2 = self._idx_to_edge[edge_idx]
                agent.build_road(v1, v2, board, is_free=True)
                self.game.check_longest_road(agent)
                self.road_building_roads_left -= 1
                return self._get_obs(), 0.0, False, False, info
            # No legal road -> mask should force EndTurn; fall through to main.
            self.road_building_roads_left = 0

        # ---------------------- Main turn ----------------------
        if action_type == ActionType.END_TURN:
            # END_TURN carries turn-control side effects (run the opponent
            # turn, advance the turn counter) that are NOT part of the shared
            # apply path — they stay here in the agent's step loop.
            self.game.check_longest_road(agent)
            self.game.check_largest_army(agent)
            terminated, truncated = self._check_terminal()
            if terminated or truncated:
                info = self._terminal_info(terminated)
                return self._get_obs(), self._terminal_reward(), terminated, truncated, info
            self._run_opponent_turn()
            terminated, truncated = self._check_terminal()
            if not terminated and not truncated and not self.discard_pending:
                self.roll_pending = True
                self._turn_count += 1
            if terminated or truncated:
                info = self._terminal_info(terminated)
                reward = self._terminal_reward()
                return self._get_obs(), reward, terminated, truncated, info
        else:
            # All non-END_TURN main-phase actions go through the shared apply
            # path (T003), used by both the agent here and the snapshot
            # opponent's turn-driver. Follow-on sub-turn state (robber after a
            # knight, free roads after road-builder) round-trips through a
            # player-local _TurnState so it never clobbers the other seat.
            ts = _TurnState(
                robber_placement_pending=self.robber_placement_pending,
                road_building_roads_left=self.road_building_roads_left,
            )
            self._apply_main_action(
                agent,
                action_type=action_type,
                corner_idx=corner_idx,
                edge_idx=edge_idx,
                tile_idx=tile_idx,
                res1_idx=res1_idx,
                res2_idx=res2_idx,
                ts=ts,
            )
            self.robber_placement_pending = ts.robber_placement_pending
            self.road_building_roads_left = ts.road_building_roads_left

        obs = self._get_obs()
        if terminated or truncated:
            reward = self._terminal_reward()
            info = self._terminal_info(terminated)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Shared main-phase action application (T003)
    # ------------------------------------------------------------------

    def _apply_main_action(
        self,
        player: Any,
        *,
        action_type: int,
        corner_idx: int,
        edge_idx: int,
        tile_idx: int,
        res1_idx: int,
        res2_idx: int,
        ts: _TurnState,
    ) -> None:
        """Apply ONE main-phase action's effects for ``player``.

        The single shared rule-application path for both the agent (via
        ``step``) and the snapshot opponent's turn-driver, so game-rule
        application lives in exactly one place (Constitution II — no rules
        drift). Follow-on sub-turn triggers (robber placement after a knight,
        free roads after road-builder) are written into the player-local
        ``ts`` rather than the env's agent-centric ``self.*`` flags.

        Does NOT handle ROLL_DICE, discard resolution, robber-placement
        *resolution*, or END_TURN's turn-control transition — those stay
        caller-managed (the agent's ``step`` sub-phases / the opponent driver).
        """
        assert self.game is not None
        board = self.game.board
        if action_type == ActionType.BUILD_SETTLEMENT:
            player.build_settlement(self._idx_to_vertex[corner_idx], board)
            self.game.check_longest_road(player)
        elif action_type == ActionType.BUILD_CITY:
            player.build_city(self._idx_to_vertex[corner_idx], board)
        elif action_type == ActionType.BUILD_ROAD:
            v1, v2 = self._idx_to_edge[edge_idx]
            player.build_road(v1, v2, board)
            self.game.check_longest_road(player)
        elif action_type == ActionType.MOVE_ROBBER:
            self._apply_robber_placement(player, tile_idx)
        elif action_type == ActionType.BUY_DEV_CARD:
            player.draw_devCard(board)
        elif action_type == ActionType.PLAY_KNIGHT:
            if player.devCards.get("KNIGHT", 0) > 0 and not player.devCardPlayedThisTurn:
                player.devCards["KNIGHT"] -= 1
                player.knightsPlayed += 1
                player.devCardPlayedThisTurn = True
                ts.robber_placement_pending = True
                self.game.check_largest_army(player)
        elif action_type == ActionType.PLAY_YOP:
            if player.devCards.get("YEAROFPLENTY", 0) > 0 and not player.devCardPlayedThisTurn:
                player.devCards["YEAROFPLENTY"] -= 1
                player.yopPlayed += 1
                player.devCardPlayedThisTurn = True
                r1, r2 = RESOURCES_CW[res1_idx], RESOURCES_CW[res2_idx]
                # spec 009: draw each YoP card from the finite bank; a pick the
                # bank cannot supply is not granted (apply-time gate, matches TS).
                granted = []
                for r in (r1, r2):
                    if board.resourceBank.get(r, 0) > 0:
                        player.resources[r] += 1
                        board.bank_draw({r: 1})
                        granted.append(r)
                self.game.log_yop(player, granted)
        elif action_type == ActionType.PLAY_MONOPOLY:
            if player.devCards.get("MONOPOLY", 0) > 0 and not player.devCardPlayedThisTurn:
                player.devCards["MONOPOLY"] -= 1
                player.monopolyPlayed += 1
                player.devCardPlayedThisTurn = True
                r = RESOURCES_CW[res1_idx]
                total_stolen = 0
                for other in list(self.game.playerQueue.queue):
                    if other is not player:
                        stolen = other.resources.get(r, 0)
                        if stolen > 0:
                            other.resources[r] = 0
                            player.resources[r] += stolen
                            total_stolen += stolen
                            self.game.broadcast.resource_change(
                                other.name, {r: -stolen}, "MONOPOLY"
                            )
                            self.game.broadcast.resource_change(
                                player.name, {r: +stolen}, "MONOPOLY"
                            )
                # Phase 0.5: emit a single structural MONOPOLY event carrying
                # the total transferred. The per-victim RESOURCE_CHANGE events
                # above fire BEFORE this structural event; a consumer that
                # groups them should buffer RESOURCE_CHANGEs until MONOPOLY
                # arrives. Fires even when total_stolen == 0 so the viewer
                # renders a step-bar marker rather than swallowing the action.
                self.game.broadcast.monopoly(player.name, r, total_stolen)
        elif action_type == ActionType.PLAY_ROAD_BUILDER:
            if player.devCards.get("ROADBUILDER", 0) > 0 and not player.devCardPlayedThisTurn:
                player.devCards["ROADBUILDER"] -= 1
                player.roadBuilderPlayed += 1
                player.devCardPlayedThisTurn = True
                ts.road_building_roads_left = 2
        elif action_type == ActionType.BANK_TRADE:
            r1, r2 = RESOURCES_CW[res1_idx], RESOURCES_CW[res2_idx]
            player.trade_with_bank(r1, r2, board)
        elif action_type == ActionType.DISCARD:
            # Fallback path (discard_pending branch is normally taken).
            res_name = RESOURCES_CW[res1_idx]
            if player.resources.get(res_name, 0) > 0:
                player.resources[res_name] -= 1
                board.bank_recirculate({res_name: 1})  # spec 009: discard -> bank
                self.game.log_discard(player, [res_name])
        # ROLL_DICE outside roll_pending is a no-op (mask should prevent it).

    # ------------------------------------------------------------------
    # Setup-phase state machine
    # ------------------------------------------------------------------

    def _handle_setup_step(
        self, action_type: int, corner_idx: int, edge_idx: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.game is not None and self.agent_player is not None
        assert self.opponent_player is not None

        agent = self.agent_player
        opponent = self.opponent_player
        board = self.game.board

        if self._setup_step == 0:  # agent settle 1
            v_pixel = self._idx_to_vertex[corner_idx]
            agent.build_settlement(v_pixel, board, is_free=True)
            self._setup_step = 1
        elif self._setup_step == 1:  # agent road 1
            v1, v2 = self._idx_to_edge[edge_idx]
            agent.build_road(v1, v2, board, is_free=True)
            self._setup_step = 2
            if self._agent_seat == 0:
                # Snake middle: P2 places BOTH (settle+road)x2 in a row.
                # Resources are granted from P2's second settlement.
                self._opponent_setup_placement()
                self._opponent_setup_placement()
                self._grant_setup_resources(opponent)
            # agent_seat=1: opponent's second setup is deferred to step 3.
        elif self._setup_step == 2:  # agent settle 2
            v_pixel = self._idx_to_vertex[corner_idx]
            agent.build_settlement(v_pixel, board, is_free=True)
            self._setup_step = 3
        elif self._setup_step == 3:  # agent road 2
            v1, v2 = self._idx_to_edge[edge_idx]
            agent.build_road(v1, v2, board, is_free=True)
            self._grant_setup_resources(agent)
            if self._agent_seat == 1:
                # Snake tail: opponent (P1) now places their second
                # settle+road. Resources are granted from this second
                # settlement.
                self._opponent_setup_placement()
                self._grant_setup_resources(opponent)
            self.initial_placement_phase = False
            self.game.gameSetup = False
            self._setup_step = 4
            # Boundary marker for the replay recorder. Fires once per
            # game, after all setup placements + grants, BEFORE any
            # main-phase logic (seat=1 then runs opp's first main
            # turn inside this same env.step).
            self.game.broadcast.setup_complete()
            if self._agent_seat == 1:
                # Opponent is snake-P1; they take the first main turn.
                self._run_opponent_turn()
                terminated, truncated = self._check_terminal()
                if terminated or truncated:
                    info = self._terminal_info(terminated)
                    return (
                        self._get_obs(),
                        self._terminal_reward(),
                        terminated,
                        truncated,
                        info,
                    )
                if not self.discard_pending:
                    self.roll_pending = True
                    self._turn_count += 1
            else:
                self.roll_pending = True
        return self._get_obs(), 0.0, False, False, {}

    def _grant_setup_resources(self, p: PlainPlayer) -> None:
        """Grant the starting resources from a player's *second* settlement."""
        assert self.game is not None
        if not p.buildGraph["SETTLEMENTS"]:
            return
        last_settle = p.buildGraph["SETTLEMENTS"][-1]
        for adj_hex in self.game.board.boardGraph[last_settle].adjacent_hex_indices:
            res_type = self.game.board.hexTileDict[adj_hex].resource_type
            if res_type != "DESERT":
                p.resources[res_type] += 1
                # spec 009: setup 2nd-settlement grant draws from the finite bank.
                self.game.board.bank_draw({res_type: 1})
                self.game.broadcast.resource_change(p.name, {res_type: 1}, "SETUP")

    # ------------------------------------------------------------------
    # Roll + opponent turn
    # ------------------------------------------------------------------

    def _do_roll_for_agent(self) -> None:
        assert self.game is not None and self.agent_player is not None
        assert self.opponent_player is not None
        agent = self.agent_player
        opponent = self.opponent_player
        board = self.game.board

        agent.updateDevCards()
        agent.devCardPlayedThisTurn = False
        self.game.currentPlayer = agent
        dice = self.game.rollDice()
        self.last_dice_roll = dice
        if dice == 7:
            if sum(opponent.resources.values()) > 9:
                self._opponent_discard()
            if sum(agent.resources.values()) > 9:
                self._cards_to_discard = sum(agent.resources.values()) // 2
                self.discard_pending = True
                self.roll_pending = False
                return
            self.robber_placement_pending = True
        else:
            self.game.update_playerResources(dice, agent)
        self.roll_pending = False

    def _apply_robber_placement(self, p: PlainPlayer, tile_idx: int) -> None:
        assert self.game is not None
        board = self.game.board
        players_to_rob = board.get_players_to_rob(tile_idx)
        player_robbed = None
        for victim in players_to_rob:
            if victim is not p:
                player_robbed = victim
                break
        p.move_robber(tile_idx, board, player_robbed)
        self.game.check_largest_army(p)

    def _run_opponent_turn(self) -> None:
        assert self.game is not None and self.opponent_player is not None
        assert self.agent_player is not None
        opp = self.opponent_player
        agent = self.agent_player
        board = self.game.board

        self.game.currentPlayer = opp
        opp.updateDevCards()
        opp.devCardPlayedThisTurn = False

        dice = self.game.rollDice()
        # Keep the obs dice scalar faithful on the opponent's turn too (the
        # snapshot's POV must see ITS roll, not the agent's stale one).
        self.last_dice_roll = dice
        if dice != 7:
            self.game.update_playerResources(dice, opp)
        else:
            if sum(opp.resources.values()) > 9:
                self._opponent_discard()
            if sum(agent.resources.values()) > 9:
                # Agent must discard before the opponent can place the robber.
                self._cards_to_discard = sum(agent.resources.values()) // 2
                self.discard_pending = True
                self._opp_pending_robber = True
                return
            self._opponent_move_robber()

        if opp.victoryPoints >= self.game.maxPoints:
            return

        self._run_opponent_main_turn()

    def _run_opponent_main_turn(self) -> None:
        assert self.game is not None and self.opponent_player is not None
        if self._snapshot_opponent is not None:
            self._drive_snapshot_main_turn()
            return
        opp = self.opponent_player
        opp.move(self.game.board)
        self.game.check_longest_road(opp)
        self.game.check_largest_army(opp)

    # ------------------------------------------------------------------
    # Snapshot-opponent driver (US1 / T015-T018, full-game scope)
    # ------------------------------------------------------------------

    def set_snapshot_opponent(self, opponent: SnapshotOpponent | None) -> None:
        """Inject the frozen policy that drives the opponent seat.

        ``None`` (empty league / evicted snapshot) leaves the heuristic body in
        charge — graceful fallback (FR-011). The injected opponent drives BOTH
        the snake-draft setup and the main turns.
        """
        self._snapshot_opponent = opponent

    @property
    def has_snapshot_opponent(self) -> bool:
        return self._snapshot_opponent is not None

    def _opponent_env_state(
        self,
        *,
        roll_pending: bool = False,
        discard_pending: bool = False,
        robber_placement_pending: bool = False,
        road_building_roads_left: int = 0,
        initial_placement_phase: bool = False,
        setup_step: int = 0,
    ) -> EnvObsState:
        """Build the OPPONENT-LOCAL ``EnvObsState`` (T005/T015).

        Carries the opponent's OWN sub-turn phase (never the agent's ``self.*``
        flags). Per the Phase-2 review, the opp-id fields describe the AGENT
        from the opponent's POV — the learner is ``OPP_KIND_SELF_LATEST``.
        """
        return EnvObsState(
            initial_placement_phase=initial_placement_phase,
            setup_step=setup_step,
            roll_pending=roll_pending,
            discard_pending=discard_pending,
            robber_placement_pending=robber_placement_pending,
            road_building_roads_left=road_building_roads_left,
            last_dice_roll=self.last_dice_roll,
            opp_kind=OPP_KIND_SELF_LATEST,
            opp_policy_id=N_OPP_POLICY_SLOTS - 1,
        )

    def _opponent_discard(self) -> None:
        """Opponent's forced 7-roll discard.

        Snapshot-driven when a frozen policy is injected — the opponent uses its
        OWN learned Discard head (one card per sample, masked to held resources),
        exactly like the agent's discard sub-phase — so the self-play opponent
        plays its discards as well as it learned to, not via the heuristic.
        Falls back to the heuristic ``discardResources`` when no snapshot is set
        (empty-pool fallback / non-self-play opponents).
        """
        assert self.opponent_player is not None and self.game is not None
        opp = self.opponent_player
        if self._snapshot_opponent is None:
            opp.discardResources(self.game)
            return
        n_to_discard = sum(opp.resources.values()) // 2
        discarded = 0
        # Safety cap: the discard mask only offers held resources, so each
        # sample should succeed, but bound the loop against a degenerate mask.
        for _ in range(max(1, n_to_discard) * 4):
            if discarded >= n_to_discard:
                break
            action = self._sample_snapshot_action(self._opponent_env_state(discard_pending=True))
            res_name = RESOURCES_CW[int(action[4])]
            if opp.resources.get(res_name, 0) > 0:
                opp.resources[res_name] -= 1
                self.game.board.bank_recirculate({res_name: 1})  # spec 009: discard -> bank
                self.game.log_discard(opp, [res_name])
                discarded += 1

    def _sample_snapshot_action(self, env_state: EnvObsState) -> np.ndarray:
        """Build the opponent-POV obs + masks for ``env_state`` and sample the
        frozen snapshot. Returns a length-6 numpy action vector."""
        # Lazy import keeps torch off the module-level engine path (headless).
        from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

        assert self._snapshot_opponent is not None
        assert self.opponent_player is not None and self.agent_player is not None
        masks = self._compute_masks(self.opponent_player, env_state)
        obs = self._build_obs_for(self.opponent_player, self.agent_player, env_state)
        device = self._snapshot_opponent.device
        obs_t = obs_to_torch(obs, device, add_batch=True)
        masks_t = masks_to_torch(masks, device, add_batch=True)
        action_t = self._snapshot_opponent.sample(obs_t, masks_t)
        return action_t[0].detach().cpu().numpy().astype(np.int64)

    def _opponent_setup_placement(self) -> None:
        """Place ONE opponent settlement + adjacent road for the snake draft.

        Snapshot-driven (settlement then road, via the setup masks) when a
        frozen policy is injected; otherwise the heuristic body's
        ``initial_setup`` — so self-play openings are symmetric (full-game
        scope) and the empty-pool fallback still plays (FR-011).
        """
        assert self.opponent_player is not None and self.game is not None
        opp = self.opponent_player
        board = self.game.board
        if self._snapshot_opponent is None:
            opp.initial_setup(board)
            return
        s_action = self._sample_snapshot_action(
            self._opponent_env_state(initial_placement_phase=True, setup_step=0)
        )
        opp.build_settlement(self._idx_to_vertex[int(s_action[1])], board, is_free=True)
        r_action = self._sample_snapshot_action(
            self._opponent_env_state(initial_placement_phase=True, setup_step=1)
        )
        v1, v2 = self._idx_to_edge[int(r_action[2])]
        opp.build_road(v1, v2, board, is_free=True)

    def _opponent_move_robber(self) -> None:
        """Opponent robber placement — snapshot-driven if present, else heuristic."""
        assert self.opponent_player is not None and self.game is not None
        if self._snapshot_opponent is None:
            self.opponent_player.heuristic_move_robber(self.game.board)
            return
        action = self._sample_snapshot_action(
            self._opponent_env_state(robber_placement_pending=True)
        )
        self._apply_robber_placement(self.opponent_player, int(action[3]))

    def _drive_snapshot_main_turn(self) -> None:
        """Drive the opponent's full main turn via the frozen policy (T015).

        Mirrors the agent's ``step`` sub-phase machine for the opponent: the
        knight→robber-placement and road-builder free-road sub-phases are
        resolved before the next main action (the masks reflect them via the
        opponent-local ``EnvObsState``), and a hard action cap (T016, FR-013)
        force-ends a turn whose policy never samples EndTurn.
        """
        assert self.opponent_player is not None and self.game is not None
        opp = self.opponent_player
        board = self.game.board
        ts = _TurnState()
        for _ in range(self._opp_turn_action_cap):
            if ts.robber_placement_pending:
                # Single snapshot-robber path (also used by the 7-roll resume).
                self._opponent_move_robber()
                ts.robber_placement_pending = False
                continue
            if ts.road_building_roads_left > 0:
                action = self._sample_snapshot_action(
                    self._opponent_env_state(road_building_roads_left=ts.road_building_roads_left)
                )
                if int(action[0]) == ActionType.BUILD_ROAD:
                    v1, v2 = self._idx_to_edge[int(action[2])]
                    opp.build_road(v1, v2, board, is_free=True)
                    self.game.check_longest_road(opp)
                    ts.road_building_roads_left -= 1
                    continue
                # Policy abandoned free roads early — mirror the agent's machine:
                # drop the road-builder phase and apply this sampled action as a
                # normal main action (do NOT discard it).
                ts.road_building_roads_left = 0
            else:
                action = self._sample_snapshot_action(self._opponent_env_state())
            if int(action[0]) == ActionType.END_TURN:
                break
            self._apply_main_action(
                opp,
                action_type=int(action[0]),
                corner_idx=int(action[1]),
                edge_idx=int(action[2]),
                tile_idx=int(action[3]),
                res1_idx=int(action[4]),
                res2_idx=int(action[5]),
                ts=ts,
            )
        else:
            logger.warning(
                "snapshot opponent turn hit the %d-action cap; forcing EndTurn",
                self._opp_turn_action_cap,
            )
        self.game.check_longest_road(opp)
        self.game.check_largest_army(opp)

    # ------------------------------------------------------------------
    # Terminal conditions + reward
    # ------------------------------------------------------------------

    def _check_terminal(self) -> tuple[bool, bool]:
        assert self.game is not None and self.agent_player is not None
        assert self.opponent_player is not None
        if (
            self.agent_player.victoryPoints >= self.game.maxPoints
            or self.opponent_player.victoryPoints >= self.game.maxPoints
        ):
            # Fire GAME_END exactly once even if _check_terminal is
            # called repeatedly — the env returns ``True, False`` and
            # the rollout loop short-circuits, so this branch only
            # executes once per game in practice. The ``_game_over``
            # guard pins that contract.
            if not self._game_over:
                winner = (
                    self.agent_player
                    if self.agent_player.victoryPoints >= self.game.maxPoints
                    else self.opponent_player
                )
                self.game.broadcast.game_end(
                    winner.name,
                    vp_breakdown={
                        self.agent_player.name: int(self.agent_player.victoryPoints),
                        self.opponent_player.name: int(self.opponent_player.victoryPoints),
                    },
                )
            self._game_over = True
            return True, False
        if self._turn_count >= self.max_turns:
            return False, True
        return False, False

    def _terminal_reward(self) -> float:
        assert self.game is not None and self.agent_player is not None
        assert self.opponent_player is not None
        a_vp = self.agent_player.victoryPoints
        o_vp = self.opponent_player.victoryPoints
        margin = (a_vp - o_vp) * self.vp_margin_bonus
        if a_vp >= self.game.maxPoints:
            return 1.0 + margin
        if o_vp >= self.game.maxPoints:
            return -1.0 + margin  # margin is negative when opp wins
        return margin

    def _terminal_info(self, terminated: bool) -> dict[str, Any]:
        assert self.game is not None and self.agent_player is not None
        assert self.opponent_player is not None
        a_vp = self.agent_player.victoryPoints
        o_vp = self.opponent_player.victoryPoints
        return {
            "is_success": terminated and a_vp >= self.game.maxPoints,
            "terminal_stats": {
                "agent_vp": a_vp,
                "opponent_vp": o_vp,
                "game_length": self._turn_count,
            },
        }

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    def get_action_masks(self) -> dict[str, np.ndarray]:
        return self._compute_masks(self.agent_player)

    def _compute_masks(
        self,
        acting_player: PlainPlayer | None,
        env_state: EnvObsState | None = None,
    ) -> dict[str, np.ndarray]:
        """Thin wrapper around :func:`catan_rl.env.masks.compute_action_masks`.

        When ``env_state`` is None, builds one from the env's current
        (agent-centric) state-machine flags — the agent path. The snapshot
        opponent's turn-driver passes an opponent-local ``env_state`` (T005)
        so the opponent's masks reflect ITS sub-turn phase, not the agent's.
        The standalone function lives in ``masks.py`` so the BC dataset
        generator can produce the same masks without instantiating an env.
        """
        from catan_rl.env.masks import compute_action_masks

        assert acting_player is not None and self.game is not None
        if env_state is None:
            env_state = EnvObsState(
                initial_placement_phase=self.initial_placement_phase,
                setup_step=self._setup_step,
                roll_pending=self.roll_pending,
                discard_pending=self.discard_pending,
                robber_placement_pending=self.robber_placement_pending,
                road_building_roads_left=self.road_building_roads_left,
                last_dice_roll=self.last_dice_roll,
            )
        return compute_action_masks(
            self.game,
            acting_player,
            env_state,
            self._vertex_to_idx,
            self._edge_to_idx,
        )

    # ------------------------------------------------------------------
    # Observations (Phase 1.5: v2 schema via ObsEncoder + BroadcastHandTracker)
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, np.ndarray]:
        assert self._obs_encoder is not None and self.game is not None
        assert self.agent_player is not None and self.opponent_player is not None
        env_state = EnvObsState(
            initial_placement_phase=self.initial_placement_phase,
            setup_step=self._setup_step,
            roll_pending=self.roll_pending,
            discard_pending=self.discard_pending,
            robber_placement_pending=self.robber_placement_pending,
            road_building_roads_left=self.road_building_roads_left,
            last_dice_roll=self.last_dice_roll,
            opp_kind=self._opp_kind,
            opp_policy_id=self._opp_policy_id,
            # Stochastic opp-id mask: roll once per obs so within-step calls
            # are consistent. Phase 3.6 default of 0.0 -> never mask.
            opp_id_masked=(
                self._opp_id_mask_prob > 0.0 and bool(np.random.random() < self._opp_id_mask_prob)
            ),
        )
        return self._build_obs_for(self.agent_player, self.opponent_player, env_state)

    def _build_obs_for(
        self,
        acting_player: Any,
        other_player: Any,
        env_state: EnvObsState,
    ) -> dict[str, np.ndarray]:
        """Build the obs dict from ``acting_player``'s point of view (T005).

        ``build_obs`` derives the entire POV from its (agent_player,
        opponent_player) argument order: the first player sees its OWN hidden
        dev cards (``current_dev_counts``) and own resources; the second
        contributes only observable info (played dev cards only, tracker-
        visible resources). The agent path passes (agent, opponent); the
        snapshot opponent's turn-driver passes (opponent, agent) with an
        opponent-local ``env_state`` — so the opponent sees its own hand and
        NONE of the agent's hidden information (FR-012, no leak).
        """
        assert self._obs_encoder is not None and self.game is not None
        return self._obs_encoder.build_obs(
            self.game,
            acting_player,
            other_player,
            env_state,
            hand_tracker=self._hand_tracker,
        )

    def belief_target(self) -> np.ndarray:
        """Ground-truth target for the belief head (aux loss).

        The OPPONENT's HIDDEN (bought-but-unplayed) dev-card composition over
        the four SECRET buyable types (KNIGHT, ROADBUILDER, YEAROFPLENTY,
        MONOPOLY) as a probability vector, or all-zeros when the opponent holds
        none of them (the masked case — most of the early game).

        VP is deliberately EXCLUDED: a VP card is observable in 1v1 because
        drawing one raises ``victoryPoints`` while ``visibleVictoryPoints =
        victoryPoints - devCards['VP']`` stays put, so the hidden-VP count is
        recoverable (and the obs's hidden-dev-count one-hot likewise excludes
        VP). Making the head predict an observable quantity would let it read
        its own answer; we keep the target to the genuinely-hidden TYPE split.

        This is a TRAINING-ONLY signal — the information the agent cannot
        observe (the obs carries the opponent's non-VP hidden *count* and
        *played* cards, never the hidden *types*). It must NEVER enter the obs.
        The head still emits ``N_DEV_TYPES`` logits; the VP target is always 0,
        so it learns to put no mass there.
        """
        assert self.opponent_player is not None
        return hidden_belief_target(self.opponent_player)

    # ------------------------------------------------------------------
    # Vertex / edge index maps
    # ------------------------------------------------------------------

    def _build_index_maps(self, board: Any) -> None:
        self._vertex_to_idx = {}
        self._idx_to_vertex = {}
        for idx, px in board.vertex_index_to_pixel_dict.items():
            self._vertex_to_idx[px] = idx
            self._idx_to_vertex[idx] = px

        seen: set[tuple[str, str]] = set()
        edges: list[tuple[Any, Any]] = []
        for v_px, v_obj in board.boardGraph.items():
            for nb_px in v_obj.neighbors:
                key = self._edge_key(v_px, nb_px)
                if key not in seen:
                    seen.add(key)
                    edges.append((v_px, nb_px))

        self._edge_to_idx = {}
        self._idx_to_edge = {}
        for i, (v1, v2) in enumerate(edges):
            self._edge_to_idx[self._edge_key(v1, v2)] = i
            self._idx_to_edge[i] = (v1, v2)

    @staticmethod
    def _edge_key(v1: Any, v2: Any) -> tuple[str, str]:
        s1, s2 = str(v1), str(v2)
        return (s1, s2) if s1 < s2 else (s2, s1)


# Silence "unused import" complaints for re-exports.
_ = GameBroadcast
