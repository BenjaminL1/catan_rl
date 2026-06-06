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

import queue
import random
from collections.abc import Sequence
from typing import Any

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
)
from catan_rl.policy.obs_schema import (
    CURR_PLAYER_DIM,
    N_DEV_TYPES,
    N_OPP_KINDS,
    N_OPP_POLICY_SLOTS,
    NEXT_PLAYER_DIM,
    OPP_KIND_HEURISTIC,
    OPP_KIND_RANDOM,
    OPP_KIND_UNKNOWN,
    TILE_DIM,
)

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

# Map ``opponent_type`` string → Phase 3.6 ``opp_kind`` int. Used when a
# caller doesn't pass ``opponent_kind`` explicitly via reset options.
_OPP_TYPE_TO_KIND: dict[str, int] = {
    "random": OPP_KIND_RANDOM,
    "heuristic": OPP_KIND_HEURISTIC,
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
                "CatanEnv(engine_backend='rust') is Phase-4 scaffolding only. "
                "The RustCatanEnvAdapter exists at "
                "catan_rl.env.rust_adapter.RustCatanEnvAdapter but does not "
                "yet implement the catanGame attribute proxies the env reads. "
                "Phase 5 / 6 of the remediation plan land the proxies; until "
                "then the production training loop must use the Python engine. "
                "See docs/plans/rust_engine_actual_state.md."
            )
        self.engine_backend = engine_backend
        # Whitelist includes ``"snapshot"`` so the
        # :mod:`catan_rl.selfplay.league` sentinel can reach env
        # construction without triggering a misleading "unsupported
        # opponent_type" error from this file. The actual snapshot
        # opponent inference path lands in Phase 8 of the training-
        # infra build-out; until then constructing a snapshot env
        # raises ``NotImplementedError`` with a clear pointer.
        if opponent_type not in {"random", "heuristic", "snapshot"}:
            raise ValueError(
                f"opponent_type={opponent_type!r}; supported: "
                "'random', 'heuristic', 'snapshot' (snapshot path Phase 8+)"
            )
        if opponent_type == "snapshot":
            raise NotImplementedError(
                "opponent_type='snapshot' is reserved by "
                "catan_rl.selfplay.league but not yet wired; Phase 8 "
                "checkpoint loading lands the snapshot opponent inference path."
            )
        self.opponent_type = opponent_type
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

        if opp_type == "heuristic":
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

        # Snake draft P1→P2→P2→P1. If agent is seat 1 (P2), opponent
        # places their FIRST settle+road before the agent acts. The
        # opponent's second placement and starting resources are deferred
        # until after the agent's second setup (handled in
        # _handle_setup_step at step 3).
        if self._agent_seat == 1:
            self.opponent_player.initial_setup(board)

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
                self.game.log_discard(agent, [res_name])
                self._cards_to_discard -= 1
            if self._cards_to_discard <= 0:
                self.discard_pending = False
                if self._opp_pending_robber:
                    # Opponent rolled the 7; resume their robber + main turn.
                    self._opp_pending_robber = False
                    opponent.heuristic_move_robber(board)
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
        if action_type == ActionType.BUILD_SETTLEMENT:
            v_pixel = self._idx_to_vertex[corner_idx]
            agent.build_settlement(v_pixel, board)
            self.game.check_longest_road(agent)
        elif action_type == ActionType.BUILD_CITY:
            v_pixel = self._idx_to_vertex[corner_idx]
            agent.build_city(v_pixel, board)
        elif action_type == ActionType.BUILD_ROAD:
            v1, v2 = self._idx_to_edge[edge_idx]
            agent.build_road(v1, v2, board)
            self.game.check_longest_road(agent)
        elif action_type == ActionType.END_TURN:
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
        elif action_type == ActionType.MOVE_ROBBER:
            self._apply_robber_placement(agent, tile_idx)
        elif action_type == ActionType.BUY_DEV_CARD:
            agent.draw_devCard(board)
        elif action_type == ActionType.PLAY_KNIGHT:
            if agent.devCards.get("KNIGHT", 0) > 0 and not agent.devCardPlayedThisTurn:
                agent.devCards["KNIGHT"] -= 1
                agent.knightsPlayed += 1
                agent.devCardPlayedThisTurn = True
                self.robber_placement_pending = True
                self.game.check_largest_army(agent)
        elif action_type == ActionType.PLAY_YOP:
            if agent.devCards.get("YEAROFPLENTY", 0) > 0 and not agent.devCardPlayedThisTurn:
                agent.devCards["YEAROFPLENTY"] -= 1
                agent.yopPlayed += 1
                agent.devCardPlayedThisTurn = True
                r1, r2 = RESOURCES_CW[res1_idx], RESOURCES_CW[res2_idx]
                agent.resources[r1] += 1
                agent.resources[r2] += 1
                self.game.log_yop(agent, [r1, r2])
        elif action_type == ActionType.PLAY_MONOPOLY:
            if agent.devCards.get("MONOPOLY", 0) > 0 and not agent.devCardPlayedThisTurn:
                agent.devCards["MONOPOLY"] -= 1
                agent.monopolyPlayed += 1
                agent.devCardPlayedThisTurn = True
                r = RESOURCES_CW[res1_idx]
                total_stolen = 0
                for other in list(self.game.playerQueue.queue):
                    if other is not agent:
                        stolen = other.resources.get(r, 0)
                        if stolen > 0:
                            other.resources[r] = 0
                            agent.resources[r] += stolen
                            total_stolen += stolen
                            self.game.broadcast.resource_change(
                                other.name, {r: -stolen}, "MONOPOLY"
                            )
                            self.game.broadcast.resource_change(
                                agent.name, {r: +stolen}, "MONOPOLY"
                            )
                # Phase 0.5: emit a single structural MONOPOLY event
                # carrying the total resources transferred. The
                # per-victim ``RESOURCE_CHANGE`` events above (one
                # ``-stolen`` from each victim, one ``+stolen`` into
                # the agent) fire BEFORE this structural event. A
                # consumer that needs to group RESOURCE_CHANGE +
                # MONOPOLY into a single transaction should buffer
                # the RESOURCE_CHANGEs until MONOPOLY arrives, then
                # treat the buffered set as the monopoly play.
                # Fires even when ``total_stolen == 0`` so the viewer
                # renders the play as a step-bar marker rather than
                # swallowing the action silently.
                self.game.broadcast.monopoly(agent.name, r, total_stolen)
        elif action_type == ActionType.PLAY_ROAD_BUILDER:
            if agent.devCards.get("ROADBUILDER", 0) > 0 and not agent.devCardPlayedThisTurn:
                agent.devCards["ROADBUILDER"] -= 1
                agent.roadBuilderPlayed += 1
                agent.devCardPlayedThisTurn = True
                self.road_building_roads_left = 2
        elif action_type == ActionType.BANK_TRADE:
            r1, r2 = RESOURCES_CW[res1_idx], RESOURCES_CW[res2_idx]
            agent.trade_with_bank(r1, r2)
        elif action_type == ActionType.DISCARD:
            # Fallback path (discard_pending branch above is normally taken).
            res_name = RESOURCES_CW[res1_idx]
            if agent.resources.get(res_name, 0) > 0:
                agent.resources[res_name] -= 1
                self.game.log_discard(agent, [res_name])
        elif action_type == ActionType.ROLL_DICE:
            # Outside roll_pending: no-op; mask should prevent this.
            pass

        obs = self._get_obs()
        if terminated or truncated:
            reward = self._terminal_reward()
            info = self._terminal_info(terminated)
        return obs, reward, terminated, truncated, info

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
                opponent.initial_setup(board)
                opponent.initial_setup(board)
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
                opponent.initial_setup(board)
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
                opponent.discardResources(self.game)
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
        if dice != 7:
            self.game.update_playerResources(dice, opp)
        else:
            if sum(opp.resources.values()) > 9:
                opp.discardResources(self.game)
            if sum(agent.resources.values()) > 9:
                # Agent must discard before the opponent can place the robber.
                self._cards_to_discard = sum(agent.resources.values()) // 2
                self.discard_pending = True
                self._opp_pending_robber = True
                return
            opp.heuristic_move_robber(board)

        if opp.victoryPoints >= self.game.maxPoints:
            return

        self._run_opponent_main_turn()

    def _run_opponent_main_turn(self) -> None:
        assert self.game is not None and self.opponent_player is not None
        opp = self.opponent_player
        opp.move(self.game.board)
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

    def _compute_masks(self, acting_player: PlainPlayer | None) -> dict[str, np.ndarray]:
        """Thin wrapper around :func:`catan_rl.env.masks.compute_action_masks`.

        Builds an ``EnvObsState`` from the env's current state-machine flags
        and delegates. The standalone function lives in ``masks.py`` so the
        BC dataset generator (Step 3) can produce the same masks without
        instantiating an env.
        """
        from catan_rl.env.masks import compute_action_masks
        from catan_rl.policy.obs_encoder import EnvObsState

        assert acting_player is not None and self.game is not None
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
        return self._obs_encoder.build_obs(
            self.game,
            self.agent_player,
            self.opponent_player,
            env_state,
            hand_tracker=self._hand_tracker,
        )

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
