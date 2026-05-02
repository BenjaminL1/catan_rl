"""
CatanEnv — Gymnasium environment for 1v1 Catan RL training.

Observation: dict with keys:
  tile_representations  (19, 79)
  current_player_main   (166,)
  next_player_main      (173,)
  current_player_hidden_dev  (MAX_DEV_SEQ,) int, 0-padded
  current_player_played_dev  (MAX_DEV_SEQ,) int, 0-padded
  next_player_played_dev     (MAX_DEV_SEQ,) int, 0-padded

Action: 6-element int array [type, corner, edge, tile, resource1, resource2]
  type:       0=BuildSettlement,1=BuildCity,2=BuildRoad,3=EndTurn,
              4=MoveRobber,5=BuyDevCard,6=PlayKnight,7=PlayYoP,
              8=PlayMonopoly,9=PlayRoadBuilder,10=BankTrade,11=Discard,
              12=RollDice
  corner:     0-53 vertex index
  edge:       0-71 edge index
  tile:       0-18 hex index
  resource1:  0-4  (WOOD,BRICK,WHEAT,ORE,SHEEP)
  resource2:  0-4  same

Action masks returned by get_action_masks() — dict:
  type (13), corner_settlement (54), corner_city (54), edge (72), tile (19),
  resource1_trade (5), resource1_discard (5), resource1_default (5),
  resource2_default (5)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from catan_rl.engine.player import player
from catan_rl.engine.game import catanGame
from catan_rl.agents.heuristic import heuristicAIPlayer
from catan_rl.agents.random_ai import RandomAIPlayer
from catan_rl.env.hand_tracker import BroadcastHandTracker, RESOURCES_CW

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOURCES = ["BRICK", "ORE", "SHEEP", "WHEAT", "WOOD"]

YOP_COMBINATIONS = [
    ("BRICK", "BRICK"),
    ("BRICK", "ORE"),
    ("BRICK", "SHEEP"),
    ("BRICK", "WHEAT"),
    ("BRICK", "WOOD"),
    ("ORE", "ORE"),
    ("ORE", "SHEEP"),
    ("ORE", "WHEAT"),
    ("ORE", "WOOD"),
    ("SHEEP", "SHEEP"),
    ("SHEEP", "WHEAT"),
    ("SHEEP", "WOOD"),
    ("WHEAT", "WHEAT"),
    ("WHEAT", "WOOD"),
    ("WOOD", "WOOD"),
]

TRADE_ACTIONS = []
for _give in RESOURCES:
    for _get in RESOURCES:
        if _give != _get:
            TRADE_ACTIONS.append((_give, _get))

# Resource order used in all observations (Charlesworth order)
# RESOURCES_CW = ['WOOD', 'BRICK', 'WHEAT', 'ORE', 'SHEEP']  (imported from hand_tracker)

# Dev card IDs for embedding: 0=padding, 1=KNIGHT, 2=VP, 3=ROADBUILDER, 4=YEAROFPLENTY, 5=MONOPOLY
DEV_CARD_ORDER = ["KNIGHT", "VP", "ROADBUILDER", "YEAROFPLENTY", "MONOPOLY"]
DEV_CARD_ID = {c: i + 1 for i, c in enumerate(DEV_CARD_ORDER)}

# Observation feature dims
MAX_DEV_SEQ = 15
N_TILES = 19
TILE_DIM = 79
CURR_PLAYER_DIM = 166
NEXT_PLAYER_DIM = 173

# Number-token dot counts (out of 36 ways to roll)
DOTS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1, None: 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bucket8(v: float, max_v: float) -> np.ndarray:
    """8-threshold binary encoding: bit i=1 if v >= (i+1)*max_v/8.

    Legacy thermometer encoding used when ``use_thermometer_encoding=True``.
    Phase 1.3 introduces ``compact_scalar`` as the lower-dim alternative.
    """
    arr = np.zeros(8, dtype=np.float32)
    if max_v <= 0:
        return arr
    for i in range(8):
        if v >= (i + 1) * max_v / 8.0:
            arr[i] = 1.0
    return arr


def compact_scalar(v: float, max_v: float) -> list[float]:
    """Phase 1.3 compact replacement for ``bucket8``: a single ``v / max_v``.

    Returned as a single-element list so the caller can use the same
    ``f.extend(...)`` interface as the legacy thermometer path. Values are
    clipped to ``[0, max_v]`` then divided so the resulting feature is in
    ``[0, 1]`` matching the rest of the per-player obs dynamic range.
    """
    if max_v <= 0:
        return [0.0]
    clipped = max(0.0, min(float(v), float(max_v)))
    return [clipped / float(max_v)]


def _compute_income(p, board) -> list:
    """Expected resource income per roll per resource (RESOURCES_CW order)."""
    income = {r: 0.0 for r in RESOURCES_CW}
    for v_pixel in p.buildGraph["SETTLEMENTS"]:
        for adj_hex in board.boardGraph[v_pixel].adjacent_hex_indices:
            tile = board.hexTileDict[adj_hex]
            if tile.resource_type != "DESERT":
                income[tile.resource_type] += DOTS.get(tile.number_token, 0) / 36.0
    for v_pixel in p.buildGraph["CITIES"]:
        for adj_hex in board.boardGraph[v_pixel].adjacent_hex_indices:
            tile = board.hexTileDict[adj_hex]
            if tile.resource_type != "DESERT":
                income[tile.resource_type] += 2 * DOTS.get(tile.number_token, 0) / 36.0
    return [income[r] for r in RESOURCES_CW]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CatanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    # Phase 3.6 opponent-kind enum, kept stable so the embedding layer
    # indices don't drift across runs. Order is load-bearing (CatanPolicy
    # checkpoints store the embedding weight matrix indexed by these).
    OPP_KIND_UNKNOWN = 0
    OPP_KIND_RANDOM = 1
    OPP_KIND_HEURISTIC = 2
    OPP_KIND_SELF_LATEST = 3
    OPP_KIND_LEAGUE = 4
    OPP_KIND_MAIN_EXPLOITER = 5
    N_OPP_KINDS = 6

    def __init__(
        self,
        render_mode=None,
        opponent_type="random",
        max_turns=500,
        use_thermometer_encoding: bool = True,
        use_opponent_id_emb: bool = False,
        opp_id_mask_prob: float = 0.40,
        league_maxlen: int = 100,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.opponent_type = opponent_type
        self.max_turns = max_turns

        # Phase 1.3: choose between the legacy bucket8-thermometer player
        # encoding and the compact normalized-scalar version. Default True
        # preserves checkpoint_07390040.pt loadability; phase1_full.yaml
        # opts in to the compact encoding.
        self.use_thermometer_encoding = bool(use_thermometer_encoding)
        self._curr_player_dim = CURR_PLAYER_DIM if self.use_thermometer_encoding else 54
        self._next_player_dim = NEXT_PLAYER_DIM if self.use_thermometer_encoding else 61

        # Phase 3.6: opponent identity embedding inputs. Off by default so
        # legacy obs schema stays stable. When on, the env emits two extra
        # int scalars (``opponent_kind``, ``opponent_policy_id``) and applies
        # random masking with prob ``opp_id_mask_prob`` to the *kind* input
        # so the model stays robust to the eval-time "unknown" distribution.
        # ``league_maxlen`` bounds the policy-id embedding range (must match
        # the trainer's league capacity to avoid index errors).
        self.use_opponent_id_emb = bool(use_opponent_id_emb)
        self.opp_id_mask_prob = float(opp_id_mask_prob)
        self._league_maxlen = int(league_maxlen)
        # Current opponent's (kind, policy_id) — set by reset() options.
        self._opp_kind: int = self.OPP_KIND_UNKNOWN
        self._opp_policy_id: int = 0

        # Spaces
        self.action_space = spaces.MultiDiscrete([13, 54, 72, 19, 5, 5])
        obs_spaces: dict = {
            "tile_representations": spaces.Box(0.0, 1.0, (N_TILES, TILE_DIM), dtype=np.float32),
            "current_player_main": spaces.Box(
                -1.0, 2.0, (self._curr_player_dim,), dtype=np.float32
            ),
            "next_player_main": spaces.Box(-1.0, 2.0, (self._next_player_dim,), dtype=np.float32),
            "current_player_hidden_dev": spaces.Box(0, 5, (MAX_DEV_SEQ,), dtype=np.int32),
            "current_player_played_dev": spaces.Box(0, 5, (MAX_DEV_SEQ,), dtype=np.int32),
            "next_player_played_dev": spaces.Box(0, 5, (MAX_DEV_SEQ,), dtype=np.int32),
        }
        if self.use_opponent_id_emb:
            obs_spaces["opponent_kind"] = spaces.Discrete(self.N_OPP_KINDS)
            obs_spaces["opponent_policy_id"] = spaces.Discrete(self._league_maxlen + 1)
        self.observation_space = spaces.Dict(obs_spaces)

        # Runtime state — populated in reset()
        self.game = None
        self.agent_player = None
        self.opponent_player = None
        self._opponent_policy = None
        self._hand_tracker = None

        # Edge/vertex index maps — rebuilt each reset
        self._edge_to_idx = {}  # tuple(sorted v1_px, v2_px) → int
        self._idx_to_edge = {}  # int → (v1_px, v2_px)
        self._vertex_to_idx = {}  # pixel_coord → int
        self._idx_to_vertex = {}  # int → pixel_coord

        # Obs caches — rebuilt each reset (board layout is fixed, tile resources vary per game)
        self._tile_corners: list = None  # list[list[Point]], shape (19, 6)
        self._tile_static: np.ndarray = None  # (19, 19) float32 — resource/number/dots dims

        # State machine flags
        self.initial_placement_phase = False
        self._setup_pending = None  # (player, 'settlement'|'road')
        self._setup_step = 0  # 0-3
        self.roll_pending = False
        self.discard_pending = False
        self._cards_to_discard = 0
        self.robber_placement_pending = False
        self.road_building_roads_left = 0
        self._opp_pending_robber = False  # opponent needs to place robber after agent discards

        # Deferred opponent NN: set by _run_policy_opponent_turn(); cleared by apply_opponent_action()
        self._opp_turn_in_progress = False
        self._opp_steps_this_turn = 0
        self._pending_opp_vp_before = 0

        self.last_dice_roll = 0
        self._turn_count = 0
        self._game_over = False

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        options = options or {}
        opp_type = options.get("opponent_type", self.opponent_type)
        opp_policy = options.get("opponent_policy", None)
        human_first = options.get("human_first", False)
        self._opponent_policy = opp_policy

        # Phase 3.6: capture opponent identity for the optional embedding.
        # Caller (GameManager) passes ``opponent_kind`` and
        # ``opponent_policy_id``; we map missing values to "unknown".
        kind_str = options.get("opponent_kind")
        if kind_str is None:
            # Infer from opponent_type when explicit kind not provided.
            kind_str = {
                "random": "random",
                "heuristic": "heuristic",
                "policy": "league",
                "current_self": "self_latest",
                "main_exploiter": "main_exploiter",
            }.get(opp_type, "unknown")
        self._opp_kind = self._kind_to_int(kind_str)
        opp_id_raw = options.get("opponent_policy_id", -1)
        # Ranges: [-1, league_maxlen-1] from caller. We map -1 / out-of-range
        # to a stable "unknown" slot at index ``league_maxlen``.
        self._opp_policy_id = (
            int(opp_id_raw) if 0 <= int(opp_id_raw) < self._league_maxlen else self._league_maxlen
        )

        # Create game (no render for RL)
        self.game = catanGame(render_mode=None)
        board = self.game.board

        # Create players
        self.agent_player = player("Agent", "black")
        self.agent_player.game = self.game
        self.agent_player.isAI = False

        if opp_type == "heuristic":
            self.opponent_player = heuristicAIPlayer("Opponent", "darkslateblue")
            self.opponent_player.updateAI()
        else:
            # random or policy — use RandomAI for game-loop compatibility
            self.opponent_player = RandomAIPlayer("Opponent", "darkslateblue")
        self.opponent_player.game = self.game
        self.opponent_player.isAI = True

        if opp_type == "policy":
            self._opp_type_runtime = "policy"
        else:
            self._opp_type_runtime = opp_type

        # Build player queue: agent first (P1), opponent second (P2)
        import queue

        self.game.playerQueue = queue.Queue(2)
        if human_first:
            self.game.playerQueue.put(self.agent_player)
            self.game.playerQueue.put(self.opponent_player)
        else:
            self.game.playerQueue.put(self.agent_player)
            self.game.playerQueue.put(self.opponent_player)

        # Build vertex/edge index maps
        self._build_index_maps(board)

        # Cache tile corners (pure geometry — constant for this board layout)
        self._tile_corners = [board.hexTileDict[i].get_corners(board.flat) for i in range(N_TILES)]

        # Cache static per-tile dims: resource one-hot (0-5), number one-hot (6-16), dots (18)
        # dim 17 (has_robber) is dynamic and intentionally left 0 here; filled per-step below
        _RES_NAMES = ["BRICK", "ORE", "SHEEP", "WHEAT", "WOOD", "DESERT"]
        _NUM_MAP = {None: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10}
        self._tile_static = np.zeros((N_TILES, 19), dtype=np.float32)
        for _i in range(N_TILES):
            _tile = board.hexTileDict[_i]
            for _j, _r in enumerate(_RES_NAMES):
                self._tile_static[_i, _j] = 1.0 if _tile.resource_type == _r else 0.0
            self._tile_static[_i, 6 + _NUM_MAP.get(_tile.number_token, 0)] = 1.0
            self._tile_static[_i, 18] = DOTS.get(_tile.number_token, 0) / 5.0

        # Hand tracker
        self._hand_tracker = BroadcastHandTracker(["Agent", "Opponent"])
        self._hand_tracker.subscribe(self.game.broadcast)

        # State machine reset
        self.initial_placement_phase = True
        self._setup_step = 0
        self.roll_pending = False
        self.discard_pending = False
        self._cards_to_discard = 0
        self.robber_placement_pending = False
        self.road_building_roads_left = 0
        self._opp_pending_robber = False
        self.last_dice_roll = 0
        self._turn_count = 0
        self._game_over = False

        # Agent places first settlement (setup step 0)
        self._setup_pending = (self.agent_player, "settlement")

        obs = self._get_obs()
        info = {}
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
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
        info = {}
        # VP shaping: capture VP at step start for delta-based reward signal
        agent_vp_before = agent.victoryPoints

        # ---- Setup phase ----
        if self.initial_placement_phase:
            if self._setup_step == 0:  # agent settle 1
                v_pixel = self._idx_to_vertex[corner_idx]
                agent.build_settlement(v_pixel, board, is_free=True)
                self._setup_step = 1
                self._setup_pending = (agent, "road")
            elif self._setup_step == 1:  # agent road 1
                v1, v2 = self._idx_to_edge[edge_idx]
                agent.build_road(v1, v2, board, is_free=True)
                self._setup_step = 2
                # Opponent does full setup (settle1+road1+settle2+road2)
                self._run_opponent_setup()
                # Grant initial resources to opponent (for their 2nd settlement)
                self._grant_setup_resources(opponent)
                self._setup_pending = (agent, "settlement")
            elif self._setup_step == 2:  # agent settle 2
                v_pixel = self._idx_to_vertex[corner_idx]
                agent.build_settlement(v_pixel, board, is_free=True)
                self._setup_step = 3
                self._setup_pending = (agent, "road")
            elif self._setup_step == 3:  # agent road 2
                v1, v2 = self._idx_to_edge[edge_idx]
                agent.build_road(v1, v2, board, is_free=True)
                # Grant initial resources to agent (for their 2nd settlement)
                self._grant_setup_resources(agent)
                self._hand_tracker.seed_from_player(agent)
                self._hand_tracker.seed_from_player(opponent)
                self.initial_placement_phase = False
                self._setup_pending = None
                self._setup_step = 4
                self.game.gameSetup = False
                self.roll_pending = True
                self._turn_count = 0

            obs = self._get_obs()
            reward += 0.05 * (agent.victoryPoints - agent_vp_before)
            return obs, reward, terminated, truncated, info

        # ---- Discard phase ----
        if self.discard_pending:
            res_name = RESOURCES_CW[res1_idx]
            if agent.resources.get(res_name, 0) > 0:
                agent.resources[res_name] -= 1
                self.game.log_discard(agent, [res_name])
                self._cards_to_discard -= 1
            if self._cards_to_discard <= 0:
                self.discard_pending = False
                if self._opp_pending_robber:
                    # Opponent was mid-turn waiting for us to discard
                    self._opp_pending_robber = False
                    opponent.heuristic_move_robber(board)
                    # Continue opponent's main turn
                    self._run_opponent_main_turn()
                    terminated, truncated = self._check_terminal()
                    if not terminated and not truncated:
                        self.roll_pending = True
                        self._turn_count += 1
                else:
                    self.robber_placement_pending = True

            obs = self._get_obs()
            if terminated or truncated:
                info = self._make_terminal_info(terminated)
            return obs, reward, terminated, truncated, info

        # ---- Robber placement phase ----
        if self.robber_placement_pending:
            hex_pixel_list = list(board.hexTileDict[tile_idx].get_corners(board.flat))
            players_to_rob = board.get_players_to_rob(tile_idx)
            player_robbed = None
            for p in players_to_rob:
                if p is not agent:
                    player_robbed = p
                    break
            agent.move_robber(tile_idx, board, player_robbed)
            self.game.check_largest_army(agent)
            self.robber_placement_pending = False

            obs = self._get_obs()
            reward += 0.05 * (agent.victoryPoints - agent_vp_before)
            return obs, reward, terminated, truncated, info

        # ---- Roll pending ----
        if self.roll_pending:
            if action_type != 12:
                obs = self._get_obs()
                return obs, 0.0, False, False, {}
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
                    obs = self._get_obs()
                    return obs, 0.0, False, False, {}
                else:
                    self.robber_placement_pending = True
            else:
                self.game.update_playerResources(dice, agent)
            self.roll_pending = False
            obs = self._get_obs()
            return obs, 0.0, False, False, {}

        # ---- Road building phase ----
        if self.road_building_roads_left > 0:
            if action_type == 2:
                v1, v2 = self._idx_to_edge[edge_idx]
                agent.build_road(v1, v2, board, is_free=True)
                self.game.check_longest_road(agent)
                self.road_building_roads_left -= 1
                reward += 0.05 * (agent.victoryPoints - agent_vp_before)
                obs = self._get_obs()
                return obs, reward, False, False, {}
            else:
                # No valid roads — mask forced EndTurn; exit phase and fall through
                self.road_building_roads_left = 0

        # ---- Main turn actions ----
        if action_type == 0:  # BuildSettlement
            v_pixel = self._idx_to_vertex[corner_idx]
            agent.build_settlement(v_pixel, board)
            self.game.check_longest_road(agent)

        elif action_type == 1:  # BuildCity
            v_pixel = self._idx_to_vertex[corner_idx]
            agent.build_city(v_pixel, board)

        elif action_type == 2:  # BuildRoad
            v1, v2 = self._idx_to_edge[edge_idx]
            agent.build_road(v1, v2, board)
            self.game.check_longest_road(agent)

        elif action_type == 3:  # EndTurn
            self.game.check_longest_road(agent)
            self.game.check_largest_army(agent)
            terminated, truncated = self._check_terminal()
            if terminated or truncated:
                info = self._make_terminal_info(terminated)
                obs = self._get_obs()
                return obs, self._compute_terminal_reward(), terminated, truncated, info
            # Run opponent turn; shape reward by opponent VP gain this turn
            opp_vp_before = opponent.victoryPoints
            self._run_opponent_turn()
            if self._opp_turn_in_progress:
                # Policy opponent's NN phase is deferred: GameManager will batch it.
                # Store vp snapshot for reward shaping once the turn completes.
                self._pending_opp_vp_before = opp_vp_before
                obs = self._get_obs()
                return obs, reward, False, False, {"opp_turn_pending": True}
            reward -= 0.025 * (opponent.victoryPoints - opp_vp_before)
            terminated, truncated = self._check_terminal()
            if not terminated and not truncated:
                self.roll_pending = True
                self._turn_count += 1
            if terminated or truncated:
                info = self._make_terminal_info(terminated)

        elif action_type == 4:  # MoveRobber
            players_to_rob = board.get_players_to_rob(tile_idx)
            player_robbed = None
            for p in players_to_rob:
                if p is not agent:
                    player_robbed = p
                    break
            agent.move_robber(tile_idx, board, player_robbed)
            self.game.check_largest_army(agent)
            self.robber_placement_pending = False

        elif action_type == 5:  # BuyDevCard
            agent.draw_devCard(board)

        elif action_type == 6:  # PlayKnight
            if agent.devCards.get("KNIGHT", 0) > 0 and not agent.devCardPlayedThisTurn:
                agent.devCards["KNIGHT"] -= 1
                agent.knightsPlayed += 1
                agent.devCardPlayedThisTurn = True
                self.robber_placement_pending = True
                self.game.check_largest_army(agent)

        elif action_type == 7:  # PlayYoP
            if agent.devCards.get("YEAROFPLENTY", 0) > 0 and not agent.devCardPlayedThisTurn:
                agent.devCards["YEAROFPLENTY"] -= 1
                agent.yopPlayed += 1
                agent.devCardPlayedThisTurn = True
                r1 = RESOURCES_CW[res1_idx]
                r2 = RESOURCES_CW[res2_idx]
                agent.resources[r1] += 1
                agent.resources[r2] += 1
                self.game.log_yop(agent, [r1, r2])

        elif action_type == 8:  # PlayMonopoly
            if agent.devCards.get("MONOPOLY", 0) > 0 and not agent.devCardPlayedThisTurn:
                agent.devCards["MONOPOLY"] -= 1
                agent.monopolyPlayed += 1
                agent.devCardPlayedThisTurn = True
                r = RESOURCES_CW[res1_idx]
                for p in list(self.game.playerQueue.queue):
                    if p is not agent:
                        stolen = p.resources.get(r, 0)
                        if stolen > 0:
                            p.resources[r] = 0
                            agent.resources[r] += stolen
                            self.game.broadcast.resource_change(p.name, {r: -stolen}, "MONOPOLY")
                            self.game.broadcast.resource_change(
                                agent.name, {r: +stolen}, "MONOPOLY"
                            )

        elif action_type == 9:  # PlayRoadBuilder
            if agent.devCards.get("ROADBUILDER", 0) > 0 and not agent.devCardPlayedThisTurn:
                agent.devCards["ROADBUILDER"] -= 1
                agent.roadBuilderPlayed += 1
                agent.devCardPlayedThisTurn = True
                self.road_building_roads_left = 2

        elif action_type == 10:  # BankTrade
            r1 = RESOURCES_CW[res1_idx]
            r2 = RESOURCES_CW[res2_idx]
            agent.trade_with_bank(r1, r2)

        elif action_type == 11:  # Discard (fallback, should be handled above)
            res_name = RESOURCES_CW[res1_idx]
            if agent.resources.get(res_name, 0) > 0:
                agent.resources[res_name] -= 1
                self.game.log_discard(agent, [res_name])

        elif action_type == 12:  # RollDice (shouldn't happen in main turn)
            pass

        obs = self._get_obs()
        if terminated or truncated:
            reward = self._compute_terminal_reward()
            info = self._make_terminal_info(terminated)
        else:
            reward += 0.05 * (agent.victoryPoints - agent_vp_before)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action masks
    # ------------------------------------------------------------------

    def get_action_masks(self) -> dict:
        return self._compute_masks(self.agent_player)

    def _compute_masks(self, acting_player) -> dict:
        board = self.game.board
        p = acting_player
        is_agent = p is self.agent_player
        is_opp = not is_agent

        type_mask = np.zeros(13, dtype=bool)
        corner_set = np.zeros(54, dtype=bool)
        corner_city = np.zeros(54, dtype=bool)
        edge_mask = np.zeros(72, dtype=bool)
        tile_mask = np.zeros(19, dtype=bool)
        res1_trade = np.zeros(5, dtype=bool)
        res1_disc = np.zeros(5, dtype=bool)
        res1_def = np.zeros(5, dtype=bool)
        res2_def = np.zeros(5, dtype=bool)

        # Setup phase
        if self.initial_placement_phase:
            step = self._setup_step
            if step == 0 or step == 2:  # settle
                type_mask[0] = True
                for v_px, _ in board.get_setup_settlements(p).items():
                    idx = self._vertex_to_idx.get(v_px)
                    if idx is not None:
                        corner_set[idx] = True
            elif step == 1 or step == 3:  # road
                type_mask[2] = True
                for v1, v2 in board.get_setup_roads(p).keys():
                    key = (min(v1, v2, key=str), max(v1, v2, key=str))
                    idx = self._edge_to_idx.get(self._edge_key(v1, v2))
                    if idx is not None:
                        edge_mask[idx] = True
            return {
                "type": type_mask,
                "corner_settlement": corner_set,
                "corner_city": corner_city,
                "edge": edge_mask,
                "tile": tile_mask,
                "resource1_trade": res1_trade,
                "resource1_discard": res1_disc,
                "resource1_default": res1_def,
                "resource2_default": res2_def,
            }

        # Discard phase
        if self.discard_pending and is_agent:
            type_mask[11] = True
            for i, r in enumerate(RESOURCES_CW):
                if p.resources.get(r, 0) > 0:
                    res1_disc[i] = True
            if not res1_disc.any():
                res1_disc[:] = True  # fallback
            return {
                "type": type_mask,
                "corner_settlement": corner_set,
                "corner_city": corner_city,
                "edge": edge_mask,
                "tile": tile_mask,
                "resource1_trade": res1_trade,
                "resource1_discard": res1_disc,
                "resource1_default": res1_def,
                "resource2_default": res2_def,
            }

        # Robber placement
        if self.robber_placement_pending and is_agent:
            type_mask[4] = True
            for hex_idx in board.get_robber_spots():
                tile_mask[hex_idx] = True
            if not tile_mask.any():
                tile_mask[:] = True
            return {
                "type": type_mask,
                "corner_settlement": corner_set,
                "corner_city": corner_city,
                "edge": edge_mask,
                "tile": tile_mask,
                "resource1_trade": res1_trade,
                "resource1_discard": res1_disc,
                "resource1_default": res1_def,
                "resource2_default": res2_def,
            }

        # Roll pending
        if self.roll_pending and is_agent:
            type_mask[12] = True
            return {
                "type": type_mask,
                "corner_settlement": corner_set,
                "corner_city": corner_city,
                "edge": edge_mask,
                "tile": tile_mask,
                "resource1_trade": res1_trade,
                "resource1_discard": res1_disc,
                "resource1_default": res1_def,
                "resource2_default": res2_def,
            }

        # Road building
        if self.road_building_roads_left > 0 and is_agent:
            type_mask[2] = True
            for v1, v2 in board.get_potential_roads(p).keys():
                idx = self._edge_to_idx.get(self._edge_key(v1, v2))
                if idx is not None:
                    edge_mask[idx] = True
            if not edge_mask.any():
                # No roads to build, skip
                type_mask[2] = False
                type_mask[3] = True  # force end
            return {
                "type": type_mask,
                "corner_settlement": corner_set,
                "corner_city": corner_city,
                "edge": edge_mask,
                "tile": tile_mask,
                "resource1_trade": res1_trade,
                "resource1_discard": res1_disc,
                "resource1_default": res1_def,
                "resource2_default": res2_def,
            }

        # --- Normal main-turn actions ---
        res = p.resources

        # EndTurn always valid
        type_mask[3] = True

        # BuildSettlement
        pot_settle = board.get_potential_settlements(p)
        can_settle = (
            res.get("BRICK", 0) >= 1
            and res.get("WOOD", 0) >= 1
            and res.get("SHEEP", 0) >= 1
            and res.get("WHEAT", 0) >= 1
            and p.settlementsLeft > 0
            and pot_settle
        )
        if can_settle:
            type_mask[0] = True
            for v_px in pot_settle:
                idx = self._vertex_to_idx.get(v_px)
                if idx is not None:
                    corner_set[idx] = True

        # BuildCity
        pot_city = board.get_potential_cities(p)
        can_city = (
            res.get("ORE", 0) >= 3 and res.get("WHEAT", 0) >= 2 and p.citiesLeft > 0 and pot_city
        )
        if can_city:
            type_mask[1] = True
            for v_px in pot_city:
                idx = self._vertex_to_idx.get(v_px)
                if idx is not None:
                    corner_city[idx] = True

        # BuildRoad
        pot_roads = board.get_potential_roads(p)
        can_road = (
            res.get("BRICK", 0) >= 1 and res.get("WOOD", 0) >= 1 and p.roadsLeft > 0 and pot_roads
        )
        if can_road:
            type_mask[2] = True
            for v1, v2 in pot_roads.keys():
                idx = self._edge_to_idx.get(self._edge_key(v1, v2))
                if idx is not None:
                    edge_mask[idx] = True

        # BuyDevCard
        deck_total = sum(self.game.board.devCardStack.values())
        if (
            res.get("ORE", 0) >= 1
            and res.get("WHEAT", 0) >= 1
            and res.get("SHEEP", 0) >= 1
            and deck_total > 0
        ):
            type_mask[5] = True

        # PlayKnight
        if p.devCards.get("KNIGHT", 0) > 0 and not p.devCardPlayedThisTurn:
            type_mask[6] = True

        # PlayYoP
        if p.devCards.get("YEAROFPLENTY", 0) > 0 and not p.devCardPlayedThisTurn:
            type_mask[7] = True
            res1_def[:] = True
            res2_def[:] = True

        # PlayMonopoly
        if p.devCards.get("MONOPOLY", 0) > 0 and not p.devCardPlayedThisTurn:
            type_mask[8] = True
            res1_def[:] = True

        # PlayRoadBuilder
        if p.devCards.get("ROADBUILDER", 0) > 0 and not p.devCardPlayedThisTurn:
            type_mask[9] = True

        # BankTrade
        for i, r in enumerate(RESOURCES_CW):
            r_port = "2:1 " + r
            if r_port in p.portList and res.get(r, 0) >= 2:
                res1_trade[i] = True
            elif "3:1 PORT" in p.portList and res.get(r, 0) >= 3:
                res1_trade[i] = True
            elif res.get(r, 0) >= 4:
                res1_trade[i] = True
        if res1_trade.any():
            type_mask[10] = True
            for i, r in enumerate(RESOURCES_CW):
                res2_def[i] = True  # any resource can be received

        # Fallback: always allow EndTurn
        if not type_mask.any():
            type_mask[3] = True

        return {
            "type": type_mask,
            "corner_settlement": corner_set,
            "corner_city": corner_city,
            "edge": edge_mask,
            "tile": tile_mask,
            "resource1_trade": res1_trade,
            "resource1_discard": res1_disc,
            "resource1_default": res1_def,
            "resource2_default": res2_def,
        }

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _kind_to_int(self, kind: str) -> int:
        """Phase 3.6: map a string opponent kind to its embedding index."""
        return {
            "unknown": self.OPP_KIND_UNKNOWN,
            "random": self.OPP_KIND_RANDOM,
            "heuristic": self.OPP_KIND_HEURISTIC,
            "self_latest": self.OPP_KIND_SELF_LATEST,
            "league": self.OPP_KIND_LEAGUE,
            "main_exploiter": self.OPP_KIND_MAIN_EXPLOITER,
        }.get(kind, self.OPP_KIND_UNKNOWN)

    def _opponent_id_obs(self) -> tuple[int, int]:
        """Phase 3.6: emit (kind, policy_id), with stochastic masking.

        With probability ``opp_id_mask_prob``, both fields are set to the
        UNKNOWN sentinel — keeps the policy robust to eval-time games where
        we never reveal opponent identity (champion-bench, exploitability).
        """
        if self.np_random.random() < self.opp_id_mask_prob:
            return self.OPP_KIND_UNKNOWN, self._league_maxlen
        return self._opp_kind, self._opp_policy_id

    def _get_obs(self) -> dict:
        return self._get_obs_for_player(self.agent_player)

    def _get_obs_for_player(self, acting_player) -> dict:
        other_player = (
            self.opponent_player if acting_player is self.agent_player else self.agent_player
        )
        board = self.game.board

        tile_feats = self._build_tile_features(acting_player)
        curr_main = self._build_current_player_main(acting_player)
        next_main = self._build_next_player_main(other_player, acting_player)
        hid_dev, played_dev = self._build_dev_sequences(acting_player, hidden=True)
        _, opp_played_dev = self._build_dev_sequences(other_player, hidden=False)

        obs = {
            "tile_representations": tile_feats,
            "current_player_main": curr_main,
            "next_player_main": next_main,
            "current_player_hidden_dev": hid_dev,
            "current_player_played_dev": played_dev,
            "next_player_played_dev": opp_played_dev,
        }
        if self.use_opponent_id_emb:
            kind, pid = self._opponent_id_obs()
            obs["opponent_kind"] = np.int64(kind)
            obs["opponent_policy_id"] = np.int64(pid)
        return obs

    def _build_tile_features(self, acting_player) -> np.ndarray:
        """Build (19, 79) tile feature array."""
        board = self.game.board
        agent = acting_player

        tiles = np.zeros((N_TILES, TILE_DIM), dtype=np.float32)

        # Copy static dims 0-16 (resource + number one-hots) and dim 18 (dots) from cache
        tiles[:, :19] = self._tile_static

        for hex_idx in range(N_TILES):
            tile = board.hexTileDict[hex_idx]

            # dim 17: has_robber (dynamic)
            tiles[hex_idx, 17] = 1.0 if tile.has_robber else 0.0

            # Per-vertex features: 6 vertices × 6 = 36  (dims 19-54)
            corners = self._tile_corners[hex_idx]
            v_start = 19
            for corner_px in corners:
                v_obj = board.boardGraph.get(corner_px)
                if v_obj is None:
                    tiles[hex_idx, v_start : v_start + 6] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                else:
                    owner = v_obj.owner
                    tiles[hex_idx, v_start] = 1.0 if owner is None else 0.0
                    tiles[hex_idx, v_start + 1] = 1.0 if owner is agent else 0.0
                    tiles[hex_idx, v_start + 2] = (
                        1.0 if (owner is not None and owner is not agent) else 0.0
                    )
                    btype = v_obj.building_type
                    tiles[hex_idx, v_start + 3] = 1.0 if btype is None else 0.0
                    tiles[hex_idx, v_start + 4] = 1.0 if btype == "Settlement" else 0.0
                    tiles[hex_idx, v_start + 5] = (
                        1.0 if (btype is not None and btype != "Settlement") else 0.0
                    )
                v_start += 6

            # Per-edge features: 6 edges × 4 = 24  (dims 55-78)
            e_start = 55
            for i in range(6):
                c1 = corners[i]
                c2 = corners[(i + 1) % 6]
                v1_obj = board.boardGraph.get(c1)
                has_road = False
                road_owner = None
                if v1_obj is not None:
                    for j, nb in enumerate(v1_obj.neighbors):
                        if nb == c2:
                            has_road = v1_obj.edge_state[j][1]
                            road_owner = v1_obj.edge_state[j][0]
                            break
                tiles[hex_idx, e_start] = 1.0 if (not has_road or road_owner is None) else 0.0
                tiles[hex_idx, e_start + 1] = 1.0 if (has_road and road_owner is agent) else 0.0
                tiles[hex_idx, e_start + 2] = (
                    1.0
                    if (has_road and road_owner is not None and road_owner is not agent)
                    else 0.0
                )
                tiles[hex_idx, e_start + 3] = 1.0 if has_road else 0.0
                e_start += 4

        return tiles

    def _build_current_player_main(self, p) -> np.ndarray:
        """Build the per-player feature vector.

        Dim depends on ``use_thermometer_encoding``: 166 (legacy bucket8)
        or 53 (Phase 1.3 compact). The encoding is identical in information
        content; only the representation of integer-valued counts differs.
        """
        board = self.game.board
        f = []
        # Phase 1.3: switch between the legacy 8-threshold bucket and a
        # single normalized scalar. Module-level helpers keep both call
        # sites identical (both use ``f.extend``).
        encode = bucket8 if self.use_thermometer_encoding else compact_scalar

        # Resources in RESOURCES_CW order (5 × {8 | 1})
        for r in RESOURCES_CW:
            f.extend(encode(p.resources.get(r, 0), 8))

        # VP ({8 | 1})
        f.extend(encode(p.victoryPoints, 15))

        # Income (5 floats)
        income = _compute_income(p, board)
        f.extend(income)

        # Trade rates per resource (5 floats: best ratio as fraction of 1)
        for r in RESOURCES_CW:
            r_port = "2:1 " + r
            if r_port in p.portList:
                f.append(0.5)
            elif "3:1 PORT" in p.portList:
                f.append(1.0 / 3.0)
            else:
                f.append(0.25)

        # Ports (6 bools: 3:1, 2:1 WOOD, 2:1 BRICK, 2:1 WHEAT, 2:1 ORE, 2:1 SHEEP)
        port_names = ["3:1 PORT", "2:1 WOOD", "2:1 BRICK", "2:1 WHEAT", "2:1 ORE", "2:1 SHEEP"]
        for pn in port_names:
            f.append(1.0 if pn in p.portList else 0.0)

        # LR + LA (2 bools)
        f.append(1.0 if p.longestRoadFlag else 0.0)
        f.append(1.0 if p.largestArmyFlag else 0.0)

        # Road length ({8 | 1})
        f.extend(encode(p.maxRoadLength, 15))

        # Knights played ({8 | 1})
        f.extend(encode(p.knightsPlayed, 8))

        # Settlements left ({8 | 1})
        f.extend(encode(p.settlementsLeft, 5))

        # Cities left ({8 | 1})
        f.extend(encode(p.citiesLeft, 4))

        # Roads left (1 normalized; same in both modes)
        f.append(p.roadsLeft / 15.0)

        # Dev cards in DEV_CARD_ORDER (5 × {8 | 1})
        for c in DEV_CARD_ORDER:
            f.extend(encode(p.devCards.get(c, 0), 8))

        # Phase flags (5)
        f.append(1.0 if self.initial_placement_phase else 0.0)
        f.append(1.0 if self.roll_pending else 0.0)
        f.append(
            1.0
            if (
                not self.initial_placement_phase
                and not self.roll_pending
                and not self.discard_pending
                and not self.robber_placement_pending
                and self.road_building_roads_left == 0
            )
            else 0.0
        )
        f.append(1.0 if self.robber_placement_pending else 0.0)
        f.append(1.0 if self.discard_pending else 0.0)

        # Deck remaining ({8 | 1})
        deck_total = sum(board.devCardStack.values())
        f.extend(encode(deck_total, 25))

        # Dice roll one-hot 2-12 (11)
        dice_oh = [0.0] * 11
        if 2 <= self.last_dice_roll <= 12:
            dice_oh[self.last_dice_roll - 2] = 1.0
        f.extend(dice_oh)

        # Karma (2 bools: agent last rolled 7, opponent last rolled 7)
        last7 = self.game.last_player_to_roll_7
        f.append(1.0 if last7 is p else 0.0)
        f.append(1.0 if (last7 is not None and last7 is not p) else 0.0)

        # devCardPlayedThisTurn (1)
        f.append(1.0 if p.devCardPlayedThisTurn else 0.0)

        arr = np.array(f, dtype=np.float32)
        assert len(arr) == self._curr_player_dim, (
            f"curr_player_main dim={len(arr)} expected {self._curr_player_dim} "
            f"(use_thermometer_encoding={self.use_thermometer_encoding})"
        )
        return arr

    def _build_next_player_main(self, p, acting_player) -> np.ndarray:
        """Build the per-opponent feature vector.

        Dim depends on ``use_thermometer_encoding``: 173 (legacy) or 60
        (Phase 1.3 compact). Layout = current-player vector (166 or 53)
        + 6 hidden-dev-count one-hot + 1 total-resources/20.
        Uses ``BroadcastHandTracker`` for exact resource tracking.
        """
        # Temporarily override p.resources with tracker data for encoding
        tracked = self._hand_tracker.get_hand(p.name)
        original_res = dict(p.resources)
        p.resources = dict(tracked)

        base = self._build_current_player_main(p)

        p.resources = original_res

        # Hidden dev card count: cards in p.newDevCards + playable non-VP devCards
        hidden_count = len(p.newDevCards) + sum(
            p.devCards.get(c, 0) for c in DEV_CARD_ORDER if c != "VP"
        )
        hidden_oh = np.zeros(6, dtype=np.float32)
        hidden_oh[min(hidden_count, 5)] = 1.0

        # Total resource count normalized
        total_res = sum(tracked.values())
        total_res_norm = np.array([total_res / 20.0], dtype=np.float32)

        arr = np.concatenate([base, hidden_oh, total_res_norm])
        assert len(arr) == self._next_player_dim, (
            f"next_player_main dim={len(arr)} expected {self._next_player_dim} "
            f"(use_thermometer_encoding={self.use_thermometer_encoding})"
        )
        return arr

    def _build_dev_sequences(self, p, hidden: bool):
        """Build (MAX_DEV_SEQ,) int arrays for hidden and played dev cards."""
        # Hidden cards = new cards + playable cards
        hidden_seq = []
        for c in p.newDevCards:
            cid = DEV_CARD_ID.get(c, 0)
            if cid:
                hidden_seq.append(cid)
        for c in DEV_CARD_ORDER:
            hidden_seq.extend([DEV_CARD_ID[c]] * p.devCards.get(c, 0))

        # Played cards reconstructed from counts
        played_seq = []
        played_seq.extend([DEV_CARD_ID["KNIGHT"]] * p.knightsPlayed)
        played_seq.extend([DEV_CARD_ID["YEAROFPLENTY"]] * p.yopPlayed)
        played_seq.extend([DEV_CARD_ID["MONOPOLY"]] * p.monopolyPlayed)
        played_seq.extend([DEV_CARD_ID["ROADBUILDER"]] * p.roadBuilderPlayed)

        def pad(seq):
            arr = np.zeros(MAX_DEV_SEQ, dtype=np.int32)
            for i, v in enumerate(seq[:MAX_DEV_SEQ]):
                arr[i] = v
            return arr

        return pad(hidden_seq), pad(played_seq)

    # ------------------------------------------------------------------
    # Index maps
    # ------------------------------------------------------------------

    def _build_index_maps(self, board):
        """Build vertex and edge index maps from board state."""
        # Vertex map: vertex_index → pixel already in board
        self._vertex_to_idx = {}
        self._idx_to_vertex = {}
        for idx, px in board.vertex_index_to_pixel_dict.items():
            self._vertex_to_idx[px] = idx
            self._idx_to_vertex[idx] = px

        # Edge map: collect all unique edges
        seen = set()
        edges = []
        for v_px, v_obj in board.boardGraph.items():
            for nb_px in v_obj.neighbors:
                key = self._edge_key(v_px, nb_px)
                if key not in seen:
                    seen.add(key)
                    edges.append((v_px, nb_px))

        self._edge_to_idx = {}
        self._idx_to_edge = {}
        for i, (v1, v2) in enumerate(edges):
            key = self._edge_key(v1, v2)
            self._edge_to_idx[key] = i
            self._idx_to_edge[i] = (v1, v2)

    @staticmethod
    def _edge_key(v1, v2):
        """Canonical edge key: sorted tuple of string reps."""
        s1, s2 = str(v1), str(v2)
        return (s1, s2) if s1 < s2 else (s2, s1)

    # ------------------------------------------------------------------
    # Game helpers
    # ------------------------------------------------------------------

    def _run_opponent_setup(self):
        """Run opponent's full setup: settle1+road1+settle2+road2."""
        opp = self.opponent_player
        board = self.game.board
        # Forward: settle1 + road1
        opp.initial_setup(board)
        # Reverse: settle2 + road2
        opp.initial_setup(board)

    def _grant_setup_resources(self, p):
        """Grant resources from the player's most recently placed settlement."""
        if not p.buildGraph["SETTLEMENTS"]:
            return
        last_settle = p.buildGraph["SETTLEMENTS"][-1]
        for adj_hex in self.game.board.boardGraph[last_settle].adjacent_hex_indices:
            res_type = self.game.board.hexTileDict[adj_hex].resource_type
            if res_type != "DESERT":
                p.resources[res_type] += 1
                self.game.broadcast.resource_change(p.name, {res_type: 1}, "SETUP")

    def _run_opponent_turn(self):
        """Execute the opponent's full turn (roll → actions → end)."""
        opp = self.opponent_player
        agent = self.agent_player
        board = self.game.board

        self.game.currentPlayer = opp
        opp.updateDevCards()
        opp.devCardPlayedThisTurn = False

        # Roll
        dice = self.game.rollDice()
        if dice != 7:
            self.game.update_playerResources(dice, opp)
        else:
            # Opponent discards if needed
            if sum(opp.resources.values()) > 9:
                opp.discardResources(self.game)
            # Agent discards if needed — interrupt and return
            if sum(agent.resources.values()) > 9:
                self._cards_to_discard = sum(agent.resources.values()) // 2
                self.discard_pending = True
                self._opp_pending_robber = True
                return  # agent must discard; robber + move deferred
            # Move robber
            opp.heuristic_move_robber(board)

        # Check for win before opponent moves
        if opp.victoryPoints >= self.game.maxPoints:
            return

        # Opponent makes moves
        self._run_opponent_main_turn()

    def _run_opponent_main_turn(self):
        """Execute the opponent's build/trade/dev phase."""
        opp = self.opponent_player
        board = self.game.board

        if self._opp_type_runtime == "policy":
            self._run_policy_opponent_turn()
            if self._opp_turn_in_progress:
                return  # post-turn checks deferred to apply_opponent_action()
        else:
            opp.move(board)

        self.game.check_longest_road(opp)
        self.game.check_largest_army(opp)

    def _run_policy_opponent_turn(self):
        """Signal that opponent NN turn is deferred; GameManager will batch inference."""
        self._opp_turn_in_progress = True
        self._opp_steps_this_turn = 0

    def _execute_action_for_player(self, p, action):
        """Execute one action for any player (used in policy opponent mode)."""
        action_type = int(action[0])
        board = self.game.board

        if action_type == 0:
            v_pixel = self._idx_to_vertex.get(int(action[1]))
            if v_pixel and board.get_potential_settlements(p).get(v_pixel):
                p.build_settlement(v_pixel, board)
        elif action_type == 1:
            v_pixel = self._idx_to_vertex.get(int(action[1]))
            if v_pixel and board.get_potential_cities(p).get(v_pixel):
                p.build_city(v_pixel, board)
        elif action_type == 2:
            v1, v2 = self._idx_to_edge.get(int(action[2]), (None, None))
            if v1 is not None:
                p.build_road(v1, v2, board)
        elif action_type == 5:
            p.draw_devCard(board)
        elif action_type == 6:
            if p.devCards.get("KNIGHT", 0) > 0 and not p.devCardPlayedThisTurn:
                p.devCards["KNIGHT"] -= 1
                p.knightsPlayed += 1
                p.devCardPlayedThisTurn = True
                p.heuristic_move_robber(board)
                self.game.check_largest_army(p)
        elif action_type == 7:
            if p.devCards.get("YEAROFPLENTY", 0) > 0 and not p.devCardPlayedThisTurn:
                p.devCards["YEAROFPLENTY"] -= 1
                p.yopPlayed += 1
                p.devCardPlayedThisTurn = True
                r1 = RESOURCES_CW[int(action[4])]
                r2 = RESOURCES_CW[int(action[5])]
                p.resources[r1] += 1
                p.resources[r2] += 1
                self.game.log_yop(p, [r1, r2])
        elif action_type == 8:
            if p.devCards.get("MONOPOLY", 0) > 0 and not p.devCardPlayedThisTurn:
                p.devCards["MONOPOLY"] -= 1
                p.monopolyPlayed += 1
                p.devCardPlayedThisTurn = True
                r = RESOURCES_CW[int(action[4])]
                for other in list(self.game.playerQueue.queue):
                    if other is not p:
                        stolen = other.resources.get(r, 0)
                        if stolen > 0:
                            other.resources[r] = 0
                            p.resources[r] += stolen
        elif action_type == 10:
            r1 = RESOURCES_CW[int(action[4])]
            r2 = RESOURCES_CW[int(action[5])]
            p.trade_with_bank(r1, r2)

    # ------------------------------------------------------------------
    # Terminal conditions
    # ------------------------------------------------------------------

    def _check_terminal(self):
        agent_won = self.agent_player.victoryPoints >= self.game.maxPoints
        opp_won = self.opponent_player.victoryPoints >= self.game.maxPoints
        if agent_won or opp_won:
            self._game_over = True
            return True, False
        if self._turn_count >= self.max_turns:
            return False, True
        return False, False

    def _compute_terminal_reward(self) -> float:
        agent_vp = self.agent_player.victoryPoints
        opp_vp = self.opponent_player.victoryPoints
        if agent_vp >= self.game.maxPoints:
            return 1.0 + (agent_vp - opp_vp) / self.game.maxPoints
        elif opp_vp >= self.game.maxPoints:
            return -1.0 - (opp_vp - agent_vp) / self.game.maxPoints
        return (agent_vp - opp_vp) / self.game.maxPoints

    def _make_terminal_info(self, terminated: bool) -> dict:
        agent_vp = self.agent_player.victoryPoints
        opp_vp = self.opponent_player.victoryPoints
        is_success = terminated and agent_vp >= self.game.maxPoints
        return {
            "is_success": is_success,
            "terminal_stats": {
                "agent_vp": agent_vp,
                "opponent_vp": opp_vp,
                "game_length": self._turn_count,
            },
        }

    # ------------------------------------------------------------------
    # Deferred opponent NN interface (called by GameManager for batching)
    # ------------------------------------------------------------------

    def get_opponent_obs_masks(self):
        """Return (obs, masks) for opponent. Only valid when _opp_turn_in_progress."""
        opp = self.opponent_player
        return self._get_obs_for_player(opp), self._compute_masks(opp)

    def apply_opponent_action(self, action: np.ndarray):
        """Apply one NN-decided action for the opponent.

        Returns (turn_complete, obs, reward, terminated, truncated, info).
        When turn_complete=False the other values are meaningless — caller
        should query get_opponent_obs_masks() and call again next step.
        When turn_complete=True the full transition is ready.
        """
        action_type = int(action[0])
        self._opp_steps_this_turn += 1
        turn_ends = action_type == 3 or self._opp_steps_this_turn >= 200

        if not turn_ends:
            self._execute_action_for_player(self.opponent_player, action)
            return False, None, 0.0, False, False, {}

        # Safety: execute the last action if it wasn't EndTurn
        if action_type != 3:
            self._execute_action_for_player(self.opponent_player, action)

        # Post-turn bookkeeping (normally done by _run_opponent_main_turn)
        opp = self.opponent_player
        self.game.check_longest_road(opp)
        self.game.check_largest_army(opp)
        self._opp_turn_in_progress = False
        self._opp_steps_this_turn = 0

        reward = -0.025 * (opp.victoryPoints - self._pending_opp_vp_before)

        terminated, truncated = self._check_terminal()
        if not terminated and not truncated:
            self.roll_pending = True
            self._turn_count += 1
        if terminated or truncated:
            info = self._make_terminal_info(terminated)
            reward = self._compute_terminal_reward()
        else:
            info = {}

        obs = self._get_obs()
        return True, obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render (stub)
    # ------------------------------------------------------------------

    def render(self):
        pass

    def close(self):
        pass
