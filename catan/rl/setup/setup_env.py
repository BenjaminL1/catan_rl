"""
Setup-phase Gymnasium environment for decoupled setup training.

The agent makes exactly 4 sequential decisions per episode:
  Step 0: Place settlement 1   → choose vertex index (0-53)
  Step 1: Place road 1         → choose edge index (0-71)
  Step 2: Place settlement 2   → choose vertex index (0-53)
  Step 3: Place road 2         → choose edge index (0-71)

Snake-draft order (agent = Player 1):
  Agent settle1 → Agent road1 →
  [Opponent settle1 + road1 + settle2 + road2 — automatic heuristic] →
  Agent settle2 → Agent road2

After all 4 agent decisions, K full-game heuristic rollouts are run
to estimate the value of the starting position.

Reward = mean normalized margin of victory across K rollouts:
  margin = (agent_vp - opp_vp) / max_vp, in [-1, 1]

Implementation note: deepcopy is avoided because queue.Queue contains
thread locks that cannot be pickled. Instead, we save the numpy RNG state
before board creation so the same board can be reproduced for each rollout.
The agent's setup decisions are stored as pixel coordinates and replayed.

Observation: 1417-dim flat float32 vector:
  - tile_features   19 × 8  = 152
  - vertex_features 54 × 18 = 972
  - edge_features   72 × 4  = 288
  - meta            5       =   5
    Total                   1417

Action space: Discrete(126)
  - 0-53:   vertex placement (active at settle steps 0, 2)
  - 54-125: edge placement, edge_idx = action - 54 (active at road steps 1, 3)
"""
import queue
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any

from catan.engine.game import catanGame
from catan.agents.heuristic import heuristicAIPlayer

# ── Observation dimensions ─────────────────────────────────────────────────────
RESOURCES_ORDER = ['WOOD', 'BRICK', 'WHEAT', 'ORE', 'SHEEP', 'DESERT']
N_TILES      = 19
N_VERTICES   = 54
N_EDGES      = 72
N_RESOURCES  = 5   # playable resources (excluding DESERT)

TILE_FEAT_DIM    = 8    # 6 resource one-hot + number/12 + pip/5
VERTEX_FEAT_DIM  = 18   # 5 ownership + 7 port + 5 pip income + 1 is_valid
EDGE_FEAT_DIM    = 4    # 3 ownership one-hot + 1 is_valid
META_DIM         = 5    # 4 step one-hot + 1 is_first_player

OBS_DIM = (N_TILES * TILE_FEAT_DIM
           + N_VERTICES * VERTEX_FEAT_DIM
           + N_EDGES * EDGE_FEAT_DIM
           + META_DIM)  # = 1417

# Pip expectation per dice number (outcomes out of 36)
PIP_MAP = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


class SetupEnv(gym.Env):
    """Gymnasium environment for the Catan setup phase.

    Each episode covers 4 agent decisions (2 settlements + 2 roads).
    After the episode ends, K full-game heuristic rollouts estimate
    the starting position's value. This reward signal trains the
    setup model independently from the main-game model.

    Parameters
    ----------
    n_rollouts : int
        Number of full-game rollouts used to estimate position value.
        Recommended range: 20–50. Start at 20; raise to 50 if noisy.
    max_game_turns : int
        Safety cap on game length per rollout.
    """

    metadata = {"render_modes": [None]}

    def __init__(self, n_rollouts: int = 20, max_game_turns: int = 500):
        super().__init__()
        self.n_rollouts    = n_rollouts
        self.max_game_turns = max_game_turns

        # Action: Discrete(126). See module docstring for layout.
        self.action_space = spaces.Discrete(N_VERTICES + N_EDGES)

        # Observation: flat float32 vector
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Build canonical edge list once from a temporary board (pixel coord pairs)
        self._edge_list: List[Tuple] = _build_edge_list()

        # Runtime state (populated in reset())
        self.game: Optional[catanGame]              = None
        self.agent_player: Optional[heuristicAIPlayer] = None
        self.opponent_player: Optional[heuristicAIPlayer] = None
        self._setup_step: int                       = 0   # 0-3

        # State saved for rollout replay
        self._rng_state: Optional[Any]     = None   # numpy RNG before board creation
        self._agent_decisions: List        = []     # [(type, pixel_or_edge), ...]
        # type: 'settle' → (px,), 'road' → (v1_px, v2_px)

    # ── Gymnasium interface ────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Save RNG state BEFORE board creation so rollouts reproduce same board
        self._rng_state = np.random.get_state()

        # Fresh game (randomises board using np.random)
        self.game = catanGame(render_mode=None)
        self._setup_step    = 0
        self._agent_decisions = []

        # Both players are heuristicAIPlayer so rollout replay works uniformly.
        self.agent_player = heuristicAIPlayer("Agent", "black")
        self.agent_player.updateAI()
        self.agent_player.game = self.game

        self.opponent_player = heuristicAIPlayer("Opponent", "darkslateblue")
        self.opponent_player.updateAI()
        self.opponent_player.game = self.game

        # Agent is always P1 (first in queue) → places first
        with self.game.playerQueue.mutex:
            self.game.playerQueue.queue.clear()
        self.game.playerQueue.put(self.agent_player)
        self.game.playerQueue.put(self.opponent_player)
        self.game.currentPlayer = self.agent_player
        self.game.gameSetup = True

        return self._get_obs(), {}

    def step(self, action: int):
        """Execute one setup decision.

        At settlement steps (0, 2): action is a vertex index (0-53).
        At road steps (1, 3):       action is an edge action (54-125),
                                    edge_idx = action - 54.

        After the final decision (step 3), runs K game rollouts and
        returns the average margin of victory as the reward.
        """
        reward   = 0.0
        done     = False
        info: Dict = {}

        step = self._setup_step

        if step == 0:
            # Place settlement 1 for agent
            vertex_idx = int(action)
            pixel = self.game.board.vertex_index_to_pixel_dict[vertex_idx]
            self.agent_player.build_settlement(pixel, self.game.board, is_free=True)
            self._agent_decisions.append(('settle', pixel))
            self._setup_step = 1

        elif step == 1:
            # Place road 1 for agent
            edge_idx = int(action) - N_VERTICES
            v1_px, v2_px = self._edge_list[edge_idx]
            self.agent_player.build_road(v1_px, v2_px, self.game.board, is_free=True)
            self._agent_decisions.append(('road', (v1_px, v2_px)))

            # Opponent does full forward + reverse setup (settle1+road1+settle2+road2)
            self.opponent_player.setupResources = []
            self.opponent_player.initial_setup(self.game.board)  # settle1 + road1
            self.opponent_player.initial_setup(self.game.board)  # settle2 + road2

            self._setup_step = 2

        elif step == 2:
            # Place settlement 2 for agent
            vertex_idx = int(action)
            pixel = self.game.board.vertex_index_to_pixel_dict[vertex_idx]
            self.agent_player.build_settlement(pixel, self.game.board, is_free=True)
            self._agent_decisions.append(('settle', pixel))
            self._setup_step = 3

        elif step == 3:
            # Place road 2 for agent
            edge_idx = int(action) - N_VERTICES
            v1_px, v2_px = self._edge_list[edge_idx]
            self.agent_player.build_road(v1_px, v2_px, self.game.board, is_free=True)
            self._agent_decisions.append(('road', (v1_px, v2_px)))

            # Grant starting resources from the 2nd settlement (snake-draft rule)
            _grant_starting_resources(self.game, self.agent_player)
            _grant_starting_resources(self.game, self.opponent_player)
            self.game.gameSetup = False

            # Run K rollouts to evaluate the position
            margin = self._evaluate_position()
            reward = float(margin)
            done   = True
            info["margin"]     = margin
            info["is_success"] = margin > 0.0

        obs = self._get_obs()
        return obs, reward, done, False, info

    def get_action_masks(self) -> np.ndarray:
        """Return a (126,) boolean mask of valid actions for the current step."""
        mask = np.zeros(N_VERTICES + N_EDGES, dtype=bool)
        step = self._setup_step

        if step == 0 or step == 2:
            valid = self.game.board.get_setup_settlements(self.agent_player)
            for px_coord in valid.keys():
                v_idx = self.game.board.boardGraph[px_coord].vertex_index
                if 0 <= v_idx < N_VERTICES:
                    mask[v_idx] = True

        elif step == 1 or step == 3:
            valid = self.game.board.get_setup_roads(self.agent_player)
            for (v1_px, v2_px) in valid.keys():
                e = tuple(sorted((
                    self.game.board.boardGraph[v1_px].vertex_index,
                    self.game.board.boardGraph[v2_px].vertex_index,
                )))
                if e in self._edge_index_map:
                    mask[N_VERTICES + self._edge_index_map[e]] = True

        # Fallback: if no valid action found, allow all (shouldn't happen)
        if not mask.any():
            mask[:] = True
        return mask

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the 1417-dim flat observation vector."""
        obs   = []
        board = self.game.board
        step  = self._setup_step

        # ── 1. Tile features (19 × 8 = 152) ──────────────────────────────────
        for tile_idx in range(N_TILES):
            tile = board.hexTileDict.get(tile_idx)
            res  = tile.resource_type if tile else 'DESERT'
            num  = tile.number_token  if tile else None
            pip  = PIP_MAP.get(num, 0) if num else 0

            for r in RESOURCES_ORDER:
                obs.append(1.0 if res == r else 0.0)
            obs.append((num / 12.0) if num else 0.0)
            obs.append(pip / 5.0)

        # ── Per-vertex pip income lookup (built once per obs call) ────────────
        vertex_pip: Dict[int, Dict[str, float]] = {}
        for tile_idx in range(N_TILES):
            tile = board.hexTileDict.get(tile_idx)
            if tile is None or tile.resource_type == 'DESERT':
                continue
            pip = PIP_MAP.get(tile.number_token, 0) / 5.0
            for px in tile.get_corners(board.flat):
                vtx = board.boardGraph.get(px)
                if vtx is None:
                    continue
                vi = vtx.vertex_index
                if vi not in vertex_pip:
                    vertex_pip[vi] = {r: 0.0 for r in RESOURCES_ORDER[:N_RESOURCES]}
                if tile.resource_type in vertex_pip[vi]:
                    vertex_pip[vi][tile.resource_type] = min(
                        vertex_pip[vi][tile.resource_type] + pip, 1.0
                    )

        # Valid settle vertices for current step
        valid_settle = (
            board.get_setup_settlements(self.agent_player)
            if step in (0, 2) else {}
        )

        # ── 2. Vertex features (54 × 18 = 972) ───────────────────────────────
        for v_idx in range(N_VERTICES):
            px  = board.vertex_index_to_pixel_dict[v_idx]
            vtx = board.boardGraph[px]

            owner = vtx.owner
            btype = vtx.building_type

            # Ownership one-hot (5): none / self-settle / self-city / opp-settle / opp-city
            ownership = [0.0] * 5
            if owner is None:
                ownership[0] = 1.0
            elif owner is self.agent_player:
                ownership[1 if btype == 'Settlement' else 2] = 1.0
            else:
                ownership[3 if btype == 'Settlement' else 4] = 1.0
            obs.extend(ownership)

            # Port type one-hot (7): no-port / 3:1 / 2:1 WOOD / BRICK / WHEAT / ORE / SHEEP
            port_vec = [0.0] * 7
            port = vtx.port
            if port is None or port == '':
                port_vec[0] = 1.0
            elif '3:1' in str(port):
                port_vec[1] = 1.0
            else:
                port_res_map = {'WOOD': 2, 'BRICK': 3, 'WHEAT': 4, 'ORE': 5, 'SHEEP': 6}
                found = False
                for rname, idx in port_res_map.items():
                    if rname in str(port):
                        port_vec[idx] = 1.0
                        found = True
                        break
                if not found:
                    port_vec[0] = 1.0
            obs.extend(port_vec)

            # Expected pip income per resource (5 floats)
            pip_income = vertex_pip.get(v_idx, {})
            for rname in RESOURCES_ORDER[:N_RESOURCES]:
                obs.append(pip_income.get(rname, 0.0))

            # Is valid settlement placement (1 float)
            obs.append(1.0 if px in valid_settle else 0.0)

        # Valid road edges for current step
        valid_roads_raw = {}
        if step in (1, 3) and self.agent_player.buildGraph['SETTLEMENTS']:
            valid_roads_raw = board.get_setup_roads(self.agent_player)

        # ── 3. Edge features (72 × 4 = 288) ──────────────────────────────────
        for e_idx, (v1_px, v2_px) in enumerate(self._edge_list):
            self_road = _has_road(self.agent_player, v1_px, v2_px)
            opp_road  = _has_road(self.opponent_player, v1_px, v2_px)

            obs.append(1.0 if self_road else 0.0)
            obs.append(1.0 if opp_road  else 0.0)
            obs.append(0.0 if (self_road or opp_road) else 1.0)

            is_valid = 1.0 if (
                (v1_px, v2_px) in valid_roads_raw or
                (v2_px, v1_px) in valid_roads_raw
            ) else 0.0
            obs.append(is_valid)

        # ── 4. Meta (5) ───────────────────────────────────────────────────────
        step_onehot = [0.0] * 4
        step_onehot[min(step, 3)] = 1.0
        obs.extend(step_onehot)
        obs.append(1.0)   # agent is always P1

        arr = np.array(obs, dtype=np.float32)
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return arr

    # ── Rollout evaluation ─────────────────────────────────────────────────────

    def _evaluate_position(self) -> float:
        """Run n_rollouts heuristic games and return the mean margin."""
        if self._rng_state is None or not self._agent_decisions:
            return 0.0

        max_vp  = self.game.maxPoints
        margins = []

        for _ in range(self.n_rollouts):
            margin = _run_rollout_from_decisions(
                rng_state        = self._rng_state,
                agent_decisions  = self._agent_decisions,
                agent_name       = "Agent",
                opp_name         = "Opponent",
                max_vp           = max_vp,
                max_turns        = self.max_game_turns,
            )
            margins.append(margin)

        return float(np.mean(margins))

    # ── Cached edge index lookup ───────────────────────────────────────────────

    @property
    def _edge_index_map(self) -> Dict:
        """Cached {(v_idx_small, v_idx_large): edge_idx} lookup."""
        if not hasattr(self, '_eim_cache'):
            # Build using a fresh game's boardGraph to get vertex indices
            # (all Catan boards share the same topology)
            tmp = catanGame(render_mode=None)
            board = tmp.board
            eim = {}
            for e_idx, (v1_px, v2_px) in enumerate(self._edge_list):
                i1 = board.boardGraph[v1_px].vertex_index
                i2 = board.boardGraph[v2_px].vertex_index
                eim[tuple(sorted((i1, i2)))] = e_idx
            self._eim_cache = eim
        return self._eim_cache


# ── Module-level helpers ───────────────────────────────────────────────────────

def _build_edge_list() -> List[Tuple]:
    """Build the canonical 72-edge list as pixel-coord pairs from a temp board."""
    tmp   = catanGame(render_mode=None)
    board = tmp.board

    # Collect unique edges as (vertex_idx_low, vertex_idx_high)
    edges: set = set()
    for v_idx in range(N_VERTICES):
        px  = board.vertex_index_to_pixel_dict[v_idx]
        vtx = board.boardGraph[px]
        for nb_px in vtx.neighbors:   # nb_px is a pixel coord (boardGraph key)
            nb_idx = board.boardGraph[nb_px].vertex_index
            edges.add(tuple(sorted((v_idx, nb_idx))))

    # Sort for deterministic ordering, then map back to pixel coords
    edge_idx_pairs = sorted(edges)
    pixel_list = []
    for (i1, i2) in edge_idx_pairs:
        px1 = board.vertex_index_to_pixel_dict[i1]
        px2 = board.vertex_index_to_pixel_dict[i2]
        pixel_list.append((px1, px2))
    return pixel_list


def _grant_starting_resources(game: catanGame, p) -> None:
    """Grant resources adjacent to the second settlement (snake-draft rule)."""
    settlements = p.buildGraph['SETTLEMENTS']
    if len(settlements) < 2:
        return
    last_coord = settlements[-1]
    vertex = game.board.boardGraph[last_coord]
    for hex_idx in vertex.adjacent_hex_indices:
        tile = game.board.hexTileDict[hex_idx]
        if tile.resource_type != 'DESERT':
            p.resources[tile.resource_type] += 1


def _has_road(p, v1_px, v2_px) -> bool:
    """Return True if player p has a road on the given edge."""
    roads = p.buildGraph.get('ROADS', [])
    return (v1_px, v2_px) in roads or (v2_px, v1_px) in roads


def _run_rollout_from_decisions(
    rng_state: Any,
    agent_decisions: List,
    agent_name: str,
    opp_name: str,
    max_vp: int = 15,
    max_turns: int = 500,
) -> float:
    """Create a fresh game from saved RNG state, replay agent decisions,
    run opponent heuristic setup, then play the main game with heuristics.

    Returns: normalised margin (agent_vp - opp_vp) / max_vp in [-1, 1].
    """
    # Restore numpy RNG to state before board creation → same board layout
    np.random.set_state(rng_state)
    game = catanGame(render_mode=None)

    # Create fresh heuristic players
    agent = heuristicAIPlayer(agent_name, "black")
    agent.updateAI()
    agent.game = game

    opp = heuristicAIPlayer(opp_name, "darkslateblue")
    opp.updateAI()
    opp.game = game

    with game.playerQueue.mutex:
        game.playerQueue.queue.clear()
    game.playerQueue.put(agent)
    game.playerQueue.put(opp)
    game.currentPlayer = agent
    game.gameSetup = True

    # Replay agent's setup decisions in the same order they were taken
    # Decisions list format: [('settle', px), ('road', (v1, v2)), ...]
    settle_count = 0
    road_count   = 0

    for dec_type, data in agent_decisions:
        if dec_type == 'settle':
            agent.build_settlement(data, game.board, is_free=True)
            settle_count += 1
        elif dec_type == 'road':
            v1_px, v2_px = data
            agent.build_road(v1_px, v2_px, game.board, is_free=True)
            road_count += 1

            # After agent's first road (road_count==1), opponent does all 4 setup moves
            if road_count == 1:
                opp.setupResources = []
                opp.initial_setup(game.board)  # opp settle1 + road1
                opp.initial_setup(game.board)  # opp settle2 + road2

    # Grant starting resources from 2nd settlements
    _grant_starting_resources(game, agent)
    _grant_starting_resources(game, opp)
    game.gameSetup = False
    game.gameOver  = False

    # Run main game: both players use heuristic
    players = list(game.playerQueue.queue)
    for _turn in range(max_turns):
        for curr in players:
            curr.updateDevCards()
            curr.devCardPlayedThisTurn = False

            dice = game.rollDice()
            game.update_playerResources(dice, curr)

            if hasattr(curr, 'move'):
                curr.move(game.board)

            game.check_longest_road(curr)
            game.check_largest_army(curr)

            if curr.victoryPoints >= max_vp:
                game.gameOver = True
                break

        if game.gameOver:
            break

    return (agent.victoryPoints - opp.victoryPoints) / max_vp
