from catan.agents.heuristic import heuristicAIPlayer
from catan.engine.player import player
from catan.engine.game import catanGame
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os
import queue

# Add current directory to path so we can import catan modules
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

YOP_COMBINATIONS = [
    ('BRICK', 'BRICK'), ('BRICK', 'ORE'), ('BRICK', 'SHEEP'), ('BRICK', 'WHEAT'), ('BRICK', 'WOOD'),
    ('ORE', 'ORE'), ('ORE', 'SHEEP'), ('ORE', 'WHEAT'), ('ORE', 'WOOD'),
    ('SHEEP', 'SHEEP'), ('SHEEP', 'WHEAT'), ('SHEEP', 'WOOD'),
    ('WHEAT', 'WHEAT'), ('WHEAT', 'WOOD'),
    ('WOOD', 'WOOD')
]  # Indices 202 to 216

RESOURCES = ['BRICK', 'ORE', 'SHEEP', 'WHEAT', 'WOOD']
TRADE_ACTIONS = []
for give in RESOURCES:
    for get in RESOURCES:
        if give != get:
            TRADE_ACTIONS.append((give, get))
# Indices 226-245 (20 actions)


class RandomAIPlayer(player):
    def __init__(self, name, color):
        super().__init__(name, color)
        self.isAI = True

    def initial_setup(self, board):
        # Build random settlement
        possibleVertices = board.get_setup_settlements(self)
        if possibleVertices:
            vertexToBuild = list(possibleVertices.keys())[np.random.randint(len(possibleVertices))]
            self.build_settlement(vertexToBuild, board)

        # Build random road
        possibleRoads = board.get_setup_roads(self)
        if possibleRoads:
            roadToBuild = list(possibleRoads.keys())[np.random.randint(len(possibleRoads))]
            self.build_road(roadToBuild[0], roadToBuild[1], board)

    def heuristic_move_robber(self, board):
        # Randomly move robber
        potential_spots = board.get_robber_spots()
        if not potential_spots:
            return

        hex_idx = list(potential_spots.keys())[np.random.randint(len(potential_spots))]

        # Randomly choose player to rob
        players_to_rob = board.get_players_to_rob(hex_idx)
        player_robbed = None
        if players_to_rob:
            # Filter out self if present (though usually you can't rob yourself, but good to be safe)
            opponents = [p for p in players_to_rob.keys() if p != self]
            if opponents:
                player_robbed = opponents[np.random.randint(len(opponents))]

        self.move_robber(hex_idx, board, player_robbed)

    def move(self, board):
        # 1. Randomly try to build/buy Dev Card (30% chance)
        if np.random.random() < 0.3:
            self.draw_devCard(board)
        # 2. Randomly try to build City
        if self.resources['ORE'] >= 3 and self.resources['WHEAT'] >= 2:
            potential_cities = board.get_potential_cities(self)
            if potential_cities:
                v_pixel = list(potential_cities.keys())[np.random.randint(len(potential_cities))]
                self.build_city(v_pixel, board)
        # 3. Randomly try to build Settlement
        if self.resources['BRICK'] >= 1 and self.resources['WOOD'] >= 1 and \
           self.resources['SHEEP'] >= 1 and self.resources['WHEAT'] >= 1:
            potential_settlements = board.get_potential_settlements(self)
            if potential_settlements:
                v_pixel = list(potential_settlements.keys())[
                    np.random.randint(len(potential_settlements))]
                self.build_settlement(v_pixel, board)
        # 4. Randomly try to build Road
        if self.resources['BRICK'] >= 1 and self.resources['WOOD'] >= 1:
            potential_roads = board.get_potential_roads(self)
            if potential_roads:
                r_coords = list(potential_roads.keys())[np.random.randint(len(potential_roads))]
                self.build_road(r_coords[0], r_coords[1], board)

    def discardResources(self, game):
        # Random discard logic
        total = sum(self.resources.values())
        if total > 7:
            num_discard = total // 2
            for _ in range(num_discard):
                available = [r for r, c in self.resources.items() if c > 0]
                if available:
                    res = available[np.random.randint(len(available))]
                    self.resources[res] -= 1


class CatanEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.game = None

        # Define Action Space
        # 0-53: Build Settlement
        # 54-107: Build City
        # 108-179: Build Road
        # 180: End Turn
        # 181-199: Move Robber to Hex 0-18
        # 200: Buy Dev Card
        # 201: Play Knight
        # 202-216: Play Year of Plenty
        # 217-221: Play Monopoly
        # 222: Play Road Building
        # 223-225: Unused
        # 226-245: Bank Trade (Give X, Get Y)
        self.action_space = spaces.Discrete(246)

        # Define Observation Space
        # Resources: 2 players * 5 = 10
        # VPs: 2 players = 2
        # Hexes: 19 * (6 one-hot resource + 1 number + 1 robber) = 19 * 8 = 152
        # Vertices: 54 * (3 one-hot owner + 3 one-hot type + 6 one-hot port + 2 binary valid spots) = 54 * 14 = 756
        # Edges: 72 * (3 one-hot owner + 1 binary valid spot) = 72 * 4 = 288
        # Dev Cards: 7 (5 Agent + 2 Opponent)
        # Bonus VPs: 6 (3 Longest Road + 3 Largest Army)
        # Expert Features:
        #   Expected Income: 5 floats (Brick, Ore, Sheep, Wheat, Wood)
        #   Trade Rates: 5 floats (Brick, Ore, Sheep, Wheat, Wood)
        #   Opponent Hand Size: 1 float
        #   Karma: 1 float
        #   New Context Features: 19 floats
        #   Tactical Lookahead: 2 floats
        #   Distance to Connect: 1 float
        #   Race Stats: 4 floats (Agent Knights, Opp Knights, Agent Road, Opp Road)
        #   Grand Total: 1258

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1258,), dtype=np.float32
        )

        self.edge_list = []  # To store canonical edge indices
        self._init_edge_list()

        self.robber_placement_pending = False
        self.road_building_roads_left = 0
        self.played_dev_card_this_turn = False
        self.max_steps = 500

    def _init_edge_list(self):
        # We need a temporary board to calculate edges
        temp_game = catanGame(render_mode=None)
        board = temp_game.board

        edges = set()
        for v_idx in range(54):
            pixel = board.vertex_index_to_pixel_dict[v_idx]
            vertex = board.boardGraph[pixel]
            for neighbor_pixel in vertex.neighbors:
                neighbor_vertex = board.boardGraph[neighbor_pixel]
                n_idx = neighbor_vertex.vertex_index

                edge = tuple(sorted((v_idx, n_idx)))
                edges.add(edge)

        self.edge_list = sorted(list(edges))
        # assert len(self.edge_list) == 72 # Verify standard Catan edges

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game = catanGame(render_mode=self.render_mode)
        self.robber_placement_pending = False
        self.road_building_roads_left = 0
        self.played_dev_card_this_turn = False
        self.current_step = 0

        # Setup players
        with self.game.playerQueue.mutex:
            self.game.playerQueue.queue.clear()

        self.agent_player = player("Agent", 'black')
        # Monkeypatch discard
        self.agent_player.discardResources = self._agent_discard_resources

        # self.opponent_player = heuristicAIPlayer("Opponent", 'darkslateblue')
        self.opponent_player = RandomAIPlayer("Opponent", 'darkslateblue')  # <--- NEW DUMMY BOT
        # self.opponent_player.updateAI() # Not needed for random bot
        # Monkeypatch opponent discard
        # self.opponent_player.discardResources = self._agent_discard_resources

        self.game.playerQueue.put(self.agent_player)
        self.game.playerQueue.put(self.opponent_player)

        # Initial Setup
        self._perform_initial_setup()

        # Start the game loop logic
        self.game.gameOver = False
        self.game.currentPlayer = self.agent_player
        self.prev_vps = 0

        # Roll dice for the first turn
        dice = self.game.rollDice()
        if dice == 7:
            self.robber_placement_pending = True
            # Discard resources if needed
            for p in [self.agent_player, self.opponent_player]:
                p.discardResources(self.game)
        else:
            self.game.update_playerResources(dice, self.game.currentPlayer)

        return self._get_obs(), {}

    def _agent_discard_resources(self, game):
        # Balanced Hand Heuristic
        total = sum(self.agent_player.resources.values())
        if total > 9:
            num_to_discard = total // 2
            # Priority for discarding (Most discardable -> Least discardable)
            # Keep Rare (Ore/Wheat) > Common (Sheep)
            discard_priority = ['SHEEP', 'WOOD', 'BRICK', 'WHEAT', 'ORE']

            for _ in range(num_to_discard):
                # Find resource(s) with max count
                max_count = max(self.agent_player.resources.values())
                candidates = [r for r, c in self.agent_player.resources.items() if c == max_count]

                # Tie-break: Discard the one that appears earliest in discard_priority
                # We sort candidates based on their index in discard_priority and pick the first one
                resource_to_discard = sorted(candidates, key=lambda x: discard_priority.index(x))[0]

                self.agent_player.resources[resource_to_discard] -= 1

    def _perform_initial_setup(self):
        # Agent Setup (Random)
        for _ in range(2):
            possibleVertices = self.game.board.get_setup_settlements(self.agent_player)
            if possibleVertices:
                v = list(possibleVertices.keys())[np.random.randint(len(possibleVertices))]
                self.agent_player.build_settlement(v, self.game.board)

                possibleRoads = self.game.board.get_setup_roads(self.agent_player)
                if possibleRoads:
                    r = list(possibleRoads.keys())[np.random.randint(len(possibleRoads))]
                    self.agent_player.build_road(r[0], r[1], self.game.board)

        # Opponent Setup (Heuristic)
        for _ in range(2):
            self.opponent_player.initial_setup(self.game.board)
        # Give initial resources based on second settlement
        for p in [self.agent_player, self.opponent_player]:
            # Get last settlement
            if p.buildGraph['SETTLEMENTS']:
                last_settle_coord = p.buildGraph['SETTLEMENTS'][-1]
                vertex = self.game.board.boardGraph[last_settle_coord]
                for hex_idx in vertex.adjacent_hex_indices:
                    hex_tile = self.game.board.hexTileDict[hex_idx]
                    res = hex_tile.resource_type
                    if res != 'DESERT':
                        p.resources[res] += 1

    def step(self, action):
        self.current_step += 1
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.current_step >= self.max_steps:
            truncated = True
            # CRITICAL CHANGE: Penalize running out of time!
            # If we don't punish this, the agent just stalls forever to be safe.
            # Stalling should be worse than losing (-0.5) to force action.
            reward -= 1.0

        player = self.agent_player
        mask = self.get_action_mask()

        if mask[action] == 0:
            reward = -0.001
            # Invalid move, turn continues? Or end turn?
            # Usually in RL, invalid move ends episode or just penalizes.
            # Here we just penalize and return same state.
            return self._get_obs(), reward, terminated, truncated, info

        # 1. Priority: Road Building Card Phase
        if self.road_building_roads_left > 0:
            if 108 <= action <= 179:
                edge_idx = action - 108
                v1_idx, v2_idx = self.edge_list[edge_idx]
                v1_pixel = self.game.board.vertex_index_to_pixel_dict[v1_idx]
                v2_pixel = self.game.board.vertex_index_to_pixel_dict[v2_idx]

                # Build FREE road
                player.build_road(v1_pixel, v2_pixel, self.game.board, is_free=True)
                self.road_building_roads_left -= 1

                self.game.check_longest_road(player)
                if player.victoryPoints >= self.game.maxPoints:
                    reward += 2.0
                    terminated = True
                    info['is_success'] = True

                return self._get_obs(), reward, terminated, truncated, info
            else:
                # Should be impossible due to mask
                return self._get_obs(), -0.001, terminated, truncated, info

        # 2. Priority: Robber Placement
        if self.robber_placement_pending:
            if 181 <= action <= 199:
                hex_idx = action - 181
                # Move robber
                # Also need to rob a player if possible
                players_to_rob = self.game.board.get_players_to_rob(hex_idx)
                player_robbed = None
                if players_to_rob:
                    # Heuristic: Rob opponent if possible
                    if self.opponent_player in players_to_rob:
                        player_robbed = self.opponent_player
                    else:
                        player_robbed = list(players_to_rob.keys())[0]

                player.move_robber(hex_idx, self.game.board, player_robbed)
                self.robber_placement_pending = False
                reward += 0.001  # Small reward for completing action
                return self._get_obs(), reward, terminated, truncated, info
            else:
                # Should be caught by mask, but just in case
                reward = -0.001
                return self._get_obs(), reward, terminated, truncated, info

        # 3. Normal Actions
        if 0 <= action <= 53:  # Build Settlement
            vertex_idx = action
            pixel = self.game.board.vertex_index_to_pixel_dict[vertex_idx]

            player.build_settlement(pixel, self.game.board)
            reward += 0.1  # Bootcamp Reward
            if player.victoryPoints >= self.game.maxPoints:
                reward += 1.0  # Bootcamp Reward
                terminated = True
                info['is_success'] = True

        elif 54 <= action <= 107:  # Build City
            vertex_idx = action - 54
            pixel = self.game.board.vertex_index_to_pixel_dict[vertex_idx]

            player.build_city(pixel, self.game.board)
            reward += 0.15  # Bootcamp Reward
            if player.victoryPoints >= self.game.maxPoints:
                reward += 1.0  # Bootcamp Reward
                terminated = True
                info['is_success'] = True

        elif 108 <= action <= 179:  # Build Road
            edge_idx = action - 108
            v1_idx, v2_idx = self.edge_list[edge_idx]
            v1_pixel = self.game.board.vertex_index_to_pixel_dict[v1_idx]
            v2_pixel = self.game.board.vertex_index_to_pixel_dict[v2_idx]

            player.build_road(v1_pixel, v2_pixel, self.game.board)
            reward += 0.02  # Bootcamp Reward
            self.game.check_longest_road(player)
            if player.victoryPoints >= self.game.maxPoints:
                reward += 1.0  # Bootcamp Reward
                terminated = True
                info['is_success'] = True

        elif action == 180:  # End Turn
            # Soft Exit for Road Building Phase
            if self.road_building_roads_left > 0:
                self.road_building_roads_left = 0
                return self._get_obs(), reward, terminated, truncated, info

            self.played_dev_card_this_turn = False  # Reset for next turn
            # End Agent Turn
            # Run Opponent Turn
            self._run_opponent_turn()

            # Start Agent Turn
            if not self.game.gameOver:
                self.game.currentPlayer = self.agent_player
                self.agent_player.updateDevCards()
                dice = self.game.rollDice()

                if dice == 7:
                    self.robber_placement_pending = True
                    # Discard resources if needed
                    for p in [self.agent_player, self.opponent_player]:
                        p.discardResources(self.game)
                else:
                    self.game.update_playerResources(dice, self.agent_player)

                # Check if opponent won
                if self.opponent_player.victoryPoints >= self.game.maxPoints:
                    terminated = True
                    reward -= 1.0  # Bootcamp Reward
                    info['is_success'] = False
            else:
                terminated = True
                if self.opponent_player.victoryPoints >= self.game.maxPoints:
                    reward -= 1.0  # Bootcamp Reward
                    info['is_success'] = False

        elif action == 200:  # Buy Dev Card
            player.draw_devCard(self.game.board)
            reward += 0.01

        elif action == 201:  # Play Knight
            player.devCards['KNIGHT'] -= 1
            self.played_dev_card_this_turn = True
            self.robber_placement_pending = True
            player.knightsPlayed += 1
            self.game.check_largest_army(player)
            reward += 0.01
            if player.victoryPoints >= self.game.maxPoints:
                reward += 1.0  # Bootcamp Reward
                terminated = True
                info['is_success'] = True

        elif 202 <= action <= 216:  # Play Year of Plenty
            player.devCards['YEAROFPLENTY'] -= 1
            self.played_dev_card_this_turn = True
            r1, r2 = YOP_COMBINATIONS[action - 202]
            player.resources[r1] += 1
            player.resources[r2] += 1
            self.game.log_yop(player, [r1, r2])
            reward += 0.01

        elif 217 <= action <= 221:  # Play Monopoly
            player.devCards['MONOPOLY'] -= 1
            self.played_dev_card_this_turn = True
            target_res = ['BRICK', 'ORE', 'SHEEP', 'WHEAT', 'WOOD'][action - 217]

            # Steal from opponent
            stolen_count = self.opponent_player.resources[target_res]
            self.opponent_player.resources[target_res] = 0
            player.resources[target_res] += stolen_count
            reward += 0.01

        elif action == 222:  # Play Road Building
            player.devCards['ROADBUILDER'] -= 1
            self.played_dev_card_this_turn = True
            self.road_building_roads_left = 2
            reward += 0.01

        elif 226 <= action <= 245:  # Bank Trading
            give, get = TRADE_ACTIONS[action - 226]
            player.trade_with_bank(give, get)
            reward += 0.0  # Neutral reward for trading

        # Add VP reward
        current_vp = player.victoryPoints

        vp_diff = current_vp - self.prev_vps
        if vp_diff > 0:
            reward += 0.2 * vp_diff  # Bootcamp Reward
        self.prev_vps = current_vp

        return self._get_obs(), reward, terminated, truncated, info

    def _run_opponent_turn(self):
        opponent = self.opponent_player
        self.game.currentPlayer = opponent

        dice = self.game.rollDice()
        self.game.update_playerResources(dice, opponent)

        opponent.move(self.game.board)

        self.game.check_longest_road(opponent)
        self.game.check_largest_army(opponent)

        if opponent.victoryPoints >= self.game.maxPoints:
            self.game.gameOver = True

    def get_action_mask(self):
        # 1. Initialize as boolean False
        mask = np.zeros(246, dtype=bool)
        player = self.agent_player

        # Priority 1: Road Building Card Phase
        if self.road_building_roads_left > 0:
            # Only allow road building
            if player.roadsLeft > 0:
                potential_roads = self.game.board.get_potential_roads(player)
                for (v1_pixel, v2_pixel) in potential_roads.keys():
                    v1_idx = self.game.board.boardGraph[v1_pixel].vertex_index
                    v2_idx = self.game.board.boardGraph[v2_pixel].vertex_index
                    edge = tuple(sorted((v1_idx, v2_idx)))
                    try:
                        edge_idx = self.edge_list.index(edge)
                        mask[108 + edge_idx] = True
                    except ValueError:
                        pass

            # Soft Exit: If no roads can be built, allow skipping (Action 180)
            if not np.any(mask):
                mask[180] = True

            return mask

        # Priority 2: Robber Placement
        if self.robber_placement_pending:
            mask[181:200] = True
            return mask

        # Normal Phase
        mask[180] = True  # End Turn always valid

        resources = player.resources

        # Settlements (0-53)
        if (resources['BRICK'] >= 1 and resources['WOOD'] >= 1 and
            resources['SHEEP'] >= 1 and resources['WHEAT'] >= 1 and
                player.settlementsLeft > 0):
            potential_settlements = self.game.board.get_potential_settlements(player)
            for v_pixel in potential_settlements.keys():
                v_idx = self.game.board.boardGraph[v_pixel].vertex_index
                mask[v_idx] = True

        # Cities (54-107)
        if (resources['ORE'] >= 3 and resources['WHEAT'] >= 2 and
                player.citiesLeft > 0):
            potential_cities = self.game.board.get_potential_cities(player)
            for v_pixel in potential_cities.keys():
                v_idx = self.game.board.boardGraph[v_pixel].vertex_index
                mask[54 + v_idx] = True

        # Roads (108-179)
        if (resources['BRICK'] >= 1 and resources['WOOD'] >= 1 and
                player.roadsLeft > 0):
            potential_roads = self.game.board.get_potential_roads(player)
            for (v1_pixel, v2_pixel) in potential_roads.keys():
                v1_idx = self.game.board.boardGraph[v1_pixel].vertex_index
                v2_idx = self.game.board.boardGraph[v2_pixel].vertex_index
                edge = tuple(sorted((v1_idx, v2_idx)))
                try:
                    edge_idx = self.edge_list.index(edge)
                    mask[108 + edge_idx] = True
                except ValueError:
                    pass

        # Dev Cards
        def can_play_dev(type):
            if self.played_dev_card_this_turn:
                return False
            # player.devCards only contains cards from previous turns (playable)
            return player.devCards[type] > 0

        # Buy Dev Card (200)
        deck_size = sum(self.game.board.devCardStack.values())
        if (resources['ORE'] >= 1 and resources['WHEAT'] >= 1 and resources['SHEEP'] >= 1 and deck_size > 0):
            mask[200] = True

        # Play Knight (201)
        if can_play_dev('KNIGHT'):
            mask[201] = True

        # Play Year of Plenty (202-216)
        if can_play_dev('YEAROFPLENTY'):
            mask[202:217] = True

        # Play Monopoly (217-221)
        if can_play_dev('MONOPOLY'):
            mask[217:222] = True

        # Play Road Building (222)
        if can_play_dev('ROADBUILDER'):
            potential_roads = self.game.board.get_potential_roads(player)
            if len(potential_roads) >= 2 and player.roadsLeft >= 2:
                mask[222] = True

        # Bank Trading (226-245)
        # Not allowed during Road Building or Robber Placement (already handled by priority checks)
        for i, (give, get) in enumerate(TRADE_ACTIONS):
            cost = 4
            if '3:1 PORT' in player.portList:
                cost = 3
            if f'2:1 {give}' in player.portList:
                cost = 2

            if resources[give] >= cost:
                mask[226 + i] = True

        # NUCLEAR FAILSAFE
        if not np.any(mask):
            # Force 'End Turn' to be valid no matter what.
            mask[180] = True
        return mask

    def _toggle_road_on_board(self, v1, v2, player, on=True):
        # Helper to toggle road state without full engine overhead
        # v1, v2 are pixel coordinates

        # Update v1
        for indx, v in enumerate(self.game.board.boardGraph[v1].neighbors):
            if v == v2:
                self.game.board.boardGraph[v1].edge_state[indx][0] = player if on else None
                self.game.board.boardGraph[v1].edge_state[indx][1] = on

        # Update v2
        for indx, v in enumerate(self.game.board.boardGraph[v2].neighbors):
            if v == v1:
                self.game.board.boardGraph[v2].edge_state[indx][0] = player if on else None
                self.game.board.boardGraph[v2].edge_state[indx][1] = on

    def _get_2_step_road_potential(self):
        player = self.agent_player
        board = self.game.board

        # Base max length
        base_len = player.get_road_length(board)
        best_1 = base_len
        best_2 = base_len

        # Get potential roads (Step 1)
        potential_roads_1 = board.get_potential_roads(player)

        for (v1, v2) in potential_roads_1.keys():
            # Apply R1
            self._toggle_road_on_board(v1, v2, player, on=True)
            player.buildGraph['ROADS'].append((v1, v2))

            # Measure R1
            len_1 = player.get_road_length(board)
            if len_1 > best_1:
                best_1 = len_1

            # Step 2
            potential_roads_2 = board.get_potential_roads(player)
            for (u1, u2) in potential_roads_2.keys():
                # Apply R2
                self._toggle_road_on_board(u1, u2, player, on=True)
                player.buildGraph['ROADS'].append((u1, u2))

                # Measure R2
                len_2 = player.get_road_length(board)
                if len_2 > best_2:
                    best_2 = len_2

                # Revert R2
                player.buildGraph['ROADS'].pop()
                self._toggle_road_on_board(u1, u2, player, on=False)

            # Revert R1
            player.buildGraph['ROADS'].pop()
            self._toggle_road_on_board(v1, v2, player, on=False)

        return float(best_1), float(best_2)

    def _get_min_connect_distance(self):
        player = self.agent_player
        board = self.game.board

        # 1. Identify Components
        # Build adjacency list for player's roads
        adj = {}
        all_vertices = set()
        for (v1, v2) in player.buildGraph['ROADS']:
            if v1 not in adj:
                adj[v1] = []
            if v2 not in adj:
                adj[v2] = []
            adj[v1].append(v2)
            adj[v2].append(v1)
            all_vertices.add(v1)
            all_vertices.add(v2)

        if not all_vertices:
            return 15.0  # No roads

        # Find connected components
        visited = set()
        components = []
        for v in all_vertices:
            if v not in visited:
                component = set()
                q = [v]
                visited.add(v)
                while q:
                    curr = q.pop(0)
                    component.add(curr)
                    if curr in adj:
                        for neighbor in adj[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                components.append(component)

        if len(components) <= 1:
            return 0.0  # Already connected or empty

        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        largest_comp = components[0]
        other_nodes = set()
        for c in components[1:]:
            other_nodes.update(c)

        # BFS from largest component
        queue = []
        visited_bfs = set()

        for v in largest_comp:
            queue.append((v, 0))
            visited_bfs.add(v)

        while queue:
            curr, dist = queue.pop(0)

            if dist >= 15:
                continue

            # Check if we can build FROM curr
            curr_obj = board.boardGraph[curr]
            if curr_obj.owner is not None and curr_obj.owner != player:
                continue

            for i, neighbor in enumerate(curr_obj.neighbors):
                # Check edge availability
                p, is_road = curr_obj.edge_state[i]
                if is_road:
                    # If it's my road, it's already in a component.
                    # If it's opponent road, blocked.
                    continue
                else:
                    # Empty edge, can build here
                    # Check if neighbor is target
                    if neighbor in other_nodes:
                        return float(dist + 1)

                    if neighbor not in visited_bfs:
                        visited_bfs.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return 15.0  # No path found

    def _get_obs(self):
        # Flatten state
        obs = []

        # Resources (Normalize by dividing by 20.0 to keep roughly in 0-1 range)
        for p in [self.agent_player, self.opponent_player]:
            obs.extend([p.resources[r] / 20.0 for r in ['BRICK', 'ORE', 'WHEAT', 'WOOD', 'SHEEP']])

        # VPs (Normalize by dividing by 10.0)
        obs.append(self.agent_player.victoryPoints / 10.0)
        obs.append(self.opponent_player.victoryPoints / 10.0)

        # Hexes
        resource_map = {'DESERT': 0, 'ORE': 1, 'BRICK': 2, 'WHEAT': 3, 'WOOD': 4, 'SHEEP': 5}
        for i in range(19):
            hex_tile = self.game.board.hexTileDict[i]
            res_one_hot = [0]*6
            res_one_hot[resource_map[hex_tile.resource_type]] = 1
            obs.extend(res_one_hot)

            num = hex_tile.number_token if hex_tile.number_token is not None else 0
            obs.append(num / 12.0)  # Normalize

            obs.append(1.0 if hex_tile.has_robber else 0.0)

        # Vertices
        port_map = {
            '2:1 BRICK': 0,
            '2:1 ORE': 1,
            '2:1 WHEAT': 2,
            '2:1 WOOD': 3,
            '2:1 SHEEP': 4,
            '3:1 PORT': 5
        }

        # Pre-calculate valid spots (ignoring resources)
        potential_settlements = self.game.board.get_potential_settlements(self.agent_player)
        potential_cities = self.game.board.get_potential_cities(self.agent_player)

        for i in range(54):
            pixel = self.game.board.vertex_index_to_pixel_dict[i]
            vertex = self.game.board.boardGraph[pixel]

            owner_one_hot = [0]*3
            type_one_hot = [0]*3
            port_one_hot = [0]*6
            valid_spot_one_hot = [0]*2  # [IsSettlementSpot, IsCitySpot]

            if vertex.building_type == 'Settlement':
                type_one_hot[1] = 1
                if vertex.owner == self.agent_player:
                    owner_one_hot[1] = 1
                else:
                    owner_one_hot[2] = 1
            elif vertex.building_type == 'City':
                type_one_hot[2] = 1
                if vertex.owner == self.agent_player:
                    owner_one_hot[1] = 1
                else:
                    owner_one_hot[2] = 1
            else:
                type_one_hot[0] = 1
                owner_one_hot[0] = 1

            # Port
            if vertex.port:
                if vertex.port in port_map:
                    port_one_hot[port_map[vertex.port]] = 1

            # Valid Spots
            if pixel in potential_settlements:
                valid_spot_one_hot[0] = 1
            if pixel in potential_cities:
                valid_spot_one_hot[1] = 1

            obs.extend(owner_one_hot)
            obs.extend(type_one_hot)
            obs.extend(port_one_hot)
            obs.extend(valid_spot_one_hot)

        # Edges
        potential_roads = self.game.board.get_potential_roads(self.agent_player)

        for v1_idx, v2_idx in self.edge_list:
            v1_pixel = self.game.board.vertex_index_to_pixel_dict[v1_idx]
            v2_pixel = self.game.board.vertex_index_to_pixel_dict[v2_idx]
            v1 = self.game.board.boardGraph[v1_pixel]

            owner_one_hot = [0]*3
            owner_one_hot[0] = 1
            valid_road_spot = [0]

            for i, neighbor_pixel in enumerate(v1.neighbors):
                if neighbor_pixel == v2_pixel:
                    p = v1.edge_state[i][0]
                    is_road = v1.edge_state[i][1]
                    if is_road:
                        owner_one_hot[0] = 0
                        if p == self.agent_player:
                            owner_one_hot[1] = 1
                        else:
                            owner_one_hot[2] = 1
                    break

            # Check if this edge is a valid road spot
            # potential_roads keys are (v1, v2) or (v2, v1)
            if (v1_pixel, v2_pixel) in potential_roads or (v2_pixel, v1_pixel) in potential_roads:
                valid_road_spot[0] = 1

            obs.extend(owner_one_hot)
            obs.extend(valid_road_spot)

        # Dev Cards
        dev_card_order = ['KNIGHT', 'VP', 'ROADBUILDER', 'YEAROFPLENTY', 'MONOPOLY']
        for card in dev_card_order:
            obs.append(self.agent_player.devCards[card] / 5.0)

        opp_hidden_total = sum(self.opponent_player.devCards.values())
        obs.append(opp_hidden_total / 10.0)
        # Removed Opponent Knights from here (moved to Race Stats)

        # Race Stats (4 Normalized Floats)
        obs.append(self.agent_player.knightsPlayed / 5.0)
        obs.append(self.opponent_player.knightsPlayed / 5.0)
        obs.append(self.agent_player.maxRoadLength / 15.0)
        obs.append(self.opponent_player.maxRoadLength / 15.0)

        # Bonus VPs
        lr_one_hot = [0]*3
        if self.agent_player.longestRoadFlag:
            lr_one_hot[1] = 1
        elif self.opponent_player.longestRoadFlag:
            lr_one_hot[2] = 1
        else:
            lr_one_hot[0] = 1
        obs.extend(lr_one_hot)

        la_one_hot = [0]*3
        if self.agent_player.largestArmyFlag:
            la_one_hot[1] = 1
        elif self.opponent_player.largestArmyFlag:
            la_one_hot[2] = 1
        else:
            la_one_hot[0] = 1
        obs.extend(la_one_hot)

        # Expert Features
        # 1. Expected Income
        dots_map = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
        income = {'BRICK': 0, 'ORE': 0, 'SHEEP': 0, 'WHEAT': 0, 'WOOD': 0}

        for i in range(19):
            hex_tile = self.game.board.hexTileDict[i]
            res_type = hex_tile.resource_type
            num = hex_tile.number_token

            if res_type != 'DESERT' and num is not None:
                # Get vertices
                corners = hex_tile.get_corners(self.game.board.flat)
                for corner in corners:
                    # Find vertex object
                    # This is slow, but we need to match pixel coords
                    # Optimization: Use pre-calculated vertex-hex map if available
                    # For now, iterate boardGraph
                    for v_pixel, v_obj in self.game.board.boardGraph.items():
                        if self.game.board.vertexDistance(corner, v_pixel) == 0:
                            if v_obj.owner == self.agent_player:
                                multiplier = 2 if v_obj.building_type == 'City' else 1
                                income[res_type] += dots_map[num] * multiplier
                            break

        for r in ['BRICK', 'ORE', 'SHEEP', 'WHEAT', 'WOOD']:
            obs.append(income[r] / 36.0)

        # 2. Trade Rates
        rates = {'BRICK': 4.0, 'ORE': 4.0, 'SHEEP': 4.0, 'WHEAT': 4.0, 'WOOD': 4.0}
        ports = self.agent_player.portList

        if '3:1 PORT' in ports:
            for r in rates:
                rates[r] = 3.0

        for r in rates:
            if f'2:1 {r}' in ports:
                rates[r] = 2.0

        for r in ['BRICK', 'ORE', 'SHEEP', 'WHEAT', 'WOOD']:
            obs.append(rates[r] / 4.0)

        # 3. Opponent Hand Size
        opp_hand_size = sum(self.opponent_player.resources.values())
        obs.append(opp_hand_size / 20.0)

        # 4. Karma
        karma_val = 0.5
        if self.game.last_player_to_roll_7 is not None:
            if self.game.last_player_to_roll_7 == self.agent_player:
                karma_val = 1.0
            elif self.game.last_player_to_roll_7 == self.opponent_player:
                karma_val = 0.0
        obs.append(karma_val)

        # 5. New Context Features (19 floats)

        # A. Affordability (5 Binary Floats)
        # Road: 1 Brick, 1 Wood
        can_buy_road = 1.0 if (
            self.agent_player.resources['BRICK'] >= 1 and self.agent_player.resources['WOOD'] >= 1) else 0.0
        # Settlement: 1 Brick, 1 Wood, 1 Sheep, 1 Wheat
        can_buy_settle = 1.0 if (self.agent_player.resources['BRICK'] >= 1 and self.agent_player.resources['WOOD']
                                 >= 1 and self.agent_player.resources['SHEEP'] >= 1 and self.agent_player.resources['WHEAT'] >= 1) else 0.0
        # City: 3 Ore, 2 Wheat
        can_buy_city = 1.0 if (
            self.agent_player.resources['ORE'] >= 3 and self.agent_player.resources['WHEAT'] >= 2) else 0.0
        # DevCard: 1 Ore, 1 Sheep, 1 Wheat
        can_buy_dev = 1.0 if (self.agent_player.resources['ORE'] >= 1 and self.agent_player.resources['SHEEP']
                              >= 1 and self.agent_player.resources['WHEAT'] >= 1) else 0.0
        # Road Building Card Active
        rb_active = 1.0 if self.road_building_roads_left > 0 else 0.0

        obs.extend([can_buy_road, can_buy_settle, can_buy_city, can_buy_dev, rb_active])

        # B. Piece Scarcity (3 Normalized Floats)
        obs.append(self.agent_player.roadsLeft / 15.0)
        obs.append(self.agent_player.settlementsLeft / 5.0)
        obs.append(self.agent_player.citiesLeft / 4.0)

        # C. Deck Health (1 Normalized Float)
        obs.append(sum(self.game.board.devCardStack.values()) / 25.0)

        # D. Game Phase (5 One-Hot Floats)
        # [Main, Robber, RoadBuilding, Discard, GameOver]
        phase_one_hot = [0.0] * 5
        if self.game.gameOver:
            phase_one_hot[4] = 1.0
        elif self.road_building_roads_left > 0:
            phase_one_hot[2] = 1.0
        elif self.robber_placement_pending:
            phase_one_hot[1] = 1.0
        # Discard phase is instantaneous in this env, so we skip index 3 (keep 0)
        else:
            phase_one_hot[0] = 1.0  # Main Turn

        obs.extend(phase_one_hot)

        # E. Exchange Rates (5 Normalized Floats)
        # Recalculate to ensure it's at the end as requested
        ex_rates = {'BRICK': 4.0, 'ORE': 4.0, 'SHEEP': 4.0, 'WHEAT': 4.0, 'WOOD': 4.0}
        if '3:1 PORT' in self.agent_player.portList:
            for r in ex_rates:
                ex_rates[r] = 3.0
        for r in ex_rates:
            if f'2:1 {r}' in self.agent_player.portList:
                ex_rates[r] = 2.0

        for r in ['BRICK', 'ORE', 'SHEEP', 'WHEAT', 'WOOD']:
            obs.append(ex_rates[r] / 4.0)

        # F. Spatial Reasoning (3 Normalized Floats)
        pot_1, pot_2 = self._get_2_step_road_potential()
        connect_dist = self._get_min_connect_distance()

        obs.append(pot_1 / 15.0)
        obs.append(pot_2 / 15.0)
        obs.append(connect_dist / 15.0)

        obs_array = np.array(obs, dtype=np.float32)
        # Sanity check for NaNs or Infs
        if not np.isfinite(obs_array).all():
            # print("Warning: NaN or Inf in observation. Replacing with 0.")
            obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs_array
