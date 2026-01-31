
import sys
import os
import numpy as np
import pygame
from sb3_contrib import MaskablePPO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catan.engine.game import catanGame
from catan.engine.player import player
from catan.rl.env import CatanEnv  # Import purely for constants or we can redefine them


class PPOAgentPlayer(player):
    def __init__(self, name, color, model_path, env_dummy):
        super().__init__(name, color)
        self.isAI = True
        self.model = MaskablePPO.load(model_path, device='cpu')
        # We need an env instance to access helper methods for observation/masking
        # simpler than copying 1000 lines of code.
        self.env = env_dummy

    def initial_setup(self, board):
        # We can just run random setup for now, or use the model if it supports setup phase actions (it usually doesn't in this env)
        # The training env uses random setup for agent.

        # Build random settlement
        possibleVertices = board.get_setup_settlements(self)
        if possibleVertices:
            vertexToBuild = list(possibleVertices.keys())[np.random.randint(len(possibleVertices))]
            self.build_settlement(vertexToBuild, board, is_free=True)

        # Build random road
        possibleRoads = board.get_setup_roads(self)
        if possibleRoads:
            roadToBuild = list(possibleRoads.keys())[np.random.randint(len(possibleRoads))]
            self.build_road(roadToBuild[0], roadToBuild[1], board, is_free=True)

    def move(self, board):
        # This function is called by the game loop when it is AI's turn
        print(f"AI {self.name} is thinking...")

        MAX_MOVES = 20
        moves = 0

        while moves < MAX_MOVES:
            # 1. Sync the dummy env with current game state
            # We simply inject the current game instance into the env
            # The game loop in game.py has updated 'self.game' but our env needs to point to it

            # CRITICAL: The Board object in game.py is passed to us.
            # We need to ensure self.env.game points to the active game object.
            # But we don't have access to the 'game' object here, only 'board'.
            # Wait, 'player.move(board)' is the signature in game.py.
            # But the 'env' needs access to 'game' (for game.currentPlayer, game.dice, etc).
            # We will attach 'game' to this player during setup.

            self.env.game = self.game_ref  # Attached externally
            self.env.agent_player = self
            # Opponent is the other player
            self.env.opponent_player = [p for p in self.game_ref.playerQueue.queue if p != self][0]

            # 2. Get Observation and Mask
            obs = self.env._get_obs()
            action_mask = self.env.get_action_mask()

            # 3. Predict
            action, _ = self.model.predict(obs, action_masks=action_mask)
            action = int(action)

            print(f"AI chose action: {action}")

            # 4. Execute Action
            # We reuse Env's physics engine logic, but stripped of reward/termination checks
            # We can basically call a modified step, or just execute the action logic here.

            if action == 180:  # End Turn
                print("AI Ends Turn.")
                break

            # Execute action using the mapping logic from CatanEnv
            self.execute_action(action)
            moves += 1

            # Refresh screen
            self.game_ref.boardView.displayGameScreen()
            pygame.time.wait(500)  # Small delay to see moves

    def execute_action(self, action):
        # Copied/Simplified from CatanEnv.step()
        # Note: We use self.game_ref instead of self.env.game
        game = self.game_ref
        board = game.board

        if 0 <= action <= 53:  # Build Settlement
            pixel = board.vertex_index_to_pixel_dict[action]
            self.build_settlement(pixel, board)

        elif 54 <= action <= 107:  # Build City
            pixel = board.vertex_index_to_pixel_dict[action - 54]
            self.build_city(pixel, board)

        elif 108 <= action <= 179:  # Build Road
            edge_idx = action - 108
            v1_idx, v2_idx = self.env.edge_list[edge_idx]
            v1_pixel = board.vertex_index_to_pixel_dict[v1_idx]
            v2_pixel = board.vertex_index_to_pixel_dict[v2_idx]
            self.build_road(v1_pixel, v2_pixel, board)

        elif 181 <= action <= 199:  # Move Robber
            hex_idx = action - 181
            players_to_rob = board.get_players_to_rob(hex_idx)
            player_robbed = None
            if players_to_rob and self.env.opponent_player in players_to_rob:
                player_robbed = self.env.opponent_player
            self.move_robber(hex_idx, board, player_robbed)
            # Handle resource discard if pending? handled in game loop usually

        elif action == 200:  # Buy Dev
            self.draw_devCard(board)

        elif action == 201:  # Play Knight
            self.devCards['KNIGHT'] -= 1
            self.knightsPlayed += 1
            game.check_largest_army(self)
            # Logic for robber placement follows in next step usually,
            # but in Env it handles it immediately.
            # We need to set a flag or force robber move?
            # In Env, 201 sets 'robber_placement_pending'.
            # If we set it in env, get_action_mask handles it.
            self.env.robber_placement_pending = True

        # ... Other Dev cards (YOP, Monopoly, RB) ...
        # For brevity, implementing minimal set.
        # Ideally we copy the whole block.

        elif 202 <= action <= 216:  # YOP
            # ...
            pass  # Implementation needed for full play

        # Mapping trade, etc.


def main():
    # Initialize Helper Env
    # We create a dummy env just to use its methods (_get_obs, get_action_mask, edge_list)
    env_dummy = CatanEnv(render_mode=None)

    # Initialize Game in HEADLESS mode to avoid auto-setup
    game = catanGame(render_mode=None)

    with game.playerQueue.mutex:
        game.playerQueue.queue.clear()

    # Initialize Human Player
    human = player("Human", "black")
    human.game_ref = game

    # Initialize AI Player (PPO)
    try:
        ai = PPOAgentPlayer("PPO_Bot", "darkslateblue", "models/catan_ppo_final.zip", env_dummy)
    except:
        print("Could not load catan_ppo_final.zip. Using random agent logic fallback (not implemented fully).")
        return

    ai.game_ref = game

    # Manually populate queue
    # Note: catanGame init created a queue of size 2.
    game.playerQueue.put(human)
    game.playerQueue.put(ai)

    game.numPlayers = 2
    game.playerList = [human, ai]
    game.resource_tracker = None

    # Init View
    from catan.gui.view import catanGameView
    game.boardView = catanGameView(game.board, game)

    # Initial Setup Phase (Interactive)
    print("Setup Phase...")
    game.boardView.displayInitialBoard()
    pygame.display.flip()

    # Forward Pass
    for p in game.playerList:
        game.currentPlayer = p
        if p.isAI:
            print(f"AI {p.name} Setup (Forward)")
            p.initial_setup(game.board)
            game.boardView.displayGameScreen()
        else:
            print(f"Human {p.name} Setup (Forward)")
            game.build(p, 'SETTLE', is_free=True)
            game.boardView.displayGameScreen()
            game.build(p, 'ROAD', is_free=True)
            game.boardView.displayGameScreen()

    # Backward Pass
    for p in reversed(game.playerList):
        game.currentPlayer = p
        if p.isAI:
            print(f"AI {p.name} Setup (Reverse)")
            p.initial_setup(game.board)
            game.boardView.displayGameScreen()
        else:
            print(f"Human {p.name} Setup (Reverse)")
            game.build(p, 'SETTLE', is_free=True)
            game.boardView.displayGameScreen()
            game.build(p, 'ROAD', is_free=True)
            game.boardView.displayGameScreen()

    # Initial Resources
    for p in [human, ai]:
        if p.buildGraph['SETTLEMENTS']:
            last = p.buildGraph['SETTLEMENTS'][-1]
            v = game.board.boardGraph[last]
            for h in v.adjacent_hex_indices:
                res = game.board.hexTileDict[h].resource_type
                if res != 'DESERT':
                    p.resources[res] += 1

    print("Setup Complete. Starting Game Loop.")

    # CRITICAL FIX: Mark setup as complete so build() uses normal rules
    game.gameSetup = False

    game.boardView.displayGameScreen()

    # Run Game
    game.playCatan()


if __name__ == "__main__":
    main()
