"""Random AI opponent for training and evaluation."""

import numpy as np
from catan.engine.player import player


class RandomAIPlayer(player):
    def __init__(self, name, color):
        super().__init__(name, color)
        self.isAI = True

    def initial_setup(self, board):
        possibleVertices = board.get_setup_settlements(self)
        if possibleVertices:
            vertexToBuild = list(possibleVertices.keys())[np.random.randint(len(possibleVertices))]
            self.build_settlement(vertexToBuild, board, is_free=True)

        possibleRoads = board.get_setup_roads(self)
        if possibleRoads:
            roadToBuild = list(possibleRoads.keys())[np.random.randint(len(possibleRoads))]
            self.build_road(roadToBuild[0], roadToBuild[1], board, is_free=True)

    def heuristic_move_robber(self, board):
        potential_spots = board.get_robber_spots()
        if not potential_spots:
            return

        hex_idx = list(potential_spots.keys())[np.random.randint(len(potential_spots))]
        players_to_rob = board.get_players_to_rob(hex_idx)
        player_robbed = None
        if players_to_rob:
            opponents = [p for p in players_to_rob.keys() if p != self]
            if opponents:
                player_robbed = opponents[np.random.randint(len(opponents))]

        self.move_robber(hex_idx, board, player_robbed)

    def move(self, board):
        if np.random.random() < 0.3:
            self.draw_devCard(board)
        if self.resources['ORE'] >= 3 and self.resources['WHEAT'] >= 2:
            potential_cities = board.get_potential_cities(self)
            if potential_cities:
                v_pixel = list(potential_cities.keys())[np.random.randint(len(potential_cities))]
                self.build_city(v_pixel, board)
        if (self.resources['BRICK'] >= 1 and self.resources['WOOD'] >= 1 and
                self.resources['SHEEP'] >= 1 and self.resources['WHEAT'] >= 1):
            potential_settlements = board.get_potential_settlements(self)
            if potential_settlements:
                v_pixel = list(potential_settlements.keys())[
                    np.random.randint(len(potential_settlements))]
                self.build_settlement(v_pixel, board)
        if self.resources['BRICK'] >= 1 and self.resources['WOOD'] >= 1:
            potential_roads = board.get_potential_roads(self)
            if potential_roads:
                r_coords = list(potential_roads.keys())[np.random.randint(len(potential_roads))]
                self.build_road(r_coords[0], r_coords[1], board)

    def discardResources(self, game):
        total = sum(self.resources.values())
        if total > 9:
            num_discard = total // 2
            discarded_resources = []
            for _ in range(num_discard):
                available = [r for r, c in self.resources.items() if c > 0]
                if not available:
                    break
                res = available[np.random.randint(len(available))]
                self.resources[res] -= 1
                discarded_resources.append(res)

            if discarded_resources:
                # Reuse central logging/broadcasting in catanGame
                game.log_discard(self, discarded_resources)
