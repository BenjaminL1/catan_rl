# Settlers of Catan
# Heuristic AI class implementation

import numpy as np

from catan_rl.engine.board import *
from catan_rl.engine.player import *

# Class definition for an AI player


class heuristicAIPlayer(player):
    # Update AI player flag and resources
    def updateAI(self):
        self.isAI = True
        self.setupResources = []  # List to keep track of setup resources
        # Initialize resources with just correct number needed for set up
        self.resources = {
            "ORE": 0,
            "BRICK": 0,
            "WHEAT": 0,
            "WOOD": 0,
            "SHEEP": 0,
        }  # Dictionary that keeps track of resource amounts
        # print("Added new AI Player:", self.name)

    # Function to build an initial settlement - just choose random spot for now

    def initial_setup(self, board):
        # Build random settlement
        possibleVertices = board.get_setup_settlements(self)

        # Simple heuristic for choosing initial spot
        diceRoll_expectation = {
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            8: 5,
            9: 4,
            10: 3,
            11: 2,
            12: 1,
            None: 0,
        }
        vertexValues = []

        # Get the adjacent hexes for each hex
        for v in possibleVertices.keys():
            vertexNumValue = 0
            resourcesAtVertex = []
            # For each adjacent hex get its value and overall resource diversity for that vertex
            for adjacentHex in board.boardGraph[v].adjacent_hex_indices:
                resourceType = board.hexTileDict[adjacentHex].resource_type
                if resourceType not in resourcesAtVertex:
                    resourcesAtVertex.append(resourceType)
                numValue = board.hexTileDict[adjacentHex].number_token
                # Add to total value of this vertex
                vertexNumValue += diceRoll_expectation[numValue]

            # basic heuristic for resource diversity
            vertexNumValue += len(resourcesAtVertex) * 2
            for r in resourcesAtVertex:
                if r != "DESERT" and r not in self.setupResources:
                    vertexNumValue += 2.5  # Every new resource gets a bonus

            vertexValues.append(vertexNumValue)

        vertexToBuild_index = vertexValues.index(max(vertexValues))
        vertexToBuild = list(possibleVertices.keys())[vertexToBuild_index]

        # Add to setup resources
        for adjacentHex in board.boardGraph[vertexToBuild].adjacent_hex_indices:
            resourceType = board.hexTileDict[adjacentHex].resource_type
            if resourceType not in self.setupResources and resourceType != "DESERT":
                self.setupResources.append(resourceType)

        self.build_settlement(vertexToBuild, board, is_free=True)

        # Build random road
        possibleRoads = board.get_setup_roads(self)
        randomEdge = np.random.randint(0, len(possibleRoads.keys()))
        self.build_road(
            list(possibleRoads.keys())[randomEdge][0],
            list(possibleRoads.keys())[randomEdge][1],
            board,
            is_free=True,
        )

    def move(self, board):
        # print("AI Player {} playing...".format(self.name))
        # Play a development card first (D2) — a Knight can unblock a hex and a
        # YoP/Monopoly can complete a build that the rest of this turn spends.
        self.heuristic_play_dev_card(board)
        # Trade resources if there are excessive amounts of a particular resource
        self.trade()
        # Build a settlements, city and few roads
        possibleVertices = board.get_potential_settlements(self)
        if possibleVertices != {} and (
            self.resources["BRICK"] > 0
            and self.resources["WOOD"] > 0
            and self.resources["SHEEP"] > 0
            and self.resources["WHEAT"] > 0
        ):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            self.build_settlement(list(possibleVertices.keys())[randomVertex], board)

        # Build a City
        possibleVertices = board.get_potential_cities(self)
        if possibleVertices != {} and (self.resources["WHEAT"] >= 2 and self.resources["ORE"] >= 3):
            randomVertex = np.random.randint(0, len(possibleVertices.keys()))
            self.build_city(list(possibleVertices.keys())[randomVertex], board)

        # Build a couple roads
        for i in range(2):
            if self.resources["BRICK"] > 0 and self.resources["WOOD"] > 0:
                possibleRoads = board.get_potential_roads(self)
                if not possibleRoads:
                    break
                randomEdge = np.random.randint(0, len(possibleRoads.keys()))
                self.build_road(
                    list(possibleRoads.keys())[randomEdge][0],
                    list(possibleRoads.keys())[randomEdge][1],
                    board,
                )

        # Draw a Dev Card with 1/3 probability
        devCardNum = np.random.randint(0, 3)
        if devCardNum == 0:
            self.draw_devCard(board)

        return

    # Wrapper function to control all trading
    def trade(self):
        for r1, r1_amount in self.resources.items():
            if (
                r1_amount >= 6
            ):  # heuristic to trade if a player has more than 5 of a particular resource
                for r2, r2_amount in self.resources.items():
                    if r2_amount < 1:
                        self.trade_with_bank(r1, r2, self.game.board)  # spec 009
                        break

    # Choose which player to rob

    def choose_player_to_rob(self, board):
        """Heuristic function to choose the player with maximum points.
        Choose hex with maximum other players, Avoid blocking own resource
        args: game board object
        returns: hex index and player to rob
        """
        # Get list of robber spots
        robberHexDict = board.get_robber_spots()

        # Choose a hexTile with maximum adversary settlements
        maxHexScore = -float("inf")  # Keep only the best hex to rob

        # Default values if no better option is found
        if robberHexDict:
            hexToRob_index = list(robberHexDict.keys())[0]
        else:
            # Should not happen if get_robber_spots returns valid spots
            hexToRob_index = 0

        playerToRob_hex = None

        for hex_ind, hexTile in robberHexDict.items():
            # Extract all 6 vertices of this hexTile
            vertexList = hexTile.get_corners(board.flat)

            hexScore = 0  # Heuristic score for hexTile
            playerToRob_VP = 0
            playerToRob = None
            for vertex in vertexList:
                playerAtVertex = board.boardGraph[vertex].owner
                if playerAtVertex == self:
                    hexScore -= self.victoryPoints
                elif playerAtVertex != None:  # There is an adversary on this vertex
                    # Compute visible VP LIVE (victoryPoints − hidden VP cards):
                    # player.visibleVictoryPoints is a cache only refreshed at init
                    # + VP-card buy, so it goes stale after settlement/city/LA/LR
                    # changes — mirrors the obs_encoder + board.get_robber_spots fix.
                    vis_vp = playerAtVertex.victoryPoints - playerAtVertex.devCards.get("VP", 0)
                    hexScore += vis_vp
                    # Find strongest other player at this hex, provided player has resources
                    if vis_vp >= playerToRob_VP and sum(playerAtVertex.resources.values()) > 0:
                        playerToRob_VP = vis_vp
                        playerToRob = playerAtVertex
                else:
                    pass

            if hexScore >= maxHexScore and playerToRob != None:
                hexToRob_index = hex_ind
                playerToRob_hex = playerToRob
                maxHexScore = hexScore

        return hexToRob_index, playerToRob_hex

    def heuristic_move_robber(self, board):
        """Function to control heuristic AI robber
        Calls the choose_player_to_rob and move_robber functions
        args: board object
        """
        # Get the best hex and player to rob
        hex_i, playerRobbed = self.choose_player_to_rob(board)

        # Move the robber
        self.move_robber(hex_i, board, playerRobbed)

        return

    # ------------------------------------------------------------------
    # Development-card policy (spec bc-coverage-and-bank-legality D2)
    # ------------------------------------------------------------------
    #
    # This used to be a commented-out stub, which meant the practice
    # opponent BOUGHT development cards and never PLAYED one: the BC corpus
    # held ~50k BUY_DEV_CARD rows and ZERO plays, so action types 6/7/8/9
    # (Knight / YoP / Monopoly / Road Builder) had never appeared as a
    # positive example. No filter change can recover an action the teacher
    # never took.
    #
    # The transitions below deliberately do NOT call ``player.play_devCard``:
    # that engine method routes through ``game.boardView.get_dev_card_selection``
    # (a GUI dependency) and the engine tree is pinned. They mirror the env's
    # apply paths in ``env/catan_env.py`` exactly instead —
    # ``devCardPlayedThisTurn = True`` BEFORE any grant (which is what keeps
    # YoP livelock-free), every resource delta routed through
    # ``board.bank_draw`` / ``bank_recirculate`` (spec-009 conservation), and
    # the matching broadcast so ``BroadcastHandTracker`` stays in sync.

    #: Build costs in ENGINE resource order/naming (not Charlesworth).
    _BUILD_COSTS = {
        "CITY": {"ORE": 3, "WHEAT": 2},
        "SETTLEMENT": {"BRICK": 1, "WOOD": 1, "SHEEP": 1, "WHEAT": 1},
        "DEVCARD": {"ORE": 1, "WHEAT": 1, "SHEEP": 1},
        "ROAD": {"BRICK": 1, "WOOD": 1},
    }

    def _shortfall(self, cost):
        """Resources still missing to afford ``cost`` (engine naming)."""
        return {
            r: n - self.resources.get(r, 0) for r, n in cost.items() if self.resources.get(r, 0) < n
        }

    def _robber_hex_index(self, board):
        for idx, hex_tile in board.hexTileDict.items():
            if hex_tile.has_robber:
                return idx
        return None

    def _robber_sits_on_own_hex(self, board):
        idx = self._robber_hex_index(board)
        if idx is None:
            return False
        for vertex in board.hexTileDict[idx].get_corners(board.flat):
            if board.boardGraph[vertex].owner is self:
                return True
        return False

    def _knight_is_useful(self, board):
        """Play a Knight when the robber sits on one of our own hexes (it is
        costing us production every roll) or when there is an opponent worth
        robbing (blocking their best number)."""
        if self.devCards.get("KNIGHT", 0) < 1 or self.devCardPlayedThisTurn:
            return False
        if self._robber_sits_on_own_hex(board):
            return True
        _hex_i, player_to_rob = self.choose_player_to_rob(board)
        return player_to_rob is not None

    def heuristic_play_knight(self, board):
        """Play a Knight: consume the card, move the robber, re-check army."""
        self.devCards["KNIGHT"] -= 1
        self.knightsPlayed += 1
        self.devCardPlayedThisTurn = True
        self.heuristic_move_robber(board)
        if getattr(self, "game", None) is not None:
            self.game.check_largest_army(self)

    def heuristic_pre_roll(self, board):
        """Pre-roll dev-card window — Knight only.

        Playing the Knight BEFORE the roll is strictly better than after when
        the robber is blocking our own production: the block is lifted in time
        for this turn's dice. Subsumes ``preroll-dev-cards-r1`` D6 on the
        opponent side. Returns True if a card was played.

        Scoped to the case where pre-roll is genuinely better than post-roll —
        the robber sitting on OUR hex. A purely offensive Knight (denying the
        opponent's best number) is left to the main phase, where it is also a
        recordable decision for the BC corpus.
        """
        if (
            self.devCards.get("KNIGHT", 0) >= 1
            and not self.devCardPlayedThisTurn
            and self._robber_sits_on_own_hex(board)
        ):
            self.heuristic_play_knight(board)
            return True
        return False

    def _yop_picks(self, board):
        """Two resources that would complete a build, or None.

        Only picks the bank can actually supply are returned — a pick the bank
        cannot honour is simply not granted, so asking for it wastes the card.
        """
        bank = board.resourceBank
        for kind in ("CITY", "SETTLEMENT", "DEVCARD", "ROAD"):
            missing = self._shortfall(self._BUILD_COSTS[kind])
            needed = [r for r, n in missing.items() for _ in range(n)]
            if not needed or len(needed) > 2:
                continue
            if len(needed) == 1:
                # One card short: take the missing resource plus the scarcest
                # resource of the next-most-expensive build we can see.
                needed.append(needed[0])
            first, second = needed[0], needed[1]
            if first == second:
                if bank.get(first, 0) >= 2:
                    return first, second
                # Bank cannot double up — fall back to a DIFFERENT second pick.
                # Excluding ``first`` is load-bearing: without it the generator
                # re-selects the very resource this branch exists to reject and
                # returns the (first, first) pair the bank cannot honour, so the
                # card grants one card and the BC row teaches a wasteful play.
                if bank.get(first, 0) < 1:
                    continue
                second = next(
                    (
                        r
                        for r in ("ORE", "WHEAT", "BRICK", "WOOD", "SHEEP")
                        if r != first and bank.get(r, 0) >= 1
                    ),
                    None,
                )
                if second is None:
                    continue
                return first, second
            if bank.get(first, 0) >= 1 and bank.get(second, 0) >= 1:
                return first, second
        return None

    def heuristic_play_year_of_plenty(self, board, r1, r2):
        """Play Year of Plenty for ``r1`` + ``r2`` (engine resource names)."""
        self.devCards["YEAROFPLENTY"] -= 1
        self.yopPlayed += 1
        # Mirrors env/catan_env.py: the flag is set BEFORE the grant so a
        # bank-starved YoP still consumes the card (no fixed point).
        self.devCardPlayedThisTurn = True
        granted = []
        for r in (r1, r2):
            if board.resourceBank.get(r, 0) > 0:
                self.resources[r] += 1
                board.bank_draw({r: 1})
                granted.append(r)
        if getattr(self, "game", None) is not None:
            self.game.log_yop(self, granted)

    def _monopoly_pick(self):
        """Resource to monopolise, or None when the steal is not worth a card.

        Worth it when it completes a build, or when it nets >= 3 cards.
        """
        game = getattr(self, "game", None)
        if game is None:
            return None
        holdings = {}
        for other in list(game.playerQueue.queue):
            if other is self:
                continue
            for r, n in other.resources.items():
                holdings[r] = holdings.get(r, 0) + n
        if not holdings:
            return None
        wanted = {}
        for kind in ("CITY", "SETTLEMENT", "DEVCARD", "ROAD"):
            for r, n in self._shortfall(self._BUILD_COSTS[kind]).items():
                wanted[r] = max(wanted.get(r, 0), n)
        best, best_gain = None, 0
        for r, gain in holdings.items():
            completes = gain >= wanted.get(r, 10**6)
            if gain >= 3 or completes:
                if gain > best_gain:
                    best, best_gain = r, gain
        return best

    def heuristic_play_monopoly(self, board, resource):
        """Play Monopoly on ``resource`` (engine resource name)."""
        del board  # monopoly moves cards between hands; the bank is untouched
        self.devCards["MONOPOLY"] -= 1
        self.monopolyPlayed += 1
        self.devCardPlayedThisTurn = True
        game = self.game
        total_stolen = 0
        for other in list(game.playerQueue.queue):
            if other is self:
                continue
            stolen = other.resources.get(resource, 0)
            if stolen > 0:
                other.resources[resource] = 0
                self.resources[resource] += stolen
                total_stolen += stolen
                game.broadcast.resource_change(other.name, {resource: -stolen}, "MONOPOLY")
                game.broadcast.resource_change(self.name, {resource: +stolen}, "MONOPOLY")
        game.broadcast.monopoly(self.name, resource, total_stolen)

    def _road_builder_is_useful(self, board):
        """Road Builder pays off when two free roads can actually extend our
        network (and therefore our longest road)."""
        if self.roadsLeft < 1:
            return False
        return bool(board.get_potential_roads(self))

    def heuristic_play_road_builder(self, board):
        """Play Road Builder: two free roads, then recompute longest road."""
        self.devCards["ROADBUILDER"] -= 1
        self.roadBuilderPlayed += 1
        self.devCardPlayedThisTurn = True
        for _ in range(2):
            possibleRoads = board.get_potential_roads(self)
            if not possibleRoads or self.roadsLeft < 1:
                break
            edges = list(possibleRoads.keys())
            chosen = edges[np.random.randint(0, len(edges))]
            self.build_road(chosen[0], chosen[1], board, is_free=True)
        if getattr(self, "game", None) is not None:
            self.game.check_longest_road(self)

    def heuristic_play_dev_card(self, board):
        """Choose and play at most one development card this turn.

        Priority: unblock our own production (Knight) > complete a build
        (Year of Plenty, then Monopoly) > expand (Road Builder) > deny the
        opponent (Knight). Returns True if a card was played.
        """
        if self.devCardPlayedThisTurn:
            return False

        if self.devCards.get("KNIGHT", 0) >= 1 and self._robber_sits_on_own_hex(board):
            self.heuristic_play_knight(board)
            return True

        if self.devCards.get("YEAROFPLENTY", 0) >= 1:
            picks = self._yop_picks(board)
            if picks is not None:
                self.heuristic_play_year_of_plenty(board, picks[0], picks[1])
                return True

        if self.devCards.get("MONOPOLY", 0) >= 1:
            resource = self._monopoly_pick()
            if resource is not None:
                self.heuristic_play_monopoly(board, resource)
                return True

        if self.devCards.get("ROADBUILDER", 0) >= 1 and self._road_builder_is_useful(board):
            self.heuristic_play_road_builder(board)
            return True

        if self._knight_is_useful(board):
            self.heuristic_play_knight(board)
            return True

        return False

    def resources_needed_for_settlement(self):
        """Function to return the resources needed for a settlement
        args: player object - use self.resources
        returns: list of resources needed for a settlement
        """
        resourcesNeededDict = {}
        for resourceName in self.resources.keys():
            if resourceName != "ORE" and self.resources[resourceName] == 0:
                resourcesNeededDict[resourceName] = 1

        return resourcesNeededDict

    def resources_needed_for_city(self):
        """Function to return the resources needed for a city
        args: player object - use self.resources
        returns: list of resources needed for a city
        """
        resourcesNeededDict = {}
        if self.resources["ORE"] < 3:
            resourcesNeededDict["ORE"] = 3 - self.resources["ORE"]

        if self.resources["WHEAT"] < 2:
            resourcesNeededDict["ORE"] = 2 - self.resources["WHEAT"]

        return resourcesNeededDict

    def discardResources(self, game):
        """Overridden function for AI to discard cards"""
        maxCards = 9
        totalResourceCount = sum(self.resources.values())

        if totalResourceCount > maxCards:
            numCardsToDiscard = int(totalResourceCount / 2)
            # print("\nAI Player {} has {} cards and discards {} cards...".format(
            #     self.name, totalResourceCount, numCardsToDiscard))

            # Simple heuristic: discard random cards
            # Create a list of all resources
            all_resources = []
            for res, count in self.resources.items():
                all_resources.extend([res] * count)

            # Randomly choose cards to discard
            discarded = np.random.choice(all_resources, numCardsToDiscard, replace=False)

            discarded_resources = []
            for res in discarded:
                self.resources[res] -= 1
                game.board.bank_recirculate({res: 1})  # spec 009: discard -> bank
                discarded_resources.append(res)
                # print("AI Discarded:", res)

            game.log_discard(self, discarded_resources)

        else:
            pass
            # print("\nAI Player {} has {} cards and does not need to discard!".format(
            #     self.name, totalResourceCount))

    def heuristic_discard(self):
        """Function for the AI to choose a set of cards to discard upon rolling a 7"""
        return

    # Function to propose a trade -> give r1 and get r2
    # Propose a trade as a dictionary with {r1:amt_1, r2: amt_2} specifying the trade
    # def propose_trade_with_players(self):

    # Function to accept/reject trade - return True if accept
    # def accept_trade(self, r1_dict, r2_dict):

    # Function to find best action - based on gamestate

    def get_action(self):
        return

    # Function to execute the player's action
    def execute_action(self):
        return
