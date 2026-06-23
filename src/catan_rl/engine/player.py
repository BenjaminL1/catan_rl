# Settlers of Catan
# Player class implementation

import numpy as np

from catan_rl.engine.board import *

# Class definition for a player


class player:
    "Class Definition for Game Player"

    # R1 INVARIANT: do not add ``__eq__`` / ``__hash__`` to this class.
    # ``engine/dice.py`` shim relies on identity-based ``!=`` to
    # decide whether Karma fires (``last_7_roller_obj is not current``).
    # If a future PR adds value-equality, the Rust StackedDice
    # shim must be updated in lockstep.

    # Initialize a game player, we use A, B and C to identify
    def __init__(self, playerName, playerColor):
        self.name = playerName
        self.color = playerColor
        self.victoryPoints = 0
        self.isAI = False
        # Optional back-reference to owning game; set by catanGame / env.
        self.game = None

        self.settlementsLeft = 5
        self.roadsLeft = 15
        self.citiesLeft = 4
        # Dictionary that keeps track of resource amounts
        self.resources = {"ORE": 0, "BRICK": 0, "WHEAT": 0, "WOOD": 0, "SHEEP": 0}

        self.knightsPlayed = 0
        self.monopolyPlayed = 0
        self.yopPlayed = 0
        self.roadBuilderPlayed = 0
        self.largestArmyFlag = False

        self.maxRoadLength = 0
        self.longestRoadFlag = False

        # Undirected Graph to keep track of which vertices and edges player has colonised
        # Every time a player's build graph is updated the gameBoardGraph must also be updated

        # Each of the 3 lists store vertex information - Roads are stores with tuples of vertex pairs
        self.buildGraph = {"ROADS": [], "SETTLEMENTS": [], "CITIES": []}
        self.portList = []  # List of ports acquired

        # Dev cards in possession
        # List to keep the new dev cards draw - update the main list every turn
        self.newDevCards = []
        self.devCards = {"KNIGHT": 0, "VP": 0, "MONOPOLY": 0, "ROADBUILDER": 0, "YEAROFPLENTY": 0}
        self.devCardPlayedThisTurn = False

        self.visibleVictoryPoints = self.victoryPoints - self.devCards["VP"]

    # function to build a road from vertex v1 to vertex v2

    def build_road(self, v1, v2, board, is_free=False):
        "Update buildGraph to add a road on edge v1 - v2"

        # Check if player has resources available
        if is_free or (self.resources["BRICK"] > 0 and self.resources["WOOD"] > 0):
            if self.roadsLeft > 0:  # Check if player has roads left
                self.buildGraph["ROADS"].append((v1, v2))
                self.roadsLeft -= 1

                if not is_free:
                    # Update player resources
                    self.resources["BRICK"] -= 1
                    self.resources["WOOD"] -= 1
                    # spec 009: a PAID road returns its cost to the finite bank
                    # (a free Road-Builder road recirculates nothing).
                    board.bank_recirculate({"BRICK": 1, "WOOD": 1})
                    # Emit resource change event if game/broadcast is available
                    if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
                        self.game.broadcast.resource_change(
                            self.name,
                            delta={"BRICK": -1, "WOOD": -1},
                            source="BUILD_ROAD",
                        )

                # update the overall boardGraph
                board.updateBoardGraph_road(v1, v2, self)

                # Phase 0.5: structural-side build event so the replay
                # recorder can attribute the road to a specific edge.
                # The edge's integer index is resolved by the env (the
                # engine itself keys edges by pixel-coord tuples); the
                # recorder bridges to env-side edge_to_idx when consuming.
                if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
                    self.game.broadcast.build(self.name, kind="ROAD", location=-1)

                # Calculate current max road length and update. Main-phase builds
                # also recompute this inside game.check_longest_road (global pass),
                # so this is redundant there — but it is the ONLY refresh during
                # SETUP (setup road builds have no following check_longest_road),
                # and the obs reads maxRoadLength. Do not remove.
                maxRoads = self.get_road_length(board)
                self.maxRoadLength = maxRoads

                # print('{} Built a Road. MaxRoadLength: {}'.format(
                #     self.name, self.maxRoadLength))

            else:
                pass
                # print("No roads available to build")

        else:
            pass
            # print("Insufficient Resources to Build Road - Need 1 BRICK, 1 WOOD")

    # function to build a settlement on vertex with coordinates vCoord

    def build_settlement(self, vCoord, board, is_free=False):
        "Update player buildGraph and boardgraph to add a settlement on vertex v"
        # Take input from Player on where to build settlement
        # Check if player has correct resources
        # Update player resources and boardGraph with transaction

        # Check if player has resources available
        if is_free or (
            self.resources["BRICK"] > 0
            and self.resources["WOOD"] > 0
            and self.resources["SHEEP"] > 0
            and self.resources["WHEAT"] > 0
        ):
            if self.settlementsLeft > 0:  # Check if player has settlements left
                self.buildGraph["SETTLEMENTS"].append(vCoord)
                self.settlementsLeft -= 1

                if not is_free:
                    # Update player resources
                    self.resources["BRICK"] -= 1
                    self.resources["WOOD"] -= 1
                    self.resources["SHEEP"] -= 1
                    self.resources["WHEAT"] -= 1
                    # spec 009: settlement cost returns to the finite bank.
                    board.bank_recirculate({"BRICK": 1, "WOOD": 1, "SHEEP": 1, "WHEAT": 1})
                    if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
                        self.game.broadcast.resource_change(
                            self.name,
                            delta={"BRICK": -1, "WOOD": -1, "SHEEP": -1, "WHEAT": -1},
                            source="BUILD_SETTLEMENT",
                        )

                self.victoryPoints += 1
                # update the overall boardGraph
                board.updateBoardGraph_settlement(vCoord, self)

                # Phase 0.5: structural-side build event. ``location``
                # is set to ``-1`` because vertex keys are pixel-tuples
                # at engine level; the env's ``_vertex_to_idx`` map
                # bridges to integer indices and the recorder back-fills
                # the location at observe time.
                if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
                    self.game.broadcast.build(self.name, kind="SETTLEMENT", location=-1)

                # print('{} Built a Settlement'.format(self.name))

                # Add port to players port list if it is a new port
                if (board.boardGraph[vCoord].port != False) and (
                    board.boardGraph[vCoord].port not in self.portList
                ):
                    self.portList.append(board.boardGraph[vCoord].port)
                    # print("{} now has {} Port access".format(
                    #     self.name, board.boardGraph[vCoord].port))

            else:
                pass
                # print("No settlements available to build")

        else:
            pass
            # print(
            #     "Insufficient Resources to Build Settlement. Build Cost: 1 BRICK, 1 WOOD, 1 WHEAT, 1 SHEEP")

    # function to build a city on vertex v
    def build_city(self, vCoord, board):
        "Upgrade existing settlement to city in buildGraph"
        # Check if player has resources available
        if self.resources["WHEAT"] >= 2 and self.resources["ORE"] >= 3:
            if self.citiesLeft > 0:
                self.buildGraph["CITIES"].append(vCoord)
                # Increase number of settlements and decrease number of cities
                self.settlementsLeft += 1
                self.citiesLeft -= 1

                # Update player resources
                self.resources["ORE"] -= 3
                self.resources["WHEAT"] -= 2
                # spec 009: city cost returns to the finite bank.
                board.bank_recirculate({"ORE": 3, "WHEAT": 2})
                if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
                    self.game.broadcast.resource_change(
                        self.name,
                        delta={"ORE": -3, "WHEAT": -2},
                        source="BUILD_CITY",
                    )
                self.victoryPoints += 1

                # update the overall boardGraph
                board.updateBoardGraph_city(vCoord, self)

                # Phase 0.5: structural-side build event for the city
                # upgrade. ``location`` follows the same convention as
                # settle/road (env's vertex_to_idx bridges).
                if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
                    self.game.broadcast.build(self.name, kind="CITY", location=-1)
                # print('{} Built a City'.format(self.name))

            else:
                pass
                # print("No cities available to build")

        else:
            pass
            # print("Insufficient Resources to Build City. Build Cost: 3 ORE, 2 WHEAT")

    # function to move robber to a specific hex and steal from a player
    def move_robber(self, hexIndex, board, player_robbed):
        "Update boardGraph with Robber and steal resource"
        board.updateBoardGraph_robber(hexIndex)

        # Phase 0.5: fire the structural MOVE_ROBBER event BEFORE
        # the steal attempt so subscribers see the move + any
        # resulting STEAL in order.
        if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
            self.game.broadcast.move_robber(self.name, int(hexIndex))

        # Steal a random resource from other players
        self.steal_resource(player_robbed)

        return

    # Function to steal a random resource from player_2

    def steal_resource(self, player_2):
        if player_2 == None:
            # print("No Player on this hex to Rob")
            return

        # Get all resources player 2 has in a list and use random list index to steal
        p2_resources = []
        for resourceName, resourceAmount in player_2.resources.items():
            p2_resources += [resourceName] * resourceAmount

        if len(p2_resources) == 0:
            return

        resourceIndexToSteal = np.random.randint(0, len(p2_resources))

        # Get a random permutation and steal a card
        p2_resources = np.random.permutation(p2_resources)
        resourceStolen = p2_resources[resourceIndexToSteal]

        # Update resources of both players
        player_2.resources[resourceStolen] -= 1
        self.resources[resourceStolen] += 1

        # Broadcast resource changes for robber and victim if possible
        if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
            self.game.broadcast.resource_change(
                player_2.name,
                delta={resourceStolen: -1},
                source="STEAL",
            )
            self.game.broadcast.resource_change(
                self.name,
                delta={resourceStolen: +1},
                source="STEAL",
            )
            # Phase 0.5: structural STEAL event carrying the actual
            # resource (omniscient — the replay system records
            # perfect-information state, so a "?" is unnecessary here).
            self.game.broadcast.steal(
                robber_name=self.name,
                victim_name=player_2.name,
                resource=str(resourceStolen),
            )
        # print("Stole 1 {} from Player {}".format(
        #     resourceStolen, player_2.name))

        return

    def get_road_length(self, board):
        """Longest continuous road (the Longest-Road VP input) = the longest
        TRAIL over the player's road segments: each segment (edge) is traversed
        at most once, but vertices MAY be revisited, so a closed loop counts in
        full (e.g. a ringed hex's 6 segments + a spur = 7). The road is BROKEN at
        any vertex carrying an OPPONENT settlement/city — the trail cannot
        continue THROUGH it — but it passes freely through empty vertices and the
        player's own buildings.
        """
        roads = self.buildGraph["ROADS"]
        if not roads:
            return 0

        # Adjacency keyed by the player's own segments; the integer edge index
        # enforces "each segment once" (vertices are unconstrained -> trail).
        adjacency = {}  # vertex -> list of (neighbor_vertex, edge_index)
        for idx, (a, b) in enumerate(roads):
            adjacency.setdefault(a, []).append((b, idx))
            adjacency.setdefault(b, []).append((a, idx))

        boardGraph = board.boardGraph

        def can_continue_through(vertex):
            owner = boardGraph[vertex].owner
            return owner is None or owner is self  # an opponent building breaks here

        best = 0

        def walk(vertex, used_edges, length):
            nonlocal best
            if length > best:
                best = length
            if not can_continue_through(vertex):
                return  # road broken at an opponent settlement/city (segment in still counts)
            for neighbor, edge_idx in adjacency[vertex]:
                if edge_idx not in used_edges:
                    used_edges.add(edge_idx)
                    walk(neighbor, used_edges, length + 1)
                    used_edges.discard(edge_idx)  # backtrack so siblings explore freely

        # A trail's endpoint is a vertex; starting from every vertex covers all
        # trails (incl. pure cycles, which have no degree-1 endpoint).
        for start in adjacency:
            walk(start, set(), 0)
        return best

    # function to end turn

    def end_turn():
        "Pass turn to next player and update game state"

    # function to draw a Development Card
    def draw_devCard(self, board):
        "Draw a random dev card from stack and update self.devcards"
        # Check if player has resources available
        if (
            self.resources["WHEAT"] >= 1
            and self.resources["ORE"] >= 1
            and self.resources["SHEEP"] >= 1
        ):
            # Get alldev cards available
            devCardsToDraw = []
            for cardName, cardAmount in board.devCardStack.items():
                devCardsToDraw += [cardName] * cardAmount

            # IF there are no devCards left
            if devCardsToDraw == []:
                # print("No Dev Cards Left!")
                return

            devCardIndex = np.random.randint(0, len(devCardsToDraw))

            # Get a random permutation and draw a card
            devCardsToDraw = np.random.permutation(devCardsToDraw)
            cardDrawn = devCardsToDraw[devCardIndex]

            # Update player resources
            self.resources["ORE"] -= 1
            self.resources["WHEAT"] -= 1
            self.resources["SHEEP"] -= 1
            # spec 009: dev-card resource cost returns to the finite bank (the
            # dev card itself comes from the separate devCardStack supply).
            board.bank_recirculate({"ORE": 1, "WHEAT": 1, "SHEEP": 1})
            if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
                self.game.broadcast.resource_change(
                    self.name,
                    delta={"ORE": -1, "WHEAT": -1, "SHEEP": -1},
                    source="BUY_DEV_CARD",
                )

            # If card is a victory point apply immediately, else add to new card list
            if cardDrawn == "VP":
                self.victoryPoints += 1
                board.devCardStack[cardDrawn] -= 1
                self.devCards[cardDrawn] += 1
                self.visibleVictoryPoints = self.victoryPoints - self.devCards["VP"]

            else:  # Update player dev card and the stack
                self.newDevCards.append(cardDrawn)
                board.devCardStack[cardDrawn] -= 1

            # print("{} drew a {} from Development Card Stack".format(
            #     self.name, cardDrawn))

        else:
            # print("Insufficient Resources for Dev Card. Cost: 1 ORE, 1 WHEAT, 1 SHEEP")
            pass

    # Function to update dev card stack with dev cards drawn from prior turn
    def updateDevCards(self):
        for newCard in self.newDevCards:
            self.devCards[newCard] += 1

        # Reset the new card list to blank
        self.newDevCards = []

    # function to play a development card
    def play_devCard(self, game):
        "Update game state"
        # Check if player can play a devCard this turn
        if self.devCardPlayedThisTurn:
            # print('Already played 1 Dev Card this turn!')
            return

        # Get a list of all the unique dev cards this player can play
        devCardsAvailable = []
        for cardName, cardAmount in self.devCards.items():
            if cardName != "VP" and cardAmount >= 1:  # Exclude Victory points
                devCardsAvailable.append((cardName, cardAmount))

        if devCardsAvailable == []:
            # print("No Development Cards available to play")
            return

        # Use GUI to play the Dev Card
        devCardPlayed = game.boardView.get_dev_card_selection(self)

        if devCardPlayed is None:
            # print("Dev Card selection cancelled")
            return

        self.devCardPlayedThisTurn = True

        # print("Playing Dev Card:", devCardPlayed)
        self.devCards[devCardPlayed] -= 1

        # Logic for each Dev Card
        if devCardPlayed == "KNIGHT":
            game.robber(self)
            self.knightsPlayed += 1

        if devCardPlayed == "ROADBUILDER":
            self.roadBuilderPlayed += 1
            game.build(self, "ROAD", is_free=True)
            game.boardView.displayGameScreen()
            game.build(self, "ROAD", is_free=True)
            game.boardView.displayGameScreen()

        # Resource List for Year of Plenty and Monopoly
        resource_list = ["BRICK", "WOOD", "WHEAT", "SHEEP", "ORE"]

        if devCardPlayed == "YEAROFPLENTY":
            self.yopPlayed += 1
            # print("Resources available:", resource_list)
            if self.isAI:
                # AI Logic for YOP - Pick 2 random resources. The np.random.choice
                # call is over the full list (RNG-identical to pre-bank); a pick
                # the bank cannot supply is simply not granted (spec 009; matches
                # the TS reject, and never fires in non-depleting games).
                resources_picked = np.random.choice(resource_list, 2, replace=True)
                yop_resources = []
                for res in resources_picked:
                    if game.board.resourceBank.get(res, 0) > 0:
                        self.resources[res] += 1
                        game.board.bank_draw({res: 1})
                        yop_resources.append(res)
                    # print("AI chose YOP resource:", res)
                game.log_yop(self, yop_resources)
            else:
                # Use GUI to select 2 resources
                result = game.boardView.get_resource_selection(self, "YOP", num_to_select=2)
                if result is None:
                    # Cancelled
                    self.devCards[devCardPlayed] += 1
                    self.devCardPlayedThisTurn = False
                    # print("Year of Plenty cancelled")
                    return
                game.log_yop(self, result)

        if devCardPlayed == "MONOPOLY":
            self.monopolyPlayed += 1
            # print("Resources to Monopolize:", resource_list)
            if self.isAI:
                # AI Logic for Monopoly - Pick random resource
                resourceToMonopolize = np.random.choice(resource_list)
                # print("AI Monopolizing:", resourceToMonopolize)
            else:
                # Use GUI to select resource
                resourceToMonopolize = game.boardView.get_resource_selection(self, "MONOPOLY")

                if resourceToMonopolize is None:
                    # Cancelled
                    self.devCards[devCardPlayed] += 1
                    self.devCardPlayedThisTurn = False
                    # print("Monopoly cancelled")
                    return

            # Loop over each player to Monopolize all resources
            for player in list(game.playerQueue.queue):
                if player != self:
                    numLost = player.resources[resourceToMonopolize]
                    if numLost <= 0:
                        continue
                    player.resources[resourceToMonopolize] = 0
                    self.resources[resourceToMonopolize] += numLost
                    # Broadcast per-player resource changes if broadcaster exists
                    if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
                        # Victim loses numLost of that resource
                        self.game.broadcast.resource_change(
                            player.name,
                            delta={resourceToMonopolize: -numLost},
                            source="MONOPOLY",
                        )
                        # Monopoly player gains numLost
                        self.game.broadcast.resource_change(
                            self.name,
                            delta={resourceToMonopolize: +numLost},
                            source="MONOPOLY",
                        )

        return

    # Function to basic trade 4:1 with bank, or use ports to trade

    def trade_with_bank(self, r1, r2, board):
        """Function to implement trading with bank
        r1: resource player wants to trade away
        r2: resource player wants to receive
        Automatically give player the best available trade ratio

        ``board`` carries the finite resource bank (spec 009): the give side
        (r1 x ratio) recirculates into the bank and the receive side (r2 x 1)
        is drawn from it. If the bank cannot supply ``r2`` the trade is a no-op,
        mirroring the TS ``applyBankTrade`` reject-on-empty-receive.
        """
        # spec 009 apply-time gate: cannot receive a resource the bank lacks.
        if board.resourceBank.get(r2, 0) <= 0:
            return

        def _emit_trade_delta(give: int) -> None:
            # Build the delta as a sum so r1 == r2 trades (degenerate but
            # legal under the action mask) don't lose their give-side
            # value to dict-key collision. The dict-literal form
            # ``{r1: -give, r2: +1}`` produces {r1: +1} when r1 == r2,
            # mis-reporting the actual net change of ``+1 - give`` and
            # silently corrupting any broadcast hand-tracker downstream.
            delta: dict[str, int] = {}
            delta[r1] = delta.get(r1, 0) - give
            delta[r2] = delta.get(r2, 0) + 1
            if getattr(self, "game", None) is not None and hasattr(self.game, "broadcast"):
                self.game.broadcast.resource_change(
                    self.name,
                    delta=delta,
                    source="TRADE_BANK",
                )

        # Get r1 port string
        r1_port = "2:1 " + r1
        # Can use 2:1 port with r1
        if r1_port in self.portList and self.resources[r1] >= 2:
            self.resources[r1] -= 2
            self.resources[r2] += 1
            _emit_trade_delta(2)
            board.bank_recirculate({r1: 2})  # spec 009: give side -> bank
            board.bank_draw({r2: 1})  # spec 009: receive side <- bank
            # print("Traded 2 {} for 1 {} using {} Port".format(r1, r2, r1))
            return

        # Check for 3:1 Port
        elif "3:1 PORT" in self.portList and self.resources[r1] >= 3:
            self.resources[r1] -= 3
            self.resources[r2] += 1
            _emit_trade_delta(3)
            board.bank_recirculate({r1: 3})  # spec 009: give side -> bank
            board.bank_draw({r2: 1})  # spec 009: receive side <- bank
            # print("Traded 3 {} for 1 {} using 3:1 Port".format(r1, r2))
            return

        # Check 4:1 port
        elif self.resources[r1] >= 4:
            self.resources[r1] -= 4
            self.resources[r2] += 1
            _emit_trade_delta(4)
            board.bank_recirculate({r1: 4})  # spec 009: give side -> bank
            board.bank_draw({r2: 1})  # spec 009: receive side <- bank
            # print("Traded 4 {} for 1 {}".format(r1, r2))
            return

        else:
            # print("Insufficient resource {} to trade with Bank".format(r1))
            return

    # Function to initate a trade - with bank or other players

    def initiate_trade(self, game, trade_type):
        """Wrapper function to initiate a trade with the bank.

        Player-to-player trading has been removed for the 1v1 RL setting.
        """
        if trade_type != "BANK":
            # Player-to-player trades and other modes are disabled.
            return

        # Use GUI to select bank trade in human games
        trade_result = game.boardView.get_resource_selection(self, "BANK")
        if trade_result:
            resourceToTrade, resourceToReceive = trade_result
            # Try and trade with Bank
            self.trade_with_bank(resourceToTrade, resourceToReceive, game.board)

        return

    # Function to discard cards

    def discardResources(self, game):
        """Function to enable a player to select cards to discard when a 7 is rolled"""
        maxCards = 9  # Default is 7, but can be changed for testing

        # Calculate resources to discard
        totalResourceCount = 0
        for resource, amount in self.resources.items():
            totalResourceCount += amount

        # Logic to calculate number of cards to discard and allow player to select
        if totalResourceCount > maxCards:
            numCardsToDiscard = int(totalResourceCount / 2)
            # print("\nPlayer {} has {} cards and MUST choose {} cards to discard...".format(
            #     self.name, totalResourceCount, numCardsToDiscard))

            # Use GUI to discard cards
            discarded_resources = game.boardView.get_resource_selection(
                self, "DISCARD", num_to_select=numCardsToDiscard
            )
            game.log_discard(self, discarded_resources)

        else:
            # print("\nPlayer {} has {} cards and does not need to discard any cards!".format(
            #     self.name, totalResourceCount))
            return
