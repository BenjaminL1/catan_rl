# Settlers of Catan
# Gameplay class with pygame

import copy
import queue
import sys

try:
    import pygame  # type: ignore[import-not-found]
except ImportError:
    pygame = None  # type: ignore[assignment]

from catan_rl.agents.heuristic import *
from catan_rl.engine.board import *
from catan_rl.engine.broadcast import GameBroadcast
from catan_rl.engine.dice import StackedDice
from catan_rl.engine.player import *
from catan_rl.engine.tracker import ResourceTracker


class _HeadlessView:
    """No-op stand-in for the pygame boardView when running headless.

    Lifted out of the ``catanGame.__init__`` body so that ``deepcopy`` can
    safely round-trip a game instance — locally-defined classes can confuse
    ``copy.deepcopy`` on some Python implementations.
    """

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def __deepcopy__(self, memo):
        return self  # stateless singleton-like; sharing is safe.


# Catan gameplay class definition


class catanGame:
    # Create new gameboard
    def __init__(self, render_mode="human"):
        # print("Initializing Settlers of Catan Board...")
        self.board = catanBoard()
        self.dice = StackedDice()
        self.last_player_to_roll_7 = None
        # Centralised broadcaster for all game events (resource changes, dice, etc.)
        self.broadcast = GameBroadcast()
        # Backwards-compat: mirror last event on the game object
        self.last_broadcast_event = None
        self.resource_tracker = None  # Will be initialized after players are set up

        # Game State variables
        self.gameOver = False
        self.maxPoints = 15
        self.numPlayers = 2

        # Initialize blank player queue and initial set up of roads + settlements
        self.playerQueue = queue.Queue(self.numPlayers)
        self.gameSetup = True  # Boolean to take care of setup phase
        self.currentPlayer = None  # Keep track of current player

        # Initialize boardview object
        if render_mode == "human":
            from catan_rl.gui.view import catanGameView

            self.boardView = catanGameView(self.board, self)
        else:
            self.boardView = _HeadlessView()

        # Run functions to view board and vertex graph
        # self.board.printGraph()

        # Functiont to go through initial set up
        if render_mode == "human":
            self.build_initial_settlements()
            # Display initial board
            self.boardView.displayGameScreen()
        else:
            self.setup_players()

    def __deepcopy__(self, memo):
        # queue.Queue holds a `_thread.lock`, which copy.deepcopy refuses to
        # pickle. We snapshot the queue's items, deepcopy the rest of the
        # state through the shared memo (preserves the player <-> game cycle),
        # and rebuild a fresh Queue on the copy.
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        queue_items = list(self.playerQueue.queue) if self.playerQueue is not None else []
        for key, value in self.__dict__.items():
            if key == "playerQueue":
                continue  # rebuilt below from the snapshot
            setattr(new, key, copy.deepcopy(value, memo))
        new.playerQueue = queue.Queue(self.numPlayers)
        for item in copy.deepcopy(queue_items, memo):
            new.playerQueue.put(item)
        return new

    def copy(self) -> "catanGame":
        """Return an independent deep copy of the current game state.

        Required by the AlphaZero-style MCTS in ``catan_rl/search/`` — search
        needs to advance a hypothetical line of play without mutating the live
        game. ``pygame`` boardViews can't be deepcopied; this method requires
        the game to be running headless (``render_mode != 'human'``).

        Reference behavior: the returned instance shares NO mutable state
        with the original — stepping it does not affect ``self`` and vice
        versa. The dice's internal bag (``StackedDice.bag``) is a plain list
        and round-trips through ``deepcopy`` cleanly; the module-level
        ``random`` module is not part of the deepcopied state, so consumed
        randomness between original and copy will diverge as expected.
        """
        if not isinstance(self.boardView, _HeadlessView):
            raise RuntimeError(
                "catanGame.copy() is only supported when running headless "
                "(render_mode != 'human'); the pygame boardView holds "
                "unpicklable surface references."
            )
        return copy.deepcopy(self)

    def setup_players(self):
        playerColors = ["black", "darkslateblue", "magenta4", "orange1"]
        for i in range(self.numPlayers):
            newPlayer = player(f"Player {i + 1}", playerColors[i])
            newPlayer.game = self
            self.playerQueue.put(newPlayer)

    # Function to initialize players + build initial settlements for players

    def build_initial_settlements(self):
        # Initialize new players with names and colors
        playerColors = ["black", "darkslateblue", "magenta4", "orange1"]
        for i in range(self.numPlayers - 1):
            # playerNameInput = input("Enter Player {} name: ".format(i+1))
            playerNameInput = f"Player {i + 1}"
            newPlayer = player(playerNameInput, playerColors[i])
            newPlayer.game = self
            self.playerQueue.put(newPlayer)

        test_AI_player = heuristicAIPlayer(
            "Random-Greedy-AI", playerColors[self.numPlayers - 1]
        )  # Add the AI Player last
        test_AI_player.game = self
        test_AI_player.updateAI()
        self.playerQueue.put(test_AI_player)

        playerList = list(self.playerQueue.queue)

        # Initialize Resource Tracker
        self.resource_tracker = ResourceTracker([p.name for p in playerList])

        self.boardView.displayGameScreen()  # display the initial gameScreen
        # print("Displaying Initial GAMESCREEN!")

        # Build Settlements and roads of each player forwards
        for player_i in playerList:
            self.currentPlayer = player_i
            if player_i.isAI:
                print(f"DEBUG: Calling AI {player_i.name} setup (Forward)...", flush=True)
                player_i.initial_setup(self.board)

            else:
                print(f"DEBUG: Human {player_i.name} setup (Forward)...", flush=True)
                self.build(player_i, "SETTLE", is_free=True)
                self.boardView.displayGameScreen()

                self.build(player_i, "ROAD", is_free=True)
                self.boardView.displayGameScreen()

        # Build Settlements and roads of each player reverse
        playerList.reverse()
        for player_i in playerList:
            self.currentPlayer = player_i
            if player_i.isAI:
                print(f"DEBUG: Calling AI {player_i.name} setup (Reverse)...", flush=True)
                player_i.initial_setup(self.board)
                self.boardView.displayGameScreen()

            else:
                print(f"DEBUG: Human {player_i.name} setup (Reverse)...", flush=True)
                self.build(player_i, "SETTLE", is_free=True)
                self.boardView.displayGameScreen()

                self.build(player_i, "ROAD", is_free=True)
                self.boardView.displayGameScreen()

            # Initial resource generation
            # check each adjacent hex to latest settlement
            initial_resources = []
            for adjacentHex in self.board.boardGraph[
                player_i.buildGraph["SETTLEMENTS"][-1]
            ].adjacent_hex_indices:
                resourceGenerated = self.board.hexTileDict[adjacentHex].resource_type
                if resourceGenerated != "DESERT":
                    player_i.resources[resourceGenerated] += 1
                    initial_resources.append(resourceGenerated)
                    # print("{} collects 1 {} from Settlement".format(
                    #     player_i.name, resourceGenerated))

            # Track initial resources if tracker exists
            if hasattr(self, "resource_tracker"):
                self.resource_tracker.track_initial_resources(player_i.name, initial_resources)

        self.gameSetup = False

        return

    # Generic function to handle all building in the game - interface with gameView

    def build(self, player, build_flag, is_free=False):
        if build_flag == "ROAD":  # Show screen with potential roads
            if self.gameSetup:
                potentialRoadDict = self.board.get_setup_roads(player)
            else:
                potentialRoadDict = self.board.get_potential_roads(player)

            roadToBuild = self.boardView.buildRoad_display(player, potentialRoadDict)
            if roadToBuild != None:
                player.build_road(roadToBuild[0], roadToBuild[1], self.board, is_free)

        if build_flag == "SETTLE":  # Show screen with potential settlements
            if self.gameSetup:
                potentialVertexDict = self.board.get_setup_settlements(player)
            else:
                potentialVertexDict = self.board.get_potential_settlements(player)

            vertexSettlement = self.boardView.buildSettlement_display(player, potentialVertexDict)
            if vertexSettlement != None:
                player.build_settlement(vertexSettlement, self.board, is_free)

        if build_flag == "CITY":
            potentialCityVertexDict = self.board.get_potential_cities(player)
            vertexCity = self.boardView.buildSettlement_display(player, potentialCityVertexDict)
            if vertexCity != None:
                player.build_city(vertexCity, self.board)

    # Wrapper Function to handle robber functionality

    def robber(self, player):
        potentialRobberDict = self.board.get_robber_spots()
        # print("DEBUG: Move Robber! Click on a hex to move the robber.")

        hex_i, playerRobbed = self.boardView.moveRobber_display(player, potentialRobberDict)
        player.move_robber(hex_i, self.board, playerRobbed)

    # Function to roll dice

    def rollDice(self):
        # Reset last broadcast event each roll; broadcaster keeps full history.
        self.broadcast.clear_last()
        self.last_broadcast_event = None
        diceRoll = self.dice.roll(self.currentPlayer, self.last_player_to_roll_7)
        # print(f"DEBUG: Rolled {diceRoll}")
        if diceRoll == 7:
            self.last_player_to_roll_7 = self.currentPlayer
            # print("DEBUG: Rolled a 7! Robber logic initiated.")
        # print("Dice Roll = ", diceRoll)

        # Broadcast dice roll event
        dice_event = self.broadcast.dice_roll(self.currentPlayer.name, diceRoll)
        self.last_broadcast_event = dice_event

        self.boardView.displayDiceRoll(diceRoll)

        return diceRoll

    def log_discard(self, player, resource_list):
        """Broadcast discard event via GameBroadcast (and mirror on game).

        Also emits a RESOURCE_CHANGE event with negative deltas for each
        discarded resource, so downstream code (e.g. RL env) can track
        hand changes numerically.
        """
        event = self.broadcast.discard(player.name, resource_list)
        self.last_broadcast_event = event

        # Aggregate negative deltas per resource
        delta = {}
        for res in resource_list:
            delta[res] = delta.get(res, 0) - 1
        if delta:
            self.broadcast.resource_change(player.name, delta=delta, source="DISCARD")
        # print(f"BROADCAST: Player {player.name} discarded {resource_list}")

    def log_yop(self, player, resource_list):
        """Broadcast Year of Plenty event via GameBroadcast (and mirror on game).

        Also emits a RESOURCE_CHANGE event with positive deltas for each
        resource the player gained from the bank.
        """
        event = self.broadcast.year_of_plenty(player.name, resource_list)
        self.last_broadcast_event = event

        # Aggregate positive deltas per resource
        delta = {}
        for res in resource_list:
            delta[res] = delta.get(res, 0) + 1
        if delta:
            self.broadcast.resource_change(player.name, delta=delta, source="YOP")
        # print(f"BROADCAST: Player {player.name} used YOP to get {resource_list}")

    # Function to update resources for all players
    def update_playerResources(self, diceRoll, currentPlayer):
        if diceRoll != 7:  # Collect resources if not a 7
            # First get the hex or hexes corresponding to diceRoll
            hexResourcesRolled = self.board.getHexResourceRolled(diceRoll)
            # print('Resources rolled this turn:', hexResourcesRolled)

            # Check for each player
            for player_i in list(self.playerQueue.queue):
                per_player_delta = {}
                # Check each settlement the player has
                for settlementCoord in player_i.buildGraph["SETTLEMENTS"]:
                    # check each adjacent hex to a settlement
                    for adjacentHex in self.board.boardGraph[settlementCoord].adjacent_hex_indices:
                        # This player gets a resource if hex is adjacent and no robber
                        if (
                            adjacentHex in hexResourcesRolled
                            and self.board.hexTileDict[adjacentHex].has_robber == False
                        ):
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource_type
                            player_i.resources[resourceGenerated] += 1
                            per_player_delta[resourceGenerated] = (
                                per_player_delta.get(resourceGenerated, 0) + 1
                            )
                            # print("{} collects 1 {} from Settlement".format(
                            #     player_i.name, resourceGenerated))

                # Check each City the player has
                for cityCoord in player_i.buildGraph["CITIES"]:
                    # check each adjacent hex to a settlement
                    for adjacentHex in self.board.boardGraph[cityCoord].adjacent_hex_indices:
                        # This player gets a resource if hex is adjacent and no robber
                        if (
                            adjacentHex in hexResourcesRolled
                            and self.board.hexTileDict[adjacentHex].has_robber == False
                        ):
                            resourceGenerated = self.board.hexTileDict[adjacentHex].resource_type
                            player_i.resources[resourceGenerated] += 2
                            per_player_delta[resourceGenerated] = (
                                per_player_delta.get(resourceGenerated, 0) + 2
                            )
                            # print("{} collects 2 {} from City".format(
                            #     player_i.name, resourceGenerated))

                # Emit a RESOURCE_CHANGE broadcast for this player, if any deltas
                if per_player_delta:
                    self.broadcast.resource_change(
                        player_i.name, delta=per_player_delta, source="DICE"
                    )

        # Logic for a 7 roll
        else:
            # Implement discarding cards
            # Check for each player
            for player_i in list(self.playerQueue.queue):
                # Player must discard resources
                player_i.discardResources(self)

            # Logic for robber
            if currentPlayer.isAI:
                if hasattr(currentPlayer, "heuristic_move_robber"):
                    # print("AI using heuristic robber...")
                    currentPlayer.heuristic_move_robber(self.board)
                else:
                    # Assuming PPO/Model Agent will handle robber movement in its move() function
                    pass
            else:
                self.robber(currentPlayer)
                self.boardView.displayGameScreen()  # Update back to original gamescreen

    # function to check if a player has the longest road - after building latest road

    def check_longest_road(self, player_i):
        """Re-evaluate the Longest Road bonus after any build that can change
        road lengths — a road build EXTENDS a road, an opponent settlement can
        SPLIT one. RECOMPUTES every player's road length first (so a split is
        modelled, not just an extension), then applies the Colonist rules:

        * a player TAKES the 2-VP card only with road length >= 5 AND strictly
          longer than everyone else;
        * the current holder KEEPS it on a tie (ties never transfer);
        * it is REVOKED when a split drops the holder below 5 or below another
          player — passing to a new SOLE leader (>=5), or to nobody if the new
          lead is tied.

        ``player_i`` is the builder that triggered the check (kept for call-site
        compatibility); the evaluation itself is global. Emits
        ``longest_road_change`` only on an actual holder change (incl. revocation
        to no holder).
        """
        players = list(self.playerQueue.queue)
        for p in players:
            p.maxRoadLength = p.get_road_length(self.board)

        holder = next((p for p in players if p.longestRoadFlag), None)
        eligible = [p for p in players if p.maxRoadLength >= 5]
        max_len = max((p.maxRoadLength for p in eligible), default=0)
        leaders = [p for p in eligible if p.maxRoadLength == max_len]

        # Holder keeps the card while still (tied for) the longest at >= 5.
        if holder is not None and holder in leaders:
            return
        # Otherwise the holder (if any) has lost it. A sole leader at >= 5 takes
        # it; a tie among non-holders leaves the card unheld.
        new_holder = leaders[0] if len(leaders) == 1 else None
        if holder is new_holder:
            return  # no change (e.g. no holder + a tie -> still nobody)

        if holder is not None:
            holder.longestRoadFlag = False
            holder.victoryPoints -= 2
        if new_holder is not None:
            new_holder.longestRoadFlag = True
            new_holder.victoryPoints += 2
        self.broadcast.longest_road_change(
            prev_owner=holder.name if holder is not None else None,
            new_owner=new_holder.name if new_holder is not None else None,
            length=int(max_len) if new_holder is not None else 0,
        )

    # function to check if a player has the largest army - after playing latest knight
    def check_largest_army(self, player_i):
        if player_i.knightsPlayed >= 3:  # Only eligible if at least 3 knights are player
            largestArmy = True
            for p in list(self.playerQueue.queue):
                # Check if any other players have more knights played
                if p.knightsPlayed >= player_i.knightsPlayed and p != player_i:
                    largestArmy = False

            # if player_i takes largest army and didn't already have it
            if largestArmy and player_i.largestArmyFlag == False:
                # Set previous players flag to false and give player_i the largest points
                prev_owner_name: str | None = None
                for p in list(self.playerQueue.queue):
                    if p.largestArmyFlag:
                        p.largestArmyFlag = False
                        p.victoryPoints -= 2
                        prev_owner_name = p.name

                player_i.largestArmyFlag = True
                player_i.victoryPoints += 2

                # Phase 0.5: emit only on actual holder change.
                self.broadcast.largest_army_change(
                    prev_owner=prev_owner_name,
                    new_owner=player_i.name,
                    knights=int(player_i.knightsPlayed),
                )

                # print("Player {} takes Largest Army {}".format(
                #     player_i.name, prevPlayer))

    # Function that runs the main game loop with all players and pieces

    def playCatan(self):
        # self.board.displayBoard() #Display updated board
        clock = pygame.time.Clock()

        while self.gameOver == False:
            # Loop for each player's turn -> iterate through the player queue
            for currPlayer in self.playerQueue.queue:
                self.currentPlayer = currPlayer

                # print(
                #     "---------------------------------------------------------------------------")
                # print("Current Player:", currPlayer.name)

                turnOver = False  # boolean to keep track of turn
                diceRolled = False  # Boolean for dice roll status

                # Update Player's dev card stack with dev cards drawn in previous turn and reset devCardPlayedThisTurn
                currPlayer.updateDevCards()
                currPlayer.devCardPlayedThisTurn = False
                self.boardView.displayGameScreen()
                while turnOver == False:
                    clock.tick(60)

                    # TO-DO: Add logic for AI Player to move
                    # TO-DO: Add option of AI Player playing a dev card prior to dice roll
                    if currPlayer.isAI:
                        # Roll Dice
                        diceNum = self.rollDice()
                        diceRolled = True
                        self.update_playerResources(diceNum, currPlayer)

                        # AI Player makes all its moves
                        currPlayer.move(self.board)
                        # Check if AI player gets longest road/largest army and update Victory points
                        self.check_longest_road(currPlayer)
                        self.check_largest_army(currPlayer)
                        print(
                            f"Player:{currPlayer.name}, Resources:{currPlayer.resources}, Points: {currPlayer.victoryPoints}"
                        )

                        self.boardView.displayGameScreen()  # Update back to original gamescreen
                        turnOver = True

                    else:  # Game loop for human players
                        for e in pygame.event.get():  # Get player actions/in-game events
                            # print(e)
                            if e.type == pygame.QUIT:
                                sys.exit(0)

                            # Check mouse click in rollDice
                            if e.type == pygame.MOUSEBUTTONDOWN:
                                print(f"DEBUG: Mouse Click at {e.pos}")
                                # Check if player rolled the dice
                                if self.boardView.rollDice_button.collidepoint(e.pos):
                                    print("DEBUG: Clicked Roll Dice Button")
                                    if diceRolled == False:  # Only roll dice once
                                        diceNum = self.rollDice()
                                        diceRolled = True

                                        self.boardView.displayDiceRoll(diceNum)
                                        # Code to update player resources with diceNum
                                        self.update_playerResources(diceNum, currPlayer)
                                        self.boardView.displayGameScreen()
                                    else:
                                        print("DEBUG: Dice already rolled this turn")

                                # Check if player wants to build road
                                if self.boardView.buildRoad_button.collidepoint(e.pos):
                                    print("DEBUG: Clicked Build Road Button")
                                    # Code to check if road is legal and build
                                    if diceRolled == True:  # Can only build after rolling dice
                                        self.build(currPlayer, "ROAD")

                                        # Check if player gets longest road and update Victory points
                                        self.check_longest_road(currPlayer)

                                        self.boardView.displayGameScreen()  # Update back to original gamescreen
                                        # Show updated points and resources
                                        print(
                                            f"Player:{currPlayer.name}, Resources:{currPlayer.resources}, Points: {currPlayer.victoryPoints}"
                                        )
                                    else:
                                        print("DEBUG: Must roll dice before building")

                                # Check if player wants to build settlement
                                if self.boardView.buildSettlement_button.collidepoint(e.pos):
                                    print("DEBUG: Clicked Build Settlement Button")
                                    # Can only build settlement after rolling dice
                                    if diceRolled == True:
                                        self.build(currPlayer, "SETTLE")
                                        # A settlement can SPLIT an opponent's road -> re-evaluate
                                        # Longest Road (recompute + revoke), mirroring the ROAD button.
                                        self.check_longest_road(currPlayer)
                                        self.boardView.displayGameScreen()  # Update back to original gamescreen
                                        # Show updated points and resources
                                        print(
                                            f"Player:{currPlayer.name}, Resources:{currPlayer.resources}, Points: {currPlayer.victoryPoints}"
                                        )

                                # Check if player wants to build city
                                if self.boardView.buildCity_button.collidepoint(e.pos):
                                    print("DEBUG: Clicked Build City Button")
                                    if diceRolled == True:  # Can only build city after rolling dice
                                        self.build(currPlayer, "CITY")
                                        self.boardView.displayGameScreen()  # Update back to original gamescreen
                                        # Show updated points and resources
                                        print(
                                            f"Player:{currPlayer.name}, Resources:{currPlayer.resources}, Points: {currPlayer.victoryPoints}"
                                        )

                                # Check if player wants to draw a development card
                                if self.boardView.devCard_button.collidepoint(e.pos):
                                    print("DEBUG: Clicked Dev Card Button")
                                    if (
                                        diceRolled == True
                                    ):  # Can only draw devCard after rolling dice
                                        currPlayer.draw_devCard(self.board)
                                        self.boardView.displayGameScreen()
                                        # Show updated points and resources
                                        print(
                                            f"Player:{currPlayer.name}, Resources:{currPlayer.resources}, Points: {currPlayer.victoryPoints}"
                                        )
                                        print("Available Dev Cards:", currPlayer.devCards)

                                # Check if player wants to play a development card - can play devCard whenever after rolling dice
                                if self.boardView.playDevCard_button.collidepoint(e.pos):
                                    print("DEBUG: Clicked Play Dev Card Button")
                                    currPlayer.play_devCard(self)

                                    # Check for Largest Army and longest road
                                    self.check_largest_army(currPlayer)
                                    self.check_longest_road(currPlayer)

                                    self.boardView.displayGameScreen()  # Update back to original gamescreen

                                    # Show updated points and resources
                                    print(
                                        f"Player:{currPlayer.name}, Resources:{currPlayer.resources}, Points: {currPlayer.victoryPoints}"
                                    )
                                    print("Available Dev Cards:", currPlayer.devCards)

                                # Check if player wants to trade with the bank
                                if self.boardView.tradeBank_button.collidepoint(e.pos):
                                    print("DEBUG: Clicked Trade Bank Button")
                                    currPlayer.initiate_trade(self, "BANK")
                                    self.boardView.displayGameScreen()
                                    # Show updated points and resources
                                    print(
                                        f"Player:{currPlayer.name}, Resources:{currPlayer.resources}, Points: {currPlayer.victoryPoints}"
                                    )

                                # Check if player wants to end turn
                                if self.boardView.endTurn_button.collidepoint(e.pos):
                                    print("DEBUG: Clicked End Turn Button")
                                    if diceRolled == True:  # Can only end turn after rolling dice
                                        print("Ending Turn!")
                                        turnOver = True  # Update flag to nextplayer turn
                                    else:
                                        print("DEBUG: Must roll dice before ending turn")

                    # Update the display
                    # self.displayGameScreen(None, None)
                    pygame.display.update()

                    # Check if game is over
                    if currPlayer.victoryPoints >= self.maxPoints:
                        self.gameOver = True
                        self.turnOver = True
                        print("====================================================")
                        print(f"PLAYER {currPlayer.name} WINS!")
                        print("Exiting game in 10 seconds...")
                        break

                if self.gameOver:
                    startTime = pygame.time.get_ticks()
                    runTime = 0
                    while runTime < 10000:  # 10 second delay prior to quitting
                        runTime = pygame.time.get_ticks() - startTime

                    break

    # ------------------------------------------------------------------
    # Replay-system accessor (Phase 0.5)
    # ------------------------------------------------------------------

    def snapshot_state(
        self,
        seat_to_actor: dict,
        vertex_pixel_to_idx: dict,
        edge_key_to_idx: dict,
    ) -> dict:
        """Return a JSON-safe state snapshot mirroring
        ``replay.schema.StepStateSnapshot``.

        Args:
            seat_to_actor: maps engine player name → JSON actor name
                (``"player_a"`` or ``"player_b"``). The recorder
                builds this once at reset based on ``agent_seat``.
            vertex_pixel_to_idx: maps engine pixel-coord vertex keys
                to the JSON integer ``vertex_idx``. Owned by the env
                in ``catan_env._vertex_to_idx``.
            edge_key_to_idx: maps engine's lex-sorted edge-key tuples
                (the ``(s1, s2)`` shape used in
                ``catan_env._edge_key``) to integer edge_idx.

        The deep-copy contract from Phase 0 (every leaf must be a
        primitive int/str/None, never an engine object reference) is
        upheld here so subsequent engine mutations cannot retroactively
        alter the snapshot.

        A future Rust/C++ engine port only has to reimplement this
        method (and ``board_static`` on the board) — the replay
        recorder calls these accessors and never reaches into engine
        internals.
        """
        import copy

        def _edge_key(v1, v2):
            s1, s2 = str(v1), str(v2)
            return (s1, s2) if s1 < s2 else (s2, s1)

        # Validate ``seat_to_actor`` covers every player in the queue
        # — a silent miss would produce an empty ``players`` dict and
        # the resulting replay would render an empty board in the
        # viewer (per Phase 0.5 review).
        live_names = {p.name for p in list(self.playerQueue.queue)}
        missing = live_names - set(seat_to_actor.keys())
        if missing:
            raise ValueError(
                f"snapshot_state: seat_to_actor missing engine player(s) "
                f"{sorted(missing)}; received keys {sorted(seat_to_actor.keys())}"
            )

        # Per-player state.
        settlements: dict[str, list[int]] = {}
        cities: dict[str, list[int]] = {}
        roads: dict[str, list[int]] = {}
        players_snap: dict[str, dict] = {}
        lr_holder: str | None = None
        la_holder: str | None = None
        for p in list(self.playerQueue.queue):
            actor = seat_to_actor.get(p.name)
            if actor is None:
                continue
            # Settlements + cities — vertex pixel → idx.
            settle_indices: list[int] = []
            for v_px in p.buildGraph.get("SETTLEMENTS", []):
                if v_px in vertex_pixel_to_idx:
                    settle_indices.append(int(vertex_pixel_to_idx[v_px]))
            city_indices: list[int] = []
            for v_px in p.buildGraph.get("CITIES", []):
                if v_px in vertex_pixel_to_idx:
                    city_indices.append(int(vertex_pixel_to_idx[v_px]))
            # A vertex with a city has had its settlement upgraded —
            # the engine keeps the original entry in SETTLEMENTS, so
            # we de-dup here for the JSON convention "settlements and
            # cities are disjoint sets".
            settle_indices = [i for i in settle_indices if i not in city_indices]
            # Roads — edge pixel-pair → idx.
            road_indices: list[int] = []
            for v1, v2 in p.buildGraph.get("ROADS", []):
                key = _edge_key(v1, v2)
                if key in edge_key_to_idx:
                    road_indices.append(int(edge_key_to_idx[key]))
            settlements[actor] = settle_indices
            cities[actor] = city_indices
            roads[actor] = road_indices

            # Resources — deep copy to immutable dict of primitives.
            resources = copy.deepcopy(dict(p.resources))
            # Omniscient dev hand — merge ``devCards`` (dict) +
            # ``newDevCards`` (list[str] of cards drawn this turn) at
            # capture time per Phase 0 review (H2). The engine uses
            # slightly different key shapes for VP cards across
            # vintages; normalise here.
            dev_hand_raw = dict(getattr(p, "devCards", {}) or {})
            raw_new = getattr(p, "newDevCards", []) or []
            if isinstance(raw_new, dict):
                # Future-proof: if a port reshapes newDevCards into a
                # dict, accept that too.
                new_devs_counts: dict[str, int] = dict(raw_new)
            else:
                new_devs_counts = {}
                for card_name in raw_new:
                    key_name = str(card_name)
                    new_devs_counts[key_name] = new_devs_counts.get(key_name, 0) + 1
            legacy_keys = {
                "ROAD_BUILDER": "ROADBUILDER",
                "YEAR_OF_PLENTY": "YEAROFPLENTY",
            }
            # VP cards live in ``dev_cards_played`` ONLY (per the
            # schema's ``PlayerStateSnapshot.dev_cards_played``
            # docstring: "VP here is the count of VP cards the
            # player owns — treated as 'played' since they're
            # permanent points"). The engine keeps them in
            # ``player.devCards["VP"]`` permanently, so we report
            # ``dev_cards_hand["VP"] = 0`` to avoid double-counting.
            dev_hand: dict[str, int] = {"VP": 0}
            for key in (
                "KNIGHT",
                "ROAD_BUILDER",
                "YEAR_OF_PLENTY",
                "MONOPOLY",
            ):
                lookup = legacy_keys.get(key, key)
                dev_hand[key] = int(dev_hand_raw.get(lookup, dev_hand_raw.get(key, 0))) + int(
                    new_devs_counts.get(lookup, new_devs_counts.get(key, 0))
                )
            dev_played = {
                "KNIGHT": int(getattr(p, "knightsPlayed", 0)),
                "VP": int(dev_hand_raw.get("VP", 0)),
                "ROAD_BUILDER": int(getattr(p, "roadBuilderPlayed", 0)),
                "YEAR_OF_PLENTY": int(getattr(p, "yopPlayed", 0)),
                "MONOPOLY": int(getattr(p, "monopolyPlayed", 0)),
            }

            players_snap[actor] = {
                "name": str(p.name),
                "vp": int(p.victoryPoints),
                "resources": resources,
                "dev_cards_hand": dev_hand,
                "dev_cards_played": dev_played,
            }

            if getattr(p, "longestRoadFlag", False):
                lr_holder = actor
            if getattr(p, "largestArmyFlag", False):
                la_holder = actor

        # Robber location — scan board.hexTileDict for has_robber.
        robber_hex = -1
        for hex_idx, tile in self.board.hexTileDict.items():
            if getattr(tile, "has_robber", False):
                robber_hex = int(hex_idx)
                break

        # Karma armed-against-whom: persistent ``last_player_to_roll_7``
        # is the OTHER player's identity for Karma purposes. Map it
        # through ``seat_to_actor`` to keep the snapshot
        # engine-implementation-free.
        last_seven = (
            seat_to_actor.get(self.last_player_to_roll_7.name)
            if getattr(self.last_player_to_roll_7, "name", None) is not None
            else None
        )

        return {
            "settlements": settlements,
            "cities": cities,
            "roads": roads,
            "robber_hex": robber_hex,
            "players": players_snap,
            "longest_road_holder": lr_holder,
            "largest_army_holder": la_holder,
            "last_seven_roller": last_seven,
        }


# Initialize new game and run
if __name__ == "__main__":
    newGame = catanGame()
    newGame.playCatan()
