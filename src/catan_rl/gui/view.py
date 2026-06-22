# Settlers of Catan
# Game view class implementation with pygame

import math
import sys

import pygame

from catan_rl.engine.geometry import *
from catan_rl.gui import render
from catan_rl.gui import render_constants as RC

pygame.init()

# Class to handle catan board display


class catanGameView:
    "Class definition for Catan board display"

    def __init__(self, catanBoardObject, catanGameObject):
        self.board = catanBoardObject
        self.game = catanGameObject

        # #Use pygame to display the board
        self.screen = pygame.display.set_mode(self.board.size)
        pygame.display.set_caption("Settlers of Catan")
        self.font_resource = pygame.font.SysFont("cambria", 15)
        self.font_ports = pygame.font.SysFont("cambria", 10)

        self.font_button = pygame.font.SysFont("cambria", 12)
        self.font_diceRoll = pygame.font.SysFont("cambria", 25)  # dice font
        self.font_Robber = pygame.font.SysFont("arialblack", 50)  # robber font
        self.font_menu = pygame.font.SysFont("cambria", 20)
        self.font_broadcast = pygame.font.SysFont("cambria", 18)  # broadcast font

        self.diceRoll = 0  # Initialize dice roll

        # Optional whose-turn banner: (text, bg_color_name) or None. Drawn by
        # displayGameScreen only when set (additive — engine paths leave it None
        # so rendering is unchanged). Set by interactive harnesses.
        self.turn_banner: tuple[str, str] | None = None

        # Optional: the human's player object. When set, displayPlayerStats always
        # shows THIS player's hand (the human sits in the AI-flagged opponent seat
        # in the vs-bot harness, so the default current-non-AI logic would hide it).
        # Engine playCatan leaves it None -> unchanged behavior.
        self.human_player = None

        # Optional: the bot/opponent player object. When set, displayPlayerStats
        # also shows the OPPONENT's hand SIZE (resource + dev card counts only, not
        # types — mirrors real Catan visibility). Engine playCatan leaves it None.
        self.bot_player = None

        # Optional: friendly display-name overrides keyed by player.name (e.g.
        # {"Opponent": "You", "Agent": "Bot"}). Used by displayPlayerStats and
        # displayBroadcastMessage. Engine playCatan leaves it empty -> raw names.
        self.name_display: dict[str, str] = {}

        return None

    # Function to display the initial board

    def displayInitialBoard(self):
        # Back-compat: external code may still read ``self.colorDict``.
        self.colorDict = dict(RC.TILE_COLORS)

        render.draw_water(self.screen, (self.board.width, self.board.height))
        hex_centers = [self.board.hexTileDict[i].to_pixel(self.board.flat) for i in range(19)]
        render.draw_island_outline(self.screen, hex_centers)

        for hexTile in self.board.hexTileDict.values():
            render.draw_hex_tile(self.screen, hexTile, self.board, with_bevel=True)
            center = hexTile.to_pixel(self.board.flat)
            cx_int = int(center.x)
            cy_int = int(center.y)
            render.draw_resource_symbol(
                self.screen,
                (cx_int, cy_int + RC.RESOURCE_SYMBOL_VERTICAL_OFFSET),
                hexTile.resource_type,
            )
            num = getattr(hexTile, "number_token", None)
            if num is not None and num != 0 and hexTile.resource_type != "DESERT":
                render.draw_number_token(
                    self.screen,
                    (cx_int, cy_int + RC.NUMBER_TOKEN_VERTICAL_OFFSET),
                    num,
                )

        board_cx = self.board.width / 2.0
        board_cy = self.board.height / 2.0
        vertex_pixel = self.board.vertex_index_to_pixel_dict
        for v1_idx, v2_idx, ratio, resource in render.collect_port_edges(self.board):
            v1_px = vertex_pixel[v1_idx]
            v2_px = vertex_pixel[v2_idx]
            mid_x = (float(v1_px.x) + float(v2_px.x)) / 2.0
            mid_y = (float(v1_px.y) + float(v2_px.y)) / 2.0
            dx = mid_x - board_cx
            dy = mid_y - board_cy
            # Guard the centroid-degenerate case; ports near the board
            # center are impossible by engine construction but cheap to
            # defend against.
            d = math.hypot(dx, dy) or 1.0
            ax = int(mid_x + dx * RC.PORT_PUSH_DISTANCE / d)
            ay = int(mid_y + dy * RC.PORT_PUSH_DISTANCE / d)
            render.draw_port_planks(
                self.screen,
                (ax, ay),
                (int(v1_px.x), int(v1_px.y)),
                (int(v2_px.x), int(v2_px.y)),
            )
            render.draw_port_ship(self.screen, ratio, resource, (ax, ay))

        pygame.display.update()

        return None

    # Function to draw a road on the board

    def draw_road(self, edgeToDraw, roadColor):
        pygame.draw.line(self.screen, pygame.Color(roadColor), edgeToDraw[0], edgeToDraw[1], 10)

    # Function to draw a potential road on the board - thin

    def draw_possible_road(self, edgeToDraw, roadColor):
        roadRect = pygame.draw.line(
            self.screen, pygame.Color(roadColor), edgeToDraw[0], edgeToDraw[1], 5
        )
        return roadRect

    # Function to draw a settlement on the board at vertexToDraw

    def draw_settlement(self, vertexToDraw, color):
        newSettlement = pygame.Rect(vertexToDraw.x - 10, vertexToDraw.y - 10, 25, 25)
        pygame.draw.rect(self.screen, pygame.Color(color), newSettlement)

    # Function to draw a potential settlement on the board - thin

    def draw_possible_settlement(self, vertexToDraw, color):
        possibleSettlement = pygame.draw.circle(
            self.screen, pygame.Color(color), (int(vertexToDraw.x), int(vertexToDraw.y)), 20, 3
        )
        return possibleSettlement

    # Function to draw a settlement on the board at vertexToDraw

    def draw_city(self, vertexToDraw, color):
        pygame.draw.circle(
            self.screen, pygame.Color(color), (int(vertexToDraw.x), int(vertexToDraw.y)), 24
        )

    # Function to draw a potential settlement on the board - thin

    def draw_possible_city(self, vertexToDraw, color):
        possibleCity = pygame.draw.circle(
            self.screen, pygame.Color(color), (int(vertexToDraw.x), int(vertexToDraw.y)), 25, 5
        )
        return possibleCity

    # Function to draw the possible spots for a robber

    def draw_possible_robber(self, vertexToDraw):
        possibleRobber = pygame.draw.circle(
            self.screen, pygame.Color("black"), (int(vertexToDraw.x), int(vertexToDraw.y)), 50, 5
        )
        return possibleRobber

    # Function to draw possible players to rob
    def draw_possible_players_to_rob(self, vertexCoord):
        possiblePlayer = pygame.draw.circle(
            self.screen, pygame.Color("black"), (int(vertexCoord.x), int(vertexCoord.y)), 35, 5
        )
        return possiblePlayer

    # ------------------------------------------------------------------
    # Animated highlight helpers (Colonist-style pulsating glow on the
    # spots the current player can act on). Additive: only used by the
    # interactive picker loops below.
    # ------------------------------------------------------------------

    def _pulse(self) -> float:
        """0..1 sine pulse keyed to the wall clock (≈0.8 Hz)."""
        return (math.sin(pygame.time.get_ticks() * 0.006) + 1.0) / 2.0

    def _blit_halo(self, x: int, y: int, radius: int, color: str, alpha: int) -> None:
        """Blit a translucent filled circle (a soft glow) centered at (x, y)."""
        radius = max(1, radius)
        surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        c = pygame.Color(color)
        c.a = max(0, min(255, alpha))
        pygame.draw.circle(surf, c, (radius + 2, radius + 2), radius)
        self.screen.blit(surf, (x - radius - 2, y - radius - 2))

    # Available-spot highlights use a bright gold glow (high contrast on the
    # board and on the dark player colors) with a player-color accent ring. The
    # returned click Rect is a STABLE size so clicks register regardless of pulse.
    _GLOW = "gold"

    def _glow_settlement(self, vertex, color, pulse):
        x, y = int(vertex.x), int(vertex.y)
        self._blit_halo(x, y, int(22 + 12 * pulse), self._GLOW, int(60 + 150 * pulse))
        pygame.draw.circle(self.screen, pygame.Color(color), (x, y), 18, 4)
        return pygame.draw.circle(self.screen, pygame.Color(self._GLOW), (x, y), 20, 3)

    def _glow_city(self, vertex, color, pulse):
        x, y = int(vertex.x), int(vertex.y)
        self._blit_halo(x, y, int(27 + 13 * pulse), self._GLOW, int(60 + 150 * pulse))
        pygame.draw.circle(self.screen, pygame.Color(color), (x, y), 23, 5)
        return pygame.draw.circle(self.screen, pygame.Color(self._GLOW), (x, y), 25, 3)

    def _glow_robber(self, vertexCoord, pulse):
        x, y = int(vertexCoord.x), int(vertexCoord.y)
        self._blit_halo(x, y, int(48 + 16 * pulse), self._GLOW, int(55 + 130 * pulse))
        return pygame.draw.circle(self.screen, pygame.Color("black"), (x, y), 50, 5)

    def _glow_road(self, edge, color, pulse):
        p0, p1 = edge[0], edge[1]
        glow_w = int(10 + 8 * pulse)  # pulsing bright underlay
        rect = pygame.draw.line(self.screen, pygame.Color(self._GLOW), p0, p1, glow_w)
        pygame.draw.line(self.screen, pygame.Color(color), p0, p1, max(2, glow_w // 2))
        return rect

    def _animated_pick(self, spots, draw_fn, allow_cancel):
        """Render the base board once, then each frame restore it and draw a
        pulsating glow on every ``spots`` entry (``draw_fn(spot, pulse)`` returns
        the click Rect). Blocks until the user clicks a spot (returns it) or — if
        ``allow_cancel`` — clicks empty space (returns None). QUIT exits.
        """
        self.displayGameScreen()
        base = self.screen.copy()  # cache the static board for cheap per-frame redraws
        clock = pygame.time.Clock()
        while True:
            self.screen.blit(base, (0, 0))
            pulse = self._pulse()
            rects = {spot: draw_fn(spot, pulse) for spot in spots}
            pygame.display.update()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)
                if e.type == pygame.MOUSEBUTTONDOWN:
                    for spot, rect in rects.items():
                        if rect.collidepoint(e.pos):
                            return spot
                    if allow_cancel:
                        return None
            clock.tick(30)

    # Function to render basic gameplay buttons

    def displayGameButtons(self):
        # Basic GamePlay Buttons
        diceRollText = self.font_button.render("ROLL DICE", False, (0, 0, 0))
        buildRoadText = self.font_button.render("ROAD", False, (0, 0, 0))
        buildSettleText = self.font_button.render("SETTLE", False, (0, 0, 0))
        buildCityText = self.font_button.render("CITY", False, (0, 0, 0))
        endTurnText = self.font_button.render("END TURN", False, (0, 0, 0))
        devCardText = self.font_button.render("DEV CARD", False, (0, 0, 0))
        playDevCardText = self.font_button.render("PLAY DEV", False, (0, 0, 0))
        tradeBankText = self.font_button.render("BANK TRADE", False, (0, 0, 0))

        self.rollDice_button = pygame.Rect(20, 10, 80, 40)
        self.buildRoad_button = pygame.Rect(20, 70, 80, 40)
        self.buildSettlement_button = pygame.Rect(20, 120, 80, 40)
        self.buildCity_button = pygame.Rect(20, 170, 80, 40)

        self.devCard_button = pygame.Rect(20, 300, 80, 40)
        self.playDevCard_button = pygame.Rect(20, 400, 80, 40)

        self.tradeBank_button = pygame.Rect(self.board.width - 125, 400, 100, 40)

        self.endTurn_button = pygame.Rect(20, 700, 80, 40)

        pygame.draw.rect(self.screen, pygame.Color("darkgreen"), self.rollDice_button)
        pygame.draw.rect(self.screen, pygame.Color("gray33"), self.buildRoad_button)
        pygame.draw.rect(self.screen, pygame.Color("gray33"), self.buildSettlement_button)
        pygame.draw.rect(self.screen, pygame.Color("gray33"), self.buildCity_button)
        pygame.draw.rect(self.screen, pygame.Color("gold"), self.devCard_button)
        pygame.draw.rect(self.screen, pygame.Color("gold"), self.playDevCard_button)
        pygame.draw.rect(self.screen, pygame.Color("magenta"), self.tradeBank_button)

        pygame.draw.rect(self.screen, pygame.Color("burlywood"), self.endTurn_button)

        self.screen.blit(diceRollText, (30, 20))
        self.screen.blit(buildRoadText, (30, 80))
        self.screen.blit(buildSettleText, (30, 130))
        self.screen.blit(buildCityText, (30, 180))
        self.screen.blit(devCardText, (30, 310))
        self.screen.blit(playDevCardText, (30, 410))
        self.screen.blit(tradeBankText, (self.board.width - 115, 410))

        self.screen.blit(endTurnText, (30, 710))

    # Function to display robber

    def displayRobber(self):
        for hexTile in self.board.hexTileDict.values():
            if hexTile.has_robber:
                render.draw_robber_pawn(self.screen, hexTile, self.board)
                return

    def displayPlayerStats(self):
        # Harness sets human_player so the human (who sits in the AI-flagged
        # opponent seat) always sees their OWN hand. Engine playCatan leaves it
        # None and falls back to the current non-AI player.
        player = self.human_player
        if player is None:
            if not hasattr(self.game, "currentPlayer") or self.game.currentPlayer is None:
                return
            player = self.game.currentPlayer
            if player.isAI:
                return

        label = self.name_display.get(player.name, player.name)
        title = f"YOUR HAND ({label})" if self.human_player is not None else f"Player: {label}"
        self._draw_hand_panel(player, self.board.width - 160, 15, title)

        # vs-bot ANALYSIS view: reveal the BOT's FULL hand (resources by type +
        # hidden dev cards by type + VP) so the human can judge its decisions.
        # Only shown when the harness sets bot_player; engine playCatan leaves it
        # None (so normal play keeps the opponent's hand hidden).
        if self.bot_player is not None:
            bot_label = self.name_display.get(self.bot_player.name, self.bot_player.name)
            self._draw_hand_panel(
                self.bot_player, self.board.width - 160, 460, f"{bot_label} — FULL HAND"
            )

    def _draw_hand_panel(self, player, x, y, title):
        """Render a full hand panel (resources by type, VP, dev cards by type) for
        ``player`` at (x, y) with a readable backdrop. The dev-card counts include
        hidden / just-bought cards (devCards + newDevCards)."""
        line_height = 20
        panel = pygame.Rect(x - 12, y - 8, 156, line_height * 15 + 12)
        backdrop = pygame.Surface((panel.width, panel.height), pygame.SRCALPHA)
        backdrop.fill((245, 245, 235, 222))
        self.screen.blit(backdrop, (panel.x, panel.y))
        pygame.draw.rect(self.screen, (0, 0, 0), panel, 2, border_radius=6)

        self.screen.blit(self.font_resource.render(title, False, (0, 0, 0)), (x, y))
        y += line_height * 1.5
        for resource, count in player.resources.items():
            self.screen.blit(
                self.font_resource.render(f"{resource}: {count}", False, (0, 0, 0)), (x, y)
            )
            y += line_height
        y += line_height * 0.5
        self.screen.blit(
            self.font_resource.render(f"Victory Points: {player.victoryPoints}", False, (0, 0, 0)),
            (x, y),
        )
        y += line_height * 1.5
        self.screen.blit(self.font_resource.render("Dev Cards:", False, (0, 0, 0)), (x, y))
        y += line_height
        total_dev_cards = player.devCards.copy()
        for card in player.newDevCards:
            total_dev_cards[card] += 1
        dev_map = {
            "KNIGHT": "Knight",
            "VP": "VP",
            "MONOPOLY": "Mono",
            "ROADBUILDER": "RB",
            "YEAROFPLENTY": "YOP",
        }
        for card_type, display_name in dev_map.items():
            count = total_dev_cards.get(card_type, 0)
            self.screen.blit(
                self.font_resource.render(f"{display_name}: {count}", False, (0, 0, 0)), (x, y)
            )
            y += line_height

    # Function to display the gameState board - use to display intermediate build screens
    # gameScreenState specifies which type of screen is to be shown

    def displayGameScreen(self):
        # First display all initial hexes and regular buttons
        self.displayInitialBoard()
        self.displayGameButtons()
        self.displayRobber()
        self.displayPlayerStats()
        self.displayBroadcastMessage()

        # Display Dice Roll
        if self.diceRoll > 0:
            pygame.draw.rect(
                self.screen, pygame.Color("royalblue2"), (100, 20, 50, 50)
            )  # blue background
            diceNum = self.font_diceRoll.render(str(self.diceRoll), False, (0, 0, 0))
            self.screen.blit(diceNum, (110, 20))

        # Loop through and display all existing buildings from players build graphs
        # Build Settlements and roads of each player
        for player_i in list(self.game.playerQueue.queue):
            for existingRoad in player_i.buildGraph["ROADS"]:
                self.draw_road(existingRoad, player_i.color)

            for settlementCoord in player_i.buildGraph["SETTLEMENTS"]:
                self.draw_settlement(settlementCoord, player_i.color)

            for cityCoord in player_i.buildGraph["CITIES"]:
                self.draw_city(cityCoord, player_i.color)

        # Whose-turn banner (top-center) — only when an interactive harness set it.
        if self.turn_banner is not None:
            text, bg_color = self.turn_banner
            surf = self.font_menu.render(text, True, (255, 255, 255))
            rect = surf.get_rect(center=(self.board.width // 2, 22))
            bg_rect = rect.inflate(34, 16)
            pygame.draw.rect(self.screen, pygame.Color(bg_color), bg_rect, border_radius=8)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 2, border_radius=8)
            self.screen.blit(surf, rect)

        pygame.display.update()
        return
        # TO-DO Add screens for trades

    # Function to display dice roll

    def displayDiceRoll(self, diceNums):
        self.diceRoll = diceNums
        self.displayGameScreen()
        return None

    def displayBroadcastMessage(self):
        """Display the last broadcast event on the screen"""
        if not self.game.last_broadcast_event:
            return
        event = self.game.last_broadcast_event
        event_type = event.get("type", "")
        player_name = event.get("player", "")
        player_name = self.name_display.get(player_name, player_name)

        msg_text = ""
        text_color = (0, 0, 0)

        if event_type == "DICE_ROLL":
            value = event.get("value", 0)
            msg_text = f"Dice: {player_name} rolled {value}"
        elif event_type == "DISCARD":
            resources = event.get("resources", [])
            msg_text = f"DISCARD: {player_name} lost {resources}"
            text_color = (255, 0, 0)  # Red
        elif event_type == "YOP":
            resources = event.get("resources", [])
            msg_text = f"YOP: {player_name} gained {resources}"
            text_color = (0, 100, 0)  # Dark Green

        if msg_text:
            text_surface = self.font_broadcast.render(msg_text, True, text_color)
            text_rect = text_surface.get_rect(center=(self.board.width // 2, 60))
            bg_rect = text_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (255, 255, 255), bg_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 2)
            self.screen.blit(text_surface, text_rect)

    def buildRoad_display(self, currentPlayer, roadsPossibleDict):
        """Function to control build-road action with display
        args: player, who is building road; roadsPossibleDict - possible roads
        returns: road edge of road to be built
        """
        # Pulsating glow on every legal road edge; click one to build. Outside
        # the setup phase a click on empty space cancels (returns None).
        roads = [edge for edge in roadsPossibleDict if roadsPossibleDict[edge]]
        return self._animated_pick(
            roads,
            lambda edge, pulse: self._glow_road(edge, currentPlayer.color, pulse),
            allow_cancel=not self.game.gameSetup,
        )

    def buildSettlement_display(self, currentPlayer, verticesPossibleDict):
        """Function to control build-settlement action with display
        args: player, who is building settlement; verticesPossibleDict - dictionary of possible settlement vertices
        returns: vertex of settlement to be built
        """
        # Pulsating glow on every legal settlement vertex (also reused for city
        # upgrades); click one to build. Outside setup a miss cancels (None).
        vertices = [v for v in verticesPossibleDict if verticesPossibleDict[v]]
        return self._animated_pick(
            vertices,
            lambda v, pulse: self._glow_settlement(v, currentPlayer.color, pulse),
            allow_cancel=not self.game.gameSetup,
        )

    def buildCity_display(self, currentPlayer, verticesPossibleDict):
        """Function to control build-city action with display
        args: player, who is building city; verticesPossibleDict - dictionary of possible city vertices
        returns: city vertex of city to be built
        """
        # Pulsating glow on every legal city vertex; click one to upgrade, or
        # click empty space to cancel (returns None).
        vertices = [c for c in verticesPossibleDict if verticesPossibleDict[c]]
        return self._animated_pick(
            vertices,
            lambda v, pulse: self._glow_city(v, currentPlayer.color, pulse),
            allow_cancel=True,
        )

    # Function to control the move-robber action with display

    def moveRobber_display(self, currentPlayer, possibleRobberDict):
        # Pulsating glow on every legal robber hex; click one, then pick a victim.
        pix = {R: possibleRobberDict[R].to_pixel(self.board.flat) for R in possibleRobberDict}
        hexIndex = self._animated_pick(
            list(pix.keys()),
            lambda R, pulse: self._glow_robber(pix[R], pulse),
            allow_cancel=False,
        )
        possiblePlayerDict = self.board.get_players_to_rob(hexIndex)
        playerToRob = self.choosePlayerToRob_display(possiblePlayerDict)
        return hexIndex, playerToRob

    # Function to control the choice of player to rob with display
    # Returns the choice of player to rob

    def choosePlayerToRob_display(self, possiblePlayerDict):
        # Get all other players the player can move robber to and show circles
        for player, vertex in possiblePlayerDict.items():
            possiblePlayerDict[player] = self.draw_possible_players_to_rob(vertex)

        pygame.display.update()

        if possiblePlayerDict == {}:
            return None

        mouseClicked = False
        clock = pygame.time.Clock()
        while mouseClicked == False:
            for e in pygame.event.get():
                if e.type == pygame.MOUSEBUTTONDOWN:
                    for playerToRob, playerCircleRect in possiblePlayerDict.items():
                        if playerCircleRect.collidepoint(e.pos):
                            return playerToRob
            clock.tick(30)

    def get_resource_selection(self, player, mode, num_to_select=1):
        """
        Displays a resource selection menu and handles user interaction.
        mode: 'DISCARD', 'YOP', 'MONOPOLY', 'BANK'
        num_to_select: Number of resources to select (for DISCARD/YOP)
        """
        resources = ["BRICK", "ORE", "SHEEP", "WHEAT", "WOOD"]

        # Menu Geometry
        menu_width = 500
        menu_height = 150
        menu_x = (self.board.width - menu_width) // 2
        menu_y = (self.board.height - menu_height) // 2

        res_size = 80
        spacing = 15
        start_x = menu_x + (menu_width - (5 * res_size + 4 * spacing)) // 2
        res_y = menu_y + (menu_height - res_size) // 2

        # Create Rects for resources
        res_rects = {}
        for i, res in enumerate(resources):
            res_rects[res] = pygame.Rect(
                start_x + i * (res_size + spacing), res_y, res_size, res_size
            )

        # State variables
        selected_resources = []  # For YOP/Discard
        trade_in_res = None  # For Bank
        receive_res = None  # For Bank
        monopoly_res = None  # For Monopoly
        result = None

        running = True
        clock = pygame.time.Clock()
        while running:
            # Draw Menu Background
            pygame.draw.rect(
                self.screen, (200, 200, 200), (menu_x, menu_y, menu_width, menu_height)
            )
            pygame.draw.rect(self.screen, (0, 0, 0), (menu_x, menu_y, menu_width, menu_height), 2)

            # Draw Title
            title_text = ""
            if mode == "DISCARD":
                title_text = f"Discard {num_to_select - len(selected_resources)} cards"
            elif mode == "YOP":
                title_text = f"Select {num_to_select - len(selected_resources)} resources"
            elif mode == "MONOPOLY":
                title_text = "Select resource to monopolize"
            elif mode == "BANK":
                title_text = "Select Trade-In (Red) then Receive (Green)"

            text_surf = self.font_menu.render(title_text, True, (0, 0, 0))
            self.screen.blit(text_surf, (menu_x + 10, menu_y + 10))

            # Draw Resources
            for res in resources:
                rect = res_rects[res]
                color = self.colorDict[res]
                pygame.draw.rect(self.screen, color, rect)

                # Draw Count (Player's current amount)
                count = player.resources[res]
                count_text = self.font_button.render(str(count), True, (0, 0, 0))
                self.screen.blit(count_text, (rect.centerx - 5, rect.centery - 5))

                # Draw Outlines based on state
                if mode == "MONOPOLY":
                    if res == monopoly_res:
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 4)  # Black outline

                elif mode == "BANK":
                    if res == trade_in_res:
                        pygame.draw.rect(self.screen, (255, 0, 0), rect, 4)  # Red outline
                    elif res == receive_res:
                        pygame.draw.rect(self.screen, (0, 255, 0), rect, 4)  # Green outline

            pygame.display.update()

            # Event Handling
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)

                if e.type == pygame.MOUSEBUTTONDOWN:
                    # Check for click outside menu to cancel (except DISCARD)
                    menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
                    if not menu_rect.collidepoint(e.pos) and mode != "DISCARD":
                        if mode == "YOP":
                            # Revert resources added so far
                            for res in selected_resources:
                                player.resources[res] -= 1

                        result = None
                        running = False
                        break

                    clicked_res = None
                    for res, rect in res_rects.items():
                        if rect.collidepoint(e.pos):
                            clicked_res = res
                            break

                    if mode == "DISCARD":
                        if clicked_res and player.resources[clicked_res] > 0:
                            player.resources[clicked_res] -= 1
                            selected_resources.append(clicked_res)
                            if len(selected_resources) >= num_to_select:
                                result = selected_resources
                                running = False

                    elif mode == "YOP":
                        if clicked_res:
                            player.resources[clicked_res] += 1
                            selected_resources.append(clicked_res)
                            if len(selected_resources) >= num_to_select:
                                result = selected_resources
                                running = False

                    elif mode == "MONOPOLY":
                        if clicked_res:
                            if clicked_res == monopoly_res:
                                result = clicked_res  # Confirm
                                running = False
                            else:
                                monopoly_res = clicked_res

                    elif mode == "BANK":
                        if trade_in_res is None:
                            if clicked_res and player.resources[clicked_res] > 0:
                                trade_in_res = clicked_res
                        else:
                            # Trade-in is selected
                            if receive_res is None:
                                if clicked_res == trade_in_res:
                                    trade_in_res = None  # Deselect
                                elif clicked_res:
                                    receive_res = clicked_res
                                else:
                                    # Clicked elsewhere
                                    trade_in_res = None
                            else:
                                # Receive is selected
                                if clicked_res == receive_res:
                                    result = (trade_in_res, receive_res)  # Confirm
                                    running = False
                                elif clicked_res == trade_in_res:
                                    receive_res = None  # Deselect receive
                                elif clicked_res:
                                    receive_res = clicked_res  # Change receive
                                else:
                                    receive_res = None  # Deselect receive

            clock.tick(30)  # Limit redraw rate to prevent flickering
        self.displayGameScreen()
        return result

    def get_dev_card_selection(self, player):
        """
        Displays a dev card selection menu and handles user interaction.
        """
        dev_cards = ["KNIGHT", "VP", "MONOPOLY", "ROADBUILDER", "YEAROFPLENTY"]
        abbreviations = {
            "KNIGHT": "K",
            "VP": "VP",
            "MONOPOLY": "M",
            "ROADBUILDER": "RB",
            "YEAROFPLENTY": "YOP",
        }

        # Menu Geometry
        menu_width = 500
        menu_height = 150
        menu_x = (self.board.width - menu_width) // 2
        menu_y = (self.board.height - menu_height) // 2

        card_size = 80
        spacing = 15
        start_x = menu_x + (menu_width - (5 * card_size + 4 * spacing)) // 2
        card_y = menu_y + (menu_height - card_size) // 2

        # Create Rects for dev cards
        card_rects = {}
        for i, card in enumerate(dev_cards):
            card_rects[card] = pygame.Rect(
                start_x + i * (card_size + spacing), card_y, card_size, card_size
            )

        selected_card = None
        result = None
        running = True

        while running:
            # Draw Menu Background
            pygame.draw.rect(
                self.screen, (200, 200, 200), (menu_x, menu_y, menu_width, menu_height)
            )
            pygame.draw.rect(self.screen, (0, 0, 0), (menu_x, menu_y, menu_width, menu_height), 2)

            # Draw Title
            title_text = "Select Development Card to Play"
            text_surf = self.font_menu.render(title_text, True, (0, 0, 0))
            self.screen.blit(text_surf, (menu_x + 10, menu_y + 10))

            # Draw Dev Cards
            for card in dev_cards:
                rect = card_rects[card]
                pygame.draw.rect(self.screen, (128, 128, 128), rect)  # Gray square

                # Draw Abbreviation (Top)
                abbr = abbreviations[card]
                abbr_text = self.font_button.render(abbr, True, (0, 0, 0))
                self.screen.blit(
                    abbr_text, (rect.centerx - abbr_text.get_width() // 2, rect.top + 5)
                )

                # Draw Count (Middle)
                count = player.devCards[card]
                count_text = self.font_button.render(str(count), True, (0, 0, 0))
                self.screen.blit(
                    count_text,
                    (
                        rect.centerx - count_text.get_width() // 2,
                        rect.centery - count_text.get_height() // 2,
                    ),
                )

                # Draw Selection Outline
                if card == selected_card:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect, 4)  # Green outline

            pygame.display.update()

            # Event Handling
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)

                if e.type == pygame.MOUSEBUTTONDOWN:
                    # Check for click outside menu to cancel
                    menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
                    if not menu_rect.collidepoint(e.pos):
                        result = None
                        running = False
                        break

                    clicked_card = None
                    for card, rect in card_rects.items():
                        if rect.collidepoint(e.pos):
                            clicked_card = card
                            break

                    if clicked_card:
                        if clicked_card == "VP":
                            continue  # Do nothing for VP

                        if player.devCards[clicked_card] > 0:
                            if clicked_card == selected_card:
                                # Double click (click on already selected) -> Confirm
                                result = clicked_card
                                running = False
                            else:
                                # Select new card
                                selected_card = clicked_card
                        else:
                            # Player doesn't have this card
                            pass

        self.displayGameScreen()
        return result
