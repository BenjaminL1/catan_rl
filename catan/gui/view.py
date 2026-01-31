# Settlers of Catan
# Game view class implementation with pygame

import pygame
import sys
from catan.engine.geometry import *

pygame.init()

# Class to handle catan board display


class catanGameView():
    'Class definition for Catan board display'

    def __init__(self, catanBoardObject, catanGameObject):
        self.board = catanBoardObject
        self.game = catanGameObject

        # #Use pygame to display the board
        self.screen = pygame.display.set_mode(self.board.size)
        pygame.display.set_caption('Settlers of Catan')
        self.font_resource = pygame.font.SysFont('cambria', 15)
        self.font_ports = pygame.font.SysFont('cambria', 10)

        self.font_button = pygame.font.SysFont('cambria', 12)
        self.font_diceRoll = pygame.font.SysFont('cambria', 25)  # dice font
        self.font_Robber = pygame.font.SysFont('arialblack', 50)  # robber font
        self.font_menu = pygame.font.SysFont('cambria', 20)
        self.font_broadcast = pygame.font.SysFont('cambria', 18)  # broadcast font

        self.diceRoll = 0  # Initialize dice roll

        return None

    # Function to display the initial board

    def displayInitialBoard(self):
        # Dictionary to store RGB Color values
        colorDict_RGB = {"BRICK": (255, 51, 51), "ORE": (128, 128, 128), "WHEAT": (
            255, 255, 51), "WOOD": (0, 153, 0), "SHEEP": (51, 255, 51), "DESERT": (255, 255, 204)}
        self.colorDict = colorDict_RGB
        pygame.draw.rect(self.screen, pygame.Color(
            'royalblue2'), (0, 0, self.board.width, self.board.height))  # blue background

        # Render each hexTile
        for hexTile in self.board.hexTileDict.values():
            hexTileCorners = hexTile.get_corners(self.board.flat)

            hexTileColor_rgb = colorDict_RGB[hexTile.resource_type]
            pygame.draw.polygon(self.screen, pygame.Color(
                hexTileColor_rgb[0], hexTileColor_rgb[1], hexTileColor_rgb[2]), hexTileCorners, self.board.width == 0)
            # print(hexTile.index, hexTileCorners)

            # Get pixel center coordinates of hex
            hexTile.pixelCenter = hexTile.to_pixel(self.board.flat)
            if (hexTile.resource_type != 'DESERT'):  # skip desert text/number
                resourceText = self.font_resource.render(str(
                    hexTile.resource_type) + " (" + str(hexTile.number_token) + ")", False, (0, 0, 0))
                # add text to hex
                self.screen.blit(
                    resourceText, (hexTile.pixelCenter.x - 25, hexTile.pixelCenter.y))

        # Display the Ports - update images/formatting later
        for vCoord, vertexInfo in self.board.boardGraph.items():
            if (vertexInfo.port != None):
                portText = self.font_ports.render(
                    vertexInfo.port, False, (0, 0, 0))
                # print("Displaying {} port with coordinates x ={} and y={}".format(vertexInfo.port, vCoord.x, vCoord.y))

                if (vCoord.x < 430 and vCoord.y > 130):
                    self.screen.blit(portText, (vCoord.x - 50, vCoord.y))
                elif (vCoord.x > 430 and vCoord.y < 130):
                    self.screen.blit(portText, (vCoord.x, vCoord.y - 15))
                elif (vCoord.x < 430 and vCoord.y < 130):
                    self.screen.blit(portText, (vCoord.x - 50, vCoord.y - 15))
                else:
                    self.screen.blit(portText, (vCoord.x, vCoord.y))

        pygame.display.update()

        return None

    # Function to draw a road on the board

    def draw_road(self, edgeToDraw, roadColor):
        pygame.draw.line(self.screen, pygame.Color(roadColor),
                         edgeToDraw[0], edgeToDraw[1], 10)

    # Function to draw a potential road on the board - thin

    def draw_possible_road(self, edgeToDraw, roadColor):
        roadRect = pygame.draw.line(self.screen, pygame.Color(
            roadColor), edgeToDraw[0], edgeToDraw[1], 5)
        return roadRect

    # Function to draw a settlement on the board at vertexToDraw

    def draw_settlement(self, vertexToDraw, color):
        newSettlement = pygame.Rect(
            vertexToDraw.x - 10, vertexToDraw.y - 10, 25, 25)
        pygame.draw.rect(self.screen, pygame.Color(color), newSettlement)

    # Function to draw a potential settlement on the board - thin

    def draw_possible_settlement(self, vertexToDraw, color):
        possibleSettlement = pygame.draw.circle(self.screen, pygame.Color(
            color), (int(vertexToDraw.x), int(vertexToDraw.y)), 20, 3)
        return possibleSettlement

    # Function to draw a settlement on the board at vertexToDraw

    def draw_city(self, vertexToDraw, color):
        pygame.draw.circle(self.screen, pygame.Color(
            color), (int(vertexToDraw.x), int(vertexToDraw.y)), 24)

    # Function to draw a potential settlement on the board - thin

    def draw_possible_city(self, vertexToDraw, color):
        possibleCity = pygame.draw.circle(self.screen, pygame.Color(
            color), (int(vertexToDraw.x), int(vertexToDraw.y)), 25, 5)
        return possibleCity

    # Function to draw the possible spots for a robber

    def draw_possible_robber(self, vertexToDraw):
        possibleRobber = pygame.draw.circle(self.screen, pygame.Color(
            'black'), (int(vertexToDraw.x), int(vertexToDraw.y)), 50, 5)
        return possibleRobber

    # Function to draw possible players to rob
    def draw_possible_players_to_rob(self, vertexCoord):
        possiblePlayer = pygame.draw.circle(self.screen, pygame.Color(
            'black'), (int(vertexCoord.x), int(vertexCoord.y)), 35, 5)
        return possiblePlayer

    # Function to render basic gameplay buttons

    def displayGameButtons(self):
        # Basic GamePlay Buttons
        diceRollText = self.font_button.render("ROLL DICE", False, (0, 0, 0))
        buildRoadText = self.font_button.render("ROAD", False, (0, 0, 0))
        buildSettleText = self.font_button.render("SETTLE", False, (0, 0, 0))
        buildCityText = self.font_button.render("CITY", False, (0, 0, 0))
        endTurnText = self.font_button.render("END TURN", False, (0, 0, 0))
        devCardText = self.font_button.render("DEV CARD", False, (0, 0, 0))
        playDevCardText = self.font_button.render(
            "PLAY DEV", False, (0, 0, 0))
        tradeBankText = self.font_button.render(
            "BANK TRADE", False, (0, 0, 0))

        self.rollDice_button = pygame.Rect(20, 10, 80, 40)
        self.buildRoad_button = pygame.Rect(20, 70, 80, 40)
        self.buildSettlement_button = pygame.Rect(20, 120, 80, 40)
        self.buildCity_button = pygame.Rect(20, 170, 80, 40)

        self.devCard_button = pygame.Rect(20, 300, 80, 40)
        self.playDevCard_button = pygame.Rect(20, 400, 80, 40)

        self.tradeBank_button = pygame.Rect(
            self.board.width - 125, 400, 100, 40)

        self.endTurn_button = pygame.Rect(20, 700, 80, 40)

        pygame.draw.rect(self.screen, pygame.Color(
            'darkgreen'), self.rollDice_button)
        pygame.draw.rect(self.screen, pygame.Color(
            'gray33'), self.buildRoad_button)
        pygame.draw.rect(self.screen, pygame.Color(
            'gray33'), self.buildSettlement_button)
        pygame.draw.rect(self.screen, pygame.Color(
            'gray33'), self.buildCity_button)
        pygame.draw.rect(self.screen, pygame.Color(
            'gold'), self.devCard_button)
        pygame.draw.rect(self.screen, pygame.Color(
            'gold'), self.playDevCard_button)
        pygame.draw.rect(self.screen, pygame.Color(
            'magenta'), self.tradeBank_button)

        pygame.draw.rect(self.screen, pygame.Color(
            'burlywood'), self.endTurn_button)

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
        # Robber text
        robberText = self.font_Robber.render("R", False, (0, 0, 0))
        # Get the coordinates for the robber
        for hexTile in self.board.hexTileDict.values():
            if (hexTile.has_robber):
                robberCoords = hexTile.pixelCenter

        self.screen.blit(robberText, (int(robberCoords.x) -
                         20, int(robberCoords.y) - 35))

    def displayPlayerStats(self):
        if not hasattr(self.game, 'currentPlayer') or self.game.currentPlayer is None:
            return

        player = self.game.currentPlayer

        if player.isAI:
            return

        # Define starting position for stats
        x_offset = self.board.width - 150
        y_offset = 20
        line_height = 20

        # Display Player Name
        nameText = self.font_resource.render(
            f"Player: {player.name}", False, (0, 0, 0))
        self.screen.blit(nameText, (x_offset, y_offset))
        y_offset += line_height * 1.5

        # Display Resources
        for resource, count in player.resources.items():
            resText = self.font_resource.render(
                f"{resource}: {count}", False, (0, 0, 0))
            self.screen.blit(resText, (x_offset, y_offset))
            y_offset += line_height

        y_offset += line_height * 0.5

        # Display Victory Points
        vpText = self.font_resource.render(
            f"Victory Points: {player.victoryPoints}", False, (0, 0, 0))
        self.screen.blit(vpText, (x_offset, y_offset))
        y_offset += line_height * 1.5

        # Display Dev Cards
        devCardText = self.font_resource.render("Dev Cards:", False, (0, 0, 0))
        self.screen.blit(devCardText, (x_offset, y_offset))
        y_offset += line_height

        devCardMap = {'KNIGHT': 'Knight', 'VP': 'VP', 'MONOPOLY': 'Mono',
                      'ROADBUILDER': 'RB', 'YEAROFPLENTY': 'YOP'}

        # Calculate total dev cards (playable + new)
        total_dev_cards = player.devCards.copy()
        for card in player.newDevCards:
            total_dev_cards[card] += 1

        for card_type, display_name in devCardMap.items():
            count = total_dev_cards.get(card_type, 0)
            cardText = self.font_resource.render(
                f"{display_name}: {count}", False, (0, 0, 0))
            self.screen.blit(cardText, (x_offset, y_offset))
            y_offset += line_height

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
            pygame.draw.rect(self.screen, pygame.Color(
                'royalblue2'), (100, 20, 50, 50))  # blue background
            diceNum = self.font_diceRoll.render(str(self.diceRoll), False, (0, 0, 0))
            self.screen.blit(diceNum, (110, 20))

        # Loop through and display all existing buildings from players build graphs
        # Build Settlements and roads of each player
        for player_i in list(self.game.playerQueue.queue):
            for existingRoad in player_i.buildGraph['ROADS']:
                self.draw_road(existingRoad, player_i.color)

            for settlementCoord in player_i.buildGraph['SETTLEMENTS']:
                self.draw_settlement(settlementCoord, player_i.color)

            for cityCoord in player_i.buildGraph['CITIES']:
                self.draw_city(cityCoord, player_i.color)

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
        if self.game.last_broadcast_event:
            event_type = self.game.last_broadcast_event['type']
            player_name = self.game.last_broadcast_event['player']
            resources = self.game.last_broadcast_event['resources']

            msg_text = ""
            text_color = (0, 0, 0)

            if event_type == 'DISCARD':
                msg_text = f"DISCARD: {player_name} lost {resources}"
                text_color = (255, 0, 0)  # Red
            elif event_type == 'YOP':
                msg_text = f"YOP: {player_name} gained {resources}"
                text_color = (0, 100, 0)  # Dark Green

            # Render text
            text_surface = self.font_broadcast.render(msg_text, True, text_color)

            # Position at the top center or somewhere visible
            # Assuming board width is around 800-1000 based on other coords
            text_rect = text_surface.get_rect(center=(self.board.width // 2, 60))

            # Draw background for readability
            bg_rect = text_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (255, 255, 255), bg_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 2)

            self.screen.blit(text_surface, text_rect)

    def buildRoad_display(self, currentPlayer, roadsPossibleDict):
        '''Function to control build-road action with display
        args: player, who is building road; roadsPossibleDict - possible roads
        returns: road edge of road to be built
        '''
        # Get all spots the player can build a road and display thin lines
        # Get Rect representation of roads and draw possible roads
        for roadEdge in roadsPossibleDict.keys():
            if roadsPossibleDict[roadEdge]:
                roadsPossibleDict[roadEdge] = self.draw_possible_road(
                    roadEdge, currentPlayer.color)
                # print("displaying road")

        pygame.display.update()

        mouseClicked = False  # Get player actions until a mouse is clicked
        while (mouseClicked == False):
            if (self.game.gameSetup):  # during gameSetup phase only exit if road is built
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        sys.exit(0)
                    if (e.type == pygame.MOUSEBUTTONDOWN):
                        for road, roadRect in roadsPossibleDict.items():
                            if (roadRect.collidepoint(e.pos)):
                                # currentPlayer.build_road(road[0], road[1], self.board)
                                mouseClicked = True
                                return road

            else:
                for e in pygame.event.get():
                    if (e.type == pygame.MOUSEBUTTONDOWN):  # Exit this loop on mouseclick
                        for road, roadRect in roadsPossibleDict.items():
                            if (roadRect.collidepoint(e.pos)):
                                # currentPlayer.build_road(road[0], road[1], self.board)
                                return road

                        mouseClicked = True
                        return None

    def buildSettlement_display(self, currentPlayer, verticesPossibleDict):
        '''Function to control build-settlement action with display
        args: player, who is building settlement; verticesPossibleDict - dictionary of possible settlement vertices
        returns: vertex of settlement to be built
        '''
        # Get all spots the player can build a settlement and display thin circles
        # Add in the Rect representations of possible settlements
        for v in verticesPossibleDict.keys():
            if verticesPossibleDict[v]:
                verticesPossibleDict[v] = self.draw_possible_settlement(
                    v, currentPlayer.color)

        pygame.display.update()

        mouseClicked = False  # Get player actions until a mouse is clicked

        while (mouseClicked == False):
            if (self.game.gameSetup):  # during gameSetup phase only exit if settlement is built
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        sys.exit(0)
                    if (e.type == pygame.MOUSEBUTTONDOWN):
                        for vertex, vertexRect in verticesPossibleDict.items():
                            if (vertexRect.collidepoint(e.pos)):
                                # currentPlayer.build_settlement(vertex, self.board)
                                mouseClicked = True
                                return vertex
            else:
                for e in pygame.event.get():
                    if (e.type == pygame.MOUSEBUTTONDOWN):  # Exit this loop on mouseclick
                        for vertex, vertexRect in verticesPossibleDict.items():
                            if (vertexRect.collidepoint(e.pos)):
                                # currentPlayer.build_settlement(vertex, self.board)
                                return vertex

                        mouseClicked = True
                        return None

    def buildCity_display(self, currentPlayer, verticesPossibleDict):
        '''Function to control build-city action with display
        args: player, who is building city; verticesPossibleDict - dictionary of possible city vertices
        returns: city vertex of city to be built
        '''
        # Get all spots the player can build a city and display circles
        # Get Rect representation of roads and draw possible roads
        for c in verticesPossibleDict.keys():
            if verticesPossibleDict[c]:
                verticesPossibleDict[c] = self.draw_possible_city(
                    c, currentPlayer.color)

        pygame.display.update()

        # Get player actions until a mouse is clicked - whether a city is built or not
        mouseClicked = False

        while (mouseClicked == False):
            for e in pygame.event.get():
                if (e.type == pygame.MOUSEBUTTONDOWN):  # Exit this loop on mouseclick
                    for vertex, vertexRect in verticesPossibleDict.items():
                        if (vertexRect.collidepoint(e.pos)):
                            # currentPlayer.build_city(vertex, self.board)
                            return vertex

                    mouseClicked = True
                    return None

    # Function to control the move-robber action with display

    def moveRobber_display(self, currentPlayer, possibleRobberDict):
        # Get all spots the player can move robber to and show circles
        # Add in the Rect representations of possible robber spots
        for R in possibleRobberDict.keys():
            possibleRobberDict[R] = self.draw_possible_robber(
                possibleRobberDict[R].pixelCenter)

        pygame.display.update()

        # Get player actions until a mouse is clicked - whether a road is built or not
        mouseClicked = False

        while (mouseClicked == False):
            for e in pygame.event.get():
                if (e.type == pygame.MOUSEBUTTONDOWN):  # Exit this loop on mouseclick
                    for hexIndex, robberCircleRect in possibleRobberDict.items():
                        if (robberCircleRect.collidepoint(e.pos)):
                            # Add code to choose which player to rob depending on hex clicked on
                            possiblePlayerDict = self.board.get_players_to_rob(
                                hexIndex)

                            playerToRob = self.choosePlayerToRob_display(
                                possiblePlayerDict)

                            # Move robber to that hex and rob
                            # currentPlayer.move_robber(hexIndex, self.board, playerToRob) #Player moved robber to this hex
                            mouseClicked = True  # Only exit out once a correct robber spot is chosen
                            return hexIndex, playerToRob

    # Function to control the choice of player to rob with display
    # Returns the choice of player to rob

    def choosePlayerToRob_display(self, possiblePlayerDict):
        # Get all other players the player can move robber to and show circles
        for player, vertex in possiblePlayerDict.items():
            possiblePlayerDict[player] = self.draw_possible_players_to_rob(
                vertex)

        pygame.display.update()

        # If dictionary is empty return None
        if (possiblePlayerDict == {}):
            return None

        # Get player actions until a mouse is clicked - whether a road is built or not
        mouseClicked = False
        while (mouseClicked == False):
            for e in pygame.event.get():
                if (e.type == pygame.MOUSEBUTTONDOWN):  # Exit this loop on mouseclick
                    for playerToRob, playerCircleRect in possiblePlayerDict.items():
                        if (playerCircleRect.collidepoint(e.pos)):
                            return playerToRob

    def get_resource_selection(self, player, mode, num_to_select=1):
        """
        Displays a resource selection menu and handles user interaction.
        mode: 'DISCARD', 'YOP', 'MONOPOLY', 'BANK'
        num_to_select: Number of resources to select (for DISCARD/YOP)
        """
        resources = ['BRICK', 'ORE', 'SHEEP', 'WHEAT', 'WOOD']

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
                start_x + i * (res_size + spacing), res_y, res_size, res_size)

        # State variables
        selected_resources = []  # For YOP/Discard
        trade_in_res = None  # For Bank
        receive_res = None  # For Bank
        monopoly_res = None  # For Monopoly
        result = None

        running = True
        while running:
            # Draw Menu Background
            pygame.draw.rect(self.screen, (200, 200, 200),
                             (menu_x, menu_y, menu_width, menu_height))
            pygame.draw.rect(self.screen, (0, 0, 0), (menu_x, menu_y, menu_width, menu_height), 2)

            # Draw Title
            title_text = ""
            if mode == 'DISCARD':
                title_text = f"Discard {num_to_select - len(selected_resources)} cards"
            elif mode == 'YOP':
                title_text = f"Select {num_to_select - len(selected_resources)} resources"
            elif mode == 'MONOPOLY':
                title_text = "Select resource to monopolize"
            elif mode == 'BANK':
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
                if mode == 'MONOPOLY':
                    if res == monopoly_res:
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 4)  # Black outline

                elif mode == 'BANK':
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
                    if not menu_rect.collidepoint(e.pos) and mode != 'DISCARD':
                        if mode == 'YOP':
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

                    if mode == 'DISCARD':
                        if clicked_res and player.resources[clicked_res] > 0:
                            player.resources[clicked_res] -= 1
                            selected_resources.append(clicked_res)
                            if len(selected_resources) >= num_to_select:
                                result = selected_resources
                                running = False

                    elif mode == 'YOP':
                        if clicked_res:
                            player.resources[clicked_res] += 1
                            selected_resources.append(clicked_res)
                            if len(selected_resources) >= num_to_select:
                                result = selected_resources
                                running = False

                    elif mode == 'MONOPOLY':
                        if clicked_res:
                            if clicked_res == monopoly_res:
                                result = clicked_res  # Confirm
                                running = False
                            else:
                                monopoly_res = clicked_res

                    elif mode == 'BANK':
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

        self.displayGameScreen()
        return result

    def get_dev_card_selection(self, player):
        """
        Displays a dev card selection menu and handles user interaction.
        """
        dev_cards = ['KNIGHT', 'VP', 'MONOPOLY', 'ROADBUILDER', 'YEAROFPLENTY']
        abbreviations = {'KNIGHT': 'K', 'VP': 'VP', 'MONOPOLY': 'M',
                         'ROADBUILDER': 'RB', 'YEAROFPLENTY': 'YOP'}

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
                start_x + i * (card_size + spacing), card_y, card_size, card_size)

        selected_card = None
        result = None
        running = True

        while running:
            # Draw Menu Background
            pygame.draw.rect(self.screen, (200, 200, 200),
                             (menu_x, menu_y, menu_width, menu_height))
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
                self.screen.blit(abbr_text, (rect.centerx -
                                 abbr_text.get_width() // 2, rect.top + 5))

                # Draw Count (Middle)
                count = player.devCards[card]
                count_text = self.font_button.render(str(count), True, (0, 0, 0))
                self.screen.blit(count_text, (rect.centerx - count_text.get_width() //
                                 2, rect.centery - count_text.get_height() // 2))

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
                        if clicked_card == 'VP':
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
