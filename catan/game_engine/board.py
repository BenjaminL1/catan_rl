# Settlers of Catan
# Game board class implementation

from string import *
import numpy as np
from catan.game_engine.hex_grid import *
from catan.game_engine.player import *
# import networkx as nx
# import matplotlib.pyplot as plt
import pygame

pygame.init()

# Define Resource namedtuple here as it was previously in hexTile.py
Resource = collections.namedtuple("Resource", ["type", "num"])

# Class to implement Catan board logic
# Use a graph representation for the board


class catanBoard:
    'Class Definition for Catan Board Logic'
    # Object Creation - creates a random board configuration with hexTiles

    def __init__(self):
        self.hexTileDict = {}  # Dict to store all hextiles, with hexIndex as key
        # Dict to store the Vertices coordinates with vertex indices as keys
        self.vertex_index_to_pixel_dict = {}
        self.boardGraph = {}  # Dict to store the vertex objects with the pixelCoordinates as keys

        self.edgeLength = 80  # Specify for hex size
        self.size = self.width, self.height = 1000, 800
        # specify Layout
        self.flat = HexLayout(Point(self.edgeLength, self.edgeLength),
                              Point(self.width/2, self.height/2))

        ## INITIALIZE BOARD##
        # print("Initializing Catan Game Board...")
        # Assign resources numbers randomly
        self.resourcesList = self.getRandomResourceList()

        # Get a random permutation of indices 0-18 to use with the resource list
        randomIndices = np.random.permutation(
            [i for i in range(len(self.resourcesList))])

        reinitializeCount = 0
        # Initialize a valid resource list that does not allow adjacent 6's and 8's
        while (self.checkHexNeighbors(randomIndices) == False):
            reinitializeCount += 1
            randomIndices = np.random.permutation(
                [i for i in range(len(self.resourcesList))])

        # print("Re-initialized random board {} times".format(reinitializeCount))

        hexIndex_i = 0  # initialize hexIndex at 0
        # Neighbors are specified in adjacency matrix - hard coded

        # Generate the hexes and the graphs with the Index, Centers and Resources defined
        for rand_i in randomIndices:
            # Get the coordinates of the new hex, indexed by hexIndex_i
            hexCoords = self.getHexCoords(hexIndex_i)
            resource = self.resourcesList[rand_i]

            # Create the new BoardTile with index and append + increment index
            newHexTile = BoardTile(
                hexCoords.q, hexCoords.r, resource.type, resource.num, hexIndex_i)

            if (newHexTile.resource_type == 'DESERT'):  # Initialize robber on Desert
                newHexTile.has_robber = True

            self.hexTileDict[hexIndex_i] = newHexTile
            hexIndex_i += 1

        # Create the vertex graph
        self.vertexIndexCount = 0  # initialize vertex index count to 0
        self.generateVertexGraph()

        self.updatePorts()  # Add the ports to the graph

        # Initialize DevCardStack
        self.devCardStack = {'KNIGHT': 14, 'VP': 5,
                             'MONOPOLY': 2, 'ROADBUILDER': 2, 'YEAROFPLENTY': 2}

        return None

    def getHexCoords(self, hexInd):
        # Dictionary to store Axial Coordinates (q, r) by hexIndex
        coordDict = {0: HexCoordinates(0, 0), 1: HexCoordinates(0, -1), 2: HexCoordinates(1, -1), 3: HexCoordinates(1, 0), 4: HexCoordinates(0, 1), 5: HexCoordinates(-1, 1), 6: HexCoordinates(-1, 0), 7: HexCoordinates(0, -2), 8: HexCoordinates(1, -2), 9: HexCoordinates(2, -2), 10: HexCoordinates(2, -1),
                     11: HexCoordinates(2, 0), 12: HexCoordinates(1, 1), 13: HexCoordinates(0, 2), 14: HexCoordinates(-1, 2), 15: HexCoordinates(-2, 2), 16: HexCoordinates(-2, 1), 17: HexCoordinates(-2, 0), 18: HexCoordinates(-1, -1)}
        return coordDict[hexInd]

    # Function to generate a random permutation of resources

    def getRandomResourceList(self):
        # Define Resources as a dict
        Resource_Dict = {'DESERT': 1, 'ORE': 3,
                         'BRICK': 3, 'WHEAT': 4, 'WOOD': 4, 'SHEEP': 4}
        # Get a random permutation of the numbers
        NumberList = np.random.permutation(
            [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
        numIndex = 0

        resourceList = []
        for r in Resource_Dict.keys():
            numberofResource = Resource_Dict[r]
            if (r != 'DESERT'):
                for n in range(numberofResource):
                    resourceList.append(Resource(r, NumberList[numIndex]))
                    numIndex += 1
            else:
                resourceList.append(Resource(r, None))

        return resourceList

    # Function to check neighboring hexTiles
    # Takes a list of rnadom indices as an input, and the resource list
    def checkHexNeighbors(self, randomIndices):
        # store a list of neighbors as per the axial coordinate -> numeric indexing system
        hexNeighborIndexList = {0: [1, 2, 3, 4, 5, 6], 1: [0, 2, 6, 7, 8, 18], 2: [0, 1, 3, 8, 9, 10],
                                3: [0, 2, 4, 10, 11, 12], 4: [0, 3, 5, 12, 13, 14], 5: [0, 4, 6, 14, 15, 16],
                                6: [0, 1, 5, 16, 17, 18], 7: [1, 8, 18], 8: [1, 2, 7, 9],
                                9: [2, 8, 10], 10: [2, 3, 9, 11], 11: [3, 10, 12],
                                12: [3, 4, 11, 13], 13: [4, 12, 14], 14: [4, 5, 13, 15],
                                15: [5, 14, 16], 16: [5, 6, 15, 17], 17: [6, 16, 18], 18: [1, 6, 7, 17]}

        # Check each position, random index pair for its resource roll value
        for pos, random_Index in enumerate(randomIndices):
            rollValueOnHex = self.resourcesList[random_Index].num

            # Check each neighbor in the position and check if number is legal
            for neighbor_index in hexNeighborIndexList[pos]:
                rollValueOnNeighbor = self.resourcesList[randomIndices[neighbor_index]].num
                if rollValueOnHex in [6, 8] and rollValueOnNeighbor in [6, 8]:
                    return False

        # Return true if it legal for all hexes
        return True

    # Function to generate the entire board graph

    def generateVertexGraph(self):
        for hexTile in self.hexTileDict.values():
            hexTileCorners = hexTile.get_corners(self.flat)  # Get vertices of each hex
            # Create vertex graph with this list of corners
            self.updateVertexGraph(hexTileCorners, hexTile.index_id)

        # Once all hexTiles have been added  get edges
        self.updateGraphEdges()

    # Function to update a graph of the board with each vertex as a node

    def updateVertexGraph(self, vertexCoordList, hexIndx):
        for v in vertexCoordList:
            # Check if vertex already exists - update adjacentHexList if it does
            if v in self.vertex_index_to_pixel_dict.values():
                for existingVertex in self.boardGraph.keys():
                    if (existingVertex == v):
                        self.boardGraph[v].adjacent_hex_indices.append(hexIndx)

            else:  # Create new vertex if it doesn't exist
                # print('Adding Vertex:', v)
                newVertex = BoardVertex(v, vertex_index=self.vertexIndexCount,
                                        adjacent_hex_indices=[hexIndx])
                # Create the index-pixel key value pair
                self.vertex_index_to_pixel_dict[self.vertexIndexCount] = v
                self.boardGraph[v] = newVertex
                self.vertexIndexCount += 1  # Increment index for future

    # Function to add adges to graph given all vertices

    def updateGraphEdges(self):
        for v1 in self.boardGraph.keys():
            for v2 in self.boardGraph.keys():
                if self.boardGraph[v1].is_adjacent_to(self.boardGraph[v2], self.edgeLength):
                    self.boardGraph[v1].neighbors.append(v2)

    @staticmethod
    def vertexDistance(v1, v2):
        dist = ((v1.x - v2.x)**2 + (v1.y - v2.y)**2)**0.5
        return round(dist)

    # View the board graph info
    def printGraph(self):
        print(len(self.boardGraph))
        for node in self.boardGraph.keys():
            print("Pixel:{}, Index:{}, NeighborVertexCount:{}, AdjacentHexes:{}".format(
                node, self.boardGraph[node].vertex_index, len(self.boardGraph[node].neighbors), self.boardGraph[node].adjacent_hex_indices))

    # Update Board vertices with Port info
    def updatePorts(self):
        # Define ports by (HexIndex, CornerIndex1, CornerIndex2)
        # Derived from analysis of flat layout to support rotation
        port_hex_corners = [
            (7, 2, 3),   # Top Left
            (8, 1, 2),   # Top Right
            (10, 1, 2),  # Right
            (11, 0, 1),  # Bottom Right
            (12, 5, 0),  # Bottom Right
            (14, 0, 5),  # Bottom Left
            (15, 4, 5),  # Bottom Left
            (16, 3, 4),  # Left
            (18, 3, 4)   # Top Left
        ]

        port_pair_list = []
        # Helper to find vertex index for a hex corner

        def get_vertex_index(hex_idx, corner_idx):
            hex_tile = self.hexTileDict[hex_idx]
            corners = hex_tile.get_corners(self.flat)
            target_pt = corners[corner_idx]

            # Find closest vertex in boardGraph
            for v_pt, v_obj in self.boardGraph.items():
                if self.vertexDistance(target_pt, v_pt) == 0:
                    return v_obj.vertex_index
            return None

        for h_idx, c1, c2 in port_hex_corners:
            v1 = get_vertex_index(h_idx, c1)
            v2 = get_vertex_index(h_idx, c2)
            if v1 is not None and v2 is not None:
                port_pair_list.append([v1, v2])
            else:
                pass
                # print(
                #     f"Warning: Could not find vertices for port at Hex {h_idx}, Corners {c1}, {c2}")

        # Get a random permutation of indices of ports
        randomPortIndices = np.random.permutation(
            [i for i in range(len(port_pair_list))])
        randomPortIndex_counter = 0

        # Initialize port dictionary with counts
        # Also use this dictionary to map vertex indices to specific ports as per the game board
        port_dict = {'2:1 BRICK': 1, '2:1 SHEEP': 1, '2:1 WOOD': 1,
                     '2:1 WHEAT': 1, '2:1 ORE': 1, '3:1 PORT': 4}

        # Assign random port vertex pairs for each port type
        for portType, portVertexPair_count in port_dict.items():
            portVertices = []
            for i in range(portVertexPair_count):  # Number of ports to assign
                # Add randomized port
                portVertices += port_pair_list[randomPortIndices[randomPortIndex_counter]]
                randomPortIndex_counter += 1

            port_dict[portType] = portVertices

        # Iterate thru each port and update vertex info
        for portType, portVertexIndex_list in port_dict.items():
            for v_index in portVertexIndex_list:  # Each vertex
                # Get the pixel coordinates to update the boardgraph
                vertexPixel = self.vertex_index_to_pixel_dict[v_index]
                # Update the port type
                self.boardGraph[vertexPixel].port = portType

    # Function to Display Catan Board Info

    def displayBoardInfo(self):
        for tile in self.hexTileDict.values():
            tile.debug_info()
        return None

    # Function to get the list of potential roads a player can build.
    # Return these roads as a dictionary where key=vertex coordinates and values is the rect
    def get_potential_roads(self, player):
        colonisableRoads = {}
        # Check potential roads from each road the player already has
        for existingRoad in player.buildGraph['ROADS']:
            for vertex_i in existingRoad:  # Iterate over both vertices of this road
                # Check neighbors from this vertex
                for indx, v_i in enumerate(self.boardGraph[vertex_i].neighbors):
                    # Edge currently does not have a road and vertex isn't colonised by another player
                    # edge_state is [Player, is_road]
                    if ((self.boardGraph[vertex_i].edge_state[indx][1] == False) and (self.boardGraph[vertex_i].owner in [None, player])):
                        # If the edge isn't already there in both its regular + opposite orientation
                        if ((v_i, vertex_i) not in colonisableRoads.keys() and (vertex_i, v_i) not in colonisableRoads.keys()):
                            # Use boolean to keep track of potential roads
                            colonisableRoads[(vertex_i, v_i)] = True
                            # print(vertex_i, v_i)

        return colonisableRoads

    # Function to get available settlements for colonisation for a particular player
    # Return these settlements as a dict of vertices with their Rects

    def get_potential_settlements(self, player):
        colonisableVertices = {}
        # Check starting from each road the player already has
        for existingRoad in player.buildGraph['ROADS']:
            for vertex_i in existingRoad:  # Iterate over both vertices of this road
                # Check if vertex isn't already in the potential settlements - to remove double checks
                if (vertex_i not in colonisableVertices.keys()):
                    # Check if this vertex is already colonised
                    if (self.boardGraph[vertex_i].owner is not None):
                        break

                    canColonise = True
                    # Check each of the neighbors from this vertex
                    for v_neighbor in self.boardGraph[vertex_i].neighbors:
                        if (self.boardGraph[v_neighbor].owner is not None):
                            canColonise = False
                            break

                # If all checks are good add this vertex and its rect as the value
                if (canColonise):
                    # colonisableVertices[vertex_i] = self.draw_possible_settlement(vertex_i, player.color)
                    colonisableVertices[vertex_i] = True

        return colonisableVertices

    # Function to get available cities for colonisation for a particular player
    # Return these cities as a dict of vertex-vertexRect key value pairs

    def get_potential_cities(self, player):
        colonisableVertices = {}
        # Check starting from each settlement the player already has
        for existingSettlement in player.buildGraph['SETTLEMENTS']:
            # colonisableVertices[existingSettlement] = self.draw_possible_city(existingSettlement, player.color)
            colonisableVertices[existingSettlement] = True

        return colonisableVertices

    # Special function to get potential first settlements during setup phase
    def get_setup_settlements(self, player):
        colonisableVertices = {}
        # Check every vertex and every neighbor of that vertex, amd if both are open then we can build a settlement there
        for vertexCoord in self.boardGraph.keys():
            canColonise = True
            potentialVertex = self.boardGraph[vertexCoord]
            if (potentialVertex.owner is not None):  # First check if vertex is colonised
                canColonise = False

            # Check each neighbor
            for v_neighbor in potentialVertex.neighbors:
                # Check if any of first neighbors are colonised
                if (self.boardGraph[v_neighbor].owner is not None):
                    canColonise = False
                    break

            if (canColonise):  # If the vertex is colonisable add it to the dict with its Rect
                # colonisableVertices[vertexCoord] = self.draw_possible_settlement(vertexCoord, player.color)
                colonisableVertices[vertexCoord] = True

        return colonisableVertices

    # Special function to get potential first roads during setup phase

    def get_setup_roads(self, player):
        colonisableRoads = {}
        # Can only build roads next to the latest existing player settlement
        latestSettlementCoords = player.buildGraph['SETTLEMENTS'][-1]
        for v_neighbor in self.boardGraph[latestSettlementCoords].neighbors:
            possibleRoad = (latestSettlementCoords, v_neighbor)
            # colonisableRoads[possibleRoad] = self.draw_possible_road(possibleRoad, player.color)
            colonisableRoads[possibleRoad] = True

        return colonisableRoads

    # Function to update boardGraph with Road by player

    def updateBoardGraph_road(self, v_coord1, v_coord2, player):
        # Update edge from first vertex v1
        for indx, v in enumerate(self.boardGraph[v_coord1].neighbors):
            if (v == v_coord2):
                self.boardGraph[v_coord1].edge_state[indx][0] = player
                self.boardGraph[v_coord1].edge_state[indx][1] = True

        # Update edge from second vertex v2
        for indx, v in enumerate(self.boardGraph[v_coord2].neighbors):
            if (v == v_coord1):
                self.boardGraph[v_coord2].edge_state[indx][0] = player
                self.boardGraph[v_coord2].edge_state[indx][1] = True

        # self.draw_road([v_coord1, v_coord2], player.color) #Draw the settlement

    # Function to update boardGraph with settlement on vertex v

    def updateBoardGraph_settlement(self, v_coord, player):
        self.boardGraph[v_coord].owner = player
        self.boardGraph[v_coord].building_type = 'Settlement'
        # self.boardGraph[v_coord].isColonised = True # Implied by owner

        # self.draw_settlement(v_coord, player.color) #Draw the settlement

    # Function to update boardGraph with settlement on vertex v
    def updateBoardGraph_city(self, v_coord, player):
        self.boardGraph[v_coord].owner = player
        self.boardGraph[v_coord].building_type = 'City'

        # Remove settlement from player's buildGraph
        player.buildGraph['SETTLEMENTS'].remove(v_coord)

    # Function to update boardGraph with Robber on hexTile
    def updateBoardGraph_robber(self, hexIndex):
        # Set all flags to false
        for hex_tile in self.hexTileDict.values():
            hex_tile.has_robber = False

        self.hexTileDict[hexIndex].has_robber = True

    # Function to get possible robber hexTiles
    # Return robber hex spots with their hexIndex - rect representations as key-value pairs
    def get_robber_spots(self):
        robberHexDict = {}
        for indx, hex_tile in self.hexTileDict.items():
            if (hex_tile.has_robber == False):
                # Friendly Robber Rule:
                # Cannot place robber on a hex if any adjacent player has < 3 visible VPs
                is_friendly_spot = True
                vertexList = hex_tile.get_corners(self.flat)

                for vertex in vertexList:
                    if vertex in self.boardGraph:
                        player = self.boardGraph[vertex].owner
                        if player is not None:
                            visible_vps = player.victoryPoints - player.devCards['VP']
                            if visible_vps < 3:
                                is_friendly_spot = False
                                break

                if is_friendly_spot:
                    robberHexDict[indx] = hex_tile

        return robberHexDict

    # Get a Dict of players to rob based on the hexIndex of the robber, with the circle Rect as the value
    def get_players_to_rob(self, hexIndex):
        # Extract all 6 vertices of this hexTile
        hexTile = self.hexTileDict[hexIndex]
        vertexList = hexTile.get_corners(self.flat)

        playersToRobDict = {}

        for vertex in vertexList:
            # There is a settlement on this vertex
            if (self.boardGraph[vertex].owner != None):
                playerToRob = self.boardGraph[vertex].owner
                # only add a player once with his/her first settlement/city
                if (playerToRob not in playersToRobDict.keys()):
                    # playersToRobDict[playerToRob] = self.draw_possible_players_to_rob(vertex)
                    playersToRobDict[playerToRob] = vertex

        return playersToRobDict

    # Function to get a hexTile with a particular number

    def getHexResourceRolled(self, diceRollNum):
        # Empty list to store the hex index rolled (min 1, max 2)
        hexesRolled = []
        for hexTile in self.hexTileDict.values():
            if hexTile.number_token == diceRollNum:
                hexesRolled.append(hexTile.index_id)

        return hexesRolled
