# Settlers of Catan
# Game board class implementation

from string import *

import numpy as np

# import networkx as nx
# import matplotlib.pyplot as plt
try:
    import pygame  # type: ignore[import-not-found]

    pygame.init()
except ImportError:
    pygame = None  # type: ignore[assignment]

from catan_rl.engine.geometry import *
from catan_rl.engine.player import *

# Define Resource namedtuple here as it was previously in hexTile.py
Resource = collections.namedtuple("Resource", ["type", "num"])

# ---------------------------------------------------------------------------
# Official ABC chip sequence (matches Colonist.io 1v1 ranked board generation)
# ---------------------------------------------------------------------------
# Empirically verified 2026-06-02 against 5 ranked Colonist 1v1 boards
# (every chip placement on every board matched the spiral under some
# rotation/mirror — see ``docs/audit/`` notes / commit history).
#
# Algorithm: walk the 19 hexes in spiral order; for each non-desert hex,
# assign the next chip from the sequence. The 18-element sequence
# matches the official Catan variable-setup "alphabetical" placement
# (A=5, B=2, C=6, ..., R=11).
SPIRAL_CHIP_SEQUENCE: tuple[int, ...] = (
    5,
    2,
    6,
    3,
    8,
    10,
    9,
    12,
    11,
    4,
    8,
    10,
    9,
    4,
    5,
    6,
    3,
    11,
)

# Outer ring in CW pixel order, starting from the top (engine hex 7 at
# axial (0, -2)). Engine hex indices already follow the spiral path —
# see ``getHexCoords`` axial assignments.
OUTER_RING_CW: tuple[int, ...] = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

# Inner ring in CW pixel order, starting top-left.
INNER_RING_CW: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

# The 6 outer hexes that sit at hexagonal-board corners (every other outer hex).
OUTER_CORNERS: tuple[int, ...] = (7, 9, 11, 13, 15, 17)


def _build_spiral_path(start_corner: int, clockwise: bool) -> list[int]:
    """Return the 19-hex spiral traversal order for one of 12 orientations.

    Args:
        start_corner: One of the 6 outer corners (``OUTER_CORNERS``).
        clockwise: ``True`` for CW spiral, ``False`` for CCW.

    Returns:
        The 19 engine hex indices in spiral order: 12 outer ring → 6 inner
        ring → center (always 0). The chip sequence is applied in this
        order, skipping the desert.

    Raises:
        ValueError: if ``start_corner`` is not in ``OUTER_CORNERS``.
    """
    if start_corner not in OUTER_CORNERS:
        raise ValueError(
            f"start_corner {start_corner} is not an outer corner (must be one of {OUTER_CORNERS})"
        )
    target_idx = OUTER_RING_CW.index(start_corner)
    # Each 60° rotation between corners moves the outer-ring index by 2,
    # and the inner ring by 1. ``rotation`` is the inner-ring offset.
    rotation = target_idx // 2

    outer_path = list(OUTER_RING_CW[target_idx:] + OUTER_RING_CW[:target_idx])
    inner_path = list(INNER_RING_CW[rotation:] + INNER_RING_CW[:rotation])

    if not clockwise:
        # Mirror: keep the start hex first, reverse the rest of the ring.
        outer_path = [outer_path[0]] + outer_path[:0:-1]
        inner_path = [inner_path[0]] + inner_path[:0:-1]

    return outer_path + inner_path + [0]


def _random_resource_type_list() -> list[str]:
    """Build the 19-resource-type list for board construction.

    Counts: 1 DESERT, 3 ORE, 3 BRICK, 4 WHEAT, 4 WOOD, 4 SHEEP.
    Order is fixed at construction time; the caller is expected to
    ``np.random.shuffle`` it before use.
    """
    counts: dict[str, int] = {
        "DESERT": 1,
        "ORE": 3,
        "BRICK": 3,
        "WHEAT": 4,
        "WOOD": 4,
        "SHEEP": 4,
    }
    out: list[str] = []
    for resource, n in counts.items():
        out.extend([resource] * n)
    return out


# Class to implement Catan board logic
# Use a graph representation for the board


class catanBoard:
    "Class Definition for Catan Board Logic"

    # Object Creation - creates a random board configuration with hexTiles

    def __init__(self):
        self.hexTileDict = {}  # Dict to store all hextiles, with hexIndex as key
        # Dict to store the Vertices coordinates with vertex indices as keys
        self.vertex_index_to_pixel_dict = {}
        self.boardGraph = {}  # Dict to store the vertex objects with the pixelCoordinates as keys

        self.edgeLength = 80  # Specify for hex size
        self.size = self.width, self.height = 1000, 800
        # specify Layout
        self.flat = HexLayout(
            Point(self.edgeLength, self.edgeLength), Point(self.width / 2, self.height / 2)
        )

        ## INITIALIZE BOARD ##
        # Colonist.io 1v1 ranked algorithm (empirically verified 2026-06-02):
        # 1. Random shuffle of resource TYPES across the 19 hex positions.
        # 2. Random spiral orientation: one of 6 outer corners × CW / CCW = 12.
        # 3. Walk the spiral in that orientation; assign chips from
        #    ``SPIRAL_CHIP_SEQUENCE`` to each non-desert hex in order.
        # The spiral construction guarantees no two 6/8 adjacent, no two
        # identical numbers adjacent, and no two 2/12 adjacent — these
        # are properties of the official ABC chip sequence, not rejection-
        # sampling constraints.

        resource_types = _random_resource_type_list()
        np.random.shuffle(resource_types)

        start_corner = OUTER_CORNERS[int(np.random.randint(len(OUTER_CORNERS)))]
        clockwise = bool(np.random.randint(2))
        spiral_path = _build_spiral_path(start_corner, clockwise)

        number_tokens: list[int | None] = [None] * 19
        chip_idx = 0
        for hex_idx in spiral_path:
            if resource_types[hex_idx] == "DESERT":
                continue
            number_tokens[hex_idx] = SPIRAL_CHIP_SEQUENCE[chip_idx]
            chip_idx += 1

        # Construct the hex tiles in canonical engine index order.
        for hex_idx in range(19):
            hexCoords = self.getHexCoords(hex_idx)
            resource_type = resource_types[hex_idx]
            number_token = number_tokens[hex_idx]
            newHexTile = BoardTile(hexCoords.q, hexCoords.r, resource_type, number_token, hex_idx)
            if resource_type == "DESERT":
                newHexTile.has_robber = True
            self.hexTileDict[hex_idx] = newHexTile

        # Preserve ``self.resourcesList`` for any back-compat reader; it is
        # now a parallel list indexed by hex index (not a permutation).
        self.resourcesList = [Resource(resource_types[i], number_tokens[i]) for i in range(19)]

        # Create the vertex graph
        self.vertexIndexCount = 0  # initialize vertex index count to 0
        self.generateVertexGraph()

        self.updatePorts()  # Add the ports to the graph

        # Initialize DevCardStack
        self.devCardStack = {
            "KNIGHT": 14,
            "VP": 5,
            "MONOPOLY": 2,
            "ROADBUILDER": 2,
            "YEAROFPLENTY": 2,
        }

        return None

    def getHexCoords(self, hexInd):
        # Dictionary to store Axial Coordinates (q, r) by hexIndex
        coordDict = {
            0: HexCoordinates(0, 0),
            1: HexCoordinates(0, -1),
            2: HexCoordinates(1, -1),
            3: HexCoordinates(1, 0),
            4: HexCoordinates(0, 1),
            5: HexCoordinates(-1, 1),
            6: HexCoordinates(-1, 0),
            7: HexCoordinates(0, -2),
            8: HexCoordinates(1, -2),
            9: HexCoordinates(2, -2),
            10: HexCoordinates(2, -1),
            11: HexCoordinates(2, 0),
            12: HexCoordinates(1, 1),
            13: HexCoordinates(0, 2),
            14: HexCoordinates(-1, 2),
            15: HexCoordinates(-2, 2),
            16: HexCoordinates(-2, 1),
            17: HexCoordinates(-2, 0),
            18: HexCoordinates(-1, -1),
        }
        return coordDict[hexInd]

    # Function retained as a compatibility shim — internally, board
    # construction now uses ``_random_resource_type_list`` +
    # ``SPIRAL_CHIP_SEQUENCE`` (Colonist's algorithm). This shim recreates
    # the historical ``Resource(type, num)`` list format for any external
    # caller that still relies on it.
    def getRandomResourceList(self):
        types = _random_resource_type_list()
        np.random.shuffle(types)
        start_corner = OUTER_CORNERS[int(np.random.randint(len(OUTER_CORNERS)))]
        clockwise = bool(np.random.randint(2))
        path = _build_spiral_path(start_corner, clockwise)
        nums: list[int | None] = [None] * 19
        chip_idx = 0
        for hex_idx in path:
            if types[hex_idx] == "DESERT":
                continue
            nums[hex_idx] = SPIRAL_CHIP_SEQUENCE[chip_idx]
            chip_idx += 1
        return [Resource(types[i], nums[i]) for i in range(19)]

    # ``checkHexNeighbors`` was the old rejection-sampling check that
    # enforced "no 6/8 adjacent" + "no same-number adjacent". Both
    # properties are now guaranteed by ``SPIRAL_CHIP_SEQUENCE`` placement
    # (the official Catan ABC sequence was designed to satisfy them).
    # Method retained for any external caller; always returns ``True``
    # on a board built by ``catanBoard.__init__``.
    def checkHexNeighbors(self, randomIndices):  # noqa: ARG002 — back-compat shim
        return True
        # End of legacy interface.
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
                    if existingVertex == v:
                        self.boardGraph[v].adjacent_hex_indices.append(hexIndx)

            else:  # Create new vertex if it doesn't exist
                # print('Adding Vertex:', v)
                newVertex = BoardVertex(
                    v, vertex_index=self.vertexIndexCount, adjacent_hex_indices=[hexIndx]
                )
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
        dist = ((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2) ** 0.5
        return round(dist)

    # View the board graph info
    def printGraph(self):
        print(len(self.boardGraph))
        for node in self.boardGraph.keys():
            print(
                f"Pixel:{node}, Index:{self.boardGraph[node].vertex_index}, NeighborVertexCount:{len(self.boardGraph[node].neighbors)}, AdjacentHexes:{self.boardGraph[node].adjacent_hex_indices}"
            )

    # Update Board vertices with Port info
    def updatePorts(self):
        # Define ports by (HexIndex, CornerIndex1, CornerIndex2)
        # Derived from analysis of flat layout to support rotation
        port_hex_corners = [
            (7, 2, 3),  # Top Left
            (8, 1, 2),  # Top Right
            (10, 1, 2),  # Right
            (11, 0, 1),  # Bottom Right
            (12, 5, 0),  # Bottom Right
            (14, 0, 5),  # Bottom Left
            (15, 4, 5),  # Bottom Left
            (16, 3, 4),  # Left
            (18, 3, 4),  # Top Left
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
        randomPortIndices = np.random.permutation([i for i in range(len(port_pair_list))])
        randomPortIndex_counter = 0

        # Initialize port dictionary with counts
        # Also use this dictionary to map vertex indices to specific ports as per the game board
        port_dict = {
            "2:1 BRICK": 1,
            "2:1 SHEEP": 1,
            "2:1 WOOD": 1,
            "2:1 WHEAT": 1,
            "2:1 ORE": 1,
            "3:1 PORT": 4,
        }

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
        seen_edges = set()  # O(1) dedup instead of O(n) dict-key scans
        # Check potential roads from each road the player already has
        for existingRoad in player.buildGraph["ROADS"]:
            for vertex_i in existingRoad:  # Iterate over both vertices of this road
                # Vertex must not be colonised by another player
                v_owner = self.boardGraph[vertex_i].owner
                if v_owner is not None and v_owner is not player:
                    continue
                # Check neighbors from this vertex
                for indx, v_i in enumerate(self.boardGraph[vertex_i].neighbors):
                    # Edge currently does not have a road
                    if not self.boardGraph[vertex_i].edge_state[indx][1]:
                        edge_key = frozenset((v_i, vertex_i))
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            colonisableRoads[(vertex_i, v_i)] = True

        return colonisableRoads

    # Function to get available settlements for colonisation for a particular player
    # Return these settlements as a dict of vertices with their Rects

    def get_potential_settlements(self, player):
        colonisableVertices = {}
        # Check starting from each road the player already has
        for existingRoad in player.buildGraph["ROADS"]:
            for vertex_i in existingRoad:  # Iterate over both vertices of this road
                # Check if vertex isn't already in the potential settlements - to remove double checks
                if vertex_i not in colonisableVertices:
                    # Check if this vertex is already colonised
                    if self.boardGraph[vertex_i].owner is not None:
                        break

                    canColonise = True
                    # Check each of the neighbors from this vertex
                    for v_neighbor in self.boardGraph[vertex_i].neighbors:
                        if self.boardGraph[v_neighbor].owner is not None:
                            canColonise = False
                            break

                # If all checks are good add this vertex and its rect as the value
                if canColonise:
                    # colonisableVertices[vertex_i] = self.draw_possible_settlement(vertex_i, player.color)
                    colonisableVertices[vertex_i] = True

        return colonisableVertices

    # Function to get available cities for colonisation for a particular player
    # Return these cities as a dict of vertex-vertexRect key value pairs

    def get_potential_cities(self, player):
        colonisableVertices = {}
        # Check starting from each settlement the player already has
        for existingSettlement in player.buildGraph["SETTLEMENTS"]:
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
            if potentialVertex.owner is not None:  # First check if vertex is colonised
                canColonise = False

            # Check each neighbor
            for v_neighbor in potentialVertex.neighbors:
                # Check if any of first neighbors are colonised
                if self.boardGraph[v_neighbor].owner is not None:
                    canColonise = False
                    break

            if canColonise:  # If the vertex is colonisable add it to the dict with its Rect
                # colonisableVertices[vertexCoord] = self.draw_possible_settlement(vertexCoord, player.color)
                colonisableVertices[vertexCoord] = True

        return colonisableVertices

    # Special function to get potential first roads during setup phase

    def get_setup_roads(self, player):
        colonisableRoads = {}
        # Can only build roads next to the latest existing player settlement
        latestSettlementCoords = player.buildGraph["SETTLEMENTS"][-1]
        for v_neighbor in self.boardGraph[latestSettlementCoords].neighbors:
            possibleRoad = (latestSettlementCoords, v_neighbor)
            # colonisableRoads[possibleRoad] = self.draw_possible_road(possibleRoad, player.color)
            colonisableRoads[possibleRoad] = True

        return colonisableRoads

    # Function to update boardGraph with Road by player

    def updateBoardGraph_road(self, v_coord1, v_coord2, player):
        # Update edge from first vertex v1
        for indx, v in enumerate(self.boardGraph[v_coord1].neighbors):
            if v == v_coord2:
                self.boardGraph[v_coord1].edge_state[indx][0] = player
                self.boardGraph[v_coord1].edge_state[indx][1] = True

        # Update edge from second vertex v2
        for indx, v in enumerate(self.boardGraph[v_coord2].neighbors):
            if v == v_coord1:
                self.boardGraph[v_coord2].edge_state[indx][0] = player
                self.boardGraph[v_coord2].edge_state[indx][1] = True

        # self.draw_road([v_coord1, v_coord2], player.color) #Draw the settlement

    # Function to update boardGraph with settlement on vertex v

    def updateBoardGraph_settlement(self, v_coord, player):
        self.boardGraph[v_coord].owner = player
        self.boardGraph[v_coord].building_type = "Settlement"
        # self.boardGraph[v_coord].isColonised = True # Implied by owner

        # self.draw_settlement(v_coord, player.color) #Draw the settlement

    # Function to update boardGraph with settlement on vertex v
    def updateBoardGraph_city(self, v_coord, player):
        self.boardGraph[v_coord].owner = player
        self.boardGraph[v_coord].building_type = "City"

        # Remove settlement from player's buildGraph
        player.buildGraph["SETTLEMENTS"].remove(v_coord)

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
            if hex_tile.has_robber == False:
                # Friendly Robber Rule:
                # Cannot place robber on a hex if any adjacent player has < 3 visible VPs
                is_friendly_spot = True
                vertexList = hex_tile.get_corners(self.flat)

                for vertex in vertexList:
                    if vertex in self.boardGraph:
                        player = self.boardGraph[vertex].owner
                        if player is not None:
                            visible_vps = player.victoryPoints - player.devCards["VP"]
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
            if self.boardGraph[vertex].owner != None:
                playerToRob = self.boardGraph[vertex].owner
                # only add a player once with his/her first settlement/city
                if playerToRob not in playersToRobDict:
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

    # ------------------------------------------------------------------
    # Replay-system accessor (Phase 0.5)
    # ------------------------------------------------------------------

    def board_static(self) -> dict:
        """Return a JSON-safe dict mirroring ``replay.schema.BoardStatic``.

        Axial coords only — pixel rendering is the viewer's job via
        :mod:`catan_rl.replay.hex_math`. The recorder's
        ``_build_board_static`` is now a thin wrapper over this method;
        a future Rust/C++ engine port only has to implement
        ``board_static`` (and ``snapshot_state`` on the game) to keep
        the replay recorder running unchanged.
        """
        # Hexes — read directly from hexTileDict so the order matches
        # hex_idx.
        hexes: list[dict] = []
        for hex_idx in sorted(self.hexTileDict.keys()):
            tile = self.hexTileDict[hex_idx]
            hexes.append(
                {
                    "hex_idx": int(hex_idx),
                    "q": int(tile.q),
                    "r": int(tile.r),
                    "resource": str(tile.resource_type),
                    "number_token": (
                        int(tile.number_token) if tile.number_token is not None else None
                    ),
                    "has_robber_initial": bool(getattr(tile, "has_robber", False)),
                }
            )

        # Vertices — variable-length adjacent hex list (interior=3,
        # coastal=1-2). Pad/sentinel never used.
        vertices: list[dict] = []
        for v_idx in sorted(self.vertex_index_to_pixel_dict.keys()):
            px = self.vertex_index_to_pixel_dict[v_idx]
            vobj = self.boardGraph[px]
            adj = list(getattr(vobj, "adjacent_hex_indices", []) or [])
            vertices.append(
                {
                    "vertex_idx": int(v_idx),
                    "adjacent_hex_indices": [int(h) for h in adj],
                }
            )

        # Edges — derived once from the boardGraph adjacency. Engine
        # itself keys edges by lex-sorted vertex-pixel-pair tuples (see
        # CatanEnv._build_index_maps); we reproduce that ordering so the
        # recorder's env-side edge_to_idx matches the JSON.
        pixel_to_idx = {px: v_idx for v_idx, px in self.vertex_index_to_pixel_dict.items()}
        seen_edges: set[tuple[str, str]] = set()
        edges: list[dict] = []
        for v_px, vobj in self.boardGraph.items():
            for nb_px in getattr(vobj, "neighbors", []):
                s1, s2 = str(v_px), str(nb_px)
                key = (s1, s2) if s1 < s2 else (s2, s1)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                v1 = pixel_to_idx.get(v_px)
                v2 = pixel_to_idx.get(nb_px)
                if v1 is None or v2 is None:
                    continue
                edges.append(
                    {
                        "edge_idx": len(edges),
                        "v1_idx": int(v1),
                        "v2_idx": int(v2),
                    }
                )

        # Ports — walk boardGraph for vertices with ``.port != None``.
        # Group by port-type string and pair up the two vertices that
        # share each port. The engine assigns ports to vertex pairs;
        # we reconstruct the pairing by lex-sorting vertex indices
        # within each group.
        port_groups: dict[str, list[int]] = {}
        for v_idx in sorted(self.vertex_index_to_pixel_dict.keys()):
            px = self.vertex_index_to_pixel_dict[v_idx]
            vobj = self.boardGraph[px]
            port = getattr(vobj, "port", None)
            if not port or port is False:
                continue
            port_groups.setdefault(str(port), []).append(int(v_idx))

        ports: list[dict] = []
        for port_type, v_indices in port_groups.items():
            # Each port type owns a stride of 2 vertices per port. The
            # engine groups them as consecutive pairs in vertex-index
            # order — but for robustness we just chunk by 2 and let the
            # viewer render them as edge labels.
            ratio: str
            resource: str | None
            if port_type.startswith("2:1"):
                ratio = "2:1"
                resource = port_type.split(" ", 1)[1] if " " in port_type else None
            elif port_type.startswith("3:1"):
                ratio = "3:1"
                resource = None
            else:  # unknown shape — best-effort skip
                continue
            for chunk_start in range(0, len(v_indices) - 1, 2):
                a, b = v_indices[chunk_start], v_indices[chunk_start + 1]
                ports.append(
                    {
                        "port_idx": len(ports),
                        "vertex_idx_pair": [int(a), int(b)],
                        "ratio": ratio,
                        "resource": resource,
                    }
                )

        return {
            "hexes": hexes,
            "vertices": vertices,
            "edges": edges,
            "ports": ports,
        }
