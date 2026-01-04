# Adapted from Red Blob Games Hex Grid logic
# Generated code -- CC0 -- No Rights Reserved -- http://www.redblobgames.com/grids/hexagons/

import math
import collections

# Point class for pixel coordinates
Point = collections.namedtuple("Point", ["x", "y"])


class HexLayout:
    """
    Configuration for the Hex Grid Layout.
    Stores size, origin, and orientation matrix for pointy-topped hexes.
    """

    def __init__(self, size, origin):
        self.size = size  # Point(x, y)
        self.origin = origin  # Point(x, y)

        # Standard 3x2 matrix for pointy-topped hexes
        # Forward matrix (Hex -> Pixel)
        self.f0 = math.sqrt(3.0)
        self.f1 = math.sqrt(3.0) / 2.0
        self.f2 = 0.0
        self.f3 = 3.0 / 2.0

        # Inverse matrix (Pixel -> Hex)
        self.b0 = math.sqrt(3.0) / 3.0
        self.b1 = -1.0 / 3.0
        self.b2 = 0.0
        self.b3 = 2.0 / 3.0

        # Start angle for corners (0.5 * 60 degrees = 30 degrees)
        self.start_angle = 0.5


class HexCoordinates:
    """
    Represents a location on the hexagonal grid using Cube Coordinates (q, r, s).
    Replaces the functional logic from hexLib.py with object-oriented methods.
    """

    # Directions for neighbors (0 to 5)
    # Corresponds to: E, NE, NW, W, SW, SE (depending on orientation)
    _directions = [
        (1, 0, -1), (1, -1, 0), (0, -1, 1),
        (-1, 0, 1), (-1, 1, 0), (0, 1, -1)
    ]

    def __init__(self, q, r):
        self.q = q
        self.r = r
        self.s = -q - r
        # Floating point arithmetic might cause slight deviations, so we round for the check if inputs are floats
        # But typically q and r are ints.
        assert round(self.q + self.r + self.s) == 0, "q + r + s must be 0"

    def __eq__(self, other):
        if isinstance(other, HexCoordinates):
            return self.q == other.q and self.r == other.r and self.s == other.s
        return False

    def __hash__(self):
        return hash((self.q, self.r, self.s))

    def __add__(self, other):
        if isinstance(other, HexCoordinates):
            return HexCoordinates(self.q + other.q, self.r + other.r)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, HexCoordinates):
            return HexCoordinates(self.q - other.q, self.r - other.r)
        return NotImplemented

    def __repr__(self):
        return f"HexCoordinates(q={self.q}, r={self.r}, s={self.s})"

    def get_neighbor(self, direction_index):
        """Returns a new HexCoordinates object for the neighbor in the given direction index (0-5)."""
        dq, dr, ds = self._directions[direction_index % 6]
        return HexCoordinates(self.q + dq, self.r + dr)

    def distance_to(self, other):
        """Calculates grid distance to another HexCoordinates object."""
        if not isinstance(other, HexCoordinates):
            raise ValueError("Argument must be of type HexCoordinates")
        return (abs(self.q - other.q) + abs(self.r - other.r) + abs(self.s - other.s)) // 2

    def to_pixel(self, layout_config):
        """Converts grid coords to (x,y) pixel center using the provided layout configuration."""
        x = (layout_config.f0 * self.q + layout_config.f1 * self.r) * layout_config.size.x
        y = (layout_config.f2 * self.q + layout_config.f3 * self.r) * layout_config.size.y
        return Point(x + layout_config.origin.x, y + layout_config.origin.y)

    def get_corners(self, layout_config):
        """Returns a list of 6 (x,y) tuples representing the corners of the hex."""
        corners = []
        center = self.to_pixel(layout_config)
        for i in range(6):
            angle = 2.0 * math.pi * (layout_config.start_angle - i) / 6.0
            offset_x = layout_config.size.x * math.cos(angle)
            offset_y = layout_config.size.y * math.sin(angle)
            corners.append(Point(round(center.x + offset_x, 2), round(center.y + offset_y, 2)))
        return corners


class BoardTile(HexCoordinates):
    """
    Represents a game tile on the board.
    Inherits from HexCoordinates so a Tile is a location.
    """

    def __init__(self, q, r, resource_type, number_token, index_id):
        super().__init__(q, r)
        self.resource_type = resource_type
        self.number_token = number_token
        self.index_id = index_id
        self.has_robber = False

    def debug_info(self):
        """Prints the tile status."""
        print(f"Index:{self.index_id}; Resource:{self.resource_type}; Number:{self.number_token}; Coords:({self.q}, {self.r}, {self.s})")


class BoardVertex:
    """
    Represents a vertex on the board where settlements/cities can be built.
    """

    def __init__(self, pixel_location, owner=None, building_type=None, vertex_index=None, adjacent_hex_indices=None):
        self.pixel_location = pixel_location  # tuple (x, y) or Point
        self.owner = owner  # Player object or None
        self.building_type = building_type  # None, 'Settlement', 'City'

        # Additional attributes for game logic
        self.vertex_index = vertex_index
        self.adjacent_hex_indices = adjacent_hex_indices if adjacent_hex_indices is not None else []
        self.neighbors = []  # List of adjacent BoardVertex objects (connected by edges)
        self.port = None  # Port type if any

        # Edge state to track roads: List of [Player, is_road] for edges connected to this vertex
        self.edge_state = [[None, False], [None, False], [None, False]]

    def is_adjacent_to(self, other_vertex, threshold=80):
        """
        Checks if this vertex is adjacent to another vertex based on distance.
        Default threshold is 80 pixels (standard edge length).
        """
        x1, y1 = self.pixel_location
        x2, y2 = other_vertex.pixel_location
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return round(dist) == threshold
