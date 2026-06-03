"""Constants for the shared rendering module (plan §B).

Pure data — colors, sizes, font hints, lookup tables. No logic.
Separated from ``render.py`` so the draw module stays focused on draw
calls. Both ``labeling/ui.py`` and ``gui/view.py`` import from here
when they need to compose layouts (e.g., when deciding how far above a
hex to place a number token).

Layer order (drawn back to front by ``render.py``):

    water  → island  → hex tiles (with bevel)  → number tokens
          → resource symbols  → ports  → vertex markers
          → prior picks  → robber  → top-bar overlay

Colors are RGB triples in 0..255.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Background + island
# ---------------------------------------------------------------------------

WATER_COLOR: tuple[int, int, int] = (35, 90, 150)
"""Deep ocean blue filling the entire canvas."""

SAND_COLOR: tuple[int, int, int] = (235, 215, 165)
"""Tan/beach color for the island outline."""

ISLAND_OUTLINE_BUFFER: int = 56
"""Outward push (px) from each hex center when building the island polygon."""

ISLAND_OUTLINE_JITTER: int = 6
"""Perpendicular jitter (px) for the "coastline" look."""

ISLAND_OUTLINE_BORDER_COLOR: tuple[int, int, int] = (200, 175, 125)
"""Slightly darker tan for the coast line."""

# ---------------------------------------------------------------------------
# Hex tile colors (one per resource type)
# ---------------------------------------------------------------------------

TILE_COLORS: dict[str, tuple[int, int, int]] = {
    "WOOD": (40, 130, 50),
    "BRICK": (190, 80, 50),
    "SHEEP": (160, 215, 110),
    "WHEAT": (240, 200, 70),
    "ORE": (120, 120, 140),
    "DESERT": (220, 200, 150),
}
"""RGB triples per resource. Falls back to gray for unknown types."""

TILE_FALLBACK_COLOR: tuple[int, int, int] = (90, 90, 90)
"""Used when ``hex_tile.resource_type`` is missing or unrecognised."""

# Bevel adds a lighter shade to the top half and a darker shade to the
# bottom half, faking soft 3D shading without textures.
HEX_BEVEL_LIGHTEN: float = 0.18
"""Fractional brightness increase applied to the top half of each hex."""

HEX_BEVEL_DARKEN: float = 0.20
"""Fractional brightness decrease applied to the bottom half."""

HEX_OUTLINE_COLOR: tuple[int, int, int] = (15, 15, 20)
"""Dark border around each hex polygon."""

# ---------------------------------------------------------------------------
# Number tokens (Colonist-style: square white tile, red 6/8, pip dots)
# ---------------------------------------------------------------------------

NUMBER_TOKEN_SIZE: int = 40
"""Side length (px) of the square white token tile."""

NUMBER_TOKEN_CORNER_RADIUS: int = 4
"""Border radius (px) for the token tile rectangle."""

NUMBER_TOKEN_BG: tuple[int, int, int] = (250, 246, 232)
"""Cream-white background for the token tile."""

NUMBER_TOKEN_BORDER: tuple[int, int, int] = (50, 50, 50)
"""Token tile border color."""

NUMBER_TEXT_RED: tuple[int, int, int] = (190, 30, 30)
"""Used for 6 and 8 — the high-pip danger numbers."""

NUMBER_TEXT_DEFAULT: tuple[int, int, int] = (30, 60, 30)
"""Dark green for non-danger numbers."""

PIP_DOT_COLOR: tuple[int, int, int] = (30, 60, 30)
"""Color of the pip dots under the number."""

PIP_DOT_COLOR_RED: tuple[int, int, int] = (190, 30, 30)
"""Pip dot color for 6 and 8 — matches the number text."""

PIP_DOT_RADIUS: int = 2
"""Radius (px) of each pip dot."""

PIP_DOT_GAP: int = 5
"""Spacing (px) between adjacent pip dot centers."""

PIP_COUNTS: dict[int, int] = {
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
}
"""Number of ways to roll each sum on 2d6 (7 is reserved for the
robber and has no token, so it's not in the table)."""

# ---------------------------------------------------------------------------
# Resource symbols (unicode + drawn-polygon fallback)
# ---------------------------------------------------------------------------

RESOURCE_EMOJI: dict[str, str] = {
    "WOOD": "🌲",
    "BRICK": "🧱",
    "SHEEP": "🐑",
    "WHEAT": "🌾",
    "ORE": "⛰",
    "DESERT": "🌵",
}
"""Per-resource emoji. Falls back to a drawn polygon if the system font
lacks the glyph (probe at module load — see ``render.py`` §3)."""

RESOURCE_SYMBOL_FONT_SIZE: int = 28
"""Pixel size requested from pygame when rendering emoji symbols."""

RESOURCE_SYMBOL_VERTICAL_OFFSET: int = -24
"""Pixels above the hex center to place the symbol. Colonist layout:
resource icon in the upper half, number token in the lower half."""

NUMBER_TOKEN_VERTICAL_OFFSET: int = 16
"""Pixels below the hex center to place the number token. Pairs with
``RESOURCE_SYMBOL_VERTICAL_OFFSET`` to give symbol + token clean
vertical separation (~3 px gap, no overlap)."""

# Candidate emoji-capable system fonts probed in order at module load.
EMOJI_FONT_CANDIDATES: tuple[str, ...] = (
    "Apple Color Emoji",  # macOS
    "Segoe UI Emoji",  # Windows
    "Noto Color Emoji",  # Linux
    "Twitter Color Emoji",
    "Symbola",
)

EMOJI_PROBE_MIN_BBOX_WIDTH: int = 8
"""A rendered probe surface must have at least this many non-background
pixels in its bounding box width to count as "the font has this glyph."
Below this threshold we flip to the polygon fallback."""

# Fallback polygon icon colors per resource — kept distinct from the
# tile base colors so the icon reads even on its own hex.
ICON_FALLBACK_COLORS: dict[str, tuple[int, int, int]] = {
    "WOOD": (20, 80, 30),
    "BRICK": (140, 60, 35),
    "SHEEP": (250, 250, 250),
    "WHEAT": (220, 180, 50),
    "ORE": (80, 80, 100),
    "DESERT": (90, 130, 60),
}

# ---------------------------------------------------------------------------
# Ports (ship-shaped markers with wooden-plank connectors)
# ---------------------------------------------------------------------------

PORT_SHIP_WIDTH: int = 42
"""Width (px) of the ship icon."""

PORT_SHIP_HEIGHT: int = 36
"""Height (px) of the ship icon (hull + mast + sail)."""

PORT_PUSH_DISTANCE: float = 60.0
"""Distance from the port edge midpoint outward to the ship anchor."""

PORT_HULL_COLOR: tuple[int, int, int] = (140, 90, 50)
"""Wooden brown for the ship hull."""

PORT_SAIL_COLOR: tuple[int, int, int] = (245, 240, 215)
"""Cream for the ship sail (so the label text reads)."""

PORT_MAST_COLOR: tuple[int, int, int] = (90, 60, 30)
"""Dark brown for the mast."""

PORT_PLANK_COLOR: tuple[int, int, int] = (170, 125, 70)
"""Light tan for the wooden plank lines connecting ship to vertices."""

PORT_PLANK_WIDTH: int = 4
"""Thickness (px) of each plank line."""

PORT_LABEL_COLOR: tuple[int, int, int] = (30, 30, 30)
"""Text color for the trade-ratio label on the badge below the ship."""

PORT_LABEL_BG_COLOR: tuple[int, int, int] = (250, 246, 220)
"""Cream background for the label badge — high contrast over water."""

PORT_LABEL_BG_BORDER_COLOR: tuple[int, int, int] = (90, 60, 30)
"""Dark wooden border around the label badge."""

PORT_LABEL_VERTICAL_OFFSET: int = 36
"""Pixels below the ship anchor where the label badge sits. Bumped from
22 to 36 to clear the hull when the badge holds both text + icon
(wider/taller than the original text-only badge)."""

PORT_LABEL_PADDING: int = 5
"""Padding (px) inside the label badge background rect."""

PORT_LABEL_ICON_SIZE: int = 18
"""Pixel font size for the resource icon inside the port label badge.
Smaller than ``RESOURCE_SYMBOL_FONT_SIZE`` because the badge is compact."""

PORT_LABEL_GENERIC_GLYPH: str = "?"
"""Drawn on 3:1 generic ports in place of a resource icon."""

# Resource short codes used in port labels.
PORT_RESOURCE_SHORT: dict[str, str] = {
    "BRICK": "B",
    "WOOD": "W",
    "WHEAT": "Wh",
    "SHEEP": "S",
    "ORE": "O",
}

# ---------------------------------------------------------------------------
# Vertex markers (state-keyed)
# ---------------------------------------------------------------------------

VERTEX_STATE_LEGAL = "legal"
VERTEX_STATE_SELECTED = "selected"
VERTEX_STATE_SETTLED_P1 = "settled_p1"
VERTEX_STATE_SETTLED_P2 = "settled_p2"
VERTEX_STATE_IDLE = "idle"

# All valid vertex states. New code should rely on this tuple for
# parametrised tests rather than reaching for the individual constants.
VERTEX_STATES: tuple[str, ...] = (
    VERTEX_STATE_LEGAL,
    VERTEX_STATE_SELECTED,
    VERTEX_STATE_SETTLED_P1,
    VERTEX_STATE_SETTLED_P2,
    VERTEX_STATE_IDLE,
)

VERTEX_COLORS: dict[str, tuple[int, int, int]] = {
    VERTEX_STATE_LEGAL: (255, 200, 60),  # amber gold
    VERTEX_STATE_SELECTED: (255, 240, 100),  # brighter yellow
    VERTEX_STATE_SETTLED_P1: (50, 100, 220),  # P1 blue
    VERTEX_STATE_SETTLED_P2: (220, 80, 80),  # P2 red
    VERTEX_STATE_IDLE: (90, 90, 95),  # gray
}

VERTEX_RADII: dict[str, int] = {
    VERTEX_STATE_LEGAL: 8,
    VERTEX_STATE_SELECTED: 14,
    VERTEX_STATE_SETTLED_P1: 12,
    VERTEX_STATE_SETTLED_P2: 12,
    VERTEX_STATE_IDLE: 5,
}

VERTEX_BORDER_COLOR: tuple[int, int, int] = (20, 20, 25)
"""Thin border around vertex markers for contrast against sand."""

# ---------------------------------------------------------------------------
# Robber pawn
# ---------------------------------------------------------------------------

ROBBER_COLOR: tuple[int, int, int] = (50, 50, 50)
"""Dark gray for the robber pawn."""

ROBBER_BASE_WIDTH: int = 18
"""Width (px) of the robber pawn base."""

ROBBER_HEIGHT: int = 26
"""Total height (px) of the robber pawn."""

# ---------------------------------------------------------------------------
# Player colors (mirror labeling/ui.py defaults — kept here so the live
# GUI and the labeling tool agree)
# ---------------------------------------------------------------------------

PLAYER_COLORS: dict[int, tuple[int, int, int]] = {
    0: (50, 100, 220),  # P1 blue
    1: (220, 80, 80),  # P2 red
}
