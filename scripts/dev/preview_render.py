"""Preview script for the shared rendering primitives.

Boots a window showing the complete scene (water + island + tiles +
bevel + number tokens + resource symbols + ports + vertex markers +
robber) so the user can spot-check the visuals before the labeling-tool
and live-GUI refactors land.

Usage::

    PYTHONPATH=src python scripts/preview_render.py
    PYTHONPATH=src python scripts/preview_render.py --seed 42

Press Q or close the window to exit.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np

# REPO_ROOT preserved for any downstream default-path usage. The
# previous ``sys.path.insert(REPO_ROOT/'src')`` shim was a no-op
# (its target was ``scripts/src/`` due to ``parent.parent`` only
# reaching ``scripts/``) and is dropped in the maturin sole-backend
# cutover — ``catan_rl`` is importable via the install path now.
REPO_ROOT = Path(__file__).resolve().parent.parent

import pygame

from catan_rl.engine.board import catanBoard
from catan_rl.gui import render, render_constants
from catan_rl.gui.render_constants import (
    PORT_PUSH_DISTANCE,
    VERTEX_STATE_LEGAL,
)


def _parse_port(label: str) -> tuple[str, str | None]:
    """Split an engine port string like ``"2:1 BRICK"`` / ``"3:1 PORT"``
    into ``(ratio, resource_type_or_None)``."""
    if "3:1" in label:
        return ("3:1", None)
    for full in ("BRICK", "WOOD", "WHEAT", "SHEEP", "ORE"):
        if full in label:
            return ("2:1", full)
    return ("2:1", None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--screen-width", type=int, default=1100)
    parser.add_argument("--screen-height", type=int, default=900)
    parser.add_argument("--no-bevel", action="store_true", help="Disable hex bevel shading.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    board = catanBoard()

    pygame.init()
    pygame.display.set_caption(f"Render Preview (seed={args.seed})")
    screen = pygame.display.set_mode((args.screen_width, args.screen_height))

    centers = [board.hexTileDict[i].to_pixel(board.flat) for i in range(19)]
    cx = sum(float(c.x) for c in centers) / len(centers)
    cy = sum(float(c.y) for c in centers) / len(centers)

    # Collect port edges from the board for ship rendering.
    port_edges: list[tuple[int, int, str, str | None]] = []
    seen: set[tuple[int, int]] = set()
    for v_obj in board.boardGraph.values():
        port = getattr(v_obj, "port", None)
        if not port:
            continue
        v_idx = v_obj.vertex_index
        for nb_pt in v_obj.neighbors:
            nb_obj = board.boardGraph[nb_pt]
            if getattr(nb_obj, "port", None) != port:
                continue
            nb_idx = nb_obj.vertex_index
            key = (min(v_idx, nb_idx), max(v_idx, nb_idx))
            if key in seen:
                continue
            seen.add(key)
            ratio, resource = _parse_port(port)
            port_edges.append((key[0], key[1], ratio, resource))

    def render_scene() -> None:
        render.draw_water(screen, (args.screen_width, args.screen_height))
        render.draw_island_outline(screen, centers)
        for h_idx in range(19):
            hex_tile = board.hexTileDict[h_idx]
            render.draw_hex_tile(screen, hex_tile, board, with_bevel=not args.no_bevel)
            tile_center = hex_tile.to_pixel(board.flat)
            sym_y = int(tile_center.y) + render_constants.RESOURCE_SYMBOL_VERTICAL_OFFSET
            render.draw_resource_symbol(
                screen,
                (int(tile_center.x), sym_y),
                hex_tile.resource_type,
            )
            num = getattr(hex_tile, "number_token", None)
            if num is not None and num != 0:
                tok_y = int(tile_center.y) + render_constants.NUMBER_TOKEN_VERTICAL_OFFSET
                render.draw_number_token(screen, (int(tile_center.x), tok_y), num)
        # Ports.
        for v1_idx, v2_idx, ratio, resource in port_edges:
            v1_px = board.vertex_index_to_pixel_dict[v1_idx]
            v2_px = board.vertex_index_to_pixel_dict[v2_idx]
            mid_x = (float(v1_px.x) + float(v2_px.x)) / 2.0
            mid_y = (float(v1_px.y) + float(v2_px.y)) / 2.0
            dx = mid_x - cx
            dy = mid_y - cy
            d = math.hypot(dx, dy) or 1.0
            ax = mid_x + dx / d * PORT_PUSH_DISTANCE
            ay = mid_y + dy / d * PORT_PUSH_DISTANCE
            render.draw_port_planks(
                screen,
                (int(ax), int(ay)),
                (int(v1_px.x), int(v1_px.y)),
                (int(v2_px.x), int(v2_px.y)),
            )
            render.draw_port_ship(screen, ratio, resource, (int(ax), int(ay)))
        # Vertex markers: highlight every legal-to-settle vertex on
        # an empty board (i.e., all 54).
        for v_pt in board.boardGraph:
            v_obj = board.boardGraph[v_pt]
            render.draw_vertex_marker(screen, (int(v_pt.x), int(v_pt.y)), VERTEX_STATE_LEGAL)
        # Robber.
        robber_hex = next(t for t in board.hexTileDict.values() if t.has_robber)
        render.draw_robber_pawn(screen, robber_hex, board)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            quit_press = event.type == pygame.KEYDOWN and event.unicode.lower() == "q"
            if event.type == pygame.QUIT or quit_press:
                running = False
        render_scene()
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
