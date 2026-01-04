import pygame
import sys
from catan.engine.board import catanBoard
from catan.engine.geometry import *


class DebugBoardView:
    def __init__(self):
        pygame.init()
        self.board = catanBoard()
        self.screen = pygame.display.set_mode(self.board.size)
        pygame.display.set_caption('Catan Debug - Hex, Corner & Vertex Indices')

        self.font_hex = pygame.font.SysFont('arial', 24, bold=True)
        self.font_corner = pygame.font.SysFont('arial', 14, bold=True)
        self.font_vertex = pygame.font.SysFont('arial', 12)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((230, 230, 230))  # Light gray background

            # Draw Hexes and Labels
            for hex_idx, hex_tile in self.board.hexTileDict.items():
                corners = hex_tile.get_corners(self.board.flat)
                center = hex_tile.to_pixel(self.board.flat)

                # Draw Hex Polygon
                pygame.draw.polygon(self.screen, (255, 255, 255), corners, 0)
                pygame.draw.polygon(self.screen, (0, 0, 0), corners, 2)  # Outline

                # Draw Hex Index (Center, Black)
                text_hex = self.font_hex.render(str(hex_idx), True, (0, 0, 0))
                self.screen.blit(text_hex, (center.x - text_hex.get_width() //
                                 2, center.y - text_hex.get_height()//2))

                # Draw Corner Indices (Inside corners, Red)
                for i, corner in enumerate(corners):
                    # Vector from corner to center
                    vec_x = center.x - corner.x
                    vec_y = center.y - corner.y
                    # Normalize and scale to push text inside
                    mag = (vec_x**2 + vec_y**2)**0.5
                    offset_x = (vec_x / mag) * 25
                    offset_y = (vec_y / mag) * 25

                    text_corner = self.font_corner.render(str(i), True, (255, 0, 0))
                    self.screen.blit(text_corner, (corner.x + offset_x - text_corner.get_width() //
                                     2, corner.y + offset_y - text_corner.get_height()//2))

            # Draw Global Vertex Indices (On vertices, Blue)
            for pixel_pt, vertex in self.board.boardGraph.items():
                # Draw small circle at vertex
                pygame.draw.circle(self.screen, (0, 0, 255), (int(pixel_pt.x), int(pixel_pt.y)), 4)

                # Draw Vertex Index
                text_vertex = self.font_vertex.render(str(vertex.vertex_index), True, (0, 0, 255))
                # Offset slightly up/right to not cover the corner number
                self.screen.blit(text_vertex, (pixel_pt.x + 5, pixel_pt.y - 15))

            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    view = DebugBoardView()
    view.run()
