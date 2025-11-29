"""
Renderer for the Mojo 2D fluid simulation.

Usage:

    mojo run fluid_sim_core.mojo | python fluid_render_client.py

The Mojo program outputs one line per frame with particle positions; this
Python script reads those lines from stdin and renders them with pygame.

This keeps Mojo focused purely on simulation logic, and Python focused on rendering.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

import pygame


Vec2 = Tuple[float, float]


def parse_frame(line: str) -> Tuple[int, int, float, float, float, float, float, float, List[Vec2], List[Vec2]]:
    """
    Parse a single frame line from the Mojo sim.

    Format:
        F <frame_index> <collisions> <bounds_w> <bounds_h> <obstacle_x> <obstacle_y> <obstacle_w> <obstacle_h> <particle_count> x0 y0 vx0 vy0 x1 y1 vx1 vy1 ...
    """
    parts = line.strip().split()
    if not parts or parts[0] != "F":
        raise ValueError("Invalid frame line")

    frame = int(parts[1])
    collisions = int(parts[2])
    bounds_w = float(parts[3])
    bounds_h = float(parts[4])
    obstacle_x = float(parts[5])
    obstacle_y = float(parts[6])
    obstacle_w = float(parts[7])
    obstacle_h = float(parts[8])
    count = int(parts[9])

    coords = parts[10:]
    if len(coords) < count * 4:  # x, y, vx, vy per particle
        raise ValueError("Not enough coordinate data in frame")

    positions: List[Vec2] = []
    velocities: List[Vec2] = []
    for i in range(count):
        x = float(coords[4 * i])
        y = float(coords[4 * i + 1])
        vx = float(coords[4 * i + 2])
        vy = float(coords[4 * i + 3])
        positions.append((x, y))
        velocities.append((vx, vy))

    return frame, collisions, bounds_w, bounds_h, obstacle_x, obstacle_y, obstacle_w, obstacle_h, positions, velocities


def world_to_screen(
    p: Vec2,
    world_w: float,
    world_h: float,
    width_px: int,
    height_px: int,
) -> Tuple[int, int]:
    x = (p[0] / world_w + 0.5) * width_px
    y = (0.5 - p[1] / world_h) * height_px
    return int(x), int(y)


def main() -> None:
    pygame.init()

    width_px, height_px = 1200, 800
    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("Mojo FluidSim Renderer (Python + pygame)")
    clock = pygame.time.Clock()

    particle_radius_px = 2
    running = True

    # Last frame data
    frame_idx = 0
    collisions = 0
    bounds_w = 8.0
    bounds_h = 8.0
    obstacle_x = 0.0
    obstacle_y = -0.8
    obstacle_w = 2.0
    obstacle_h = 0.6
    positions: List[Vec2] = []
    velocities: List[Vec2] = []

    # Simple stats history for graphs
    fps_history: List[float] = []
    coll_history: List[float] = []
    max_history = 240  # ~4 seconds at 60fps

    show_graphs = True

    def v_length(v: Vec2) -> float:
        return (v[0] ** 2 + v[1] ** 2) ** 0.5

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    show_graphs = not show_graphs

        # Read next frame from stdin (blocking)
        line = sys.stdin.readline()
        if not line:
            running = False
            continue

        try:
            frame_idx, collisions, bounds_w, bounds_h, obstacle_x, obstacle_y, obstacle_w, obstacle_h, positions, velocities = parse_frame(line)
        except Exception:
            continue

        # Update histories
        fps = clock.get_fps()
        if fps > 0:
            fps_history.append(fps)
        coll_history.append(float(collisions))
        if len(fps_history) > max_history:
            fps_history = fps_history[-max_history:]
        if len(coll_history) > max_history:
            coll_history = coll_history[-max_history:]

        # Clear and draw
        screen.fill((10, 10, 30))

        # Draw bounds box
        half_w = bounds_w * 0.5
        half_h = bounds_h * 0.5
        top_left = world_to_screen((-half_w, half_h), bounds_w, bounds_h, width_px, height_px)
        bottom_right = world_to_screen((half_w, -half_h), bounds_w, bounds_h, width_px, height_px)

        rect = pygame.Rect(
            top_left[0],
            top_left[1],
            bottom_right[0] - top_left[0],
            bottom_right[1] - top_left[1],
        )
        pygame.draw.rect(screen, (80, 80, 80), rect, 1)

        # Draw obstacle
        ohx = obstacle_w * 0.5
        ohy = obstacle_h * 0.5
        obs_tl = world_to_screen((obstacle_x - ohx, obstacle_y + ohy), bounds_w, bounds_h, width_px, height_px)
        obs_br = world_to_screen((obstacle_x + ohx, obstacle_y - ohy), bounds_w, bounds_h, width_px, height_px)
        obs_rect = pygame.Rect(
            obs_tl[0],
            obs_tl[1],
            obs_br[0] - obs_tl[0],
            obs_br[1] - obs_tl[1],
        )
        pygame.draw.rect(screen, (120, 120, 120), obs_rect, 1)

        # Draw particles with speed-based coloring (like original)
        max_speed = 6.0
        for i, p in enumerate(positions):
            sx, sy = world_to_screen(p, bounds_w, bounds_h, width_px, height_px)
            
            # Calculate speed and map to color gradient
            if i < len(velocities):
                vel = velocities[i]
                spd = min(v_length(vel) / max_speed, 1.0)
                
                # Approximate gradient: dark blue -> cyan -> yellow -> white
                if spd < 0.33:
                    t = spd / 0.33
                    color = (
                        int(30 * (1 - t) + 0 * t),
                        int(60 * (1 - t) + 200 * t),
                        int(120 * (1 - t) + 255 * t),
                    )
                elif spd < 0.66:
                    t = (spd - 0.33) / 0.33
                    color = (
                        int(0 * (1 - t) + 255 * t),
                        int(200 * (1 - t) + 255 * t),
                        int(255 * (1 - t) + 150 * t),
                    )
                else:
                    t = (spd - 0.66) / 0.34
                    color = (
                        int(255 * (1 - t) + 255 * t),
                        int(255 * (1 - t) + 255 * t),
                        int(150 * (1 - t) + 255 * t),
                    )
            else:
                # Fallback color if no velocity data
                color = (120, 200, 255)
            
            pygame.draw.circle(screen, color, (sx, sy), particle_radius_px)

        # Draw simple FPS and collision graphs at top-right
        if show_graphs:
            draw_graphs(screen, width_px, height_px, fps_history, coll_history)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def draw_graphs(
    screen: pygame.Surface,
    width_px: int,
    height_px: int,
    fps_history: List[float],
    coll_history: List[float],
) -> None:
    margin = 10
    graph_width = 260
    graph_height = 80
    right = width_px - margin
    top = margin

    bg_color = (5, 5, 20)
    border_color = (180, 180, 180)
    grid_color = (60, 60, 90)
    fps_color = (80, 220, 80)
    col_color = (220, 180, 80)
    text_color = (230, 230, 230)
    font = pygame.font.SysFont("consolas", 12)

    def draw_single(data: List[float], rect: pygame.Rect, color, y_label: str) -> None:
        pygame.draw.rect(screen, bg_color, rect)
        pygame.draw.rect(screen, border_color, rect, 1)

        step_x = rect.width // 4
        step_y = rect.height // 3
        for i in range(1, 4):
            x = rect.left + i * step_x
            pygame.draw.line(screen, grid_color, (x, rect.top), (x, rect.bottom))
        for i in range(1, 3):
            y = rect.top + i * step_y
            pygame.draw.line(screen, grid_color, (rect.left, y), (rect.right, y))

        max_val = 1.0
        n = len(data)
        if data:
            max_val = max(max(data), 1e-3)
        max_val *= 1.1

        for i in range(1, n):
            x0 = rect.left + int(rect.width * (i - 1) / max(n - 1, 1))
            x1 = rect.left + int(rect.width * i / max(n - 1, 1))
            y0 = rect.bottom - int(rect.height * (data[i - 1] / max_val))
            y1 = rect.bottom - int(rect.height * (data[i] / max_val))
            pygame.draw.line(screen, color, (x0, y0), (x1, y1), 2)

        # Y-axis numeric labels
        for frac in (0.0, 0.5, 1.0):
            value = max_val * frac
            y = rect.bottom - int(rect.height * frac)
            label = f"{value:.0f}"
            surf = font.render(label, True, text_color)
            screen.blit(surf, (rect.left - surf.get_width() - 4, y - surf.get_height() // 2))

        # X-axis numeric labels: 0, mid, last index
        if n > 0:
            for idx in (0, n // 2, n - 1):
                frac_x = idx / max(n - 1, 1)
                x = rect.left + int(rect.width * frac_x)
                surf = font.render(str(idx), True, text_color)
                screen.blit(surf, (x - surf.get_width() // 2, rect.bottom + 2))

        # Y label
        label_surf = font.render(y_label, True, text_color)
        screen.blit(label_surf, (rect.left - label_surf.get_width() - 4, rect.top - 2))

    # FPS graph (top)
    fps_rect = pygame.Rect(right - graph_width, top, graph_width, graph_height)
    draw_single(fps_history, fps_rect, fps_color, "FPS")

    # Collisions graph (below)
    col_rect = pygame.Rect(
        right - graph_width, top + graph_height + 24, graph_width, graph_height
    )
    draw_single(coll_history, col_rect, col_color, "Coll/s")


if __name__ == "__main__":
    main()


