"""
Renderer for the Mojo 3D fluid simulation.

Usage:

    mojo run fluid_sim_core_3d.mojo | python fluid_render_client_3d.py

Configuration:
    Edit fluid_sim_core_3d.mojo to change:
    - particles_per_axis (default: 12)
    - max_frames (default: 10000)
    - scenario_type (default: "whirlpool")
    - scenario_activation_time (default: 5.0)

The Mojo program outputs one line per frame with particle positions; this
Python script reads those lines from stdin and renders them with pygame using 3D projection.

This keeps Mojo focused purely on simulation logic, and Python focused on rendering.
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple, Optional

import pygame

Vec3 = Tuple[float, float, float]


def parse_frame(line: str) -> Tuple[int, int, float, float, float, int, float, float, float, float, List[Vec3], List[Vec3]]:
    """
    Parse a single frame line from the Mojo sim.

    Format:
        F <frame_index> <collisions> <bounds_w> <bounds_h> <bounds_d> <scenario_active> <whirlpool_x> <whirlpool_y> <whirlpool_z> <whirlpool_radius> <particle_count> x0 y0 z0 vx0 vy0 vz0 x1 y1 z1 vx1 vy1 vz1 ...
    """
    parts = line.strip().split()
    if not parts or parts[0] != "F":
        raise ValueError("Invalid frame line")

    frame = int(parts[1])
    collisions = int(parts[2])
    bounds_w = float(parts[3])
    bounds_h = float(parts[4])
    bounds_d = float(parts[5])
    scenario_active = int(parts[6])
    whirlpool_x = float(parts[7])
    whirlpool_y = float(parts[8])
    whirlpool_z = float(parts[9])
    whirlpool_radius = float(parts[10])
    count = int(parts[11])

    coords = parts[12:]
    if len(coords) < count * 6:  # x, y, z, vx, vy, vz per particle
        raise ValueError("Not enough coordinate data in frame")

    positions: List[Vec3] = []
    velocities: List[Vec3] = []
    for i in range(count):
        x = float(coords[6 * i])
        y = float(coords[6 * i + 1])
        z = float(coords[6 * i + 2])
        vx = float(coords[6 * i + 3])
        vy = float(coords[6 * i + 4])
        vz = float(coords[6 * i + 5])
        positions.append((x, y, z))
        velocities.append((vx, vy, vz))

    return frame, collisions, bounds_w, bounds_h, bounds_d, scenario_active, whirlpool_x, whirlpool_y, whirlpool_z, whirlpool_radius, positions, velocities


def v3_length(v: Vec3) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def v3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v3_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v3_normalize(v: Vec3) -> Vec3:
    length = v3_length(v)
    if length <= 1e-8:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / length
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def v3_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def project_3d_to_2d(
    pos: Vec3,
    camera_distance: float,
    camera_angle_x: float,
    camera_angle_y: float,
    camera_zoom: float,
    width_px: int,
    height_px: int,
    fov: float = math.pi / 4.0,
    near_plane: float = 0.1,
) -> Tuple[int, int, float]:
    """Project 3D world position to 2D screen coordinates using perspective projection."""
    cam_x = camera_distance * math.cos(camera_angle_y) * math.sin(camera_angle_x)
    cam_y = camera_distance * math.sin(camera_angle_y)
    cam_z = camera_distance * math.cos(camera_angle_y) * math.cos(camera_angle_x)
    camera_pos = (cam_x, cam_y, cam_z)

    look_at = (0.0, 0.0, 0.0)
    forward = v3_normalize(v3_sub(look_at, camera_pos))

    up_world = (0.0, 1.0, 0.0)
    right = v3_normalize(v3_cross(forward, up_world))
    up = v3_normalize(v3_cross(right, forward))

    to_point = v3_sub(pos, camera_pos)
    x_cam = v3_dot(to_point, right)
    y_cam = v3_dot(to_point, up)
    z_cam = v3_dot(to_point, forward)

    if z_cam < near_plane:
        return (-9999, -9999, -9999.0)

    aspect = width_px / height_px
    fov_scale = 1.0 / math.tan(fov / 2.0)

    # Fixed aspect ratio: divide X by aspect to keep square pixels
    x_proj = (x_cam / z_cam) * fov_scale * camera_zoom / aspect
    y_proj = (y_cam / z_cam) * fov_scale * camera_zoom

    screen_x = int(width_px * 0.5 + x_proj * width_px * 0.5)
    screen_y = int(height_px * 0.5 - y_proj * height_px * 0.5)

    return screen_x, screen_y, z_cam


def main() -> None:
    pygame.init()

    width_px, height_px = 1200, 800
    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("Mojo 3D FluidSim Renderer (Python + pygame)")
    clock = pygame.time.Clock()

    particle_radius_px = 3
    running = True

    # Camera state
    camera_distance = 10.0
    camera_angle_x = 0.5
    camera_angle_y = 0.3
    camera_zoom = 1.0
    mouse_dragging = False
    last_mouse_pos = (0, 0)

    # Last frame data
    frame_idx = 0
    collisions = 0
    bounds_w = 4.0
    bounds_h = 4.0
    bounds_d = 4.0
    scenario_active = 0
    whirlpool_x = 0.0
    whirlpool_y = -1.0
    whirlpool_z = 0.0
    whirlpool_radius = 3.0
    positions: List[Vec3] = []
    velocities: List[Vec3] = []

    # Simple stats history for graphs
    fps_history: List[float] = []
    coll_history: List[float] = []
    max_history = 240  # ~4 seconds at 60fps

    show_graphs = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    show_graphs = not show_graphs
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    camera_zoom = min(camera_zoom * 1.1, 3.0)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    camera_zoom = max(camera_zoom / 1.1, 0.3)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_dragging = True
                    last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    rotation_speed = 0.005
                    camera_angle_x += dx * rotation_speed
                    camera_angle_y = max(
                        min(camera_angle_y - dy * rotation_speed, math.pi / 2 - 0.1),
                        -math.pi / 2 + 0.1
                    )
                    last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = 1.1 if event.y > 0 else 1.0 / 1.1
                camera_zoom = max(0.3, min(3.0, camera_zoom * zoom_factor))

        keys = pygame.key.get_pressed()
        rotation_speed = 0.02
        if keys[pygame.K_LEFT]:
            camera_angle_x -= rotation_speed
        if keys[pygame.K_RIGHT]:
            camera_angle_x += rotation_speed
        if keys[pygame.K_UP]:
            camera_angle_y = max(camera_angle_y - rotation_speed, -math.pi / 2 + 0.1)
        if keys[pygame.K_DOWN]:
            camera_angle_y = min(camera_angle_y + rotation_speed, math.pi / 2 - 0.1)

        # Read next frame from stdin (non-blocking)
        line = sys.stdin.readline()
        if not line:
            continue

        try:
            frame_idx, collisions, bounds_w, bounds_h, bounds_d, scenario_active, whirlpool_x, whirlpool_y, whirlpool_z, whirlpool_radius, positions, velocities = parse_frame(line)
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

        # Draw bounds box (wireframe)
        half_w = bounds_w * 0.5
        half_h = bounds_h * 0.5
        half_d = bounds_d * 0.5

        corners = [
            (-half_w, -half_h, -half_d),
            (half_w, -half_h, -half_d),
            (half_w, half_h, -half_d),
            (-half_w, half_h, -half_d),
            (-half_w, -half_h, half_d),
            (half_w, -half_h, half_d),
            (half_w, half_h, half_d),
            (-half_w, half_h, half_d),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            p1_x, p1_y, p1_z = project_3d_to_2d(
                corners[i], camera_distance, camera_angle_x, camera_angle_y, camera_zoom, width_px, height_px
            )
            p2_x, p2_y, p2_z = project_3d_to_2d(
                corners[j], camera_distance, camera_angle_x, camera_angle_y, camera_zoom, width_px, height_px
            )
            if p1_z > 0 and p2_z > 0:
                pygame.draw.line(screen, (80, 80, 80), (p1_x, p1_y), (p2_x, p2_y), 1)

        # Draw scenario visualization (whirlpool center and radius)
        if scenario_active:
            centre_x, centre_y, centre_z = whirlpool_x, whirlpool_y, whirlpool_z
            sx, sy, depth = project_3d_to_2d(
                (centre_x, centre_y, centre_z), camera_distance, camera_angle_x, camera_angle_y, camera_zoom, width_px, height_px
            )
            if depth > 0:
                # Draw whirlpool center indicator
                pygame.draw.circle(screen, (255, 100, 100), (int(sx), int(sy)), 8, 2)
                # Draw radius indicator (circle in XZ plane)
                radius_points = []
                for angle in range(0, 360, 10):
                    rad = math.radians(angle)
                    px = centre_x + whirlpool_radius * math.cos(rad)
                    pz = centre_z + whirlpool_radius * math.sin(rad)
                    py = centre_y
                    psx, psy, pdepth = project_3d_to_2d(
                        (px, py, pz), camera_distance, camera_angle_x, camera_angle_y, camera_zoom, width_px, height_px
                    )
                    if pdepth > 0:
                        radius_points.append((int(psx), int(psy)))
                if len(radius_points) > 2:
                    pygame.draw.lines(screen, (255, 150, 150), True, radius_points, 1)

        # Draw particles with speed-based coloring and depth sorting
        max_speed = 6.0
        particles_to_draw = []
        for i in range(len(positions)):
            pos = positions[i]
            vel = velocities[i] if i < len(velocities) else (0.0, 0.0, 0.0)
            spd = min(v3_length(vel) / max_speed, 1.0)

            # Color gradient based on speed
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

            sx, sy, depth = project_3d_to_2d(
                pos, camera_distance, camera_angle_x, camera_angle_y, camera_zoom, width_px, height_px
            )
            if depth > 0 and sx >= 0 and sx < width_px and sy >= 0 and sy < height_px:
                particles_to_draw.append((sx, sy, depth, color))

        # Sort by depth (back to front)
        particles_to_draw.sort(key=lambda p: p[2], reverse=True)
        for sx, sy, depth, color in particles_to_draw:
            pygame.draw.circle(screen, color, (int(sx), int(sy)), particle_radius_px)

        # Draw scenario status
        font = pygame.font.SysFont("consolas", 14)
        if scenario_active:
            status_text = f"WHIRLPOOL ACTIVE - Frame: {frame_idx}"
            status_color = (255, 100, 100)
        else:
            status_text = f"Whirlpool pending - Frame: {frame_idx}"
            status_color = (200, 200, 200)
        text_surf = font.render(status_text, True, status_color)
        screen.blit(text_surf, (10, 10))

        # Draw graphs
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
    top = margin + 30  # Leave space for scenario status

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

        # X-axis numeric labels
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

