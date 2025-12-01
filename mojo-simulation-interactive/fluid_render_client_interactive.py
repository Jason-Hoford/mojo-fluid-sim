"""
Interactive Renderer for the Mojo 3D fluid simulation.

Usage:
    python fluid_render_client_interactive.py

Requirements:
    - pygame
    - mojo (in PATH)
"""

import sys
import struct
import subprocess
import math
import time
from typing import List, Tuple, Optional
import pygame

# Constants
MOJO_CMD = ["mojo", "run", "fluid_sim_core_3d_interactive.mojo"]
WIDTH, HEIGHT = 1200, 800

# Protocol IDs
CMD_STEP = 1
CMD_RESIZE = 2
CMD_QUIT = 3

Vec3 = Tuple[float, float, float]

class MojoSimulator:
    def __init__(self):
        self.proc = subprocess.Popen(
            MOJO_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr
        )
        self.frame_idx = 0
        self.particle_count = 0
        self.collision_events = 0
        self.bounds = [4.0, 4.0, 4.0]
        self.scenario_active = 0
        self.whirlpool_radius = 3.0
        self.positions: List[Vec3] = []
        self.velocities: List[Vec3] = []

    def step(self):
        if self.proc.poll() is not None:
            return False

        # Send STEP command
        try:
            self.proc.stdin.write(struct.pack("<I", CMD_STEP))
            self.proc.stdin.flush()
        except BrokenPipeError:
            return False

        # Read Header (64 bytes)
        # qqqdddqd
        header_size = 64
        header_data = self.proc.stdout.read(header_size)
        if len(header_data) < header_size:
            return False

        (
            self.frame_idx,
            self.particle_count,
            self.collision_events,
            self.bounds[0],
            self.bounds[1],
            self.bounds[2],
            self.scenario_active,
            self.whirlpool_radius
        ) = struct.unpack("<qqqdddqd", header_data)

        # Read Positions
        # particle_count * 4 doubles (x, y, z, pad)
        doubles_per_vec = 4
        bytes_per_double = 8
        block_size = self.particle_count * doubles_per_vec * bytes_per_double
        
        pos_data = self.proc.stdout.read(block_size)
        if len(pos_data) < block_size:
            return False
            
        vel_data = self.proc.stdout.read(block_size)
        if len(vel_data) < block_size:
            return False

        # Parse data
        # We can use struct.iter_unpack if available (Python 3.4+) or just unpack in a loop/chunk
        # For speed, let's unpack all at once
        fmt = f"<{self.particle_count * 4}d"
        raw_pos = struct.unpack(fmt, pos_data)
        raw_vel = struct.unpack(fmt, vel_data)
        
        self.positions = []
        self.velocities = []
        
        for i in range(self.particle_count):
            base = i * 4
            self.positions.append((raw_pos[base], raw_pos[base+1], raw_pos[base+2]))
            self.velocities.append((raw_vel[base], raw_vel[base+1], raw_vel[base+2]))
            
        return True

    def resize(self, w: float, h: float, d: float):
        if self.proc.poll() is not None:
            return
        
        # Send RESIZE command
        # ID (4) + 3 doubles (24) = 28 bytes
        payload = struct.pack("<Iddd", CMD_RESIZE, w, h, d)
        try:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
            self.bounds = [w, h, d]
        except BrokenPipeError:
            pass

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(struct.pack("<I", CMD_QUIT))
                self.proc.stdin.flush()
            except:
                pass
            self.proc.terminate()
            self.proc.wait()


# Rendering Utils
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

    x_proj = (x_cam / z_cam) * fov_scale * camera_zoom / aspect
    y_proj = (y_cam / z_cam) * fov_scale * camera_zoom

    screen_x = int(width_px * 0.5 + x_proj * width_px * 0.5)
    screen_y = int(height_px * 0.5 - y_proj * height_px * 0.5)

    return screen_x, screen_y, z_cam


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mojo 3D FluidSim Interactive")
    clock = pygame.time.Clock()

    sim = MojoSimulator()
    
    # Camera
    camera_distance = 10.0
    camera_angle_x = 0.5
    camera_angle_y = 0.3
    camera_zoom = 1.0
    mouse_dragging = False
    last_mouse_pos = (0, 0)

    running = True
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Resize controls
                elif event.key == pygame.K_LEFTBRACKET:
                    # Shrink bounds
                    new_w = max(2.0, sim.bounds[0] - 0.5)
                    sim.resize(new_w, sim.bounds[1], sim.bounds[2])
                elif event.key == pygame.K_RIGHTBRACKET:
                    # Grow bounds
                    new_w = min(10.0, sim.bounds[0] + 0.5)
                    sim.resize(new_w, sim.bounds[1], sim.bounds[2])
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_dragging = True
                    last_mouse_pos = event.pos
                elif event.button == 4: # Scroll Up
                    camera_zoom = min(3.0, camera_zoom * 1.1)
                elif event.button == 5: # Scroll Down
                    camera_zoom = max(0.3, camera_zoom / 1.1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    camera_angle_x += dx * 0.005
                    camera_angle_y = max(-1.5, min(1.5, camera_angle_y - dy * 0.005))
                    last_mouse_pos = event.pos

        # Simulation Step
        if not sim.step():
            print("Simulation ended or failed to step.")
            running = False
            continue

        # Rendering
        screen.fill((10, 10, 30))

        # Draw Bounds
        half_w = sim.bounds[0] * 0.5
        half_h = sim.bounds[1] * 0.5
        half_d = sim.bounds[2] * 0.5
        
        corners = [
            (-half_w, -half_h, -half_d), (half_w, -half_h, -half_d),
            (half_w, half_h, -half_d), (-half_w, half_h, -half_d),
            (-half_w, -half_h, half_d), (half_w, -half_h, half_d),
            (half_w, half_h, half_d), (-half_w, half_h, half_d),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        
        for i, j in edges:
            p1 = project_3d_to_2d(corners[i], camera_distance, camera_angle_x, camera_angle_y, camera_zoom, WIDTH, HEIGHT)
            p2 = project_3d_to_2d(corners[j], camera_distance, camera_angle_x, camera_angle_y, camera_zoom, WIDTH, HEIGHT)
            if p1[2] > 0 and p2[2] > 0:
                pygame.draw.line(screen, (80, 80, 80), (p1[0], p1[1]), (p2[0], p2[1]), 1)

        # Draw Particles
        particles_to_draw = []
        for i in range(len(sim.positions)):
            pos = sim.positions[i]
            vel = sim.velocities[i]
            speed = v3_length(vel)
            
            # Color based on speed
            t = min(speed / 6.0, 1.0)
            color = (
                int(30 * (1-t) + 255 * t),
                int(60 * (1-t) + 255 * t),
                int(255 * (1-t) + 150 * t)
            )
            
            sx, sy, depth = project_3d_to_2d(pos, camera_distance, camera_angle_x, camera_angle_y, camera_zoom, WIDTH, HEIGHT)
            if depth > 0 and 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
                particles_to_draw.append((sx, sy, depth, color))
                
        particles_to_draw.sort(key=lambda p: p[2], reverse=True)
        
        for sx, sy, depth, color in particles_to_draw:
            pygame.draw.circle(screen, color, (sx, sy), 3)

        # UI
        font = pygame.font.SysFont("consolas", 14)
        info = [
            f"FPS: {clock.get_fps():.1f}",
            f"Particles: {sim.particle_count}",
            f"Bounds: {sim.bounds[0]:.1f} x {sim.bounds[1]:.1f} x {sim.bounds[2]:.1f}",
            "Controls: Mouse Drag=Rotate, Scroll=Zoom, [ ]=Resize Width"
        ]
        
        for i, line in enumerate(info):
            surf = font.render(line, True, (200, 200, 200))
            screen.blit(surf, (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    sim.close()
    pygame.quit()

if __name__ == "__main__":
    main()
