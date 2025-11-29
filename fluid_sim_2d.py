"""
2D particle-based fluid simulator inspired by the Unity GPU implementation in
`Fluid-Sim-Episode-01/Assets/Scripts/Sim 2D`.

This is written as a single, large Python file so it can later be ported to Mojo
by copying over the simulation functions and data structures.

Original Unity/C# + compute shader functionality covered:

- Simulation settings:
  - time scale
  - fixed timestep vs frame-based timestep
  - iterations-per-frame substeps
  - gravity
  - collision damping
  - smoothing radius
  - target density
  - pressure multiplier & near-pressure multiplier
  - viscosity strength
  - world bounds
  - rectangular obstacle (size + centre)
- Interaction settings:
  - mouse interaction radius
  - attraction (LMB) / repulsion (RMB) forces
- Particle spawning:
  - rectangular spawn region
  - configurable particle count
  - configurable initial velocity
  - random jitter per particle
- Fluid dynamics (ported from `FluidSim2D.compute` + `FluidMaths2D.hlsl`):
  - external forces (gravity + mouse interaction)
  - spatial hashing (2D grid, 9-cell neighborhood search)
  - density & near-density kernels (spiky kernels)
  - pressure from density
  - pressure forces between neighbours
  - viscosity forces between neighbours
  - position integration and collision with:
      - outer bounds
      - inner rectangular obstacle

Additional Python/pygame functionality:

- Pygame 2D visualization:
  - circles for particles
  - simple color mapping by speed
- Keyboard controls:
  - Space: pause / resume
  - Right arrow: single-step one simulation frame while paused
  - R: reset to initial spawn state
  - Esc or window close: quit

Structure notes for Mojo-compatibility:

- All simulation state is stored in plain Python lists of tuples/floats
  rather than classes or NumPy arrays.
- Simulation logic is organized into small, pure functions that operate
  on primitive data structures in-place. These map cleanly to Mojo structs
  and for-loops later.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Tuple

import pygame

# ---------------------------------------------
# Type aliases (helps later Mojo translation)
# ---------------------------------------------

Vec2 = Tuple[float, float]


# ---------------------------------------------
# Central simulation configuration
# (Edit this block to tweak behaviour)
# ---------------------------------------------


class SimConfig:
    # Particle/bounds defaults
    particle_count: int = 2000
    bounds_size: Vec2 = (8.0, 8.0)  # bigger box by default

    # Screen/window size in pixels (can be non-square)
    screen_width: int = 1200
    screen_height: int = 800

    # Time integration
    time_scale: float = 1
    fixed_time_step: bool = True
    iterations_per_frame: int = 10

    # Physical behaviour
    gravity: float = -5.0
    collision_damping: float = 0.97
    smoothing_radius: float = 0.12
    target_density: float = 12.0
    pressure_multiplier: float = 120.0
    near_pressure_multiplier: float = 20.0
    viscosity_strength: float = 0.2

    # Spawn region
    initial_velocity: Vec2 = (0.0, 0.0)
    spawn_centre: Vec2 = (0.0, 1.0)
    spawn_size: Vec2 = (3.0, 1.2)
    jitter_strength: float = 0.02

    # Interaction
    interaction_radius: float = 0.6
    interaction_strength: float = 45.0

    # Obstacle
    obstacle_size: Vec2 = (2.0, 1.2)
    obstacle_centre: Vec2 = (0.0, -0.9)


# ---------------------------------------------
# Utility vector functions
# ---------------------------------------------

def v_add(a: Vec2, b: Vec2) -> Vec2:
    return a[0] + b[0], a[1] + b[1]


def v_sub(a: Vec2, b: Vec2) -> Vec2:
    return a[0] - b[0], a[1] - b[1]


def v_mul(a: Vec2, s: float) -> Vec2:
    return a[0] * s, a[1] * s


def v_length_sq(a: Vec2) -> float:
    return a[0] * a[0] + a[1] * a[1]


def v_length(a: Vec2) -> float:
    return math.sqrt(v_length_sq(a))


def v_normalize(a: Vec2) -> Vec2:
    length = v_length(a)
    if length <= 1e-8:
        return 0.0, 0.0
    inv = 1.0 / length
    return a[0] * inv, a[1] * inv


# ---------------------------------------------
# Kernel functions (FluidMaths2D.hlsl)
# ---------------------------------------------

def smoothing_kernel_poly6(dst: float, radius: float, poly6_scaling: float) -> float:
    """SmoothingKernelPoly6 from FluidMaths2D.hlsl."""
    if dst < radius:
        v = radius * radius - dst * dst
        return v * v * v * poly6_scaling
    return 0.0


def spiky_kernel_pow3(dst: float, radius: float, scale: float) -> float:
    """SpikyKernelPow3."""
    if dst < radius:
        v = radius - dst
        return v * v * v * scale
    return 0.0


def spiky_kernel_pow2(dst: float, radius: float, scale: float) -> float:
    """SpikyKernelPow2."""
    if dst < radius:
        v = radius - dst
        return v * v * scale
    return 0.0


def derivative_spiky_pow3(dst: float, radius: float, scale: float) -> float:
    """DerivativeSpikyPow3."""
    if dst <= radius:
        v = radius - dst
        return -v * v * scale
    return 0.0


def derivative_spiky_pow2(dst: float, radius: float, scale: float) -> float:
    """DerivativeSpikyPow2."""
    if dst <= radius:
        v = radius - dst
        return -v * scale
    return 0.0


# ---------------------------------------------
# Spatial hashing (SpatialHash.hlsl)
# ---------------------------------------------

OFFSETS_2D: List[Tuple[int, int]] = [
    (-1, 1),
    (0, 1),
    (1, 1),
    (-1, 0),
    (0, 0),
    (1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]

HASH_K1 = 15823
HASH_K2 = 9737333


def get_cell_2d(position: Vec2, radius: float) -> Tuple[int, int]:
    return int(math.floor(position[0] / radius)), int(math.floor(position[1] / radius))


def hash_cell_2d(cell: Tuple[int, int]) -> int:
    # Cast to "unsigned" by masking to 32 bits (Python ints are unbounded)
    cx = cell[0] & 0xFFFFFFFF
    cy = cell[1] & 0xFFFFFFFF
    a = (cx * HASH_K1) & 0xFFFFFFFF
    b = (cy * HASH_K2) & 0xFFFFFFFF
    return (a + b) & 0xFFFFFFFF


# ---------------------------------------------
# Simulation state and parameters
# ---------------------------------------------

class FluidSim2D:
    """
    CPU implementation of the Unity 2D fluid sim, adapted for pygame.

    The layout of this class is intentionally simple so that its fields and
    methods can be ported to Mojo with minimal structural changes.
    """

    def __init__(
        self,
        particle_count: int = SimConfig.particle_count,
        bounds_size: Vec2 = SimConfig.bounds_size,
    ) -> None:
        # --- Simulation settings (mirrors Simulation2D.cs) ---
        self.time_scale: float = SimConfig.time_scale
        self.fixed_time_step: bool = SimConfig.fixed_time_step
        self.iterations_per_frame: int = SimConfig.iterations_per_frame
        self.gravity: float = SimConfig.gravity
        self.collision_damping: float = SimConfig.collision_damping
        self.smoothing_radius: float = SimConfig.smoothing_radius
        self.target_density: float = SimConfig.target_density
        self.pressure_multiplier: float = SimConfig.pressure_multiplier
        self.near_pressure_multiplier: float = SimConfig.near_pressure_multiplier
        self.viscosity_strength: float = SimConfig.viscosity_strength

        # Bounds and obstacle are in "world units". We'll map those to pixels.
        self.bounds_size: Vec2 = bounds_size
        self.obstacle_size: Vec2 = SimConfig.obstacle_size
        self.obstacle_centre: Vec2 = SimConfig.obstacle_centre

        # Interaction settings
        self.interaction_radius: float = SimConfig.interaction_radius
        self.interaction_strength: float = SimConfig.interaction_strength

        # Kernel scaling factors (same as in Simulation2D.UpdateSettings)
        r = self.smoothing_radius
        self.poly6_scaling_factor: float = 4.0 / (math.pi * (r ** 8))
        self.spiky_pow3_scaling_factor: float = 10.0 / (math.pi * (r ** 5))
        self.spiky_pow2_scaling_factor: float = 6.0 / (math.pi * (r ** 4))
        self.spiky_pow3_derivative_scaling_factor: float = 30.0 / ((r ** 5) * math.pi)
        self.spiky_pow2_derivative_scaling_factor: float = 12.0 / ((r ** 4) * math.pi)

        # Spawn settings (mirrors ParticleSpawner.cs conceptually)
        self.particle_count: int = particle_count
        self.initial_velocity: Vec2 = SimConfig.initial_velocity
        self.spawn_centre: Vec2 = SimConfig.spawn_centre
        self.spawn_size: Vec2 = SimConfig.spawn_size
        self.jitter_strength: float = SimConfig.jitter_strength

        # Internal state arrays (parallel arrays for easy Mojo translation)
        self.positions: List[Vec2] = []
        self.predicted_positions: List[Vec2] = []
        self.velocities: List[Vec2] = []
        # densities[i] = (density, near_density)
        self.densities: List[Tuple[float, float]] = []

        # Spatial hash: maps (cell_x, cell_y) -> list of particle indices
        self.cell_particles: Dict[Tuple[int, int], List[int]] = {}

        # Mouse interaction state (set externally by the pygame loop)
        self.mouse_world_pos: Vec2 = (0.0, 0.0)
        self.mouse_interaction_strength: float = 0.0

        # Collision statistics (updated each simulation frame)
        self.collision_events: int = 0

        # For reset
        self._initial_positions: List[Vec2] = []
        self._initial_velocities: List[Vec2] = []

        self._init_spawn()

    # -----------------------------
    # Spawn logic (ParticleSpawner)
    # -----------------------------

    def _init_spawn(self) -> None:
        """Compute initial particle positions/velocities like ParticleSpawner."""
        self.positions = [(0.0, 0.0)] * self.particle_count
        self.predicted_positions = [(0.0, 0.0)] * self.particle_count
        self.velocities = [self.initial_velocity] * self.particle_count
        self.densities = [(0.0, 0.0)] * self.particle_count

        sx, sy = self.spawn_size
        # Same math idea as in ParticleSpawner.GetSpawnData:
        # Determine numX/numY grid such that we can fit particle_count
        # (we'll use a simpler heuristic)
        aspect = sx / max(sy, 1e-6)
        approx_rows = int(math.sqrt(self.particle_count / max(aspect, 1e-3)))
        approx_rows = max(1, approx_rows)
        approx_cols = max(1, int(math.ceil(self.particle_count / approx_rows)))

        i = 0
        rng = random.Random(42)
        for y in range(approx_rows):
            for x in range(approx_cols):
                if i >= self.particle_count:
                    break
                tx = x / max(approx_cols - 1, 1)
                ty = y / max(approx_rows - 1, 1)
                # Position inside spawn rect, centre + jitter
                angle = rng.random() * math.tau
                dir_vec = (math.cos(angle), math.sin(angle))
                jitter_mag = (rng.random() - 0.5) * self.jitter_strength
                jitter = (dir_vec[0] * jitter_mag, dir_vec[1] * jitter_mag)

                px = (tx - 0.5) * sx + self.spawn_centre[0] + jitter[0]
                py = (ty - 0.5) * sy + self.spawn_centre[1] + jitter[1]
                self.positions[i] = (px, py)
                self.predicted_positions[i] = (px, py)
                self.velocities[i] = self.initial_velocity
                i += 1

        # Save reset copies
        self._initial_positions = list(self.positions)
        self._initial_velocities = list(self.velocities)

    def reset(self) -> None:
        """Reset to initial spawn state, like Simulation2D.HandleInput (R key)."""
        self.positions = list(self._initial_positions)
        self.predicted_positions = list(self._initial_positions)
        self.velocities = list(self._initial_velocities)
        self.densities = [(0.0, 0.0)] * self.particle_count

    # ----------------------------------------
    # External forces (gravity + interactions)
    # ----------------------------------------

    def _external_forces(self, pos: Vec2, vel: Vec2) -> Vec2:
        """
        Port of ExternalForces() in FluidSim2D.compute.
        Gravity + optional mouse attraction/repulsion.
        """
        gravity_accel = (0.0, self.gravity)

        s = self.mouse_interaction_strength
        if abs(s) > 1e-5:
            offset = v_sub(self.mouse_world_pos, pos)
            sqr_dst = v_length_sq(offset)
            r = self.interaction_radius
            if sqr_dst < r * r and sqr_dst > 1e-12:
                dst = math.sqrt(sqr_dst)
                edge_t = dst / r
                centre_t = 1.0 - edge_t
                dir_to_centre = v_mul(offset, 1.0 / dst)

                gravity_weight = 1.0 - (centre_t * max(min(s / 10.0, 1.0), -1.0))
                accel_gravity = v_mul(gravity_accel, gravity_weight)
                accel_interact = v_mul(dir_to_centre, centre_t * s)
                accel_damping = v_mul(vel, -centre_t)

                ax = accel_gravity[0] + accel_interact[0] + accel_damping[0]
                ay = accel_gravity[1] + accel_interact[1] + accel_damping[1]
                return ax, ay

        return gravity_accel

    # -------------------------
    # Collision handling
    # -------------------------

    def _handle_collisions(self, index: int) -> None:
        """Port of HandleCollisions() in FluidSim2D.compute."""
        pos = self.positions[index]
        vel = self.velocities[index]
        collided = False

        half_w = self.bounds_size[0] * 0.5
        half_h = self.bounds_size[1] * 0.5

        # Outer bounds
        if pos[0] <= -half_w or pos[0] >= half_w:
            pos = (max(min(pos[0], half_w), -half_w), pos[1])
            vel = (-vel[0] * self.collision_damping, vel[1])
            collided = True
        if pos[1] <= -half_h or pos[1] >= half_h:
            pos = (pos[0], max(min(pos[1], half_h), -half_h))
            vel = (vel[0], -vel[1] * self.collision_damping)
            collided = True

        # Obstacle: axis-aligned box
        ox, oy = self.obstacle_centre
        ohx = self.obstacle_size[0] * 0.5
        ohy = self.obstacle_size[1] * 0.5

        # Early exit: quick distance check to skip obstacle collision if particle is far away
        dx = abs(pos[0] - ox)
        dy = abs(pos[1] - oy)
        if dx <= ohx + 0.5 and dy <= ohy + 0.5:
            # Particle might be near obstacle, do detailed check
            rel_x = pos[0] - ox
            rel_y = pos[1] - oy
            edge_x = ohx - abs(rel_x)
            edge_y = ohy - abs(rel_y)

            if edge_x >= 0.0 and edge_y >= 0.0:
                if edge_x < edge_y:
                    # Hit vertical side
                    pos = (ohx * math.copysign(1.0, rel_x) + ox, pos[1])
                    vel = (-vel[0] * self.collision_damping, vel[1])
                else:
                    # Hit horizontal side
                    pos = (pos[0], ohy * math.copysign(1.0, rel_y) + oy)
                    vel = (vel[0], -vel[1] * self.collision_damping)
                collided = True

        self.positions[index] = pos
        self.velocities[index] = vel
        if collided:
            self.collision_events += 1

    # -------------------------
    # Spatial hash construction
    # -------------------------

    def _build_spatial_hash(self) -> None:
        """Fill cell_particles with indices based on predicted_positions."""
        self.cell_particles.clear()
        r = self.smoothing_radius
        for i, pos in enumerate(self.predicted_positions):
            cell = get_cell_2d(pos, r)
            bucket = self.cell_particles.get(cell)
            if bucket is None:
                bucket = []
                self.cell_particles[cell] = bucket
            bucket.append(i)

    # -------------------------
    # Density and near-density
    # -------------------------

    def _calculate_density_for_particle(self, index: int) -> Tuple[float, float]:
        pos = self.predicted_positions[index]
        r = self.smoothing_radius
        sqr_radius = r * r

        density = 0.0
        near_density = 0.0

        origin_cell = get_cell_2d(pos, r)

        for ox, oy in OFFSETS_2D:
            neighbour_cell = (origin_cell[0] + ox, origin_cell[1] + oy)
            indices = self.cell_particles.get(neighbour_cell)
            if not indices:
                continue
            for j in indices:
                neighbour_pos = self.predicted_positions[j]
                offset = v_sub(neighbour_pos, pos)
                sqr_dst = v_length_sq(offset)
                if sqr_dst > sqr_radius:
                    continue
                dst = math.sqrt(sqr_dst)
                density += spiky_kernel_pow2(dst, r, self.spiky_pow2_scaling_factor)
                near_density += spiky_kernel_pow3(dst, r, self.spiky_pow3_scaling_factor)

        return density, near_density

    def _update_densities(self) -> None:
        for i in range(self.particle_count):
            self.densities[i] = self._calculate_density_for_particle(i)

    # -------------------------
    # Pressure forces
    # -------------------------

    def _pressure_from_density(self, density: float) -> float:
        return (density - self.target_density) * self.pressure_multiplier

    def _near_pressure_from_density(self, near_density: float) -> float:
        return self.near_pressure_multiplier * near_density

    def _apply_pressure_forces(self, dt: float) -> None:
        r = self.smoothing_radius
        sqr_radius = r * r

        accel: List[Vec2] = [(0.0, 0.0)] * self.particle_count

        for i in range(self.particle_count):
            pos = self.predicted_positions[i]
            origin_cell = get_cell_2d(pos, r)

            density, near_density = self.densities[i]
            if density <= 1e-8:
                continue

            pressure = self._pressure_from_density(density)
            near_pressure_val = self._near_pressure_from_density(near_density)

            total_force = (0.0, 0.0)

            for ox, oy in OFFSETS_2D:
                neighbour_cell = (origin_cell[0] + ox, origin_cell[1] + oy)
                indices = self.cell_particles.get(neighbour_cell)
                if not indices:
                    continue
                for j in indices:
                    if j == i:
                        continue
                    neighbour_pos = self.predicted_positions[j]
                    offset = v_sub(neighbour_pos, pos)
                    sqr_dst = v_length_sq(offset)
                    if sqr_dst > sqr_radius:
                        continue
                    dst = math.sqrt(sqr_dst)
                    if dst <= 1e-8:
                        dir_to_neighbour = (0.0, 1.0)
                    else:
                        dir_to_neighbour = v_mul(offset, 1.0 / dst)

                    nd, n_near = self.densities[j]
                    if nd <= 1e-8 or n_near <= 0.0:
                        continue

                    neighbour_pressure = self._pressure_from_density(nd)
                    neighbour_near_pressure = self._near_pressure_from_density(n_near)

                    shared_pressure = 0.5 * (pressure + neighbour_pressure)
                    shared_near = 0.5 * (near_pressure_val + neighbour_near_pressure)

                    d_kernel = derivative_spiky_pow2(
                        dst, r, self.spiky_pow2_derivative_scaling_factor
                    )
                    nd_kernel = derivative_spiky_pow3(
                        dst, r, self.spiky_pow3_derivative_scaling_factor
                    )

                    # pressureForce += dir * DensityDerivative * sharedPressure / neighbourDensity
                    # pressureForce += dir * NearDensityDerivative * sharedNearPressure / neighbourNearDensity
                    scale1 = (d_kernel * shared_pressure) / max(nd, 1e-8)
                    scale2 = (nd_kernel * shared_near) / max(n_near, 1e-8)
                    force_x = dir_to_neighbour[0] * (scale1 + scale2)
                    force_y = dir_to_neighbour[1] * (scale1 + scale2)

                    total_force = (total_force[0] + force_x, total_force[1] + force_y)

            # acceleration = pressureForce / density
            ax = total_force[0] / density
            ay = total_force[1] / density
            accel[i] = (ax, ay)

        # Apply accelerations to velocities
        for i in range(self.particle_count):
            vx, vy = self.velocities[i]
            ax, ay = accel[i]
            self.velocities[i] = (vx + ax * dt, vy + ay * dt)

    # -------------------------
    # Viscosity forces
    # -------------------------

    def _apply_viscosity(self, dt: float) -> None:
        r = self.smoothing_radius
        sqr_radius = r * r
        viscosity_strength = self.viscosity_strength

        delta_v: List[Vec2] = [(0.0, 0.0)] * self.particle_count

        for i in range(self.particle_count):
            pos = self.predicted_positions[i]
            origin_cell = get_cell_2d(pos, r)
            vi = self.velocities[i]
            visc_force = (0.0, 0.0)

            for ox, oy in OFFSETS_2D:
                neighbour_cell = (origin_cell[0] + ox, origin_cell[1] + oy)
                indices = self.cell_particles.get(neighbour_cell)
                if not indices:
                    continue

                for j in indices:
                    if j == i:
                        continue
                    neighbour_pos = self.predicted_positions[j]
                    offset = v_sub(neighbour_pos, pos)
                    sqr_dst = v_length_sq(offset)
                    if sqr_dst > sqr_radius:
                        continue
                    dst = math.sqrt(sqr_dst)
                    vj = self.velocities[j]
                    dv = (vj[0] - vi[0], vj[1] - vi[1])
                    w = smoothing_kernel_poly6(dst, r, self.poly6_scaling_factor)
                    visc_force = (visc_force[0] + dv[0] * w, visc_force[1] + dv[1] * w)

            delta_v[i] = (
                visc_force[0] * viscosity_strength * dt,
                visc_force[1] * viscosity_strength * dt,
            )

        for i in range(self.particle_count):
            vx, vy = self.velocities[i]
            dvx, dvy = delta_v[i]
            self.velocities[i] = (vx + dvx, vy + dvy)

    # -------------------------
    # Simulation step
    # -------------------------

    def step(self, frame_time: float) -> None:
        """Run one simulation frame, with substeps."""
        if self.iterations_per_frame <= 0:
            return
        # Reset collision counter for this frame
        self.collision_events = 0
        dt = (frame_time / self.iterations_per_frame) * self.time_scale
        # match shader predictionFactor (1/120) by using a separate factor
        prediction_factor = 1.0 / 120.0

        for _ in range(self.iterations_per_frame):
            # External forces and prediction
            for i in range(self.particle_count):
                pos = self.positions[i]
                vel = self.velocities[i]
                accel = self._external_forces(pos, vel)
                vel = (vel[0] + accel[0] * dt, vel[1] + accel[1] * dt)
                self.velocities[i] = vel
                self.predicted_positions[i] = (
                    pos[0] + vel[0] * prediction_factor,
                    pos[1] + vel[1] * prediction_factor,
                )

            # Spatial hash
            self._build_spatial_hash()

            # Densities
            self._update_densities()

            # Pressure
            self._apply_pressure_forces(dt)

            # Viscosity
            self._apply_viscosity(dt)

            # Update positions and collisions
            for i in range(self.particle_count):
                pos = self.positions[i]
                vel = self.velocities[i]
                self.positions[i] = (pos[0] + vel[0] * dt, pos[1] + vel[1] * dt)
                self._handle_collisions(i)


# ---------------------------------------------
# Pygame visualization and input handling
# ---------------------------------------------

class FluidSimApp:
    """Pygame app wrapper around FluidSim2D."""

    def __init__(
        self,
        particle_count: int = SimConfig.particle_count,
        bounds_size: Vec2 = SimConfig.bounds_size,
        screen_size: Tuple[int, int] = (SimConfig.screen_width, SimConfig.screen_height),
        zoom: float = 1.0,
    ) -> None:
        pygame.init()
        self.clock = pygame.time.Clock()

        # Window
        self.pixel_width = screen_size[0]
        self.pixel_height = screen_size[1]
        self.screen = pygame.display.set_mode((self.pixel_width, self.pixel_height))
        pygame.display.set_caption("2D Fluid Sim (CPU, pygame)")

        # Simulation
        self.sim = FluidSim2D(particle_count=particle_count, bounds_size=bounds_size)

        # Map world to screen: centre bounds at window centre
        self.world_width = self.sim.bounds_size[0]
        self.world_height = self.sim.bounds_size[1]

        # Zoom factor (1.0 = show full bounds; >1 = zoom in; <1 = zoom out)
        self.zoom_factor = max(0.1, zoom)
        self._update_pixels_per_unit()

        self.running = True
        self.paused = False
        self.step_once = False

        # Precompute particle radius in pixels
        self._update_particle_radius()

        # Graph display state
        self.show_graphs = False
        self.fps_history: List[float] = []
        self.collision_history: List[float] = []
        self.max_history_points = 240  # about 4 seconds at 60 fps

    def _update_pixels_per_unit(self) -> None:
        # Effective world size shrinks when zooming in (zoom_factor > 1)
        effective_w = self.world_width / self.zoom_factor
        effective_h = self.world_height / self.zoom_factor
        self.pixels_per_unit = min(
            self.pixel_width / effective_w, self.pixel_height / effective_h
        )

    def _update_particle_radius(self) -> None:
        self.particle_radius_px = max(
            2, int(self.sim.smoothing_radius * self.pixels_per_unit * 0.35)
        )

    # ---------------
    # World <-> screen
    # ---------------

    def world_to_screen(self, p: Vec2) -> Tuple[int, int]:
        effective_w = self.world_width / self.zoom_factor
        effective_h = self.world_height / self.zoom_factor
        x = (p[0] / effective_w + 0.5) * self.pixel_width
        # Invert y because screen y grows downward
        y = (0.5 - p[1] / effective_h) * self.pixel_height
        return int(x), int(y)

    def screen_to_world(self, x: int, y: int) -> Vec2:
        effective_w = self.world_width / self.zoom_factor
        effective_h = self.world_height / self.zoom_factor
        wx = (x / self.pixel_width - 0.5) * effective_w
        wy = (0.5 - y / self.pixel_height) * effective_h
        return wx, wy

    # ---------------
    # Input handling
    # ---------------

    def _handle_events(self) -> None:
        self.sim.mouse_interaction_strength = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT:
                    # Step one frame while paused
                    self.step_once = True
                elif event.key == pygame.K_r:
                    self.sim.reset()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    # Zoom in
                    self.zoom_factor = min(self.zoom_factor * 1.1, 10.0)
                    self._update_pixels_per_unit()
                    self._update_particle_radius()
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    # Zoom out
                    self.zoom_factor = max(self.zoom_factor / 1.1, 0.2)
                    self._update_pixels_per_unit()
                    self._update_particle_radius()
                elif event.key == pygame.K_g:
                    # Toggle FPS/collision graphs
                    self.show_graphs = not self.show_graphs

        # Mouse interaction (like Simulation2D.UpdateSettings)
        mouse_buttons = pygame.mouse.get_pressed(3)
        mx, my = pygame.mouse.get_pos()
        self.sim.mouse_world_pos = self.screen_to_world(mx, my)

        is_pull = mouse_buttons[0]
        is_push = mouse_buttons[2]
        if is_pull or is_push:
            strength = self.sim.interaction_strength
            if is_push:
                strength = -strength
            self.sim.mouse_interaction_strength = strength

    # ---------------
    # Rendering
    # ---------------

    def _draw(self) -> None:
        self.screen.fill((10, 10, 30))

        # Draw outer bounds
        half_w = self.sim.bounds_size[0] * 0.5
        half_h = self.sim.bounds_size[1] * 0.5
        top_left = self.world_to_screen((-half_w, half_h))
        bottom_right = self.world_to_screen((half_w, -half_h))
        rect = pygame.Rect(
            top_left[0],
            top_left[1],
            bottom_right[0] - top_left[0],
            bottom_right[1] - top_left[1],
        )
        pygame.draw.rect(self.screen, (80, 80, 80), rect, 1)

        # Draw obstacle
        ox, oy = self.sim.obstacle_centre
        ohx = self.sim.obstacle_size[0] * 0.5
        ohy = self.sim.obstacle_size[1] * 0.5
        obs_tl = self.world_to_screen((ox - ohx, oy + ohy))
        obs_br = self.world_to_screen((ox + ohx, oy - ohy))
        obs_rect = pygame.Rect(
            obs_tl[0],
            obs_tl[1],
            obs_br[0] - obs_tl[0],
            obs_br[1] - obs_tl[1],
        )
        pygame.draw.rect(self.screen, (120, 120, 120), obs_rect, 1)

        # Draw particles (color by velocity magnitude, like gradient in original)
        max_speed = 6.0
        for i in range(self.sim.particle_count):
            pos = self.sim.positions[i]
            vel = self.sim.velocities[i]
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
            sx, sy = self.world_to_screen(pos)
            pygame.draw.circle(self.screen, color, (sx, sy), self.particle_radius_px)

        # Interaction radius visualization
        if abs(self.sim.mouse_interaction_strength) > 0.0:
            mx, my = self.sim.mouse_world_pos
            sx, sy = self.world_to_screen((mx, my))
            r_px = int(self.sim.interaction_radius * self.pixels_per_unit)
            col = (0, 255, 0) if self.sim.mouse_interaction_strength > 0 else (255, 0, 0)
            pygame.draw.circle(self.screen, col, (sx, sy), r_px, 1)

        # Overlay graphs (FPS and collisions/sec) at top-right
        if self.show_graphs:
            self._draw_graphs()

        pygame.display.flip()

    def _draw_graphs(self) -> None:
        """Draw FPS and collisions-per-second graphs at top-right with grids and axis labels."""
        # Layout
        margin = 10
        graph_width = 260
        graph_height = 80
        right = self.pixel_width - margin
        top = margin

        # Shared settings
        bg_color = (5, 5, 20)
        border_color = (180, 180, 180)
        grid_color = (60, 60, 90)
        fps_color = (80, 220, 80)
        col_color = (220, 180, 80)

        font = pygame.font.SysFont("consolas", 12)

        def draw_single_graph(
            data: List[float],
            rect: pygame.Rect,
            color: Tuple[int, int, int],
            y_label: str,
            x_label: str,
        ) -> None:
            pygame.draw.rect(self.screen, bg_color, rect)
            pygame.draw.rect(self.screen, border_color, rect, 1)

            # Grid: 4 vertical + 3 horizontal lines
            step_x = rect.width // 4
            step_y = rect.height // 3
            for i in range(1, 4):
                x = rect.left + i * step_x
                pygame.draw.line(self.screen, grid_color, (x, rect.top), (x, rect.bottom))
            for i in range(1, 3):
                y = rect.top + i * step_y
                pygame.draw.line(self.screen, grid_color, (rect.left, y), (rect.right, y))

            max_val = 1.0
            n = len(data)
            if data:
                max_val = max(max(data), 1e-3)
            # Leave some headroom
            max_val *= 1.1

            for i in range(1, n):
                x0 = rect.right - rect.width + int(rect.width * (i - 1) / max(n - 1, 1))
                x1 = rect.right - rect.width + int(rect.width * i / max(n - 1, 1))
                y0 = rect.bottom - int(rect.height * (data[i - 1] / max_val))
                y1 = rect.bottom - int(rect.height * (data[i] / max_val))
                pygame.draw.line(self.screen, color, (x0, y0), (x1, y1), 2)

            # Axis numeric ticks and labels
            text_color = (230, 230, 230)
            # Y-axis numeric labels (min/mid/max)
            for iy, frac in enumerate([0.0, 0.5, 1.0]):
                value = max_val * frac
                y = rect.bottom - int(rect.height * frac)
                label = f"{value:.0f}"
                surf = font.render(label, True, text_color)
                self.screen.blit(surf, (rect.left - surf.get_width() - 4, y - surf.get_height() // 2))

            # X-axis numeric labels: show 0, mid, max frame index
            if n > 0:
                x_indices = [0, n // 2, n - 1]
                for idx in x_indices:
                    if idx < 0 or idx >= n:
                        continue
                    frac_x = idx / max(n - 1, 1)
                    x = rect.left + int(rect.width * frac_x)
                    label = f"{idx}"
                    surf = font.render(label, True, text_color)
                    self.screen.blit(surf, (x - surf.get_width() // 2, rect.bottom + 2))

            # Axis labels (text)
            y_label_surf = font.render(y_label, True, text_color)
            self.screen.blit(
                y_label_surf, (rect.left - y_label_surf.get_width() - 4, rect.top - 2)
            )
            x_label_surf = font.render(x_label, True, text_color)
            self.screen.blit(
                x_label_surf,
                (rect.right - x_label_surf.get_width(), rect.bottom + 16),
            )

        # FPS graph (top)
        fps_rect = pygame.Rect(
            right - graph_width,
            top,
            graph_width,
            graph_height,
        )
        draw_single_graph(
            self.fps_history,
            fps_rect,
            fps_color,
            "FPS",
            "time (frames)",
        )

        # Collisions/s graph (below)
        col_rect = pygame.Rect(
            right - graph_width,
            top + graph_height + 24,  # leave space for labels
            graph_width,
            graph_height,
        )
        draw_single_graph(
            self.collision_history,
            col_rect,
            col_color,
            "coll/s",
            "time (frames)",
        )

    # ---------------
    # Main loop
    # ---------------

    def run(self) -> None:
        target_fps = 60
        while self.running:
            frame_time = self.clock.tick(target_fps) / 1000.0
            self._handle_events()

            # Fixed or variable timestep behaviour
            if not self.paused or self.step_once:
                self.sim.step(1.0 / target_fps if self.sim.fixed_time_step else frame_time)
                self.step_once = False

                # Update stats histories
                fps = self.clock.get_fps()
                if fps > 0:
                    self.fps_history.append(fps)
                collisions_per_sec = 0.0
                if frame_time > 0:
                    collisions_per_sec = self.sim.collision_events / frame_time
                self.collision_history.append(collisions_per_sec)

                # Trim history
                if len(self.fps_history) > self.max_history_points:
                    self.fps_history = self.fps_history[-self.max_history_points :]
                if len(self.collision_history) > self.max_history_points:
                    self.collision_history = self.collision_history[-self.max_history_points :]

            self._draw()

        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="2D Fluid Simulator (CPU, pygame)")
    parser.add_argument(
        "--particles",
        type=int,
        default=1000,
        help="Total number of particles (balls) in the simulation.",
    )
    parser.add_argument(
        "--box-width",
        type=float,
        default=6.0,
        help="Width of the simulation bounds in world units.",
    )
    parser.add_argument(
        "--box-height",
        type=float,
        default=6.0,
        help="Height of the simulation bounds in world units.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Initial zoom factor (1.0 = show full box, >1 = zoom in, <1 = zoom out).",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=SimConfig.screen_width,
        help="Window width in pixels.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=SimConfig.screen_height,
        help="Window height in pixels.",
    )
    args = parser.parse_args()

    bounds = (args.box_width, args.box_height)
    screen_size = (args.screen_width, args.screen_height)
    app = FluidSimApp(
        particle_count=args.particles,
        bounds_size=bounds,
        screen_size=screen_size,
        zoom=args.zoom,
    )
    app.run()


if __name__ == "__main__":
    main()


