"""
3D particle-based fluid simulator with scenarios support.

This extends fluid_sim_3d.py with scenario-based simulations.
Currently includes:
- Whirlpool scenario: Normal simulation for first 300 seconds, then whirlpool forces activate

All functionality from fluid_sim_3d.py is preserved.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from typing import Dict, List, Tuple, Optional

import pygame

# Try to import imageio and numpy for video saving, fall back to warning if not available
try:
    import imageio
    import numpy as np
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Import all the core simulation code from the base implementation
# (In a real scenario, you might want to refactor into a shared module)
# For now, we'll include everything here

# ---------------------------------------------
# Type aliases (helps later Mojo translation)
# ---------------------------------------------

Vec3 = Tuple[float, float, float]


# ---------------------------------------------
# Central simulation configuration
# ---------------------------------------------


class SimConfig:
    # Particle/bounds defaults
    particles_per_axis: int = 12  # Creates particles_per_axis^3 particles
    bounds_size: Vec3 = (4.0, 4.0, 4.0)  # 3D box

    # Screen/window size in pixels (can be non-square)
    screen_width: int = 1200
    screen_height: int = 800

    # Time integration
    time_scale: float = 1
    fixed_time_step: bool = True
    iterations_per_frame: int = 10

    # Physical behaviour
    gravity: float = -10.0
    collision_damping: float = 0.05
    smoothing_radius: float = 0.2
    target_density: float = 12.0
    pressure_multiplier: float = 120.0
    near_pressure_multiplier: float = 20.0
    viscosity_strength: float = 0.2

    # Spawn region
    initial_velocity: Vec3 = (0.0, 0.0, 0.0)
    spawn_centre: Vec3 = (0.0, 1.0, 0.0)
    spawn_size: float = 1.5  # Cubic spawn region
    jitter_strength: float = 0.02

    # Scenario settings
    scenario_type: str = "whirlpool"
    scenario_activation_time: float = 5.0  # Seconds before scenario activates


# ---------------------------------------------
# Utility vector functions (3D)
# ---------------------------------------------

def v3_add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def v3_sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def v3_mul(a: Vec3, s: float) -> Vec3:
    return a[0] * s, a[1] * s, a[2] * s


def v3_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v3_length_sq(a: Vec3) -> float:
    return v3_dot(a, a)


def v3_length(a: Vec3) -> float:
    return math.sqrt(v3_length_sq(a))


def v3_normalize(a: Vec3) -> Vec3:
    length = v3_length(a)
    if length <= 1e-8:
        return 0.0, 0.0, 1.0  # Default to (0,0,1) if zero
    inv = 1.0 / length
    return a[0] * inv, a[1] * inv, a[2] * inv


def v3_cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# ---------------------------------------------
# Kernel functions (FluidMaths3D.hlsl)
# ---------------------------------------------

PI = math.pi


def smoothing_kernel_poly6_3d(dst: float, radius: float) -> float:
    """SmoothingKernelPoly6 for 3D from FluidMaths3D.hlsl."""
    if dst < radius:
        scale = 315.0 / (64.0 * PI * (radius ** 9))
        v = radius * radius - dst * dst
        return v * v * v * scale
    return 0.0


def spiky_kernel_pow3_3d(dst: float, radius: float) -> float:
    """SpikyKernelPow3 for 3D."""
    if dst < radius:
        scale = 15.0 / (PI * (radius ** 6))
        v = radius - dst
        return v * v * v * scale
    return 0.0


def spiky_kernel_pow2_3d(dst: float, radius: float) -> float:
    """SpikyKernelPow2 for 3D."""
    if dst < radius:
        scale = 15.0 / (2.0 * PI * (radius ** 5))
        v = radius - dst
        return v * v * scale
    return 0.0


def derivative_spiky_pow3_3d(dst: float, radius: float) -> float:
    """DerivativeSpikyPow3 for 3D."""
    if dst <= radius:
        scale = 45.0 / ((radius ** 6) * PI)
        v = radius - dst
        return -v * v * scale
    return 0.0


def derivative_spiky_pow2_3d(dst: float, radius: float) -> float:
    """DerivativeSpikyPow2 for 3D."""
    if dst <= radius:
        scale = 15.0 / ((radius ** 5) * PI)
        v = radius - dst
        return -v * scale
    return 0.0


# ---------------------------------------------
# Spatial hashing (SpatialHash3D.hlsl)
# ---------------------------------------------

# 27 offsets for 3D (3x3x3 neighborhood)
OFFSETS_3D: List[Tuple[int, int, int]] = [
    (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
    (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
    (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
    (0, -1, -1), (0, -1, 0), (0, -1, 1),
    (0, 0, -1), (0, 0, 0), (0, 0, 1),
    (0, 1, -1), (0, 1, 0), (0, 1, 1),
    (1, -1, -1), (1, -1, 0), (1, -1, 1),
    (1, 0, -1), (1, 0, 0), (1, 0, 1),
    (1, 1, -1), (1, 1, 0), (1, 1, 1),
]

HASH_K1 = 15823
HASH_K2 = 9737333
HASH_K3 = 440817757


def get_cell_3d(position: Vec3, radius: float) -> Tuple[int, int, int]:
    """GetCell3D from SpatialHash3D.hlsl."""
    return (
        int(math.floor(position[0] / radius)),
        int(math.floor(position[1] / radius)),
        int(math.floor(position[2] / radius)),
    )


def hash_cell_3d(cell: Tuple[int, int, int]) -> int:
    """HashCell3D from SpatialHash3D.hlsl."""
    # Cast to "unsigned" by masking to 32 bits
    cx = cell[0] & 0xFFFFFFFF
    cy = cell[1] & 0xFFFFFFFF
    cz = cell[2] & 0xFFFFFFFF
    return ((cx * HASH_K1) + (cy * HASH_K2) + (cz * HASH_K3)) & 0xFFFFFFFF


def key_from_hash(hash_val: int, table_size: int) -> int:
    """KeyFromHash from SpatialHash3D.hlsl."""
    return hash_val % table_size


# ---------------------------------------------
# Scenario system
# ---------------------------------------------

class WhirlpoolScenario:
    """Whirlpool scenario: Creates a rotating vortex force."""
    
    def __init__(self):
        self.whirlpool_centre: Vec3 = (0.0, -1.0, 0.0)  # Center of the whirlpool
        self.whirlpool_radius: float = 3.0  # Maximum radius of effect
        self.whirlpool_strength: float = 15.0  # Strength of rotational force
        self.whirlpool_sink_strength: float = 8.0  # Strength of downward pull
        self.whirlpool_rotation_speed: float = 2.0  # Angular velocity (radians per second)
        self.whirlpool_time: float = 0.0  # Time accumulator for rotation
        
    def update(self, dt: float) -> None:
        """Update scenario state."""
        self.whirlpool_time += dt * self.whirlpool_rotation_speed
        
    def get_force(self, pos: Vec3, vel: Vec3) -> Vec3:
        """
        Calculate whirlpool force at a given position.
        Returns force vector to apply to particle.
        """
        # Vector from whirlpool centre to particle
        to_particle = v3_sub(pos, self.whirlpool_centre)
        dist_sq = v3_length_sq(to_particle)
        dist = math.sqrt(dist_sq)
        
        if dist < 0.1:  # Too close to center, avoid division by zero
            return (0.0, 0.0, 0.0)
        
        # Check if particle is within whirlpool radius
        if dist > self.whirlpool_radius:
            return (0.0, 0.0, 0.0)
        
        # Normalize direction
        to_particle_norm = v3_normalize(to_particle)
        
        # Calculate tangential direction (perpendicular to radial direction)
        # For a whirlpool, we want rotation around the Y axis
        # Tangential vector = cross product of (0,1,0) and radial direction
        up = (0.0, 1.0, 0.0)
        tangential = v3_cross(up, to_particle_norm)
        tangential = v3_normalize(tangential)
        
        # Add rotation based on time for dynamic effect
        # Rotate tangential vector around Y axis
        cos_t = math.cos(self.whirlpool_time)
        sin_t = math.sin(self.whirlpool_time)
        # Rotate in XZ plane
        tangential_rotated = (
            tangential[0] * cos_t - tangential[2] * sin_t,
            tangential[1],
            tangential[0] * sin_t + tangential[2] * cos_t,
        )
        
        # Force strength decreases with distance
        force_factor = 1.0 - (dist / self.whirlpool_radius)
        force_factor = max(0.0, force_factor)  # Clamp to [0, 1]
        
        # Tangential force (rotation)
        tangential_force = v3_mul(tangential_rotated, self.whirlpool_strength * force_factor)
        
        # Radial inward force (sink)
        radial_inward = v3_mul(to_particle_norm, -self.whirlpool_sink_strength * force_factor)
        
        # Downward force (sink)
        downward_force = (0.0, -self.whirlpool_sink_strength * force_factor * 0.5, 0.0)
        
        # Combine forces
        total_force = v3_add(tangential_force, radial_inward)
        total_force = v3_add(total_force, downward_force)
        
        return total_force


# ---------------------------------------------
# Simulation state and parameters
# ---------------------------------------------

class FluidSim3D:
    """
    CPU implementation of the Unity 3D fluid sim with scenario support.
    """

    def __init__(
        self,
        particles_per_axis: int = SimConfig.particles_per_axis,
        bounds_size: Vec3 = SimConfig.bounds_size,
        scenario_type: str = "none",
    ) -> None:
        # --- Simulation settings ---
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

        # Bounds are in "world units"
        self.bounds_size: Vec3 = bounds_size

        # Spawn settings
        self.particles_per_axis: int = particles_per_axis
        self.particle_count: int = particles_per_axis ** 3
        self.initial_velocity: Vec3 = SimConfig.initial_velocity
        self.spawn_centre: Vec3 = SimConfig.spawn_centre
        self.spawn_size: float = SimConfig.spawn_size
        self.jitter_strength: float = SimConfig.jitter_strength

        # Internal state arrays
        self.positions: List[Vec3] = []
        self.predicted_positions: List[Vec3] = []
        self.velocities: List[Vec3] = []
        self.densities: List[Tuple[float, float]] = []

        # Spatial hash
        self.cell_particles: Dict[int, List[int]] = {}

        # Collision statistics
        self.collision_events: int = 0

        # For reset
        self._initial_positions: List[Vec3] = []
        self._initial_velocities: List[Vec3] = []

        # Scenario system
        self.scenario_type = scenario_type
        self.scenario: Optional[WhirlpoolScenario] = None
        if scenario_type == "whirlpool":
            self.scenario = WhirlpoolScenario()
        self.scenario_active = False
        self.simulation_time = 0.0

        self._init_spawn()

    # -----------------------------
    # Spawn logic
    # -----------------------------

    def _init_spawn(self) -> None:
        """Compute initial particle positions/velocities."""
        self.positions = [(0.0, 0.0, 0.0)] * self.particle_count
        self.predicted_positions = [(0.0, 0.0, 0.0)] * self.particle_count
        self.velocities = [self.initial_velocity] * self.particle_count
        self.densities = [(0.0, 0.0)] * self.particle_count

        i = 0
        rng = random.Random(42)
        for x in range(self.particles_per_axis):
            for y in range(self.particles_per_axis):
                for z in range(self.particles_per_axis):
                    tx = x / max(self.particles_per_axis - 1, 1)
                    ty = y / max(self.particles_per_axis - 1, 1)
                    tz = z / max(self.particles_per_axis - 1, 1)

                    px = (tx - 0.5) * self.spawn_size + self.spawn_centre[0]
                    py = (ty - 0.5) * self.spawn_size + self.spawn_centre[1]
                    pz = (tz - 0.5) * self.spawn_size + self.spawn_centre[2]

                    angle1 = rng.random() * math.tau
                    angle2 = rng.random() * math.tau
                    jitter_mag = rng.random() * self.jitter_strength
                    jitter_x = jitter_mag * math.sin(angle1) * math.cos(angle2)
                    jitter_y = jitter_mag * math.sin(angle1) * math.sin(angle2)
                    jitter_z = jitter_mag * math.cos(angle1)

                    self.positions[i] = (
                        px + jitter_x,
                        py + jitter_y,
                        pz + jitter_z,
                    )
                    self.predicted_positions[i] = self.positions[i]
                    self.velocities[i] = self.initial_velocity
                    i += 1

        self._initial_positions = list(self.positions)
        self._initial_velocities = list(self.velocities)

    def reset(self) -> None:
        """Reset to initial spawn state."""
        self.positions = list(self._initial_positions)
        self.predicted_positions = list(self._initial_positions)
        self.velocities = list(self._initial_velocities)
        self.densities = [(0.0, 0.0)] * self.particle_count
        self.simulation_time = 0.0
        self.scenario_active = False

    # ----------------------------------------
    # External forces (gravity + scenarios)
    # ----------------------------------------

    def _external_forces(self, pos: Vec3, vel: Vec3) -> Vec3:
        """
        Calculate external forces including gravity and scenario forces.
        """
        # Base gravity
        gravity_accel = (0.0, self.gravity, 0.0)
        
        # Add scenario forces if active
        if self.scenario_active and self.scenario:
            scenario_force = self.scenario.get_force(pos, vel)
            return v3_add(gravity_accel, scenario_force)
        
        return gravity_accel

    # -------------------------
    # Collision handling
    # -------------------------

    def _handle_collisions(self, index: int) -> None:
        """Handle collisions with bounds."""
        pos = self.positions[index]
        vel = self.velocities[index]
        collided = False

        half_w = self.bounds_size[0] * 0.5
        half_h = self.bounds_size[1] * 0.5
        half_d = self.bounds_size[2] * 0.5

        edge_dst_x = half_w - abs(pos[0])
        edge_dst_y = half_h - abs(pos[1])
        edge_dst_z = half_d - abs(pos[2])

        if edge_dst_x <= 0:
            pos = (half_w * math.copysign(1.0, pos[0]), pos[1], pos[2])
            vel = (-vel[0] * self.collision_damping, vel[1], vel[2])
            collided = True
        if edge_dst_y <= 0:
            pos = (pos[0], half_h * math.copysign(1.0, pos[1]), pos[2])
            vel = (vel[0], -vel[1] * self.collision_damping, vel[2])
            collided = True
        if edge_dst_z <= 0:
            pos = (pos[0], pos[1], half_d * math.copysign(1.0, pos[2]))
            vel = (vel[0], vel[1], -vel[2] * self.collision_damping)
            collided = True

        self.positions[index] = pos
        self.velocities[index] = vel
        if collided:
            self.collision_events += 1

    # -------------------------
    # Spatial hash construction
    # -------------------------

    def _build_spatial_hash(self) -> None:
        """Build spatial hash table for neighbor lookup."""
        self.cell_particles.clear()
        r = self.smoothing_radius
        for i, pos in enumerate(self.predicted_positions):
            cell = get_cell_3d(pos, r)
            hash_val = hash_cell_3d(cell)
            key = key_from_hash(hash_val, self.particle_count)
            bucket = self.cell_particles.get(key)
            if bucket is None:
                bucket = []
                self.cell_particles[key] = bucket
            bucket.append((i, hash_val))

    # -------------------------
    # Density and near-density
    # -------------------------

    def _calculate_density_for_particle(self, index: int) -> Tuple[float, float]:
        """Calculate density for a particle."""
        pos = self.predicted_positions[index]
        r = self.smoothing_radius
        sqr_radius = r * r

        density = 0.0
        near_density = 0.0

        origin_cell = get_cell_3d(pos, r)

        for ox, oy, oz in OFFSETS_3D:
            neighbour_cell = (origin_cell[0] + ox, origin_cell[1] + oy, origin_cell[2] + oz)
            hash_val = hash_cell_3d(neighbour_cell)
            key = key_from_hash(hash_val, self.particle_count)
            bucket = self.cell_particles.get(key)
            if not bucket:
                continue

            for j, j_hash in bucket:
                if j_hash != hash_val:
                    continue
                neighbour_pos = self.predicted_positions[j]
                offset = v3_sub(neighbour_pos, pos)
                sqr_dst = v3_length_sq(offset)
                if sqr_dst > sqr_radius:
                    continue
                dst = math.sqrt(sqr_dst)
                density += spiky_kernel_pow2_3d(dst, r)
                near_density += spiky_kernel_pow3_3d(dst, r)

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
        """Apply pressure forces."""
        r = self.smoothing_radius
        sqr_radius = r * r

        accel: List[Vec3] = [(0.0, 0.0, 0.0)] * self.particle_count

        for i in range(self.particle_count):
            pos = self.predicted_positions[i]
            origin_cell = get_cell_3d(pos, r)

            density, near_density = self.densities[i]
            if density <= 1e-8:
                continue

            pressure = self._pressure_from_density(density)
            near_pressure_val = self._near_pressure_from_density(near_density)

            total_force = (0.0, 0.0, 0.0)

            for ox, oy, oz in OFFSETS_3D:
                neighbour_cell = (origin_cell[0] + ox, origin_cell[1] + oy, origin_cell[2] + oz)
                hash_val = hash_cell_3d(neighbour_cell)
                key = key_from_hash(hash_val, self.particle_count)
                bucket = self.cell_particles.get(key)
                if not bucket:
                    continue

                for j, j_hash in bucket:
                    if j_hash != hash_val:
                        continue
                    if j == i:
                        continue
                    neighbour_pos = self.predicted_positions[j]
                    offset = v3_sub(neighbour_pos, pos)
                    sqr_dst = v3_length_sq(offset)
                    if sqr_dst > sqr_radius:
                        continue
                    dst = math.sqrt(sqr_dst)
                    if dst <= 1e-8:
                        dir_to_neighbour = (0.0, 1.0, 0.0)
                    else:
                        dir_to_neighbour = v3_normalize(offset)

                    nd, n_near = self.densities[j]
                    if nd <= 1e-8 or n_near <= 0.0:
                        continue

                    neighbour_pressure = self._pressure_from_density(nd)
                    neighbour_near_pressure = self._near_pressure_from_density(n_near)

                    shared_pressure = 0.5 * (pressure + neighbour_pressure)
                    shared_near = 0.5 * (near_pressure_val + neighbour_near_pressure)

                    d_kernel = derivative_spiky_pow2_3d(dst, r)
                    nd_kernel = derivative_spiky_pow3_3d(dst, r)

                    scale1 = (d_kernel * shared_pressure) / max(nd, 1e-8)
                    scale2 = (nd_kernel * shared_near) / max(n_near, 1e-8)
                    force_scale = scale1 + scale2
                    force_x = dir_to_neighbour[0] * force_scale
                    force_y = dir_to_neighbour[1] * force_scale
                    force_z = dir_to_neighbour[2] * force_scale

                    total_force = (
                        total_force[0] + force_x,
                        total_force[1] + force_y,
                        total_force[2] + force_z,
                    )

            ax = total_force[0] / density
            ay = total_force[1] / density
            az = total_force[2] / density
            accel[i] = (ax, ay, az)

        for i in range(self.particle_count):
            vx, vy, vz = self.velocities[i]
            ax, ay, az = accel[i]
            self.velocities[i] = (vx + ax * dt, vy + ay * dt, vz + az * dt)

    # -------------------------
    # Viscosity forces
    # -------------------------

    def _apply_viscosity(self, dt: float) -> None:
        """Apply viscosity forces."""
        r = self.smoothing_radius
        sqr_radius = r * r
        viscosity_strength = self.viscosity_strength

        delta_v: List[Vec3] = [(0.0, 0.0, 0.0)] * self.particle_count

        for i in range(self.particle_count):
            pos = self.predicted_positions[i]
            origin_cell = get_cell_3d(pos, r)
            vi = self.velocities[i]
            visc_force = (0.0, 0.0, 0.0)

            for ox, oy, oz in OFFSETS_3D:
                neighbour_cell = (origin_cell[0] + ox, origin_cell[1] + oy, origin_cell[2] + oz)
                hash_val = hash_cell_3d(neighbour_cell)
                key = key_from_hash(hash_val, self.particle_count)
                bucket = self.cell_particles.get(key)
                if not bucket:
                    continue

                for j, j_hash in bucket:
                    if j_hash != hash_val:
                        continue
                    if j == i:
                        continue
                    neighbour_pos = self.predicted_positions[j]
                    offset = v3_sub(neighbour_pos, pos)
                    sqr_dst = v3_length_sq(offset)
                    if sqr_dst > sqr_radius:
                        continue
                    dst = math.sqrt(sqr_dst)
                    vj = self.velocities[j]
                    dv = (vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2])
                    w = smoothing_kernel_poly6_3d(dst, r)
                    visc_force = (
                        visc_force[0] + dv[0] * w,
                        visc_force[1] + dv[1] * w,
                        visc_force[2] + dv[2] * w,
                    )

            delta_v[i] = (
                visc_force[0] * viscosity_strength * dt,
                visc_force[1] * viscosity_strength * dt,
                visc_force[2] * viscosity_strength * dt,
            )

        for i in range(self.particle_count):
            vx, vy, vz = self.velocities[i]
            dvx, dvy, dvz = delta_v[i]
            self.velocities[i] = (vx + dvx, vy + dvy, vz + dvz)

    # -------------------------
    # Simulation step
    # -------------------------

    def step(self, frame_time: float) -> None:
        """Run one simulation frame, with substeps."""
        if self.iterations_per_frame <= 0:
            return
        
        # Update simulation time
        self.simulation_time += frame_time
        
        # Check if scenario should activate
        if not self.scenario_active and self.scenario_type == "whirlpool":
            if self.simulation_time >= SimConfig.scenario_activation_time:
                self.scenario_active = True
                print(f"Whirlpool scenario activated at {self.simulation_time:.1f} seconds!")
        
        # Update scenario if active
        if self.scenario_active and self.scenario:
            dt = (frame_time / self.iterations_per_frame) * self.time_scale
            self.scenario.update(dt)
        
        # Reset collision counter for this frame
        self.collision_events = 0
        dt = (frame_time / self.iterations_per_frame) * self.time_scale
        prediction_factor = 1.0 / 120.0

        for _ in range(self.iterations_per_frame):
            # External forces and prediction
            for i in range(self.particle_count):
                pos = self.positions[i]
                vel = self.velocities[i]
                accel = self._external_forces(pos, vel)
                vel = (vel[0] + accel[0] * dt, vel[1] + accel[1] * dt, vel[2] + accel[2] * dt)
                self.velocities[i] = vel
                self.predicted_positions[i] = (
                    pos[0] + vel[0] * prediction_factor,
                    pos[1] + vel[1] * prediction_factor,
                    pos[2] + vel[2] * prediction_factor,
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
                self.positions[i] = (
                    pos[0] + vel[0] * dt,
                    pos[1] + vel[1] * dt,
                    pos[2] + vel[2] * dt,
                )
                self._handle_collisions(i)


# ---------------------------------------------
# Pygame visualization and input handling
# ---------------------------------------------

class FluidSimApp3D:
    """Pygame app wrapper around FluidSim3D with 3D projection and scenarios."""

    def __init__(
        self,
        particles_per_axis: int = SimConfig.particles_per_axis,
        bounds_size: Vec3 = SimConfig.bounds_size,
        screen_size: Tuple[int, int] = (SimConfig.screen_width, SimConfig.screen_height),
        scenario_type: str = "none",
        record: bool = False,
        record_duration: float = 10.0,
        playback_fps: int = 60,
        output_file: Optional[str] = None,
    ) -> None:
        pygame.init()
        self.clock = pygame.time.Clock()

        # Window
        self.pixel_width = screen_size[0]
        self.pixel_height = screen_size[1]
        self.screen = pygame.display.set_mode((self.pixel_width, self.pixel_height))
        caption = f"3D Fluid Sim - {scenario_type.capitalize() if scenario_type != 'none' else 'Normal'}"
        pygame.display.set_caption(caption)

        # Simulation
        self.sim = FluidSim3D(
            particles_per_axis=particles_per_axis,
            bounds_size=bounds_size,
            scenario_type=scenario_type,
        )

        # 3D Camera state
        self.camera_distance = 10.0
        self.camera_angle_x = 0.5
        self.camera_angle_y = 0.3
        self.camera_zoom = 1.0
        
        # Perspective projection parameters
        self.fov = math.pi / 4.0
        self.near_plane = 0.1
        self.far_plane = 100.0
        
        # Mouse control state
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)

        self.running = True
        self.paused = False
        self.step_once = False

        # Precompute particle radius in pixels
        self.particle_radius_px = 3

        # Graph display state
        self.show_graphs = False
        self.fps_history: List[float] = []
        self.collision_history: List[float] = []
        self.max_history_points = 240

        # Recording state
        self.recording = record
        self.record_duration = record_duration
        self.playback_fps = playback_fps
        self.output_file = output_file
        self.recorded_frames: List[pygame.Surface] = []
        self.recording_start_time = 0.0
        self.recording_complete = False
        
        # Scenario info
        self.scenario_type = scenario_type

    def project_3d_to_2d(self, pos: Vec3) -> Tuple[int, int, float]:
        """Project 3D world position to 2D screen coordinates using proper perspective projection."""
        cam_x = self.camera_distance * math.cos(self.camera_angle_y) * math.sin(self.camera_angle_x)
        cam_y = self.camera_distance * math.sin(self.camera_angle_y)
        cam_z = self.camera_distance * math.cos(self.camera_angle_y) * math.cos(self.camera_angle_x)
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
        
        if z_cam < self.near_plane:
            return (-9999, -9999, -9999.0)
        
        aspect = self.pixel_width / self.pixel_height
        fov_scale = 1.0 / math.tan(self.fov / 2.0)
        
        # Fixed aspect ratio: divide X by aspect to keep square pixels
        x_proj = (x_cam / z_cam) * fov_scale * self.camera_zoom / aspect
        y_proj = (y_cam / z_cam) * fov_scale * self.camera_zoom
        
        screen_x = int(self.pixel_width * 0.5 + x_proj * self.pixel_width * 0.5)
        screen_y = int(self.pixel_height * 0.5 - y_proj * self.pixel_height * 0.5)
        
        return screen_x, screen_y, z_cam

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT:
                    self.step_once = True
                elif event.key == pygame.K_r:
                    self.sim.reset()
                elif event.key == pygame.K_g:
                    self.show_graphs = not self.show_graphs
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    self.camera_zoom = min(self.camera_zoom * 1.1, 3.0)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    self.camera_zoom = max(self.camera_zoom / 1.1, 0.3)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    rotation_speed = 0.005
                    self.camera_angle_x += dx * rotation_speed
                    self.camera_angle_y = max(
                        min(self.camera_angle_y - dy * rotation_speed, math.pi / 2 - 0.1),
                        -math.pi / 2 + 0.1
                    )
                    self.last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = 1.1 if event.y > 0 else 1.0 / 1.1
                self.camera_zoom = max(0.3, min(3.0, self.camera_zoom * zoom_factor))

        keys = pygame.key.get_pressed()
        rotation_speed = 0.02
        if keys[pygame.K_LEFT]:
            self.camera_angle_x -= rotation_speed
        if keys[pygame.K_RIGHT]:
            self.camera_angle_x += rotation_speed
        if keys[pygame.K_UP]:
            self.camera_angle_y = max(self.camera_angle_y - rotation_speed, -math.pi / 2 + 0.1)
        if keys[pygame.K_DOWN]:
            self.camera_angle_y = min(self.camera_angle_y + rotation_speed, math.pi / 2 - 0.1)

    def _draw(self) -> None:
        self.screen.fill((10, 10, 30))

        # Draw bounds
        half_w = self.sim.bounds_size[0] * 0.5
        half_h = self.sim.bounds_size[1] * 0.5
        half_d = self.sim.bounds_size[2] * 0.5

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
            p1_x, p1_y, p1_z = self.project_3d_to_2d(corners[i])
            p2_x, p2_y, p2_z = self.project_3d_to_2d(corners[j])
            if p1_z > 0 and p2_z > 0:
                pygame.draw.line(self.screen, (80, 80, 80), (p1_x, p1_y), (p2_x, p2_y), 1)

        # Draw scenario visualization (whirlpool center)
        if self.scenario_type == "whirlpool" and self.sim.scenario_active and self.sim.scenario:
            centre_x, centre_y, centre_z = self.sim.scenario.whirlpool_centre
            sx, sy, depth = self.project_3d_to_2d((centre_x, centre_y, centre_z))
            if depth > 0:
                # Draw whirlpool center indicator
                pygame.draw.circle(self.screen, (255, 100, 100), (int(sx), int(sy)), 8, 2)
                # Draw radius indicator
                radius_points = []
                for angle in range(0, 360, 10):
                    rad = math.radians(angle)
                    px = centre_x + self.sim.scenario.whirlpool_radius * math.cos(rad)
                    pz = centre_z + self.sim.scenario.whirlpool_radius * math.sin(rad)
                    py = centre_y
                    psx, psy, pdepth = self.project_3d_to_2d((px, py, pz))
                    if pdepth > 0:
                        radius_points.append((int(psx), int(psy)))
                if len(radius_points) > 2:
                    pygame.draw.lines(self.screen, (255, 150, 150), True, radius_points, 1)

        # Draw particles
        max_speed = 6.0
        particles_to_draw = []
        for i in range(self.sim.particle_count):
            pos = self.sim.positions[i]
            vel = self.sim.velocities[i]
            spd = min(v3_length(vel) / max_speed, 1.0)
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
            sx, sy, depth = self.project_3d_to_2d(pos)
            if depth > 0 and sx >= 0 and sx < self.pixel_width and sy >= 0 and sy < self.pixel_height:
                particles_to_draw.append((sx, sy, depth, color))
        
        particles_to_draw.sort(key=lambda p: p[2], reverse=True)
        for sx, sy, depth, color in particles_to_draw:
            pygame.draw.circle(self.screen, color, (int(sx), int(sy)), self.particle_radius_px)

        # Draw scenario status
        font = pygame.font.SysFont("consolas", 14)
        if self.scenario_type == "whirlpool":
            if self.sim.scenario_active:
                status_text = f"WHIRLPOOL ACTIVE - Time: {self.sim.simulation_time:.1f}s"
                status_color = (255, 100, 100)
            else:
                time_remaining = SimConfig.scenario_activation_time - self.sim.simulation_time
                status_text = f"Whirlpool in {time_remaining:.1f}s - Time: {self.sim.simulation_time:.1f}s"
                status_color = (200, 200, 200)
            text_surf = font.render(status_text, True, status_color)
            self.screen.blit(text_surf, (10, 10))

        # Overlay graphs
        if self.show_graphs:
            self._draw_graphs()

        pygame.display.flip()

    def _draw_graphs(self) -> None:
        """Draw FPS and collisions-per-second graphs."""
        margin = 10
        graph_width = 260
        graph_height = 80
        right = self.pixel_width - margin
        top = margin + 30  # Leave space for scenario status

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
            max_val *= 1.1

            for i in range(1, n):
                x0 = rect.right - rect.width + int(rect.width * (i - 1) / max(n - 1, 1))
                x1 = rect.right - rect.width + int(rect.width * i / max(n - 1, 1))
                y0 = rect.bottom - int(rect.height * (data[i - 1] / max_val))
                y1 = rect.bottom - int(rect.height * (data[i] / max_val))
                pygame.draw.line(self.screen, color, (x0, y0), (x1, y1), 2)

            text_color = (230, 230, 230)
            for iy, frac in enumerate([0.0, 0.5, 1.0]):
                value = max_val * frac
                y = rect.bottom - int(rect.height * frac)
                label = f"{value:.0f}"
                surf = font.render(label, True, text_color)
                self.screen.blit(surf, (rect.left - surf.get_width() - 4, y - surf.get_height() // 2))

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

            y_label_surf = font.render(y_label, True, text_color)
            self.screen.blit(
                y_label_surf, (rect.left - y_label_surf.get_width() - 4, rect.top - 2)
            )
            x_label_surf = font.render(x_label, True, text_color)
            self.screen.blit(
                x_label_surf,
                (rect.right - x_label_surf.get_width(), rect.bottom + 16),
            )

        fps_rect = pygame.Rect(right - graph_width, top, graph_width, graph_height)
        draw_single_graph(
            self.fps_history,
            fps_rect,
            fps_color,
            "FPS",
            "time (frames)",
        )

        col_rect = pygame.Rect(
            right - graph_width,
            top + graph_height + 24,
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

    def run(self) -> None:
        target_fps = 60
        
        if self.recording:
            import time as time_module
            self.recording_start_time = time_module.time()
            print(f"Recording for {self.record_duration} seconds...")
            print("Press ESC to stop recording early.")
            
            while self.running and not self.recording_complete:
                frame_time = self.clock.tick(target_fps) / 1000.0
                self._handle_events()
                
                elapsed = time_module.time() - self.recording_start_time
                if elapsed >= self.record_duration:
                    self.recording_complete = True
                    print(f"Recording complete! Captured {len(self.recorded_frames)} frames.")
                    break

                if not self.paused or self.step_once:
                    self.sim.step(1.0 / target_fps if self.sim.fixed_time_step else frame_time)
                    self.step_once = False

                    fps = self.clock.get_fps()
                    if fps > 0:
                        self.fps_history.append(fps)
                    collisions_per_sec = 0.0
                    if frame_time > 0:
                        collisions_per_sec = self.sim.collision_events / frame_time
                    self.collision_history.append(collisions_per_sec)

                    if len(self.fps_history) > self.max_history_points:
                        self.fps_history = self.fps_history[-self.max_history_points :]
                    if len(self.collision_history) > self.max_history_points:
                        self.collision_history = self.collision_history[-self.max_history_points :]

                self._draw()
                frame_copy = self.screen.copy()
                self.recorded_frames.append(frame_copy)
                
                if len(self.recorded_frames) % 30 == 0:
                    progress = (elapsed / self.record_duration) * 100
                    print(f"Recording: {progress:.1f}% ({len(self.recorded_frames)} frames)")
            
            if self.recording_complete and len(self.recorded_frames) > 0:
                if self.output_file:
                    self._save_recording_to_file()
                
                print(f"\nPlaying back {len(self.recorded_frames)} frames at {self.playback_fps} FPS...")
                print("Press ESC to exit playback.")
                self._playback_recording()
        else:
            # Normal simulation mode
            while self.running:
                frame_time = self.clock.tick(target_fps) / 1000.0
                self._handle_events()

                if not self.paused or self.step_once:
                    self.sim.step(1.0 / target_fps if self.sim.fixed_time_step else frame_time)
                    self.step_once = False

                    fps = self.clock.get_fps()
                    if fps > 0:
                        self.fps_history.append(fps)
                    collisions_per_sec = 0.0
                    if frame_time > 0:
                        collisions_per_sec = self.sim.collision_events / frame_time
                    self.collision_history.append(collisions_per_sec)

                    if len(self.fps_history) > self.max_history_points:
                        self.fps_history = self.fps_history[-self.max_history_points :]
                    if len(self.collision_history) > self.max_history_points:
                        self.collision_history = self.collision_history[-self.max_history_points :]

                self._draw()

        pygame.quit()

    def _playback_recording(self) -> None:
        """Play back recorded frames at specified FPS."""
        playback_clock = pygame.time.Clock()
        frame_index = 0
        
        while self.running and frame_index < len(self.recorded_frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            self.screen.blit(self.recorded_frames[frame_index], (0, 0))
            pygame.display.flip()
            
            if frame_index % 30 == 0:
                progress = (frame_index / len(self.recorded_frames)) * 100
                print(f"Playback: {progress:.1f}% ({frame_index}/{len(self.recorded_frames)})")
            
            frame_index += 1
            playback_clock.tick(self.playback_fps)
        
        print("Playback complete!")

    def _save_recording_to_file(self) -> None:
        """Save recorded frames to a video file."""
        if not IMAGEIO_AVAILABLE:
            print("Warning: imageio or numpy not available.")
            print("Install with: pip install imageio imageio-ffmpeg numpy")
            print("Skipping video file save.")
            return
        
        if not self.output_file:
            return
        
        print(f"\nSaving recording to {self.output_file}...")
        print("This may take a while depending on the number of frames...")
        
        try:
            frames_data = []
            for i, frame in enumerate(self.recorded_frames):
                frame_array = pygame.surfarray.array3d(frame)
                frame_array = frame_array.transpose(1, 0, 2)
                frames_data.append(frame_array)
                
                if (i + 1) % 30 == 0:
                    progress = ((i + 1) / len(self.recorded_frames)) * 100
                    print(f"Converting frames: {progress:.1f}% ({i + 1}/{len(self.recorded_frames)})")
            
            print("Writing video file...")
            imageio.mimwrite(
                self.output_file,
                frames_data,
                fps=self.playback_fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )
            print(f"Successfully saved recording to {self.output_file}")
            print(f"Video: {len(self.recorded_frames)} frames at {self.playback_fps} FPS")
        except Exception as e:
            print(f"Error saving video file: {e}")
            print("Make sure imageio-ffmpeg is installed: pip install imageio-ffmpeg")
            import traceback
            traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Fluid Simulator with Scenarios (CPU, pygame)")
    parser.add_argument(
        "--particles-per-axis",
        type=int,
        default=12,
        help="Number of particles per axis (creates particles_per_axis^3 total particles).",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=4.0,
        help="Size of the simulation bounds in world units (cubic box).",
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
    parser.add_argument(
        "--scenario",
        type=str,
        default="whirlpool",
        choices=["none", "whirlpool"],
        help="Scenario type to run (default: whirlpool).",
    )
    parser.add_argument(
        "--scenario-activation-time",
        type=float,
        default=2.5,
        help="Time in seconds before scenario activates (default: 300.0).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable recording mode.",
    )
    parser.add_argument(
        "--record-duration",
        type=float,
        default=10.0,
        help="Duration in seconds to record (default: 10.0).",
    )
    parser.add_argument(
        "--playback-fps",
        type=int,
        default=60,
        help="FPS for playback of recorded frames (default: 60).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path for saving the recording as a video.",
    )
    args = parser.parse_args()

    # Update config with activation time
    SimConfig.scenario_activation_time = args.scenario_activation_time

    bounds = (args.box_size, args.box_size, args.box_size)
    screen_size = (args.screen_width, args.screen_height)
    app = FluidSimApp3D(
        particles_per_axis=args.particles_per_axis,
        bounds_size=bounds,
        screen_size=screen_size,
        scenario_type=args.scenario,
        record=args.record,
        record_duration=args.record_duration,
        playback_fps=args.playback_fps,
        output_file=args.output_file,
    )
    app.run()


if __name__ == "__main__":
    main()

