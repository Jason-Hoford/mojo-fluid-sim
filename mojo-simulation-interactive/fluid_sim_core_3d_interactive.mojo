from math import sqrt, floor, ceil, cos, sin
from random import random_float64, seed
from collections import List
from python import Python
from sys import argv

alias PI: Float64 = 3.14159265358979323846

# -----------------------------
# Configuration
# -----------------------------

struct SimConfig:
    var particles_per_axis: Int
    var bounds_size: SIMD[DType.float64, 4]
    var time_scale: Float64
    var iterations_per_frame: Int
    var gravity: Float64
    var collision_damping: Float64
    var smoothing_radius: Float64
    var target_density: Float64
    var pressure_multiplier: Float64
    var near_pressure_multiplier: Float64
    var viscosity_strength: Float64
    var initial_velocity: SIMD[DType.float64, 4]
    var spawn_centre: SIMD[DType.float64, 4]
    var spawn_size: Float64
    var jitter_strength: Float64
    var scenario_type: Int  # 0 = none, 1 = whirlpool
    var scenario_activation_time: Float64

    fn __init__(out self):
        self.particles_per_axis = 12
        self.bounds_size = SIMD[DType.float64, 4](4.0, 4.0, 4.0)
        self.time_scale = 1.0
        self.iterations_per_frame = 10
        self.gravity = -10.0
        self.collision_damping = 0.05
        self.smoothing_radius = 0.2
        self.target_density = 12.0
        self.pressure_multiplier = 120.0
        self.near_pressure_multiplier = 20.0
        self.viscosity_strength = 0.2
        self.initial_velocity = SIMD[DType.float64, 4](0.0, 0.0, 0.0)
        self.spawn_centre = SIMD[DType.float64, 4](0.0, 1.0, 0.0)
        self.spawn_size = 1.5
        self.jitter_strength = 0.02
        self.scenario_type = 1  # 1 = whirlpool
        self.scenario_activation_time = 2.5


# -----------------------------
# 3D Kernel functions
# -----------------------------

fn smoothing_kernel_poly6_3d(dst: Float64, radius: Float64) -> Float64:
    if dst < radius:
        var scale = 315.0 / (64.0 * PI * (radius ** 9))
        var v = radius * radius - dst * dst
        return v * v * v * scale
    return 0.0


fn spiky_kernel_pow3_3d(dst: Float64, radius: Float64) -> Float64:
    if dst < radius:
        var scale = 15.0 / (PI * (radius ** 6))
        var v = radius - dst
        return v * v * v * scale
    return 0.0


fn spiky_kernel_pow2_3d(dst: Float64, radius: Float64) -> Float64:
    if dst < radius:
        var scale = 15.0 / (2.0 * PI * (radius ** 5))
        var v = radius - dst
        return v * v * scale
    return 0.0


fn derivative_spiky_pow3_3d(dst: Float64, radius: Float64) -> Float64:
    if dst <= radius:
        var scale = 45.0 / ((radius ** 6) * PI)
        var v = radius - dst
        return -v * v * scale
    return 0.0


fn derivative_spiky_pow2_3d(dst: Float64, radius: Float64) -> Float64:
    if dst <= radius:
        var scale = 15.0 / ((radius ** 5) * PI)
        var v = radius - dst
        return -v * scale
    return 0.0


fn length_sq_3d(v: SIMD[DType.float64, 4]) -> Float64:
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]


fn length_3d(v: SIMD[DType.float64, 4]) -> Float64:
    return sqrt(length_sq_3d(v))


fn normalize_3d(v: SIMD[DType.float64, 4]) -> SIMD[DType.float64, 4]:
    var len = length_3d(v)
    if len <= 1e-8:
        return SIMD[DType.float64, 4](0.0, 0.0, 1.0)
    return v / len


fn cross_3d(a: SIMD[DType.float64, 4], b: SIMD[DType.float64, 4]) -> SIMD[DType.float64, 4]:
    return SIMD[DType.float64, 4](
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )


# -----------------------------
# 3D Spatial hashing
# -----------------------------

fn get_cell_3d(position: SIMD[DType.float64, 4], radius: Float64) -> SIMD[DType.int32, 4]:
    return SIMD[DType.int32, 4](
        Int32(floor(position[0] / radius)),
        Int32(floor(position[1] / radius)),
        Int32(floor(position[2] / radius)),
        0
    )


fn hash_cell_3d(cell: SIMD[DType.int32, 4]) -> Int:
    var HASH_K1: Int = 15823
    var HASH_K2: Int = 9737333
    var HASH_K3: Int = 440817757
    var cx = Int(cell[0]) & 0xFFFFFFFF
    var cy = Int(cell[1]) & 0xFFFFFFFF
    var cz = Int(cell[2]) & 0xFFFFFFFF
    return ((cx * HASH_K1) + (cy * HASH_K2) + (cz * HASH_K3)) & 0xFFFFFFFF


struct SpatialHash3D:
    var table_size: Int
    var cell_start: List[Int]
    var cell_end: List[Int]
    var cell_entries: List[Int]
    var particle_cells: List[Int]
    
    fn __init__(out self, capacity: Int):
        # Use a hash table size that's a power of 2 for fast modulo
        self.table_size = 32768  # Reasonable size for 3D
        self.cell_start = List[Int](capacity=self.table_size)
        self.cell_end = List[Int](capacity=self.table_size)
        self.cell_entries = List[Int](capacity=capacity)
        self.particle_cells = List[Int](capacity=capacity)
        
        for _ in range(self.table_size):
            self.cell_start.append(-1)
            self.cell_end.append(-1)
        
        for _ in range(capacity):
            self.cell_entries.append(0)
            self.particle_cells.append(0)
    
    fn clear(mut self):
        for i in range(self.table_size):
            self.cell_start[i] = -1
            self.cell_end[i] = -1
    
    fn build(mut self, positions: List[SIMD[DType.float64, 4]], radius: Float64):
        self.clear()
        var n = len(positions)
        
        # Step 1: Compute cell hash for each particle
        for i in range(n):
            var cell = get_cell_3d(positions[i], radius)
            var hash = hash_cell_3d(cell) & (self.table_size - 1)  # Fast modulo for power of 2
            self.particle_cells[i] = hash
        
        # Step 2: Count particles per cell
        var cell_count = List[Int](capacity=self.table_size)
        for _ in range(self.table_size):
            cell_count.append(0)
        
        for i in range(n):
            var hash = self.particle_cells[i]
            cell_count[hash] += 1
        
        # Step 3: Compute cell_start offsets (prefix sum)
        var offset = 0
        for hash in range(self.table_size):
            self.cell_start[hash] = offset
            self.cell_end[hash] = offset
            offset += cell_count[hash]
        
        # Step 4: Fill cell_entries with particle indices
        for i in range(n):
            var hash = self.particle_cells[i]
            var idx = self.cell_end[hash]
            self.cell_entries[idx] = i
            self.cell_end[hash] += 1
    
    fn get_cell_hash(self, cell: SIMD[DType.int32, 4]) -> Int:
        """Get hash for a cell."""
        return hash_cell_3d(cell) & (self.table_size - 1)
    
    fn get_start(self, hash: Int) -> Int:
        """Get start index for a hash."""
        return self.cell_start[hash]
    
    fn get_end(self, hash: Int) -> Int:
        """Get end index for a hash."""
        return self.cell_end[hash]


# -----------------------------
# Scenario system
# -----------------------------

struct WhirlpoolScenario:
    var whirlpool_centre: SIMD[DType.float64, 4]
    var whirlpool_radius: Float64
    var whirlpool_strength: Float64
    var whirlpool_sink_strength: Float64
    var whirlpool_rotation_speed: Float64
    var whirlpool_time: Float64

    fn __init__(out self):
        self.whirlpool_centre = SIMD[DType.float64, 4](0.0, -1.0, 0.0)
        self.whirlpool_radius = 3.0
        self.whirlpool_strength = 15.0
        self.whirlpool_sink_strength = 8.0
        self.whirlpool_rotation_speed = 2.0
        self.whirlpool_time = 0.0

    fn update(mut self, dt: Float64):
        self.whirlpool_time += dt * self.whirlpool_rotation_speed

    fn get_force(self, pos: SIMD[DType.float64, 4], vel: SIMD[DType.float64, 4]) -> SIMD[DType.float64, 4]:
        var to_particle = pos - self.whirlpool_centre
        var dist_sq = length_sq_3d(to_particle)
        var dist = sqrt(dist_sq)

        if dist < 0.1:
            return SIMD[DType.float64, 4](0.0, 0.0, 0.0)

        if dist > self.whirlpool_radius:
            return SIMD[DType.float64, 4](0.0, 0.0, 0.0)

        var to_particle_norm = normalize_3d(to_particle)
        var up = SIMD[DType.float64, 4](0.0, 1.0, 0.0)
        var tangential = cross_3d(up, to_particle_norm)
        tangential = normalize_3d(tangential)

        var cos_t = cos(self.whirlpool_time)
        var sin_t = sin(self.whirlpool_time)
        var tangential_rotated = SIMD[DType.float64, 4](
            tangential[0] * cos_t - tangential[2] * sin_t,
            tangential[1],
            tangential[0] * sin_t + tangential[2] * cos_t
        )

        var force_factor = 1.0 - (dist / self.whirlpool_radius)
        if force_factor < 0.0:
            force_factor = 0.0

        var tangential_force = tangential_rotated * (self.whirlpool_strength * force_factor)
        var radial_inward = to_particle_norm * (-self.whirlpool_sink_strength * force_factor)
        var downward_force = SIMD[DType.float64, 4](0.0, -self.whirlpool_sink_strength * force_factor * 0.5, 0.0)

        return tangential_force + radial_inward + downward_force


# -----------------------------
# 3D Fluid simulation
# -----------------------------

struct FluidSim3D:
    var time_scale: Float64
    var iterations_per_frame: Int
    var gravity: Float64
    var collision_damping: Float64
    var smoothing_radius: Float64
    var target_density: Float64
    var pressure_multiplier: Float64
    var near_pressure_multiplier: Float64
    var viscosity_strength: Float64
    var bounds_size: SIMD[DType.float64, 4]
    var particles_per_axis: Int
    var particle_count: Int
    var initial_velocity: SIMD[DType.float64, 4]
    var spawn_centre: SIMD[DType.float64, 4]
    var spawn_size: Float64
    var jitter_strength: Float64
    var positions: List[SIMD[DType.float64, 4]]
    var predicted_positions: List[SIMD[DType.float64, 4]]
    var velocities: List[SIMD[DType.float64, 4]]
    var densities: List[SIMD[DType.float64, 2]]
    var spatial_hash: SpatialHash3D
    var collision_events: Int
    var scenario_type: Int  # 0 = none, 1 = whirlpool
    var scenario: WhirlpoolScenario
    var has_scenario: Bool
    var scenario_active: Bool
    var simulation_time: Float64
    var poly6_scaling_factor: Float64
    var spiky_pow3_scaling_factor: Float64
    var spiky_pow2_scaling_factor: Float64
    var spiky_pow3_derivative_scaling_factor: Float64
    var spiky_pow2_derivative_scaling_factor: Float64
    var scenario_activation_time: Float64

    fn __init__(out self, var cfg: SimConfig):
        self.time_scale = cfg.time_scale
        self.iterations_per_frame = cfg.iterations_per_frame
        self.gravity = cfg.gravity
        self.collision_damping = cfg.collision_damping
        self.smoothing_radius = cfg.smoothing_radius
        self.target_density = cfg.target_density
        self.pressure_multiplier = cfg.pressure_multiplier
        self.near_pressure_multiplier = cfg.near_pressure_multiplier
        self.viscosity_strength = cfg.viscosity_strength
        self.bounds_size = cfg.bounds_size
        self.particles_per_axis = cfg.particles_per_axis
        self.particle_count = cfg.particles_per_axis ** 3
        self.initial_velocity = cfg.initial_velocity
        self.spawn_centre = cfg.spawn_centre
        self.spawn_size = cfg.spawn_size
        self.jitter_strength = cfg.jitter_strength
        self.scenario_type = cfg.scenario_type
        self.scenario_activation_time = cfg.scenario_activation_time

        var r = cfg.smoothing_radius
        self.poly6_scaling_factor = 315.0 / (64.0 * PI * (r ** 9))
        self.spiky_pow3_scaling_factor = 15.0 / (PI * (r ** 6))
        self.spiky_pow2_scaling_factor = 15.0 / (2.0 * PI * (r ** 5))
        self.spiky_pow3_derivative_scaling_factor = 45.0 / ((r ** 6) * PI)
        self.spiky_pow2_derivative_scaling_factor = 15.0 / ((r ** 5) * PI)

        self.positions = List[SIMD[DType.float64, 4]]()
        self.predicted_positions = List[SIMD[DType.float64, 4]]()
        self.velocities = List[SIMD[DType.float64, 4]]()
        self.densities = List[SIMD[DType.float64, 2]]()
        self.spatial_hash = SpatialHash3D(self.particle_count)
        self.collision_events = 0
        self.scenario_active = False
        self.simulation_time = 0.0

        if cfg.scenario_type == 1:
            self.scenario = WhirlpoolScenario()
            self.has_scenario = True
        else:
            self.scenario = WhirlpoolScenario()
            self.has_scenario = False

        seed()
        
        # Initialize spawn
        var i = 0
        for x in range(self.particles_per_axis):
            for y in range(self.particles_per_axis):
                for z in range(self.particles_per_axis):
                    var tx = Float64(x) / max(Float64(self.particles_per_axis - 1), 1.0)
                    var ty = Float64(y) / max(Float64(self.particles_per_axis - 1), 1.0)
                    var tz = Float64(z) / max(Float64(self.particles_per_axis - 1), 1.0)

                    var px = (tx - 0.5) * self.spawn_size + self.spawn_centre[0]
                    var py = (ty - 0.5) * self.spawn_size + self.spawn_centre[1]
                    var pz = (tz - 0.5) * self.spawn_size + self.spawn_centre[2]

                    var angle1 = random_float64() * 2.0 * PI
                    var angle2 = random_float64() * 2.0 * PI
                    var jitter_mag = random_float64() * self.jitter_strength
                    var jitter_x = jitter_mag * sin(angle1) * cos(angle2)
                    var jitter_y = jitter_mag * sin(angle1) * sin(angle2)
                    var jitter_z = jitter_mag * cos(angle1)

                    self.positions.append(SIMD[DType.float64, 4](px + jitter_x, py + jitter_y, pz + jitter_z))
                    self.predicted_positions.append(SIMD[DType.float64, 4](px + jitter_x, py + jitter_y, pz + jitter_z))
                    self.velocities.append(self.initial_velocity)
                    self.densities.append(SIMD[DType.float64, 2](0.0, 0.0))
                    i += 1

    fn _external_forces(self, pos: SIMD[DType.float64, 4], vel: SIMD[DType.float64, 4]) -> SIMD[DType.float64, 4]:
        var gravity_accel = SIMD[DType.float64, 4](0.0, self.gravity, 0.0)

        if self.scenario_active and self.has_scenario:
            var scenario_force = self.scenario.get_force(pos, vel)
            return gravity_accel + scenario_force

        return gravity_accel

    fn _handle_collisions(mut self, index: Int):
        var pos = self.positions[index]
        var vel = self.velocities[index]
        var collided = False

        var half_w = self.bounds_size[0] * 0.5
        var half_h = self.bounds_size[1] * 0.5
        var half_d = self.bounds_size[2] * 0.5

        var edge_dst_x = half_w - abs(pos[0])
        var edge_dst_y = half_h - abs(pos[1])
        var edge_dst_z = half_d - abs(pos[2])

        if edge_dst_x <= 0:
            pos[0] = half_w * (1.0 if pos[0] >= 0.0 else -1.0)
            vel[0] = -vel[0] * self.collision_damping
            collided = True
        if edge_dst_y <= 0:
            pos[1] = half_h * (1.0 if pos[1] >= 0.0 else -1.0)
            vel[1] = -vel[1] * self.collision_damping
            collided = True
        if edge_dst_z <= 0:
            pos[2] = half_d * (1.0 if pos[2] >= 0.0 else -1.0)
            vel[2] = -vel[2] * self.collision_damping
            collided = True

        self.positions[index] = pos
        self.velocities[index] = vel
        if collided:
            self.collision_events += 1

    fn _build_spatial_hash(mut self):
        self.spatial_hash.build(self.predicted_positions, self.smoothing_radius)

    fn _update_densities(mut self):
        var r = self.smoothing_radius
        var sqr_radius = r * r

        for i in range(self.particle_count):
            var pos = self.predicted_positions[i]
            var origin_cell = get_cell_3d(pos, r)
            var density: Float64 = 0.0
            var near_density: Float64 = 0.0

            # Check 27 neighboring cells (3x3x3 grid)
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        var neighbor_cell = SIMD[DType.int32, 4](origin_cell[0] + dx, origin_cell[1] + dy, origin_cell[2] + dz, 0)
                        var hash = self.spatial_hash.get_cell_hash(neighbor_cell)
                        var start = self.spatial_hash.get_start(hash)
                        var end = self.spatial_hash.get_end(hash)
                        
                        if start < 0:
                            continue
                        
                        for idx in range(start, end):
                            var j = self.spatial_hash.cell_entries[idx]
                            var npos = self.predicted_positions[j]
                            var delta = npos - pos
                            var sqr_dst = length_sq_3d(delta)
                            if sqr_dst > sqr_radius:
                                continue

                            var dst = sqrt(sqr_dst)
                            density += spiky_kernel_pow2_3d(dst, r)
                            near_density += spiky_kernel_pow3_3d(dst, r)

            self.densities[i] = SIMD[DType.float64, 2](density, near_density)

    fn _apply_pressure_forces(mut self, dt: Float64):
        var r = self.smoothing_radius
        var sqr_radius = r * r

        for i in range(self.particle_count):
            var pos = self.predicted_positions[i]
            var origin_cell = get_cell_3d(pos, r)
            var density = self.densities[i][0]
            var near_density = self.densities[i][1]
            if density <= 1e-8:
                continue

            var pressure = (density - self.target_density) * self.pressure_multiplier
            var near_pressure_val = near_density * self.near_pressure_multiplier

            var total_force = SIMD[DType.float64, 4](0.0, 0.0, 0.0)

            # Check 27 neighboring cells
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        var neighbor_cell = SIMD[DType.int32, 4](origin_cell[0] + dx, origin_cell[1] + dy, origin_cell[2] + dz, 0)
                        var hash = self.spatial_hash.get_cell_hash(neighbor_cell)
                        var start = self.spatial_hash.get_start(hash)
                        var end = self.spatial_hash.get_end(hash)
                        
                        if start < 0:
                            continue
                        
                        for idx in range(start, end):
                            var j = self.spatial_hash.cell_entries[idx]
                            if j == i:
                                continue
                            var npos = self.predicted_positions[j]
                            var delta = npos - pos
                            var sqr_dst = length_sq_3d(delta)
                            if sqr_dst > sqr_radius:
                                continue

                            var dst = sqrt(sqr_dst)
                            var dir = SIMD[DType.float64, 4](0.0, 1.0, 0.0)
                            if dst > 1e-8:
                                dir = normalize_3d(delta)

                            var nd = self.densities[j][0]
                            var n_near = self.densities[j][1]
                            if nd <= 1e-8 or n_near <= 0.0:
                                continue

                            var neighbour_pressure = (nd - self.target_density) * self.pressure_multiplier
                            var neighbour_near_pressure = n_near * self.near_pressure_multiplier

                            var shared_pressure = 0.5 * (pressure + neighbour_pressure)
                            var shared_near = 0.5 * (near_pressure_val + neighbour_near_pressure)

                            var d_kernel = derivative_spiky_pow2_3d(dst, r)
                            var nd_kernel = derivative_spiky_pow3_3d(dst, r)

                            var scale1 = (d_kernel * shared_pressure) / max(nd, 1e-8)
                            var scale2 = (nd_kernel * shared_near) / max(n_near, 1e-8)
                            var force_scale = scale1 + scale2
                            
                            # FIXED: Apply force in the direction of neighbor (repulsive when pressure is high)
                            # This matches the Python implementation exactly
                            total_force = total_force + dir * force_scale

            var accel = total_force / density
            self.velocities[i] = self.velocities[i] + accel * dt

    fn _apply_viscosity(mut self, dt: Float64):
        var r = self.smoothing_radius
        var sqr_radius = r * r
        var viscosity_strength = self.viscosity_strength
        var delta_v = List[SIMD[DType.float64, 4]]()

        for _ in range(self.particle_count):
            delta_v.append(SIMD[DType.float64, 4](0.0, 0.0, 0.0))

        for i in range(self.particle_count):
            var pos = self.predicted_positions[i]
            var origin_cell = get_cell_3d(pos, r)
            var vi = self.velocities[i]
            var visc_force = SIMD[DType.float64, 4](0.0, 0.0, 0.0)

            # Check 27 neighboring cells
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        var neighbor_cell = SIMD[DType.int32, 4](origin_cell[0] + dx, origin_cell[1] + dy, origin_cell[2] + dz, 0)
                        var hash = self.spatial_hash.get_cell_hash(neighbor_cell)
                        var start = self.spatial_hash.get_start(hash)
                        var end = self.spatial_hash.get_end(hash)
                        
                        if start < 0:
                            continue
                        
                        for idx in range(start, end):
                            var j = self.spatial_hash.cell_entries[idx]
                            if j == i:
                                continue
                            var npos = self.predicted_positions[j]
                            var delta = npos - pos
                            var sqr_dst = length_sq_3d(delta)
                            if sqr_dst > sqr_radius:
                                continue

                            var dst = sqrt(sqr_dst)
                            var vj = self.velocities[j]
                            var dv = vj - vi
                            var w = smoothing_kernel_poly6_3d(dst, r)
                            visc_force = visc_force + dv * w

            delta_v[i] = visc_force * viscosity_strength * dt

        for i in range(self.particle_count):
            self.velocities[i] = self.velocities[i] + delta_v[i]

    fn step(mut self, frame_time: Float64):
        if self.iterations_per_frame <= 0:
            return

        self.simulation_time += frame_time

        if not self.scenario_active and self.scenario_type == 1:
            if self.simulation_time >= self.scenario_activation_time:
                self.scenario_active = True

        if self.scenario_active and self.has_scenario:
            var dt = (frame_time / Float64(self.iterations_per_frame)) * self.time_scale
            self.scenario.update(dt)

        self.collision_events = 0
        var dt = (frame_time / Float64(self.iterations_per_frame)) * self.time_scale
        var prediction_factor = 1.0 / 120.0

        for _ in range(self.iterations_per_frame):
            for i in range(self.particle_count):
                var pos = self.positions[i]
                var vel = self.velocities[i]
                var accel = self._external_forces(pos, vel)
                var dt_scalar = Float64(dt)
                vel = SIMD[DType.float64, 4](
                    vel[0] + accel[0] * dt_scalar,
                    vel[1] + accel[1] * dt_scalar,
                    vel[2] + accel[2] * dt_scalar
                )
                self.velocities[i] = vel
                var pf_scalar = Float64(prediction_factor)
                self.predicted_positions[i] = SIMD[DType.float64, 4](
                    pos[0] + vel[0] * pf_scalar,
                    pos[1] + vel[1] * pf_scalar,
                    pos[2] + vel[2] * pf_scalar
                )

            # densities (with spatial hashing)
            self._build_spatial_hash()
            self._update_densities()
            self._apply_pressure_forces(dt)
            self._apply_viscosity(dt)

            for i in range(self.particle_count):
                var pos = self.positions[i]
                var vel = self.velocities[i]
                var dt_scalar = Float64(dt)
                self.positions[i] = SIMD[DType.float64, 4](
                    pos[0] + vel[0] * dt_scalar,
                    pos[1] + vel[1] * dt_scalar,
                    pos[2] + vel[2] * dt_scalar
                )
                self._handle_collisions(i)


# -----------------------------
# Entry point
# -----------------------------

fn main() raises:
    # Use Python for binary I/O
    var py_sys = Python.import_module("sys")
    var py_struct = Python.import_module("struct")
    
    var cfg = SimConfig()
    var sim = FluidSim3D(cfg^)

    var target_fps: Float64 = 60.0
    var dt = 1.0 / target_fps
    var frame_idx = 0

    # Command loop
    while True:
        # Read 4 bytes command ID
        var cmd_bytes = py_sys.stdin.buffer.read(4)
        if len(cmd_bytes) < 4:
            break
            
        var cmd_id = py_struct.unpack("<I", cmd_bytes)[0]
        
        if cmd_id == 1: # STEP
            sim.step(dt)
            frame_idx += 1
            
            # Prepare header
            var scenario_active = 1 if sim.scenario_active else 0
            var whirlpool_radius = 3.0
            if sim.has_scenario:
                whirlpool_radius = sim.scenario.whirlpool_radius
                
            var header = py_struct.pack(
                "<qqqdddqd",
                frame_idx,
                sim.particle_count,
                sim.collision_events,
                sim.bounds_size[0],
                sim.bounds_size[1],
                sim.bounds_size[2],
                scenario_active,
                whirlpool_radius
            )
            py_sys.stdout.buffer.write(header)
            
            # Write positions and velocities directly from memory
            # We need to be careful here. Mojo List storage is not guaranteed to be contiguous 
            # in the way we might expect if we just grab a pointer, but for simple scalar types it often is.
            # However, `List` in Mojo is a dynamic array.
            # A safer way without unsafe pointer magic that might break is to pack them.
            # BUT, for performance, we want to write raw bytes.
            
            # Let's use a slightly slower but safer approach for now: pack into a large bytes object using python
            # Or better, construct a bytes object from the raw data if possible.
            # Since we can't easily cast Mojo List to Python bytes directly without unsafe,
            # we will iterate and pack. It's slower but safe.
            # OPTIMIZATION: In a real high-perf scenario, we'd use UnsafePointer to the underlying data
            # and write it out. Given the constraints, let's try to be reasonably efficient.
            
            var pos_data = Python.evaluate("bytearray()")
            var vel_data = Python.evaluate("bytearray()")
            
            # We can't easily bulk copy without unsafe. 
            # Let's use a python list and struct.pack_into? No, that's slow.
            # Let's try to construct a list of floats and pack them in python.
            
            var all_pos = Python.evaluate("[]")
            var all_vel = Python.evaluate("[]")
            
            for i in range(sim.particle_count):
                var p = sim.positions[i]
                var v = sim.velocities[i]
                all_pos.append(p[0])
                all_pos.append(p[1])
                all_pos.append(p[2])
                all_pos.append(0.0) # padding
                
                all_vel.append(v[0])
                all_vel.append(v[1])
                all_vel.append(v[2])
                all_vel.append(0.0) # padding
                
            # Pack in python using array module to avoid unpacking arguments issue
            var py_array = Python.import_module("array")
            
            # Create array('d', all_pos)
            var pos_arr = py_array.array("d", all_pos)
            var vel_arr = py_array.array("d", all_vel)
            
            var packed_pos = pos_arr.tobytes()
            var packed_vel = vel_arr.tobytes()
            
            py_sys.stdout.buffer.write(packed_pos)
            py_sys.stdout.buffer.write(packed_vel)
            py_sys.stdout.buffer.flush()
            
        elif cmd_id == 2: # RESIZE
            var payload = py_sys.stdin.buffer.read(24) # 3 doubles
            var dims = py_struct.unpack("<ddd", payload)
            
            # Convert Python float -> Python str -> Mojo String -> Mojo Float64
            # This is a workaround for direct casting issues
            var py_builtins = Python.import_module("builtins")
            
            var s0 = py_builtins.str(dims[0])
            var s1 = py_builtins.str(dims[1])
            var s2 = py_builtins.str(dims[2])
            
            sim.bounds_size[0] = atof(String(s0))
            sim.bounds_size[1] = atof(String(s1))
            sim.bounds_size[2] = atof(String(s2))
            
        elif cmd_id == 3: # QUIT
            break
