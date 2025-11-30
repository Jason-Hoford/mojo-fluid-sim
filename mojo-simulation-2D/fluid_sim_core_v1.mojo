from math import sqrt, floor, ceil, cos, sin
from random import random_float64, seed
from sys import argv
from collections import List

alias PI: Float64 = 3.14159265358979323846

# -----------------------------
# Configuration
# -----------------------------

struct SimConfig:
    var particle_count: Int
    var bounds_size: SIMD[DType.float64, 2]
    var time_scale: Float64
    var iterations_per_frame: Int
    var gravity: Float64
    var collision_damping: Float64
    var smoothing_radius: Float64
    var target_density: Float64
    var pressure_multiplier: Float64
    var near_pressure_multiplier: Float64
    var viscosity_strength: Float64
    var initial_velocity: SIMD[DType.float64, 2]
    var spawn_centre: SIMD[DType.float64, 2]
    var spawn_size: SIMD[DType.float64, 2]
    var jitter_strength: Float64
    var interaction_radius: Float64
    var interaction_strength: Float64
    var obstacle_size: SIMD[DType.float64, 2]
    var obstacle_centre: SIMD[DType.float64, 2]

    fn __init__(out self):
        self.particle_count = 1000
        self.bounds_size = SIMD[DType.float64, 2](8.0, 8.0)
        self.time_scale = 1.0
        self.iterations_per_frame = 4
        self.gravity = -9.0
        self.collision_damping = 0.97
        self.smoothing_radius = 0.12
        self.target_density = 12.0
        self.pressure_multiplier = 120.0
        self.near_pressure_multiplier = 20.0
        self.viscosity_strength = 0.2
        self.initial_velocity = SIMD[DType.float64, 2](0.0, 0.0)
        self.spawn_centre = SIMD[DType.float64, 2](0.0, 1.0)
        self.spawn_size = SIMD[DType.float64, 2](3.0, 1.2)
        self.jitter_strength = 0.02
        self.interaction_radius = 0.6
        self.interaction_strength = 15.0
        self.obstacle_size = SIMD[DType.float64, 2](2.0, 2.6)
        self.obstacle_centre = SIMD[DType.float64, 2](0.0, -2.0)


# -----------------------------
# Kernel functions
# -----------------------------

fn smoothing_kernel_poly6(dst: Float64, radius: Float64, poly6_scaling: Float64) -> Float64:
    if dst < radius:
        var v = radius * radius - dst * dst
        return v * v * v * poly6_scaling
    return 0.0


fn spiky_kernel_pow3(dst: Float64, radius: Float64, scale: Float64) -> Float64:
    if dst < radius:
        var v = radius - dst
        return v * v * v * scale
    return 0.0


fn spiky_kernel_pow2(dst: Float64, radius: Float64, scale: Float64) -> Float64:
    if dst < radius:
        var v = radius - dst
        return v * v * scale
    return 0.0


fn derivative_spiky_pow3(dst: Float64, radius: Float64, scale: Float64) -> Float64:
    if dst <= radius:
        var v = radius - dst
        return -v * v * scale
    return 0.0


fn derivative_spiky_pow2(dst: Float64, radius: Float64, scale: Float64) -> Float64:
    if dst <= radius:
        var v = radius - dst
        return -v * scale
    return 0.0


fn get_cell_2d(position: SIMD[DType.float64, 2], radius: Float64) -> SIMD[DType.int32, 2]:
    return SIMD[DType.int32, 2](
        Int32(floor(position[0] / radius)),
        Int32(floor(position[1] / radius))
    )


fn hash_cell_2d(cell: SIMD[DType.int32, 2]) -> Int:
    # Simple hash function for spatial hashing
    var h = Int(cell[0]) * 73856093 ^ Int(cell[1]) * 19349663
    return h


fn length_sq(v: SIMD[DType.float64, 2]) -> Float64:
    return v[0] * v[0] + v[1] * v[1]


# -----------------------------
# Spatial Hash Grid
# -----------------------------

struct SpatialHash:
    var table_size: Int
    var cell_start: List[Int]  # cell_start[hash] = start index in cell_entries
    var cell_end: List[Int]    # cell_end[hash] = end index in cell_entries
    var cell_entries: List[Int]  # particle indices sorted by cell
    var particle_cells: List[Int]  # particle_cells[i] = cell hash for particle i
    
    fn __init__(out self, capacity: Int):
        # Use a hash table size that's a power of 2 for fast modulo
        self.table_size = 4096  # Reasonable size for up to ~1000 particles
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
    
    fn build(mut self, positions: List[SIMD[DType.float64, 2]], radius: Float64):
        self.clear()
        var n = len(positions)
        
        # Step 1: Compute cell hash for each particle
        for i in range(n):
            var cell = get_cell_2d(positions[i], radius)
            var hash = hash_cell_2d(cell) & (self.table_size - 1)  # Fast modulo for power of 2
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
    
    fn get_cell_hash(self, cell: SIMD[DType.int32, 2]) -> Int:
        """Get hash for a cell."""
        return hash_cell_2d(cell) & (self.table_size - 1)
    
    fn get_start(self, hash: Int) -> Int:
        """Get start index for a hash."""
        return self.cell_start[hash]
    
    fn get_end(self, hash: Int) -> Int:
        """Get end index for a hash."""
        return self.cell_end[hash]


# -----------------------------
# Fluid simulation
# -----------------------------

struct FluidSim2D:
    var time_scale: Float64
    var iterations_per_frame: Int
    var gravity: Float64
    var collision_damping: Float64
    var smoothing_radius: Float64
    var target_density: Float64
    var pressure_multiplier: Float64
    var near_pressure_multiplier: Float64
    var viscosity_strength: Float64
    var bounds_size: SIMD[DType.float64, 2]
    var obstacle_size: SIMD[DType.float64, 2]
    var obstacle_centre: SIMD[DType.float64, 2]
    var interaction_radius: Float64
    var interaction_strength: Float64
    var mouse_world_pos: SIMD[DType.float64, 2]
    var mouse_interaction_strength: Float64
    var poly6_scaling_factor: Float64
    var spiky_pow3_scaling_factor: Float64
    var spiky_pow2_scaling_factor: Float64
    var spiky_pow3_derivative_scaling_factor: Float64
    var spiky_pow2_derivative_scaling_factor: Float64
    var particle_count: Int
    var initial_velocity: SIMD[DType.float64, 2]
    var spawn_centre: SIMD[DType.float64, 2]
    var spawn_size: SIMD[DType.float64, 2]
    var jitter_strength: Float64
    var positions: List[SIMD[DType.float64, 2]]
    var predicted_positions: List[SIMD[DType.float64, 2]]
    var velocities: List[SIMD[DType.float64, 2]]
    var densities: List[SIMD[DType.float64, 2]]
    var spatial_hash: SpatialHash
    var collision_events: Int

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
        self.obstacle_size = cfg.obstacle_size
        self.obstacle_centre = cfg.obstacle_centre
        self.interaction_radius = cfg.interaction_radius
        self.interaction_strength = cfg.interaction_strength
        self.mouse_world_pos = SIMD[DType.float64, 2](0.0, 0.0)
        self.mouse_interaction_strength = 0.0

        var r = cfg.smoothing_radius
        self.poly6_scaling_factor = 4.0 / (PI * (r ** 8))
        self.spiky_pow3_scaling_factor = 10.0 / (PI * (r ** 5))
        self.spiky_pow2_scaling_factor = 6.0 / (PI * (r ** 4))
        self.spiky_pow3_derivative_scaling_factor = 30.0 / ((r ** 5) * PI)
        self.spiky_pow2_derivative_scaling_factor = 12.0 / ((r ** 4) * PI)

        self.particle_count = cfg.particle_count
        self.initial_velocity = cfg.initial_velocity
        self.spawn_centre = cfg.spawn_centre
        self.spawn_size = cfg.spawn_size
        self.jitter_strength = cfg.jitter_strength

        self.positions = List[SIMD[DType.float64, 2]](capacity=cfg.particle_count)
        self.predicted_positions = List[SIMD[DType.float64, 2]](capacity=cfg.particle_count)
        self.velocities = List[SIMD[DType.float64, 2]](capacity=cfg.particle_count)
        self.densities = List[SIMD[DType.float64, 2]](capacity=cfg.particle_count)
        self.spatial_hash = SpatialHash(cfg.particle_count)
        self.collision_events = 0

        seed()
        var n = self.particle_count
        
        var sx = self.spawn_size[0]
        var sy = self.spawn_size[1]
        var aspect = sx / max(sy, 1e-6)
        var approx_rows = Int(sqrt(Float64(n) / max(aspect, 1e-3)))
        if approx_rows == 0:
            approx_rows = 1
        var approx_cols = Int(ceil(Float64(n) / Float64(approx_rows)))

        for i in range(n):
            var y = i // approx_cols
            var x = i % approx_cols
            
            var tx = Float64(x) / max(Float64(approx_cols - 1), 1.0)
            var ty = Float64(y) / max(Float64(approx_rows - 1), 1.0)
            
            var angle = random_float64() * 2.0 * PI
            var dir_x = cos(angle)
            var dir_y = sin(angle)
            var jitter_mag = (random_float64() - 0.5) * self.jitter_strength
            var jitter_x = dir_x * jitter_mag
            var jitter_y = dir_y * jitter_mag

            var px = (tx - 0.5) * sx + self.spawn_centre[0] + jitter_x
            var py = (ty - 0.5) * sy + self.spawn_centre[1] + jitter_y
            
            self.positions.append(SIMD[DType.float64, 2](px, py))
            self.predicted_positions.append(SIMD[DType.float64, 2](px, py))
            self.velocities.append(self.initial_velocity)
            self.densities.append(SIMD[DType.float64, 2](0.0, 0.0))

    fn _external_forces(self, pos: SIMD[DType.float64, 2], vel: SIMD[DType.float64, 2]) -> SIMD[DType.float64, 2]:
        var gravity_accel = SIMD[DType.float64, 2](0.0, self.gravity)
        var s = self.mouse_interaction_strength
        
        if abs(s) > 1e-5:
            var offset = self.mouse_world_pos - pos
            var sqr_dst = length_sq(offset)
            var r = self.interaction_radius
            
            if sqr_dst < r * r and sqr_dst > 1e-12:
                var dst = sqrt(sqr_dst)
                var edge_t = dst / r
                var centre_t = 1.0 - edge_t
                var dir_to_centre = offset / dst

                var gravity_weight = 1.0 - (centre_t * max(min(s / 10.0, 1.0), -1.0))
                var accel_gravity = gravity_accel * gravity_weight
                var accel_interact = dir_to_centre * centre_t * s
                var accel_damping = vel * (-centre_t)

                return accel_gravity + accel_interact + accel_damping
        
        return gravity_accel

    fn _handle_collisions(mut self, index: Int):
        var pos = self.positions[index]
        var vel = self.velocities[index]
        var collided = False

        var half_w = self.bounds_size[0] * 0.5
        var half_h = self.bounds_size[1] * 0.5

        if pos[0] <= -half_w or pos[0] >= half_w:
            pos[0] = max(min(pos[0], half_w), -half_w)
            vel[0] = -vel[0] * self.collision_damping
            collided = True
        
        if pos[1] <= -half_h or pos[1] >= half_h:
            pos[1] = max(min(pos[1], half_h), -half_h)
            vel[1] = -vel[1] * self.collision_damping
            collided = True

        var ox = self.obstacle_centre[0]
        var oy = self.obstacle_centre[1]
        var ohx = self.obstacle_size[0] * 0.5
        var ohy = self.obstacle_size[1] * 0.5

        var dx = abs(pos[0] - ox)
        var dy = abs(pos[1] - oy)
        if dx <= ohx + 0.5 and dy <= ohy + 0.5:
            var rel_x = pos[0] - ox
            var rel_y = pos[1] - oy
            var edge_x = ohx - abs(rel_x)
            var edge_y = ohy - abs(rel_y)

            if edge_x >= 0.0 and edge_y >= 0.0:
                if edge_x < edge_y:
                    pos[0] = ohx * (1.0 if rel_x >= 0.0 else -1.0) + ox
                    vel[0] = -vel[0] * self.collision_damping
                else:
                    pos[1] = ohy * (1.0 if rel_y >= 0.0 else -1.0) + oy
                    vel[1] = -vel[1] * self.collision_damping
                collided = True

        self.positions[index] = pos
        self.velocities[index] = vel
        if collided:
            self.collision_events += 1

    fn _update_densities(mut self):
        var r = self.smoothing_radius
        var sqr_radius = r * r

        for i in range(self.particle_count):
            var pos = self.predicted_positions[i]
            var origin_cell = get_cell_2d(pos, r)
            var density: Float64 = 0.0
            var near_density: Float64 = 0.0

            # Check 9 neighboring cells (3x3 grid)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    var neighbor_cell = SIMD[DType.int32, 2](origin_cell[0] + dx, origin_cell[1] + dy)
                    var hash = self.spatial_hash.get_cell_hash(neighbor_cell)
                    var start = self.spatial_hash.get_start(hash)
                    var end = self.spatial_hash.get_end(hash)
                    
                    if start < 0:
                        continue
                    
                    for idx in range(start, end):
                        var j = self.spatial_hash.cell_entries[idx]
                        var npos = self.predicted_positions[j]
                        var delta = npos - pos
                        var sqr_dst = length_sq(delta)
                        if sqr_dst > sqr_radius:
                            continue

                        var dst = sqrt(sqr_dst)
                        density += spiky_kernel_pow2(dst, r, self.spiky_pow2_scaling_factor)
                        near_density += spiky_kernel_pow3(dst, r, self.spiky_pow3_scaling_factor)

            self.densities[i] = SIMD[DType.float64, 2](density, near_density)

    fn _apply_pressure_forces(mut self, dt: Float64):
        var r = self.smoothing_radius
        var sqr_radius = r * r

        for i in range(self.particle_count):
            var pos = self.predicted_positions[i]
            var origin_cell = get_cell_2d(pos, r)
            var density = self.densities[i][0]
            var near_density = self.densities[i][1]
            if density <= 1e-8:
                continue

            var pressure = (density - self.target_density) * self.pressure_multiplier
            var near_pressure_val = near_density * self.near_pressure_multiplier

            var total_force = SIMD[DType.float64, 2](0.0, 0.0)

            # Check 9 neighboring cells
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    var neighbor_cell = SIMD[DType.int32, 2](origin_cell[0] + dx, origin_cell[1] + dy)
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
                        var sqr_dst = length_sq(delta)
                        if sqr_dst > sqr_radius:
                            continue

                        var dst = sqrt(sqr_dst)
                        var dir = SIMD[DType.float64, 2](0.0, 1.0)
                        if dst > 1e-8:
                            dir = delta / dst

                        var nd = self.densities[j][0]
                        var n_near = self.densities[j][1]
                        if nd <= 1e-8 or n_near <= 0.0:
                            continue

                        var neighbour_pressure = (nd - self.target_density) * self.pressure_multiplier
                        var neighbour_near_pressure = n_near * self.near_pressure_multiplier

                        var shared_pressure = 0.5 * (pressure + neighbour_pressure)
                        var shared_near = 0.5 * (near_pressure_val + neighbour_near_pressure)

                        var d_kernel = derivative_spiky_pow2(dst, r, self.spiky_pow2_derivative_scaling_factor)
                        var nd_kernel = derivative_spiky_pow3(dst, r, self.spiky_pow3_derivative_scaling_factor)

                        var scale1 = (d_kernel * shared_pressure) / max(nd, 1e-8)
                        var scale2 = (nd_kernel * shared_near) / max(n_near, 1e-8)
                        var s_total = scale1 + scale2
                        total_force = total_force + dir * s_total

            var accel = total_force / density
            self.velocities[i] = self.velocities[i] + accel * dt

    fn _apply_viscosity(mut self, dt: Float64):
        var r = self.smoothing_radius
        var sqr_radius = r * r
        var viscosity_strength = self.viscosity_strength
        var delta_v = List[SIMD[DType.float64, 2]](capacity=self.particle_count)
        
        for _ in range(self.particle_count):
            delta_v.append(SIMD[DType.float64, 2](0.0, 0.0))

        for i in range(self.particle_count):
            var pos = self.predicted_positions[i]
            var origin_cell = get_cell_2d(pos, r)
            var vi = self.velocities[i]
            var visc_force = SIMD[DType.float64, 2](0.0, 0.0)

            # Check 9 neighboring cells
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    var neighbor_cell = SIMD[DType.int32, 2](origin_cell[0] + dx, origin_cell[1] + dy)
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
                        var sqr_dst = length_sq(delta)
                        if sqr_dst > sqr_radius:
                            continue

                        var dst = sqrt(sqr_dst)
                        var vj = self.velocities[j]
                        var dv = vj - vi
                        var w = smoothing_kernel_poly6(dst, r, self.poly6_scaling_factor)
                        visc_force = visc_force + dv * w

            delta_v[i] = visc_force * viscosity_strength * dt

        for i in range(self.particle_count):
            self.velocities[i] = self.velocities[i] + delta_v[i]

    fn step(mut self, frame_time: Float64):
        if self.iterations_per_frame <= 0:
            return
        self.collision_events = 0

        var dt = (frame_time / Float64(self.iterations_per_frame)) * self.time_scale
        var prediction_factor = 1.0 / 120.0

        for _ in range(self.iterations_per_frame):
            # external forces + prediction
            for i in range(self.particle_count):
                var pos = self.positions[i]
                var vel = self.velocities[i]
                var accel = self._external_forces(pos, vel)
                vel = vel + accel * dt
                self.velocities[i] = vel
                self.predicted_positions[i] = pos + vel * prediction_factor

            # Build spatial hash for neighbor queries
            self.spatial_hash.build(self.predicted_positions, self.smoothing_radius)
            
            # densities (with spatial hashing)
            self._update_densities()
            # pressure
            self._apply_pressure_forces(dt)
            # viscosity
            self._apply_viscosity(dt)

            # integrate + collisions
            for i in range(self.particle_count):
                var pos = self.positions[i]
                var vel = self.velocities[i]
                self.positions[i] = pos + vel * dt
                self._handle_collisions(i)


# -----------------------------
# Entry point
# -----------------------------

fn main() raises:
    var args = argv()
    var max_frames = 10000
    
    if len(args) > 1:
        try:
            max_frames = atol(args[1])
        except:
            pass

    var cfg = SimConfig()
    var sim = FluidSim2D(cfg^)

    var target_fps: Float64 = 60.0
    var dt = 1.0 / target_fps

    for frame in range(max_frames):
        sim.step(dt)

        var w = sim.bounds_size[0]
        var h = sim.bounds_size[1]
        var n = sim.particle_count

        var ox = sim.obstacle_centre[0]
        var oy = sim.obstacle_centre[1]
        var ow = sim.obstacle_size[0]
        var oh = sim.obstacle_size[1]
        
        print("F", frame, sim.collision_events, w, h, ox, oy, ow, oh, n, end=" ")
        
        for i in range(n):
            var pos = sim.positions[i]
            var vel = sim.velocities[i]
            print(pos[0], pos[1], vel[0], vel[1], end=" " if i < n - 1 else "")
        
        print()