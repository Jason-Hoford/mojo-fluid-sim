import taichi as ti
import math
import time

# Initialize Taichi on GPU
try:
    ti.init(arch=ti.cuda, device_memory_fraction=0.8)
except:
    print("CUDA not found, falling back to Vulkan or CPU...")
    ti.init(arch=ti.gpu, device_memory_fraction=0.8)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
NUM_PARTICLES = 8000
BOUNDS = [2.3, 8.0, 2.3]  # Initial bounds (can be resized)

# Physics Constants (Matched to Mojo)
SMOOTHING_RADIUS = 0.2
PARTICLE_RADIUS = 0.05
TARGET_DENSITY = 12.0
PRESSURE_MULTIPLIER = 120.0
NEAR_PRESSURE_MULTIPLIER = 20.0
VISCOSITY_STRENGTH = 0.2
COLLISION_DAMPING = 0.95
GRAVITY = -10.0
DT = 0.002
SUBSTEPS = 10

# Whirlpool Scenario Settings
SCENARIO_ACTIVATION_TIME = 2000  # seconds
WHIRLPOOL_CENTER = [0.0, -1.0, 0.0]
WHIRLPOOL_RADIUS = 3.0
WHIRLPOOL_STRENGTH = 15.0
WHIRLPOOL_SINK_STRENGTH = 8.0
WHIRLPOOL_ROTATION_SPEED = 2.0

# Grid for Spatial Hashing
GRID_CELL_SIZE = SMOOTHING_RADIUS


def compute_grid_res(bounds):
    return (
        int(math.ceil(bounds[0] / GRID_CELL_SIZE)) + 1,
        int(math.ceil(bounds[1] / GRID_CELL_SIZE)) + 1,
        int(math.ceil(bounds[2] / GRID_CELL_SIZE)) + 1
    )


GRID_RES = compute_grid_res(BOUNDS)
print(f"Initial Grid Resolution: {GRID_RES}")

# ----------------------------------------------------------------------------
# Data Fields
# ----------------------------------------------------------------------------
pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
dens = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
pressure = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)
colors = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)

# Spatial Grid (allocate max size)
MAX_GRID_RES = (100, 100, 100)
grid_num_particles = ti.field(dtype=ti.i32, shape=MAX_GRID_RES)
grid2particles = ti.field(dtype=ti.i32, shape=(MAX_GRID_RES[0], MAX_GRID_RES[1], MAX_GRID_RES[2], 100))

# Dynamic parameters
bounds = ti.Vector.field(3, dtype=ti.f32, shape=())
grid_res = ti.Vector.field(3, dtype=ti.i32, shape=())
whirlpool_active = ti.field(dtype=ti.i32, shape=())
whirlpool_time = ti.field(dtype=ti.f32, shape=())
simulation_time = ti.field(dtype=ti.f32, shape=())


# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------
@ti.func
def get_cell(p):
    cell = (p / GRID_CELL_SIZE).cast(int)
    res = grid_res[None]
    return ti.Vector([
        ti.max(0, ti.min(cell[0], res[0] - 1)),
        ti.max(0, ti.min(cell[1], res[1] - 1)),
        ti.max(0, ti.min(cell[2], res[2] - 1))
    ])


@ti.func
def is_in_grid(c):
    res = grid_res[None]
    return 0 <= c[0] < res[0] and 0 <= c[1] < res[1] and 0 <= c[2] < res[2]


@ti.func
def cross_product(a, b):
    return ti.Vector([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])


@ti.func
def whirlpool_force(pos_i, vel_i):
    """Calculate whirlpool force matching Mojo implementation - OPTIMIZED."""
    force = ti.Vector([0.0, 0.0, 0.0])

    # Single return at end - required for Taichi GPU kernels
    if whirlpool_active[None] == 1:
        center = ti.Vector(WHIRLPOOL_CENTER)
        to_particle = pos_i - center
        dist_sq = to_particle.norm_sqr()

        # Check if within valid radius range
        radius_sq = WHIRLPOOL_RADIUS * WHIRLPOOL_RADIUS
        if dist_sq <= radius_sq and dist_sq >= 0.01:  # 0.1^2 = 0.01
            dist = ti.sqrt(dist_sq)
            to_particle_norm = to_particle / dist  # Faster than .normalized()

            up = ti.Vector([0.0, 1.0, 0.0])
            tangential = cross_product(up, to_particle_norm)
            tangential_norm = tangential.norm()

            # Avoid division by zero
            if tangential_norm > 1e-8:
                tangential = tangential / tangential_norm

                # Rotate tangential vector over time
                wt = whirlpool_time[None]
                cos_t = ti.cos(wt)
                sin_t = ti.sin(wt)
                tangential_rotated = ti.Vector([
                    tangential[0] * cos_t - tangential[2] * sin_t,
                    tangential[1],
                    tangential[0] * sin_t + tangential[2] * cos_t
                ])

                force_factor = 1.0 - (dist / WHIRLPOOL_RADIUS)

                tangential_force = tangential_rotated * (WHIRLPOOL_STRENGTH * force_factor)
                radial_inward = to_particle_norm * (-WHIRLPOOL_SINK_STRENGTH * force_factor)
                downward_force = ti.Vector([0.0, -WHIRLPOOL_SINK_STRENGTH * force_factor * 0.5, 0.0])

                force = tangential_force + radial_inward + downward_force

    return force


@ti.kernel
def init_particles():
    dim_count = int(ti.ceil(NUM_PARTICLES ** (1 / 3)))
    spacing = 0.2
    b = bounds[None]
    offset = ti.Vector([b[0] / 2 - dim_count * spacing / 2, b[1] / 2, b[2] / 2 - dim_count * spacing / 2])

    for i in range(NUM_PARTICLES):
        x = i % dim_count
        y = (i // dim_count) % dim_count
        z = i // (dim_count * dim_count)

        jitter = ti.Vector([ti.random(), ti.random(), ti.random()]) * 0.01
        pos[i] = offset + ti.Vector([x, y, z]) * spacing + jitter

        # Ensure inside bounds
        for k in ti.static(range(3)):
            pos[i][k] = ti.max(0.1, ti.min(pos[i][k], b[k] - 0.1))

        vel[i] = ti.Vector([0.0, 0.0, 0.0])
        colors[i] = ti.Vector([0.2, 0.5, 1.0])


@ti.kernel
def update_grid():
    res = grid_res[None]
    for i, j, k in ti.ndrange(res[0], res[1], res[2]):
        grid_num_particles[i, j, k] = 0

    for i in range(NUM_PARTICLES):
        cell = get_cell(pos[i])
        idx = ti.atomic_add(grid_num_particles[cell], 1)
        if idx < 100:
            grid2particles[cell, idx] = i


@ti.func
def spiky_kernel_pow2(dst, r):
    result = 0.0
    if dst < r:
        v = r - dst
        result = v * v
    return result


@ti.func
def spiky_kernel_pow3(dst, r):
    result = 0.0
    if dst < r:
        v = r - dst
        result = v * v * v
    return result


@ti.func
def smoothing_kernel_poly6(dst, r):
    result = 0.0
    if dst < r:
        v = r * r - dst * dst
        result = v * v * v
    return result


@ti.kernel
def compute_densities():
    scale_pow2 = 15.0 / (2.0 * math.pi * (SMOOTHING_RADIUS ** 5))
    scale_pow3 = 15.0 / (math.pi * (SMOOTHING_RADIUS ** 6))

    for i in range(NUM_PARTICLES):
        p = pos[i]
        cell = get_cell(p)

        d = 0.0
        dn = 0.0

        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    off = ti.Vector([dx, dy, dz])
                    neighbor_cell = cell + off
                    if is_in_grid(neighbor_cell):
                        count = grid_num_particles[neighbor_cell]
                        if count > 100:
                            count = 100

                        for k in range(count):
                            j = grid2particles[neighbor_cell, k]

                            r_vec = pos[j] - p
                            dst_sq = r_vec.norm_sqr()
                            if dst_sq < SMOOTHING_RADIUS ** 2:
                                dst = ti.sqrt(dst_sq)
                                d += spiky_kernel_pow2(dst, SMOOTHING_RADIUS) * scale_pow2
                                dn += spiky_kernel_pow3(dst, SMOOTHING_RADIUS) * scale_pow3

        dens[i] = ti.Vector([d, dn])
        pressure[i][0] = (d - TARGET_DENSITY) * PRESSURE_MULTIPLIER
        pressure[i][1] = dn * NEAR_PRESSURE_MULTIPLIER


@ti.kernel
def compute_forces():
    scale_poly6 = 315.0 / (64.0 * math.pi * (SMOOTHING_RADIUS ** 9))
    scale_d_pow2 = 15.0 / ((SMOOTHING_RADIUS ** 5) * math.pi)
    scale_d_pow3 = 45.0 / ((SMOOTHING_RADIUS ** 6) * math.pi)

    for i in range(NUM_PARTICLES):
        p = pos[i]
        cell = get_cell(p)

        pressure_force = ti.Vector([0.0, 0.0, 0.0])
        viscosity_force = ti.Vector([0.0, 0.0, 0.0])

        d_i = dens[i][0]
        dn_i = dens[i][1]
        p_i = pressure[i][0]
        pn_i = pressure[i][1]

        if d_i > 1e-8:
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        off = ti.Vector([dx, dy, dz])
                        neighbor_cell = cell + off
                        if is_in_grid(neighbor_cell):
                            count = grid_num_particles[neighbor_cell]
                            if count > 100:
                                count = 100

                            for k in range(count):
                                j = grid2particles[neighbor_cell, k]
                                if i == j:
                                    continue

                                r_vec = pos[j] - p
                                dst_sq = r_vec.norm_sqr()

                                if dst_sq < SMOOTHING_RADIUS ** 2 and dst_sq > 1e-10:
                                    dst = ti.sqrt(dst_sq)
                                    dir = r_vec / dst

                                    d_j = dens[j][0]
                                    dn_j = dens[j][1]
                                    p_j = pressure[j][0]
                                    pn_j = pressure[j][1]

                                    if d_j > 1e-8:
                                        shared_p = (p_i + p_j) * 0.5
                                        shared_pn = (pn_i + pn_j) * 0.5

                                        v_pow2 = SMOOTHING_RADIUS - dst
                                        v_pow3 = SMOOTHING_RADIUS - dst
                                        grad_k = -v_pow2 * scale_d_pow2
                                        grad_kn = -v_pow3 * v_pow3 * scale_d_pow3

                                        term1 = shared_p * grad_k / d_j
                                        term2 = 0.0
                                        if dn_j > 1e-8:
                                            term2 = shared_pn * grad_kn / dn_j

                                        pressure_force += dir * (term1 + term2)

                                        v_rel = vel[j] - vel[i]
                                        w = smoothing_kernel_poly6(dst, SMOOTHING_RADIUS) * scale_poly6
                                        viscosity_force += v_rel * w

            pressure_accel = pressure_force / d_i
            visc_accel = viscosity_force * VISCOSITY_STRENGTH
            gravity_accel = ti.Vector([0.0, GRAVITY, 0.0])

            # Add whirlpool force
            whirlpool_accel = whirlpool_force(p, vel[i])

            accel = pressure_accel + visc_accel + gravity_accel + whirlpool_accel
            vel[i] += accel * DT


@ti.kernel
def integrate():
    epsilon = 0.05
    b = bounds[None]

    for i in range(NUM_PARTICLES):
        pos[i] += vel[i] * DT

        p = pos[i]
        v = vel[i]

        # Dynamic bounds collision
        for k in ti.static(range(3)):
            if p[k] < epsilon:
                p[k] = epsilon
                v[k] = -v[k] * COLLISION_DAMPING
            elif p[k] > b[k] - epsilon:
                p[k] = b[k] - epsilon
                v[k] = -v[k] * COLLISION_DAMPING

        pos[i] = p
        vel[i] = v

        speed = v.norm()
        colors[i] = ti.Vector([ti.min(1.0, speed * 0.1), 0.5, ti.max(0.0, 1.0 - speed * 0.1)])


@ti.kernel
def update_whirlpool(dt: ti.f32):
    if whirlpool_active[None] == 1:
        whirlpool_time[None] += dt * WHIRLPOOL_ROTATION_SPEED


# ----------------------------------------------------------------------------
# Helper functions for dynamic bounds
# ----------------------------------------------------------------------------
def set_bounds(new_bounds):
    """Update simulation bounds dynamically."""
    global BOUNDS
    BOUNDS = list(new_bounds)
    new_grid_res = compute_grid_res(BOUNDS)

    # Update Taichi fields
    bounds[None] = ti.Vector([float(BOUNDS[0]), float(BOUNDS[1]), float(BOUNDS[2])])
    grid_res[None] = ti.Vector([new_grid_res[0], new_grid_res[1], new_grid_res[2]])

    print(
        f"Bounds updated to: X={BOUNDS[0]:.1f}, Y={BOUNDS[1]:.1f}, Z={BOUNDS[2]:.1f}, Grid Resolution: {new_grid_res}")
    return new_grid_res


def create_box_visualization(b):
    """Create box vertices and indices for given bounds."""
    box_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
    box_indices = ti.field(dtype=ti.i32, shape=24)

    # FIXED: Use parameter b instead of global BOUNDS
    box_vertices[0] = ti.Vector([0.0, 0.0, 0.0])
    box_vertices[1] = ti.Vector([b[0], 0.0, 0.0])
    box_vertices[2] = ti.Vector([b[0], b[1], 0.0])
    box_vertices[3] = ti.Vector([0.0, b[1], 0.0])
    box_vertices[4] = ti.Vector([0.0, 0.0, b[2]])
    box_vertices[5] = ti.Vector([b[0], 0.0, b[2]])
    box_vertices[6] = ti.Vector([b[0], b[1], b[2]])
    box_vertices[7] = ti.Vector([0.0, b[1], b[2]])

    indices = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
    for i in range(24):
        box_indices[i] = indices[i]

    return box_vertices, box_indices


# ----------------------------------------------------------------------------
# Main Loop
# ----------------------------------------------------------------------------
def main():
    print("Initializing particles...")

    # Initialize dynamic parameters
    bounds[None] = ti.Vector(BOUNDS)
    grid_res[None] = ti.Vector(GRID_RES)
    whirlpool_active[None] = 0
    whirlpool_time[None] = 0.0
    simulation_time[None] = 0.0

    init_particles()
    print("Particles initialized.")

    print("\n=== CONTROLS ===")
    print("Right Mouse Button + Drag: Rotate camera")
    print("W/A/S/D: Move camera forward/left/backward/right (hold RMB)")
    print("Q/E: Move camera down/up")
    print("Mouse Wheel: Zoom in/out")
    print("SPACE: Pause/Resume simulation")
    print("\n--- Boundary Controls (All Axes) ---")
    print("N: Decrease all boundaries uniformly")
    print("M: Increase all boundaries uniformly")
    print("\n--- Individual Axis Controls ---")
    print("Z/X: Decrease/Increase X-axis boundary")
    print("C/V: Decrease/Increase Y-axis boundary")
    print("B/G: Decrease/Increase Z-axis boundary")
    print("\nH: Toggle whirlpool scenario (H for Hurricane)")
    print("ESC: Exit")
    print("================\n")

    window = ti.ui.Window("Taichi GPU Fluid Sim - Whirlpool & Dynamic Bounds", (1024, 768), vsync=True)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    camera.position(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] * 2.0)
    camera.lookat(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] / 2)
    camera.up(0, 1, 0)

    box_vertices, box_indices = create_box_visualization(BOUNDS)

    print("Starting simulation loop...")
    frame_count = 0
    last_time = time.time()
    paused = False
    show_whirlpool_marker = False
    needs_box_update = False  # Flag to track if box needs updating

    while window.running:
        # Handle keyboard input
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == ti.ui.SPACE:
                paused = not paused
                print(f"Simulation {'PAUSED' if paused else 'RESUMED'}")
            elif window.event.key == 'h':
                # Toggle whirlpool
                current = whirlpool_active[None]
                whirlpool_active[None] = 1 - current
                show_whirlpool_marker = (whirlpool_active[None] == 1)
                print(f"Whirlpool {'ACTIVATED' if show_whirlpool_marker else 'DEACTIVATED'}")

            # Uniform boundary changes
            elif window.event.key == 'n':
                # Decrease all bounds
                new_bounds = [max(2.0, b * 0.9) for b in BOUNDS]
                set_bounds(new_bounds)
                needs_box_update = True
                camera.lookat(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] / 2)
            elif window.event.key == 'm':
                # Increase all bounds
                new_bounds = [min(20.0, b * 1.1) for b in BOUNDS]
                set_bounds(new_bounds)
                needs_box_update = True
                camera.lookat(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] / 2)

            # Individual axis controls
            elif window.event.key == 'z':
                # Decrease X-axis
                new_bounds = BOUNDS.copy()
                new_bounds[0] = max(2.0, new_bounds[0] * 0.9)
                set_bounds(new_bounds)
                needs_box_update = True
                camera.lookat(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] / 2)
            elif window.event.key == 'x':
                # Increase X-axis
                new_bounds = BOUNDS.copy()
                new_bounds[0] = min(20.0, new_bounds[0] * 1.1)
                set_bounds(new_bounds)
                needs_box_update = True
                camera.lookat(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] / 2)

            elif window.event.key == 'c':
                # Decrease Y-axis
                new_bounds = BOUNDS.copy()
                new_bounds[1] = max(2.0, new_bounds[1] * 0.9)
                set_bounds(new_bounds)
                needs_box_update = True
                camera.lookat(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] / 2)
            elif window.event.key == 'v':
                # Increase Y-axis
                new_bounds = BOUNDS.copy()
                new_bounds[1] = min(20.0, new_bounds[1] * 1.1)
                set_bounds(new_bounds)
                needs_box_update = True
                camera.lookat(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] / 2)

            elif window.event.key == 'b':
                # Decrease Z-axis
                new_bounds = BOUNDS.copy()
                new_bounds[2] = max(2.0, new_bounds[2] * 0.9)
                set_bounds(new_bounds)
                needs_box_update = True
                camera.lookat(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] / 2)
            elif window.event.key == 'g':
                # Increase Z-axis
                new_bounds = BOUNDS.copy()
                new_bounds[2] = min(20.0, new_bounds[2] * 1.1)
                set_bounds(new_bounds)
                needs_box_update = True
                camera.lookat(BOUNDS[0] / 2, BOUNDS[1] / 2, BOUNDS[2] / 2)

        # Recreate box if bounds changed
        if needs_box_update:
            box_vertices, box_indices = create_box_visualization(BOUNDS)
            needs_box_update = False

        # Update simulation
        if not paused:
            for s in range(SUBSTEPS):
                update_grid()
                compute_densities()
                compute_forces()
                update_whirlpool(DT)
                integrate()

            # Update simulation time
            simulation_time[None] += (1.0 / 60.0)

            # Auto-activate whirlpool after delay (like Mojo)
            if whirlpool_active[None] == 0 and simulation_time[None] >= SCENARIO_ACTIVATION_TIME:
                whirlpool_active[None] = 1
                show_whirlpool_marker = True
                print("Whirlpool AUTO-ACTIVATED at t={:.1f}s".format(simulation_time[None]))

        # Camera controls
        camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # Lighting
        scene.point_light(pos=(BOUNDS[0] / 2, BOUNDS[1] * 2, BOUNDS[2] / 2), color=(1, 1, 1))
        scene.ambient_light((0.2, 0.2, 0.2))

        # Render particles
        scene.particles(pos, radius=PARTICLE_RADIUS, per_vertex_color=colors)

        # Render bounds box
        scene.lines(box_vertices, indices=box_indices, color=(1, 1, 1), width=2.0)

        # Render whirlpool marker
        if show_whirlpool_marker:
            # Create whirlpool visualization (circle at center)
            whirlpool_verts = ti.Vector.field(3, dtype=ti.f32, shape=32)
            whirlpool_indices = ti.field(dtype=ti.i32, shape=64)
            for i in range(32):
                angle = (i / 32.0) * 2 * math.pi
                whirlpool_verts[i] = ti.Vector([
                    WHIRLPOOL_CENTER[0] + WHIRLPOOL_RADIUS * math.cos(angle),
                    WHIRLPOOL_CENTER[1],
                    WHIRLPOOL_CENTER[2] + WHIRLPOOL_RADIUS * math.sin(angle)
                ])
                whirlpool_indices[i * 2] = i
                whirlpool_indices[i * 2 + 1] = (i + 1) % 32

            scene.lines(whirlpool_verts, indices=whirlpool_indices, color=(1, 0, 0), width=3.0)

        canvas.scene(scene)
        window.show()

        frame_count += 1
        if frame_count % 60 == 0:
            curr_time = time.time()
            fps = 60.0 / (curr_time - last_time)
            status = '[PAUSED]' if paused else ''
            whirlpool_status = '[WHIRLPOOL ON]' if show_whirlpool_marker else ''
            print(
                f"FPS: {fps:.1f} {status} {whirlpool_status} Bounds: X={BOUNDS[0]:.1f} Y={BOUNDS[1]:.1f} Z={BOUNDS[2]:.1f}")
            last_time = curr_time


if __name__ == "__main__":
    main()