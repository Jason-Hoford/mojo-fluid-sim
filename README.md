# 2D Fluid Simulation: Python & Mojo Implementation

A high-performance 2D particle-based fluid simulator implemented in both **pure Python** and **Mojo** (with Python rendering). This project demonstrates the performance difference between interpreted Python and compiled Mojo code, while maintaining identical simulation physics.

| ![Python Version](Assets/python_fluid_sim.gif) | ![Mojo Version](Assets/mojo_fluid_sim.gif) |
|:----------------------------------------------:|:------------------------------------------:|
|           **Python Implementation**            |          **Mojo Implementation**           |

> **Inspired by**: [Sebastian Lague's Fluid Simulation Episode 01](https://www.youtube.com/watch?v=rSKMYc1CQHE)  
> Special thanks to Sebastian Lague for the excellent tutorial and Unity implementation that served as the foundation for this project.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Core Functions & Components](#-core-functions--components)
- [Performance Comparison](#-performance-comparison)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Further Reading](#-further-reading)

---

## 🎯 Overview

This project implements a **Smoothed Particle Hydrodynamics (SPH)**-based 2D fluid simulation with two implementations:

1. **Pure Python Version** (`fluid_sim_2d.py`): A complete, self-contained implementation using pygame for visualization. Great for learning and understanding the algorithm.

2. **Mojo + Python Hybrid** (`mojo-simulation/`): A high-performance implementation where:
   - **Mojo** (`fluid_sim_core.mojo`) handles all computation-intensive simulation logic
   - **Python** (`fluid_render_client.py`) handles rendering and visualization
   - Communication happens via **stdin/stdout pipe**, keeping concerns separated

The Mojo version demonstrates significant performance improvements while maintaining identical physics behavior.

---

## 🏗️ Architecture

### How Mojo and Python Work Together

The hybrid implementation uses a **producer-consumer pattern** via Unix pipes:

```
┌─────────────────────┐         stdout         ┌──────────────────────┐
│  fluid_sim_core.mojo│ ────────────────────>  │ fluid_render_client  │
│  (Simulation Logic) │                        │    (Rendering)       │
│                     │ <────────────────────  │                      │
└─────────────────────┘         stdin          └──────────────────────┘
```

#### Data Flow

1. **Mojo Simulation** (`fluid_sim_core.mojo`):
   - Runs the physics simulation at 60 FPS
   - Calculates particle positions, velocities, densities, and collisions
   - Outputs one line per frame to **stdout** with format:
     ```
     F <frame> <collisions> <bounds_w> <bounds_h> <obstacle_x> <obstacle_y> <obstacle_w> <obstacle_h> <particle_count> x0 y0 vx0 vy0 x1 y1 vx1 vy1 ...
     ```

2. **Python Renderer** (`fluid_render_client.py`):
   - Reads frame data from **stdin** (blocking read)
   - Parses particle positions and velocities
   - Renders using pygame:
     - Draws bounds box
     - Draws obstacle rectangle
     - Draws particles with speed-based color gradient
     - Displays FPS and collision graphs (toggle with `G` key)

3. **Communication Protocol**:
   - **One-way**: Mojo → Python (simulation → rendering)
   - **Synchronous**: Python blocks waiting for each frame
   - **Text-based**: Space-separated values for easy parsing
   - **No shared memory**: Complete process isolation

#### Why This Architecture?

- **Separation of Concerns**: Simulation logic (Mojo) is completely independent from rendering (Python)
- **Performance**: Mojo's compiled code handles the O(n²) neighbor searches efficiently
- **Flexibility**: Easy to swap renderers (could use OpenGL, WebGL, etc.)
- **Debugging**: Can run Mojo standalone and inspect output, or run Python with test data
- **Portability**: Standard Unix pipes work on Linux, macOS, and Windows (WSL)

---

## ✨ Features

### Simulation Features

- **SPH Fluid Dynamics**:
  - Density calculation using Spiky kernels (pow2 and pow3)
  - Pressure forces based on density deviation from target
  - Viscosity forces for smooth particle interactions
  - Spatial neighbor search (brute force in current implementation)

- **Physics**:
  - Gravity
  - Collision detection with bounds and obstacles
  - Collision damping
  - Mouse interaction (attract/repel particles)

- **Visualization**:
  - Speed-based color gradient (blue → cyan → yellow → white)
  - Real-time FPS and collision rate graphs
  - Configurable particle count, bounds size, and zoom

### Controls

#### Basic Controls
- **Space**: Pause/Resume simulation
- **Right Arrow**: Single-step one frame (when paused)
- **R**: Reset to initial spawn state
- **Esc**: Quit

#### Zoom Controls
- **`+` or `=`**: Zoom in (increases zoom by 10%, maximum 10x)
- **`-` or `_`**: Zoom out (decreases zoom by 10%, minimum 0.2x)
- Zoom affects the visualization scale, allowing you to see more detail (zoom in) or a wider view (zoom out)
- The zoom factor is applied to both particle rendering and world-to-screen coordinate conversion

#### Visualization Controls
- **`G`**: Toggle FPS and collision graphs on/off
  - When enabled, displays two real-time graphs in the top-right corner:
    - **FPS Graph** (top): Shows frames per second over time
      - X-axis: Time (frame index)
      - Y-axis: FPS value
      - Helps monitor simulation performance
    - **Collisions/s Graph** (bottom): Shows collision rate per second over time
      - X-axis: Time (frame index)
      - Y-axis: Collisions per second
      - Useful for understanding simulation activity and detecting performance bottlenecks
  - Both graphs include:
    - Grid lines for easy reading
    - Numeric labels on both X and Y axes
    - Real-time updates as the simulation runs

#### Mouse Interaction
- **LMB (Left Mouse Button)**: Attract particles toward mouse cursor
- **RMB (Right Mouse Button)**: Repel particles from mouse cursor
- Interaction strength and radius are configurable in `SimConfig`

---

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **pygame** (`pip install pygame`)
- **Mojo** (for the hybrid version) - [Install Mojo](https://docs.modular.com/mojo/manual/get-started/)

### Setup

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For Mojo version** (optional):
   - **Windows Users**: See the detailed tutorial in `mojo-simulation/TotrialWindowsMoJo.md` for step-by-step instructions on installing Mojo using WSL and pixi
   - **Linux/macOS Users**: Install Mojo following the [official guide](https://docs.modular.com/mojo/manual/get-started/)
   - Ensure `mojo` is in your PATH

### Windows Installation Guide

If you're on Windows and want to run the Mojo version, we've included a comprehensive tutorial that covers:

- Installing WSL (Windows Subsystem for Linux)
- Setting up pixi for environment management
- Installing Mojo in a project environment
- Adding Python and pygame to the same environment
- Using VS Code with WSL for development

**See**: `mojo-simulation/TotrialWindowsMoJo.md` for the complete step-by-step guide.

This tutorial is especially helpful if you're new to Mojo or prefer a managed environment setup using pixi.

---

## 🚀 Usage

### Pure Python Version

Run the standalone Python implementation:

```bash
python fluid_sim_2d.py
```

With custom parameters:

```bash
python fluid_sim_2d.py --particles 2000 --box-width 10 --box-height 8 --zoom 1.5
```

### Mojo + Python Hybrid

#### Prerequisites
- **Windows Users**: Follow the installation guide in `mojo-simulation/TotrialWindowsMoJo.md` to set up Mojo using WSL and pixi
- **Linux/macOS Users**: Install Mojo from the [official guide](https://docs.modular.com/mojo/manual/get-started/)

#### Running the Hybrid Version

Run the Mojo simulation piped to the Python renderer:

```bash
cd mojo-simulation
mojo run fluid_sim_core.mojo | python fluid_render_client.py
```

The Mojo program will run the simulation and stream data to Python for rendering.

**Note for Windows/WSL users**: Make sure you're running this command inside your WSL environment (after running `wsl` and entering your pixi shell if using pixi).

---

## 🔧 Core Functions & Components

### Pure Python Implementation (`fluid_sim_2d.py`)

#### Vector Utilities
- `v_add(a, b)`: Vector addition
- `v_sub(a, b)`: Vector subtraction
- `v_mul(a, s)`: Vector scalar multiplication
- `v_length(a)`: Vector magnitude
- `v_length_sq(a)`: Squared magnitude (optimization)
- `v_normalize(a)`: Normalize vector

#### Kernel Functions (SPH)
- `smoothing_kernel_poly6(dst, radius, poly6_scaling)`: Poly6 smoothing kernel for viscosity
- `spiky_kernel_pow2(dst, radius, scale)`: Spiky kernel (pow2) for density
- `spiky_kernel_pow3(dst, radius, scale)`: Spiky kernel (pow3) for near-density
- `derivative_spiky_pow2(dst, radius, scale)`: Derivative for pressure forces
- `derivative_spiky_pow3(dst, radius, scale)`: Derivative for near-pressure forces

#### Spatial Hashing
- `get_cell_2d(position, radius)`: Convert world position to grid cell
- `hash_cell_2d(cell)`: Hash cell coordinates
- `_build_spatial_hash()`: Build spatial hash table for neighbor lookup

#### Simulation Core (`FluidSim2D` class)
- `__init__()`: Initialize simulation with config
- `_init_spawn()`: Spawn particles in grid pattern with jitter
- `reset()`: Reset to initial spawn state
- `_external_forces(pos, vel)`: Calculate gravity + mouse interaction
- `_handle_collisions(index)`: Handle bounds and obstacle collisions
- `_build_spatial_hash()`: Build spatial hash for neighbor search
- `_calculate_density_for_particle(index)`: Calculate density and near-density
- `_update_densities()`: Update all particle densities
- `_pressure_from_density(density)`: Convert density to pressure
- `_near_pressure_from_density(near_density)`: Convert near-density to pressure
- `_apply_pressure_forces(dt)`: Apply pressure forces between neighbors
- `_apply_viscosity(dt)`: Apply viscosity forces between neighbors
- `step(frame_time)`: Run one simulation frame with substeps

#### Visualization (`FluidSimApp` class)
- `__init__()`: Initialize pygame window and simulation
- `world_to_screen(p)`: Convert world coordinates to screen pixels
- `screen_to_world(x, y)`: Convert screen pixels to world coordinates
- `_handle_events()`: Process keyboard and mouse input
- `_draw()`: Render simulation state
- `_draw_graphs()`: Draw FPS and collision graphs
- `run()`: Main game loop

### Mojo Implementation (`mojo-simulation/fluid_sim_core.mojo`)

The Mojo version implements the same functions with identical logic, but uses:
- `SIMD[DType.float64, 2]` for 2D vectors (instead of tuples)
- `List[T]` for dynamic arrays
- Mojo's type system for better optimization
- Same function names and structure for easy comparison

### Python Renderer (`mojo-simulation/fluid_render_client.py`)

- `parse_frame(line)`: Parse frame data from Mojo output
- `world_to_screen(p, ...)`: Convert world to screen coordinates
- `v_length(v)`: Calculate velocity magnitude for coloring
- `draw_graphs(...)`: Render FPS and collision graphs
- `main()`: Main loop that reads from stdin and renders

---

## 📊 Performance Comparison

The Mojo implementation demonstrates significant performance improvements:

| Metric | Python             | Mojo                 | Improvement |
|--------|--------------------|----------------------|-------------|
| **1000 particles** | ~3-5 FPS           | ~50-55 FPS           | **10-18x faster** |
| **2000 particles** | ~1-3 FPS           | ~10-15 FPS           | **5-15x faster** |
| **Neighbor search** | O(n²) Python loops | O(n²) compiled loops | **10-20x faster** |

### Why Mojo is Faster

1. **Compiled Code**: Mojo compiles to native machine code, eliminating Python interpreter overhead
2. **Type System**: Strong typing allows better optimizations
3. **SIMD Support**: Vector operations can use CPU SIMD instructions
4. **Memory Layout**: Better cache locality with structured data types
5. **No GIL**: No Global Interpreter Lock limiting parallelism

### Visual Evidence

Check out the included GIF files:
- `python_fluid_sim.gif`: Shows the Python version running
- `mojo_fluid_sim.gif`: Shows the Mojo version running at higher FPS

The difference in smoothness and particle count handling is immediately apparent!

---

## 📁 Project Structure

```
FluidSim/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── fluid_sim_2d.py             # Pure Python implementation (standalone)
│
└── mojo-simulation/            # Mojo + Python hybrid implementation
    ├── fluid_sim_core.mojo      # Mojo simulation logic
    ├── fluid_render_client.py   # Python renderer
    └── TotrialWindowsMoJo.md    # Windows installation tutorial (WSL + pixi)
│
└── Assets/                      # GIF files and other assets
    ├── python_fluid_sim.gif     # Python version demo
    └── mojo_fluid_sim.gif       # Mojo version demo
```

---

## 🔬 Technical Details

### SPH Algorithm Overview

The simulation uses **Smoothed Particle Hydrodynamics**:

1. **Density Calculation**: For each particle, sum kernel contributions from neighbors within smoothing radius
2. **Pressure Forces**: Calculate pressure from density deviation, apply forces using kernel derivatives
3. **Viscosity Forces**: Smooth velocity differences between neighboring particles
4. **Integration**: Update positions using velocity, handle collisions

### Key Parameters (in `SimConfig`)

- `smoothing_radius`: Interaction radius for SPH kernels
- `target_density`: Desired particle density (affects pressure)
- `pressure_multiplier`: Strength of pressure forces
- `viscosity_strength`: Strength of viscosity forces
- `collision_damping`: Energy loss on collisions (0-1)

### Optimization Techniques

- **Early Exit Collision Detection**: Skip obstacle checks for particles far away
- **Squared Distance Checks**: Avoid `sqrt()` in hot loops
- **Spatial Hashing**: (In Python version) Use grid-based neighbor lookup
- **Precomputed Scaling Factors**: Cache kernel normalization constants

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add spatial hashing to Mojo version (currently brute force)
- [ ] Implement GPU acceleration (CUDA/OpenCL)
- [ ] Add 3D version
- [ ] Improve visualization (shaders, better gradients)
- [ ] Add more interaction modes
- [ ] Performance profiling and optimization
- [ ] Better documentation and comments

---

## 📄 License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for the full license text, or visit [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0) for details.

---

## 🙏 Acknowledgments

- **Sebastian Lague**: For the excellent [Fluid Simulation tutorial](https://www.youtube.com/watch?v=rSKMYc1CQHE) that inspired this project
- **Modular AI**: For creating Mojo and making high-performance computing more accessible
- **Pygame Community**: For the excellent 2D graphics library

---

## 📚 Further Reading

- [SPH Method Overview](https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics)
- [Mojo Documentation](https://docs.modular.com/mojo/)
- [Sebastian Lague's YouTube Channel](https://www.youtube.com/@SebastianLague)

---

**Enjoy simulating fluids! 🌊**
