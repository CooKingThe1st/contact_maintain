# Contact Maintain

Contact maintenance with holonomic and differential-drive robots in PyBullet simulation.

## Overview

This package implements controllers for maintaining contact between robots and objects. The goal is to explore contact maintenance under different conditions:

- **Robot types**: Holonomic (omni-directional) and differential-drive
- **Sensing**: With and without force sensor access
- **Simulation**: PyBullet physics engine

## Installation

This code has been tested on Ubuntu 20.04 with ROS Noetic and Python 3.8.

Clone this repository into your catkin workspace:

```bash
cd ~/catkin_ws/src
git clone <repository_url> contact_maintain
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build the workspace:

```bash
catkin build contact_maintain
```

## Project Structure

```
contact_maintain/
├── config/                    # Configuration files
├── data/                      # Logged experiment data
├── scripts/
│   └── simulation/
│       ├── basic_scene.py    # Single robot scene with web observer
│       ├── multi_robot_scene.py  # Multi-robot scene with T-shape
│       ├── test_robot.py     # Robot testing script
│       └── contact_maintain_sim.py  # Contact maintenance sim
├── src/contact_maintain/     # Python package
│   ├── __init__.py
│   ├── util.py               # Utility functions
│   ├── pyb_simulation.py     # PyBullet simulation primitives
│   ├── robots.py             # Robot classes (Holonomic, DiffDrive)
│   ├── robot_factory.py      # Factory for creating robots (all 4 types)
│   ├── omniwheel_robot.py    # Realistic 4-wheel omni robot
│   ├── diffdrive_wheel_robot.py # Realistic 2-wheel diff-drive robot
│   ├── objects.py            # Non-convex objects (TShape, LShape)
│   ├── object_bridge.py      # Bridge between GenericObject and PyBullet
│   ├── control.py            # Basic velocity controllers
│   ├── contact_maintain_controller.py # Contact maintenance controllers
│   ├── solvers.py            # Contact maintenance algorithms
│   ├── observer.py           # Contact observation and tracking
│   ├── logging.py            # Data logging utilities
│   ├── visualization.py      # Plotting and visualization
│   └── web_observer.py       # Flask web dashboard (multi-robot)
├── urdf/
│   ├── holonomic_robot.urdf       # Compiled Robotino 3 model (dummy holo)
│   ├── omniwheel_robot.urdf       # Compiled 4-wheel omni robot (realistic)
│   ├── diffdrive_wheel_robot.urdf # Compiled 2-wheel diff drive (realistic)
│   ├── compile_xacro.sh           # URDF compilation script
│   └── xacro/
│       ├── holonomic_robot.urdf.xacro     # Dummy holonomic model
│       ├── omniwheel_robot.urdf.xacro     # Realistic omniwheel model
│       └── diffdrive_wheel_robot.urdf.xacro  # Realistic diff-drive model
├── CMakeLists.txt
├── package.xml
├── setup.py
└── README.md
```

## Usage

## Notes on force “feedback” in PyBullet contact tests

Some of the test scripts (e.g. `scripts/test/test_force_vs_speed.py`) log a “contact force”
using `get_contact_force(...)`.

This is useful as a **diagnostic**, but it is easy to misinterpret:

- **Single pusher, steady sliding**: once the object is moving at (roughly) steady velocity,
  the measured contact force is dominated by **Coulomb friction** (plus drive limits / solver
  constraints). Increasing commanded speed often does **not** increase the *steady* measured force,
  because the object accelerates until friction + drive limits balance the motion.
- **Transients matter**: if you want to see force change with actuation, you should look at
  **acceleration/impact/ramp** phases, not long steady-velocity pushing.
- **Two robots (especially opposite sides)**: the reported contact “force” is not a clean
  per-robot “applied force” measurement. It can include **impulse/constraint** effects from the
  rigid-body contact solver as the object and robots become a coupled system.

### Basic Scene with Web Observer

The main simulation script with full state tracking and web-based monitoring:

```bash
# Run basic scene (robot approaches and contacts object)
rosrun contact_maintain basic_scene.py

# Run without GUI (headless)
rosrun contact_maintain basic_scene.py --no-gui

# Run without web observer
rosrun contact_maintain basic_scene.py --no-web

# Set custom duration
rosrun contact_maintain basic_scene.py --duration 120
```

Then open http://localhost:5000 in your browser to see the real-time dashboard.

### Multi-Robot Scene

Run multiple robots approaching a non-convex T-shaped object:

```bash
# Run with 3 robots (default)
rosrun contact_maintain multi_robot_scene.py

# Run with custom number of robots
rosrun contact_maintain multi_robot_scene.py --num-robots 5

# Use different object shapes
rosrun contact_maintain multi_robot_scene.py --object tshape  # T-shape (default)
rosrun contact_maintain multi_robot_scene.py --object box     # Rectangular box
rosrun contact_maintain multi_robot_scene.py --object cylinder

# Run headless
rosrun contact_maintain multi_robot_scene.py --no-gui --no-web
```

The web dashboard supports:
- **Robot Selection**: Choose which robot to focus on in detailed plots
- **Object Selection**: Choose which object to track
- **Robot Status Bar**: Live badges showing all robots' contact status
- **Contact Count**: "X/N robots in contact" metric

### Test Robot Motion

```bash
rosrun contact_maintain test_robot.py --test all
```

Options:
- `--test holonomic`: Test holonomic robot only
- `--test diffdrive`: Test differential-drive robot only
- `--test contact`: Test contact scenario only
- `--test all`: Run all tests (default)

### Run Contact Maintenance Simulation

```bash
rosrun contact_maintain contact_maintain_sim.py --mode with_force --robot holonomic
```

Options:
- `--mode`: `with_force`, `without_force`, `compare`, or `all`
- `--robot`: `holonomic`, `diffdrive`, or `both`
- `--no-gui`: Run without GUI
- `--save`: Save logged data
- `--plot`: Show analysis plots after simulation

### Examples

```bash
# Run force-based contact maintenance with holonomic robot
rosrun contact_maintain contact_maintain_sim.py --mode with_force --robot holonomic

# Compare force vs position-based control with both robot types
rosrun contact_maintain contact_maintain_sim.py --mode compare --robot both --plot

# Run all experiments without GUI and save data
rosrun contact_maintain contact_maintain_sim.py --mode all --robot both --no-gui --save
```

## Solvers

### Force-Based (`ForceBasedContactSolver`)
Uses measured contact force to maintain a target contact force with PID control.

### Position-Based (`PositionBasedContactSolver`)
Uses position estimation to maintain contact distance without force feedback.

### Adaptive (`AdaptiveContactSolver`)
Hybrid approach that blends force and position control based on signal reliability.

### Differential-Drive Variants
- `DiffDriveForceBasedSolver`
- `DiffDrivePositionBasedSolver`
- `DiffDriveAdaptiveSolver`

## Contact Optimization: Stochastic Magnum Search

### Overview

A major improvement in contact configuration search: **stochastic Latin square-based search** that finds sufficient contact configurations in **sub-second time** (typically 0.1-0.5s) compared to exhaustive search (50+ seconds), while maintaining high validity.

### The Problem

Previous exhaustive search methods (`find_the_magnum_four_v3`) were too slow for real-time multi-robot systems (MRS):
- **Star shape example**: Found best solution at iteration 912 but continued for 50+ seconds testing 21,000+ configurations
- **Not suitable for reactive robotics**: Need "good enough" solutions quickly, not optimal solutions slowly

### The Solution: Stochastic "Anytime" Algorithm

**Philosophy**: Move from "finding the highest number" (optimization) to "finding the first solution that works" (engineering sufficiency).

**Key Innovation**: Use **Limit Surface (LS) containment** as the physical ground truth instead of arbitrary quality scores:
- **Grasp Wrench Space (GWS)**: Set of wrenches the current 4-robot configuration can apply
- **Limit Surface (LS)**: 3D boundary in wrench space representing maximum static friction
- **Success criterion**: `GWS ⊇ threshold × LS` (typically threshold = 1.0 = 100% coverage)

### Algorithm: Latin Square-Based Stochastic Search

1. **Strategic Sampling**: Generate strategic contact points on object edges:
   - Near-corner points (epsilon away from vertices)
   - Edge midpoints and quartiles
   - No-torque points (where τ(t) = 0)
   - Tangency points (from max inscribed circles)

2. **Latin Square Sampling**: Create uniform combinations without clustering:
   - 4 columns (one per robot)
   - N rows (N = number of strategic points)
   - Each column is a random permutation of [0, 1, ..., N-1]
   - Ensures each strategic point is used exactly once per batch

3. **Early Pruning**: Fast geometric checks before expensive wrench space calculation:
   - Distinct points check
   - Non-parallel normals
   - Quick force closure check
   - Robot spacing validation

4. **Sufficiency Check**: 3-plane projection test:
   - Project GWS and LS onto (Fx, Fy), (Fx, τ), (Fy, τ)
   - Check that GWS convex hull contains scaled LS ellipse in all projections
   - **Early termination**: Return immediately when first sufficient configuration found

### Performance

**Speed Improvement**:
- **Exhaustive search**: 50+ seconds (Star shape, 21,000+ configs)
- **Stochastic search**: 0.1-0.5 seconds (typically finds solution in first batch)
- **Speedup**: ~100-500× faster

**Validity**:
- Solutions satisfy physical sufficiency: `GWS ⊇ threshold × LS`
- Same validity criteria as exhaustive search (just faster to find)
- Success rate: Typically 90%+ on standard shapes

**Real-Time Capability**:
- Suitable for reactive MRS applications
- Configurable timeout (default: 10s)
- "Anytime" algorithm: returns best solution found so far if timeout reached

### Usage

```python
from contact_optimizer_utils_test_ver import find_the_magnum_stochastic
from object_utils import create_standard_objects

# Get object
standard_objects = create_standard_objects()
obj = standard_objects['star']

# Run stochastic search
result = find_the_magnum_stochastic(
    obj,
    threshold=1.0,              # 100% LS coverage (default)
    timeout=10.0,               # Search budget in seconds
    force_range_scalar=2.0,     # Robots can exert 2× static friction
    robot_radius=0.06,          # Robot radius for spacing checks (engineering mode)
    theory_mode=False,          # True: skip robot/FC prune; keep sufficiency check
    verbose=True
)

if result['success']:
    contacts = result['contacts']
    print(f"Found in {result['elapsed_time']:.3f}s")
    print(f"Configs tested: {result['configs_tested']}")
else:
    print("No sufficient configuration found")
```

### Comprehensive Testing

Run the test suite to evaluate performance across all standard shapes:

```bash
cd scripts/test
python test_stochastic_magnum.py
```

**Output**:
- Tests all standard shapes (rectangle, triangle, star, L-shape, etc.)
- Records success rate, elapsed time, configs tested
- Generates visualization plots showing:
  - Object with contact points
  - 2D wrench space projections (Fx-Fy, Fx-τ, Fy-τ)
  - GWS coverage of Limit Surface
- Saves plots to `/tmp/basic_test/stochastic_{shape_name}.jpg`

**Example Output**:
```
📊 SUMMARY STATISTICS
============================================================
   Total shapes tested    : 15
   Successful            : 14 (93.3%)
   Failed                 : 1 (6.7%)
   Average time per shape: 0.234 s
   
✅ Successful Searches:
   Average time          : 0.198 s
   Min time               : 0.089 s
   Max time               : 0.512 s
   Average configs tested: 12.3
```

### Technical Details

**Force Range Calculation**:
- Based on object's static friction: `max_force = force_range_scalar × static_friction × (mass × 9.81)`
- Default `force_range_scalar = 2.0` assumes robots can exert 2× static friction limit

**Limit Surface Calculation**:
- Uses numerical integration for accurate moment calculation
- Ellipsoid in (Fx, Fy, τ) space
- Scaled by `threshold` parameter (1.0 = 100% coverage)

**Visualization**:
- 2×2 grid layout:
  - Top-left: Object with contact points
  - Top-right: Fx vs Fy projection
  - Bottom-left: Fx vs Torque projection
  - Bottom-right: Fy vs Torque projection
- Shows GWS points, convex hull, and Limit Surface ellipse/circle

### Files

| File | Description |
|------|-------------|
| `src/legacy/contact_optimizer_utils_test_ver.py` | Main implementation: `find_the_magnum_stochastic()` |
| `src/legacy/object_utils.py` | `WrenchSpaceVisualizer`: Wrench space and limit surface calculation |
| `scripts/test/test_stochastic_magnum.py` | Comprehensive test suite |

### References

- **Limit Surface Theory**: Maximum static friction boundary in wrench space
- **Latin Hypercube Sampling**: Uniform sampling without clustering
- **Anytime Algorithms**: Return best solution found so far, improve with time

## API

### Creating a Solver

```python
from contact_maintain import create_solver

# Holonomic robot with force feedback
solver = create_solver(
    solver_type='force',
    robot_type='holonomic',
    target_force=5.0,
    kp_force=0.02
)

# Differential-drive robot without force feedback
solver = create_solver(
    solver_type='position',
    robot_type='diffdrive',
    robot_radius=0.06,
    object_radius=0.5
)
```

### Computing Velocity Commands

```python
# For holonomic robot: returns (vx, vy, omega)
cmd = solver.compute_velocity(robot_pos, robot_theta, object_pos, contact_force)
robot.command_velocity(cmd)

# For differential-drive robot: returns (v, omega)
cmd = solver.compute_velocity(robot_pos, robot_theta, object_pos, contact_force)
robot.command_velocity(cmd)
```

## Progress Tracking

### Phase 1: Package Setup and Robot Design - COMPLETE

- [x] Clone force_push to contact_maintain package (Jan 8, 2026)
- [x] Update package metadata (package.xml, CMakeLists.txt, setup.py)
- [x] Clean out mobile-manipulator-specific code
- [x] Keep core simulation utilities
- [x] Create Robotino 3 URDF (cylinder body + bumper)

### Phase 2: Basic Robot Controller + State Tracking - COMPLETE

- [x] Create HolonomicRobot class for PyBullet
- [x] Create DifferentialDriveRobot class
- [x] Implement state tracking:
  - [x] Robot pose (x, y, heading) and velocity (vx, vy, omega)
  - [x] Object pose and velocity
  - [x] Contact state (in_contact, force magnitude, direction)
- [x] Web-based observer (Flask + SocketIO):
  - [x] Real-time dashboard at http://localhost:5000
  - [x] Live plots (trajectory, velocity, contact force, etc.)
  - [x] Metrics display (contact time, losses, etc.)
- [x] Simple P controller for approach and contact

### Phase 3: Contact Observation and Analysis - COMPLETE

- [x] ContactObserver for tracking contact state
- [x] ContactPointTracker for position-based estimation
- [x] DataLogger for experiment logging
- [x] Visualization utilities for analysis
- [x] Web-based real-time observer (WebObserver):
  - [x] Flask + SocketIO dashboard at http://localhost:5000
  - [x] Live plots: trajectory, velocity, force, heading, distance
  - [x] Pushing direction tracking and visualization
  - [x] Object velocity components (vx, vy, ω) plot
  - [x] Metrics display: contact time, losses, force magnitude
- [x] Keyboard quit (press 'Q' in PyBullet window for safe exit)

### Phase 3.5: Multi-Robot System - COMPLETE

- [x] Multi-robot scene (`multi_robot_scene.py`):
  - [x] Configurable number of robots (default: 3)
  - [x] Circular placement around object (no overlapping)
  - [x] Robot IDs: R_01, R_02, ..., R_XX
- [x] Non-convex objects (`objects.py`):
  - [x] TShapeObject: T-shaped composite body
  - [x] LShapeObject: L-shaped composite body
- [x] Enhanced web observer for multi-robot:
  - [x] Robot selection dropdown (choose which robot to track)
  - [x] Object selection dropdown
  - [x] Robot status bar with live badges (contact status per robot)
  - [x] "X/N robots in contact" metric
  - [x] Selected robot force and push direction display

### Phase 4: Contact Maintenance Problem - COMPLETE

- [x] **Case A - With force sensor**:
  - [x] Holonomic robot (ForceBasedContactSolver)
  - [x] Differential-drive robot (DiffDriveForceBasedSolver)
- [x] **Case B - Without force sensor**:
  - [x] Holonomic robot (PositionBasedContactSolver)
  - [x] Differential-drive robot (DiffDrivePositionBasedSolver)

### Phase 4 Pre-Plan: Contact Maintenance Foundation - COMPLETE

Foundation work for boundary point tracking contact maintenance.

- [x] **Step 1: Object Bridge** (`object_bridge.py`):
  - [x] `generic_to_pybullet()`: Convert GenericObject to PyBullet body
  - [x] `pybullet_to_generic()`: Convert PyBullet body to GenericObject
  - [x] `BridgedObject`: Wrapper maintaining both representations
  - [x] Support for compound shapes (L, T) via convex decomposition
- [x] **Step 2: Physics Validation** (`scripts/validation/physics_validation.py`):
  - [x] Compare BoundaryMotionPredictor vs PyBullet ground truth
  - [x] Track object and boundary point motion
  - [x] Plot position, velocity, and error metrics
  - [x] NOTE: Validation shows discrepancies between object_utils.py 2D quasi-static
    friction model and PyBullet's full 3D rigid body dynamics. This is expected
    and informs controller design decisions.
- [x] **Step 3a: Realistic Omniwheel Robot**:
  - [x] `omniwheel_robot.urdf.xacro`: 4-wheel omni robot (45°, 135°, 225°, 315°)
  - [x] `OmniwheelRobot` class with wheel velocity conversion
  - [x] `compare_robots.py`: Test dummy vs realistic holonomic robot trajectories
- [x] **Step 3b: Realistic Diff-Drive Robot**:
  - [x] `diffdrive_wheel_robot.urdf.xacro`: 2-wheel diff drive + caster
  - [x] `DiffDriveWheelRobot` class with wheel velocity conversion
  - [x] `compare_diffdrive.py`: Test dummy vs realistic diff-drive trajectories
- [x] **Step 3c: Pre-Pushing Coordination (Swarm State Machine)** - COMPLETE:
  - [x] `SwarmHost` decentralized coordinator (state tracking, goal coordination)
  - [x] `RobotAgent` class with navigation and pushing modules
  - [x] ORCA navigation (RVO2) for optimal collision avoidance
  - [x] `test_swarm_coordination.py`: Validate multi-robot coordination
  - [x] `test_orca_navigation.py`: Test ORCA with formation and obstacles
  - [x] States: IDLE → REACHING → WAITING → PUSHING → CHANGING
  - [x] Reconfiguration only when all robots at target AND in contact
- [x] **Step 4: Contact Maintenance Controllers** (`contact_maintain_controller.py`):
  - [x] `InstantVelocityMatcher`: Match robot velocity to boundary point velocity
  - [x] `WrenchTrackingController`: Apply desired wrench through contact point
  - [x] `SimpleVelocityTracker`: Pure velocity tracking for validation
  - [x] `test_contact_maintain.py`: Validate contact maintenance

#### Usage

```bash
# Physics validation (headless by default, saves plot to /tmp/)
python scripts/validation/physics_validation.py --shape rectangle --duration 3
python scripts/validation/physics_validation.py --shape circle --gui  # With GUI

# Compare dummy vs omniwheel robot (holonomic)
python scripts/test/compare_robots.py --trajectory circle --duration 10
python scripts/test/compare_robots.py --trajectory figure8 --no-gui --save-plot /tmp/fig8.png

# Compare dummy vs wheel-based differential drive
python scripts/test/compare_diffdrive.py --trajectory arc --duration 10
python scripts/test/compare_diffdrive.py --trajectory scurve --no-gui

# Test contact maintenance
python scripts/test/test_contact_maintain.py --controller velocity --duration 10
python scripts/test/test_contact_maintain.py --controller wrench --perturbation continuous
```

**PyBullet GUI Interaction:**
- Left-click + drag: Rotate camera
- Middle-click + drag: Pan camera
- Scroll wheel: Zoom in/out
- Press Enter in terminal to exit (after simulation completes)

### Step 3c: Pre-Pushing Coordination (Swarm State Machine) - COMPLETE

Decentralized multi-robot system where each robot agent computes its own navigation and
pushing velocities. The SwarmHost only tracks state and coordinates goals (t_params).
All robots must reach their target contact positions and establish contact before pushing begins.

#### Swarm State Machine

```
┌─────────┐    spawn     ┌──────────┐   all at    ┌─────────┐   all in    ┌─────────┐
│  IDLE   │ ──────────▶  │ REACHING │ ──────────▶ │ WAITING │ ──────────▶ │ PUSHING │
└─────────┘              └──────────┘   target    └─────────┘   contact   └─────────┘
                              │                        │                       │
                              │ collision              │ contact lost          │ reconfig
                              ▼ avoidance              ▼                       ▼
                         ┌──────────┐            ┌──────────┐           ┌──────────┐
                         │ (avoid)  │            │ (recover)│           │ CHANGING │
                         └──────────┘            └──────────┘           └──────────┘
                                                                              │
                                                                              │ new t_params
                                                                              ▼
                                                                        ┌──────────┐
                                                                        │ REACHING │
                                                                        └──────────┘
```

**States:**

| State | Description |
|-------|-------------|
| `IDLE` | Robot spawned, waiting for target assignment |
| `REACHING` | Moving toward target t_param position on object boundary |
| `WAITING` | At target position, waiting for all robots to reach their positions |
| `PUSHING` | All robots in contact, coordinated pushing active |
| `CHANGING` | Transitioning to new configuration (new t_params assigned) |

#### Decentralized Architecture

**RobotAgent**: Each robot is a decentralized agent that:
- Computes its own navigation velocity (using ORCA)
- Computes its own pushing velocity (using contact controller)
- Communicates with SwarmHost for goal updates

**SwarmHost**: Only tracks state and coordinates goals:
- Monitors robot positions and contact states
- Assigns target t_params to robots
- Transitions swarm state when conditions are met
- Triggers reconfiguration only when all robots are at target AND in contact

```python
from contact_maintain.robot_agent import RobotAgent
from contact_maintain.swarm import SwarmHost, SwarmState

# Create robot agents (each computes its own velocity)
robot_agents = {}
for name, robot in robots.items():
    agent = RobotAgent(
        robot=robot,
        name=name,
        object_uid=object_uid,
        generic_object=generic_object,
        navigation_type='orca',  # ORCA navigation
        pushing_type='velocity',  # Velocity-based pushing
    )
    robot_agents[name] = agent

# Create host (only tracks state, coordinates goals)
host = SwarmHost(
    robot_agents=robot_agents,
    object_uid=object_uid,
    generic_object=generic_object,
)

# Assign initial t_params
host.assign_targets({
    'R_01': 0.0,
    'R_02': 0.33,
    'R_03': 0.66,
})

# Main loop
while running:
    # Update host (tracks state, updates goals)
    host.update(dt, object_state)
    
    # Each agent computes its own velocity
    for name, agent in robot_agents.items():
        other_positions = [
            robot_agents[other_name].robot.get_state()[0]
            for other_name in robot_agents.keys()
            if other_name != name
        ]
        
        cmd = agent.compute_velocity(object_state, other_positions)
        agent.robot.command_velocity(cmd)
    
    # Check swarm state
    if host.swarm_state == SwarmState.PUSHING:
        print("All robots at target and in contact - pushing!")
    
    # Reconfigure only when all robots are at target AND in contact
    if (host.swarm_state == SwarmState.PUSHING and 
        all_robots_at_target_and_in_contact):
        host.reconfigure({
            'R_01': 0.1,
            'R_02': 0.5,
            'R_03': 0.9,
        })
```

#### ORCA Navigation (Optimal Reciprocal Collision Avoidance)

During REACHING and CHANGING states, robots use ORCA (RVO2) for optimal collision avoidance.
ORCA avoids local minima that potential fields can fall into.

**Installation:**
```bash
pip install git+https://github.com/sybrenstuvel/Python-RVO2.git
```

**Features:**
- Optimal reciprocal collision avoidance (no local minima)
- Handles static obstacles
- Works with multiple robots simultaneously
- Each robot computes its own velocity independently

**Parameters:**
- `time_step`: Simulation time step (default: 1/60s)
- `neighbor_dist`: Maximum distance to consider neighbors (default: 2.0m)
- `time_horizon`: Time horizon for collision avoidance (default: 2.0s)
- `radius`: Robot radius (default: 0.06m)
- `max_speed`: Maximum speed (default: 0.5 m/s)

**Testing ORCA:**
```bash
# Test formation (2 side lines, opposite goals)
python scripts/test/test_orca_navigation.py --test formation

# Test with static obstacles
python scripts/test/test_orca_navigation.py --test obstacles

# Both tests
python scripts/test/test_orca_navigation.py --test all
```

#### Test Scripts

```bash
# Test basic swarm coordination (3 robots)
python scripts/test/test_swarm_coordination.py --num-robots 3

# Test with reconfiguration
python scripts/test/test_swarm_coordination.py --num-robots 5 --reconfig-interval 10

# Test collision avoidance during reconfiguration
python scripts/test/test_swarm_coordination.py --test collision_avoidance

# Full swarm test with output
python scripts/test/test_swarm_coordination.py \
    --num-robots 4 \
    --duration 30 \
    --reconfig-interval 10 \
    --save-dir /tmp/swarm_test/
```

#### Expected Output

```
============================================================
  SWARM COORDINATION TEST
============================================================
  Robots: 4
  Initial t_params: [0.0, 0.25, 0.5, 0.75]
============================================================

[t=0.00s] SWARM STATE: IDLE → REACHING
  R_01: IDLE → REACHING (target t=0.00)
  R_02: IDLE → REACHING (target t=0.25)
  R_03: IDLE → REACHING (target t=0.50)
  R_04: IDLE → REACHING (target t=0.75)

[t=2.34s] R_01: REACHING → WAITING (at target)
[t=2.51s] R_03: REACHING → WAITING (at target)
[t=2.89s] R_02: REACHING → WAITING (at target)
[t=3.12s] R_04: REACHING → WAITING (at target)

[t=3.12s] SWARM STATE: REACHING → WAITING
  All robots at target positions

[t=3.45s] R_01: contact established (force=2.3N)
[t=3.52s] R_03: contact established (force=1.9N)
[t=3.61s] R_02: contact established (force=2.1N)
[t=3.78s] R_04: contact established (force=2.4N)

[t=3.78s] SWARM STATE: WAITING → PUSHING
  All robots in contact - pushing enabled!

[t=10.0s] RECONFIGURATION TRIGGERED
  New t_params: [0.1, 0.4, 0.6, 0.9]

[t=10.0s] SWARM STATE: PUSHING → CHANGING
  R_01: PUSHING → CHANGING (new target t=0.10)
  R_02: PUSHING → CHANGING (new target t=0.40)
  ...

============================================================
  SWARM METRICS
============================================================
  Total time to first push: 3.78s
  Reconfiguration time: 2.45s
  Collision events: 0
  Contact losses during push: 0
============================================================
```

#### Files

| File | Description |
|------|-------------|
| `src/contact_maintain/swarm.py` | SwarmHost (state tracking, goal coordination) |
| `src/contact_maintain/robot_agent.py` | RobotAgent (decentralized navigation & pushing) |
| `src/contact_maintain/orca_navigation.py` | ORCA navigation using RVO2 |
| `src/contact_maintain/potential_field.py` | Potential field navigation (legacy) |
| `scripts/test/test_swarm_coordination.py` | Integration test script |
| `scripts/test/test_orca_navigation.py` | ORCA navigation test (formation & obstacles) |

### Comprehensive Contact Maintenance Testing

The comprehensive test framework supports all robot configurations in a single script.

#### Test Matrix

| Dimension | Options |
|-----------|---------|
| Robot Count | Single (1) / Multi (N) |
| Robot Model | Dummy (direct velocity) / Wheel (realistic physics) |
| Kinematics | Holonomic / Diff-drive |
| Controller | Velocity / Wrench |

**Total combinations**: 2 × 2 × 2 × 2 = 16 scenarios

#### Single Test

```bash
# Single holonomic dummy robot with velocity controller:
python scripts/test/comprehensive_contact_test.py \
  --num-robots 1 --kinematics holonomic --model dummy --controller velocity

# Multi-robot (3) with wheel model:
python scripts/test/comprehensive_contact_test.py \
  --num-robots 3 --model wheel --t-params 0.1,0.4,0.7

# Diff-drive with wrench controller:
python scripts/test/comprehensive_contact_test.py \
  --kinematics diffdrive --controller wrench

# Full test with output:
python scripts/test/comprehensive_contact_test.py \
  --num-robots 3 --model wheel --save-dir /tmp/results/ --duration 15
```

#### Batch Testing (All Combinations)

```bash
# Run all 16 combinations:
python scripts/test/run_all_tests.py --output-dir /tmp/full_test/

# Run only holonomic tests:
python scripts/test/run_all_tests.py --kinematics holonomic --output-dir /tmp/holo/

# Run only wheel model tests:
python scripts/test/run_all_tests.py --model wheel --output-dir /tmp/wheel/

# Quick test (5s duration):
python scripts/test/run_all_tests.py --duration 5 --output-dir /tmp/quick/

# Parallel execution (4 workers):
python scripts/test/run_all_tests.py --parallel 4 --output-dir /tmp/parallel/
```

#### Output Structure

```
results/
├── config.yaml          # Test configuration
├── summary.png          # All robots overview plot
├── robot_R_01.png       # Detailed plots for R_01
├── robot_R_02.png       # Detailed plots for R_02
└── metrics.json         # Numerical results
```

#### Robot Factory

Create any robot type programmatically:

```python
from contact_maintain import create_robot

# Holonomic dummy (direct velocity control)
robot = create_robot('holonomic', 'dummy', position=(0, 0), orientation=0.0)

# Holonomic wheel (4-wheel omni physics)
robot = create_robot('holonomic', 'wheel', position=(1, 0), orientation=0.0)

# Diff-drive dummy
robot = create_robot('diffdrive', 'dummy', position=(0, 1), orientation=np.pi/2)

# Diff-drive wheel (2-wheel + caster)
robot = create_robot('diffdrive', 'wheel', position=(1, 1), orientation=np.pi)
```

## Next Steps

The following can be customized when user provides:
1. **Custom URDF**: Replace `urdf/xacro/holonomic_robot.urdf.xacro` with Webots model
2. **Custom Controller**: Port Webots controller to extend existing classes
3. **Observer Scripts**: Integrate user's Python observation code

## License

MIT
