#!/usr/bin/env python3
"""
Swarm Coordination Test

Tests the coordinated multi-robot navigation phase where:
1. All robots move to their target t_param positions (REACHING)
2. Wait until all robots are at positions (WAITING)
3. Optionally reconfigure to new t_param positions (CHANGING)
4. Repeat navigation cycle

By default, this tests NAVIGATION ONLY (no pushing, object is static).
Use --enable-pushing to test the full pushing phase.

Usage:
    # Basic navigation test with 3 robots (object static)
    python test_swarm_coordination.py --num-robots 3
    
    # Test with multiple reconfigurations
    python test_swarm_coordination.py --num-robots 4 --reconfig-interval 5 --duration 20
    
    # Test collision avoidance
    python test_swarm_coordination.py --test collision_avoidance
    
    # Enable actual pushing (object will move)
    python test_swarm_coordination.py --num-robots 3 --enable-pushing
    
    # Full test with output
    python test_swarm_coordination.py --num-robots 5 --duration 30 --save-dir /tmp/swarm/
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Use non-interactive backend for headless mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import create_standard_objects
from contact_maintain.robot_factory import create_robot
from contact_maintain.object_bridge import generic_to_pybullet
from contact_maintain.robot_agent import RobotAgent
from contact_maintain.swarm import SwarmHost, SwarmState, RobotState


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

# Object parameters
DEFAULT_OBJECT_SHAPE = 'rectangle'
DEFAULT_OBJECT_HEIGHT = 0.05
DEFAULT_OBJECT_FRICTION = 0.3

# Robot spawn configuration
ROBOT_SPAWN_RADIUS = 1.2  # Distance from object center


# ============================================================================
# SIMULATION SETUP
# ============================================================================

def setup_pybullet(gui=True):
    """Initialize PyBullet."""
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    
    return ground


def get_object_state(object_uid):
    """Get object state from PyBullet."""
    pos, orn = pyb.getBasePositionAndOrientation(object_uid)
    vel_lin, vel_ang = pyb.getBaseVelocity(object_uid)
    euler = pyb.getEulerFromQuaternion(orn)
    
    return {
        'position': np.array([pos[0], pos[1]]),
        'orientation': euler[2],
        'velocity': np.array([vel_lin[0], vel_lin[1]]),
        'angular_velocity': vel_ang[2],
    }


def get_object_as_obstacle(generic_object, object_position, object_orientation):
    """Convert object boundary to obstacle format for APF navigation.
    
    Parameters
    ----------
    generic_object : GenericObject
        The object to convert.
    object_position : np.ndarray
        Object center position (x, y) in world frame.
    object_orientation : float
        Object orientation (radians) in world frame.
    
    Returns
    -------
    list
        Obstacle as list of (x, y) tuples in world coordinates.
    """
    # Get boundary vertices in local frame
    boundary_coords = list(generic_object.geometry.exterior.coords)
    
    # Transform to world frame
    c, s = np.cos(object_orientation), np.sin(object_orientation)
    R = np.array([[c, -s], [s, c]])
    
    world_vertices = []
    for local_vertex in boundary_coords:
        # Convert to numpy array (skip z if present)
        local_2d = np.array([local_vertex[0], local_vertex[1]])
        # Rotate and translate
        world_2d = R @ local_2d + object_position
        world_vertices.append((float(world_2d[0]), float(world_2d[1])))
    
    return [world_vertices]  # Return as list of obstacles (one obstacle)


# ============================================================================
# TEST CLASS
# ============================================================================

class SwarmCoordinationTest:
    """Test class for swarm coordination.
    
    By default, tests navigation only (object is static).
    Set enable_pushing=True to test actual pushing behavior.
    """
    
    def __init__(
        self,
        num_robots: int = 3,
        object_shape: str = DEFAULT_OBJECT_SHAPE,
        kinematics: str = 'holonomic',
        model: str = 'dummy',
        duration: float = 30.0,
        reconfig_interval: Optional[float] = 5.0,  # Default: reconfigure every 5s
        enable_pushing: bool = False,  # Default: navigation only
    ):
        self.num_robots = num_robots
        self.object_shape = object_shape
        self.kinematics = kinematics
        self.model = model
        self.duration = duration
        self.reconfig_interval = reconfig_interval
        self.enable_pushing = enable_pushing
        
        # Create object
        print(f"\nCreating {object_shape} object...")
        standard_objects = create_standard_objects()
        self.generic_object = standard_objects[object_shape]
        
        self.object_uid = generic_to_pybullet(
            self.generic_object,
            height=DEFAULT_OBJECT_HEIGHT,
            position=(0, 0, 0),
            color=(0.4, 0.7, 0.4, 1.0)
        )
        pyb.changeDynamics(self.object_uid, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
        
        # If navigation only mode, make object static (very high mass or fixed)
        if not enable_pushing:
            # Make object essentially immovable
            pyb.changeDynamics(self.object_uid, -1, mass=1.0)
            print("  Mode: NAVIGATION ONLY (object is static)")
        
        # Create robots and agents
        self.robots: Dict[str, object] = {}
        self.robot_agents: Dict[str, RobotAgent] = {}
        self._create_robots()
        
        # Create swarm host with robot agents
        self.host = SwarmHost(
            robot_agents=self.robot_agents,
            object_uid=self.object_uid,
            generic_object=self.generic_object,
        )
        
        # History tracking
        self.history = {
            'times': [],
            'swarm_states': [],
            'robot_positions': {name: [] for name in self.robots},
            'robot_states': {name: [] for name in self.robots},
            'contacts': {name: [] for name in self.robots},
            'contact_forces': {name: [] for name in self.robots},
            'distances_to_target': {name: [] for name in self.robots},
        }
    
    def _create_robots(self):
        """Create all robots around the object."""
        print(f"\nCreating {self.num_robots} robots ({self.kinematics}, {self.model})...")
        
        for i in range(self.num_robots):
            name = f"R_{i+1:02d}"
            
            # Distribute robots around object
            spawn_angle = 2 * np.pi * i / self.num_robots
            robot_x = ROBOT_SPAWN_RADIUS * np.cos(spawn_angle)
            robot_y = ROBOT_SPAWN_RADIUS * np.sin(spawn_angle)
            robot_heading = spawn_angle + np.pi  # Face inward
            
            robot = create_robot(
                kinematics=self.kinematics,
                model=self.model,
                position=(robot_x, robot_y),
                orientation=robot_heading,
                name=name
            )
            
            self.robots[name] = robot
            
            # Create robot agent
            agent = RobotAgent(
                robot=robot,
                name=name,
                object_uid=self.object_uid,
                generic_object=self.generic_object,
                navigation_type='apf',  # Use APF navigation
                pushing_type='velocity',
            )
            self.robot_agents[name] = agent
            
            print(f"  {name}: pos=({robot_x:.2f}, {robot_y:.2f})")
    
    def run_test(self, gui=True) -> Dict:
        """Run the swarm coordination test.
        
        In navigation-only mode (default), tests:
        - REACHING: Robots move to t_param positions
        - WAITING: All robots at positions
        - CHANGING: Reconfigure to new t_params
        
        In pushing mode (--enable-pushing), also tests:
        - PUSHING: Coordinated pushing with contact
        
        Returns
        -------
        dict
            Test results and metrics.
        """
        n_steps = int(self.duration / TIMESTEP)
        step_count = 0
        t = 0.0
        last_reconfig_time = 0.0
        reconfig_count = 0
        
        # Generate initial t_params (uniform distribution with offset)
        # but swap at first
        offset = 0.17  # Choose desired offset in [0,1)

        initial_t_param = [(i / self.num_robots + offset) % 1.0 for i, _ in enumerate(self.robots.keys())]

        # one bug here where the robot 1 and 3 are stuck and velocity direction are swap continuously  and opposite
        # for i in range(0, len(initial_t_param) - 1, 2):
        #     initial_t_param[i], initial_t_param[i + 1] = initial_t_param[i + 1], initial_t_param[i]

        # not technically a bug, but need to update the waiting logic of the apf to be better when robot are close to each other, yielding 
        initial_t_param[1], initial_t_param[3] = initial_t_param[3], initial_t_param[1]

        initial_t_param = [0.0179, 0.4750, 0.6458, 0.9208]

        initial_t_params = {
            name: initial_t_param[i]
            for i, name in enumerate(self.robots.keys())
        }
        
        
        print("\n" + "=" * 60)
        print("  SWARM COORDINATION TEST")
        print("=" * 60)
        print(f"  Mode: {'PUSHING ENABLED' if self.enable_pushing else 'NAVIGATION ONLY'}")
        print(f"  Robots: {self.num_robots}")
        print(f"  Kinematics: {self.kinematics} | Model: {self.model}")
        print(f"  Initial t_params: {[f'{v:.2f}' for v in initial_t_params.values()]}")

        if self.reconfig_interval:
            print(f"  Reconfig interval: {self.reconfig_interval}s")
        print("=" * 60 + "\n")
        
        # Assign initial targets (triggers REACHING)
        self.host.assign_targets(initial_t_params)
        
        for step in range(n_steps):
            # Get object state
            obj_state = get_object_state(self.object_uid)
            
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Update swarm host (tracks state, coordinates goals)
                self.host.update(1.0 / CTRL_FREQ, obj_state)
                
                # Each robot agent computes its own velocity
                for name, agent in self.robot_agents.items():
                    robot = agent.robot
                    
                    # Get other robot positions for collision avoidance
                    other_positions = [
                        self.robot_agents[other_name].robot.get_state()[0]
                        for other_name in self.robot_agents.keys()
                        if other_name != name
                    ]
                    
                    # Object acts as obstacle during REACHING or CHANGING phase (navigate goal)
                    # but not during APPROACHING or PUSHING phases
                    obstacles = None
                    if agent.goal_type == 'navigate':
                        # Convert object boundary to obstacle format
                        obstacles = get_object_as_obstacle(
                            self.generic_object,
                            obj_state['position'],
                            obj_state['orientation']
                        )
                    
                    # Agent computes velocity based on its goal
                    cmd = agent.compute_velocity(
                        obj_state,
                        other_positions,
                        obstacles=obstacles,
                    )
                    
                    # Handle diff-drive conversion
                    if self.kinematics == 'diffdrive' and len(cmd) == 3:
                        pos, heading, _ = robot.get_state()
                        v_forward = cmd[0] * np.cos(heading) + cmd[1] * np.sin(heading)
                        robot.command_velocity(np.array([v_forward, cmd[2]]))
                    else:
                        robot.command_velocity(cmd)
                
                # Record history
                self.history['times'].append(t)
                self.history['swarm_states'].append(self.host.swarm_state.name)
                for name, robot in self.robots.items():
                    pos, _, _ = robot.get_state()
                    self.history['robot_positions'][name].append(pos.copy())
                    self.history['robot_states'][name].append(
                        self.host.robot_states[name].name
                    )
                    status = self.host.robot_statuses.get(name)
                    self.history['contacts'][name].append(
                        status.in_contact if status else False
                    )
                    self.history['contact_forces'][name].append(
                        status.contact_force if status else 0.0
                    )
                    self.history['distances_to_target'][name].append(
                        status.distance_to_target if status else float('inf')
                    )
                
                # Check for reconfiguration trigger
                # In pushing mode: reconfigure when all robots are at target AND in contact (PUSHING state)
                # In navigation-only mode: reconfigure when all robots are at target (WAITING state)
                if (self.reconfig_interval and 
                    ((self.enable_pushing and self.host.swarm_state == SwarmState.PUSHING) or
                     (not self.enable_pushing and self.host.swarm_state == SwarmState.WAITING)) and
                    t - last_reconfig_time >= self.reconfig_interval):
                    
                    # Generate new t_params
                    reconfig_count += 1
                    new_t_params = self._generate_new_t_params(reconfig_count)
                    
                    print(f"\n[t={t:.1f}s] RECONFIGURATION #{reconfig_count}")
                    print(f"  New t_params: {[f'{v:.2f}' for v in new_t_params.values()]}")
                    
                    self.host.reconfigure(new_t_params)
                    last_reconfig_time = t
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.3)
        
        # Add reconfig count to history
        self.history['reconfig_count'] = reconfig_count
        
        return self._compute_metrics()
    
    def _generate_new_t_params(self, seed: int) -> Dict[str, float]:
        """Swap t_params between robots for reconfiguration (forces more movement)."""
        robot_names = list(self.robots.keys())
        
        # Get current t_params
        current_t_params = [self.host.target_t_params.get(name, i / self.num_robots) for i, name in enumerate(robot_names)]
        
        # # Swap them: rotate assignments by one position
        # new_params = np.roll(current_t_params, 1)

        # Swap adjacent pairs
        new_params = current_t_params.copy()
        n = len(new_params)
        # swap adjacent elements (1<->2, 3<->4, etc.)
        for i in range(0, n - 1, 2):
            new_params[i], new_params[i + 1] = new_params[i + 1], new_params[i]

        return {
            name: float(new_params[i])
            for i, name in enumerate(robot_names)
        }

    
    def _compute_metrics(self) -> Dict:
        """Compute test metrics."""
        # Get host metrics
        host_metrics = self.host.get_metrics()
        
        # Compute per-robot metrics
        robot_metrics = {}
        for name in self.robots:
            contacts = np.array(self.history['contacts'][name])
            contact_ratio = np.mean(contacts) if len(contacts) > 0 else 0.0
            
            # Time to first contact
            time_to_contact = None
            for i, c in enumerate(contacts):
                if c:
                    time_to_contact = self.history['times'][i]
                    break
            
            robot_metrics[name] = {
                'contact_ratio': float(contact_ratio),
                'time_to_first_contact': time_to_contact,
            }
        
        return {
            'swarm': host_metrics,
            'robots': robot_metrics,
            'history': self.history,
        }
    
    def save_results(self, save_dir: str, metrics: Dict):
        """Save test results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics (without numpy arrays)
        metrics_clean = {
            'swarm': metrics['swarm'],
            'robots': metrics['robots'],
        }
        with open(save_path / "metrics.json", 'w') as f:
            json.dump(metrics_clean, f, indent=2, default=str)
        
        # Generate plot
        self._generate_plot(save_path / "swarm_test.png")
        
        # Save event log
        with open(save_path / "events.txt", 'w') as f:
            f.write("SWARM EVENT LOG\n")
            f.write("=" * 60 + "\n")
            for event in self.host.events:
                f.write(str(event) + "\n")
        
        print(f"\nResults saved to {save_path}")
    
    def _generate_plot(self, save_path: Path):
        """Generate summary plot."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        fig.suptitle(f'Swarm Coordination Test\n'
                    f'{self.num_robots} robots | {self.kinematics} | {self.model}',
                    fontsize=14)
        
        times = np.array(self.history['times'])
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_robots))
        
        # Trajectories
        ax = axes[0, 0]
        obj_bounds = self.generic_object.geometry.bounds
        ax.fill([obj_bounds[0], obj_bounds[2], obj_bounds[2], obj_bounds[0]],
               [obj_bounds[1], obj_bounds[1], obj_bounds[3], obj_bounds[3]],
               alpha=0.3, color='green', label='Object')
        
        for (name, positions), color in zip(self.history['robot_positions'].items(), colors):
            pos_arr = np.array(positions)
            if len(pos_arr) > 0:
                ax.plot(pos_arr[:, 0], pos_arr[:, 1], '-', 
                       color=color, linewidth=1.5, label=name)
                ax.plot(pos_arr[0, 0], pos_arr[0, 1], 'o', color=color, markersize=8)
                ax.plot(pos_arr[-1, 0], pos_arr[-1, 1], 's', color=color, markersize=8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectories')
        ax.legend(fontsize=8, loc='upper right')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # State timeline
        ax = axes[0, 1]
        state_colors = {
            'IDLE': 'gray',
            'REACHING': 'blue',
            'APPROACHING': 'cyan',
            'WAITING': 'orange',
            'PUSHING': 'green',
            'CHANGING': 'purple',
        }
        
        for i, (name, states) in enumerate(self.history['robot_states'].items()):
            y_offset = i * 0.15
            for j, state in enumerate(states):
                if j < len(times):
                    ax.scatter(times[j], y_offset, 
                              c=state_colors.get(state, 'black'),
                              s=5, alpha=0.7)
        
        # Legend for states
        for state_name, color in state_colors.items():
            ax.scatter([], [], c=color, label=state_name)
        ax.legend(fontsize=8, loc='upper right')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Robot (offset)')
        ax.set_title('Robot State Timeline')
        ax.grid(True, alpha=0.3)
        
        # Distance to target
        ax = axes[0, 2]
        for (name, distances), color in zip(self.history['distances_to_target'].items(), colors):
            distances_arr = np.array(distances)
            # Replace inf with NaN for plotting
            distances_arr = np.where(np.isfinite(distances_arr), distances_arr, np.nan)
            ax.plot(times, distances_arr, '-', color=color, linewidth=1.5, label=name)
        
        ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Threshold (0.05m)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to Target (m)')
        ax.set_title('Distance to Target')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Contact status
        ax = axes[1, 0]
        for (name, contacts), color in zip(self.history['contacts'].items(), colors):
            contacts_arr = np.array(contacts).astype(float)
            ax.fill_between(times, 0, contacts_arr, alpha=0.3, color=color)
            ax.plot(times, contacts_arr, '-', color=color, linewidth=1, label=name)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('In Contact')
        ax.set_title('Contact Status')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)
        
        # Contact forces
        ax = axes[1, 1]
        for (name, forces), color in zip(self.history['contact_forces'].items(), colors):
            forces_arr = np.array(forces)
            ax.plot(times, forces_arr, '-', color=color, linewidth=1.5, label=name)
        
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Min (0.1N)')
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Max (2.0N)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact Force (N)')
        ax.set_title('Contact Forces')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Swarm state
        ax = axes[1, 2]
        swarm_states = self.history['swarm_states']
        state_nums = {
            'IDLE': 0, 'REACHING': 1, 'APPROACHING': 2, 'WAITING': 3, 'PUSHING': 4, 'CHANGING': 5
        }
        state_values = [state_nums.get(s, 0) for s in swarm_states]
        ax.step(times, state_values, where='post', linewidth=2, color='purple')
        ax.set_yticks(list(state_nums.values()))
        ax.set_yticklabels(list(state_nums.keys()))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Swarm State')
        ax.set_title('Swarm State Machine')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Swarm Coordination Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic navigation test with 3 robots (object static)
    python test_swarm_coordination.py --num-robots 3
    
    # Test with faster reconfiguration 
    python test_swarm_coordination.py --num-robots 4 --reconfig-interval 3
    
    # Test collision avoidance in isolation
    python test_swarm_coordination.py --test collision_avoidance
    
    # Enable actual pushing (object will move)
    python test_swarm_coordination.py --num-robots 3 --enable-pushing
    
    # Full test with output
    python test_swarm_coordination.py --num-robots 5 --duration 30 --save-dir /tmp/swarm/
        """
    )
    parser.add_argument("--num-robots", "-n", type=int, default=3,
                       help="Number of robots (default: 3)")
    parser.add_argument("--kinematics", "-k", default="holonomic",
                       choices=['holonomic', 'diffdrive'],
                       help="Kinematics type (default: holonomic)")
    parser.add_argument("--model", "-m", default="dummy",
                       choices=['dummy', 'wheel'],
                       help="Robot model (default: dummy)")
    parser.add_argument("--duration", "-d", type=float, default=30.0,
                       help="Test duration in seconds (default: 30.0)")
    parser.add_argument("--reconfig-interval", type=float, default=5.0,
                       help="Reconfiguration interval in seconds (default: 5.0)")
    parser.add_argument("--no-reconfig", action="store_true",
                       help="Disable automatic reconfiguration")
    parser.add_argument("--object", type=str, default="rectangle",
                       help="Object shape (default: rectangle)")
    parser.add_argument("--enable-pushing", action="store_true",
                       help="Enable pushing mode (object will move)")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run without GUI")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save results")
    args = parser.parse_args()
    

    
    # Setup PyBullet
    print("\nInitializing PyBullet...")
    ground = setup_pybullet(gui=not args.no_gui)
    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
    
    # Handle reconfig interval
    reconfig_interval = None if args.no_reconfig else args.reconfig_interval
    
    # Create and run test
    test = SwarmCoordinationTest(
        num_robots=args.num_robots,
        object_shape=args.object,
        kinematics=args.kinematics,
        model=args.model,
        duration=args.duration,
        reconfig_interval=reconfig_interval,
        enable_pushing=args.enable_pushing,
    )
    
    metrics = test.run_test(gui=not args.no_gui)
    
    # Print results
    print("\n" + "=" * 60)
    print("  SWARM METRICS")
    print("=" * 60)
    print(f"  Mode: {'PUSHING' if args.enable_pushing else 'NAVIGATION ONLY'}")
    print(f"  Final swarm state: {metrics['swarm']['swarm_state']}")
    print(f"  Reconfigurations: {test.history.get('reconfig_count', 0)}")
    if args.enable_pushing:
        print(f"  Time to first push: {metrics['swarm']['time_to_first_push']}")
        print(f"  Robots in contact: {metrics['swarm']['robots_in_contact']}/{args.num_robots}")
    print(f"  Total events: {metrics['swarm']['total_events']}")
    print("-" * 60)
    for name, m in metrics['robots'].items():
        status = test.host.robot_statuses.get(name)
        robot_state = test.host.robot_states.get(name, RobotState.IDLE)
        at_target = robot_state in [RobotState.WAITING, RobotState.PUSHING]
        dist = status.distance_to_target if status else float('inf')
        print(f"  {name}: state={robot_state.name}, dist={dist:.3f}m, contact_ratio={m['contact_ratio']*100:.0f}%", end="")
        if m['time_to_first_contact']:
            print(f", first_contact={m['time_to_first_contact']:.2f}s", end="")
        print()
    print("=" * 60)
    
    # Save results
    if args.save_dir:
        test.save_results(args.save_dir, metrics)
    
    # Print event log
    test.host.print_events()
    
    # Keep open if GUI
    if not args.no_gui:
        print("\nPress Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()

