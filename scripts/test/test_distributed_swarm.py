#!/usr/bin/env python3
"""
Distributed Swarm Architecture Test

Tests the new distributed swarm architecture with three navigation schemes:
- APF: Rewritten APF navigation optimized for pushing
- Static Single: Only one robot moves at a time
- Divide-n-Conquer: Each robot manages non-overlapping consecutive edges

This test script demonstrates the new architecture:
1. OverworldSimulator coordinates all robots
2. Each robot has its own DistributedMonitor
3. Navigation and pushing controllers are modular and swappable

Usage:
    python test_distributed_swarm.py --navigation-scheme apf --object rectangle --duration 30
    python test_distributed_swarm.py --navigation-scheme static_single --object right_triangle
    python test_distributed_swarm.py --navigation-scheme divide_conquer --no-gui --duration 20
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

# Use non-interactive backend for headless mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import ContactPointParameterization, ContactPoint
from contact_maintain.robot_factory import create_robot
from contact_maintain.object_bridge import obj_to_generic
from contact_maintain.overworld_sim import OverworldSimulator
from contact_optimizer_utils import find_the_magnum_four_v3


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

DEFAULT_OBJECT_HEIGHT = 0.08
DEFAULT_OBJECT_FRICTION = 0.3
ROBOT_RADIUS = 0.06
APPROACH_DISTANCE = ROBOT_RADIUS + 0.02


def setup_pybullet(gui: bool = True):
    """Setup PyBullet simulation."""
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    
    urdf_dir = Path(pkg_path) / "urdf"
    pyb.setAdditionalSearchPath(str(urdf_dir))
    
    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=4,
            cameraYaw=-5,
            cameraPitch=-85,
            cameraTargetPosition=[0, 0, 0]
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_TINY_RENDERER, 0)
    
    return ground


def get_object_state(object_uid):
    """Get object state from PyBullet."""
    pos, orn = pyb.getBasePositionAndOrientation(object_uid)
    vel_lin, vel_ang = pyb.getBaseVelocity(object_uid)
    euler = pyb.getEulerFromQuaternion(orn)
    return {
        "position": np.array([pos[0], pos[1]]),
        "orientation": euler[2],
        "velocity": np.array([vel_lin[0], vel_lin[1]]),
        "angular_velocity": vel_ang[2],
    }


def main():
    parser = argparse.ArgumentParser(description="Distributed Swarm Architecture Test")
    parser.add_argument("--object", type=str, default="right_triangle",
                        choices=["right_triangle", "bolt", "pi", "root", "rect", "hourglass", "meteor"],
                        help="Object shape name")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Test duration in seconds")
    parser.add_argument("--no-gui", action="store_true", help="Run headless")
    parser.add_argument("--kinematics", "-k", default="holonomic",
                        choices=["holonomic", "diffdrive"])
    parser.add_argument("--model", "-m", default="wheel", choices=["wheel"],
                        help="Robot model")
    parser.add_argument("--navigation-scheme", type=str, default="apf",
                        choices=["apf", "static_single", "divide_conquer"],
                        help="Navigation scheme to test")
    parser.add_argument(
        "--startup-mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help=(
            "Startup behavior: 'quick' uses direct rotate-then-creep approach to contact; "
            "'full' uses navigation-scheme startup."
        ),
    )
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Setup PyBullet
    print("\nInitializing PyBullet...")
    setup_pybullet(gui=not args.no_gui)
    
    # Load object
    obj_file_map = {
        'right_triangle': 'right_triangle.obj',
        'bolt': 'bolt.obj',
        'pi': 'pi.obj',
        'root': 'root.obj',
        'rect': 'rect.obj',
        "hourglass": "hourglass.obj",
        "meteor": "meteor.obj",
    }
    
    if args.object not in obj_file_map:
        raise ValueError(f"Unknown object '{args.object}'")
    
    obj_file = obj_file_map[args.object]
    print(f"Loading object: {obj_file}...")
    
    generic_object, object_uid = obj_to_generic(
        obj_path=obj_file,
        shape_name=args.object,
        position=(0, 0, 0.2),
        orientation=0.0,
        mass=1.0,
        lateral_friction=DEFAULT_OBJECT_FRICTION,
        blind_test=True,
    )
    
    contact_point_parameterization = ContactPointParameterization(generic_object)
    print(f"✓ Loaded object: {args.object}")
    
    # Compute Magnum Four contacts
    print(f"\nComputing Magnum Four contact points...")
    magnum_result = find_the_magnum_four_v3(
        generic_object,
        verbose=False,
        visualize=False,
        weighting_scheme="balanced",
        torque_method=3,
    )
    
    if not magnum_result or not magnum_result.get("success", False):
        raise RuntimeError("Magnum Four solver failed")
    
    contacts = magnum_result["best_solution"]["contacts"]
    t_params = [float(c.parameter) % 1.0 for c in contacts]
    
    if len(t_params) != 4:
        raise RuntimeError(f"Expected 4 contacts, got {len(t_params)}")
    
    print(f"Magnum Four t_params: {[f'{v:.4f}' for v in t_params]}")
    
    # Create robots
    robots: Dict[str, object] = {}
    for i in range(4):
        name = f"R_{i+1:02d}"
        target_t_param = t_params[i]
        
        # Get contact point info
        contact_info = contact_point_parameterization.get_contact_info(target_t_param)
        contact_point_body = np.array(contact_info['point'], dtype=float)
        normal_outward = np.array(contact_info['normal_outward'], dtype=float)
        normal_inward = np.array(contact_info['normal_inward'], dtype=float)
        
        # Calculate spawn position
        spawn_position_body = contact_point_body + APPROACH_DISTANCE * normal_outward
        robot_x = float(spawn_position_body[0])
        robot_y = float(spawn_position_body[1])
        robot_heading = float(np.arctan2(normal_inward[1], normal_inward[0]))
        
        robot = create_robot(
            kinematics=args.kinematics,
            model=args.model,
            position=(robot_x, robot_y),
            orientation=robot_heading,
            name=name,
        )
        robots[name] = robot
        print(f"Spawned {name} at ({robot_x:.3f}, {robot_y:.3f}), target t={target_t_param:.4f}")
    
    # Create OverworldSimulator
    print(f"\nCreating OverworldSimulator with navigation scheme: {args.navigation_scheme}")
    overworld = OverworldSimulator(
        robots=robots,
        object_uid=object_uid,
        generic_object=generic_object,
        navigation_scheme=args.navigation_scheme,
        push_controller_type='phase7',
        startup_mode=args.startup_mode,
    )
    print(f"Startup mode: {args.startup_mode}")
    
    # Assign initial targets
    target_map = {name: t_params[i] for i, name in enumerate(robots.keys())}
    print(f"Assigning targets: { {k: round(v, 4) for k, v in target_map.items()} }")
    overworld.assign_targets(target_map)
    
    # Run simulation
    print(f"\n{'='*60}")
    print(f"  RUNNING DISTRIBUTED SWARM TEST")
    print(f"{'='*60}")
    print(f"  Navigation scheme: {args.navigation_scheme}")
    print(f"  Duration: {args.duration}s")
    print(f"  Object: {args.object}")
    print(f"{'='*60}\n")
    
    n_steps = int(args.duration / TIMESTEP)
    step_count = 0
    t = 0.0
    
    for _ in range(n_steps):
        obj_state = get_object_state(object_uid)
        
        if step_count % CTRL_STEP == 0:
            # Update overworld simulator
            overworld.update(1.0 / CTRL_FREQ, obj_state)
            
            # Compute velocities for all robots
            velocities = overworld.compute_velocities(obj_state)
            
            # Command velocities to robots
            for name, robot in robots.items():
                cmd = velocities[name]
                
                if args.kinematics == "diffdrive" and len(cmd) == 3:
                    pos, heading, _ = robot.get_state()
                    v_forward = cmd[0] * np.cos(heading) + cmd[1] * np.sin(heading)
                    robot.command_velocity(np.array([v_forward, cmd[2]]))
                else:
                    robot.command_velocity(cmd)
            
            # Print status periodically
            if step_count % (CTRL_STEP * 50) == 0:  # Every 0.5 seconds
                status = overworld.get_status()
                all_pushing = overworld.get_all_in_pushing()
                all_at_target = overworld.get_all_at_target()
                print(f"[t={t:.1f}s] All pushing: {all_pushing}, All at target: {all_at_target}")
                for name, s in status.items():
                    print(f"  {name}: {s['state']}, contact={s['in_contact']}, "
                          f"force={s['contact_force']:.2f}N, dist={s['distance_to_target']:.3f}m")
        
        pyb.stepSimulation()
        t += TIMESTEP
        step_count += 1
        
        if not args.no_gui:
            time.sleep(TIMESTEP * 0.3)
    
    # Final status
    print(f"\n{'='*60}")
    print(f"  FINAL STATUS")
    print(f"{'='*60}")
    final_status = overworld.get_status()
    for name, s in final_status.items():
        print(f"  {name}: {s['state']}, contact={s['in_contact']}, "
              f"force={s['contact_force']:.2f}N")
    print(f"{'='*60}\n")
    
    pyb.disconnect()
    print("Done!")


if __name__ == "__main__":
    main()
