#!/usr/bin/env python3
"""Test script for holonomic robot in PyBullet simulation."""
import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as pyb
import pybullet_data

import rospkg

# Add the package to path
import sys
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))

from contact_maintain.robots import HolonomicRobot, DifferentialDriveRobot
from contact_maintain.control import HolonomicVelocityController, DifferentialDriveController
from contact_maintain.pyb_simulation import BulletCircleSlider


def setup_simulation(timestep=0.01, gui=True):
    """Initialize PyBullet simulation."""
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(timestep)
    
    # Camera setup
    pyb.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0.5, 0],
    )
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    
    # Ground plane
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_uid = pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.changeDynamics(ground_uid, -1, lateralFriction=0.5)
    
    return client_id, ground_uid


def test_holonomic_robot():
    """Test holonomic robot motion."""
    print("Testing Holonomic Robot...")
    
    # Setup
    client_id, ground_uid = setup_simulation()
    timestep = 0.01
    
    # Load robot
    urdf_path = str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf")
    robot = HolonomicRobot(urdf_path, position=(0, 0), orientation=0)
    
    # Controller
    controller = HolonomicVelocityController(
        kp_pos=1.0, 
        kp_theta=1.0,
        max_linear_vel=0.5,
        max_angular_vel=1.0
    )
    
    # Target waypoints
    waypoints = [
        (1.0, 0.0, 0),
        (1.0, 1.0, np.pi/2),
        (0.0, 1.0, np.pi),
        (0.0, 0.0, 0),
    ]
    
    current_waypoint = 0
    t = 0
    duration = 20.0
    
    print(f"Moving through {len(waypoints)} waypoints...")
    
    while t < duration:
        # Get current state
        pos, theta, vel = robot.get_state()
        
        # Get target
        target_pos = waypoints[current_waypoint][:2]
        target_theta = waypoints[current_waypoint][2]
        
        # Check if waypoint reached
        dist_to_target = np.linalg.norm(np.array(target_pos) - pos)
        if dist_to_target < 0.1:
            print(f"  Reached waypoint {current_waypoint + 1}: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
            current_waypoint = (current_waypoint + 1) % len(waypoints)
            if current_waypoint == 0:
                print("  Completed all waypoints!")
        
        # Compute control
        cmd_vel = controller.position_control(pos, target_pos, theta, target_theta)
        robot.command_velocity(cmd_vel)
        
        # Step simulation
        pyb.stepSimulation()
        t += timestep
        time.sleep(timestep)
    
    pyb.disconnect()
    print("Holonomic robot test completed!\n")


def test_differential_drive_robot():
    """Test differential-drive robot motion."""
    print("Testing Differential-Drive Robot...")
    
    # Setup
    client_id, ground_uid = setup_simulation()
    timestep = 0.01
    
    # Load robot (uses same URDF but with diff-drive constraints)
    urdf_path = str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf")
    robot = DifferentialDriveRobot(urdf_path, position=(0, 0), orientation=0)
    
    # Controller
    controller = DifferentialDriveController(
        kp_distance=1.0,
        kp_heading=2.0,
        max_linear_vel=0.5,
        max_angular_vel=1.0,
        goal_tolerance=0.1
    )
    
    # Target waypoints
    waypoints = [
        (1.0, 0.0, None),
        (1.0, 1.0, None),
        (0.0, 1.0, None),
        (0.0, 0.0, 0),
    ]
    
    current_waypoint = 0
    t = 0
    duration = 30.0
    
    print(f"Moving through {len(waypoints)} waypoints...")
    
    while t < duration:
        # Get current state
        pos, theta, vel = robot.get_state()
        
        # Get target
        target_pos = waypoints[current_waypoint][:2]
        target_theta = waypoints[current_waypoint][2]
        
        # Check if waypoint reached
        dist_to_target = np.linalg.norm(np.array(target_pos) - pos)
        if dist_to_target < 0.15:
            print(f"  Reached waypoint {current_waypoint + 1}: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
            current_waypoint = (current_waypoint + 1) % len(waypoints)
            if current_waypoint == 0:
                print("  Completed all waypoints!")
        
        # Compute control
        v, omega = controller.position_control(pos, theta, target_pos, target_theta)
        robot.command_velocity((v, omega))
        
        # Step simulation
        pyb.stepSimulation()
        t += timestep
        time.sleep(timestep)
    
    pyb.disconnect()
    print("Differential-drive robot test completed!\n")


def test_contact_scenario():
    """Test robot approaching and contacting an object."""
    print("Testing Contact Scenario...")
    
    # Setup
    client_id, ground_uid = setup_simulation()
    timestep = 0.01
    
    # Load robot
    urdf_path = str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf")
    robot = HolonomicRobot(urdf_path, position=(0, 0), orientation=0, contact_mu=1.0)
    
    # Create object to push
    object_pos = [0.5, 0.0, 0.15]
    slider = BulletCircleSlider(
        position=object_pos,
        mass=1.0,
        radius=0.15,
        height=0.3
    )
    
    # Controller
    controller = HolonomicVelocityController(max_linear_vel=0.3)
    
    t = 0
    duration = 15.0
    
    print("Robot approaching object...")
    
    while t < duration:
        # Get states
        robot_pos, robot_theta, _ = robot.get_state()
        contact_pos = robot.get_contact_position()
        contact_force = robot.get_contact_force([slider.uid])
        object_position = slider.get_pose()[0][:2]
        
        # Move toward object
        direction = np.array(object_position) - robot_pos
        dist = np.linalg.norm(direction)
        
        force_magnitude = np.linalg.norm(contact_force[:2])
        
        if force_magnitude > 1.0:
            # In contact - maintain gentle pressure
            vel = 0.05 * direction / (dist + 0.01)
            status = f"IN CONTACT - Force: {force_magnitude:.2f}N"
        else:
            # Approach object
            vel = 0.2 * direction / (dist + 0.01)
            status = f"Approaching - Dist: {dist:.3f}m"
        
        cmd_vel = controller.velocity_tracking(np.append(vel, 0))
        robot.command_velocity(cmd_vel)
        
        # Print status every second
        if int(t * 10) % 10 == 0:
            print(f"  t={t:.1f}s: {status}")
        
        # Step simulation
        pyb.stepSimulation()
        t += timestep
        time.sleep(timestep)
    
    pyb.disconnect()
    print("Contact scenario test completed!\n")


def main():
    parser = argparse.ArgumentParser(description="Test robots in PyBullet simulation")
    parser.add_argument(
        "--test", 
        choices=["holonomic", "diffdrive", "contact", "all"],
        default="all",
        help="Which test to run"
    )
    args = parser.parse_args()
    
    if args.test == "holonomic" or args.test == "all":
        test_holonomic_robot()
    
    if args.test == "diffdrive" or args.test == "all":
        test_differential_drive_robot()
    
    if args.test == "contact" or args.test == "all":
        test_contact_scenario()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()

