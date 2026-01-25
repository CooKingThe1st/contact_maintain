#!/usr/bin/env python3
"""Contact maintenance simulation with observation and analysis."""
import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as pyb
import pybullet_data
import matplotlib.pyplot as plt

import rospkg

# Add the package to path
import sys
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))

from contact_maintain.robots import HolonomicRobot, DifferentialDriveRobot
from contact_maintain.control import (
    HolonomicVelocityController, 
    DifferentialDriveController,
    ContactMaintainController
)
from contact_maintain.solvers import (
    ForceBasedContactSolver,
    PositionBasedContactSolver,
    AdaptiveContactSolver,
    DiffDriveForceBasedSolver,
    DiffDrivePositionBasedSolver,
    create_solver,
)
from contact_maintain.observer import ContactObserver, ContactPointTracker
from contact_maintain.logging import DataLogger
from contact_maintain.visualization import plot_contact_analysis, plot_scene_snapshot
from contact_maintain.pyb_simulation import BulletCircleSlider, BulletSquareSlider


# Simulation parameters
TIMESTEP = 0.01
DURATION = 30.0

# Robot parameters (scaled small robot)
ROBOT_RADIUS = 0.06   # Body radius
BUMPER_OFFSET = 0.055 # Bumper distance from center
CONTACT_MU = 0.8
SURFACE_MU = 0.3

# Object parameters (scaled for small robot, objects are 2.5x larger)
OBJECT_MASS = 10.0
OBJECT_RADIUS = 0.5   # For circular slider (0.2 * 2.5)

# Contact parameters
FORCE_THRESHOLD = 0.5
TARGET_FORCE = 5.0


def setup_simulation(timestep=TIMESTEP, gui=True):
    """Initialize PyBullet simulation."""
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(timestep)
    
    # Camera setup
    pyb.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=0,
        cameraPitch=-45,
        cameraTargetPosition=[0.5, 0.0, 0],
    )
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    
    # Ground plane
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_uid = pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.changeDynamics(ground_uid, -1, lateralFriction=SURFACE_MU)
    
    return client_id, ground_uid


def run_contact_maintain_with_force(use_diff_drive=False, gui=True, save_data=False):
    """Run contact maintenance simulation with force sensor feedback.
    
    Parameters
    ----------
    use_diff_drive : bool
        If True, use differential-drive robot. Otherwise, use holonomic.
    gui : bool
        Whether to show GUI.
    save_data : bool
        Whether to save logged data.
    
    Returns
    -------
    dict
        Dictionary containing observer data and statistics.
    """
    print(f"\n{'='*60}")
    robot_type = "Differential-Drive" if use_diff_drive else "Holonomic"
    print(f"Contact Maintenance WITH Force Sensor - {robot_type} Robot")
    print(f"{'='*60}")
    
    # Setup
    client_id, ground_uid = setup_simulation(gui=gui)
    
    # Load robot
    urdf_path = str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf")
    
    if use_diff_drive:
        robot = DifferentialDriveRobot(
            urdf_path, position=(0, 0), orientation=0, contact_mu=CONTACT_MU
        )
    else:
        robot = HolonomicRobot(
            urdf_path, position=(0, 0), orientation=0, contact_mu=CONTACT_MU
        )
    
    # Create object
    object_pos = [0.6, 0.0, OBJECT_RADIUS]
    slider = BulletCircleSlider(
        position=object_pos,
        mass=OBJECT_MASS,
        radius=OBJECT_RADIUS,
        height=OBJECT_RADIUS * 2
    )
    
    # Observer and logger
    observer = ContactObserver(force_threshold=FORCE_THRESHOLD)
    logger = DataLogger(log_dir=Path(pkg_path) / "data", 
                       experiment_name=f"contact_with_force_{robot_type.lower()}")
    logger.add_metadata("robot_type", robot_type)
    logger.add_metadata("has_force_sensor", True)
    logger.add_metadata("duration", DURATION)
    
    # Solver for force-based contact maintenance
    solver = create_solver(
        solver_type='force',
        robot_type='diffdrive' if use_diff_drive else 'holonomic',
        target_force=TARGET_FORCE,
        kp_force=0.02,
        ki_force=0.001,
        kd_force=0.005,
        force_threshold=FORCE_THRESHOLD,
        approach_speed=0.15,
        maintain_speed=0.1
    )
    
    print(f"Robot starting at origin, object at ({object_pos[0]:.2f}, {object_pos[1]:.2f})")
    print(f"Target contact force: {TARGET_FORCE}N")
    print("Running simulation...")
    
    t = 0
    while t < DURATION:
        # Get robot state
        robot_pos, robot_theta, robot_vel = robot.get_state()
        contact_pos = robot.get_contact_position()
        
        # Get contact force
        contact_force = robot.get_contact_force([slider.uid])
        
        # Get object state
        object_position = slider.get_pose()[0]
        object_vel_lin, object_vel_ang = pyb.getBaseVelocity(slider.uid)
        object_vel = np.array(object_vel_lin)
        
        # Update observer
        state = observer.update(
            timestamp=t,
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            robot_vel=robot_vel,
            contact_pos=contact_pos,
            contact_force=contact_force,
            object_pos=object_position,
            object_vel=object_vel
        )
        
        # Compute control using force-based solver
        cmd = solver.compute_velocity(
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            object_pos=object_position[:2],
            contact_force=contact_force,
            dt=TIMESTEP
        )
        
        # Apply control
        if use_diff_drive:
            # Solver returns (v, omega) tuple for diff-drive
            robot.command_velocity(cmd)
        else:
            # Solver returns (vx, vy, omega) array for holonomic
            robot.command_velocity(cmd)
        
        # Log data
        logger.log(
            t=t,
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            contact_force=contact_force,
            object_pos=object_position,
            in_contact=state.in_contact
        )
        
        # Step simulation
        pyb.stepSimulation()
        t += TIMESTEP
        
        if gui:
            time.sleep(TIMESTEP * 0.5)  # Slow down for visualization
    
    # Get results
    stats = observer.get_statistics()
    history = observer.get_history_arrays()
    
    print(f"\n--- Results ---")
    print(f"Contact ratio: {stats['contact_ratio']*100:.1f}%")
    print(f"Contact events: {stats['contact_count']}")
    print(f"Contact lost: {stats['contact_lost_count']}")
    print(f"Mean force: {stats['mean_force']:.2f}N")
    print(f"Max force: {stats['max_force']:.2f}N")
    
    if save_data:
        logger.save_pickle()
    
    pyb.disconnect()
    
    return {'history': history, 'stats': stats}


def run_contact_maintain_without_force(use_diff_drive=False, gui=True, save_data=False):
    """Run contact maintenance simulation WITHOUT force sensor feedback.
    
    Uses position-based estimation to maintain contact.
    
    Parameters
    ----------
    use_diff_drive : bool
        If True, use differential-drive robot. Otherwise, use holonomic.
    gui : bool
        Whether to show GUI.
    save_data : bool
        Whether to save logged data.
    
    Returns
    -------
    dict
        Dictionary containing observer data and statistics.
    """
    print(f"\n{'='*60}")
    robot_type = "Differential-Drive" if use_diff_drive else "Holonomic"
    print(f"Contact Maintenance WITHOUT Force Sensor - {robot_type} Robot")
    print(f"{'='*60}")
    
    # Setup
    client_id, ground_uid = setup_simulation(gui=gui)
    
    # Load robot
    urdf_path = str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf")
    
    if use_diff_drive:
        robot = DifferentialDriveRobot(
            urdf_path, position=(0, 0), orientation=0, contact_mu=CONTACT_MU
        )
    else:
        robot = HolonomicRobot(
            urdf_path, position=(0, 0), orientation=0, contact_mu=CONTACT_MU
        )
    
    # Create object
    object_pos = [0.6, 0.0, OBJECT_RADIUS]
    slider = BulletCircleSlider(
        position=object_pos,
        mass=OBJECT_MASS,
        radius=OBJECT_RADIUS,
        height=OBJECT_RADIUS * 2
    )
    
    # Observer and logger
    observer = ContactObserver(force_threshold=FORCE_THRESHOLD)
    contact_tracker = ContactPointTracker(object_radius=OBJECT_RADIUS)
    logger = DataLogger(log_dir=Path(pkg_path) / "data",
                       experiment_name=f"contact_no_force_{robot_type.lower()}")
    logger.add_metadata("robot_type", robot_type)
    logger.add_metadata("has_force_sensor", False)
    logger.add_metadata("duration", DURATION)
    
    # Position-based solver (no force feedback)
    solver = create_solver(
        solver_type='position',
        robot_type='diffdrive' if use_diff_drive else 'holonomic',
        robot_radius=ROBOT_RADIUS,
        object_radius=OBJECT_RADIUS,
        contact_offset=BUMPER_OFFSET,  # Bumper is at this offset from robot center
        kp_distance=1.0,
        approach_speed=0.1,
        maintain_speed=0.05
    )
    
    print(f"Robot starting at origin, object at ({object_pos[0]:.2f}, {object_pos[1]:.2f})")
    print("Using position-based contact estimation (no force sensor)")
    print("Running simulation...")
    
    t = 0
    while t < DURATION:
        # Get robot state
        robot_pos, robot_theta, robot_vel = robot.get_state()
        contact_pos = robot.get_contact_position()
        
        # Get actual contact force (for logging/analysis, not used in control)
        contact_force = robot.get_contact_force([slider.uid])
        
        # Get object state
        object_position = slider.get_pose()[0]
        object_vel_lin, object_vel_ang = pyb.getBaseVelocity(slider.uid)
        object_vel = np.array(object_vel_lin)
        
        # Update observer
        state = observer.update(
            timestamp=t,
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            robot_vel=robot_vel,
            contact_pos=contact_pos,
            contact_force=contact_force,
            object_pos=object_position,
            object_vel=object_vel
        )
        
        # Position-based contact maintenance (no force feedback)
        # Note: contact_force is passed for logging only, not used in control
        cmd = solver.compute_velocity(
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            object_pos=object_position[:2],
            contact_force=contact_force  # For tracking actual contact, not control
        )
        
        # Apply control
        if use_diff_drive:
            # Solver returns (v, omega) tuple for diff-drive
            robot.command_velocity(cmd)
        else:
            # Solver returns (vx, vy, omega) array for holonomic
            robot.command_velocity(cmd)
        
        # Log data
        logger.log(
            t=t,
            robot_pos=robot_pos,
            robot_theta=robot_theta,
            contact_force=contact_force,
            object_pos=object_position,
            in_contact=state.in_contact,
            solver_in_contact=solver.in_contact
        )
        
        # Step simulation
        pyb.stepSimulation()
        t += TIMESTEP
        
        if gui:
            time.sleep(TIMESTEP * 0.5)
    
    # Get results
    stats = observer.get_statistics()
    history = observer.get_history_arrays()
    
    print(f"\n--- Results ---")
    print(f"Contact ratio: {stats['contact_ratio']*100:.1f}%")
    print(f"Contact events: {stats['contact_count']}")
    print(f"Contact lost: {stats['contact_lost_count']}")
    print(f"Mean force: {stats['mean_force']:.2f}N")
    print(f"Max force: {stats['max_force']:.2f}N")
    
    if save_data:
        logger.save_pickle()
    
    pyb.disconnect()
    
    return {'history': history, 'stats': stats}


def main():
    parser = argparse.ArgumentParser(
        description="Contact maintenance simulation"
    )
    parser.add_argument(
        "--mode",
        choices=["with_force", "without_force", "compare", "all"],
        default="with_force",
        help="Simulation mode"
    )
    parser.add_argument(
        "--robot",
        choices=["holonomic", "diffdrive", "both"],
        default="holonomic",
        help="Robot type to use"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save logged data"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show analysis plots after simulation"
    )
    args = parser.parse_args()
    
    gui = not args.no_gui
    results = {}
    
    robots = []
    if args.robot == "both":
        robots = [False, True]  # holonomic, diff-drive
    else:
        robots = [args.robot == "diffdrive"]
    
    for use_diff_drive in robots:
        robot_name = "diffdrive" if use_diff_drive else "holonomic"
        
        if args.mode in ["with_force", "compare", "all"]:
            key = f"with_force_{robot_name}"
            results[key] = run_contact_maintain_with_force(
                use_diff_drive=use_diff_drive, gui=gui, save_data=args.save
            )
        
        if args.mode in ["without_force", "compare", "all"]:
            key = f"without_force_{robot_name}"
            results[key] = run_contact_maintain_without_force(
                use_diff_drive=use_diff_drive, gui=gui, save_data=args.save
            )
    
    # Show comparison plots
    if args.plot and results:
        print("\nGenerating analysis plots...")
        
        for name, data in results.items():
            if 'history' in data and data['history']:
                fig = plot_contact_analysis(data['history'])
                fig.suptitle(f"Contact Analysis: {name.replace('_', ' ').title()}")
                plt.show()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, data in results.items():
        stats = data['stats']
        print(f"\n{name}:")
        print(f"  Contact ratio: {stats['contact_ratio']*100:.1f}%")
        print(f"  Mean force: {stats['mean_force']:.2f}N")


if __name__ == "__main__":
    main()

