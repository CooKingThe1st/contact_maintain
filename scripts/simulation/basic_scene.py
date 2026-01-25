#!/usr/bin/env python3
"""
Basic PyBullet scene with Robotino 3 robot and object.

Features:
- Full state tracking (robot pose/velocity, object pose/velocity, contact state)
- Web-based observer for real-time monitoring
- Simple P controller to move toward and contact object
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as pyb
import pybullet_data
import pyb_utils

import rospkg

# Add the package to path
import sys
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))

from contact_maintain.robots import HolonomicRobot
from contact_maintain.pyb_simulation import BulletSquareSlider, get_contact_force
from contact_maintain.web_observer import WebObserver


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

TIMESTEP = 1.0 / 240.0  # 240 Hz physics
CTRL_FREQ = 100  # 100 Hz control
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

# Robot parameters (Robotino 3)
ROBOT_NAME = "robotino_01"
ROBOT_START_POS = (0, 0)
ROBOT_START_HEADING = 0.0

# Object parameters
OBJECT_NAME = "box_01"
OBJECT_SIZE = (0.5, 0.3, 0.15)  # half-extents (1m x 0.6m x 0.3m box)
OBJECT_MASS = 20.0
OBJECT_START_POS = [1.2, 0.0, 0.15]

# Friction
SURFACE_MU = 0.3
CONTACT_MU = 0.8
OBJECT_MU = 0.4

# Controller parameters
APPROACH_SPEED = 0.15  # m/s
KP_POSITION = 1.0  # Position gain
KP_HEADING = 2.0  # Heading alignment gain

# Contact detection
FORCE_THRESHOLD = 0.5  # N


# ============================================================================
# STATE TRACKING
# ============================================================================

class SimulationState:
    """Tracks all simulation state."""
    
    def __init__(self):
        # Robot state
        self.robot_position = np.zeros(2)
        self.robot_velocity = np.zeros(3)  # vx, vy, omega
        self.robot_heading = 0.0
        self.robot_bumper_pos = np.zeros(3)
        
        # Object state
        self.object_position = np.zeros(3)
        self.object_velocity = np.zeros(3)
        self.object_orientation = 0.0
        self.object_angular_velocity = 0.0
        
        # Contact state
        self.in_contact = False
        self.contact_force = np.zeros(3)
        self.contact_force_magnitude = 0.0
        self.contact_direction = np.zeros(2)
        
        # Time
        self.time = 0.0
    
    def update_robot(self, robot: HolonomicRobot):
        """Update robot state from PyBullet."""
        pos, heading, vel = robot.get_state()
        self.robot_position = pos
        self.robot_heading = heading
        self.robot_velocity = vel
        self.robot_bumper_pos = robot.get_contact_position()
    
    def update_object(self, slider_uid):
        """Update object state from PyBullet."""
        pos, orn = pyb.getBasePositionAndOrientation(slider_uid)
        vel_lin, vel_ang = pyb.getBaseVelocity(slider_uid)
        
        self.object_position = np.array(pos)
        self.object_velocity = np.array(vel_lin)
        
        # Extract yaw from quaternion
        euler = pyb.getEulerFromQuaternion(orn)
        self.object_orientation = euler[2]  # yaw
        self.object_angular_velocity = vel_ang[2]  # omega_z
    
    def update_contact(self, robot: HolonomicRobot, slider_uid):
        """Update contact state from PyBullet."""
        # Get contact force
        self.contact_force = robot.get_contact_force([slider_uid])
        self.contact_force_magnitude = np.linalg.norm(self.contact_force[:2])
        self.in_contact = self.contact_force_magnitude > FORCE_THRESHOLD
        
        if self.contact_force_magnitude > 0:
            self.contact_direction = self.contact_force[:2] / self.contact_force_magnitude
        else:
            self.contact_direction = np.zeros(2)
    
    def print_state(self):
        """Print current state summary."""
        print(f"\n{'='*50}")
        print(f"Time: {self.time:.2f}s")
        print(f"Robot: pos=({self.robot_position[0]:.3f}, {self.robot_position[1]:.3f}), "
              f"heading={np.degrees(self.robot_heading):.1f}°")
        print(f"Object: pos=({self.object_position[0]:.3f}, {self.object_position[1]:.3f})")
        print(f"Contact: {'YES' if self.in_contact else 'NO'}, force={self.contact_force_magnitude:.2f}N")
        print(f"{'='*50}")


# ============================================================================
# CONTROLLER
# ============================================================================

class SimpleApproachController:
    """Simple P controller to approach and contact object."""
    
    def __init__(self, approach_speed=APPROACH_SPEED, kp_pos=KP_POSITION, kp_heading=KP_HEADING):
        self.approach_speed = approach_speed
        self.kp_pos = kp_pos
        self.kp_heading = kp_heading
    
    def compute_velocity(self, state: SimulationState) -> np.ndarray:
        """Compute velocity command to approach object.
        
        Returns (vx, vy, omega) in world frame.
        """
        # Direction to object
        to_object = state.object_position[:2] - state.robot_position
        distance = np.linalg.norm(to_object)
        
        if distance < 0.01:
            return np.array([0, 0, 0])
        
        direction = to_object / distance
        
        # Desired heading (face the object)
        desired_heading = np.arctan2(direction[1], direction[0])
        heading_error = np.arctan2(np.sin(desired_heading - state.robot_heading),
                                   np.cos(desired_heading - state.robot_heading))
        
        # Angular velocity for heading alignment
        omega = self.kp_heading * heading_error
        omega = np.clip(omega, -1.5, 1.5)
        
        # Linear velocity toward object
        # Reduce speed if not well aligned
        alignment = np.cos(heading_error)
        speed = self.approach_speed * max(0.3, alignment)
        
        # If in contact, reduce speed but keep pushing
        if state.in_contact:
            speed = min(speed, 0.05)
        
        vel = speed * direction
        
        return np.array([vel[0], vel[1], omega])


# ============================================================================
# SIMULATION SETUP
# ============================================================================

def setup_simulation(gui=True):
    """Initialize PyBullet simulation."""
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    
    # Camera setup
    pyb.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=30,
        cameraPitch=-40,
        cameraTargetPosition=[0.6, 0.0, 0.1],
    )
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 1)
    
    # Ground plane
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_uid = pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.changeDynamics(ground_uid, -1, lateralFriction=SURFACE_MU)
    
    return client_id, ground_uid


def check_quit_key():
    """Check if 'Q' key was pressed to quit simulation.
    
    Returns True if should quit, False otherwise.
    """
    keys = pyb.getKeyboardEvents()
    # 'q' key code is 113, 'Q' is 81
    if 113 in keys or 81 in keys:
        if keys.get(113, 0) & pyb.KEY_WAS_TRIGGERED or keys.get(81, 0) & pyb.KEY_WAS_TRIGGERED:
            return True
    return False


def create_robot():
    """Create the Robotino 3 robot."""
    urdf_path = str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf")
    robot = HolonomicRobot(
        urdf_path,
        position=ROBOT_START_POS,
        orientation=ROBOT_START_HEADING,
        contact_mu=CONTACT_MU
    )
    return robot


def create_object():
    """Create the box object."""
    slider = BulletSquareSlider(
        position=OBJECT_START_POS,
        mass=OBJECT_MASS,
        half_extents=OBJECT_SIZE,
    )
    pyb.changeDynamics(slider.uid, -1, lateralFriction=OBJECT_MU)
    return slider


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Basic contact maintenance scene")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--no-web", action="store_true", help="Disable web observer")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration (seconds)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  CONTACT MAINTENANCE - Basic Scene")
    print("="*60)
    
    # Setup simulation
    print("\nInitializing PyBullet...")
    gui = not args.no_gui
    client_id, ground_uid = setup_simulation(gui=gui)
    
    # Create robot and object
    print("Creating robot and object...")
    robot = create_robot()
    slider = create_object()
    
    # Initialize state tracker
    state = SimulationState()
    
    # Initialize controller
    controller = SimpleApproachController()
    
    # Initialize web observer
    observer = None
    if not args.no_web:
        print("Starting web observer...")
        observer = WebObserver(port=5000)
        observer.register_robot(ROBOT_NAME)
        observer.register_object(OBJECT_NAME)
        observer.start()
        print(f"\n>>> Web dashboard: http://localhost:5000 <<<\n")
    
    # Let simulation settle
    print("Settling simulation...")
    for _ in range(100):
        pyb.stepSimulation()
    
    print("\nSimulation running...")
    print("  - Robot will approach and contact the object")
    print("  - Press 'Q' in PyBullet window or Ctrl+C in terminal to stop")
    if observer:
        print(f"  - View real-time data at http://localhost:5000")
    print()
    
    # Main simulation loop
    t = 0.0
    step_count = 0
    last_print_time = 0.0
    
    try:
        while t < args.duration:
            # Check for quit key (Q)
            if gui and check_quit_key():
                print("\n'Q' pressed - stopping simulation...")
                break
            
            # Update state
            state.time = t
            state.update_robot(robot)
            state.update_object(slider.uid)
            state.update_contact(robot, slider.uid)
            
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Compute control
                cmd_vel = controller.compute_velocity(state)
                robot.command_velocity(cmd_vel)
                
                # Update web observer
                if observer:
                    observer.update_robot(
                        name=ROBOT_NAME,
                        position=state.robot_position,
                        velocity=state.robot_velocity,
                        heading=state.robot_heading,
                        bumper_pos=state.robot_bumper_pos,
                        in_contact=state.in_contact,
                        contact_force=state.contact_force,
                        timestamp=t,
                        object_position=state.object_position  # For pushing direction
                    )
                    observer.update_object(
                        name=OBJECT_NAME,
                        position=state.object_position,
                        velocity=state.object_velocity,
                        orientation=state.object_orientation,
                        angular_velocity=state.object_angular_velocity,
                        timestamp=t
                    )
            
            # Print status periodically
            if t - last_print_time >= 2.0:
                contact_str = f"CONTACT (F={state.contact_force_magnitude:.1f}N)" if state.in_contact else "no contact"
                dist = np.linalg.norm(state.object_position[:2] - state.robot_position)
                print(f"t={t:.1f}s | dist={dist:.3f}m | {contact_str}")
                last_print_time = t
            
            # Step simulation
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            # Real-time pacing (if GUI)
            if gui:
                time.sleep(TIMESTEP * 0.5)
                
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
    
    # Final state
    state.print_state()
    
    # Cleanup
    if observer:
        observer.stop()
    pyb.disconnect()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
