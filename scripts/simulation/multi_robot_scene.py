#!/usr/bin/env python3
"""
Multi-robot PyBullet scene for contact maintenance.

Features:
- Multiple homogeneous robots (R_01, R_02, ...)
- T-shaped non-convex object
- Web-based observer with robot/object selection
"""
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

from contact_maintain.robots import HolonomicRobot
from contact_maintain.pyb_simulation import BulletSquareSlider, BulletCircleSlider
from contact_maintain.objects import TShapeObject
from contact_maintain.web_observer import WebObserver


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

TIMESTEP = 1.0 / 240.0  # 240 Hz physics
CTRL_FREQ = 100  # 100 Hz control
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

# Default number of robots
DEFAULT_NUM_ROBOTS = 3

# Robot parameters (scaled small robot)
ROBOT_RADIUS = 0.06   # Body radius
ROBOT_SPACING = 0.2   # Minimum spacing between robots

# Object parameters
OBJECT_TYPE = "tshape"  # "tshape", "box", or "cylinder"
OBJECT_POSITION = [0.0, 0.0, 0.2]

# Friction
SURFACE_MU = 0.3
CONTACT_MU = 0.8
OBJECT_MU = 0.5

# Controller parameters
APPROACH_SPEED = 0.1
KP_HEADING = 2.0

# Contact detection
FORCE_THRESHOLD = 0.5


# ============================================================================
# ROBOT PLACEMENT
# ============================================================================

def compute_robot_positions(num_robots, center=(0, 0), radius=1.5):
    """Compute starting positions for robots arranged in a circle.
    
    Parameters
    ----------
    num_robots : int
        Number of robots to place.
    center : tuple
        Center of the circle.
    radius : float
        Radius of the circle.
    
    Returns
    -------
    list of tuple
        List of (x, y, heading) for each robot.
    """
    positions = []
    for i in range(num_robots):
        angle = 2 * np.pi * i / num_robots
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        # Face toward center
        heading = angle + np.pi
        positions.append((x, y, heading))
    return positions


# ============================================================================
# STATE TRACKING
# ============================================================================

class RobotState:
    """State for a single robot."""
    def __init__(self, name):
        self.name = name
        self.position = np.zeros(2)
        self.velocity = np.zeros(3)
        self.heading = 0.0
        self.bumper_pos = np.zeros(3)
        self.in_contact = False
        self.contact_force = np.zeros(3)
        self.contact_force_magnitude = 0.0


class ObjectState:
    """State for an object."""
    def __init__(self, name):
        self.name = name
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = 0.0
        self.angular_velocity = 0.0


class MultiRobotSimState:
    """Tracks all simulation state for multi-robot scene."""
    
    def __init__(self):
        self.robots = {}
        self.objects = {}
        self.time = 0.0
    
    def add_robot(self, name):
        self.robots[name] = RobotState(name)
    
    def add_object(self, name):
        self.objects[name] = ObjectState(name)
    
    def update_robot(self, name, robot: HolonomicRobot, object_uids):
        """Update robot state from PyBullet."""
        state = self.robots[name]
        pos, heading, vel = robot.get_state()
        state.position = pos
        state.heading = heading
        state.velocity = vel
        state.bumper_pos = robot.get_contact_position()
        
        # Check contact with all objects
        total_force = np.zeros(3)
        for uid in object_uids:
            force = robot.get_contact_force([uid])
            total_force += force
        
        state.contact_force = total_force
        state.contact_force_magnitude = np.linalg.norm(total_force[:2])
        state.in_contact = state.contact_force_magnitude > FORCE_THRESHOLD
    
    def update_object(self, name, uid):
        """Update object state from PyBullet."""
        state = self.objects[name]
        pos, orn = pyb.getBasePositionAndOrientation(uid)
        vel_lin, vel_ang = pyb.getBaseVelocity(uid)
        
        state.position = np.array(pos)
        state.velocity = np.array(vel_lin)
        
        euler = pyb.getEulerFromQuaternion(orn)
        state.orientation = euler[2]
        state.angular_velocity = vel_ang[2]


# ============================================================================
# CONTROLLER
# ============================================================================

class MultiRobotApproachController:
    """Simple approach controller for multiple robots."""
    
    def __init__(self, approach_speed=APPROACH_SPEED, kp_heading=KP_HEADING):
        self.approach_speed = approach_speed
        self.kp_heading = kp_heading
    
    def compute_velocity(self, robot_state: RobotState, target_pos):
        """Compute velocity to approach target."""
        to_target = np.array(target_pos)[:2] - robot_state.position
        distance = np.linalg.norm(to_target)
        
        if distance < 0.01:
            return np.array([0, 0, 0])
        
        direction = to_target / distance
        
        # Desired heading
        desired_heading = np.arctan2(direction[1], direction[0])
        heading_error = np.arctan2(np.sin(desired_heading - robot_state.heading),
                                   np.cos(desired_heading - robot_state.heading))
        
        omega = self.kp_heading * heading_error
        omega = np.clip(omega, -1.5, 1.5)
        
        # Linear velocity
        alignment = np.cos(heading_error)
        speed = self.approach_speed * max(0.3, alignment)
        
        if robot_state.in_contact:
            speed = min(speed, 0.03)
        
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
    
    pyb.resetDebugVisualizerCamera(
        cameraDistance=4.0,
        cameraYaw=45,
        cameraPitch=-50,
        cameraTargetPosition=[0.0, 0.0, 0.0],
    )
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 1)
    
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_uid = pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.changeDynamics(ground_uid, -1, lateralFriction=SURFACE_MU)
    
    return client_id, ground_uid


def create_robots(num_robots, urdf_path):
    """Create multiple robots arranged in a circle."""
    positions = compute_robot_positions(num_robots, center=(0, 0), radius=1.5)
    robots = {}
    
    for i, (x, y, heading) in enumerate(positions):
        name = f"R_{i+1:02d}"
        robot = HolonomicRobot(
            urdf_path,
            position=(x, y),
            orientation=heading,
            contact_mu=CONTACT_MU
        )
        robots[name] = robot
        print(f"  Created {name} at ({x:.2f}, {y:.2f}), heading={np.degrees(heading):.1f}°")
    
    return robots


def create_object(object_type="tshape"):
    """Create the object to push."""
    if object_type == "tshape":
        obj = TShapeObject(
            position=OBJECT_POSITION,
            horizontal_size=(1.05, 0.24, 0.08), # Top of the T
            vertical_size=(0.24, 0.6, 0.08),   # Stem of the T
            mass=5.0,
            mu=OBJECT_MU,
            color=(0.2, 0.7, 0.3, 1.0)
        )
        return obj, obj.uid
    elif object_type == "box":
        obj = BulletSquareSlider(
            position=OBJECT_POSITION,
            mass=2.0,
            half_extents=(0.9, 0.3, 0.15),
        )
        pyb.changeDynamics(obj.uid, -1, lateralFriction=OBJECT_MU)
        return obj, obj.uid
    elif object_type == "cylinder":
        obj = BulletCircleSlider(
            position=OBJECT_POSITION,
            mass=15.0,
            radius=0.25,
            height=0.3,
        )
        pyb.changeDynamics(obj.uid, -1, lateralFriction=OBJECT_MU)
        return obj, obj.uid
    else:
        raise ValueError(f"Unknown object type: {object_type}")


def check_quit_key():
    """Check if 'Q' key was pressed."""
    keys = pyb.getKeyboardEvents()
    if 113 in keys or 81 in keys:
        if keys.get(113, 0) & pyb.KEY_WAS_TRIGGERED or keys.get(81, 0) & pyb.KEY_WAS_TRIGGERED:
            return True
    return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-robot contact maintenance scene")
    parser.add_argument("--num-robots", "-n", type=int, default=DEFAULT_NUM_ROBOTS,
                       help="Number of robots")
    parser.add_argument("--object", "-o", choices=["tshape", "box", "cylinder"],
                       default="tshape", help="Object type")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--no-web", action="store_true", help="Disable web observer")
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  MULTI-ROBOT CONTACT MAINTENANCE")
    print("="*60)
    print(f"  Robots: {args.num_robots}")
    print(f"  Object: {args.object}")
    print("="*60)
    
    # Setup simulation
    print("\nInitializing PyBullet...")
    gui = not args.no_gui
    client_id, ground_uid = setup_simulation(gui=gui)
    
    # Create robots
    print(f"\nCreating {args.num_robots} robots...")
    urdf_path = str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf")
    robots = create_robots(args.num_robots, urdf_path)
    
    # Create object
    print(f"\nCreating {args.object} object...")
    obj, obj_uid = create_object(args.object)
    object_name = f"OBJ_{args.object}"
    
    # Initialize state tracker
    state = MultiRobotSimState()
    for name in robots.keys():
        state.add_robot(name)
    state.add_object(object_name)
    
    # Initialize controllers
    controllers = {name: MultiRobotApproachController() for name in robots.keys()}
    
    # Initialize web observer
    observer = None
    if not args.no_web:
        print("\nStarting web observer...")
        observer = WebObserver(port=5000)
        for name in robots.keys():
            observer.register_robot(name)
        observer.register_object(object_name)
        observer.start()
        print(f"\n>>> Web dashboard: http://localhost:5000 <<<")
    
    # Let simulation settle
    print("\nSettling simulation...")
    for _ in range(100):
        pyb.stepSimulation()
    
    print("\nSimulation running...")
    print(f"  - {args.num_robots} robots approaching {args.object}")
    print("  - Press 'Q' in PyBullet window or Ctrl+C to stop")
    if observer:
        print(f"  - View at http://localhost:5000")
    print()
    
    # Get object position for controllers
    obj_pos, _ = obj.get_pose()
    
    # Main loop
    t = 0.0
    step_count = 0
    last_print_time = 0.0
    
    try:
        while t < args.duration:
            if gui and check_quit_key():
                print("\n'Q' pressed - stopping...")
                break
            
            state.time = t
            
            # Update object state
            state.update_object(object_name, obj_uid)
            obj_pos = state.objects[object_name].position
            
            # Update each robot
            for name, robot in robots.items():
                state.update_robot(name, robot, [obj_uid])
            
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Compute and apply velocity commands
                cmd_vels = {}
                for name, robot in robots.items():
                    robot_state = state.robots[name]
                    cmd_vel = controllers[name].compute_velocity(robot_state, obj_pos)
                    robot.command_velocity(cmd_vel)
                    cmd_vels[name] = cmd_vel
                
                # Update web observer
                if observer:
                    for name, robot in robots.items():
                        rs = state.robots[name]
                        observer.update_robot(
                            name=name,
                            position=rs.position,
                            velocity=rs.velocity,
                            heading=rs.heading,
                            bumper_pos=rs.bumper_pos,
                            in_contact=rs.in_contact,
                            contact_force=rs.contact_force,
                            timestamp=t,
                            object_position=obj_pos,
                            cmd_velocity=cmd_vels[name]
                        )
                    
                    os = state.objects[object_name]
                    observer.update_object(
                        name=object_name,
                        position=os.position,
                        velocity=os.velocity,
                        orientation=os.orientation,
                        angular_velocity=os.angular_velocity,
                        timestamp=t
                    )
            
            # Print status
            if t - last_print_time >= 3.0:
                contacts = sum(1 for rs in state.robots.values() if rs.in_contact)
                print(f"t={t:.1f}s | {contacts}/{args.num_robots} robots in contact")
                last_print_time = t
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.3)
                
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    # Final status
    print(f"\n{'='*40}")
    print("FINAL STATE")
    print(f"{'='*40}")
    for name, rs in state.robots.items():
        contact_str = f"CONTACT (F={rs.contact_force_magnitude:.1f}N)" if rs.in_contact else "no contact"
        print(f"{name}: pos=({rs.position[0]:.2f}, {rs.position[1]:.2f}), {contact_str}")
    
    if observer:
        observer.stop()
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()

