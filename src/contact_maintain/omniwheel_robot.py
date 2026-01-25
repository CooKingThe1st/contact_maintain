"""Omniwheel Robot class for PyBullet simulation.

A realistic holonomic robot with 4 omniwheels that converts body velocity
commands (vx, vy, omega) to individual wheel angular velocities.

Based on the wheel velocity equations from the Webots Robotino controller.
"""
import numpy as np
import pybullet as pyb
from pathlib import Path

import rospkg


# ============================================================================
# CONSTANTS
# ============================================================================

# Wheel configuration (scaled to realistic small robot)
WHEEL_RADIUS = 0.02   # meters (scaled from 0.063)
ROBOT_RADIUS = 0.045  # distance from center to wheel (scaled from 0.16)
WHEEL_ANGLES = [
    np.pi/4,      # Front-Right (45°)
    3*np.pi/4,    # Front-Left (135°)
    5*np.pi/4,    # Rear-Left (225°)
    7*np.pi/4,    # Rear-Right (315°)
]

# Maximum speeds (adjusted for smaller wheels)
MAX_WHEEL_SPEED = 60.0  # rad/s (increased to compensate for smaller wheels)
MAX_LINEAR_SPEED = WHEEL_RADIUS * MAX_WHEEL_SPEED  # m/s
MAX_ANGULAR_SPEED = WHEEL_RADIUS * MAX_WHEEL_SPEED / ROBOT_RADIUS  # rad/s


# ============================================================================
# WHEEL VELOCITY CALCULATION
# ============================================================================

def compute_wheel_velocities(vx, vy, omega, robot_heading=0.0):
    """Convert body velocity (vx, vy, omega) to 4 wheel angular velocities.
    
    Based on the omniwheel kinematics from temp.txt:
    - Wheels at 45°, 135°, 225°, 315° from forward direction
    - Each wheel contributes based on its angle to the velocity direction
    
    Parameters
    ----------
    vx : float
        Forward velocity in world frame (m/s).
    vy : float
        Lateral velocity in world frame (m/s).
    omega : float
        Angular velocity (rad/s).
    robot_heading : float
        Current robot heading in world frame (rad).
    
    Returns
    -------
    list of float
        Angular velocities for wheels [FR, FL, RL, RR] (rad/s).
    """
    # Scale velocities
    vx_scaled = vx / WHEEL_RADIUS
    vy_scaled = vy / WHEEL_RADIUS
    omega_scaled = omega * ROBOT_RADIUS / WHEEL_RADIUS
    
    # Wheel angles relative to robot heading
    theta = robot_heading
    
    # Calculate wheel velocities using omniwheel kinematics
    # Each wheel velocity = projection of body velocity onto wheel axis + rotation
    w = []
    for angle in WHEEL_ANGLES:
        wheel_angle = theta + angle
        # Wheel axis perpendicular to wheel orientation
        # Using the formula from temp.txt
        w_i = (vx_scaled * -np.sin(wheel_angle) + 
               vy_scaled * np.cos(wheel_angle) + 
               omega_scaled)
        w.append(w_i)
    
    return w


def compute_body_velocity(wheel_velocities, robot_heading=0.0):
    """Inverse: convert wheel velocities to body velocity.
    
    Parameters
    ----------
    wheel_velocities : list of float
        Angular velocities for wheels [FR, FL, RL, RR] (rad/s).
    robot_heading : float
        Current robot heading in world frame (rad).
    
    Returns
    -------
    tuple
        (vx, vy, omega) body velocity.
    """
    theta = robot_heading
    
    # Build the forward kinematics matrix
    # v_wheel = J * v_body, so v_body = J_pinv * v_wheel
    J = []
    for angle in WHEEL_ANGLES:
        wheel_angle = theta + angle
        row = [
            -np.sin(wheel_angle) / WHEEL_RADIUS,
            np.cos(wheel_angle) / WHEEL_RADIUS,
            ROBOT_RADIUS / WHEEL_RADIUS
        ]
        J.append(row)
    
    J = np.array(J)
    w = np.array(wheel_velocities)
    
    # Pseudo-inverse solution
    J_pinv = np.linalg.pinv(J)
    v_body = J_pinv @ w
    
    return v_body[0], v_body[1], v_body[2]


# ============================================================================
# OMNIWHEEL ROBOT CLASS
# ============================================================================

class OmniwheelRobot:
    """Realistic omniwheel robot with 4-wheel control.
    
    Unlike the dummy HolonomicRobot that uses direct velocity control,
    this robot converts body velocity commands to individual wheel speeds.
    
    Parameters
    ----------
    urdf_path : str
        Path to the omniwheel robot URDF file.
    position : tuple
        Initial (x, y) position.
    orientation : float
        Initial heading in radians.
    contact_mu : float
        Friction coefficient for bumper.
    wheel_lateral_friction : float
        Lateral friction for wheels (0 for ideal omniwheels).
    """
    
    # Joint indices
    X_JOINT_IDX = 0
    Y_JOINT_IDX = 1
    THETA_JOINT_IDX = 2
    
    # Wheel joint names (from URDF)
    WHEEL_JOINT_NAMES = ['wheel_fr_joint', 'wheel_fl_joint', 
                         'wheel_rl_joint', 'wheel_rr_joint']
    
    def __init__(self, urdf_path=None, position=(0, 0), orientation=0.0,
                 contact_mu=0.8, wheel_lateral_friction=0.0):
        
        # Default URDF path
        if urdf_path is None:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("contact_maintain")
            urdf_path = str(Path(pkg_path) / "urdf" / "omniwheel_robot.urdf")
        
        # Load URDF
        self.uid = pyb.loadURDF(urdf_path, [0, 0, 0], [0, 0, 0, 1], 
                                useFixedBase=True)
        
        # Store initial config
        self.pos_init = np.array(position, dtype=float)
        self.orn_init = float(orientation)
        
        # Find joint indices
        self._build_joint_info()
        
        # Set friction
        self.set_contact_friction(contact_mu)
        self.set_wheel_friction(wheel_lateral_friction)
        
        # Reset to initial position
        self.reset()
        
        # Command history for debugging
        self.last_cmd_vel = np.zeros(3)
        self.last_wheel_speeds = np.zeros(4)
    
    def _build_joint_info(self):
        """Build joint index mappings."""
        self.planar_joint_indices = [self.X_JOINT_IDX, self.Y_JOINT_IDX, 
                                     self.THETA_JOINT_IDX]
        
        # Find wheel joint indices
        self.wheel_joint_indices = []
        for i in range(pyb.getNumJoints(self.uid)):
            info = pyb.getJointInfo(self.uid, i)
            joint_name = info[1].decode('utf-8')
            if joint_name in self.WHEEL_JOINT_NAMES:
                self.wheel_joint_indices.append(i)
        
        if len(self.wheel_joint_indices) != 4:
            raise RuntimeError(
                f"Expected 4 wheel joints, found {len(self.wheel_joint_indices)}")
        
        # Find bumper link
        self.bumper_link_idx = None
        for i in range(pyb.getNumJoints(self.uid)):
            info = pyb.getJointInfo(self.uid, i)
            link_name = info[12].decode('utf-8')
            if link_name == 'bumper':
                self.bumper_link_idx = i
                break
        
        if self.bumper_link_idx is None:
            print("Warning: bumper link not found")
    
    def set_contact_friction(self, mu):
        """Set bumper friction."""
        if self.bumper_link_idx is not None:
            pyb.changeDynamics(self.uid, self.bumper_link_idx, 
                              lateralFriction=mu)
    
    def set_wheel_friction(self, lateral_friction):
        """Set wheel lateral friction (0 for ideal omniwheels)."""
        for idx in self.wheel_joint_indices:
            pyb.changeDynamics(self.uid, idx, lateralFriction=lateral_friction)
    
    def get_state(self):
        """Get current robot state.
        
        Returns
        -------
        position : np.ndarray, shape (2,)
            Current (x, y) position.
        heading : float
            Current heading (theta) in radians.
        velocity : np.ndarray, shape (3,)
            Current velocity (vx, vy, omega).
        """
        states = pyb.getJointStates(self.uid, self.planar_joint_indices)
        position = np.array([states[0][0], states[1][0]])
        heading = states[2][0]
        velocity = np.array([states[0][1], states[1][1], states[2][1]])
        return position, heading, velocity
    
    def get_pose(self):
        """Get current pose (position, heading)."""
        position, heading, _ = self.get_state()
        return position, heading
    
    def get_wheel_velocities(self):
        """Get current wheel angular velocities."""
        velocities = []
        for idx in self.wheel_joint_indices:
            state = pyb.getJointState(self.uid, idx)
            velocities.append(state[1])
        return np.array(velocities)
    
    def get_contact_position(self):
        """Get bumper position in world frame."""
        if self.bumper_link_idx is not None:
            state = pyb.getLinkState(self.uid, self.bumper_link_idx)
            return np.array(state[0])
        else:
            pos, heading = self.get_pose()
            # Estimate bumper position (scaled for small robot)
            return np.array([pos[0] + 0.055 * np.cos(heading),
                            pos[1] + 0.055 * np.sin(heading), 0.025])
    
    def get_contact_force(self, object_uids, max_contacts=1):
        """Get contact force with objects."""
        from contact_maintain.pyb_simulation import get_contact_force
        
        total_force = np.zeros(3)
        for uid in object_uids:
            force = get_contact_force(
                self.uid, uid,
                linkIndexA=self.bumper_link_idx if self.bumper_link_idx else -1,
                max_contacts=max_contacts
            )
            total_force += force
        return total_force
    
    def command_velocity(self, velocity):
        """Command body velocity using wheel control.
        
        Parameters
        ----------
        velocity : array-like, shape (3,)
            Desired velocity (vx, vy, omega) in world frame.
        """
        vx, vy, omega = velocity
        self.last_cmd_vel = np.array([vx, vy, omega])
        
        # Get current heading
        _, heading, _ = self.get_state()
        
        # Compute wheel velocities
        wheel_speeds = compute_wheel_velocities(vx, vy, omega, heading)
        self.last_wheel_speeds = np.array(wheel_speeds)
        
        # Clamp wheel speeds
        wheel_speeds = np.clip(wheel_speeds, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        
        # Apply wheel velocities
        for i, (idx, speed) in enumerate(zip(self.wheel_joint_indices, wheel_speeds)):
            pyb.setJointMotorControl2(
                self.uid, idx,
                controlMode=pyb.VELOCITY_CONTROL,
                targetVelocity=speed,
                force=100.0
            )
        
        # Also set planar joint velocities for the carrier mechanism
        # This ensures the robot actually moves in the simulation
        pyb.setJointMotorControlArray(
            self.uid,
            self.planar_joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=[vx, vy, omega]
        )
    
    def command_wheel_velocities(self, wheel_velocities):
        """Command wheel velocities directly.
        
        Parameters
        ----------
        wheel_velocities : array-like, shape (4,)
            Angular velocities for [FR, FL, RL, RR] wheels (rad/s).
        """
        self.last_wheel_speeds = np.array(wheel_velocities)
        
        for idx, speed in zip(self.wheel_joint_indices, wheel_velocities):
            pyb.setJointMotorControl2(
                self.uid, idx,
                controlMode=pyb.VELOCITY_CONTROL,
                targetVelocity=speed,
                force=100.0
            )
        
        # Compute body velocity for carrier
        _, heading, _ = self.get_state()
        vx, vy, omega = compute_body_velocity(wheel_velocities, heading)
        
        pyb.setJointMotorControlArray(
            self.uid,
            self.planar_joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=[vx, vy, omega]
        )
    
    def reset(self, position=None, orientation=None):
        """Reset robot to initial or specified configuration."""
        if position is not None:
            self.pos_init = np.array(position, dtype=float)
        if orientation is not None:
            self.orn_init = float(orientation)
        
        # Stop all motors
        self.command_velocity([0, 0, 0])
        
        # Reset planar joints
        pyb.resetJointState(self.uid, self.X_JOINT_IDX, self.pos_init[0], 0)
        pyb.resetJointState(self.uid, self.Y_JOINT_IDX, self.pos_init[1], 0)
        pyb.resetJointState(self.uid, self.THETA_JOINT_IDX, self.orn_init, 0)
        
        # Reset wheel joints
        for idx in self.wheel_joint_indices:
            pyb.resetJointState(self.uid, idx, 0, 0)
    
    def get_debug_info(self):
        """Get debug information about current state."""
        pos, heading, vel = self.get_state()
        wheel_vels = self.get_wheel_velocities()
        
        return {
            'position': pos,
            'heading': heading,
            'heading_deg': np.degrees(heading),
            'velocity': vel,
            'wheel_velocities': wheel_vels,
            'last_cmd_vel': self.last_cmd_vel,
            'last_wheel_speeds': self.last_wheel_speeds,
        }

