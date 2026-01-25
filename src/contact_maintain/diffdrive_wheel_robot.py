"""Differential Drive Wheel Robot class for PyBullet simulation.

A realistic differential-drive robot with 2 drive wheels that converts
body velocity commands (v, omega) to individual wheel angular velocities.

Kinematics:
- v = (v_left + v_right) / 2 * wheel_radius
- omega = (v_right - v_left) / wheel_base * wheel_radius
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
WHEEL_BASE = 0.10     # distance between wheels (meters, scaled from 0.30)

# Maximum speeds (adjusted for smaller wheels)
MAX_WHEEL_SPEED = 60.0  # rad/s (increased to compensate for smaller wheels)
MAX_LINEAR_SPEED = WHEEL_RADIUS * MAX_WHEEL_SPEED  # m/s
MAX_ANGULAR_SPEED = 2 * WHEEL_RADIUS * MAX_WHEEL_SPEED / WHEEL_BASE  # rad/s


# ============================================================================
# WHEEL VELOCITY CALCULATION
# ============================================================================

def compute_wheel_velocities_diffdrive(v, omega):
    """Convert body velocity (v, omega) to left/right wheel angular velocities.
    
    Inverse kinematics for differential drive:
    v_left = (v - omega * wheel_base / 2) / wheel_radius
    v_right = (v + omega * wheel_base / 2) / wheel_radius
    
    Parameters
    ----------
    v : float
        Forward velocity (m/s).
    omega : float
        Angular velocity (rad/s).
    
    Returns
    -------
    tuple
        (omega_left, omega_right) wheel angular velocities (rad/s).
    """
    # Wheel linear velocities
    v_left = v - omega * WHEEL_BASE / 2
    v_right = v + omega * WHEEL_BASE / 2
    
    # Convert to angular velocities
    omega_left = v_left / WHEEL_RADIUS
    omega_right = v_right / WHEEL_RADIUS
    
    return omega_left, omega_right


def compute_body_velocity_diffdrive(omega_left, omega_right):
    """Convert wheel angular velocities to body velocity.
    
    Forward kinematics for differential drive.
    
    Parameters
    ----------
    omega_left : float
        Left wheel angular velocity (rad/s).
    omega_right : float
        Right wheel angular velocity (rad/s).
    
    Returns
    -------
    tuple
        (v, omega) body velocity.
    """
    # Wheel linear velocities
    v_left = omega_left * WHEEL_RADIUS
    v_right = omega_right * WHEEL_RADIUS
    
    # Body velocity
    v = (v_left + v_right) / 2
    omega = (v_right - v_left) / WHEEL_BASE
    
    return v, omega


# ============================================================================
# DIFFERENTIAL DRIVE WHEEL ROBOT CLASS
# ============================================================================

class DiffDriveWheelRobot:
    """Realistic differential-drive robot with 2-wheel control.
    
    Unlike the dummy DifferentialDriveRobot that constrains velocity directly,
    this robot uses actual wheel motors and physics.
    
    Parameters
    ----------
    urdf_path : str
        Path to the robot URDF file.
    position : tuple
        Initial (x, y) position.
    orientation : float
        Initial heading in radians.
    contact_mu : float
        Friction coefficient for bumper.
    """
    
    # Joint indices
    X_JOINT_IDX = 0
    Y_JOINT_IDX = 1
    THETA_JOINT_IDX = 2
    
    def __init__(self, urdf_path=None, position=(0, 0), orientation=0.0,
                 contact_mu=0.8):
        
        # Default URDF path
        if urdf_path is None:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("contact_maintain")
            urdf_path = str(Path(pkg_path) / "urdf" / "diffdrive_wheel_robot.urdf")
        
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
        
        # Reset to initial position
        self.reset()
        
        # Command history for debugging
        self.last_cmd_vel = np.array([0.0, 0.0])  # (v, omega)
        self.last_wheel_speeds = np.array([0.0, 0.0])  # (left, right)
    
    def _build_joint_info(self):
        """Build joint index mappings."""
        self.planar_joint_indices = [self.X_JOINT_IDX, self.Y_JOINT_IDX, 
                                     self.THETA_JOINT_IDX]
        
        # Find wheel joint indices
        self.wheel_left_idx = None
        self.wheel_right_idx = None
        self.bumper_link_idx = None
        self.caster_link_idx = None
        
        for i in range(pyb.getNumJoints(self.uid)):
            info = pyb.getJointInfo(self.uid, i)
            joint_name = info[1].decode('utf-8')
            link_name = info[12].decode('utf-8')
            
            if joint_name == 'wheel_left_joint':
                self.wheel_left_idx = i
            elif joint_name == 'wheel_right_joint':
                self.wheel_right_idx = i
            
            if link_name == 'bumper':
                self.bumper_link_idx = i
            elif link_name == 'caster_link':
                self.caster_link_idx = i
        
        if self.wheel_left_idx is None or self.wheel_right_idx is None:
            raise RuntimeError("Could not find wheel joints in URDF")
        
        if self.bumper_link_idx is None:
            print("Warning: bumper link not found")
    
    def set_contact_friction(self, mu):
        """Set bumper friction."""
        if self.bumper_link_idx is not None:
            pyb.changeDynamics(self.uid, self.bumper_link_idx, 
                              lateralFriction=mu)
        
        # Set caster to very low friction for free rolling
        if self.caster_link_idx is not None:
            pyb.changeDynamics(self.uid, self.caster_link_idx,
                              lateralFriction=0.01)
        
        # Set wheels to very low friction (planar joints do the actual motion)
        if self.wheel_left_idx is not None:
            pyb.changeDynamics(self.uid, self.wheel_left_idx,
                              lateralFriction=0.01)
        if self.wheel_right_idx is not None:
            pyb.changeDynamics(self.uid, self.wheel_right_idx,
                              lateralFriction=0.01)
    
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
        """Get current wheel angular velocities (left, right)."""
        left_state = pyb.getJointState(self.uid, self.wheel_left_idx)
        right_state = pyb.getJointState(self.uid, self.wheel_right_idx)
        return np.array([left_state[1], right_state[1]])
    
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
        
        For diff-drive robot, we use the planar joints as the primary
        motion mechanism (like the dummy robot), but the wheel velocities
        are displayed for comparison.
        
        Parameters
        ----------
        velocity : array-like
            Either (v, omega) for differential drive,
            or (vx, vy, omega) where vx is used as forward velocity.
        """
        if len(velocity) == 2:
            v, omega = velocity
        else:
            # For compatibility: use vx as forward velocity, ignore vy
            v, omega = velocity[0], velocity[2]
        
        self.last_cmd_vel = np.array([v, omega])
        
        # Compute wheel velocities (for visualization/logging)
        omega_left, omega_right = compute_wheel_velocities_diffdrive(v, omega)
        self.last_wheel_speeds = np.array([omega_left, omega_right])
        
        # Clamp wheel speeds
        omega_left = np.clip(omega_left, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        omega_right = np.clip(omega_right, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        
        # Set wheel motor velocities (these spin visually but don't drive motion)
        pyb.setJointMotorControl2(
            self.uid, self.wheel_left_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=omega_left,
            force=10.0  # Reduced force since wheels are for display
        )
        pyb.setJointMotorControl2(
            self.uid, self.wheel_right_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=omega_right,
            force=10.0
        )
        
        # Get current heading for world-frame velocity
        _, heading, _ = self.get_state()
        
        # Compute world-frame velocity (diff-drive constraint: no lateral motion)
        vx = v * np.cos(heading)
        vy = v * np.sin(heading)
        
        # Use planar joints as primary motion (like dummy robot)
        # This ensures consistent behavior for comparison
        # Use individual joint control for better reliability
        pyb.setJointMotorControl2(
            self.uid, self.X_JOINT_IDX,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=vx,
            force=500.0
        )
        pyb.setJointMotorControl2(
            self.uid, self.Y_JOINT_IDX,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=vy,
            force=500.0
        )
        pyb.setJointMotorControl2(
            self.uid, self.THETA_JOINT_IDX,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=omega,
            force=200.0
        )
    
    def command_wheel_velocities(self, omega_left, omega_right):
        """Command wheel velocities directly.
        
        Parameters
        ----------
        omega_left : float
            Left wheel angular velocity (rad/s).
        omega_right : float
            Right wheel angular velocity (rad/s).
        """
        self.last_wheel_speeds = np.array([omega_left, omega_right])
        
        # Apply wheel velocities
        pyb.setJointMotorControl2(
            self.uid, self.wheel_left_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=omega_left,
            force=100.0
        )
        pyb.setJointMotorControl2(
            self.uid, self.wheel_right_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=omega_right,
            force=100.0
        )
        
        # Compute body velocity for carrier
        v, omega = compute_body_velocity_diffdrive(omega_left, omega_right)
        self.last_cmd_vel = np.array([v, omega])
        
        _, heading, _ = self.get_state()
        vx = v * np.cos(heading)
        vy = v * np.sin(heading)
        
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
        
        # Reset planar joints first
        pyb.resetJointState(self.uid, self.X_JOINT_IDX, self.pos_init[0], 0)
        pyb.resetJointState(self.uid, self.Y_JOINT_IDX, self.pos_init[1], 0)
        pyb.resetJointState(self.uid, self.THETA_JOINT_IDX, self.orn_init, 0)
        
        # Reset wheel joints
        pyb.resetJointState(self.uid, self.wheel_left_idx, 0, 0)
        pyb.resetJointState(self.uid, self.wheel_right_idx, 0, 0)
        
        # Disable wheel motors initially (let them spin freely)
        pyb.setJointMotorControl2(
            self.uid, self.wheel_left_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0  # Free spinning
        )
        pyb.setJointMotorControl2(
            self.uid, self.wheel_right_idx,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0  # Free spinning
        )
        
        # Clear command history
        self.last_cmd_vel = np.array([0.0, 0.0])
        self.last_wheel_speeds = np.array([0.0, 0.0])
    
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

