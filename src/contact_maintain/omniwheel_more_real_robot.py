"""More Realistic Omniwheel Robot class for PyBullet simulation.

⚠️  UNSUPPORTED / EXPERIMENTAL ⚠️

This robot class attempts to simulate realistic omniwheel/mecanum wheel behavior
by using ONLY wheel motors (no planar joint "magic" control) with anisotropic
friction. However, PyBullet does not fully support mecanum/omniwheel physics,
so this implementation may not work correctly.

Key differences from OmniwheelRobot:
- NO planar joint velocity control (wheels drive everything)
- PI controller uses wheel encoder feedback to track desired body velocity
- More realistic physics where wheels must overcome forces
- Attempts to use anisotropic friction for directional sliding

Status: This class is experimental and may not function as intended due to
PyBullet limitations with omniwheel/mecanum wheel physics. Use OmniwheelRobot
for reliable simulation.
"""
import numpy as np
import pybullet as pyb
from pathlib import Path

import rospkg

# Import wheel velocity computation from original
from contact_maintain.omniwheel_robot import (
    WHEEL_RADIUS,
    ROBOT_RADIUS,
    WHEEL_ANGLES,
    MAX_WHEEL_SPEED,
    compute_wheel_velocities,
    compute_body_velocity,
)


# ============================================================================
# OMNIWHEEL MORE REAL ROBOT CLASS
# ============================================================================

class OmniwheelMoreRealRobot:
    """More realistic omniwheel robot using ONLY wheel motors (no planar joints).
    
    Unlike OmniwheelRobot that uses planar joints to "ensure" motion,
    this robot relies entirely on wheel motors. Uses PI controller with
    encoder feedback to track desired body velocities.
    
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
        Lateral friction for wheels (deprecated, use wheel_friction_x/y instead).
    wheel_friction_x : float
        Friction coefficient in x-direction for wheels (default: 1.0, for traction).
    wheel_friction_y : float
        Friction coefficient in y-direction for wheels (default: 0.0, for free sliding like real omniwheels).
    wheel_motor_force : float
        Maximum force for wheel motors (default: 500.0, higher than original 100.0).
    kp_vel : float
        Proportional gain for velocity PI controller (default: 2.0).
    ki_vel : float
        Integral gain for velocity PI controller (default: 0.5).
    dt_ctrl : float
        Control time step for PI controller (default: 0.01, 100Hz).
    """
    
    # Joint indices (same as original)
    X_JOINT_IDX = 0
    Y_JOINT_IDX = 1
    THETA_JOINT_IDX = 2
    
    # Wheel joint names (from URDF)
    WHEEL_JOINT_NAMES = ['wheel_fr_joint', 'wheel_fl_joint', 
                         'wheel_rl_joint', 'wheel_rr_joint']
    
    def __init__(self, urdf_path=None, position=(0, 0), orientation=0.0,
                 contact_mu=0.8, wheel_lateral_friction=0.0,
                 wheel_friction_x=1.0, wheel_friction_y=0.0,
                 wheel_motor_force=500.0, kp_vel=2.0, ki_vel=0.001, dt_ctrl=0.01):
        
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
        
        # Motor parameters
        self.wheel_motor_force = wheel_motor_force
        
        # Wheel friction parameters (for anisotropic friction)
        self.wheel_friction_x = wheel_friction_x  # Friction in x-direction (traction)
        self.wheel_friction_y = wheel_friction_y  # Friction in y-direction (0 for free sliding)
        
        # PI controller parameters
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.dt_ctrl = dt_ctrl
        
        # PI controller state (velocity error integrals)
        self.velocity_error_int = np.zeros(3)  # [vx_err_int, vy_err_int, omega_err_int]
        self.velocity_error_int_max = 5.0  # Anti-windup limit
        
        # Find joint indices
        self._build_joint_info()
        
        # Set friction
        self.set_contact_friction(contact_mu)
        self.set_wheel_friction_anisotropic(wheel_friction_x, wheel_friction_y)
        
        # Disable planar joint motors (let them be passive/free)
        # This is key: we don't want planar joints to force motion
        for idx in self.planar_joint_indices:
            pyb.setJointMotorControl2(
                self.uid, idx,
                controlMode=pyb.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0  # No force = passive/free joint
            )
        
        # Reset to initial position
        self.reset()
        
        # Command history for debugging
        self.last_cmd_vel = np.zeros(3)
        self.last_wheel_speeds = np.zeros(4)
    
    def _build_joint_info(self):
        """Build joint index mappings."""
        self.planar_joint_indices = [self.X_JOINT_IDX, self.Y_JOINT_IDX, 
                                     self.THETA_JOINT_IDX]
        
        # Find wheel joint indices and their corresponding link indices
        self.wheel_joint_indices = []
        self.wheel_link_indices = []  # Child link index for each wheel joint
        for i in range(pyb.getNumJoints(self.uid)):
            info = pyb.getJointInfo(self.uid, i)
            joint_name = info[1].decode('utf-8')
            if joint_name in self.WHEEL_JOINT_NAMES:
                self.wheel_joint_indices.append(i)
                # Joint info: [0]=jointIndex, [16]=childLinkIndex
                child_link_idx = info[16]
                self.wheel_link_indices.append(child_link_idx)
        
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
        """Set wheel lateral friction (isotropic, for backward compatibility)."""
        for idx in self.wheel_joint_indices:
            pyb.changeDynamics(self.uid, idx, lateralFriction=lateral_friction)
    
    def set_wheel_friction_anisotropic(self, friction_x, friction_y):
        """Set wheel friction with asymmetric values (like mecanum/omniwheels).
        
        Parameters
        ----------
        friction_x : float
            Friction coefficient in x-direction (for traction).
        friction_y : float
            Friction coefficient in y-direction (0 for free sliding like real omniwheels).
        """
        # PyBullet anisotropicFriction: [fx, fy, fz] in local frame
        # For wheels, we want different friction in x and y directions
        # z-direction (normal) friction is typically 1.0
        anisotropic_friction = [friction_x, friction_y, 0.0]
        
        for joint_idx, link_idx in zip(self.wheel_joint_indices, self.wheel_link_indices):
            # Set anisotropic friction on the wheel link
            # Note: anisotropicFriction is in the local frame of the link
            # The local frame x/y directions depend on the wheel orientation in the URDF
            pyb.changeDynamics(
                self.uid, 
                link_idx,
                lateralFriction=max(friction_x, friction_y),  # Use max for compatibility
                anisotropicFriction=anisotropic_friction
            )
    
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
        """Get current wheel angular velocities (encoder readings)."""
        velocities = []
        for idx in self.wheel_joint_indices:
            state = pyb.getJointState(self.uid, idx)
            velocities.append(state[1])  # Joint velocity (angular velocity)
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
        """Command body velocity using wheel control with PI feedback.
        
        This method:
        1. Reads actual wheel velocities (encoders)
        2. Computes actual body velocity from wheels
        3. Compares with desired body velocity
        4. Uses PI controller to adjust wheel commands
        5. Applies wheel motor commands (NO planar joint control)
        
        Parameters
        ----------
        velocity : array-like, shape (3,)
            Desired velocity (vx, vy, omega) in world frame.
        """
        desired_vx, desired_vy, desired_omega = velocity
        self.last_cmd_vel = np.array([desired_vx, desired_vy, desired_omega])
        
        # Get current heading and actual wheel velocities (encoder feedback)
        _, heading, _ = self.get_state()
        actual_wheel_vels = self.get_wheel_velocities()
        
        # Compute actual body velocity from wheel encoders (forward kinematics)
        actual_vx, actual_vy, actual_omega = compute_body_velocity(
            actual_wheel_vels, heading
        )
        actual_body_vel = np.array([actual_vx, actual_vy, actual_omega])
        desired_body_vel = np.array([desired_vx, desired_vy, desired_omega])
        
        # PI controller: compute velocity error and integral
        velocity_error = desired_body_vel - actual_body_vel
        
        # Update integral term (with anti-windup)
        self.velocity_error_int += velocity_error * self.dt_ctrl
        self.velocity_error_int = np.clip(
            self.velocity_error_int,
            -self.velocity_error_int_max,
            self.velocity_error_int_max
        )
        
        # PI control output: feed-forward + proportional + integral
        # Feed-forward: desired wheel speeds from desired body velocity
        wheel_speeds_ff = compute_wheel_velocities(
            desired_vx, desired_vy, desired_omega, heading
        )
        
        # PI correction: convert velocity error to wheel speed correction
        # Use inverse kinematics to map body velocity error to wheel speed error
        wheel_error = compute_wheel_velocities(
            velocity_error[0], velocity_error[1], velocity_error[2], heading
        )
        wheel_error_int = compute_wheel_velocities(
            self.velocity_error_int[0],
            self.velocity_error_int[1],
            self.velocity_error_int[2],
            heading
        )
        
        # Total wheel command: feed-forward + P correction + I correction
        wheel_speeds = (
            np.array(wheel_speeds_ff) +
            self.kp_vel * np.array(wheel_error) +
            self.ki_vel * np.array(wheel_error_int)
        )
        
        # Clamp wheel speeds
        wheel_speeds = np.clip(wheel_speeds, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        self.last_wheel_speeds = wheel_speeds.copy()
        
        # Apply wheel motor commands ONLY (no planar joint control)
        for i, (idx, speed) in enumerate(zip(self.wheel_joint_indices, wheel_speeds)):
            pyb.setJointMotorControl2(
                self.uid, idx,
                controlMode=pyb.VELOCITY_CONTROL,
                targetVelocity=speed,
                force=self.wheel_motor_force
            )
        
        # NOTE: We do NOT set planar joint velocities here!
        # The robot moves purely through wheel-ground interaction.
    
    def command_wheel_velocities(self, wheel_velocities):
        """Command wheel velocities directly (bypasses PI controller).
        
        Parameters
        ----------
        wheel_velocities : array-like, shape (4,)
            Angular velocities for [FR, FL, RL, RR] wheels (rad/s).
        """
        self.last_wheel_speeds = np.array(wheel_velocities)
        
        # Clamp wheel speeds
        wheel_velocities = np.clip(
            wheel_velocities, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED
        )
        
        # Apply wheel motor commands
        for idx, speed in zip(self.wheel_joint_indices, wheel_velocities):
            pyb.setJointMotorControl2(
                self.uid, idx,
                controlMode=pyb.VELOCITY_CONTROL,
                targetVelocity=speed,
                force=self.wheel_motor_force
            )
        
        # Reset PI integral when using direct wheel control
        self.velocity_error_int = np.zeros(3)
    
    def reset(self, position=None, orientation=None):
        """Reset robot to initial or specified configuration."""
        if position is not None:
            self.pos_init = np.array(position, dtype=float)
        if orientation is not None:
            self.orn_init = float(orientation)
        
        # Stop all wheel motors
        for idx in self.wheel_joint_indices:
            pyb.setJointMotorControl2(
                self.uid, idx,
                controlMode=pyb.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )
        
        # Reset planar joints (position and velocity)
        pyb.resetJointState(self.uid, self.X_JOINT_IDX, self.pos_init[0], 0)
        pyb.resetJointState(self.uid, self.Y_JOINT_IDX, self.pos_init[1], 0)
        pyb.resetJointState(self.uid, self.THETA_JOINT_IDX, self.orn_init, 0)
        
        # Reset wheel joints
        for idx in self.wheel_joint_indices:
            pyb.resetJointState(self.uid, idx, 0, 0)
        
        # Reset PI controller state
        self.velocity_error_int = np.zeros(3)
        
        # Clear command history
        self.last_cmd_vel = np.zeros(3)
        self.last_wheel_speeds = np.zeros(4)
    
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
            'velocity_error_int': self.velocity_error_int.copy(),
        }
