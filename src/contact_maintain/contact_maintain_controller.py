"""Contact Maintenance Controllers - FIXED VERSION

Velocity-based controllers for maintaining contact at a specific boundary point
on an object.

Two approaches:
1. InstantVelocityMatcher: Match robot velocity to boundary point velocity
2. WrenchTrackingController: Track a desired wrench through the contact point

Both require knowing the boundary point parameter (t_param) to track.

MAJOR BUGS FIXED:
- Heading control now points toward contact point from robot position
- Adaptive position gain based on velocity magnitude
- Added desired object velocity tracking mode with angular velocity support
- WrenchTrackingController uses ForceDistributorPro API for proper force distribution
"""
import numpy as np
from pathlib import Path
import sys

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import (
    ContactPointParameterization, ContactPoint,
    BoundaryMotionPredictor, DynamicObjectModel
)
from wrench_solver import ForceDistributorPro


# ============================================================================
# INSTANT VELOCITY MATCHING - FIXED
# ============================================================================

class InstantVelocityMatcher:
    """Match robot velocity to boundary point velocity.
    
    This controller computes the velocity of a boundary point on the object
    and commands the robot to match that velocity to maintain contact.
    
    Works by:
    1. Getting current object state (position, orientation, velocity)
    2. Computing boundary point velocity using rigid body kinematics
    3. Commanding robot to match that velocity
    
    FIXES:
    - Heading now points toward contact point from robot position
    - Position gain is now adaptive based on velocity
    - Added mode to track desired object velocity (linear + angular)
    
    Parameters
    ----------
    generic_object : GenericObject
        The object being tracked (from object_utils.py).
    t_param : float
        Boundary parameter (0-1) for the contact point to track.
    kp_position : float
        Base proportional gain for position error correction.
    max_velocity : float
        Maximum velocity magnitude.
    mode : str
        'track_current' (default): Match current boundary point velocity
        'drive_desired': Drive object toward desired velocity
    desired_object_velocity : np.ndarray, optional
        Desired object linear velocity [vx, vy] when mode='drive_desired'
    desired_object_angular_velocity : float, optional
        Desired object angular velocity (omega) when mode='drive_desired'
    """
    
    def __init__(self, generic_object, t_param, 
                 kp_position=2.0, max_velocity=0.5,
                 mode='track_current', 
                 desired_object_velocity=None,
                 desired_object_angular_velocity=0.0):
        self.object = generic_object
        self.t_param = t_param
        self.kp_position_base = kp_position
        self.max_velocity = max_velocity
        self.mode = mode
        self.desired_object_velocity = desired_object_velocity if desired_object_velocity is not None else np.zeros(2)
        self.desired_object_angular_velocity = desired_object_angular_velocity
        
        # Create parameterization for boundary point calculations
        self.parameterization = ContactPointParameterization(generic_object)
        
        # Get contact point info in body frame
        contact_info = self.parameterization.get_contact_info(t_param)
        self.contact_point_body = contact_info['point']
        self.normal_inward = contact_info['normal_inward']
    
    def set_mode(self, mode, desired_object_velocity=None, desired_object_angular_velocity=None):
        """Change tracking mode.
        
        Parameters
        ----------
        mode : str
            'track_current' or 'drive_desired'
        desired_object_velocity : np.ndarray, optional
            Desired linear velocity [vx, vy] when mode='drive_desired'
        desired_object_angular_velocity : float, optional
            Desired angular velocity (omega) when mode='drive_desired'
        """
        self.mode = mode
        if desired_object_velocity is not None:
            self.desired_object_velocity = desired_object_velocity
        if desired_object_angular_velocity is not None:
            self.desired_object_angular_velocity = desired_object_angular_velocity
    
    def compute_robot_velocity(self, robot_pos, robot_heading,
                               object_pos, object_orientation,
                               object_velocity, object_angular_velocity):
        """Compute velocity command for robot to maintain contact.
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Robot (x, y) position.
        robot_heading : float
            Robot heading in radians.
        object_pos : np.ndarray
            Object center (x, y) position.
        object_orientation : float
            Object orientation in radians.
        object_velocity : np.ndarray
            Object velocity (vx, vy) in world frame.
        object_angular_velocity : float
            Object angular velocity (omega) in rad/s.
        
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy, omega) for holonomic robot.
        """
        # Compute boundary point position in world frame
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        point_world = R @ self.contact_point_body + object_pos
        
        # Mode-dependent target velocity at contact point
        if self.mode == 'track_current':
            # Use current object velocities
            v_object_target = object_velocity
            omega_object_target = object_angular_velocity
        elif self.mode == 'drive_desired':
            # Use desired object velocities
            v_object_target = self.desired_object_velocity
            omega_object_target = self.desired_object_angular_velocity
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Compute boundary point velocity using rigid body kinematics
        # v_point = v_object + omega × r
        r = point_world - object_pos
        v_rotation = omega_object_target * np.array([-r[1], r[0]])
        v_target = v_object_target + v_rotation
        
        # Position error correction (to stay on the boundary point)
        position_error = point_world - robot_pos
        
        # ADAPTIVE GAIN: Scale based on velocity magnitude
        # Higher velocity → higher gain to maintain contact
        # Lower velocity → lower gain to avoid oscillation
        v_mag = np.linalg.norm(v_target)
        adaptive_gain = self.kp_position_base * (1.0 + 0.5 * v_mag)
        v_correction = adaptive_gain * position_error
        
        v_correction = [0.0, 0.0] # disable for now
        # Total velocity command
        v_cmd = v_target + v_correction
        
        # Clamp velocity magnitude
        v_mag_cmd = np.linalg.norm(v_cmd)
        if v_mag_cmd > self.max_velocity:
            v_cmd = v_cmd * self.max_velocity / v_mag_cmd
        
        # FIXED: Compute desired heading pointing FROM robot TO contact point
        to_contact_point = point_world - robot_pos
        desired_heading = np.arctan2(to_contact_point[1], to_contact_point[0])
        
        # Angular velocity to track heading
        heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                   np.cos(desired_heading - robot_heading))
        omega_cmd = 20.0 * heading_error  # Simple P control on heading
        omega_cmd = np.clip(omega_cmd, -1.5, 1.5)
        # omega_cmd = 0.0 # disable for now

        print(f"v and omega from equations: {v_target} and {heading_error}")
        print(f"v_cmd: {v_cmd}, omega_cmd: {omega_cmd} for t_param: {self.t_param} and for object velocity: {object_velocity} and object angular velocity: {object_angular_velocity}")
        print(f"desired object velocity: {v_object_target} and desired object angular velocity: {omega_object_target}")
        
        return np.array([v_cmd[0], v_cmd[1], omega_cmd])
    
    def get_target_point(self, object_pos, object_orientation):
        """Get the target boundary point position in world frame."""
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        return R @ self.contact_point_body + object_pos


# ============================================================================
# WRENCH TRACKING CONTROLLER - USING FORCE DISTRIBUTOR API
# ============================================================================

class WrenchTrackingController:
    """Track a desired wrench on the object through the contact point.
    
    This controller applies a desired wrench (Fx, Fy, tau) to the object
    by computing the required force at the contact point using ForceDistributorPro,
    then commanding robot velocity to achieve that force.
    
    NEW APPROACH:
    - Use ForceDistributorPro API to compute desired contact forces from wrench
    - Track desired force magnitude with proportional control
    - Much cleaner than previous unit wrench projection method
    
    Parameters
    ----------
    generic_object : GenericObject
        The object being manipulated.
    t_param : float
        Boundary parameter for contact point.
    desired_wrench : np.ndarray
        Desired wrench [Fx, Fy, tau] to apply to object center.
    kp_force : float
        Proportional gain for force tracking.
    kp_position : float
        Proportional gain for position tracking.
    max_velocity : float
        Maximum velocity magnitude.
    """
    
    def __init__(self, generic_object, t_param,
                 desired_wrench=None,
                 kp_force=0.1, kp_position=2.0, max_velocity=0.3,
                 force_distributor=None):
        self.object = generic_object
        self.t_param = t_param
        self.desired_wrench = desired_wrench if desired_wrench is not None else np.zeros(3)
        self.kp_force = kp_force
        self.kp_position_base = kp_position
        self.max_velocity = max_velocity
        
        # Create parameterization
        self.parameterization = ContactPointParameterization(generic_object)
        
        # Get contact point info in body frame
        contact_info = self.parameterization.get_contact_info(t_param)
        self.contact_point_body = contact_info['point']
        self.normal_inward = contact_info['normal_inward']
        self.tangent = contact_info['tangent']
        
        # Cache for grasp matrix (single-contact case)
        self._grasp_matrix = self._compute_grasp_matrix()
        
        # Initialize / attach ForceDistributorPro
        if force_distributor is None:
            # Default: single-contact distributor owned by this controller
            assert False, "ForceDistributorPro is not supposed to be initialized here"
        else:
            # Shared distributor across multiple controllers (e.g., Magnum Four)
            self.force_distributor = force_distributor
            # Determine which contact this controller corresponds to
            # based on t_param vs distributor.t_params (if available).
            if hasattr(self.force_distributor, "t_params") and self.force_distributor.t_params:
                t_list = self.force_distributor.t_params
                # Use circular distance on [0,1)
                diffs = [min(abs(tp - t_param), 1.0 - abs(tp - t_param)) for tp in t_list]
                self.contact_index = int(np.argmin(diffs))
                print(f"contact_index: {self.contact_index} for t_param: {t_param}")
            else:
                # Fallback: assume single-contact if no t_params defined
                self.contact_index = 0
    
    def _compute_grasp_matrix(self):
        """Compute grasp matrix for single contact point in body frame.
        
        For a single contact point at position r = [rx, ry] with normal n = [nx, ny]:
        Grasp matrix G maps force magnitude to wrench:
        [Fx, Fy, tau]^T = G * f
        
        where G is 3x1: [nx, ny, rx*ny - ry*nx]^T
        """
        r = self.contact_point_body
        n = self.normal_inward
        
        # Moment arm: tau = r × f = rx*fy - ry*fx = (rx*ny - ry*nx)*f
        moment_coefficient = r[0] * n[1] - r[1] * n[0]
        
        # Grasp matrix is 3x1 for single contact
        G = np.array([
            [n[0]],
            [n[1]],
            [moment_coefficient]
        ])
        
        return G
    
    def get_target_point(self, object_pos, object_orientation):
        """Get the target boundary point position in world frame."""
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        return R @ self.contact_point_body + object_pos

    def set_desired_wrench(self, wrench):
        """Update the desired wrench."""
        self.desired_wrench = np.array(wrench)
    
    def compute_robot_velocity(self, robot_pos, robot_heading,
                               object_pos, object_orientation,
                               object_velocity, object_angular_velocity,
                               measured_force=None):
        """Compute velocity command to track desired wrench.
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Robot position.
        robot_heading : float
            Robot heading.
        object_pos : np.ndarray
            Object position.
        object_orientation : float
            Object orientation.
        object_velocity : np.ndarray
            Object velocity.
        object_angular_velocity : float
            Object angular velocity.
        measured_force : np.ndarray, optional
            Measured contact force [fx, fy, fz] (if available).
        
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy, omega).
        """
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Step 1: Convert desired wrench from world frame to body frame
        # For planar case: rotate force components, moment stays same
        desired_wrench_body = np.zeros(3)
        desired_wrench_body[:2] = R.T @ self.desired_wrench[:2]  # Rotate force to body frame
        desired_wrench_body[2] = self.desired_wrench[2]  # Moment is same in both frames
        
        # Step 2: Use ForceDistributorPro to compute desired contact forces.
        # The distributor may manage one or many contacts internally.
        result = self.force_distributor.distribute_forces(
            desired_wrench=desired_wrench_body,
            version='v2',  # With force magnitude constraints
            method='rf'     # Refined method (most stable)
        )
        
        if not result['success']:
            # Fallback: zero force if distribution fails
            desired_force_magnitude = 0.0
        else:
            force_magnitudes = result.get('force_magnitudes', None)
            if force_magnitudes is None or len(force_magnitudes) == 0:
                desired_force_magnitude = 0.0
            else:
                idx = getattr(self, "contact_index", 0)
                if idx < len(force_magnitudes):
                    desired_force_magnitude = force_magnitudes[idx]
                else:
                    # Safety fallback: use first contact's force
                    desired_force_magnitude = force_magnitudes[0]
        
        # Step 3: Get boundary point position and velocity
        point_world = self.get_target_point(object_pos, object_orientation)
        
        r = point_world - object_pos
        v_rotation = object_angular_velocity * np.array([-r[1], r[0]])
        v_point = object_velocity + v_rotation
        
        # Step 4: Force direction in world frame
        force_dir_world = R @ self.normal_inward
        
        # Step 5: Base velocity - track the boundary point with adaptive gain
        position_error = point_world - robot_pos
        v_mag = np.linalg.norm(v_point)
        adaptive_gain = self.kp_position_base * (1.0 + 0.5 * v_mag)
        v_base = adaptive_gain * position_error + v_point

        v_base = v_point # disable the correction term for now
        
        # Step 6: Force tracking - push harder/softer to match desired force
        if measured_force is not None:
            # Use measured force for feedback control
            current_force_mag = np.linalg.norm(measured_force[:2])
            force_error = desired_force_magnitude - current_force_mag
            v_force = self.kp_force * force_error * force_dir_world
            v_cmd = v_base + v_force
        else:
            # Without force feedback, use feedforward control
            # Push proportionally to desired force magnitude
            v_force = self.kp_force * desired_force_magnitude * force_dir_world
            v_cmd = v_base + v_force
        
        # Step 7: Clamp velocity
        v_mag_cmd = np.linalg.norm(v_cmd)
        if v_mag_cmd > self.max_velocity:
            v_cmd = v_cmd * self.max_velocity / v_mag_cmd
        
        # Step 8: FIXED - Heading control points FROM robot TO contact point
        to_contact_point = point_world - robot_pos
        desired_heading = np.arctan2(to_contact_point[1], to_contact_point[0])
        
        heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                   np.cos(desired_heading - robot_heading))
        omega_cmd = 2.0 * heading_error
        omega_cmd = np.clip(omega_cmd, -1.5, 1.5)
        
        return np.array([v_cmd[0], v_cmd[1], omega_cmd])


# ============================================================================
# CONTACT MAINTENANCE STATE
# ============================================================================

class ContactMaintenanceState:
    """Tracks state for contact maintenance.
    
    Stores robot state, object state, and contact information.
    """
    
    def __init__(self):
        # Robot state
        self.robot_pos = np.zeros(2)
        self.robot_heading = 0.0
        self.robot_velocity = np.zeros(3)
        
        # Object state
        self.object_pos = np.zeros(2)
        self.object_orientation = 0.0
        self.object_velocity = np.zeros(2)
        self.object_angular_velocity = 0.0
        
        # Contact state
        self.in_contact = False
        self.contact_force = np.zeros(3)
        self.contact_force_magnitude = 0.0
        self.contact_point_world = np.zeros(2)
        
        # Target
        self.target_t_param = 0.0
        self.target_point_world = np.zeros(2)
        
        # Errors
        self.position_error = 0.0
        self.t_param_error = 0.0
    
    def update_from_simulation(self, robot, object_uid, controller):
        """Update state from PyBullet simulation.
        
        Parameters
        ----------
        robot : HolonomicRobot or OmniwheelRobot
            The robot instance.
        object_uid : int
            PyBullet body UID for the object.
        controller : InstantVelocityMatcher or WrenchTrackingController
            The controller (to get target point).
        """
        import pybullet as pyb
        
        # Robot state
        self.robot_pos, self.robot_heading, self.robot_velocity = robot.get_state()
        
        # Object state
        obj_pos, obj_orn = pyb.getBasePositionAndOrientation(object_uid)
        obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(object_uid)
        euler = pyb.getEulerFromQuaternion(obj_orn)
        
        self.object_pos = np.array([obj_pos[0], obj_pos[1]])
        self.object_orientation = euler[2]
        self.object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
        self.object_angular_velocity = obj_vel_ang[2]
        
        # Contact state
        self.contact_force = robot.get_contact_force([object_uid])
        self.contact_force_magnitude = np.linalg.norm(self.contact_force[:2])
        self.in_contact = self.contact_force_magnitude > 0.5
        self.contact_point_world = robot.get_contact_position()[:2]
        
        # Target point
        self.target_point_world = controller.get_target_point(
            self.object_pos, self.object_orientation
        )
        
        # Errors
        self.position_error = np.linalg.norm(
            self.contact_point_world - self.target_point_world
        )
    
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            'robot_pos': self.robot_pos.tolist(),
            'robot_heading': float(self.robot_heading),
            'robot_velocity': self.robot_velocity.tolist(),
            'object_pos': self.object_pos.tolist(),
            'object_orientation': float(self.object_orientation),
            'object_velocity': self.object_velocity.tolist(),
            'object_angular_velocity': float(self.object_angular_velocity),
            'in_contact': self.in_contact,
            'contact_force_magnitude': float(self.contact_force_magnitude),
            'position_error': float(self.position_error),
        }