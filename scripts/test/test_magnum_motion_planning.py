#!/usr/bin/env python3
"""
Swarm + Magnum Four Tracking Base Test

⚠️  NOTE: Currently only supported with the standard OmniwheelRobot (not OmniwheelMoreRealRobot) ⚠️

The OmniwheelMoreRealRobot class is experimental and unsupported due to PyBullet limitations
with omniwheel/mecanum wheel physics. This test script uses the standard OmniwheelRobot
from robot_factory, which uses planar joint control for reliable simulation.

Wheel robot only version with actual wheel velocity tracking:
1) Phase7BetaVerDecouple controller only (removed other Phase 7 variants)
2) ObjectVelocityPIDController: Simple PID controller for object velocity control to reach a goal state
3) Actual wheel velocity tracking and plotting (not commanded velocities)

1) Create an object (GenericObject) in PyBullet
2) Use legacy Magnum Four solver to find 4 best boundary contact points (t_params)
3) Spawn 4 wheel robots and command them to NAVIGATE to those t_params (REACHING phase)
4) Uses Phase7BetaVerDecouple controller for contact maintenance (pushing phase)
5) Uses ObjectVelocityPIDController to compute desired object velocities based on goal position/orientation
6) Tracks and plots actual wheel velocities from PyBullet simulation

Usage:
  python test_magnum_tracking_base.py --object rectangle --duration 20
  python test_magnum_tracking_base.py --no-gui --object rectangle --duration 10
  python test_magnum_tracking_base.py --kinematics holonomic --object rectangle
  python3 test_magnum_motion_planning.py  --save-dir /tmp/hybrid_control/ --duration 60 --desired-path p_trajectory --record-video --object right_triangle
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Use non-interactive backend for headless mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
from contact_maintain.robot_agent import RobotAgent
from contact_maintain.swarm import SwarmHost
from contact_maintain.pyb_simulation import get_contact_force

# Motion planner with PathFollowingController
from contact_maintain.motion_planner import PathFollowingController

# Path creation functions from paths_lib
from paths_lib import create_p_trajectory_hybrid_path, create_rectangle_hybrid_path

# Magnum Four solver (legacy)
from contact_optimizer_utils import find_the_magnum_four_v3


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)
# Run the high-level object velocity PID at a lower rate than Phase 7
PID_DECIMATION = 5  # Update ObjectVelocityPIDController every 5 Phase 7 control cycles

# Object height: taller for wheel robots to avoid multi-contact issues
DEFAULT_OBJECT_HEIGHT_WHEEL = 0.08   # For wheel robots (taller to avoid multi-contact)
DEFAULT_OBJECT_FRICTION = 0.3
ROBOT_RADIUS = 0.06  # Robot radius for position offset calculation
APPROACH_DISTANCE = ROBOT_RADIUS + 0.02  # Distance from contact point to spawn robot (for faster testing)


@dataclass
class Phase7History:
    """History for Phase 7 controller plotting."""
    times: List[float] = field(default_factory=list)
    robot_positions: List[np.ndarray] = field(default_factory=list)
    robot_headings: List[float] = field(default_factory=list)
    robot_velocities: List[np.ndarray] = field(default_factory=list)
    intended_positions: List[np.ndarray] = field(default_factory=list)
    position_errors: List[np.ndarray] = field(default_factory=list)
    desired_headings: List[float] = field(default_factory=list)
    heading_errors: List[float] = field(default_factory=list)
    contact_point_positions: List[np.ndarray] = field(default_factory=list)
    contact_point_velocities: List[np.ndarray] = field(default_factory=list)
    object_positions: List[np.ndarray] = field(default_factory=list)
    object_velocities: List[np.ndarray] = field(default_factory=list)
    object_angular_velocities: List[float] = field(default_factory=list)  # Object angular velocity (omega)
    contact_forces: List[float] = field(default_factory=list)
    in_contact: List[bool] = field(default_factory=list)
    v_base_history: List[float] = field(default_factory=list)
    v_ff_history: List[float] = field(default_factory=list)
    v_pi_history: List[float] = field(default_factory=list)
    desired_contact_point_speeds: List[float] = field(default_factory=list)
    wheel_velocities: List[np.ndarray] = field(default_factory=list)  # Actual wheel velocities from PyBullet
    wheel_cmd_velocities: List[np.ndarray] = field(default_factory=list)  # Commanded wheel velocities (if available)


class WaypointController:
    """Waypoint controller that manages multiple waypoints and switches between them.
    
    This controller cycles through waypoints in order, switching to the next waypoint
    when the object is close enough to the current waypoint or when velocity is very small.
    """
    
    def __init__(
        self,
        waypoints: List,
        position_tolerance: float = 0.05,  # Switch waypoint when within this distance (m)
        orientation_tolerance: float = 0.1,  # Switch waypoint when orientation error < this (rad)
        velocity_threshold: float = 0.02,  # Switch waypoint when velocity < this (m/s)
        angular_velocity_threshold: float = 0.05,  # Switch waypoint when angular velocity < this (rad/s)
    ):
        """
        Parameters
        ----------
        waypoints : List[tuple[np.ndarray, float]]
            List of (position, orientation) tuples for each waypoint
        position_tolerance : float
            Distance threshold for position-based waypoint switching (m)
        orientation_tolerance : float
            Orientation error threshold for waypoint switching (rad)
        velocity_threshold : float
            Linear velocity threshold for waypoint switching (m/s)
        angular_velocity_threshold : float
            Angular velocity threshold for waypoint switching (rad/s)
        """
        self.waypoints = waypoints
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.velocity_threshold = velocity_threshold
        self.angular_velocity_threshold = angular_velocity_threshold
        
        self.current_waypoint_idx = 0
        self.num_waypoints = len(waypoints)
    
    def get_current_waypoint(self):
        """Get the current waypoint (position, orientation)."""
        return self.waypoints[self.current_waypoint_idx]
    
    def update(
        self,
        current_position: np.ndarray,
        current_orientation: float,
        current_velocity: np.ndarray,
        current_angular_velocity: float,
    ) -> bool:
        """Update waypoint controller and check if waypoint should be switched.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current object position (x, y)
        current_orientation : float
            Current object orientation (radians)
        current_velocity : np.ndarray
            Current object linear velocity (vx, vy)
        current_angular_velocity : float
            Current object angular velocity (rad/s)
        
        Returns
        -------
        bool
            True if waypoint was switched, False otherwise
        """
        goal_position, goal_orientation = self.get_current_waypoint()
        
        # Compute errors
        position_error = np.linalg.norm(goal_position - current_position)
        orientation_error = np.abs(np.arctan2(
            np.sin(goal_orientation - current_orientation),
            np.cos(goal_orientation - current_orientation)
        ))
        velocity_magnitude = np.linalg.norm(current_velocity)
        angular_velocity_magnitude = np.abs(current_angular_velocity)
        
        # Check if we should switch to next waypoint.
        # With multi-rate control (PID slower than Phase 7), switching purely on "velocity_small"
        # can cause rapid waypoint cycling when the object is stationary but far from the waypoint.
        # So we only allow velocity-based switching when we're also reasonably near the waypoint.
        close_enough = (position_error < self.position_tolerance and 
                       orientation_error < self.orientation_tolerance)
        velocity_small = (velocity_magnitude < self.velocity_threshold and
                          angular_velocity_magnitude < self.angular_velocity_threshold and
                          position_error < (2.0 * self.position_tolerance) and
                          orientation_error < (2.0 * self.orientation_tolerance))
        
        if close_enough or velocity_small:
            # Switch to next waypoint (circular: wrap around)
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % self.num_waypoints
            return True
        
        return False






class ObjectVelocityPIDController:
    """Simple PID controller for object velocity control to reach a goal state.
    
    This controller computes desired object velocity (vx, vy, omega) based on:
    - Position error: difference between current and goal position
    - Orientation error: difference between current and goal orientation
    - Current object velocity: for damping/feed-forward
    
    Control Law:
        desired_vx = Kp_pos_x * (goal_x - current_x) + Kd_vel_x * (0 - current_vx)
        desired_vy = Kp_pos_y * (goal_y - current_y) + Kd_vel_y * (0 - current_vy)
        desired_omega = Kp_orient * (goal_theta - current_theta) + Kd_omega * (0 - current_omega)
    
    The desired velocities are then passed to Phase7BetaVerDecouple controllers
    which execute the motion via robot contact forces.
    """
    
    def __init__(
        self,
        goal_position: np.ndarray,
        goal_orientation: float = 0.0,
    ):
        """
        Parameters
        ----------
        goal_position : np.ndarray
            Goal position (x, y) in world frame
        goal_orientation : float
            Goal orientation (radians) in world frame
        """
        self.goal_position = np.array(goal_position, dtype=float)
        self.goal_orientation = float(goal_orientation)
        
        # Position control gains
        self.kp_pos_x = 0.5  # Proportional gain for x position
        self.kp_pos_y = 0.5  # Proportional gain for y position
        self.kd_vel_x = 0.3  # Derivative gain for x velocity (damping)
        self.kd_vel_y = 0.3  # Derivative gain for y velocity (damping)
        
        # Orientation control gains
        self.kp_orient = 0.8  # Proportional gain for orientation
        self.kd_omega = 0.2   # Derivative gain for angular velocity (damping)
        
        # Velocity limits
        self.max_linear_velocity = 0.15  # Max desired linear velocity (m/s)
        self.max_angular_velocity = 0.15  # Max desired angular velocity (rad/s)
        
        # Control time step
        self.dt_ctrl = 1.0 / CTRL_FREQ
    
    def compute_desired_velocity(
        self,
        current_position: np.ndarray,
        current_orientation: float,
        current_velocity: np.ndarray,
        current_angular_velocity: float,
    ) :
        """Compute desired object velocity to reach goal state.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current object position (x, y)
        current_orientation : float
            Current object orientation (radians)
        current_velocity : np.ndarray
            Current object linear velocity (vx, vy)
        current_angular_velocity : float
            Current object angular velocity (rad/s)
        
        Returns
        -------
        tuple[np.ndarray, float]
            (desired_linear_velocity, desired_angular_velocity)
            desired_linear_velocity: (vx, vy) in world frame
            desired_angular_velocity: omega (rad/s)
        """
        # Position error
        position_error = self.goal_position - current_position
        
        # Orientation error (wrap to [-pi, pi])
        orientation_error = self.goal_orientation - current_orientation
        orientation_error = np.arctan2(
            np.sin(orientation_error),
            np.cos(orientation_error)
        )
        
        # PID control for linear velocity
        # P term: position error
        vx_p = self.kp_pos_x * position_error[0]
        vy_p = self.kp_pos_y * position_error[1]
        
        # D term: velocity damping (drive velocity toward zero when close to goal)
        vx_d = -self.kd_vel_x * current_velocity[0]
        vy_d = -self.kd_vel_y * current_velocity[1]
        
        desired_vx = vx_p + vx_d
        desired_vy = vy_p + vy_d
        
        # PID control for angular velocity
        # P term: orientation error
        omega_p = self.kp_orient * orientation_error
        
        # D term: angular velocity damping
        omega_d = -self.kd_omega * current_angular_velocity
        
        desired_omega = omega_p + omega_d
        
        # Clamp velocities
        desired_linear_velocity = np.array([desired_vx, desired_vy])
        speed = np.linalg.norm(desired_linear_velocity)
        if speed > self.max_linear_velocity:
            desired_linear_velocity = desired_linear_velocity * (self.max_linear_velocity / speed)
        
        desired_omega = np.clip(desired_omega, -self.max_angular_velocity, self.max_angular_velocity)
        
        return desired_linear_velocity, desired_omega
    
    def set_goal(self, goal_position: np.ndarray, goal_orientation: float):
        """Update the goal position and orientation.
        
        Parameters
        ----------
        goal_position : np.ndarray
            New goal position (x, y) in world frame
        goal_orientation : float
            New goal orientation (radians) in world frame
        """
        self.goal_position = np.array(goal_position, dtype=float)
        self.goal_orientation = float(goal_orientation)


class Phase7BetaVerDecouple:
    """Phase 7 Beta Version Decouple: Tripartite Decoupled Structure in Contact Frame.
    
    This controller solves the "slipping out" and "mismatch" issues by treating the robot's
    effort as three independent axes in the Contact Frame (C_i), separating the physics of
    the object's motion from the geometry of contact maintenance.
    
    Control Structure:
    ------------------
    1. Longitudinal Axis (v_x, contact frame): Controls pushing force (clamping) against inner normal
       - v_along = v_along_ff + Kp_pos * error_along + PI_Velocity_Correction
       - High priority: maintains contact via "Virtual Spring"
    
    2. Lateral Axis (v_y, contact frame): Governs lateral position along object's edge
       - v_perp = v_perp_ff + Kp_perp * error_perp
       - Prevents "slipping out" by matching object rotation
    
    3. Angular (ω): Manages robot orientation so force sensor/bumper points along normal
       - Standard heading control toward contact point
    
    Key Features:
    -------------
    - High-Fidelity Velocity Feed-Forward ("Rotation Lock"): Projects desired contact point
      velocity onto local normal and tangent before PID corrections
    - Hierarchical Priority: Corrections are "delta" adjustments to base speeds
    - Target Penetration: Virtual spring maintains constant clamping pressure (2-5mm penetration)
    - No Fighting: Position and velocity controllers work on same side of equation (bias)
    - Dynamic Response: v_perp_ff allows "crab-walk" sideways in sync with object rotation
    """
    
    def __init__(
        self,
        robot_uid: int,
        object_uid: int,
        generic_object: Any,
        t_param: float,
        desired_object_velocity: np.ndarray,
        desired_object_angular_velocity: float,
    ):
        """
        Parameters
        ----------
        robot_uid : int
            PyBullet UID of the robot
        object_uid : int
            PyBullet UID of the object
        generic_object : GenericObject
            Object model for boundary parameterization
        t_param : float
            Target t_param on object boundary
        desired_object_velocity : np.ndarray
            Desired object linear velocity (vx, vy) in world frame
        desired_object_angular_velocity : float
            Desired object angular velocity (rad/s)
        """
        self.robot_uid = robot_uid
        self.object_uid = object_uid
        self.generic_object = generic_object
        self.t_param = t_param
        self.desired_object_velocity = np.array(desired_object_velocity, dtype=float)
        self.desired_object_angular_velocity = float(desired_object_angular_velocity)
        
        # Get contact point info at t_param
        from object_utils import ContactPointParameterization
        param = ContactPointParameterization(generic_object)
        contact_info = param.get_contact_info(t_param)
        self.contact_point_body = contact_info['point']
        self.normal_outward = contact_info['normal_outward']
        self.normal_inward = -self.normal_outward
        self.tangent = contact_info['tangent']  # Unit tangent vector along boundary
        
        # Target penetration depth (Virtual Spring for force proxy)
        # This is the intended depth inside object boundary to ensure bumper compression
        self.target_penetration = 0.003  # 3mm penetration (adjustable: 2-5mm)
        
        # Longitudinal Axis Controller (The "Cling" Controller)
        # Controls pushing force/clamping along normal_inward
        self.kp_along = 2.5  # Position error gain along normal (Virtual Spring stiffness)
        self.kp_vel_along = 1.0  # Velocity error proportional gain
        self.ki_vel_along = 0.3  # Velocity error integral gain
        self.velocity_error_int_along = 0.0
        self.velocity_error_int_max_along = 0.8
        
        # Lateral Axis Controller (The "Sliding" Controller)
        # Controls lateral position along object's edge (tangent direction)
        self.kp_perp = 1.5  # Position error gain along tangent
        
        # Feed-forward gains (plant compensation)
        self.K_static = 0.03  # Static friction compensation
        self.K_alpha = 0.6    # Viscous friction coefficient
        
        # Contact detection with hysteresis
        self.contact_threshold_on = 2.0
        self.contact_threshold_off = 0.2
        self.in_contact_prev = False
        
        # Integral decay when not in contact
        self.integral_decay_rate = 0.95
        
        # Heading control (Angular axis)
        self.kp_heading = 10.0
        
        # Orientation filtering to reduce swiggling
        self.orientation_filter_alpha = 0.7
        self.object_orientation_filtered = None
        
        # Limits
        self.max_linear_speed = 0.5
        self.max_along_speed = 0.4  # Max speed along normal (clamping)
        self.max_perp_speed = 0.3  # Max speed along tangent (sliding)
        
        # Control time step
        self.dt_ctrl = 1.0 / CTRL_FREQ
        
        # History for plotting
        self.history = Phase7History()
    
    def _compute_desired_contact_point_velocity(
        self,
        object_pos: np.ndarray,
        object_orientation: float,
    ) -> np.ndarray:
        """Compute desired contact point velocity vector from desired object motion.
        
        Same as other Phase7 controllers - computes in body frame first, then transforms to world.
        """
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        R_T = R.T
        
        r_cp_body = self.contact_point_body
        v_obj_desired_body = R_T @ self.desired_object_velocity
        v_rotation_body = self.desired_object_angular_velocity * np.array([-r_cp_body[1], r_cp_body[0]])
        v_cp_desired_body = v_obj_desired_body + v_rotation_body
        v_cp_desired = R @ v_cp_desired_body
        
        return v_cp_desired
    
    def compute_velocity(
        self,
        robot_pos: np.ndarray,
        robot_heading: float,
        object_pos: np.ndarray,
        object_orientation: float,
        object_velocity: np.ndarray,
        object_angular_velocity: float,
        contact_force: float,
        in_contact: bool,
        t: float = 0.0,
        record_history: bool = False,
        robot: Optional[Any] = None,  # Robot object to get actual wheel velocities
    ) -> np.ndarray:
        """Compute velocity command using Tripartite Decoupled Structure.
        
        Control Law:
        ------------
        In Contact Frame (C_i):
            v_along = v_along_ff + Kp_along * error_along + PI_velocity_correction
            v_perp  = v_perp_ff  + Kp_perp * error_perp
            ω       = Kp_heading * heading_error
        
        Where:
            - v_along_ff: Feed-forward along normal (from desired contact point velocity)
            - v_perp_ff:  Feed-forward along tangent (from desired contact point velocity)
            - error_along: Position error along normal (with target penetration)
            - error_perp:  Position error along tangent
        """
        # Apply hysteresis-based contact detection
        if self.in_contact_prev:
            in_contact_hyst = contact_force > self.contact_threshold_off
        else:
            in_contact_hyst = contact_force > self.contact_threshold_on
        
        self.in_contact_prev = in_contact_hyst
        in_contact = in_contact_hyst
        
        # Filter object orientation to reduce swiggling
        if self.object_orientation_filtered is None:
            self.object_orientation_filtered = object_orientation
        else:
            self.object_orientation_filtered = (
                self.orientation_filter_alpha * object_orientation +
                (1 - self.orientation_filter_alpha) * self.object_orientation_filtered
            )
        
        filtered_orientation = self.object_orientation_filtered
        
        # Contact point & frame vectors in world (using filtered orientation)
        cos_t = np.cos(filtered_orientation)
        sin_t = np.sin(filtered_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        contact_point_world = R @ self.contact_point_body + object_pos
        normal_outward_world = R @ self.normal_outward
        normal_inward_world = -normal_outward_world
        tangent_world = R @ self.tangent
        
        # Intended position: contact point + robot_radius * normal_outward - target_penetration * normal_inward
        # The target_penetration creates a "Virtual Spring" that maintains clamping pressure
        intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world - self.target_penetration * normal_inward_world
        position_error = intended_pos - robot_pos
        
        # Heading control: point toward contact point
        desired_heading = np.arctan2(
            (contact_point_world - robot_pos)[1],
            (contact_point_world - robot_pos)[0]
        )
        heading_error = np.arctan2(
            np.sin(desired_heading - robot_heading),
            np.cos(desired_heading - robot_heading)
        )
        
        # ===== TRIPARTITE DECOUPLED CONTROL IN CONTACT FRAME =====
        
        # STEP 1: High-Fidelity Velocity Feed-Forward ("Rotation Lock")
        # Project desired contact point velocity onto local contact frame axes
        v_cp_desired = self._compute_desired_contact_point_velocity(
            object_pos, filtered_orientation
        )
        desired_contact_point_speed = np.linalg.norm(v_cp_desired)
        
        # Project onto contact frame axes (normal_inward and tangent)
        v_along_ff = np.dot(v_cp_desired, normal_inward_world)  # Feed-forward along normal
        v_perp_ff = np.dot(v_cp_desired, tangent_world)          # Feed-forward along tangent
        
        # STEP 2: Decompose position error in contact frame
        error_along = np.dot(position_error, normal_inward_world)  # Error along normal (longitudinal)
        error_perp = np.dot(position_error, tangent_world)         # Error along tangent (lateral)
        
        # STEP 3: Longitudinal Axis Controller (The "Cling" Controller)
        # Virtual Spring: maintains constant clamping pressure via position error
        v_along_pos = self.kp_along * error_along
        
        # Velocity error along normal (for PI correction)
        # Actual contact point velocity
        r_cp = contact_point_world - object_pos
        v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
        contact_point_velocity = object_velocity + v_rotation
        
        # Project actual velocity onto normal
        v_along_actual = np.dot(contact_point_velocity, normal_inward_world)
        velocity_error_along = v_along_actual - v_along_ff
        
        # PI control on velocity error along normal
        if in_contact:
            self.velocity_error_int_along += velocity_error_along * self.dt_ctrl
            self.velocity_error_int_along = np.clip(
                self.velocity_error_int_along,
                -self.velocity_error_int_max_along,
                self.velocity_error_int_max_along
            )
        else:
            # Decay integral when not in contact
            self.velocity_error_int_along *= self.integral_decay_rate
        
        v_along_pi = self.kp_vel_along * velocity_error_along + self.ki_vel_along * self.velocity_error_int_along
        
        # Total longitudinal velocity (feed-forward + position correction + PI)
        # v_along = v_along_ff + v_along_pos + v_along_pi
        v_along = v_along_ff
        v_along = np.clip(v_along, -self.max_along_speed, self.max_along_speed)
        
        # STEP 4: Lateral Axis Controller (The "Sliding" Controller)
        # Simple proportional control along tangent to prevent "slipping out"
        v_perp_pos = self.kp_perp * error_perp
        
        # Total lateral velocity (feed-forward + position correction)
        # v_perp = v_perp_ff + v_perp_pos
        v_perp = v_perp_ff * 1 + v_perp_pos

        if (self.t_param == 0.62 or self.t_param == 0.36):
            print(f"for this t_param: {self.t_param}")
            print(f" the contact point body is: {self.contact_point_body} and the desired object velocity is: {self.desired_object_velocity} and the desired object angular velocity is: {self.desired_object_angular_velocity}")
            print(f"error_perp: {error_perp} and v_perp: {v_perp} and v_along: {v_along} and v_cp_desired: {v_cp_desired} and normal_inward_world: {normal_inward_world} and tangent_world: {tangent_world}") 

        v_perp = np.clip(v_perp, -self.max_perp_speed, self.max_perp_speed)
        
        # STEP 5: Transform from contact frame to world frame
        # v_along is along normal_inward, v_perp is along tangent
        vel_cmd_xy = v_along * normal_inward_world + v_perp * tangent_world
        
        # Clamp total speed
        speed = np.linalg.norm(vel_cmd_xy)
        if speed > self.max_linear_speed:
            vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)
        
        # STEP 6: Angular control (heading)
        omega = self.kp_heading * heading_error
        omega = np.clip(omega, -1.0, 1.0)
        
        # Record history if requested
        if record_history:
            robot_vel = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
            
            # Get actual wheel velocities if robot is provided and has wheel velocity method
            actual_wheel_vels = None
            if robot is not None and hasattr(robot, 'get_wheel_velocities'):
                try:
                    actual_wheel_vels = robot.get_wheel_velocities()
                except:
                    actual_wheel_vels = None

            # Get commanded wheel velocities if available (similar to compare_robots.py)
            cmd_wheel_vels = None
            if robot is not None and hasattr(robot, "last_wheel_speeds"):
                try:
                    cmd_wheel_vels = robot.last_wheel_speeds.copy()
                except Exception:
                    cmd_wheel_vels = None
            
            self.history.times.append(t)
            self.history.robot_positions.append(robot_pos.copy())
            self.history.robot_headings.append(robot_heading)
            self.history.robot_velocities.append(robot_vel.copy())
            self.history.intended_positions.append(intended_pos.copy())
            self.history.position_errors.append(position_error.copy())
            self.history.desired_headings.append(desired_heading)
            self.history.heading_errors.append(heading_error)
            self.history.contact_point_positions.append(contact_point_world.copy())
            self.history.contact_point_velocities.append(contact_point_velocity.copy())
            self.history.object_positions.append(object_pos.copy())
            self.history.object_velocities.append(object_velocity.copy())
            self.history.object_angular_velocities.append(object_angular_velocity)
            self.history.contact_forces.append(contact_force)
            self.history.in_contact.append(in_contact)
            # Store components for plotting
            self.history.v_base_history.append(v_along)  # Longitudinal velocity
            self.history.v_ff_history.append(np.linalg.norm([v_along_ff, v_perp_ff]))  # Feed-forward magnitude
            self.history.v_pi_history.append(v_along_pi)  # PI correction
            self.history.desired_contact_point_speeds.append(desired_contact_point_speed)
            # Store actual wheel velocities (or None if not available)
            if actual_wheel_vels is not None:
                self.history.wheel_velocities.append(actual_wheel_vels.copy())
            else:
                self.history.wheel_velocities.append(np.array([]))  # Empty array if not available

            # Store commanded wheel velocities (or empty if not available)
            if cmd_wheel_vels is not None:
                self.history.wheel_cmd_velocities.append(cmd_wheel_vels.copy())
            else:
                self.history.wheel_cmd_velocities.append(np.array([]))
        
        return np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])


def setup_pybullet(gui: bool = True):
    
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)

    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    
    # IMPORTANT: Set search paths BEFORE loading URDFs
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    # Add urdf directory for OBJ files
    urdf_dir = Path(pkg_path) / "urdf"
    pyb.setAdditionalSearchPath(str(urdf_dir))
    

    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
    
    if gui:
        # Configure camera
        pyb.resetDebugVisualizerCamera(
            cameraDistance=4,
            cameraYaw=-5,
            cameraPitch=-85,
            cameraTargetPosition=[0, 0, 0]
        )
        # Disable GUI elements (tabs and tree view sidebar) - keep only the scene
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_TINY_RENDERER, 0)
    
    return ground

def setup_video_recording(video_path: Path, object_uid: int):
    """Setup PyBullet video recording from fixed top-down view.
    
    Sets the camera to a fixed top-down view at the start and keeps it there.
    No camera tracking during simulation to avoid interfering with video recording.
    
    Parameters
    ----------
    video_path : Path
        Absolute path to save the video file (should end with .mp4)
    object_uid : int
        PyBullet UID of the object (for initial camera positioning)
    
    Returns
    -------
    int
        Video logging ID
    """
    # Get object position for initial camera positioning
    pos, _ = pyb.getBasePositionAndOrientation(object_uid)
    
    
    
    # Ensure path is absolute and parent directory exists
    video_path = video_path.resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing file if it exists
    if video_path.exists():
        print(f"Removing existing video file at {video_path}")
        video_path.unlink()
    
    # Start video logging
    video_path_str = str(video_path)
    print(f"Starting video recording to: {video_path_str}")
    print(f"  Camera: Fixed top-down view (no tracking)")
    
    video_log_id = pyb.startStateLogging(
        pyb.STATE_LOGGING_VIDEO_MP4,
        video_path_str
    )
    
    if video_log_id < 0:
        raise RuntimeError(f"Failed to start video recording (log_id={video_log_id})")
    
    print(f"✓ Video recording started (log_id={video_log_id})")
    
    return video_log_id

def stop_video_recording(video_log_id: int, video_path: Path):
    """Stop PyBullet video recording and ensure file is saved.
    
    Parameters
    ----------
    video_log_id : int
        Video logging ID returned from setup_video_recording
    video_path : Path
        Absolute path where video should be saved
    """
    # Ensure path is absolute
    video_path = video_path.resolve()
    
    if video_log_id < 0:
        print(f"Error: Invalid video_log_id ({video_log_id})")
        return
    
    # Stop the logging
    pyb.stopStateLogging(video_log_id)
    print(f"Stopped video logging (ID: {video_log_id})")



    
    
    # Give it time to write the file
    import time
    time.sleep(3.0)  # Increased wait time
    
    # Verify file was created
    if video_path.exists():
        file_size = video_path.stat().st_size
        if file_size > 0:
            print(f"✓ Video saved successfully to {video_path} ({file_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"⚠ Warning: Video file is empty (0 bytes)")
    else:
        print(f"✗ Error: Video file not found at {video_path}")
        # Check what files are in the directory
        mp4_files = list(video_path.parent.glob("*.mp4"))
        if mp4_files:
            print(f"  Found these .mp4 files: {mp4_files}")

def get_object_state(object_uid):
    pos, orn = pyb.getBasePositionAndOrientation(object_uid)
    vel_lin, vel_ang = pyb.getBaseVelocity(object_uid)
    euler = pyb.getEulerFromQuaternion(orn)
    return {
        "position": np.array([pos[0], pos[1]]),
        "orientation": euler[2],
        "velocity": np.array([vel_lin[0], vel_lin[1]]),
        "angular_velocity": vel_ang[2],
    }


def get_object_as_obstacle(generic_object, object_position, object_orientation):
    # Get boundary vertices in local frame
    boundary_coords = list(generic_object.geometry.exterior.coords)

    # Transform to world frame
    c, s = np.cos(object_orientation), np.sin(object_orientation)
    R = np.array([[c, -s], [s, c]])

    world_vertices = []
    for local_vertex in boundary_coords:
        local_2d = np.array([local_vertex[0], local_vertex[1]])
        world_2d = R @ local_2d + object_position
        world_vertices.append((float(world_2d[0]), float(world_2d[1])))

    return [world_vertices]


def remove_outliers(data: np.ndarray, method: str = "percentile", lower_percentile: float = 1.0, upper_percentile: float = 99.0, iqr_factor: float = 1.5) -> np.ndarray:
    """Remove or clip outliers from data array.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array (1D or 2D)
    method : str
        Method to use: "percentile" (clip to percentiles) or "iqr" (IQR-based filtering)
    lower_percentile : float
        Lower percentile for clipping (default: 1.0)
    upper_percentile : float
        Upper percentile for clipping (default: 99.0)
    iqr_factor : float
        IQR factor for outlier detection (default: 1.5)
    
    Returns
    -------
    np.ndarray
        Data with outliers handled (clipped or filtered)
    """
    if len(data) == 0:
        return data
    
    data = np.asarray(data)
    original_shape = data.shape
    
    # Flatten for processing
    data_flat = data.flatten()
    
    # Remove NaN and Inf values first
    valid_mask = np.isfinite(data_flat)
    if not np.any(valid_mask):
        return data
    
    valid_data = data_flat[valid_mask]
    
    if method == "percentile":
        # Clip to percentiles
        lower_bound = np.percentile(valid_data, lower_percentile)
        upper_bound = np.percentile(valid_data, upper_percentile)
        data_flat[valid_mask] = np.clip(valid_data, lower_bound, upper_bound)
    elif method == "iqr":
        # IQR-based outlier detection
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr
        data_flat[valid_mask] = np.clip(valid_data, lower_bound, upper_bound)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Restore original shape
    return data_flat.reshape(original_shape)


def plot_phase7_velocities(
    histories: Dict[str, Phase7History],
    t_params: Dict[str, float],
    desired_obj_velocity: np.ndarray,
    desired_obj_omega: float,
    save_path: Optional[Path] = None,
):
    """Plot Phase 7 velocity tracking for all robots.
    
    Parameters
    ----------
    histories : Dict[str, Phase7History]
        History for each robot
    t_params : Dict[str, float]
        t_param for each robot
    desired_obj_velocity : np.ndarray
        Desired object linear velocity
    desired_obj_omega : float
        Desired object angular velocity
    save_path : Optional[Path]
        Path to save plot
    """
    if len(histories) == 0:
        print("No history to plot for Phase 7 velocities.")
        return
    
    # Create subplots: one row per robot, 7 columns (added object velocity plots)
    n_robots = len(histories)
    fig, axes = plt.subplots(n_robots, 7, figsize=(28, 4 * n_robots))
    if n_robots == 1:
        axes = axes.reshape(1, -1)
    
    # Compute overall contact percentage across robots (only where data exists)
    contact_percents = []
    for _, h in histories.items():
        if len(h.in_contact) > 0:
            contact_percents.append(100.0 * float(np.mean(np.array(h.in_contact, dtype=bool))))
    overall_contact_pct = float(np.mean(contact_percents)) if len(contact_percents) > 0 else 0.0

    fig.suptitle(
        f'Phase 7 Velocities: Multi-Robot Swarm (Contact: {overall_contact_pct:.1f}%)\n'
        f'Desired object velocity: {desired_obj_velocity}, omega: {desired_obj_omega:.3f} rad/s',
        fontsize=14, fontweight="bold",
    )
    
    for idx, (name, history) in enumerate(histories.items()):
        if len(history.times) == 0:
            continue
        
        times = np.array(history.times)
        robot_vels = np.array(history.robot_velocities)
        robot_speeds = np.linalg.norm(robot_vels[:, :2], axis=1)
        obj_vels = np.array(history.object_velocities)
        obj_angular_vels = np.array(history.object_angular_velocities)
        cp_vels = np.array(history.contact_point_velocities)
        cp_speeds = np.linalg.norm(cp_vels, axis=1)
        desired_cp_speeds = np.array(history.desired_contact_point_speeds)
        in_contact = np.array(history.in_contact)
        
        v_base = np.array(history.v_base_history)
        v_ff = np.array(history.v_ff_history)
        v_pi = np.array(history.v_pi_history)
        contact_forces = np.array(history.contact_forces)
        
        # Remove outliers from data before plotting
        robot_speeds = remove_outliers(robot_speeds, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        contact_forces = remove_outliers(contact_forces, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
        # Plot 1: Desired vs actual contact point speed
        ax = axes[idx, 0]
        ax.plot(times, desired_cp_speeds, 'g--', label='desired CP speed', linewidth=2)
        ax.plot(times, cp_speeds, 'r-', label='actual CP speed', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title(f'{name} - Contact Point Speed (t_param={t_params[name]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Robot speed
        ax = axes[idx, 1]
        ax.plot(times, robot_speeds, 'b-', label='robot speed', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title(f'{name} - Robot Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Velocity components
        ax = axes[idx, 2]
        ax.plot(times, v_base, 'c-', label='v_base', linewidth=1.5, alpha=0.7)
        ax.plot(times, v_ff, 'm-', label='v_ff (feed-forward)', linewidth=1.5, alpha=0.7)
        ax.plot(times, v_pi, 'orange', label='v_pi', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Component (m/s)')
        ax.set_title(f'{name} - Velocity Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Contact force
        ax = axes[idx, 3]
        ax.plot(times, contact_forces, 'r-', linewidth=1.5, label='contact force')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact Force (N)')
        ax.set_title(f'{name} - Contact Force')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Contact state
        ax = axes[idx, 4]
        ax.fill_between(times, 0, 1, where=in_contact, alpha=0.3, color='green', label='in contact')
        ax.fill_between(times, 0, 1, where=~in_contact, alpha=0.3, color='red', label='not in contact')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact State')
        contact_pct = 100.0 * float(np.mean(in_contact)) if len(in_contact) > 0 else 0.0
        ax.set_title(f'{name} - Contact State ({contact_pct:.1f}%)')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No Contact', 'In Contact'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Object linear velocity (x, y)
        ax = axes[idx, 5]
        ax.plot(times, obj_vels[:, 0], 'b-', label='vx', linewidth=1.5)
        ax.plot(times, obj_vels[:, 1], 'r-', label='vy', linewidth=1.5)
        ax.axhline(y=desired_obj_velocity[0], color='b', linestyle='--', alpha=0.5, label='desired vx')
        ax.axhline(y=desired_obj_velocity[1], color='r', linestyle='--', alpha=0.5, label='desired vy')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'{name}  Object Linear Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 7: Object angular velocity (omega)
        ax = axes[idx, 6]
        ax.plot(times, obj_angular_vels, 'g-', label='omega', linewidth=1.5)
        ax.axhline(y=desired_obj_omega, color='g', linestyle='--', alpha=0.5, label='desired omega')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title(f'{name} - Object Angular Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Phase 7 velocity plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_phase_1_results(
    histories: Dict[str, Phase7History],
    t_params: Dict[str, float],
    contact_threshold: float = 0.5,
    save_path: Optional[Path] = None,
):
    """Plot Phase 1 style results (trajectories, position errors, heading errors, etc.) for all robots.
    
    Parameters
    ----------
    histories : Dict[str, Phase7History]
        History for each robot
    t_params : Dict[str, float]
        t_param for each robot
    contact_threshold : float
        Contact force threshold for plotting
    save_path : Optional[Path]
        Path to save plot
    """
    if len(histories) == 0:
        print("No history to plot for Phase 1 results.")
        return
    
    # Create subplots: one row per robot, 6 columns (2x3 grid per robot)
    n_robots = len(histories)
    fig, axes = plt.subplots(n_robots, 6, figsize=(24, 4 * n_robots))
    if n_robots == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'Phase 7 Trajectories and Metrics: Multi-Robot Swarm',
        fontsize=14, fontweight='bold'
    )
    
    for idx, (name, history) in enumerate(histories.items()):
        if len(history.times) == 0:
            continue
        
        times = np.array(history.times)
        robot_positions = np.array(history.robot_positions)
        intended_positions = np.array(history.intended_positions)
        contact_points = np.array(history.contact_point_positions)
        object_positions = np.array(history.object_positions)
        position_errors = np.array(history.position_errors)
        heading_errors = np.array(history.heading_errors)
        contact_forces = np.array(history.contact_forces)
        
        # Remove outliers from data before plotting
        # Position error: compute magnitude and remove outliers
        error_mags = np.linalg.norm(position_errors, axis=1)
        error_mags = remove_outliers(error_mags, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
        # Heading error: remove outliers (already in radians)
        heading_errors = remove_outliers(heading_errors, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
        # Contact force: remove outliers
        contact_forces = remove_outliers(contact_forces, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
        # Robot speed: compute and remove outliers
        robot_vels = np.array(history.robot_velocities)
        robot_speeds = np.linalg.norm(robot_vels[:, :2], axis=1)
        robot_speeds = remove_outliers(robot_speeds, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
        # Plot 1: Trajectory
        ax = axes[idx, 0]
        ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', linewidth=1.5, label='Robot')
        ax.plot(intended_positions[:, 0], intended_positions[:, 1], 'g--', linewidth=1, alpha=0.7, label='Intended pos')
        ax.plot(contact_points[:, 0], contact_points[:, 1], 'r--', linewidth=1, alpha=0.7, label='Contact point')
        ax.plot(object_positions[:, 0], object_positions[:, 1], 'k-', linewidth=1.5, alpha=0.8, label='Object')
        if len(robot_positions) > 0:
            ax.plot(robot_positions[0, 0], robot_positions[0, 1], 'go', markersize=8, label='Robot Start')
            ax.plot(robot_positions[-1, 0], robot_positions[-1, 1], 'ro', markersize=8, label='Robot End')
            if len(object_positions) > 0:
                ax.plot(object_positions[0, 0], object_positions[0, 1], 'ks', markersize=8, label='Object Start')
                ax.plot(object_positions[-1, 0], object_positions[-1, 1], 'rs', markersize=8, label='Object End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{name} - Trajectories (t_param={t_params[name]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot 2: Position error (already computed and filtered above)
        ax = axes[idx, 1]
        ax.plot(times, error_mags * 100, 'b-', linewidth=1.5)  # Convert to cm
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (cm)')
        ax.set_title(f'{name} - Position Error')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Heading error
        ax = axes[idx, 2]
        ax.plot(times, np.degrees(heading_errors), 'r-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading Error (deg)')
        ax.set_title(f'{name} - Heading Error')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Contact force
        ax = axes[idx, 3]
        ax.plot(times, contact_forces, 'r-', linewidth=1.5)
        ax.axhline(y=contact_threshold, color='g', linestyle='--', label='Threshold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact Force (N)')
        ax.set_title(f'{name} - Contact Force')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Robot speed (already computed and filtered above)
        ax = axes[idx, 4]
        ax.plot(times, robot_speeds, 'b-', linewidth=1.5, label='robot speed')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title(f'{name} - Robot Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Contact state
        ax = axes[idx, 5]
        in_contact = np.array(history.in_contact)
        ax.fill_between(times, 0, 1, where=in_contact, alpha=0.3, color='green', label='in contact')
        ax.fill_between(times, 0, 1, where=~in_contact, alpha=0.3, color='red', label='not in contact')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact State')
        ax.set_title(f'{name} - Contact State')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No Contact', 'In Contact'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Phase 1 results plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_phase_7beta(
    histories: Dict[str, Phase7History],
    t_params: Dict[str, float],
    save_path: Optional[Path] = None,
        ):
    """Plot Phase 7 Beta: Object-focused trajectory visualization with robot subplots.
    
    Layout (2x3 grid):
    - Object trajectory (x-y) spans subplots 1,2 (top row, left 2 columns) - MAIN FOCUS
    - Robot 1 trajectory in subplot 3 (top-right)
    - Robot 2 trajectory in subplot 4 (bottom-left)
    - Robot 3 trajectory in subplot 5 (bottom-middle)
    - Robot 4 trajectory in subplot 6 (bottom-right)
    
    Parameters
    ----------
    histories : Dict[str, Phase7History]
        History for each robot (all should have same object data)
    t_params : Dict[str, float]
        t_param for each robot
    save_path : Optional[Path]
        Path to save plot
    """
    if len(histories) == 0:
        print("No history to plot for Phase 7 Beta.")
        return
    
    # Use GridSpec for custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(
        f'Phase 7 Beta: Object and Robot Trajectories with Headings',
        fontsize=16, fontweight='bold'
    )
    
    # Get object data from first robot's history (all robots share same object)
    first_history = list(histories.values())[0]
    if len(first_history.times) == 0:
        print("No data in history for Phase 7 Beta.")
        return
    
    times = np.array(first_history.times)
    object_positions = np.array(first_history.object_positions)
    
    object_velocities = np.array(first_history.object_velocities)
    object_angular_velocities = np.array(first_history.object_angular_velocities)
    
    # Compute object orientation by integrating angular velocity
    # Start with initial orientation: compute from first velocity direction if available
    object_orientations = np.zeros_like(times)
    if len(times) > 0:
        # Initial orientation: use velocity direction if velocity is significant, otherwise 0
        if len(object_velocities) > 0 and np.linalg.norm(object_velocities[0]) > 0.01:
            object_orientations[0] = np.arctan2(object_velocities[0, 1], object_velocities[0, 0])
        else:
            object_orientations[0] = 0.0
        
        # Integrate angular velocity to get orientation over time
        for i in range(1, len(times)):
            dt_actual = times[i] - times[i-1]
            object_orientations[i] = object_orientations[i-1] + object_angular_velocities[i-1] * dt_actual
            # Normalize to [-pi, pi]
            object_orientations[i] = np.arctan2(np.sin(object_orientations[i]), np.cos(object_orientations[i]))
    
    # PLOT 1 & 2: Object trajectory (spans top-left and top-middle, 2 columns)
    ax_obj_traj = fig.add_subplot(gs[0, :2])  # Top row, first 2 columns
    
    # Plot object trajectory with heading arrows
    ax_obj_traj.plot(object_positions[:, 0], object_positions[:, 1], 'k-', linewidth=2.5, label='Object Trajectory', alpha=0.8)
    
    # Add heading arrows along trajectory (every Nth point)
    arrow_interval = max(1, len(object_positions) // 20)  # Show ~20 arrows
    for i in range(0, len(object_positions), arrow_interval):
        if i < len(object_positions) - 1:
            dx = 0.05 * np.cos(object_orientations[i])  # Arrow length
            dy = 0.05 * np.sin(object_orientations[i])
            ax_obj_traj.arrow(
                object_positions[i, 0], object_positions[i, 1],
                dx, dy,
                head_width=0.02, head_length=0.015,
                fc='red', ec='red', alpha=0.6, zorder=5
            )
    
    # Mark start and end
    if len(object_positions) > 0:
        ax_obj_traj.plot(object_positions[0, 0], object_positions[0, 1], 'go', markersize=10, label='Object Start', zorder=6)
        ax_obj_traj.plot(object_positions[-1, 0], object_positions[-1, 1], 'ro', markersize=10, label='Object End', zorder=6)
    
    ax_obj_traj.set_xlabel('X (m)', fontsize=12)
    ax_obj_traj.set_ylabel('Y (m)', fontsize=12)
    ax_obj_traj.set_title('Object Trajectory with Heading (Main Focus)', fontsize=14, fontweight='bold')
    ax_obj_traj.legend(fontsize=10)
    ax_obj_traj.grid(True, alpha=0.3)
    ax_obj_traj.axis('equal')
    
    # PLOT 3, 4, 5, 6: Robot trajectories (top-right, bottom-left, bottom-middle, bottom-right)
    robot_names = list(histories.keys())
    robot_subplot_positions = [
        (0, 2),  # Robot 1: top-right
        (1, 0),  # Robot 2: bottom-left
        (1, 1),  # Robot 3: bottom-middle
        (1, 2),  # Robot 4: bottom-right
    ]
    
    for idx, (robot_idx, (row, col)) in enumerate(zip(range(min(4, len(robot_names))), robot_subplot_positions)):
        if robot_idx >= len(robot_names):
            break
        
        name = robot_names[robot_idx]
        history = histories[name]
        
        if len(history.times) == 0:
            continue
        
        robot_times = np.array(history.times)
        robot_positions = np.array(history.robot_positions)
        robot_headings = np.array(history.robot_headings)
        contact_points = np.array(history.contact_point_positions)
        
        ax_robot = fig.add_subplot(gs[row, col])
        
        # Plot robot trajectory with heading arrows
        ax_robot.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', linewidth=2, label=f'{name} Trajectory', alpha=0.8)
        ax_robot.plot(contact_points[:, 0], contact_points[:, 1], 'r--', linewidth=1, alpha=0.5, label='Contact Points')
        
        # Add heading arrows along robot trajectory
        arrow_interval = max(1, len(robot_positions) // 15)  # Show ~15 arrows
        for i in range(0, len(robot_positions), arrow_interval):
            if i < len(robot_positions):
                dx = 0.03 * np.cos(robot_headings[i])
                dy = 0.03 * np.sin(robot_headings[i])
                ax_robot.arrow(
                    robot_positions[i, 0], robot_positions[i, 1],
                    dx, dy,
                    head_width=0.015, head_length=0.01,
                    fc='blue', ec='blue', alpha=0.5, zorder=5
                )
        
        # Mark start and end
        if len(robot_positions) > 0:
            ax_robot.plot(robot_positions[0, 0], robot_positions[0, 1], 'go', markersize=8, label='Start', zorder=6)
            ax_robot.plot(robot_positions[-1, 0], robot_positions[-1, 1], 'ro', markersize=8, label='End', zorder=6)
        
        ax_robot.set_xlabel('X (m)', fontsize=10)
        ax_robot.set_ylabel('Y (m)', fontsize=10)
        ax_robot.set_title(f'{name} Trajectory (t_param={t_params[name]:.3f})', fontsize=11, fontweight='bold')
        ax_robot.legend(fontsize=8, loc='upper right')
        ax_robot.grid(True, alpha=0.3)
        ax_robot.axis('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Phase 7 Beta plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_phase7_wheel_plot(
    histories: Dict[str, Phase7History],
    t_params: Dict[str, float],
    save_path: Optional[Path] = None,
        ):
    """Plot Phase 7 wheel velocities for all robots.
    
    Parameters
    ----------
    histories : Dict[str, Phase7History]
        History for each robot
    t_params : Dict[str, float]
        t_param for each robot
    save_path : Optional[Path]
        Path to save plot
    """
    if len(histories) == 0:
        print("No history to plot for Phase 7 wheel velocities.")
        return
    
    # Determine number of wheels from first robot's history
    first_history = list(histories.values())[0]
    if len(first_history.wheel_velocities) == 0:
        print("No wheel velocity data available.")
        return
    
    # Check if we have valid wheel velocity data
    valid_wheel_data = [wv for wv in first_history.wheel_velocities if len(wv) > 0]
    if len(valid_wheel_data) == 0:
        print("No valid wheel velocity data available.")
        return
    
    num_wheels = len(valid_wheel_data[0])
    
    # Create subplots: one row per robot, num_wheels + 1 columns (one per wheel + summary)
    n_robots = len(histories)
    fig, axes = plt.subplots(n_robots, num_wheels + 1, figsize=(5 * (num_wheels + 1), 4 * n_robots))
    if n_robots == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'Phase 7 Wheel Velocities: Multi-Robot Swarm (Commanded = solid, Actual = dashed)',
        fontsize=14, fontweight='bold'
    )
    
    for idx, (name, history) in enumerate(histories.items()):
        if len(history.times) == 0:
            continue
        
        times = np.array(history.times)
        wheel_vels_list = history.wheel_velocities
        wheel_cmd_list = getattr(history, "wheel_cmd_velocities", [])
        
        # Filter out empty arrays and convert to numpy array
        valid_indices = [i for i, wv in enumerate(wheel_vels_list) if len(wv) > 0]
        if len(valid_indices) == 0:
            continue
        
        # Extract valid wheel velocities
        wheel_vels_array = np.array([wheel_vels_list[i] for i in valid_indices])
        valid_times = times[valid_indices]

        # Extract commanded wheel velocities if present and aligned
        wheel_cmd_array = None
        if len(wheel_cmd_list) == len(wheel_vels_list):
            if all(len(wheel_cmd_list[i]) > 0 for i in valid_indices):
                wheel_cmd_array = np.array([wheel_cmd_list[i] for i in valid_indices])

        # Quick sanity stats (helps diagnose "flat" wheel plots)
        try:
            w_min = float(np.min(wheel_vels_array))
            w_max = float(np.max(wheel_vels_array))
            w_std = float(np.std(wheel_vels_array))
            print(f"[wheel_debug] {name}: min={w_min:.3f}, max={w_max:.3f}, std={w_std:.3f} rad/s, samples={len(valid_times)}")
        except Exception:
            pass
        
        # Plot individual wheel velocities
        # Focus on commanded velocities (solid line) with actual velocities as reference (dashed)
        colors = ['b', 'r', 'g', 'm', 'c', 'orange']
        for wheel_idx in range(num_wheels):
            ax = axes[idx, wheel_idx]
            if wheel_idx < wheel_vels_array.shape[1]:
                # Plot commanded velocities as solid line (primary focus)
                if wheel_cmd_array is not None and wheel_idx < wheel_cmd_array.shape[1]:
                    wheel_cmd_clean = remove_outliers(
                        wheel_cmd_array[:, wheel_idx],
                        method="percentile",
                        lower_percentile=1.0,
                        upper_percentile=99.0,
                    )
                    wheel_cmd_clean = np.round(wheel_cmd_clean, 4)
                    ax.plot(
                        valid_times,
                        wheel_cmd_clean,
                        colors[wheel_idx % len(colors)],
                        linewidth=2.0,
                        label=f'Wheel {wheel_idx+1} cmd',
                    )
                
                # Plot actual velocities as dashed line (reference)
                wheel_vels = wheel_vels_array[:, wheel_idx]
                wheel_vels_clean = remove_outliers(
                    wheel_vels,
                    method="percentile",
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                )
                wheel_vels_clean = np.round(wheel_vels_clean, 4)
                ax.plot(valid_times, wheel_vels_clean, colors[wheel_idx % len(colors)] + "--", 
                       linewidth=1.5, alpha=0.7, label=f'Wheel {wheel_idx+1} actual')
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Angular Velocity (rad/s)')
                ax.set_title(f'{name} - Wheel {wheel_idx+1} (t_param={t_params[name]:.3f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Plot summary: all wheels together
        # Focus on commanded velocities (solid) with actual velocities as reference (dashed)
        ax_summary = axes[idx, num_wheels]
        for wheel_idx in range(num_wheels):
            if wheel_idx < wheel_vels_array.shape[1]:
                # Plot commanded velocities as solid line (primary focus)
                if wheel_cmd_array is not None and wheel_idx < wheel_cmd_array.shape[1]:
                    wheel_cmd_clean = remove_outliers(
                        wheel_cmd_array[:, wheel_idx],
                        method="percentile",
                        lower_percentile=1.0,
                        upper_percentile=99.0,
                    )
                    wheel_cmd_clean = np.round(wheel_cmd_clean, 4)
                    ax_summary.plot(
                        valid_times,
                        wheel_cmd_clean,
                        colors[wheel_idx % len(colors)],
                        linewidth=2.0,
                        alpha=0.8,
                        label=f'Wheel {wheel_idx+1} cmd',
                    )
                
                # Plot actual velocities as dashed line (reference)
                wheel_vels = wheel_vels_array[:, wheel_idx]
                wheel_vels_clean = remove_outliers(
                    wheel_vels,
                    method="percentile",
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                )
                wheel_vels_clean = np.round(wheel_vels_clean, 4)
                ax_summary.plot(valid_times, wheel_vels_clean, colors[wheel_idx % len(colors)] + "--", 
                              linewidth=1.5, alpha=0.5, label=f'Wheel {wheel_idx+1} actual')
        ax_summary.set_xlabel('Time (s)')
        ax_summary.set_ylabel('Angular Velocity (rad/s)')
        ax_summary.set_title(f'{name} - All Wheels Summary')
        ax_summary.legend()
        ax_summary.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Phase 7 wheel velocity plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def export_histories(
    histories: Dict[str, Phase7History],
    t_params: Dict[str, float],
    save_path: Path,
        ):
    """Export histories and t_params to JSON file.
    
    Parameters
    ----------
    histories : Dict[str, Phase7History]
        History for each robot
    t_params : Dict[str, float]
        t_param for each robot
    save_path : Path
        Path to save JSON file
    """
    # Convert Phase7History dataclass to JSON-serializable dict
    export_data = {
        "t_params": t_params,
        "histories": {}
    }
    
    for name, history in histories.items():
        # Convert numpy arrays to lists for JSON serialization
        # Also convert numpy booleans to Python booleans
        history_dict = {
            "times": [float(t) for t in history.times],
            "robot_positions": [pos.tolist() for pos in history.robot_positions],
            "robot_headings": [float(h) for h in history.robot_headings],
            "robot_velocities": [vel.tolist() for vel in history.robot_velocities],
            "intended_positions": [pos.tolist() for pos in history.intended_positions],
            "position_errors": [err.tolist() for err in history.position_errors],
            "desired_headings": [float(h) for h in history.desired_headings],
            "heading_errors": [float(e) for e in history.heading_errors],
            "contact_point_positions": [pos.tolist() for pos in history.contact_point_positions],
            "contact_point_velocities": [vel.tolist() for vel in history.contact_point_velocities],
            "object_positions": [pos.tolist() for pos in history.object_positions],
            "object_velocities": [vel.tolist() for vel in history.object_velocities],
            "object_angular_velocities": [float(omega) for omega in history.object_angular_velocities],
            "contact_forces": [float(f) for f in history.contact_forces],
            "in_contact": [bool(ic) for ic in history.in_contact],  # Convert numpy bool_ to Python bool
            "v_base_history": [float(v) for v in history.v_base_history],
            "v_ff_history": [float(v) for v in history.v_ff_history],
            "v_pi_history": [float(v) for v in history.v_pi_history],
            "desired_contact_point_speeds": [float(s) for s in history.desired_contact_point_speeds],
            "wheel_velocities": [wv.tolist() if len(wv) > 0 else [] for wv in history.wheel_velocities],
            "wheel_cmd_velocities": [wv.tolist() if len(wv) > 0 else [] for wv in getattr(history, "wheel_cmd_velocities", [])],
        }
        export_data["histories"][name] = history_dict
    
    # Save to JSON
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported histories to {save_path}")


def import_histories(
    load_path: Path,
        ):
    """Import histories and t_params from JSON file.
    
    Parameters
    ----------
    load_path : Path
        Path to load JSON file from
    
    Returns
    -------
    histories : Dict[str, Phase7History]
        History for each robot
    t_params : Dict[str, float]
        t_param for each robot
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"History file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        import_data = json.load(f)
    
    t_params = import_data["t_params"]
    histories = {}
    
    for name, history_dict in import_data["histories"].items():
        # Convert lists back to numpy arrays
        history = Phase7History(
            times=history_dict["times"],
            robot_positions=[np.array(pos) for pos in history_dict["robot_positions"]],
            robot_headings=history_dict["robot_headings"],
            robot_velocities=[np.array(vel) for vel in history_dict["robot_velocities"]],
            intended_positions=[np.array(pos) for pos in history_dict["intended_positions"]],
            position_errors=[np.array(err) for err in history_dict["position_errors"]],
            desired_headings=history_dict["desired_headings"],
            heading_errors=history_dict["heading_errors"],
            contact_point_positions=[np.array(pos) for pos in history_dict["contact_point_positions"]],
            contact_point_velocities=[np.array(vel) for vel in history_dict["contact_point_velocities"]],
            object_positions=[np.array(pos) for pos in history_dict["object_positions"]],
            object_velocities=[np.array(vel) for vel in history_dict["object_velocities"]],
            object_angular_velocities=history_dict["object_angular_velocities"],
            contact_forces=history_dict["contact_forces"],
            in_contact=history_dict["in_contact"],
            v_base_history=history_dict["v_base_history"],
            v_ff_history=history_dict["v_ff_history"],
            v_pi_history=history_dict["v_pi_history"],
            desired_contact_point_speeds=history_dict["desired_contact_point_speeds"],
            wheel_velocities=[np.array(wv) if len(wv) > 0 else np.array([]) for wv in history_dict.get("wheel_velocities", [])],
            wheel_cmd_velocities=[np.array(wv) if len(wv) > 0 else np.array([]) for wv in history_dict.get("wheel_cmd_velocities", [])],
        )
        histories[name] = history
    
    print(f"Imported histories from {load_path}")
    return histories, t_params


def main():
    parser = argparse.ArgumentParser(description="Swarm Magnum Four navigation / contact test")
    parser.add_argument("--object", type=str, default="right_triangle",
                        choices=["right_triangle", "bolt", "pi", "root", "rect", "hourglass", "meteor"],
                        help="Object shape name (must have OBJ and DXF files in urdf directory)")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Test duration in seconds")
    parser.add_argument("--no-gui", action="store_true", help="Run headless")
    parser.add_argument("--kinematics", "-k", default="holonomic",
                        choices=["holonomic", "diffdrive"])
    parser.add_argument("--model", "-m", default="wheel", choices=["wheel"],
                        help="Robot model (wheel only)")
    parser.add_argument("--magnum-verbose", action="store_true", help="Verbose Magnum Four search logs")
    parser.add_argument("--magnum-visualize", action="store_true", help="Visualize Magnum Four search (matplotlib)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save plots (only for Phase 7 velocity controller)")
    parser.add_argument("--record-video", action="store_true",
                        help="Record PyBullet simulation as video (top-down view). Requires --save-dir.")
    parser.add_argument("--load-histories", type=str, default=None,
                        help="Load histories from JSON file and plot (skip simulation). Requires --save-dir.")
    parser.add_argument("--desired-path", type=str, default="square",
                        choices=["square", "rectangle", "sine", "zigzag", "p_trajectory"],
                        help="Desired path type: 'square', 'rectangle', 'sine', 'zigzag', or 'p_trajectory' (uses PathFollowingController)")
    parser.add_argument(
        "--startup-mode",
        type=str,
        default="quick",
        choices=["quick", "full"],
        help=(
            "Startup behavior: 'quick' skips APF reaching and starts direct approach "
            "(rotate then creep); 'full' keeps REACHING->APPROACHING pipeline."
        ),
    )
    
    args = parser.parse_args()
    
    # If loading histories, handle it separately
    if args.load_histories is not None:
        if args.save_dir is None:
            parser.error("--save-dir is required when using --load-histories")
        
        load_path = Path(args.load_histories)
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Load histories
        histories, t_params = import_histories(load_path)
        
        # Extract path type and object name from filename if possible
        # Filename format: histories_{path_type}_w_{object}.json
        path_type = "square"  # default
        object_name = "unknown"  # default
        load_path_str = str(load_path)
        if "_" in load_path_str:
            # Remove .json extension and split by underscores
            parts = load_path_str.replace(".json", "").split("_")
            # Look for pattern: histories_{path_type}_w_{object}
            if len(parts) >= 4 and parts[0] == "histories" and parts[2] == "w":
                path_type = parts[1]
                object_name = parts[3]
            elif len(parts) > 1:
                # Fallback: try to find path_type in last part
                if parts[-1] in ["square", "rectangle", "sine", "zigzag"]:
                    path_type = parts[-1]
        
        # Generate plots
        plot_phase7_velocities(
            histories=histories,
            t_params=t_params,
            desired_obj_velocity=np.array([0.0, 0.0]),  # Reference for plotting
            desired_obj_omega=0.0,
            save_path=save_path / f"phase7_swarm_velocities_{path_type}_w_{object_name}.png",
        )
        
        plot_phase_1_results(
            histories=histories,
            t_params=t_params,
            contact_threshold=2.0,
            save_path=save_path / f"phase7_swarm_trajectories_{path_type}_w_{object_name}.png",
        )
        
        plot_phase_7beta(
            histories=histories,
            t_params=t_params,
            save_path=save_path / f"phase7_beta_trajectories_{path_type}_w_{object_name}.png",
        )

        plot_phase7_wheel_plot(
            histories=histories,
            t_params=t_params,
            save_path=save_path / f"phase7_wheel_velocities_{path_type}_w_{object_name}.png",
        )
        
        print(f"Generated plots from loaded histories to {save_path}")
        return

    # ROS package path setup (same as other tests)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("contact_maintain")
    sys.path.insert(0, str(Path(pkg_path) / "src"))
    sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))


    setup_pybullet(gui=not args.no_gui)

    selected_name = args.object
    
    # Mapping from shape names to OBJ file names
    obj_file_map = {
        'right_triangle': 'right_triangle.obj',
        'bolt': 'bolt.obj',
        'pi': 'pi.obj',
        'root': 'root.obj',
        'rect': 'rect.obj',
        "hourglass": "hourglass.obj",
        "meteor": "meteor.obj",
    }
    
    # All shapes use OBJ method (requires both OBJ and DXF files)
    if selected_name not in obj_file_map:
        raise ValueError(
            f"Unknown object '{selected_name}'. "
            f"Available: {', '.join(obj_file_map.keys())}"
        )
    


    obj_file = obj_file_map[selected_name]
    print(f"Loading OBJ file: {obj_file} for shape '{selected_name}'...")
    
    generic_object, object_uid = obj_to_generic(
        obj_path=obj_file,
        shape_name=selected_name,
        position=(0, 0, 0.2),
        orientation=0.0,
        mass=1.0,
        lateral_friction=DEFAULT_OBJECT_FRICTION,
        blind_test=True,
    )
    contact_point_parameterization = ContactPointParameterization(generic_object)
    print(f"✓ Loaded object: {selected_name}")
    print(f"  Mass: {generic_object.mass:.3f} kg")
    print(f"  Moment of inertia: {generic_object.moment_of_inertia:.6f} kg·m²")
    print(f"  Boundary length: {generic_object.boundary_length:.3f} m")

    # Compute Magnum Four contacts / t_params
    # Check cache first to avoid recomputing optimal contact points
    cache_file = Path(pkg_path) / "urdf" / "magnum_four_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load cache if it exists
    cached_t_params = None
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                if selected_name in cache_data:
                    cached_t_params = cache_data[selected_name]
                    print(f"\nFound cached Magnum Four t_params for '{selected_name}': {cached_t_params}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load cache: {e}")
    
    if cached_t_params is not None:
        # Use cached solution (similar to hardcoded solution approach)
        t_params = cached_t_params
        print(f"Using cached Magnum Four t_params for '{selected_name}': {[f'{v:.4f}' for v in t_params]}")
        
        # Create ContactPoint objects from cached t_params (similar to hardcoded solution)
        contacts = []
        for t_param in t_params:
            temp_contact = contact_point_parameterization.get_contact_info(t_param)
            contacts.append(ContactPoint(
                position=temp_contact['point'],
                tangent=temp_contact['tangent'],
                normal_outward=temp_contact['normal_outward'],
                normal_inward=temp_contact['normal_inward'],
                parameter=t_param,
                force_direction=None,
                object_ref=generic_object,
            ))
    else:
        # No cache found, run solver and save result
        print(f"\nComputing Magnum Four contact points for '{selected_name}'...")
        magnum_result = find_the_magnum_four_v3(
            generic_object,
            verbose=args.magnum_verbose,
            visualize=args.magnum_visualize and (not args.no_gui),
            weighting_scheme="balanced",
            torque_method=3,
        )
        if not magnum_result or not magnum_result.get("success", False):
            raise RuntimeError("Magnum Four solver failed to produce a solution.")

        contacts = magnum_result["best_solution"]["contacts"]
        t_params = [float(c.parameter) for c in contacts]
        t_params = [tp % 1.0 for tp in t_params]
        t_params = np.array(t_params)
        t_params = t_params.tolist()
        
        if len(t_params) != 4:
            raise RuntimeError(f"Expected 4 contacts from Magnum Four, got {len(t_params)}")
        
        print(f"\nMagnum Four t_params: {[f'{v:.4f}' for v in t_params]}")
        
        # Save to cache
        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
        
        cache_data[selected_name] = t_params
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved Magnum Four solution to cache: {cache_file}")

    # Create 4 robots - spawn near their target t_params for faster testing
    robots: Dict[str, object] = {}
    robot_agents: Dict[str, RobotAgent] = {}
    for i in range(4):
        name = f"R_{i+1:02d}"
        # Get target t_param for this robot
        target_t_param = t_params[i]
        
        # Get contact point info at target t_param (in object body frame)
        contact_info = contact_point_parameterization.get_contact_info(target_t_param)
        contact_point_body = np.array(contact_info['point'], dtype=float)
        normal_outward = np.array(contact_info['normal_outward'], dtype=float)
        normal_inward = np.array(contact_info['normal_inward'], dtype=float)
        
        # Calculate spawn position: contact_point + approach_distance * normal_outward
        # Object is at origin (0, 0, 0), so body frame = world frame initially
        spawn_position_body = contact_point_body + APPROACH_DISTANCE * normal_outward
        robot_x = float(spawn_position_body[0])
        robot_y = float(spawn_position_body[1])
        
        # Robot heading: point toward contact point (normal_inward direction)
        robot_heading = float(np.arctan2(normal_inward[1], normal_inward[0]))

        robot = create_robot(
            kinematics=args.kinematics,
            model=args.model,
            position=(robot_x, robot_y),
            orientation=robot_heading,
            name=name,
        )
        robots[name] = robot
        print(f"Spawned {name} at ({robot_x:.3f}, {robot_y:.3f}) with heading {robot_heading:.3f} rad, "
              f"target t_param={target_t_param:.4f}")

        # Create agent for this robot (velocity-based pushing only)
        agent = RobotAgent(
            robot=robot,
            name=name,
            object_uid=object_uid,
            generic_object=generic_object,
            navigation_type="apf",
            pushing_type="velocity",
            force_distributor=None,
        )
        robot_agents[name] = agent

    # Configure Phase 7 controllers for VELOCITY mode (only mode used here)
    phase7_controllers = {}
    object_velocity_pid = None
    waypoint_controller = None
    path_following_controller = None  # For p_trajectory path type
    
    # Generate waypoints based on desired path type (velocity pushing only)
    def generate_waypoints(path_type: str) :
            """Generate waypoints for different path types.
            
            Parameters
            ----------
            path_type : str
                Path type: 'square', 'rectangle', 'sine', or 'zigzag'
            
            Returns
            -------
            List[tuple[np.ndarray, float]]
                List of (position, orientation) tuples
            """
            if path_type == "square":
                # Square: 2m x 2m (from -1 to 1 in both axes)
                # Starting from top-right corner, going clockwise
                return [
                    (np.array([1.0, 1.0]), 0.0),           # Top-right: pointing right (0°)
                    (np.array([1.0, -1.0]), 0),     # Bottom-right: pointing down (-90°)
                    (np.array([-1.0, -1.0]), 0),       # Bottom-left: pointing left (180°)
                    (np.array([-1.0, 1.0]), 0),      # Top-left: pointing up (90°)
                ]
            
            elif path_type == "rectangle":
                # Rectangle: 3m x 1.5m (from -1.5 to 1.5 in x, -0.75 to 0.75 in y)
                # Starting from top-right corner, going clockwise
                return [
                    (np.array([1.5, 0.75]), 0.0),          # Top-right: pointing right (0°)
                    (np.array([1.5, -0.75]), -np.pi/2),     # Bottom-right: pointing down (-90°)
                    (np.array([-1.5, -0.75]), np.pi),      # Bottom-left: pointing left (180°)
                    (np.array([-1.5, 0.75]), np.pi/2),     # Top-left: pointing up (90°)
                ]
            
            elif path_type == "sine":
                # Sine wave: Generate waypoints along a sine wave path
                # Path: x from 0 to 4, y = 1.0 * sin(pi * x / 2) (starting from y=0 at x=0)
                num_waypoints = 12  # More waypoints for smooth sine wave
                waypoints = []
                x_range = np.linspace(0.0, 4.0, num_waypoints)
                
                for i, x in enumerate(x_range):
                    y = 1.0 * np.sin(np.pi * x / 2.0)  # Amplitude increased to 1.0, starts at y=0
                    
                    # Compute orientation: tangent to the sine wave using analytical derivative
                    # dy/dx = 1.0 * (pi/2) * cos(pi * x / 2) = (pi/2) * cos(pi * x / 2)
                    dy_dx = (np.pi / 2.0) * np.cos(np.pi * x / 2.0)
                    orientation = np.arctan2(dy_dx, 1.0)  # atan2(dy/dx, 1) = atan(dy/dx)
                    
                    waypoints.append((np.array([x, y]), orientation))
                
                return waypoints
            
            elif path_type == "zigzag":
                # Zigzag/square wave: Generate waypoints along a zigzag path
                # Path: alternating horizontal and diagonal segments, starting from x=0, y=0
                # Pattern: right -> down-right -> right -> up-right -> right -> down-right -> ...
                num_segments = 8  # Number of zigzag segments
                waypoints = []
                
                x_start = 0.0  # Start from x=0
                x_end = 4.0
                y_center = 0.0  # Center y position
                y_amplitude = 0.75  # Vertical amplitude of zigzag
                segment_length = (x_end - x_start) / num_segments
                
                for i in range(num_segments + 1):
                    x = x_start + i * segment_length
                    
                    # Alternate between top and bottom, starting from y=0 (i=0 at center, then alternate)
                    if i == 0:
                        y = y_center  # First waypoint at y=0
                    elif i % 2 == 1:
                        y = y_center + y_amplitude  # Top (odd indices)
                    else:
                        y = y_center - y_amplitude  # Bottom (even indices after first)
                    
                    # Compute orientation based on direction to next waypoint
                    if i < num_segments:
                        x_next = x_start + (i + 1) * segment_length
                        if i == 0:
                            # Next waypoint after first (i=0) goes to top
                            y_next = y_center + y_amplitude
                        elif (i + 1) % 2 == 1:
                            y_next = y_center + y_amplitude
                        else:
                            y_next = y_center - y_amplitude
                        
                        dx = x_next - x
                        dy = y_next - y
                        orientation = np.arctan2(dy, dx)
                    else:
                        # Last waypoint: keep same orientation as previous segment
                        orientation = waypoints[-1][1] if waypoints else 0.0
                    
                    waypoints.append((np.array([x, y]), orientation))
                
                return waypoints
            
            else:
                raise ValueError(f"Unknown path type: {path_type}")

    # Branch based on path type: p_trajectory uses PathFollowingController, others use waypoint system
    if args.desired_path == "p_trajectory":
        # Create P-trajectory hybrid path using PathFollowingController
        # The path starts at object's current position
        obj_state_init = get_object_state(object_uid)
        start_point = obj_state_init["position"].tolist()
        
        p_path = create_p_trajectory_hybrid_path(
            start_point=start_point,
            stem_height=0.8,  # 0.8m vertical stem
            arc_radius=0.3,   # 0.3m arc radius (smaller for simulation)
            arc_center_offset=0.3,  # Match radius
        )
        
        # Create PathFollowingController
        path_following_controller = PathFollowingController(
            hybrid_path=p_path,
            a_max=0.15,        # m/s² (conservative for contact maintenance)
            a_lat_max=0.08,    # m/s² (conservative for contact maintenance)
            v_user_max=0.1,    # m/s (slow for pushing)
            look_ahead=2,      # Smart mode (considers transition curvature)
            use_tracking=False,
        )
        
        print(f"\n{'='*60}")
        print(f"USING PathFollowingController for P-trajectory path")
        print(f"{'='*60}")
        print(f"  Path: P-trajectory")
        print(f"  Start point: {start_point}")
        print(f"  Stem height: 0.8 m")
        print(f"  Arc radius: 0.3 m")
        print(f"  Total path length: {p_path.total_length:.3f} m")
        print(f"  Estimated traversal time: {path_following_controller.velocity_planner.get_total_time():.3f} s")
        print(f"  Number of segments: {p_path.num_components}")
        print(f"{'='*60}\n")
        
    else:
        # Use waypoint-based controller for square, rectangle, sine, zigzag
        waypoints = generate_waypoints(args.desired_path)

        # Create waypoint controller
        waypoint_controller = WaypointController(
            waypoints=waypoints,
            position_tolerance=0.05,  # Switch when within 5cm
            orientation_tolerance=0.1,  # Switch when orientation error < 0.1 rad (~6°)
            velocity_threshold=0.02,  # Switch when velocity < 2cm/s
            angular_velocity_threshold=0.05,  # Switch when angular velocity < 0.05 rad/s
        )
        
        # Get initial waypoint
        initial_goal_position, initial_goal_orientation = waypoint_controller.get_current_waypoint()
        
        # Create PID controller for object velocity control
        object_velocity_pid = ObjectVelocityPIDController(
            goal_position=initial_goal_position,
            goal_orientation=initial_goal_orientation,
        )
        print(f"Created WaypointController with {len(waypoints)} waypoints")
        print(f"  Waypoint 0: position={initial_goal_position}, orientation={initial_goal_orientation:.3f} rad")
        print(
            f"Created ObjectVelocityPIDController with initial goal_position={initial_goal_position}, "
            f"goal_orientation={initial_goal_orientation:.3f} rad"
        )
    
    # Phase 7 controllers will be created in the control loop with dynamic desired velocities
    # from the PID controller
    for name, agent in robot_agents.items():
            # Get the target t_param for this robot
            robot_idx = list(robot_agents.keys()).index(name)
            target_t_param = t_params[robot_idx]
            
            # Create Phase 7 controller for this robot (desired velocities will be updated in loop)
            # Initial desired velocities (will be updated by PID controller)
            initial_desired_obj_velocity = np.array([0.0, 0.0])
            initial_desired_obj_omega = 0.0
            phase7_controllers[name] = Phase7BetaVerDecouple(
                robot_uid=robots[name].uid,
                object_uid=object_uid,
                generic_object=generic_object,
                t_param=target_t_param,
                desired_object_velocity=initial_desired_obj_velocity,
                desired_object_angular_velocity=initial_desired_obj_omega,
            )
            print(f"Created Phase 7 controller for {name} with t_param={target_t_param:.4f}")

    host = SwarmHost(
        robot_agents=robot_agents,
        object_uid=object_uid,
        generic_object=generic_object,
        startup_mode=args.startup_mode,
    )
    print(f"Startup mode: {args.startup_mode}")

    # Assign targets: map robot names in order to Magnum contacts
    target_map = {name: t_params[i] for i, name in enumerate(robots.keys())}
    print(f"Assigned targets: { {k: round(v, 4) for k, v in target_map.items()} }")
    host.assign_targets(target_map)

    # Setup video recording if requested
    video_log_id = None
    video_path = None
    if args.record_video and not args.no_gui:
        if not args.save_dir:
            raise ValueError("--record-video requires --save-dir to be specified")

        # Always save as phase7_topview_{path_type}_w_{object}.mp4 in save_dir
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_path = save_dir / f"phase7_topview_{args.desired_path}_w_{args.object}.mp4"

        video_log_id = setup_video_recording(video_path, object_uid)


    # Run sim loop
    n_steps = int(args.duration / TIMESTEP)
    step_count = 0
    t = 0.0
    # Counter to decimate high-level PID updates relative to Phase 7 controller
    pid_cycle_count = 0
    
    # Flag to delay PathFollowingController start until all robots are in pushing mode
    # This ensures we don't miss the beginning of the path while robots are still navigating
    path_following_started = False
    
    for _ in range(n_steps):
        obj_state = get_object_state(object_uid)

        if step_count % CTRL_STEP == 0:
            host.update(1.0 / CTRL_FREQ, obj_state)

            # Update velocity controller (PathFollowingController or Waypoint+PID)
            pid_cycle_count += 1
            if pid_cycle_count % PID_DECIMATION == 0:
                
                if path_following_controller is not None:
                    # Use PathFollowingController for p_trajectory
                    # But only start after ALL robots are in pushing mode
                    
                    # Check if all robots are in pushing mode
                    all_pushing = all(agent.goal_type == "push" for agent in robot_agents.values())
                    
                    if not path_following_started and all_pushing:
                        # All robots are now pushing - start the path following!
                        path_following_started = True
                        print(f"\n{'='*60}")
                        print("ALL ROBOTS IN PUSHING MODE - STARTING PATH FOLLOWING!")
                        print(f"  Simulation time: {t:.2f} s")
                        print(f"  Path length: {path_following_controller.hybrid_path.total_length:.3f} m")
                        print(f"  Estimated traversal time: {path_following_controller.velocity_planner.get_total_time():.3f} s")
                        print(f"{'='*60}\n")
                    
                    if path_following_started:
                        # Compute velocity command from the path following controller
                        dt_pid = (1.0 / CTRL_FREQ) * PID_DECIMATION  # Time since last PID update
                        
                        # Get velocity command (vx, vy, omega) from path following controller
                        velocity_cmd = path_following_controller.compute_velocity(dt=dt_pid)
                        
# IMPORTANT, always remind me here lets try to see what can go wrong
                        velocity_cmd[2] = 0
                        # Extract linear velocity and angular velocity
                        desired_obj_velocity = np.array([velocity_cmd[0], velocity_cmd[1]])
                        desired_obj_omega = velocity_cmd[2]
                        
                        # Progress report (every 100 PID cycles ~= every 2 seconds at 100Hz/20)
                        if pid_cycle_count % (PID_DECIMATION * 100) == 0:
                            progress = path_following_controller.get_progress_fraction() * 100
                            current_s = path_following_controller.get_current_s()
                            elapsed = path_following_controller.get_elapsed_time()
                            print(f"[PathFollowing] t={elapsed:.1f}s, s={current_s:.3f}m ({progress:.1f}%), "
                                  f"cmd=[{velocity_cmd[0]:.3f}, {velocity_cmd[1]:.3f}, {velocity_cmd[2]:.3f}]")
                        
                        # Check if path following is complete
                        if path_following_controller.is_completed():
                            print(f"\n{'='*60}")
                            print("PATH FOLLOWING COMPLETED!")
                            print(f"  Total time: {path_following_controller.get_elapsed_time():.3f} s")
                            print(f"  Total distance: {path_following_controller.hybrid_path.total_length:.3f} m")
                            print(f"{'='*60}\n")
                            # Keep desired velocity at zero after completion
                            desired_obj_velocity = np.array([0.0, 0.0])
                            desired_obj_omega = 0.0
                    else:
                        # Not yet started - robots still navigating to object
                        # Output zero velocity (no object movement during approach)
                        desired_obj_velocity = np.array([0.0, 0.0])
                        desired_obj_omega = 0.0
                    
                    # Update all Phase 7 controllers with new desired velocities
                    for controller in phase7_controllers.values():
                        controller.desired_object_velocity = desired_obj_velocity
                        controller.desired_object_angular_velocity = desired_obj_omega
                
                elif waypoint_controller is not None and object_velocity_pid is not None:
                    # Use Waypoint + PID controller for square, rectangle, sine, zigzag
                    # Check if we should switch to next waypoint
                    waypoint_switched = waypoint_controller.update(
                        current_position=obj_state["position"],
                        current_orientation=obj_state["orientation"],
                        current_velocity=obj_state["velocity"],
                        current_angular_velocity=obj_state["angular_velocity"],
                    )

                    # If waypoint switched, update PID controller goal
                    if waypoint_switched:
                        new_goal_position, new_goal_orientation = waypoint_controller.get_current_waypoint()
                        object_velocity_pid.set_goal(new_goal_position, new_goal_orientation)
                        print(
                            f"Switched to waypoint {waypoint_controller.current_waypoint_idx}: "
                            f"position={new_goal_position}, orientation={new_goal_orientation:.3f} rad"
                        )

                    # Compute desired velocities using PID controller
                    desired_obj_velocity, desired_obj_omega = object_velocity_pid.compute_desired_velocity(
                        current_position=obj_state["position"],
                        current_orientation=obj_state["orientation"],
                        current_velocity=obj_state["velocity"],
                        current_angular_velocity=obj_state["angular_velocity"],
                    )
                    # Update all Phase 7 controllers with new desired velocities
                    for controller in phase7_controllers.values():
                        controller.desired_object_velocity = desired_obj_velocity
                        controller.desired_object_angular_velocity = desired_obj_omega

            for name, agent in robot_agents.items():
                other_positions = [
                    robot_agents[other_name].robot.get_state()[0]
                    for other_name in robot_agents.keys()
                    if other_name != name
                ]

                # Use Phase 7 controller if in pushing mode
                if agent.goal_type == "push" and name in phase7_controllers:
                    # Use Phase 7 controller instead of agent's internal controller
                    # Update contact state from RobotAgent (uses correct link index)
                    agent.update_contact_state()
                    robot_pos, robot_heading, _ = agent.robot.get_state()
                    controller = phase7_controllers[name]
                    record_history = (args.save_dir is not None)
                    cmd = controller.compute_velocity(
                        robot_pos=robot_pos,
                        robot_heading=robot_heading,
                        object_pos=obj_state["position"],
                        object_orientation=obj_state["orientation"],
                        object_velocity=obj_state["velocity"],
                        object_angular_velocity=obj_state["angular_velocity"],
                        contact_force=agent.contact_force,
                        in_contact=agent.in_contact,
                        t=t,
                        record_history=record_history,
                        robot=agent.robot,  # Pass robot object to get actual wheel velocities
                    )
                else:
                    # Use normal agent controller (navigation, approach)
                    obstacles = None
                    if agent.goal_type == "navigate":
                        obstacles = get_object_as_obstacle(
                            generic_object, obj_state["position"], obj_state["orientation"]
                        )

                    cmd = agent.compute_velocity(obj_state, other_positions, obstacles=obstacles)

                if args.kinematics == "diffdrive" and len(cmd) == 3:
                    pos, heading, _ = agent.robot.get_state()
                    v_forward = cmd[0] * np.cos(heading) + cmd[1] * np.sin(heading)
                    agent.robot.command_velocity(np.array([v_forward, cmd[2]]))
                else:
                    agent.robot.command_velocity(cmd)

                # Wheel velocity sanity: record *post-command* actual wheel velocities so it aligns
                # with what was just commanded. (The Phase7 controller records inside compute_velocity,
                # which happens before command_velocity.)
                if (
                    args.save_dir is not None
                    and agent.goal_type == "push"
                    and name in phase7_controllers
                    and hasattr(agent.robot, "get_wheel_velocities")
                    and len(phase7_controllers[name].history.wheel_velocities) > 0
                ):
                    try:
                        phase7_controllers[name].history.wheel_velocities[-1] = (
                            agent.robot.get_wheel_velocities().copy()
                        )
                    except Exception:
                        pass

        # Step physics every timestep
        pyb.stepSimulation()
        t += TIMESTEP
        step_count += 1

        if not args.no_gui:
            time.sleep(TIMESTEP * 0.3)

    # Finalize video recording (if any)
    if video_log_id is not None:
        print("Video recording will be finalized on PyBullet disconnect...")
        stop_video_recording(video_log_id, video_path)

    # Plot results if save_dir is provided and using Phase 7 controller
    if args.save_dir and len(phase7_controllers) > 0:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Collect histories and t_params
        histories = {name: controller.history for name, controller in phase7_controllers.items()}
        t_params_dict = {name: controller.t_param for name, controller in phase7_controllers.items()}

        plot_phase7_velocities(
            histories=histories,
            t_params=t_params_dict,
            desired_obj_velocity=np.array([0.0, 0.0]),
            desired_obj_omega=0.0,
            save_path=save_path / f"phase7_swarm_velocities_{args.desired_path}_w_{args.object}.png",
        )

        plot_phase_1_results(
            histories=histories,
            t_params=t_params_dict,
            contact_threshold=2.0,
            save_path=save_path / f"phase7_swarm_trajectories_{args.desired_path}_w_{args.object}.png",
        )

        plot_phase_7beta(
            histories=histories,
            t_params=t_params_dict,
            save_path=save_path / f"phase7_beta_trajectories_{args.desired_path}_w_{args.object}.png",
        )

        plot_phase7_wheel_plot(
            histories=histories,
            t_params=t_params_dict,
            save_path=save_path / f"phase7_wheel_velocities_{args.desired_path}_w_{args.object}.png",
        )

        export_histories(
            histories=histories,
            t_params=t_params_dict,
            save_path=save_path / f"histories_{args.desired_path}_w_{args.object}.json",
        )

        print(f"Saved plots and histories to {save_path}")

    pyb.disconnect()

    # Verify video file was created (after disconnect, PyBullet should have finalized it)
    if video_log_id is not None and video_path is not None:
        time.sleep(1.0)
        video_path = video_path.resolve()
        if video_path.exists():
            file_size = video_path.stat().st_size
            if file_size > 0:
                print(f"✓ Video saved successfully to {video_path} ({file_size / 1024 / 1024:.2f} MB)")
            else:
                print(f"✗ Warning: Video file exists but is empty (0 bytes) at {video_path}")
        else:
            print(f"✗ Warning: Video file not found at {video_path}")

    print("Done.")


if __name__ == "__main__":
    main()

