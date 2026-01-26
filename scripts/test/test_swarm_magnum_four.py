#!/usr/bin/env python3
"""
Swarm + Magnum Four Test

1) Create an object (GenericObject) in PyBullet
2) Use legacy Magnum Four solver to find 4 best boundary contact points (t_params)
3) Spawn 4 robots and command them to NAVIGATE to those t_params (REACHING phase)
4) Uses WRENCH controller for contact maintenance (pushing phase)

Note: This test specifically uses the wrench controller for contact maintenance.

Usage:
  python test_swarm_magnum_four.py --object rectangle --duration 20
  python test_swarm_magnum_four.py --no-gui --object rectangle --duration 10
  python test_swarm_magnum_four.py --model wheel --object rectangle
"""

import argparse
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

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))


from object_utils import create_standard_objects, GraspMatrixCalculator, ContactPointParameterization, ContactPoint
from contact_maintain.robot_factory import create_robot
from contact_maintain.object_bridge import generic_to_pybullet
from contact_maintain.robot_agent import RobotAgent
from contact_maintain.swarm import SwarmHost
from contact_maintain.pyb_simulation import get_contact_force

# Magnum Four solver (legacy)
from contact_optimizer_utils import find_the_magnum_four_v3
# Wrench force distributor (legacy)
from wrench_solver import ForceDistributorPro


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

# Object height: taller for wheel robots to avoid multi-contact issues
DEFAULT_OBJECT_HEIGHT_DUMMY = 0.05   # For dummy robots (shorter, scaled for small robot)
DEFAULT_OBJECT_HEIGHT_WHEEL = 0.08   # For wheel robots (taller to avoid multi-contact)
DEFAULT_OBJECT_FRICTION = 0.3
ROBOT_RADIUS = 0.06  # Robot radius for position offset calculation
APPROACH_DISTANCE = 0.2  # Distance from contact point to spawn robot (for faster testing)


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


class Phase7ContactPointSpeedController:
    """Phase 7 style controller for Multi-Robot Swarm (MRS): Vector-based contact point tracking.
    
    For MRS, this controller tracks the full desired contact point velocity vector (not just speed).
    The desired contact point velocity is computed from:
        v_cp_desired = v_obj_desired + omega_desired × r_cp
    
    Control Law:
        vel_cmd_xy = v_base * normal_inward + v_cp_desired
    
    Where:
        - v_base: Position feedback along normal (ensures contact maintenance)
        - v_cp_desired: Full 2D desired contact point velocity vector (provides motion direction)
    
    This differs from single-pusher Phase 7 which tracks scalar speed with PI control.
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
        
        # Feed-forward gains (plant compensation)
        self.K_static = 0.03  # Static friction compensation (m/s)
        self.K_alpha = 0.6    # Viscous friction coefficient
        
        # PI controller gains
        self.kp_vel = 0.9     # Proportional gain
        self.ki_vel = 0.2     # Integral gain
        self.velocity_error_int = 0.0  # Integral accumulator
        self.velocity_error_int_max = 0.7  # Clamp to prevent windup
        
        # Control gain for base velocity from position error
        self.kp_approach = 2.0  # Gain for position-based approach velocity
        self.max_linear_speed = 0.5
        
        # Contact detection with hysteresis
        self.contact_threshold_on = 2.0  # Enter contact when force > this
        self.contact_threshold_off = 0.2  # Exit contact when force < this
        self.contact_threshold = 0.5  # Base threshold
        self.in_contact_prev = False
        
        # Integral decay when not in contact
        self.integral_decay_rate = 0.95  # Decay per control step
        
        # Heading control
        self.kp_heading = 10.0
        
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
        
        For MRS (Multi-Robot Swarm), we track the full velocity vector, not just speed magnitude.
        
        IMPORTANT: This method computes the desired velocity in body frame first (where it's
        constant for a fixed t_param), then transforms to world frame. This ensures the
        magnitude remains constant as the object rotates.
        
        Parameters
        ----------
        object_pos : np.ndarray
            Object center position (x, y)
        object_orientation : float
            Object orientation (radians)
        
        Returns
        -------
        np.ndarray
            Desired contact point velocity vector (vx, vy) in world frame
        """
        # Build rotation matrix
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        R_T = R.T  # Transpose for inverse rotation (world to body)
        
        # Compute desired contact point velocity in BODY FRAME (constant for fixed t_param)
        # r_cp_body is fixed (contact_point_body relative to object center in body frame)
        r_cp_body = self.contact_point_body
        
        # Transform desired object velocity from world frame to body frame
        v_obj_desired_body = R_T @ self.desired_object_velocity
        
        # Rotational velocity component in body frame (constant)
        # omega × r_cp_body = omega * [-r_cp_body[1], r_cp_body[0]]
        v_rotation_body = self.desired_object_angular_velocity * np.array([-r_cp_body[1], r_cp_body[0]])
        
        # Desired contact point velocity in body frame (constant)
        v_cp_desired_body = v_obj_desired_body + v_rotation_body
        
        # Transform to world frame (direction rotates with object, but magnitude is constant)
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
    ) -> np.ndarray:
        """Compute velocity command using MRS Phase 7 control strategy.
        
        Control Law:
            vel_cmd_xy = v_base * normal_inward + v_cp_desired
        
        Where:
            - v_base: Position feedback along normal (scalar, ensures contact maintenance)
            - v_cp_desired: Full 2D desired contact point velocity vector (from desired object motion)
        
        This differs from single-pusher Phase 7 which uses PI control on speed error.
        For MRS, we directly use the desired contact point velocity vector.
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Robot position (x, y)
        robot_heading : float
            Robot heading (radians)
        object_pos : np.ndarray
            Object center position (x, y)
        object_orientation : float
            Object orientation (radians)
        object_velocity : np.ndarray
            Object linear velocity (vx, vy)
        object_angular_velocity : float
            Object angular velocity (rad/s)
        contact_force : float
            Contact force magnitude (N) - provided by RobotAgent
        in_contact : bool
            Contact state - provided by RobotAgent (with hysteresis applied)
        t : float
            Current time (for history recording)
        record_history : bool
            Whether to record history for plotting
        
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy, omega)
        """
        # Apply hysteresis-based contact detection using provided contact_force
        # The RobotAgent already provides in_contact with basic threshold, but we apply
        # additional hysteresis here for smoother control transitions
        if self.in_contact_prev:
            # Was in contact: use lower threshold to exit
            in_contact_hyst = contact_force > self.contact_threshold_off
        else:
            # Was not in contact: use higher threshold to enter
            in_contact_hyst = contact_force > self.contact_threshold_on
        
        self.in_contact_prev = in_contact_hyst
        # Use the hysteresis-based contact state for control
        in_contact = in_contact_hyst
        
        # Contact point & normals in world
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        contact_point_world = R @ self.contact_point_body + object_pos
        normal_outward_world = R @ self.normal_outward
        normal_inward_world = -normal_outward_world
        
        # Requirement 1: Intended position (contact point + robot radius * normal_outward)
        intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
        position_error = intended_pos - robot_pos
        
        # Requirement 2: Heading toward contact point
        desired_heading = np.arctan2(
            (contact_point_world - robot_pos)[1],
            (contact_point_world - robot_pos)[0]
        )
        heading_error = np.arctan2(
            np.sin(desired_heading - robot_heading),
            np.cos(desired_heading - robot_heading)
        )
        
        # Calculate desired contact point velocity vector from desired object motion
        # For MRS: track the full vector, not just speed magnitude
        v_cp_desired = self._compute_desired_contact_point_velocity(
            object_pos, object_orientation
        )
        
        # Contact point velocity calculation (actual, for history/plotting)
        r_cp = contact_point_world - object_pos
        v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
        contact_point_velocity = object_velocity + v_rotation
        contact_point_speed = np.linalg.norm(contact_point_velocity)
        desired_contact_point_speed = np.linalg.norm(v_cp_desired)  # For history/plotting
        
        # print(f" at t_param {self.t_param}, v_cp_desired: {v_cp_desired}, with norm is {np.linalg.norm(v_cp_desired)} or {desired_contact_point_speed} from desired velocity is {self.desired_object_velocity} and desired angular velocity is {self.desired_object_angular_velocity}")

        # Decompose position error along normal_inward for contact maintenance
        normal_along = normal_inward_world
        error_along = np.dot(position_error, normal_along)

        # normal_perp = np.array([-normal_along[1], normal_along[0]])
        # error_perp = np.dot(position_error, normal_perp)
        
        # BASE VELOCITY: Position feedback along normal (ensures approach/maintenance)
        # This is a scalar that pushes robot toward contact point
        v_base = self.kp_approach * position_error
        
        # MRS CONTROL LAW: v_base + v_cp_desired (full vector)
        # v_base ensures contact maintenance, v_cp_desired provides desired motion direction
        if in_contact:
            # In contact: combine base velocitywith desired contact point velocity
            vel_cmd_xy = v_base + v_cp_desired
        else:
            # Not in contact: use base velocity only (approach object)
            vel_cmd_xy = v_base*1.2 + v_cp_desired 
        
        # Clamp total speed
        speed = np.linalg.norm(vel_cmd_xy)
        if speed > self.max_linear_speed:
            vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)
        
        # Heading control
        omega = self.kp_heading * heading_error
        omega = np.clip(omega, -1.0, 1.0)
        
        # Record history if requested
        if record_history:
            robot_vel = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
            # Note: contact_force and in_contact are provided as parameters from RobotAgent
            
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
            self.history.v_base_history.append(v_base)
            # For MRS, v_ff and v_pi are not used (we use v_cp_desired vector instead)
            self.history.v_ff_history.append(0.0)  # Not used in MRS control law
            self.history.v_pi_history.append(0.0)  # Not used in MRS control law
            self.history.desired_contact_point_speeds.append(desired_contact_point_speed)
        
        return np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])


class Phase7AlphaController:
    """Phase 7 Alpha: Decoupled direction/magnitude controller to fix swiggling.
    
    Key improvements over original Phase7ContactPointSpeedController:
    1. Decouples "Direction" from "Magnitude" - prevents v_base from fighting velocity PI
    2. Velocity-sourced position control - position error biases velocity PI target instead of directly adding
    3. Filtered orientation - reduces swiggling from object rotation jitter
    4. Priority stack approach - feed-forward sets base rhythm, PI handles speed error, position fine-tunes
    
    Control Law:
        v_ideal = v_ff_static + (K_alpha * desired_contact_point_speed)
        bias_speed_error = speed_error + (kp_approach * error_along)
        v_pi = kp_vel * bias_speed_error + ki_vel * velocity_error_int
        v_along = v_ideal + v_pi
        vel_cmd_xy = v_along * direction_normalized + v_perp_correction
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
        
        # Feed-forward gains (plant compensation)
        self.K_static = 0.03  # Static friction compensation (m/s)
        self.K_alpha = 0.6    # Viscous friction coefficient
        
        # PI controller gains
        self.kp_vel = 0.9     # Proportional gain
        self.ki_vel = 0.2     # Integral gain
        self.velocity_error_int = 0.0  # Integral accumulator
        self.velocity_error_int_max = 0.7  # Clamp to prevent windup
        
        # Position-based velocity bias gain (velocity-sourced position control)
        self.kp_approach = 0.5  # Reduced from 2.0 - now biases velocity PI instead of direct addition
        self.max_linear_speed = 0.5
        
        # Contact detection with hysteresis
        self.contact_threshold_on = 2.0  # Enter contact when force > this
        self.contact_threshold_off = 0.2  # Exit contact when force < this
        self.contact_threshold = 0.5  # Base threshold
        self.in_contact_prev = False
        
        # Integral decay when not in contact
        self.integral_decay_rate = 0.95  # Decay per control step
        
        # Heading control
        self.kp_heading = 10.0
        
        # Orientation filtering to reduce swiggling
        self.orientation_filter_alpha = 0.7  # Low-pass filter for object orientation
        self.object_orientation_filtered = None  # Will be initialized on first call
        
        # Perpendicular correction gain (for fine-tuning direction)
        self.kp_perp = 0.3  # Small correction perpendicular to desired direction
        
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
        
        Same as original Phase7 - computes in body frame first, then transforms to world.
        """
        # Build rotation matrix
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
    ) -> np.ndarray:
        """Compute velocity command using Phase 7 Alpha control strategy.
        
        Key difference: Position error biases velocity PI target instead of directly adding to velocity.
        This prevents the two controllers from fighting each other.
        """
        # Apply hysteresis-based contact detection
        if self.in_contact_prev:
            in_contact_hyst = contact_force > self.contact_threshold_off
        else:
            in_contact_hyst = contact_force > self.contact_threshold_on
        
        self.in_contact_prev = in_contact_hyst
        in_contact = in_contact_hyst
        
        # Filter object orientation to reduce swiggling from jitter
        if self.object_orientation_filtered is None:
            self.object_orientation_filtered = object_orientation
        else:
            # Low-pass filter: smooth orientation changes
            self.object_orientation_filtered = (
                self.orientation_filter_alpha * object_orientation +
                (1 - self.orientation_filter_alpha) * self.object_orientation_filtered
            )
        
        # Use filtered orientation for contact point calculations
        filtered_orientation = self.object_orientation_filtered
        
        # Contact point & normals in world (using filtered orientation)
        cos_t = np.cos(filtered_orientation)
        sin_t = np.sin(filtered_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        contact_point_world = R @ self.contact_point_body + object_pos
        normal_outward_world = R @ self.normal_outward
        normal_inward_world = -normal_outward_world
        
        # Intended position (contact point + robot radius * normal_outward)
        intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
        position_error = intended_pos - robot_pos
        
        # Heading toward contact point
        desired_heading = np.arctan2(
            (contact_point_world - robot_pos)[1],
            (contact_point_world - robot_pos)[0]
        )
        heading_error = np.arctan2(
            np.sin(desired_heading - robot_heading),
            np.cos(desired_heading - robot_heading)
        )
        
        # Calculate desired contact point velocity vector
        v_cp_desired = self._compute_desired_contact_point_velocity(
            object_pos, filtered_orientation
        )
        desired_contact_point_speed = np.linalg.norm(v_cp_desired)
        
        # Actual contact point velocity (for error calculation)
        r_cp = contact_point_world - object_pos
        v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
        contact_point_velocity = object_velocity + v_rotation
        contact_point_speed = np.linalg.norm(contact_point_velocity)
        
        # Decompose position error along normal_inward
        normal_along = normal_inward_world
        error_along = np.dot(position_error, normal_along)
        
        # ===== PHASE 7 ALPHA CONTROL LAW =====
        # 1. Calculate ideal speed required by physics (feed-forward)
        v_ideal = self.K_static + (self.K_alpha * desired_contact_point_speed)
        
        # 2. Calculate speed error (actual vs desired)
        speed_error = contact_point_speed - desired_contact_point_speed
        
        # 3. Use position error to bias the velocity PI target (velocity-sourced position control)
        # This prevents v_base from 'overpowering' the velocity loop
        bias_speed_error = speed_error + (self.kp_approach * error_along)
        
        # 4. Update integral term (with decay when not in contact)
        if in_contact:
            self.velocity_error_int += bias_speed_error * self.dt_ctrl
            self.velocity_error_int = np.clip(
                self.velocity_error_int,
                -self.velocity_error_int_max,
                self.velocity_error_int_max
            )
        else:
            # Decay integral when not in contact
            self.velocity_error_int *= self.integral_decay_rate
        
        # 5. PI controller output
        v_pi = self.kp_vel * bias_speed_error + self.ki_vel * self.velocity_error_int
        
        # 6. Total velocity magnitude along desired direction
        v_along = v_ideal + v_pi
        v_along = np.clip(v_along, -self.max_linear_speed, self.max_linear_speed)
        
        # 7. Direction: normalize desired contact point velocity vector
        if desired_contact_point_speed > 1e-6:
            direction = v_cp_desired / desired_contact_point_speed
        else:
            # If no desired motion, use normal_inward direction
            direction = normal_inward_world
        
        # 8. Perpendicular correction for fine-tuning (small correction perpendicular to direction)
        normal_perp = np.array([-direction[1], direction[0]])
        error_perp = np.dot(position_error, normal_perp)
        v_perp_correction = self.kp_perp * error_perp * normal_perp
        
        # 9. Final velocity command
        if in_contact:
            vel_cmd_xy = v_along * direction + v_perp_correction
        else:
            # Not in contact: approach with higher gain
            vel_cmd_xy = v_along * direction + v_perp_correction * 1.5
        
        # Clamp total speed
        speed = np.linalg.norm(vel_cmd_xy)
        if speed > self.max_linear_speed:
            vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)
        
        # Heading control
        omega = self.kp_heading * heading_error
        omega = np.clip(omega, -1.0, 1.0)
        
        # Record history if requested
        if record_history:
            robot_vel = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
            
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
            self.history.v_base_history.append(v_along)  # Store v_along as v_base for plotting
            self.history.v_ff_history.append(v_ideal)  # Feed-forward component
            self.history.v_pi_history.append(v_pi)  # PI component
            self.history.desired_contact_point_speeds.append(desired_contact_point_speed)
        
        return np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])


class Phase7BetaController:
    """Phase 7 Beta: Hierarchical dual-controller approach for two-objective problem.
    
    This controller uses a hierarchical structure to separate contact maintenance (geometry)
    from velocity tracking (kinematics). Several approaches are possible:
    
    Approach 1: Outer-Loop Position, Inner-Loop Velocity
        - Outer loop: Position controller generates desired velocity setpoint
        - Inner loop: Velocity controller tracks the setpoint
        - Problem: Still can have coupling issues
    
    Approach 2: Priority-Based Control
        - High priority: Contact maintenance (safety constraint)
        - Low priority: Velocity tracking (performance objective)
        - Use null-space projection or weighted combination
    
    Approach 3: State Machine / Mode Switching
        - Mode 1: Approach mode (position control dominant)
        - Mode 2: Contact mode (velocity control dominant)
        - Smooth transition between modes
    
    Approach 4: Constrained Optimization
        - Minimize velocity tracking error
        - Subject to: position error < threshold
        - Solve QP problem at each step
    
    Approach 5: Dual-Loop with Feed-Forward
        - Position loop: Generates correction velocity
        - Velocity loop: Tracks desired + correction
        - Feed-forward: Direct desired velocity injection
    
    Current Implementation: Approach 5 (Dual-Loop with Feed-Forward)
        - Position controller generates correction term
        - Velocity controller tracks desired velocity + correction
        - Feed-forward provides direct desired velocity
        - Weighted combination based on contact state
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
        
        # Position controller gains (outer loop / geometry objective)
        self.kp_pos = 2.0  # Position error gain
        self.kd_pos = 0.5  # Position error derivative gain (optional)
        self.position_error_prev = np.array([0.0, 0.0])
        
        # Velocity controller gains (inner loop / kinematics objective)
        self.kp_vel = 1.2  # Velocity error gain
        self.ki_vel = 0.3  # Velocity error integral gain
        self.velocity_error_int = 0.0
        self.velocity_error_int_max = 0.8
        
        # Feed-forward gains
        self.K_static = 0.03
        self.K_alpha = 0.6
        
        # Weighting factors for combining objectives
        self.w_position = 0.4  # Weight for position correction (geometry)
        self.w_velocity = 0.6   # Weight for velocity tracking (kinematics)
        # Alternative: Adaptive weighting based on contact state
        self.w_position_contact = 0.2  # Lower position weight when in contact
        self.w_velocity_contact = 0.8   # Higher velocity weight when in contact
        self.w_position_approach = 0.7  # Higher position weight when approaching
        self.w_velocity_approach = 0.3  # Lower velocity weight when approaching
        
        # Contact detection
        self.contact_threshold_on = 2.0
        self.contact_threshold_off = 0.2
        self.in_contact_prev = False
        
        # Heading control
        self.kp_heading = 10.0
        
        # Orientation filtering
        self.orientation_filter_alpha = 0.7
        self.object_orientation_filtered = None
        
        # Limits
        self.max_linear_speed = 0.5
        self.max_position_correction = 0.3  # Max correction from position controller
        
        # Control time step
        self.dt_ctrl = 1.0 / CTRL_FREQ
        
        # History for plotting
        self.history = Phase7History()
    
    def _compute_desired_contact_point_velocity(
        self,
        object_pos: np.ndarray,
        object_orientation: float,
    ) -> np.ndarray:
        """Compute desired contact point velocity vector from desired object motion."""
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
    ) -> np.ndarray:
        """Compute velocity command using Phase 7 Beta hierarchical control strategy.
        
        Dual-loop approach:
        1. Position controller (outer loop) generates correction velocity
        2. Velocity controller (inner loop) tracks desired + correction
        3. Feed-forward provides direct desired velocity
        4. Weighted combination based on contact state
        """
        # Apply hysteresis-based contact detection
        if self.in_contact_prev:
            in_contact_hyst = contact_force > self.contact_threshold_off
        else:
            in_contact_hyst = contact_force > self.contact_threshold_on
        
        self.in_contact_prev = in_contact_hyst
        in_contact = in_contact_hyst
        
        # Filter object orientation
        if self.object_orientation_filtered is None:
            self.object_orientation_filtered = object_orientation
        else:
            self.object_orientation_filtered = (
                self.orientation_filter_alpha * object_orientation +
                (1 - self.orientation_filter_alpha) * self.object_orientation_filtered
            )
        
        filtered_orientation = self.object_orientation_filtered
        
        # Contact point & normals in world
        cos_t = np.cos(filtered_orientation)
        sin_t = np.sin(filtered_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        contact_point_world = R @ self.contact_point_body + object_pos
        normal_outward_world = R @ self.normal_outward
        normal_inward_world = -normal_outward_world
        
        # Intended position
        intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
        position_error = intended_pos - robot_pos
        
        # Heading control
        desired_heading = np.arctan2(
            (contact_point_world - robot_pos)[1],
            (contact_point_world - robot_pos)[0]
        )
        heading_error = np.arctan2(
            np.sin(desired_heading - robot_heading),
            np.cos(desired_heading - robot_heading)
        )
        
        # Desired contact point velocity
        v_cp_desired = self._compute_desired_contact_point_velocity(
            object_pos, filtered_orientation
        )
        desired_contact_point_speed = np.linalg.norm(v_cp_desired)
        
        # Actual contact point velocity
        r_cp = contact_point_world - object_pos
        v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
        contact_point_velocity = object_velocity + v_rotation
        contact_point_speed = np.linalg.norm(contact_point_velocity)
        
        # ===== PHASE 7 BETA: DUAL-LOOP CONTROL =====
        
        # LOOP 1: Position Controller (Geometry Objective)
        # Generates correction velocity to maintain contact
        v_pos_correction = self.kp_pos * position_error
        # Optional: Add derivative term for damping
        # position_error_derivative = (position_error - self.position_error_prev) / self.dt_ctrl
        # v_pos_correction += self.kd_pos * position_error_derivative
        # self.position_error_prev = position_error.copy()
        
        # Limit position correction magnitude
        pos_corr_mag = np.linalg.norm(v_pos_correction)
        if pos_corr_mag > self.max_position_correction:
            v_pos_correction = v_pos_correction * (self.max_position_correction / pos_corr_mag)
        
        # LOOP 2: Velocity Controller (Kinematics Objective)
        # Tracks desired contact point velocity
        velocity_error = contact_point_velocity - v_cp_desired
        velocity_error_mag = np.linalg.norm(velocity_error)
        
        # Get direction of velocity error (for integral term)
        if velocity_error_mag > 1e-6:
            velocity_error_dir = velocity_error / velocity_error_mag
        else:
            velocity_error_dir = np.array([0.0, 0.0])
        
        # Update integral term (accumulate magnitude error)
        if in_contact:
            self.velocity_error_int += velocity_error_mag * self.dt_ctrl
            self.velocity_error_int = np.clip(
                self.velocity_error_int,
                -self.velocity_error_int_max,
                self.velocity_error_int_max
            )
        else:
            self.velocity_error_int *= 0.95  # Decay when not in contact
        
        # PI control on velocity error
        # Proportional: direct velocity error
        # Integral: accumulated magnitude error applied in error direction
        v_vel_correction = self.kp_vel * velocity_error + self.ki_vel * self.velocity_error_int * velocity_error_dir
        
        # FEED-FORWARD: Direct desired velocity injection
        v_ff = v_cp_desired  # Direct feed-forward of desired velocity
        
        # WEIGHTED COMBINATION: Adaptive weights based on contact state
        if in_contact:
            w_pos = self.w_position_contact
            w_vel = self.w_velocity_contact
        else:
            w_pos = self.w_position_approach
            w_vel = self.w_velocity_approach
        
        # Combine: feed-forward + weighted corrections
        vel_cmd_xy = v_ff + w_pos * v_pos_correction + w_vel * v_vel_correction
        
        # Clamp total speed
        speed = np.linalg.norm(vel_cmd_xy)
        if speed > self.max_linear_speed:
            vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)
        
        # Heading control
        omega = self.kp_heading * heading_error
        omega = np.clip(omega, -1.0, 1.0)
        
        # Record history
        if record_history:
            robot_vel = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
            
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
            self.history.v_base_history.append(np.linalg.norm(v_pos_correction))
            self.history.v_ff_history.append(np.linalg.norm(v_ff))
            self.history.v_pi_history.append(np.linalg.norm(v_vel_correction))
            self.history.desired_contact_point_speeds.append(desired_contact_point_speed)
        
        return np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])


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
        
        return np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])


class Phase7BetaVerDecoupleVerLocal:
    """Phase 7 Beta Version Decouple VerLocal: Tripartite Decoupled Structure with Local Frame Desired Velocity.
    
    This is the same as Phase7BetaVerDecouple, but with a key correction:
    - desired_object_velocity is interpreted as being in the OBJECT'S BODY FRAME, not world frame.
    
    This makes more sense for specifying desired motion relative to the object's orientation.
    For example, to push the object forward along its current heading, you would specify
    desired_object_velocity = [0.05, 0.0] (forward in body frame), regardless of the object's
    current orientation in the world.
    
    Control Structure:
    ------------------
    Same as Phase7BetaVerDecouple:
    1. Longitudinal Axis (v_x, contact frame): Controls pushing force (clamping) against inner normal
    2. Lateral Axis (v_y, contact frame): Governs lateral position along object's edge
    3. Angular (ω): Manages robot orientation so force sensor/bumper points along normal
    
    Key Difference:
    ---------------
    - desired_object_velocity is in OBJECT BODY FRAME (not world frame)
    - No transformation needed when computing desired contact point velocity in body frame
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
            Desired object linear velocity (vx, vy) in OBJECT BODY FRAME (not world frame)
        desired_object_angular_velocity : float
            Desired object angular velocity (rad/s)
        """
        self.robot_uid = robot_uid
        self.object_uid = object_uid
        self.generic_object = generic_object
        self.t_param = t_param
        # Store desired velocity as-is (already in body frame)
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
        
        KEY DIFFERENCE FROM Phase7BetaVerDecouple:
        - desired_object_velocity is already in BODY FRAME, so no transformation needed.
        - We use it directly in body frame calculations.
        
        Parameters
        ----------
        object_pos : np.ndarray
            Object center position (x, y) in world frame
        object_orientation : float
            Object orientation (radians) in world frame
        
        Returns
        -------
        np.ndarray
            Desired contact point velocity vector (vx, vy) in world frame
        """
        # Build rotation matrix (body to world)
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # desired_object_velocity is already in body frame, use it directly
        v_obj_desired_body = self.desired_object_velocity
        
        # Rotational velocity component in body frame (constant)
        r_cp_body = self.contact_point_body
        v_rotation_body = self.desired_object_angular_velocity * np.array([-r_cp_body[1], r_cp_body[0]])
        
        # Desired contact point velocity in body frame
        v_cp_desired_body = v_obj_desired_body + v_rotation_body
        
        # Transform to world frame
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
    ) -> np.ndarray:
        """Compute velocity command using Tripartite Decoupled Structure.
        
        Same control law as Phase7BetaVerDecouple, but desired_object_velocity is in body frame.
        
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
        # v_perp = v_perp_ff * 0.8 + v_perp_pos
        v_perp = v_perp_ff * 1 + v_perp_pos
        # v_perp = v_perp_pos
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
        
        return np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])


def setup_pybullet(gui: bool = True):
    
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)

    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
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
    
    fig.suptitle(
        f'Phase 7 Velocities: Multi-Robot Swarm\n'
        f'Desired object velocity: {desired_obj_velocity}, omega: {desired_obj_omega:.3f} rad/s',
        fontsize=14, fontweight='bold'
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
        ax.set_title(f'{name} - Contact State')
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
        
        # Plot 2: Position error
        ax = axes[idx, 1]
        error_mags = np.linalg.norm(position_errors, axis=1)
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
        
        # Plot 5: Robot speed
        ax = axes[idx, 4]
        robot_vels = np.array(history.robot_velocities)
        robot_speeds = np.linalg.norm(robot_vels[:, :2], axis=1)
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


def main():
    parser = argparse.ArgumentParser(description="Swarm Magnum Four navigation / contact test")
    parser.add_argument("--object", type=str, default="rectangle",
                        help="Object shape from create_standard_objects()")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Test duration in seconds")
    parser.add_argument("--no-gui", action="store_true", help="Run headless")
    parser.add_argument("--kinematics", "-k", default="holonomic",
                        choices=["holonomic", "diffdrive"])
    parser.add_argument("--model", "-m", default="dummy", choices=["dummy", "wheel"],
                        help="Robot model (default: dummy)")
    parser.add_argument(
        "--controller", "-c",
        default="velocity",
        choices=["velocity", "wrench"],
        help="Contact controller type: 'velocity' (InstantVelocityMatcher, drive_desired) "
             "or 'wrench' (WrenchTrackingController).",
    )
    parser.add_argument("--magnum-verbose", action="store_true", help="Verbose Magnum Four search logs")
    parser.add_argument("--magnum-visualize", action="store_true", help="Visualize Magnum Four search (matplotlib)")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save plots (only for Phase 7 velocity controller)")
    parser.add_argument("--record-video", action="store_true",
                       help="Record PyBullet simulation as video (top-down view). Requires --save-dir.")
    
    args = parser.parse_args()

    # ROS package path setup (same as other tests)
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path("contact_maintain")
    sys.path.insert(0, str(Path(pkg_path) / "src"))
    sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))


    setup_pybullet(gui=not args.no_gui)

    # Create object
    standard_objects = create_standard_objects()
    if args.object not in standard_objects:
        raise ValueError(f"Unknown object '{args.object}'. Available: {list(standard_objects.keys())}")
    generic_object = standard_objects[args.object]
    contact_point_parameterization = ContactPointParameterization(generic_object)

    # Use taller object for wheel robots to avoid multi-contact issues
    object_height = DEFAULT_OBJECT_HEIGHT_WHEEL if args.model == 'wheel' else DEFAULT_OBJECT_HEIGHT_DUMMY

    object_uid = generic_to_pybullet(
        generic_object,
        height=object_height,
        position=(0, 0, 0),
        color=(0.4, 0.7, 0.4, 1.0),
    )
    pyb.changeDynamics(object_uid, -1, lateralFriction=DEFAULT_OBJECT_FRICTION, mass=1.0)

    # Decide controller type
    pushing_type = args.controller  # 'velocity' or 'wrench'

    # Compute Magnum Four contacts / t_params
    contacts = None
    if args.object == "rectangle" :
        # Use hardcoded solution for rectangle to save solver time
        t_params = [0.4750, 0.6458, 0.9208, 0.0179]
        print("\nUsing hardcoded Magnum Four t_params for rectangle "
              "(velocity controller):", [f"{v:.4f}" for v in t_params])

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
        t_params = np.roll(t_params, -1)
        t_params = t_params.tolist()

    if len(t_params) != 4:
        raise RuntimeError(f"Expected 4 contacts from Magnum Four, got {len(t_params)}")

    print(f"\nMagnum Four t_params: {[f'{v:.4f}' for v in t_params]}")

    # OPTIONAL: Build grasp matrix and shared ForceDistributorPro
    # Only needed for wrench controller
    force_distributor = None
    if pushing_type == "wrench":
        if contacts is None:
            raise RuntimeError("Wrench controller requires full Magnum Four contact solution.")
        grasp_calculator = GraspMatrixCalculator()
        grasp_matrix = grasp_calculator.build_wrench_matrix(contacts)
        force_distributor = ForceDistributorPro(
            max_force=10.0,
            max_rate_increase=4.0,
            max_rate_decrease=6.0,
            contact_points=contacts,
            grasp_matrix=grasp_matrix,
            t_params=t_params,
        )

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

        # Use selected controller for contact maintenance
        agent = RobotAgent(
            robot=robot,
            name=name,
            object_uid=object_uid,
            generic_object=generic_object,
            navigation_type="apf",
            pushing_type=pushing_type,
            force_distributor=force_distributor,
        )
        robot_agents[name] = agent

    # Configure Phase 7 controllers for VELOCITY mode
    # OR configure desired wrench for WRENCH mode
    phase7_controllers = {}
    if pushing_type == "velocity":
        # Phase 7 controller: track contact point speed calculated from desired object motion
        desired_obj_velocity = np.array([0.03, 0.05])  # Desired object linear velocity (vx, vy)
        desired_obj_omega = 0.2  # Desired object angular velocity (rad/s)
        
        for name, agent in robot_agents.items():
            # Get the target t_param for this robot
            robot_idx = list(robot_agents.keys()).index(name)
            target_t_param = t_params[robot_idx]
            
            # Create Phase 7 controller for this robot
            # It will calculate desired contact point speed from desired object motion
            phase7_controllers[name] = Phase7BetaVerDecoupleVerLocal(
                robot_uid=robots[name].uid,
                object_uid=object_uid,
                generic_object=generic_object,
                t_param=target_t_param,
                desired_object_velocity=desired_obj_velocity,
                desired_object_angular_velocity=desired_obj_omega,
            )
            print(f"Created Phase 7 controller for {name} with t_param={target_t_param:.4f}, "
                  f"desired_obj_velocity={desired_obj_velocity}, "
                  f"desired_obj_omega={desired_obj_omega:.3f} rad/s")
    else:
        desired_wrench = np.array([5, 0.0, 0.0])
        for agent in robot_agents.values():
            agent.set_desired_wrench(desired_wrench)
    host = SwarmHost(
        robot_agents=robot_agents,
        object_uid=object_uid,
        generic_object=generic_object,
    )

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
        
        # Always save as phase7_topview.mp4 in save_dir
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_path = save_dir / "phase7_topview.mp4"
        
        

        video_log_id = setup_video_recording(video_path, object_uid)


    # Run sim loop
    n_steps = int(args.duration / TIMESTEP)
    step_count = 0
    t = 0.0
    for _ in range(n_steps):
        obj_state = get_object_state(object_uid)

        if step_count % CTRL_STEP == 0:
            host.update(1.0 / CTRL_FREQ, obj_state)

            for name, agent in robot_agents.items():
                other_positions = [
                    robot_agents[other_name].robot.get_state()[0]
                    for other_name in robot_agents.keys()
                    if other_name != name
                ]

                # Use Phase 7 controller if in pushing mode and velocity controller type
                if pushing_type == "velocity" and agent.goal_type == "push" and name in phase7_controllers:
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
                    )
                else:
                    # Use normal agent controller (navigation, approach, or wrench pushing)
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


        pyb.stepSimulation()
        t += TIMESTEP
        step_count += 1
        
        if not args.no_gui:
            time.sleep(TIMESTEP * 0.3)

    # Note: PyBullet video recording is finalized automatically when we disconnect
    # We don't need to stop it manually - PyBullet will handle it
    if video_log_id is not None:
        print(f"Video recording will be finalized on PyBullet disconnect...")
        stop_video_recording(video_log_id, video_path)

    # Plot results if save_dir is provided and using Phase 7 controller
    if args.save_dir and pushing_type == "velocity" and len(phase7_controllers) > 0:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Collect histories and t_params
        histories = {name: controller.history for name, controller in phase7_controllers.items()}
        t_params_dict = {name: controller.t_param for name, controller in phase7_controllers.items()}
        
        # Plot Phase 7 velocities
        plot_phase7_velocities(
            histories=histories,
            t_params=t_params_dict,
            desired_obj_velocity=desired_obj_velocity,
            desired_obj_omega=desired_obj_omega,
            save_path=save_path / "phase7_swarm_velocities.png",
        )
        
        # Plot Phase 1 style results (trajectories, errors, etc.)
        plot_phase_1_results(
            histories=histories,
            t_params=t_params_dict,
            contact_threshold=2.0,  # Use same threshold as controller
            save_path=save_path / "phase7_swarm_trajectories.png",
        )
        
        print(f"Saved plots to {save_path}")

    pyb.disconnect()
    # Verify video file was created (after disconnect, PyBullet should have finalized it)
    if video_log_id is not None and video_path is not None:
        time.sleep(1.0)  # Give PyBullet time to finalize file after disconnect
        
        video_path = video_path.resolve()
        if video_path.exists():
            file_size = video_path.stat().st_size
            if file_size > 0:
                print(f"✓ Video saved successfully to {video_path} ({file_size / 1024 / 1024:.2f} MB)")
            else:
                print(f"✗ Warning: Video file exists but is empty (0 bytes) at {video_path}")
        else:
            print(f"✗ Warning: Video file not found at {video_path}")
            print(f"  Expected path: {video_path.absolute()}")
            print(f"  Parent directory exists: {video_path.parent.exists()}")
            # List files in parent directory to see if it was created with different name
            if video_path.parent.exists():
                print(f"  Files in directory: {list(video_path.parent.glob('*.mp4'))}")
            print(f"  This might indicate an issue with PyBullet video recording.")
            print(f"  Make sure GUI is enabled (not --no-gui) and path is writable.")
    
    print("Done.")


if __name__ == "__main__":
    main()

