#!/usr/bin/env python3
"""Revised-owned holonomic sim helpers (no dependency on test_magnum_holonomic_control).

Constants, spawn pose, Phase7BetaVerDecouple, PyBullet setup / video / object state.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg

_rospack = rospkg.RosPack()
_pkg_path = Path(_rospack.get_path("contact_maintain"))

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)
# Run the high-level object velocity PID at a lower rate than Phase 7
PID_DECIMATION = 5  # Update ObjectVelocityPIDController every 5 Phase 7 control cycles

# Object height: taller for wheel robots to avoid multi-contact issues
DEFAULT_OBJECT_HEIGHT_WHEEL = 0.08   # For wheel robots (taller to avoid multi-contact)
DEFAULT_OBJECT_FRICTION = 0.3
ROBOT_RADIUS = 0.06  # Robot radius for position offset calculation
APPROACH_DISTANCE = ROBOT_RADIUS + 0.1  # Distance from contact point to spawn robot (for faster testing)


def robot_spawn_pose_world(
    contact_point_body: np.ndarray,
    normal_outward: np.ndarray,
    normal_inward: np.ndarray,
    object_xy: Tuple[float, float],
    object_yaw_rad: float,
    approach_distance: float = APPROACH_DISTANCE,
) -> Tuple[float, float, float]:
    """Body-frame Magnum contact offset → world spawn pose for the object at ``object_xy``."""
    spawn_body = np.asarray(contact_point_body, dtype=float) + float(approach_distance) * np.asarray(
        normal_outward, dtype=float
    )
    c = math.cos(float(object_yaw_rad))
    s = math.sin(float(object_yaw_rad))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    spawn_world = rot @ spawn_body + np.asarray(object_xy, dtype=float)
    n_in_w = rot @ np.asarray(normal_inward, dtype=float)
    heading = float(math.atan2(n_in_w[1], n_in_w[0]))
    return float(spawn_world[0]), float(spawn_world[1]), heading


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
    urdf_dir = Path(_pkg_path) / "urdf"
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
