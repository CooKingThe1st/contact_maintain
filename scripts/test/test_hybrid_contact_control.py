#!/usr/bin/env python3
"""
Hybrid Contact Control Test

Tests a new contact maintenance controller that addresses accumulated errors
from open-loop velocity control when facing disturbances.

Key Insight:
- Robot has closed-loop control, but small instant errors from obstacles
  accumulate over time, causing position deviation
- Need hybrid control that combines position/force/velocity requirements

Controller Requirements:
1. Position requirement: Robot should be at contact_point + robot_radius * normal_outward
2. Heading requirement: Robot heading should point toward contact point
   (adapts to closest point on boundary if robot slides)
3. Force/Velocity requirement: Additional control based on desired force or velocity

Phases:
- Phase 1: Calculate requirements 1 & 2, plus closest point on boundary
- Phase 2: Force-position hybrid control (spring model)
- Phase 2.5: Force-position hybrid control with PI feedback
- Phase 3: Velocity-position hybrid control
- Phase 4: Force-position hybrid control using Newton's law (F = m*a) - OPEN-LOOP
- Phase 5: Closed-loop object velocity control using feedback (single robot) - CLOSED-LOOP

Usage:
    # Phase 1: Basic requirements calculation
    python test_hybrid_contact_control.py --phase 1 --t-param 0.125
    
    # Phase 2: Force-position hybrid control
    python test_hybrid_contact_control.py --phase 2 --t-param 0.125 --desired-force 5.0
    
    # Phase 3: Velocity-position hybrid control
    python test_hybrid_contact_control.py --phase 3 --t-param 0.125
"""
import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Use non-interactive backend for headless mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import create_standard_objects, create_pybullet_objects, ContactPointParameterization
from contact_maintain.robot_factory import create_robot, is_wheel_robot, get_wheel_velocities
from contact_maintain.object_bridge import generic_to_pybullet, obj_to_generic
from contact_maintain.pyb_simulation import get_contact_force


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

DEFAULT_OBJECT_SHAPE = 'rectangle'
DEFAULT_OBJECT_HEIGHT = 0.2
DEFAULT_OBJECT_FRICTION = 0.8

ROBOT_RADIUS = 0.06  # Robot radius for position offset calculation


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class HybridControlHistory:
    """History for hybrid contact control test."""
    times: List[float] = field(default_factory=list)
    robot_positions: List[np.ndarray] = field(default_factory=list)
    robot_headings: List[float] = field(default_factory=list)
    robot_velocities: List[np.ndarray] = field(default_factory=list)
    robot_cmd_velocities: List[np.ndarray] = field(default_factory=list)
    
    # Requirement 1: Intended position
    intended_positions: List[np.ndarray] = field(default_factory=list)
    position_errors: List[np.ndarray] = field(default_factory=list)
    
    # Requirement 2: Heading
    desired_headings: List[float] = field(default_factory=list)
    heading_errors: List[float] = field(default_factory=list)
    
    # Closest point on boundary
    closest_points_on_desired_segment: List[np.ndarray] = field(default_factory=list)
    closest_u_on_desired_segment: List[float] = field(default_factory=list)
    
    # Object state
    object_positions: List[np.ndarray] = field(default_factory=list)
    object_orientations: List[float] = field(default_factory=list)
    object_velocities: List[np.ndarray] = field(default_factory=list)
    object_angular_velocities: List[float] = field(default_factory=list)
    
    # Contact point (from desired t_param)
    contact_point_positions: List[np.ndarray] = field(default_factory=list)
    contact_point_velocities: List[np.ndarray] = field(default_factory=list)
    
    # Contact forces
    contact_forces: List[float] = field(default_factory=list)
    in_contact: List[bool] = field(default_factory=list)
    
    # Phase 5 specific: velocity components
    v_base_history: List[float] = field(default_factory=list)
    v_constant_history: List[float] = field(default_factory=list)
    v_velo_error_pi_history: List[float] = field(default_factory=list)


# ============================================================================
# SIMULATION SETUP
# ============================================================================

def setup_pybullet(gui: bool = True):
    """Initialize PyBullet."""
    if gui:
        client_id = pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    
    # Set search paths BEFORE loading any URDF files (PyBullet requirement)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])

    # Add URDF directory to search path for custom URDF files
    urdf_dir = Path(pkg_path) / "urdf"
    if urdf_dir.exists():
        pyb.setAdditionalSearchPath(str(urdf_dir))
    
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    
    return ground


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_closest_t_param(robot_pos: np.ndarray, generic_object, parameterization: ContactPointParameterization,
                         object_pos: np.ndarray, object_orientation: float) -> Tuple[float, np.ndarray]:
    """Deprecated: previously used a full-boundary search for closest t_param.

    We intentionally avoid this in Phase 1 of the hybrid controller because we want
    the controller to stay on the *desired edge*. Use `closest_point_on_desired_segment()`.
    """
    raise RuntimeError(
        "find_closest_t_param() is deprecated in this test. "
        "Use closest_point_on_desired_segment() with the desired t_param's segment."
    )


def closest_point_on_desired_segment(
    robot_pos: np.ndarray,
    object_pos: np.ndarray,
    object_orientation: float,
    seg_p1_body: np.ndarray,
    seg_p2_body: np.ndarray,
    *,
    assert_on_segment: bool = True,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Compute closest point from robot to the *desired t_param's segment*.

    This intentionally does NOT search the full boundary. It projects the robot position
    onto the segment that contains the desired t_param (in the object's body frame).

    Returns the closest point in WORLD frame and the unclamped segment parameter u.

    If assert_on_segment is True, we assert that the projection lies within the segment
    (u in [0,1]) up to tolerance. If it does not, the controller has likely drifted
    away from the intended edge.
    """
    # Transform robot position to object body frame
    cos_t = np.cos(-object_orientation)
    sin_t = np.sin(-object_orientation)
    R_inv = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    robot_local = R_inv @ (robot_pos - object_pos)

    # Project onto segment in body frame
    seg_vec = seg_p2_body - seg_p1_body
    seg_len2 = float(seg_vec @ seg_vec)
    if seg_len2 < 1e-12:
        # Degenerate segment: treat as a point
        u_unclamped = 0.0
        closest_body = seg_p1_body.copy()
    else:
        u_unclamped = float(((robot_local - seg_p1_body) @ seg_vec) / seg_len2)
        u_clamped = float(np.clip(u_unclamped, 0.0, 1.0))
        closest_body = seg_p1_body + u_clamped * seg_vec

    if assert_on_segment:
        if not (-tol <= u_unclamped <= 1.0 + tol):
            raise AssertionError(
                f"Robot projection left desired segment: u={u_unclamped:.4f} "
                f"(expected in [0,1]). This indicates controller drift from the desired edge."
            )

    # Transform closest point back to world frame
    cos_t = np.cos(object_orientation)
    sin_t = np.sin(object_orientation)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    closest_world = R @ closest_body + object_pos

    return closest_world, u_unclamped


# ============================================================================
# HYBRID CONTACT CONTROL TEST
# ============================================================================

class HybridContactControlTest:
    """Test hybrid contact control with position/force/velocity requirements."""
    
    def __init__(self, kinematics: str, t_param: float, approach_distance: float = 0.5, 
                 object_name: Optional[str] = None, object_index: Optional[int] = None,
                 obj_shape: Optional[str] = None, obj_file: Optional[str] = None):
        """
        Parameters
        ----------
        kinematics : str
            'holonomic' or 'diffdrive'
        t_param : float
            Desired t_param on object boundary to track
        approach_distance : float
            Initial distance from object to spawn robot
        object_name : str, optional
            Object name from create_standard_objects() (e.g., 'rectangle')
        object_index : int, optional
            Object index from create_pybullet_objects() (1-based, e.g., 3 for right_triangle, 6 for l_shape)
        obj_shape : str, optional
            Shape name for OBJ mode (must exist in shape_data.json, e.g., 'right_triangle')
        obj_file : str, optional
            OBJ file path (relative to urdf directory or absolute). If None, uses '{obj_shape}.obj'
        """
        self.kinematics = kinematics
        self.desired_t_param = t_param
        self.approach_distance = approach_distance
        
        # Create object - support OBJ mode, object_index, object_name, or default
        print(f"\nCreating object with desired t_param={t_param:.3f}...")
        
        if obj_shape is not None:
            # OBJ mode: Load from OBJ file and create GenericObject from shape data
            if obj_file is None:
                obj_file = f"{obj_shape}.obj"
            
            print(f"Loading object from OBJ: {obj_file} (shape: {obj_shape})")
            self.generic_object, self.object_uid = obj_to_generic(
                obj_path=obj_file,
                shape_name=obj_shape,
                position=(0.0, 0.0, 1),
                orientation=0.0,
                mass=1.0,
                lateral_friction=DEFAULT_OBJECT_FRICTION
            )
            print(f"✓ Loaded OBJ object: {obj_shape}")
            print(f"  Mass: {self.generic_object.mass:.3f} kg")
            print(f"  Moment of inertia: {self.generic_object.moment_of_inertia:.6f} kg·m²")
            print(f"  Lateral friction: {self.generic_object.lateral_friction:.3f}")
            
            # Set dynamics (ensure friction matches)
            pyb.changeDynamics(self.object_uid, -1, 
                              lateralFriction=DEFAULT_OBJECT_FRICTION)
        elif object_index is not None:
            # Use create_pybullet_objects() and select by index (1-based)
            pybullet_objects = create_pybullet_objects()
            object_names = list(pybullet_objects.keys())
            
            if object_index < 1 or object_index > len(object_names):
                raise ValueError(
                    f"Invalid object index {object_index}. "
                    f"Valid range: 1-{len(object_names)}\n"
                    f"Available objects:\n" +
                    "\n".join([f"  {i+1}. {name}" for i, name in enumerate(object_names)])
                )
            
            selected_name = object_names[object_index - 1]  # Convert to 0-based
            self.generic_object = pybullet_objects[selected_name]
            print(f"Selected object by index {object_index}: '{selected_name}'")
            
            # Create object in PyBullet at origin
            self.object_uid = generic_to_pybullet(
                self.generic_object,
                height=DEFAULT_OBJECT_HEIGHT,
                position=(0.0, 0.0, 0),
                orientation=0.0,
                color=(0.4, 0.7, 0.4, 1.0)
            )
            pyb.changeDynamics(self.object_uid, -1, 
                              lateralFriction=DEFAULT_OBJECT_FRICTION, 
                              mass=1.0)
        elif object_name is not None:
            # Use create_standard_objects() and select by name
            standard_objects = create_standard_objects()
            if object_name not in standard_objects:
                raise ValueError(f"Unknown object '{object_name}'. Available: {list(standard_objects.keys())}")
            self.generic_object = standard_objects[object_name]
            print(f"Selected object by name: '{object_name}'")
            
            # Create object in PyBullet at origin
            self.object_uid = generic_to_pybullet(
                self.generic_object,
                height=DEFAULT_OBJECT_HEIGHT,
                position=(0.0, 0.0, 0),
                orientation=0.0,
                color=(0.4, 0.7, 0.4, 1.0)
            )
            pyb.changeDynamics(self.object_uid, -1, 
                              lateralFriction=DEFAULT_OBJECT_FRICTION, 
                              mass=1.0)
        else:
            # Default: use rectangle from create_standard_objects()
            standard_objects = create_standard_objects()
            self.generic_object = standard_objects[DEFAULT_OBJECT_SHAPE]
            print(f"Using default object: '{DEFAULT_OBJECT_SHAPE}'")
            
            # Create object in PyBullet at origin
            self.object_uid = generic_to_pybullet(
                self.generic_object,
                height=DEFAULT_OBJECT_HEIGHT,
                position=(0.0, 0.0, 0),
                orientation=0.0,
                color=(0.4, 0.7, 0.4, 1.0)
            )
            pyb.changeDynamics(self.object_uid, -1, 
                              lateralFriction=DEFAULT_OBJECT_FRICTION, 
                              mass=1.0)
        
        self.parameterization = ContactPointParameterization(self.generic_object)
        
        # Get contact point info at desired t_param
        contact_info = self.parameterization.get_contact_info(t_param)
        self.contact_point_body = contact_info['point']
        self.normal_outward = contact_info['normal_outward']
        self.normal_inward = -self.normal_outward
        self.tangent = contact_info['tangent']  # Unit tangent vector along boundary (for Phase 7 Beta)

        # Precompute the segment (two vertices) that contains desired t_param (in body frame)
        _, seg_idx, _ = self.parameterization.parameter_to_point(t_param)
        self.desired_seg_idx = int(seg_idx)
        self.desired_seg_p1_body = np.array(self.parameterization.boundary_coords[self.desired_seg_idx], dtype=float)
        self.desired_seg_p2_body = np.array(self.parameterization.boundary_coords[self.desired_seg_idx + 1], dtype=float)
        
        # Calculate robot spawn position
        spawn_position_body = self.contact_point_body + self.approach_distance * self.normal_outward
        
        
        # Create robot at spawn position
        print(f"Creating {kinematics} wheel robot at spawn position...")
        print(f"Spawn position body: {spawn_position_body}")
        
        self.robot = create_robot(
            kinematics=kinematics,
            model='wheel',  # Always use wheel model
            position=(spawn_position_body[0], spawn_position_body[1], 0),
            orientation=np.arctan2(self.normal_inward[1], self.normal_inward[0]),
            name="hybrid_control_robot"
        )
        
        # History
        self.history = HybridControlHistory()
        
        # Contact detection
        self.in_contact = False
        self.contact_threshold = 0.5  # N

        # Force control params (for Phase 2)
        self.spring_stiffness = 100.0  # N/m (tunable)
        self.kp_heading = 10.0  # Heading P gain
        self.max_linear_speed = 0.5

        # Force PI gains for Phase 2_5 (force-position hybrid with PI)
        self.kp_force = 0.02   # m/N (how much to move per unit force error)
        self.ki_force = 0.005  # m/(N*s)
        self.force_error_int = 0.0

        # Phase 3: desired object motion snapshot (captured at first contact)
        self.phase3_desired_object_velocity = None       # np.ndarray[2]
        self.phase3_desired_object_angular_velocity = 0.0
    
    def compute_requirement_1(self, object_pos: np.ndarray, object_orientation: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Requirement 1: Intended robot position.
        
        Intended position = contact_point + robot_radius * normal_outward
        
        Parameters
        ----------
        object_pos : np.ndarray
            Object center position
        object_orientation : float
            Object orientation
        
        Returns
        -------
        intended_position : np.ndarray
            Intended robot position in world frame
        contact_point_world : np.ndarray
            Contact point position in world frame
        """
        # Get contact point in world frame
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        contact_point_world = R @ self.contact_point_body + object_pos
        
        # Get normal outward in world frame
        normal_outward_world = R @ self.normal_outward
        
        # Intended position = contact_point + robot_radius * normal_outward
        intended_position = contact_point_world + ROBOT_RADIUS * normal_outward_world
        
        return intended_position, contact_point_world
    
    def compute_requirement_2(self, robot_pos: np.ndarray, closest_point: np.ndarray) -> Tuple[float, float]:
        """Compute Requirement 2: Desired heading.
        
        Robot heading should point toward contact point (or closest point if robot slides).
        
        Parameters
        ----------
        robot_pos : np.ndarray
            Robot position
        closest_point : np.ndarray
            Closest point on boundary (or desired contact point)
        
        Returns
        -------
        desired_heading : float
            Desired heading angle (radians)
        heading_error : float
            Heading error (radians)
        """
        # Direction from robot to contact point
        to_contact = closest_point - robot_pos
        desired_heading = np.arctan2(to_contact[1], to_contact[0])
        
        return desired_heading, 0.0  # heading_error computed later with current heading
    
    def run_phase_1(self, gui: bool = True, duration: float = 10.0) -> Dict:
        """Phase 1: Calculate requirements 1 & 2, plus closest point on boundary.
        
        This phase:
        - Calculates intended position (requirement 1)
        - Calculates desired heading (requirement 2)
        - Finds closest point on boundary to robot
        - Prints all values for analysis
        - Uses simple approach controller to get to contact
        """
        print(f"\n{'='*60}")
        print(f"PHASE 1: Basic Requirements Calculation")
        print(f"{'='*60}")
        print(f"  Desired t_param: {self.desired_t_param:.3f}")
        print(f"  Duration: {duration:.1f} s")
        
        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        
        # Approach phase parameters
        approach_speed = 0.15  # m/s
        approach_kp = 1.0
        
        for step in range(n_steps):
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Get robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()
                
                # Get object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]
                
                # Check contact
                contact_force = self._get_contact_force()
                self.in_contact = contact_force > self.contact_threshold
                
                # REQUIREMENT 1: Calculate intended position
                intended_pos, contact_point_world = self.compute_requirement_1(object_pos, object_orientation)
                position_error = intended_pos - robot_pos
                
                # Closest point on the DESIRED segment only (controller correctness check)
                closest_point, closest_u = closest_point_on_desired_segment(
                    robot_pos=robot_pos,
                    object_pos=object_pos,
                    object_orientation=object_orientation,
                    seg_p1_body=self.desired_seg_p1_body,
                    seg_p2_body=self.desired_seg_p2_body,
                    assert_on_segment=True,
                    tol=1e-4,
                )
                
                # REQUIREMENT 2: Calculate desired heading
                # Use closest point if robot has slid, otherwise use desired contact point
                reference_point = closest_point if self.in_contact else contact_point_world
                desired_heading, _ = self.compute_requirement_2(robot_pos, reference_point)
                heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                          np.cos(desired_heading - robot_heading))
                
                # Calculate contact point velocity (for reference)
                r = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r[1], r[0]])
                contact_point_velocity = object_velocity + v_rotation
                
                # Simple approach controller
                if not self.in_contact:
                    # Drive toward intended position
                    to_intended = intended_pos - robot_pos
                    distance = np.linalg.norm(to_intended)
                    
                    if distance > 0.01:
                        direction = to_intended / distance
                        speed = min(approach_kp * distance, approach_speed)
                        vel_2d = direction * speed
                        
                        # Heading control
                        omega = 10.0 * heading_error
                        omega = np.clip(omega, -1.0, 1.0)
                        
                        cmd = np.array([vel_2d[0], vel_2d[1], omega])
                    else:
                        cmd = np.zeros(3)
                else:
                    # In contact - just maintain heading for now
                    omega = 2.0 * heading_error
                    omega = np.clip(omega, -1.0, 1.0)
                    cmd = np.array([0.0, 0.0, omega])
                
                # Command velocity
                self.robot.command_velocity(cmd)
                
                # Record history
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(desired_heading)
                self.history.heading_errors.append(heading_error)
                self.history.closest_points_on_desired_segment.append(closest_point.copy())
                self.history.closest_u_on_desired_segment.append(float(closest_u))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)
                
                # Print analysis (every 10 control steps)
                if step_count % (CTRL_STEP * 10) == 0:
                    print(f"\n[t={t:.2f}s] Phase 1 Analysis:")
                    print(f"  Robot pos: {robot_pos}, heading: {robot_heading:.3f} rad")
                    print(f"  REQUIREMENT 1:")
                    print(f"    Intended pos: {intended_pos}")
                    print(f"    Position error: {position_error} (mag: {np.linalg.norm(position_error)*100:.2f} cm)")
                    print(f"  REQUIREMENT 2:")
                    print(f"    Desired heading: {desired_heading:.3f} rad ({np.degrees(desired_heading):.1f} deg)")
                    print(f"    Heading error: {heading_error:.3f} rad ({np.degrees(heading_error):.1f} deg)")
                    print(f"  Closest point on DESIRED segment (seg_idx={self.desired_seg_idx}):")
                    print(f"    Closest point: {closest_point}")
                    print(f"    Segment u (unclamped): {closest_u:.4f}")
                    print(f"    Distance to closest: {np.linalg.norm(robot_pos - closest_point)*100:.2f} cm")
                    print(f"  Contact: {self.in_contact}, force: {contact_force:.2f} N")
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.3)
        
        return self._compute_phase_1_metrics()
    
    def _get_contact_force(self) -> float:
        """Get contact force magnitude."""
        try:
            link_idx = -1
            if hasattr(self.robot, 'bumper_link_idx') and self.robot.bumper_link_idx is not None:
                link_idx = self.robot.bumper_link_idx
            elif hasattr(self.robot, 'contact_link_idx') and self.robot.contact_link_idx is not None:
                link_idx = self.robot.contact_link_idx
            
            force = get_contact_force(
                self.robot.uid, self.object_uid,
                linkIndexA=link_idx,
                max_contacts=4
            )
            return np.linalg.norm(force[:2])
        except:
            return 0.0
    
    def _compute_phase_1_metrics(self) -> Dict:
        """Compute Phase 1 metrics."""
        position_errors = np.array(self.history.position_errors)
        heading_errors = np.array(self.history.heading_errors)
        closest_u = np.array(self.history.closest_u_on_desired_segment)
        
        # Find contact phase
        in_contact = np.array(self.history.in_contact)
        contact_start_idx = None
        for i, contact in enumerate(in_contact):
            if contact:
                contact_start_idx = i
                break
        
        if contact_start_idx is None:
            return {
                'contact_achieved': False,
                'avg_position_error': float(np.mean(np.linalg.norm(position_errors, axis=1))),
                'avg_heading_error': float(np.mean(np.abs(heading_errors))),
            }
        
        # Metrics after contact
        contact_position_errors = position_errors[contact_start_idx:]
        contact_heading_errors = heading_errors[contact_start_idx:]
        contact_closest_u = closest_u[contact_start_idx:]
        
        return {
            'contact_achieved': True,
            'contact_time': float(self.history.times[contact_start_idx]),
            'avg_position_error': float(np.mean(np.linalg.norm(contact_position_errors, axis=1))),
            'max_position_error': float(np.max(np.linalg.norm(contact_position_errors, axis=1))),
            'avg_heading_error': float(np.mean(np.abs(contact_heading_errors))),
            'max_heading_error': float(np.max(np.abs(contact_heading_errors))),
            'avg_segment_u_error': float(np.mean(np.maximum(0.0, np.abs(contact_closest_u - np.clip(contact_closest_u, 0.0, 1.0))))),
        }
    
    def plot_phase_1_results(self, save_path: Optional[Path] = None):
        """Plot Phase 1 results."""
        times = np.array(self.history.times)
        robot_positions = np.array(self.history.robot_positions)
        intended_positions = np.array(self.history.intended_positions)
        closest_points = np.array(self.history.closest_points_on_desired_segment)
        contact_points = np.array(self.history.contact_point_positions)
        position_errors = np.array(self.history.position_errors)
        heading_errors = np.array(self.history.heading_errors)
        closest_u = np.array(self.history.closest_u_on_desired_segment)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Phase 1: Basic Requirements - {self.kinematics} wheel, t_param={self.desired_t_param:.3f}',
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Trajectory
        ax = axes[0, 0]
        ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', linewidth=1.5, label='Robot')
        ax.plot(intended_positions[:, 0], intended_positions[:, 1], 'g--', linewidth=1, alpha=0.7, label='Intended pos')
        ax.plot(contact_points[:, 0], contact_points[:, 1], 'r--', linewidth=1, alpha=0.7, label='Contact point')
        ax.plot(closest_points[:, 0], closest_points[:, 1], 'm:', linewidth=1, alpha=0.5, label='Closest point')
        ax.plot(robot_positions[0, 0], robot_positions[0, 1], 'go', markersize=8, label='Start')
        ax.plot(robot_positions[-1, 0], robot_positions[-1, 1], 'ro', markersize=8, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot 2: Position error
        ax = axes[0, 1]
        error_mags = np.linalg.norm(position_errors, axis=1)
        ax.plot(times, error_mags * 100, 'b-', linewidth=1.5)  # Convert to cm
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (cm)')
        ax.set_title('Position Error (Robot to Intended)')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Heading error
        ax = axes[0, 2]
        ax.plot(times, np.degrees(heading_errors), 'r-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading Error (deg)')
        ax.set_title('Heading Error')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: t_param tracking
        ax = axes[1, 0]
        ax.plot(times, closest_u, 'm-', linewidth=1.5, label='Closest u on desired segment')
        ax.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('u (segment coordinate)')
        ax.set_title('Closest Projection on Desired Segment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Contact force
        ax = axes[1, 1]
        contact_forces = np.array(self.history.contact_forces)
        ax.plot(times, contact_forces, 'r-', linewidth=1.5)
        ax.axhline(y=self.contact_threshold, color='g', linestyle='--', label='Threshold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact Force (N)')
        ax.set_title('Contact Force')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Metrics
        ax = axes[1, 2]
        ax.axis('off')
        metrics = self._compute_phase_1_metrics()
        if metrics['contact_achieved']:
            text = f"""PHASE 1 METRICS

Contact Achieved: Yes
Contact Time: {metrics['contact_time']:.2f} s

Position Error:
  Avg: {metrics['avg_position_error']*100:.2f} cm
  Max: {metrics['max_position_error']*100:.2f} cm

Heading Error:
  Avg: {np.degrees(metrics['avg_heading_error']):.2f} deg
  Max: {np.degrees(metrics['max_heading_error']):.2f} deg

Segment projection check:
  seg_idx: {self.desired_seg_idx}
  (if projection leaves [0,1], we assert-fail)
"""
        else:
            text = f"""PHASE 1 METRICS

Contact Achieved: No
Avg Position Error: {metrics['avg_position_error']*100:.2f} cm
Avg Heading Error: {np.degrees(metrics['avg_heading_error']):.2f} deg
"""
        ax.text(0.1, 0.9, text, fontsize=11, family='monospace',
               verticalalignment='top', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

    # -------------------------------------------------------------------------
    # Additional diagnostic plot for Phase 3: object vs robot velocities
    # -------------------------------------------------------------------------
    def plot_phase3_velocities(self, save_path: Optional[Path] = None):
        """Plot object vs robot velocities (vx, vy, omega) over time."""
        if len(self.history.times) == 0:
            print("No history to plot for Phase 3 velocities.")
            return

        times = np.array(self.history.times)
        robot_vels = np.array(self.history.robot_velocities)
        obj_vels = np.array(self.history.object_velocities)
        obj_omegas = np.array(self.history.object_angular_velocities)
        robot_omegas = robot_vels[:, 2] if robot_vels.shape[1] >= 3 else np.zeros_like(times)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'Phase 3 Velocities: {self.kinematics} wheel, t_param={self.desired_t_param:.3f}',
                     fontsize=14, fontweight='bold')

        # vx
        ax = axes[0, 0]
        ax.plot(times, robot_vels[:, 0], 'b-', label='robot vx')
        ax.plot(times, obj_vels[:, 0], 'r--', label='object vx')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('vx (m/s)')
        ax.set_title('Linear Velocity X')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # vy
        ax = axes[0, 1]
        ax.plot(times, robot_vels[:, 1], 'b-', label='robot vy')
        ax.plot(times, obj_vels[:, 1], 'r--', label='object vy')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('vy (m/s)')
        ax.set_title('Linear Velocity Y')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # omega
        ax = axes[1, 0]
        ax.plot(times, robot_omegas, 'b-', label='robot ω')
        ax.plot(times, obj_omegas, 'r--', label='object ω')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # contact force for context
        ax = axes[1, 1]
        contact_forces = np.array(self.history.contact_forces)
        ax.plot(times, contact_forces, 'm-', label='|contact force|')
        ax.axhline(y=self.contact_threshold, color='k', linestyle='--', label='threshold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Contact Force (context)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved Phase 3 velocity plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_phase_5_velocities(self, desired_speed: float, save_path: Optional[Path] = None, phase: int = 5):
        """Plot Phase 5/6 velocity tracking: desired speed, object speed, robot speed, and velocity components.
        
        Parameters
        ----------
        desired_speed : float
            Desired object speed
        save_path : Optional[Path]
            Path to save plot
        phase : int
            Phase number (5 or 6) for title and labels
        """
        if len(self.history.times) == 0:
            print(f"No history to plot for Phase {phase} velocities.")
            return

        times = np.array(self.history.times)
        robot_vels = np.array(self.history.robot_velocities)
        obj_vels = np.array(self.history.object_velocities)
        obj_speeds = np.linalg.norm(obj_vels, axis=1)
        robot_speeds = np.linalg.norm(robot_vels[:, :2], axis=1)
        
        # Velocity components
        v_base = np.array(self.history.v_base_history)
        v_constant = np.array(self.history.v_constant_history)
        v_velo_error_pi = np.array(self.history.v_velo_error_pi_history)
        in_contact = np.array(self.history.in_contact)

        # Label for the constant/feed-forward component
        v_constant_label = 'v_ff (feed-forward)' if phase == 6 else 'v_constant'
        v_pi_label = 'v_pi' if phase == 6 else 'v_velo_error_pi'

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Phase {phase} Velocities: {self.kinematics} wheel, t_param={self.desired_t_param:.3f}',
                     fontsize=14, fontweight='bold')

        # Plot 1: Desired vs actual speed (object speed for phase 5/6, contact point speed for phase 7)
        ax = axes[0, 0]
        ax.plot(times, np.full_like(times, desired_speed), 'g--', label='desired speed', linewidth=2)
        if phase == 7:
            # For Phase 7, plot contact point speeds
            cp_speeds = np.linalg.norm(np.array(self.history.contact_point_velocities), axis=1)
            ax.plot(times, cp_speeds, 'r-', label='contact point speed', linewidth=1.5)
            ax.set_title('Contact Point Speed Tracking')
        else:
            # For Phase 5/6, plot object speeds
            ax.plot(times, obj_speeds, 'r-', label='object speed', linewidth=1.5)
            ax.set_title('Object Speed Tracking')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=desired_speed, color='g', linestyle='--', alpha=0.5)

        # Plot 2: Robot speed
        ax = axes[0, 1]
        ax.plot(times, robot_speeds, 'b-', label='robot speed', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Robot Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Velocity components (v_base, v_constant/v_ff, v_velo_error_pi/v_pi)
        ax = axes[1, 0]
        ax.plot(times, v_base, 'c-', label='v_base', linewidth=1.5, alpha=0.7)
        ax.plot(times, v_constant, 'm-', label=v_constant_label, linewidth=1.5, alpha=0.7)
        ax.plot(times, v_velo_error_pi, 'orange', label=v_pi_label, linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Component (m/s)')
        ax.set_title('Velocity Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Contact state
        ax = axes[1, 1]
        ax.fill_between(times, 0, 1, where=in_contact, alpha=0.3, color='green', label='in contact')
        ax.fill_between(times, 0, 1, where=~in_contact, alpha=0.3, color='red', label='not in contact')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact State')
        ax.set_title('Contact State Over Time')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No Contact', 'In Contact'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved Phase {phase} velocity plot to {save_path}")
        else:
            plt.show()

        plt.close()

    # ----------------------------------------------------------------------
    # PHASE 2: Force-position hybrid control
    # ----------------------------------------------------------------------
    def run_phase_2(self, desired_force: float, gui: bool = True, duration: float = 10.0) -> Dict:
        """Phase 2: Force-position hybrid control (simple spring model).

        - Requirement 1 & 2 still enforced (position offset + heading)
        - Desired force achieved by adjusting the offset along normal_outward using a spring model:
            offset_adjust = desired_force / spring_stiffness
            intended_pos = contact_point + (ROBOT_RADIUS - offset_adjust) * normal_outward
        - If offset_adjust is large, the intended position moves closer to the object to generate force.
        """
        print(f"\n{'='*60}")
        print(f"PHASE 2: Force-Position Hybrid Control")
        print(f"{'='*60}")
        print(f"  Desired t_param: {self.desired_t_param:.3f}")
        print(f"  Desired force: {desired_force:.2f} N")
        print(f"  Duration: {duration:.1f} s")
        print(f"  Spring stiffness: {self.spring_stiffness:.1f} N/m")

        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0

        for step in range(n_steps):
            if step_count % CTRL_STEP == 0:
                # Robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()

                # Object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]

                # Contact detection
                contact_force = self._get_contact_force()
                self.in_contact = contact_force > self.contact_threshold

                # Contact point & normals in world
                cos_t = np.cos(object_orientation)
                sin_t = np.sin(object_orientation)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                contact_point_world = R @ self.contact_point_body + object_pos
                normal_outward_world = R @ self.normal_outward

                # Closest point on desired segment
                closest_point, closest_u = closest_point_on_desired_segment(
                    robot_pos=robot_pos,
                    object_pos=object_pos,
                    object_orientation=object_orientation,
                    seg_p1_body=self.desired_seg_p1_body,
                    seg_p2_body=self.desired_seg_p2_body,
                    assert_on_segment=True,
                    tol=1e-4,
                )

                # Requirement 1: Intended position with force-based offset
                offset_adjust = desired_force / self.spring_stiffness  # meters
                desired_offset = max(0.01, ROBOT_RADIUS - offset_adjust)
                intended_pos = contact_point_world + desired_offset * normal_outward_world
                position_error = intended_pos - robot_pos

                # Requirement 2: Heading toward reference (closest point if sliding)
                reference_point = closest_point if self.in_contact else contact_point_world
                desired_heading = np.arctan2((reference_point - robot_pos)[1],
                                             (reference_point - robot_pos)[0])
                heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                          np.cos(desired_heading - robot_heading))

                # Contact point velocity (for reference)
                r_cp = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
                contact_point_velocity = object_velocity + v_rotation

                # Control: simple P on position, heading P
                kp_pos = 2.0
                vel_cmd_xy = kp_pos * position_error
                speed = np.linalg.norm(vel_cmd_xy)
                if speed > self.max_linear_speed:
                    vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)

                omega = self.kp_heading * heading_error
                omega = np.clip(omega, -1.0, 1.0)

                cmd = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
                self.robot.command_velocity(cmd)

                # Record history
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(desired_heading)
                self.history.heading_errors.append(heading_error)
                self.history.closest_points_on_desired_segment.append(closest_point.copy())
                self.history.closest_u_on_desired_segment.append(float(closest_u))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)

                # Debug print occasionally
                if step_count % (CTRL_STEP * 10) == 0:
                    print(f"\n[t={t:.2f}s] Phase 2 Analysis:")
                    print(f"  Robot pos: {robot_pos}, heading: {robot_heading:.3f} rad")
                    print(f"  Desired force: {desired_force:.2f} N, offset_adjust: {offset_adjust:.4f} m, desired_offset: {desired_offset:.4f} m")
                    print(f"  Intended pos: {intended_pos}")
                    print(f"  Position error: {position_error} (mag: {np.linalg.norm(position_error)*100:.2f} cm)")
                    print(f"  Heading error: {heading_error:.3f} rad ({np.degrees(heading_error):.1f} deg)")
                    print(f"  Closest u on desired segment: {closest_u:.4f}")
                    print(f"  Contact: {self.in_contact}, force: {contact_force:.2f} N")

            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1

            if gui:
                time.sleep(TIMESTEP * 0.3)

        return self._compute_phase_1_metrics()  # reuse metrics structure

    # ----------------------------------------------------------------------
    # PHASE 2.5: Force-position hybrid with PI on force
    # ----------------------------------------------------------------------
    def run_phase_25(self, desired_force: float, gui: bool = True, duration: float = 10.0) -> Dict:
        """Phase 2.5: Force-position hybrid control with PI feedback on force.

        - Requirement 1 & 2 still enforced (position + heading)
        - Desired force tracked by adjusting the offset along normal_outward with PI:
            error = F_desired - F_sensed
            offset_cmd += -(kp * error + ki * integral(error))
          Positive error (need more force) moves robot closer (smaller offset).
        """
        print(f"\n{'='*60}")
        print(f"PHASE 2.5: Force-Position Hybrid with PI")
        print(f"{'='*60}")
        print(f"  Desired t_param: {self.desired_t_param:.3f}")
        print(f"  Desired force: {desired_force:.2f} N")
        print(f"  Duration: {duration:.1f} s")
        print(f"  PI gains: Kp={self.kp_force:.4f} m/N, Ki={self.ki_force:.4f} m/(N*s)")

        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0

        # Reset PI integral
        self.force_error_int = 0.0

        # Start with nominal offset = ROBOT_RADIUS
        offset_cmd = ROBOT_RADIUS

        for step in range(n_steps):
            if step_count % CTRL_STEP == 0:
                dt_ctrl = CTRL_STEP * TIMESTEP

                # Robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()

                # Object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]

                # Contact detection
                contact_force = self._get_contact_force()
                self.in_contact = contact_force > self.contact_threshold

                # Contact point & normals in world
                cos_t = np.cos(object_orientation)
                sin_t = np.sin(object_orientation)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                contact_point_world = R @ self.contact_point_body + object_pos
                normal_outward_world = R @ self.normal_outward

                # Closest point on desired segment (for heading reference)
                closest_point, closest_u = closest_point_on_desired_segment(
                    robot_pos=robot_pos,
                    object_pos=object_pos,
                    object_orientation=object_orientation,
                    seg_p1_body=self.desired_seg_p1_body,
                    seg_p2_body=self.desired_seg_p2_body,
                    assert_on_segment=True,
                    tol=1e-4,
                )

                # Force PI update (only when in contact to avoid windup on noise)
                if self.in_contact:
                    force_error = desired_force - contact_force
                    self.force_error_int += force_error * dt_ctrl
                    delta_offset = -(self.kp_force * force_error + self.ki_force * self.force_error_int)

                    print(f"  Delta offset: {delta_offset:.4f} m from force error {force_error:.2f} N and integral {self.force_error_int:.2f}")
                    offset_cmd += delta_offset
                else:
                    # When not in contact, slowly drive offset back toward ROBOT_RADIUS
                    offset_cmd += (ROBOT_RADIUS - offset_cmd) * 0.1

                # Clamp offset to reasonable range
                offset_cmd = float(np.clip(offset_cmd, -0.3, ROBOT_RADIUS * 1.5))
                

                print(f"  Offset cmd: {offset_cmd:.4f} m")

                # Requirement 1: Intended position with PI-controlled offset
                intended_pos = contact_point_world + offset_cmd * normal_outward_world
                position_error = intended_pos - robot_pos

                # Requirement 2: Heading toward reference (closest point if sliding)
                reference_point = closest_point if self.in_contact else contact_point_world
                desired_heading = np.arctan2((reference_point - robot_pos)[1],
                                             (reference_point - robot_pos)[0])
                heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                          np.cos(desired_heading - robot_heading))

                # Contact point velocity (for reference)
                r_cp = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
                contact_point_velocity = object_velocity + v_rotation

                # Control: simple P on position, heading P
                kp_pos = 2.0
                vel_cmd_xy = kp_pos * position_error
                speed = np.linalg.norm(vel_cmd_xy)

                print(f"  speed {speed:.4f} m/s and max speed {self.max_linear_speed:.4f} m/s")
                if speed > self.max_linear_speed:
                    vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)

                omega = self.kp_heading * heading_error
                omega = np.clip(omega, -1.0, 1.0)

                cmd = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
                self.robot.command_velocity(cmd)

                # Record history (same structure as Phase 1 for reuse of plots/metrics)
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(desired_heading)
                self.history.heading_errors.append(heading_error)
                self.history.closest_points_on_desired_segment.append(closest_point.copy())
                self.history.closest_u_on_desired_segment.append(float(closest_u))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)

                # Debug print occasionally
                if step_count % (CTRL_STEP * 10) == 0:
                    print(f"\n[t={t:.2f}s] Phase 2.5 Analysis:")
                    print(f"  Robot pos: {robot_pos}, heading: {robot_heading:.3f} rad")
                    print(f"  Desired force: {desired_force:.2f} N, sensed: {contact_force:.2f} N")
                    if self.in_contact:
                        print(f"  Force error: {force_error:.2f} N, integral: {self.force_error_int:.2f}")
                    print(f"  Offset_cmd: {offset_cmd:.4f} m (ROBOT_RADIUS={ROBOT_RADIUS:.4f} m)")
                    print(f"  Intended pos: {intended_pos}")
                    print(f"  Position error: {position_error} (mag: {np.linalg.norm(position_error)*100:.2f} cm)")
                    print(f"  Heading error: {heading_error:.3f} rad ({np.degrees(heading_error):.1f} deg)")
                    print(f"  Closest u on desired segment: {closest_u:.4f}")
                    print(f"  Contact: {self.in_contact}, force: {contact_force:.2f} N")

            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1

            if gui:
                time.sleep(TIMESTEP * 0.3)

        return self._compute_phase_1_metrics()  # reuse metrics structure

    # ----------------------------------------------------------------------
    # PHASE 4: Force-position hybrid control using Newton's law
    # ----------------------------------------------------------------------
    def run_phase_4(self, desired_force: float, gui: bool = True, duration: float = 10.0) -> Dict:
        """Phase 4: Force-position hybrid control using Newton's law (F = m*a).

        - Requirement 1 & 2 still enforced (position offset + heading)
        - Desired force achieved by computing acceleration from Newton's law:
            a = F_desired / m_robot
        - Integrate acceleration to get velocity in robot's local X direction:
            v_x_local = v_x_local + a * dt  (or use a * dt directly)
        - Transform to world frame and combine with position control for y and heading control for omega.

        Note: This is a simplified model that ignores object dynamics, friction, etc.
        """
        print(f"\n{'='*60}")
        print(f"PHASE 4: Force-Position Hybrid Control (Newton's Law)")
        print(f"{'='*60}")
        print(f"  Desired t_param: {self.desired_t_param:.3f}")
        print(f"  Desired force: {desired_force:.2f} N")
        print(f"  Duration: {duration:.1f} s")

        # Get robot mass from PyBullet
        # Note: With useFixedBase=True, base link has mass=0, so we sum all link masses
        total_mass = 0.0
        for i in range(pyb.getNumJoints(self.robot.uid)):
            link_dynamics = pyb.getDynamicsInfo(self.robot.uid, i)
            total_mass += link_dynamics[0]  # mass is first element
        
        # Fallback: if still zero, use reasonable default
        if total_mass < 0.001:
            total_mass = 1.0  # Default mass for small robot
            print(f"Warning: Robot mass was zero, using default {total_mass} kg")
        
        robot_mass = total_mass
        print(f"  Robot mass: {robot_mass:.3f} kg")

        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        dt_ctrl = CTRL_STEP * TIMESTEP

        # Velocity accumulator for Newton's law integration
        v_newton = 0.0  # Velocity from Newton's law (F = m*a, integrated)
        
        # Control gain for base velocity from position error
        kp_approach = 2.0  # Gain for position-based approach velocity

        for step in range(n_steps):
            if step_count % CTRL_STEP == 0:
                # Robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()

                # Object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]

                # Contact detection
                contact_force = self._get_contact_force()
                self.in_contact = contact_force > self.contact_threshold

                # Contact point & normals in world
                cos_t = np.cos(object_orientation)
                sin_t = np.sin(object_orientation)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                contact_point_world = R @ self.contact_point_body + object_pos
                normal_outward_world = R @ self.normal_outward
                normal_inward_world = -normal_outward_world

                # Closest point on desired segment
                closest_point, closest_u = closest_point_on_desired_segment(
                    robot_pos=robot_pos,
                    object_pos=object_pos,
                    object_orientation=object_orientation,
                    seg_p1_body=self.desired_seg_p1_body,
                    seg_p2_body=self.desired_seg_p2_body,
                    assert_on_segment=True,
                    tol=1e-4,
                )

                # Requirement 1: Intended position (no force-based offset here, just nominal)
                intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
                position_error = intended_pos - robot_pos

                # Requirement 2: Heading toward reference (closest point if sliding)
                reference_point = closest_point if self.in_contact else contact_point_world
                desired_heading = np.arctan2((reference_point - robot_pos)[1],
                                             (reference_point - robot_pos)[0])
                heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                          np.cos(desired_heading - robot_heading))

                # Contact point velocity (for reference)
                r_cp = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
                contact_point_velocity = object_velocity + v_rotation

                # Decompose position error into components:
                # - Along normal_inward: base velocity + Newton's law velocity
                # - Perpendicular to normal_inward: position control
                normal_along = normal_inward_world
                normal_perp = np.array([-normal_along[1], normal_along[0]])  # Rotate 90 deg CCW
                
                # Project position error onto these directions
                error_along = np.dot(position_error, normal_along)
                error_perp = np.dot(position_error, normal_perp)
                
                # BASE VELOCITY: Always compute from position error (ensures approach/maintenance)
                # This provides a base velocity that tries to reduce position error along normal.
                # Inspired by Henrik's PushController approach: base velocity from path/position,
                # which ensures the robot always tries to maintain/achieve contact position.
                v_base = kp_approach * error_along
                
                # NEWTON'S LAW: Compute acceleration and integrate to velocity
                # NOTE: This is OPEN-LOOP control - we command velocity based on desired force,
                # but there's no feedback from actual measured force to adjust the command.
                # The "force sensor" (get_contact_force) is actually an impulse/constraint force sensor
                # and may not reflect the true force the robot is applying, especially in multi-robot
                # scenarios where constraint forces can dominate. Therefore, we use open-loop
                # velocity commands based on desired force rather than closed-loop force feedback.
                if self.in_contact:
                    # F = m * a  =>  a = F / m
                    acceleration = desired_force / max(robot_mass, 0.001)  # Avoid division by zero
                    
                    v_newton = 0
                    # Integrate: v = v + a * dt (open-loop integration)
                    v_newton = v_newton + acceleration * dt_ctrl
                    
                    # Optional: Add damping to prevent unbounded growth
                    v_newton = v_newton * 0.99  # Small damping factor

                    
                    # Clamp Newton velocity to reasonable limits
                    v_newton = np.clip(v_newton, -self.max_linear_speed, self.max_linear_speed)
                    
                    # HYBRID VELOCITY: Combine base velocity (position feedback) with Newton's law velocity (force-based)
                    # This ensures we always try to maintain position while adding force-based acceleration.
                    # The base velocity provides stability and ensures contact maintenance,
                    # while Newton's law velocity adds the desired force component.

                    v_along = v_base + v_newton
                else:
                    # When not in contact: use base velocity only to approach object
                    # Reset Newton velocity accumulator when contact is lost (no force applied when not touching)
                    v_newton = 0.0
                    v_along = v_base
                
                # Clamp along-normal velocity
                v_along = np.clip(v_along, -self.max_linear_speed, self.max_linear_speed)
                
                # Perpendicular velocity: position control
                v_perp = kp_approach * error_perp
                v_perp = np.clip(v_perp, -self.max_linear_speed, self.max_linear_speed)
                
                # Combine velocities in world frame
                vel_cmd_xy = v_along * normal_along + v_perp * normal_perp
                
                # Clamp total speed
                speed = np.linalg.norm(vel_cmd_xy)
                if speed > self.max_linear_speed:
                    vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)

                omega = self.kp_heading * heading_error
                omega = np.clip(omega, -1.0, 1.0)

                cmd = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
                self.robot.command_velocity(cmd)

                # Record history
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(desired_heading)
                self.history.heading_errors.append(heading_error)
                self.history.closest_points_on_desired_segment.append(closest_point.copy())
                self.history.closest_u_on_desired_segment.append(float(closest_u))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)

                # Debug print occasionally
                if step_count % (CTRL_STEP * 10) == 0:
                    acceleration_calc = desired_force / max(robot_mass, 0.001) if self.in_contact else 0.0
                    print(f"\n[t={t:.2f}s] Phase 4 Analysis:")
                    print(f"  Robot pos: {robot_pos}, heading: {robot_heading:.3f} rad")
                    print(f"  Desired force: {desired_force:.2f} N, sensed: {contact_force:.2f} N")
                    print(f"  Robot mass: {robot_mass:.3f} kg")
                    print(f"  Acceleration (F/m): {acceleration_calc:.4f} m/s²")
                    print(f"  v_base (from position error): {v_base:.4f} m/s")
                    print(f"  v_newton (integrated): {v_newton:.4f} m/s")
                    print(f"  v_along (combined): {v_along:.4f} m/s")
                    print(f"  error_along: {error_along:.4f} m")
                    print(f"  Intended pos: {intended_pos}")
                    print(f"  Position error: {position_error} (mag: {np.linalg.norm(position_error)*100:.2f} cm)")
                    print(f"  Heading error: {heading_error:.3f} rad ({np.degrees(heading_error):.1f} deg)")
                    print(f"  Closest u on desired segment: {closest_u:.4f}")
                    print(f"  Contact: {self.in_contact}, force: {contact_force:.2f} N")

            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1

            if gui:
                time.sleep(TIMESTEP * 0.3)

        return self._compute_phase_1_metrics()  # reuse metrics structure

    # ----------------------------------------------------------------------
    # PHASE 5: Closed-loop object velocity control (single robot)
    # ----------------------------------------------------------------------
    def run_phase_5(self, desired_speed: float, gui: bool = True, duration: float = 10.0) -> Dict:
        """Phase 5: Closed-loop object velocity control using feedback (single robot).

        This phase replaces Phase 4's open-loop Newton's law with closed-loop feedback
        from the object's actual velocity. The structure is still v_base + v_velo_error_pi,
        but v_velo_error_pi is now controlled by a PI controller on object velocity error.

        Control Strategy:
        - v_base: Position feedback (always active, ensures contact maintenance)
        - v_velo_error_pi: PI-controlled velocity based on object speed error
          - For single robot: Higher v_velo_error_pi → faster object movement
          - PI controller: v_velo_error_pi = kp * error + ki * integral(error)
          - Error: desired_speed - ||object_velocity|| (speed magnitude error)

        NOTE: This is designed for SINGLE robot pushing. Multi-robot scenarios are
        more complex because:
        - Object velocity depends on combined forces from all robots
        - Each robot's contribution is not independent
        - Coordination/communication may be needed

        NOTE: We only control linear speed magnitude, not direction or angular velocity,
        because in single pusher scenarios, the object's motion direction and rotation
        are coupled to the contact geometry and friction.

        Parameters
        ----------
        desired_speed : float
            Desired object speed magnitude (m/s)
        gui : bool
            Show GUI
        duration : float
            Test duration (s)
        """
        print(f"\n{'='*60}")
        print(f"PHASE 5: Closed-Loop Object Velocity Control (Single Robot)")
        print(f"{'='*60}")
        print(f"  Desired t_param: {self.desired_t_param:.3f}")
        print(f"  Desired object speed: {desired_speed:.3f} m/s")
        print(f"  Duration: {duration:.1f} s")

        # PI controller gains for object velocity tracking
        kp_vel = 5   # Proportional gain (m/s per m/s error)
        ki_vel = 0.1   # Integral gain (m/s per (m/s)*s error)
        velocity_error_int = 0.0  # Integral accumulator

        # Control gain for base velocity from position error
        kp_approach = 2.0  # Gain for position-based approach velocity

        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        dt_ctrl = CTRL_STEP * TIMESTEP

        # Velocity command accumulator (v_velo_error_pi)
        v_velo_error_pi = 0.0  # Velocity component controlled by object velocity feedback

        for step in range(n_steps):
            if step_count % CTRL_STEP == 0:
                # Robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()

                # Object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]

                # Contact detection
                contact_force = self._get_contact_force()
                self.in_contact = contact_force > self.contact_threshold

                # Contact point & normals in world
                cos_t = np.cos(object_orientation)
                sin_t = np.sin(object_orientation)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                contact_point_world = R @ self.contact_point_body + object_pos
                normal_outward_world = R @ self.normal_outward
                normal_inward_world = -normal_outward_world

                # Closest point on desired segment
                closest_point, closest_u = closest_point_on_desired_segment(
                    robot_pos=robot_pos,
                    object_pos=object_pos,
                    object_orientation=object_orientation,
                    seg_p1_body=self.desired_seg_p1_body,
                    seg_p2_body=self.desired_seg_p2_body,
                    assert_on_segment=True,
                    tol=1e-4,
                )

                # Requirement 1: Intended position (no force-based offset here, just nominal)
                intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
                position_error = intended_pos - robot_pos

                # Requirement 2: Heading toward reference (closest point if sliding)
                reference_point = closest_point if self.in_contact else contact_point_world
                desired_heading = np.arctan2((reference_point - robot_pos)[1],
                                             (reference_point - robot_pos)[0])
                heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                          np.cos(desired_heading - robot_heading))

                # Contact point velocity (for reference)
                r_cp = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
                contact_point_velocity = object_velocity + v_rotation

                # Decompose position error into components:
                # - Along normal_inward: base velocity + velocity feedback control
                # - Perpendicular to normal_inward: position control
                normal_along = normal_inward_world
                normal_perp = np.array([-normal_along[1], normal_along[0]])  # Rotate 90 deg CCW
                
                # Project position error onto these directions
                error_along = np.dot(position_error, normal_along)
                error_perp = np.dot(position_error, normal_perp)
                
                # BASE VELOCITY: Always compute from position error (ensures approach/maintenance)
                # This provides a base velocity that tries to reduce position error along normal.
                # Inspired by Henrik's PushController approach: base velocity from path/position,
                # which ensures the robot always tries to maintain/achieve contact position.
                v_base = kp_approach * error_along
                
                # CLOSED-LOOP OBJECT VELOCITY CONTROL: PI controller on object speed error
                # NOTE: This is CLOSED-LOOP control - we measure actual object speed and
                # adjust v_velo_error_pi based on the error between desired and actual speed.
                # For single robot: Higher v_velo_error_pi → faster object movement (monotonic relationship).
                # This relationship allows us to use a PI controller to track desired object speed.
                # We only control speed magnitude, not direction, because in single pusher scenarios,
                # the object's motion direction and angular velocity are coupled to contact geometry.
                v_constant_velocity = 0.1  # constant velocity component
                if self.in_contact:
                    # Compute actual object speed magnitude
                    v_obj_actual_speed = np.linalg.norm(object_velocity)
                    
                    # Speed error (desired - actual)
                    speed_error = desired_speed - v_obj_actual_speed
                    
                    # PI controller update
                    velocity_error_int += speed_error * dt_ctrl
                    
                    # Compute v_velo_error_pi from PI controller
                    # Positive error (object too slow) → increase v_velo_error_pi (push harder)
                    # Negative error (object too fast) → decrease v_velo_error_pi (push softer)
                    v_velo_error_pi = kp_vel * speed_error + ki_vel * velocity_error_int

                    print(f"  v_velo_error_pi: {v_velo_error_pi:.4f} m/s and speed error: {speed_error:.4f} m/s and velocity error integral: {velocity_error_int:.4f} m/s*s")
                    
                    # Clamp v_velo_error_pi to reasonable limits
                    v_velo_error_pi = np.clip(v_velo_error_pi, v_constant_velocity, self.max_linear_speed)
                    
                    # HYBRID VELOCITY: Combine base velocity (position feedback) with velocity feedback control
                    # This ensures we always try to maintain position while tracking desired object speed.
                    # The base velocity provides stability and ensures contact maintenance,
                    # while v_velo_error_pi adds the speed tracking component.
                    v_along = v_base + v_velo_error_pi + v_constant_velocity
                else:
                    # When not in contact: use base velocity only to approach object
                    # Reset velocity feedback accumulator when contact is lost
                    velocity_error_int = 0.0
                    v_velo_error_pi = 0.0
                    v_along = v_base + v_constant_velocity * 2
                
                # Clamp along-normal velocity
                v_along = np.clip(v_along, -self.max_linear_speed, self.max_linear_speed)
                
                # Perpendicular velocity: position control
                v_perp = kp_approach * error_perp
                v_perp = np.clip(v_perp, -self.max_linear_speed, self.max_linear_speed)
                
                # Combine velocities in world frame
                vel_cmd_xy = v_along * normal_along + v_perp * normal_perp
                
                # Clamp total speed
                speed = np.linalg.norm(vel_cmd_xy)
                if speed > self.max_linear_speed:
                    vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)

                omega = self.kp_heading * heading_error
                omega = np.clip(omega, -1.0, 1.0)

                cmd = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
                self.robot.command_velocity(cmd)

                # Record history
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(desired_heading)
                self.history.heading_errors.append(heading_error)
                self.history.closest_points_on_desired_segment.append(closest_point.copy())
                self.history.closest_u_on_desired_segment.append(float(closest_u))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)
                # Phase 5 specific: velocity components
                self.history.v_base_history.append(v_base)
                self.history.v_constant_history.append(v_constant_velocity)
                self.history.v_velo_error_pi_history.append(v_velo_error_pi)

                # Debug print occasionally
                if step_count % (CTRL_STEP * 10) == 0:
                    if self.in_contact:
                        v_obj_actual_speed = np.linalg.norm(object_velocity)
                        speed_error = desired_speed - v_obj_actual_speed
                    else:
                        v_obj_actual_speed = 0.0
                        speed_error = 0.0
                    
                    print(f"\n[t={t:.2f}s] Phase 5 Analysis:")
                    print(f"  Robot pos: {robot_pos}, heading: {robot_heading:.3f} rad")
                    print(f"  Desired object speed: {desired_speed:.4f} m/s")
                    print(f"  Actual object speed: {np.linalg.norm(object_velocity):.4f} m/s")
                    print(f"  Actual object velocity: {object_velocity} m/s")
                    if self.in_contact:
                        print(f"  Speed error: {speed_error:.4f} m/s")
                        print(f"  Speed error integral: {velocity_error_int:.4f} m/s*s")
                    print(f"  v_base (from position error): {v_base:.4f} m/s")
                    print(f"  v_velo_error_pi (PI controlled): {v_velo_error_pi:.4f} m/s")
                    print(f"  v_along (combined): {v_along:.4f} m/s")
                    print(f"  error_along: {error_along:.4f} m")
                    print(f"  Intended pos: {intended_pos}")
                    print(f"  Position error: {position_error} (mag: {np.linalg.norm(position_error)*100:.2f} cm)")
                    print(f"  Heading error: {heading_error:.3f} rad ({np.degrees(heading_error):.1f} deg)")
                    print(f"  Closest u on desired segment: {closest_u:.4f}")
                    print(f"  Contact: {self.in_contact}, force: {contact_force:.2f} N")

            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1

            if gui:
                time.sleep(TIMESTEP * 0.3)

        return self._compute_phase_1_metrics()  # reuse metrics structure

    # ----------------------------------------------------------------------
    # PHASE 6: Feed-Forward + PI Control with Smooth Contact Transitions
    # ----------------------------------------------------------------------
    def run_phase_6(self, desired_speed: float, gui: bool = True, duration: float = 10.0) -> Dict:
        """Phase 6: Feed-Forward + PI Control with smooth contact transitions.

        This phase improves Phase 5 by adding:
        1. Feed-forward component (plant compensation):
           - Static friction: v_ff_static = K_static
           - Viscous friction: v_ff_viscous = K_alpha * desired_speed
           - Total: v_ff = v_ff_static + v_ff_viscous
        2. Smooth contact transitions:
           - Hysteresis for contact detection (different thresholds for enter/exit)
           - Smooth integral reset (gradual decay instead of instant reset)
           - Contact state filtering (exponential moving average)
        3. Combined control: v_along = v_base + v_ff + v_pi

        Parameters
        ----------
        desired_speed : float
            Desired object speed magnitude (m/s)
        gui : bool
            Show GUI
        duration : float
            Test duration (s)
        """
        print(f"\n{'='*60}")
        print(f"PHASE 6: Feed-Forward + PI Control (Single Robot)")
        print(f"{'='*60}")
        print(f"  Desired t_param: {self.desired_t_param:.3f}")
        print(f"  Desired object speed: {desired_speed:.3f} m/s")
        print(f"  Duration: {duration:.1f} s")

        # Feed-forward gains (plant compensation)
        K_static = 0.03  # Static friction compensation (m/s)
        K_alpha = 0.8    # Viscous friction coefficient

        # PI controller gains
        kp_vel = 1.0     # Proportional gain (reduced since FF does heavy lifting)
        ki_vel = 0.1     # Integral gain
        velocity_error_int = 0.0  # Integral accumulator
        velocity_error_int_max = 0.5  # Clamp to prevent windup

        # Control gain for base velocity from position error
        kp_approach = 2.0  # Gain for position-based approach velocity

        # Contact detection with hysteresis
        contact_threshold_on = 2.0  # Enter contact when force > this
        contact_threshold_off = 0.2  # Exit contact when force < this
        contact_state_filtered = 0.0  # Filtered contact state (0-1)
        contact_filter_alpha = 0.1   # Exponential moving average coefficient

        # Integral decay when not in contact
        integral_decay_rate = 0.95  # Decay per control step

        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        dt_ctrl = CTRL_STEP * TIMESTEP

        # Track previous contact state for hysteresis
        in_contact_prev = False

        for step in range(n_steps):
            if step_count % CTRL_STEP == 0:
                # Robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()

                # Object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]

                # Contact detection with filtering
                contact_force = self._get_contact_force()
                
                # Filter contact force (exponential moving average)
                contact_state_filtered = (1.0 - contact_filter_alpha) * contact_state_filtered + \
                                        contact_filter_alpha * (1.0 if contact_force > 0.5 else 0.0)
                
                # Hysteresis-based contact detection
                if in_contact_prev:
                    # Was in contact: use lower threshold to exit
                    self.in_contact = contact_force > contact_threshold_off
                else:
                    # Was not in contact: use higher threshold to enter
                    self.in_contact = contact_force > contact_threshold_on

                in_contact_prev = self.in_contact

                # Contact point & normals in world
                cos_t = np.cos(object_orientation)
                sin_t = np.sin(object_orientation)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                contact_point_world = R @ self.contact_point_body + object_pos
                normal_outward_world = R @ self.normal_outward
                normal_inward_world = -normal_outward_world

                # Closest point on desired segment
                closest_point, closest_u = closest_point_on_desired_segment(
                    robot_pos=robot_pos,
                    object_pos=object_pos,
                    object_orientation=object_orientation,
                    seg_p1_body=self.desired_seg_p1_body,
                    seg_p2_body=self.desired_seg_p2_body,
                    assert_on_segment=True,
                    tol=1e-4,
                )

                # Requirement 1: Intended position (no force-based offset here, just nominal)
                intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
                position_error = intended_pos - robot_pos

                # Requirement 2: Heading toward reference (closest point if sliding)
                reference_point = closest_point if self.in_contact else contact_point_world
                desired_heading = np.arctan2((reference_point - robot_pos)[1],
                                             (reference_point - robot_pos)[0])
                heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                          np.cos(desired_heading - robot_heading))

                # Contact point velocity (for reference)
                r_cp = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
                contact_point_velocity = object_velocity + v_rotation

                # Decompose position error into components:
                # - Along normal_inward: base velocity + feed-forward + PI control
                # - Perpendicular to normal_inward: position control
                normal_along = normal_inward_world
                normal_perp = np.array([-normal_along[1], normal_along[0]])  # Rotate 90 deg CCW
                
                # Project position error onto these directions
                error_along = np.dot(position_error, normal_along)
                error_perp = np.dot(position_error, normal_perp)
                
                # BASE VELOCITY: Always compute from position error (ensures approach/maintenance)
                v_base = kp_approach * error_along
                
                # FEED-FORWARD COMPONENT (Plant Compensation)
                # Static friction: constant offset to overcome static friction
                v_ff_static = K_static
                # Viscous friction: proportional to desired speed
                v_ff_viscous = K_alpha * desired_speed
                # Total feed-forward
                v_ff = v_ff_static + v_ff_viscous
                
                # PI CONTROLLER (Error Correction)
                if self.in_contact:
                    # Compute actual object speed magnitude
                    v_obj_actual_speed = np.linalg.norm(object_velocity)
                    
                    # Speed error (desired - actual)
                    speed_error = desired_speed - v_obj_actual_speed
                    
                    # PI controller update
                    velocity_error_int += speed_error * dt_ctrl
                    
                    # Clamp integral to prevent windup
                    velocity_error_int = np.clip(velocity_error_int, -velocity_error_int_max, velocity_error_int_max)
                    
                    # Compute v_pi from PI controller
                    # Positive error (object too slow) → increase v_pi (push harder)
                    # Negative error (object too fast) → decrease v_pi (push softer)
                    v_pi = kp_vel * speed_error + ki_vel * velocity_error_int
                    
                    # Clamp PI output to reasonable limits
                    v_pi = np.clip(v_pi, -self.max_linear_speed, self.max_linear_speed)
                    
                    # COMBINED VELOCITY: Feed-forward + PI
                    # Feed-forward does the "heavy lifting" (overcomes friction),
                    # PI only needs to correct small errors
                    v_along = v_base + v_ff + v_pi
                else:
                    # When not in contact: use base velocity + feed-forward (for approach)
                    # Smoothly decay integral instead of instant reset
                    velocity_error_int *= integral_decay_rate
                    v_pi = 0.0
                    # Still use feed-forward for approach (helps overcome static friction)
                    v_along = v_base + v_ff * 0.5  # Reduced FF when not in contact
                
                # Clamp along-normal velocity
                v_along = np.clip(v_along, -self.max_linear_speed, self.max_linear_speed)
                
                # Perpendicular velocity: position control
                v_perp = kp_approach * error_perp
                v_perp = np.clip(v_perp, -self.max_linear_speed, self.max_linear_speed)
                
                # Combine velocities in world frame
                vel_cmd_xy = v_along * normal_along + v_perp * normal_perp
                
                # Clamp total speed
                speed = np.linalg.norm(vel_cmd_xy)
                if speed > self.max_linear_speed:
                    vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)

                omega = self.kp_heading * heading_error
                omega = np.clip(omega, -1.0, 1.0)

                cmd = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
                self.robot.command_velocity(cmd)

                # Record history (reuse Phase 5 structure)
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(desired_heading)
                self.history.heading_errors.append(heading_error)
                self.history.closest_points_on_desired_segment.append(closest_point.copy())
                self.history.closest_u_on_desired_segment.append(float(closest_u))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)
                # Phase 6 velocity components (reuse Phase 5 fields)
                # v_base, v_ff (as v_constant), v_pi (as v_velo_error_pi)
                self.history.v_base_history.append(v_base)
                self.history.v_constant_history.append(v_ff)  # Store feed-forward as "constant"
                self.history.v_velo_error_pi_history.append(v_pi)

                # Debug print occasionally
                if step_count % (CTRL_STEP * 10) == 0:
                    if self.in_contact:
                        v_obj_actual_speed = np.linalg.norm(object_velocity)
                        speed_error = desired_speed - v_obj_actual_speed
                    else:
                        v_obj_actual_speed = 0.0
                        speed_error = 0.0
                    
                    print(f"\n[t={t:.2f}s] Phase 6 Analysis:")
                    print(f"  Robot pos: {robot_pos}, heading: {robot_heading:.3f} rad")
                    print(f"  Desired object speed: {desired_speed:.4f} m/s")
                    print(f"  Actual object speed: {np.linalg.norm(object_velocity):.4f} m/s")
                    print(f"  Actual object velocity: {object_velocity} m/s")
                    if self.in_contact:
                        print(f"  Speed error: {speed_error:.4f} m/s")
                        print(f"  Speed error integral: {velocity_error_int:.4f} m/s*s")
                    print(f"  v_base (position feedback): {v_base:.4f} m/s")
                    print(f"  v_ff (feed-forward): {v_ff:.4f} m/s (static: {v_ff_static:.4f}, viscous: {v_ff_viscous:.4f})")
                    print(f"  v_pi (PI controlled): {v_pi:.4f} m/s")
                    print(f"  v_along (combined): {v_along:.4f} m/s")
                    print(f"  error_along: {error_along:.4f} m")
                    print(f"  Contact: {self.in_contact}, force: {contact_force:.2f} N, filtered: {contact_state_filtered:.3f}")
                    print(f"  Intended pos: {intended_pos}")
                    print(f"  Position error: {position_error} (mag: {np.linalg.norm(position_error)*100:.2f} cm)")
                    print(f"  Heading error: {heading_error:.3f} rad ({np.degrees(heading_error):.1f} deg)")
                    print(f"  Closest u on desired segment: {closest_u:.4f}")

            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1

            if gui:
                time.sleep(TIMESTEP * 0.3)

        return self._compute_phase_1_metrics()  # reuse metrics structure

    # ----------------------------------------------------------------------
    # PHASE 7: Feed-Forward + PI Control tracking Contact Point Speed
    # ----------------------------------------------------------------------
    def run_phase_7(self, desired_contact_point_speed: float, gui: bool = True, duration: float = 10.0) -> Dict:
        """Phase 7: Feed-Forward + PI Control tracking contact point speed (not object speed).

        This phase is similar to Phase 6, but tracks contact point speed instead of object speed.
        The contact point speed accounts for both object linear and angular velocities:
            v_cp = v_obj + omega × r_cp
            speed_cp = ||v_cp||

        This is more appropriate when the object rotates, as the contact point moves relative
        to the object center.

        Parameters
        ----------
        desired_contact_point_speed : float
            Desired contact point speed magnitude (m/s)
        gui : bool
            Show GUI
        duration : float
            Test duration (s)
        """
        print(f"\n{'='*60}")
        print(f"PHASE 7: Feed-Forward + PI Control (Contact Point Speed)")
        print(f"{'='*60}")
        print(f"  Desired t_param: {self.desired_t_param:.3f}")
        print(f"  Desired contact point speed: {desired_contact_point_speed:.3f} m/s")
        print(f"  Duration: {duration:.1f} s")

        # Feed-forward gains (plant compensation)
        K_static = 0.03  # Static friction compensation (m/s)
        K_alpha = 1.2    # Viscous friction coefficient

        # PI controller gains
        kp_vel = 0.9     # Proportional gain (reduced since FF does heavy lifting)
        ki_vel = 0.2     # Integral gain
        velocity_error_int = 0.0  # Integral accumulator
        velocity_error_int_max = 0.7  # Clamp to prevent windup

        # Control gain for base velocity from position error
        kp_approach = 2.0  # Gain for position-based approach velocity

        # Contact detection with hysteresis
        contact_threshold_on = 2.0  # Enter contact when force > this
        contact_threshold_off = 0.2  # Exit contact when force < this
        contact_state_filtered = 0.0  # Filtered contact state (0-1)
        contact_filter_alpha = 0.1   # Exponential moving average coefficient

        # Integral decay when not in contact
        integral_decay_rate = 0.95  # Decay per control step

        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        dt_ctrl = CTRL_STEP * TIMESTEP

        # Track previous contact state for hysteresis
        in_contact_prev = False

        for step in range(n_steps):
            if step_count % CTRL_STEP == 0:
                # Robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()

                # Object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]

                # Contact detection with filtering
                contact_force = self._get_contact_force()
                
                # Filter contact force (exponential moving average)
                contact_state_filtered = (1.0 - contact_filter_alpha) * contact_state_filtered + \
                                        contact_filter_alpha * (1.0 if contact_force > 0.5 else 0.0)
                
                # Hysteresis-based contact detection
                if in_contact_prev:
                    # Was in contact: use lower threshold to exit
                    self.in_contact = contact_force > contact_threshold_off
                else:
                    # Was not in contact: use higher threshold to enter
                    self.in_contact = contact_force > contact_threshold_on

                in_contact_prev = self.in_contact

                # Contact point & normals in world
                cos_t = np.cos(object_orientation)
                sin_t = np.sin(object_orientation)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                contact_point_world = R @ self.contact_point_body + object_pos
                normal_outward_world = R @ self.normal_outward
                normal_inward_world = -normal_outward_world

                # Closest point on desired segment
                closest_point, closest_u = closest_point_on_desired_segment(
                    robot_pos=robot_pos,
                    object_pos=object_pos,
                    object_orientation=object_orientation,
                    seg_p1_body=self.desired_seg_p1_body,
                    seg_p2_body=self.desired_seg_p2_body,
                    assert_on_segment=True,
                    tol=1e-4,
                )

                # Requirement 1: Intended position (no force-based offset here, just nominal)
                intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
                position_error = intended_pos - robot_pos

                # Requirement 2: Heading toward reference (closest point if sliding)
                reference_point = closest_point if self.in_contact else contact_point_world
                desired_heading = np.arctan2((reference_point - robot_pos)[1],
                                             (reference_point - robot_pos)[0])
                heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                          np.cos(desired_heading - robot_heading))

                # Contact point velocity calculation (like Phase 3)
                r_cp = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
                contact_point_velocity = object_velocity + v_rotation
                contact_point_speed = np.linalg.norm(contact_point_velocity)

                # Decompose position error into components:
                # - Along normal_inward: base velocity + feed-forward + PI control
                # - Perpendicular to normal_inward: position control
                normal_along = normal_inward_world
                normal_perp = np.array([-normal_along[1], normal_along[0]])  # Rotate 90 deg CCW
                
                # Project position error onto these directions
                error_along = np.dot(position_error, normal_along)
                error_perp = np.dot(position_error, normal_perp)
                
                # BASE VELOCITY: Always compute from position error (ensures approach/maintenance)
                v_base = kp_approach * error_along
                
                # FEED-FORWARD COMPONENT (Plant Compensation)
                # Static friction: constant offset to overcome static friction
                v_ff_static = K_static
                # Viscous friction: proportional to desired contact point speed
                v_ff_viscous = K_alpha * desired_contact_point_speed
                # Total feed-forward
                v_ff = v_ff_static + v_ff_viscous
                
                # PI CONTROLLER (Error Correction) - tracking contact point speed
                if self.in_contact:
                    # Speed error (desired contact point speed - actual contact point speed)
                    speed_error = desired_contact_point_speed - contact_point_speed
                    
                    # PI controller update
                    velocity_error_int += speed_error * dt_ctrl
                    
                    # Clamp integral to prevent windup
                    velocity_error_int = np.clip(velocity_error_int, -velocity_error_int_max, velocity_error_int_max)
                    
                    # Compute v_pi from PI controller
                    # Positive error (contact point too slow) → increase v_pi (push harder)
                    # Negative error (contact point too fast) → decrease v_pi (push softer)
                    v_pi = kp_vel * speed_error + ki_vel * velocity_error_int
                    
                    # Clamp PI output to reasonable limits
                    v_pi = np.clip(v_pi, -self.max_linear_speed, self.max_linear_speed)
                    
                    # COMBINED VELOCITY: Feed-forward + PI
                    # Feed-forward does the "heavy lifting" (overcomes friction),
                    # PI only needs to correct small errors
                    v_along = v_base + v_ff + v_pi
                    # v_along = v_base + desired_contact_point_speed + v_ff + v_pi
                else:
                    # When not in contact: use base velocity + feed-forward (for approach)
                    # Smoothly decay integral instead of instant reset
                    velocity_error_int *= integral_decay_rate
                    v_pi = 0.0
                    # Still use feed-forward for approach (helps overcome static friction)
                    v_along = v_base + v_ff * 0.5  # Reduced FF when not in contact
                
                # Clamp along-normal velocity
                v_along = np.clip(v_along, -self.max_linear_speed, self.max_linear_speed)
                
                # Perpendicular velocity: position control
                v_perp = kp_approach * error_perp
                v_perp = np.clip(v_perp, -self.max_linear_speed, self.max_linear_speed)
                
                # Combine velocities in world frame
                vel_cmd_xy = v_along * normal_along + v_perp * normal_perp
                
                # Clamp total speed
                speed = np.linalg.norm(vel_cmd_xy)
                if speed > self.max_linear_speed:
                    vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)

                omega = self.kp_heading * heading_error
                omega = np.clip(omega, -1.0, 1.0)

                cmd = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
                self.robot.command_velocity(cmd)

                # Record history (reuse Phase 5 structure)
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(desired_heading)
                self.history.heading_errors.append(heading_error)
                self.history.closest_points_on_desired_segment.append(closest_point.copy())
                self.history.closest_u_on_desired_segment.append(float(closest_u))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)
                # Phase 7 velocity components (reuse Phase 5 fields)
                # v_base, v_ff (as v_constant), v_pi (as v_velo_error_pi)
                self.history.v_base_history.append(v_base)
                self.history.v_constant_history.append(v_ff)  # Store feed-forward as "constant"
                self.history.v_velo_error_pi_history.append(v_pi)

                # Debug print occasionally
                if step_count % (CTRL_STEP * 10) == 0:
                    if self.in_contact:
                        speed_error = desired_contact_point_speed - contact_point_speed
                    else:
                        speed_error = 0.0
                    
                    print(f"\n[t={t:.2f}s] Phase 7 Analysis:")
                    print(f"  Robot pos: {robot_pos}, heading: {robot_heading:.3f} rad")
                    print(f"  Desired contact point speed: {desired_contact_point_speed:.4f} m/s")
                    print(f"  Desired contact point position: {contact_point_world}")
                    print(f"  Actual contact point speed: {contact_point_speed:.4f} m/s")
                    print(f"  Actual contact point velocity: {contact_point_velocity} m/s")
                    print(f"  Object velocity: {object_velocity} m/s, omega: {object_angular_velocity:.4f} rad/s")
                    if self.in_contact:
                        print(f"  Speed error: {speed_error:.4f} m/s")
                        print(f"  Speed error integral: {velocity_error_int:.4f} m/s*s")
                    print(f"  v_base (position feedback): {v_base:.4f} m/s")
                    print(f"  v_ff (feed-forward): {v_ff:.4f} m/s (static: {v_ff_static:.4f}, viscous: {v_ff_viscous:.4f})")
                    print(f"  v_pi (PI controlled): {v_pi:.4f} m/s")
                    print(f"  v_along (combined): {v_along:.4f} m/s")
                    print(f"  error_along: {error_along:.4f} m")
                    print(f"  Contact: {self.in_contact}, force: {contact_force:.2f} N, filtered: {contact_state_filtered:.3f}")
                    print(f"  Intended pos: {intended_pos}")
                    print(f"  Position error: {position_error} (mag: {np.linalg.norm(position_error)*100:.2f} cm)")
                    print(f"  Heading error: {heading_error:.3f} rad ({np.degrees(heading_error):.1f} deg)")
                    print(f"  Closest u on desired segment: {closest_u:.4f}")

            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1

            if gui:
                time.sleep(TIMESTEP * 0.3)

        return self._compute_phase_1_metrics()  # reuse metrics structure

    # ----------------------------------------------------------------------
    # PHASE 7 BETA: Simplified Tripartite Decoupled Control (Single Robot)
    # ----------------------------------------------------------------------
    def run_phase_7_beta(self, desired_object_velocity: np.ndarray, desired_object_angular_velocity: float, 
                          gui: bool = True, duration: float = 10.0) -> Dict:
        """Phase 7 Beta: Simplified Tripartite Decoupled Control for single robot pushing.
        
        This phase uses a simplified version of Phase7BetaVerDecouple adapted for single robot:
        - v_along = v_ff_along (feed-forward only, no corrections)
        - v_tangent = v_ff_tangent + K * error (feed-forward + proportional error correction)
        - omega = K * error (proportional heading error)
        
        Control Structure:
        ------------------
        1. Longitudinal Axis (v_along): Feed-forward only from desired contact point velocity
        2. Lateral Axis (v_tangent): Feed-forward + proportional position error correction
        3. Angular (ω): Proportional heading error control
        
        Parameters
        ----------
        desired_object_velocity : np.ndarray
            Desired object linear velocity (vx, vy) in world frame
        desired_object_angular_velocity : float
            Desired object angular velocity (rad/s)
        gui : bool
            Show GUI
        duration : float
            Test duration (s)
        """
        print(f"\n{'='*60}")
        print(f"PHASE 7 BETA: Simplified Tripartite Decoupled Control (Single Robot)")
        print(f"{'='*60}")
        print(f"  Desired t_param: {self.desired_t_param:.3f}")
        print(f"  Desired object velocity: {desired_object_velocity}")
        print(f"  Desired object angular velocity: {desired_object_angular_velocity:.3f} rad/s")
        print(f"  Duration: {duration:.1f} s")
        
        # Control gains
        kp_tangent = 1.5  # Proportional gain for tangent position error
        kp_heading = 10.0  # Proportional gain for heading error
        
        # Limits
        max_linear_speed = 0.5
        max_along_speed = 0.4
        max_tangent_speed = 0.3
        
        # Contact detection with hysteresis
        contact_threshold_on = 2.0
        contact_threshold_off = 0.2
        in_contact_prev = False
        
        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        
        for step in range(n_steps):
            if step_count % CTRL_STEP == 0:
                # Robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()
                
                # Object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]
                
                # Contact detection with hysteresis
                contact_force = self._get_contact_force()
                if in_contact_prev:
                    self.in_contact = contact_force > contact_threshold_off
                else:
                    self.in_contact = contact_force > contact_threshold_on
                in_contact_prev = self.in_contact
                
                # Contact point & frame vectors in world
                cos_t = np.cos(object_orientation)
                sin_t = np.sin(object_orientation)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                contact_point_world = R @ self.contact_point_body + object_pos
                normal_outward_world = R @ self.normal_outward
                normal_inward_world = -normal_outward_world
                tangent_world = R @ self.tangent
                
                # Intended position: contact point + robot_radius * normal_outward
                intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
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
                
                # ===== SIMPLIFIED TRIPARTITE DECOUPLED CONTROL =====
                
                # STEP 1: Compute desired contact point velocity from desired object motion
                # Transform desired object velocity to body frame
                R_T = R.T
                r_cp_body = self.contact_point_body
                v_obj_desired_body = R_T @ desired_object_velocity
                v_rotation_body = desired_object_angular_velocity * np.array([-r_cp_body[1], r_cp_body[0]])
                v_cp_desired_body = v_obj_desired_body + v_rotation_body
                v_cp_desired = R @ v_cp_desired_body
                
                # STEP 2: Project desired contact point velocity onto contact frame axes
                v_ff_along = np.dot(v_cp_desired, normal_inward_world)  # Feed-forward along normal
                v_ff_tangent = np.dot(v_cp_desired, tangent_world)       # Feed-forward along tangent
                
                # STEP 3: Decompose position error in contact frame
                error_along = np.dot(position_error, normal_inward_world)  # Error along normal
                error_tangent = np.dot(position_error, tangent_world)      # Error along tangent
                
                # STEP 4: Control Law (Simplified)
                # v_along = v_ff_along (feed-forward only, no corrections)
                v_along = v_ff_along + error_along * kp_tangent
                # v_along = v_ff_along
                v_along = np.clip(v_along, -max_along_speed, max_along_speed)
                
                # v_tangent = v_ff_tangent + K * error (feed-forward + proportional error)
                v_tangent = v_ff_tangent + kp_tangent * error_tangent
                v_tangent = np.clip(v_tangent, -max_tangent_speed, max_tangent_speed)
                
                # omega = K * error (proportional heading error)
                omega = kp_heading * heading_error
                omega = np.clip(omega, -1.0, 1.0)
                
                # STEP 5: Transform from contact frame to world frame
                vel_cmd_xy = v_along * normal_inward_world + v_tangent * tangent_world
                
                # Clamp total speed
                speed = np.linalg.norm(vel_cmd_xy)
                if speed > max_linear_speed:
                    vel_cmd_xy = vel_cmd_xy * (max_linear_speed / speed)
                
                cmd = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])
                self.robot.command_velocity(cmd)
                
                # Contact point velocity (for reference)
                r_cp = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
                contact_point_velocity = object_velocity + v_rotation
                
                # Record history (reuse Phase 5 structure)
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(desired_heading)
                self.history.heading_errors.append(heading_error)
                self.history.closest_points_on_desired_segment.append(contact_point_world.copy())
                self.history.closest_u_on_desired_segment.append(0.0)  # Not used in this phase
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)
                # Phase 7 Beta velocity components (reuse Phase 5 fields)
                self.history.v_base_history.append(v_along)  # Longitudinal velocity
                self.history.v_constant_history.append(v_ff_tangent)  # Feed-forward tangent
                self.history.v_velo_error_pi_history.append(kp_tangent * error_tangent)  # Tangent error correction
                
                # Debug print occasionally
                if step_count % (CTRL_STEP * 10) == 0:
                    print(f"\n[t={t:.2f}s] Phase 7 Beta Analysis:")
                    print(f"  Robot pos: {robot_pos}, heading: {robot_heading:.3f} rad")
                    print(f"  Desired object velocity: {desired_object_velocity}")
                    print(f"  Actual object velocity: {object_velocity}")
                    print(f"  v_ff_along: {v_ff_along:.4f} m/s, v_along: {v_along:.4f} m/s")
                    print(f"  v_ff_tangent: {v_ff_tangent:.4f} m/s, error_tangent: {error_tangent:.4f} m")
                    print(f"  v_tangent: {v_tangent:.4f} m/s")
                    print(f"  omega: {omega:.4f} rad/s (heading error: {heading_error:.3f} rad)")
                    print(f"  Contact: {self.in_contact}, force: {contact_force:.2f} N")
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.3)
        
        return self._compute_phase_1_metrics()  # reuse metrics structure

    # ----------------------------------------------------------------------
    # PHASE 3: Velocity-position hybrid control
    # ----------------------------------------------------------------------
    def run_phase_3(self, gui: bool = True, duration: float = 10.0) -> Dict:
        """Phase 3: Velocity-position hybrid control.

        Idea:
        - Phase 1 already keeps relative pose (position + heading) very good.
        - Here we additionally try to track the *instant* object velocity at first contact:
            1. When contact is first detected, snapshot:
                v_obj_desired = object_velocity
                omega_desired = object_angular_velocity
            2. At each step, compute desired contact point velocity from these:
                v_cp_desired = v_obj_desired + omega_desired × r_cp
            3. Command robot velocity:
                v_cmd_xy = v_cp_desired + Kp_pos * position_error
                omega_cmd from heading error (same as before)
        """
        print(f"\n{'='*60}")
        print(f"PHASE 3: Velocity-Position Hybrid Control")
        print(f"{'='*60}")
        print(f"  Desired t_param: {self.desired_t_param:.3f}")
        print(f"  Duration: {duration:.1f} s")

        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0

        # Reset Phase 3 desired velocities
        self.phase3_desired_object_velocity = None
        self.phase3_desired_object_angular_velocity = 0.0

        # Simple approach controller reused from Phase 1
        approach_speed = 0.15
        approach_kp = 1.0

        for step in range(n_steps):
            if step_count % CTRL_STEP == 0:
                # Robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()

                # Object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]

                # Contact detection
                contact_force = self._get_contact_force()
                in_contact = contact_force > self.contact_threshold

                # Contact point & normals in world
                cos_t = np.cos(object_orientation)
                sin_t = np.sin(object_orientation)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                contact_point_world = R @ self.contact_point_body + object_pos
                normal_outward_world = R @ self.normal_outward

                # Closest point on desired segment
                closest_point, closest_u = closest_point_on_desired_segment(
                    robot_pos=robot_pos,
                    object_pos=object_pos,
                    object_orientation=object_orientation,
                    seg_p1_body=self.desired_seg_p1_body,
                    seg_p2_body=self.desired_seg_p2_body,
                    assert_on_segment=True,
                    tol=1e-4,
                )

                # Snapshot desired object motion at FIRST contact
                if in_contact and self.phase3_desired_object_velocity is None:
                    self.phase3_desired_object_velocity = object_velocity.copy()
                    self.phase3_desired_object_angular_velocity = float(object_angular_velocity)

                    small_gain = 10
                    self.phase3_desired_object_velocity *= small_gain
                    self.phase3_desired_object_angular_velocity *= small_gain
                    print(f"\n[t={t:.2f}s] Phase 3: Contact detected, snapshot desired object motion:")
                    print(f"  v_obj_desired = {self.phase3_desired_object_velocity}")
                    print(f"  omega_desired = {self.phase3_desired_object_angular_velocity:.3f} rad/s")

                # Requirement 1: Intended position (no force scaling here)
                intended_pos = contact_point_world + ROBOT_RADIUS * normal_outward_world
                position_error = intended_pos - robot_pos

                # Requirement 2: Heading toward reference (closest point if sliding)
                reference_point = closest_point if in_contact else contact_point_world
                desired_heading = np.arctan2((reference_point - robot_pos)[1],
                                             (reference_point - robot_pos)[0])
                heading_error = np.arctan2(np.sin(desired_heading - robot_heading),
                                          np.cos(desired_heading - robot_heading))

                # Desired contact point velocity (only defined after snapshot)
                if self.phase3_desired_object_velocity is not None:
                    r_cp = contact_point_world - object_pos
                    v_rotation_des = self.phase3_desired_object_angular_velocity * np.array([-r_cp[1], r_cp[0]])
                    v_cp_desired = self.phase3_desired_object_velocity + v_rotation_des
                else:
                    # Before snapshot, just use zero desired CP velocity
                    v_cp_desired = np.zeros(2)

                # Contact point velocity for logging (actual)
                r_cp_actual = contact_point_world - object_pos
                v_rotation_actual = object_angular_velocity * np.array([-r_cp_actual[1], r_cp_actual[0]])
                contact_point_velocity_actual = object_velocity + v_rotation_actual

                # CONTROLLER:
                if not in_contact:
                    # Approach: same as Phase 1 but using intended_pos
                    to_intended = intended_pos - robot_pos
                    distance = np.linalg.norm(to_intended)
                    if distance > 0.01:
                        direction = to_intended / distance
                        speed = min(approach_kp * distance, approach_speed)
                        vel_2d = direction * speed
                        omega = self.kp_heading * heading_error
                        omega = np.clip(omega, -1.0, 1.0)
                        cmd = np.array([vel_2d[0], vel_2d[1], omega])
                    else:
                        cmd = np.zeros(3)
                else:
                    # Velocity-position hybrid:
                    kp_pos = 2.0
                    vel_cmd_xy = v_cp_desired + kp_pos * position_error
                    speed = np.linalg.norm(vel_cmd_xy)
                    if speed > self.max_linear_speed:
                        vel_cmd_xy = vel_cmd_xy * (self.max_linear_speed / speed)

                    omega = self.kp_heading * heading_error
                    omega = np.clip(omega, -1.0, 1.0)
                    cmd = np.array([vel_cmd_xy[0], vel_cmd_xy[1], omega])

                # Command velocity
                self.robot.command_velocity(cmd)

                # Record history (reuse Phase 1 structure)
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.intended_positions.append(intended_pos.copy())
                self.history.position_errors.append(position_error.copy())
                self.history.desired_headings.append(desired_heading)
                self.history.heading_errors.append(heading_error)
                self.history.closest_points_on_desired_segment.append(closest_point.copy())
                self.history.closest_u_on_desired_segment.append(float(closest_u))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity_actual.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(in_contact)

                # Debug print occasionally
                if step_count % (CTRL_STEP * 10) == 0:
                    print(f"\n[t={t:.2f}s] Phase 3 Analysis:")
                    print(f"  Robot pos: {robot_pos}, heading: {robot_heading:.3f} rad")
                    print(f"  Intended pos: {intended_pos}")
                    print(f"  Position error: {position_error} (mag: {np.linalg.norm(position_error)*100:.2f} cm)")
                    if self.phase3_desired_object_velocity is not None:
                        print(f"  v_cp_desired: {v_cp_desired}")
                        print(f"  v_cp_actual:  {contact_point_velocity_actual}")
                    print(f"  Heading error: {heading_error:.3f} rad ({np.degrees(heading_error):.1f} deg)")
                    print(f"  Closest u on desired segment: {closest_u:.4f}")
                    print(f"  Contact: {in_contact}, force: {contact_force:.2f} N")

            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1

            if gui:
                time.sleep(TIMESTEP * 0.3)

        return self._compute_phase_1_metrics()  # reuse metrics structure


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Contact Control Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Phase 1: Basic requirements
    python test_hybrid_contact_control.py --phase 1 --t-param 0.125
    
    # Phase 2: Force-position hybrid (open-loop offset)
    python test_hybrid_contact_control.py --phase 2 --t-param 0.125 --desired-force 5.0

    # Phase 2.5: Force-position hybrid with PI on force
    python test_hybrid_contact_control.py --phase 25 --t-param 0.125 --desired-force 20.0
    
    # Phase 3: Velocity-position hybrid
    python test_hybrid_contact_control.py --phase 3 --t-param 0.125
    
    # Phase 4: Force-position hybrid using Newton's law (F = m*a) - OPEN-LOOP
    python test_hybrid_contact_control.py --phase 4 --t-param 0.125 --desired-force 5.0
    
    # Phase 5: Closed-loop object velocity control - CLOSED-LOOP (single robot)
    python test_hybrid_contact_control.py --phase 5 --t-param 0.125 --desired-speed 0.1
    
    # Phase 6: Feed-Forward + PI Control with smooth contact transitions
    python test_hybrid_contact_control.py --phase 6 --t-param 0.125 --desired-speed 0.1
    
    # Phase 7: Feed-Forward + PI Control tracking contact point speed
    python test_hybrid_contact_control.py --phase 7 --t-param 0.125 --desired-speed 0.1
        """
    )
    parser.add_argument("--phase", type=str, default="1", 
                       choices=["1", "2", "25", "3", "4", "5", "6", "7", "7beta"],
                       help="Test phase (default: 1). Use '7beta' for Phase 7 Beta.")
    parser.add_argument("--kinematics", "-k", default="holonomic",
                       choices=['holonomic', 'diffdrive'],
                       help="Kinematics type (default: holonomic)")
    parser.add_argument("--t-param", type=float, default=0.125,
                       help="Desired t_param on object boundary (default: 0.125)")
    parser.add_argument("--approach-distance", type=float, default=0.5,
                       help="Initial distance from object (default: 0.5 m)")
    parser.add_argument("--desired-force", type=float, default=5.0,
                       help="Desired contact force for Phase 2/4 (default: 5.0 N)")
    parser.add_argument("--desired-speed", type=float, default=0.1,
                       help="Desired speed magnitude for Phase 5/6/7 (object speed for 5/6, contact point speed for 7) (default: 0.1 m/s)")
    parser.add_argument("--desired-object-velocity", type=str, default="0.03,0.05",
                       help="Desired object velocity (vx,vy) for Phase 7beta as 'vx,vy' (default: 0.03,0.05)")
    parser.add_argument("--desired-object-angular-velocity", type=float, default=0.2,
                       help="Desired object angular velocity (rad/s) for Phase 7beta (default: 0.2)")
    parser.add_argument("--object", type=str, default=None,
                       help="Object name from create_standard_objects() (e.g., 'rectangle')")
    parser.add_argument("--object-index", type=int, default=None,
                       help="Object index from create_pybullet_objects() (1-based, e.g., 3 for right_triangle, 6 for l_shape)")
    parser.add_argument("--obj-shape", type=str, default=None,
                       help="Shape name for OBJ mode (must exist in shape_data.json, e.g., 'right_triangle')")
    parser.add_argument("--obj-file", type=str, default=None,
                       help="OBJ file path (relative to urdf directory or absolute). If None, uses '{obj-shape}.obj'")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Test duration (default: 10.0 s)")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run headless")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save results")
    args = parser.parse_args()
    
    # Setup PyBullet
    print("\nInitializing PyBullet...")
    ground = setup_pybullet(gui=not args.no_gui)
    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
    
    # Create test
    test = HybridContactControlTest(
        kinematics=args.kinematics,
        t_param=args.t_param,
        approach_distance=args.approach_distance,
        object_name=args.object,
        object_index=args.object_index,
        obj_shape=args.obj_shape,
        obj_file=args.obj_file
    )
    
    # Convert phase string to int for backward compatibility
    phase_str = args.phase
    phase_int = int(phase_str) if phase_str.isdigit() else None
    
    if phase_str == "1" or phase_int == 1:
        # Phase 1: Basic requirements
        results = test.run_phase_1(gui=not args.no_gui, duration=args.duration)
        
        print("\n" + "="*60)
        print("PHASE 1 RESULTS")
        print("="*60)
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg heading error: {np.degrees(results['avg_heading_error']):.2f} deg")
        print("="*60)
        
        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_phase_1_results(save_path / "phase1_results.png")
        else:
            test.plot_phase_1_results()
    
    elif phase_str == "2" or phase_int == 2:
        results = test.run_phase_2(desired_force=args.desired_force, gui=not args.no_gui, duration=args.duration)

        print("\n" + "="*60)
        print("PHASE 2 RESULTS (Force-Position Hybrid)")
        print("="*60)
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg heading error: {np.degrees(results['avg_heading_error']):.2f} deg")
            print(f"  Avg segment projection error (u off [0,1]): {results['avg_segment_u_error']:.4f}")
        print("="*60)

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_phase_1_results(save_path / "phase2_results.png")  # reuse plot (shows same tracked signals)
        else:
            test.plot_phase_1_results()
    
    elif phase_str == "25":
        results = test.run_phase_25(desired_force=args.desired_force, gui=not args.no_gui, duration=args.duration)

        print("\n" + "="*60)
        print("PHASE 2.5 RESULTS (Force-Position Hybrid + PI)")
        print("="*60)
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg heading error: {np.degrees(results['avg_heading_error']):.2f} deg")
            print(f"  Avg segment projection error (u off [0,1]): {results['avg_segment_u_error']:.4f}")
        print("="*60)

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_phase_1_results(save_path / "phase2_5_results.png")  # reuse plot layout
        else:
            test.plot_phase_1_results()

    elif phase_str == "3" or phase_int == 3:
        results = test.run_phase_3(gui=not args.no_gui, duration=args.duration)

        print("\n" + "="*60)
        print("PHASE 3 RESULTS (Velocity-Position Hybrid)")
        print("="*60)
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg heading error: {np.degrees(results['avg_heading_error']):.2f} deg")
            print(f"  Avg segment projection error (u off [0,1]): {results['avg_segment_u_error']:.4f}")
        print("="*60)

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            # Geometry/pose diagnostics
            test.plot_phase_1_results(save_path / "phase3_results.png")
            # Additional velocity diagnostics
            test.plot_phase3_velocities(save_path / "phase3_velocities.png")
        else:
            test.plot_phase_1_results()
            test.plot_phase3_velocities()
    
    elif phase_str == "4" or phase_int == 4:
        results = test.run_phase_4(desired_force=args.desired_force, gui=not args.no_gui, duration=args.duration)

        print("\n" + "="*60)
        print("PHASE 4 RESULTS (Force-Position Hybrid using Newton's Law)")
        print("="*60)
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg heading error: {np.degrees(results['avg_heading_error']):.2f} deg")
            print(f"  Avg segment projection error (u off [0,1]): {results['avg_segment_u_error']:.4f}")
        print("="*60)

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_phase_1_results(save_path / "phase4_results.png")  # reuse plot layout
        else:
            test.plot_phase_1_results()
    
    elif phase_str == "5" or phase_int == 5:
        results = test.run_phase_5(
            desired_speed=args.desired_speed,
            gui=not args.no_gui,
            duration=args.duration
        )

        print("\n" + "="*60)
        print("PHASE 5 RESULTS (Closed-Loop Object Velocity Control)")
        print("="*60)
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg heading error: {np.degrees(results['avg_heading_error']):.2f} deg")
            print(f"  Avg segment projection error (u off [0,1]): {results['avg_segment_u_error']:.4f}")
        print("="*60)

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_phase_1_results(save_path / "phase5_results.png")  # reuse plot layout
            # Also plot Phase 5 specific velocity tracking
            test.plot_phase_5_velocities(args.desired_speed, save_path / "phase5_velocities.png")
        else:
            test.plot_phase_1_results()
            test.plot_phase_5_velocities(args.desired_speed)
    
    elif phase_str == "6" or phase_int == 6:
        results = test.run_phase_6(
            desired_speed=args.desired_speed,
            gui=not args.no_gui,
            duration=args.duration
        )

        print("\n" + "="*60)
        print("PHASE 6 RESULTS (Feed-Forward + PI Control)")
        print("="*60)
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg heading error: {np.degrees(results['avg_heading_error']):.2f} deg")
            print(f"  Avg segment projection error (u off [0,1]): {results['avg_segment_u_error']:.4f}")
        print("="*60)

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_phase_1_results(save_path / "phase6_results.png")  # reuse plot layout
            # Use Phase 5 velocity plot (shows v_base, v_ff, v_pi) with phase=6 for correct labels
            test.plot_phase_5_velocities(args.desired_speed, save_path / "phase6_velocities.png", phase=6)
        else:
            test.plot_phase_1_results()
            test.plot_phase_5_velocities(args.desired_speed, phase=6)
    
    elif phase_str == "7" or phase_int == 7:
        results = test.run_phase_7(
            desired_contact_point_speed=args.desired_speed,
            gui=not args.no_gui,
            duration=args.duration
        )

        print("\n" + "="*60)
        print("PHASE 7 RESULTS (Feed-Forward + PI Control - Contact Point Speed)")
        print("="*60)
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg heading error: {np.degrees(results['avg_heading_error']):.2f} deg")
            print(f"  Avg segment projection error (u off [0,1]): {results['avg_segment_u_error']:.4f}")
        print("="*60)

        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_phase_1_results(save_path / "phase7_results.png")  # reuse plot layout
            # Use Phase 5 velocity plot (shows v_base, v_ff, v_pi) with phase=7 for correct labels
            test.plot_phase_5_velocities(args.desired_speed, save_path / "phase7_velocities.png", phase=7)
        else:
            test.plot_phase_1_results()
            test.plot_phase_5_velocities(args.desired_speed, phase=7)
    
    elif phase_str == "7beta":
        # Parse desired object velocity
        try:
            vel_parts = [float(x.strip()) for x in args.desired_object_velocity.split(',')]
            if len(vel_parts) != 2:
                raise ValueError("Desired object velocity must have 2 components (vx, vy)")
            desired_obj_velocity = np.array(vel_parts)
        except Exception as e:
            print(f"Error parsing desired object velocity: {e}")
            print("Using default: [0.03, 0.05]")
            desired_obj_velocity = np.array([0.03, 0.05])
        
        results = test.run_phase_7_beta(
            desired_object_velocity=desired_obj_velocity,
            desired_object_angular_velocity=args.desired_object_angular_velocity,
            gui=not args.no_gui,
            duration=args.duration
        )
        
        print("\n" + "="*60)
        print("PHASE 7 BETA RESULTS (Simplified Tripartite Decoupled Control)")
        print("="*60)
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg heading error: {np.degrees(results['avg_heading_error']):.2f} deg")
            print(f"  Avg segment projection error (u off [0,1]): {results['avg_segment_u_error']:.4f}")
        print("="*60)
        
        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_phase_1_results(save_path / "phase7beta_results.png")
            # Use Phase 5 velocity plot with phase=7 for correct labels
            test.plot_phase_5_velocities(
                np.linalg.norm(desired_obj_velocity), 
                save_path / "phase7beta_velocities.png", 
                phase=7
            )
        else:
            test.plot_phase_1_results()
            test.plot_phase_5_velocities(np.linalg.norm(desired_obj_velocity), phase=7)
    
    if not args.no_gui:
        print("\nPress Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
