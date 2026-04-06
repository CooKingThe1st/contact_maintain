#!/usr/bin/env python3
"""
Single Pusher Test for Contact Maintenance and Stability

Tests a single robot pushing an object loaded from OBJ mesh files.
Focuses on:
1. Contact maintenance: Robot maintains contact at a specific boundary point
2. Stability testing: Object stability under single pusher (especially for non-convex shapes)

This test is designed to work with:
- Convex objects (rectangles, circles, etc.)
- Non-convex objects (L-shapes, T-shapes, etc.) - tests stability

Usage:
    # Test with Body1.obj (non-convex):
    python test_single_pusher.py --mesh meshes/Body1.obj --shape-name body1
    
    # Test with custom t_param and approach distance:
    python test_single_pusher.py --mesh meshes/Body1.obj --shape-name body1 --t-param 0.25 --approach-distance 0.6
    
    # Test stability only (no contact maintenance):
    python test_single_pusher.py --mesh meshes/Body1.obj --shape-name body1 --test-mode stability
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

from object_utils import ContactPointParameterization
from contact_maintain.robot_factory import create_robot, is_wheel_robot
from contact_maintain.object_bridge import obj_to_generic, generic_to_pybullet
from contact_maintain.contact_maintain_controller import InstantVelocityMatcher
from contact_maintain.pyb_simulation import get_contact_force


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

DEFAULT_OBJECT_HEIGHT = 0.2
DEFAULT_OBJECT_FRICTION = 0.8
DEFAULT_OBJECT_MASS = 2.0  # Match reference file
APPROACH_DISTANCE = 0.4  # Match reference file


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SinglePusherHistory:
    """History for single pusher test."""
    times: List[float] = field(default_factory=list)
    robot_positions: List[np.ndarray] = field(default_factory=list)
    robot_headings: List[float] = field(default_factory=list)
    robot_velocities: List[np.ndarray] = field(default_factory=list)
    robot_cmd_velocities: List[np.ndarray] = field(default_factory=list)
    object_positions: List[np.ndarray] = field(default_factory=list)
    object_orientations: List[float] = field(default_factory=list)
    object_velocities: List[np.ndarray] = field(default_factory=list)
    object_angular_velocities: List[float] = field(default_factory=list)
    contact_point_positions: List[np.ndarray] = field(default_factory=list)
    contact_point_velocities: List[np.ndarray] = field(default_factory=list)
    contact_forces: List[float] = field(default_factory=list)
    in_contact: List[bool] = field(default_factory=list)
    phase: List[str] = field(default_factory=list)  # 'approach', 'maintain', 'stability'
    # Stability metrics
    object_linear_velocity_magnitude: List[float] = field(default_factory=list)
    object_angular_velocity_magnitude: List[float] = field(default_factory=list)
    position_tracking_error: List[float] = field(default_factory=list)
    velocity_tracking_error: List[float] = field(default_factory=list)


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
    
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
    
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
# SINGLE PUSHER TEST
# ============================================================================

class SinglePusherTest:
    """Test single robot pushing an object loaded from mesh file.
    
    Tests:
    1. Contact Maintenance: Robot maintains contact at a specific boundary point
    2. Stability: Object stability under single pusher (especially for non-convex shapes)
    """
    
    def __init__(self, 
                 mesh_path: str,
                 shape_name: str,
                 t_param: float = 0.125,
                 approach_distance: float = 0.5,
                 kinematics: str = 'diffdrive',
                 test_mode: str = 'contact_maintenance',
                 object_mass: float = DEFAULT_OBJECT_MASS,
                 object_friction: float = DEFAULT_OBJECT_FRICTION):
        """
        Parameters
        ----------
        mesh_path : str
            Path to OBJ file, relative to urdf directory or absolute
        shape_name : str
            Name of the shape (used for object creation)
        t_param : float
            Target t_param on object boundary to track (0-1)
        approach_distance : float
            Distance from object to spawn robot (along normal outward)
        kinematics : str
            'holonomic' or 'diffdrive' (default: 'diffdrive')
        test_mode : str
            'contact_maintenance': Test contact maintenance
            'stability': Test object stability only
        object_mass : float
            Mass of the object (default: 1.0 kg)
        object_friction : float
            Friction coefficient (default: 0.8)
        """
        self.shape_name = shape_name
        self.t_param = t_param
        self.approach_distance = approach_distance
        self.kinematics = kinematics
        self.test_mode = test_mode
        self.object_mass = object_mass
        self.object_friction = object_friction
        
        print(f"\nLoading object from OBJ file: {mesh_path}")
        print(f"  Shape name: {shape_name}")
        print(f"  t_param: {t_param:.3f}")
        print(f"  Test mode: {test_mode}")
        
        # Only support OBJ files (STL support removed)
        if not mesh_path.endswith('.obj'):
            raise ValueError(
                f"Only OBJ files are supported. Please convert STL to OBJ format. "
                f"File: {mesh_path}"
            )
        
        # Load object from OBJ file (match reference: position z=0.6, mass=2.0)
        # obj_to_generic handles path resolution (checks urdf directory, etc.)
        self.generic_object, self.object_uid_pybullet = obj_to_generic(
            obj_path=mesh_path,
            shape_name=shape_name,
            position=(0.0, 0.0, 0.6),  # Match reference: object at z=0.6
            orientation=0.0,
            mass=object_mass,
            lateral_friction=object_friction,
            blind_test=True
        )
        print(f"✓ Loaded OBJ object: {shape_name}")
        print(f"  Mass: {self.generic_object.mass:.3f} kg")
        print(f"  Moment of inertia: {self.generic_object.moment_of_inertia:.6f} kg·m²")
        print(f"  Lateral friction: {self.generic_object.lateral_friction:.3f}")
        
        # Create parameterization for contact point calculations
        self.parameterization = ContactPointParameterization(self.generic_object)
        
        # Get contact point info at t_param
        contact_info = self.parameterization.get_contact_info(t_param)
        self.contact_point_body = np.array(contact_info['point'], dtype=float)
        self.normal_outward = np.array(contact_info['normal_outward'], dtype=float)
        self.normal_inward = -self.normal_outward
        
        # Calculate robot spawn position (match reference file)
        # Object is at (0, 0, 0.6) but for 2D spawn calculation, we use body frame (x,y at origin)
        # Spawn robot at: contact_point + approach_distance * normal_outward
        spawn_position_body = self.contact_point_body + self.approach_distance * self.normal_outward
        robot_x = float(spawn_position_body[0])
        robot_y = float(spawn_position_body[1])
        
        # Robot heading: point toward contact point (normal_inward direction)
        robot_heading = float(np.arctan2(self.normal_inward[1], self.normal_inward[0]))
        
        print(f"  Contact point (body frame): {self.contact_point_body}")
        print(f"  Normal outward: {self.normal_outward}")
        print(f"  Spawn position (body frame): ({robot_x:.3f}, {robot_y:.3f})")
        
        # Create robot at spawn position (match reference: z defaults to 0)
        print(f"\nCreating {kinematics} wheel robot...")
        self.robot = create_robot(
            kinematics=kinematics,
            model='wheel',  # Always use wheel model for pushing
            position=(robot_x, robot_y),  # z defaults to 0 in create_robot
            orientation=robot_heading,
            name="single_pusher"
        )
        print(f"Spawned robot at ({robot_x:.3f}, {robot_y:.3f}) with heading {robot_heading:.3f} rad, "
              f"target t_param={t_param:.4f}")
        
        # Contact controller (will be initialized when contact is detected)
        self.contact_controller = None
        self.in_contact = False
        self.contact_threshold = 0.5  # N
        
        # Desired velocities for contact maintenance (captured at first contact)
        self.desired_object_velocity_at_contact = None
        self.desired_object_angular_velocity_at_contact = None
        
        # History
        self.history = SinglePusherHistory()
    
    def run_test(self, gui: bool = True, duration: float = 15.0) -> Dict:
        """Run single pusher test.
        
        Parameters
        ----------
        gui : bool
            Show PyBullet GUI
        duration : float
            Test duration in seconds
        
        Returns
        -------
        dict
            Test results with metrics
        """
        print(f"\nRunning single pusher test...")
        print(f"  Duration: {duration:.1f} s")
        print(f"  Test mode: {self.test_mode}")
        
        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        
        # Approach phase parameters
        approach_speed = 0.15  # m/s
        approach_kp = 1.0  # P gain for approach
        
        for step in range(n_steps):
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Get robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()
                
                # Get object state
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid_pybullet)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid_pybullet)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]
                
                # Check contact
                contact_force = self._get_contact_force()
                was_in_contact = self.in_contact
                self.in_contact = contact_force > self.contact_threshold
                
                # Compute velocity command based on test mode
                if self.test_mode == 'stability':
                    # Stability test: Just push forward, no contact maintenance
                    cmd = self._compute_stability_command(
                        robot_pos, robot_heading, object_pos, object_orientation
                    )
                    phase = 'stability'
                else:
                    # Contact maintenance test
                    # Initialize contact controller on first contact
                    if self.in_contact and not was_in_contact:
                        # Capture object velocities at the moment of first contact
                        self.desired_object_velocity_at_contact = object_velocity.copy() * 10
                        self.desired_object_angular_velocity_at_contact = object_angular_velocity * 10
                        
                        print(f"\n[t={t:.2f}s] Contact detected! Initializing contact controller...")
                        print(f"  Captured object velocity: {self.desired_object_velocity_at_contact}")
                        print(f"  Captured object angular velocity: {self.desired_object_angular_velocity_at_contact:.3f} rad/s")
                        
                        self.contact_controller = InstantVelocityMatcher(
                            self.generic_object, self.t_param
                        )
                        # Set desired velocities to match object's velocity at contact moment
                        self.contact_controller.set_mode(
                            mode='drive_desired',
                            desired_object_velocity=self.desired_object_velocity_at_contact,
                            desired_object_angular_velocity=self.desired_object_angular_velocity_at_contact
                        )
                        print(f"  Controller set to drive_desired mode with captured velocities")
                    
                    # Compute velocity command
                    if self.in_contact and self.contact_controller is not None:
                        # MAINTAIN CONTACT phase: Use InstantVelocityMatcher
                        cmd = self.contact_controller.compute_robot_velocity(
                            robot_pos, robot_heading,
                            object_pos, object_orientation,
                            object_velocity, object_angular_velocity
                        )
                        phase = 'maintain'
                    else:
                        # APPROACH phase: Drive toward contact point along normal inward
                        cmd = self._compute_approach_command(
                            robot_pos, robot_heading, object_pos, object_orientation,
                            approach_speed, approach_kp
                        )
                        phase = 'approach'
                
                # Command velocity
                self.robot.command_velocity(cmd)
                
                # Calculate contact point position and velocity for history
                cos_t = np.cos(object_orientation)
                sin_t = np.sin(object_orientation)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                contact_point_world = R @ self.contact_point_body + object_pos
                
                # Contact point velocity (from rigid body kinematics)
                r = contact_point_world - object_pos
                v_rotation = object_angular_velocity * np.array([-r[1], r[0]])
                contact_point_velocity = object_velocity + v_rotation
                
                # Calculate tracking errors
                position_error = np.linalg.norm(robot_pos - contact_point_world)
                velocity_error = np.linalg.norm(robot_vel[:2] - contact_point_velocity) if len(robot_vel) >= 2 else 0.0
                
                # Record history
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy() if isinstance(cmd, np.ndarray) else np.array(cmd))
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)
                self.history.phase.append(phase)
                self.history.object_linear_velocity_magnitude.append(np.linalg.norm(object_velocity))
                self.history.object_angular_velocity_magnitude.append(abs(object_angular_velocity))
                self.history.position_tracking_error.append(position_error)
                self.history.velocity_tracking_error.append(velocity_error)
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.3)
        
        return self._compute_metrics()
    
    def _compute_approach_command(self, robot_pos, robot_heading, object_pos, object_orientation,
                                   approach_speed, approach_kp):
        """Compute approach command to drive toward contact point."""
        # Get current contact point position in world frame
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        contact_point_world = R @ self.contact_point_body + object_pos
        
        # Direction from robot to contact point
        to_contact = contact_point_world - robot_pos
        distance_to_contact = np.linalg.norm(to_contact)
        
        if distance_to_contact > 0.01:
            # Drive toward contact point
            direction = to_contact / distance_to_contact
            speed = min(approach_kp * distance_to_contact, approach_speed)
            vel_2d = direction * speed
            
            # Heading control: point toward contact point
            target_heading = np.arctan2(to_contact[1], to_contact[0])
            heading_error = np.arctan2(np.sin(target_heading - robot_heading),
                                      np.cos(target_heading - robot_heading))
            omega = 2.0 * heading_error
            omega = np.clip(omega, -1.0, 1.0)
            
            # Use 3-element command: [vx, vy, omega]
            cmd = np.array([vel_2d[0], vel_2d[1], omega])
        else:
            # Very close, stop
            cmd = np.zeros(3)
        
        return cmd
    
    def _compute_stability_command(self, robot_pos, robot_heading, object_pos, object_orientation):
        """Compute stability test command: simple forward push."""
        # Get contact point position
        cos_t = np.cos(object_orientation)
        sin_t = np.sin(object_orientation)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        contact_point_world = R @ self.contact_point_body + object_pos
        
        # Drive toward contact point with constant speed
        to_contact = contact_point_world - robot_pos
        distance_to_contact = np.linalg.norm(to_contact)
        
        if distance_to_contact > 0.05:
            direction = to_contact / distance_to_contact
            speed = 0.2  # Constant push speed
            vel_2d = direction * speed
            
            # Heading control
            target_heading = np.arctan2(to_contact[1], to_contact[0])
            heading_error = np.arctan2(np.sin(target_heading - robot_heading),
                                      np.cos(target_heading - robot_heading))
            omega = 2.0 * heading_error
            omega = np.clip(omega, -1.0, 1.0)
            
            cmd = np.array([vel_2d[0], vel_2d[1], omega])
        else:
            # Maintain contact with forward push
            # Push along normal inward direction
            normal_inward_world = R @ self.normal_inward
            push_speed = 0.15
            vel_2d = normal_inward_world * push_speed
            omega = 0.0  # No rotation during stability test
            cmd = np.array([vel_2d[0], vel_2d[1], omega])
        
        return cmd
    
    def _get_contact_force(self) -> float:
        """Get contact force magnitude."""
        try:
            # Find bumper/contact link index
            link_idx = -1
            if hasattr(self.robot, 'bumper_link_idx') and self.robot.bumper_link_idx is not None:
                link_idx = self.robot.bumper_link_idx
            elif hasattr(self.robot, 'contact_link_idx') and self.robot.contact_link_idx is not None:
                link_idx = self.robot.contact_link_idx
            
            force = get_contact_force(
                self.robot.uid, self.object_uid_pybullet,
                linkIndexA=link_idx,
                max_contacts=4
            )
            return np.linalg.norm(force[:2])
        except:
            return 0.0
    
    def _compute_metrics(self) -> Dict:
        """Compute test metrics."""
        times = np.array(self.history.times)
        in_contact = np.array(self.history.in_contact)
        contact_forces = np.array(self.history.contact_forces)
        
        # Find contact phase
        contact_start_idx = None
        for i, contact in enumerate(in_contact):
            if contact:
                contact_start_idx = i
                break
        
        metrics = {
            'contact_achieved': contact_start_idx is not None,
            'contact_time': float(times[contact_start_idx]) if contact_start_idx is not None else None,
            'avg_contact_force': float(np.mean(contact_forces)) if len(contact_forces) > 0 else 0.0,
            'max_contact_force': float(np.max(contact_forces)) if len(contact_forces) > 0 else 0.0,
        }
        
        if contact_start_idx is not None:
            contact_times = times[contact_start_idx:]
            contact_force_values = contact_forces[contact_start_idx:]
            
            # Position tracking error (robot to contact point)
            position_errors = np.array(self.history.position_tracking_error[contact_start_idx:])
            velocity_errors = np.array(self.history.velocity_tracking_error[contact_start_idx:])
            
            # Object motion metrics (stability)
            object_vel_mags = np.array(self.history.object_linear_velocity_magnitude[contact_start_idx:])
            object_omega_mags = np.array(self.history.object_angular_velocity_magnitude[contact_start_idx:])
            
            metrics.update({
                'contact_duration': float(times[-1] - times[contact_start_idx]),
                'avg_position_error': float(np.mean(position_errors)),
                'max_position_error': float(np.max(position_errors)),
                'avg_velocity_error': float(np.mean(velocity_errors)),
                'max_velocity_error': float(np.max(velocity_errors)),
                # Stability metrics
                'avg_object_velocity': float(np.mean(object_vel_mags)),
                'max_object_velocity': float(np.max(object_vel_mags)),
                'avg_object_angular_velocity': float(np.mean(object_omega_mags)),
                'max_object_angular_velocity': float(np.max(object_omega_mags)),
                'object_displacement': float(np.linalg.norm(
                    self.history.object_positions[-1] - self.history.object_positions[contact_start_idx]
                )),
                'object_rotation': float(abs(
                    self.history.object_orientations[-1] - self.history.object_orientations[contact_start_idx]
                )),
            })
        else:
            metrics.update({
                'contact_duration': 0.0,
                'avg_position_error': 0.0,
                'max_position_error': 0.0,
                'avg_velocity_error': 0.0,
                'max_velocity_error': 0.0,
                'avg_object_velocity': 0.0,
                'max_object_velocity': 0.0,
                'avg_object_angular_velocity': 0.0,
                'max_object_angular_velocity': 0.0,
                'object_displacement': 0.0,
                'object_rotation': 0.0,
            })
        
        return metrics
    
    def plot_results(self, save_path: Optional[Path] = None):
        """Plot test results."""
        times = np.array(self.history.times)
        robot_positions = np.array(self.history.robot_positions)
        object_positions = np.array(self.history.object_positions)
        contact_positions = np.array(self.history.contact_point_positions)
        in_contact = np.array(self.history.in_contact)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Single Pusher Test: {self.shape_name}, t_param={self.t_param:.3f}, mode={self.test_mode}',
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Trajectory overview
        ax = axes[0, 0]
        ax.plot(object_positions[:, 0], object_positions[:, 1], 'g-', linewidth=2, label='Object trajectory')
        ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', linewidth=1.5, label='Robot trajectory')
        ax.plot(contact_positions[:, 0], contact_positions[:, 1], 'r--', linewidth=1, alpha=0.7, label='Contact point')
        
        # Mark contact start
        contact_start_idx = np.argmax(in_contact) if np.any(in_contact) else None
        if contact_start_idx is not None:
            ax.plot(robot_positions[contact_start_idx, 0], robot_positions[contact_start_idx, 1], 
                   'go', markersize=10, label='Contact start')
        
        ax.plot(robot_positions[0, 0], robot_positions[0, 1], 'bs', markersize=8, label='Start')
        ax.plot(robot_positions[-1, 0], robot_positions[-1, 1], 'rs', markersize=8, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot 2: Contact force
        ax = axes[0, 1]
        contact_forces = np.array(self.history.contact_forces)
        ax.plot(times, contact_forces, 'r-', linewidth=1.5)
        ax.axhline(y=self.contact_threshold, color='g', linestyle='--', label='Contact threshold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact Force (N)')
        ax.set_title('Contact Force')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Position error
        ax = axes[0, 2]
        position_errors = np.array(self.history.position_tracking_error)
        ax.plot(times, position_errors * 100, 'b-', linewidth=1.5)  # Convert to cm
        if contact_start_idx is not None:
            ax.axvline(x=times[contact_start_idx], color='g', linestyle='--', label='Contact start')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (cm)')
        ax.set_title('Robot to Contact Point Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Velocity tracking
        ax = axes[1, 0]
        robot_vels = np.array(self.history.robot_velocities)
        contact_vels = np.array(self.history.contact_point_velocities)
        robot_vel_mags = np.linalg.norm(robot_vels[:, :2], axis=1)
        contact_vel_mags = np.linalg.norm(contact_vels, axis=1)
        ax.plot(times, robot_vel_mags, 'b-', linewidth=1.5, label='Robot velocity')
        ax.plot(times, contact_vel_mags, 'r--', linewidth=1.5, label='Contact point velocity')
        if contact_start_idx is not None:
            ax.axvline(x=times[contact_start_idx], color='g', linestyle='--', label='Contact start')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Magnitude (m/s)')
        ax.set_title('Velocity Tracking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Object motion (linear velocity)
        ax = axes[1, 1]
        object_vels = np.array(self.history.object_velocities)
        object_vel_mags = np.linalg.norm(object_vels, axis=1)
        ax.plot(times, object_vel_mags, 'g-', linewidth=1.5, label='Object linear velocity')
        if contact_start_idx is not None:
            ax.axvline(x=times[contact_start_idx], color='r', linestyle='--', label='Contact start')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Object Linear Velocity (Stability)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Object motion (angular velocity)
        ax = axes[1, 2]
        object_angular_vels = np.array(self.history.object_angular_velocities)
        ax.plot(times, np.abs(object_angular_vels), 'g--', linewidth=1.5, label='Object angular velocity (abs)')
        if contact_start_idx is not None:
            ax.axvline(x=times[contact_start_idx], color='r', linestyle='--', label='Contact start')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Object Angular Velocity (Stability)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 7: Object orientation
        ax = axes[2, 0]
        object_orientations = np.array(self.history.object_orientations)
        ax.plot(times, np.degrees(object_orientations), 'g-', linewidth=1.5)
        if contact_start_idx is not None:
            ax.axvline(x=times[contact_start_idx], color='r', linestyle='--', label='Contact start')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Orientation (deg)')
        ax.set_title('Object Orientation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 8: Velocity error
        ax = axes[2, 1]
        velocity_errors = np.array(self.history.velocity_tracking_error)
        ax.plot(times, velocity_errors * 100, 'orange', linewidth=1.5)  # Convert to cm/s
        if contact_start_idx is not None:
            ax.axvline(x=times[contact_start_idx], color='g', linestyle='--', label='Contact start')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Error (cm/s)')
        ax.set_title('Velocity Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 9: Metrics text
        ax = axes[2, 2]
        ax.axis('off')
        metrics = self._compute_metrics()
        if metrics['contact_achieved']:
            text = f"""TEST METRICS

Contact Maintenance:
  Contact Achieved: Yes
  Contact Time: {metrics['contact_time']:.2f} s
  Contact Duration: {metrics['contact_duration']:.2f} s
  Avg Contact Force: {metrics['avg_contact_force']:.2f} N
  Max Contact Force: {metrics['max_contact_force']:.2f} N

Tracking Performance:
  Avg Position Error: {metrics['avg_position_error']*100:.2f} cm
  Max Position Error: {metrics['max_position_error']*100:.2f} cm
  Avg Velocity Error: {metrics['avg_velocity_error']*100:.2f} cm/s
  Max Velocity Error: {metrics['max_velocity_error']*100:.2f} cm/s

Stability Metrics:
  Object Displacement: {metrics['object_displacement']*100:.2f} cm
  Object Rotation: {np.degrees(metrics['object_rotation']):.2f} deg
  Avg Object Velocity: {metrics['avg_object_velocity']:.3f} m/s
  Max Object Velocity: {metrics['max_object_velocity']:.3f} m/s
  Avg Object Angular Vel: {metrics['avg_object_angular_velocity']:.3f} rad/s
  Max Object Angular Vel: {metrics['max_object_angular_velocity']:.3f} rad/s
"""
        else:
            text = f"""TEST METRICS

Contact Achieved: No
Robot did not make contact
with the object.

Check:
- Approach distance
- Robot spawn position
- Object geometry
"""
        ax.text(0.1, 0.9, text, fontsize=10, family='monospace',
               verticalalignment='top', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single Pusher Test for Contact Maintenance and Stability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with default shape (right_triangle):
    python test_single_pusher.py --obj-shape right_triangle
    
    # Test with custom shape:
    python test_single_pusher.py --obj-shape body1
    
    # Test with custom OBJ file path:
    python test_single_pusher.py --obj-shape body1 --obj-file meshes/Body1.obj
    
    # Test with custom t_param and approach distance:
    python test_single_pusher.py --obj-shape body1 --t-param 0.25 --approach-distance 0.6
    
    # Test stability only (no contact maintenance):
    python test_single_pusher.py --obj-shape body1 --test-mode stability
    
    # Save results:
    python test_single_pusher.py --obj-shape hourglass --kinematics holonomic --approach-distance 1 --duration 20 --t-param 0.1
# python basic_test/test_single_pusher.py  --kinematics holonomic --approach-distance 0.2 --duration 20 --t-param 0.3 --obj-shape root
        """
    )
    parser.add_argument(
        "--obj-shape",
        type=str,
        default="right_triangle",
        help="Shape name (right_triangle,rect, bolt, hourglass, pi, root; default: right_triangle)",
    )
    parser.add_argument("--obj-file", type=str, default=None,
                       help="OBJ file path (relative to urdf directory). If None, uses '{obj-shape}.obj'")
    parser.add_argument("--t-param", type=float, default=0.125,
                       help="Target t_param on object boundary to track (0-1, default: 0.125)")
    parser.add_argument("--approach-distance", type=float, default=APPROACH_DISTANCE,
                       help=f"Distance from object to spawn robot (default: {APPROACH_DISTANCE} m)")
    parser.add_argument("--kinematics", "-k", default="diffdrive",
                       choices=['holonomic', 'diffdrive'],
                       help="Kinematics type (default: diffdrive)")
    parser.add_argument("--test-mode", default="contact_maintenance",
                       choices=['contact_maintenance', 'stability'],
                       help="Test mode (default: contact_maintenance)")
    parser.add_argument("--object-mass", type=float, default=DEFAULT_OBJECT_MASS,
                       help=f"Object mass in kg (default: {DEFAULT_OBJECT_MASS}, reference uses 2.0)")
    parser.add_argument("--object-friction", type=float, default=DEFAULT_OBJECT_FRICTION,
                       help=f"Object friction coefficient (default: {DEFAULT_OBJECT_FRICTION})")
    parser.add_argument("--duration", type=float, default=15.0,
                       help="Test duration in seconds (default: 15.0)")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run headless")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save results")
    args = parser.parse_args()
    
    # Determine OBJ file path (match reference pattern)
    if args.obj_file is None:
        obj_file = f"{args.obj_shape}.obj"
    else:
        obj_file = args.obj_file
    
    # Setup PyBullet
    print("\nInitializing PyBullet...")
    ground = setup_pybullet(gui=not args.no_gui)
    
    # Create and run test
    test = SinglePusherTest(
        mesh_path=obj_file,
        shape_name=args.obj_shape,
        t_param=args.t_param,
        approach_distance=args.approach_distance,
        kinematics=args.kinematics,
        test_mode=args.test_mode,
        object_mass=args.object_mass,
        object_friction=args.object_friction
    )
    
    results = test.run_test(gui=not args.no_gui, duration=args.duration)
    
    # Print results
    print("\n" + "="*60)
    print("SINGLE PUSHER TEST RESULTS")
    print("="*60)
    print(f"  OBJ file: {obj_file}")
    print(f"  Shape: {args.obj_shape}")
    print(f"  t_param: {args.t_param:.3f}")
    print(f"  Test mode: {args.test_mode}")
    print(f"  Contact achieved: {results['contact_achieved']}")
    if results['contact_achieved']:
        print(f"  Contact time: {results['contact_time']:.2f} s")
        print(f"  Contact duration: {results['contact_duration']:.2f} s")
        print(f"  Avg contact force: {results['avg_contact_force']:.2f} N")
        print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
        print(f"  Avg velocity error: {results['avg_velocity_error']*100:.2f} cm/s")
        print(f"\n  Stability Metrics:")
        print(f"    Object displacement: {results['object_displacement']*100:.2f} cm")
        print(f"    Object rotation: {np.degrees(results['object_rotation']):.2f} deg")
        print(f"    Avg object velocity: {results['avg_object_velocity']:.3f} m/s")
        print(f"    Avg object angular velocity: {results['avg_object_angular_velocity']:.3f} rad/s")
    print("="*60)
    
    # Save results
    if args.save_dir:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        test.plot_results(save_path / "single_pusher_test.png")
    else:
        test.plot_results()
    
    if not args.no_gui:
        print("\nPress Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
