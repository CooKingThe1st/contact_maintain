#!/usr/bin/env python3
"""
Velocity Tracking Test

Tests how well robots track commanded velocities under different conditions:
- No load: Robot moves freely
- With load: Robot pushes against an object (disturbance)

Tests all combinations:
- Model: dummy / wheel
- Kinematics: holonomic / diffdrive

Measures:
- Steady-state error
- Rise time
- Tracking accuracy under load

Usage:
    # Test holonomic dummy robot:
    python test_velocity_tracking.py --kinematics holonomic --model dummy
    
    # Test diffdrive wheel robot with load:
    python test_velocity_tracking.py --kinematics diffdrive --model wheel --with-load
    
    # Test all combinations:
    python test_velocity_tracking.py --test-all
"""
import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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

from object_utils import create_standard_objects, ContactPointParameterization
from contact_maintain.robot_factory import (
    create_robot, is_wheel_robot, get_wheel_velocities
)
from contact_maintain.object_bridge import generic_to_pybullet
from contact_maintain.contact_maintain_controller import InstantVelocityMatcher
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

# Test parameters
SETTLE_TIME = 1.0  # Time to wait before starting velocity command (s)
TEST_DURATION = 10.0  # Duration of velocity tracking test (s)
TARGET_VELOCITY_MAGNITUDE = 0.2  # Target velocity magnitude (m/s)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class VelocityTrackingHistory:
    """History for velocity tracking test."""
    times: List[float] = field(default_factory=list)
    cmd_velocities: List[np.ndarray] = field(default_factory=list)
    actual_velocities: List[np.ndarray] = field(default_factory=list)
    velocity_errors: List[float] = field(default_factory=list)
    positions: List[np.ndarray] = field(default_factory=list)
    headings: List[float] = field(default_factory=list)
    # For wheel robots
    wheel_velocities: List[np.ndarray] = field(default_factory=list)
    # Contact point analysis (from InstantVelocityMatcher equations)
    object_velocities: List[np.ndarray] = field(default_factory=list)
    object_angular_velocities: List[float] = field(default_factory=list)
    contact_point_positions: List[np.ndarray] = field(default_factory=list)
    contact_point_velocities: List[np.ndarray] = field(default_factory=list)
    heading_errors: List[float] = field(default_factory=list)
    t_params: List[float] = field(default_factory=list)


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
# VELOCITY TRACKING TEST
# ============================================================================

class VelocityTrackingTest:
    """Test velocity tracking performance."""
    
    def __init__(self, kinematics: str, model: str, with_load: bool = False):
        self.kinematics = kinematics
        self.model = model
        self.with_load = with_load
        
        # Create robot
        print(f"\nCreating {kinematics} {model} robot...")
        self.robot = create_robot(
            kinematics=kinematics,
            model=model,
            position=(0, 0, 0),
            orientation=0.0,
            name="test_robot"
        )
        
        # Create object if testing with load
        self.object_uid = None
        self.generic_object = None
        self.parameterization = None
        if with_load:
            print("Creating object for load testing...")
            standard_objects = create_standard_objects()
            self.generic_object = standard_objects[DEFAULT_OBJECT_SHAPE]
            
            self.object_uid = generic_to_pybullet(
                self.generic_object,
                height=DEFAULT_OBJECT_HEIGHT,
                position=(1.0, 0, 0),  # Place object 1m in front
                color=(0.4, 0.7, 0.4, 1.0)
            )
            pyb.changeDynamics(self.object_uid, -1, lateralFriction=DEFAULT_OBJECT_FRICTION, mass=1.0)
            
            # Create parameterization for contact point calculations
            self.parameterization = ContactPointParameterization(self.generic_object)
        else:
            # Even without load, create object for analysis (but don't place it)
            standard_objects = create_standard_objects()
            self.generic_object = standard_objects[DEFAULT_OBJECT_SHAPE]
            self.parameterization = ContactPointParameterization(self.generic_object)
        
        # Use a simple t_param for analysis (e.g., 0.0 = first vertex)
        self.test_t_param = 0.1
        
        # History
        self.history = VelocityTrackingHistory()
    
    def run_test(self, gui: bool = True) -> Dict:
        """Run velocity tracking test.
        
        Returns
        -------
        dict
            Test results with metrics.
        """
        print(f"\nRunning velocity tracking test...")
        print(f"  Kinematics: {self.kinematics}")
        print(f"  Model: {self.model}")
        print(f"  With load: {self.with_load}")
        print(f"  Target velocity: {TARGET_VELOCITY_MAGNITUDE} m/s")
        
        n_steps = int((SETTLE_TIME + TEST_DURATION) / TIMESTEP)
        settle_steps = int(SETTLE_TIME / TIMESTEP)
        step_count = 0
        t = 0.0
        
        # Determine target velocity command based on kinematics
        if self.kinematics == 'holonomic':
            # Command velocity in +X direction
            target_cmd = np.array([TARGET_VELOCITY_MAGNITUDE, 0.0, 0.0])  # [vx, vy, omega]
        else:  # diffdrive
            # Command forward velocity
            target_cmd = np.array([TARGET_VELOCITY_MAGNITUDE, 0.0])  # [v_forward, omega]
        
        for step in range(n_steps):
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Get robot state
                pos, heading, actual_vel = self.robot.get_state()
                
                # Get object state (if object exists)
                object_pos = np.zeros(2)
                object_orientation = 0.0
                object_velocity = np.zeros(2)
                object_angular_velocity = 0.0
                if self.object_uid is not None:
                    obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                    obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                    euler = pyb.getEulerFromQuaternion(obj_orn)
                    object_pos = np.array([obj_pos[0], obj_pos[1]])
                    object_orientation = euler[2]
                    object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                    object_angular_velocity = obj_vel_ang[2]
                
                # Calculate contact point position and velocity (using InstantVelocityMatcher equations)
                contact_point_world = None
                contact_point_velocity = None
                heading_error_calc = None
                if self.parameterization is not None:
                    # Get contact point in body frame
                    contact_info = self.parameterization.get_contact_info(self.test_t_param)
                    contact_point_body = contact_info['point']
                    
                    # Transform to world frame (same as InstantVelocityMatcher)
                    cos_t = np.cos(object_orientation)
                    sin_t = np.sin(object_orientation)
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    contact_point_world = R @ contact_point_body + object_pos
                    
                    # Compute boundary point velocity using rigid body kinematics
                    # v_point = v_object + omega × r (same as InstantVelocityMatcher line 156-159)
                    r = contact_point_world - object_pos
                    v_rotation = object_angular_velocity * np.array([-r[1], r[0]])
                    contact_point_velocity = object_velocity + v_rotation
                    
                    # Calculate heading error (same as InstantVelocityMatcher line 180-186)
                    to_contact_point = contact_point_world - pos
                    desired_heading = np.arctan2(to_contact_point[1], to_contact_point[0])
                    heading_error_calc = np.arctan2(np.sin(desired_heading - heading),
                                                   np.cos(desired_heading - heading))
                
                # Command velocity after settle time
                if step >= settle_steps:
                    # Command target velocity
                    self.robot.command_velocity(target_cmd)
                    cmd_vel = target_cmd.copy()
                else:
                    # Settle phase: command zero velocity
                    self.robot.command_velocity(np.zeros_like(target_cmd))
                    cmd_vel = np.zeros_like(target_cmd)
                
                # Record history
                self.history.times.append(t)
                self.history.cmd_velocities.append(cmd_vel.copy())
                self.history.actual_velocities.append(actual_vel.copy())
                self.history.positions.append(pos.copy())
                self.history.headings.append(heading)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.t_params.append(self.test_t_param)
                
                if contact_point_world is not None:
                    self.history.contact_point_positions.append(contact_point_world.copy())
                    self.history.contact_point_velocities.append(contact_point_velocity.copy())
                    self.history.heading_errors.append(heading_error_calc)
                else:
                    self.history.contact_point_positions.append(np.zeros(2))
                    self.history.contact_point_velocities.append(np.zeros(2))
                    self.history.heading_errors.append(0.0)
                
                # Calculate velocity error
                if self.kinematics == 'holonomic':
                    # Compare 2D velocity magnitude
                    cmd_mag = np.linalg.norm(cmd_vel[:2])
                    actual_mag = np.linalg.norm(actual_vel[:2])
                    error = abs(cmd_mag - actual_mag)
                else:  # diffdrive
                    # Compare forward velocity component
                    cmd_forward = cmd_vel[0]
                    # Actual forward velocity in body frame
                    actual_forward = actual_vel[0] * np.cos(heading) + actual_vel[1] * np.sin(heading)
                    error = abs(cmd_forward - actual_forward)
                
                self.history.velocity_errors.append(error)
                
                # Print detailed analysis (only during test phase, every 10 control steps)
                if step >= settle_steps and step_count % (CTRL_STEP * 10) == 0:
                    print(f"\n[t={t:.2f}s] Velocity Tracking Analysis:")
                    print(f"  Robot pos: {pos}, heading: {heading:.3f} rad")
                    print(f"  Commanded velocity: {cmd_vel}")
                    print(f"  Actual velocity: {actual_vel}")
                    print(f"  Velocity error: {error:.4f} m/s")
                    if self.object_uid is not None:
                        print(f"  Object pos: {object_pos}, orientation: {object_orientation:.3f} rad")
                        print(f"  Object velocity: {object_velocity}, angular: {object_angular_velocity:.3f} rad/s")
                        if contact_point_world is not None:
                            print(f"  Contact point (t_param={self.test_t_param:.3f}):")
                            print(f"    Position: {contact_point_world}")
                            print(f"    Velocity (from eq): {contact_point_velocity}")
                            print(f"    r = {r}, v_rotation = {v_rotation}")
                            print(f"    Heading error: {heading_error_calc:.3f} rad")
                
                # Record wheel velocities if available
                if is_wheel_robot(self.robot):
                    wheel_vel = get_wheel_velocities(self.robot)
                    self.history.wheel_velocities.append(wheel_vel.copy() if wheel_vel is not None else np.array([]))
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.3)
        
        return self._compute_metrics()
    
    def _compute_metrics(self) -> Dict:
        """Compute tracking metrics."""
        times = np.array(self.history.times)
        errors = np.array(self.history.velocity_errors)
        cmd_vels = np.array(self.history.cmd_velocities)
        actual_vels = np.array(self.history.actual_velocities)
        
        # Find test phase (after settle time)
        settle_idx = int(SETTLE_TIME / (1.0 / CTRL_FREQ))
        test_times = times[settle_idx:]
        test_errors = errors[settle_idx:]
        test_cmd = cmd_vels[settle_idx:]
        test_actual = actual_vels[settle_idx:]
        
        # Steady-state error (average error in last 50% of test)
        steady_start_idx = len(test_errors) // 2
        steady_state_error = np.mean(test_errors[steady_start_idx:])
        steady_state_error_std = np.std(test_errors[steady_start_idx:])
        
        # Rise time (time to reach 90% of target)
        if self.kinematics == 'holonomic':
            target_mag = np.linalg.norm(test_cmd[0, :2])
            actual_mags = np.linalg.norm(test_actual[:, :2], axis=1)
        else:
            target_mag = test_cmd[0, 0]
            actual_mags = test_actual[:, 0] * np.cos(self.history.headings[settle_idx:]) + \
                         test_actual[:, 1] * np.sin(self.history.headings[settle_idx:])
        
        target_90 = 0.9 * target_mag
        rise_time = None
        for i, actual_mag in enumerate(actual_mags):
            if actual_mag >= target_90:
                rise_time = test_times[i]
                break
        
        # Maximum error
        max_error = np.max(test_errors)
        
        # RMS error
        rms_error = np.sqrt(np.mean(test_errors**2))
        
        # Tracking accuracy (1 - normalized error)
        if target_mag > 1e-6:
            tracking_accuracy = 1.0 - (steady_state_error / target_mag)
        else:
            tracking_accuracy = 0.0
        
        return {
            'steady_state_error': float(steady_state_error),
            'steady_state_error_std': float(steady_state_error_std),
            'rise_time': float(rise_time) if rise_time is not None else None,
            'max_error': float(max_error),
            'rms_error': float(rms_error),
            'tracking_accuracy': float(tracking_accuracy),
            'target_velocity': float(target_mag),
        }
    
    def plot_results(self, save_path: Optional[Path] = None):
        """Plot velocity tracking results."""
        times = np.array(self.history.times)
        cmd_vels = np.array(self.history.cmd_velocities)
        actual_vels = np.array(self.history.actual_velocities)
        errors = np.array(self.history.velocity_errors)
        
        # Find test phase
        settle_idx = int(SETTLE_TIME / (1.0 / CTRL_FREQ))
        test_times = times[settle_idx:] - times[settle_idx]
        
        # Use 3x2 grid if we have object/contact point data, otherwise 2x2
        has_contact_data = self.object_uid is not None and len(self.history.contact_point_velocities) > 0
        if has_contact_data:
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Velocity Tracking Test: {self.kinematics} {self.model} '
                     f'({"with load" if self.with_load else "no load"})',
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Velocity magnitude over time
        ax = axes[0, 0]
        if self.kinematics == 'holonomic':
            cmd_mags = np.linalg.norm(cmd_vels[settle_idx:, :2], axis=1)
            actual_mags = np.linalg.norm(actual_vels[settle_idx:, :2], axis=1)
        else:
            cmd_mags = cmd_vels[settle_idx:, 0]
            headings = np.array(self.history.headings[settle_idx:])
            actual_mags = (actual_vels[settle_idx:, 0] * np.cos(headings) + 
                          actual_vels[settle_idx:, 1] * np.sin(headings))
        
        ax.plot(test_times, cmd_mags, 'b--', linewidth=2, label='Commanded')
        ax.plot(test_times, actual_mags, 'r-', linewidth=1.5, label='Actual')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Magnitude (m/s)')
        ax.set_title('Velocity Tracking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Velocity error over time
        ax = axes[0, 1]
        test_errors = errors[settle_idx:]
        ax.plot(test_times, test_errors, 'g-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Error (m/s)')
        ax.set_title('Tracking Error')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Trajectory
        ax = axes[1, 0]
        positions = np.array(self.history.positions)
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, label='Trajectory')
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
        if self.object_uid is not None:
            obj_pos, _ = pyb.getBasePositionAndOrientation(self.object_uid)
            ax.plot(obj_pos[0], obj_pos[1], 's', color='green', markersize=10, label='Object')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot 4: Metrics text (always in bottom-right for 2x2, or bottom-left for 3x2)
        if has_contact_data:
            ax = axes[2, 0]
        else:
            ax = axes[1, 1]
        ax.axis('off')
        metrics = self._compute_metrics()
        text = f"""TRACKING METRICS

            Steady-State Error: {metrics['steady_state_error']*100:.2f} cm/s
            (std: {metrics['steady_state_error_std']*100:.2f} cm/s)

            Rise Time: {metrics['rise_time']:.3f} s
            (time to 90% of target)

            Max Error: {metrics['max_error']*100:.2f} cm/s
            RMS Error: {metrics['rms_error']*100:.2f} cm/s

            Tracking Accuracy: {metrics['tracking_accuracy']*100:.1f}%
            Target Velocity: {metrics['target_velocity']:.3f} m/s
            """
        ax.text(0.1, 0.9, text, fontsize=11, family='monospace',
               verticalalignment='top', transform=ax.transAxes)
        
        # Plot 5 & 6: Contact point analysis (if available)
        if has_contact_data:
            # Plot 5: Heading error over time (bottom-right)
            ax = axes[1, 1]
            heading_errors = np.array(self.history.heading_errors[settle_idx:])
            ax.plot(test_times, np.degrees(heading_errors), 'orange', linewidth=1.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Heading Error (deg)')
            ax.set_title('Heading Error (robot to contact point)')
            ax.grid(True, alpha=0.3)
            
            # Plot 6: Object and contact point velocities (bottom-right of 3x2)
            ax = axes[2, 1]
            obj_vels = np.array(self.history.object_velocities[settle_idx:])
            contact_vels = np.array(self.history.contact_point_velocities[settle_idx:])
            
            obj_vel_mags = np.linalg.norm(obj_vels, axis=1)
            contact_vel_mags = np.linalg.norm(contact_vels, axis=1)
            
            ax.plot(test_times, obj_vel_mags, 'g--', linewidth=1.5, label='Object velocity')
            ax.plot(test_times, contact_vel_mags, 'm-', linewidth=1.5, label='Contact point velocity (from eq)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity Magnitude (m/s)')
            ax.set_title('Object & Contact Point Velocities')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


# ============================================================================
# SINGLE CONTACT TRACKING TEST
# ============================================================================

@dataclass
class SingleTrackingHistory:
    """History for single contact tracking test."""
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
    phase: List[str] = field(default_factory=list)  # 'approach' or 'maintain'


class SingleContactTrackingTest:
    """Test single robot tracking a contact point on a moving object.
    
    Scenario:
    1. Robot spawns away from object along normal outward direction
    2. Robot approaches object along normal inward direction
    3. Once in contact, robot maintains contact using InstantVelocityMatcher
    4. Object moves in arc/straight line due to contact force
    """
    
    def __init__(self, kinematics: str, t_param: float, approach_distance: float = 0.5):
        """
        Parameters
        ----------
        kinematics : str
            'holonomic' or 'diffdrive' (but only wheel model is used)
        t_param : float
            Target t_param on object boundary to track
        approach_distance : float
            Distance from object to spawn robot (along normal outward)
        """
        self.kinematics = kinematics
        self.t_param = t_param
        self.approach_distance = approach_distance
        
        # Create object
        print(f"\nCreating object with t_param={t_param:.3f}...")
        standard_objects = create_standard_objects()
        self.generic_object = standard_objects[DEFAULT_OBJECT_SHAPE]
        self.parameterization = ContactPointParameterization(self.generic_object)
        
        # Get contact point info at t_param
        contact_info = self.parameterization.get_contact_info(t_param)
        self.contact_point_body = contact_info['point']
        self.normal_outward = contact_info['normal_outward']
        self.normal_inward = -self.normal_outward
        
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
        
        # Calculate robot spawn position
        # Object is at origin, contact point in body frame is contact_point_body
        # Spawn robot at: contact_point_body + approach_distance * normal_outward
        spawn_position_body = self.contact_point_body + self.approach_distance * self.normal_outward
        
        # Create robot at spawn position
        print(f"Creating {kinematics} wheel robot at spawn position...")

        print(f"Spawn position body: {spawn_position_body}")

        self.robot = create_robot(
            kinematics=kinematics,
            model='wheel',  # Always use wheel model
            position=(spawn_position_body[0], spawn_position_body[1], 0),
            orientation=np.arctan2(self.normal_inward[1], self.normal_inward[0]),
            name="tracking_robot"
        )
        
        # Contact controller (will be initialized when contact is detected)
        self.contact_controller = None
        self.in_contact = False
        self.contact_threshold = 0.5  # N
        # Desired velocities captured at first contact
        self.desired_object_velocity_at_contact = None
        self.desired_object_angular_velocity_at_contact = None
        
        # History
        self.history = SingleTrackingHistory()
    
    def run_test(self, gui: bool = True, duration: float = 10.0) -> Dict:
        """Run single contact tracking test.
        
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
        print(f"\nRunning single contact tracking test...")
        print(f"  Kinematics: {self.kinematics}")
        print(f"  t_param: {self.t_param:.3f}")
        print(f"  Approach distance: {self.approach_distance:.2f} m")
        print(f"  Duration: {duration:.1f} s")
        
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
                obj_pos, obj_orn = pyb.getBasePositionAndOrientation(self.object_uid)
                obj_vel_lin, obj_vel_ang = pyb.getBaseVelocity(self.object_uid)
                euler = pyb.getEulerFromQuaternion(obj_orn)
                object_pos = np.array([obj_pos[0], obj_pos[1]])
                object_orientation = euler[2]
                object_velocity = np.array([obj_vel_lin[0], obj_vel_lin[1]])
                object_angular_velocity = obj_vel_ang[2]
                
                # Check contact
                contact_force = self._get_contact_force()
                was_in_contact = self.in_contact
                self.in_contact = contact_force > self.contact_threshold
                
                # Initialize contact controller on first contact
                if self.in_contact and not was_in_contact:
                    # Capture object velocities at the moment of first contact
                    self.desired_object_velocity_at_contact = object_velocity.copy()
                    self.desired_object_angular_velocity_at_contact = object_angular_velocity
                    
                    print(f"\n[t={t:.2f}s] Contact detected! Initializing contact controller...")
                    print(f"  Captured object velocity: {self.desired_object_velocity_at_contact}")
                    print(f"  Captured object angular velocity: {self.desired_object_angular_velocity_at_contact:.3f} rad/s")
                    
                    self.contact_controller = InstantVelocityMatcher(
                        self.generic_object, self.t_param
                    )
                    # Set desired velocities to match object's velocity at contact moment
                    # This should be achievable since object was already moving at this velocity
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
                        # For diffdrive, vx will be used as forward velocity
                        cmd = np.array([vel_2d[0], vel_2d[1], omega])

                        print(f" Approaching phase, cmd: {cmd}")
                    else:
                        # Very close, stop
                        cmd = np.zeros(3)
                    
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
                
                # Record history
                self.history.times.append(t)
                self.history.robot_positions.append(robot_pos.copy())
                self.history.robot_headings.append(robot_heading)
                self.history.robot_velocities.append(robot_vel.copy())
                self.history.robot_cmd_velocities.append(cmd.copy())
                self.history.object_positions.append(object_pos.copy())
                self.history.object_orientations.append(object_orientation)
                self.history.object_velocities.append(object_velocity.copy())
                self.history.object_angular_velocities.append(object_angular_velocity)
                self.history.contact_point_positions.append(contact_point_world.copy())
                self.history.contact_point_velocities.append(contact_point_velocity.copy())
                self.history.contact_forces.append(contact_force)
                self.history.in_contact.append(self.in_contact)
                self.history.phase.append(phase)
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.3)
        
        return self._compute_metrics()
    
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
                self.robot.uid, self.object_uid,
                linkIndexA=link_idx,
                max_contacts=4
            )
            return np.linalg.norm(force[:2])
        except:
            return 0.0
    
    def _compute_metrics(self) -> Dict:
        """Compute tracking metrics."""
        times = np.array(self.history.times)
        in_contact = np.array(self.history.in_contact)
        contact_forces = np.array(self.history.contact_forces)
        
        # Find contact phase
        contact_start_idx = None
        for i, contact in enumerate(in_contact):
            if contact:
                contact_start_idx = i
                break
        
        if contact_start_idx is None:
            return {
                'contact_achieved': False,
                'contact_time': None,
                'avg_contact_force': 0.0,
                'max_contact_force': 0.0,
            }
        
        contact_times = times[contact_start_idx:]
        contact_force_values = contact_forces[contact_start_idx:]
        
        # Calculate position tracking error (robot to contact point)
        robot_positions = np.array(self.history.robot_positions[contact_start_idx:])
        contact_positions = np.array(self.history.contact_point_positions[contact_start_idx:])
        position_errors = np.linalg.norm(robot_positions - contact_positions, axis=1)
        
        # Calculate velocity tracking error
        robot_velocities = np.array(self.history.robot_velocities[contact_start_idx:])
        contact_velocities = np.array(self.history.contact_point_velocities[contact_start_idx:])
        velocity_errors = np.linalg.norm(robot_velocities[:, :2] - contact_velocities, axis=1)
        
        return {
            'contact_achieved': True,
            'contact_time': float(times[contact_start_idx]),
            'avg_contact_force': float(np.mean(contact_force_values)),
            'max_contact_force': float(np.max(contact_force_values)),
            'avg_position_error': float(np.mean(position_errors)),
            'max_position_error': float(np.max(position_errors)),
            'avg_velocity_error': float(np.mean(velocity_errors)),
            'max_velocity_error': float(np.max(velocity_errors)),
            'contact_duration': float(times[-1] - times[contact_start_idx]),
        }
    
    def plot_results(self, save_path: Optional[Path] = None):
        """Plot single contact tracking results."""
        times = np.array(self.history.times)
        robot_positions = np.array(self.history.robot_positions)
        object_positions = np.array(self.history.object_positions)
        contact_positions = np.array(self.history.contact_point_positions)
        in_contact = np.array(self.history.in_contact)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Single Contact Tracking Test: {self.kinematics} wheel, t_param={self.t_param:.3f}',
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
        
        # Plot 2: Contact force over time
        ax = axes[0, 1]
        contact_forces = np.array(self.history.contact_forces)
        ax.plot(times, contact_forces, 'r-', linewidth=1.5)
        ax.axhline(y=self.contact_threshold, color='g', linestyle='--', label='Contact threshold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact Force (N)')
        ax.set_title('Contact Force')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Position error (robot to contact point)
        ax = axes[0, 2]
        position_errors = np.linalg.norm(robot_positions - contact_positions, axis=1)
        ax.plot(times, position_errors * 100, 'b-', linewidth=1.5)  # Convert to cm
        ax.axvline(x=times[contact_start_idx] if contact_start_idx is not None else 0, 
                  color='g', linestyle='--', label='Contact start')
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
        
        # Plot 5: Object motion
        ax = axes[1, 1]
        object_vels = np.array(self.history.object_velocities)
        object_angular_vels = np.array(self.history.object_angular_velocities)
        object_vel_mags = np.linalg.norm(object_vels, axis=1)
        ax.plot(times, object_vel_mags, 'g-', linewidth=1.5, label='Object linear velocity')
        ax.plot(times, np.abs(object_angular_vels), 'g--', linewidth=1.5, label='Object angular velocity (abs)')
        if contact_start_idx is not None:
            ax.axvline(x=times[contact_start_idx], color='r', linestyle='--', label='Contact start')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s) or Angular Vel (rad/s)')
        ax.set_title('Object Motion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Metrics text
        ax = axes[1, 2]
        ax.axis('off')
        metrics = self._compute_metrics()
        if metrics['contact_achieved']:
            text = f"""TRACKING METRICS

        Contact Achieved: Yes
        Contact Time: {metrics['contact_time']:.2f} s
        Contact Duration: {metrics['contact_duration']:.2f} s

        Avg Contact Force: {metrics['avg_contact_force']:.2f} N
        Max Contact Force: {metrics['max_contact_force']:.2f} N

        Avg Position Error: {metrics['avg_position_error']*100:.2f} cm
        Max Position Error: {metrics['max_position_error']*100:.2f} cm

        Avg Velocity Error: {metrics['avg_velocity_error']*100:.2f} cm/s
        Max Velocity Error: {metrics['max_velocity_error']*100:.2f} cm/s
        """
        else:
            text = f"""TRACKING METRICS

            Contact Achieved: No
            Robot did not make contact
            with the object.
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


# ============================================================================
# ROBOT VELOCITY TRACKING TEST (Open-loop)
# ============================================================================

@dataclass
class RobotVelocityTrackingHistory:
    """History for robot velocity tracking test."""
    times: List[float] = field(default_factory=list)
    cmd_velocities: List[np.ndarray] = field(default_factory=list)
    actual_velocities: List[np.ndarray] = field(default_factory=list)
    velocity_errors: List[np.ndarray] = field(default_factory=list)
    positions: List[np.ndarray] = field(default_factory=list)
    headings: List[float] = field(default_factory=list)


class RobotVelocityTrackingTest:
    """Test robot's ability to track commanded velocities.
    
    Tests:
    1. Open-loop velocity tracking without obstacles
    2. Velocity tracking with obstacles (disturbances)
    """
    
    def __init__(self, kinematics: str, target_velocity: np.ndarray):
        """
        Parameters
        ----------
        kinematics : str
            'holonomic' or 'diffdrive'
        target_velocity : np.ndarray
            Target velocity command [vx, vy, omega] or [v_forward, omega] for diffdrive
        """
        self.kinematics = kinematics
        self.target_velocity = np.array(target_velocity)
        
        # Create robot at origin
        print(f"\nCreating {kinematics} wheel robot...")
        self.robot = create_robot(
            kinematics=kinematics,
            model='wheel',  # Always use wheel model
            position=(0.0, 0.0, 0),
            orientation=0.0,
            name="velocity_test_robot"
        )
        
        # Obstacles (will be added later)
        self.obstacle_uids = []
        
        # History
        self.history = RobotVelocityTrackingHistory()
    
    def add_obstacle(self, position: np.ndarray, size: float = 0.3):
        """Add a box obstacle for disturbance testing.
        
        Parameters
        ----------
        position : np.ndarray
            Obstacle (x, y) position
        size : float
            Obstacle size (side length of cube)
        """
        obstacle_shape = pyb.createCollisionShape(pyb.GEOM_BOX, halfExtents=[size/2, size/2, DEFAULT_OBJECT_HEIGHT/2])
        obstacle_visual = pyb.createVisualShape(pyb.GEOM_BOX, halfExtents=[size/2, size/2, DEFAULT_OBJECT_HEIGHT/2],
                                               rgbaColor=[0.8, 0.2, 0.2, 1.0])
        obstacle_uid = pyb.createMultiBody(baseMass=5.0,  # Light obstacle
                                           baseCollisionShapeIndex=obstacle_shape,
                                           baseVisualShapeIndex=obstacle_visual,
                                           basePosition=[position[0], position[1], DEFAULT_OBJECT_HEIGHT/2])
        pyb.changeDynamics(obstacle_uid, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
        self.obstacle_uids.append(obstacle_uid)
        print(f"Added obstacle at {position} with size {size}m")
    
    def run_test(self, gui: bool = True, duration: float = 10.0, 
                 add_obstacle_at: Optional[float] = None) -> Dict:
        """Run velocity tracking test.
        
        Parameters
        ----------
        gui : bool
            Show PyBullet GUI
        duration : float
            Test duration in seconds
        add_obstacle_at : float, optional
            Time (seconds) to add obstacle. If None, no obstacle added.
        
        Returns
        -------
        dict
            Test results with metrics
        """
        print(f"\nRunning robot velocity tracking test...")
        print(f"  Kinematics: {self.kinematics}")
        print(f"  Target velocity: {self.target_velocity}")
        print(f"  Duration: {duration:.1f} s")
        if add_obstacle_at is not None:
            print(f"  Obstacle will be added at t={add_obstacle_at:.1f} s")
        
        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        obstacle_added = False
        
        for step in range(n_steps):
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Get robot state
                robot_pos, robot_heading, robot_vel = self.robot.get_state()
                
                # Add obstacle at specified time
                if add_obstacle_at is not None and not obstacle_added and t >= add_obstacle_at:
                    # Add obstacle in front of robot
                    obstacle_pos = robot_pos[:2] + np.array([0.5, 0.0])  # 0.5m in front
                    self.add_obstacle(obstacle_pos, size=0.3)
                    obstacle_added = True
                    print(f"\n[t={t:.2f}s] Obstacle added at {obstacle_pos}")
                
                # Command target velocity
                self.robot.command_velocity(self.target_velocity)
                
                # Record history
                self.history.times.append(t)
                self.history.cmd_velocities.append(self.target_velocity.copy())
                self.history.actual_velocities.append(robot_vel.copy())
                self.history.positions.append(robot_pos.copy())
                self.history.headings.append(robot_heading)
                
                # Calculate velocity error
                if self.kinematics == 'holonomic':
                    # Compare all 3 components
                    error = robot_vel - self.target_velocity
                else:  # diffdrive
                    # For diffdrive, compare forward velocity and angular
                    cmd_forward = self.target_velocity[0]
                    actual_forward = robot_vel[0] * np.cos(robot_heading) + robot_vel[1] * np.sin(robot_heading)
                    cmd_omega = self.target_velocity[1] if len(self.target_velocity) > 1 else 0.0
                    error = np.array([actual_forward - cmd_forward, 0.0, robot_vel[2] - cmd_omega])
                
                self.history.velocity_errors.append(error)
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.3)
        
        return self._compute_metrics(add_obstacle_at)
    
    def _compute_metrics(self, obstacle_time: Optional[float] = None) -> Dict:
        """Compute tracking metrics."""
        times = np.array(self.history.times)
        cmd_vels = np.array(self.history.cmd_velocities)
        actual_vels = np.array(self.history.actual_velocities)
        errors = np.array(self.history.velocity_errors)
        
        # Split into phases if obstacle was added
        if obstacle_time is not None:
            obstacle_idx = int(obstacle_time / (1.0 / CTRL_FREQ))
            before_times = times[:obstacle_idx]
            after_times = times[obstacle_idx:]
            before_errors = errors[:obstacle_idx]
            after_errors = errors[obstacle_idx:]
            
            # Metrics before obstacle
            before_steady_start = len(before_errors) // 2
            before_steady_error = np.mean(np.linalg.norm(before_errors[before_steady_start:, :2], axis=1))
            before_max_error = np.max(np.linalg.norm(before_errors[:, :2], axis=1))
            before_rms_error = np.sqrt(np.mean(np.linalg.norm(before_errors[:, :2], axis=1)**2))
            
            # Metrics after obstacle
            if len(after_errors) > 0:
                after_steady_start = len(after_errors) // 2
                after_steady_error = np.mean(np.linalg.norm(after_errors[after_steady_start:, :2], axis=1))
                after_max_error = np.max(np.linalg.norm(after_errors[:, :2], axis=1))
                after_rms_error = np.sqrt(np.mean(np.linalg.norm(after_errors[:, :2], axis=1)**2))
            else:
                after_steady_error = 0.0
                after_max_error = 0.0
                after_rms_error = 0.0
            
            return {
                'before_obstacle': {
                    'steady_state_error': float(before_steady_error),
                    'max_error': float(before_max_error),
                    'rms_error': float(before_rms_error),
                },
                'after_obstacle': {
                    'steady_state_error': float(after_steady_error),
                    'max_error': float(after_max_error),
                    'rms_error': float(after_rms_error),
                },
                'obstacle_time': float(obstacle_time),
            }
        else:
            # No obstacle - single phase
            steady_start_idx = len(errors) // 2
            steady_state_error = np.mean(np.linalg.norm(errors[steady_start_idx:, :2], axis=1))
            max_error = np.max(np.linalg.norm(errors[:, :2], axis=1))
            rms_error = np.sqrt(np.mean(np.linalg.norm(errors[:, :2], axis=1)**2))
            
            return {
                'steady_state_error': float(steady_state_error),
                'max_error': float(max_error),
                'rms_error': float(rms_error),
            }
    
    def plot_results(self, save_path: Optional[Path] = None):
        """Plot velocity tracking results."""
        times = np.array(self.history.times)
        cmd_vels = np.array(self.history.cmd_velocities)
        actual_vels = np.array(self.history.actual_velocities)
        errors = np.array(self.history.velocity_errors)
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Robot Velocity Tracking Test: {self.kinematics} wheel, target={self.target_velocity}',
                     fontsize=14, fontweight='bold')
        
        # Plot 1: vx component
        ax = axes[0, 0]
        if self.kinematics == 'holonomic':
            ax.plot(times, cmd_vels[:, 0], 'b--', linewidth=2, label='Cmd vx')
            ax.plot(times, actual_vels[:, 0], 'b-', linewidth=1.5, label='Actual vx', alpha=0.7)
        else:
            # For diffdrive, show forward velocity component in world frame
            cmd_forward = cmd_vels[:, 0]
            headings = np.array(self.history.headings)
            actual_vx = actual_vels[:, 0] * np.cos(headings) - actual_vels[:, 1] * np.sin(headings)
            cmd_vx = cmd_forward * np.cos(headings)
            ax.plot(times, cmd_vx, 'b--', linewidth=2, label='Cmd vx (from v_forward)')
            ax.plot(times, actual_vx, 'b-', linewidth=1.5, label='Actual vx', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('vx (m/s)')
        ax.set_title('Velocity X Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: vy component
        ax = axes[0, 1]
        if self.kinematics == 'holonomic':
            ax.plot(times, cmd_vels[:, 1], 'g--', linewidth=2, label='Cmd vy')
            ax.plot(times, actual_vels[:, 1], 'g-', linewidth=1.5, label='Actual vy', alpha=0.7)
        else:
            # For diffdrive, show lateral velocity component in world frame
            cmd_forward = cmd_vels[:, 0]
            headings = np.array(self.history.headings)
            actual_vy = actual_vels[:, 0] * np.sin(headings) + actual_vels[:, 1] * np.cos(headings)
            cmd_vy = cmd_forward * np.sin(headings)
            ax.plot(times, cmd_vy, 'g--', linewidth=2, label='Cmd vy (from v_forward)')
            ax.plot(times, actual_vy, 'g-', linewidth=1.5, label='Actual vy', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('vy (m/s)')
        ax.set_title('Velocity Y Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Angular velocity
        ax = axes[1, 0]
        ax.plot(times, cmd_vels[:, 2], 'r--', linewidth=2, label='Cmd omega')
        ax.plot(times, actual_vels[:, 2], 'r-', linewidth=1.5, label='Actual omega', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Velocity error magnitude
        ax = axes[1, 1]
        error_mags = np.linalg.norm(errors[:, :2], axis=1)
        ax.plot(times, error_mags * 100, 'r-', linewidth=1.5)  # Convert to cm/s
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Error Magnitude (cm/s)')
        ax.set_title('Velocity Tracking Error')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Trajectory
        ax = axes[2, 0]
        positions = np.array(self.history.positions)
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, label='Robot trajectory')
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
        
        # Plot obstacles
        for obs_uid in self.obstacle_uids:
            obs_pos, _ = pyb.getBasePositionAndOrientation(obs_uid)
            ax.plot(obs_pos[0], obs_pos[1], 'rs', markersize=12, label='Obstacle')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot 6: Metrics text
        ax = axes[2, 1]
        ax.axis('off')
        metrics = self._compute_metrics(None if not self.obstacle_uids else 5.0)
        
        if 'before_obstacle' in metrics:
            text = f"""TRACKING METRICS

BEFORE OBSTACLE (t=0 to {metrics['obstacle_time']:.1f}s):
  Steady-state error: {metrics['before_obstacle']['steady_state_error']*100:.2f} cm/s
  Max error: {metrics['before_obstacle']['max_error']*100:.2f} cm/s
  RMS error: {metrics['before_obstacle']['rms_error']*100:.2f} cm/s

AFTER OBSTACLE (t>{metrics['obstacle_time']:.1f}s):
  Steady-state error: {metrics['after_obstacle']['steady_state_error']*100:.2f} cm/s
  Max error: {metrics['after_obstacle']['max_error']*100:.2f} cm/s
  RMS error: {metrics['after_obstacle']['rms_error']*100:.2f} cm/s
"""
        else:
            text = f"""TRACKING METRICS

Steady-state error: {metrics['steady_state_error']*100:.2f} cm/s
Max error: {metrics['max_error']*100:.2f} cm/s
RMS error: {metrics['rms_error']*100:.2f} cm/s

Target velocity: {self.target_velocity}
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Velocity Tracking Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Test holonomic dummy robot:
                python test_velocity_tracking.py --kinematics holonomic --model dummy
                
                # Test diffdrive wheel robot with load:
                python test_velocity_tracking.py --kinematics diffdrive --model wheel --with-load
                
                # Test all combinations:
                python test_velocity_tracking.py --test-all
                
                 # Single contact tracking test (wheel robots only):
                 python test_velocity_tracking.py --test-single-tracking --kinematics holonomic --t-param 0.125
                 python test_velocity_tracking.py --test-single-tracking --kinematics diffdrive --t-param 0.25 --approach-distance 0.6
                 
                 # Robot velocity tracking test (open-loop):
                 python test_velocity_tracking.py --test-robot-velocity --kinematics holonomic --target-velocity 0.5,0,0.2
                 python test_velocity_tracking.py --test-robot-velocity --kinematics diffdrive --target-velocity 0.5,0.2 --add-obstacle-at 5.0
                 
                 # Save results:
                 python test_velocity_tracking.py --kinematics holonomic --model dummy --save-dir /tmp/results/
                     """
    )
    parser.add_argument("--kinematics", "-k", default="holonomic",
                       choices=['holonomic', 'diffdrive'],
                       help="Kinematics type (default: holonomic)")
    parser.add_argument("--model", "-m", default="dummy",
                       choices=['dummy', 'wheel'],
                       help="Robot model (default: dummy)")
    parser.add_argument("--with-load", action="store_true",
                       help="Test with object load (robot pushes against object)")
    parser.add_argument("--test-all", action="store_true",
                       help="Test all combinations of kinematics and models")
    parser.add_argument("--test-single-tracking", action="store_true",
                       help="Run single contact tracking test (wheel robots only)")
    parser.add_argument("--test-robot-velocity", action="store_true",
                       help="Test robot velocity tracking (open-loop)")
    parser.add_argument("--target-velocity", type=str, default="0.5,0,0.2",
                       help="Target velocity as 'vx,vy,omega' or 'v_forward,omega' (default: 0.5,0,0.2)")
    parser.add_argument("--add-obstacle-at", type=float, default=None,
                       help="Time (s) to add obstacle for disturbance test")
    parser.add_argument("--t-param", type=float, default=0.125,
                       help="t_param for single tracking test (default: 0.125)")
    parser.add_argument("--approach-distance", type=float, default=0.5,
                       help="Distance from object to spawn robot (default: 0.5 m)")
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
    
    if args.test_robot_velocity:
        # Robot velocity tracking test
        print(f"\n{'='*60}")
        print(f"Robot Velocity Tracking Test")
        print(f"{'='*60}")
        
        # Parse target velocity
        try:
            vel_parts = [float(x.strip()) for x in args.target_velocity.split(',')]
            if len(vel_parts) == 2:
                # Diffdrive format: [v_forward, omega]
                target_velocity = np.array([vel_parts[0], vel_parts[1]])
            elif len(vel_parts) == 3:
                # Holonomic format: [vx, vy, omega]
                target_velocity = np.array(vel_parts)
            else:
                raise ValueError("Target velocity must have 2 or 3 components")
        except Exception as e:
            print(f"Error parsing target velocity: {e}")
            print("Using default: [0.5, 0, 0.2]")
            target_velocity = np.array([0.5, 0.0, 0.2])
        
        test = RobotVelocityTrackingTest(
            kinematics=args.kinematics,
            target_velocity=target_velocity
        )
        results = test.run_test(
            gui=not args.no_gui, 
            duration=args.duration,
            add_obstacle_at=args.add_obstacle_at
        )
        
        # Print results
        print("\n" + "="*60)
        print("ROBOT VELOCITY TRACKING RESULTS")
        print("="*60)
        print(f"  Kinematics: {args.kinematics}")
        print(f"  Target velocity: {target_velocity}")
        if 'before_obstacle' in results:
            print(f"\n  BEFORE OBSTACLE:")
            print(f"    Steady-state error: {results['before_obstacle']['steady_state_error']*100:.2f} cm/s")
            print(f"    Max error: {results['before_obstacle']['max_error']*100:.2f} cm/s")
            print(f"    RMS error: {results['before_obstacle']['rms_error']*100:.2f} cm/s")
            print(f"\n  AFTER OBSTACLE (added at t={results['obstacle_time']:.1f}s):")
            print(f"    Steady-state error: {results['after_obstacle']['steady_state_error']*100:.2f} cm/s")
            print(f"    Max error: {results['after_obstacle']['max_error']*100:.2f} cm/s")
            print(f"    RMS error: {results['after_obstacle']['rms_error']*100:.2f} cm/s")
        else:
            print(f"  Steady-state error: {results['steady_state_error']*100:.2f} cm/s")
            print(f"  Max error: {results['max_error']*100:.2f} cm/s")
            print(f"  RMS error: {results['rms_error']*100:.2f} cm/s")
        print("="*60)
        
        # Save results
        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_results(save_path / "robot_velocity_tracking.png")
        else:
            test.plot_results()
    
    elif args.test_single_tracking:
        # Single contact tracking test
        print(f"\n{'='*60}")
        print(f"Single Contact Tracking Test")
        print(f"{'='*60}")
        
        test = SingleContactTrackingTest(
            kinematics=args.kinematics,
            t_param=args.t_param,
            approach_distance=args.approach_distance
        )
        results = test.run_test(gui=not args.no_gui, duration=args.duration)
        
        # Print results
        print("\n" + "="*60)
        print("SINGLE CONTACT TRACKING RESULTS")
        print("="*60)
        print(f"  Kinematics: {args.kinematics}")
        print(f"  t_param: {args.t_param:.3f}")
        print(f"  Contact achieved: {results['contact_achieved']}")
        if results['contact_achieved']:
            print(f"  Contact time: {results['contact_time']:.2f} s")
            print(f"  Contact duration: {results['contact_duration']:.2f} s")
            print(f"  Avg contact force: {results['avg_contact_force']:.2f} N")
            print(f"  Avg position error: {results['avg_position_error']*100:.2f} cm")
            print(f"  Avg velocity error: {results['avg_velocity_error']*100:.2f} cm/s")
        print("="*60)
        
        # Save results
        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_results(save_path / "single_tracking.png")
        else:
            test.plot_results()
    
    elif args.test_all:
        # Test all combinations
        configs = [
            ('holonomic', 'dummy', False),
            ('holonomic', 'dummy', True),
            ('holonomic', 'wheel', False),
            ('holonomic', 'wheel', True),
            ('diffdrive', 'dummy', False),
            ('diffdrive', 'dummy', True),
            ('diffdrive', 'wheel', False),
            ('diffdrive', 'wheel', True),
        ]
        
        all_results = {}
        for kinematics, model, with_load in configs:
            print(f"\n{'='*60}")
            print(f"Testing: {kinematics} {model} {'(with load)' if with_load else '(no load)'}")
            print(f"{'='*60}")
            
            test = VelocityTrackingTest(kinematics, model, with_load)
            results = test.run_test(gui=not args.no_gui)
            
            config_name = f"{kinematics}_{model}_{'load' if with_load else 'noload'}"
            all_results[config_name] = results
            
            if args.save_dir:
                save_path = Path(args.save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                test.plot_results(save_path / f"{config_name}.png")
            else:
                test.plot_results()
            
            pyb.resetSimulation()
            ground = setup_pybullet(gui=not args.no_gui)
            pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY OF ALL TESTS")
        print("="*60)
        for config_name, results in all_results.items():
            print(f"\n{config_name}:")
            print(f"  Steady-state error: {results['steady_state_error']*100:.2f} cm/s")
            print(f"  Tracking accuracy: {results['tracking_accuracy']*100:.1f}%")
            if results['rise_time']:
                print(f"  Rise time: {results['rise_time']:.3f} s")
        
    else:
        # Single test
        test = VelocityTrackingTest(args.kinematics, args.model, args.with_load)
        results = test.run_test(gui=not args.no_gui)
        
        # Print results
        print("\n" + "="*60)
        print("VELOCITY TRACKING RESULTS")
        print("="*60)
        print(f"  Kinematics: {args.kinematics}")
        print(f"  Model: {args.model}")
        print(f"  With load: {args.with_load}")
        print(f"  Steady-state error: {results['steady_state_error']*100:.2f} cm/s")
        print(f"  Tracking accuracy: {results['tracking_accuracy']*100:.1f}%")
        if results['rise_time']:
            print(f"  Rise time: {results['rise_time']:.3f} s")
        print(f"  Max error: {results['max_error']*100:.2f} cm/s")
        print(f"  RMS error: {results['rms_error']*100:.2f} cm/s")
        print("="*60)
        
        # Save results
        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            test.plot_results(save_path / "velocity_tracking.png")
        else:
            test.plot_results()
    
    if not args.no_gui:
        print("\nPress Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()