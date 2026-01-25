#!/usr/bin/env python3
"""
Comprehensive Contact Maintenance Test

Flexible testing framework supporting all robot configurations:
- Single/Multi robot (1 to N robots)
- Dummy/Wheel model (direct velocity vs wheel physics)
- Holonomic/Diff-drive kinematics
- Velocity/Wrench controller

Each robot tracks a different boundary point (t_param).

PyBullet GUI Interaction:
- Left-click + drag: Rotate camera
- Middle-click + drag: Pan camera
- Scroll wheel: Zoom in/out

Usage:
    # Single holonomic dummy robot with velocity controller:
    python comprehensive_contact_test.py --num-robots 1 --kinematics holonomic --model dummy

    # Multi-robot (3) with wheel model:
    python comprehensive_contact_test.py --num-robots 3 --model wheel --t-params 0.1,0.4,0.7

    # Diff-drive with wrench controller:
    python comprehensive_contact_test.py --kinematics diffdrive --controller wrench

    # Full test with output:
    python comprehensive_contact_test.py --num-robots 3 --model wheel --save-dir /tmp/results/
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

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

from object_utils import create_standard_objects
from contact_maintain.robot_factory import (
    create_robot, is_wheel_robot, get_wheel_velocities, get_command_wheel_velocities
)
from contact_maintain.object_bridge import generic_to_pybullet
from contact_maintain.contact_maintain_controller import (
    InstantVelocityMatcher, WrenchTrackingController
)
from contact_maintain.pyb_simulation import get_contact_force


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

# Default object parameters (objects are now 2.5x larger)
DEFAULT_OBJECT_SHAPE = 'rectangle'
DEFAULT_OBJECT_HEIGHT_DUMMY = 0.05   # For dummy robots (shorter, scaled for small robot)
DEFAULT_OBJECT_HEIGHT_WHEEL = 0.08   # For wheel robots (taller to avoid multi-contact)
DEFAULT_OBJECT_FRICTION = 0.3

# Robot spawn configuration (increased for larger objects)
ROBOT_SPAWN_RADIUS = 1.5  # Distance from object center


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RobotHistory:
    """Per-robot history tracking."""
    name: str
    t_param: float
    times: List[float] = field(default_factory=list)
    positions: List[np.ndarray] = field(default_factory=list)
    headings: List[float] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    cmd_velocities: List[np.ndarray] = field(default_factory=list)
    in_contact: List[bool] = field(default_factory=list)
    contact_forces: List[float] = field(default_factory=list)
    position_errors: List[float] = field(default_factory=list)
    target_points: List[np.ndarray] = field(default_factory=list)
    # Wheel velocities (only for wheel models)
    wheel_velocities: List[np.ndarray] = field(default_factory=list)
    cmd_wheel_velocities: List[np.ndarray] = field(default_factory=list)


@dataclass
class ObjectHistory:
    """Object state history."""
    times: List[float] = field(default_factory=list)
    positions: List[np.ndarray] = field(default_factory=list)
    orientations: List[float] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    angular_velocities: List[float] = field(default_factory=list)


@dataclass
class TestConfig:
    """Test configuration."""
    num_robots: int
    kinematics: str
    model: str
    controller: str
    t_params: List[float]
    duration: float
    perturbation: str
    object_shape: str
    save_dir: Optional[str]
    
    def to_dict(self):
        return {
            'num_robots': self.num_robots,
            'kinematics': self.kinematics,
            'model': self.model,
            'controller': self.controller,
            't_params': self.t_params,
            'duration': self.duration,
            'perturbation': self.perturbation,
            'object_shape': self.object_shape,
        }
    
    def get_name(self):
        """Generate a descriptive name for this configuration."""
        return f"{self.kinematics}_{self.model}_{self.controller}_n{self.num_robots}"


# ============================================================================
# SIMULATION SETUP
# ============================================================================

def setup_pybullet(gui=True):
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
            cameraDistance=2.5,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    
    return ground


def apply_perturbation(object_uid, t, perturbation_type='pulse'):
    """Apply perturbation to object."""
    if perturbation_type == 'none':
        return
    
    if perturbation_type == 'pulse':
        if 2.0 <= t < 2.1:
            force = [3.0, 0, 0]
            pos, _ = pyb.getBasePositionAndOrientation(object_uid)
            pyb.applyExternalForce(object_uid, -1, force, pos, pyb.WORLD_FRAME)
        elif 4.0 <= t < 4.1:
            force = [0, 3.0, 0]
            pos, _ = pyb.getBasePositionAndOrientation(object_uid)
            pyb.applyExternalForce(object_uid, -1, force, pos, pyb.WORLD_FRAME)
        elif 6.0 <= t < 6.1:
            torque = [0, 0, 1.0]
            pyb.applyExternalTorque(object_uid, -1, torque, pyb.WORLD_FRAME)
    
    elif perturbation_type == 'continuous':
        fx = 1.0 * np.sin(0.5 * t)
        fy = 1.0 * np.cos(0.5 * t)
        force = [fx, fy, 0]
        pos, _ = pyb.getBasePositionAndOrientation(object_uid)
        pyb.applyExternalForce(object_uid, -1, force, pos, pyb.WORLD_FRAME)


# ============================================================================
# COMPREHENSIVE TEST CLASS
# ============================================================================

class ComprehensiveContactTest:
    """Comprehensive contact maintenance test."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        
        # Create object
        print(f"\nCreating {config.object_shape} object...")
        standard_objects = create_standard_objects()
        self.generic_object = standard_objects[config.object_shape]
        
        # Use taller object for wheel robots to avoid multi-contact issues
        object_height = DEFAULT_OBJECT_HEIGHT_WHEEL if config.model == 'wheel' else DEFAULT_OBJECT_HEIGHT_DUMMY
        
        self.object_uid = generic_to_pybullet(
            self.generic_object,
            height=object_height,
            position=(0, 0, 0),
            color=(0.4, 0.7, 0.4, 1.0)
        )
        pyb.changeDynamics(self.object_uid, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
        
        # Create robots and controllers
        self.robots: Dict[str, object] = {}
        self.controllers: Dict[str, object] = {}
        self.robot_histories: Dict[str, RobotHistory] = {}
        
        self._create_robots_and_controllers()
        
        # Object history
        self.object_history = ObjectHistory()
    
    def _create_robots_and_controllers(self):
        """Create all robots and their controllers."""
        n = self.config.num_robots
        t_params = self.config.t_params
        
        print(f"\nCreating {n} robots ({self.config.kinematics}, {self.config.model})...")
        
        for i in range(n):
            name = f"R_{i+1:02d}"
            t_param = t_params[i]
            
            # Create controller first to get target point
            if self.config.controller == 'velocity':
                controller = InstantVelocityMatcher(
                    self.generic_object, t_param,
                    kp_position=3.0, max_velocity=0.5
                )
            else:
                controller = WrenchTrackingController(
                    self.generic_object, t_param,
                    desired_wrench=np.array([2.0, 0, 0]),
                    kp_force=0.2, max_velocity=0.3
                )
            
            # Get target point for initial positioning
            target_point = controller.get_target_point(np.zeros(2), 0.0)
            
            # Calculate spawn position (around the object)
            if n == 1:
                spawn_angle = np.arctan2(target_point[1], target_point[0])
            else:
                # Distribute robots around the object
                spawn_angle = 2 * np.pi * i / n
            
            robot_x = ROBOT_SPAWN_RADIUS * np.cos(spawn_angle)
            robot_y = ROBOT_SPAWN_RADIUS * np.sin(spawn_angle)
            robot_heading = spawn_angle + np.pi  # Face inward
            
            # Create robot using factory
            robot = create_robot(
                kinematics=self.config.kinematics,
                model=self.config.model,
                position=(robot_x, robot_y),
                orientation=robot_heading,
                name=name
            )
            
            self.robots[name] = robot
            self.controllers[name] = controller
            self.robot_histories[name] = RobotHistory(name=name, t_param=t_param)
            
            print(f"  {name}: t_param={t_param:.2f}, pos=({robot_x:.2f}, {robot_y:.2f})")
    
    def _get_robot_state(self, robot):
        """Get robot state in a unified format."""
        pos, heading, vel = robot.get_state()
        return pos, heading, vel
    
    def _get_object_state(self):
        """Get object state."""
        pos, orn = pyb.getBasePositionAndOrientation(self.object_uid)
        vel_lin, vel_ang = pyb.getBaseVelocity(self.object_uid)
        euler = pyb.getEulerFromQuaternion(orn)
        
        return {
            'position': np.array([pos[0], pos[1]]),
            'orientation': euler[2],
            'velocity': np.array([vel_lin[0], vel_lin[1]]),
            'angular_velocity': vel_ang[2],
        }
    
    def _get_contact_force(self, robot):
        """Get contact force for a robot."""
        # Find bumper/contact link index
        # Note: HolonomicRobot uses 'contact_link_idx', wheel robots use 'bumper_link_idx'
        link_idx = -1
        if hasattr(robot, 'bumper_link_idx') and robot.bumper_link_idx is not None:
            link_idx = robot.bumper_link_idx
        elif hasattr(robot, 'contact_link_idx') and robot.contact_link_idx is not None:
            link_idx = robot.contact_link_idx
        
        # Allow multiple contacts for wheel robots (bumper geometry can cause this)
        max_contacts = 4 if self.config.model == 'wheel' else 1
        
        force = get_contact_force(
            robot.uid, self.object_uid,
            linkIndexA=link_idx,
            max_contacts=max_contacts
        )
        return force
    
    def run_test(self, gui=True) -> Dict:
        """Run the comprehensive test.
        
        Returns
        -------
        dict
            Per-robot metrics and summary.
        """
        n_steps = int(self.config.duration / TIMESTEP)
        step_count = 0
        t = 0.0
        
        print(f"\nRunning test for {self.config.duration}s...")
        print(f"  Perturbation: {self.config.perturbation}")
        print(f"  Controller: {self.config.controller}")
        
        for step in range(n_steps):
            # Apply perturbation
            apply_perturbation(self.object_uid, t, self.config.perturbation)
            
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Get object state
                obj_state = self._get_object_state()
                
                # Record object history
                self.object_history.times.append(t)
                self.object_history.positions.append(obj_state['position'].copy())
                self.object_history.orientations.append(obj_state['orientation'])
                self.object_history.velocities.append(obj_state['velocity'].copy())
                self.object_history.angular_velocities.append(obj_state['angular_velocity'])
                
                # Update each robot
                for name, robot in self.robots.items():
                    controller = self.controllers[name]
                    history = self.robot_histories[name]
                    
                    # Get robot state
                    robot_pos, robot_heading, robot_vel = self._get_robot_state(robot)
                    
                    # Get contact force
                    contact_force = self._get_contact_force(robot)
                    force_mag = np.linalg.norm(contact_force[:2])
                    in_contact = force_mag > 0.5
                    
                    # Get target point
                    target_point = controller.get_target_point(
                        obj_state['position'], obj_state['orientation']
                    )
                    
                    # Calculate position error
                    bumper_pos = robot.get_contact_position()[:2] if hasattr(robot, 'get_contact_position') else robot_pos
                    position_error = np.linalg.norm(bumper_pos - target_point)
                    
                    # Compute velocity command
                    if self.config.controller == 'velocity':
                        cmd_vel = controller.compute_robot_velocity(
                            robot_pos, robot_heading,
                            obj_state['position'], obj_state['orientation'],
                            obj_state['velocity'], obj_state['angular_velocity']
                        )
                    else:
                        cmd_vel = controller.compute_robot_velocity(
                            robot_pos, robot_heading,
                            obj_state['position'], obj_state['orientation'],
                            obj_state['velocity'], obj_state['angular_velocity'],
                            measured_force=contact_force[:2] if in_contact else None
                        )
                    
                    # For diff-drive, convert 3-DOF to 2-DOF command if needed
                    if self.config.kinematics == 'diffdrive' and len(cmd_vel) == 3:
                        # Use vx as forward velocity (body frame)
                        v_forward = cmd_vel[0] * np.cos(robot_heading) + cmd_vel[1] * np.sin(robot_heading)
                        cmd_vel_dd = np.array([v_forward, cmd_vel[2]])
                        robot.command_velocity(cmd_vel_dd)
                    else:
                        robot.command_velocity(cmd_vel)
                    
                    # Record history
                    history.times.append(t)
                    history.positions.append(robot_pos.copy())
                    history.headings.append(robot_heading)
                    history.velocities.append(robot_vel.copy())
                    history.cmd_velocities.append(cmd_vel.copy())
                    history.in_contact.append(in_contact)
                    history.contact_forces.append(force_mag)
                    history.position_errors.append(position_error)
                    history.target_points.append(target_point.copy())
                    
                    # Record wheel velocities if applicable
                    if is_wheel_robot(robot):
                        wheel_vel = get_wheel_velocities(robot)
                        cmd_wheel_vel = get_command_wheel_velocities(robot)
                        history.wheel_velocities.append(wheel_vel.copy() if wheel_vel is not None else np.array([]))
                        history.cmd_wheel_velocities.append(cmd_wheel_vel.copy() if cmd_wheel_vel is not None else np.array([]))
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.3)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate per-robot and summary metrics."""
        metrics = {
            'config': self.config.to_dict(),
            'robots': {},
            'summary': {},
        }
        
        # Per-robot metrics
        total_contact_ratio = 0
        total_contact_losses = 0
        all_errors = []
        
        for name, history in self.robot_histories.items():
            in_contact = np.array(history.in_contact)
            contact_forces = np.array(history.contact_forces)
            position_errors = np.array(history.position_errors)
            
            # Contact ratio
            contact_ratio = np.mean(in_contact)
            total_contact_ratio += contact_ratio
            
            # Contact losses
            contact_losses = 0
            was_in_contact = False
            for c in in_contact:
                if was_in_contact and not c:
                    contact_losses += 1
                was_in_contact = c
            total_contact_losses += contact_losses
            
            # Position error (when in contact)
            if np.any(in_contact):
                error_in_contact = position_errors[in_contact]
                error_rmse = np.sqrt(np.mean(error_in_contact**2))
                error_max = np.max(error_in_contact)
                all_errors.extend(error_in_contact.tolist())
            else:
                error_rmse = np.nan
                error_max = np.nan
            
            # Force stats (when in contact)
            if np.any(in_contact):
                force_in_contact = contact_forces[in_contact]
                force_mean = np.mean(force_in_contact)
                force_std = np.std(force_in_contact)
            else:
                force_mean = 0.0
                force_std = 0.0
            
            metrics['robots'][name] = {
                't_param': history.t_param,
                'contact_ratio': float(contact_ratio),
                'contact_losses': int(contact_losses),
                'position_error_rmse': float(error_rmse) if not np.isnan(error_rmse) else None,
                'position_error_max': float(error_max) if not np.isnan(error_max) else None,
                'force_mean': float(force_mean),
                'force_std': float(force_std),
            }
        
        # Summary metrics
        n = len(self.robot_histories)
        metrics['summary'] = {
            'num_robots': n,
            'avg_contact_ratio': float(total_contact_ratio / n),
            'total_contact_losses': int(total_contact_losses),
            'overall_position_error_rmse': float(np.sqrt(np.mean(np.array(all_errors)**2))) if all_errors else None,
        }
        
        return metrics
    
    def save_results(self, metrics: Dict, save_dir: str):
        """Save test results to directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = save_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)
        print(f"Saved config to {config_path}")
        
        # Save metrics
        metrics_path = save_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
        
        # Generate plots
        self._generate_plots(save_path)
    
    def _generate_plots(self, save_path: Path):
        """Generate all plots."""
        # Summary plot
        self._plot_summary(save_path / "summary.png")
        
        # Per-robot plots
        for name, history in self.robot_histories.items():
            robot = self.robots[name]
            self._plot_robot(history, robot, save_path / f"robot_{name}.png")
    
    def _plot_summary(self, save_path: Path):
        """Generate summary plot with all robots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Contact Maintenance Test Summary\n'
                    f'{self.config.kinematics} | {self.config.model} | {self.config.controller} | '
                    f'{self.config.num_robots} robots', fontsize=14)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.robot_histories)))
        
        # Trajectories
        ax = axes[0, 0]
        obj_pos = np.array(self.object_history.positions)
        ax.plot(obj_pos[:, 0], obj_pos[:, 1], 'k-', linewidth=2, label='Object')
        for (name, history), color in zip(self.robot_histories.items(), colors):
            pos = np.array(history.positions)
            ax.plot(pos[:, 0], pos[:, 1], '-', color=color, linewidth=1.5, 
                   label=f'{name} (t={history.t_param:.2f})')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectories')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Contact status
        ax = axes[0, 1]
        times = np.array(self.object_history.times)
        for (name, history), color in zip(self.robot_histories.items(), colors):
            contact = np.array(history.in_contact).astype(float)
            ax.plot(times, contact + list(self.robot_histories.keys()).index(name) * 0.1, 
                   '-', color=color, linewidth=1.5, label=name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact Status (offset)')
        ax.set_title('Contact Status per Robot')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Position errors
        ax = axes[0, 2]
        for (name, history), color in zip(self.robot_histories.items(), colors):
            errors = np.array(history.position_errors) * 100  # to cm
            ax.plot(times, errors, '-', color=color, linewidth=1.5, label=name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (cm)')
        ax.set_title('Position Error (robot to target point)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Contact forces
        ax = axes[1, 0]
        for (name, history), color in zip(self.robot_histories.items(), colors):
            forces = np.array(history.contact_forces)
            ax.plot(times, forces, '-', color=color, linewidth=1.5, label=name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Contact Force Magnitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Object motion
        ax = axes[1, 1]
        ax.plot(times, obj_pos[:, 0], 'r-', linewidth=1.5, label='x')
        ax.plot(times, obj_pos[:, 1], 'g-', linewidth=1.5, label='y')
        obj_orn = np.array(self.object_history.orientations)
        ax.plot(times, obj_orn, 'b-', linewidth=1.5, label='θ (rad)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.set_title('Object State')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Metrics text
        ax = axes[1, 2]
        ax.axis('off')
        metrics = self.calculate_metrics()
        text = "TEST METRICS\n\n"
        text += f"Avg Contact Ratio: {metrics['summary']['avg_contact_ratio']*100:.1f}%\n"
        text += f"Total Contact Losses: {metrics['summary']['total_contact_losses']}\n"
        if metrics['summary'].get('overall_position_error_rmse'):
            text += f"Overall Error RMSE: {metrics['summary']['overall_position_error_rmse']*100:.2f} cm\n"
        text += "\nPer-Robot:\n"
        for name, m in metrics['robots'].items():
            text += f"  {name}: {m['contact_ratio']*100:.0f}% contact"
            if m.get('position_error_rmse'):
                text += f", err={m['position_error_rmse']*100:.1f}cm"
            text += "\n"
        ax.text(0.1, 0.9, text, fontsize=10, family='monospace',
               verticalalignment='top', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved summary plot to {save_path}")
    
    def _plot_robot(self, history: RobotHistory, robot, save_path: Path):
        """Generate detailed plot for a single robot."""
        has_wheels = is_wheel_robot(robot)
        
        # Adjust grid based on whether we have wheel data
        if has_wheels:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        else:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        fig.suptitle(f'Robot {history.name} | t_param={history.t_param:.2f}\n'
                    f'{self.config.kinematics} | {self.config.model}', fontsize=14)
        
        times = np.array(history.times)
        positions = np.array(history.positions)
        target_points = np.array(history.target_points)
        headings = np.array(history.headings)
        velocities = np.array(history.velocities)
        cmd_velocities = np.array(history.cmd_velocities)
        in_contact = np.array(history.in_contact)
        contact_forces = np.array(history.contact_forces)
        position_errors = np.array(history.position_errors)
        
        obj_pos = np.array(self.object_history.positions)
        
        # Trajectory
        ax = axes[0, 0]
        ax.plot(obj_pos[:, 0], obj_pos[:, 1], 'g-', linewidth=2, label='Object')
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, label='Robot')
        ax.plot(target_points[:, 0], target_points[:, 1], 'r--', linewidth=1, alpha=0.7, label='Target')
        ax.plot(positions[0, 0], positions[0, 1], 'bo', markersize=8, label='Start')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Contact status
        ax = axes[0, 1]
        ax.fill_between(times, 0, in_contact.astype(float), alpha=0.5, color='green')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact')
        ax.set_title('Contact Status')
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)
        
        # Contact force
        ax = axes[0, 2]
        ax.plot(times, contact_forces, 'r-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Contact Force')
        ax.grid(True, alpha=0.3)
        
        # Position error
        ax = axes[1, 0]
        ax.plot(times, position_errors * 100, 'b-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (cm)')
        ax.set_title('Position Error')
        ax.grid(True, alpha=0.3)
        
        # Velocity commands
        ax = axes[1, 1]
        if cmd_velocities.shape[1] >= 3:
            ax.plot(times, cmd_velocities[:, 0], 'r-', linewidth=1.5, label='cmd vx')
            ax.plot(times, cmd_velocities[:, 1], 'g-', linewidth=1.5, label='cmd vy')
            ax.plot(times, cmd_velocities[:, 2], 'b-', linewidth=1.5, label='cmd ω')
        else:
            ax.plot(times, cmd_velocities[:, 0], 'r-', linewidth=1.5, label='cmd v')
            ax.plot(times, cmd_velocities[:, 1], 'b-', linewidth=1.5, label='cmd ω')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity')
        ax.set_title('Velocity Commands')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Heading
        ax = axes[1, 2]
        ax.plot(times, np.degrees(headings), 'purple', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (deg)')
        ax.set_title('Robot Heading')
        ax.grid(True, alpha=0.3)
        
        # Wheel velocities (if applicable)
        if has_wheels and len(history.wheel_velocities) > 0:
            wheel_vels = np.array(history.wheel_velocities)
            cmd_wheel_vels = np.array(history.cmd_wheel_velocities)
            
            ax = axes[2, 0]
            n_wheels = wheel_vels.shape[1] if len(wheel_vels.shape) > 1 else 0
            if n_wheels > 0:
                wheel_labels = ['FR', 'FL', 'RL', 'RR'] if n_wheels == 4 else ['L', 'R']
                for i in range(min(n_wheels, len(wheel_labels))):
                    ax.plot(times, wheel_vels[:, i], '-', linewidth=1.5, label=f'{wheel_labels[i]}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Wheel Vel (rad/s)')
            ax.set_title('Actual Wheel Velocities')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            ax = axes[2, 1]
            if n_wheels > 0:
                for i in range(min(n_wheels, len(wheel_labels))):
                    ax.plot(times, cmd_wheel_vels[:, i], '--', linewidth=1.5, label=f'{wheel_labels[i]} cmd')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Wheel Vel (rad/s)')
            ax.set_title('Commanded Wheel Velocities')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            # Actual velocity
            ax = axes[2, 0]
            if velocities.shape[1] >= 3:
                ax.plot(times, velocities[:, 0], 'r-', linewidth=1.5, label='vx')
                ax.plot(times, velocities[:, 1], 'g-', linewidth=1.5, label='vy')
                ax.plot(times, velocities[:, 2], 'b-', linewidth=1.5, label='ω')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity')
            ax.set_title('Actual Velocity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Distance to object
            ax = axes[2, 1]
            distance = np.linalg.norm(positions - obj_pos, axis=1)
            ax.plot(times, distance * 100, 'purple', linewidth=1.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Distance (cm)')
            ax.set_title('Robot-Object Distance')
            ax.grid(True, alpha=0.3)
        
        # Metrics
        ax = axes[2, 2]
        ax.axis('off')
        metrics = self.calculate_metrics()
        m = metrics['robots'][history.name]
        
        # Handle None values for error metrics
        err_rmse = f"{m['position_error_rmse']*100:.2f} cm" if m['position_error_rmse'] else "N/A"
        err_max = f"{m['position_error_max']*100:.2f} cm" if m['position_error_max'] else "N/A"
        
        text = f"""ROBOT {history.name} METRICS

                t_param: {history.t_param:.2f}

                Contact Ratio: {m['contact_ratio']*100:.1f}%
                Contact Losses: {m['contact_losses']}

                Position Error:
                RMSE: {err_rmse}
                Max:  {err_max}

                Contact Force:
                Mean: {m['force_mean']:.2f} N
                Std:  {m['force_std']:.2f} N
                """
        ax.text(0.1, 0.9, text, fontsize=11, family='monospace',
               verticalalignment='top', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved robot plot to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def parse_t_params(t_params_str: str, num_robots: int) -> List[float]:
    """Parse t_params from string or generate defaults."""
    if t_params_str:
        params = [float(x.strip()) for x in t_params_str.split(',')]
        if len(params) < num_robots:
            # Extend with uniform distribution
            remaining = num_robots - len(params)
            for i in range(remaining):
                params.append((i + 1) / (remaining + 1))
        return params[:num_robots]
    else:
        # Generate uniform distribution
        return [i / num_robots for i in range(num_robots)]


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Contact Maintenance Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Single holonomic dummy robot:
                python comprehensive_contact_test.py --num-robots 1 --kinematics holonomic --model dummy

                # Multi-robot (3) with wheel model:
                python comprehensive_contact_test.py --num-robots 3 --model wheel --t-params 0.1,0.4,0.7

                # Diff-drive with wrench controller:
                python comprehensive_contact_test.py --kinematics diffdrive --controller wrench

                # Full test with output:
                python comprehensive_contact_test.py --num-robots 3 --save-dir /tmp/results/

                Test Matrix:
                - Robot Count: 1 (single) or N (multi)
                - Model: dummy (direct velocity) or wheel (realistic physics)
                - Kinematics: holonomic or diffdrive
                - Controller: velocity or wrench
                """
            )
    parser.add_argument("--num-robots", "-n", type=int, default=1,
                       help="Number of robots (default: 1)")
    parser.add_argument("--kinematics", "-k", default="holonomic",
                       choices=['holonomic', 'diffdrive'],
                       help="Kinematics type (default: holonomic)")
    parser.add_argument("--model", "-m", default="dummy",
                       choices=['dummy', 'wheel'],
                       help="Robot model (default: dummy)")
    parser.add_argument("--controller", "-c", default="velocity",
                       choices=['velocity', 'wrench'],
                       help="Controller type (default: velocity)")
    parser.add_argument("--t-params", type=str, default=None,
                       help="Comma-separated t_params for each robot (default: uniform)")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                       help="Test duration in seconds (default: 10.0)")
    parser.add_argument("--perturbation", "-p", default="pulse",
                       choices=['pulse', 'continuous', 'none'],
                       help="Perturbation type (default: pulse)")
    parser.add_argument("--object", type=str, default="rectangle",
                       help="Object shape (default: rectangle)")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run without GUI")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save results")
    args = parser.parse_args()
    
    # Parse t_params
    t_params = parse_t_params(args.t_params, args.num_robots)
    
    # Create config
    config = TestConfig(
        num_robots=args.num_robots,
        kinematics=args.kinematics,
        model=args.model,
        controller=args.controller,
        t_params=t_params,
        duration=args.duration,
        perturbation=args.perturbation,
        object_shape=args.object,
        save_dir=args.save_dir,
    )
    
    # Default save dir if headless
    if args.no_gui and args.save_dir is None:
        config.save_dir = f"/tmp/contact_test_{config.get_name()}"
    
    # Print configuration
    print("="*60)
    print("  COMPREHENSIVE CONTACT MAINTENANCE TEST")
    print("="*60)
    print(f"  Robots: {config.num_robots}")
    print(f"  Kinematics: {config.kinematics}")
    print(f"  Model: {config.model}")
    print(f"  Controller: {config.controller}")
    print(f"  t_params: {config.t_params}")
    print(f"  Duration: {config.duration}s")
    print(f"  Perturbation: {config.perturbation}")
    print(f"  Object: {config.object_shape}")
    print(f"  Mode: {'Headless' if args.no_gui else 'GUI'}")
    if config.save_dir:
        print(f"  Save Dir: {config.save_dir}")
    print("="*60)
    
    # Setup PyBullet
    print("\nInitializing PyBullet...")
    ground = setup_pybullet(gui=not args.no_gui)
    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)
    
    # Create and run test
    test = ComprehensiveContactTest(config)
    metrics = test.run_test(gui=not args.no_gui)
    
    # Print results
    print("\n" + "="*60)
    print("  TEST RESULTS")
    print("="*60)
    print(f"  Avg Contact Ratio: {metrics['summary']['avg_contact_ratio']*100:.1f}%")
    print(f"  Total Contact Losses: {metrics['summary']['total_contact_losses']}")
    if metrics['summary']['overall_position_error_rmse']:
        print(f"  Overall Error RMSE: {metrics['summary']['overall_position_error_rmse']*100:.2f} cm")
    print("-"*60)
    for name, m in metrics['robots'].items():
        print(f"  {name}: contact={m['contact_ratio']*100:.0f}%", end="")
        if m.get('position_error_rmse'):
            print(f", err_rmse={m['position_error_rmse']*100:.1f}cm", end="")
        print()
    print("="*60)
    
    # Save results
    if config.save_dir:
        test.save_results(metrics, config.save_dir)
    
    # Keep open if GUI
    if not args.no_gui:
        print("\nPress Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()

