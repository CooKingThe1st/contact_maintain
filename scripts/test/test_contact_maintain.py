#!/usr/bin/env python3
"""
Test Contact Maintenance Controller

Tests the contact maintenance controllers:
1. InstantVelocityMatcher - matches robot velocity to boundary point velocity
2. WrenchTrackingController - applies desired wrench through contact point

Uses the dummy holonomic robot to validate that velocity control is sufficient
for maintaining contact at a specific boundary point.

PyBullet GUI Interaction:
- Left-click + drag: Rotate camera
- Middle-click + drag: Pan camera
- Scroll wheel: Zoom in/out

Usage:
    # Test with InstantVelocityMatcher (default):
    python test_contact_maintain.py --controller velocity
    
    # Test with WrenchTrackingController:
    python test_contact_maintain.py --controller wrench
    
    # Headless mode with plot:
    python test_contact_maintain.py --no-gui --save-plot /tmp/contact_maintain.png
"""
import argparse
import time
from pathlib import Path
import sys
import os

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

from object_utils import create_standard_objects
from contact_maintain.robots import HolonomicRobot
from contact_maintain.object_bridge import generic_to_pybullet
from contact_maintain.contact_maintain_controller import (
    InstantVelocityMatcher, WrenchTrackingController, ContactMaintenanceState
)


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

# Object parameters
OBJECT_SHAPE = 'rectangle'
OBJECT_HEIGHT = 0.1
OBJECT_FRICTION = 0.3

# Robot parameters
ROBOT_START_OFFSET = 0.5  # Distance from object center to start


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
    pyb.changeDynamics(ground, -1, lateralFriction=OBJECT_FRICTION)
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=0,
            cameraPitch=-60,
            cameraTargetPosition=[0, 0, 0]
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    
    return client_id


# ============================================================================
# PERTURBATION FUNCTIONS
# ============================================================================

def apply_perturbation(object_uid, t, perturbation_type='pulse'):
    """Apply perturbation to object to test contact maintenance.
    
    Parameters
    ----------
    object_uid : int
        PyBullet body UID.
    t : float
        Current time.
    perturbation_type : str
        Type of perturbation ('pulse', 'continuous', 'none').
    """
    if perturbation_type == 'none':
        return
    
    if perturbation_type == 'pulse':
        # Apply pulse forces at specific times
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
        # Apply continuous sinusoidal force
        fx = 1.0 * np.sin(0.5 * t)
        fy = 1.0 * np.cos(0.5 * t)
        force = [fx, fy, 0]
        pos, _ = pyb.getBasePositionAndOrientation(object_uid)
        pyb.applyExternalForce(object_uid, -1, force, pos, pyb.WORLD_FRAME)


# ============================================================================
# TEST CLASS
# ============================================================================

class ContactMaintenanceTest:
    """Test contact maintenance controllers."""
    
    def __init__(self, controller_type='velocity', t_param=0.25):
        """Initialize test.
        
        Parameters
        ----------
        controller_type : str
            'velocity' for InstantVelocityMatcher, 'wrench' for WrenchTrackingController
        t_param : float
            Boundary parameter for contact point (0-1).
        """
        self.controller_type = controller_type
        self.t_param = t_param
        
        # Create object
        print(f"\nCreating {OBJECT_SHAPE} object...")
        standard_objects = create_standard_objects()
        self.generic_object = standard_objects[OBJECT_SHAPE]
        
        self.object_uid = generic_to_pybullet(
            self.generic_object,
            height=OBJECT_HEIGHT,
            position=(0, 0, 0),
            color=(0.4, 0.7, 0.4, 1.0)
        )
        pyb.changeDynamics(self.object_uid, -1, lateralFriction=OBJECT_FRICTION)
        
        # Create controller
        print(f"Creating {controller_type} controller at t_param={t_param}...")
        if controller_type == 'velocity':
            self.controller = InstantVelocityMatcher(
                self.generic_object, t_param,
                kp_position=3.0, max_velocity=0.5
            )
        else:
            self.controller = WrenchTrackingController(
                self.generic_object, t_param,
                desired_wrench=np.array([2.0, 0, 0]),
                kp_force=0.2, max_velocity=0.3
            )
        
        # Get initial target point to position robot
        target_point = self.controller.get_target_point(np.zeros(2), 0.0)
        
        # Position robot outside the object, pointing toward it
        robot_offset = target_point + ROBOT_START_OFFSET * target_point / (np.linalg.norm(target_point) + 0.01)
        robot_heading = np.arctan2(-target_point[1], -target_point[0])
        
        print(f"Creating robot at ({robot_offset[0]:.2f}, {robot_offset[1]:.2f})...")
        self.robot = HolonomicRobot(
            str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf"),
            position=robot_offset,
            orientation=robot_heading
        )
        
        # State tracker
        self.state = ContactMaintenanceState()
        self.state.target_t_param = t_param
        
        # History
        self.history = {
            'times': [],
            'robot_pos': [],
            'robot_heading': [],
            'object_pos': [],
            'object_orientation': [],
            'in_contact': [],
            'contact_force': [],
            'position_error': [],
            'target_point': [],
            'cmd_vel': [],
        }
    
    def run_test(self, duration=10.0, perturbation='pulse', gui=True):
        """Run contact maintenance test.
        
        Parameters
        ----------
        duration : float
            Test duration in seconds.
        perturbation : str
            Type of perturbation ('pulse', 'continuous', 'none').
        gui : bool
            Whether to show visualization.
        
        Returns
        -------
        dict
            Test metrics.
        """
        n_steps = int(duration / TIMESTEP)
        step_count = 0
        t = 0.0
        
        print(f"\nRunning {self.controller_type} controller test...")
        print(f"  Duration: {duration}s")
        print(f"  Perturbation: {perturbation}")
        print(f"  Target boundary point: t_param={self.t_param}")
        
        for step in range(n_steps):
            # Apply perturbation
            apply_perturbation(self.object_uid, t, perturbation)
            
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Update state
                self.state.update_from_simulation(
                    self.robot, self.object_uid, self.controller
                )
                
                # Compute velocity command
                if self.controller_type == 'velocity':
                    cmd_vel = self.controller.compute_robot_velocity(
                        self.state.robot_pos,
                        self.state.robot_heading,
                        self.state.object_pos,
                        self.state.object_orientation,
                        self.state.object_velocity,
                        self.state.object_angular_velocity
                    )
                else:
                    cmd_vel = self.controller.compute_robot_velocity(
                        self.state.robot_pos,
                        self.state.robot_heading,
                        self.state.object_pos,
                        self.state.object_orientation,
                        self.state.object_velocity,
                        self.state.object_angular_velocity,
                        measured_force=self.state.contact_force[:2] if self.state.in_contact else None
                    )
                
                # Apply command
                self.robot.command_velocity(cmd_vel)
                
                # Record history
                self.history['times'].append(t)
                self.history['robot_pos'].append(self.state.robot_pos.copy())
                self.history['robot_heading'].append(self.state.robot_heading)
                self.history['object_pos'].append(self.state.object_pos.copy())
                self.history['object_orientation'].append(self.state.object_orientation)
                self.history['in_contact'].append(self.state.in_contact)
                self.history['contact_force'].append(self.state.contact_force_magnitude)
                self.history['position_error'].append(self.state.position_error)
                self.history['target_point'].append(self.state.target_point_world.copy())
                self.history['cmd_vel'].append(cmd_vel.copy())
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.5)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate test metrics."""
        times = np.array(self.history['times'])
        in_contact = np.array(self.history['in_contact'])
        contact_force = np.array(self.history['contact_force'])
        position_error = np.array(self.history['position_error'])
        
        # Contact statistics
        contact_ratio = np.mean(in_contact)
        
        # Find contact loss events
        contact_losses = 0
        was_in_contact = False
        for c in in_contact:
            if was_in_contact and not c:
                contact_losses += 1
            was_in_contact = c
        
        # Position error statistics (only when in contact)
        if np.any(in_contact):
            error_in_contact = position_error[in_contact]
            error_rmse = np.sqrt(np.mean(error_in_contact**2))
            error_max = np.max(error_in_contact)
        else:
            error_rmse = np.nan
            error_max = np.nan
        
        # Force statistics (only when in contact)
        if np.any(in_contact):
            force_in_contact = contact_force[in_contact]
            force_mean = np.mean(force_in_contact)
            force_std = np.std(force_in_contact)
        else:
            force_mean = 0.0
            force_std = 0.0
        
        metrics = {
            'contact_ratio': contact_ratio,
            'contact_losses': contact_losses,
            'position_error_rmse': error_rmse,
            'position_error_max': error_max,
            'force_mean': force_mean,
            'force_std': force_std,
        }
        
        return metrics
    
    def plot_results(self, save_path=None, show=False):
        """Plot test results."""
        times = np.array(self.history['times'])
        robot_pos = np.array(self.history['robot_pos'])
        object_pos = np.array(self.history['object_pos'])
        in_contact = np.array(self.history['in_contact'])
        contact_force = np.array(self.history['contact_force'])
        position_error = np.array(self.history['position_error'])
        target_point = np.array(self.history['target_point'])
        cmd_vel = np.array(self.history['cmd_vel'])
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Contact Maintenance Test: {self.controller_type.title()} Controller\n'
                    f't_param={self.t_param}', fontsize=14)
        
        # Trajectory
        ax = axes[0, 0]
        ax.plot(robot_pos[:, 0], robot_pos[:, 1], 'b-', linewidth=1.5, label='Robot')
        ax.plot(object_pos[:, 0], object_pos[:, 1], 'g-', linewidth=1.5, label='Object')
        ax.plot(target_point[:, 0], target_point[:, 1], 'r--', linewidth=1, alpha=0.7, label='Target Point')
        ax.plot(robot_pos[0, 0], robot_pos[0, 1], 'bo', markersize=10, label='Robot Start')
        ax.plot(object_pos[0, 0], object_pos[0, 1], 'go', markersize=10, label='Object Start')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectories')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Contact status
        ax = axes[0, 1]
        ax.fill_between(times, 0, in_contact.astype(float), alpha=0.5, color='green', label='In Contact')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Contact')
        ax.set_title('Contact Status')
        ax.set_ylim(-0.1, 1.5)
        ax.grid(True, alpha=0.3)
        
        # Contact force
        ax = axes[0, 2]
        ax.plot(times, contact_force, 'r-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Contact Force Magnitude')
        ax.grid(True, alpha=0.3)
        
        # Position error
        ax = axes[1, 0]
        ax.plot(times, position_error * 100, 'b-', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (cm)')
        ax.set_title('Position Error (robot to target point)')
        ax.grid(True, alpha=0.3)
        
        # Velocity commands
        ax = axes[1, 1]
        ax.plot(times, cmd_vel[:, 0], 'r-', linewidth=1.5, label='vx')
        ax.plot(times, cmd_vel[:, 1], 'g-', linewidth=1.5, label='vy')
        ax.plot(times, cmd_vel[:, 2], 'b-', linewidth=1.5, label='omega')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity')
        ax.set_title('Velocity Commands')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Object motion
        ax = axes[1, 2]
        ax.plot(times, object_pos[:, 0], 'r-', linewidth=1.5, label='x')
        ax.plot(times, object_pos[:, 1], 'g-', linewidth=1.5, label='y')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.set_title('Object Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Robot-object distance
        ax = axes[2, 0]
        distance = np.linalg.norm(robot_pos - object_pos, axis=1)
        ax.plot(times, distance * 100, 'purple', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (cm)')
        ax.set_title('Robot-Object Distance')
        ax.grid(True, alpha=0.3)
        
        # Error statistics text
        ax = axes[2, 1]
        ax.axis('off')
        metrics = self.calculate_metrics()
        metrics_text = f"""
            TEST METRICS

            Contact Ratio: {metrics['contact_ratio']*100:.1f}%
            Contact Losses: {metrics['contact_losses']}

            Position Error (when in contact):
            RMSE: {metrics['position_error_rmse']*100:.2f} cm
            Max:  {metrics['position_error_max']*100:.2f} cm

            Contact Force:
            Mean: {metrics['force_mean']:.2f} N
            Std:  {metrics['force_std']:.2f} N
            """
        ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
               verticalalignment='center')
        
        # Contact force histogram
        ax = axes[2, 2]
        force_in_contact = contact_force[in_contact] if np.any(in_contact) else [0]
        ax.hist(force_in_contact, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Force (N)')
        ax.set_ylabel('Count')
        ax.set_title('Contact Force Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        # Show plot if requested
        if show:
            try:
                plt.switch_backend('TkAgg')
                plt.show()
            except Exception as e:
                print(f"Could not display plot interactively: {e}")
        
        return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Contact Maintenance Controllers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test InstantVelocityMatcher with GUI:
  python test_contact_maintain.py --controller velocity
  
  # Test WrenchTrackingController headless:
  python test_contact_maintain.py --controller wrench --no-gui
  
  # Test with different boundary point:
  python test_contact_maintain.py --t-param 0.5 --duration 15

PyBullet GUI Interaction:
  - Left-click + drag: Rotate camera
  - Middle-click + drag: Pan camera  
  - Scroll wheel: Zoom in/out
"""
    )
    parser.add_argument("--controller", "-c", default="velocity",
                       choices=['velocity', 'wrench'],
                       help="Controller type (default: velocity)")
    parser.add_argument("--t-param", type=float, default=0.25,
                       help="Boundary parameter for contact point (default: 0.25)")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                       help="Test duration in seconds (default: 10.0)")
    parser.add_argument("--perturbation", "-p", default="pulse",
                       choices=['pulse', 'continuous', 'none'],
                       help="Perturbation type (default: pulse)")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run without GUI (headless mode)")
    parser.add_argument("--save-plot", type=str, default=None,
                       help="Save plot to file")
    args = parser.parse_args()
    
    # Determine save path for headless mode
    if args.no_gui and args.save_plot is None:
        args.save_plot = f"/tmp/contact_maintain_{args.controller}.png"
    
    print("="*60)
    print("  CONTACT MAINTENANCE TEST")
    print("="*60)
    print(f"  Controller: {args.controller}")
    print(f"  t_param: {args.t_param}")
    print(f"  Duration: {args.duration}s")
    print(f"  Perturbation: {args.perturbation}")
    print(f"  Mode: {'Headless' if args.no_gui else 'GUI'}")
    if args.save_plot:
        print(f"  Save Plot: {args.save_plot}")
    print("="*60)
    
    # Setup PyBullet
    print("\nInitializing PyBullet...")
    setup_pybullet(gui=not args.no_gui)
    
    # Create test
    test = ContactMaintenanceTest(
        controller_type=args.controller,
        t_param=args.t_param
    )
    
    # Run test
    metrics = test.run_test(
        duration=args.duration,
        perturbation=args.perturbation,
        gui=not args.no_gui
    )
    
    # Print results
    print("\n" + "="*60)
    print("  TEST RESULTS")
    print("="*60)
    print(f"  Contact Ratio:        {metrics['contact_ratio']*100:.1f}%")
    print(f"  Contact Losses:       {metrics['contact_losses']}")
    print(f"  Position Error RMSE:  {metrics['position_error_rmse']*100:.2f} cm")
    print(f"  Position Error Max:   {metrics['position_error_max']*100:.2f} cm")
    print(f"  Force Mean:           {metrics['force_mean']:.2f} N")
    print(f"  Force Std:            {metrics['force_std']:.2f} N")
    print("="*60)
    
    # Plot results
    test.plot_results(save_path=args.save_plot, show=not args.no_gui)
    
    # Keep PyBullet open for inspection if GUI mode
    if not args.no_gui:
        print("\nPress Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
