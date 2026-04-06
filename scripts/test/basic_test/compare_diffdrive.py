#!/usr/bin/env python3
"""
Compare Differential Drive Robots Script

Compares the dummy differential-drive robot (direct velocity control) against
the realistic diff-drive robot with 2 wheel control.

Tests:
1. Straight line motion
2. Pure rotation  
3. Arc trajectory
4. S-curve trajectory

Shows:
- Trajectory comparison
- Velocity tracking error
- Wheel velocities (for wheel-based robot)

NOTE: Differential drive cannot strafe laterally (non-holonomic constraint).

PyBullet GUI Interaction:
- Left-click + drag: Rotate camera
- Middle-click + drag: Pan camera
- Scroll wheel: Zoom in/out

Usage:
    # Run with GUI (default):
    python compare_diffdrive.py --trajectory arc
    
    # Run headless with plot output:
    python compare_diffdrive.py --trajectory scurve --no-gui --save-plot /tmp/compare_dd.png
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

from contact_maintain.robots import DifferentialDriveRobot
from contact_maintain.diffdrive_wheel_robot import DiffDriveWheelRobot


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)


# ============================================================================
# TRAJECTORY GENERATORS (for differential drive: v, omega)
# ============================================================================

def straight_trajectory(t, speed=0.3):
    """Move straight forward."""
    return np.array([speed, 0])


def pure_rotation_trajectory(t, omega=1.0):
    """Rotate in place."""
    return np.array([0, omega])


def arc_trajectory(t, v=0.3, omega=0.3):
    """Move in an arc (constant curvature)."""
    return np.array([v, omega])


def scurve_trajectory(t, v=0.3, period=4.0, omega_max=0.5):
    """Move in an S-curve pattern."""
    omega = omega_max * np.sin(2 * np.pi * t / period)
    return np.array([v, omega])


def zigzag_trajectory(t, v=0.3, period=2.0, omega_max=1.0):
    """Move in a zigzag pattern."""
    omega = omega_max * np.sign(np.sin(2 * np.pi * t / period))
    return np.array([v, omega])


TRAJECTORIES = {
    'straight': straight_trajectory,
    'rotation': pure_rotation_trajectory,
    'arc': arc_trajectory,
    'scurve': scurve_trajectory,
    'zigzag': zigzag_trajectory,
}


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
    pyb.changeDynamics(ground, -1, lateralFriction=0.5)
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=0,
            cameraPitch=-60,
            cameraTargetPosition=[0, 0, 0]
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    
    return client_id


# ============================================================================
# ROBOT COMPARISON TEST
# ============================================================================

class DiffDriveComparisonTest:
    """Test comparing dummy vs wheel-based differential drive robot."""
    
    def __init__(self):
        # Create robots at different positions
        self.dummy_robot = DifferentialDriveRobot(
            str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf"),
            position=(-1.0, 0),
            orientation=0.0
        )
        
        self.wheel_robot = DiffDriveWheelRobot(
            str(Path(pkg_path) / "urdf" / "diffdrive_wheel_robot.urdf"),
            position=(1.0, 0),
            orientation=0.0
        )
        
        # History
        self.history = {
            'times': [],
            'cmd_vel': [],
            # Dummy robot
            'dummy_pos': [],
            'dummy_heading': [],
            'dummy_vel': [],
            # Wheel robot
            'wheel_pos': [],
            'wheel_heading': [],
            'wheel_vel': [],
            'wheel_wheel_vel': [],
            'wheel_cmd_wheel_vel': [],
        }
    
    def reset(self):
        """Reset both robots."""
        self.dummy_robot.reset(position=(-1.0, 0), orientation=0.0)
        self.wheel_robot.reset(position=(1.0, 0), orientation=0.0)
        
        for key in self.history:
            self.history[key] = []
    
    def run_test(self, trajectory_name='arc', duration=10.0, gui=True):
        """Run comparison test."""
        self.reset()
        
        trajectory_fn = TRAJECTORIES.get(trajectory_name, arc_trajectory)
        n_steps = int(duration / TIMESTEP)
        
        print(f"\nRunning {trajectory_name} trajectory for {duration}s")
        print("  Dummy robot (blue, uses holonomic URDF) at x=-1")
        print("  Wheel robot (actual wheels) at x=+1")
        
        step_count = 0
        t = 0.0
        
        for step in range(n_steps):
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Get trajectory command (v, omega)
                cmd_vel = trajectory_fn(t)
                
                # Apply to both robots
                self.dummy_robot.command_velocity(cmd_vel)
                self.wheel_robot.command_velocity(cmd_vel)
                
                # Record state
                dummy_pos, dummy_heading, dummy_vel = self.dummy_robot.get_state()
                wheel_pos, wheel_heading, wheel_vel = self.wheel_robot.get_state()
                wheel_wheel_vel = self.wheel_robot.get_wheel_velocities()
                
                self.history['times'].append(t)
                self.history['cmd_vel'].append(cmd_vel.copy())
                self.history['dummy_pos'].append(dummy_pos.copy())
                self.history['dummy_heading'].append(dummy_heading)
                self.history['dummy_vel'].append(dummy_vel.copy())
                self.history['wheel_pos'].append(wheel_pos.copy())
                self.history['wheel_heading'].append(wheel_heading)
                self.history['wheel_vel'].append(wheel_vel.copy())
                self.history['wheel_wheel_vel'].append(wheel_wheel_vel.copy())
                self.history['wheel_cmd_wheel_vel'].append(
                    self.wheel_robot.last_wheel_speeds.copy())
            
            pyb.stepSimulation()
            t += TIMESTEP
            step_count += 1
            
            if gui:
                time.sleep(TIMESTEP * 0.5)
        
        return self.calculate_errors()
    
    def calculate_errors(self):
        """Calculate comparison metrics."""
        times = np.array(self.history['times'])
        cmd_vel = np.array(self.history['cmd_vel'])
        
        dummy_pos = np.array(self.history['dummy_pos'])
        dummy_vel = np.array(self.history['dummy_vel'])
        dummy_heading = np.array(self.history['dummy_heading'])
        
        wheel_pos = np.array(self.history['wheel_pos'])
        wheel_vel = np.array(self.history['wheel_vel'])
        wheel_heading = np.array(self.history['wheel_heading'])
        
        # Forward velocity (vx in body frame) - need to compute from world vel
        dummy_v = np.array([
            np.cos(h) * v[0] + np.sin(h) * v[1] 
            for h, v in zip(dummy_heading, dummy_vel)
        ])
        wheel_v = np.array([
            np.cos(h) * v[0] + np.sin(h) * v[1] 
            for h, v in zip(wheel_heading, wheel_vel)
        ])
        
        # Velocity tracking error
        dummy_v_error = np.abs(dummy_v - cmd_vel[:, 0])
        wheel_v_error = np.abs(wheel_v - cmd_vel[:, 0])
        dummy_omega_error = np.abs(dummy_vel[:, 2] - cmd_vel[:, 1])
        wheel_omega_error = np.abs(wheel_vel[:, 2] - cmd_vel[:, 1])
        
        # Trajectory deviation (relative to starting offset)
        dummy_traj = dummy_pos - dummy_pos[0]
        wheel_traj = wheel_pos - wheel_pos[0]
        traj_diff = np.linalg.norm(dummy_traj - wheel_traj, axis=1)
        
        # Heading deviation
        heading_diff = np.abs(np.arctan2(
            np.sin(dummy_heading - wheel_heading),
            np.cos(dummy_heading - wheel_heading)
        ))
        
        errors = {
            'dummy_v_rmse': np.sqrt(np.mean(dummy_v_error**2)),
            'wheel_v_rmse': np.sqrt(np.mean(wheel_v_error**2)),
            'dummy_omega_rmse': np.sqrt(np.mean(dummy_omega_error**2)),
            'wheel_omega_rmse': np.sqrt(np.mean(wheel_omega_error**2)),
            'trajectory_diff_rmse': np.sqrt(np.mean(traj_diff**2)),
            'trajectory_diff_max': np.max(traj_diff),
            'heading_diff_max': np.max(heading_diff),
        }
        
        return errors
    
    def plot_results(self, save_path=None, show=False):
        """Plot comparison results."""
        times = np.array(self.history['times'])
        cmd_vel = np.array(self.history['cmd_vel'])
        
        dummy_pos = np.array(self.history['dummy_pos'])
        dummy_heading = np.array(self.history['dummy_heading'])
        dummy_vel = np.array(self.history['dummy_vel'])
        
        wheel_pos = np.array(self.history['wheel_pos'])
        wheel_heading = np.array(self.history['wheel_heading'])
        wheel_vel = np.array(self.history['wheel_vel'])
        wheel_wheel_vel = np.array(self.history['wheel_wheel_vel'])
        wheel_cmd_wheel_vel = np.array(self.history['wheel_cmd_wheel_vel'])
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Diff-Drive Comparison: Dummy (blue) vs Wheel-based (red)', fontsize=14)
        
        # Trajectories (relative)
        ax = axes[0, 0]
        dummy_traj = dummy_pos - dummy_pos[0]
        wheel_traj = wheel_pos - wheel_pos[0]
        ax.plot(dummy_traj[:, 0], dummy_traj[:, 1], 'b-', linewidth=2, label='Dummy')
        ax.plot(wheel_traj[:, 0], wheel_traj[:, 1], 'r--', linewidth=2, label='Wheel')
        ax.plot(0, 0, 'go', markersize=10, label='Start')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Relative Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Forward velocity
        ax = axes[0, 1]
        ax.plot(times, cmd_vel[:, 0], 'k-', linewidth=2, label='Command v')
        # Compute forward velocity in body frame
        dummy_v = [np.cos(h) * v[0] + np.sin(h) * v[1] 
                   for h, v in zip(dummy_heading, dummy_vel)]
        wheel_v = [np.cos(h) * v[0] + np.sin(h) * v[1] 
                   for h, v in zip(wheel_heading, wheel_vel)]
        ax.plot(times, dummy_v, 'b-', linewidth=1.5, alpha=0.8, label='Dummy')
        ax.plot(times, wheel_v, 'r--', linewidth=1.5, alpha=0.8, label='Wheel')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Forward Velocity (m/s)')
        ax.set_title('Forward Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Angular velocity
        ax = axes[0, 2]
        ax.plot(times, cmd_vel[:, 1], 'k-', linewidth=2, label='Command ω')
        ax.plot(times, dummy_vel[:, 2], 'b-', linewidth=1.5, alpha=0.8, label='Dummy')
        ax.plot(times, wheel_vel[:, 2], 'r--', linewidth=1.5, alpha=0.8, label='Wheel')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Heading
        ax = axes[1, 0]
        ax.plot(times, np.degrees(dummy_heading), 'b-', linewidth=2, label='Dummy')
        ax.plot(times, np.degrees(wheel_heading), 'r--', linewidth=2, label='Wheel')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (deg)')
        ax.set_title('Robot Heading')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Trajectory difference
        ax = axes[1, 1]
        traj_diff = np.linalg.norm(dummy_traj - wheel_traj, axis=1)
        ax.plot(times, traj_diff * 100, 'g-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Difference (cm)')
        ax.set_title('Trajectory Difference')
        ax.grid(True, alpha=0.3)
        
        # Heading difference
        ax = axes[1, 2]
        heading_diff = np.degrees(np.arctan2(
            np.sin(dummy_heading - wheel_heading),
            np.cos(dummy_heading - wheel_heading)
        ))
        ax.plot(times, heading_diff, 'm-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading Diff (deg)')
        ax.set_title('Heading Difference')
        ax.grid(True, alpha=0.3)
        
        # Wheel velocities
        ax = axes[2, 0]
        ax.plot(times, wheel_wheel_vel[:, 0], 'b-', linewidth=1.5, label='Left actual')
        ax.plot(times, wheel_wheel_vel[:, 1], 'r-', linewidth=1.5, label='Right actual')
        ax.plot(times, wheel_cmd_wheel_vel[:, 0], 'b--', linewidth=1, alpha=0.5, label='Left cmd')
        ax.plot(times, wheel_cmd_wheel_vel[:, 1], 'r--', linewidth=1, alpha=0.5, label='Right cmd')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Wheel Velocity (rad/s)')
        ax.set_title('Wheel Velocities')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Velocity tracking error (forward)
        ax = axes[2, 1]
        dummy_v_err = np.abs(np.array(dummy_v) - cmd_vel[:, 0])
        wheel_v_err = np.abs(np.array(wheel_v) - cmd_vel[:, 0])
        ax.plot(times, dummy_v_err * 100, 'b-', linewidth=2, label='Dummy')
        ax.plot(times, wheel_v_err * 100, 'r-', linewidth=2, label='Wheel')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (cm/s)')
        ax.set_title('Forward Velocity Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity tracking error (angular)
        ax = axes[2, 2]
        dummy_omega_err = np.abs(dummy_vel[:, 2] - cmd_vel[:, 1])
        wheel_omega_err = np.abs(wheel_vel[:, 2] - cmd_vel[:, 1])
        ax.plot(times, dummy_omega_err, 'b-', linewidth=2, label='Dummy')
        ax.plot(times, wheel_omega_err, 'r-', linewidth=2, label='Wheel')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (rad/s)')
        ax.set_title('Angular Velocity Error')
        ax.legend()
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
        description="Compare Dummy vs Wheel-based Differential Drive Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run arc trajectory with GUI:
  python compare_diffdrive.py --trajectory arc
  
  # Run S-curve headless and save plot:
  python compare_diffdrive.py --trajectory scurve --no-gui --save-plot /tmp/scurve.png
  
  # Available trajectories: straight, rotation, arc, scurve, zigzag

PyBullet GUI Interaction:
  - Left-click + drag: Rotate camera
  - Middle-click + drag: Pan camera  
  - Scroll wheel: Zoom in/out
"""
    )
    parser.add_argument("--trajectory", "-t", default="arc",
                       choices=list(TRAJECTORIES.keys()),
                       help="Trajectory to test (default: arc)")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                       help="Test duration in seconds (default: 10.0)")
    parser.add_argument("--no-gui", action="store_true", 
                       help="Run without GUI (headless mode)")
    parser.add_argument("--save-plot", type=str, default=None, 
                       help="Save plot to file")
    args = parser.parse_args()
    
    # Determine save path for headless mode
    if args.no_gui and args.save_plot is None:
        args.save_plot = f"/tmp/compare_diffdrive_{args.trajectory}.png"
    
    print("="*60)
    print("  DIFFERENTIAL DRIVE COMPARISON TEST")
    print("="*60)
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Duration: {args.duration}s")
    print(f"  Mode: {'Headless' if args.no_gui else 'GUI'}")
    if args.save_plot:
        print(f"  Save Plot: {args.save_plot}")
    print("="*60)
    
    # Setup PyBullet
    print("\nInitializing PyBullet...")
    setup_pybullet(gui=not args.no_gui)
    
    # Create test
    test = DiffDriveComparisonTest()
    
    # Run test
    errors = test.run_test(
        trajectory_name=args.trajectory,
        duration=args.duration,
        gui=not args.no_gui
    )
    
    # Print results
    print("\n" + "="*60)
    print("  COMPARISON RESULTS")
    print("="*60)
    print(f"  Dummy forward vel RMSE:  {errors['dummy_v_rmse']*100:.3f} cm/s")
    print(f"  Wheel forward vel RMSE:  {errors['wheel_v_rmse']*100:.3f} cm/s")
    print(f"  Dummy angular vel RMSE:  {errors['dummy_omega_rmse']:.4f} rad/s")
    print(f"  Wheel angular vel RMSE:  {errors['wheel_omega_rmse']:.4f} rad/s")
    print("-"*60)
    print(f"  Trajectory diff RMSE:    {errors['trajectory_diff_rmse']*100:.3f} cm")
    print(f"  Trajectory diff max:     {errors['trajectory_diff_max']*100:.3f} cm")
    print(f"  Heading diff max:        {np.degrees(errors['heading_diff_max']):.2f} deg")
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

