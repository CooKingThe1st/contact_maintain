#!/usr/bin/env python3
"""
Compare Robots Script

Compares the dummy holonomic robot (direct velocity control) against
the realistic omniwheel robot (4-wheel velocity control).

Tests:
1. Straight line motion
2. Pure rotation
3. Circular trajectory
4. Figure-8 trajectory
5. Lateral strafe motion

Shows:
- Trajectory comparison
- Velocity tracking error
- Wheel velocities (for omniwheel robot)

PyBullet GUI Interaction:
- Left-click + drag: Rotate camera
- Middle-click + drag: Pan camera
- Scroll wheel: Zoom in/out
- Press 'Q' key in terminal to quit (if running with GUI)

Usage:
    # Run with GUI (default):
    python compare_robots.py --trajectory circle
    
    # Run headless with plot output:
    python compare_robots.py --trajectory figure8 --no-gui --save-plot /tmp/compare.png
    
    # Available trajectories: straight, rotation, circle, figure8, strafe
"""
import argparse
import time
from pathlib import Path
import sys
import os

import numpy as np

# Use non-interactive backend for headless mode
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))

from contact_maintain.robots import HolonomicRobot
from contact_maintain.omniwheel_robot import OmniwheelRobot, compute_wheel_velocities


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 100  # Hz
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)


# ============================================================================
# TRAJECTORY GENERATORS
# ============================================================================

def straight_line_trajectory(t, speed=0.5):
    """Move straight forward."""
    return np.array([speed, 0, 0])


def pure_rotation_trajectory(t, omega=1.0):
    """Rotate in place."""
    return np.array([0, 0, omega])


def circular_trajectory(t, radius=0.5, omega=0.5):
    """Move in a circle."""
    v = omega * radius
    return np.array([v, 0, omega])


def figure8_trajectory(t, scale=0.5, period=8.0):
    """Move in a figure-8 pattern."""
    freq = 2 * np.pi / period
    
    # Figure-8 parametric: x = sin(t), y = sin(2t)/2
    vx = scale * freq * np.cos(freq * t)
    vy = scale * freq * np.cos(2 * freq * t)
    
    # Heading rate based on velocity direction change
    speed = np.sqrt(vx**2 + vy**2)
    if speed > 0.01:
        heading = np.arctan2(vy, vx)
        # Simple heading tracking
        omega = 0.5 * np.sin(freq * t)
    else:
        omega = 0
    
    return np.array([vx, vy, omega])


def strafe_trajectory(t, speed=0.3):
    """Move sideways (lateral motion)."""
    return np.array([0, speed, 0])


TRAJECTORIES = {
    'straight': straight_line_trajectory,
    'rotation': pure_rotation_trajectory,
    'circle': circular_trajectory,
    'figure8': figure8_trajectory,
    'strafe': strafe_trajectory,
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

class RobotComparisonTest:
    """Test comparing dummy vs omniwheel robot."""
    
    def __init__(self):
        # Create robots at different positions
        self.dummy_robot = HolonomicRobot(
            str(Path(pkg_path) / "urdf" / "holonomic_robot.urdf"),
            position=(-1.0, 0),
            orientation=0.0
        )
        
        self.omni_robot = OmniwheelRobot(
            str(Path(pkg_path) / "urdf" / "omniwheel_robot.urdf"),
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
            # Omniwheel robot
            'omni_pos': [],
            'omni_heading': [],
            'omni_vel': [],
            'omni_wheel_vel': [],
            'omni_cmd_wheel_vel': [],
        }
    
    def reset(self):
        """Reset both robots."""
        self.dummy_robot.reset(position=(-1.0, 0), orientation=0.0)
        self.omni_robot.reset(position=(1.0, 0), orientation=0.0)
        
        for key in self.history:
            self.history[key] = []
    
    def run_test(self, trajectory_name='circle', duration=10.0, gui=True):
        """Run comparison test.
        
        Parameters
        ----------
        trajectory_name : str
            Name of trajectory to follow.
        duration : float
            Test duration in seconds.
        gui : bool
            Whether to show visualization.
        
        Returns
        -------
        dict
            Error metrics.
        """
        self.reset()
        
        trajectory_fn = TRAJECTORIES.get(trajectory_name, circular_trajectory)
        n_steps = int(duration / TIMESTEP)
        
        print(f"\nRunning {trajectory_name} trajectory for {duration}s")
        print("  Dummy robot (blue) at x=-1")
        print("  Omniwheel robot (orange) at x=+1")
        
        step_count = 0
        t = 0.0
        
        for step in range(n_steps):
            # Control at lower frequency
            if step_count % CTRL_STEP == 0:
                # Get trajectory command
                cmd_vel = trajectory_fn(t)
                
                # Apply to both robots
                self.dummy_robot.command_velocity(cmd_vel)
                self.omni_robot.command_velocity(cmd_vel)
                
                # Record state
                dummy_pos, dummy_heading, dummy_vel = self.dummy_robot.get_state()
                omni_pos, omni_heading, omni_vel = self.omni_robot.get_state()
                omni_wheel_vel = self.omni_robot.get_wheel_velocities()
                
                self.history['times'].append(t)
                self.history['cmd_vel'].append(cmd_vel.copy())
                self.history['dummy_pos'].append(dummy_pos.copy())
                self.history['dummy_heading'].append(dummy_heading)
                self.history['dummy_vel'].append(dummy_vel.copy())
                self.history['omni_pos'].append(omni_pos.copy())
                self.history['omni_heading'].append(omni_heading)
                self.history['omni_vel'].append(omni_vel.copy())
                self.history['omni_wheel_vel'].append(omni_wheel_vel.copy())
                self.history['omni_cmd_wheel_vel'].append(
                    self.omni_robot.last_wheel_speeds.copy())
            
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
        omni_pos = np.array(self.history['omni_pos'])
        omni_vel = np.array(self.history['omni_vel'])
        
        # Velocity tracking error
        dummy_vel_error = np.linalg.norm(dummy_vel - cmd_vel, axis=1)
        omni_vel_error = np.linalg.norm(omni_vel - cmd_vel, axis=1)
        
        # Trajectory deviation (relative to starting offset)
        dummy_traj = dummy_pos - dummy_pos[0]
        omni_traj = omni_pos - omni_pos[0]
        traj_diff = np.linalg.norm(dummy_traj - omni_traj, axis=1)
        
        errors = {
            'dummy_vel_rmse': np.sqrt(np.mean(dummy_vel_error**2)),
            'omni_vel_rmse': np.sqrt(np.mean(omni_vel_error**2)),
            'trajectory_diff_rmse': np.sqrt(np.mean(traj_diff**2)),
            'trajectory_diff_max': np.max(traj_diff),
        }
        
        return errors
    
    def plot_results(self, save_path=None, show=False):
        """Plot comparison results.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot.
        show : bool
            If True, display plot interactively.
        """
        times = np.array(self.history['times'])
        cmd_vel = np.array(self.history['cmd_vel'])
        
        dummy_pos = np.array(self.history['dummy_pos'])
        dummy_heading = np.array(self.history['dummy_heading'])
        dummy_vel = np.array(self.history['dummy_vel'])
        
        omni_pos = np.array(self.history['omni_pos'])
        omni_heading = np.array(self.history['omni_heading'])
        omni_vel = np.array(self.history['omni_vel'])
        omni_wheel_vel = np.array(self.history['omni_wheel_vel'])
        omni_cmd_wheel_vel = np.array(self.history['omni_cmd_wheel_vel'])
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Robot Comparison: Dummy (blue) vs Omniwheel (orange)', fontsize=14)
        
        # Trajectories (relative)
        ax = axes[0, 0]
        dummy_traj = dummy_pos - dummy_pos[0]
        omni_traj = omni_pos - omni_pos[0]
        ax.plot(dummy_traj[:, 0], dummy_traj[:, 1], 'b-', linewidth=2, label='Dummy')
        ax.plot(omni_traj[:, 0], omni_traj[:, 1], 'r--', linewidth=2, label='Omniwheel')
        ax.plot(0, 0, 'go', markersize=10, label='Start')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Relative Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Velocity X
        ax = axes[0, 1]
        ax.plot(times, cmd_vel[:, 0], 'k-', linewidth=2, label='Command')
        ax.plot(times, dummy_vel[:, 0], 'b-', linewidth=1.5, alpha=0.8, label='Dummy')
        ax.plot(times, omni_vel[:, 0], 'r--', linewidth=1.5, alpha=0.8, label='Omni')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity X (m/s)')
        ax.set_title('Linear Velocity X')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Velocity Y
        ax = axes[0, 2]
        ax.plot(times, cmd_vel[:, 1], 'k-', linewidth=2, label='Command')
        ax.plot(times, dummy_vel[:, 1], 'b-', linewidth=1.5, alpha=0.8, label='Dummy')
        ax.plot(times, omni_vel[:, 1], 'r--', linewidth=1.5, alpha=0.8, label='Omni')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity Y (m/s)')
        ax.set_title('Linear Velocity Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Angular Velocity
        ax = axes[1, 0]
        ax.plot(times, cmd_vel[:, 2], 'k-', linewidth=2, label='Command')
        ax.plot(times, dummy_vel[:, 2], 'b-', linewidth=1.5, alpha=0.8, label='Dummy')
        ax.plot(times, omni_vel[:, 2], 'r--', linewidth=1.5, alpha=0.8, label='Omni')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Heading
        ax = axes[1, 1]
        ax.plot(times, np.degrees(dummy_heading), 'b-', linewidth=2, label='Dummy')
        ax.plot(times, np.degrees(omni_heading), 'r--', linewidth=2, label='Omniwheel')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (deg)')
        ax.set_title('Robot Heading')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Trajectory difference
        ax = axes[1, 2]
        traj_diff = np.linalg.norm(dummy_traj - omni_traj, axis=1)
        ax.plot(times, traj_diff * 100, 'g-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Difference (cm)')
        ax.set_title('Trajectory Difference')
        ax.grid(True, alpha=0.3)
        
        # Wheel velocities
        ax = axes[2, 0]
        labels = ['FR', 'FL', 'RL', 'RR']
        colors = ['r', 'g', 'b', 'm']
        for i in range(4):
            ax.plot(times, omni_wheel_vel[:, i], colors[i] + '-', 
                   linewidth=1.5, label=f'{labels[i]} actual')
            ax.plot(times, omni_cmd_wheel_vel[:, i], colors[i] + '--', 
                   linewidth=1, alpha=0.5, label=f'{labels[i]} cmd')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Wheel Velocity (rad/s)')
        ax.set_title('Omniwheel Wheel Velocities')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Velocity tracking error
        ax = axes[2, 1]
        dummy_error = np.linalg.norm(dummy_vel - cmd_vel, axis=1)
        omni_error = np.linalg.norm(omni_vel - cmd_vel, axis=1)
        ax.plot(times, dummy_error * 100, 'b-', linewidth=2, label='Dummy')
        ax.plot(times, omni_error * 100, 'r-', linewidth=2, label='Omniwheel')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (cm/s)')
        ax.set_title('Velocity Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Command visualization (polar plot of velocity)
        ax = axes[2, 2]
        cmd_speed = np.sqrt(cmd_vel[:, 0]**2 + cmd_vel[:, 1]**2)
        cmd_angle = np.arctan2(cmd_vel[:, 1], cmd_vel[:, 0])
        ax.scatter(cmd_angle, cmd_speed, c=times, cmap='viridis', s=5, alpha=0.5)
        ax.set_xlabel('Direction (rad)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title('Command Velocity Profile')
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
        description="Compare Dummy vs Omniwheel Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run circular trajectory with GUI:
  python compare_robots.py --trajectory circle
  
  # Run figure-8 headless and save plot:
  python compare_robots.py --trajectory figure8 --no-gui --save-plot /tmp/fig8.png
  
  # Test lateral motion:
  python compare_robots.py --trajectory strafe --duration 5

PyBullet GUI Interaction:
  - Left-click + drag: Rotate camera
  - Middle-click + drag: Pan camera  
  - Scroll wheel: Zoom in/out
"""
    )
    parser.add_argument("--trajectory", "-t", default="circle",
                       choices=list(TRAJECTORIES.keys()),
                       help="Trajectory to test (default: circle)")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                       help="Test duration in seconds (default: 10.0)")
    parser.add_argument("--no-gui", action="store_true", 
                       help="Run without GUI (headless mode)")
    parser.add_argument("--save-plot", type=str, default=None, 
                       help="Save plot to file. If not specified with --no-gui, saves to /tmp/")
    args = parser.parse_args()
    
    # Determine save path for headless mode
    if args.no_gui and args.save_plot is None:
        args.save_plot = f"/tmp/compare_robots_{args.trajectory}.png"
    
    print("="*60)
    print("  ROBOT COMPARISON TEST")
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
    test = RobotComparisonTest()
    
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
    print(f"  Dummy velocity RMSE:     {errors['dummy_vel_rmse']*100:.3f} cm/s")
    print(f"  Omniwheel velocity RMSE: {errors['omni_vel_rmse']*100:.3f} cm/s")
    print(f"  Trajectory diff RMSE:    {errors['trajectory_diff_rmse']*100:.3f} cm")
    print(f"  Trajectory diff max:     {errors['trajectory_diff_max']*100:.3f} cm")
    print("="*60)
    
    # Plot results (always save to file in headless mode)
    test.plot_results(save_path=args.save_plot, show=not args.no_gui)
    
    # Keep PyBullet open for inspection if GUI mode
    if not args.no_gui:
        print("\nPress Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()

