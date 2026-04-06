#!/usr/bin/env python3
"""
Test script for the Path Velocity Planner.

This script demonstrates:
1. Creating a hybrid path (rectangle)
2. Planning velocity profiles with different look-ahead modes
3. Querying velocities at various points
4. Printing planning summaries
"""

import sys
from pathlib import Path
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

import rospkg

# ROS package path setup
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

# Import the planner and path creation functions
from contact_maintain.motion_planner import (
    PathVelocityPlanner,
    PathDirectionProvider,
    PathFollowingController,
)
from paths_lib import (
    create_rectangle_hybrid_path,
    create_p_trajectory_hybrid_path,
    create_catenary_hybrid_path
)


def test_rectangle_path():
    """Test velocity planning on a rectangle path."""
    print("\n" + "="*80)
    print("TEST 1: Rectangle Path - Look-ahead Mode 1 (Continuous)")
    print("="*80)
    
    # Create a rectangle path
    path = create_rectangle_hybrid_path(
        corner1=[0, 0],
        corner2=[3, 0],
        corner3=[3, 2],
        corner4=[0, 2]
    )
    
    print(f"Path created: {path.num_components} segments, total length: {path.total_length:.4f} m")
    
    # Create planner with continuous mode
    planner = PathVelocityPlanner(
        hybrid_path=path,
        a_max=2.0,        # m/s²
        a_lat_max=1.5,   # m/s²
        v_user_max=1.0,  # m/s
        look_ahead=0     # Continuous mode
    )
    
    # Print summary
    planner.print_summary()
    
    # Query velocities at various points
    print("\n" + "-"*80)
    print("Velocity Queries:")
    print("-"*80)
    test_points = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for s in test_points:
        if s <= path.total_length:
            v = planner.get_velocity_at_arc_length(s)
            print(f"  s = {s:6.2f} m  ->  v = {v:.4f} m/s")
    
    return planner


def test_rectangle_path_independent():
    """Test velocity planning on a rectangle path with independent mode."""
    print("\n" + "="*80)
    print("TEST 2: Rectangle Path - Look-ahead Mode 0 (Independent/Stops)")
    print("="*80)
    
    # Create a rectangle path
    path = create_rectangle_hybrid_path(
        corner1=[0, 0],
        corner2=[3, 0],
        corner3=[3, 2],
        corner4=[0, 2]
    )
    
    # Create planner with independent mode (stops at boundaries)
    planner = PathVelocityPlanner(
        hybrid_path=path,
        a_max=2.0,        # m/s²
        a_lat_max=1.5,   # m/s²
        v_user_max=3.0,  # m/s
        look_ahead=0     # Independent mode (stops at boundaries)
    )
    
    # Print summary
    planner.print_summary()
    
    print(f"\nTotal time (independent mode): {planner.get_total_time():.4f} s")
    
    return planner


def test_p_trajectory():
    """Test velocity planning on a P-trajectory path."""
    print("\n" + "="*80)
    print("TEST 3: P Trajectory Path - Look-ahead Mode 1 (Continuous)")
    print("="*80)
    
    # Create a P trajectory path
    path = create_p_trajectory_hybrid_path(
        start_point=[0, 0],
        stem_height=2.0,
        arc_radius=1.0,
        arc_center_offset=1.0
    )
    
    print(f"Path created: {path.num_components} segments, total length: {path.total_length:.4f} m")
    
    # Create planner
    planner = PathVelocityPlanner(
        hybrid_path=path,
        a_max=2.0,        # m/s²
        a_lat_max=1.5,   # m/s²
        v_user_max=3.0,  # m/s
        look_ahead=1     # Continuous mode
    )
    
    # Print summary
    planner.print_summary()
    
    return planner


def test_catenary_path():
    """Test velocity planning on a catenary path."""
    print("\n" + "="*80)
    print("TEST 4: Catenary Path - Look-ahead Mode 1 (Continuous)")
    print("="*80)
    
    # Create a catenary path
    path = create_catenary_hybrid_path(
        x_start=-2.0,
        x_end=2.0,
        a=5.0,  # High alpha for flatter curve
        y_offset=0.0,
        num_points=100
    )
    
    print(f"Path created: {path.num_components} segments, total length: {path.total_length:.4f} m")
    
    # Create planner
    planner = PathVelocityPlanner(
        hybrid_path=path,
        a_max=2.0,        # m/s²
        a_lat_max=1.5,   # m/s²
        v_user_max=3.0,  # m/s
        look_ahead=1     # Continuous mode
    )
    
    # Print summary
    planner.print_summary()
    
    return planner


def plot_velocity_profiles(path, planner_0, planner_1, planner_2=None, save_dir="/tmp/hybrid_path"):
    """
    Plot velocity profiles for all look-ahead modes over path length.
    
    Args:
        path: HybridPath instance
        planner_0: PathVelocityPlanner with look_ahead=0
        planner_1: PathVelocityPlanner with look_ahead=1
        planner_2: PathVelocityPlanner with look_ahead=2 (optional)
        save_dir: Directory to save the plot
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate boundary arc lengths (transition points)
    boundary_s = [0.0]
    cumulative = 0.0
    for component in path.components:
        cumulative += component.get_path_length()
        boundary_s.append(cumulative)
    boundary_s = np.array(boundary_s)
    
    # Sample velocities along the path, ensuring we include boundary points
    num_samples = 500
    s_regular = np.linspace(0.0, path.total_length, num_samples)
    
    # Merge regular samples with boundary points and sort
    s_samples = np.unique(np.sort(np.concatenate([s_regular, boundary_s])))
    
    velocities_0 = []
    velocities_1 = []
    velocities_2 = [] if planner_2 else None
    
    for s in s_samples:
        v0 = planner_0.get_velocity_at_arc_length(s)
        v1 = planner_1.get_velocity_at_arc_length(s)
        velocities_0.append(v0)
        velocities_1.append(v1)
        if planner_2:
            v2 = planner_2.get_velocity_at_arc_length(s)
            velocities_2.append(v2)
    
    velocities_0 = np.array(velocities_0)
    velocities_1 = np.array(velocities_1)
    if velocities_2:
        velocities_2 = np.array(velocities_2)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot all profiles
    ax.plot(s_samples, velocities_0, 'r-', linewidth=2, label='Mode 0 (Independent - Stops at boundaries)', alpha=0.8)
    ax.plot(s_samples, velocities_1, 'b-', linewidth=2, label='Mode 1 (Naive - Ignores transition curvature)', alpha=0.8)
    if planner_2:
        ax.plot(s_samples, velocities_2, 'g-', linewidth=2, label='Mode 2 (Smart - Considers transition curvature)', alpha=0.8)
    
    # Mark segment boundaries with vertical lines and plot boundary velocities
    print("\n--- Boundary Velocities Debug ---")
    for i, s_bound in enumerate(boundary_s):
        v0_bound = planner_0.get_velocity_at_arc_length(s_bound)
        v1_bound = planner_1.get_velocity_at_arc_length(s_bound)
        v2_bound = planner_2.get_velocity_at_arc_length(s_bound) if planner_2 else None
        
        if planner_2:
            print(f"Boundary {i}: s={s_bound:.4f}m, v0={v0_bound:.6f}, v1={v1_bound:.6f}, v2={v2_bound:.6f} m/s")
        else:
            print(f"Boundary {i}: s={s_bound:.4f}m, v0={v0_bound:.6f} m/s, v1={v1_bound:.6f} m/s")
        
        if i > 0 and i < len(boundary_s) - 1:  # Internal boundaries only
            ax.axvline(x=s_bound, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            # Plot boundary velocity points
            ax.scatter([s_bound], [v0_bound], color='darkred', s=50, zorder=5, marker='o')
            ax.scatter([s_bound], [v1_bound], color='darkblue', s=50, zorder=5, marker='s')
            if planner_2:
                ax.scatter([s_bound], [v2_bound], color='darkgreen', s=50, zorder=5, marker='^')
    
    # Mark start and end
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=path.total_length, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Path Length (m)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    title = 'Velocity Profiles: Mode 0 vs Mode 1 vs Mode 2' if planner_2 else 'Velocity Profiles: Mode 0 vs Mode 1'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with summary info
    time_0 = planner_0.get_total_time()
    time_1 = planner_1.get_total_time()
    avg_speed_0 = path.total_length / time_0 if time_0 > 0 else 0
    avg_speed_1 = path.total_length / time_1 if time_1 > 0 else 0
    
    if planner_2:
        time_2 = planner_2.get_total_time()
        avg_speed_2 = path.total_length / time_2 if time_2 > 0 else 0
        info_text = (
            f"Path Length: {path.total_length:.3f} m\n"
            f"Mode 0: {time_0:.3f} s (avg: {avg_speed_0:.3f} m/s)\n"
            f"Mode 1: {time_1:.3f} s (avg: {avg_speed_1:.3f} m/s)\n"
            f"Mode 2: {time_2:.3f} s (avg: {avg_speed_2:.3f} m/s)\n"
            f"Time saved (0→2): {time_0 - time_2:.3f} s"
        )
    else:
        time_saved = time_0 - time_1
        info_text = (
            f"Path Length: {path.total_length:.3f} m\n"
            f"Mode 0: {time_0:.3f} s (avg: {avg_speed_0:.3f} m/s)\n"
            f"Mode 1: {time_1:.3f} s (avg: {avg_speed_1:.3f} m/s)\n"
            f"Time Saved: {time_saved:.3f} s ({100*time_saved/time_0:.1f}% faster)"
        )
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, "velocity_profiles_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVelocity profile plot saved to: {save_path}")


def compare_modes():
    """Compare all look-ahead modes (0, 1, 2) on the same path."""
    print("\n" + "="*80)
    print("COMPARISON: Look-ahead Mode 0 vs Mode 1 vs Mode 2")
    print("="*80)
    
    # Create a rectangle path
    path = create_rectangle_hybrid_path(
        corner1=[0, 0],
        corner2=[3, 0],
        corner3=[3, 2],
        corner4=[0, 2]
    )


    # Create a P trajectory path
    # path = create_p_trajectory_hybrid_path(
    #     start_point=[0, 0],
    #     stem_height=2.0,
    #     arc_radius=1.0,
    #     arc_center_offset=1.0
    # )

    
    # Create a catenary path
    # path = create_catenary_hybrid_path(
    #     x_start=-2.0,
    #     x_end=2.0,
    #     a=5.0,  # High alpha for flatter curve
    #     y_offset=0.0,
    #     num_points=100
    # )
    
    # Print transition curvatures for debugging
    if hasattr(path, 'transition_curvatures'):
        print("\nTransition curvatures at corners:")
        for i, k in enumerate(path.transition_curvatures):
            print(f"  Transition {i+1}: κ = {k:.4f} rad/m")
    
    # Mode 0 (Independent - stops at every boundary)
    planner_0 = PathVelocityPlanner(
        hybrid_path=path,
        a_max=0.3,
        a_lat_max=0.1,
        v_user_max=0.5,
        look_ahead=0
    )
    
    # Mode 1 (Naive - ignores transition curvature)
    planner_1 = PathVelocityPlanner(
        hybrid_path=path,
        a_max=0.3,
        a_lat_max=0.1,
        v_user_max=0.5,
        look_ahead=1
    )
    
    # Mode 2 (Smart - considers transition curvature)
    planner_2 = PathVelocityPlanner(
        hybrid_path=path,
        a_max=0.3,
        a_lat_max=0.1,
        v_user_max=0.5,
        look_ahead=2
    )
    
    print(f"\nPath length: {path.total_length:.4f} m")
    
    print(f"\nMode 0 (Independent - stops at every boundary):")
    print(f"  Total time: {planner_0.get_total_time():.4f} s")
    print(f"  Average speed: {path.total_length / planner_0.get_total_time():.4f} m/s")
    
    print(f"\nMode 1 (Naive - ignores transition curvature):")
    print(f"  Total time: {planner_1.get_total_time():.4f} s")
    print(f"  Average speed: {path.total_length / planner_1.get_total_time():.4f} m/s")
    
    print(f"\nMode 2 (Smart - considers transition curvature):")
    print(f"  Total time: {planner_2.get_total_time():.4f} s")
    print(f"  Average speed: {path.total_length / planner_2.get_total_time():.4f} m/s")
    
    # Compare modes
    time_0 = planner_0.get_total_time()
    time_1 = planner_1.get_total_time()
    time_2 = planner_2.get_total_time()
    
    print(f"\n--- Comparison ---")
    print(f"Mode 0 → Mode 1: {time_0 - time_1:.4f} s faster ({100*(time_0-time_1)/time_0:.1f}%)")
    print(f"Mode 0 → Mode 2: {time_0 - time_2:.4f} s faster ({100*(time_0-time_2)/time_0:.1f}%)")
    print(f"Mode 1 → Mode 2: {time_1 - time_2:.4f} s difference")
    
    # Plot velocity profiles
    plot_velocity_profiles(path, planner_0, planner_1, planner_2)


def test_path_following_controller():
    """Test the PathFollowingController with rectangle and P-trajectory paths."""
    print("\n" + "="*80)
    print("PATH FOLLOWING CONTROLLER TEST")
    print("="*80)
    
    # Test 1: Rectangle path
    print("\n--- Test 1: Rectangle Path ---")
    rect_path = create_rectangle_hybrid_path(
        corner1=[0, 0],
        corner2=[3, 0],
        corner3=[3, 2],
        corner4=[0, 2]
    )
    
    rect_controller = PathFollowingController(
        hybrid_path=rect_path,
        a_max=0.5,
        a_lat_max=0.3,
        v_user_max=0.5,
        look_ahead=2,  # Smart mode
        use_tracking=False,
    )
    
    print(f"Path: Rectangle, {rect_path.num_components} segments, length: {rect_path.total_length:.3f} m")
    print(f"Estimated time: {rect_controller.velocity_planner.get_total_time():.3f} s")
    
    # Simulate path following
    print("\nSimulating path following (dt=0.1s):")
    dt = 0.1
    step = 0
    while not rect_controller.is_completed() and step < 200:
        s = rect_controller.get_current_s()
        target = rect_controller.get_target_point()
        tangent = rect_controller.get_target_tangent()
        cmd = rect_controller.compute_velocity(dt=dt)
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: s={s:6.3f}m ({100*rect_controller.get_progress_fraction():5.1f}%) "
                  f"cmd=[{cmd[0]:6.3f}, {cmd[1]:6.3f}, {cmd[2]:6.3f}]")
        step += 1
    
    print(f"Completed in {step} steps ({step * dt:.1f}s simulated)")
    
    # Test 2: P-trajectory path
    print("\n--- Test 2: P-Trajectory Path ---")
    p_path = create_p_trajectory_hybrid_path(
        start_point=[0, 0],
        stem_height=2.0,
        arc_radius=0.5,
        arc_center_offset=0.5
    )
    
    p_controller = PathFollowingController(
        hybrid_path=p_path,
        a_max=0.5,
        a_lat_max=0.3,
        v_user_max=0.5,
        look_ahead=2,
        use_tracking=False,
    )
    
    print(f"Path: P-trajectory, {p_path.num_components} segments, length: {p_path.total_length:.3f} m")
    print(f"Estimated time: {p_controller.velocity_planner.get_total_time():.3f} s")
    
    # Simulate path following
    print("\nSimulating path following (dt=0.1s):")
    dt = 0.1
    step = 0
    while not p_controller.is_completed() and step < 200:
        s = p_controller.get_current_s()
        cmd = p_controller.compute_velocity(dt=dt)
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: s={s:6.3f}m ({100*p_controller.get_progress_fraction():5.1f}%) "
                  f"cmd=[{cmd[0]:6.3f}, {cmd[1]:6.3f}, {cmd[2]:6.3f}]")
        step += 1
    
    print(f"Completed in {step} steps ({step * dt:.1f}s simulated)")
    
    return rect_controller, p_controller


def plot_single_path_following_commands(path, path_name, filename, save_dir="/tmp/hybrid_path"):
    """
    Plot velocity commands for a single path.
    
    Args:
        path: HybridPath instance
        path_name: Human-readable name for the path (e.g., "Rectangle")
        filename: Output filename (without .png extension)
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    controller = PathFollowingController(
        hybrid_path=path,
        a_max=0.5,
        a_lat_max=0.3,
        v_user_max=0.5,
        look_ahead=2,
        use_tracking=False,
    )
    
    # Collect data
    dt = 0.05
    s_list = []
    vx_list = []
    vy_list = []
    omega_list = []
    speed_list = []
    
    while not controller.is_completed():
        s = controller.get_current_s()
        cmd = controller.compute_velocity(dt=dt)
        speed = np.sqrt(cmd[0]**2 + cmd[1]**2)
        
        s_list.append(s)
        vx_list.append(cmd[0])
        vy_list.append(cmd[1])
        omega_list.append(cmd[2])
        speed_list.append(speed)
    
    s_arr = np.array(s_list)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Speed (magnitude)
    ax = axes[0, 0]
    ax.plot(s_arr, speed_list, 'b-', linewidth=2)
    ax.set_xlabel('Arc Length (m)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed |v| along Path')
    ax.grid(True, alpha=0.3)
    
    # Mark segment boundaries
    for i in range(1, path.num_components):
        s_bound = path.cumulative_lengths[i]
        ax.axvline(x=s_bound, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 2: vx and vy
    ax = axes[0, 1]
    ax.plot(s_arr, vx_list, 'r-', linewidth=2, label='vx')
    ax.plot(s_arr, vy_list, 'g-', linewidth=2, label='vy')
    ax.set_xlabel('Arc Length (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Linear Velocity Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i in range(1, path.num_components):
        s_bound = path.cumulative_lengths[i]
        ax.axvline(x=s_bound, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: omega
    ax = axes[1, 0]
    ax.plot(s_arr, omega_list, 'm-', linewidth=2)
    ax.set_xlabel('Arc Length (m)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Angular Velocity ω')
    ax.grid(True, alpha=0.3)
    
    for i in range(1, path.num_components):
        s_bound = path.cumulative_lengths[i]
        ax.axvline(x=s_bound, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: Path with velocity arrows
    ax = axes[1, 1]
    
    # Plot path
    num_pts = 500
    s_samples = np.linspace(0, path.total_length, num_pts)
    path_x = []
    path_y = []
    for s in s_samples:
        pt = path.get_point_at_arc_length(s)
        path_x.append(pt[0])
        path_y.append(pt[1])
    
    ax.plot(path_x, path_y, 'k-', linewidth=2, label='Path')
    
    # Add velocity arrows every N points
    arrow_interval = max(1, len(s_list) // 20)
    for i in range(0, len(s_list), arrow_interval):
        pt = path.get_point_at_arc_length(s_list[i])
        scale = 0.3  # Arrow scale
        ax.arrow(pt[0], pt[1], vx_list[i]*scale, vy_list[i]*scale,
                head_width=0.05, head_length=0.03, fc='blue', ec='blue', alpha=0.7)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Path with Velocity Direction Arrows')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f'PathFollowingController Output: {path_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  {path_name} plot saved to: {save_path}")


def plot_path_following_commands(save_dir="/tmp/hybrid_path"):
    """Plot velocity commands for Rectangle and P-trajectory paths.
    
    Saves plots with naming convention: path_following_commands_{path_type}.png
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("PLOTTING PATH FOLLOWING COMMANDS")
    print("="*80)
    
    # Plot 1: Rectangle path
    rect_path = create_rectangle_hybrid_path(
        corner1=[0, 0],
        corner2=[3, 0],
        corner3=[3, 2],
        corner4=[0, 2]
    )
    plot_single_path_following_commands(
        rect_path, 
        "Rectangle Path", 
        "path_following_commands_rectangle",  # Consistent naming format
        save_dir
    )
    
    # Plot 2: P-trajectory path
    p_path = create_p_trajectory_hybrid_path(
        start_point=[0, 0],
        stem_height=2.0,
        arc_radius=0.5,
        arc_center_offset=0.5
    )
    plot_single_path_following_commands(
        p_path,
        "P-Trajectory Path",
        "path_following_commands_p_trajectory",  # Consistent naming format
        save_dir
    )
    
    print(f"\nAll path following command plots saved to: {save_dir}")


if __name__ == "__main__":
    print("="*80)
    print("PATH VELOCITY PLANNER TEST SUITE")
    print("="*80)
    
    try:
        # Compare all look-ahead modes (0, 1, 2)
        compare_modes()
        
        # Test PathFollowingController
        test_path_following_controller()
        
        # Plot path following commands
        plot_path_following_commands()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
