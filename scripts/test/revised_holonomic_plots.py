#!/usr/bin/env python3
"""Revised-owned Phase7 plotting / history I/O (no dependency on test_magnum_holonomic_control)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from revised_holonomic_core import Phase7History

def remove_outliers(data: np.ndarray, method: str = "percentile", lower_percentile: float = 1.0, upper_percentile: float = 99.0, iqr_factor: float = 1.5) -> np.ndarray:
    """Remove or clip outliers from data array.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array (1D or 2D)
    method : str
        Method to use: "percentile" (clip to percentiles) or "iqr" (IQR-based filtering)
    lower_percentile : float
        Lower percentile for clipping (default: 1.0)
    upper_percentile : float
        Upper percentile for clipping (default: 99.0)
    iqr_factor : float
        IQR factor for outlier detection (default: 1.5)
    
    Returns
    -------
    np.ndarray
        Data with outliers handled (clipped or filtered)
    """
    if len(data) == 0:
        return data
    
    data = np.asarray(data)
    original_shape = data.shape
    
    # Flatten for processing
    data_flat = data.flatten()
    
    # Remove NaN and Inf values first
    valid_mask = np.isfinite(data_flat)
    if not np.any(valid_mask):
        return data
    
    valid_data = data_flat[valid_mask]
    
    if method == "percentile":
        # Clip to percentiles
        lower_bound = np.percentile(valid_data, lower_percentile)
        upper_bound = np.percentile(valid_data, upper_percentile)
        data_flat[valid_mask] = np.clip(valid_data, lower_bound, upper_bound)
    elif method == "iqr":
        # IQR-based outlier detection
        q1 = np.percentile(valid_data, 25)
        q3 = np.percentile(valid_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr
        data_flat[valid_mask] = np.clip(valid_data, lower_bound, upper_bound)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Restore original shape
    return data_flat.reshape(original_shape)


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
    
    # Compute overall contact percentage across robots (only where data exists)
    contact_percents = []
    for _, h in histories.items():
        if len(h.in_contact) > 0:
            contact_percents.append(100.0 * float(np.mean(np.array(h.in_contact, dtype=bool))))
    overall_contact_pct = float(np.mean(contact_percents)) if len(contact_percents) > 0 else 0.0

    fig.suptitle(
        f'Phase 7 Velocities: Multi-Robot Swarm (Contact: {overall_contact_pct:.1f}%)\n'
        f'Desired object velocity: {desired_obj_velocity}, omega: {desired_obj_omega:.3f} rad/s',
        fontsize=14, fontweight="bold",
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
        
        # Remove outliers from data before plotting
        robot_speeds = remove_outliers(robot_speeds, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        contact_forces = remove_outliers(contact_forces, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
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
        contact_pct = 100.0 * float(np.mean(in_contact)) if len(in_contact) > 0 else 0.0
        ax.set_title(f'{name} - Contact State ({contact_pct:.1f}%)')
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
        
        # Remove outliers from data before plotting
        # Position error: compute magnitude and remove outliers
        error_mags = np.linalg.norm(position_errors, axis=1)
        error_mags = remove_outliers(error_mags, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
        # Heading error: remove outliers (already in radians)
        heading_errors = remove_outliers(heading_errors, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
        # Contact force: remove outliers
        contact_forces = remove_outliers(contact_forces, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
        # Robot speed: compute and remove outliers
        robot_vels = np.array(history.robot_velocities)
        robot_speeds = np.linalg.norm(robot_vels[:, :2], axis=1)
        robot_speeds = remove_outliers(robot_speeds, method="percentile", lower_percentile=1.0, upper_percentile=99.0)
        
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
        
        # Plot 2: Position error (already computed and filtered above)
        ax = axes[idx, 1]
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
        
        # Plot 5: Robot speed (already computed and filtered above)
        ax = axes[idx, 4]
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


def plot_phase_7beta(
    histories: Dict[str, Phase7History],
    t_params: Dict[str, float],
    save_path: Optional[Path] = None,
        ):
    """Plot Phase 7 Beta: Object-focused trajectory visualization with robot subplots.
    
    Layout (2x3 grid):
    - Object trajectory (x-y) spans subplots 1,2 (top row, left 2 columns) - MAIN FOCUS
    - Robot 1 trajectory in subplot 3 (top-right)
    - Robot 2 trajectory in subplot 4 (bottom-left)
    - Robot 3 trajectory in subplot 5 (bottom-middle)
    - Robot 4 trajectory in subplot 6 (bottom-right)
    
    Parameters
    ----------
    histories : Dict[str, Phase7History]
        History for each robot (all should have same object data)
    t_params : Dict[str, float]
        t_param for each robot
    save_path : Optional[Path]
        Path to save plot
    """
    if len(histories) == 0:
        print("No history to plot for Phase 7 Beta.")
        return
    
    # Use GridSpec for custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(
        f'Phase 7 Beta: Object and Robot Trajectories with Headings',
        fontsize=16, fontweight='bold'
    )
    
    # Get object data from first robot's history (all robots share same object)
    first_history = list(histories.values())[0]
    if len(first_history.times) == 0:
        print("No data in history for Phase 7 Beta.")
        return
    
    times = np.array(first_history.times)
    object_positions = np.array(first_history.object_positions)
    
    object_velocities = np.array(first_history.object_velocities)
    object_angular_velocities = np.array(first_history.object_angular_velocities)
    
    # Compute object orientation by integrating angular velocity
    # Start with initial orientation: compute from first velocity direction if available
    object_orientations = np.zeros_like(times)
    if len(times) > 0:
        # Initial orientation: use velocity direction if velocity is significant, otherwise 0
        if len(object_velocities) > 0 and np.linalg.norm(object_velocities[0]) > 0.01:
            object_orientations[0] = np.arctan2(object_velocities[0, 1], object_velocities[0, 0])
        else:
            object_orientations[0] = 0.0
        
        # Integrate angular velocity to get orientation over time
        for i in range(1, len(times)):
            dt_actual = times[i] - times[i-1]
            object_orientations[i] = object_orientations[i-1] + object_angular_velocities[i-1] * dt_actual
            # Normalize to [-pi, pi]
            object_orientations[i] = np.arctan2(np.sin(object_orientations[i]), np.cos(object_orientations[i]))
    
    # PLOT 1 & 2: Object trajectory (spans top-left and top-middle, 2 columns)
    ax_obj_traj = fig.add_subplot(gs[0, :2])  # Top row, first 2 columns
    
    # Plot object trajectory with heading arrows
    ax_obj_traj.plot(object_positions[:, 0], object_positions[:, 1], 'k-', linewidth=2.5, label='Object Trajectory', alpha=0.8)
    
    # Add heading arrows along trajectory (every Nth point)
    arrow_interval = max(1, len(object_positions) // 20)  # Show ~20 arrows
    for i in range(0, len(object_positions), arrow_interval):
        if i < len(object_positions) - 1:
            dx = 0.05 * np.cos(object_orientations[i])  # Arrow length
            dy = 0.05 * np.sin(object_orientations[i])
            ax_obj_traj.arrow(
                object_positions[i, 0], object_positions[i, 1],
                dx, dy,
                head_width=0.02, head_length=0.015,
                fc='red', ec='red', alpha=0.6, zorder=5
            )
    
    # Mark start and end
    if len(object_positions) > 0:
        ax_obj_traj.plot(object_positions[0, 0], object_positions[0, 1], 'go', markersize=10, label='Object Start', zorder=6)
        ax_obj_traj.plot(object_positions[-1, 0], object_positions[-1, 1], 'ro', markersize=10, label='Object End', zorder=6)
    
    ax_obj_traj.set_xlabel('X (m)', fontsize=12)
    ax_obj_traj.set_ylabel('Y (m)', fontsize=12)
    ax_obj_traj.set_title('Object Trajectory with Heading (Main Focus)', fontsize=14, fontweight='bold')
    ax_obj_traj.legend(fontsize=10)
    ax_obj_traj.grid(True, alpha=0.3)
    ax_obj_traj.axis('equal')
    
    # PLOT 3, 4, 5, 6: Robot trajectories (top-right, bottom-left, bottom-middle, bottom-right)
    robot_names = list(histories.keys())
    robot_subplot_positions = [
        (0, 2),  # Robot 1: top-right
        (1, 0),  # Robot 2: bottom-left
        (1, 1),  # Robot 3: bottom-middle
        (1, 2),  # Robot 4: bottom-right
    ]
    
    for idx, (robot_idx, (row, col)) in enumerate(zip(range(min(4, len(robot_names))), robot_subplot_positions)):
        if robot_idx >= len(robot_names):
            break
        
        name = robot_names[robot_idx]
        history = histories[name]
        
        if len(history.times) == 0:
            continue
        
        robot_times = np.array(history.times)
        robot_positions = np.array(history.robot_positions)
        robot_headings = np.array(history.robot_headings)
        contact_points = np.array(history.contact_point_positions)
        
        ax_robot = fig.add_subplot(gs[row, col])
        
        # Plot robot trajectory with heading arrows
        ax_robot.plot(robot_positions[:, 0], robot_positions[:, 1], 'b-', linewidth=2, label=f'{name} Trajectory', alpha=0.8)
        ax_robot.plot(contact_points[:, 0], contact_points[:, 1], 'r--', linewidth=1, alpha=0.5, label='Contact Points')
        
        # Add heading arrows along robot trajectory
        arrow_interval = max(1, len(robot_positions) // 15)  # Show ~15 arrows
        for i in range(0, len(robot_positions), arrow_interval):
            if i < len(robot_positions):
                dx = 0.03 * np.cos(robot_headings[i])
                dy = 0.03 * np.sin(robot_headings[i])
                ax_robot.arrow(
                    robot_positions[i, 0], robot_positions[i, 1],
                    dx, dy,
                    head_width=0.015, head_length=0.01,
                    fc='blue', ec='blue', alpha=0.5, zorder=5
                )
        
        # Mark start and end
        if len(robot_positions) > 0:
            ax_robot.plot(robot_positions[0, 0], robot_positions[0, 1], 'go', markersize=8, label='Start', zorder=6)
            ax_robot.plot(robot_positions[-1, 0], robot_positions[-1, 1], 'ro', markersize=8, label='End', zorder=6)
        
        ax_robot.set_xlabel('X (m)', fontsize=10)
        ax_robot.set_ylabel('Y (m)', fontsize=10)
        ax_robot.set_title(f'{name} Trajectory (t_param={t_params[name]:.3f})', fontsize=11, fontweight='bold')
        ax_robot.legend(fontsize=8, loc='upper right')
        ax_robot.grid(True, alpha=0.3)
        ax_robot.axis('equal')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Phase 7 Beta plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_phase7_wheel_plot(
    histories: Dict[str, Phase7History],
    t_params: Dict[str, float],
    save_path: Optional[Path] = None,
        ):
    """Plot Phase 7 wheel velocities for all robots.
    
    Parameters
    ----------
    histories : Dict[str, Phase7History]
        History for each robot
    t_params : Dict[str, float]
        t_param for each robot
    save_path : Optional[Path]
        Path to save plot
    """
    if len(histories) == 0:
        print("No history to plot for Phase 7 wheel velocities.")
        return
    
    # Determine number of wheels from first robot's history
    first_history = list(histories.values())[0]
    if len(first_history.wheel_velocities) == 0:
        print("No wheel velocity data available.")
        return
    
    # Check if we have valid wheel velocity data
    valid_wheel_data = [wv for wv in first_history.wheel_velocities if len(wv) > 0]
    if len(valid_wheel_data) == 0:
        print("No valid wheel velocity data available.")
        return
    
    num_wheels = len(valid_wheel_data[0])
    
    # Create subplots: one row per robot, num_wheels + 1 columns (one per wheel + summary)
    n_robots = len(histories)
    fig, axes = plt.subplots(n_robots, num_wheels + 1, figsize=(5 * (num_wheels + 1), 4 * n_robots))
    if n_robots == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'Phase 7 Wheel Velocities: Multi-Robot Swarm (Commanded = solid, Actual = dashed)',
        fontsize=14, fontweight='bold'
    )
    
    for idx, (name, history) in enumerate(histories.items()):
        if len(history.times) == 0:
            continue
        
        times = np.array(history.times)
        wheel_vels_list = history.wheel_velocities
        wheel_cmd_list = getattr(history, "wheel_cmd_velocities", [])
        
        # Filter out empty arrays and convert to numpy array
        valid_indices = [i for i, wv in enumerate(wheel_vels_list) if len(wv) > 0]
        if len(valid_indices) == 0:
            continue
        
        # Extract valid wheel velocities
        wheel_vels_array = np.array([wheel_vels_list[i] for i in valid_indices])
        valid_times = times[valid_indices]

        # Extract commanded wheel velocities if present and aligned
        wheel_cmd_array = None
        if len(wheel_cmd_list) == len(wheel_vels_list):
            if all(len(wheel_cmd_list[i]) > 0 for i in valid_indices):
                wheel_cmd_array = np.array([wheel_cmd_list[i] for i in valid_indices])

        # Quick sanity stats (helps diagnose "flat" wheel plots)
        try:
            w_min = float(np.min(wheel_vels_array))
            w_max = float(np.max(wheel_vels_array))
            w_std = float(np.std(wheel_vels_array))
            print(f"[wheel_debug] {name}: min={w_min:.3f}, max={w_max:.3f}, std={w_std:.3f} rad/s, samples={len(valid_times)}")
        except Exception:
            pass
        
        # Plot individual wheel velocities
        # Focus on commanded velocities (solid line) with actual velocities as reference (dashed)
        colors = ['b', 'r', 'g', 'm', 'c', 'orange']
        for wheel_idx in range(num_wheels):
            ax = axes[idx, wheel_idx]
            if wheel_idx < wheel_vels_array.shape[1]:
                # Plot commanded velocities as solid line (primary focus)
                if wheel_cmd_array is not None and wheel_idx < wheel_cmd_array.shape[1]:
                    wheel_cmd_clean = remove_outliers(
                        wheel_cmd_array[:, wheel_idx],
                        method="percentile",
                        lower_percentile=1.0,
                        upper_percentile=99.0,
                    )
                    wheel_cmd_clean = np.round(wheel_cmd_clean, 4)
                    ax.plot(
                        valid_times,
                        wheel_cmd_clean,
                        colors[wheel_idx % len(colors)],
                        linewidth=2.0,
                        label=f'Wheel {wheel_idx+1} cmd',
                    )
                
                # Plot actual velocities as dashed line (reference)
                wheel_vels = wheel_vels_array[:, wheel_idx]
                wheel_vels_clean = remove_outliers(
                    wheel_vels,
                    method="percentile",
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                )
                wheel_vels_clean = np.round(wheel_vels_clean, 4)
                ax.plot(valid_times, wheel_vels_clean, colors[wheel_idx % len(colors)] + "--", 
                       linewidth=1.5, alpha=0.7, label=f'Wheel {wheel_idx+1} actual')
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Angular Velocity (rad/s)')
                ax.set_title(f'{name} - Wheel {wheel_idx+1} (t_param={t_params[name]:.3f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Plot summary: all wheels together
        # Focus on commanded velocities (solid) with actual velocities as reference (dashed)
        ax_summary = axes[idx, num_wheels]
        for wheel_idx in range(num_wheels):
            if wheel_idx < wheel_vels_array.shape[1]:
                # Plot commanded velocities as solid line (primary focus)
                if wheel_cmd_array is not None and wheel_idx < wheel_cmd_array.shape[1]:
                    wheel_cmd_clean = remove_outliers(
                        wheel_cmd_array[:, wheel_idx],
                        method="percentile",
                        lower_percentile=1.0,
                        upper_percentile=99.0,
                    )
                    wheel_cmd_clean = np.round(wheel_cmd_clean, 4)
                    ax_summary.plot(
                        valid_times,
                        wheel_cmd_clean,
                        colors[wheel_idx % len(colors)],
                        linewidth=2.0,
                        alpha=0.8,
                        label=f'Wheel {wheel_idx+1} cmd',
                    )
                
                # Plot actual velocities as dashed line (reference)
                wheel_vels = wheel_vels_array[:, wheel_idx]
                wheel_vels_clean = remove_outliers(
                    wheel_vels,
                    method="percentile",
                    lower_percentile=1.0,
                    upper_percentile=99.0,
                )
                wheel_vels_clean = np.round(wheel_vels_clean, 4)
                ax_summary.plot(valid_times, wheel_vels_clean, colors[wheel_idx % len(colors)] + "--", 
                              linewidth=1.5, alpha=0.5, label=f'Wheel {wheel_idx+1} actual')
        ax_summary.set_xlabel('Time (s)')
        ax_summary.set_ylabel('Angular Velocity (rad/s)')
        ax_summary.set_title(f'{name} - All Wheels Summary')
        ax_summary.legend()
        ax_summary.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Phase 7 wheel velocity plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def export_histories(
    histories: Dict[str, Phase7History],
    t_params: Dict[str, float],
    save_path: Path,
        ):
    """Export histories and t_params to JSON file.
    
    Parameters
    ----------
    histories : Dict[str, Phase7History]
        History for each robot
    t_params : Dict[str, float]
        t_param for each robot
    save_path : Path
        Path to save JSON file
    """
    # Convert Phase7History dataclass to JSON-serializable dict
    export_data = {
        "t_params": t_params,
        "histories": {}
    }
    
    for name, history in histories.items():
        # Convert numpy arrays to lists for JSON serialization
        # Also convert numpy booleans to Python booleans
        history_dict = {
            "times": [float(t) for t in history.times],
            "robot_positions": [pos.tolist() for pos in history.robot_positions],
            "robot_headings": [float(h) for h in history.robot_headings],
            "robot_velocities": [vel.tolist() for vel in history.robot_velocities],
            "intended_positions": [pos.tolist() for pos in history.intended_positions],
            "position_errors": [err.tolist() for err in history.position_errors],
            "desired_headings": [float(h) for h in history.desired_headings],
            "heading_errors": [float(e) for e in history.heading_errors],
            "contact_point_positions": [pos.tolist() for pos in history.contact_point_positions],
            "contact_point_velocities": [vel.tolist() for vel in history.contact_point_velocities],
            "object_positions": [pos.tolist() for pos in history.object_positions],
            "object_velocities": [vel.tolist() for vel in history.object_velocities],
            "object_angular_velocities": [float(omega) for omega in history.object_angular_velocities],
            "contact_forces": [float(f) for f in history.contact_forces],
            "in_contact": [bool(ic) for ic in history.in_contact],  # Convert numpy bool_ to Python bool
            "v_base_history": [float(v) for v in history.v_base_history],
            "v_ff_history": [float(v) for v in history.v_ff_history],
            "v_pi_history": [float(v) for v in history.v_pi_history],
            "desired_contact_point_speeds": [float(s) for s in history.desired_contact_point_speeds],
            "wheel_velocities": [wv.tolist() if len(wv) > 0 else [] for wv in history.wheel_velocities],
            "wheel_cmd_velocities": [wv.tolist() if len(wv) > 0 else [] for wv in getattr(history, "wheel_cmd_velocities", [])],
        }
        export_data["histories"][name] = history_dict
    
    # Save to JSON
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported histories to {save_path}")


def import_histories(
    load_path: Path,
        ):
    """Import histories and t_params from JSON file.
    
    Parameters
    ----------
    load_path : Path
        Path to load JSON file from
    
    Returns
    -------
    histories : Dict[str, Phase7History]
        History for each robot
    t_params : Dict[str, float]
        t_param for each robot
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"History file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        import_data = json.load(f)
    
    t_params = import_data["t_params"]
    histories = {}
    
    for name, history_dict in import_data["histories"].items():
        # Convert lists back to numpy arrays
        history = Phase7History(
            times=history_dict["times"],
            robot_positions=[np.array(pos) for pos in history_dict["robot_positions"]],
            robot_headings=history_dict["robot_headings"],
            robot_velocities=[np.array(vel) for vel in history_dict["robot_velocities"]],
            intended_positions=[np.array(pos) for pos in history_dict["intended_positions"]],
            position_errors=[np.array(err) for err in history_dict["position_errors"]],
            desired_headings=history_dict["desired_headings"],
            heading_errors=history_dict["heading_errors"],
            contact_point_positions=[np.array(pos) for pos in history_dict["contact_point_positions"]],
            contact_point_velocities=[np.array(vel) for vel in history_dict["contact_point_velocities"]],
            object_positions=[np.array(pos) for pos in history_dict["object_positions"]],
            object_velocities=[np.array(vel) for vel in history_dict["object_velocities"]],
            object_angular_velocities=history_dict["object_angular_velocities"],
            contact_forces=history_dict["contact_forces"],
            in_contact=history_dict["in_contact"],
            v_base_history=history_dict["v_base_history"],
            v_ff_history=history_dict["v_ff_history"],
            v_pi_history=history_dict["v_pi_history"],
            desired_contact_point_speeds=history_dict["desired_contact_point_speeds"],
            wheel_velocities=[np.array(wv) if len(wv) > 0 else np.array([]) for wv in history_dict.get("wheel_velocities", [])],
            wheel_cmd_velocities=[np.array(wv) if len(wv) > 0 else np.array([]) for wv in history_dict.get("wheel_cmd_velocities", [])],
        )
        histories[name] = history
    
    print(f"Imported histories from {load_path}")
    return histories, t_params
