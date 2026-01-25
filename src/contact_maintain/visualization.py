"""Visualization utilities for contact maintenance analysis."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
from matplotlib.collections import LineCollection
from typing import Optional, Dict, List


def plot_trajectory(positions, orientations=None, ax=None, 
                    color='blue', label='Robot', show_orientation=True,
                    orientation_interval=10):
    """Plot a 2D trajectory.
    
    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Array of (x, y) positions.
    orientations : np.ndarray, shape (N,), optional
        Array of orientations (radians).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    color : str
        Color for the trajectory.
    label : str
        Label for the legend.
    show_orientation : bool
        Whether to show orientation arrows.
    orientation_interval : int
        Show orientation every N points.
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    positions = np.array(positions)
    
    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], '-', color=color, 
            label=label, linewidth=1.5)
    
    # Plot start and end markers
    ax.plot(positions[0, 0], positions[0, 1], 'o', color=color, 
            markersize=10, label=f'{label} Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 's', color=color, 
            markersize=10, label=f'{label} End')
    
    # Plot orientation arrows
    if show_orientation and orientations is not None:
        orientations = np.array(orientations)
        arrow_length = 0.1
        for i in range(0, len(positions), orientation_interval):
            dx = arrow_length * np.cos(orientations[i])
            dy = arrow_length * np.sin(orientations[i])
            ax.arrow(positions[i, 0], positions[i, 1], dx, dy,
                    head_width=0.03, head_length=0.02, fc=color, ec=color)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_contact_forces(timestamps, forces, in_contact=None, ax=None):
    """Plot contact force over time.
    
    Parameters
    ----------
    timestamps : np.ndarray, shape (N,)
        Time values.
    forces : np.ndarray, shape (N, 3) or (N,)
        Force values. If 3D, computes magnitude.
    in_contact : np.ndarray, shape (N,), optional
        Boolean array indicating contact.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    forces = np.array(forces)
    if forces.ndim > 1:
        force_mag = np.linalg.norm(forces[:, :2], axis=1)
    else:
        force_mag = forces
    
    ax.plot(timestamps, force_mag, 'b-', linewidth=1, label='Force Magnitude')
    
    # Highlight contact regions
    if in_contact is not None:
        in_contact = np.array(in_contact)
        ax.fill_between(timestamps, 0, force_mag.max(), 
                       where=in_contact, alpha=0.2, color='green',
                       label='In Contact')
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax.set_title('Contact Force Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([timestamps[0], timestamps[-1]])
    ax.set_ylim([0, None])
    
    return ax


def plot_contact_analysis(observer_data: Dict, figsize=(14, 10)):
    """Create a comprehensive contact analysis plot.
    
    Parameters
    ----------
    observer_data : dict
        Dictionary from ContactObserver.get_history_arrays().
    figsize : tuple
        Figure size.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    timestamps = observer_data['timestamps']
    robot_pos = observer_data['robot_positions']
    robot_orn = observer_data['robot_orientations']
    contact_forces = observer_data['contact_forces']
    object_pos = observer_data['object_positions']
    in_contact = observer_data['in_contact']
    
    # Top-left: Trajectories
    ax = axes[0, 0]
    ax.plot(robot_pos[:, 0], robot_pos[:, 1], 'b-', label='Robot', linewidth=1.5)
    ax.plot(object_pos[:, 0], object_pos[:, 1], 'r-', label='Object', linewidth=1.5)
    ax.plot(robot_pos[0, 0], robot_pos[0, 1], 'bo', markersize=8)
    ax.plot(object_pos[0, 0], object_pos[0, 1], 'ro', markersize=8)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajectories')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Top-right: Force magnitude
    force_mag = np.linalg.norm(contact_forces[:, :2], axis=1)
    ax = axes[0, 1]
    ax.plot(timestamps, force_mag, 'b-', linewidth=1)
    ax.fill_between(timestamps, 0, force_mag.max(), 
                   where=in_contact, alpha=0.2, color='green')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax.set_title('Contact Force Magnitude')
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Force components
    ax = axes[1, 0]
    ax.plot(timestamps, contact_forces[:, 0], 'r-', label='Fx', linewidth=1)
    ax.plot(timestamps, contact_forces[:, 1], 'g-', label='Fy', linewidth=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax.set_title('Force Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Contact ratio over time
    ax = axes[1, 1]
    window = 50  # Rolling window for contact ratio
    contact_ratio = np.convolve(in_contact.astype(float), 
                                np.ones(window)/window, mode='valid')
    t_ratio = timestamps[:len(contact_ratio)]
    ax.plot(t_ratio, contact_ratio * 100, 'g-', linewidth=1.5)
    ax.axhline(y=100, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Contact Ratio [%]')
    ax.set_title(f'Rolling Contact Ratio (window={window})')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_scene_snapshot(robot_pos, robot_theta, object_pos, object_radius=0.5,
                       robot_radius=0.06, contact_force=None, ax=None):
    """Plot a snapshot of the scene.
    
    Parameters
    ----------
    robot_pos : np.ndarray, shape (2,)
        Robot position.
    robot_theta : float
        Robot orientation.
    object_pos : np.ndarray, shape (2,)
        Object position.
    object_radius : float
        Object radius.
    robot_radius : float
        Robot radius.
    contact_force : np.ndarray, shape (2,) or (3,), optional
        Contact force to visualize.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw object
    object_circle = Circle(object_pos[:2], object_radius, 
                          fill=True, facecolor='lightblue', 
                          edgecolor='blue', linewidth=2)
    ax.add_patch(object_circle)
    
    # Draw robot
    robot_circle = Circle(robot_pos[:2], robot_radius,
                         fill=True, facecolor='lightcoral',
                         edgecolor='red', linewidth=2)
    ax.add_patch(robot_circle)
    
    # Draw robot orientation
    arrow_length = robot_radius * 1.5
    dx = arrow_length * np.cos(robot_theta)
    dy = arrow_length * np.sin(robot_theta)
    ax.arrow(robot_pos[0], robot_pos[1], dx, dy,
            head_width=0.03, head_length=0.02, fc='darkred', ec='darkred')
    
    # Draw contact force
    if contact_force is not None:
        force = np.array(contact_force)[:2]
        force_mag = np.linalg.norm(force)
        if force_mag > 0.1:
            # Scale force for visualization
            scale = 0.05  # meters per Newton
            ax.arrow(robot_pos[0] + robot_radius * np.cos(robot_theta),
                    robot_pos[1] + robot_radius * np.sin(robot_theta),
                    force[0] * scale, force[1] * scale,
                    head_width=0.02, head_length=0.01, 
                    fc='green', ec='green', linewidth=2)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable limits
    all_x = [robot_pos[0], object_pos[0]]
    all_y = [robot_pos[1], object_pos[1]]
    margin = max(robot_radius, object_radius) * 2
    ax.set_xlim([min(all_x) - margin, max(all_x) + margin])
    ax.set_ylim([min(all_y) - margin, max(all_y) + margin])
    
    return ax


def create_animation_frames(observer_data: Dict, output_dir: str, 
                           object_radius=0.5, robot_radius=0.06,
                           frame_interval=10):
    """Create animation frames from observer data.
    
    Parameters
    ----------
    observer_data : dict
        Dictionary from ContactObserver.get_history_arrays().
    output_dir : str
        Directory to save frames.
    object_radius : float
        Object radius for visualization.
    robot_radius : float
        Robot radius for visualization.
    frame_interval : int
        Save every N-th frame.
    
    Returns
    -------
    list
        List of saved frame paths.
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamps = observer_data['timestamps']
    robot_pos = observer_data['robot_positions']
    robot_orn = observer_data['robot_orientations']
    contact_forces = observer_data['contact_forces']
    object_pos = observer_data['object_positions']
    
    saved_frames = []
    
    for i in range(0, len(timestamps), frame_interval):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        plot_scene_snapshot(
            robot_pos[i], robot_orn[i], object_pos[i],
            object_radius=object_radius, robot_radius=robot_radius,
            contact_force=contact_forces[i], ax=ax
        )
        ax.set_title(f't = {timestamps[i]:.2f}s')
        
        frame_path = output_path / f"frame_{i:05d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        saved_frames.append(frame_path)
    
    print(f"Saved {len(saved_frames)} frames to {output_dir}")
    return saved_frames

