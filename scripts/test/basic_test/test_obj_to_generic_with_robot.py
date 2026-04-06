#!/usr/bin/env python3
"""
Simple test to verify OBJ-to-GenericObject conversion.

This script:
1. Loads an object from OBJ file and converts it to GenericObject
2. Periodically traverses through t_param (over 2 seconds)
3. Draws markers to visualize:
   - Contact point position (what GenericObject thinks it is)
   - Normal outward vector
   - Tangent vector
4. This helps verify if the conversion from OBJ to GenericObject is correct
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg

# Use non-interactive backend for headless mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ROS package path setup
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import ContactPointParameterization, ContactPoint
from contact_maintain.object_bridge import obj_to_generic
from contact_maintain.robot_factory import create_robot
from contact_maintain.pyb_simulation import get_object_state

# Magnum Four solver (legacy)
from contact_optimizer_utils import find_the_magnum_four_v3

# Constants
TIMESTEP = 1.0 / 240.0
DEFAULT_OBJECT_FRICTION = 0.8
T_PARAM_INCREMENT_INTERVAL = 2.0  # seconds between t_param increments
T_PARAM_INCREMENT = 0.1  # increment amount

# Single-robot navigation constants
ROBOT_APPROACH_DISTANCE = 0.2  # distance from contact point along normal_outward
APPROACH_DISTANCE = 0.4  # Distance from contact point to spawn robot (for Magnum Four)
ROBOT_LINEAR_GAIN = 1.0       # gain for position error (in body frame)
ROBOT_ANGULAR_GAIN = 4.0      # gain for heading error
ROBOT_MAX_LINEAR_SPEED = 0.6  # clamp translational speed
ROBOT_MAX_ANG_SPEED = 1.0     # clamp angular speed


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

    # Add URDF directory to search path for custom OBJ files
    urdf_dir = Path(pkg_path) / "urdf"
    if urdf_dir.exists():
        pyb.setAdditionalSearchPath(str(urdf_dir))
    
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


def create_marker(position: np.ndarray, color: tuple = (1.0, 0.0, 0.0, 1.0), radius: float = 0.03):
    """Create a visual marker at position."""
    marker_uid = pyb.createVisualShape(
        shapeType=pyb.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )
    body_uid = pyb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=marker_uid,
        basePosition=[position[0], position[1], 0.1],
        baseOrientation=[0, 0, 0, 1]
    )
    return body_uid


def create_arrow_marker(start_pos: np.ndarray, direction: np.ndarray, color: tuple = (0.0, 1.0, 0.0, 1.0), scale: float = 0.2):
    """Create an arrow marker to visualize a vector direction."""
    # Normalize direction and scale
    direction_norm = direction / (np.linalg.norm(direction) + 1e-10)
    end_pos = start_pos + direction_norm * scale
    
    # Create a cylinder for the arrow shaft
    arrow_length = np.linalg.norm(end_pos - start_pos)
    mid_pos = (start_pos + end_pos) / 2
    
    # Calculate rotation to align with direction
    angle = np.arctan2(direction_norm[1], direction_norm[0])
    orn = pyb.getQuaternionFromEuler([0, 0, angle])
    
    arrow_uid = pyb.createVisualShape(
        shapeType=pyb.GEOM_CYLINDER,
        radius=0.01,
        length=arrow_length,
        rgbaColor=color
    )
    body_uid = pyb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=arrow_uid,
        basePosition=[mid_pos[0], mid_pos[1], 0.1],
        baseOrientation=orn
    )
    return body_uid


def create_cube_marker(position: np.ndarray, color: tuple = (0.0, 1.0, 0.0, 1.0), size: float = 0.05):
    """Create a cube marker at position."""
    marker_uid = pyb.createVisualShape(
        shapeType=pyb.GEOM_BOX,
        halfExtents=[size/2, size/2, size/2],
        rgbaColor=color
    )
    body_uid = pyb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=marker_uid,
        basePosition=[position[0], position[1], size/2 + 0.05],
        baseOrientation=[0, 0, 0, 1]
    )
    return body_uid

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OBJ+DXF-to-GenericObject conversion with a single robot")
    parser.add_argument(
        "--obj-shape",
        type=str,
        default="right_triangle",
        help="Shape name (right_triangle, bolt, hourglass, pi, root; default: right_triangle)",
    )
    parser.add_argument("--obj-file", type=str, default=None,
                       help="OBJ file path (relative to urdf directory). If None, uses '{obj-shape}.obj'")
    parser.add_argument("--duration", type=float, default=20.0,
                       help="Total test duration (default: 20.0 s)")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run headless")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save visualization plots")
    args = parser.parse_args()
    
    print("="*60)
    print("  OBJ-to-GenericObject + Single-Robot Boundary-Tracking Test")
    print("="*60)
    print(f"  OBJ shape: {args.obj_shape}")
    print(f"  t_param increment: {T_PARAM_INCREMENT:.1f} every {T_PARAM_INCREMENT_INTERVAL:.1f} s")
    print(f"  Total duration: {args.duration:.1f} s")
    print("="*60)
    
    # Setup PyBullet
    print("\nInitializing PyBullet...")
    setup_pybullet(gui=not args.no_gui)
    
    # Load OBJ file and create GenericObject
    if args.obj_file is None:
        obj_file = f"{args.obj_shape}.obj"
    else:
        obj_file = args.obj_file
    
    print(f"\nLoading OBJ file: {obj_file}...")
    try:
        generic_object, object_uid = obj_to_generic(
            obj_path=obj_file,
            shape_name=args.obj_shape,
            position=(0.0, 0.0, 0.6),
            orientation=0,
            mass=2.0,
            lateral_friction=DEFAULT_OBJECT_FRICTION,
            blind_test=True
        )
        print(f"✓ Loaded OBJ object: {args.obj_shape}")
        print(f"  Mass: {generic_object.mass:.3f} kg")
        print(f"  Moment of inertia: {generic_object.moment_of_inertia:.6f} kg·m²")
        print(f"  Lateral friction: {generic_object.lateral_friction:.3f}")
    except Exception as e:
        print(f"✗ Failed to load OBJ: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create contact point parameterization
    print(f"\nCreating contact point parameterization...")
    parameterization = ContactPointParameterization(generic_object)
    boundary_length = generic_object.boundary_length
    print(f"  Boundary length: {boundary_length:.3f} m")
    
    # Visualize parameterization using matplotlib
    print(f"\nGenerating parameterization visualization...")
    try:
        ax = parameterization.visualize_parameterization(n_test_points=30)
        
        # Save plot if save-dir is provided
        if args.save_dir:
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            plot_file = save_path / f"{args.obj_shape}_parameterization.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved parameterization plot to: {plot_file}")
        else:
            # Save to current directory as fallback
            plot_file = Path(f"{args.obj_shape}_parameterization.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved parameterization plot to: {plot_file}")
        
        plt.close()
    except Exception as e:
        print(f"  ⚠ Failed to generate visualization: {e}")
        import traceback
        traceback.print_exc()
    
    # Compute Magnum Four contacts / t_params
    print(f"\n{'='*60}")
    print("  Computing Magnum Four optimal contact points...")
    print(f"{'='*60}")
    
    # Check cache first to avoid recomputing optimal contact points
    cache_file = Path(pkg_path) / "urdf" / "magnum_four_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load cache if it exists
    cached_t_params = None
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                if args.obj_shape in cache_data:
                    cached_t_params = cache_data[args.obj_shape]
                    print(f"\nFound cached Magnum Four t_params for '{args.obj_shape}': {cached_t_params}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load cache: {e}")
    
    if cached_t_params is not None:
        # Use cached solution
        t_params = cached_t_params
        print(f"Using cached Magnum Four t_params for '{args.obj_shape}': {[f'{v:.4f}' for v in t_params]}")
        
        # Create ContactPoint objects from cached t_params
        contacts = []
        for t_param in t_params:
            temp_contact = parameterization.get_contact_info(t_param)
            contacts.append(ContactPoint(
                position=temp_contact['point'],
                tangent=temp_contact['tangent'],
                normal_outward=temp_contact['normal_outward'],
                normal_inward=temp_contact['normal_inward'],
                parameter=t_param,
                force_direction=None,
                object_ref=generic_object,
            ))
    else:
        # No cache found, run solver and save result
        print(f"\nComputing Magnum Four contact points for '{args.obj_shape}'...")
        magnum_result = find_the_magnum_four_v3(
            generic_object,
            verbose=False,
            visualize=False,
            weighting_scheme="balanced",
            torque_method=3,
        )
        if not magnum_result or not magnum_result.get("success", False):
            raise RuntimeError("Magnum Four solver failed to produce a solution.")

        contacts = magnum_result["best_solution"]["contacts"]
        t_params = [float(c.parameter) for c in contacts]
        t_params = [tp % 1.0 for tp in t_params]
        t_params = np.array(t_params)
        t_params = t_params.tolist()
        
        if len(t_params) != 4:
            raise RuntimeError(f"Expected 4 contacts from Magnum Four, got {len(t_params)}")
        
        print(f"\nMagnum Four t_params: {[f'{v:.4f}' for v in t_params]}")
        
        # Save to cache
        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass
        
        cache_data[args.obj_shape] = t_params
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved Magnum Four solution to cache: {cache_file}")
    
    # Mark the 4 Magnum Four solution points in the scene with different colors
    print(f"\nMarking Magnum Four solution points in scene...")
    solution_colors = [
        (1.0, 0.0, 0.0, 1.0),  # Red for first
        (0.0, 0.0, 1.0, 1.0),  # Blue for second
        (0.0, 1.0, 0.0, 1.0),  # Green for third
        (1.0, 1.0, 0.0, 1.0),  # Yellow for fourth
    ]
    solution_markers = []
    
    # Get object's initial state (should be at origin with orientation 0)
    obj_state = get_object_state(object_uid)
    obj_position = obj_state["position"]
    obj_orientation = obj_state["orientation"]
    
    # Transform rotation matrix
    cos_theta = np.cos(obj_orientation)
    sin_theta = np.sin(obj_orientation)
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    
    for i, t_param in enumerate(t_params):
        # Get contact point info in body frame
        contact_info = parameterization.get_contact_info(t_param)
        contact_point_body = np.array(contact_info['point'], dtype=float)
        
        # Transform to world frame
        contact_point_world = R @ contact_point_body + obj_position
        
        # Create marker with appropriate color
        marker = create_marker(
            contact_point_world,
            color=solution_colors[i],
            radius=0.06,  # Larger radius to make them more visible
        )
        solution_markers.append(marker)
        print(f"  Solution {i+1} (t_param={t_param:.4f}): {solution_colors[i][:3]} marker at ({contact_point_world[0]:.3f}, {contact_point_world[1]:.3f})")
    
    # Delay 1 second to let physics stabilize
    print(f"\nWaiting 1 second for physics to stabilize...")
    for _ in range(int(1.0 / TIMESTEP)):
        pyb.stepSimulation()
        if not args.no_gui:
            time.sleep(TIMESTEP)
    print(f"  ✓ Physics stabilized")
    
    # === Single robot + boundary traversal test ===
    print(f"\n{'='*60}")
    print("  Single robot: navigate to first Magnum Four solution point")
    print("  Then traverse along boundary as t_param increases.")
    print(f"{'='*60}")
    
    # Use first Magnum Four solution as target
    target_t_param = t_params[0]
    print(f"  Target t_param (first solution): {target_t_param:.4f}")
    
    # Get contact point info at target t_param (in object body frame)
    contact_info = parameterization.get_contact_info(target_t_param)
    contact_point_body = np.array(contact_info['point'], dtype=float)
    normal_outward = np.array(contact_info['normal_outward'], dtype=float)
    normal_inward = np.array(contact_info['normal_inward'], dtype=float)
    
    # Calculate spawn position: contact_point + approach_distance * normal_outward
    # Object is at origin (0, 0, 0), so body frame = world frame initially
    spawn_position_body = contact_point_body + APPROACH_DISTANCE * normal_outward
    robot_x = float(spawn_position_body[0])
    robot_y = float(spawn_position_body[1])
    
    # Robot heading: point toward contact point (normal_inward direction)
    robot_heading = float(np.arctan2(normal_inward[1], normal_inward[0]))
    
    # Spawn robot
    robot = create_robot(
        kinematics="holonomic",
        model="wheel",
        position=(robot_x, robot_y),
        orientation=robot_heading,
        name="R_01",
    )
    print(f"Spawned robot at ({robot_x:.3f}, {robot_y:.3f}) with heading {robot_heading:.3f} rad, "
          f"target t_param={target_t_param:.4f}")
    
    # Initial t_param for traversal (start with first solution)
    current_t_param = target_t_param
    
    # Traverse through t_param and visualize + control robot
    print(f"\n{'='*60}")
    print(f"  Traversing t_param: increment by {T_PARAM_INCREMENT:.1f} every {T_PARAM_INCREMENT_INTERVAL:.1f} seconds...")
    print(f"  Starting from first Magnum Four solution (t_param={current_t_param:.4f})")
    print(f"{'='*60}")
    
    n_steps = int(args.duration / TIMESTEP)
    step_count = 0
    t = 0.0
    last_increment_time = 0.0
    
    # Initialize markers (will be created in the loop)
    contact_marker = None
    normal_outward_marker = None
    normal_inward_marker = None
    tangent_marker = None
    target_marker = None
    
    for step in range(n_steps):
        # Increment t_param every T_PARAM_INCREMENT_INTERVAL seconds
        t_param_changed = False
        if t - last_increment_time >= T_PARAM_INCREMENT_INTERVAL:
            current_t_param = (current_t_param + T_PARAM_INCREMENT) % 1.0
            last_increment_time = t
            t_param_changed = True
        
        # --- Compute transformation every step (object may move/rotate) ---
        # Get object's current state for transformation
        obj_state = get_object_state(object_uid)
        obj_position = obj_state["position"]
        obj_orientation = obj_state["orientation"]
        
        # Get contact info in object's body frame
        contact_info = parameterization.get_contact_info(current_t_param)
        contact_point_body = np.array(contact_info['point'], dtype=float)
        normal_outward_body = np.array(contact_info['normal_outward'], dtype=float)
        normal_inward_body = np.array(contact_info['normal_inward'], dtype=float)
        tangent_body = np.array(contact_info['tangent'], dtype=float)
        
        # Transform from body frame to world frame
        # Rotation matrix for object orientation
        cos_theta = np.cos(obj_orientation)
        sin_theta = np.sin(obj_orientation)
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        # Transform contact point (position: rotation + translation)
        contact_point_world = R @ contact_point_body + obj_position
        
        # Transform vectors (rotation only, no translation)
        normal_outward_world = R @ normal_outward_body
        normal_inward_world = R @ normal_inward_body
        tangent_world = R @ tangent_body
        
        # Update target position and heading based on current object pose
        target_position = contact_point_world + ROBOT_APPROACH_DISTANCE * normal_outward_world
        target_heading = float(np.arctan2(normal_inward_world[1], normal_inward_world[0]))
        
        if t_param_changed:
            # Remove old markers
            if contact_marker is not None:
                pyb.removeBody(contact_marker)
            if normal_outward_marker is not None:
                pyb.removeBody(normal_outward_marker)
            if normal_inward_marker is not None:
                pyb.removeBody(normal_inward_marker)
            if tangent_marker is not None:
                pyb.removeBody(tangent_marker)
            if target_marker is not None:
                pyb.removeBody(target_marker)
            
            # Create new markers in world frame
            contact_marker = create_marker(
                contact_point_world,
                color=(1.0, 0.0, 0.0, 1.0),
                radius=0.04,
            )
            normal_outward_marker = create_arrow_marker(
                contact_point_world,
                normal_outward_world,
                color=(0.0, 1.0, 0.0, 1.0),
                scale=0.15,
            )
            normal_inward_marker = create_arrow_marker(
                contact_point_world,
                normal_inward_world,
                color=(1.0, 1.0, 0.0, 1.0),
                scale=0.15,
            )
            tangent_marker = create_arrow_marker(
                contact_point_world,
                tangent_world,
                color=(0.0, 0.0, 1.0, 1.0),
                scale=0.15,
            )
            # Green cube: robot target position
            target_marker = create_cube_marker(
                target_position,
                color=(0.0, 1.0, 0.0, 1.0),
                size=0.08,
            )
            
            print(
                f"  [t={t:.2f}s] t_param={current_t_param:.3f}, "
                f"contact_point_world=({contact_point_world[0]:.3f}, {contact_point_world[1]:.3f}), "
                f"target_position=({target_position[0]:.3f}, {target_position[1]:.3f})"
            )
        
        # # --- Simple PID robot controller: drive toward (target_position, target_heading) ---
        # # (target_position and target_heading are already computed above using object's current pose)
        # robot_pos, robot_heading, _ = robot.get_state()
        # robot_pos = np.array(robot_pos, dtype=float)
        
        # # Position error in world frame
        # pos_error_world = target_position - robot_pos

        
        # # Heading error (wrap to [-pi, pi])
        # heading_error = target_heading - robot_heading
        # heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # # Proportional control in world frame
        # vx_body = ROBOT_LINEAR_GAIN * pos_error_world[0]
        # vy_body = ROBOT_LINEAR_GAIN * pos_error_world[1]
        # omega = ROBOT_ANGULAR_GAIN * heading_error
        
        # # Clamp speeds
        # lin_speed = np.linalg.norm([vx_body, vy_body])
        # if lin_speed > ROBOT_MAX_LINEAR_SPEED:
        #     scale = ROBOT_MAX_LINEAR_SPEED / (lin_speed + 1e-9)
        #     vx_body *= scale
        #     vy_body *= scale
        # omega = float(np.clip(omega, -ROBOT_MAX_ANG_SPEED, ROBOT_MAX_ANG_SPEED))
        
        # # Command robot (holonomic model expects world-frame [vx, vy, omega])
        # # robot.command_velocity(np.array([vx_body, vy_body, omega]))
        
        # Step simulation
        pyb.stepSimulation()
        t += TIMESTEP
        step_count += 1
        
        if not args.no_gui:
            time.sleep(TIMESTEP * 0.5)
    
    print(f"\n{'='*60}")
    print("  VISUALIZATION + ROBOT TRACKING COMPLETE")
    print(f"{'='*60}")
    print(f"  Magnum Four solution markers (static):")
    print(f"    Red: First solution (robot target)")
    print(f"    Blue: Second solution")
    print(f"    Green: Third solution")
    print(f"    Yellow: Fourth solution")
    print(f"  Dynamic markers (update as t_param changes):")
    print(f"    Red sphere: Current contact point position")
    print(f"    Green cube: Robot target position (contact point + approach distance)")
    print(f"    Green arrow: Normal outward vector")
    print(f"    Yellow arrow: Normal inward vector")
    print(f"    Blue arrow: Tangent vector")
    print(f"  Robot R_01 uses simple PID control to track")
    print(f"  the boundary contact point + approach distance as t_param traverses.")
    print(f"{'='*60}")
    
    if not args.no_gui:
        print("\nMarkers and robot will remain visible. Press Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
