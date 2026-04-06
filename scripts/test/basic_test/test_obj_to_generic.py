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

from object_utils import ContactPointParameterization
from contact_maintain.object_bridge import obj_to_generic

# Constants
TIMESTEP = 1.0 / 240.0
DEFAULT_OBJECT_FRICTION = 0.8
T_PARAM_INCREMENT_INTERVAL = 2.0  # seconds between t_param increments
T_PARAM_INCREMENT = 0.1  # increment amount


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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OBJ+DXF-to-GenericObject conversion")
    parser.add_argument(
        "--obj-shape",
        type=str,
        default="right_triangle",
        help="Shape name (right_triangle, bolt, pi, root; default: right_triangle)",
    )
    parser.add_argument("--obj-file", type=str, default=None,
                       help="OBJ file path (relative to urdf directory). If None, uses '{obj-shape}.obj'")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Total test duration (default: 5.0 s)")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run headless")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save visualization plots")
    args = parser.parse_args()
    
    print("="*60)
    print("  OBJ-to-GenericObject Conversion Test")
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
            position=(0.0, 0.0, 0.5),
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
    
    # Delay 1 second to let physics stabilize
    print(f"\nWaiting 1 second for physics to stabilize...")
    for _ in range(int(1.0 / TIMESTEP)):
        pyb.stepSimulation()
        if not args.no_gui:
            time.sleep(TIMESTEP)
    print(f"  ✓ Physics stabilized")
    
    # Traverse through t_param and visualize
    print(f"\n{'='*60}")
    print(f"  Traversing t_param: increment by {T_PARAM_INCREMENT:.1f} every {T_PARAM_INCREMENT_INTERVAL:.1f} seconds...")
    print(f"{'='*60}")
    
    n_steps = int(args.duration / TIMESTEP)
    step_count = 0
    t = 0.0
    current_t_param = 0.0
    last_increment_time = 0.0
    
    # Markers (will be updated each step)
    contact_marker = None
    normal_outward_marker = None
    normal_inward_marker = None
    tangent_marker = None
    
    for step in range(n_steps):
        # Increment t_param every T_PARAM_INCREMENT_INTERVAL seconds
        t_param_changed = False
        if t - last_increment_time >= T_PARAM_INCREMENT_INTERVAL:
            current_t_param = (current_t_param + T_PARAM_INCREMENT) % 1.0
            last_increment_time = t
            t_param_changed = True
        
        # Only update markers when t_param changes
        if t_param_changed:
            # Get contact point info at current t_param.
            # NOTE: ContactPointParameterization already works in world coordinates
            # for the current GenericObject geometry, so we use these directly.
            contact_info = parameterization.get_contact_info(current_t_param)
            contact_point_world = np.array(contact_info['point'], dtype=float)
            normal_outward_world = np.array(contact_info['normal_outward'], dtype=float)
            normal_inward_world = np.array(contact_info['normal_inward'], dtype=float)
            tangent_world = np.array(contact_info['tangent'], dtype=float)
            
            # Remove old markers
            if contact_marker is not None:
                pyb.removeBody(contact_marker)
            if normal_outward_marker is not None:
                pyb.removeBody(normal_outward_marker)
            if normal_inward_marker is not None:
                pyb.removeBody(normal_inward_marker)
            if tangent_marker is not None:
                pyb.removeBody(tangent_marker)
            
            # Create new markers in world frame
            # Red sphere: contact point position
            contact_marker = create_marker(
                contact_point_world, 
                color=(1.0, 0.0, 0.0, 1.0), 
                radius=0.04
            )
            
            # Green arrow: normal outward vector
            normal_outward_marker = create_arrow_marker(
                contact_point_world,
                normal_outward_world,
                color=(0.0, 1.0, 0.0, 1.0),
                scale=0.15
            )
            
            # Yellow arrow: normal inward vector (for comparison)
            normal_inward_marker = create_arrow_marker(
                contact_point_world,
                normal_inward_world,
                color=(1.0, 1.0, 0.0, 1.0),
                scale=0.15
            )
            
            # Blue arrow: tangent vector
            tangent_marker = create_arrow_marker(
                contact_point_world,
                tangent_world,
                color=(0.0, 0.0, 1.0, 1.0),
                scale=0.15
            )
            
            # Print status when t_param changes
            print(f"  [t={t:.2f}s] t_param={current_t_param:.3f}, "
                  f"contact_point_world=({contact_point_world[0]:.3f}, {contact_point_world[1]:.3f})")
        
        pyb.stepSimulation()
        t += TIMESTEP
        step_count += 1
        
        if not args.no_gui:
            time.sleep(TIMESTEP * 0.5)
    
    print(f"\n{'='*60}")
    print("  VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Red sphere: Contact point position (GenericObject's understanding)")
    print(f"  Green arrow: Normal outward vector (from GenericObject)")
    print(f"  Yellow arrow: Normal inward vector (for comparison)")
    print(f"  Blue arrow: Tangent vector")
    print(f"  Compare markers with actual 3D object to verify conversion correctness")
    print(f"{'='*60}")
    
    if not args.no_gui:
        print("\nMarkers will remain visible. Press Enter to exit...")
        input()
    
    pyb.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
