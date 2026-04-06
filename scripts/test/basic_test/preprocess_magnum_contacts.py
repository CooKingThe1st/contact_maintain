#!/usr/bin/env python3
"""
Precompute Magnum Four contact configurations for a set of OBJ-backed shapes.

This script:
  1. Forcefully clears the existing Magnum Four cache at:
       urdf/magnum_four_cache.json
  2. Loads each target object via obj_to_generic
  3. Runs find_the_magnum_four_v3 with a C-space reachability filter
     (via robot_radius → get_reachable_contact_intervals)
  4. Saves the resulting t_params into the cache JSON.

Usage examples:
  # Default safety scale (1.5x robot radius)
  python preprocess_magnum_contacts.py

  # More conservative (2x robot radius)
  python preprocess_magnum_contacts.py --safety-scale 2.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg


# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "src"))
sys.path.insert(0, str(pkg_path / "src" / "legacy"))

from contact_maintain.object_bridge import obj_to_generic  # noqa: E402
from contact_optimizer_utils import find_the_magnum_four_v3  # noqa: E402


DEFAULT_OBJECT_MASS = 2.0
DEFAULT_OBJECT_FRICTION = 0.8
DEFAULT_ROBOT_RADIUS = 0.06  # should match other tests (e.g., test_magnum_motion_planning)

TARGET_SHAPES: List[str] = [
    "right_triangle",
    "pi",
    "root",
    "rect",
    "meteor",
    "hourglass",
]


def setup_pybullet(gui: bool = False):
    """Initialize a minimal PyBullet scene for obj_to_generic."""
    if gui:
        pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        pyb.connect(pyb.DIRECT)

    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(1.0 / 240.0)
    pyb.setRealTimeSimulation(0)

    # Search paths: default URDFs + this package's urdf folder
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_dir = pkg_path / "urdf"
    if urdf_dir.exists():
        pyb.setAdditionalSearchPath(str(urdf_dir))

    # Basic ground plane
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.changeDynamics(ground, -1, lateralFriction=DEFAULT_OBJECT_FRICTION)

    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0],
        )
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

    return ground


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Magnum Four contact points for standard OBJ-backed shapes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Shapes:
  right_triangle, pi, root, rect, meteor, hourglass

Examples:
  # Default safety scale (1.5x robot radius):
  python preprocess_magnum_contacts.py

  # More conservative reachability (2x robot radius):
  python preprocess_magnum_contacts.py --safety-scale 2.0
        """,
    )
    parser.add_argument(
        "--safety-scale",
        type=float,
        default=1.5,
        help="Multiplier on the nominal robot radius (0.06 m) used for "
             "C-space reachability filtering. Effective radius = safety_scale * 0.06.",
    )
    args = parser.parse_args()

    effective_radius = DEFAULT_ROBOT_RADIUS * float(args.safety_scale)
    print("=" * 60)
    print(" MAGNUM FOUR PREPROCESSING (with C-space reachability filter) ")
    print("=" * 60)
    print(f"Package path: {pkg_path}")
    print(f"Robot nominal radius: {DEFAULT_ROBOT_RADIUS:.4f} m")
    print(f"Safety scale: {args.safety_scale:.3f}")
    print(f"Effective reachability radius: {effective_radius:.4f} m")

    cache_path = pkg_path / "urdf" / "magnum_four_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Forcefully clear existing cache
    if cache_path.exists():
        print(f"\nClearing existing cache file: {cache_path}")
        cache_path.unlink()
    else:
        print(f"\nNo existing cache file found at: {cache_path} (nothing to clear)")

    # 2. Initialize PyBullet for obj_to_generic
    print("\nInitializing PyBullet (headless) for OBJ loading...")
    setup_pybullet(gui=False)

    cache_data = {}

    # 3. Process each target shape
    for shape_name in TARGET_SHAPES:
        print("\n" + "-" * 60)
        print(f"Processing shape: {shape_name}")

        obj_file = f"{shape_name}.obj"
        print(f"  OBJ file: {obj_file}")

        try:
            generic_object, _ = obj_to_generic(
                obj_path=obj_file,
                shape_name=shape_name,
                position=(0.0, 0.0, 0.2),
                orientation=0.0,
                mass=DEFAULT_OBJECT_MASS,
                lateral_friction=DEFAULT_OBJECT_FRICTION,
                blind_test=True,
            )
        except Exception as e:
            print(f"  ✗ Failed to load OBJ for '{shape_name}': {e}")
            continue

        print(f"  ✓ Loaded GenericObject: '{generic_object.name}'")
        print(f"    Mass: {generic_object.mass:.3f} kg")
        print(f"    Moment of inertia: {generic_object.moment_of_inertia:.6f} kg·m²")

        # Run Magnum Four with reachability filtering
        print(f"  Computing Magnum Four contact points (with reachability filter)...")
        magnum_result = find_the_magnum_four_v3(
            generic_object,
            verbose=False,
            visualize=False,
            weighting_scheme="balanced",
            torque_method=3,
            robot_radius=effective_radius,
        )

        if not magnum_result or not magnum_result.get("success", False):
            print(f"  ✗ Magnum Four solver failed for '{shape_name}'")
            continue

        contacts = magnum_result["best_solution"]["contacts"]
        t_params = [float(c.parameter) for c in contacts]
        # Normalize into [0, 1) for cache
        t_params = [float(tp % 1.0) for tp in t_params]
        t_params = np.array(t_params, dtype=float).tolist()

        if len(t_params) != 4:
            print(f"  ✗ Expected 4 contacts, got {len(t_params)} for '{shape_name}' – skipping")
            continue

        print(f"  ✓ Magnum Four t_params for '{shape_name}': {[f'{v:.6f}' for v in t_params]}")
        cache_data[shape_name] = t_params

    # 4. Save updated cache
    if cache_data:
        with cache_path.open("w") as f:
            json.dump(cache_data, f, indent=2)
        print("\n" + "=" * 60)
        print(f"Saved Magnum Four cache to: {cache_path}")
        print("Shapes included:")
        for k in cache_data.keys():
            print(f"  - {k}")
        print("=" * 60)
    else:
        print("\n⚠️ No successful Magnum Four solutions; cache file not written.")

    pyb.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()

