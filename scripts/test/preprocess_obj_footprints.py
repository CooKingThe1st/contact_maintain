#!/usr/bin/env python3
"""
Precompute 2D footprints for OBJ-backed holonomic shapes.

This script:
  1. Starts a minimal headless PyBullet session (same as preprocess_magnum_contacts.py)
  2. Loads each target OBJ via obj_to_generic (authoritative geometry pipeline)
  3. Extracts the 2D boundary from the resulting GenericObject
  4. Saves vertices to urdf/obj_footprint_cache.json for fast runtime load

After preprocessing, HA_draw / mod_grid_SE / obj_to_generic read the cache and no
longer need trimesh slicing (or a PyBullet session) just to obtain footprints.

Usage:
  python preprocess_obj_footprints.py
  python preprocess_obj_footprints.py --shapes rect hourglass
  python preprocess_obj_footprints.py --gui
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pybullet as pyb
import pybullet_data
import rospkg


rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "src"))
sys.path.insert(0, str(pkg_path / "src" / "legacy"))

from contact_maintain.footprint_cache import CACHE_FILENAME, save_cache  # noqa: E402
from contact_maintain.object_bridge import obj_to_generic  # noqa: E402


DEFAULT_OBJECT_MASS = 2.0
DEFAULT_OBJECT_FRICTION = 0.8

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

    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])

    urdf_dir = pkg_path / "urdf"
    if urdf_dir.exists():
        pyb.setAdditionalSearchPath(str(urdf_dir))

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


def footprint_entry_from_generic(shape_name: str, obj_file: str, generic_object) -> Dict[str, Any]:
    coords = list(generic_object.geometry.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    vertices = [[float(x), float(y)] for x, y in coords]
    return {
        "obj_file": obj_file,
        "vertex_count": len(vertices),
        "area": float(generic_object.geometry.area),
        "boundary_length": float(getattr(generic_object, "boundary_length", 0.0)),
        "vertices": vertices,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute 2D OBJ footprints into urdf/obj_footprint_cache.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Shapes (default all):
  {", ".join(TARGET_SHAPES)}

Examples:
  python preprocess_obj_footprints.py
  python preprocess_obj_footprints.py --shapes rect hourglass --force
        """,
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=list(TARGET_SHAPES),
        help="Subset of shape names to process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Rewrite {CACHE_FILENAME} from scratch (default: merge/update selected shapes)",
    )
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI while preprocessing")
    args = parser.parse_args()

    unknown = [s for s in args.shapes if s not in TARGET_SHAPES]
    if unknown:
        parser.error(f"Unknown shapes: {unknown}. Available: {', '.join(TARGET_SHAPES)}")

    cache_path = pkg_path / "urdf" / CACHE_FILENAME
    print("=" * 60)
    print(" OBJ FOOTPRINT PREPROCESSING ")
    print("=" * 60)
    print(f"Package path: {pkg_path}")
    print(f"Cache path:   {cache_path}")
    print(f"Shapes:       {', '.join(args.shapes)}")

    if args.force and cache_path.exists():
        print(f"\n--force: removing existing cache {cache_path}")
        cache_path.unlink()

    existing: Dict[str, Any] = {}
    if cache_path.is_file() and not args.force:
        import json

        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        existing = dict(data.get("shapes", {}))
        print(f"\nLoaded {len(existing)} existing cache entries (will merge)")

    print("\nInitializing PyBullet (headless) for OBJ loading...")
    setup_pybullet(gui=args.gui)

    out_shapes = dict(existing)
    t0_all = time.time()

    for shape_name in args.shapes:
        print("\n" + "-" * 60)
        print(f"Processing shape: {shape_name}")
        obj_file = f"{shape_name}.obj"
        print(f"  OBJ file: {obj_file}")

        try:
            generic_object, body_uid = obj_to_generic(
                obj_path=obj_file,
                shape_name=shape_name,
                position=(0.0, 0.0, 0.2),
                orientation=0.0,
                mass=DEFAULT_OBJECT_MASS,
                lateral_friction=DEFAULT_OBJECT_FRICTION,
                blind_test=True,
            )
            entry = footprint_entry_from_generic(shape_name, obj_file, generic_object)
            out_shapes[shape_name] = entry
            print(
                f"  ✓ {entry['vertex_count']} vertices, "
                f"area={entry['area']:.4f}, boundary={entry['boundary_length']:.3f} m"
            )
            pyb.removeBody(body_uid)
        except Exception as e:
            print(f"  ✗ Failed for '{shape_name}': {e}")

    pyb.disconnect()

    if not out_shapes:
        print("\nNo footprints generated; cache not written.")
        return 1

    save_cache(out_shapes, cache_path)
    elapsed = time.time() - t0_all
    print("\n" + "=" * 60)
    print(f"Saved {len(out_shapes)} shape footprints to {cache_path}")
    print(f"Total time: {elapsed:.2f} s")
    print("=" * 60)

    try:
        if pyb.isConnected():
            pyb.disconnect()
    except Exception:
        pass
    # PyBullet + scipy teardown can segfault on normal interpreter exit; hard-exit is fine here.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
