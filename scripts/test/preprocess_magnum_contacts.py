#!/usr/bin/env python3
"""
Precompute Magnum Four contact configurations for a set of OBJ-backed shapes.

This script:
  1. Forcefully clears the existing Magnum Four cache at:
       urdf/magnum_four_cache.json
  2. Loads each target object via obj_to_generic
  3. Runs find_the_magnum_stochastic (default) or legacy find_the_magnum_four_v3
     with C-space reachability filter (robot_radius → get_reachable_contact_intervals)
  4. Saves the resulting t_params into the cache JSON.

Usage examples:
  # Default safety scale (1.5x robot radius)
  python preprocess_magnum_contacts.py

  # More conservative (2x robot radius)
  python preprocess_magnum_contacts.py --safety-scale 2.0
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
from stochastic_magnum_finder import find_the_magnum_stochastic  # noqa: E402
from grasp_covariance import (  # noqa: E402
    DEFAULT_SOFT_DEGENERACY_THRESHOLD,
    calculate_grasp_covariance,
    format_grasp_covariance_report,
    recommend_tangent_fallback,
    screening_fields_for_log,
)


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
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])

    urdf_dir = pkg_path / "urdf"
    if urdf_dir.exists():
        pyb.setAdditionalSearchPath(str(urdf_dir))

    # Basic ground plane
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
        help=(
            "Multiplier on the nominal robot radius (0.06 m) used for "
            "C-space reachability filtering. Effective radius = "
            "safety_scale * 0.06."
        ),
    )
    parser.add_argument(
        "--solver",
        choices=["stochastic", "v3"],
        default="stochastic",
        help=(
            "Magnum solver to use. "
            "'stochastic' uses the newer Latin-square-based search "
            "(`find_the_magnum_stochastic`), which is typically faster; "
            "'v3' uses the legacy deterministic solver "
            "(`find_the_magnum_four_v3`)."
        ),
    )
    parser.add_argument(
        "--soft-threshold",
        type=float,
        default=DEFAULT_SOFT_DEGENERACY_THRESHOLD,
        help=(
            f"D >= this value recommends tangent fallback before search "
            f"(default {DEFAULT_SOFT_DEGENERACY_THRESHOLD}, Section 11)."
        ),
    )
    parser.add_argument(
        "--samples-per-edge",
        type=int,
        default=4,
        help="Interior boundary samples per edge for grasp covariance (default 4)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "CSV log path for degeneracy screening and solver results "
            "(default: urdf/magnum_preprocess_screening.csv)"
        ),
    )
    parser.add_argument(
        "--force-tangent",
        action="store_true",
        help="Always enable used_tangent_as_fallback (ignore D gate)",
    )
    parser.add_argument(
        "--ignore-degeneracy-gate",
        action="store_true",
        help="Never enable tangent from D screening (normal-only stochastic pass only)",
    )
    parser.add_argument(
        "--no-retry-tangent-on-failure",
        action="store_true",
        help=(
            "Disable Section 11 step 3: do not retry with tangent when D gate "
            "said normal-only (default: retry enabled for OBJ preprocessing)"
        ),
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
    print(f"Solver: {args.solver}")
    print(f"Degeneracy gate: D_soft={args.soft_threshold:.1f}")

    cache_path = pkg_path / "urdf" / "magnum_four_cache.json"
    csv_path = args.csv or (pkg_path / "urdf" / "magnum_preprocess_screening.csv")
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
    csv_rows: List[Dict[str, Any]] = []

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

        cov = calculate_grasp_covariance(
            generic_object,
            samples_per_edge=args.samples_per_edge,
            soft_degeneracy_threshold=args.soft_threshold,
        )
        tangent_rec = recommend_tangent_fallback(
            cov,
            soft_degeneracy_threshold=args.soft_threshold,
        )
        screening = screening_fields_for_log(cov, tangent_rec)
        print(f"    {format_grasp_covariance_report(cov, shape_name)}")
        if tangent_rec["recommend_tangent_fallback"]:
            print(
                f"    🔶 Tangent recommended: {tangent_rec['reason']} "
                f"(D={screening['degeneracy_index']:.2f})"
            )
        else:
            print(
                f"    🟢 Normal-only by D gate "
                f"(D={screening['degeneracy_index']:.2f}, "
                f"class={screening['degeneracy_classification']})"
            )

        if args.force_tangent:
            tangent_required = True
            use_tangent_fallback = True
        elif args.ignore_degeneracy_gate:
            tangent_required = False
            use_tangent_fallback = not args.no_retry_tangent_on_failure
        else:
            tangent_required = tangent_rec["recommend_tangent_fallback"]
            use_tangent_fallback = (
                tangent_required or not args.no_retry_tangent_on_failure
            )

        row: Dict[str, Any] = {
            "shape_name": shape_name,
            "solver": args.solver,
            "safety_scale": args.safety_scale,
            "effective_robot_radius_m": effective_radius,
            "force_range_scalar": 2.0,
            "tangent_required": tangent_required,
            "used_tangent_as_fallback_requested": use_tangent_fallback,
            "cache_saved": False,
            "success": False,
            "search_used_tangent_fallback": False,
            "elapsed_time_s": None,
            "t_params": "",
            "error": "",
            **screening,
        }

        # Run Magnum solver with reachability filtering
        if args.solver == "stochastic":
            print(
                "  Computing Magnum Four contact points using stochastic "
                f"solver (tangent_required={tangent_required}, "
                f"tangent_fallback={use_tangent_fallback and not tangent_required})..."
            )
            t0 = time.time()
            magnum_result = find_the_magnum_stochastic(
                generic_object,
                threshold=1.0,
                timeout=10.0,
                force_range_scalar=2.0,
                robot_radius=effective_radius,
                used_tangent_as_fallback=use_tangent_fallback and not tangent_required,
                tangent_required=tangent_required,
                verbose=False,
            )
            row["elapsed_time_s"] = time.time() - t0

            if not magnum_result or not magnum_result.get("success", False):
                row["error"] = "stochastic_solver_failed"
                csv_rows.append(row)
                print(f"  ✗ Stochastic Magnum solver failed for '{shape_name}'")
                continue

            row["success"] = True
            row["search_used_tangent_fallback"] = bool(
                magnum_result.get("used_tangent_fallback", False)
            )
            contacts = magnum_result.get("contacts", [])
            if not contacts:
                row["success"] = False
                row["error"] = "no_contacts_returned"
                csv_rows.append(row)
                print(
                    f"  ✗ Stochastic Magnum solver returned no contacts for "
                    f"'{shape_name}'"
                )
                continue
        else:
            print(
                "  Computing Magnum Four contact points using legacy v3 solver "
                "(with reachability filter)..."
            )
            t0 = time.time()
            magnum_result = find_the_magnum_four_v3(
                generic_object,
                verbose=False,
                visualize=False,
                weighting_scheme="balanced",
                torque_method=3,
                robot_radius=effective_radius,
            )
            row["elapsed_time_s"] = time.time() - t0

            if not magnum_result or not magnum_result.get("success", False):
                row["error"] = "v3_solver_failed"
                csv_rows.append(row)
                print(f"  ✗ Magnum Four v3 solver failed for '{shape_name}'")
                continue

            row["success"] = True
            contacts = magnum_result["best_solution"]["contacts"]
        t_params = [float(c.parameter) for c in contacts]
        # Normalize into [0, 1) for cache
        t_params = [float(tp % 1.0) for tp in t_params]
        t_params = np.array(t_params, dtype=float).tolist()

        if len(t_params) != 4:
            row["error"] = f"expected_4_contacts_got_{len(t_params)}"
            csv_rows.append(row)
            print(f"  ✗ Expected 4 contacts, got {len(t_params)} for '{shape_name}' – skipping")
            continue

        row["t_params"] = ",".join(f"{v:.6f}" for v in t_params)
        row["cache_saved"] = True
        csv_rows.append(row)

        print(f"  ✓ Magnum Four t_params for '{shape_name}': {[f'{v:.6f}' for v in t_params]}")
        if row.get("search_used_tangent_fallback"):
            print("    (solution found with tangent-force fallback)")
        cache_data[shape_name] = t_params

    # 4. Save updated cache
    if csv_rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(csv_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved screening CSV: {csv_path}")

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

