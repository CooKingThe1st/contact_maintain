import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from contact_maintain.object_bridge import obj_to_generic
from object_utils import (
    get_reachable_contact_points,
    get_reachable_contact_intervals,
)


DEFAULT_OBJECT_MASS = 2.0
DEFAULT_OBJECT_FRICTION = 0.3
DEFAULT_ROBOT_RADIUS = 0.06


def setup_pybullet(gui: bool = False):
    """Initialize a minimal PyBullet scene for obj_to_generic."""
    if gui:
        pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    else:
        pyb.connect(pyb.DIRECT)

    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(1.0 / 240.0)
    pyb.setRealTimeSimulation(0)

    # Basic ground plane (matches other tests)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
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


def load_generic_object(shape_name: str, obj_file: Optional[str]):
    """
    Load a GenericObject from an OBJ-backed shape via obj_to_generic.
    """
    if obj_file is None:
        obj_file = f"{shape_name}.obj"

    generic_object, _ = obj_to_generic(
        obj_path=obj_file,
        shape_name=shape_name,
        position=(0.0, 0.0, 0.0),
        orientation=0.0,
        mass=DEFAULT_OBJECT_MASS,
        lateral_friction=DEFAULT_OBJECT_FRICTION,
        blind_test=True,
    )
    return generic_object


def compute_full_boundary_samples(geom, n_samples: int = 1024):
    """
    Uniformly sample the entire exterior boundary of a Polygon.
    """
    boundary = geom.exterior
    L = boundary.length
    if L <= 0:
        return np.zeros((0, 2), dtype=float)

    ts = np.linspace(0.0, L, int(n_samples), endpoint=False)
    pts = []
    for t in ts:
        p = boundary.interpolate(t)
        pts.append((p.x, p.y))
    return np.asarray(pts, dtype=float)


def sample_reachable_intervals_on_boundary(geom, intervals, n_samples: int):
    """
    Convert reachable t-intervals into sampled points on the boundary.

    Args:
        geom: Shapely Polygon geometry.
        intervals: list of (t_start, t_end) tuples in [0, 1].
        n_samples: total number of samples to draw across all intervals.
    """
    if not intervals or n_samples <= 0:
        return np.zeros((0, 2), dtype=float)

    boundary = geom.exterior
    L = boundary.length
    if L <= 0:
        return np.zeros((0, 2), dtype=float)

    # Total parameter measure covered by intervals
    total_t = sum(max(0.0, t1 - t0) for t0, t1 in intervals)
    if total_t <= 0:
        return np.zeros((0, 2), dtype=float)

    pts = []
    remaining_samples = n_samples

    for idx, (t0, t1) in enumerate(intervals):
        dt = max(0.0, t1 - t0)
        if dt <= 0:
            continue
        # Allocate samples proportional to interval length
        if idx == len(intervals) - 1:
            k = max(1, remaining_samples)
        else:
            k = max(1, int(round(n_samples * (dt / total_t))))
            remaining_samples -= k

        # Uniform samples in [t0, t1)
        for tau in np.linspace(t0, t1, k, endpoint=False):
            s = tau * L
            p = boundary.interpolate(s)
            pts.append((p.x, p.y))

    if not pts:
        return np.zeros((0, 2), dtype=float)

    return np.asarray(pts, dtype=float)


def plot_comparison(generic_object, reachable_pts, save_path: Optional[Path] = None):
    geom = generic_object.geometry
    full_pts = compute_full_boundary_samples(geom, n_samples=2048)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: full boundary (idealized: all non-vertex points reachable)
    ax = axes[0]
    x, y = geom.exterior.xy
    ax.fill(x, y, facecolor="lightgray", edgecolor="black", alpha=0.5)
    if len(full_pts) > 0:
        ax.plot(full_pts[:, 0], full_pts[:, 1], "b.", markersize=1, label="All boundary points")
    ax.set_aspect("equal")
    ax.set_title("Original object boundary\n(all boundary points)")
    ax.grid(True)

    # Right: reachable subset from C-space method
    ax = axes[1]
    ax.fill(x, y, facecolor="lightgray", edgecolor="black", alpha=0.5)
    if len(reachable_pts) > 0:
        ax.plot(reachable_pts[:, 0], reachable_pts[:, 1], "r.", markersize=2, label="Reachable contacts")
    ax.set_aspect("equal")
    ax.set_title("Reachable contact points\n(C-space / buffer-based)")
    ax.grid(True)

    for ax in axes:
        ax.legend(loc="best")

    fig.suptitle(f"Reachable contacts for '{generic_object.name}'")
    fig.tight_layout()

    # Always save plot; default to /tmp/basic_test if no explicit path is given
    if save_path is None:
        save_dir = Path("/tmp/basic_test")
        save_path = save_dir / f"{generic_object.name}_reachable_contacts.png"
    else:
        save_dir = save_path.parent

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved reachable contact comparison plot to: {save_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visual test for reachable contact points on GenericObject boundary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OBJ-based shape (default file: right_triangle.obj):
  python test_reachable_contacts.py --obj-shape right_triangle

  # Explicit OBJ file:
  python test_reachable_contacts.py --obj-shape right_triangle --obj-file meshes/right_triangle.obj
        """,
    )
    parser.add_argument(
        "--obj-shape",
        type=str,
        default="triangle",
        help="Shape name (must match create_standard_objects or an OBJ-based name).",
    )
    parser.add_argument(
        "--obj-file",
        type=str,
        default=None,
        help="OBJ file path (relative to urdf directory). If None, uses '{obj-shape}.obj' when not a standard shape.",
    )
    parser.add_argument(
        "--robot-radius",
        type=float,
        default=DEFAULT_ROBOT_RADIUS,
        help="Robot radius for C-space contact computation (default: 0.06).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=512,
        help="Number of samples along the reachable locus (default: 512).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save the comparison plot instead of showing it.",
    )
    parser.add_argument(
        "--use-intervals",
        action="store_true",
        help="Use get_reachable_contact_intervals and render reachable arcs instead of discrete C-space points.",
    )
    args = parser.parse_args()

    print("Initializing PyBullet (headless) for OBJ loading...")
    setup_pybullet(gui=False)

    generic_object = load_generic_object(args.obj_shape, args.obj_file)

    print(f"Loaded object '{generic_object.name}' with shape key '{args.obj_shape}'")
    print(f"  Geometry bounds: {generic_object.geometry.bounds}")
    print(f"  Robot radius: {args.robot_radius}")

    if args.use_intervals:
        intervals = get_reachable_contact_intervals(
            generic_object.geometry,
            robot_radius=args.robot_radius,
            n_samples=args.n_samples * 4,
        )
        print(f"Computed {len(intervals)} reachable intervals.")
        print(f"The intervals are {intervals}")
        reachable_pts = sample_reachable_intervals_on_boundary(
            generic_object.geometry,
            intervals,
            n_samples=args.n_samples,
        )
        print(f"Sampled {len(reachable_pts)} points from reachable intervals.")
    else:
        reachable_pts = get_reachable_contact_points(
            generic_object.geometry,
            robot_radius=args.robot_radius,
            n_samples=args.n_samples,
        )
        print(f"Computed {len(reachable_pts)} reachable contact samples.")

    save_path = Path(args.save_path) if args.save_path is not None else None
    plot_comparison(generic_object, reachable_pts, save_path=save_path)

    pyb.disconnect()


if __name__ == "__main__":
    main()

