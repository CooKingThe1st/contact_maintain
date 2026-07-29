#!/usr/bin/env python3
"""
Compute wrench covariance M, degeneracy index D, and spectral floor λ_lb.

Section 11: D and σ₃ gate friction; λ_lb is diagnostic only (see doc warning).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import rospkg

rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "src"))
sys.path.insert(0, str(pkg_path / "src" / "legacy"))

from grasp_covariance import (  # noqa: E402
    calculate_grasp_covariance,
    format_grasp_covariance_report,
)
from object_utils import create_standard_objects  # noqa: E402


def _fmt_bound(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    if value >= 1000:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _fmt_lam(r: dict) -> str:
    lam = _fmt_bound(r["lambda_shape_lower_bound"])
    if not r.get("lambda_floor_trusted", False):
        return f"{lam}*"
    return lam
# Shapes that typically need tangent fallback in engineering search (reference).
TANGENT_FALLBACK_EXPECTED = frozenset({
    "circle",
    "crescent",
    "narrow_triangle",
    "symmetric_crescent",
})


def parse_args():
    p = argparse.ArgumentParser(description="Wrench covariance / degeneracy index D")
    p.add_argument(
        "--samples-per-edge",
        type=int,
        default=4,
        help="Interior samples per boundary edge (endpoints excluded); default 4",
    )
    p.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Run a single shape from create_standard_objects()",
    )
    p.add_argument(
        "--com",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Override center of torque (default: geometry centroid)",
    )
    p.add_argument(
        "--no-normalize-radius",
        action="store_true",
        help="Do not scale r = x - com by max boundary radius",
    )
    p.add_argument(
        "--soft-threshold",
        type=float,
        default=100.0,
        help="D >= this value ⇒ soft_degenerate (default 100)",
    )
    p.add_argument(
        "--sweep-density",
        action="store_true",
        help="For --shape only: print D vs samples_per_edge",
    )
    return p.parse_args()


def run_one(name, obj, args):
    result = calculate_grasp_covariance(
        obj,
        com=args.com,
        samples_per_edge=args.samples_per_edge,
        normalize_radius=not args.no_normalize_radius,
        soft_degeneracy_threshold=args.soft_threshold,
    )
    tangent_hint = " [tangent typical]" if name in TANGENT_FALLBACK_EXPECTED else ""
    print(format_grasp_covariance_report(result, name) + tangent_hint)
    return result


def main():
    args = parse_args()
    objects = create_standard_objects()

    if args.shape is not None:
        if args.shape not in objects:
            print(f"Unknown shape '{args.shape}'. Available: {sorted(objects)}")
            sys.exit(1)
        if args.sweep_density:
            print(f"Density sweep for {args.shape} (com={args.com}, T=1)")
            print(f"{'N/edge':>8} {'D':>12} {'K':>12} {'λ_lb':>12} {'class':>18}")
            for n in (1, 2, 4, 8, 16, 32):
                r = calculate_grasp_covariance(
                    objects[args.shape],
                    com=args.com,
                    samples_per_edge=n,
                    normalize_radius=not args.no_normalize_radius,
                    soft_degeneracy_threshold=args.soft_threshold,
                )
                d = r["degeneracy_index"]
                d_str = "inf" if not np.isfinite(d) else f"{d:.2f}"
                K_str = _fmt_bound(r["sobolev_K"])
                lam_str = _fmt_bound(r["lambda_shape_lower_bound"])
                print(
                    f"{n:8d} {d_str:>12} {K_str:>12} {lam_str:>12} "
                    f"{r['classification']:>18}"
                )
            return
        run_one(args.shape, objects[args.shape], args)
        return

    names = sorted(objects.keys())
    print("=" * 118)
    print(
        f"Grasp covariance / spectral bounds (T=1)  "
        f"(samples_per_edge={args.samples_per_edge}, "
        f"normalize_radius={not args.no_normalize_radius})"
    )
    print("  λ_floor = (T/(4√(K·σ₁)))√D — DIAGNOSTIC ONLY; gate on D and σ₃ (see Section 10–11)")
    print("=" * 118)

    rows = []
    for name in names:
        r = run_one(name, objects[name], args)
        rows.append((name, r))

    print("=" * 118)
    print(
        f"{'shape':18} {'σ₁':>9} {'σ₃':>9} {'D':>8} {'K':>7} {'Kd':>7} "
        f"{'λ_lb':>7} {'trust':>5} {'class':>16}"
    )
    print("-" * 118)
    for name, r in rows:
        d = r["degeneracy_index"]
        d_str = "inf" if not np.isfinite(d) else f"{d:.1f}"
        trust = "yes" if r.get("lambda_floor_trusted") else "no"
        print(
            f"{name:18} {r['sigma1']:9.3f} {r['sigma3']:9.3f} {d_str:>8} "
            f"{_fmt_bound(r['sobolev_K']):>7} "
            f"{_fmt_bound(r.get('sobolev_K_deriv', float('nan'))):>7} "
            f"{_fmt_lam(r):>7} {trust:>5} "
            f"{r['classification']:>16}"
        )


if __name__ == "__main__":
    main()
