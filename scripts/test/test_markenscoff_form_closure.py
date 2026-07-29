#!/usr/bin/env python3
"""
Verify Markenscoff / friction-reduced form-closure claims via stochastic search.

Experiments:
  - markenscoff-4: 4 frictionless contacts
  - friction-3:    3 contacts with friction cone (lateral_friction at contact)

Viz/CSV names use shape-first tags so JPGs for the same object sort together, e.g.:
  rectangle_n4_markenscoff_T1e-3.jpg
  rectangle_n3_friction_T1e-3.jpg
  rectangle_n3_friction_T1_mu0.2_lam2.jpg
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import rospkg

rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "src"))
sys.path.insert(0, str(pkg_path / "src" / "legacy"))
sys.path.insert(0, str(pkg_path / "scripts" / "test"))

from stochastic_magnum_finder import find_the_magnum_stochastic  # noqa: E402
from object_utils import create_standard_objects  # noqa: E402
from grasp_covariance import (  # noqa: E402
    DEFAULT_SOFT_DEGENERACY_THRESHOLD,
    calculate_grasp_covariance,
    format_grasp_covariance_report,
    recommend_tangent_fallback,
    screening_fields_for_log,
)
from test_stochastic_magnum import visualize_stochastic_solution  # noqa: E402


EXPERIMENT_PRESETS = {
    "markenscoff-4": {
        "n_contacts": 4,
        "tangent_required": False,
        "used_tangent_as_fallback": False,
        "description": "Markenscoff: 4 frictionless contacts",
        "expected_fail_shapes": {"circle"},
    },
    "friction-3": {
        "n_contacts": 3,
        "tangent_required": True,
        "used_tangent_as_fallback": False,
        "description": "3 contacts with friction cone at contact",
        "expected_fail_shapes": set(),
    },
}


def format_threshold_tag(threshold: float) -> str:
    if abs(threshold - 1.0) < 1e-12:
        return "T1"
    if abs(threshold - 1e-3) < 1e-15:
        return "T1e-3"
    return f"T{threshold:g}"


def build_run_tag(
    experiment_name: str,
    n_contacts: int,
    threshold: float,
    contact_friction=None,
    force_range_scalar=None,
) -> str:
    """Shape-first filename suffix: n4_markenscoff_T1e-3, n3_friction_T1_mu0.2_lam2."""
    if experiment_name == "markenscoff-4":
        exp_part = "markenscoff"
    else:
        exp_part = "friction"
    tag = f"n{n_contacts}_{exp_part}_{format_threshold_tag(threshold)}"
    if contact_friction is not None and experiment_name == "friction-3":
        tag += f"_mu{contact_friction:g}"
    if force_range_scalar is not None:
        tag += f"_lam{force_range_scalar:g}"
    return tag


def build_viz_title(
    shape_name: str,
    run_tag: str,
    n_contacts: int,
    threshold: float,
    force_range_scalar: float,
    contact_friction,
    ground_static_friction: float,
    search_used_tangent: bool,
) -> str:
    parts = [
        shape_name.upper(),
        f"n={n_contacts}",
        f"T={threshold:g}",
        f"λ={force_range_scalar:g}",
        f"μ_gnd={ground_static_friction:g}",
    ]
    if search_used_tangent and contact_friction is not None:
        parts.append(f"μ_contact={contact_friction:g}")
    elif search_used_tangent:
        parts.append("μ_contact=default")
    parts.append(run_tag)
    return " | ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Markenscoff / friction-3 form-closure verification.",
    )
    parser.add_argument(
        "--experiment",
        choices=["markenscoff-4", "friction-3", "both"],
        default="both",
        help="Which experiment(s) to run (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/markenscoff_benchmark"),
        help="Directory for visualizations and CSV",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path (default: <output-dir>/<run-tag>.csv)",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Suffix for viz/CSV names (auto-generated from params if omitted)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="LS scale factor T (default 1e-3)",
    )
    parser.add_argument(
        "--force-range-scalar",
        type=float,
        default=10.0,
        help="Force cap multiplier λ_hw (default 10.0)",
    )
    parser.add_argument(
        "--contact-friction",
        type=float,
        default=None,
        help="Override lateral_friction (μ_contact) for friction-3 runs",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Per-shape search timeout in seconds",
    )
    parser.add_argument(
        "--robot-radius",
        type=float,
        default=0.06,
        help="Robot radius for reachability/spacing filters",
    )
    parser.add_argument(
        "--theory-mode",
        action="store_true",
        help="Skip robot-spacing and quick FC prune",
    )
    parser.add_argument(
        "--soft-threshold",
        type=float,
        default=DEFAULT_SOFT_DEGENERACY_THRESHOLD,
        help="D soft threshold for degeneracy logging",
    )
    parser.add_argument(
        "--samples-per-edge",
        type=int,
        default=4,
        help="Boundary samples per edge for grasp covariance screening",
    )
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=None,
        help="Optional subset of shape names",
    )
    parser.add_argument(
        "--no-expected-fail",
        action="store_true",
        help="Treat all failures as FAIL (no EXPECTED_FAIL for circle)",
    )
    return parser.parse_args()


def write_results_csv(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n📄 Wrote CSV: {csv_path}")


def run_experiment(
    experiment_name,
    preset,
    standard_objects,
    shape_names,
    args,
    output_dir,
    run_tag,
    expected_fail_shapes,
):
    """Run one experiment preset over all shapes; return result rows."""
    contact_friction = (
        args.contact_friction if experiment_name == "friction-3" else None
    )

    print("\n" + "=" * 80)
    print(f"🔬 EXPERIMENT: {experiment_name}  [{run_tag}]")
    print(f"   {preset['description']}")
    print(
        f"   n_contacts={preset['n_contacts']}, T={args.threshold}, "
        f"λ_hw={args.force_range_scalar}, theory_mode={args.theory_mode}"
    )
    if contact_friction is not None:
        print(f"   μ_contact (override)={contact_friction}")
    print("=" * 80)

    rows = []
    for idx, shape_name in enumerate(shape_names, 1):
        obj = standard_objects[shape_name]
        orig_lateral = obj.lateral_friction
        orig_static = obj.static_friction
        if contact_friction is not None:
            obj.lateral_friction = contact_friction

        print(f"\n[{idx}/{len(shape_names)}] {experiment_name} / {shape_name.upper()}")
        print("-" * 80)
        print(
            f"   Object friction: μ_contact={obj.lateral_friction:g}, "
            f"μ_ground(static)={obj.static_friction:g}"
        )

        try:
            cov = calculate_grasp_covariance(
                obj,
                samples_per_edge=args.samples_per_edge,
                soft_degeneracy_threshold=args.soft_threshold,
            )
            tangent_rec = recommend_tangent_fallback(
                cov,
                soft_degeneracy_threshold=args.soft_threshold,
            )
            screening = screening_fields_for_log(cov, tangent_rec)
            print(f"   {format_grasp_covariance_report(cov, shape_name)}")
            print(
                f"   D gate (logged): recommend_tangent={tangent_rec['recommend_tangent_fallback']} "
                f"(D={screening['degeneracy_index']:.2f})"
            )

            t0 = time.time()
            result = find_the_magnum_stochastic(
                obj,
                verbose=False,
                threshold=args.threshold,
                timeout=args.timeout,
                force_range_scalar=args.force_range_scalar,
                robot_radius=args.robot_radius,
                n_contacts=preset["n_contacts"],
                theory_mode=args.theory_mode,
                tangent_required=preset["tangent_required"],
                used_tangent_as_fallback=preset["used_tangent_as_fallback"],
            )
            elapsed = time.time() - t0

            success = result.get("success", False)
            search_used_tangent = result.get("used_tangent_fallback", False)
            pruned_count = result.get("pruned_count", {}) or {}
            total_pruned = sum(pruned_count.values())
            expected_fail = shape_name in expected_fail_shapes
            verdict = "EXPECTED_FAIL" if (not success and expected_fail) else (
                "PASS" if success else "FAIL"
            )

            viz_path = None
            if success:
                contacts = result.get("contacts", [])
                if contacts:
                    viz_filename = f"{shape_name}_{run_tag}.jpg"
                    viz_path = output_dir / viz_filename
                    viz_title = build_viz_title(
                        shape_name,
                        run_tag,
                        preset["n_contacts"],
                        args.threshold,
                        args.force_range_scalar,
                        obj.lateral_friction if search_used_tangent else None,
                        obj.static_friction,
                        search_used_tangent,
                    )
                    try:
                        # Cone drawing policy (visual only; search logic unchanged):
                        #   n=3: always draw ±atan(μ) friction cone
                        #   n=4: if T < 1 never draw; if T >= 1 draw only when
                        #        tangent forces were needed (degenerate / fallback)
                        n_contacts = preset["n_contacts"]
                        if n_contacts == 3:
                            show_friction_cones = True
                        elif n_contacts == 4:
                            show_friction_cones = (
                                args.threshold >= 1.0 and search_used_tangent
                            )
                        else:
                            show_friction_cones = False

                        visualize_stochastic_solution(
                            obj,
                            contacts,
                            viz_title,
                            str(viz_path),
                            threshold=args.threshold,
                            force_range_scalar=args.force_range_scalar,
                            enable_tangent_forces=search_used_tangent,
                            show_friction_cones=show_friction_cones,
                        )
                        print(f"   💾 Saved visualization: {viz_filename}")
                    except Exception as exc:
                        print(f"   ⚠️  Visualization failed: {exc}")
                        viz_path = None

            row = {
                "run_tag": run_tag,
                "experiment": experiment_name,
                "shape_name": shape_name,
                "n_contacts": preset["n_contacts"],
                "threshold": args.threshold,
                "force_range_scalar": args.force_range_scalar,
                "contact_friction": obj.lateral_friction if search_used_tangent else 0.0,
                "contact_friction_override": contact_friction,
                "ground_static_friction": obj.static_friction,
                "theory_mode": args.theory_mode,
                "tangent_required": preset["tangent_required"],
                "search_used_tangent_fallback": search_used_tangent,
                "success": success,
                "verdict": verdict,
                "expected_fail": expected_fail,
                "elapsed_time": elapsed,
                "configs_tested": result.get("configs_tested", 0),
                "batches_tested": result.get("batches_tested", 0),
                "total_pruned": total_pruned,
                "pruned_count": str(pruned_count),
                "viz_path": str(viz_path) if viz_path else None,
                **screening,
            }
            rows.append(row)

            status = "✅" if success else ("⚠️ " if expected_fail else "❌")
            print(
                f"   {status} {verdict} | {elapsed:.3f}s | "
                f"configs={row['configs_tested']} | pruned={total_pruned}"
            )
        finally:
            obj.lateral_friction = orig_lateral
            obj.static_friction = orig_static

    return rows


def print_summary(all_rows):
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    for run_tag in sorted({r["run_tag"] for r in all_rows}):
        tag_rows = [r for r in all_rows if r["run_tag"] == run_tag]
        passed = [r for r in tag_rows if r["verdict"] == "PASS"]
        expected = [r for r in tag_rows if r["verdict"] == "EXPECTED_FAIL"]
        failed = [r for r in tag_rows if r["verdict"] == "FAIL"]
        rate = len(passed) / len(tag_rows) * 100 if tag_rows else 0.0
        sample = tag_rows[0]

        print(f"\n[{run_tag}]")
        print(
            f"   T={sample['threshold']}, λ={sample['force_range_scalar']}, "
            f"μ_contact={sample.get('contact_friction_override', 'default')}"
        )
        print(f"   PASS            : {len(passed)}/{len(tag_rows)} ({rate:.1f}%)")
        print(f"   EXPECTED_FAIL   : {len(expected)}")
        print(f"   UNEXPECTED_FAIL : {len(failed)}")
        if failed:
            print(f"   Failed shapes   : {', '.join(r['shape_name'] for r in failed)}")

    print("\n" + "-" * 100)
    print(
        f"{'RunTag':<32} {'Shape':<14} {'Verdict':<14} "
        f"{'μ_c':<6} {'λ':<5} {'Tan':<4} {'Time':<7} {'Cfg':<6}"
    )
    print("-" * 100)
    for r in all_rows:
        mu_c = r.get("contact_friction_override")
        mu_str = f"{mu_c:g}" if mu_c is not None else "-"
        tan_used = "yes" if r["search_used_tangent_fallback"] else "no"
        print(
            f"{r['run_tag']:<32} {r['shape_name']:<14} {r['verdict']:<14} "
            f"{mu_str:<6} {r['force_range_scalar']:<5.1f} {tan_used:<4} "
            f"{r['elapsed_time']:<7.3f} {r['configs_tested']:<6}"
        )


def resolve_run_tag(args, experiment_name, preset):
    if args.run_tag:
        return args.run_tag
    return build_run_tag(
        experiment_name,
        preset["n_contacts"],
        args.threshold,
        args.contact_friction if experiment_name == "friction-3" else None,
        args.force_range_scalar,
    )


def main(argv=None):
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_objects = create_standard_objects()
    if args.shapes:
        shape_names = [s for s in args.shapes if s in standard_objects]
        unknown = set(args.shapes) - set(shape_names)
        if unknown:
            print(f"⚠️  Unknown shapes ignored: {', '.join(sorted(unknown))}")
    else:
        shape_names = sorted(standard_objects.keys())

    experiments = (
        ["markenscoff-4", "friction-3"]
        if args.experiment == "both"
        else [args.experiment]
    )

    print("\n" + "=" * 80)
    print("🧪 MARKENSCOFF / FRICTION-3 FORM-CLOSURE VERIFICATION")
    print("=" * 80)
    print(f"Shapes ({len(shape_names)}): {', '.join(shape_names)}")
    print(f"Output: {output_dir}")

    all_rows = []
    total_t0 = time.time()
    for exp_name in experiments:
        preset = dict(EXPERIMENT_PRESETS[exp_name])
        if args.no_expected_fail:
            preset["expected_fail_shapes"] = set()
        run_tag = resolve_run_tag(args, exp_name, preset)
        csv_path = args.csv or (output_dir / f"{run_tag}.csv")

        rows = run_experiment(
            exp_name,
            preset,
            standard_objects,
            shape_names,
            args,
            output_dir,
            run_tag,
            preset["expected_fail_shapes"],
        )
        all_rows.extend(rows)
        write_results_csv(csv_path, rows)

    print_summary(all_rows)

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print(f"✅ Completed in {total_elapsed:.3f} s")
    print("=" * 80)
    return all_rows


if __name__ == "__main__":
    main()
