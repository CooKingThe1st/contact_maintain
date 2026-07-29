#!/usr/bin/env python3
"""
Full Markenscoff benchmark orchestrator.

Phase 1: markenscoff-4 + friction-3 at T=1e-3 (form-closure proxy)
Phase 2: friction-3 at T=1, μ_contact=0.2, λ=2 (full AFC engineering)
Phase 3: retry Phase-2 failures at μ=0.5
Phase 4: retry Phase-3 failures at μ=0.8
"""

import subprocess
import sys
from pathlib import Path

import rospkg

rospack = rospkg.RosPack()
PKG = Path(rospack.get_path("contact_maintain"))
TEST_SCRIPT = PKG / "scripts" / "test" / "test_markenscoff_form_closure.py"
OUTPUT_DIR = Path("/tmp/markenscoff_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_phase(cmd, label):
    print("\n" + "#" * 80)
    print(f"# PHASE: {label}")
    print("#" * 80)
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PKG))
    if proc.returncode != 0:
        print(f"⚠️  Phase exited with code {proc.returncode}: {label}")
    return proc.returncode


def read_failures(csv_path):
    import csv

    if not csv_path.exists():
        return []
    failed = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("verdict") == "FAIL":
                failed.append(row["shape_name"])
    return failed


def main():
    py = sys.executable
    base = [
        py,
        str(TEST_SCRIPT),
        "--output-dir",
        str(OUTPUT_DIR),
        "--timeout",
        "15",
    ]

    # ------------------------------------------------------------------
    # Phase 1: small threshold — shape-first JPGs for n4 vs n3 comparison
    # ------------------------------------------------------------------
    run_phase(
        base
        + [
            "--experiment",
            "markenscoff-4",
            "--threshold",
            "1e-3",
            "--force-range-scalar",
            "10",
        ],
        "Phase 1a — markenscoff-4 T=1e-3 (auto tag)",
    )
    run_phase(
        base
        + [
            "--experiment",
            "friction-3",
            "--threshold",
            "1e-3",
            "--force-range-scalar",
            "10",
        ],
        "Phase 1b — friction-3 T=1e-3 (auto tag)",
    )

    # ------------------------------------------------------------------
    # Phase 2: full AFC friction-3 at T=1, μ=0.2, λ=2
    # ------------------------------------------------------------------
    tag2 = "n3_friction_T1_mu0.2_lam2"
    run_phase(
        base
        + [
            "--experiment",
            "friction-3",
            "--threshold",
            "1",
            "--force-range-scalar",
            "2",
            "--contact-friction",
            "0.2",
            "--no-expected-fail",
        ],
        "Phase 2 — friction-3 full AFC T=1 μ=0.2 λ=2",
    )
    csv2 = OUTPUT_DIR / f"{tag2}.csv"
    failed_02 = read_failures(csv2)
    print(f"\nPhase 2 failures at μ=0.2: {failed_02 or '(none)'}")

    # ------------------------------------------------------------------
    # Phase 3: retry failures at μ=0.5
    # ------------------------------------------------------------------
    failed_05 = []
    if failed_02:
        tag3 = "n3_friction_T1_mu0.5_lam2"
        run_phase(
            base
            + [
                "--experiment",
                "friction-3",
                "--threshold",
                "1",
                "--force-range-scalar",
                "2",
                "--contact-friction",
                "0.5",
                "--shapes",
                *failed_02,
                "--no-expected-fail",
            ],
            "Phase 3 — retry μ=0.5 for Phase-2 failures",
        )
        failed_05 = read_failures(OUTPUT_DIR / f"{tag3}.csv")
        print(f"\nPhase 3 still failing at μ=0.5: {failed_05 or '(none)'}")

    # ------------------------------------------------------------------
    # Phase 4: retry remaining at μ=0.8
    # ------------------------------------------------------------------
    if failed_05:
        run_phase(
            base
            + [
                "--experiment",
                "friction-3",
                "--threshold",
                "1",
                "--force-range-scalar",
                "2",
                "--contact-friction",
                "0.8",
                "--shapes",
                *failed_05,
                "--no-expected-fail",
            ],
            "Phase 4 — retry μ=0.8 for Phase-3 failures",
        )

    print("\n" + "=" * 80)
    print(f"All outputs in: {OUTPUT_DIR}")
    print("JPG naming: <shape>_<run_tag>.jpg  (sort by shape to compare n3 vs n4)")
    print("=" * 80)


if __name__ == "__main__":
    main()
