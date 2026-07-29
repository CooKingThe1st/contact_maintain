#!/usr/bin/env python3
"""
Offline plotter for revised holonomic Magnum history JSON.

Use after a run (or after Ctrl-C left histories_live_*.json):

  python3 plot_revised_holonomic_histories.py \\
    --histories /tmp/revised_holo/histories_live_....json \\
    --save-dir /tmp/revised_holo/

Also accepts histories_final / histories_*.json from export_histories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rospkg

rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "scripts" / "test"))

from revised_holonomic_plots import (  # noqa: E402
    import_histories,
    plot_phase7_velocities,
    plot_phase7_wheel_plot,
    plot_phase_1_results,
    plot_phase_7beta,
)


def _stem_bits(hist_path: Path):
    """Parse run_tag / object from histories_*_w_*.json names when possible."""
    stem = hist_path.stem
    # histories_live_<tag>_w_<obj>  or  histories_<tag>_w_<obj>
    for prefix in ("histories_live_", "histories_"):
        if stem.startswith(prefix):
            rest = stem[len(prefix) :]
            if "_w_" in rest:
                tag, obj = rest.rsplit("_w_", 1)
                return tag, obj
    return "offline", "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Phase7 figures from revised holonomic history JSON."
    )
    parser.add_argument(
        "--histories",
        type=Path,
        required=True,
        help="Path to histories_*.json or histories_live_*.json",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: same dir as histories)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Override run tag used in output filenames",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Override object name used in output filenames",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=2.0,
        help="Contact force threshold for trajectory plots",
    )
    args = parser.parse_args()

    hist_path = Path(args.histories)
    if not hist_path.is_file():
        raise FileNotFoundError(hist_path)

    save_dir = Path(args.save_dir) if args.save_dir else hist_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    tag_guess, obj_guess = _stem_bits(hist_path)
    run_tag = args.tag or tag_guess
    object_name = args.object or obj_guess

    print(f"Loading {hist_path}")
    histories, t_params = import_histories(hist_path)
    n = len(histories)
    n_samp = max((len(h.times) for h in histories.values()), default=0)
    print(f"  robots={n}  samples≈{n_samp}  tag={run_tag}  object={object_name}")
    if n_samp == 0:
        raise RuntimeError("History file has no samples — nothing to plot.")

    prefix = f"{run_tag}_w_{object_name}"
    outs = {
        "velocities": save_dir / f"phase7_swarm_velocities_{prefix}.png",
        "trajectories": save_dir / f"phase7_swarm_trajectories_{prefix}.png",
        "beta": save_dir / f"phase7_beta_trajectories_{prefix}.png",
        "wheels": save_dir / f"phase7_wheel_velocities_{prefix}.png",
    }

    plot_phase7_velocities(
        histories=histories,
        t_params=t_params,
        desired_obj_velocity=np.array([0.0, 0.0]),
        desired_obj_omega=0.0,
        save_path=outs["velocities"],
    )
    plot_phase_1_results(
        histories=histories,
        t_params=t_params,
        contact_threshold=args.contact_threshold,
        save_path=outs["trajectories"],
    )
    plot_phase_7beta(
        histories=histories,
        t_params=t_params,
        save_path=outs["beta"],
    )
    plot_phase7_wheel_plot(
        histories=histories,
        t_params=t_params,
        save_path=outs["wheels"],
    )

    print("Wrote:")
    for p in outs.values():
        print(f"  {p}")


if __name__ == "__main__":
    main()
