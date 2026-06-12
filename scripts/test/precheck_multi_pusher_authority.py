#!/usr/bin/env python3
"""Pre-push authority diagnostic for multi-pusher diff-drive fixed-ref control.

This script does not spawn or push robots.  It loads the object geometry, chooses
contact parameters, solves the fixed-ref diff-drive command for each contact, and
classifies the expected role from the normal component of the commanded motion.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import ContactPointParameterization
from contact_maintain.object_bridge import obj_to_generic
from contact_optimizer_utils import find_the_magnum_four_v3
from contact_maintain.diffdrive_path_control import solve_constant_body_twist_from_SE2


DEFAULT_OBJECT_SHAPE = "rect"
DEFAULT_OBJECT_HEIGHT = 0.08
DEFAULT_OBJECT_FRICTION = 0.8
ROBOT_RADIUS = 0.06

OBJECT_FILE_MAP = {
    "right_triangle": "right_triangle.obj",
    "pi": "pi.obj",
    "root": "root.obj",
    "rect": "rect.obj",
    "hourglass": "hourglass.obj",
    "meteor": "meteor.obj",
}


@dataclass
class AuthorityRow:
    robot: str
    t_param: float
    cp_body: List[float]
    n_in_body: List[float]
    v_cp_body: List[float]
    cp_speed: float
    vr_ff: float
    omega_ff: float
    zeta0: float
    alpha_star: float
    true_alpha_deg: float
    normal_ratio: float
    normal_authority: float
    tangential_authority: float
    branch_sign: float
    role: str


@dataclass
class SegmentTwist:
    segment_idx: int
    t_start: float
    t_end: float
    p_start: List[float]
    p_end: List[float]
    theta_start: float
    theta_end: float
    v_ref_body: List[float]
    omega_ref: float
    duration: float


def _wrap_angle(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def _compute_body_cp_velocity(contact_point_body: np.ndarray, v_ref_body: np.ndarray, omega_ref: float) -> np.ndarray:
    r_b = np.asarray(contact_point_body, dtype=float).reshape(2)
    return np.asarray(v_ref_body, dtype=float).reshape(2) + float(omega_ref) * np.array([-r_b[1], r_b[0]], dtype=float)


def _init_segment_reference(
    phi0: float,
    v_cp_ref_world: np.ndarray,
    omega_ref: float,
    robot_heading: float,
    branch_sign: Optional[float] = None,
) -> Dict:
    a = float(v_cp_ref_world[0] + omega_ref * ROBOT_RADIUS * np.sin(phi0))
    b = float(v_cp_ref_world[1] - omega_ref * ROBOT_RADIUS * np.cos(phi0))
    vr_mag = float(np.hypot(a, b))
    zeta_fwd = float(np.arctan2(b, a))
    zeta_bwd = zeta_fwd + float(np.pi)

    if branch_sign is not None:
        use_forward = branch_sign >= 0.0
    else:
        use_forward = abs(_wrap_angle(zeta_fwd - robot_heading)) <= abs(_wrap_angle(zeta_bwd - robot_heading))

    if use_forward:
        zeta0, vr_ff = zeta_fwd, vr_mag
    else:
        zeta0, vr_ff = zeta_bwd, -vr_mag

    return {
        "vr_ff": float(vr_ff),
        "omega_ff": float(omega_ref),
        "zeta0": float(zeta0),
        "alpha_star": float(_wrap_angle(phi0 - zeta0)),
        "branch_sign": 1.0 if vr_ff >= 0.0 else -1.0,
    }


def _setup_pybullet() -> None:
    pyb.connect(pyb.DIRECT)
    pyb.setGravity(0, 0, -9.81)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf")


def _parse_t_params(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    vals = [float(part.strip()) % 1.0 for part in raw.split(",") if part.strip()]
    if not vals:
        raise ValueError("--t-params was provided but no values were parsed.")
    return vals


def _load_cached_t_params(object_name: str) -> Optional[List[float]]:
    cache_file = Path(pkg_path) / "urdf" / "magnum_four_cache.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    vals = cache_data.get(object_name)
    if vals is None:
        return None
    return [float(v) % 1.0 for v in vals]


def _save_t_params_to_cache(object_name: str, t_params: List[float]) -> None:
    cache_file = Path(pkg_path) / "urdf" / "magnum_four_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache_data = {}
    cache_data[object_name] = [float(t) % 1.0 for t in t_params]
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


def _classify_role(normal_ratio: float, vr_ff: float, passive_ratio: float, speed_floor: float) -> str:
    if abs(vr_ff) < speed_floor:
        return "passive_low_speed"
    if normal_ratio > passive_ratio:
        return "active_pusher"
    if normal_ratio < -passive_ratio:
        return "obstructing_candidate"
    return "passive_tangential"


def _load_object_states_csv(csv_path: Path) -> List[Dict[str, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Object states CSV not found: {csv_path}")
    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # We relax the CSV time constraint: only (x, y, theta) is required.
        # If a `t` column exists, we keep it only for labeling/export.
        required = {"x", "y", "theta"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"CSV must include columns {sorted(required)}; missing {sorted(missing)}"
            )
        for i, row in enumerate(reader, start=2):
            try:
                t_val = float(row["t"]) if row.get("t") not in (None, "") else float(i)
                rows.append(
                    {
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "theta": float(row["theta"]),
                        "t": t_val,
                    }
                )
            except Exception as exc:
                raise ValueError(f"Invalid numeric value in {csv_path} line {i}: {exc}") from exc
    if len(rows) < 2:
        raise ValueError("CSV must contain at least two object states.")
    return rows


def _build_segment_twists_from_states(
    states: List[Dict[str, float]],
    v_speed: float,
) -> List[SegmentTwist]:
    segments: List[SegmentTwist] = []
    for idx in range(len(states) - 1):
        s0 = states[idx]
        s1 = states[idx + 1]
        p0 = np.array([s0["x"], s0["y"]], dtype=float)
        p1 = np.array([s1["x"], s1["y"]], dtype=float)
        th0 = float(s0["theta"])
        th1 = float(s1["theta"])

        c, ss = np.cos(th0), np.sin(th0)
        Rinv = np.array([[c, ss], [-ss, c]], dtype=float)
        local_delta = Rinv @ (p1 - p0)
        dx, dy = float(local_delta[0]), float(local_delta[1])
        dtheta = _wrap_angle(th1 - th0)

        v_body_unit, omega_unit, T_unit = solve_constant_body_twist_from_SE2(
            dx, dy, dtheta, v_speed=v_speed
        )
        v_body_exec = np.asarray(v_body_unit, dtype=float)
        omega_exec = float(omega_unit)
        T_exec = float(T_unit)

        segments.append(
            SegmentTwist(
                segment_idx=idx,
                t_start=float(s0["t"]),
                t_end=float(s1["t"]),
                p_start=[float(p0[0]), float(p0[1])],
                p_end=[float(p1[0]), float(p1[1])],
                theta_start=th0,
                theta_end=th1,
                v_ref_body=[float(v_body_exec[0]), float(v_body_exec[1])],
                omega_ref=omega_exec,
                duration=T_exec,
            )
        )
    return segments


def analyze_authority(
    parameterization: ContactPointParameterization,
    t_params: List[float],
    v_ref_body: np.ndarray,
    omega_ref: float,
    passive_ratio: float,
    speed_floor: float,
) -> List[AuthorityRow]:
    rows: List[AuthorityRow] = []
    for i, t_param in enumerate(t_params):
        info = parameterization.get_contact_info(float(t_param))
        cp_b = np.array(info["point"], dtype=float)
        n_out_b = np.array(info["normal_outward"], dtype=float)
        n_in_b = -n_out_b
        phi0 = float(np.arctan2(n_in_b[1], n_in_b[0]))
        initial_heading = phi0
        v_cp_b = _compute_body_cp_velocity(cp_b, v_ref_body, omega_ref)

        seg_ref = _init_segment_reference(
            phi0=phi0,
            v_cp_ref_world=v_cp_b,
            omega_ref=omega_ref,
            robot_heading=initial_heading,
        )
        vr_ff = float(seg_ref["vr_ff"])
        zeta0 = float(seg_ref["zeta0"])
        drive_dir = np.array([np.cos(zeta0), np.sin(zeta0)], dtype=float)

        if abs(vr_ff) >= 1e-12:
            move_dir = float(np.sign(vr_ff)) * drive_dir
            normal_ratio = float(np.dot(n_in_b, move_dir))
            tangential_authority = float(abs(vr_ff) * np.sqrt(max(0.0, 1.0 - normal_ratio**2)))
            true_alpha = float(np.arctan2(
                n_in_b[0] * move_dir[1] - n_in_b[1] * move_dir[0],
                np.dot(n_in_b, move_dir),
            ))
        else:
            normal_ratio = 0.0
            tangential_authority = 0.0
            true_alpha = 0.0

        normal_authority = float(abs(vr_ff) * normal_ratio)
        role = _classify_role(normal_ratio, vr_ff, passive_ratio, speed_floor)

        rows.append(AuthorityRow(
            robot=f"R_{i + 1:02d}",
            t_param=float(t_param),
            cp_body=[float(cp_b[0]), float(cp_b[1])],
            n_in_body=[float(n_in_b[0]), float(n_in_b[1])],
            v_cp_body=[float(v_cp_b[0]), float(v_cp_b[1])],
            cp_speed=float(np.linalg.norm(v_cp_b)),
            vr_ff=vr_ff,
            omega_ff=float(seg_ref["omega_ff"]),
            zeta0=zeta0,
            alpha_star=float(seg_ref["alpha_star"]),
            true_alpha_deg=float(np.degrees(abs(true_alpha))),
            normal_ratio=normal_ratio,
            normal_authority=normal_authority,
            tangential_authority=tangential_authority,
            branch_sign=float(seg_ref["branch_sign"]),
            role=role,
        ))
    return rows


def _role_color(role: str) -> str:
    return {
        "active_pusher": "tab:green",
        "passive_tangential": "tab:orange",
        "passive_low_speed": "tab:gray",
        "obstructing_candidate": "tab:red",
    }.get(role, "tab:blue")


def _plot_authority_panel(
    axes: np.ndarray,
    rows: List[AuthorityRow],
    parameterization: ContactPointParameterization,
    v_ref_body: np.ndarray,
    omega_ref: float,
    passive_ratio: float,
    speed_floor: float,
    title_prefix: str,
) -> None:
    names = [r.robot for r in rows]
    x = np.arange(len(rows))
    colors = [_role_color(r.role) for r in rows]

    normal_authority = np.array([r.normal_authority for r in rows], dtype=float)
    normal_ratio = np.array([r.normal_ratio for r in rows], dtype=float)
    true_alpha = np.array([r.true_alpha_deg for r in rows], dtype=float)
    vr_ff = np.array([r.vr_ff for r in rows], dtype=float)
    cp_speed = np.array([r.cp_speed for r in rows], dtype=float)
    tangential_authority = np.array([r.tangential_authority for r in rows], dtype=float)

    ax_geo = axes[0][0]
    boundary = np.asarray(parameterization.boundary_coords, dtype=float)
    if len(boundary):
        closed = np.vstack([boundary, boundary[0]])
        ax_geo.plot(closed[:, 0], closed[:, 1], "k-", lw=1.5, label="object boundary")
    for row, color in zip(rows, colors):
        cp = np.array(row.cp_body, dtype=float)
        n_in = np.array(row.n_in_body, dtype=float)
        move_dir = n_in * 0.0
        if abs(row.vr_ff) > 1e-12:
            drive_dir = np.array([np.cos(row.zeta0), np.sin(row.zeta0)], dtype=float)
            move_dir = np.sign(row.vr_ff) * drive_dir
        ax_geo.scatter(cp[0], cp[1], color=color, s=60)
        ax_geo.text(cp[0], cp[1], f" {row.robot}", fontsize=9, color=color)
        ax_geo.arrow(cp[0], cp[1], 0.12 * n_in[0], 0.12 * n_in[1],
                     head_width=0.015, head_length=0.02, color=color, alpha=0.7)
        ax_geo.arrow(cp[0], cp[1], 0.12 * move_dir[0], 0.12 * move_dir[1],
                     head_width=0.015, head_length=0.02, color=color, alpha=0.35, linestyle="--")
    ax_geo.set_title(f"{title_prefix}\ncontact geometry: inward normal and movement direction")
    ax_geo.set_xlabel("object body x (m)")
    ax_geo.set_ylabel("object body y (m)")
    ax_geo.axis("equal")
    ax_geo.grid(True, alpha=0.3)

    ax_auth = axes[0][1]
    ax_auth.axhspan(-speed_floor, speed_floor, color="tab:gray", alpha=0.12, label="abs authority speed floor")
    ax_auth.axhline(0.0, color="k", lw=0.8)
    ax_auth.bar(x, normal_authority, color=colors, alpha=0.85, label="normal authority")
    ax_auth.plot(x, tangential_authority, "o--", color="tab:blue", label="tangential authority")
    ax_auth.set_xticks(x, names)
    ax_auth.set_ylabel("m/s")
    ax_auth.set_title(
        f"normal authority = |vr_ff| * cos(true alpha)\n"
        f"v=({v_ref_body[0]:+.3f},{v_ref_body[1]:+.3f}) omega={omega_ref:+.3f}"
    )
    ax_auth.legend(fontsize=8)
    ax_auth.grid(True, axis="y", alpha=0.3)

    ax_ratio = axes[1][0]
    ax_ratio.axhspan(-passive_ratio, passive_ratio, color="tab:gray", alpha=0.12, label="passive band")
    ax_ratio.axhline(0.0, color="k", lw=0.8)
    ax_ratio.bar(x - 0.18, normal_ratio, width=0.36, color=colors, alpha=0.85, label="normal ratio")
    ax_alpha = ax_ratio.twinx()
    ax_alpha.plot(x + 0.18, true_alpha, "o-", color="tab:purple", label="true alpha")
    ax_ratio.set_xticks(x, names)
    ax_ratio.set_ylabel("normal ratio = cos(true alpha)")
    ax_alpha.set_ylabel("abs true alpha (deg)")
    ax_ratio.set_ylim(-1.1, 1.1)
    ax_alpha.set_ylim(0.0, 190.0)
    ax_ratio.set_title("role ratio and true alpha")
    lines1, labels1 = ax_ratio.get_legend_handles_labels()
    lines2, labels2 = ax_alpha.get_legend_handles_labels()
    ax_ratio.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax_ratio.grid(True, axis="y", alpha=0.3)

    ax_table = axes[1][1]
    ax_table.axis("off")
    ax_table.set_title("fixed-ref solve summary")
    ax_table.bar(x - 0.18, np.abs(vr_ff), width=0.36, alpha=0.25, color="tab:blue", label="|vr_ff|")
    ax_table.bar(x + 0.18, cp_speed, width=0.36, alpha=0.25, color="tab:orange", label="|v_cp_body|")
    ax_table.legend(fontsize=8, loc="upper left")
    table_rows = [
        [
            r.robot,
            f"{r.t_param:.3f}",
            f"{r.vr_ff:+.4f}",
            f"{r.normal_authority:+.4f}",
            f"{r.normal_ratio:+.2f}",
            f"{r.true_alpha_deg:.1f}",
            r.role.replace("_", " "),
        ]
        for r in rows
    ]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["robot", "t", "vr_ff", "normal", "ratio", "alpha", "role"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)


def plot_authority(
    rows: List[AuthorityRow],
    parameterization: ContactPointParameterization,
    v_ref_body: np.ndarray,
    omega_ref: float,
    passive_ratio: float,
    speed_floor: float,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Pre-push diff-drive authority diagnostic\n"
        f"object twist body: v=({v_ref_body[0]:+.3f}, {v_ref_body[1]:+.3f}) m/s, "
        f"omega={omega_ref:+.3f} rad/s | passive_ratio={passive_ratio:.2f}, "
        f"speed_floor={speed_floor:.4f} m/s",
        fontsize=12,
    )
    _plot_authority_panel(
        axes=axes,
        rows=rows,
        parameterization=parameterization,
        v_ref_body=v_ref_body,
        omega_ref=omega_ref,
        passive_ratio=passive_ratio,
        speed_floor=speed_floor,
        title_prefix="single twist",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_authority_segments(
    all_rows: List[List[AuthorityRow]],
    segments: List[SegmentTwist],
    parameterization: ContactPointParameterization,
    passive_ratio: float,
    speed_floor: float,
    save_path: Path,
) -> None:
    n = len(all_rows)
    fig, axes = plt.subplots(2 * n, 2, figsize=(16, 10 * n))
    if n == 1:
        axes = np.asarray(axes).reshape(2, 2)
    fig.suptitle(
        "Pre-push diff-drive authority diagnostic (CSV segments)\n"
        f"{n} consecutive object-state segments",
        fontsize=12,
    )
    for i in range(n):
        sub_axes = axes[(2 * i):(2 * i + 2), :]
        seg = segments[i]
        v_ref_body = np.asarray(seg.v_ref_body, dtype=float)
        _plot_authority_panel(
            axes=sub_axes,
            rows=all_rows[i],
            parameterization=parameterization,
            v_ref_body=v_ref_body,
            omega_ref=float(seg.omega_ref),
            passive_ratio=passive_ratio,
            speed_floor=speed_floor,
            title_prefix=(
                f"segment {seg.segment_idx}: t[{seg.t_start:.2f},{seg.t_end:.2f}] "
                f"dt={seg.duration:.3f}s"
            ),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_outputs(rows: List[AuthorityRow], save_dir: Path, prefix: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / f"{prefix}.json"
    csv_path = save_dir / f"{prefix}.csv"
    data = [row.__dict__ for row in rows]
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute multi-pusher diff-drive normal authority roles.")
    parser.add_argument("--object", type=str, default=DEFAULT_OBJECT_SHAPE, choices=sorted(OBJECT_FILE_MAP))
    parser.add_argument("--v-ref-x", type=float, default=0.02)
    parser.add_argument("--v-ref-y", type=float, default=0.0)
    parser.add_argument("--omega-ref", type=float, default=0.02)
    parser.add_argument(
        "--t-params",
        type=str,
        default=None,
        help="Comma-separated contact parameters. If omitted, use cached Magnum Four contacts.",
    )
    parser.add_argument(
        "--compute-magnum-if-missing",
        action="store_true",
        help="Run Magnum Four if no cache entry exists for the object.",
    )
    parser.add_argument("--passive-ratio", type=float, default=0.25)
    parser.add_argument("--speed-floor", type=float, default=0.003)
    parser.add_argument("--save-dir", type=str, default="/tmp/multi_pusher_authority")
    parser.add_argument("--prefix", type=str, default="authority_precheck")
    parser.add_argument(
        "--object-states-csv",
        type=str,
        default=None,
        help="Optional CSV with columns x,y,theta[,t]. When set, evaluate one authority page per consecutive state segment (t is only used for labeling).",
    )
    parser.add_argument(
        "--primitive-speed",
        type=float,
        default=0.1,
        help="Unit-speed used by primitive inversion for CSV segment extraction.",
    )
    args = parser.parse_args()

    t_params = _parse_t_params(args.t_params)
    v_ref_body = np.array([args.v_ref_x, args.v_ref_y], dtype=float)
    save_dir = Path(args.save_dir)

    _setup_pybullet()
    try:
        generic_object, _object_uid = obj_to_generic(
            obj_path=OBJECT_FILE_MAP[args.object],
            shape_name=args.object,
            position=(0.0, 0.0, DEFAULT_OBJECT_HEIGHT),
            orientation=0.0,
            mass=1.0,
            lateral_friction=DEFAULT_OBJECT_FRICTION,
            blind_test=True,
        )
        if t_params is None:
            t_params = _load_cached_t_params(args.object)
            if t_params is not None:
                print(f"[magnum] loaded cached t_params for {args.object}: {[round(t, 4) for t in t_params]}")
        if t_params is None:
            if not args.compute_magnum_if_missing:
                raise RuntimeError(
                    "No --t-params provided and no cached Magnum Four contacts found. "
                    "Pass --compute-magnum-if-missing to solve them."
                )
            print(f"[magnum] computing Magnum Four contacts for {args.object}...")
            magnum_result = find_the_magnum_four_v3(
                generic_object,
                verbose=False,
                visualize=False,
                weighting_scheme="balanced",
                torque_method=3,
            )
            if not magnum_result or not magnum_result.get("success", False):
                raise RuntimeError("Magnum Four solver failed.")
            contacts = magnum_result["best_solution"]["contacts"]
            t_params = [float(c.parameter) % 1.0 for c in contacts]
            _save_t_params_to_cache(args.object, t_params)
            print(f"[magnum] solved t_params: {[round(t, 4) for t in t_params]}")

        parameterization = ContactPointParameterization(generic_object)
        if args.object_states_csv:
            states = _load_object_states_csv(Path(args.object_states_csv).expanduser().resolve())
            segments = _build_segment_twists_from_states(states, v_speed=float(args.primitive_speed))
            all_rows: List[List[AuthorityRow]] = []
            rows_for_export: List[Dict[str, object]] = []
            for seg in segments:
                seg_v = np.asarray(seg.v_ref_body, dtype=float)
                seg_rows = analyze_authority(
                    parameterization=parameterization,
                    t_params=t_params,
                    v_ref_body=seg_v,
                    omega_ref=float(seg.omega_ref),
                    passive_ratio=float(args.passive_ratio),
                    speed_floor=float(args.speed_floor),
                )
                all_rows.append(seg_rows)
                print(
                    f"[segment {seg.segment_idx}] t=[{seg.t_start:.3f},{seg.t_end:.3f}] "
                    f"v_body=({seg_v[0]:+.4f},{seg_v[1]:+.4f}) omega={seg.omega_ref:+.4f}"
                )
                for row in seg_rows:
                    print(
                        f"  {row.robot}: vr_ff={row.vr_ff:+.4f} normal={row.normal_authority:+.4f} "
                        f"ratio={row.normal_ratio:+.3f} role={row.role}"
                    )
                    payload = dict(row.__dict__)
                    payload.update(
                        {
                            "segment_idx": seg.segment_idx,
                            "t_start": seg.t_start,
                            "t_end": seg.t_end,
                            "segment_duration": seg.duration,
                            "segment_v_ref_x": float(seg_v[0]),
                            "segment_v_ref_y": float(seg_v[1]),
                            "segment_omega_ref": float(seg.omega_ref),
                        }
                    )
                    rows_for_export.append(payload)

            plot_path = save_dir / f"{args.prefix}.png"
            plot_authority_segments(
                all_rows=all_rows,
                segments=segments,
                parameterization=parameterization,
                passive_ratio=float(args.passive_ratio),
                speed_floor=float(args.speed_floor),
                save_path=plot_path,
            )
            print(f"Saved {plot_path}")

            save_dir.mkdir(parents=True, exist_ok=True)
            json_path = save_dir / f"{args.prefix}.json"
            csv_path = save_dir / f"{args.prefix}.csv"
            with open(json_path, "w") as f:
                json.dump(rows_for_export, f, indent=2)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows_for_export[0].keys()))
                writer.writeheader()
                writer.writerows(rows_for_export)
            print(f"Saved {json_path}")
            print(f"Saved {csv_path}")
        else:
            rows = analyze_authority(
                parameterization=parameterization,
                t_params=t_params,
                v_ref_body=v_ref_body,
                omega_ref=float(args.omega_ref),
                passive_ratio=float(args.passive_ratio),
                speed_floor=float(args.speed_floor),
            )
            for row in rows:
                print(
                    f"{row.robot}: t={row.t_param:.4f} vr_ff={row.vr_ff:+.4f} "
                    f"normal={row.normal_authority:+.4f} ratio={row.normal_ratio:+.3f} "
                    f"true_alpha={row.true_alpha_deg:.1f}deg role={row.role}"
                )

            plot_path = save_dir / f"{args.prefix}.png"
            plot_authority(
                rows=rows,
                parameterization=parameterization,
                v_ref_body=v_ref_body,
                omega_ref=float(args.omega_ref),
                passive_ratio=float(args.passive_ratio),
                speed_floor=float(args.speed_floor),
                save_path=plot_path,
            )
            print(f"Saved {plot_path}")
            write_outputs(rows, save_dir, args.prefix)
    finally:
        # In this geometry-only diagnostic, explicit disconnect can trigger a
        # PyBullet cleanup abort in this environment after the files are saved.
        # Let process teardown reclaim the DIRECT connection instead.
        pass


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
