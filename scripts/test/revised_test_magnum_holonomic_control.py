#!/usr/bin/env python3
"""
Revised holonomic Magnum test: stochastic AFC + scenario obstacles + HA_draw path planning.

Pipeline:
  1) Object material friction + D/σ₃ screening
  2) Bumper µ from (n_contacts, degeneracy) → µ_contact = material × bumper
  3) Soft hardware sanity check (product µ) + find_the_magnum_stochastic(n_contacts=...)
  4) Load obstacles from HA_draw scenario JSON or --planned-path bundle
  5) mod_grid / mod_grid_SE path → HybridPath or pure pursuit
  6) Phase7 holonomic push with optional stray replan monitor

Friction model (PyBullet product):
  object material µ × robot bumper µ = effective contact µ (search + sim cone).

Usage:
  python3 revised_test_magnum_holonomic_control.py \
    --n-contacts 4 --planned-path HA_draw/rectObs_scenario_root_SE_minprime.planned.json \
    --planner hybrid --theta-mode waypoint --duration 60 --no-boundary-walls \
    --headless --save-dir /tmp/revised_holo/

  # Debug: reuse / write n-contact t_params in urdf/magnum_afc_cache.json
  python3 revised_test_magnum_holonomic_control.py \
    --debug --n-contacts 3 --planned-path ... --headless --save-dir /tmp/revised_holo/

  # After Ctrl-C (or full run), replot from live checkpoint:
  python3 plot_revised_holonomic_histories.py \
    --histories /tmp/revised_holo/histories_live_<tag>_w_<obj>.json \
    --save-dir /tmp/revised_holo/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as pyb

import rospkg

rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "src"))
sys.path.insert(0, str(pkg_path / "src" / "legacy"))
sys.path.insert(0, str(pkg_path / "scripts" / "test"))
sys.path.insert(0, str(pkg_path / "scripts" / "test" / "basic_test"))
sys.path.insert(0, str(pkg_path / "scripts" / "PathPlanning" / "Search_based_Planning" / "HA_draw"))

from object_utils import ContactPoint, ContactPointParameterization
from contact_maintain.robot_factory import create_robot
from contact_maintain.object_bridge import obj_to_generic
from contact_maintain.robot_agent import RobotAgent
from contact_maintain.swarm import SwarmHost, RobotState
from contact_maintain.motion_planner import PathFollowingController
from contact_maintain.holonomic_path_control import (
    ThetaMode,
    HolonomicPurePursuitPolyline,
    build_hybrid_path_from_planned,
    cumulative_vertex_s,
    lateral_error_to_polyline,
    nearest_s_on_hybrid_path,
    orientation_pid_omega,
    theta_goal_at_segment_endpoint,
    final_theta_goal_for_mode,
    holonomic_path_xy_completed,
    apply_path_completion_to_desired_motion,
    apply_orientation_hold,
    HolonomicSegmentOrientGate,
    resolve_segment_theta_specs,
    completed_segment_at_s_crossing,
    mandated_theta_at_segment_end,
)
from stochastic_magnum_finder import find_the_magnum_stochastic
from friction_model import recommend_bumper_friction
from grasp_covariance import (
    DEFAULT_SOFT_DEGENERACY_THRESHOLD,
    calculate_grasp_covariance,
    format_grasp_covariance_report,
    recommend_tangent_fallback,
)
from afc_hardware import check_robot_afc_hardware_feasible, estimate_robot_max_push_force
from holonomic_run_logger import HolonomicRunLogger
from magnum_contact_cache import (
    contacts_from_t_params,
    default_cache_path,
    load_cached_contacts,
    save_cached_contacts,
)
from revised_holonomic_core import (
    APPROACH_DISTANCE,
    CTRL_FREQ,
    CTRL_STEP,
    DEFAULT_OBJECT_FRICTION,
    PID_DECIMATION,
    ROBOT_RADIUS,
    TIMESTEP,
    Phase7BetaVerDecouple,
    get_object_as_obstacle,
    get_object_state,
    robot_spawn_pose_world,
    setup_pybullet,
    setup_video_recording,
    stop_video_recording,
)
from revised_holonomic_plots import (
    export_histories,
    plot_phase7_velocities,
    plot_phase7_wheel_plot,
    plot_phase_1_results,
    plot_phase_7beta,
)

from load_json_to_obstacles import spawn_scenario_obstacles, OBJ_SHAPE_FILES
from scenario_planner_bridge import (
    PlannedPath,
    load_scenario_file,
    plan_scenario_path,
    resolve_planned_bundle_paths,
    save_planned_path,
    theta_milestones_from_df_primitives,
)


def _resolve_scenario_path(path: Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    cand = pkg_path / "scripts" / "PathPlanning" / "Search_based_Planning" / "HA_draw" / p.name
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"Scenario not found: {path}")


def _resolve_planned_path_file(path: Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    ha_draw = pkg_path / "scripts" / "PathPlanning" / "Search_based_Planning" / "HA_draw"
    for folder in (Path.cwd(), ha_draw):
        cand = folder / p.name
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"Planned path not found: {path}")


def _load_planned_path_context(planned_arg: str) -> Dict[str, Any]:
    ha_draw = pkg_path / "scripts" / "PathPlanning" / "Search_based_Planning" / "HA_draw"
    planned_file = _resolve_planned_path_file(Path(planned_arg))
    scenario_path, _, scenario, planned, bundle = resolve_planned_bundle_paths(
        planned_file, search_dirs=[ha_draw, planned_file.parent]
    )
    shape_name = str((scenario.get("robot", {}) or {}).get("shape_name", "right_triangle"))
    if shape_name not in OBJ_SHAPE_FILES:
        raise ValueError(
            f"Scenario shape '{shape_name}' is not supported by revised_test_magnum_holonomic_control."
        )
    return {
        "planned_file": planned_file,
        "scenario_path": scenario_path,
        "scenario": scenario,
        "planned": planned,
        "bundle": bundle,
        "shape_name": shape_name,
    }


def _stochastic_contacts(
    generic_object,
    *,
    n_contacts: int,
    force_range_scalar: float,
    timeout: float,
    soft_threshold: float,
    samples_per_edge: int,
    force_tangent: bool,
    ignore_degeneracy_gate: bool,
    retry_tangent_on_failure: bool,
    robot_max_force: Optional[float],
    material_friction: float,
    bumper_friction: Optional[float],
    target_contact_friction: Optional[float],
    strict_hw_gate: bool,
    shape_name: str,
    use_contact_cache: bool = False,
    contact_cache_path: Optional[Path] = None,
) -> Tuple[List[ContactPoint], List[float], dict]:
    """
    Screen degeneracy, choose bumper/contact µ, resolve contacts.

    With ``use_contact_cache`` (``--debug``): load n-contact t_params from JSON if
    present; otherwise search once and write the cache for consistent reruns.
    """
    generic_object.set_material_friction(
        material_friction,
        sync_legacy_lateral=True,
        sync_legacy_static=True,  # revised floor story ≈ material µ
    )

    cov = calculate_grasp_covariance(
        generic_object,
        samples_per_edge=samples_per_edge,
        soft_degeneracy_threshold=soft_threshold,
    )
    tangent_rec = recommend_tangent_fallback(cov, soft_degeneracy_threshold=soft_threshold)
    print(f"   {format_grasp_covariance_report(cov, getattr(generic_object, 'name', ''))}")

    n_contacts = int(n_contacts)
    if n_contacts < 2:
        raise ValueError(f"n_contacts must be >= 2, got {n_contacts}")

    # n=3 always needs friction cone; n=4 follows D-gate unless CLI overrides.
    if n_contacts <= 3:
        tangent_required = True
        use_tangent_fallback = False
        gate_note = "n<=3: tangent required"
    elif force_tangent:
        tangent_required = True
        use_tangent_fallback = True
        gate_note = "CLI --force-tangent"
    elif ignore_degeneracy_gate:
        tangent_required = False
        use_tangent_fallback = retry_tangent_on_failure
        gate_note = "CLI ignore D gate"
    else:
        tangent_required = bool(tangent_rec["recommend_tangent_fallback"])
        use_tangent_fallback = tangent_required or retry_tangent_on_failure
        gate_note = "Section 11 D/σ₃ gate"

    bumper_plan = recommend_bumper_friction(
        n_contacts,
        material_friction=generic_object.material_friction,
        tangent_required=tangent_required,
        bumper_override=bumper_friction,
        target_contact_override=target_contact_friction,
    )
    mu_contact = generic_object.apply_bumper_contact_model(bumper_plan.bumper_friction)
    print(
        f"   Friction plan [{gate_note}]: {bumper_plan.reason}\n"
        f"      µ_material={bumper_plan.material_friction:g}  "
        f"µ_bumper={bumper_plan.bumper_friction:g}  "
        f"µ_contact={mu_contact:g} (= product)"
    )

    hw = check_robot_afc_hardware_feasible(
        generic_object,
        force_range_scalar=force_range_scalar,
        tangent_mode=bool(tangent_required or use_tangent_fallback),
        contact_friction=mu_contact,
        robot_max_force=robot_max_force,
        warn_only=not strict_hw_gate,
    )
    print(f"   Hardware check: {hw.reason}")
    if not hw.feasible and strict_hw_gate:
        raise RuntimeError(hw.reason)

    cache_path = Path(contact_cache_path) if contact_cache_path else default_cache_path()
    cache_hit = False
    result: dict = {}

    if use_contact_cache:
        cached = load_cached_contacts(
            shape_name, n_contacts, cache_path=cache_path, allow_legacy_four=True
        )
        if cached is not None:
            t_params = list(cached["t_params"])
            contacts = contacts_from_t_params(generic_object, t_params)
            cache_hit = True
            print(
                f"   DEBUG cache HIT [{shape_name}/n={n_contacts}] "
                f"source={cached.get('source')} file={cached.get('cache_path')}"
            )
            print(f"   t_params (cached): {[f'{v:.4f}' for v in t_params]}")
            meta = {
                "cov": cov,
                "tangent_rec": tangent_rec,
                "search": {"success": True, "from_cache": True, "source": cached.get("source")},
                "hardware": hw.as_dict(),
                "bumper_plan": bumper_plan.as_dict(),
                "mu_contact": mu_contact,
                "n_contacts": n_contacts,
                "tangent_required": tangent_required,
                "cache_hit": True,
                "cache_path": str(cached.get("cache_path") or cache_path),
            }
            return contacts, t_params, meta
        print(
            f"   DEBUG cache MISS [{shape_name}/n={n_contacts}] — searching, then write {cache_path}"
        )

    result = find_the_magnum_stochastic(
        generic_object,
        verbose=True,
        threshold=1.0,
        timeout=timeout,
        force_range_scalar=force_range_scalar,
        robot_radius=ROBOT_RADIUS,
        n_contacts=n_contacts,
        used_tangent_as_fallback=use_tangent_fallback and not tangent_required,
        tangent_required=tangent_required,
    )
    if not result.get("success"):
        raise RuntimeError(
            f"Stochastic AFC search failed (n_contacts={n_contacts}, "
            f"µ_contact={mu_contact:g}) — no configuration found."
        )

    contacts = result.get("contacts", [])
    if len(contacts) != n_contacts:
        raise RuntimeError(
            f"Searcher returned {len(contacts)} contacts, expected {n_contacts}"
        )
    t_params = [float(c.parameter) % 1.0 for c in contacts]

    if use_contact_cache:
        written = save_cached_contacts(
            shape_name,
            n_contacts,
            t_params,
            cache_path=cache_path,
            mu_contact=mu_contact,
            tangent_required=tangent_required,
            source="stochastic",
        )
        print(f"   DEBUG cache WRITE [{shape_name}/n={n_contacts}] → {written}")

    meta = {
        "cov": cov,
        "tangent_rec": tangent_rec,
        "search": result,
        "hardware": hw.as_dict(),
        "bumper_plan": bumper_plan.as_dict(),
        "mu_contact": mu_contact,
        "n_contacts": n_contacts,
        "tangent_required": tangent_required,
        "cache_hit": cache_hit,
        "cache_path": str(cache_path) if use_contact_cache else None,
    }
    return contacts, t_params, meta


def _build_path_followers(
    planned: PlannedPath,
    start_xy: np.ndarray,
    *,
    planner_mode: str,
    pyaw: Optional[List[float]] = None,
) -> Tuple[Optional[PathFollowingController], Optional[HolonomicPurePursuitPolyline], object, List[float], List[float], List]:
    a_max = 0.15
    a_lat_max = 0.08
    v_user_max = 0.1
    look_ahead_hybrid = 0

    path_following_controller = None
    pursuit_controller = None
    holonomic_hybrid_path = None
    s_milestones: List[float] = []
    theta_milestones: List[float] = []
    segment_theta_specs: List = []

    prims = list(planned.primitives) if planned.primitives else []
    df_prims = list(getattr(planned, "df_primitives", None) or [])

    if planner_mode == "hybrid":
        if not prims:
            raise RuntimeError(
                "Hybrid mode requires df_primitives / primitives in the export "
                "(re-export from HA_draw after phase-3 planning)."
            )
        holonomic_hybrid_path = build_hybrid_path_from_planned(
            planned.px, planned.py, primitives=prims
        )
        path_following_controller = PathFollowingController(
            holonomic_hybrid_path,
            a_max=a_max,
            a_lat_max=a_lat_max,
            v_user_max=v_user_max,
            look_ahead=look_ahead_hybrid,
            use_tracking=False,
        )
        if df_prims:
            s_milestones, theta_milestones = theta_milestones_from_df_primitives(df_prims)
        else:
            sv = cumulative_vertex_s(holonomic_hybrid_path)
            s_milestones = [float(x) for x in sv]
            theta_milestones = list(pyaw) if pyaw else [0.0] * len(s_milestones)
            if theta_milestones and len(theta_milestones) < len(s_milestones):
                theta_milestones = theta_milestones + [theta_milestones[-1]] * (
                    len(s_milestones) - len(theta_milestones)
                )
    else:
        pts = np.column_stack([planned.px, planned.py])
        pursuit_controller = HolonomicPurePursuitPolyline(
            pts, a_max=a_max, v_user_max=v_user_max, Ld=0.25, kf=0.5
        )
        s_milestones = [float(x) for x in pursuit_controller.cum]
        if df_prims:
            s_milestones, theta_milestones = theta_milestones_from_df_primitives(df_prims)
        elif pyaw:
            theta_milestones = list(pyaw)
            if len(theta_milestones) < len(s_milestones):
                theta_milestones = theta_milestones + [theta_milestones[-1]] * (
                    len(s_milestones) - len(theta_milestones)
                )
        else:
            theta_milestones = [0.0] * len(planned.px)

    segment_theta_specs = resolve_segment_theta_specs(
        df_primitives=df_prims or None,
        s_vertices=s_milestones if not df_prims else None,
        theta_vertices=theta_milestones if not df_prims else None,
    )

    return path_following_controller, pursuit_controller, holonomic_hybrid_path, s_milestones, theta_milestones, segment_theta_specs


def _replan_from_current(
    scenario: dict,
    current_pose: Tuple[float, float, float],
    *,
    planner_arg: str,
    stop_phase: int,
) -> PlannedPath:
    goal = scenario.get("pose", {}).get("goal", [0.0, 0.0, 0.0])
    gyaw = float(goal[2]) if len(goal) >= 3 else 0.0
    planner_cfg = scenario.get("planner", {}) or {}
    path_opts = planner_cfg.get("options", {}) or {}
    return plan_scenario_path(
        scenario,
        planner=planner_arg,
        stop_phase=stop_phase,
        start_override=(current_pose[0], current_pose[1], math.degrees(current_pose[2])),
        goal_override=(float(goal[0]), float(goal[1]), gyaw),
        se_p3_primitive=str(path_opts.get("se_p3_primitive", "linear_yaw_dp")),
        se_p3_collision_mode=str(path_opts.get("se_p3_collision", "volume_bin")),
        dp_objective=str(path_opts.get("dp_objective", "length")),
        disk_collision_mode=str(path_opts.get("disk_collision_mode", "offline")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Revised holonomic Magnum + scenario planner test")
    parser.add_argument(
        "--object",
        type=str,
        default="right_triangle",
        choices=list(OBJ_SHAPE_FILES.keys()),
        help="Object shape (ignored when --planned-path is set)",
    )
    parser.add_argument("--scenario", type=str, default=None, help="HA_draw scenario JSON path")
    parser.add_argument(
        "--planned-path",
        type=str,
        default=None,
        help="HA_draw *.planned.json; loads paired scenario via scenario_ref and uses exported path",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--no-gui", action="store_true", help="PyBullet DIRECT (no window)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Alias for --no-gui; enables compact live status + history checkpoints when --save-dir is set",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=2.0,
        help="Seconds between atomic histories_live_*.json checkpoints (default 2)",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between compact status.log lines (default 1)",
    )
    parser.add_argument("--planner", type=str, default="auto", choices=["auto", "disk", "se", "hybrid", "pursuit"],
                        help="Path planner: auto/disk/se for HA_draw; hybrid/pursuit for tracking mode")
    parser.add_argument("--planner-stop-phase", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--theta-mode", type=str, default="waypoint", choices=["waypoint", "fixed", "path"])
    parser.add_argument("--fixed-theta", type=float, default=0.0)
    parser.add_argument("--path-theta-sine-amp", type=float, default=float(np.pi))
    parser.add_argument("--path-theta-sine-k", type=float, default=1.0)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--robot-max-force", type=float, default=None,
                        help="Override per-robot max push force (N); default from Omniwheel model")
    parser.add_argument("--force-range-scalar", type=float, default=2.0)
    parser.add_argument(
        "--n-contacts",
        type=int,
        default=4,
        help="Number of contact points / robots (default 4; use 3 for friction-cone Magnum)",
    )
    parser.add_argument(
        "--material-friction",
        type=float,
        default=None,
        help="Object material µ (PyBullet object+ground). Default: DEFAULT_OBJECT_FRICTION",
    )
    parser.add_argument(
        "--bumper-friction",
        type=float,
        default=None,
        help="Override robot bumper µ. Default: auto from n_contacts + D gate",
    )
    parser.add_argument(
        "--target-contact-friction",
        type=float,
        default=None,
        help="Override desired µ_contact (= material × bumper); bumper = target / material",
    )
    parser.add_argument(
        "--strict-hw-gate",
        action="store_true",
        help="Abort if actuator sanity check fails (default: warn only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Use contact-cache mode: load n-contact t_params from "
            "urdf/magnum_afc_cache.json (or --contact-cache); on miss, search once and write"
        ),
    )
    parser.add_argument(
        "--contact-cache",
        type=str,
        default=None,
        help="Override path for n-contact AFC cache JSON (default: urdf/magnum_afc_cache.json)",
    )
    parser.add_argument("--stochastic-timeout", type=float, default=15.0)
    parser.add_argument("--soft-threshold", type=float, default=DEFAULT_SOFT_DEGENERACY_THRESHOLD)
    parser.add_argument("--samples-per-edge", type=int, default=4)
    parser.add_argument("--force-tangent", action="store_true")
    parser.add_argument("--ignore-degeneracy-gate", action="store_true")
    parser.add_argument("--retry-tangent-on-failure", action="store_true")
    parser.add_argument("--stray-threshold", type=float, default=0.4, help="Cross-track error (m) before replan")
    parser.add_argument("--stray-hold-time", type=float, default=1.0, help="Seconds above threshold before replan")
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument("--hybrid-retouch-duration", type=float, default=0.55)
    parser.add_argument("--hybrid-retouch-timeout", type=float, default=2.0)
    parser.add_argument(
        "--orientation-complete-tol",
        type=float,
        default=0.1,
        help="After XY path completion, keep rotating until |theta_err| < this (rad).",
    )
    parser.add_argument("--write-planned-path", action="store_true", help="Save scenario + path JSON to save-dir")
    parser.add_argument(
        "--no-boundary-walls",
        action="store_true",
        help="Skip spawning map boundary walls (rect obstacles still spawn)",
    )
    args = parser.parse_args()

    if args.headless:
        args.no_gui = True
    if (args.headless or args.no_gui) and not args.save_dir:
        print(
            "Warning: headless/--no-gui without --save-dir — no live status.log "
            "or histories_live checkpoint will be written.",
            flush=True,
        )

    if not args.planned_path and not args.scenario:
        parser.error("one of --scenario or --planned-path is required")

    planned_ctx = None
    if args.planned_path:
        planned_ctx = _load_planned_path_context(args.planned_path)
        scenario = planned_ctx["scenario"]
        scenario_path = planned_ctx["scenario_path"]
        selected_name = planned_ctx["shape_name"]
    else:
        scenario_path = _resolve_scenario_path(Path(args.scenario))
        scenario = load_scenario_file(scenario_path)
        scenario["robot"]["shape_name"] = args.object
        selected_name = args.object

    f_robot = args.robot_max_force or estimate_robot_max_push_force()
    print(f"\nRevised holonomic test: object={selected_name} scenario={scenario_path.name}")
    if planned_ctx:
        print(f"  Planned-path mode: {planned_ctx['planned_file'].name}")
    print(f"  F_robot_max (derived/override) = {f_robot:.2f} N")
    print(f"  n_contacts = {args.n_contacts}")

    ground_uid = setup_pybullet(gui=not args.no_gui)

    material_mu = (
        float(args.material_friction)
        if args.material_friction is not None
        else float(DEFAULT_OBJECT_FRICTION)
    )
    pyb.changeDynamics(ground_uid, -1, lateralFriction=material_mu)

    spawn_scenario_obstacles(
        scenario, wall_thickness=0.2, wall_height=0.5, mu=1.0, spawn_walls=not args.no_boundary_walls
    )
    if args.no_boundary_walls:
        print("  Boundary walls: disabled (--no-boundary-walls)")

    pose = scenario.get("pose", {}) or {}
    start = pose.get("start", [0.0, 0.0, 0.0])
    sx, sy = float(start[0]), float(start[1])
    syaw = math.radians(float(start[2]) if len(start) >= 3 else 0.0)

    obj_file = OBJ_SHAPE_FILES[selected_name]
    generic_object, object_uid = obj_to_generic(
        obj_path=obj_file,
        shape_name=selected_name,
        position=(sx, sy, 0.2),
        orientation=syaw,
        mass=1.0,
        lateral_friction=material_mu,
        blind_test=True,
    )
    contact_param = ContactPointParameterization(generic_object)

    print(f"\n--- Stochastic AFC (n_contacts={args.n_contacts}) ---")
    if args.debug:
        cache_show = args.contact_cache or str(default_cache_path())
        print(f"   DEBUG contact-cache ON → {cache_show}")
    contacts, t_params, afc_meta = _stochastic_contacts(
        generic_object,
        n_contacts=args.n_contacts,
        force_range_scalar=args.force_range_scalar,
        timeout=args.stochastic_timeout,
        soft_threshold=args.soft_threshold,
        samples_per_edge=args.samples_per_edge,
        force_tangent=args.force_tangent,
        ignore_degeneracy_gate=args.ignore_degeneracy_gate,
        retry_tangent_on_failure=args.retry_tangent_on_failure,
        robot_max_force=args.robot_max_force,
        material_friction=material_mu,
        bumper_friction=args.bumper_friction,
        target_contact_friction=args.target_contact_friction,
        strict_hw_gate=args.strict_hw_gate,
        shape_name=selected_name,
        use_contact_cache=bool(args.debug),
        contact_cache_path=Path(args.contact_cache) if args.contact_cache else None,
    )
    print(f"   t_params: {[f'{v:.4f}' for v in t_params]}")
    bumper_mu = float(afc_meta["bumper_plan"]["bumper_friction"])

    tracking_mode = "pursuit" if args.planner == "pursuit" else "hybrid"

    if planned_ctx:
        planned = planned_ctx["planned"]
        if not planned.ok:
            raise RuntimeError(f"Exported planned path is invalid: {planned_ctx['planned_file']}")
        print(f"\n--- Exported planned path ---")
        print(
            f"   file={planned_ctx['planned_file'].name} "
            f"planner={planned.planner} points={len(planned.px)} "
            f"primitives={len(planned.primitives)} "
            f"df_primitives={len(getattr(planned, 'df_primitives', None) or [])}"
        )
        planner_backend = planned.planner or ("se" if planned.primitives else "disk")
    else:
        planner_backend = "auto" if args.planner in ("hybrid", "pursuit", "auto") else args.planner
        print(f"\n--- Path planning ({planner_backend}) ---")
        planner_cfg = scenario.get("planner", {}) or {}
        path_opts = planner_cfg.get("options", {}) or {}
        planned = plan_scenario_path(
            scenario,
            planner=planner_backend,
            stop_phase=args.planner_stop_phase,
            se_p3_primitive=str(path_opts.get("se_p3_primitive", "linear_yaw_dp")),
            se_p3_collision_mode=str(path_opts.get("se_p3_collision", "volume_bin")),
            dp_objective=str(path_opts.get("dp_objective", "length")),
            disk_collision_mode=str(path_opts.get("disk_collision_mode", "offline")),
        )
        if not planned.ok:
            raise RuntimeError("Path planner returned no path for scenario.")
        print(f"   planner={planned.planner} points={len(planned.px)} primitives={len(planned.primitives)}")

    if args.write_planned_path and args.save_dir:
        out = Path(args.save_dir) / f"planned_{scenario_path.stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_planned_path(scenario_path, planned, out)
        print(f"   wrote {out}")

    obj_state_init = get_object_state(object_uid)
    start_xy = np.asarray(obj_state_init["position"], dtype=float)

    path_following_controller, pursuit_controller, holonomic_hybrid_path, s_milestones, theta_milestones, segment_theta_specs = (
        _build_path_followers(planned, start_xy, planner_mode=tracking_mode, pyaw=planned.pyaw)
    )

    theta_mode_enum = {
        "waypoint": ThetaMode.WAYPOINT,
        "fixed": ThetaMode.FIXED,
        "path": ThetaMode.PATH,
    }[args.theta_mode]
    run_tag = f"scenario_{tracking_mode}_{args.theta_mode}"

    robots: Dict[str, object] = {}
    robot_agents: Dict[str, RobotAgent] = {}
    n_robots = len(t_params)
    for i in range(n_robots):
        name = f"R_{i+1:02d}"
        target_t_param = t_params[i]
        contact_info = contact_param.get_contact_info(target_t_param)
        contact_point_body = np.array(contact_info["point"], dtype=float)
        normal_outward = np.array(contact_info["normal_outward"], dtype=float)
        normal_inward = np.array(contact_info["normal_inward"], dtype=float)
        obj_xy = (float(sx), float(sy))
        robot_x, robot_y, robot_heading = robot_spawn_pose_world(
            contact_point_body,
            normal_outward,
            normal_inward,
            object_xy=obj_xy,
            object_yaw_rad=float(syaw),
        )
        robot = create_robot(
            kinematics="holonomic",
            model="wheel",
            position=(robot_x, robot_y),
            orientation=robot_heading,
            name=name,
            contact_mu=bumper_mu,
        )
        robots[name] = robot
        robot_agents[name] = RobotAgent(
            robot=robot,
            name=name,
            object_uid=object_uid,
            generic_object=generic_object,
            navigation_type="apf",
            pushing_type="velocity",
            force_distributor=None,
        )

    phase7_controllers = {}
    for name in robots:
        idx = list(robots.keys()).index(name)
        phase7_controllers[name] = Phase7BetaVerDecouple(
            robot_uid=robots[name].uid,
            object_uid=object_uid,
            generic_object=generic_object,
            t_param=t_params[idx],
            desired_object_velocity=np.array([0.0, 0.0]),
            desired_object_angular_velocity=0.0,
        )

    host = SwarmHost(
        robot_agents=robot_agents,
        object_uid=object_uid,
        generic_object=generic_object,
        startup_mode="quick",
    )
    target_map = {name: t_params[i] for i, name in enumerate(robots.keys())}
    host.assign_targets(target_map)

    run_logger = None
    if args.save_dir:
        def _snapshot():
            hist = {name: c.history for name, c in phase7_controllers.items()}
            tp = {name: float(c.t_param) for name, c in phase7_controllers.items()}
            return hist, tp

        run_logger = HolonomicRunLogger(
            Path(args.save_dir),
            run_tag=run_tag,
            object_name=selected_name,
            meta={
                "n_contacts": args.n_contacts,
                "duration_s": args.duration,
                "headless": bool(args.headless or args.no_gui),
                "planner": args.planner,
                "theta_mode": args.theta_mode,
                "force_range_scalar": args.force_range_scalar,
                "afc": {
                    "mu_contact": afc_meta.get("mu_contact"),
                    "bumper_plan": afc_meta.get("bumper_plan"),
                    "tangent_required": afc_meta.get("tangent_required"),
                    "t_params": t_params,
                    "debug_cache": bool(args.debug),
                    "cache_hit": afc_meta.get("cache_hit"),
                    "cache_path": afc_meta.get("cache_path"),
                },
            },
            checkpoint_interval_s=args.checkpoint_interval,
            status_interval_s=args.status_interval,
            get_snapshot=_snapshot,
        )

    video_log_id = None
    video_path = None
    if args.record_video and not args.no_gui and args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_path = save_dir / f"phase7_topview_{run_tag}_w_{selected_name}.mp4"
        video_log_id = setup_video_recording(video_path, object_uid)

    n_steps = int(args.duration / TIMESTEP)
    step_count = 0
    t = 0.0
    pid_cycle_count = 0
    holonomic_motion_started = False
    hybrid_retouch_active = False
    hybrid_retouch_t0 = 0.0
    hybrid_retouch_consumed_boundaries = set()
    orient_gate_consumed_boundaries = set()
    hybrid_retouch_resume_guard = False
    segment_orient_gate = HolonomicSegmentOrientGate(
        segment_specs=segment_theta_specs,
        orientation_tol=float(args.orientation_complete_tol),
    )
    tracking_prev_s = 0.0

    stray_timer = 0.0
    replan_count = 0
    ref_px, ref_py = list(planned.px), list(planned.py)

    for _ in range(n_steps):
        obj_state = get_object_state(object_uid)

        if step_count % CTRL_STEP == 0:
            host.update(1.0 / CTRL_FREQ, obj_state)
            pid_cycle_count += 1

            if pid_cycle_count % PID_DECIMATION == 0:
                has_ref = path_following_controller is not None or pursuit_controller is not None
                if has_ref:
                    all_pushing = all(agent.goal_type == "push" for agent in robot_agents.values())

                    if not holonomic_motion_started and all_pushing:
                        holonomic_motion_started = True
                        if path_following_controller is not None:
                            path_following_controller.reset()
                        if pursuit_controller is not None:
                            pursuit_controller.reset()
                        print(f"\nALL ROBOTS PUSHING — START SCENARIO PATH (t={t:.2f}s)\n")

                    if holonomic_motion_started and all_pushing and args.stray_threshold > 0:
                        e_lat = lateral_error_to_polyline(
                            np.asarray(obj_state["position"], dtype=float), ref_px, ref_py
                        )
                        if e_lat > args.stray_threshold:
                            stray_timer += (1.0 / CTRL_FREQ) * PID_DECIMATION
                        else:
                            stray_timer = 0.0
                        if stray_timer >= args.stray_hold_time and replan_count < args.max_replans:
                            print(
                                f"[stray] e_lat={e_lat:.3f}m for {stray_timer:.2f}s — replan "
                                f"({replan_count + 1}/{args.max_replans})"
                            )
                            cur_yaw = float(obj_state["orientation"])
                            planned = _replan_from_current(
                                scenario,
                                (float(obj_state["position"][0]), float(obj_state["position"][1]), cur_yaw),
                                planner_arg=planner_backend,
                                stop_phase=args.planner_stop_phase,
                            )
                            if planned.ok:
                                ref_px, ref_py = list(planned.px), list(planned.py)
                                (
                                    path_following_controller,
                                    pursuit_controller,
                                    holonomic_hybrid_path,
                                    s_milestones,
                                    theta_milestones,
                                    segment_theta_specs,
                                ) = _build_path_followers(
                                    planned,
                                    np.asarray(obj_state["position"]),
                                    planner_mode=tracking_mode,
                                    pyaw=planned.pyaw,
                                )
                                if path_following_controller is not None:
                                    path_following_controller.reset()
                                if pursuit_controller is not None:
                                    pursuit_controller.reset()
                                hybrid_retouch_consumed_boundaries.clear()
                                orient_gate_consumed_boundaries.clear()
                                segment_orient_gate = HolonomicSegmentOrientGate(
                                    segment_specs=segment_theta_specs,
                                    orientation_tol=float(args.orientation_complete_tol),
                                )
                                tracking_prev_s = 0.0
                            replan_count += 1
                            stray_timer = 0.0

                    if holonomic_motion_started:
                        dt_pid = (1.0 / CTRL_FREQ) * PID_DECIMATION
                        vx = vy = w_path = 0.0
                        current_s = 0.0

                        do_hybrid_retouch = tracking_mode == "hybrid" and path_following_controller is not None
                        if do_hybrid_retouch and hybrid_retouch_active:
                            all_in_c = all(a.in_contact for a in robot_agents.values())
                            theta_ok = segment_orient_gate.retouch_may_resume(
                                obj_state["orientation"]
                            )
                            soft_end = hybrid_retouch_t0 + args.hybrid_retouch_duration
                            hard_end = soft_end + args.hybrid_retouch_timeout
                            if t >= soft_end and (all_in_c or t >= hard_end):
                                if not theta_ok and t < hard_end:
                                    pass
                                else:
                                    hybrid_retouch_active = False
                                    hybrid_retouch_resume_guard = True
                                    for rname in robot_agents:
                                        host.robot_states[rname] = RobotState.PUSHING
                                        robot_agents[rname].set_goal("push", target_map[rname])

                        if path_following_controller is not None:
                            if segment_orient_gate.gate_active or (
                                do_hybrid_retouch and hybrid_retouch_active
                            ):
                                path_following_controller.compute_velocity(dt=None)
                                vx = vy = w_path = 0.0
                                current_s = path_following_controller.get_current_s()
                            elif do_hybrid_retouch:
                                if hybrid_retouch_resume_guard:
                                    hybrid_retouch_resume_guard = False
                                    path_following_controller.compute_velocity(dt=None)
                                    vx, vy, w_path = 0.0, 0.0, 0.0
                                    current_s = path_following_controller.get_current_s()
                                else:
                                    seg_before = path_following_controller.current_segment_idx
                                    velocity_cmd = path_following_controller.compute_velocity(dt=dt_pid)
                                    seg_after = path_following_controller.current_segment_idx
                                    if seg_after > seg_before:
                                        boundary_key = (int(seg_before), int(seg_after))
                                        orient_done = boundary_key in orient_gate_consumed_boundaries
                                        retouch_done = boundary_key in hybrid_retouch_consumed_boundaries

                                        if orient_done and retouch_done:
                                            vx, vy, w_path = map(float, velocity_cmd)
                                        elif orient_done and not retouch_done and do_hybrid_retouch:
                                            hybrid_retouch_active = True
                                            hybrid_retouch_t0 = t
                                            hybrid_retouch_consumed_boundaries.add(boundary_key)
                                            segment_orient_gate.retouch_resume_theta = (
                                                mandated_theta_at_segment_end(
                                                    segment_theta_specs, int(seg_before)
                                                )
                                            )
                                            for rname in robot_agents:
                                                host.robot_states[rname] = RobotState.APPROACHING
                                                robot_agents[rname].set_goal(
                                                    "approach", target_map[rname]
                                                )
                                            print(
                                                f"[hybrid retouch] boundary {seg_before}->{seg_after} "
                                                f"at t={t:.2f}s — path refrozen, approach"
                                            )
                                            vx, vy, w_path = 0.0, 0.0, 0.0
                                        else:
                                            path_following_controller.elapsed_time -= dt_pid
                                            path_following_controller._update_state_from_time()
                                            needs_hold = segment_orient_gate.begin_segment_end_hold(
                                                int(seg_before),
                                                boundary_key=boundary_key,
                                                do_retouch=do_hybrid_retouch,
                                                retouch_already_consumed=retouch_done,
                                                current_orientation=obj_state["orientation"],
                                                current_s=path_following_controller.get_current_s(),
                                            )
                                            if needs_hold:
                                                print(
                                                    f"[orient gate] segment {seg_before} end "
                                                    f"θ→{segment_orient_gate.gate_theta:.3f} rad "
                                                    f"at t={t:.2f}s"
                                                )
                                            else:
                                                orient_gate_consumed_boundaries.add(boundary_key)
                                                if (
                                                    segment_orient_gate.pending_retouch
                                                    and not retouch_done
                                                ):
                                                    hybrid_retouch_active = True
                                                    hybrid_retouch_t0 = t
                                                    hybrid_retouch_consumed_boundaries.add(
                                                        boundary_key
                                                    )
                                                    for rname in robot_agents:
                                                        host.robot_states[rname] = (
                                                            RobotState.APPROACHING
                                                        )
                                                        robot_agents[rname].set_goal(
                                                            "approach", target_map[rname]
                                                        )
                                                    print(
                                                        f"[hybrid retouch] boundary "
                                                        f"{seg_before}->{seg_after} at t={t:.2f}s "
                                                        f"— path refrozen, approach"
                                                    )
                                                segment_orient_gate.clear_gate()
                                            vx, vy, w_path = 0.0, 0.0, 0.0
                                    else:
                                        vx, vy, w_path = map(float, velocity_cmd)
                                    current_s = path_following_controller.get_current_s()
                            else:
                                velocity_cmd = path_following_controller.compute_velocity(dt=dt_pid)
                                vx, vy, w_path = map(float, velocity_cmd)
                                current_s = path_following_controller.get_current_s()
                                if segment_theta_specs:
                                    crossed = completed_segment_at_s_crossing(
                                        tracking_prev_s, current_s, segment_theta_specs
                                    )
                                    if crossed is not None:
                                        boundary_key = (int(crossed), int(crossed) + 1)
                                        if boundary_key not in orient_gate_consumed_boundaries:
                                            path_following_controller.elapsed_time -= dt_pid
                                            path_following_controller._update_state_from_time()
                                            needs_hold = segment_orient_gate.begin_segment_end_hold(
                                                int(crossed),
                                                boundary_key=boundary_key,
                                                do_retouch=False,
                                                retouch_already_consumed=True,
                                                current_orientation=obj_state["orientation"],
                                                current_s=path_following_controller.get_current_s(),
                                            )
                                            if needs_hold:
                                                print(
                                                    f"[orient gate] segment {crossed} end "
                                                    f"θ→{segment_orient_gate.gate_theta:.3f} rad "
                                                    f"at t={t:.2f}s"
                                                )
                                            else:
                                                orient_gate_consumed_boundaries.add(boundary_key)
                                                segment_orient_gate.clear_gate()
                                            vx, vy, w_path = 0.0, 0.0, 0.0
                        elif pursuit_controller is not None:
                            if segment_orient_gate.gate_active:
                                current_s = pursuit_controller.s_progress
                                vx = vy = 0.0
                            else:
                                vc = pursuit_controller.compute_velocity(
                                    obj_state["position"], obj_state["velocity"], dt_pid, omega_override=None
                                )
                                vx, vy = float(vc[0]), float(vc[1])
                                current_s = pursuit_controller.s_progress
                                if segment_theta_specs:
                                    crossed = completed_segment_at_s_crossing(
                                        tracking_prev_s, current_s, segment_theta_specs
                                    )
                                    if crossed is not None:
                                        boundary_key = (int(crossed), int(crossed) + 1)
                                        if boundary_key not in orient_gate_consumed_boundaries:
                                            pursuit_controller.s_along = float(
                                                segment_theta_specs[crossed].s1
                                            )
                                            needs_hold = segment_orient_gate.begin_segment_end_hold(
                                                int(crossed),
                                                boundary_key=boundary_key,
                                                do_retouch=False,
                                                retouch_already_consumed=True,
                                                current_orientation=obj_state["orientation"],
                                                current_s=path_following_controller.get_current_s(),
                                            )
                                            if needs_hold:
                                                print(
                                                    f"[orient gate] segment {crossed} end "
                                                    f"θ→{segment_orient_gate.gate_theta:.3f} rad "
                                                    f"at t={t:.2f}s"
                                                )
                                            else:
                                                orient_gate_consumed_boundaries.add(boundary_key)
                                                segment_orient_gate.clear_gate()
                                            vx, vy = 0.0, 0.0

                        if segment_orient_gate.gate_active:
                            desired_obj_velocity, desired_obj_omega, gate_ok = apply_orientation_hold(
                                current_orientation=obj_state["orientation"],
                                current_angular_velocity=obj_state["angular_velocity"],
                                hold_theta=segment_orient_gate.gate_theta,
                                orientation_tol=float(args.orientation_complete_tol),
                            )
                            if gate_ok:
                                retouch_key, orient_key = segment_orient_gate.clear_gate()
                                if orient_key is not None:
                                    orient_gate_consumed_boundaries.add(orient_key)
                                if retouch_key is not None:
                                    hybrid_retouch_active = True
                                    hybrid_retouch_t0 = t
                                    hybrid_retouch_consumed_boundaries.add(retouch_key)
                                    for rname in robot_agents:
                                        host.robot_states[rname] = RobotState.APPROACHING
                                        robot_agents[rname].set_goal("approach", target_map[rname])
                                    print(
                                        f"[hybrid retouch] boundary {retouch_key[0]}->{retouch_key[1]} "
                                        f"at t={t:.2f}s — path refrozen, approach"
                                    )
                        elif theta_mode_enum == ThetaMode.PATH:
                            desired_obj_velocity = np.array([vx, vy])
                            theta_goal = float(args.path_theta_sine_amp) * float(
                                np.sin(float(args.path_theta_sine_k) * float(current_s))
                            )
                            desired_obj_omega = orientation_pid_omega(
                                obj_state["orientation"], theta_goal, obj_state["angular_velocity"]
                            )
                        elif theta_mode_enum == ThetaMode.FIXED:
                            desired_obj_velocity = np.array([vx, vy])
                            desired_obj_omega = orientation_pid_omega(
                                obj_state["orientation"], args.fixed_theta, obj_state["angular_velocity"]
                            )
                        else:
                            th_goal = theta_goal_at_segment_endpoint(
                                current_s, segment_theta_specs, obj_state["orientation"]
                            )
                            desired_obj_velocity = np.array([vx, vy])
                            desired_obj_omega = orientation_pid_omega(
                                obj_state["orientation"], th_goal, obj_state["angular_velocity"]
                            )

                        if not segment_orient_gate.gate_active:
                            if path_following_controller is not None:
                                s_total = float(path_following_controller.hybrid_path.total_length)
                            elif pursuit_controller is not None:
                                s_total = float(pursuit_controller.L)
                            else:
                                s_total = float(current_s)
                            path_xy_done = holonomic_path_xy_completed(
                                path_following_controller, pursuit_controller
                            )
                            final_theta = final_theta_goal_for_mode(
                                theta_mode_enum,
                                s_total=s_total,
                                theta_milestones=theta_milestones,
                                segment_specs=segment_theta_specs,
                                fixed_theta=float(args.fixed_theta),
                                path_theta_sine_amp=float(args.path_theta_sine_amp),
                                path_theta_sine_k=float(args.path_theta_sine_k),
                            )
                            desired_obj_velocity, desired_obj_omega, _scenario_done = (
                                apply_path_completion_to_desired_motion(
                                    desired_obj_velocity=desired_obj_velocity,
                                    desired_obj_omega=desired_obj_omega,
                                    current_orientation=obj_state["orientation"],
                                    current_angular_velocity=obj_state["angular_velocity"],
                                    path_xy_done=path_xy_done,
                                    final_theta=final_theta,
                                    orientation_tol=float(args.orientation_complete_tol),
                                )
                            )

                        tracking_prev_s = float(current_s)

                        for controller in phase7_controllers.values():
                            controller.desired_object_velocity = desired_obj_velocity
                            controller.desired_object_angular_velocity = desired_obj_omega
                    else:
                        for controller in phase7_controllers.values():
                            controller.desired_object_velocity = np.array([0.0, 0.0])
                            controller.desired_object_angular_velocity = 0.0
                else:
                    for controller in phase7_controllers.values():
                        controller.desired_object_velocity = np.array([0.0, 0.0])
                        controller.desired_object_angular_velocity = 0.0

            for name, agent in robot_agents.items():
                other_positions = [
                    robot_agents[other_name].robot.get_state()[0]
                    for other_name in robot_agents.keys()
                    if other_name != name
                ]
                if agent.goal_type == "push" and name in phase7_controllers:
                    agent.update_contact_state()
                    robot_pos, robot_heading, _ = agent.robot.get_state()
                    controller = phase7_controllers[name]
                    record_history = args.save_dir is not None
                    cmd = controller.compute_velocity(
                        robot_pos=robot_pos,
                        robot_heading=robot_heading,
                        object_pos=obj_state["position"],
                        object_orientation=obj_state["orientation"],
                        object_velocity=obj_state["velocity"],
                        object_angular_velocity=obj_state["angular_velocity"],
                        contact_force=agent.contact_force,
                        in_contact=agent.in_contact,
                        t=t,
                        record_history=record_history,
                        robot=agent.robot,
                    )
                else:
                    obstacles = None
                    if agent.goal_type == "navigate":
                        obstacles = get_object_as_obstacle(
                            generic_object, obj_state["position"], obj_state["orientation"]
                        )
                    cmd = agent.compute_velocity(obj_state, other_positions, obstacles=obstacles)
                agent.robot.command_velocity(cmd)

        pyb.stepSimulation()
        if not args.no_gui:
            time.sleep(TIMESTEP * 0.3)
        step_count += 1
        t += TIMESTEP

        if run_logger is not None and step_count % CTRL_STEP == 0:
            obj_state_log = get_object_state(object_uid)
            n_hist = 0
            if phase7_controllers:
                n_hist = max(
                    (len(c.history.times) for c in phase7_controllers.values()),
                    default=0,
                )
            run_logger.tick(
                sim_t=t,
                obj_state=obj_state_log,
                robot_agents=robot_agents,
                n_hist=n_hist,
            )

    if run_logger is not None:
        run_logger.close()

    if video_log_id is not None and video_path is not None:
        stop_video_recording(video_log_id, video_path)

    if args.save_dir and phase7_controllers:
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        histories = {name: c.history for name, c in phase7_controllers.items()}
        t_params_dict = {name: c.t_param for name, c in phase7_controllers.items()}
        plot_phase7_velocities(
            histories=histories,
            t_params=t_params_dict,
            desired_obj_velocity=np.array([0.0, 0.0]),
            desired_obj_omega=0.0,
            save_path=save_path / f"phase7_swarm_velocities_{run_tag}_w_{selected_name}.png",
        )
        plot_phase_1_results(
            histories=histories,
            t_params=t_params_dict,
            contact_threshold=2.0,
            save_path=save_path / f"phase7_swarm_trajectories_{run_tag}_w_{selected_name}.png",
        )
        plot_phase_7beta(
            histories=histories,
            t_params=t_params_dict,
            save_path=save_path / f"phase7_beta_trajectories_{run_tag}_w_{selected_name}.png",
        )
        plot_phase7_wheel_plot(
            histories=histories,
            t_params=t_params_dict,
            save_path=save_path / f"phase7_wheel_velocities_{run_tag}_w_{selected_name}.png",
        )
        export_histories(
            histories=histories,
            t_params=t_params_dict,
            save_path=save_path / f"histories_{run_tag}_w_{selected_name}.json",
        )
        print(f"Saved plots and histories to {save_path}")

    if pyb.isConnected():
        pyb.disconnect()


if __name__ == "__main__":
    main()
