#!/usr/bin/env python3
"""
Headless HA_draw scenario path planning bridge for holonomic magnum tests.

Supports mod_grid (disk / circumradius) with automatic fallback to mod_grid_SE.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HA_DRAW = Path(__file__).resolve().parent
if str(_HA_DRAW) not in sys.path:
    sys.path.insert(0, str(_HA_DRAW))

import HybridAstarPlanner.mod_grid as mod_grid  # noqa: E402
import HybridAstarPlanner.mod_grid_SE as mod_grid_SE  # noqa: E402
from scenario_obstacles import MIN_SAFETY_MARGIN, SWARM_PUSHER_ROBOT_DIAMETER_M, clamp_safety_margin, obstacle_points_for_disk_planner, obstacle_points_from_scenario, parse_scenario_rects, rect_values_for_se, swarm_pusher_min_safety_margin_m  # noqa: E402


@dataclass
class PlannedPath:
    px: List[float]
    py: List[float]
    pyaw: List[float]
    planner: str
    primitives: List[Tuple[str, dict]] = field(default_factory=list)
    df_primitives: List[Dict[str, Any]] = field(default_factory=list)
    stop_phase: int = 3

    @property
    def ok(self) -> bool:
        return len(self.px) >= 2


def _resolve_robot_dict(robot: dict) -> dict:
    out = dict(robot)
    shape = str(out.get("shape_name", ""))
    if shape and not out.get("obj_path"):
        try:
            import rospkg

            pkg = Path(rospkg.RosPack().get_path("contact_maintain"))
            cand = pkg / "urdf" / f"{shape}.obj"
            if cand.is_file():
                out["obj_path"] = str(cand.resolve())
                return out
        except Exception:
            pass
        for parent in [_HA_DRAW, *_HA_DRAW.parents]:
            cand = parent / "urdf" / f"{shape}.obj"
            if cand.is_file():
                out["obj_path"] = str(cand.resolve())
                break
    return out


def _disk_rr(robot: dict, reso: float) -> float:
    return mod_grid._disk_planner_rr_from_scenario(robot, reso=reso)


def _min_safety_margin_from_robot(robot: dict, reso: float) -> float:
    del reso  # margin floor is the swarm pusher fleet, not the pushed-object footprint
    return swarm_pusher_min_safety_margin_m(
        margin_ge_swarm_pusher_size=bool(robot.get("margin_ge_robot_size", True))
    )


def _scenario_safety_margin(robot: dict, reso: float) -> float:
    min_m = _min_safety_margin_from_robot(robot, reso)
    return clamp_safety_margin(float(robot.get("safety_margin", 0.0)), min_margin=min_m)


def resolve_planned_bundle_paths(
    planned_path: Path,
    *,
    search_dirs: Optional[Sequence[Path]] = None,
) -> Tuple[Path, Path, dict, PlannedPath, dict]:
    """
    Load ``*.planned.json`` and its paired scenario JSON.

    Returns (scenario_path, planned_path, scenario_dict, PlannedPath, bundle_dict).
    """
    planned_file = Path(planned_path).resolve()
    scenario_ref, bundle, planned = load_planned_path_bundle(planned_file)
    dirs = [planned_file.parent]
    if search_dirs:
        dirs.extend(Path(d) for d in search_dirs)
    dirs.append(_HA_DRAW)
    scenario_path: Optional[Path] = None
    for folder in dirs:
        cand = Path(folder) / scenario_ref
        if cand.is_file():
            scenario_path = cand.resolve()
            break
    if scenario_path is None:
        raise FileNotFoundError(
            f"Scenario '{scenario_ref}' not found next to {planned_file.name} or in HA_draw."
        )
    scenario = load_scenario_file(scenario_path)
    return scenario_path, planned_file, scenario, planned, bundle


def validate_planned_path_sat(
    scenario: dict,
    px: Sequence[float],
    py: Sequence[float],
    pyaw: Optional[Sequence[float]] = None,
    *,
    safety_margin: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Dense SAT-direct collision check of a planned polyline using the scenario footprint.
    """
    from se2_grid_volume import P3_COLLISION_SAT_DIRECT, build_se2_grid_volume

    if len(px) < 2 or len(py) < 2 or len(px) != len(py):
        return {
            "ok": False,
            "method": "sat_direct",
            "reason": "path_too_short_or_mismatched",
            "n_samples_checked": 0,
        }

    robot = _resolve_robot_dict(scenario.get("robot", {}) or {})
    ox_se, oy_se, reso, map_w, map_h = obstacle_points_from_scenario(scenario)
    margin = (
        float(safety_margin)
        if safety_margin is not None
        else _scenario_safety_margin(robot, reso)
    )
    margin = clamp_safety_margin(margin, min_margin=_min_safety_margin_from_robot(robot, reso))

    rects_raw = (scenario.get("obstacles", {}) or {}).get("rects", {}) or {}
    parsed = parse_scenario_rects(rects_raw, map_w=map_w, map_h=map_h)
    map_bounds = (0.0, 0.0, float(map_w), float(map_h))
    verts = mod_grid_SE._extract_robot_footprint_vertices_local(robot, reso=reso)
    volume = build_se2_grid_volume(
        ox=ox_se,
        oy=oy_se,
        reso=float(reso),
        robot_vertices_local=verts,
        safety_margin=float(margin),
        rects=list(parsed.values()) if parsed else (),
        map_bounds=map_bounds,
    )

    yaw = list(pyaw) if pyaw is not None else []
    if not yaw:
        pose = scenario.get("pose", {}) or {}
        start = pose.get("start", [0.0, 0.0, 0.0])
        syaw = math.radians(float(start[2]) if len(start) >= 3 else 0.0)
        yaw = [syaw] * len(px)
    elif len(yaw) != len(px):
        return {
            "ok": False,
            "method": "sat_direct",
            "reason": "pyaw_length_mismatch",
            "n_samples_checked": 0,
        }

    ok = mod_grid_SE._p3_output_polyline_clear(
        [float(x) for x in px],
        [float(y) for y in py],
        [float(t) for t in yaw],
        volume,
        float(reso),
        P3_COLLISION_SAT_DIRECT,
    )
    return {
        "ok": bool(ok),
        "method": "sat_direct",
        "safety_margin_m": float(margin),
        "n_path_points": len(px),
        "direct_sat_queries": int(getattr(volume, "direct_sat_queries", 0)),
    }


DF_PATH_FORMAT = "df_xy_linear_theta_v1"


def _wrap_angle_rad(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))


def spine_to_df_primitives(
    px: Sequence[float],
    py: Sequence[float],
    pyaw: Sequence[float],
) -> List[Dict[str, Any]]:
    """Phase-1 spine as line segments with linear θ between vertices."""
    n = min(len(px), len(py), len(pyaw))
    if n < 2:
        return []
    out: List[Dict[str, Any]] = []
    for i in range(n - 1):
        out.append(
            {
                "kind": "line",
                "x0": float(px[i]),
                "y0": float(py[i]),
                "x1": float(px[i + 1]),
                "y1": float(py[i + 1]),
                "theta0": _wrap_angle_rad(float(pyaw[i])),
                "theta1": _wrap_angle_rad(float(pyaw[i + 1])),
            }
        )
    return out


def planner_primitives_to_df(
    prims: Sequence[Tuple[str, dict]],
    *,
    syaw: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Normalize disk (S/A) or SE (S/C) planner primitives to export/controller format.

    Each segment: XY line or arc; θ endpoints only (linear in arc length assumed).
    """
    out: List[Dict[str, Any]] = []
    cur_yaw = float(syaw)
    for typ, raw in prims:
        p = dict(raw)
        tag = str(typ).upper()
        if tag == "S":
            t0 = float(p["t0"]) if "t0" in p else cur_yaw
            t1 = float(p["t1"]) if "t1" in p else t0
            out.append(
                {
                    "kind": "line",
                    "x0": float(p["x0"]),
                    "y0": float(p["y0"]),
                    "x1": float(p["x1"]),
                    "y1": float(p["y1"]),
                    "theta0": _wrap_angle_rad(t0),
                    "theta1": _wrap_angle_rad(t1),
                }
            )
            cur_yaw = t1
        elif tag in ("A", "C"):
            sweep = float(p["sweep"])
            t0 = float(p["t0"]) if "t0" in p else cur_yaw
            t1 = float(p["t1"]) if "t1" in p else cur_yaw + sweep
            out.append(
                {
                    "kind": "arc",
                    "cx": float(p["ocx"]),
                    "cy": float(p["ocy"]),
                    "r": float(p["r"]),
                    "a0": float(p["a0"]),
                    "sweep": sweep,
                    "theta0": _wrap_angle_rad(t0),
                    "theta1": _wrap_angle_rad(t1),
                }
            )
            cur_yaw = t1
        else:
            raise ValueError(f"Unsupported planner primitive type {typ!r}")
    return out


def df_primitives_to_planner_tuples(df: Sequence[dict]) -> List[Tuple[str, dict]]:
    """Convert export format to planner tuples for HybridPath construction."""
    out: List[Tuple[str, dict]] = []
    for seg in df:
        kind = str(seg.get("kind", "line"))
        if kind == "line":
            out.append(
                (
                    "S",
                    {
                        "x0": float(seg["x0"]),
                        "y0": float(seg["y0"]),
                        "x1": float(seg["x1"]),
                        "y1": float(seg["y1"]),
                        "t0": float(seg["theta0"]),
                        "t1": float(seg["theta1"]),
                    },
                )
            )
        elif kind == "arc":
            out.append(
                (
                    "C",
                    {
                        "ocx": float(seg["cx"]),
                        "ocy": float(seg["cy"]),
                        "r": float(seg["r"]),
                        "a0": float(seg["a0"]),
                        "sweep": float(seg["sweep"]),
                        "t0": float(seg["theta0"]),
                        "t1": float(seg["theta1"]),
                    },
                )
            )
        else:
            raise ValueError(f"Unsupported df primitive kind {kind!r}")
    return out


def theta_milestones_from_df_primitives(
    df: Sequence[dict],
) -> Tuple[List[float], List[float]]:
    """Arc-length s and θ at primitive boundaries (for waypoint PID)."""
    if not df:
        return [], []
    s_list = [0.0]
    theta_list = [_wrap_angle_rad(float(df[0]["theta0"]))]
    cum = 0.0
    for seg in df:
        kind = str(seg.get("kind", "line"))
        if kind == "line":
            length = math.hypot(
                float(seg["x1"]) - float(seg["x0"]),
                float(seg["y1"]) - float(seg["y0"]),
            )
        else:
            length = abs(float(seg["sweep"])) * max(float(seg["r"]), 1e-9)
        cum += float(length)
        s_list.append(cum)
        theta_list.append(_wrap_angle_rad(float(seg["theta1"])))
    return s_list, theta_list


def _resolve_df_primitives(
    *,
    prims: Optional[Sequence[Tuple[str, dict]]],
    px: Sequence[float],
    py: Sequence[float],
    pyaw: Sequence[float],
    syaw: float,
) -> List[Dict[str, Any]]:
    if prims:
        return planner_primitives_to_df(prims, syaw=syaw)
    return spine_to_df_primitives(px, py, pyaw)


def plan_scenario_path(
    scenario: dict,
    *,
    planner: str = "auto",
    stop_phase: int = 3,
    start_override: Optional[Tuple[float, float, float]] = None,
    goal_override: Optional[Tuple[float, float, float]] = None,
    se_p3_primitive: str = "linear_yaw_dp",
    se_p3_collision_mode: str = "volume_bin",
    dp_objective: str = "length",
    disk_collision_mode: str = "offline",
) -> PlannedPath:
    """
    Plan a path through scenario obstacles.

    planner: 'auto' | 'disk' | 'se'
    start_override / goal_override: (x, y, yaw_deg) replace pose in scenario.
    """
    robot = _resolve_robot_dict(scenario.get("robot", {}) or {})
    ox_disk, oy_disk, reso, map_w, map_h = obstacle_points_for_disk_planner(scenario)
    safety_margin = _scenario_safety_margin(robot, reso)
    ox_se, oy_se, _, _, _ = obstacle_points_from_scenario(scenario)

    pose = scenario.get("pose", {}) or {}
    s = start_override or tuple(pose.get("start", [0.0, 0.0, 0.0]))
    g = goal_override or tuple(pose.get("goal", [0.0, 0.0, 0.0]))
    sx, sy = float(s[0]), float(s[1])
    gx, gy = float(g[0]), float(g[1])
    syaw = math.radians(float(s[2]) if len(s) >= 3 else 0.0)
    gyaw = math.radians(float(g[2]) if len(g) >= 3 else 0.0)

    rects_raw = (scenario.get("obstacles", {}) or {}).get("rects", {}) or {}
    parsed = parse_scenario_rects(rects_raw, map_w=map_w, map_h=map_h)
    obstacle_rects = rect_values_for_se(parsed) if parsed else []
    planner_rects = list(parsed.values()) if parsed else []
    map_bounds = (0.0, 0.0, float(map_w), float(map_h))

    mode = planner.lower().strip()
    use_se = mode in ("se", "mod_grid_se", "mod_grid_se2")
    px: List[float] = []
    py: List[float] = []
    pyaw: List[float] = []
    prims: List[Tuple[str, dict]] = []
    used = "mod_grid"

    if not use_se:
        rr = _disk_rr(robot, reso)
        if stop_phase == 3:
            out = mod_grid.astar_planning(
                sx,
                sy,
                gx,
                gy,
                ox_disk,
                oy_disk,
                reso,
                rr,
                stop_phase=stop_phase,
                safety_margin=safety_margin,
                obstacle_rects=planner_rects if planner_rects else None,
                dp_objective=dp_objective,
                disk_collision_mode=disk_collision_mode,
                return_primitives=True,
            )
            if len(out) == 3:
                px, py, prims = out
            else:
                px, py = out
                prims = []
        else:
            px, py = mod_grid.astar_planning(
                sx,
                sy,
                gx,
                gy,
                ox_disk,
                oy_disk,
                reso,
                rr,
                stop_phase=stop_phase,
                safety_margin=safety_margin,
                obstacle_rects=planner_rects if planner_rects else None,
                disk_collision_mode=disk_collision_mode,
            )
        if not px and mode == "auto":
            use_se = True

    if use_se or (mode == "auto" and not px):
        used = "mod_grid_SE"
        verts = mod_grid_SE._extract_robot_footprint_vertices_local(robot, reso=reso)
        path_stats: Dict[str, Any] = {}
        px, py, pyaw = mod_grid_SE.astar_planning(
            sx=sx,
            sy=sy,
            syaw_rad=syaw,
            gx=gx,
            gy=gy,
            gyaw_rad=gyaw,
            ox=ox_se,
            oy=oy_se,
            reso=reso,
            robot_vertices_local=verts,
            stop_phase=stop_phase,
            safety_margin=safety_margin,
            obstacle_rects=obstacle_rects if obstacle_rects else None,
            map_bounds=map_bounds,
            se_p3_primitive=se_p3_primitive,
            se_p3_collision_mode=se_p3_collision_mode,
            dp_objective=dp_objective,
            path_stats=path_stats if stop_phase == 3 else None,
        )
        prims_raw = path_stats.get("primitives", []) or []
        if prims_raw:
            prims = [
                (str(item["type"]), dict(item["params"]))
                for item in prims_raw
                if isinstance(item, dict) and "type" in item and "params" in item
            ]
    elif not pyaw and px:
        pyaw = [syaw] * len(px)

    df_prims = _resolve_df_primitives(prims=prims, px=px, py=py, pyaw=pyaw, syaw=syaw)
    return PlannedPath(
        px=list(px),
        py=list(py),
        pyaw=list(pyaw),
        planner=used,
        primitives=prims,
        df_primitives=df_prims,
        stop_phase=stop_phase,
    )


def path_to_json_dict(planned: PlannedPath) -> Dict[str, Any]:
    return {
        "px": planned.px,
        "py": planned.py,
        "pyaw": planned.pyaw,
        "planner": planned.planner,
        "stop_phase": planned.stop_phase,
        "primitives": [{"type": t, "params": p} for t, p in planned.primitives],
    }


def build_planned_path_bundle(
    *,
    scenario: dict,
    scenario_filename: str,
    px: Sequence[float],
    py: Sequence[float],
    pyaw: Optional[Sequence[float]] = None,
    planner: str,
    stop_phase: int,
    path_stats: Optional[Dict[str, Any]] = None,
    planner_options: Optional[Dict[str, Any]] = None,
    safety_margin: float = 0.0,
    safety_margin_ge_robot_size: bool = True,
    min_safety_margin_m: float = MIN_SAFETY_MARGIN,
    prims: Optional[Sequence[Tuple[str, dict]]] = None,
    export_sat_validation: Optional[Dict[str, Any]] = None,
    exported_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Self-describing planned-path export for holonomic controller tests.

    Saves as a sibling ``*.planned.json`` next to the scenario file; ``scenario_ref``
    names the environment JSON that must be loaded together.
    """
    yaw = list(pyaw) if pyaw is not None else []
    pose = scenario.get("pose", {}) or {}
    start = pose.get("start", [0.0, 0.0, 0.0])
    syaw = math.radians(float(start[2]) if len(start) >= 3 else 0.0)
    if not yaw and px:
        yaw = [syaw] * len(px)
    df_prims = _resolve_df_primitives(prims=prims, px=px, py=py, pyaw=yaw, syaw=syaw)
    path_block: Dict[str, Any] = {
        "format": DF_PATH_FORMAT,
        "df_primitives": df_prims,
        "dense": {
            "px": [float(x) for x in px],
            "py": [float(y) for y in py],
            "pyaw": [float(t) for t in yaw],
        },
        "planner": str(planner),
        "stop_phase": int(stop_phase),
        "recommended_tracker": "hybrid" if df_prims else "pursuit",
    }
    if prims:
        path_block["primitives"] = [{"type": str(t), "params": dict(p)} for t, p in prims]
    if path_stats:
        path_block["path_stats"] = dict(path_stats)
    if planner_options:
        path_block["planner_options"] = dict(planner_options)

    planned_name = f"{Path(scenario_filename).stem}.planned.json"
    ts = exported_at or datetime.now(timezone.utc).isoformat()
    metadata: Dict[str, Any] = {
        "exported_from": "HA_draw",
        "exported_at": ts,
        "scenario_ref": str(scenario_filename),
        "scenario_filename": str(scenario_filename),
        "planned_filename": planned_name,
        "controller_hint": f"python3 test_magnum_holonomic_control.py --planned-path {planned_name}",
    }
    if export_sat_validation is not None:
        metadata["export_sat_validation"] = dict(export_sat_validation)
    if path_stats:
        metadata["path_stats_summary"] = {
            k: path_stats[k]
            for k in (
                "output_pts",
                "polyline_length_m",
                "n_primitives",
                "p3_fallback",
                "p3_compressed",
                "direct_sat_queries",
            )
            if k in path_stats
        }
    if planner_options:
        metadata["planner_options"] = dict(planner_options)

    return {
        "version": 1,
        "scenario_ref": str(scenario_filename),
        "path": path_block,
        "safety": {
            "margin_m": float(safety_margin),
            "margin_ge_robot_size": bool(safety_margin_ge_robot_size),
            "min_margin_m": float(min_safety_margin_m),
            "swarm_pusher_diameter_m": float(SWARM_PUSHER_ROBOT_DIAMETER_M),
            "note": (
                "margin_ge_robot_size refers to the Magnum swarm pusher fleet "
                "(test ROBOT_RADIUS), not the pushed-object footprint."
            ),
        },
        "metadata": metadata,
    }


def load_planned_path_bundle(path: Path) -> Tuple[str, dict, PlannedPath]:
    """Load ``*.planned.json``; returns (scenario_ref, bundle_dict, PlannedPath)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    scenario_ref = str(data.get("scenario_ref", ""))
    path_block = data.get("path", {}) or {}

    df_raw = path_block.get("df_primitives", []) or []
    if df_raw:
        df_prims = [dict(item) for item in df_raw if isinstance(item, dict)]
        prims = df_primitives_to_planner_tuples(df_prims)
    else:
        df_prims = []
        prims_raw = path_block.get("primitives", []) or []
        prims = [
            (str(item["type"]), dict(item["params"]))
            for item in prims_raw
            if isinstance(item, dict) and "type" in item and "params" in item
        ]

    dense = path_block.get("dense", {}) or {}
    px = dense.get("px", path_block.get("px", []))
    py = dense.get("py", path_block.get("py", []))
    pyaw = dense.get("pyaw", path_block.get("pyaw", []))

    if not df_prims and prims:
        pose_block = data.get("safety", {})
        syaw = 0.0
        if pyaw:
            syaw = float(pyaw[0])
        df_prims = planner_primitives_to_df(prims, syaw=syaw)
    elif not df_prims and px and py:
        yaw = [float(t) for t in pyaw] if pyaw else [0.0] * len(px)
        df_prims = spine_to_df_primitives(px, py, yaw)

    planned = PlannedPath(
        px=[float(x) for x in px],
        py=[float(y) for y in py],
        pyaw=[float(t) for t in pyaw],
        planner=str(path_block.get("planner", "unknown")),
        primitives=prims,
        df_primitives=df_prims,
        stop_phase=int(path_block.get("stop_phase", 3)),
    )
    return scenario_ref, data, planned


def write_planned_path_export_pair(
    scenario: dict,
    scenario_filename: str,
    bundle: Dict[str, Any],
    out_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Write environment JSON + ``*.planned.json`` sibling pair."""
    folder = Path(out_dir) if out_dir is not None else _HA_DRAW
    scenario_path = folder / scenario_filename
    scenario_copy = dict(scenario)
    scenario_copy.pop("path", None)
    scenario_path.write_text(json.dumps(scenario_copy, indent=2), encoding="utf-8")
    planned_path = folder / f"{Path(scenario_filename).stem}.planned.json"
    planned_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return scenario_path, planned_path


def save_planned_path(scenario_path: Path, planned: PlannedPath, out_path: Optional[Path] = None) -> Path:
    scenario = json.loads(scenario_path.read_text())
    scenario["path"] = path_to_json_dict(planned)
    dest = out_path or scenario_path.with_suffix(".path.json")
    dest.write_text(json.dumps(scenario, indent=2))
    return dest


def load_scenario_file(path: Path) -> dict:
    return json.loads(path.read_text())
