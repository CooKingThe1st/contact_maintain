#!/usr/bin/env python3
"""
Load a HA_draw JSON scenario and spawn walls/rect obstacles in PyBullet.

This script creates:
  - 4 boundary walls for the map (width x height)
  - each rectangle obstacle as a single filled box (axis-aligned or rotated)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pybullet as pyb
import pybullet_data

from contact_maintain.pyb_simulation import BulletBlock
from contact_maintain.object_bridge import obj_to_generic

_HA_DRAW = (
    Path(__file__).resolve().parents[2]
    / "PathPlanning"
    / "Search_based_Planning"
    / "HA_draw"
)
if str(_HA_DRAW) not in sys.path:
    sys.path.insert(0, str(_HA_DRAW))

from scenario_obstacles import ObstacleRect, parse_scenario_rects  # noqa: E402


OBJ_SHAPE_FILES = {
    "right_triangle": "right_triangle.obj",
    "bolt": "bolt.obj",
    "pi": "pi.obj",
    "root": "root.obj",
    "rect": "rect.obj",
    "hourglass": "hourglass.obj",
    "meteor": "meteor.obj",
}


def load_scenario(json_path: Path) -> Tuple[float, float, Dict[str, ObstacleRect], str, Optional[Tuple[float, float, float]], dict]:
    data = json.loads(json_path.read_text())
    map_w = float(data["map"]["width"])
    map_h = float(data["map"]["height"])
    rects_raw = data.get("obstacles", {}).get("rects", {}) or {}
    rects = parse_scenario_rects(rects_raw, map_w=map_w, map_h=map_h)
    robot = data.get("robot", {}) or {}
    pose = data.get("pose", {}) or {}
    start = pose.get("start", None)
    robot_shape = str(robot.get("shape_name", "") or "")
    start_pose = None
    if isinstance(start, list) and len(start) >= 3:
        start_pose = (float(start[0]), float(start[1]), float(start[2]))
    return map_w, map_h, rects, robot_shape, start_pose, data


def spawn_boundary_walls(
    map_w: float,
    map_h: float,
    *,
    wall_thickness: float = 0.2,
    wall_height: float = 0.5,
    mu: float = 1.0,
    z0: float = 0.0,
) -> List[BulletBlock]:
    t = float(wall_thickness)
    hz = 0.5 * float(wall_height)
    blocks: List[BulletBlock] = []

    blocks.append(
        BulletBlock(
            position=[-0.5 * t, 0.5 * map_h, z0 + hz],
            half_extents=[0.5 * t, 0.5 * map_h + 0.5 * t, hz],
            mu=mu,
        )
    )
    blocks.append(
        BulletBlock(
            position=[map_w + 0.5 * t, 0.5 * map_h, z0 + hz],
            half_extents=[0.5 * t, 0.5 * map_h + 0.5 * t, hz],
            mu=mu,
        )
    )
    blocks.append(
        BulletBlock(
            position=[0.5 * map_w, -0.5 * t, z0 + hz],
            half_extents=[0.5 * map_w + 0.5 * t, 0.5 * t, hz],
            mu=mu,
        )
    )
    blocks.append(
        BulletBlock(
            position=[0.5 * map_w, map_h + 0.5 * t, z0 + hz],
            half_extents=[0.5 * map_w + 0.5 * t, 0.5 * t, hz],
            mu=mu,
        )
    )

    return blocks


def spawn_rect_obstacle(
    rect: ObstacleRect,
    *,
    height: float = 0.5,
    mu: float = 1.0,
    z0: float = 0.0,
) -> BulletBlock:
    """Create a rectangular obstacle (axis-aligned or rotated about Z)."""
    hz = 0.5 * float(height)
    w = max(float(rect.w), 1e-9)
    h = max(float(rect.h), 1e-9)
    yaw = math.radians(float(rect.angle_deg))
    orientation = pyb.getQuaternionFromEuler([0.0, 0.0, yaw])
    return BulletBlock(
        position=[rect.cx, rect.cy, z0 + hz],
        half_extents=[0.5 * w, 0.5 * h, hz],
        mu=mu,
        orientation=orientation,
    )


def spawn_scenario_obstacles(
    scenario: dict,
    *,
    wall_thickness: float = 0.2,
    wall_height: float = 0.5,
    mu: float = 1.0,
    z0: float = 0.0,
    spawn_walls: bool = True,
) -> List[BulletBlock]:
    """Spawn boundary walls and all rectangle obstacles from a scenario dict."""
    map_w = float(scenario["map"]["width"])
    map_h = float(scenario["map"]["height"])
    rects_raw = scenario.get("obstacles", {}).get("rects", {}) or {}
    rects = parse_scenario_rects(rects_raw, map_w=map_w, map_h=map_h)

    blocks: List[BulletBlock] = []
    if spawn_walls:
        blocks.extend(
            spawn_boundary_walls(
                map_w,
                map_h,
                wall_thickness=wall_thickness,
                wall_height=wall_height,
                mu=mu,
                z0=z0,
            )
        )
    for rect in rects.values():
        blocks.append(spawn_rect_obstacle(rect, height=wall_height, mu=mu, z0=z0))
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        type=Path,
        default=_HA_DRAW / "rectObs_scenario_bottleneck.json",
        help="Path to HA_draw scenario JSON",
    )
    ap.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    ap.add_argument("--wall_thickness", type=float, default=0.2)
    ap.add_argument("--wall_height", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--robot_z", type=float, default=0.2, help="Z height for spawned robot shape")
    ap.add_argument("--robot_mass", type=float, default=1.0, help="Mass for spawned robot shape")
    ap.add_argument("--robot_friction", type=float, default=0.3, help="Friction for spawned robot shape")
    ap.add_argument("--time_step", type=float, default=1.0 / 240.0)
    args = ap.parse_args()

    map_w, map_h, rects, robot_shape, start_pose, scenario = load_scenario(args.json)

    cid = pyb.connect(pyb.GUI if args.gui else pyb.DIRECT)
    try:
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pyb.resetSimulation()
        pyb.setGravity(0, 0, -9.81)
        pyb.setTimeStep(float(args.time_step))
        pyb.loadURDF("plane.urdf")

        _ = spawn_scenario_obstacles(
            scenario,
            wall_thickness=args.wall_thickness,
            wall_height=args.wall_height,
            mu=args.mu,
        )

        if start_pose is not None and robot_shape in OBJ_SHAPE_FILES:
            sx, sy, syaw_deg = start_pose
            _generic_obj, _uid = obj_to_generic(
                obj_path=OBJ_SHAPE_FILES[robot_shape],
                shape_name=robot_shape,
                position=(sx, sy, float(args.robot_z)),
                orientation=math.radians(syaw_deg),
                mass=float(args.robot_mass),
                lateral_friction=float(args.robot_friction),
                blind_test=True,
            )

        if args.gui:
            while pyb.isConnected(cid):
                pyb.stepSimulation()
    finally:
        if pyb.isConnected(cid):
            pyb.disconnect()


if __name__ == "__main__":
    main()
