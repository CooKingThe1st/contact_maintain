#!/usr/bin/env python3
"""
Load a HA_draw JSON scenario and spawn walls/rect obstacles in PyBullet.

This script creates:
  - 4 boundary walls for the map (width x height)
  - each rectangle obstacle as a single filled box
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pybullet as pyb
import pybullet_data

from contact_maintain.pyb_simulation import BulletBlock
from contact_maintain.object_bridge import obj_to_generic


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + 0.5 * self.w

    @property
    def cy(self) -> float:
        return self.y + 0.5 * self.h

    @property
    def xmin(self) -> float:
        return self.x

    @property
    def ymin(self) -> float:
        return self.y

    @property
    def xmax(self) -> float:
        return self.x + self.w

    @property
    def ymax(self) -> float:
        return self.y + self.h


def _rect_from_4_numbers(vals: List[float], map_w: float, map_h: float) -> Rect:
    """
    HA_draw has historically used a couple encodings for rects:
      - [x, y, w, h]  (this is what HA_draw/app.py uses when rendering)
      - [xmin, ymin, xmax, ymax] (older exports / other tools)

    We detect which one is more plausible given map bounds.
    """
    if len(vals) != 4:
        raise ValueError(f"Expected 4 numbers for rect, got {len(vals)}")
    a, b, c, d = (float(v) for v in vals)

    eps = 1e-9

    # Candidate A (preferred): x,y,w,h (HA_draw rendering)
    r_xywh = Rect(x=a, y=b, w=max(0.0, c), h=max(0.0, d))
    xywh_ok = (
        r_xywh.w > eps
        and r_xywh.h > eps
        and r_xywh.x >= -eps
        and r_xywh.y >= -eps
        and (r_xywh.x + r_xywh.w) <= map_w + 1e-3
        and (r_xywh.y + r_xywh.h) <= map_h + 1e-3
    )

    # Candidate B: xmin,ymin,xmax,ymax
    xmin, xmax = (a, c) if a <= c else (c, a)
    ymin, ymax = (b, d) if b <= d else (d, b)
    r_minmax = Rect(x=xmin, y=ymin, w=max(0.0, xmax - xmin), h=max(0.0, ymax - ymin))
    minmax_ok = (
        r_minmax.w > eps
        and r_minmax.h > eps
        and -map_w <= r_minmax.x <= 2 * map_w
        and -map_h <= r_minmax.y <= 2 * map_h
    )

    if xywh_ok and not minmax_ok:
        return r_xywh
    if minmax_ok and not xywh_ok:
        return r_minmax

    # If both plausible, prefer xywh since it matches HA_draw/app.py.
    return r_xywh


def load_scenario(json_path: Path) -> Tuple[float, float, Dict[str, Rect]]:
    data = json.loads(json_path.read_text())
    map_w = float(data["map"]["width"])
    map_h = float(data["map"]["height"])
    rects_raw = data.get("obstacles", {}).get("rects", {}) or {}
    rects: Dict[str, Rect] = {}
    for name, vals in rects_raw.items():
        rects[name] = _rect_from_4_numbers(vals, map_w=map_w, map_h=map_h)
    robot = data.get("robot", {}) or {}
    pose = data.get("pose", {}) or {}
    start = pose.get("start", None)
    robot_shape = str(robot.get("shape_name", "") or "")
    start_pose = None
    if isinstance(start, list) and len(start) >= 3:
        start_pose = (float(start[0]), float(start[1]), float(start[2]))  # yaw is degrees in HA_draw
    return map_w, map_h, rects, robot_shape, start_pose


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

    # Left / right walls (along y)
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

    # Bottom / top walls (along x)
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
    rect: Rect,
    *,
    height: float = 0.5,
    mu: float = 1.0,
    z0: float = 0.0,
) -> BulletBlock:
    """
    Create a single filled rectangular obstacle (box).
    """
    hz = 0.5 * float(height)
    w = max(float(rect.w), 1e-9)
    h = max(float(rect.h), 1e-9)
    return BulletBlock(
        position=[rect.cx, rect.cy, z0 + hz],
        half_extents=[0.5 * w, 0.5 * h, hz],
        mu=mu,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        type=Path,
        default=Path(
            __file__
        ).resolve().parents[3]
        / "scripts"
        / "PathPlanning"
        / "Search_based_Planning"
        / "HA_draw"
        / "rectObs_scenario_bottleneck.json",
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

    map_w, map_h, rects, robot_shape, start_pose = load_scenario(args.json)

    cid = pyb.connect(pyb.GUI if args.gui else pyb.DIRECT)
    try:
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pyb.resetSimulation()
        pyb.setGravity(0, 0, -9.81)
        pyb.setTimeStep(float(args.time_step))
        pyb.loadURDF("plane.urdf")

        # Spawn boundaries + rect obstacles
        _ = spawn_boundary_walls(
            map_w,
            map_h,
            wall_thickness=args.wall_thickness,
            wall_height=args.wall_height,
            mu=args.mu,
        )

        for _, r in rects.items():
            _ = spawn_rect_obstacle(r, height=args.wall_height, mu=args.mu)

        # Optional: spawn robot/object shape at start pose (to match HA_draw scenario).
        # Supported shapes match `test_magnum_motion_planning.py` --object choices.
        obj_file_map = {
            "right_triangle": "right_triangle.obj",
            "bolt": "bolt.obj",
            "pi": "pi.obj",
            "root": "root.obj",
            "rect": "rect.obj",
            "hourglass": "hourglass.obj",
            "meteor": "meteor.obj",
        }
        if start_pose is not None and robot_shape in obj_file_map:
            sx, sy, syaw_deg = start_pose
            _generic_obj, _uid = obj_to_generic(
                obj_path=obj_file_map[robot_shape],
                shape_name=robot_shape,
                position=(sx, sy, float(args.robot_z)),
                orientation=math.radians(syaw_deg),
                mass=float(args.robot_mass),
                lateral_friction=float(args.robot_friction),
                blind_test=True,
            )

        if args.gui:
            # simple idle loop so you can inspect the geometry
            while pyb.isConnected(cid):
                pyb.stepSimulation()
    finally:
        if pyb.isConnected(cid):
            pyb.disconnect()


if __name__ == "__main__":
    main()

