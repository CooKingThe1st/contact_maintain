import json
import math
import re
import sys
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


THIS_DIR = Path(__file__).resolve().parent


def _find_motion_planner_dir(start_dir: Path) -> Path:
    for parent in [start_dir, *start_dir.parents]:
        candidate = parent / "scripts" / "MotionPlanning"
        if (candidate / "HybridAstarPlanner").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate scripts/MotionPlanning/HybridAstarPlanner "
        f"from {start_dir}"
    )


MOTION_PLANNER_DIR = _find_motion_planner_dir(THIS_DIR)
# Prefer local HA_draw planner package first.
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
# Keep original MotionPlanning path available for CurvesGenerator imports.
if str(MOTION_PLANNER_DIR) not in sys.path:
    sys.path.append(str(MOTION_PLANNER_DIR))


def _ensure_legacy_object_utils_path() -> bool:
    """So `object_utils.create_standard_objects` matches mod_grid_SE / contact_maintain."""
    for parent in [THIS_DIR, *THIS_DIR.parents]:
        cand_src = parent / "src"
        if (cand_src / "legacy" / "object_utils.py").exists():
            if str(cand_src) not in sys.path:
                sys.path.insert(0, str(cand_src))
            if str(cand_src / "legacy") not in sys.path:
                sys.path.insert(0, str(cand_src / "legacy"))
            return True
    return False


def robot_shape_options_from_create_standard_objects() -> List[str]:
    """All keys from create_standard_objects() (sorted); used for the Robot shape combobox."""
    if _ensure_legacy_object_utils_path():
        try:
            from object_utils import create_standard_objects

            return sorted(create_standard_objects().keys())
        except Exception:
            pass
    # Fallback if legacy tree is missing or import fails (matches object_utils as of last sync).
    return sorted(
        [
            "arrow",
            "asym_l_shape",
            "asym_quad",
            "boot",
            "circle",
            "crescent",
            "crescent_asym",
            "ellipse",
            "fat_triangle",
            "l_shape",
            "narrow_triangle",
            "obese_triangle",
            "pentagon_asym",
            "plus",
            "rectangle",
            "scalene",
            "star",
            "t_shape",
            "trapezoid",
            "triangle",
            "u_shape",
            "wedge",
        ]
    )


import HybridAstarPlanner.astar as grid_astar
import HybridAstarPlanner.mod_grid as mod_grid_astar
import HybridAstarPlanner.mod_grid_SE as mod_grid_SE_astar


def _import_hybrid_astar():
    try:
        import HybridAstarPlanner.hybrid_astar as hybrid_astar_mod

        return hybrid_astar_mod
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        raise ModuleNotFoundError(
            f"{missing}. Install Hybrid A* extras, e.g. "
            "`pip install heapdict scipy` (see repo requirements.txt)."
        ) from e


@dataclass
class Pose:
    x: float
    y: float
    yaw_deg: float


@dataclass
class MapState:
    map_width: float = 60.0
    map_height: float = 40.0
    resolution: float = 1.0


@dataclass
class RobotState:
    robot_type: str = "holonomic"
    robot_width: float = 2.0
    robot_length: float = 3.0


@dataclass
class ObstacleState:
    rects: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    lines: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    rect_count: int = 0
    line_count: int = 0

    def next_rect_id(self) -> str:
        self.rect_count += 1
        return f"RECT_{self.rect_count:03d}"

    def next_line_id(self) -> str:
        self.line_count += 1
        return f"LINE_{self.line_count:03d}"


class PlannerWorkbench:
    """
    Tkinter shell + embedded Matplotlib map.
    Avoids matplotlib.widgets (TextBox/Button), which often break on WSL / some backends.
    """

    def __init__(self):
        self.map_state = MapState()
        self.robot_state = RobotState()
        self.start = Pose(5.0, 5.0, 0.0)
        self.goal = Pose(50.0, 30.0, 0.0)
        self.obstacles = ObstacleState()

        self.mode = "rect"
        self.drag_start: Optional[Tuple[float, float]] = None
        self.current_line: List[Tuple[float, float]] = []
        self.path_data: Optional[Tuple[List[float], List[float]]] = None
        # Path footprint visualization (aligned with planner geometry).
        self._path_footprint_mode: Optional[str] = None  # "disk" | "polygon"
        self._disk_radius_for_viz: Optional[float] = None
        self._path_pyaw: Optional[List[float]] = None
        self._robot_vertices_local: Optional[List[Tuple[float, float]]] = None
        # Created after tk.Tk() to avoid "Too early to create variable" errors.
        self.var_mod_grid_phase: Optional[tk.IntVar] = None
        self.status_text = "Ready"
        self.line_thickness = 5.0
        self.safety_margin = 0.0
        self._suppress_mode_cb = False

        self.root = tk.Tk()
        self.var_mod_grid_phase = tk.IntVar(master=self.root, value=3)
        self.var_robot_shape = tk.StringVar(master=self.root, value="circle")
        self.var_yaw_fill = tk.StringVar(master=self.root, value="none")
        self.root.title("Hybrid/A* Planning Workbench (Tk + Matplotlib)")
        self.root.minsize(900, 600)

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # --- Settings (left) ---
        left_frame = tk.Frame(main, width=340)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left_frame.pack_propagate(True)

        # --- Map (middle) ---
        map_frame = tk.Frame(main)
        map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")

        self.canvas = FigureCanvasTkAgg(self.fig, master=map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        hint = tk.Label(
            map_frame,
            text="LMB: draw / erase  |  Wheel: zoom  |  MMB drag: pan",
            font=("TkDefaultFont", 8),
        )
        hint.pack(anchor=tk.W)

        # --- Planner + actions (right) ---
        right_frame = tk.Frame(main, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right_frame.pack_propagate(True)

        self._build_settings_panel(left_frame)
        self._build_planner_panel(right_frame)
        self._last_planner_choice = self.var_planner.get()

        self._connect_canvas_events()
        self.reset_view()
        self.render()

    def _row_entry(self, parent, label: str, initial: str) -> tk.Entry:
        f = tk.Frame(parent)
        f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, width=12, anchor=tk.W).pack(side=tk.LEFT)
        e = tk.Entry(f, width=14)
        e.insert(0, initial)
        e.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return e

    def _build_control_panel(self, ctrl: tk.Frame):
        self.var_scenario_file = tk.StringVar(value="")

        tk.Label(ctrl, text="Start", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W)
        self.ent_sx = self._row_entry(ctrl, "sx", str(self.start.x))
        self.ent_sy = self._row_entry(ctrl, "sy", str(self.start.y))
        self.ent_syaw = self._row_entry(ctrl, "syaw°", str(self.start.yaw_deg))

        tk.Label(ctrl, text="Goal", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.ent_gx = self._row_entry(ctrl, "gx", str(self.goal.x))
        self.ent_gy = self._row_entry(ctrl, "gy", str(self.goal.y))
        self.ent_gyaw = self._row_entry(ctrl, "gyaw°", str(self.goal.yaw_deg))
        tk.Button(ctrl, text="Apply Pose", command=self.apply_pose, width=22).pack(fill=tk.X, pady=(6, 0))

        tk.Label(ctrl, text="Map", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.ent_map_w = self._row_entry(ctrl, "width", str(self.map_state.map_width))
        self.ent_map_h = self._row_entry(ctrl, "height", str(self.map_state.map_height))
        self.ent_reso = self._row_entry(ctrl, "resolution", str(self.map_state.resolution))
        tk.Button(ctrl, text="Apply Map", command=self.apply_map, width=22).pack(fill=tk.X, pady=(6, 0))

        tk.Label(ctrl, text="Scenario file to load", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.cmb_scenarios = ttk.Combobox(
            ctrl, textvariable=self.var_scenario_file, state="readonly", width=22
        )
        self.cmb_scenarios.pack(fill=tk.X, pady=(2, 0))
        self._refresh_scenario_combo()
        tk.Button(ctrl, text="Load scenario", command=self.load_scenario, width=22).pack(fill=tk.X, pady=(4, 0))
        tk.Button(ctrl, text="Save scenario", command=self.save_scenario, width=22).pack(fill=tk.X, pady=(2, 0))

        tk.Label(ctrl, text="Robot shape", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        # Use SE(2) footprint planning for non-circular shapes; names match create_standard_objects().
        self._robot_shape_options = robot_shape_options_from_create_standard_objects()
        self.cmb_robot_shape = ttk.Combobox(
            ctrl,
            textvariable=self.var_robot_shape,
            state="readonly",
            values=self._robot_shape_options,
            width=22,
        )
        self.cmb_robot_shape.pack(fill=tk.X, pady=(2, 0))
        self.cmb_robot_shape.bind("<<ComboboxSelected>>", self._on_robot_shape_changed)

        tk.Label(ctrl, text="Edit mode", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.var_mode = tk.StringVar(value="rect")
        mode_row = tk.Frame(ctrl)
        mode_row.pack(fill=tk.X, pady=(2, 0))
        mode_left = tk.Frame(mode_row)
        mode_left.pack(side=tk.LEFT)
        for m in ("rect", "line", "erase"):
            tk.Radiobutton(
                mode_left,
                text=m,
                variable=self.var_mode,
                value=m,
                command=self._on_mode_changed,
            ).pack(side=tk.LEFT, padx=(0, 8))
        thick_f = tk.Frame(mode_row)
        thick_f.pack(side=tk.RIGHT)
        tk.Label(thick_f, text="line thick").pack(side=tk.LEFT, padx=(8, 4))
        self.ent_line_th = tk.Entry(thick_f, width=8)
        self.ent_line_th.insert(0, str(self.line_thickness))
        self.ent_line_th.pack(side=tk.LEFT)

        tk.Label(ctrl, text="Planner", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.var_robot = tk.StringVar(value="holonomic")
        tk.Radiobutton(
            ctrl,
            text="holonomic (grid A*)",
            variable=self.var_robot,
            value="holonomic",
            command=self._on_robot_changed,
        ).pack(anchor=tk.W)
        tk.Radiobutton(
            ctrl,
            text="car (hybrid A*)",
            variable=self.var_robot,
            value="car",
            command=self._on_robot_changed,
        ).pack(anchor=tk.W)

        tk.Label(ctrl, text="Planner Method", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.var_planner = tk.StringVar(value="grid_astar")
        self.rb_grid = tk.Radiobutton(
            ctrl,
            text="grid_astar (baseline)",
            variable=self.var_planner,
            value="grid_astar",
            command=self._on_planner_changed,
        )
        self.rb_grid.pack(anchor=tk.W)
        self.rb_mod_grid = tk.Radiobutton(
            ctrl,
            text="mod_grid (disk safe+smooth)",
            variable=self.var_planner,
            value="mod_grid",
            command=self._on_planner_changed,
        )
        self.rb_mod_grid.pack(anchor=tk.W)
        self.rb_mod_grid_se = tk.Radiobutton(
            ctrl,
            text="mod_grid_SE (footprint SE2)",
            variable=self.var_planner,
            value="mod_grid_se",
            command=self._on_planner_changed,
        )
        self.rb_mod_grid_se.pack(anchor=tk.W)
        self.rb_hybrid = tk.Radiobutton(
            ctrl,
            text="hybrid_astar",
            variable=self.var_planner,
            value="hybrid_astar",
            command=self._on_planner_changed,
        )
        self.rb_hybrid.pack(anchor=tk.W)

        # mod_grid early-exit / optimization choice
        tk.Label(ctrl, text="mod_grid stop phase", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.rb_mod_phase1 = tk.Radiobutton(
            ctrl, text="1 (augmented A*)", variable=self.var_mod_grid_phase, value=1, command=self._on_planner_changed
        )
        self.rb_mod_phase2 = tk.Radiobutton(
            ctrl, text="2 (CHOMP final + shortcut)", variable=self.var_mod_grid_phase, value=2, command=self._on_planner_changed
        )
        self.rb_mod_phase3 = tk.Radiobutton(
            ctrl, text="2 (straight+arc alternative)", variable=self.var_mod_grid_phase, value=3, command=self._on_planner_changed
        )
        self.rb_mod_phase1.pack(anchor=tk.W)
        self.rb_mod_phase2.pack(anchor=tk.W)
        self.rb_mod_phase3.pack(anchor=tk.W)

        tk.Label(ctrl, text="Yaw Filling", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.rb_yaw_none = tk.Radiobutton(
            ctrl,
            text="none",
            variable=self.var_yaw_fill,
            value="none",
            command=self._on_planner_changed,
        )
        self.rb_yaw_none.pack(anchor=tk.W)
        self.rb_yaw_linear = tk.Radiobutton(
            ctrl,
            text="Linear Interpolate",
            variable=self.var_yaw_fill,
            value="linear",
            command=self._on_planner_changed,
        )
        self.rb_yaw_linear.pack(anchor=tk.W)
        self.rb_yaw_df = tk.Radiobutton(
            ctrl,
            text="Differential Flatness",
            variable=self.var_yaw_fill,
            value="differential_flatness",
            command=self._on_planner_changed,
        )
        self.rb_yaw_df.pack(anchor=tk.W)

        self._update_planner_options()

        tk.Label(ctrl, text="Planner actions", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(12, 0))

        # Vertical stack to avoid clipping in smaller window sizes / font differences.
        btnf = tk.Frame(ctrl)
        btnf.pack(fill=tk.X, pady=(8, 0))

        def _btn(text: str, cmd):
            tk.Button(btnf, text=text, command=cmd, width=22).pack(fill=tk.X, pady=2)

        _btn("Apply line thick", self.apply_robot)
        _btn("Plan / Replan", self.replan)
        _btn("Clear Obst", self.clear_obstacles)
        _btn("Reset View", self.reset_view)

        # Lightweight status display so it's obvious that button actions executed.
        status_wrap = tk.Frame(ctrl, width=280, height=120)
        status_wrap.pack(fill=tk.X, pady=(10, 0))
        status_wrap.pack_propagate(False)
        self.txt_status = tk.Text(status_wrap, width=38, height=7, wrap=tk.WORD, state=tk.DISABLED)
        self.txt_status.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll = tk.Scrollbar(status_wrap, command=self.txt_status.yview)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_status.configure(yscrollcommand=status_scroll.set)
        self._status_last_logged: Optional[str] = None

    def _append_status_log(self, line: str) -> None:
        """
        Append an informational line to the status text box without changing `status_text`.
        This is used for timing diagnostics so they don't overwrite the main summary.
        """
        if not hasattr(self, "txt_status"):
            return
        self.txt_status.configure(state=tk.NORMAL)
        if self.txt_status.index("end-1c") != "1.0":
            self.txt_status.insert(tk.END, "\n")
        self.txt_status.insert(tk.END, str(line))
        self.txt_status.see(tk.END)
        self.txt_status.configure(state=tk.DISABLED)

    def _build_settings_panel(self, ctrl: tk.Frame):
        self.var_scenario_file = tk.StringVar(value="")

        tk.Label(ctrl, text="Start", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W)
        self.ent_sx = self._row_entry(ctrl, "sx", str(self.start.x))
        self.ent_sy = self._row_entry(ctrl, "sy", str(self.start.y))
        self.ent_syaw = self._row_entry(ctrl, "syaw°", str(self.start.yaw_deg))

        tk.Label(ctrl, text="Goal", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.ent_gx = self._row_entry(ctrl, "gx", str(self.goal.x))
        self.ent_gy = self._row_entry(ctrl, "gy", str(self.goal.y))
        self.ent_gyaw = self._row_entry(ctrl, "gyaw°", str(self.goal.yaw_deg))
        tk.Button(ctrl, text="Apply Pose", command=self.apply_pose, width=22).pack(fill=tk.X, pady=(6, 0))

        tk.Label(ctrl, text="Map", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.ent_map_w = self._row_entry(ctrl, "width", str(self.map_state.map_width))
        self.ent_map_h = self._row_entry(ctrl, "height", str(self.map_state.map_height))
        self.ent_reso = self._row_entry(ctrl, "resolution", str(self.map_state.resolution))
        tk.Button(ctrl, text="Apply Map", command=self.apply_map, width=22).pack(fill=tk.X, pady=(6, 0))

        tk.Label(ctrl, text="Scenario file to load", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.cmb_scenarios = ttk.Combobox(ctrl, textvariable=self.var_scenario_file, state="readonly", width=22)
        self.cmb_scenarios.pack(fill=tk.X, pady=(2, 0))
        self._refresh_scenario_combo()
        tk.Button(ctrl, text="Load scenario", command=self.load_scenario, width=22).pack(fill=tk.X, pady=(4, 0))
        tk.Button(ctrl, text="Save scenario", command=self.save_scenario, width=22).pack(fill=tk.X, pady=(2, 0))

        tk.Label(ctrl, text="Robot shape", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        # Use SE(2) footprint planning for non-circular shapes; names match create_standard_objects().
        self._robot_shape_options = robot_shape_options_from_create_standard_objects()
        self.cmb_robot_shape = ttk.Combobox(
            ctrl,
            textvariable=self.var_robot_shape,
            state="readonly",
            values=self._robot_shape_options,
            width=22,
        )
        self.cmb_robot_shape.pack(fill=tk.X, pady=(2, 0))
        self.cmb_robot_shape.bind("<<ComboboxSelected>>", self._on_robot_shape_changed)

        tk.Label(ctrl, text="Edit mode", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.var_mode = tk.StringVar(value="rect")
        mode_row = tk.Frame(ctrl)
        mode_row.pack(fill=tk.X, pady=(2, 0))
        mode_left = tk.Frame(mode_row)
        mode_left.pack(side=tk.LEFT)
        for m in ("rect", "line", "erase"):
            tk.Radiobutton(
                mode_left,
                text=m,
                variable=self.var_mode,
                value=m,
                command=self._on_mode_changed,
            ).pack(side=tk.LEFT, padx=(0, 8))

        thick_f = tk.Frame(mode_row)
        thick_f.pack(side=tk.RIGHT)
        tk.Label(thick_f, text="line thick").pack(side=tk.LEFT, padx=(8, 4))
        self.ent_line_th = tk.Entry(thick_f, width=8)
        self.ent_line_th.insert(0, str(self.line_thickness))
        self.ent_line_th.pack(side=tk.LEFT)

        # Settings buttons (left column)
        btnf = tk.Frame(ctrl)
        btnf.pack(fill=tk.X, pady=(10, 0))
        tk.Button(btnf, text="Apply line thick", command=self.apply_robot, width=22).pack(fill=tk.X, pady=2)
        tk.Button(btnf, text="Clear Obst", command=self.clear_obstacles, width=22).pack(fill=tk.X, pady=2)

    def _build_planner_panel(self, ctrl: tk.Frame):
        tk.Label(ctrl, text="Planner", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.var_robot = tk.StringVar(value="holonomic")
        tk.Radiobutton(
            ctrl,
            text="holonomic (grid A*)",
            variable=self.var_robot,
            value="holonomic",
            command=self._on_robot_changed,
        ).pack(anchor=tk.W)
        tk.Radiobutton(
            ctrl,
            text="car (hybrid A*)",
            variable=self.var_robot,
            value="car",
            command=self._on_robot_changed,
        ).pack(anchor=tk.W)

        tk.Label(ctrl, text="Planner Method", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.var_planner = tk.StringVar(value="grid_astar")
        self.rb_grid = tk.Radiobutton(
            ctrl,
            text="grid_astar (baseline)",
            variable=self.var_planner,
            value="grid_astar",
            command=self._on_planner_changed,
        )
        self.rb_grid.pack(anchor=tk.W)

        self.rb_mod_grid = tk.Radiobutton(
            ctrl,
            text="mod_grid (disk safe+smooth)",
            variable=self.var_planner,
            value="mod_grid",
            command=self._on_planner_changed,
        )
        self.rb_mod_grid.pack(anchor=tk.W)

        self.rb_mod_grid_se = tk.Radiobutton(
            ctrl,
            text="mod_grid_SE (footprint SE2)",
            variable=self.var_planner,
            value="mod_grid_se",
            command=self._on_planner_changed,
        )
        self.rb_mod_grid_se.pack(anchor=tk.W)

        self.rb_hybrid = tk.Radiobutton(
            ctrl,
            text="hybrid_astar",
            variable=self.var_planner,
            value="hybrid_astar",
            command=self._on_planner_changed,
        )
        self.rb_hybrid.pack(anchor=tk.W)

        tk.Label(ctrl, text="safety_margin", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.ent_safety_margin = self._row_entry(ctrl, "margin [m]", str(self.safety_margin))

        tk.Label(ctrl, text="Yaw Filling", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.rb_yaw_none = tk.Radiobutton(
            ctrl,
            text="none",
            variable=self.var_yaw_fill,
            value="none",
            command=self._on_planner_changed,
        )
        self.rb_yaw_none.pack(anchor=tk.W)
        self.rb_yaw_linear = tk.Radiobutton(
            ctrl,
            text="Linear Interpolate",
            variable=self.var_yaw_fill,
            value="linear",
            command=self._on_planner_changed,
        )
        self.rb_yaw_linear.pack(anchor=tk.W)
        self.rb_yaw_df = tk.Radiobutton(
            ctrl,
            text="Differential Flatness",
            variable=self.var_yaw_fill,
            value="differential_flatness",
            command=self._on_planner_changed,
        )
        self.rb_yaw_df.pack(anchor=tk.W)

        # mod_grid stop phase selection
        tk.Label(ctrl, text="mod_grid stop phase", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.rb_mod_phase1 = tk.Radiobutton(
            ctrl,
            text="1 (augmented A*)",
            variable=self.var_mod_grid_phase,
            value=1,
            command=self._on_planner_changed,
        )
        self.rb_mod_phase2 = tk.Radiobutton(
            ctrl,
            text="2 (CHOMP final + shortcut)",
            variable=self.var_mod_grid_phase,
            value=2,
            command=self._on_planner_changed,
        )
        self.rb_mod_phase3 = tk.Radiobutton(
            ctrl,
            text="3 (SE(2) primitive compression)",
            variable=self.var_mod_grid_phase,
            value=3,
            command=self._on_planner_changed,
        )
        self.rb_mod_phase1.pack(anchor=tk.W)
        self.rb_mod_phase2.pack(anchor=tk.W)
        self.rb_mod_phase3.pack(anchor=tk.W)

        self._update_planner_options()

        tk.Label(ctrl, text="Planner actions", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(12, 0))

        btnf = tk.Frame(ctrl)
        btnf.pack(fill=tk.X, pady=(8, 0))

        def _btn(text: str, cmd):
            tk.Button(btnf, text=text, command=cmd, width=22).pack(fill=tk.X, pady=2)

        _btn("Plan / Replan", self.replan)
        _btn("Reset View", self.reset_view)

        status_wrap = tk.Frame(ctrl, width=280, height=120)
        status_wrap.pack(fill=tk.X, pady=(10, 0))
        status_wrap.pack_propagate(False)
        self.txt_status = tk.Text(status_wrap, width=38, height=7, wrap=tk.WORD, state=tk.DISABLED)
        self.txt_status.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll = tk.Scrollbar(status_wrap, command=self.txt_status.yview)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_status.configure(yscrollcommand=status_scroll.set)
        self._status_last_logged: Optional[str] = None

    def _on_mode_changed(self):
        if self._suppress_mode_cb:
            return

        old_mode = self.mode
        new_mode = self.var_mode.get()
        if old_mode == new_mode:
            return

        # Switching between rect <-> line: confirm, then remove obstacles of the *current* type only.
        if old_mode in ("rect", "line") and new_mode in ("rect", "line") and old_mode != new_mode:
            kind = "rectangle" if old_mode == "rect" else "line / polyline"
            if not messagebox.askyesno(
                "Change obstacle mode",
                f"You are leaving '{old_mode}' mode for '{new_mode}'.\n\n"
                f"This will delete all current {kind} obstacles.\n\n"
                "Continue?",
                parent=self.root,
            ):
                self._suppress_mode_cb = True
                self.var_mode.set(old_mode)
                self._suppress_mode_cb = False
                return

            if old_mode == "rect":
                self.obstacles.rects.clear()
                self.obstacles.rect_count = 0
                self.drag_start = None
            else:
                self.obstacles.lines.clear()
                self.obstacles.line_count = 0
                self.current_line = []

            self.status_text = f"Cleared all {kind}s; mode → {new_mode}"
            self.mode = new_mode
            self.render()
            return

        # erase involved, or same family: no confirmation
        self.mode = new_mode
        self.status_text = f"Mode: {self.mode}"
        self.render()

    def _on_robot_changed(self):
        self.robot_state.robot_type = self.var_robot.get()
        self._update_planner_options()
        self.status_text = f"Robot: {self.robot_state.robot_type}"
        self.render()

    def _on_planner_changed(self):
        cur = self.var_planner.get()
        if cur == "mod_grid_se" and getattr(self, "_last_planner_choice", None) != "mod_grid_se":
            self.var_mod_grid_phase.set(1)
        self._last_planner_choice = cur
        self.status_text = f"Planner: {cur}"
        # Recompute which controls should be enabled (e.g., mod_grid stop phase).
        self._update_planner_options()
        self.render()

    def _on_robot_shape_changed(self, _event=None):
        self.status_text = f"Robot shape: {self.var_robot_shape.get()}"
        self._update_planner_options()
        self.render()

    def _update_planner_options(self):
        # Holonomic: enable both grid planners, disable hybrid.
        if self.robot_state.robot_type == "holonomic":
            self.rb_grid.config(state=tk.NORMAL)
            self.rb_mod_grid.config(state=tk.NORMAL)
            self.rb_mod_grid_se.config(state=tk.NORMAL)
            self.rb_hybrid.config(state=tk.DISABLED)
            if self.var_planner.get() == "hybrid_astar":
                self.var_planner.set("grid_astar")

            # Enable mod_grid phase controls only when mod_grid is selected.
            if self.var_planner.get() in ("mod_grid", "mod_grid_se"):
                use_se = self.var_planner.get() == "mod_grid_se"
                # mod_grid: phases 1/2/3
                # mod_grid_SE: phases 1/3 only (phase2 disabled)
                self.rb_mod_phase1.config(state=tk.NORMAL)
                if use_se:
                    if int(self.var_mod_grid_phase.get()) == 2:
                        self.var_mod_grid_phase.set(1)
                    self.rb_mod_phase2.config(state=tk.DISABLED)
                    self.rb_mod_phase3.config(state=tk.NORMAL)
                else:
                    for rb in (self.rb_mod_phase2, self.rb_mod_phase3):
                        rb.config(state=tk.NORMAL)
            else:
                for rb in (self.rb_mod_phase1, self.rb_mod_phase2, self.rb_mod_phase3):
                    rb.config(state=tk.DISABLED)
            # Yaw-filling is a post-process for mod_grid(disk) only.
            yaw_state = tk.NORMAL if self.var_planner.get() == "mod_grid" else tk.DISABLED
            for rb in (self.rb_yaw_none, self.rb_yaw_linear, self.rb_yaw_df):
                rb.config(state=yaw_state)
        # Non-holonomic(car): disable all grid planners, enable hybrid only.
        else:
            self.rb_grid.config(state=tk.DISABLED)
            self.rb_mod_grid.config(state=tk.DISABLED)
            self.rb_mod_grid_se.config(state=tk.DISABLED)
            self.rb_hybrid.config(state=tk.NORMAL)
            if self.var_planner.get() in ("grid_astar", "mod_grid", "mod_grid_se"):
                self.var_planner.set("hybrid_astar")

            for rb in (self.rb_mod_phase1, self.rb_mod_phase2, self.rb_mod_phase3):
                rb.config(state=tk.DISABLED)
            for rb in (self.rb_yaw_none, self.rb_yaw_linear, self.rb_yaw_df):
                rb.config(state=tk.DISABLED)

    @staticmethod
    def _disk_radius_for_path_viz(rr: float, reso: float) -> float:
        """
        Disk radius used only for drawing along the path. The planner uses `rr` in meters
        against a discrete obstacle point cloud / grid; a full continuous disk often looks
        slightly larger than what the discrete model guarantees (appears to graze obstacles).
        """
        return max(float(rr) - 0.5 * float(reso), 1e-6)

    def _clear_path_footprint_viz(self) -> None:
        self._path_footprint_mode = None
        self._disk_radius_for_viz = None
        self._path_pyaw = None
        self._robot_vertices_local = None

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float(math.atan2(math.sin(a), math.cos(a)))

    def _fill_yaw_linear(self, n: int, syaw: float, gyaw: float) -> List[float]:
        if n <= 1:
            return [float(syaw)]
        dyaw = self._wrap_angle(float(gyaw) - float(syaw))
        return [float(syaw) + dyaw * (i / float(n - 1)) for i in range(n)]

    def _fill_yaw_df_from_phase3_primitives(
        self, prims: List[Tuple[str, dict]], syaw: float, n_expected: int
    ) -> List[float]:
        """
        Recover yaw by anchoring at start yaw and integrating primitive turn.
        - Straight primitive: keep yaw constant.
        - Arc primitive: yaw increments with arc sweep.
        """
        out = [float(syaw)]
        cur = float(syaw)
        for typ, p in prims:
            if typ == "S":
                out.append(cur)
                continue
            sweep = float(p["sweep"])
            n_arc = max(5, int(abs(sweep) * mod_grid_astar._ARC_POINTS_PER_RAD) + 1)
            for kk in range(1, n_arc + 1):
                t = kk / float(n_arc)
                out.append(cur + t * sweep)
            cur = out[-1]
        if len(out) > n_expected:
            out = out[:n_expected]
        elif len(out) < n_expected:
            out.extend([out[-1]] * (n_expected - len(out)))
        return [self._wrap_angle(y) for y in out]

    def _append_final_self_rotation(
        self, px: List[float], py: List[float], pyaw: List[float], goal_yaw: float, n_steps: int = 8
    ) -> Tuple[List[float], List[float], List[float]]:
        if not px or not py or not pyaw:
            return px, py, pyaw
        d = self._wrap_angle(float(goal_yaw) - float(pyaw[-1]))
        if abs(d) < 1e-3:
            return px, py, pyaw
        xg, yg = float(px[-1]), float(py[-1])
        outx = list(px)
        outy = list(py)
        outyaw = list(pyaw)
        for i in range(1, max(2, int(n_steps)) + 1):
            t = i / float(max(2, int(n_steps)))
            outx.append(xg)
            outy.append(yg)
            outyaw.append(self._wrap_angle(float(pyaw[-1]) + t * d))
        return outx, outy, outyaw

    def _disk_planner_rr(self, reso: float) -> float:
        """
        Robot radius [m] for disk-based planners (grid_astar, mod_grid with circle shape).
        True disk uses max(width,length)/2; any other shape uses the circumradius of the
        scaled footprint from mod_grid_SE (same as SE footprint sizing).
        """
        w = self.robot_state.robot_width
        l = self.robot_state.robot_length
        if self.var_robot_shape.get() == "circle":
            return max(w, l) / 2.0
        robot_spec = {
            "shape_name": self.var_robot_shape.get(),
            "width": w,
            "length": l,
        }
        verts = mod_grid_SE_astar._extract_robot_footprint_vertices_local(robot_spec, reso=reso)
        return max(math.hypot(vx, vy) for vx, vy in verts)

    @staticmethod
    def _parse_float_from_entry(entry: tk.Entry, fallback: float) -> float:
        try:
            return float(entry.get().strip())
        except ValueError:
            return fallback

    def apply_pose(self):
        self.start = Pose(
            self._parse_float_from_entry(self.ent_sx, self.start.x),
            self._parse_float_from_entry(self.ent_sy, self.start.y),
            self._parse_float_from_entry(self.ent_syaw, self.start.yaw_deg),
        )
        self.goal = Pose(
            self._parse_float_from_entry(self.ent_gx, self.goal.x),
            self._parse_float_from_entry(self.ent_gy, self.goal.y),
            self._parse_float_from_entry(self.ent_gyaw, self.goal.yaw_deg),
        )
        self.status_text = "Pose updated"
        self.render()

    def apply_map(self):
        self.map_state.map_width = max(2.0, self._parse_float_from_entry(self.ent_map_w, self.map_state.map_width))
        self.map_state.map_height = max(2.0, self._parse_float_from_entry(self.ent_map_h, self.map_state.map_height))
        self.map_state.resolution = max(0.2, self._parse_float_from_entry(self.ent_reso, self.map_state.resolution))
        self.reset_view()
        self.status_text = "Map updated"
        self.render()

    def apply_robot(self):
        self.line_thickness = max(0.1, self._parse_float_from_entry(self.ent_line_th, self.line_thickness))
        self.status_text = "Line thickness updated"
        self.render()

    def reset_view(self):
        self.ax.set_xlim(0.0, self.map_state.map_width)
        self.ax.set_ylim(0.0, self.map_state.map_height)
        self.canvas.draw_idle()

    def _connect_canvas_events(self):
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.85 if event.button == "up" else 1.15
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        nx0 = event.xdata - (event.xdata - x0) * scale
        nx1 = event.xdata + (x1 - event.xdata) * scale
        ny0 = event.ydata - (event.ydata - y0) * scale
        ny1 = event.ydata + (y1 - event.ydata) * scale
        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ny0, ny1)
        self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 2:
            self.drag_start = (event.x, event.y)
            return
        if event.button != 1:
            return
        if self.mode == "rect":
            self.drag_start = (event.xdata, event.ydata)
        elif self.mode == "line":
            self.current_line = [(event.xdata, event.ydata)]
        elif self.mode == "erase":
            self.erase_nearest(event.xdata, event.ydata)

    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return
        if self.drag_start is not None and event.button == 2:
            dx_pix = event.x - self.drag_start[0]
            dy_pix = event.y - self.drag_start[1]
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            sx = (x1 - x0) / max(self.ax.bbox.width, 1.0)
            sy = (y1 - y0) / max(self.ax.bbox.height, 1.0)
            self.ax.set_xlim(x0 - dx_pix * sx, x1 - dx_pix * sx)
            self.ax.set_ylim(y0 - dy_pix * sy, y1 - dy_pix * sy)
            self.drag_start = (event.x, event.y)
            self.canvas.draw_idle()
            return
        if self.mode == "line" and self.current_line and event.xdata is not None and event.ydata is not None:
            self.current_line.append((event.xdata, event.ydata))
            self.render()

    def on_mouse_release(self, event):
        if event.button == 2:
            self.drag_start = None
            return
        if event.inaxes != self.ax or event.button != 1:
            return
        if self.mode == "rect" and self.drag_start is not None and event.xdata is not None and event.ydata is not None:
            x0, y0 = self.drag_start
            x1, y1 = event.xdata, event.ydata
            rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            if rect[2] > 1e-3 and rect[3] > 1e-3:
                rid = self.obstacles.next_rect_id()
                self.obstacles.rects[rid] = rect
                self.status_text = f"Added {rid}"
            self.drag_start = None
            self.render()
        elif self.mode == "line" and self.current_line:
            if len(self.current_line) >= 2:
                lid = self.obstacles.next_line_id()
                self.obstacles.lines[lid] = self.current_line[:]
                self.status_text = f"Added {lid}"
            self.current_line = []
            self.render()

    def erase_nearest(self, x: float, y: float):
        best_id = None
        best_dist = float("inf")
        best_kind = ""

        for rid, rect in self.obstacles.rects.items():
            rx, ry, rw, rh = rect
            cx, cy = rx + rw / 2.0, ry + rh / 2.0
            d = math.hypot(x - cx, y - cy)
            if d < best_dist:
                best_dist = d
                best_id = rid
                best_kind = "rect"

        for lid, pts in self.obstacles.lines.items():
            if not pts:
                continue
            px, py = min(pts, key=lambda p: math.hypot(x - p[0], y - p[1]))
            d = math.hypot(x - px, y - py)
            if d < best_dist:
                best_dist = d
                best_id = lid
                best_kind = "line"

        erase_radius = 2.0 * self.map_state.resolution
        if best_id is not None and best_dist <= erase_radius:
            if best_kind == "rect":
                self.obstacles.rects.pop(best_id, None)
            else:
                self.obstacles.lines.pop(best_id, None)
            self.status_text = f"Removed {best_id}"
            self.render()

    def _boundary_points(self) -> Tuple[List[float], List[float]]:
        ox, oy = [], []
        w = int(self.map_state.map_width / self.map_state.resolution)
        h = int(self.map_state.map_height / self.map_state.resolution)
        r = self.map_state.resolution
        for i in range(w + 1):
            x = i * r
            ox += [x, x]
            oy += [0.0, h * r]
        for j in range(h + 1):
            y = j * r
            ox += [0.0, w * r]
            oy += [y, y]
        return ox, oy

    def _obstacle_points(self) -> Tuple[List[float], List[float]]:
        ox, oy = self._boundary_points()
        r = self.map_state.resolution

        for rect in self.obstacles.rects.values():
            x, y, w, h = rect
            x0 = max(0.0, x)
            y0 = max(0.0, y)
            x1 = min(self.map_state.map_width, x + w)
            y1 = min(self.map_state.map_height, y + h)
            xi = np.arange(x0, x1 + 1e-6, r)
            yi = np.arange(y0, y1 + 1e-6, r)
            for xx in xi:
                for yy in yi:
                    ox.append(float(xx))
                    oy.append(float(yy))

        thick = max(0.2, self.line_thickness)
        samples = max(2, int(thick / r))
        for pts in self.obstacles.lines.values():
            for i in range(len(pts) - 1):
                (x0, y0), (x1, y1) = pts[i], pts[i + 1]
                seg_len = max(math.hypot(x1 - x0, y1 - y0), 1e-6)
                n = max(2, int(seg_len / r) * 2)
                for t in np.linspace(0.0, 1.0, n):
                    cx = x0 + t * (x1 - x0)
                    cy = y0 + t * (y1 - y0)
                    for dx in np.linspace(-thick / 2.0, thick / 2.0, samples):
                        for dy in np.linspace(-thick / 2.0, thick / 2.0, samples):
                            ox.append(float(cx + dx))
                            oy.append(float(cy + dy))
        return ox, oy

    def _point_in_collision(self, x: float, y: float) -> bool:
        if x <= 0.0 or y <= 0.0 or x >= self.map_state.map_width or y >= self.map_state.map_height:
            return True
        for rx, ry, rw, rh in self.obstacles.rects.values():
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return True
        thick = self.line_thickness / 2.0
        for pts in self.obstacles.lines.values():
            for i in range(len(pts) - 1):
                if self._distance_to_segment((x, y), pts[i], pts[i + 1]) <= thick:
                    return True
        return False

    @staticmethod
    def _distance_to_segment(p, a, b) -> float:
        px, py = p
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        cx = ax + t * dx
        cy = ay + t * dy
        return math.hypot(px - cx, py - cy)

    def replan(self):
        self.apply_pose()
        self.apply_map()
        self.apply_robot()
        self.safety_margin = max(0.0, self._parse_float_from_entry(self.ent_safety_margin, self.safety_margin))

        if self._point_in_collision(self.start.x, self.start.y):
            self.status_text = "Start in collision"
            self.path_data = None
            self._clear_path_footprint_viz()
            self.render()
            return
        if self._point_in_collision(self.goal.x, self.goal.y):
            self.status_text = "Goal in collision"
            self.path_data = None
            self._clear_path_footprint_viz()
            self.render()
            return

        ox, oy = self._obstacle_points()
        reso = self.map_state.resolution

        try:
            self._clear_path_footprint_viz()
            if self.robot_state.robot_type == "holonomic":
                rr = self._disk_planner_rr(reso)
                planner = self.var_planner.get()
                if planner == "grid_astar":
                    self._path_footprint_mode = "disk"
                    self._disk_radius_for_viz = self._disk_radius_for_path_viz(rr, reso)
                    px, py = grid_astar.astar_planning(
                        self.start.x, self.start.y, self.goal.x, self.goal.y, ox, oy, reso, rr
                    )
                elif planner == "mod_grid":
                    # Disk-based mod_grid (uses rr, which can be circumradius for non-circle shapes).
                    # Timed sub-steps mirror HybridAstarPlanner.mod_grid.astar_planning (phases 1/2/3).
                    shape = self.var_robot_shape.get()
                    yaw_fill = self.var_yaw_fill.get()
                    phase = int(self.var_mod_grid_phase.get())
                    sm = float(self.safety_margin)
                    prims = None
                    want_prims = yaw_fill == "differential_flatness" and phase == 3 and shape != "circle"

                    t0 = time.perf_counter()
                    px, py = mod_grid_astar.phase1_augmented_astar(
                        self.start.x,
                        self.start.y,
                        self.goal.x,
                        self.goal.y,
                        ox,
                        oy,
                        reso,
                        rr,
                        safety_margin=sm,
                    )
                    t_phase1_end = time.perf_counter()

                    if len(px) < 2:
                        self._append_status_log(
                            f"[timing] mod_grid phase1={1000.0 * (t_phase1_end - t0):.1f}ms (no path)"
                        )
                    elif phase == 1:
                        self._append_status_log(
                            f"[timing] mod_grid phase1 total={1000.0 * (t_phase1_end - t0):.1f}ms"
                        )
                    elif phase == 2:
                        safe_rr = float(rr) + sm + max(
                            mod_grid_astar._INFLATION_RESO_FACTOR * float(reso), 0.15
                        )
                        eff_rr = float(rr) + sm
                        p1x, p1y = px, py
                        t_ch0 = time.perf_counter()
                        cpx, cpy = mod_grid_astar.phase2_chomp(
                            px, py, ox, oy, reso, rr, safety_margin=sm
                        )
                        t_ch1 = time.perf_counter()
                        if mod_grid_astar._path_min_segment_clearance(cpx, cpy, ox, oy) <= eff_rr:
                            cpx, cpy = p1x, p1y
                        t_sc0 = time.perf_counter()
                        spx, spy = mod_grid_astar._shortcut_path(cpx, cpy, ox, oy, safe_rr)
                        t_sc1 = time.perf_counter()
                        if mod_grid_astar._path_min_segment_clearance(spx, spy, ox, oy) <= eff_rr:
                            px, py = cpx, cpy
                        else:
                            px, py = spx, spy
                        self._append_status_log(
                            f"[timing] mod_grid phase1={1000.0 * (t_phase1_end - t0):.1f}ms | "
                            f"chomp={1000.0 * (t_ch1 - t_ch0):.1f}ms | "
                            f"shortcut={1000.0 * (t_sc1 - t_sc0):.1f}ms | "
                            f"total={1000.0 * (t_sc1 - t0):.1f}ms"
                        )
                    else:
                        # stop_phase == 3: shortcut then phase3_min_segments (same as mod_grid.astar_planning).
                        safe_rr = float(rr) + sm + 0.05
                        eff_rr = float(rr) + sm
                        t_sc0 = time.perf_counter()
                        spx, spy = mod_grid_astar._shortcut_path(px, py, ox, oy, safe_rr)
                        t_sc1 = time.perf_counter()
                        t_p3_0 = time.perf_counter()
                        if want_prims:
                            px, py, prims = mod_grid_astar.phase3_min_segments(
                                spx, spy, ox, oy, eff_rr, return_primitives=True
                            )
                            if mod_grid_astar._path_min_segment_clearance(px, py, ox, oy) <= eff_rr:
                                px, py = list(spx), list(spy)
                                prims = mod_grid_astar._polyline_straight_primitives(spx, spy)
                        else:
                            px, py = mod_grid_astar.phase3_min_segments(spx, spy, ox, oy, eff_rr)
                            if mod_grid_astar._path_min_segment_clearance(px, py, ox, oy) <= eff_rr:
                                px, py = spx, spy
                        t_p3_1 = time.perf_counter()
                        self._append_status_log(
                            f"[timing] mod_grid phase1={1000.0 * (t_phase1_end - t0):.1f}ms | "
                            f"shortcut={1000.0 * (t_sc1 - t_sc0):.1f}ms | "
                            f"phase3={1000.0 * (t_p3_1 - t_p3_0):.1f}ms | "
                            f"total={1000.0 * (t_p3_1 - t0):.1f}ms"
                        )

                    # Yaw filling is display-only for disk planner; planning still uses circumscribed disk.
                    if shape != "circle" and yaw_fill != "none":
                        robot_spec = {
                            "shape_name": shape,
                            "width": self.robot_state.robot_width,
                            "length": self.robot_state.robot_length,
                        }
                        robot_vertices_local = mod_grid_SE_astar._extract_robot_footprint_vertices_local(
                            robot_spec, reso=reso
                        )
                        syaw = math.radians(self.start.yaw_deg)
                        gyaw = math.radians(self.goal.yaw_deg)
                        if yaw_fill == "linear":
                            pyaw = self._fill_yaw_linear(len(px), syaw, gyaw)
                        else:
                            if phase != 3 or prims is None:
                                # DF fill relies on phase-3 primitives from mod_grid.
                                self._path_footprint_mode = "disk"
                                self._disk_radius_for_viz = self._disk_radius_for_path_viz(rr, reso)
                                self.status_text = "DF yaw fill needs mod_grid phase 3; showing disk footprint"
                                pyaw = None
                            else:
                                pyaw = self._fill_yaw_df_from_phase3_primitives(prims, syaw, len(px))
                                px, py, pyaw = self._append_final_self_rotation(px, py, pyaw, gyaw)

                        if pyaw is not None:
                            self._path_footprint_mode = "polygon"
                            self._robot_vertices_local = list(robot_vertices_local)
                            self._path_pyaw = list(pyaw)
                        else:
                            self._path_footprint_mode = "disk"
                            self._disk_radius_for_viz = self._disk_radius_for_path_viz(rr, reso)
                    else:
                        self._path_footprint_mode = "disk"
                        self._disk_radius_for_viz = self._disk_radius_for_path_viz(rr, reso)
                elif planner == "mod_grid_se":
                    # True footprint SE(2) planning (Phase 1 or 3).
                    syaw = math.radians(self.start.yaw_deg)
                    gyaw = math.radians(self.goal.yaw_deg)
                    phase = int(self.var_mod_grid_phase.get())
                    robot_spec = {
                        "shape_name": self.var_robot_shape.get(),
                        "width": self.robot_state.robot_width,
                        "length": self.robot_state.robot_length,
                    }
                    robot_vertices_local = mod_grid_SE_astar._extract_robot_footprint_vertices_local(robot_spec, reso=reso)

                    obstacle_rects = list(self.obstacles.rects.values())
                    map_bounds = (0.0, 0.0, float(self.map_state.map_width), float(self.map_state.map_height))

                    # Timing instrumentation: report per-phase timing in the text log.
                    if phase == 3:
                        t0 = time.perf_counter()
                        px1, py1, pyaw1 = mod_grid_SE_astar.phase1_augmented_astar_se2(
                            sx=self.start.x,
                            sy=self.start.y,
                            syaw_rad=syaw,
                            gx=self.goal.x,
                            gy=self.goal.y,
                            gyaw_rad=gyaw,
                            ox=ox,
                            oy=oy,
                            reso=reso,
                            robot_vertices_local=robot_vertices_local,
                            safety_margin=float(self.safety_margin),
                            obstacle_rects=obstacle_rects,
                            map_bounds=map_bounds,
                        )
                        t1 = time.perf_counter()

                        hard_pad = (
                            max(
                                mod_grid_SE_astar._SHAPE_MIN_HARD_CLEARANCE_PAD,
                                mod_grid_SE_astar._SHAPE_HARD_CLEARANCE_PAD_FACTOR * float(reso),
                            )
                            + float(self.safety_margin)
                        )
                        px, py, pyaw = mod_grid_SE_astar.phase3_min_segments(
                            px1,
                            py1,
                            pyaw1,
                            ox=ox,
                            oy=oy,
                            robot_vertices_local=robot_vertices_local,
                            reso=reso,
                            clearance=float(hard_pad),
                            obstacle_rects=obstacle_rects,
                            map_bounds=map_bounds,
                        )
                        t2 = time.perf_counter()
                        self._append_status_log(
                            f"[timing] mod_grid_SE phase1={1000.0*(t1-t0):.1f}ms | "
                            f"phase3={1000.0*(t2-t1):.1f}ms | total={1000.0*(t2-t0):.1f}ms"
                        )
                    else:
                        t0 = time.perf_counter()
                        px, py, pyaw = mod_grid_SE_astar.astar_planning(
                            sx=self.start.x,
                            sy=self.start.y,
                            syaw_rad=syaw,
                            gx=self.goal.x,
                            gy=self.goal.y,
                            gyaw_rad=gyaw,
                            ox=ox,
                            oy=oy,
                            reso=reso,
                            robot_vertices_local=robot_vertices_local,
                            stop_phase=phase,
                            safety_margin=float(self.safety_margin),
                            obstacle_rects=obstacle_rects,
                            map_bounds=map_bounds,
                        )
                        t1 = time.perf_counter()
                        self._append_status_log(f"[timing] mod_grid_SE phase{phase} total={1000.0*(t1-t0):.1f}ms")

                    self._path_footprint_mode = "polygon"
                    self._robot_vertices_local = list(robot_vertices_local)
                    self._path_pyaw = list(pyaw)
                else:
                    raise ValueError(f"Unknown planner: {planner}")
            else:
                hybrid_astar = _import_hybrid_astar()
                syaw = math.radians(self.start.yaw_deg)
                gyaw = math.radians(self.goal.yaw_deg)
                path = hybrid_astar.hybrid_astar_planning(
                    self.start.x,
                    self.start.y,
                    syaw,
                    self.goal.x,
                    self.goal.y,
                    gyaw,
                    ox,
                    oy,
                    hybrid_astar.C.XY_RESO,
                    hybrid_astar.C.YAW_RESO,
                )
                if path is None:
                    raise RuntimeError("Hybrid A* failed")
                px, py = list(path.x), list(path.y)
            self.path_data = (px, py)
            phase_suffix = ""
            if self.robot_state.robot_type == "holonomic" and self.var_planner.get() in ("mod_grid", "mod_grid_se"):
                if self.var_planner.get() == "mod_grid":
                    phase_suffix = (
                        f" | mod_grid(disk) mode {int(self.var_mod_grid_phase.get())}"
                        f" yaw_fill={self.var_yaw_fill.get()}"
                    )
                else:
                    phase_suffix = f" | mod_grid_SE shape {self.var_robot_shape.get()} mode {int(self.var_mod_grid_phase.get())}"
            self.status_text = f"Path found: {len(px)} pts{phase_suffix}"
        except Exception as exc:
            self.path_data = None
            self._clear_path_footprint_viz()
            self.status_text = f"Replan failed: {exc}"
        self.render()

    def clear_obstacles(self):
        self.obstacles = ObstacleState()
        self.path_data = None
        self._clear_path_footprint_viz()
        self.status_text = "Cleared obstacles"
        self.render()

    def _refresh_scenario_combo(self) -> None:
        names = sorted(
            p.name
            for p in THIS_DIR.glob("*.json")
            if p.name != "obstacles_export.json"
        )
        self.cmb_scenarios["values"] = names
        cur = self.var_scenario_file.get()
        if names:
            if cur not in names:
                self.var_scenario_file.set(names[0])
        else:
            self.var_scenario_file.set("")

    def _normalize_obstacle_ids_inplace(self) -> None:
        """Rewrite obstacle keys as RECT_### / LINE_### (sorted by previous key)."""
        rects = {
            f"RECT_{i:03d}": tuple(v)
            for i, (_, v) in enumerate(sorted(self.obstacles.rects.items(), key=lambda kv: kv[0]), start=1)
        }
        lines = {
            f"LINE_{j:03d}": [tuple(p) for p in pts]
            for j, (_, pts) in enumerate(sorted(self.obstacles.lines.items(), key=lambda kv: kv[0]), start=1)
        }
        self.obstacles = ObstacleState(
            rects=rects,
            lines=lines,
            rect_count=len(rects),
            line_count=len(lines),
        )

    def _scenario_dict(self) -> dict:
        return {
            "version": 1,
            "map": {
                "width": self.map_state.map_width,
                "height": self.map_state.map_height,
                "resolution": self.map_state.resolution,
            },
            "robot": {
                "type": self.robot_state.robot_type,
                "width": self.robot_state.robot_width,
                "length": self.robot_state.robot_length,
                "shape_name": self.var_robot_shape.get(),
                "safety_margin": float(self.safety_margin),
            },
            "pose": {
                "start": [self.start.x, self.start.y, self.start.yaw_deg],
                "goal": [self.goal.x, self.goal.y, self.goal.yaw_deg],
            },
            "draw": {"line_thickness": self.line_thickness},
            "planner": {
                "yaw_fill_mode": self.var_yaw_fill.get(),
            },
            "obstacles": {
                "rects": self.obstacles.rects,
                "lines": self.obstacles.lines,
                "rect_count": self.obstacles.rect_count,
                "line_count": self.obstacles.line_count,
            },
        }

    def save_scenario(self):
        # Simulation depends on filenames reflecting the obstacle draw mode.
        # Expected patterns:
        #   rectObs_scenario_*.json
        #   lineObs_scenario_*.json
        mode_prefix = "rectObs" if self.mode == "rect" else "lineObs"
        raw = simpledialog.askstring(
            "Save scenario",
            f"Scenario name (saved as {mode_prefix}_scenario_<name>.json):",
            parent=self.root,
            initialvalue="default",
        )
        if raw is None:
            return
        stem = raw.strip()
        if not stem:
            messagebox.showerror("Save", "Name cannot be empty.", parent=self.root)
            return
        safe = re.sub(r"[^\w\-.]+", "_", stem).strip("._")
        if not safe or safe == ".":
            messagebox.showerror("Save", "Invalid name after sanitization.", parent=self.root)
            return
        # Strip accidental .json / known prefixes from user input
        if safe.lower().endswith(".json"):
            safe = safe[:-5]
        safe_lower = safe.lower()
        rect_prefix = "rectobs_scenario_"
        line_prefix = "lineobs_scenario_"
        if safe_lower.startswith(rect_prefix):
            safe = safe[len(rect_prefix) :]
        elif safe_lower.startswith(line_prefix):
            safe = safe[len(line_prefix) :]
        elif safe_lower.startswith("scenario_"):
            safe = safe[len("scenario_") :]
        elif safe_lower.startswith("rectobs_"):
            safe = safe[len("rectobs_") :]
        elif safe_lower.startswith("lineobs_"):
            safe = safe[len("lineobs_") :]

        fname = f"{mode_prefix}_scenario_{safe}.json"
        out = THIS_DIR / fname
        if out.exists():
            if not messagebox.askyesno("Save", f"'{fname}' already exists. Overwrite?", parent=self.root):
                return

        self._normalize_obstacle_ids_inplace()
        payload = self._scenario_dict()
        payload["metadata"] = {
            "filename": fname,
            "obstacle_id_prefixes": {"rectangles": "RECT_", "polylines": "LINE_"},
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.status_text = f"Saved: {fname}"
        self._refresh_scenario_combo()
        self.var_scenario_file.set(fname)
        self.render()

    def load_scenario(self):
        name = (self.var_scenario_file.get() or "").strip()
        candidates = [p.name for p in THIS_DIR.glob("*.json") if p.name != "obstacles_export.json"]
        if not name:
            if not candidates:
                messagebox.showwarning("Load", "No scenario JSON files in this folder.", parent=self.root)
            else:
                messagebox.showwarning("Load", "Choose a file from the dropdown first.", parent=self.root)
            self.render()
            return
        src = THIS_DIR / name
        if not src.is_file():
            messagebox.showerror("Load", f"File not found: {name}", parent=self.root)
            self._refresh_scenario_combo()
            self.render()
            return
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)

        m = data.get("map", {})
        self.map_state = MapState(
            float(m.get("width", 60.0)), float(m.get("height", 40.0)), float(m.get("resolution", 1.0))
        )
        r = data.get("robot", {})
        self.robot_state = RobotState(
            str(r.get("type", "holonomic")), float(r.get("width", 2.0)), float(r.get("length", 3.0))
        )
        self.safety_margin = float(r.get("safety_margin", 0.0))
        shape_name = str(r.get("shape_name", "circle"))
        # Old GUI used "plus_shape"; library key is "plus".
        if shape_name == "plus_shape":
            shape_name = "plus"
        if hasattr(self, "_robot_shape_options") and shape_name not in self._robot_shape_options:
            shape_name = "circle"
        self.var_robot_shape.set(shape_name)
        p = data.get("pose", {})
        s = p.get("start", [5.0, 5.0, 0.0])
        g = p.get("goal", [50.0, 30.0, 0.0])
        self.start = Pose(float(s[0]), float(s[1]), float(s[2]))
        self.goal = Pose(float(g[0]), float(g[1]), float(g[2]))
        self.line_thickness = float(data.get("draw", {}).get("line_thickness", 5.0))
        yfill = str(data.get("planner", {}).get("yaw_fill_mode", "none"))
        if yfill not in ("none", "linear", "differential_flatness"):
            yfill = "none"
        self.var_yaw_fill.set(yfill)

        o = data.get("obstacles", {})
        self.obstacles = ObstacleState(
            rects={k: tuple(v) for k, v in o.get("rects", {}).items()},
            lines={k: [tuple(pt) for pt in v] for k, v in o.get("lines", {}).items()},
            rect_count=int(o.get("rect_count", len(o.get("rects", {})))),
            line_count=int(o.get("line_count", len(o.get("lines", {})))),
        )
        self._normalize_obstacle_ids_inplace()
        self.path_data = None
        self._clear_path_footprint_viz()
        self.var_robot.set(self.robot_state.robot_type)
        self._update_planner_options()
        self._sync_entries()
        self.reset_view()
        self.var_scenario_file.set(src.name)
        self.status_text = f"Loaded: {src.name}"
        self.render()

    def export_obstacles(self):
        out = THIS_DIR / "obstacles_export.json"
        payload = {"RECT": self.obstacles.rects, "LINE": self.obstacles.lines}
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.status_text = f"Exported: {out.name}"
        self.render()

    def _sync_entries(self):
        for ent, val in [
            (self.ent_sx, self.start.x),
            (self.ent_sy, self.start.y),
            (self.ent_syaw, self.start.yaw_deg),
            (self.ent_gx, self.goal.x),
            (self.ent_gy, self.goal.y),
            (self.ent_gyaw, self.goal.yaw_deg),
            (self.ent_map_w, self.map_state.map_width),
            (self.ent_map_h, self.map_state.map_height),
            (self.ent_reso, self.map_state.resolution),
            (self.ent_line_th, self.line_thickness),
            (self.ent_safety_margin, self.safety_margin),
        ]:
            ent.delete(0, tk.END)
            ent.insert(0, str(val))

    def _draw_pose(self, pose: Pose, color: str):
        self.ax.plot(pose.x, pose.y, "o", color=color, markersize=7)
        yaw = math.radians(pose.yaw_deg)
        arr = FancyArrowPatch(
            (pose.x, pose.y),
            (pose.x + 2.0 * math.cos(yaw), pose.y + 2.0 * math.sin(yaw)),
            mutation_scale=10,
            color=color,
            linewidth=2.0,
        )
        self.ax.add_patch(arr)

    def render(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        self.ax.set_title(f"Hybrid/A* Planning Workbench | {self.status_text}")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.2)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        frame = Rectangle((0.0, 0.0), self.map_state.map_width, self.map_state.map_height, fill=False, ec="black", lw=2.0)
        self.ax.add_patch(frame)

        for rid, (x, y, w, h) in self.obstacles.rects.items():
            p = Rectangle((x, y), w, h, facecolor="#777777", edgecolor="black", alpha=0.7)
            self.ax.add_patch(p)
            self.ax.text(x + w / 2.0, y + h / 2.0, rid, fontsize=7, color="white", ha="center", va="center")

        for lid, pts in self.obstacles.lines.items():
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                self.ax.plot(xs, ys, color="purple", linewidth=max(1.0, self.line_thickness))
                self.ax.text(xs[0], ys[0], lid, fontsize=7, color="purple")

        if len(self.current_line) >= 2:
            xs, ys = zip(*self.current_line)
            self.ax.plot(xs, ys, color="orange", linewidth=max(1.0, self.line_thickness), alpha=0.8)

        self._draw_pose(self.start, "green")
        self._draw_pose(self.goal, "red")

        if self.path_data is not None:
            px, py = self.path_data
            self.ax.plot(px, py, "-b", linewidth=2.0)

            n = len(px)
            max_samples = 40
            step = max(1, int(math.ceil(n / float(max_samples))))

            if self._path_footprint_mode == "polygon" and self._robot_vertices_local and self._path_pyaw:
                verts_loc = self._robot_vertices_local
                pyaw = self._path_pyaw
                for i in range(0, n, step):
                    if i >= len(pyaw):
                        break
                    yaw = float(pyaw[i])
                    c, s = math.cos(yaw), math.sin(yaw)
                    poly_xy = []
                    for lx, ly in verts_loc:
                        wx = c * lx - s * ly + px[i]
                        wy = s * lx + c * ly + py[i]
                        poly_xy.append((wx, wy))
                    self.ax.add_patch(
                        Polygon(
                            poly_xy,
                            closed=True,
                            fill=False,
                            edgecolor="cyan",
                            alpha=0.35,
                            linewidth=1.0,
                        )
                    )
            elif (
                self._path_footprint_mode == "disk"
                and self._disk_radius_for_viz is not None
                and self._disk_radius_for_viz > 0.0
            ):
                for i in range(0, n, step):
                    self.ax.add_patch(
                        Circle(
                            (px[i], py[i]),
                            radius=self._disk_radius_for_viz,
                            fill=False,
                            edgecolor="cyan",
                            alpha=0.22,
                            linewidth=1.0,
                        )
                    )

        self.fig.tight_layout()
        self.canvas.draw_idle()
        if hasattr(self, "txt_status") and self._status_last_logged != self.status_text:
            self._status_last_logged = self.status_text
            self.txt_status.configure(state=tk.NORMAL)
            if self.txt_status.index("end-1c") != "1.0":
                self.txt_status.insert(tk.END, "\n")
            self.txt_status.insert(tk.END, f"{self.status_text}")
            self.txt_status.see(tk.END)
            self.txt_status.configure(state=tk.DISABLED)

    def run(self):
        self.root.mainloop()


def main():
    PlannerWorkbench().run()


if __name__ == "__main__":
    main()
