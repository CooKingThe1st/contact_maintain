import json
import math
import re
import sys
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
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


from scenario_obstacles import (
    MIN_SAFETY_MARGIN,
    SWARM_PUSHER_ROBOT_DIAMETER_M,
    ObstacleRect,
    clamp_safety_margin,
    obstacle_points_for_disk_planner,
    obstacle_points_from_scenario,
    parse_rect_values,
    parse_scenario_rects,
    rect_values_for_se,
    swarm_pusher_min_safety_margin_m,
)
from scenario_planner_bridge import (
    build_planned_path_bundle,
    load_planned_path_bundle,
    validate_planned_path_sat,
    write_planned_path_export_pair,
)


OBJ_SHAPE_NAMES = [
    "right_triangle",
    "pi",
    "root",
    "rect",
    "hourglass",
    "meteor",
]


def robot_shape_options_from_create_standard_objects() -> List[str]:
    """OBJ mesh shapes used by holonomic magnum tests."""
    return list(OBJ_SHAPE_NAMES)


def obj_path_for_shape(shape_name: str) -> Optional[str]:
    """Package-relative OBJ path for a holonomic test shape."""
    if shape_name not in OBJ_SHAPE_NAMES:
        return None
    try:
        import rospkg

        pkg = Path(rospkg.RosPack().get_path("contact_maintain"))
        cand = pkg / "urdf" / f"{shape_name}.obj"
        if cand.is_file():
            return str(cand.resolve())
    except Exception:
        pass
    for parent in [THIS_DIR, *THIS_DIR.parents]:
        cand = parent / "urdf" / f"{shape_name}.obj"
        if cand.is_file():
            return str(cand.resolve())
    return None


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
    map_width: float = 14.0
    map_height: float = 14.0
    resolution: float = 0.2


@dataclass
class RobotState:
    robot_type: str = "holonomic"
    robot_width: float = 0.8
    robot_length: float = 1.0


@dataclass
class ObstacleState:
    rects: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
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
        self.start = Pose(3.0, 3.0, 0.0)
        self.goal = Pose(10.0, 10.0, 90.0)
        self.obstacles = ObstacleState()
        self.rect_angle_deg = 0.0

        self.mode = "rect"
        self.drag_start: Optional[Tuple[float, float]] = None
        self.current_line: List[Tuple[float, float]] = []
        self.path_data: Optional[Tuple[List[float], List[float]]] = None
        # Path footprint visualization (aligned with planner geometry).
        self._path_footprint_mode: Optional[str] = None  # "disk" | "polygon"
        self._disk_radius_for_viz: Optional[float] = None
        self._path_pyaw: Optional[List[float]] = None
        self._robot_vertices_local: Optional[List[Tuple[float, float]]] = None
        self._shape_preview_vertices_local: Optional[List[Tuple[float, float]]] = None
        # Created after tk.Tk() to avoid "Too early to create variable" errors.
        self.var_mod_grid_phase: Optional[tk.IntVar] = None
        self.status_text = "Ready"
        self.line_thickness = 5.0
        self.safety_margin = MIN_SAFETY_MARGIN
        self._suppress_mode_cb = False
        self._obstacle_undo_stack: List[ObstacleState] = []
        self._rect_drag_preview: Optional[Tuple[float, float, float, float, float]] = None
        self._last_path_stats: Dict[str, object] = {}
        self._current_scenario_filename: str = ""
        self._last_mod_grid_prims = None
        self._last_planner_phase: int = 1

        self.root = tk.Tk()
        self.var_mod_grid_phase = tk.IntVar(master=self.root, value=3)
        self.var_mod_grid_dp_objective = tk.StringVar(master=self.root, value="length")
        self.var_mod_grid_collision = tk.StringVar(master=self.root, value="offline")
        self.var_mod_grid_se_p3_primitive = tk.StringVar(master=self.root, value="linear_yaw_dp")
        self.var_mod_grid_se_p3_collision = tk.StringVar(master=self.root, value="volume_bin")
        self.var_margin_ge_robot_size = tk.BooleanVar(master=self.root, value=True)
        self.var_export_sat_validate = tk.BooleanVar(master=self.root, value=False)
        self.var_robot_shape = tk.StringVar(master=self.root, value="right_triangle")
        self.var_yaw_fill = tk.StringVar(master=self.root, value="linear")
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
        self._refresh_shape_preview_vertices()
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
        for m in ("rect", "erase"):
            tk.Radiobutton(
                mode_left,
                text=m,
                variable=self.var_mode,
                value=m,
                command=self._on_mode_changed,
            ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(mode_left, text="undo", command=self.undo_last_obstacle, width=5).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        thick_f = tk.Frame(mode_row)
        thick_f.pack(side=tk.RIGHT)
        tk.Label(thick_f, text="line thick").pack(side=tk.LEFT, padx=(8, 4))
        self.ent_line_th = tk.Entry(thick_f, width=8)
        self.ent_line_th.insert(0, str(self.line_thickness))
        self.ent_line_th.pack(side=tk.LEFT)
        thick_f.pack_forget()

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

        tk.Label(ctrl, text="mod_grid disk collision", font=("TkDefaultFont", 9, "bold")).pack(
            anchor=tk.W, pady=(8, 0)
        )
        self.rb_mod_collision_offline = tk.Radiobutton(
            ctrl,
            text="offline (build full bitmap)",
            variable=self.var_mod_grid_collision,
            value="offline",
            command=self._on_planner_changed,
        )
        self.rb_mod_collision_online = tk.Radiobutton(
            ctrl,
            text="online (lazy per-cell checks)",
            variable=self.var_mod_grid_collision,
            value="online",
            command=self._on_planner_changed,
        )
        self.rb_mod_collision_offline.pack(anchor=tk.W)
        self.rb_mod_collision_online.pack(anchor=tk.W)

        tk.Label(ctrl, text="Yaw Filling (mod_grid)", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(10, 0))
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
            text="Differential Flatness (phase 3)",
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

        _btn("Apply rect angle", self.apply_robot)
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
        if self.txt_status.index("end-1c") != "1.0":
            self.txt_status.insert(tk.END, "\n")
        self.txt_status.insert(tk.END, str(line))
        self.txt_status.see(tk.END)

    def _make_text_readonly_copyable(self, widget: tk.Text) -> None:
        """Read-only log text that still allows selection and copy (Ctrl+C / right-click)."""

        def _allow_key(event):
            ctrl = bool(event.state & 0x4)
            if ctrl and event.keysym.lower() in ("c", "a"):
                return
            if event.keysym in (
                "Left",
                "Right",
                "Up",
                "Down",
                "Home",
                "End",
                "Prior",
                "Next",
                "Shift_L",
                "Shift_R",
                "Control_L",
                "Control_R",
                "Tab",
            ):
                return
            return "break"

        widget.bind("<Key>", _allow_key)
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(
            label="Select all",
            command=lambda: (widget.tag_add("sel", "1.0", "end-1c"), widget.mark_set("insert", "1.0")),
        )

        def _popup(event):
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        widget.bind("<Button-3>", _popup)

    @staticmethod
    def _path_stats_summary(path_stats: Optional[Dict[str, object]]) -> str:
        if not path_stats:
            return ""
        poly_len = float(path_stats.get("polyline_length_m", 0.0))
        n_prims = int(path_stats.get("n_primitives", 0))
        prim_len = float(path_stats.get("primitive_length_m", 0.0))
        parts = [f"len={poly_len:.2f}m", f"prims={n_prims}"]
        if prim_len > 0.0:
            parts.append(f"prim_len={prim_len:.2f}m")
        if path_stats.get("p3_fallback"):
            parts.append("fallback")
        elif path_stats.get("p3_compressed"):
            parts.append("compressed")
        return " | " + " ".join(parts)

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
        for m in ("rect", "erase"):
            tk.Radiobutton(
                mode_left,
                text=m,
                variable=self.var_mode,
                value=m,
                command=self._on_mode_changed,
            ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(mode_left, text="undo", command=self.undo_last_obstacle, width=5).pack(
            side=tk.LEFT, padx=(0, 8)
        )

        thick_f = tk.Frame(mode_row)
        tk.Label(thick_f, text="line thick").pack(side=tk.LEFT, padx=(8, 4))
        self.ent_line_th = tk.Entry(thick_f, width=8)
        self.ent_line_th.insert(0, str(self.line_thickness))
        self.ent_line_th.pack(side=tk.LEFT)
        thick_f.pack_forget()

        tk.Label(ctrl, text="Rect angle (deg)", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))
        self.ent_rect_angle = self._row_entry(ctrl, "angle°", str(self.rect_angle_deg))

        # Settings buttons (left column)
        btnf = tk.Frame(ctrl)
        btnf.pack(fill=tk.X, pady=(10, 0))
        tk.Button(btnf, text="Apply rect angle", command=self.apply_robot, width=22).pack(fill=tk.X, pady=2)
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
        self.lbl_safety_margin = tk.Label(ctrl, text="", font=("TkDefaultFont", 8))
        self.lbl_safety_margin.pack(anchor=tk.W)
        self.ent_safety_margin = self._row_entry(ctrl, "margin [m]", str(self.safety_margin))
        self.chk_margin_ge_robot = tk.Checkbutton(
            ctrl,
            text="margin >= swarm pusher size (diameter)",
            variable=self.var_margin_ge_robot_size,
            command=self._on_margin_policy_changed,
        )
        self.chk_margin_ge_robot.pack(anchor=tk.W)
        self._update_safety_margin_label()

        self.chk_export_sat_validate = tk.Checkbutton(
            ctrl,
            text="SAT-check path before export",
            variable=self.var_export_sat_validate,
        )
        self.chk_export_sat_validate.pack(anchor=tk.W)

        tk.Label(ctrl, text="Yaw Filling (mod_grid)", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(10, 0))
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
            text="Differential Flatness (phase 3)",
            variable=self.var_yaw_fill,
            value="differential_flatness",
            command=self._on_planner_changed,
        )
        self.rb_yaw_df.pack(anchor=tk.W)

        # mod_grid stop phase + phase-3 DP objective
        tk.Label(ctrl, text="mod_grid stop phase", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.rb_mod_phase1 = tk.Radiobutton(
            ctrl, text="1 (augmented A*)", variable=self.var_mod_grid_phase, value=1, command=self._on_planner_changed
        )
        self.rb_mod_phase3 = tk.Radiobutton(
            ctrl, text="3 (shortcut + arc DP)", variable=self.var_mod_grid_phase, value=3, command=self._on_planner_changed
        )
        self.rb_mod_phase1.pack(anchor=tk.W)
        self.rb_mod_phase3.pack(anchor=tk.W)

        tk.Label(ctrl, text="mod_grid disk collision", font=("TkDefaultFont", 9, "bold")).pack(
            anchor=tk.W, pady=(8, 0)
        )
        self.rb_mod_collision_offline = tk.Radiobutton(
            ctrl,
            text="offline (build full bitmap)",
            variable=self.var_mod_grid_collision,
            value="offline",
            command=self._on_planner_changed,
        )
        self.rb_mod_collision_online = tk.Radiobutton(
            ctrl,
            text="online (lazy per-cell checks)",
            variable=self.var_mod_grid_collision,
            value="online",
            command=self._on_planner_changed,
        )
        self.rb_mod_collision_offline.pack(anchor=tk.W)
        self.rb_mod_collision_online.pack(anchor=tk.W)

        self.lbl_mod_dp_objective = tk.Label(
            ctrl, text="Phase 3 DP objective", font=("TkDefaultFont", 9, "bold")
        )
        self.lbl_mod_dp_objective.pack(anchor=tk.W, pady=(6, 0))
        self.rb_mod_dp_length = tk.Radiobutton(
            ctrl,
            text="shortest length",
            variable=self.var_mod_grid_dp_objective,
            value="length",
            command=self._on_planner_changed,
        )
        self.rb_mod_dp_count = tk.Radiobutton(
            ctrl,
            text="min primitive count",
            variable=self.var_mod_grid_dp_objective,
            value="min_segments",
            command=self._on_planner_changed,
        )
        self.rb_mod_dp_length.pack(anchor=tk.W)
        self.rb_mod_dp_count.pack(anchor=tk.W)

        self.lbl_mod_se_p3_primitive = tk.Label(
            ctrl, text="SE phase 3 primitive", font=("TkDefaultFont", 9, "bold")
        )
        self.rb_se_p3_linear_yaw = tk.Radiobutton(
            ctrl,
            text="linear yaw DP (S_interp + C_interp)",
            variable=self.var_mod_grid_se_p3_primitive,
            value="linear_yaw_dp",
            command=self._on_planner_changed,
        )
        self.rb_se_p3_body_twist = tk.Radiobutton(
            ctrl,
            text="body twist (legacy, disabled)",
            variable=self.var_mod_grid_se_p3_primitive,
            value="body_twist",
            command=self._on_planner_changed,
        )
        self.lbl_mod_se_p3_collision = tk.Label(
            ctrl, text="SE phase 3 collision check", font=("TkDefaultFont", 9, "bold")
        )
        self.rb_se_p3_volume_bin = tk.Radiobutton(
            ctrl,
            text="volume bin (conservative θ-bin)",
            variable=self.var_mod_grid_se_p3_collision,
            value="volume_bin",
            command=self._on_planner_changed,
        )
        self.rb_se_p3_sat_direct = tk.Radiobutton(
            ctrl,
            text="SAT direct (exact pose)",
            variable=self.var_mod_grid_se_p3_collision,
            value="sat_direct",
            command=self._on_planner_changed,
        )
        for w in (
            self.lbl_mod_se_p3_primitive,
            self.rb_se_p3_linear_yaw,
            self.rb_se_p3_body_twist,
            self.lbl_mod_se_p3_collision,
            self.rb_se_p3_volume_bin,
            self.rb_se_p3_sat_direct,
        ):
            w.pack_forget()

        self._update_planner_options()

        self.safety_margin = self._clamp_app_safety_margin(self.safety_margin)
        self._sync_safety_margin_entry()

        tk.Label(ctrl, text="Planner actions", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W, pady=(12, 0))

        btnf = tk.Frame(ctrl)
        btnf.pack(fill=tk.X, pady=(8, 0))

        def _btn(text: str, cmd):
            tk.Button(btnf, text=text, command=cmd, width=22).pack(fill=tk.X, pady=2)

        _btn("Plan / Replan", self.replan)
        _btn("Import planned path", self.import_planned_path)
        _btn("Export planned path", self.export_planned_path)
        _btn("Reset View", self.reset_view)

        status_wrap = tk.Frame(ctrl, width=280, height=120)
        status_wrap.pack(fill=tk.X, pady=(10, 0))
        status_wrap.pack_propagate(False)
        self.txt_status = tk.Text(status_wrap, width=38, height=7, wrap=tk.WORD, exportselection=True)
        self.txt_status.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll = tk.Scrollbar(status_wrap, command=self.txt_status.yview)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_status.configure(yscrollcommand=status_scroll.set)
        self._make_text_readonly_copyable(self.txt_status)
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
        self._clear_path_footprint_viz()
        self.path_data = None
        self._refresh_shape_preview_vertices()
        self._update_safety_margin_label()
        if hasattr(self, "ent_safety_margin"):
            self.safety_margin = self._clamp_app_safety_margin(
                self._parse_float_from_entry(self.ent_safety_margin, self.safety_margin)
            )
            self._sync_safety_margin_entry()
        self.status_text = f"Robot shape: {self.var_robot_shape.get()}"
        self._update_planner_options()
        self.render()

    def _refresh_shape_preview_vertices(self) -> None:
        """Load scaled footprint vertices for the current robot shape (same as mod_grid_SE)."""
        try:
            import mod_grid_SE as se

            shape = self.var_robot_shape.get()
            try:
                import rospkg

                src = Path(rospkg.RosPack().get_path("contact_maintain")) / "src"
                if str(src) not in sys.path:
                    sys.path.insert(0, str(src))
                from contact_maintain.footprint_cache import default_cache_path, vertices_for_shape

                if not vertices_for_shape(shape):
                    self._append_status_log(
                        f"Note: no cached footprint for '{shape}' "
                        f"({default_cache_path().name}); using OBJ/DXF fallback. "
                        "Run scripts/test/preprocess_obj_footprints.py to precompute."
                    )
            except Exception:
                pass
            spec = self._robot_dict_for_planner()
            self._shape_preview_vertices_local = se._extract_robot_footprint_vertices_local(
                spec, self.map_state.resolution
            )
        except Exception:
            self._shape_preview_vertices_local = None

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
                self.rb_mod_phase1.config(state=tk.NORMAL)
                if int(self.var_mod_grid_phase.get()) == 2:
                    self.var_mod_grid_phase.set(3)
                if use_se:
                    self.rb_mod_phase3.config(
                        text="3 (SE(2) primitive compression)", state=tk.NORMAL
                    )
                    phase = int(self.var_mod_grid_phase.get())
                    p3_on = phase == 3
                    for w in (
                        self.lbl_mod_se_p3_primitive,
                        self.rb_se_p3_linear_yaw,
                        self.rb_se_p3_body_twist,
                        self.lbl_mod_se_p3_collision,
                        self.rb_se_p3_volume_bin,
                        self.rb_se_p3_sat_direct,
                    ):
                        w.pack(anchor=tk.W)
                    se_p3_state = tk.NORMAL if p3_on else tk.DISABLED
                    for rb in (
                        self.lbl_mod_se_p3_primitive,
                        self.rb_se_p3_linear_yaw,
                        self.lbl_mod_se_p3_collision,
                        self.rb_se_p3_volume_bin,
                        self.rb_se_p3_sat_direct,
                    ):
                        rb.config(state=se_p3_state)
                    self.rb_se_p3_body_twist.config(state=tk.DISABLED)
                    if self.var_mod_grid_se_p3_primitive.get() == "body_twist":
                        self.var_mod_grid_se_p3_primitive.set("linear_yaw_dp")
                    self.lbl_mod_dp_objective.pack(anchor=tk.W, pady=(6, 0))
                    for rb in (self.rb_mod_dp_length, self.rb_mod_dp_count):
                        rb.pack(anchor=tk.W)
                        rb.config(state=se_p3_state)
                    for rb in (self.rb_mod_collision_offline, self.rb_mod_collision_online):
                        rb.pack_forget()
                else:
                    self.rb_mod_phase3.config(text="3 (shortcut + arc DP)", state=tk.NORMAL)
                    for w in (
                        self.lbl_mod_se_p3_primitive,
                        self.rb_se_p3_linear_yaw,
                        self.rb_se_p3_body_twist,
                        self.lbl_mod_se_p3_collision,
                        self.rb_se_p3_volume_bin,
                        self.rb_se_p3_sat_direct,
                    ):
                        w.pack_forget()
                    self.lbl_mod_dp_objective.pack(anchor=tk.W, pady=(6, 0))
                    for rb in (self.rb_mod_dp_length, self.rb_mod_dp_count):
                        rb.pack(anchor=tk.W)
                    dp_on = int(self.var_mod_grid_phase.get()) == 3
                    dp_state = tk.NORMAL if dp_on else tk.DISABLED
                    for rb in (
                        self.lbl_mod_dp_objective,
                        self.rb_mod_dp_length,
                        self.rb_mod_dp_count,
                    ):
                        rb.config(state=dp_state)
            else:
                for rb in (
                    self.rb_mod_phase1,
                    self.rb_mod_phase3,
                    self.lbl_mod_dp_objective,
                    self.rb_mod_dp_length,
                    self.rb_mod_dp_count,
                ):
                    rb.config(state=tk.DISABLED)
                for rb in (self.rb_mod_collision_offline, self.rb_mod_collision_online):
                    rb.config(state=tk.DISABLED)
            # Yaw-filling: mod_grid only; phase 1 = linear only, phase 3 = linear or DF.
            if self.var_planner.get() == "mod_grid":
                phase = int(self.var_mod_grid_phase.get())
                for rb in (self.rb_mod_collision_offline, self.rb_mod_collision_online):
                    rb.config(state=tk.NORMAL)
                if self.var_mod_grid_dp_objective.get() == "compare":
                    self.var_mod_grid_dp_objective.set("length")
                if self.var_yaw_fill.get() in ("none", ""):
                    self.var_yaw_fill.set("linear")
                if phase == 1:
                    if self.var_yaw_fill.get() == "differential_flatness":
                        self.var_yaw_fill.set("linear")
                    self.rb_yaw_linear.config(state=tk.NORMAL)
                    self.rb_yaw_df.config(state=tk.DISABLED)
                else:
                    self.rb_yaw_linear.config(state=tk.NORMAL)
                    self.rb_yaw_df.config(state=tk.NORMAL)
            else:
                for rb in (self.rb_yaw_linear, self.rb_yaw_df):
                    rb.config(state=tk.DISABLED)
                for rb in (self.rb_mod_collision_offline, self.rb_mod_collision_online):
                    rb.config(state=tk.DISABLED)
        else:
            self.rb_grid.config(state=tk.DISABLED)
            self.rb_mod_grid.config(state=tk.DISABLED)
            self.rb_mod_grid_se.config(state=tk.DISABLED)
            self.rb_hybrid.config(state=tk.NORMAL)
            if self.var_planner.get() in ("grid_astar", "mod_grid", "mod_grid_se"):
                self.var_planner.set("hybrid_astar")

            for rb in (
                self.rb_mod_phase1,
                self.rb_mod_phase3,
                self.lbl_mod_dp_objective,
                self.rb_mod_dp_length,
                self.rb_mod_dp_count,
            ):
                rb.config(state=tk.DISABLED)
            for rb in (self.rb_yaw_linear, self.rb_yaw_df):
                rb.config(state=tk.DISABLED)
            for rb in (self.rb_mod_collision_offline, self.rb_mod_collision_online):
                rb.config(state=tk.DISABLED)

    @staticmethod
    def _disk_radius_for_path_viz(rr: float, reso: float, safety_margin: float = 0.0) -> float:
        """Disk radius drawn along the path (matches planner disk, not shrunk)."""
        return float(rr) + float(safety_margin)

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

    def _apply_mod_grid_footprint_viz(
        self,
        px: List[float],
        py: List[float],
        *,
        phase: int,
        yaw_fill: str,
        reso: float,
        disk_viz_r: float,
        prims: Optional[List[Tuple[str, dict]]] = None,
    ) -> Tuple[List[float], List[float], Optional[List[float]]]:
        """
        Densify the planner polyline for footprint display; yaw fill never changes geometry.

        Planning clearance is decided earlier (phase 1 grid, phase 3 primitive funnel in
        ``scenario_obstacles``).  This only builds ``(vx, vy, pyaw)`` for rendering.
        """
        if len(px) < 2:
            self._path_footprint_mode = "disk"
            self._disk_radius_for_viz = disk_viz_r
            return px, py, None

        robot_spec = self._robot_dict_for_planner()
        robot_vertices_local = mod_grid_SE_astar._extract_robot_footprint_vertices_local(
            robot_spec, reso=reso
        )
        syaw = math.radians(self.start.yaw_deg)
        gyaw = math.radians(self.goal.yaw_deg)

        vx, vy = mod_grid_astar.densify_polyline(px, py, reso)

        if phase != 3 or yaw_fill != "differential_flatness":
            pyaw = self._fill_yaw_linear(len(vx), syaw, gyaw)
        else:
            if not prims:
                raise RuntimeError("Differential flatness requires phase-3 primitives")
            pyaw = mod_grid_astar.yaw_on_polyline_samples(vx, vy, prims, syaw)
            if len(pyaw) != len(vx):
                raise RuntimeError("Yaw fill length mismatch along planner path")
            vx, vy, pyaw = self._append_final_self_rotation(vx, vy, pyaw, gyaw)

        self._path_footprint_mode = "polygon"
        self._robot_vertices_local = list(robot_vertices_local)
        self._path_pyaw = list(pyaw)
        self._disk_radius_for_viz = disk_viz_r
        return vx, vy, pyaw

    def _restore_path_footprint_viz_from_planned(self, planned, bundle: dict) -> None:
        """After import, enable polygon/disk footprint overlays along the path."""
        self._clear_path_footprint_viz()
        if not planned.ok or len(planned.px) < 2:
            return

        reso = self.map_state.resolution
        robot_spec = self._robot_dict_for_planner()
        disk_r = self._disk_planner_rr(reso)

        path_block = bundle.get("path", {}) or {}
        planner = str(planned.planner or path_block.get("planner", "mod_grid_se")).lower()

        pyaw = list(planned.pyaw) if planned.pyaw else None
        n = len(planned.px)
        if pyaw is None or len(pyaw) != n:
            syaw = math.radians(self.start.yaw_deg)
            gyaw = math.radians(self.goal.yaw_deg)
            pyaw = self._fill_yaw_linear(n, syaw, gyaw)

        if planner in ("disk", "grid_astar", "mod_grid", "grid"):
            self._path_footprint_mode = "disk"
            self._disk_radius_for_viz = disk_r
            self._path_pyaw = pyaw
            return

        robot_vertices_local = mod_grid_SE_astar._extract_robot_footprint_vertices_local(
            robot_spec, reso=reso
        )
        self._path_footprint_mode = "polygon"
        self._robot_vertices_local = list(robot_vertices_local)
        self._disk_radius_for_viz = disk_r
        self._path_pyaw = pyaw

    def _swarm_pusher_diameter_m(self) -> float:
        return float(SWARM_PUSHER_ROBOT_DIAMETER_M)

    def _min_safety_margin_m(self, reso: Optional[float] = None) -> float:
        del reso
        return swarm_pusher_min_safety_margin_m(
            margin_ge_swarm_pusher_size=bool(self.var_margin_ge_robot_size.get())
        )

    def _clamp_app_safety_margin(self, margin: float, reso: Optional[float] = None) -> float:
        return clamp_safety_margin(margin, min_margin=self._min_safety_margin_m(reso))

    def _update_safety_margin_label(self) -> None:
        if not hasattr(self, "lbl_safety_margin"):
            return
        if bool(self.var_margin_ge_robot_size.get()):
            dia = self._swarm_pusher_diameter_m()
            self.lbl_safety_margin.config(
                text=f"min = swarm pusher diameter)"
            )
        else:
            self.lbl_safety_margin.config(text=f"min = {MIN_SAFETY_MARGIN:.2f} m (fixed floor)")

    def _sync_safety_margin_entry(self) -> None:
        if hasattr(self, "ent_safety_margin"):
            self.ent_safety_margin.delete(0, tk.END)
            self.ent_safety_margin.insert(0, str(self.safety_margin))

    def _on_margin_policy_changed(self) -> None:
        self._update_safety_margin_label()
        if hasattr(self, "ent_safety_margin"):
            cur = self._parse_float_from_entry(self.ent_safety_margin, self.safety_margin)
            if bool(self.var_margin_ge_robot_size.get()):
                min_m = self._min_safety_margin_m()
                if cur < min_m:
                    cur = min_m
            self.safety_margin = self._clamp_app_safety_margin(cur)
            self._sync_safety_margin_entry()

    def _planner_options_dict(self, planner: str, phase: int) -> Dict[str, object]:
        opts: Dict[str, object] = {
            "planner": planner,
            "stop_phase": int(phase),
            "yaw_fill_mode": self.var_yaw_fill.get(),
        }
        if planner == "mod_grid":
            opts["disk_collision_mode"] = self.var_mod_grid_collision.get()
            opts["dp_objective"] = self.var_mod_grid_dp_objective.get()
        if planner == "mod_grid_se" and phase == 3:
            opts["se_p3_primitive"] = self.var_mod_grid_se_p3_primitive.get()
            opts["se_p3_collision"] = self.var_mod_grid_se_p3_collision.get()
            opts["dp_objective"] = self.var_mod_grid_dp_objective.get()
        return opts

    @staticmethod
    def _sanitize_scenario_stem(raw: str, mode_prefix: str) -> Optional[str]:
        stem = raw.strip()
        if not stem:
            return None
        safe = re.sub(r"[^\w\-.]+", "_", stem).strip("._")
        if not safe or safe == ".":
            return None
        if safe.lower().endswith(".json"):
            safe = safe[:-5]
        safe_lower = safe.lower()
        for prefix in (
            "rectobs_scenario_",
            "lineobs_scenario_",
            "scenario_",
            "rectobs_",
            "lineobs_",
        ):
            if safe_lower.startswith(prefix):
                safe = safe[len(prefix) :]
                break
        return safe

    def _apply_planner_options_from_dict(self, opts: Dict[str, object]) -> None:
        if not opts:
            return
        planner = str(opts.get("planner", ""))
        if planner in ("grid_astar", "mod_grid", "mod_grid_se", "hybrid_astar"):
            self.var_planner.set(planner)
        if "stop_phase" in opts:
            self.var_mod_grid_phase.set(int(opts["stop_phase"]))
        if "yaw_fill_mode" in opts:
            yfill = str(opts["yaw_fill_mode"])
            if yfill in ("linear", "differential_flatness"):
                self.var_yaw_fill.set(yfill)
        if "disk_collision_mode" in opts:
            self.var_mod_grid_collision.set(str(opts["disk_collision_mode"]))
        if "dp_objective" in opts:
            self.var_mod_grid_dp_objective.set(str(opts["dp_objective"]))
        if "se_p3_primitive" in opts:
            self.var_mod_grid_se_p3_primitive.set(str(opts["se_p3_primitive"]))
        if "se_p3_collision" in opts:
            self.var_mod_grid_se_p3_collision.set(str(opts["se_p3_collision"]))

    def _apply_scenario_from_dict(self, data: dict, filename: str) -> None:
        m = data.get("map", {})
        self.map_state = MapState(
            float(m.get("width", 60.0)), float(m.get("height", 40.0)), float(m.get("resolution", 1.0))
        )
        r = data.get("robot", {})
        self.robot_state = RobotState(
            str(r.get("type", "holonomic")), float(r.get("width", 2.0)), float(r.get("length", 3.0))
        )
        if "margin_ge_robot_size" in r:
            self.var_margin_ge_robot_size.set(bool(r.get("margin_ge_robot_size", True)))
        else:
            self.var_margin_ge_robot_size.set(True)
        self._update_safety_margin_label()
        self.safety_margin = self._clamp_app_safety_margin(
            float(r.get("safety_margin", MIN_SAFETY_MARGIN))
        )
        shape_name = str(r.get("shape_name", "right_triangle"))
        if shape_name not in OBJ_SHAPE_NAMES:
            shape_name = OBJ_SHAPE_NAMES[0]
        self.var_robot_shape.set(shape_name)
        p = data.get("pose", {})
        s = p.get("start", [5.0, 5.0, 0.0])
        g = p.get("goal", [50.0, 30.0, 0.0])
        self.start = Pose(float(s[0]), float(s[1]), float(s[2]))
        self.goal = Pose(float(g[0]), float(g[1]), float(g[2]))
        self.line_thickness = float(data.get("draw", {}).get("line_thickness", 5.0))
        yfill = str(data.get("planner", {}).get("yaw_fill_mode", "linear"))
        if yfill in ("none", ""):
            yfill = "linear"
        if yfill not in ("linear", "differential_flatness"):
            yfill = "linear"
        self.var_yaw_fill.set(yfill)

        o = data.get("obstacles", {})
        self.obstacles = ObstacleState(
            rects={k: tuple(v) for k, v in o.get("rects", {}).items()},
            lines={k: [tuple(pt) for pt in v] for k, v in o.get("lines", {}).items()},
            rect_count=int(o.get("rect_count", len(o.get("rects", {})))),
            line_count=int(o.get("line_count", len(o.get("lines", {})))),
        )
        self._normalize_obstacle_ids_inplace()
        self._refresh_shape_preview_vertices()
        self.var_robot.set(self.robot_state.robot_type)
        self._current_scenario_filename = filename
        if self.var_planner.get() == "mod_grid_se":
            self.var_mod_grid_phase.set(1)
        self._update_planner_options()
        self._sync_entries()
        self.ent_safety_margin.delete(0, tk.END)
        self.ent_safety_margin.insert(0, str(self.safety_margin))

    def import_planned_path(self) -> None:
        path = filedialog.askopenfilename(
            title="Import planned path",
            initialdir=str(THIS_DIR),
            filetypes=[("Planned path", "*.planned.json"), ("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        planned_file = Path(path)
        try:
            scenario_ref, bundle, planned = load_planned_path_bundle(planned_file)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            messagebox.showerror("Import path", f"Failed to read planned path:\n{exc}", parent=self.root)
            return
        scenario_path = planned_file.parent / scenario_ref
        if not scenario_path.is_file():
            scenario_path = THIS_DIR / scenario_ref
        if not scenario_path.is_file():
            messagebox.showerror(
                "Import path",
                f"Scenario file not found: {scenario_ref}\n(expected next to {planned_file.name})",
                parent=self.root,
            )
            return
        try:
            with scenario_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Import path", f"Failed to read scenario:\n{exc}", parent=self.root)
            return
        if not planned.ok:
            messagebox.showwarning("Import path", "Planned path has fewer than 2 points.", parent=self.root)
            return

        safety = bundle.get("safety", {}) or {}
        if "margin_ge_robot_size" in safety:
            self.var_margin_ge_robot_size.set(bool(safety["margin_ge_robot_size"]))

        self._apply_scenario_from_dict(data, scenario_path.name)
        self.path_data = (list(planned.px), list(planned.py))
        self._path_pyaw = list(planned.pyaw) if planned.pyaw else None
        self._last_planner_phase = int(planned.stop_phase)
        path_block = bundle.get("path", {}) or {}
        self._last_path_stats = dict(path_block.get("path_stats", {}) or {})
        self._last_mod_grid_prims = list(planned.primitives) if planned.primitives else None
        self._apply_planner_options_from_dict(dict(path_block.get("planner_options", {}) or {}))
        self._update_planner_options()
        self._restore_path_footprint_viz_from_planned(planned, bundle)

        self.var_scenario_file.set(scenario_path.name)
        self._refresh_scenario_combo()
        self.status_text = f"Imported: {planned_file.name} (+ {scenario_path.name})"
        self._append_status_log(f"Imported scenario: {scenario_path.name}")
        self._append_status_log(
            f"Imported path: {planned_file.name} | planner={planned.planner} "
            f"phase={planned.stop_phase} pts={len(planned.px)}"
        )
        self.reset_view()
        self.render()

    def export_planned_path(self) -> None:
        if not self.path_data or len(self.path_data[0]) < 2:
            messagebox.showwarning(
                "Export path",
                "Plan a path first (Plan / Replan), then export.",
                parent=self.root,
            )
            return

        if not bool(self.var_margin_ge_robot_size.get()):
            if not messagebox.askyesno(
                "Export with low margin floor",
                "“margin >= swarm pusher size” is unchecked.\n\n"
                f"The minimum margin is only {MIN_SAFETY_MARGIN:.2f} m, which may be "
                "unsafe for the Magnum pusher fleet to squeeze through gaps.\n\n"
                "Export anyway?",
                parent=self.root,
            ):
                return

        self.apply_pose()
        self.apply_map()
        self.apply_robot()
        self.safety_margin = self._clamp_app_safety_margin(
            self._parse_float_from_entry(self.ent_safety_margin, self.safety_margin)
        )
        self._sync_safety_margin_entry()

        mode_prefix = "rectObs" if self.mode == "rect" else "lineObs"
        initial = ""
        if self._current_scenario_filename:
            stem = Path(self._current_scenario_filename).stem
            for prefix in ("rectObs_scenario_", "lineObs_scenario_", "scenario_"):
                if stem.startswith(prefix):
                    initial = stem[len(prefix) :]
                    break
            else:
                initial = stem
        raw = simpledialog.askstring(
            "Export planned path",
            f"Export name (writes {mode_prefix}_scenario_<name>.json + .planned.json):",
            parent=self.root,
            initialvalue=initial or "default",
        )
        if raw is None:
            return
        safe = self._sanitize_scenario_stem(raw, mode_prefix)
        if safe is None:
            messagebox.showerror("Export path", "Invalid export name.", parent=self.root)
            return

        fname = f"{mode_prefix}_scenario_{safe}.json"
        if (THIS_DIR / fname).exists():
            if not messagebox.askyesno(
                "Export path",
                f"'{fname}' already exists. Overwrite scenario + planned pair?",
                parent=self.root,
            ):
                return

        self._normalize_obstacle_ids_inplace()
        scenario = self._scenario_dict()
        scenario["metadata"] = {
            "filename": fname,
            "obstacle_id_prefixes": {"rectangles": "RECT_", "polylines": "LINE_"},
        }

        px, py = self.path_data
        pyaw = list(self._path_pyaw) if self._path_pyaw else []
        export_sat_validation = None
        if bool(self.var_export_sat_validate.get()):
            export_sat_validation = validate_planned_path_sat(
                scenario, px, py, pyaw, safety_margin=float(self.safety_margin)
            )
            export_sat_validation["checked"] = True
            if not export_sat_validation.get("ok", False):
                reason = export_sat_validation.get("reason", "collision detected")
                if not messagebox.askyesno(
                    "Export path — SAT collision",
                    "Pre-export SAT validation failed.\n\n"
                    f"Reason: {reason}\n"
                    f"Margin: {export_sat_validation.get('safety_margin_m', self.safety_margin):.2f} m\n\n"
                    "Export anyway?",
                    parent=self.root,
                ):
                    return
            else:
                self._append_status_log("Export SAT validation: passed")

        planner = self.var_planner.get() if self.robot_state.robot_type == "holonomic" else "hybrid_astar"
        bundle = build_planned_path_bundle(
            scenario=scenario,
            scenario_filename=fname,
            px=list(px),
            py=list(py),
            pyaw=pyaw,
            planner=planner,
            stop_phase=int(self._last_planner_phase),
            path_stats=dict(self._last_path_stats) if self._last_path_stats else None,
            planner_options=self._planner_options_dict(planner, int(self._last_planner_phase)),
            safety_margin=float(self.safety_margin),
            safety_margin_ge_robot_size=bool(self.var_margin_ge_robot_size.get()),
            min_safety_margin_m=self._min_safety_margin_m(),
            prims=self._last_mod_grid_prims,
            export_sat_validation=export_sat_validation,
        )
        try:
            scenario_path, planned_path = write_planned_path_export_pair(
                scenario, fname, bundle, out_dir=THIS_DIR
            )
        except OSError as exc:
            messagebox.showerror("Export path", f"Failed to write export files:\n{exc}", parent=self.root)
            return

        self._current_scenario_filename = fname
        self.var_scenario_file.set(fname)
        self._refresh_scenario_combo()
        hint = bundle.get("metadata", {}).get("controller_hint", "")
        self.status_text = f"Exported: {planned_path.name} (+ {scenario_path.name})"
        self._append_status_log(f"Exported scenario: {scenario_path.name}")
        self._append_status_log(f"Exported planned path: {planned_path.name}")
        if hint:
            self._append_status_log(f"Controller hint: {hint}")
        self.render()

    def _disk_planner_rr(self, reso: float) -> float:
        """
        Robot radius [m] for disk-based planners (grid_astar, mod_grid).
        Uses circumradius of the true OBJ / standard footprint (not scenario width/length).
        """
        robot_spec = self._robot_dict_for_planner()
        verts = mod_grid_SE_astar._extract_robot_footprint_vertices_local(robot_spec, reso=reso)
        return max(math.hypot(vx, vy) for vx, vy in verts)

    def _log_disk_phase1_report(
        self,
        timing: Dict[str, float],
        *,
        meta: Optional[Dict[str, object]] = None,
        wall_s: Optional[float] = None,
    ) -> None:
        for line in mod_grid_astar.format_disk_phase1_report(
            timing,
            meta=meta,
            wall_s=wall_s,
        ):
            self._append_status_log(line)

    def _log_se2_pipeline_report(
        self,
        timing: Dict[str, float],
        *,
        prep: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, object]] = None,
        phase3_s: Optional[float] = None,
        wall_s: Optional[float] = None,
    ) -> None:
        for line in mod_grid_SE_astar.format_se2_pipeline_report(
            timing,
            prep=prep,
            meta=meta,
            phase3_s=phase3_s,
            wall_s=wall_s,
        ):
            self._append_status_log(line)

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
        if hasattr(self, "ent_line_th"):
            self.line_thickness = max(0.1, self._parse_float_from_entry(self.ent_line_th, self.line_thickness))
        if hasattr(self, "ent_rect_angle"):
            self.rect_angle_deg = self._parse_float_from_entry(self.ent_rect_angle, self.rect_angle_deg)
        self._refresh_shape_preview_vertices()
        self.status_text = "Rect angle updated"
        self.render()

    def _parsed_obstacle_rects(self) -> Dict[str, ObstacleRect]:
        return parse_scenario_rects(
            {k: list(v) for k, v in self.obstacles.rects.items()},
            map_w=self.map_state.map_width,
            map_h=self.map_state.map_height,
        )

    def _planner_obstacle_rects(self) -> List:
        return list(self._parsed_obstacle_rects().values())

    def _obstacle_rects_for_planner(self) -> List[Tuple[float, ...]]:
        parsed = self._parsed_obstacle_rects()
        return rect_values_for_se(parsed) if parsed else []

    def _robot_dict_for_planner(self) -> dict:
        shape = self.var_robot_shape.get()
        spec = {
            "shape_name": shape,
            "width": self.robot_state.robot_width,
            "length": self.robot_state.robot_length,
        }
        obj_path = obj_path_for_shape(shape)
        if obj_path:
            spec["obj_path"] = obj_path
        return spec

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
            return
        if self.mode == "rect" and self.drag_start is not None and event.xdata is not None and event.ydata is not None:
            x0, y0 = self.drag_start
            x1, y1 = event.xdata, event.ydata
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w > 1e-3 and h > 1e-3:
                angle = (
                    self._parse_float_from_entry(self.ent_rect_angle, self.rect_angle_deg)
                    if hasattr(self, "ent_rect_angle")
                    else 0.0
                )
                if abs(angle) > 1e-6:
                    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                    self._rect_drag_preview = (cx, cy, w, h, angle)
                else:
                    self._rect_drag_preview = (min(x0, x1), min(y0, y1), w, h, 0.0)
            else:
                self._rect_drag_preview = None
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
                self._push_obstacle_undo()
                rid = self.obstacles.next_rect_id()
                angle = self._parse_float_from_entry(self.ent_rect_angle, self.rect_angle_deg) if hasattr(self, "ent_rect_angle") else 0.0
                if abs(angle) > 1e-6:
                    cx = 0.5 * (x0 + x1)
                    cy = 0.5 * (y0 + y1)
                    self.obstacles.rects[rid] = (cx, cy, rect[2], rect[3], angle)
                else:
                    self.obstacles.rects[rid] = rect
                self.status_text = f"Added {rid}"
            self.drag_start = None
            self._rect_drag_preview = None
            self.render()
        elif self.mode == "line" and self.current_line:
            if len(self.current_line) >= 2:
                self._push_obstacle_undo()
                lid = self.obstacles.next_line_id()
                self.obstacles.lines[lid] = self.current_line[:]
                self.status_text = f"Added {lid}"
            self.current_line = []
            self.render()

    def erase_nearest(self, x: float, y: float):
        best_id = None
        best_dist = float("inf")
        best_kind = ""

        for rid, rect_vals in self.obstacles.rects.items():
            parsed = parse_rect_values(rect_vals, map_w=self.map_state.map_width, map_h=self.map_state.map_height)
            cx, cy = parsed.cx, parsed.cy
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
            self._push_obstacle_undo()
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

    def _obstacle_points(self, *, disk_planner: bool = False) -> Tuple[List[float], List[float]]:
        scenario = {
            "map": {
                "width": self.map_state.map_width,
                "height": self.map_state.map_height,
                "resolution": self.map_state.resolution,
            },
            "draw": {"line_thickness": self.line_thickness},
            "obstacles": {
                "rects": {k: list(v) for k, v in self.obstacles.rects.items()},
                "lines": {k: [list(p) for p in pts] for k, pts in self.obstacles.lines.items()},
            },
        }
        if disk_planner:
            ox, oy, _, _, _ = obstacle_points_for_disk_planner(scenario)
        else:
            ox, oy, _, _, _ = obstacle_points_from_scenario(scenario)
        return ox, oy

    def _point_in_collision(self, x: float, y: float) -> bool:
        if x <= 0.0 or y <= 0.0 or x >= self.map_state.map_width or y >= self.map_state.map_height:
            return True
        for rect_vals in self.obstacles.rects.values():
            parsed = parse_rect_values(
                rect_vals, map_w=self.map_state.map_width, map_h=self.map_state.map_height
            )
            if parsed.contains_point(x, y):
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
        self.safety_margin = self._clamp_app_safety_margin(
            self._parse_float_from_entry(self.ent_safety_margin, self.safety_margin)
        )
        self._sync_safety_margin_entry()

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

        planner = self.var_planner.get() if self.robot_state.robot_type == "holonomic" else "hybrid_astar"
        use_disk_obstacle_points = (
            self.robot_state.robot_type == "holonomic"
            and planner in ("grid_astar", "mod_grid", "mod_grid_se")
        )
        ox, oy = self._obstacle_points(disk_planner=use_disk_obstacle_points)
        reso = self.map_state.resolution
        planner_rects = self._planner_obstacle_rects()

        try:
            self._clear_path_footprint_viz()
            self._last_path_stats = {}
            self._last_mod_grid_prims = None
            if self.robot_state.robot_type == "holonomic":
                rr = self._disk_planner_rr(reso)
                sm = float(self.safety_margin)
                disk_viz_r = self._disk_radius_for_path_viz(rr, reso, sm)
                if planner == "grid_astar":
                    self._last_planner_phase = 1
                    self._path_footprint_mode = "disk"
                    self._disk_radius_for_viz = disk_viz_r
                    px, py = grid_astar.astar_planning(
                        self.start.x,
                        self.start.y,
                        self.goal.x,
                        self.goal.y,
                        ox,
                        oy,
                        reso,
                        rr + sm,
                        obstacle_rects=planner_rects,
                    )
                    self._last_path_stats = {
                        "output_pts": len(px),
                        "polyline_length_m": mod_grid_SE_astar.polyline_path_length_m(px, py),
                        "n_primitives": max(0, len(px) - 1),
                        "primitive_length_m": mod_grid_SE_astar.polyline_path_length_m(px, py),
                    }
                elif planner == "mod_grid":
                    yaw_fill = self.var_yaw_fill.get()
                    if yaw_fill in ("none", ""):
                        yaw_fill = "linear"
                    phase = int(self.var_mod_grid_phase.get())
                    self._last_planner_phase = phase
                    prims = None

                    t0 = time.perf_counter()
                    p1_timing: Dict[str, float] = {}
                    disk_collision = self.var_mod_grid_collision.get()
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
                        obstacle_rects=planner_rects,
                        timing=p1_timing,
                        disk_collision_mode=disk_collision,
                    )
                    t_phase1_end = time.perf_counter()
                    disk_meta = {
                        "reso": reso,
                        "map_w": self.map_state.map_width,
                        "map_h": self.map_state.map_height,
                        "rr": rr,
                        "safety_margin": sm,
                        "obstacle_pts": len(ox),
                        "collision_mode": disk_collision,
                    }

                    if len(px) < 2:
                        self._log_disk_phase1_report(
                            p1_timing,
                            meta=disk_meta,
                            wall_s=t_phase1_end - t0,
                        )
                    elif phase == 1:
                        self._log_disk_phase1_report(
                            p1_timing,
                            meta=disk_meta,
                            wall_s=t_phase1_end - t0,
                        )
                        self._last_path_stats = {
                            "output_pts": len(px),
                            "polyline_length_m": mod_grid_SE_astar.polyline_path_length_m(px, py),
                            "n_primitives": max(0, len(px) - 1),
                            "primitive_length_m": mod_grid_SE_astar.polyline_path_length_m(px, py),
                        }
                    elif phase == 3:
                        dp_mode = self.var_mod_grid_dp_objective.get()
                        if dp_mode == "compare":
                            dp_mode = mod_grid_astar.DP_OBJECTIVE_LENGTH
                        p1x, p1y = list(px), list(py)
                        t_p3_0 = time.perf_counter()
                        px, py, prims = mod_grid_astar.phase3_polish(
                            p1x,
                            p1y,
                            ox,
                            oy,
                            rr,
                            safety_margin=sm,
                            reso=reso,
                            obstacle_rects=planner_rects,
                            dp_objective=dp_mode,
                            return_primitives=True,
                        )
                        t_p3_1 = time.perf_counter()
                        prim_len = sum(
                            mod_grid_astar._primitive_length(str(typ), params) for typ, params in prims
                        )
                        self._last_path_stats = {
                            "output_pts": len(px),
                            "polyline_length_m": mod_grid_SE_astar.polyline_path_length_m(px, py),
                            "n_primitives": len(prims),
                            "n_straight": sum(1 for typ, _ in prims if typ == "S"),
                            "n_arc": sum(1 for typ, _ in prims if typ == "A"),
                            "primitive_length_m": prim_len,
                            "p1_spine_pts": len(p1x),
                            "p3_compressed": len(prims) < max(0, len(p1x) - 1),
                        }
                        dp_note = f" dp={dp_mode}" if dp_mode != mod_grid_astar.DP_OBJECTIVE_LENGTH else ""
                        self._append_status_log(
                            f"[timing] mod_grid phase1={1000.0 * (t_phase1_end - t0):.1f}ms | "
                            f"phase3={1000.0 * (t_p3_1 - t_p3_0):.1f}ms | "
                            f"total={1000.0 * (t_p3_1 - t0):.1f}ms{dp_note}"
                        )
                        self._append_status_log(
                            f"  path: {len(px)} pts len={self._last_path_stats['polyline_length_m']:.2f}m | "
                            f"prims={len(prims)} prim_len={prim_len:.2f}m | p1={len(p1x)}"
                        )
                        self._last_mod_grid_prims = list(prims)

                    px, py, _ = self._apply_mod_grid_footprint_viz(
                        px,
                        py,
                        phase=phase,
                        yaw_fill=yaw_fill,
                        reso=reso,
                        disk_viz_r=disk_viz_r,
                        prims=prims,
                    )
                elif planner == "mod_grid_se":
                    # True footprint SE(2) planning (Phase 1 or 3).
                    syaw = math.radians(self.start.yaw_deg)
                    gyaw = math.radians(self.goal.yaw_deg)
                    phase = int(self.var_mod_grid_phase.get())
                    self._last_planner_phase = phase
                    self._last_mod_grid_prims = None
                    wall_t0 = time.perf_counter()
                    prep_timing: Dict[str, float] = {}
                    t_fp0 = time.perf_counter()
                    robot_spec = self._robot_dict_for_planner()
                    robot_vertices_local = mod_grid_SE_astar._extract_robot_footprint_vertices_local(
                        robot_spec, reso=reso
                    )
                    prep_timing["footprint_s"] = time.perf_counter() - t_fp0

                    obstacle_rects = self._obstacle_rects_for_planner()
                    map_bounds = (0.0, 0.0, float(self.map_state.map_width), float(self.map_state.map_height))
                    pipeline_meta: Dict[str, object] = {
                        "phase": phase,
                        "shape": self.var_robot_shape.get(),
                        "reso": reso,
                        "map_w": self.map_state.map_width,
                        "map_h": self.map_state.map_height,
                        "obstacle_pts": len(ox),
                        "footprint_verts": len(robot_vertices_local),
                        "safety_margin": float(self.safety_margin),
                    }

                    if phase == 3:
                        p1_timing: Dict[str, float] = {}
                        path_stats: Dict[str, object] = {}
                        t_p3_0 = time.perf_counter()
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
                            stop_phase=3,
                            safety_margin=float(self.safety_margin),
                            obstacle_rects=obstacle_rects,
                            map_bounds=map_bounds,
                            timing=p1_timing,
                            se_p3_primitive=self.var_mod_grid_se_p3_primitive.get(),
                            se_p3_collision_mode=self.var_mod_grid_se_p3_collision.get(),
                            dp_objective=self.var_mod_grid_dp_objective.get(),
                            path_stats=path_stats,
                        )
                        t_p3_1 = time.perf_counter()
                        pipeline_meta["se_p3_primitive"] = self.var_mod_grid_se_p3_primitive.get()
                        pipeline_meta["se_p3_collision"] = self.var_mod_grid_se_p3_collision.get()
                        pipeline_meta["dp_objective"] = self.var_mod_grid_dp_objective.get()
                        pipeline_meta["path_stats"] = path_stats
                        self._last_path_stats = dict(path_stats)
                        prims_raw = path_stats.get("primitives", []) or []
                        if prims_raw:
                            self._last_mod_grid_prims = [
                                (str(item["type"]), dict(item["params"]))
                                for item in prims_raw
                                if isinstance(item, dict) and "type" in item and "params" in item
                            ]
                        p1_timing["path_pts"] = float(len(px))
                        self._log_se2_pipeline_report(
                            p1_timing,
                            prep=prep_timing,
                            meta=pipeline_meta,
                            phase3_s=t_p3_1 - t_p3_0,
                            wall_s=time.perf_counter() - wall_t0,
                        )
                    else:
                        p1_timing = {}
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
                            timing=p1_timing,
                        )
                        self._last_path_stats = {
                            "output_pts": len(px),
                            "polyline_length_m": mod_grid_SE_astar.polyline_path_length_m(px, py),
                            "n_primitives": max(0, len(px) - 1),
                            "primitive_length_m": mod_grid_SE_astar.polyline_path_length_m(px, py),
                            "p1_spine_pts": len(px),
                        }
                        self._log_se2_pipeline_report(
                            p1_timing,
                            prep=prep_timing,
                            meta=pipeline_meta,
                            wall_s=time.perf_counter() - wall_t0,
                        )

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
                self._last_planner_phase = 1
            if not px or len(px) < 2:
                if self.var_planner.get() == "mod_grid_se":
                    raise RuntimeError(
                        "No feasible SE(2) path for this footprint at the current map resolution "
                        "(try smaller robot, wider clearance, or coarser grid)"
                    )
                raise RuntimeError(
                    "No feasible path for the circumscribed disk at this map resolution "
                    "(try mod_grid_SE, smaller robot, or wider clearance)"
                )
            self.path_data = (px, py)
            phase_suffix = ""
            if self.robot_state.robot_type == "holonomic" and self.var_planner.get() in ("mod_grid", "mod_grid_se"):
                if self.var_planner.get() == "mod_grid":
                    phase_suffix = f" | mod_grid(disk) phase {int(self.var_mod_grid_phase.get())}"
                    if int(self.var_mod_grid_phase.get()) == 3:
                        phase_suffix += f" dp={self.var_mod_grid_dp_objective.get()}"
                    phase_suffix += f" yaw_fill={self.var_yaw_fill.get()}"
                else:
                    phase_suffix = f" | mod_grid_SE shape {self.var_robot_shape.get()} mode {int(self.var_mod_grid_phase.get())}"
                    if int(self.var_mod_grid_phase.get()) == 3:
                        phase_suffix += f" p3={self.var_mod_grid_se_p3_primitive.get()}"
                        phase_suffix += f" coll={self.var_mod_grid_se_p3_collision.get()}"
                        phase_suffix += f" dp={self.var_mod_grid_dp_objective.get()}"
            self.status_text = (
                f"Path found: {len(px)} pts"
                f"{self._path_stats_summary(self._last_path_stats)}"
                f"{phase_suffix}"
            )
        except Exception as exc:
            self.path_data = None
            self._clear_path_footprint_viz()
            self.status_text = f"Replan failed: {exc}"
        self.render()

    def _snapshot_obstacles(self) -> ObstacleState:
        return ObstacleState(
            rects={k: tuple(v) for k, v in self.obstacles.rects.items()},
            lines={k: [tuple(p) for p in pts] for k, pts in self.obstacles.lines.items()},
            rect_count=int(self.obstacles.rect_count),
            line_count=int(self.obstacles.line_count),
        )

    def _push_obstacle_undo(self) -> None:
        self._obstacle_undo_stack.append(self._snapshot_obstacles())
        if len(self._obstacle_undo_stack) > 50:
            self._obstacle_undo_stack.pop(0)

    def undo_last_obstacle(self) -> None:
        if not self._obstacle_undo_stack:
            self.status_text = "Nothing to undo"
            self.render()
            return
        self.obstacles = self._obstacle_undo_stack.pop()
        self._rect_drag_preview = None
        self.status_text = "Undid last obstacle edit"
        self.render()

    def clear_obstacles(self):
        if self.obstacles.rects or self.obstacles.lines:
            self._push_obstacle_undo()
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
        shape = self.var_robot_shape.get()
        robot = {
            "type": self.robot_state.robot_type,
            "width": self.robot_state.robot_width,
            "length": self.robot_state.robot_length,
            "shape_name": shape,
            "safety_margin": float(self.safety_margin),
            "margin_ge_robot_size": bool(self.var_margin_ge_robot_size.get()),
        }
        obj_path = obj_path_for_shape(shape)
        if obj_path:
            robot["obj_path"] = obj_path
        return {
            "version": 1,
            "map": {
                "width": self.map_state.map_width,
                "height": self.map_state.map_height,
                "resolution": self.map_state.resolution,
            },
            "robot": robot,
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
        self._current_scenario_filename = fname
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

        self.path_data = None
        self._clear_path_footprint_viz()
        self._apply_scenario_from_dict(data, name)
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

        for rid, rect_vals in self.obstacles.rects.items():
            parsed = parse_rect_values(
                rect_vals, map_w=self.map_state.map_width, map_h=self.map_state.map_height
            )
            if abs(parsed.angle_deg) > 1e-6:
                p = Rectangle(
                    (parsed.cx, parsed.cy),
                    parsed.w,
                    parsed.h,
                    angle=parsed.angle_deg,
                    rotation_point="center",
                    facecolor="#777777",
                    edgecolor="black",
                    alpha=0.7,
                )
            else:
                p = Rectangle(
                    (parsed.cx - 0.5 * parsed.w, parsed.cy - 0.5 * parsed.h),
                    parsed.w,
                    parsed.h,
                    facecolor="#777777",
                    edgecolor="black",
                    alpha=0.7,
                )
            self.ax.add_patch(p)
            self.ax.text(parsed.cx, parsed.cy, rid, fontsize=7, color="white", ha="center", va="center")

        for lid, pts in self.obstacles.lines.items():
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                self.ax.plot(xs, ys, color="purple", linewidth=max(1.0, self.line_thickness))
                self.ax.text(xs[0], ys[0], lid, fontsize=7, color="purple")

        if len(self.current_line) >= 2:
            xs, ys = zip(*self.current_line)
            self.ax.plot(xs, ys, color="orange", linewidth=max(1.0, self.line_thickness), alpha=0.8)

        if self._rect_drag_preview is not None:
            px, py, pw, ph, pang = self._rect_drag_preview
            if abs(pang) > 1e-6:
                preview = Rectangle(
                    (px, py),
                    pw,
                    ph,
                    angle=pang,
                    rotation_point="center",
                    fill=False,
                    edgecolor="orange",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85,
                )
            else:
                preview = Rectangle(
                    (px, py),
                    pw,
                    ph,
                    fill=False,
                    edgecolor="orange",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85,
                )
            self.ax.add_patch(preview)

        self._draw_pose(self.start, "green")
        self._draw_pose(self.goal, "red")

        preview_verts = self._shape_preview_vertices_local
        if preview_verts and self.start is not None:
            sx, sy, syaw = self.start
            c, s = math.cos(syaw), math.sin(syaw)
            poly_xy = []
            for lx, ly in preview_verts:
                poly_xy.append((c * lx - s * ly + sx, s * lx + c * ly + sy))
            self.ax.add_patch(
                Polygon(
                    poly_xy,
                    closed=True,
                    fill=True,
                    facecolor="green",
                    edgecolor="darkgreen",
                    alpha=0.25,
                    linewidth=1.5,
                )
            )

        if self.path_data is not None:
            px, py = self.path_data
            self.ax.plot(px, py, "-b", linewidth=2.0)

            n = len(px)
            max_samples = 40
            step = max(1, int(math.ceil(n / float(max_samples))))

            if self._path_footprint_mode == "polygon" and self._robot_vertices_local and self._path_pyaw:
                verts_loc = self._robot_vertices_local
                pyaw = self._path_pyaw
                disk_r = self._disk_radius_for_viz
                for i in range(0, n, step):
                    if i >= len(pyaw):
                        break
                    yaw = float(pyaw[i])
                    c, s = math.cos(yaw), math.sin(yaw)
                    if disk_r is not None and disk_r > 0.0:
                        self.ax.add_patch(
                            Circle(
                                (px[i], py[i]),
                                radius=disk_r,
                                fill=False,
                                edgecolor="deepskyblue",
                                linestyle=":",
                                alpha=0.45,
                                linewidth=0.9,
                            )
                        )
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
            if self.txt_status.index("end-1c") != "1.0":
                self.txt_status.insert(tk.END, "\n")
            self.txt_status.insert(tk.END, f"{self.status_text}")
            self.txt_status.see(tk.END)

    def run(self):
        self.root.mainloop()


def main():
    PlannerWorkbench().run()


if __name__ == "__main__":
    main()
