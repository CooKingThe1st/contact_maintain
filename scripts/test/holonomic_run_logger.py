#!/usr/bin/env python3
"""
Live run logger for revised holonomic Magnum tests.

Writes as we go so Ctrl-C / kill still leaves usable data:
  - run_meta.json          small human-readable config
  - status.log             compact readable progress lines (tail -f friendly)
  - histories_live.json    full Phase7 histories checkpoint (atomic overwrite)
  - histories_final.json   same schema, written on clean/flush exit

Compatible with export_histories / import_histories in test_magnum_holonomic_control.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _history_to_dict(history) -> dict:
    return {
        "times": [float(t) for t in history.times],
        "robot_positions": [pos.tolist() for pos in history.robot_positions],
        "robot_headings": [float(h) for h in history.robot_headings],
        "robot_velocities": [vel.tolist() for vel in history.robot_velocities],
        "intended_positions": [pos.tolist() for pos in history.intended_positions],
        "position_errors": [err.tolist() for err in history.position_errors],
        "desired_headings": [float(h) for h in history.desired_headings],
        "heading_errors": [float(e) for e in history.heading_errors],
        "contact_point_positions": [pos.tolist() for pos in history.contact_point_positions],
        "contact_point_velocities": [vel.tolist() for vel in history.contact_point_velocities],
        "object_positions": [pos.tolist() for pos in history.object_positions],
        "object_velocities": [vel.tolist() for vel in history.object_velocities],
        "object_angular_velocities": [float(omega) for omega in history.object_angular_velocities],
        "contact_forces": [float(f) for f in history.contact_forces],
        "in_contact": [bool(ic) for ic in history.in_contact],
        "v_base_history": [float(v) for v in history.v_base_history],
        "v_ff_history": [float(v) for v in history.v_ff_history],
        "v_pi_history": [float(v) for v in history.v_pi_history],
        "desired_contact_point_speeds": [float(s) for s in history.desired_contact_point_speeds],
        "wheel_velocities": [
            wv.tolist() if len(wv) > 0 else [] for wv in history.wheel_velocities
        ],
        "wheel_cmd_velocities": [
            wv.tolist() if len(wv) > 0 else []
            for wv in getattr(history, "wheel_cmd_velocities", [])
        ],
    }


def histories_payload(histories: Dict[str, Any], t_params: Dict[str, float]) -> dict:
    return {
        "t_params": {k: float(v) for k, v in t_params.items()},
        "histories": {name: _history_to_dict(h) for name, h in histories.items()},
    }


def atomic_write_json(path: Path, data: dict, *, compact: bool = True) -> None:
    """Write JSON via temp file + os.replace so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    kwargs = (
        {"separators": (",", ":"), "indent": None}
        if compact
        else {"indent": 2}
    )
    with open(tmp, "w") as f:
        json.dump(data, f, **kwargs)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class HolonomicRunLogger:
    """
    Checkpoint + status logger for long holonomic runs.

    Call ``tick(...)`` from the control loop; flush is rate-limited.
    """

    def __init__(
        self,
        save_dir: Path,
        *,
        run_tag: str,
        object_name: str,
        meta: Optional[dict] = None,
        checkpoint_interval_s: float = 2.0,
        status_interval_s: float = 1.0,
        get_snapshot: Optional[Callable[[], tuple]] = None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_tag = run_tag
        self.object_name = object_name
        self.checkpoint_interval_s = float(checkpoint_interval_s)
        self.status_interval_s = float(status_interval_s)
        self.get_snapshot = get_snapshot

        self.meta_path = self.save_dir / f"run_meta_{run_tag}_w_{object_name}.json"
        self.status_path = self.save_dir / f"status_{run_tag}_w_{object_name}.log"
        self.live_path = self.save_dir / f"histories_live_{run_tag}_w_{object_name}.json"
        self.final_path = self.save_dir / f"histories_{run_tag}_w_{object_name}.json"

        self._t0 = time.time()
        self._last_checkpoint = 0.0
        self._last_status = 0.0
        self._flush_count = 0
        self._closed = False

        meta_out = {
            "run_tag": run_tag,
            "object_name": object_name,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "paths": {
                "status_log": str(self.status_path),
                "histories_live": str(self.live_path),
                "histories_final": str(self.final_path),
            },
            "meta": meta or {},
        }
        atomic_write_json(self.meta_path, meta_out, compact=False)
        with open(self.status_path, "w") as f:
            f.write(
                f"# holonomic run log  tag={run_tag}  object={object_name}\n"
                f"# started {meta_out['started_at']}\n"
                f"# columns: wall_t  sim_t  obj_xy  yaw_deg  |v|  omega  "
                f"per-robot(c=in_contact F=force)  n_hist\n"
            )
            f.flush()
        self.status(
            f"logger ready | live={self.live_path.name} | "
            f"checkpoint every {self.checkpoint_interval_s:.1f}s"
        )

        atexit.register(self.close)
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            self.status(f"signal {signum} — flushing checkpoint before exit")
            self.flush(force=True, final=True)
            # Re-raise default behavior after flush
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    def status(self, line: str) -> None:
        """Append one human-readable status line (also prints)."""
        wall = time.time() - self._t0
        msg = f"[{wall:7.2f}s] {line}"
        print(msg, flush=True)
        with open(self.status_path, "a") as f:
            f.write(msg + "\n")
            f.flush()

    def format_status_line(
        self,
        *,
        sim_t: float,
        obj_xy,
        obj_yaw_rad: float,
        obj_speed: float,
        obj_omega: float,
        robot_contact: Dict[str, bool],
        robot_force: Dict[str, float],
        n_hist: int,
    ) -> str:
        import math

        yaw_deg = math.degrees(float(obj_yaw_rad))
        parts = [
            f"t={sim_t:6.2f}s",
            f"obj=({float(obj_xy[0]):.3f},{float(obj_xy[1]):.3f})",
            f"yaw={yaw_deg:6.1f}°",
            f"|v|={obj_speed:.3f}",
            f"ω={obj_omega:.3f}",
        ]
        for name in sorted(robot_contact.keys()):
            c = 1 if robot_contact.get(name) else 0
            f = float(robot_force.get(name, 0.0))
            parts.append(f"{name}:c={c} F={f:5.2f}")
        parts.append(f"n_hist={n_hist}")
        return "  ".join(parts)

    def flush(self, *, force: bool = False, final: bool = False) -> None:
        if self._closed and not force:
            return
        if self.get_snapshot is None:
            return
        now = time.time()
        if not force and (now - self._last_checkpoint) < self.checkpoint_interval_s:
            return
        histories, t_params = self.get_snapshot()
        if not histories:
            return
        payload = histories_payload(histories, t_params)
        payload["flush_wall_s"] = now - self._t0
        payload["flush_count"] = self._flush_count + 1
        atomic_write_json(self.live_path, payload, compact=True)
        if final:
            atomic_write_json(self.final_path, payload, compact=True)
        self._flush_count += 1
        self._last_checkpoint = now

    def tick(
        self,
        *,
        sim_t: float,
        obj_state: dict,
        robot_agents: dict,
        n_hist: int,
        force: bool = False,
    ) -> None:
        now = time.time()
        if force or (now - self._last_status) >= self.status_interval_s:
            import numpy as np

            pos = obj_state["position"]
            vel = obj_state["velocity"]
            speed = float(np.linalg.norm(vel[:2])) if hasattr(vel, "__len__") else 0.0
            contacts = {n: bool(a.in_contact) for n, a in robot_agents.items()}
            forces = {n: float(getattr(a, "contact_force", 0.0) or 0.0) for n, a in robot_agents.items()}
            line = self.format_status_line(
                sim_t=sim_t,
                obj_xy=pos,
                obj_yaw_rad=float(obj_state["orientation"]),
                obj_speed=speed,
                obj_omega=float(obj_state["angular_velocity"]),
                robot_contact=contacts,
                robot_force=forces,
                n_hist=n_hist,
            )
            self.status(line)
            self._last_status = now
        self.flush(force=force)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush(force=True, final=True)
            self.status(
                f"logger closed | flushes={self._flush_count} | "
                f"final={self.final_path.name}"
            )
        except Exception as exc:
            print(f"HolonomicRunLogger.close failed: {exc}", flush=True)
