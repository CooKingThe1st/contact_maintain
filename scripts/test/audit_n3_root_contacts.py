#!/usr/bin/env python3
"""Audit cached n=3 Magnum contacts on `root`, then search mid-edge alternatives.

Checks:
  1) GWS ⊇ LS (AFC) with and without tangent forces
  2) robot intended pose outside the polygon (crotch / inside-COM slots)
  3) distance to nearest edge corner (prefer mid-edge)
  4) n_out · (cp − COM) > 0 (outward, not into a notch)

Writes a better cache entry when one is found (source=mid_edge_search).
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg
from shapely.geometry import Point

rospack = rospkg.RosPack()
pkg_path = Path(rospack.get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "src"))
sys.path.insert(0, str(pkg_path / "src" / "legacy"))
sys.path.insert(0, str(pkg_path / "scripts" / "test"))
sys.path.insert(0, str(pkg_path / "scripts" / "test" / "basic_test"))

from contact_maintain.object_bridge import obj_to_generic  # noqa: E402
from load_json_to_obstacles import OBJ_SHAPE_FILES  # noqa: E402
from magnum_contact_cache import (  # noqa: E402
    contacts_from_t_params,
    default_cache_path,
    load_cached_contacts,
    save_cached_contacts,
)
from object_utils import (  # noqa: E402
    ContactPoint,
    EdgeCharacterizer,
    get_reachable_contact_intervals,
)
from stochastic_magnum_finder import check_wrench_space_sufficiency  # noqa: E402

ROBOT_RADIUS = 0.06
MATERIAL_MU = 0.3
BUMPER_MU = 0.5 / MATERIAL_MU  # n=3 AFC target µ_contact = 0.5
FORCE_RANGE = 2.0
MIN_ROBOT_GAP = 2.0 * ROBOT_RADIUS + 0.04


def _setup_pb():
    pyb.connect(pyb.DIRECT)
    pyb.setGravity(0, 0, -9.81)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.setAdditionalSearchPath(str(pkg_path / "urdf"))


def _load_root():
    generic, uid = obj_to_generic(
        obj_path=OBJ_SHAPE_FILES["root"],
        shape_name="root",
        position=(0.0, 0.0, 0.2),
        orientation=0.0,
        mass=1.0,
        lateral_friction=MATERIAL_MU,
        blind_test=True,
    )
    generic.set_material_friction(
        MATERIAL_MU, sync_legacy_lateral=True, sync_legacy_static=True
    )
    mu_c = generic.apply_bumper_contact_model(BUMPER_MU)
    return generic, uid, mu_c


def _t_on_edge(edge, local_t):
    s0 = float(edge["start_param"]) % 1.0
    s1 = float(edge["end_param"]) % 1.0
    loc = float(np.clip(local_t, 0.0, 1.0))
    if s1 >= s0:
        return (s0 + loc * (s1 - s0)) % 1.0
    span = (1.0 - s0) + s1
    return (s0 + loc * span) % 1.0


def _edge_of_t(edges, t):
    t = float(t) % 1.0
    for i, e in enumerate(edges):
        s0, s1 = float(e["start_param"]), float(e["end_param"])
        if s1 >= s0:
            if s0 - 1e-9 <= t <= s1 + 1e-9:
                span = max(s1 - s0, 1e-12)
                return i, (t - s0) / span, e
        else:
            # wrap
            if t >= s0 or t <= s1:
                span = (1.0 - s0) + s1
                loc = (t - s0) % 1.0
                return i, loc / max(span, 1e-12), e
    return None, None, None


def _contacts_from_ts(obj, param, ts):
    return contacts_from_t_params(obj, ts, contact_point_parameterization=param)


def _metrics(obj, contacts, edges, reachable):
    poly = obj.geometry
    com = np.array([obj.get_centroid().x, obj.get_centroid().y], dtype=float)
    rows = []
    intended = []
    for c in contacts:
        t = float(c.parameter) % 1.0
        ei, loc, ed = _edge_of_t(edges, t)
        cp = np.asarray(c.position, dtype=float)[:2]
        n_in = np.asarray(c.normal_inward, dtype=float)[:2]
        n_out = np.asarray(c.normal_outward, dtype=float)[:2]
        n_in = n_in / (np.linalg.norm(n_in) + 1e-12)
        n_out = n_out / (np.linalg.norm(n_out) + 1e-12)
        r = cp - com
        r_n = float(np.linalg.norm(r))
        nout_dot = float(np.dot(n_out, r / (r_n + 1e-12)))
        ip = cp + ROBOT_RADIUS * n_out
        intended.append(ip)
        inside = bool(poly.contains(Point(float(ip[0]), float(ip[1]))))
        corner_clr = None if loc is None else min(float(loc), 1.0 - float(loc))
        on_reach = False
        if reachable:
            for a, b in reachable:
                if a - 1e-9 <= t <= b + 1e-9:
                    on_reach = True
                    break
        rows.append(
            {
                "t": t,
                "edge": ei,
                "local_t": loc,
                "corner_clr": corner_clr,
                "len": None if ed is None else float(ed.get("length", 0.0)),
                "r": r_n,
                "nout_dot": nout_dot,
                "inside": inside,
                "reachable": on_reach,
                "cp": cp,
                "n_in": n_in,
                "intended": ip,
            }
        )
    gaps = []
    for a, b in itertools.combinations(intended, 2):
        gaps.append(float(np.linalg.norm(a - b)))
    min_gap = min(gaps) if gaps else 0.0
    return rows, min_gap, com


def _kappa_xy(contacts, n_angles=360):
    ns = []
    for c in contacts:
        n = np.asarray(c.normal_inward, dtype=float)[:2]
        ns.append(n / (np.linalg.norm(n) + 1e-12))
    vals = []
    for th in np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False):
        u = np.array([math.cos(th), math.sin(th)])
        vals.append(sum(max(0.0, float(np.dot(u, n))) for n in ns))
    return float(min(vals))


def _afc(obj, contacts):
    nrm = check_wrench_space_sufficiency(
        contacts,
        obj,
        threshold=1.0,
        n_ellipse_samples=48,
        force_range_scalar=FORCE_RANGE,
        enable_tangent_forces=False,
        verbose=False,
    )
    tan = check_wrench_space_sufficiency(
        contacts,
        obj,
        threshold=1.0,
        n_ellipse_samples=48,
        force_range_scalar=FORCE_RANGE,
        enable_tangent_forces=True,
        verbose=False,
    )
    return bool(nrm.get("satisfied")), bool(tan.get("satisfied"))


def _score(rows, min_gap, kappa, afc_tan, afc_n):
    if not afc_tan:
        return -1e9
    if any(r["inside"] for r in rows):
        return -1e8
    if min_gap < MIN_ROBOT_GAP:
        return -1e7
    if kappa < 0.2:
        return -1e5
    clrs = [r["corner_clr"] for r in rows if r["corner_clr"] is not None]
    min_clr = min(clrs) if clrs else 0.0
    min_r = min(r["r"] for r in rows)
    # Prefer mid-edge, exterior, decent lever arm, then kappa.
    return (
        100.0 * min_clr
        + 20.0 * min(min_r, 0.45)
        + 8.0 * kappa
        + (5.0 if afc_n else 0.0)
        + 2.0 * min(min_gap, 0.6)
    )


def _print_cfg(label, ts, rows, min_gap, kappa, afc_n, afc_t, score=None):
    print(f"\n=== {label} ===")
    print(f"  t_params = {[round(float(t), 4) for t in ts]}")
    print(
        f"  AFC normal-only={afc_n}  AFC+tangent={afc_t}  "
        f"κ_xy={kappa:.3f}  min_robot_gap={min_gap:.3f} m"
        + (f"  score={score:.2f}" if score is not None else "")
    )
    for i, r in enumerate(rows):
        loc = "n/a" if r["local_t"] is None else f"{r['local_t']:.2f}"
        clr = "n/a" if r["corner_clr"] is None else f"{r['corner_clr']:.2f}"
        print(
            f"  c{i}: t={r['t']:.3f} edge={r['edge']} local={loc} "
            f"corner_clr={clr} |r|={r['r']:.3f} n_out·r̂={r['nout_dot']:+.3f} "
            f"inside={r['inside']} reachable={r['reachable']}"
        )


def main():
    _setup_pb()
    obj, _uid, mu_c = _load_root()
    print(f"root loaded  µ_contact={mu_c:.4f}  mass={obj.mass}  "
          f"area={obj.geometry.area:.3f}")
    char = EdgeCharacterizer(obj)
    edges = char.edges
    param = char.parameterization
    print(f"logical edges: {len(edges)}")
    for i, e in enumerate(edges):
        print(
            f"  e{i}: t=[{e['start_param']:.3f},{e['end_param']:.3f}] "
            f"len={e['length']:.3f} m"
        )
    reachable = get_reachable_contact_intervals(obj.geometry, ROBOT_RADIUS)

    cached = load_cached_contacts("root", 3)
    cached_ts = list(cached["t_params"]) if cached else []
    print(f"\ncache: {cached}")

    def eval_ts(ts, *, force_afc=False):
        contacts = _contacts_from_ts(obj, param, ts)
        rows, min_gap, _com = _metrics(obj, contacts, edges, reachable)
        kappa = _kappa_xy(contacts)
        geom_bad = (
            any(r["inside"] for r in rows)
            or min_gap < MIN_ROBOT_GAP
            or any(r["nout_dot"] < 0.02 for r in rows)
        )
        if geom_bad and not force_afc:
            return contacts, rows, min_gap, kappa, False, False, _score(
                rows, min_gap, kappa, False, False
            )
        afc_n, afc_t = _afc(obj, contacts)
        sc = _score(rows, min_gap, kappa, afc_t, afc_n)
        return contacts, rows, min_gap, kappa, afc_n, afc_t, sc

    if cached_ts:
        pack = eval_ts(cached_ts, force_afc=True)
        _print_cfg("CACHED n=3", cached_ts, pack[1], pack[2], pack[3], pack[4], pack[5], pack[6])

    n4 = load_cached_contacts("root", 4)
    if n4:
        pack4 = eval_ts(n4["t_params"], force_afc=True)
        _print_cfg("CACHED n=4 (benchmark slots)", n4["t_params"], pack4[1], pack4[2], pack4[3], pack4[4], pack4[5], pack4[6])

    # Mid-edge search: 3 distinct logical edges, local t near 0.5 then refine.
    print("\n--- mid-edge search (3 distinct edges) ---")
    local_grid = (0.50,)
    refine_grid = (0.35, 0.50, 0.65)
    candidates = []
    n_edges = len(edges)
    tested = 0
    for eidx in itertools.combinations(range(n_edges), 3):
        for locs in itertools.product(local_grid, repeat=3):
            ts = [_t_on_edge(edges[ei], loc) for ei, loc in zip(eidx, locs)]
            tested += 1
            _c, rows, min_gap, kappa, afc_n, afc_t, sc = eval_ts(ts)
            if sc > -1e5:
                candidates.append((sc, ts, rows, min_gap, kappa, afc_n, afc_t, eidx, locs))

    print(f"midpoint triples tested={tested}  passing geometry+AFC={len(candidates)}")
    candidates.sort(key=lambda x: -x[0])

    # Refine top midpoint triples on a 3^3 local grid.
    refined = list(candidates)
    seen = {tuple(round(t, 5) for t in c[1]) for c in candidates}
    for base in candidates[:12]:
        eidx = base[7]
        for locs in itertools.product(refine_grid, repeat=3):
            ts = [_t_on_edge(edges[ei], loc) for ei, loc in zip(eidx, locs)]
            key = tuple(round(t, 5) for t in ts)
            if key in seen:
                continue
            seen.add(key)
            _c, rows, min_gap, kappa, afc_n, afc_t, sc = eval_ts(ts)
            if sc > -1e5:
                refined.append((sc, ts, rows, min_gap, kappa, afc_n, afc_t, eidx, locs))

    refined.sort(key=lambda x: -x[0])
    print(f"after quartile refine: {len(refined)} feasible configs")
    for k, c in enumerate(refined[:8], 1):
        _print_cfg(
            f"CANDIDATE #{k} edges={c[7]} local={tuple(round(x, 2) for x in c[8])}",
            c[1], c[2], c[3], c[4], c[5], c[6], c[0],
        )

    if not refined:
        print("\nNo better feasible n=3 config found; cache unchanged.")
        return 1

    best = refined[0]
    cached_score = pack[6] if cached_ts else -1e12
    print(f"\nbest score={best[0]:.2f}  cached score={cached_score:.2f}")
    if best[0] > cached_score + 1.0:
        path = save_cached_contacts(
            "root",
            3,
            best[1],
            cache_path=default_cache_path(),
            mu_contact=mu_c,
            tangent_required=True,
            source="mid_edge_search",
        )
        print(f"WROTE cache {path}")
        print(f"  new t_params={[round(float(t), 6) for t in best[1]]}")
    else:
        print("Cached config is already as good or better on this metric; not overwriting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
