#!/usr/bin/env python3
"""
Verify Problem B bounds from docs/afc_problem_B.md:

1. kappa_xy and lambda_min for ideal 4-contact placements
2. Translational LP containment at lambda_min (T1)
3. Full AFC at lambda_min (T2) — circle should fail T2, pass T1
4. Stochastic search at lambda_min for T2-non-degenerate shapes (assumes robust C)
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

rospack = __import__("rospkg")
pkg_path = Path(rospack.RosPack().get_path("contact_maintain"))
sys.path.insert(0, str(pkg_path / "src"))
sys.path.insert(0, str(pkg_path / "src" / "legacy"))

from contact_optimizer_utils_test_ver import (  # noqa: E402
    check_wrench_space_sufficiency,
    find_the_magnum_stochastic,
)
from object_utils import ContactPoint, EdgeCharacterizer, create_standard_objects  # noqa: E402


def inward_normals_from_contacts(contacts):
    return [np.asarray(c.normal_inward, dtype=float) for c in contacts]


def kappa_xy(normals, n_angles=720):
  """Min over unit directions of sum_i max(0, u·n_i)."""
  normals = [n / (np.linalg.norm(n) + 1e-12) for n in normals]
  vals = []
  for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
    u = np.array([math.cos(theta), math.sin(theta)])
    vals.append(sum(max(0.0, float(np.dot(u, n))) for n in normals))
  return min(vals)


def kappa_projection(contacts, axes, n_angles=720):
  """kappa_p for projection axes (0=Fx, 1=Fy, 2=tau)."""
  cols = []
  for c in contacts:
    w = c.calculate_contact_wrench(1.0, 0.0, friction_constraint=True)
    cols.append(np.array([w["force_x"], w["force_y"], w["torque"]])[list(axes)])
  hs = [h / (np.linalg.norm(h) + 1e-12) for h in cols]
  vals = []
  for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
    u = np.array([math.cos(theta), math.sin(theta)])
    vals.append(sum(max(0.0, float(np.dot(u, h))) for h in hs))
  return min(vals)


def lambda_min_full_afc(kappas, threshold=1.0):
  kmin = min(kappas.values())
  if kmin < 1e-6:
    return float("inf")
  return threshold / kmin


def lp_translational_disk_containment(contacts, threshold, force_range_scalar):
  """T1: every point on Fx-Fy LS circle boundary feasible via capped normals?"""
  obj = contacts[0].object_ref
  nf = obj.mass * 9.81
  f_max = threshold * obj.static_friction * nf
  f_cap = force_range_scalar * obj.static_friction * nf
  normals = inward_normals_from_contacts(contacts)
  n = len(contacts)
  angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
  for th in angles:
    p = f_max * np.array([math.cos(th), math.sin(th)])
    G2 = np.column_stack(normals)
    res = linprog(
      c=np.zeros(n),
      A_eq=G2,
      b_eq=p,
      bounds=[(0.0, f_cap)] * n,
      method="highs",
    )
    if not res.success:
      return False
  return True


def contacts_from_logical_edges(edge_characterizer, edge_indices, local_ts=None):
  """Map logical edge index + local t in [0,1] to global boundary parameter."""
  if local_ts is None:
    local_ts = [0.5] * len(edge_indices)
  obj = edge_characterizer.parameterization.object
  contacts = []
  for edge_idx, local_t in zip(edge_indices, local_ts):
    edge = edge_characterizer.edges[edge_idx]
    t_global = edge["start_param"] + local_t * (edge["end_param"] - edge["start_param"])
    info = edge_characterizer.parameterization.get_contact_info(t_global)
    contacts.append(
        ContactPoint(
            position=info["point"],
            tangent=info["tangent"],
            normal_outward=info["normal_outward"],
            normal_inward=info["normal_inward"],
            parameter=t_global,
            object_ref=obj,
        )
    )
  return contacts


def main():
  threshold = 1.0
  margin = 0.05
  objs = create_standard_objects()

  # Ideal placements (edge indices depend on characterization)
  cases = {
    "rectangle": (objs["rectangle"], [0, 1, 2, 3], [0.5, 0.5, 0.5, 0.5]),
    "triangle": (objs["triangle"], [0, 1, 2, 0], [0.5, 0.5, 0.5, 0.9]),
    "circle": (objs["circle"], [0, 16, 32, 48], [0.5, 0.5, 0.5, 0.5]),
  }

  print("=" * 72)
  print("kappa_p and lambda_min for ideal 4-tuples (normal-only)")
  print("=" * 72)
  for name, (obj, eidx, lts) in cases.items():
    ec = EdgeCharacterizer(obj, force_magnitude=1.0)
    contacts = contacts_from_logical_edges(ec, eidx, lts)
    normals = inward_normals_from_contacts(contacts)
    kxy = kappa_xy(normals)
    kfxt = kappa_projection(contacts, (0, 2))
    kfyt = kappa_projection(contacts, (1, 2))
    kappas = {"xy": kxy, "Fxt": kfxt, "Fyt": kfyt}
    lam_min = lambda_min_full_afc(kappas, threshold)
    lam_test = (lam_min + margin) if math.isfinite(lam_min) else 5.0
    t1 = lp_translational_disk_containment(contacts, threshold, lam_test)
    t2 = check_wrench_space_sufficiency(
      contacts, obj, threshold=threshold, force_range_scalar=lam_test,
      enable_tangent_forces=False, verbose=False,
    )["satisfied"]
    lam_str = f"{lam_min:.4f}" if math.isfinite(lam_min) else "inf"
    print(f"{name:12} kxy={kxy:.4f}  kFxt={kfxt:.4f}  kFyt={kfyt:.4f}  "
          f"lam_min={lam_str}  T1_LP={t1}  T2_AFC={t2}")

  print("\n" + "=" * 72)
  print("Stochastic search at lambda_min + margin (engineering, 10s)")
  print("=" * 72)
  stochastic_shapes = {
    "rectangle": (False, None),
    "trapezoid": (False, None),
    "triangle": (False, None),
    "circle": (True, None),  # T2-degenerate normal-only; needs tangent
    "crescent": (True, None),
    "narrow_triangle": (True, None),
  }
  for name, (need_tangent, _) in stochastic_shapes.items():
    obj = objs[name]
    ec = EdgeCharacterizer(obj, force_magnitude=1.0)
  # recompute lambda from a generic 4-corner-like tuple on first 4 edges if possible
    n_edges = len(ec.edges)
    if n_edges >= 4:
      eidx = [0, n_edges // 4, n_edges // 2, 3 * n_edges // 4]
    else:
      eidx = list(range(n_edges)) + [0] * (4 - n_edges)
    contacts = contacts_from_logical_edges(ec, eidx[:4], [0.5] * 4)
    kxy = kappa_xy(inward_normals_from_contacts(contacts))
    lam = threshold / max(kxy, 1e-6) + margin
    r = find_the_magnum_stochastic(
      obj, verbose=False, threshold=threshold, timeout=10.0,
      force_range_scalar=lam, robot_radius=0.06, theory_mode=False,
      used_tangent_as_fallback=need_tangent,
    )
    print(f"{name:16} lambda={lam:.3f}  tangent={need_tangent}  "
          f"success={r['success']}  configs={r['configs_tested']}")


if __name__ == "__main__":
  main()
