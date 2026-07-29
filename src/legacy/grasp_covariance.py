"""
Wrench covariance matrix and shape degeneracy index for normal-only grasping.

Discretizes the continuous boundary integral from docs/afc_problem_B.md Section 9:

    M = ∮ g(x) g(x)^T ds,   D = σ₁ / σ₃

Section 10 spectral bound (T = threshold):

    f_max = max_s u₃ᵀ g(s);  K_tight = f_max²/σ₃;  K_deriv from per-edge ∫(f')²;
    K = min(K_tight, K_deriv) for λ_floor = (T/(4√(K σ₁)))√D

Sampling is strictly interior to each boundary edge (endpoints excluded).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from object_utils import ContactPointParameterization, GenericObject

ComInput = Optional[Union[Sequence[float], Tuple[float, float], np.ndarray]]

# Degeneracy labels aligned with docs/afc_problem_B.md Section 9–11.
CLASS_WELL_BEHAVED = "well_behaved"
CLASS_SOFT_DEGENERATE = "soft_degenerate"
CLASS_STRICT_DEGENERATE = "strict_degenerate"

# Gate: σ₃ below this ⇒ strict degeneracy; do not trust λ_floor (Section 11).
DEFAULT_SIGMA3_STRICT_EPS = 1e-5
# D at or above this ⇒ soft degeneracy; enable tangent fallback before search.
DEFAULT_SOFT_DEGENERACY_THRESHOLD = 100.0
# If D exceeds this, λ_floor from discrete Sobolev is untrusted (near-singularity).
DEFAULT_D_UNTRUSTED_THRESHOLD = 500.0


def _resolve_com(obj: GenericObject, com: ComInput) -> np.ndarray:
    if com is None:
        centroid = obj.get_centroid()
        return np.array([centroid.x, centroid.y], dtype=float)
    com_arr = np.asarray(com, dtype=float).reshape(2)
    return com_arr


def _max_radius_from_com(param: ContactPointParameterization, com: np.ndarray) -> float:
    coords = param.boundary_coords
    radii = np.linalg.norm(coords - com, axis=1)
    return float(np.max(radii))


def _classify_degeneracy(
    degeneracy_index: float,
    sigma3: float,
    *,
    sigma3_strict_eps: float,
    soft_threshold: float,
) -> str:
    if sigma3 < sigma3_strict_eps or not np.isfinite(degeneracy_index):
        return CLASS_STRICT_DEGENERATE
    if degeneracy_index >= soft_threshold:
        return CLASS_SOFT_DEGENERATE
    return CLASS_WELL_BEHAVED


def _lambda_floor_trusted(
    sigma3: float,
    degeneracy_index: float,
    f_max: float,
    *,
    sigma3_strict_eps: float,
    d_untrusted_threshold: float,
    f_max_strict_eps: float = 1e-8,
) -> bool:
    """
  Discrete Sobolev λ_floor collapses when σ₃ → 0 but σ₃ stays above mesh noise.
  Trust λ_floor only when σ₃ is safe and D is not in the near-singular regime.
  """
    if sigma3 < sigma3_strict_eps:
        return False
    if not np.isfinite(degeneracy_index) or degeneracy_index >= d_untrusted_threshold:
        return False
    if f_max <= f_max_strict_eps:
        return False
    return True

def _edge_df_ds(
    p1: np.ndarray,
    p2: np.ndarray,
    normal_inward: np.ndarray,
    u3: np.ndarray,
    com: np.ndarray,
    *,
    normalize_radius: bool,
    max_radius: float,
) -> float:
    """
  df/ds for f(s) = u₃ᵀ g(s) along a straight edge (inward n constant, τ linear in s).

  Only the torque component varies: df/ds = u₃,τ · dτ/ds with dτ/ds = (t×n)·ẑ.
  """
    edge = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(edge))
    if length < 1e-12:
        return 0.0
    tangent = edge / length
    n = np.asarray(normal_inward, dtype=float)
    scale = max(max_radius, 1e-12) if normalize_radius else 1.0
    tau_slope = (tangent[0] * n[1] - tangent[1] * n[0]) / scale
    return float(u3[2] * tau_slope)


def _wrench_at_point(
    point: np.ndarray,
    normal_inward: np.ndarray,
    com: np.ndarray,
    u3: np.ndarray,
    *,
    normalize_radius: bool,
    max_radius: float,
) -> float:
    g = local_normal_wrench(
        point, normal_inward, com,
        normalize_radius=normalize_radius, max_radius=max_radius,
    )
    return float(np.dot(u3, g))


def _sobolev_K_segmentwise(
    param: ContactPointParameterization,
    u3: np.ndarray,
    com: np.ndarray,
    *,
    normalize_radius: bool,
    max_radius: float,
    sigma3: float,
    sigma3_eps: float = 1e-12,
) -> Tuple[float, float]:
    """
  Sobolev constant K on a piecewise-linear boundary.

  ∫ (df/ds)² ds is summed **per edge** using the analytical derivative on each
  straight segment. Corner jumps in f are excluded — they are not part of the
  C¹ Sobolev energy and previously inflated K via periodic finite differences.

  Returns (K, f_max) with f_max taken over mesh vertices and interior samples.
  """
    if sigma3 < sigma3_eps or param.total_length < 1e-12:
        return float("inf"), 0.0

    u3 = np.asarray(u3, dtype=float)
    df_sq_integral = 0.0
    f_peak = 0.0

    for seg_i in range(param.n_segments):
        seg_len = param.segment_lengths[seg_i]
        if seg_len < 1e-12:
            continue
        p1 = param.boundary_coords[seg_i]
        p2 = param.boundary_coords[seg_i + 1]
        dist_mid = param.cumulative_distances[seg_i] + 0.5 * seg_len
        t_mid = dist_mid / param.total_length
        n_in = param.get_normal_vector(t_mid, outward=False)

        df_ds = _edge_df_ds(
            p1, p2, n_in, u3, com,
            normalize_radius=normalize_radius, max_radius=max_radius,
        )
        df_sq_integral += df_ds * df_ds * seg_len

    # Vertices: evaluate f with each incident edge's normal (n jumps at corners).
    for v_idx in range(param.n_segments):
        pt = param.boundary_coords[v_idx]
        for seg_i in (v_idx - 1, v_idx):
            si = seg_i % param.n_segments
            if v_idx == si:
                dist = param.cumulative_distances[si]
            else:
                dist = param.cumulative_distances[si + 1]
            t_g = max(0.0, min(1.0, dist / param.total_length))
            n_in = param.get_normal_vector(t_g, outward=False)
            f_peak = max(
                f_peak,
                _wrench_at_point(
                    pt, n_in, com, u3,
                    normalize_radius=normalize_radius, max_radius=max_radius,
                ),
            )

    L = float(param.total_length)
    K = (2.0 / L) + (L / (2.0 * sigma3)) * df_sq_integral
    return float(K), float(f_peak)


def _sobolev_K_tight(f_max: float, sigma3: float, boundary_length: float) -> float:
    """
  Minimum K satisfying f_max ≤ √(K σ₃) on the sampled boundary.

  Gives the strongest spectral floor: λ_lb = T / (4 √(K σ₃)) = T / (4 f_max).
  """
    if sigma3 < 1e-12 or f_max <= 0.0:
        return float("inf")
    K_peak = (f_max * f_max) / sigma3
    K_floor = 2.0 / max(boundary_length, 1e-12)
    return float(max(K_peak, K_floor))


def _sobolev_K_from_boundary(
    f_vals: Sequence[float],
    ds_vals: Sequence[float],
    sigma3: float,
    boundary_length: float,
    *,
    sigma3_eps: float = 1e-12,
) -> float:
    """
  Legacy periodic central-difference K (overestimates on polygons).

  Prefer _sobolev_K_segmentwise. Kept for comparison/debugging.
  """
    if sigma3 < sigma3_eps or boundary_length < 1e-12:
        return float("inf")
    n = len(f_vals)
    if n < 2:
        return float("inf")

    df_sq_integral = 0.0
    for i in range(n):
        f_prev = f_vals[(i - 1) % n]
        f_next = f_vals[(i + 1) % n]
        ds_prev = ds_vals[(i - 1) % n]
        ds_here = ds_vals[i]
        arc_span = ds_prev + ds_here
        if arc_span < 1e-12:
            continue
        df_ds = (f_next - f_prev) / arc_span
        df_sq_integral += df_ds * df_ds * ds_here

    L = boundary_length
    return float((2.0 / L) + (L / (2.0 * sigma3)) * df_sq_integral)


def spectral_lambda_lower_bound(
    sigma1: float,
    sigma3: float,
    sobolev_K: float,
    *,
    threshold: float = 1.0,
    sigma3_eps: float = 1e-12,
) -> float:
    """
  Spectral floor from Section 10 (original form):

      λ_floor = (T / (4 √(K σ₁))) √D = T / (4 √(K σ₃)),   D = σ₁/σ₃.

  K should be the smallest valid Sobolev constant (min of peak and derivative
  estimates) so the floor is as strong as possible.
  """
    if sigma3 < sigma3_eps or sigma1 < sigma3_eps:
        return float("inf")
    if sobolev_K <= 0 or not np.isfinite(sobolev_K):
        return float("inf")
    D = sigma1 / sigma3
    return float(
        threshold * np.sqrt(D) / (4.0 * np.sqrt(sobolev_K * sigma1))
    )


def local_normal_wrench(
    point: np.ndarray,
    normal_inward: np.ndarray,
    com: np.ndarray,
    *,
    normalize_radius: bool,
    max_radius: float,
) -> np.ndarray:
    """
  Unit-normal wrench column g(x) = (n_x, n_y, τ) relative to com.

  τ = r_x n_y - r_y n_x with r = x - com (optionally scaled by max_radius).
  """
    r = np.asarray(point, dtype=float) - com
    if normalize_radius:
        scale = max(max_radius, 1e-12)
        r = r / scale
    n = np.asarray(normal_inward, dtype=float)
    tau = r[0] * n[1] - r[1] * n[0]
    return np.array([n[0], n[1], tau], dtype=float)


def calculate_grasp_covariance(
    obj: GenericObject,
    com: ComInput = None,
    *,
    samples_per_edge: int = 4,
    normalize_radius: bool = True,
    threshold: float = 1.0,
    sigma3_strict_eps: float = DEFAULT_SIGMA3_STRICT_EPS,
    soft_degeneracy_threshold: float = DEFAULT_SOFT_DEGENERACY_THRESHOLD,
    d_untrusted_threshold: float = DEFAULT_D_UNTRUSTED_THRESHOLD,
) -> Dict[str, Any]:
    """
  Integrate the wrench covariance matrix M over the object boundary.

  Degeneracy gate (Section 11): if σ₃ < sigma3_strict_eps, the shape is
  strictly degenerate — λ_floor is not computed (set to ∞). Otherwise K and
  λ_floor are computed, but lambda_floor_trusted is False when D is huge
  (Sobolev singularity: K ∝ 1/σ₃ cancels √D).

  Args:
      obj: GenericObject with a Shapely polygon boundary.
      com: Center of torque (x, y). Defaults to geometry centroid.
      samples_per_edge: Interior samples per boundary edge (>= 1).
      normalize_radius: If True, scale r = x - com by max boundary radius.
      threshold: Engineering threshold T for λ_floor (default 1).
      sigma3_strict_eps: σ₃ gate — below this, strict degeneracy, skip λ_floor.
      soft_degeneracy_threshold: D >= this ⇒ soft_degenerate classification.
      d_untrusted_threshold: D above this ⇒ lambda_floor_trusted=False.

  Returns:
      dict including M, eigenvalues, D, sobolev_K, lambda_shape_lower_bound,
      lambda_floor_computed, lambda_floor_trusted, degeneracy_gate,
      classification, ...
  """
    if samples_per_edge < 1:
        raise ValueError("samples_per_edge must be >= 1")

    param = ContactPointParameterization(obj)
    com_xy = _resolve_com(obj, com)
    max_radius = _max_radius_from_com(param, com_xy)

    M = np.zeros((3, 3), dtype=float)
    g_samples: list = []
    ds_samples: list = []
    n_samples = 0
    total_weight = 0.0

    for seg_i in range(param.n_segments):
        seg_len = param.segment_lengths[seg_i]
        if seg_len < 1e-12:
            continue

        p1 = param.boundary_coords[seg_i]
        p2 = param.boundary_coords[seg_i + 1]
        seg_start = param.cumulative_distances[seg_i]
        ds = seg_len / float(samples_per_edge)

        for j in range(1, samples_per_edge + 1):
            local_t = j / (samples_per_edge + 1)
            point = p1 + local_t * (p2 - p1)
            dist_along = seg_start + local_t * seg_len
            t_global = dist_along / param.total_length
            n_in = param.get_normal_vector(t_global, outward=False)

            g = local_normal_wrench(
                point,
                n_in,
                com_xy,
                normalize_radius=normalize_radius,
                max_radius=max_radius,
            )
            M += np.outer(g, g) * ds
            g_samples.append(g)
            ds_samples.append(ds)
            n_samples += 1
            total_weight += ds

    boundary_length = float(param.total_length)
    eigenvalues_all, eigenvectors = np.linalg.eigh(M)
    order = np.argsort(eigenvalues_all)[::-1]
    eigenvalues = eigenvalues_all[order]
    eigenvectors = eigenvectors[:, order]
    sigma1, sigma2, sigma3 = (float(eigenvalues[i]) for i in range(3))
    u3 = eigenvectors[:, 2]

    f_vals = [float(np.dot(u3, g)) for g in g_samples]
    f_max_samples = max(f_vals) if f_vals else 0.0

    if sigma3 < sigma3_strict_eps:
        degeneracy_index = float("inf")
        degeneracy_gate = "strict_sigma3"
        lambda_floor_computed = False
        sobolev_K = float("inf")
        sobolev_K_tight = float("inf")
        sobolev_K_deriv = float("inf")
        lambda_shape_lower_bound = float("inf")
        lambda_floor_trusted = False
    else:
        degeneracy_index = sigma1 / sigma3
        degeneracy_gate = "evaluated"
        lambda_floor_computed = True
        sobolev_K_deriv, f_max_vertices = _sobolev_K_segmentwise(
            param,
            u3,
            com_xy,
            normalize_radius=normalize_radius,
            max_radius=max_radius,
            sigma3=sigma3,
            sigma3_eps=sigma3_strict_eps,
        )
        f_max = max(f_max_samples, f_max_vertices)
        sobolev_K_tight = _sobolev_K_tight(f_max, sigma3, boundary_length)
        sobolev_K = float(min(sobolev_K_tight, sobolev_K_deriv))
        lambda_shape_lower_bound = spectral_lambda_lower_bound(
            sigma1,
            sigma3,
            sobolev_K,
            threshold=threshold,
            sigma3_eps=sigma3_strict_eps,
        )
        lambda_floor_trusted = _lambda_floor_trusted(
            sigma3,
            degeneracy_index,
            f_max,
            sigma3_strict_eps=sigma3_strict_eps,
            d_untrusted_threshold=d_untrusted_threshold,
        )

    classification = _classify_degeneracy(
        degeneracy_index,
        sigma3,
        sigma3_strict_eps=sigma3_strict_eps,
        soft_threshold=soft_degeneracy_threshold,
    )

    return {
        "M": M,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "u3": u3,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
        "degeneracy_index": degeneracy_index,
        "f_max": f_max,
        "sobolev_K": sobolev_K,
        "sobolev_K_tight": sobolev_K_tight,
        "sobolev_K_deriv": sobolev_K_deriv,
        "lambda_shape_lower_bound": lambda_shape_lower_bound,
        "lambda_floor_computed": lambda_floor_computed,
        "lambda_floor_trusted": lambda_floor_trusted,
        "degeneracy_gate": degeneracy_gate,
        "classification": classification,
        "com": com_xy,
        "max_radius": max_radius,
        "samples_per_edge": samples_per_edge,
        "n_samples": n_samples,
        "total_weight": total_weight,
        "normalize_radius": normalize_radius,
        "boundary_length": boundary_length,
        "threshold": float(threshold),
        "sigma3_strict_eps": float(sigma3_strict_eps),
    }


def recommend_tangent_fallback(
    covariance_result: Dict[str, Any],
    *,
    soft_degeneracy_threshold: float = DEFAULT_SOFT_DEGENERACY_THRESHOLD,
    sigma3_strict_eps: float = DEFAULT_SIGMA3_STRICT_EPS,
) -> Dict[str, Any]:
    """
    Section 11 pipeline gate: should ``used_tangent_as_fallback`` be True?

    Returns recommend_tangent_fallback, reason, and the thresholds used.
    """
    sigma3 = float(covariance_result["sigma3"])
    classification = covariance_result["classification"]
    degeneracy_index = float(covariance_result["degeneracy_index"])

    if (
        classification == CLASS_STRICT_DEGENERATE
        or sigma3 < sigma3_strict_eps
        or not np.isfinite(degeneracy_index)
    ):
        return {
            "recommend_tangent_fallback": True,
            "reason": "strict_sigma3",
            "soft_degeneracy_threshold": float(soft_degeneracy_threshold),
            "sigma3_strict_eps": float(sigma3_strict_eps),
        }

    if (
        classification == CLASS_SOFT_DEGENERATE
        or degeneracy_index >= soft_degeneracy_threshold
    ):
        return {
            "recommend_tangent_fallback": True,
            "reason": "soft_degenerate",
            "soft_degeneracy_threshold": float(soft_degeneracy_threshold),
            "sigma3_strict_eps": float(sigma3_strict_eps),
        }

    return {
        "recommend_tangent_fallback": False,
        "reason": "well_behaved",
        "soft_degeneracy_threshold": float(soft_degeneracy_threshold),
        "sigma3_strict_eps": float(sigma3_strict_eps),
    }


def screening_fields_for_log(
    covariance_result: Dict[str, Any],
    recommendation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flat dict of degeneracy screening fields for stdout / CSV logging."""
    if recommendation is None:
        recommendation = recommend_tangent_fallback(covariance_result)

    degeneracy_index = float(covariance_result["degeneracy_index"])
    lambda_floor = float(covariance_result["lambda_shape_lower_bound"])

    return {
        "sigma1": float(covariance_result["sigma1"]),
        "sigma2": float(covariance_result["sigma2"]),
        "sigma3": float(covariance_result["sigma3"]),
        "degeneracy_index": degeneracy_index if np.isfinite(degeneracy_index) else float("inf"),
        "degeneracy_classification": covariance_result["classification"],
        "degeneracy_gate": covariance_result["degeneracy_gate"],
        "lambda_floor": lambda_floor if np.isfinite(lambda_floor) else float("inf"),
        "lambda_floor_trusted": bool(covariance_result.get("lambda_floor_trusted", False)),
        "recommend_tangent_fallback": bool(recommendation["recommend_tangent_fallback"]),
        "tangent_recommendation_reason": recommendation["reason"],
        "D_soft_threshold": float(recommendation["soft_degeneracy_threshold"]),
        "sigma3_strict_eps": float(recommendation["sigma3_strict_eps"]),
    }


def _fmt_bound(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    if value >= 1000:
        return f"{value:.2e}"
    return f"{value:.3f}"


def format_grasp_covariance_report(result: Dict[str, Any], shape_name: str = "") -> str:
    """Human-readable one-block summary."""
    prefix = f"{shape_name}: " if shape_name else ""
    d = result["degeneracy_index"]
    d_str = "inf" if not np.isfinite(d) else f"{d:.2f}"
    ev = result["eigenvalues"]
    K = result.get("sobolev_K", float("nan"))
    Kt = result.get("sobolev_K_tight", float("nan"))
    Kd = result.get("sobolev_K_deriv", float("nan"))
    lam = result.get("lambda_shape_lower_bound", float("nan"))
    s1, s3 = result.get("sigma1", float("nan")), result.get("sigma3", float("nan"))
    trusted = result.get("lambda_floor_trusted", False)
    lam_note = "" if trusted else "*"
    gate = result.get("degeneracy_gate", "")
    return (
        f"{prefix}σ₁={s1:.3e} σ₃={s3:.3e} D={d_str}  "
        f"K={_fmt_bound(K)} (Kt={_fmt_bound(Kt)} Kd={_fmt_bound(Kd)}) "
        f"λ_lb={_fmt_bound(lam)}{lam_note}  "
        f"gate={gate} class={result['classification']}  "
        f"σ=({ev[0]:.4e}, {ev[1]:.4e}, {ev[2]:.4e})  "
        f"samples={result['n_samples']} (per_edge={result['samples_per_edge']})  "
        f"weight={result['total_weight']:.4f}/{result['boundary_length']:.4f}"
    )
