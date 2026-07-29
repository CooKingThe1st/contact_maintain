#!/usr/bin/env python3
"""
Hardware feasibility checks for AFC Problem B at engineering force_range_scalar.

Verifies that holonomic pushers can supply the per-contact force cap assumed by search.

Contact friction must be the **effective robot–object Coulomb µ**
(material × bumper in PyBullet). Never use ground ``static_friction`` for the cone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from object_utils import GenericObject, WrenchSpaceVisualizer
from grasp_covariance import DEFAULT_SOFT_DEGENERACY_THRESHOLD


# OmniwheelRobot velocity motors use force=100 N (see omniwheel_robot.py).
_DEFAULT_WHEEL_MOTOR_FORCE_N = 100.0
_DEFAULT_FLOOR_FRICTION = 1.0


@dataclass(frozen=True)
class HardwareFeasibilityResult:
    feasible: bool
    f_max: float
    required_per_robot: float
    f_robot_max: float
    force_range_scalar: float
    tangent_mode: bool
    contact_friction: float
    reason: str
    warn_only: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "f_max": self.f_max,
            "required_per_robot": self.required_per_robot,
            "f_robot_max": self.f_robot_max,
            "force_range_scalar": self.force_range_scalar,
            "tangent_mode": self.tangent_mode,
            "contact_friction": self.contact_friction,
            "reason": self.reason,
            "warn_only": self.warn_only,
        }


def estimate_robot_max_push_force(
    *,
    wheel_motor_force: float = _DEFAULT_WHEEL_MOTOR_FORCE_N,
    floor_friction: float = _DEFAULT_FLOOR_FRICTION,
) -> float:
    """
    Conservative estimate of max push force per holonomic wheel robot (N).

    Uses PyBullet motor force limit × floor friction coefficient.
    """
    return float(wheel_motor_force) * max(float(floor_friction), 1e-6)


def limit_surface_f_max(
    obj: GenericObject,
    *,
    threshold: float = 1.0,
    grid_size: int = 30,
) -> float:
    vis = WrenchSpaceVisualizer()
    ls = vis.calculate_limit_surface(obj, resolution=50, scaling_factor=threshold, grid_size=grid_size)
    return float(ls["f_max"])


def _resolve_contact_friction(
    obj: GenericObject,
    contact_friction: Optional[float],
) -> float:
    """
    Prefer explicit arg, else GenericObject.get_contact_friction() (product /
    revised path), never static_friction (ground).
    """
    if contact_friction is not None:
        return max(float(contact_friction), 1e-6)
    if hasattr(obj, "get_contact_friction"):
        return max(float(obj.get_contact_friction()), 1e-6)
    return max(float(getattr(obj, "lateral_friction", 0.2)), 1e-6)


def check_robot_afc_hardware_feasible(
    obj: GenericObject,
    *,
    force_range_scalar: float = 2.0,
    tangent_mode: bool = False,
    contact_friction: Optional[float] = None,
    robot_max_force: Optional[float] = None,
    wheel_motor_force: float = _DEFAULT_WHEEL_MOTOR_FORCE_N,
    floor_friction: float = _DEFAULT_FLOOR_FRICTION,
    threshold: float = 1.0,
    warn_only: bool = False,
) -> HardwareFeasibilityResult:
    """
    Sanity-check actuator capability vs Problem B force cap.

    Normal-only: F_robot_max >= force_range_scalar * f_max
    Tangent mode: F_robot_max >= force_range_scalar * f_max / mu_contact

    ``contact_friction`` must be effective robot–object µ (material × bumper).
    Primary AFC decisions now live in D/σ₃ + stochastic search; this gate is a
    soft pre-check unless the caller treats ``feasible=False`` as fatal.
    """
    mu = _resolve_contact_friction(obj, contact_friction)
    f_max = limit_surface_f_max(obj, threshold=threshold)
    f_robot = (
        float(robot_max_force)
        if robot_max_force is not None
        else estimate_robot_max_push_force(
            wheel_motor_force=wheel_motor_force,
            floor_friction=floor_friction,
        )
    )
    required = float(force_range_scalar) * f_max
    if tangent_mode:
        required = required / mu
    feasible = f_robot > required
    if feasible:
        reason = (
            f"OK: F_robot={f_robot:.2f}N > required={required:.2f}N "
            f"(f_max={f_max:.2f}N, λ={force_range_scalar}, µ_contact={mu:g}, "
            f"tangent={tangent_mode})"
        )
    else:
        prefix = "WARN" if warn_only else "AFC not achievable"
        reason = (
            f"{prefix}: F_robot={f_robot:.2f}N <= required={required:.2f}N "
            f"(f_max={f_max:.2f}N, λ={force_range_scalar}, µ_contact={mu:g}, "
            f"tangent={tangent_mode})"
        )
    return HardwareFeasibilityResult(
        feasible=feasible,
        f_max=f_max,
        required_per_robot=required,
        f_robot_max=f_robot,
        force_range_scalar=float(force_range_scalar),
        tangent_mode=bool(tangent_mode),
        contact_friction=mu,
        reason=reason,
        warn_only=bool(warn_only),
    )


def hardware_feasible_from_screening(
    obj: GenericObject,
    *,
    recommend_tangent_fallback: bool,
    degeneracy_index: float,
    force_range_scalar: float = 2.0,
    robot_max_force: Optional[float] = None,
    soft_threshold: float = DEFAULT_SOFT_DEGENERACY_THRESHOLD,
    contact_friction: Optional[float] = None,
    warn_only: bool = False,
) -> HardwareFeasibilityResult:
    tangent = bool(recommend_tangent_fallback) or float(degeneracy_index) >= float(soft_threshold)
    return check_robot_afc_hardware_feasible(
        obj,
        force_range_scalar=force_range_scalar,
        tangent_mode=tangent,
        contact_friction=contact_friction,
        robot_max_force=robot_max_force,
        warn_only=warn_only,
    )
