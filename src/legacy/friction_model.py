#!/usr/bin/env python3
"""
PyBullet-aligned friction helpers (material × bumper = contact).

Legacy GenericObject still has static/kinetic/lateral fields. Revised paths
should use material_friction + bumper_mu → effective contact µ for search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# Engineering defaults from Markenscoff / friction-3 benchmarks (T=1, λ=2).
MU_CONTACT_N3 = 0.5
MU_CONTACT_N4_DEGENERATE = 0.25
MU_CONTACT_N4_WELL_BEHAVED = 0.05


@dataclass(frozen=True)
class BumperFrictionPlan:
    n_contacts: int
    tangent_required: bool
    material_friction: float
    target_contact_friction: float
    bumper_friction: float
    reason: str

    def as_dict(self):
        return {
            "n_contacts": self.n_contacts,
            "tangent_required": self.tangent_required,
            "material_friction": self.material_friction,
            "target_contact_friction": self.target_contact_friction,
            "bumper_friction": self.bumper_friction,
            "reason": self.reason,
        }


def product_contact_friction(material_mu: float, bumper_mu: float) -> float:
    """µ_contact = µ_material × µ_bumper (PyBullet product model)."""
    return float(material_mu) * float(bumper_mu)


def bumper_from_target_contact(
    material_mu: float,
    target_contact_mu: float,
) -> float:
    """Invert product: µ_bumper = µ_contact / µ_material."""
    m = max(float(material_mu), 1e-9)
    return float(target_contact_mu) / m


def recommend_target_contact_friction(
    n_contacts: int,
    *,
    tangent_required: bool,
) -> Tuple[float, str]:
    """
    Choose target µ_contact from contact count + degeneracy/friction need.

    n=3: always friction — need µ_contact ≳ 0.5 at T=1, λ=2.
    n=4 + tangent: µ_contact ≈ 0.2–0.3 class.
    n=4 + normal-only: small bumper (search does not rely on cone).
    """
    n = int(n_contacts)
    if n <= 3:
        return MU_CONTACT_N3, "n<=3: full AFC needs µ_contact≳0.5"
    if tangent_required:
        return (
            MU_CONTACT_N4_DEGENERATE,
            "n=4 + degenerate/tangent: µ_contact≈0.25",
        )
    return (
        MU_CONTACT_N4_WELL_BEHAVED,
        "n=4 well-behaved: low bumper; normal-only search",
    )


def recommend_bumper_friction(
    n_contacts: int,
    *,
    material_friction: float,
    tangent_required: bool,
    bumper_override: Optional[float] = None,
    target_contact_override: Optional[float] = None,
) -> BumperFrictionPlan:
    """
    Plan bumper µ (and implied contact µ) for revised holonomic scenes.

    If bumper_override is set, contact = material × bumper.
    Elif target_contact_override is set, bumper = target / material.
    Else use recommend_target_contact_friction policy.
    """
    mat = float(material_friction)
    if bumper_override is not None:
        bump = float(bumper_override)
        mu_c = product_contact_friction(mat, bump)
        reason = f"bumper override={bump:g} → µ_contact={mu_c:g}"
        return BumperFrictionPlan(
            n_contacts=int(n_contacts),
            tangent_required=bool(tangent_required),
            material_friction=mat,
            target_contact_friction=mu_c,
            bumper_friction=bump,
            reason=reason,
        )

    if target_contact_override is not None:
        mu_c = float(target_contact_override)
        reason = f"target µ_contact override={mu_c:g}"
    else:
        mu_c, reason = recommend_target_contact_friction(
            n_contacts, tangent_required=tangent_required
        )

    bump = bumper_from_target_contact(mat, mu_c)
    return BumperFrictionPlan(
        n_contacts=int(n_contacts),
        tangent_required=bool(tangent_required),
        material_friction=mat,
        target_contact_friction=mu_c,
        bumper_friction=bump,
        reason=reason,
    )
