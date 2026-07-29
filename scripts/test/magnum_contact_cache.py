#!/usr/bin/env python3
"""Persistent n-contact AFC t_param cache for revised holonomic debug runs.

Schema (urdf/magnum_afc_cache.json)::

    {
      "<shape>": {
        "<n_contacts>": {
          "t_params": [float, ...],
          "mu_contact": float | null,
          "tangent_required": bool | null,
          "source": "stochastic" | "legacy_four" | ...
        }
      }
    }

Also reads legacy ``magnum_four_cache.json`` (flat shape -> 4 floats) as a
fallback when looking up n=4 and the new cache has no entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import rospkg

_rospack = rospkg.RosPack()
_PKG = Path(_rospack.get_path("contact_maintain"))
DEFAULT_AFC_CACHE = _PKG / "urdf" / "magnum_afc_cache.json"
LEGACY_FOUR_CACHE = _PKG / "urdf" / "magnum_four_cache.json"


def default_cache_path() -> Path:
    return DEFAULT_AFC_CACHE


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to load contact cache {path}: {e}")
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _normalize_entry(raw: Any, *, n_contacts: int) -> Optional[Dict[str, Any]]:
    """Accept list[float] or dict with t_params; validate length."""
    if raw is None:
        return None
    if isinstance(raw, list):
        t_params = [float(x) % 1.0 for x in raw]
        if len(t_params) != n_contacts:
            return None
        return {
            "t_params": t_params,
            "mu_contact": None,
            "tangent_required": None,
            "source": "list",
        }
    if isinstance(raw, dict) and "t_params" in raw:
        t_params = [float(x) % 1.0 for x in raw["t_params"]]
        if len(t_params) != n_contacts:
            return None
        return {
            "t_params": t_params,
            "mu_contact": (
                float(raw["mu_contact"]) if raw.get("mu_contact") is not None else None
            ),
            "tangent_required": (
                bool(raw["tangent_required"])
                if raw.get("tangent_required") is not None
                else None
            ),
            "source": str(raw.get("source") or "dict"),
        }
    return None


def load_cached_contacts(
    shape_name: str,
    n_contacts: int,
    cache_path: Optional[Path] = None,
    *,
    allow_legacy_four: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return cached entry for (shape, n) or None."""
    n_contacts = int(n_contacts)
    path = Path(cache_path) if cache_path is not None else DEFAULT_AFC_CACHE
    data = _load_json(path)
    shape_block = data.get(shape_name)
    if isinstance(shape_block, dict):
        # Prefer string keys "3", "4"; also accept int keys after json roundtrip
        raw = shape_block.get(str(n_contacts), shape_block.get(n_contacts))
        entry = _normalize_entry(raw, n_contacts=n_contacts)
        if entry is not None:
            entry["cache_path"] = str(path)
            return entry
        # Flat legacy-in-new-file: shape -> [floats] (n=4 only)
        if n_contacts == 4:
            entry = _normalize_entry(shape_block if isinstance(shape_block, list) else None, n_contacts=4)
            # shape_block is dict here; skip

    if allow_legacy_four and n_contacts == 4:
        legacy = _load_json(LEGACY_FOUR_CACHE)
        if shape_name in legacy:
            entry = _normalize_entry(legacy[shape_name], n_contacts=4)
            if entry is not None:
                entry["source"] = "legacy_four"
                entry["cache_path"] = str(LEGACY_FOUR_CACHE)
                return entry
    return None


def save_cached_contacts(
    shape_name: str,
    n_contacts: int,
    t_params: Sequence[float],
    *,
    cache_path: Optional[Path] = None,
    mu_contact: Optional[float] = None,
    tangent_required: Optional[bool] = None,
    source: str = "stochastic",
) -> Path:
    """Upsert (shape, n) entry and write cache file. Returns path written."""
    n_contacts = int(n_contacts)
    t_list = [float(x) % 1.0 for x in t_params]
    if len(t_list) != n_contacts:
        raise ValueError(
            f"t_params length {len(t_list)} != n_contacts {n_contacts}"
        )
    path = Path(cache_path) if cache_path is not None else DEFAULT_AFC_CACHE
    data = _load_json(path)
    block = data.get(shape_name)
    if not isinstance(block, dict):
        block = {}
    block[str(n_contacts)] = {
        "t_params": t_list,
        "mu_contact": mu_contact,
        "tangent_required": tangent_required,
        "source": source,
    }
    data[shape_name] = block
    _write_json(path, data)
    return path


def contacts_from_t_params(
    generic_object,
    t_params: Sequence[float],
    contact_point_parameterization=None,
) -> List[Any]:
    """Build ContactPoint list from cached t_params (body-frame geometry)."""
    from object_utils import ContactPoint, ContactPointParameterization

    param = contact_point_parameterization or ContactPointParameterization(generic_object)
    contacts = []
    for t_param in t_params:
        t = float(t_param) % 1.0
        info = param.get_contact_info(t)
        contacts.append(
            ContactPoint(
                position=info["point"],
                tangent=info["tangent"],
                normal_outward=info["normal_outward"],
                normal_inward=info["normal_inward"],
                parameter=t,
                force_direction=None,
                object_ref=generic_object,
            )
        )
    return contacts
