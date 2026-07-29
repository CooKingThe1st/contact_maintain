"""Precomputed 2D footprints for OBJ-backed holonomic shapes.

Footprints are generated once by ``scripts/test/preprocess_obj_footprints.py``
(uses PyBullet + ``obj_to_generic``) and stored in ``urdf/obj_footprint_cache.json``.
Runtime planners and simulation use **true OBJ scale** (centroid-centered only;
scenario ``robot.width`` / ``robot.length`` are not scaling factors for OBJ shapes).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

from shapely.geometry import Polygon

try:
    import rospkg

    _ROSPKG_AVAILABLE = True
except ImportError:
    _ROSPKG_AVAILABLE = False

CACHE_FILENAME = "obj_footprint_cache.json"
CACHE_VERSION = 1
Vertex = Tuple[float, float]


def package_root() -> Path:
    if _ROSPKG_AVAILABLE:
        return Path(rospkg.RosPack().get_path("contact_maintain"))
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "urdf").is_dir() and (parent / "src" / "contact_maintain").is_dir():
            return parent
        if parent.name == "contact_maintain" and (parent / "urdf").is_dir():
            return parent
    raise FileNotFoundError("Could not locate contact_maintain package root")


def default_cache_path() -> Path:
    return package_root() / "urdf" / CACHE_FILENAME


@lru_cache(maxsize=1)
def load_cache() -> dict:
    path = default_cache_path()
    if not path.is_file():
        return {"version": CACHE_VERSION, "shapes": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid footprint cache format: {path}")
    data.setdefault("shapes", {})
    return data


def clear_cache_memory() -> None:
    load_cache.cache_clear()


def center_vertices_at_centroid(vertices: List[Vertex]) -> List[Vertex]:
    """Translate vertices so the polygon centroid is at the origin."""
    if not vertices:
        return []
    poly = Polygon(vertices)
    if not poly.is_valid or poly.area <= 0:
        cx = sum(x for x, _ in vertices) / len(vertices)
        cy = sum(y for _, y in vertices) / len(vertices)
    else:
        cx, cy = float(poly.centroid.x), float(poly.centroid.y)
    return [(float(x) - cx, float(y) - cy) for x, y in vertices]


def vertices_for_shape(shape_name: str) -> Optional[List[Vertex]]:
    """Return cached local-frame vertices for ``shape_name``, or None if missing."""
    entry = load_cache().get("shapes", {}).get(str(shape_name))
    if not entry:
        return None
    raw = entry.get("vertices")
    if not raw or len(raw) < 3:
        return None
    return [(float(x), float(y)) for x, y in raw]


def resolve_footprint_vertices(
    shape_name: Optional[str] = None,
    obj_path: Optional[Union[str, Path]] = None,
) -> List[Vertex]:
    """
    Load footprint vertices for planning.

    Order of precedence:
      1. Precomputed cache entry for ``shape_name``
      2. ``read_obj_to_vertices(obj_path)`` (trimesh / DXF fallback)
    """
    if shape_name:
        cached = vertices_for_shape(shape_name)
        if cached:
            return list(cached)

    if obj_path:
        from object_utils import read_obj_to_vertices

        return read_obj_to_vertices(obj_path)

    if shape_name:
        raise ValueError(
            f"No cached footprint for shape '{shape_name}' and no obj_path provided. "
            f"Run scripts/test/preprocess_obj_footprints.py to build "
            f"{CACHE_FILENAME}."
        )
    raise ValueError("resolve_footprint_vertices requires shape_name and/or obj_path")


def save_cache(shapes: dict, cache_path: Optional[Path] = None) -> Path:
    """Write cache JSON (used by preprocess script)."""
    out = cache_path or default_cache_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "cache_file": out.name,
        "shapes": shapes,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    clear_cache_memory()
    return out
