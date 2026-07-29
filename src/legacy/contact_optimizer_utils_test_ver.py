import numpy as np
import scipy.io as sio
import math
import time
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import transform
from shapely.affinity import rotate, translate

import socket
import struct
import pickle
import io
import base64

# =============================================================================
# MULTI-ROBOT WRENCH OPTIMIZATION FRAMEWORK
# =============================================================================

try:
    import casadi as ca
except ModuleNotFoundError:
    ca = None  # Optional dependency (only needed for some optimization routines)
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from scipy.linalg import null_space
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


# =============================================================================
# MAXIMUM INSCRIBED CIRCLE FINDER - IMPLEMENTATION
# =============================================================================

from scipy.spatial import Voronoi
from scipy.optimize import linprog
from matplotlib.patches import Circle as MPLCircle
 

# Import object geometry + parameterization utilities from this repo.
# NOTE: This file originally depended on an external `StudyPlan_ObjectLib` notebook/module.
# In this repo we provide the same primitives in `legacy/object_utils.py`.
from object_utils import (
    GenericObject,
    ContactPointParameterization,
    ContactPoint,
    GenericContactCalculator,
    GraspMatrixCalculator,
    EdgeCharacterizer,
    create_standard_objects,
    WrenchSpaceVisualizer,
    DynamicObjectModel,
    get_reachable_contact_intervals,
)

def _find_max_inscribed_circles(obj, edge_characterizer, method='auto', samples_per_edge=50, tolerance=1e-6):
    """
    Find maximum inscribed circle(s) in the object.
    
    Args:
        obj: GenericObject instance
        edge_characterizer: EdgeCharacterizer instance
        method: 'auto', 'voronoi', or 'lp' (Linear Programming)
        samples_per_edge: Number of points to sample per edge for Voronoi
        tolerance: Numerical tolerance for comparisons
    
    Returns:
        list of dict: [{
            'center': np.array([x, y]),
            'radius': float,
            'contacts': list of contact dicts,
            'num_tangents': int
        }]
    """
    print(f"🔵 Finding maximum inscribed circles using method: {method}")
    
    # STEP 1: Determine object type and method
    if method == 'auto':
        # Check if object is convex
        is_convex = _check_convexity(obj)
        if is_convex:
            print("   Object is convex - using Linear Programming method")
            method = 'lp'
        else:
            print("   Object is non-convex - using Voronoi method")
            method = 'voronoi'
    
    # STEP 2: Find all maximal circles
    if method == 'lp':
        circles = _find_circles_lp(obj, edge_characterizer, tolerance)
    else:  # voronoi
        circles = _find_circles_voronoi(obj, edge_characterizer, samples_per_edge, tolerance)
    
    # Handle edge case: no circles found
    if len(circles) == 0:
        print("   ⚠️ No valid circles found - returning centroid with tiny radius")
        centroid = obj.get_centroid()
        return [{
            'center': np.array([centroid.x, centroid.y]),
            'radius': 0.001,
            'contacts': [],
            'num_tangents': 0
        }]
    
    # STEP 3: Find contact points for each circle
    print(f"   Found {len(circles)} candidate circles, finding contact points...")
    
    for circle in circles:
        contacts = _find_circle_contacts(circle, obj, edge_characterizer, tolerance)
        circle['contacts'] = contacts
        circle['num_tangents'] = len(contacts)
    
    # Warn if very small circle
    max_radius = max(c['radius'] for c in circles)
    if max_radius < 1e-6:
        print("   ⚠️ Very small inscribed circle - object may be degenerate")
    
    print(f"   ✅ Found {len(circles)} maximal circle(s) with radius {max_radius:.6f}")
    
    return circles


def _check_convexity(obj):
    """
    Check if object is convex using cross product test.
    
    Returns:
        bool: True if convex, False otherwise
    """
    # FIX: Use obj.geometry.exterior.coords
    if hasattr(obj.geometry, 'exterior'):
        coords = list(obj.geometry.exterior.coords)
    elif hasattr(obj.geometry, 'coords'):
        coords = list(obj.geometry.coords)
    else:
        return False
    
    if len(coords) < 3:
        return False
    
    # Remove duplicate last point if closed
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    
    # Check cross products have same sign
    n = len(coords)
    sign = None
    
    for i in range(n):
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % n])
        p3 = np.array(coords[(i + 2) % n])
        
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Cross product (z-component in 2D)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        if abs(cross) > 1e-10:  # Non-degenerate
            if sign is None:
                sign = np.sign(cross)
            elif np.sign(cross) != sign:
                return False  # Mixed signs = non-convex
    
    return True

def _find_circles_lp(obj, edge_characterizer, tolerance):
    """
    Find maximum inscribed circle using Linear Programming (for convex objects).
    
    Formulation:
        Maximize: r
        Subject to: distance from (cx, cy) to each edge ≥ r
    
    Variables: [cx, cy, r]
    """
    print("   Using LP method for convex object...")
    
    # Get edges
    edges = edge_characterizer.edges
    n_edges = len(edges)
    
    if n_edges == 0:
        return []
    
    # Build LP constraints: for each edge, distance ≥ r
    # For edge from A to B with inward normal n:
    # n · (center - A) ≥ r
    # n_x * cx + n_y * cy - r ≥ n · A
    
    # Inequality constraints: A_ub @ x ≤ b_ub
    # We want: -n_x * cx - n_y * cy + r ≤ -n · A
    A_ub = []
    b_ub = []
    
    for edge_idx in range(n_edges):
        # Get edge endpoints by sampling start and end parameters
        edge_info = edges[edge_idx]
        t_start = edge_info['start_param']
        t_end = edge_info['end_param']
        
        # Sample edge start and end
        start_info = edge_characterizer.parameterization.get_contact_info(t_start)
        end_info = edge_characterizer.parameterization.get_contact_info(t_end)
        
        A = np.array(start_info['point'])
        B = np.array(end_info['point'])
        
        # Get inward normal (sample at midpoint)
        t_mid = 0.5 * (t_start + t_end)
        mid_info = edge_characterizer.parameterization.get_contact_info(t_mid)
        normal_inward = np.array(mid_info['normal_inward'])
        
        # Constraint: n · (center - A) ≥ r
        # Rearrange: -n_x * cx - n_y * cy + r ≤ -n · A
        A_ub.append([-normal_inward[0], -normal_inward[1], 1.0])
        b_ub.append(-np.dot(normal_inward, A))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Objective: maximize r (minimize -r)
    c = np.array([0.0, 0.0, -1.0])
    
    # Bounds: cx, cy unbounded, r ≥ 0
    bounds = [(None, None), (None, None), (0, None)]
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        print(f"   ❌ LP failed: {result.message}")
        return []
    
    cx, cy, r = result.x
    
    print(f"   ✅ LP solution: center=({cx:.4f}, {cy:.4f}), radius={r:.6f}")
    
    # For convex objects, typically only one circle
    # But check if there are multiple points with same radius (symmetry)
    # This is rare for LP, so return single circle
    return [{
        'center': np.array([cx, cy]),
        'radius': r,
        'contacts': [],
        'num_tangents': 0
    }]

def _find_circles_voronoi(obj, edge_characterizer, samples_per_edge, tolerance):
    """
    Find maximum inscribed circles using Voronoi diagram.
    
    This can find multiple circles for symmetric shapes.
    """
    print(f"   Using Voronoi method with {samples_per_edge} samples per edge...")
    
    # STEP 2.1: Sample points on edges
    # TODO: when len(edges) is large, consider deduping nearby sites or lowering
    # samples_per_edge — 46 edges × 50 samples = 2300 Voronoi sites for sym crescent.
    edge_samples, edge_map = _sample_edges(obj, edge_characterizer, samples_per_edge)
    
    if len(edge_samples) < 3:
        print("   ❌ Too few edge samples for Voronoi")
        return []
    
    print(f"   Sampled {len(edge_samples)} points from edges")
    
    # STEP 2.2: Compute Voronoi diagram
    try:
        vor = Voronoi(edge_samples)
    except Exception as e:
        print(f"   ❌ Voronoi computation failed: {e}")
        return []
    
    print(f"   Voronoi diagram has {len(vor.vertices)} vertices")
    
    # STEP 2.3: For each Voronoi vertex inside the shape, compute clearance radius.
    # Use prepared geometry for fast contains; Shapely boundary distance for radius.
    from shapely.prepared import prep

    prepared_geom = prep(obj.geometry)
    boundary = _get_object_boundary(obj)
    candidates = []

    for idx, vertex in enumerate(vor.vertices):
        if not prepared_geom.contains(Point(vertex[0], vertex[1])):
            continue

        radius = boundary.distance(Point(vertex[0], vertex[1]))

        candidates.append({
            'center': vertex,
            'radius': radius
        })

        if idx < 5:  # Debug: print first few inside vertices
            print(f"      Vertex {idx}: center=({vertex[0]:.3f}, {vertex[1]:.3f}), "
                  f"radius={radius:.4f}, inside=True")
    
    if len(candidates) == 0:
        print("   ⚠️ No Voronoi vertices inside object")
        print("   This is normal for very regular shapes (like rectangles)")
        print("   The maximum inscribed circle may be at the center, not at Voronoi vertices")
        
        # Fallback: Use centroid with distance to nearest edge
        centroid = obj.get_centroid()
        centroid_array = np.array([centroid.x, centroid.y])
        radius = _distance_to_nearest_edge(centroid_array, obj)
        
        print(f"   Fallback: Using centroid with radius={radius:.4f}")
        
        return [{
            'center': centroid_array,
            'radius': radius,
            'contacts': [],
            'num_tangents': 0
        }]
    
    print(f"   Found {len(candidates)} candidate circles inside object")
    
    # STEP 2.4: Find maximum radius
    max_radius = max(c['radius'] for c in candidates)
    
    # STEP 2.5: Collect ALL circles with radius ≈ max_radius
    circles = [
        c for c in candidates 
        if abs(c['radius'] - max_radius) < tolerance
    ]
    
    print(f"   Selected {len(circles)} circles with max radius {max_radius:.6f}")
    
    # Handle infinite case (too many circles)
    if len(circles) > 100:
        print(f"   ⚠️ Found {len(circles)} circles - likely infinite (annulus/rectangle)")
        print(f"   Marking for strategic sampling")
        # Add a flag for later processing
        for c in circles:
            c['infinite_set'] = True
    
    return circles


def _sample_edges(obj, edge_characterizer, samples_per_edge):
    """
    Sample points uniformly along each edge.
    
    Returns:
        edge_samples: np.array of shape (N, 2) - sampled points
        edge_map: list mapping sample_index → edge_index
    """
    edge_samples = []
    edge_map = []
    
    edges = edge_characterizer.edges
    
    for edge_idx, edge_info in enumerate(edges):
        t_start = edge_info['start_param']
        t_end = edge_info['end_param']
        
        # Sample uniformly along parameter
        t_values = np.linspace(t_start, t_end, samples_per_edge)
        
        for t in t_values:
            contact_info = edge_characterizer.parameterization.get_contact_info(t)
            point = contact_info['point']
            
            edge_samples.append([point[0], point[1]])
            edge_map.append(edge_idx)
    
    return np.array(edge_samples), edge_map


_object_boundary_cache: Dict[int, Any] = {}


def _get_object_boundary(obj):
    """Return (and cache) the Shapely boundary geometry for distance queries."""
    cache_key = id(obj)
    if cache_key not in _object_boundary_cache:
        _object_boundary_cache[cache_key] = obj.geometry.boundary
    return _object_boundary_cache[cache_key]


def _distance_to_nearest_edge(point, obj, edge_characterizer=None):
    """
    Compute distance from point to nearest edge of object.

    Uses Shapely/GEOS boundary distance (O(log n) typical) instead of looping
    all logical edges in Python.
    
    Args:
        point: np.array([x, y])
        obj: GenericObject
        edge_characterizer: Unused; kept for backward-compatible call sites.
    
    Returns:
        float: minimum distance to any boundary segment
    """
    _ = edge_characterizer
    return float(_get_object_boundary(obj).distance(Point(float(point[0]), float(point[1]))))


def _distance_point_to_segment(point, A, B):
    """
    Compute minimum distance from point to line segment AB.
    
    Args:
        point: np.array([x, y])
        A: np.array([x, y]) - segment start
        B: np.array([x, y]) - segment end
    
    Returns:
        float: minimum distance
    """
    AB = B - A
    AP = point - A
    
    AB_length = np.linalg.norm(AB)
    
    if AB_length < 1e-10:
        # Degenerate segment
        return np.linalg.norm(AP)
    
    AB_unit = AB / AB_length
    t = np.dot(AP, AB_unit)
    
    # Clamp t to [0, AB_length]
    t = np.clip(t, 0, AB_length)
    
    closest = A + t * AB_unit
    return np.linalg.norm(point - closest)


def _find_circle_contacts(circle, obj, edge_characterizer, tolerance):
    """
    Find all contact points (tangency points) between circle and object edges.
    
    Args:
        circle: dict with 'center' and 'radius'
        obj: GenericObject
        edge_characterizer: EdgeCharacterizer
        tolerance: numerical tolerance for tangency check
    
    Returns:
        list of contact dicts: [{
            'point': np.array([x, y]),
            'edge_index': int,
            'parameter': float
        }]
    """
    contacts = []
    
    center = circle['center']
    radius = circle['radius']
    
    edges = edge_characterizer.edges
    
    for edge_idx, edge_info in enumerate(edges):
        # Find tangency point on this edge
        tangency = _find_tangency_on_edge(
            edge_idx, 
            edge_characterizer, 
            center, 
            radius, 
            tolerance
        )
        
        if tangency is not None:
            contacts.append({
                'point': tangency['point'],
                'edge_index': edge_idx,
                'parameter': tangency['parameter']
            })
    
    return contacts

def _find_tangency_on_edge(edge_idx, edge_characterizer, center, radius, tolerance):
    """
    Find tangency point between circle and a specific edge.
    Enhanced to handle vertex tangencies (corners).
    
    Args:
        edge_idx: int - edge index
        edge_characterizer: EdgeCharacterizer
        center: np.array([x, y]) - circle center
        radius: float - circle radius
        tolerance: float - numerical tolerance
    
    Returns:
        dict or None: {'point': np.array, 'parameter': float} if tangent, else None
    """
    edge_info = edge_characterizer.edges[edge_idx]
    t_start = edge_info['start_param']
    t_end = edge_info['end_param']
    
    # Sample edge endpoints
    start_info = edge_characterizer.parameterization.get_contact_info(t_start)
    end_info = edge_characterizer.parameterization.get_contact_info(t_end)
    
    A = np.array(start_info['point'])
    B = np.array(end_info['point'])
    
    # 🆕 STEP 1: Check if circle touches at start vertex (corner)
    dist_to_start = np.linalg.norm(center - A)
    if abs(dist_to_start - radius) < tolerance:
        # Tangent at start vertex
        return {
            'point': A,
            'parameter': t_start
        }
    
    # 🆕 STEP 2: Check if circle touches at end vertex (corner)
    dist_to_end = np.linalg.norm(center - B)
    if abs(dist_to_end - radius) < tolerance:
        # Tangent at end vertex
        return {
            'point': B,
            'parameter': t_end
        }
    
    # STEP 3: Check tangency along the edge (not at corners)
    AB = B - A
    AB_length = np.linalg.norm(AB)
    
    if AB_length < 1e-10:
        # Degenerate edge
        return None
    
    AB_unit = AB / AB_length
    
    AP = center - A
    t_proj = np.dot(AP, AB_unit)  # Projection distance along AB
    
    # 🔧 IMPORTANT: Exclude endpoints (already checked above)
    # Only consider interior of edge: (0, AB_length) exclusive
    if t_proj <= tolerance or t_proj >= AB_length - tolerance:
        return None  # Too close to endpoints
    
    # Find closest point on segment interior
    closest = A + t_proj * AB_unit
    
    # Check if distance equals radius (tangency condition)
    distance = np.linalg.norm(center - closest)
    
    if abs(distance - radius) < tolerance:
        # Convert spatial position back to parameter t
        # Linear interpolation: t = t_start + (t_proj / AB_length) * (t_end - t_start)
        t_param = t_start + (t_proj / AB_length) * (t_end - t_start)
        
        return {
            'point': closest,
            'parameter': t_param
        }
    
    return None

def _rank_and_filter_circles(circles, obj, max_circles=4, tolerance=0.1):
    """
    Rank circles by quality and filter to top candidates.
    
    Ranking criteria:
    1. More tangency points (contact with more edges) - MOST IMPORTANT
    2. Better distribution of tangency points
    3. Closer to centroid
    
    Enhanced filtering:
    - If top circles have more contacts than others, exclude inferior circles
    
    Args:
        circles: list of circle dicts
        obj: GenericObject
        max_circles: maximum number of circles to return
        tolerance: minimum distance threshold for _too_close check
    
    Returns:
        list: filtered and ranked circles (at most max_circles)
    """
    print(f"📊 Ranking {len(circles)} circles...")
    
    # STEP 1: Handle edge cases
    if len(circles) == 0:
        return []
    
    if len(circles) == 1:
        print("   Only one circle - no ranking needed")
        return circles
    
    # STEP 1.5: Check for infinite case (annulus/rectangle with many circles)
    if len(circles) > 100:
        print(f"   ⚠️ Too many circles ({len(circles)}) - using strategic sampling")
        return _sample_strategic_circles(circles, obj, max_circles, tolerance)
    
    # STEP 2: Compute quality scores for all circles
    centroid = obj.get_centroid()
    centroid_array = np.array([centroid.x, centroid.y])
    
    scores = []
    for circle in circles:
        score = _compute_circle_quality(circle, obj, centroid_array)
        scores.append((score, circle))
    
    # STEP 3: Sort by score (descending - higher is better)
    scores.sort(reverse=True, key=lambda x: x[0])
    
    if len(set(s[0] for s in scores)) == 1:
        print("   All circles have same score - breaking tie by distance to centroid")
        # Sort by distance to centroid
        scores.sort(key=lambda x: np.linalg.norm(x[1]['center'] - centroid_array))
    
    # STEP 4: Take top max_circles
    top_circles = [circle for (score, circle) in scores[:max_circles]]
    
    # 🆕 STEP 5: Enhanced filtering - remove circles with fewer contacts if we have better ones
    if len(top_circles) > 1:
        # Find maximum number of contacts among top circles
        max_contacts = max(c['num_tangents'] for c in top_circles)
        
        # Count how many circles have this maximum
        circles_with_max_contacts = [c for c in top_circles if c['num_tangents'] == max_contacts]
        
        # If we have at least 2 circles with max contacts, filter out inferior ones
        if len(circles_with_max_contacts) >= 2:
            # Check if any circles have fewer contacts
            has_inferior = any(c['num_tangents'] < max_contacts for c in top_circles)
            
            if has_inferior:
                print(f"   🔧 Filtering: Found {len(circles_with_max_contacts)} circles with {max_contacts} contacts")
                print(f"      Removing circles with fewer contacts as they provide inferior solutions")
                
                # Keep only circles with maximum contacts
                top_circles = circles_with_max_contacts
    
    print(f"   ✅ Selected top {len(top_circles)} circles")
    for i, circle in enumerate(top_circles):
        # Find corresponding score
        score = next((s for s, c in scores if c is circle), 0.0)
        print(f"      Circle {i+1}: score={score:.2f}, contacts={circle['num_tangents']}, "
              f"center=({circle['center'][0]:.3f}, {circle['center'][1]:.3f})")
    
    return top_circles

def _compute_circle_quality(circle, obj, centroid):
    """
    Compute quality score for a circle (higher = better).
    
    Args:
        circle: circle dict with 'center', 'radius', 'contacts'
        obj: GenericObject
        centroid: np.array([x, y]) - object centroid
    
    Returns:
        float: quality score
    """
    score = 0.0
    
    # Criterion 1: Number of contact points (MOST IMPORTANT)
    # More contacts = easier to achieve force closure
    num_contacts = len(circle['contacts'])
    score += num_contacts * 100  # Weight: 100 points per contact
    
    # Criterion 2: Distribution of contact points
    if num_contacts >= 2:
        distribution_score = _compute_contact_distribution(
            circle['contacts'], 
            circle['center']
        )
        score += distribution_score * 10  # Weight: 10x
    
    # Criterion 3: Distance to centroid (prefer central circles)
    dist_to_centroid = np.linalg.norm(circle['center'] - centroid)
    
    # Normalize by object diameter
    object_diameter = _compute_object_diameter(obj)
    normalized_dist = dist_to_centroid / (object_diameter + 1e-10)
    
    # Closer = better (invert: 1 - normalized_dist)
    score += (1 - normalized_dist) * 5  # Weight: 5x
    
    return score


def _compute_contact_distribution(contacts, center):
    """
    Measure how well contacts are distributed angularly around the circle.
    
    Args:
        contacts: list of contact dicts with 'point'
        center: np.array([x, y]) - circle center
    
    Returns:
        float: distribution score from 0 (clustered) to 1 (well-spread)
    """
    if len(contacts) < 2:
        return 0.0
    
    # Compute angle of each contact from center
    angles = []
    for contact in contacts:
        point = contact['point']
        angle = np.arctan2(point[1] - center[1], point[0] - center[0])
        angles.append(angle)
    
    # Sort angles
    angles = sorted(angles)
    
    # Compute gaps between consecutive angles (including wrap-around)
    gaps = []
    n = len(angles)
    for i in range(n):
        next_i = (i + 1) % n
        gap = angles[next_i] - angles[i]
        if gap < 0:
            gap += 2 * np.pi
        gaps.append(gap)
    
    # Ideal gap = 2π / num_contacts (uniform distribution)
    ideal_gap = 2 * np.pi / len(contacts)
    
    # Compute variance from ideal
    variance = sum((gap - ideal_gap)**2 for gap in gaps)
    
    # Convert to score: lower variance = higher score
    # Use exponential decay: exp(-variance / ideal_gap²)
    distribution_score = np.exp(-variance / (ideal_gap**2 + 1e-10))
    
    return distribution_score


def _sample_strategic_circles(circles, obj, max_circles, tolerance):
    """
    For infinite circles (annulus/rectangle), sample at strategic locations.
    Sample at cardinal directions: North, South, East, West, etc.
    
    Args:
        circles: list of circle dicts
        obj: GenericObject
        max_circles: max number to return
        tolerance: minimum distance between sampled circles
    
    Returns:
        list: strategically sampled circles
    """
    print(f"   Sampling {max_circles} strategic circles from infinite set...")
    
    sampled = []
    
    centroid = obj.get_centroid()
    centroid_array = np.array([centroid.x, centroid.y])
    
    # Sample 1: Northernmost (highest y)
    north = max(circles, key=lambda c: c['center'][1])
    sampled.append(north)
    print(f"      North: y={north['center'][1]:.3f}")
    
    # Sample 2: Southernmost (lowest y)
    south = min(circles, key=lambda c: c['center'][1])
    if not _too_close(south, sampled, tolerance):
        sampled.append(south)
        print(f"      South: y={south['center'][1]:.3f}")
    
    # Sample 3: Easternmost (highest x)
    east = max(circles, key=lambda c: c['center'][0])
    if not _too_close(east, sampled, tolerance):
        sampled.append(east)
        print(f"      East: x={east['center'][0]:.3f}")
    
    # Sample 4: Westernmost (lowest x)
    west = min(circles, key=lambda c: c['center'][0])
    if not _too_close(west, sampled, tolerance):
        sampled.append(west)
        print(f"      West: x={west['center'][0]:.3f}")
    
    # If we need more circles
    if len(sampled) < max_circles:
        # Sample 5: Closest to centroid
        center_circle = min(circles, 
                          key=lambda c: np.linalg.norm(c['center'] - centroid_array))
        if not _too_close(center_circle, sampled, tolerance):
            sampled.append(center_circle)
            print(f"      Center: dist={np.linalg.norm(center_circle['center'] - centroid_array):.3f}")
    
    # If STILL need more, sample at intermediate directions (NE, NW, SE, SW)
    if len(sampled) < max_circles:
        for angle_deg, name in [(45, 'NE'), (135, 'NW'), (225, 'SW'), (315, 'SE')]:
            angle = np.radians(angle_deg)
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Find circle whose center is most aligned with this direction
            projected = max(circles,
                          key=lambda c: np.dot(c['center'] - centroid_array, direction))
            
            if not _too_close(projected, sampled, tolerance):
                sampled.append(projected)
                print(f"      {name}: angle={angle_deg}°")
                
                if len(sampled) >= max_circles:
                    break
    
    return sampled[:max_circles]


def _too_close(circle, circle_list, threshold):
    """
    Check if circle is too close to any circle in circle_list.
    
    Args:
        circle: circle dict to check
        circle_list: list of existing circle dicts
        threshold: minimum allowed distance between centers
    
    Returns:
        bool: True if too close to any existing circle
    """
    for existing in circle_list:
        dist = np.linalg.norm(circle['center'] - existing['center'])
        if dist < threshold:
            return True
    return False

def _compute_object_diameter(obj):
    """
    Approximate object diameter as distance between furthest vertices.
    
    Args:
        obj: GenericObject
    
    Returns:
        float: approximate diameter
    """
    # FIX: Use obj.geometry.exterior.coords instead of obj.get_boundary()
    if hasattr(obj.geometry, 'exterior'):
        coords = list(obj.geometry.exterior.coords)
    elif hasattr(obj.geometry, 'coords'):
        coords = list(obj.geometry.coords)
    else:
        # Fallback: use bounding box diagonal
        bbox = obj.get_bounding_box()
        return np.sqrt(bbox['width']**2 + bbox['height']**2)
    
    if len(coords) < 2:
        return 1.0  # Default
    
    # Convert to numpy array
    points = np.array(coords)
    
    # Compute pairwise distances
    max_dist = 0.0
    n = len(points)
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(points[i] - points[j])
            max_dist = max(max_dist, dist)
    
    return max_dist if max_dist > 0 else 1.0





# %%
def test_max_inscribed_circles():
    """
    Comprehensive test of maximum inscribed circle finder.
    Tests on various shapes and visualizes results.
    """
    print("\n" + "="*80)
    print("🧪 TESTING MAXIMUM INSCRIBED CIRCLE FINDER")
    print("="*80)
    
    # Get test objects
    standard_objects = create_standard_objects()
    test_objects = ['rectangle', 'triangle', 'l_shape', 'star']
    
    # Create visualization grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, obj_name in enumerate(test_objects):
        obj = standard_objects[obj_name]
        
        print(f"\n{'='*60}")
        print(f"🔷 Testing: {obj_name.upper()}")
        print(f"{'='*60}")
        
        # Create edge characterizer
        edge_characterizer = EdgeCharacterizer(obj, force_magnitude=1.0)
        
        # Test both methods
        for method_idx, method in enumerate(['voronoi', 'lp']):
            ax_idx = i * 2 + method_idx
            ax = axes[ax_idx]
            
            print(f"\n  Method: {method.upper()}")
            
            # Find circles
            circles = _find_max_inscribed_circles(
                obj, 
                edge_characterizer, 
                method=method,
                samples_per_edge=50
            )
            
            # Rank and filter
            if len(circles) > 1:
                circles = _rank_and_filter_circles(circles, obj, max_circles=4)
            
            # Visualize
            obj.visualize(ax=ax, alpha=0.3, facecolor='lightcyan', show_frame=True)
            
            # Draw circles
            for j, circle in enumerate(circles):
                center = circle['center']
                radius = circle['radius']
                
                # Draw circle
                circle_patch = MPLCircle(
                    center, radius,
                    fill=False, 
                    edgecolor='red' if j == 0 else 'orange',
                    linewidth=2,
                    linestyle='-' if j == 0 else '--',
                    label=f'Circle {j+1}' if j < 3 else ''
                )
                ax.add_patch(circle_patch)
                
                # Draw center
                ax.plot(center[0], center[1], 'r*', markersize=12, 
                       label='Center' if j == 0 else '')
                
                # Draw contact points - FIX: Use enumerate to track first contact
                for k, contact in enumerate(circle['contacts']):
                    point = contact['point']
                    # FIX: Use k == 0 instead of comparing arrays
                    ax.plot(point[0], point[1], 'go', markersize=6,
                           label='Tangency' if j == 0 and k == 0 else '')
                    
                    # Draw line from center to contact
                    ax.plot([center[0], point[0]], [center[1], point[1]], 
                           'g--', alpha=0.5, linewidth=1)
            
            # Title with results
            if len(circles) > 0:
                max_r = circles[0]['radius']
                max_contacts = circles[0]['num_tangents']
                title = f"{obj_name.title()} - {method.upper()}\n" \
                        f"r={max_r:.4f}, {len(circles)} circle(s), {max_contacts} contacts"
            else:
                title = f"{obj_name.title()} - {method.upper()}\nNo circles found"
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=8)
            
            # Print summary
            print(f"    ✅ Found {len(circles)} circle(s)")
            if len(circles) > 0:
                print(f"    Max radius: {circles[0]['radius']:.6f}")
                print(f"    Contacts: {circles[0]['num_tangents']}")
    
    plt.tight_layout()
    plt.suptitle('Maximum Inscribed Circles - Method Comparison', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.show()
    
    print("\n" + "="*80)
    print("🎉 TEST COMPLETE!")
    print("="*80)


# %%
def _remove_duplicate_parameters(samples, tolerance=0.01):
    """
    Remove sample points that are too close together in parameter space.
    
    Args:
        samples: list of (t_param, description) tuples
        tolerance: minimum parameter distance between points
    
    Returns:
        list: filtered samples with duplicates removed
    """
    if len(samples) == 0:
        return []
    
    # Sort by parameter value
    sorted_samples = sorted(samples, key=lambda x: x[0])
    
    # Keep first point, then only keep points far enough from previous kept point
    filtered = [sorted_samples[0]]
    
    for t_param, description in sorted_samples[1:]:
        last_kept_t = filtered[-1][0]
        
        if abs(t_param - last_kept_t) >= tolerance:
            filtered.append((t_param, description))
        else:
            # Too close - optionally merge descriptions
            # For now, just skip duplicate
            pass
    
    return filtered




def compute_epsilon(max_inscribed_circles, edge_characterizer):
    """
    Compute epsilon (offset distance) for strategic sampling based on multiple criteria.
    
    Epsilon should be:
    - Small enough to stay "near" original point
    - Large enough to create measurable moment difference
    - Scale with object size
    
    Args:
        max_inscribed_circles: list of circle dicts with 'radius'
        edge_characterizer: EdgeCharacterizer instance
    
    Returns:
        float: epsilon value for contact point offsets
    """
    # Strategy 1: Fraction of maximum circle radius
    if len(max_inscribed_circles) > 0:
        max_radius = max(circle['radius'] for circle in max_inscribed_circles)
        epsilon_radius = 0.05 * max_radius
    else:
        max_radius = 0.1  # Fallback
        epsilon_radius = 0.001  # Fallback
    
    # Strategy 2: Fraction of smallest edge length
    edge_lengths = [edge['length'] for edge in edge_characterizer.edges]
    min_edge_length = min(edge_lengths) if len(edge_lengths) > 0 else 0.1
    epsilon_edge = 0.2 * min_edge_length
    
    # Strategy 3: Ensure measurable moment difference
    # moment ≈ force × radius × ε/radius = force × ε
    # Want moment to be at least 1% of typical moment
    if len(max_inscribed_circles) > 0:
        typical_moment = max_radius  # Order of magnitude
        epsilon_moment = 0.05 * typical_moment / 10
    else:
        epsilon_moment = 0.0001
    
    # Take minimum of first two strategies, but ensure it's at least epsilon_moment
    epsilon = min(epsilon_radius, epsilon_edge)
    epsilon = max(epsilon, epsilon_moment)
    
    return epsilon




def generate_strategic_contact_samples(edge_characterizer, max_inscribed_circles, verbose=False):
    """
    Generate strategic contact point samples on each edge.
    
    Sampling strategy:
    1. Near-corner points (epsilon away from vertices)
    2. Edge midpoint
    3. No-torque point (where τ(t) = 0)
    4. Tangency points (contact with max inscribed circles)
    5. Epsilon offsets around special points
    
    Args:
        obj: GenericObject instance
        edge_characterizer: EdgeCharacterizer instance
        max_inscribed_circles: list of circle dicts from STEP 2
        verbose: If True, print detailed sampling info
    
    Returns:
        dict: edge_idx -> list of (t_param, description) tuples
    """
    if verbose:
        print(f"\n📍 STEP 3: Generating strategic contact point samples...")
    
    # Compute epsilon based on object geometry
    epsilon = compute_epsilon(max_inscribed_circles, edge_characterizer)
    
    if verbose:
        print(f"   Computed epsilon: {epsilon:.6f}")
        if len(max_inscribed_circles) > 0:
            max_radius = max(c['radius'] for c in max_inscribed_circles)
            print(f"   Max circle radius: {max_radius:.6f}")
            print(f"   Epsilon as % of radius: {epsilon/max_radius*100:.2f}%")
    
    edge_sample_points = {}  # Dict: edge_idx -> list of (t_param, description)
    
    for edge_idx, edge_name in enumerate(edge_characterizer.edge_characteristics.keys()):
        char = edge_characterizer.edge_characteristics[edge_name]
        edge_info = char['edge_info']
        
        t_start = edge_info['start_param']
        t_end = edge_info['end_param']
        t_range = t_end - t_start
        
        if t_range < 1e-10:
            # Degenerate edge
            if verbose:
                print(f"   ⚠️ Edge {edge_idx}: Degenerate (range={t_range:.6f}), skipping")
            edge_sample_points[edge_idx] = []
            continue
        
        samples = []
        
        # =====================================================================
        # 1. NEAR-CORNER POINTS (epsilon away from vertices)
        # =====================================================================
        # Convert epsilon from spatial distance to parameter distance
        edge_length = edge_info['length']
        epsilon_param = epsilon / edge_length if edge_length > 0 else epsilon
        
        t_corner_start = t_start + epsilon_param
        t_corner_end = t_end - epsilon_param
        
        # Only add if within bounds
        if t_corner_start < t_corner_end:
            samples.append((t_corner_start, 'near_start_corner'))
            samples.append((t_corner_end, 'near_end_corner'))
        
        # =====================================================================
        # 2. EDGE MIDPOINT
        # =====================================================================
        t_mid = 0.5 * (t_start + t_end)
        samples.append((t_mid, 'midpoint'))
        
        # =====================================================================
        # 2b. QUARTILE POINTS (first and third quartile)
        # =====================================================================
        # First quartile: midpoint between start and midpoint
        t_quarter = t_start + 0.25 * t_range
        # Third quartile: midpoint between midpoint and end
        t_three_quarter = t_start + 0.75 * t_range
        
        # Only add if within bounds (with small margin to avoid duplicates with corners)
        margin = 0.001 * t_range
        if (t_start + margin) < t_quarter < (t_end - margin):
            samples.append((t_quarter, 'first_quartile'))
        if (t_start + margin) < t_three_quarter < (t_end - margin):
            samples.append((t_three_quarter, 'third_quartile'))
        
        # =====================================================================
        # 3. NO-TORQUE POINT (where torque contribution is zero)
        # =====================================================================
        # Torque = α * (slope * t + offset) = 0
        # If slope ≠ 0: t_no_torque = -offset / slope
        slope = char['torque_slope']
        offset = char['torque_offset']
        
        if abs(slope) > 1e-6:  # Slope is significant
            t_no_torque = -offset / slope
            
            # Check if within edge bounds (with small margin)
            margin = 0.001 * t_range
            if (t_start + margin) <= t_no_torque <= (t_end - margin):
                samples.append((t_no_torque, 'no_torque_point'))
                
                # Add epsilon offsets around no-torque point
                t_no_torque_plus = min(t_no_torque + epsilon_param, t_end - margin)
                t_no_torque_minus = max(t_no_torque - epsilon_param, t_start + margin)
                
                samples.append((t_no_torque_plus, 'no_torque_plus_eps'))
                samples.append((t_no_torque_minus, 'no_torque_minus_eps'))
        
        # =====================================================================
        # 4. TANGENCY POINTS (from max inscribed circles)
        # =====================================================================
        for circle_idx, circle in enumerate(max_inscribed_circles):
            # Check if this circle has tangency on this edge
            tangency_contacts = [
                contact for contact in circle.get('contacts', [])
                if contact['edge_index'] == edge_idx
            ]
            
            for contact in tangency_contacts:
                tangency_t = contact['parameter']
                
                # Verify within edge bounds
                margin = 0.001 * t_range
                if (t_start + margin) <= tangency_t <= (t_end - margin):
                    samples.append((tangency_t, f'tangency_circle_{circle_idx}'))
                    
                    # Add epsilon offsets around tangency point
                    t_tangency_plus = min(tangency_t + epsilon_param, t_end - margin)
                    t_tangency_minus = max(tangency_t - epsilon_param, t_start + margin)
                    
                    samples.append((t_tangency_plus, f'tangency_circle_{circle_idx}_plus_eps'))
                    samples.append((t_tangency_minus, f'tangency_circle_{circle_idx}_minus_eps'))
        
        # =====================================================================
        # 5. REMOVE DUPLICATES (points too close together)
        # =====================================================================
        # Use smaller tolerance for duplicate detection (half of epsilon_param)
        duplicate_tolerance = 0.5 * epsilon_param
        samples = _remove_duplicate_parameters(samples, tolerance=duplicate_tolerance)
        
        # Store samples for this edge
        edge_sample_points[edge_idx] = samples
        
        if verbose:
            print(f"   Edge {edge_idx}: {len(samples)} sample points")
            for t_param, description in samples[:5]:  # Show first 5
                print(f"      t={t_param:.6f} ({description})")
            if len(samples) > 5:
                print(f"      ... and {len(samples) - 5} more")
    
    # Summary
    total_samples = sum(len(samples) for samples in edge_sample_points.values())
    if verbose:
        print(f"\n   ✅ Total strategic samples: {total_samples}")
        print(f"   Average per edge: {total_samples / len(edge_sample_points):.1f}")
    
    return edge_sample_points





# print("\n✅ Strategic Contact Point Sampling (STEP 3) tested successfully!")

# %%
def _build_contact_points(edge_indices, t_params, edge_characterizer):
    """
    Build ContactPoint objects from edge indices and parameters.
    """
    obj = edge_characterizer.parameterization.object
    contacts = []
    
    for edge_idx, t_param in zip(edge_indices, t_params):
        contact_info = edge_characterizer.parameterization.get_contact_info(t_param)
        
        contact_point = ContactPoint(
            position=contact_info['point'],
            tangent=contact_info['tangent'],
            normal_outward=contact_info['normal_outward'],
            normal_inward=contact_info['normal_inward'],
            parameter=t_param,
            object_ref=obj
        )
        contacts.append(contact_point)
    
    return contacts

def _check_edge_combination_seen(edge_indices, seen_combinations):
    """
    Check if this edge combination has already been tested.
    Uses sorted tuple as hash key to detect permutations.
    
    Args:
        edge_indices: list of 4 edge indices [e1, e2, e3, e4]
        seen_combinations: set of sorted tuples already tested
    
    Returns:
        bool: True if already seen, False if new combination
    """
    # Create sorted tuple as hash key
    sorted_key = tuple(sorted(edge_indices))
    
    if sorted_key in seen_combinations:
        return True  # Already tested this combination
    
    # New combination - add to set
    seen_combinations.add(sorted_key)
    return False


def _check_points_distinct(contacts, tolerance=0.01):
    """
    Check that all contact points are spatially distinct.
    
    Args:
        contacts: list of 4 ContactPoint objects
        tolerance: minimum distance between points
    
    Returns:
        bool: True if all points are distinct, False otherwise
    """
    n = len(contacts)
    
    # Check all pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            pos_i = contacts[i].position
            pos_j = contacts[j].position
            
            distance = np.linalg.norm(pos_i - pos_j)
            
            if distance < tolerance:
                return False  # Points too close together
    
    return True


_object_min_edge_length_cache: Dict[int, float] = {}


def _get_object_min_edge_length(obj) -> float:
    """Return the shortest boundary segment length for `obj` (cached per object)."""
    cache_key = id(obj)
    if cache_key not in _object_min_edge_length_cache:
        param = ContactPointParameterization(obj)
        lengths = param.segment_lengths
        _object_min_edge_length_cache[cache_key] = (
            float(min(lengths)) if len(lengths) > 0 else 0.1
        )
    return _object_min_edge_length_cache[cache_key]


def _compute_robot_spacing_buffer(robot_radius: float, min_edge_length: float) -> float:
    """
    Additional clearance beyond 2 * robot_radius for center-to-center spacing.

    Uses max(0.01 m floor, 0.5 * robot_radius, 10% of shortest object edge).
    """
    return max(0.01, 0.5 * robot_radius, 0.1 * min_edge_length)


def _check_enough_space_for_robots(
    contacts,
    robot_radius: float,
    min_edge_length: Optional[float] = None,
):
    """
    Check that robot centers have enough spacing to avoid collisions.
    
    For each contact point, the robot center is at:
        robot_center = contact.position + robot_radius * normal_outward
    
    For any two robot centers, the distance must be >= 2 * robot_radius + buffer
    to ensure the robots don't collide. The buffer is computed dynamically from
    robot size and the object's shortest edge length.
    
    Args:
        contacts: list of ContactPoint objects
        robot_radius: Radius of the circular robot
        min_edge_length: Optional shortest boundary segment length (m). If None,
            inferred from contacts[0].object_ref.
    
    Returns:
        bool: True if all robot centers have sufficient spacing, False otherwise
    """
    n = len(contacts)
    if n < 2:
        return True  # No spacing check needed for < 2 contacts

    if min_edge_length is None:
        obj = contacts[0].object_ref if contacts else None
        min_edge_length = _get_object_min_edge_length(obj) if obj is not None else 0.1

    buffer = _compute_robot_spacing_buffer(robot_radius, min_edge_length)
    
    # Compute robot center positions
    robot_centers = []
    for contact in contacts:
        robot_center = np.array(contact.position) + robot_radius * np.array(contact.normal_outward)
        robot_centers.append(robot_center)
    
    # Check all pairwise distances
    min_required_distance = 2.0 * robot_radius + buffer
    
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(robot_centers[i] - robot_centers[j])
            
            if distance < min_required_distance:
                return False  # Robots would collide
    
    return True


def _check_normals_not_parallel(contacts, angle_tolerance=2.0):
    """
    Check that normals are not all parallel or all in same direction.
    
    For force closure in 2D, we need normals that span the plane.
    This means we need at least 2 normals that are NOT parallel.
    
    Args:
        contacts: list of 4 ContactPoint objects
        angle_tolerance: minimum angle (degrees) to consider non-parallel
    
    Returns:
        bool: True if normals have good distribution, False if problematic
    """
    angle_tol_rad = np.radians(angle_tolerance)
    
    # Extract all inward normals
    normals = [contact.normal_inward for contact in contacts]
    
    # Check 1: Ensure at least one pair of normals is non-parallel
    found_non_parallel_pair = False
    
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            n_i = normals[i]
            n_j = normals[j]
            
            # Compute angle between normals using dot product
            # cos(θ) = n_i · n_j / (||n_i|| × ||n_j||)
            dot_product = np.dot(n_i, n_j)
            
            # For unit normals: cos(θ) = n_i · n_j
            # Parallel if |cos(θ)| ≈ 1 (angle ≈ 0° or 180°)
            cos_angle = np.clip(dot_product, -1.0, 1.0)
            angle = np.arccos(abs(cos_angle))  # Absolute angle (ignore direction)
            
            if angle > angle_tol_rad:
                found_non_parallel_pair = True
                break
        
        if found_non_parallel_pair:
            break
    
    if not found_non_parallel_pair:
        return False  # All normals are parallel - cannot achieve force closure
    
    # Check 2: Ensure normals are not all in same half-plane
    # This would fail force closure even if non-parallel
    
    # Compute reference direction (first normal)
    ref_normal = normals[0]
    
    # Check if all other normals point in roughly same direction
    all_same_side = True
    
    for i in range(1, len(normals)):
        dot_with_ref = np.dot(normals[i], ref_normal)
        
        if dot_with_ref < 0.0:  # Opposite side
            all_same_side = False
            break
    
    if all_same_side:
        return False  # All normals in same half-plane - cannot close
    
    return True


def _quick_force_closure_check(contacts):
    """
    Quick geometric check if normals can span the 2D plane.
    
    For force closure in 2D:
    - Need normals to positively span R²
    - Equivalent to: normals form a proper cone (not all in half-plane)
    - Check: Can we express any direction as positive combination of normals?
    
    This is a fast heuristic check before full LP solve.
    
    Args:
        contacts: list of 4 ContactPoint objects
    
    Returns:
        bool: True if quick check passes, False if definitely fails
    """
    # Extract inward normals
    normals = [contact.normal_inward for contact in contacts]
    
    # Build normal matrix: each column is a normal vector
    # N = [n_0 | n_1 | n_2 | n_3]  (2x4 matrix)
    N = np.column_stack(normals)  # Shape: (2, 4)
    
    # Quick Test 1: Check if normals span 2D (rank = 2)
    rank = np.linalg.matrix_rank(N)
    
    if rank < 2:
        return False  # Cannot span 2D plane - force closure impossible
    
    # Quick Test 2: Check if we can find positive weights that give zero sum
    # This is a simplified check - not as rigorous as LP, but much faster
    
    # Test: Can we find α_i > 0 such that Σ α_i × n_i = 0?
    # Heuristic: Check if normals form a "proper cone" (not all in same half-space)
    
    # Compute angular distribution of normals
    angles = []
    for normal in normals:
        angle = np.arctan2(normal[1], normal[0])  # Angle from +x axis
        angles.append(angle)
    
    # Sort angles
    angles_sorted = sorted(angles)
    
    # Compute gaps between consecutive angles (with wrap-around)
    gaps = []
    n = len(angles_sorted)
    for i in range(n):
        next_i = (i + 1) % n
        gap = angles_sorted[next_i] - angles_sorted[i]
        if gap < 0:
            gap += 2 * np.pi
        gaps.append(gap)
    
    # Check if any gap is >= π (180°)
    # If so, all normals are in one half-plane - force closure fails
    max_gap = max(gaps)
    
    if max_gap >= np.pi - 0.1:  # Allow small tolerance
        return False  # All normals in one half-plane
    
    # Quick check passed - normals likely can achieve force closure
    return True



# %%
def check_three_edge_force_closure(force_dirs, edge_indices=None, verbose=False):
    """
    Check if three edges can achieve force closure.
    
    This is the CORE force closure test used by both preprocessing and Magnum Three.
    
    Force closure conditions:
    1. No two force directions are parallel (threshold 0.98)
    2. Find strictly positive [a1, a2, a3] such that a1*f1 + a2*f2 + a3*f3 = 0
    
    Args:
        force_dirs: List of 3 force direction vectors (each is np.array([fx, fy]))
        edge_indices: Optional list of 3 edge indices (for verbose output)
        verbose: If True, print detailed analysis
    
    Returns:
        dict: {
            'valid': bool,
            'reason': str,
            'coefficients': np.array or None,
            'residual': float or None
        }
    """
    f1, f2, f3 = force_dirs
    
    # Normalize to unit vectors
    f1 = f1 / (np.linalg.norm(f1) + 1e-10)
    f2 = f2 / (np.linalg.norm(f2) + 1e-10)
    f3 = f3 / (np.linalg.norm(f3) + 1e-10)
    
    edge_label = f"edges {edge_indices}" if edge_indices else "3 edges"
    
    # =========================================================================
    # CHECK 1: No two force directions are parallel
    # =========================================================================
    parallel_threshold = 0.98  # cos(11.5°) ≈ 0.98
    
    dot_12 = np.abs(np.dot(f1, f2))
    dot_13 = np.abs(np.dot(f1, f3))
    dot_23 = np.abs(np.dot(f2, f3))
    
    if dot_12 > parallel_threshold:
        if verbose:
            print(f"   {edge_label}: ❌ Forces 1 and 2 are parallel (dot={dot_12:.4f})")
        return {
            'valid': False,
            'reason': 'parallel_forces_12',
            'coefficients': None,
            'residual': None
        }
    
    if dot_13 > parallel_threshold:
        if verbose:
            print(f"   {edge_label}: ❌ Forces 1 and 3 are parallel (dot={dot_13:.4f})")
        return {
            'valid': False,
            'reason': 'parallel_forces_13',
            'coefficients': None,
            'residual': None
        }
    
    if dot_23 > parallel_threshold:
        if verbose:
            print(f"   {edge_label}: ❌ Forces 2 and 3 are parallel (dot={dot_23:.4f})")
        return {
            'valid': False,
            'reason': 'parallel_forces_23',
            'coefficients': None,
            'residual': None
        }
    
    # =========================================================================
    # CHECK 2: Find strictly positive [a1, a2, a3] such that a1*f1 + a2*f2 + a3*f3 = 0
    # =========================================================================
    
    # Build force matrix F = [f1, f2, f3]^T (shape: 2x3)
    F = np.array([f1, f2, f3]).T  # Shape: (2, 3)
    
    # Check rank
    rank_F = np.linalg.matrix_rank(F)
    if rank_F >= 3:
        # Overdetermined system - forces don't span compatible subspace
        if verbose:
            print(f"   {edge_label}: ❌ Overdetermined system (rank={rank_F})")
        return {
            'valid': False,
            'reason': 'overdetermined_system',
            'coefficients': None,
            'residual': None
        }
    
    # Use LP to find positive solution
    # Minimize: sum(a_i)
    c = np.ones(3)
    
    # Equality constraint: F @ a = 0
    A_eq = F
    b_eq = np.zeros(2)
    
    # Bounds: a_i >= epsilon (strictly positive)
    epsilon = 1e-6
    bounds = [(epsilon, None)] * 3
    
    try:
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'disp': False})
        
        if result.success and np.all(result.x > 0):
            # Found valid strictly positive solution
            a_solution = result.x
            residual = np.linalg.norm(F @ a_solution)
            
            if verbose:
                print(f"   {edge_label}: ✅ VALID! Coefficients: [{a_solution[0]:.3f}, {a_solution[1]:.3f}, {a_solution[2]:.3f}], residual={residual:.2e}")
            
            return {
                'valid': True,
                'reason': 'force_closure_satisfied',
                'coefficients': a_solution,
                'residual': residual
            }
        else:
            if verbose:
                print(f"   {edge_label}: ❌ No strictly positive solution (LP failed)")
            return {
                'valid': False,
                'reason': 'no_positive_solution',
                'coefficients': None,
                'residual': None
            }
    
    except Exception as e:
        if verbose:
            print(f"   {edge_label}: ❌ LP solver error: {e}")
        return {
            'valid': False,
            'reason': 'lp_solver_error',
            'coefficients': None,
            'residual': None
        }



# %%
from collections import Counter
import itertools

def preprocess_object_force_closure(obj, force_magnitude=1.0, verbose=False, include_4edge=True):
    """
    Preprocessing: Identify all valid 3-edge and optionally 4-edge combinations for force closure.
    
    🆕 OPTIMIZED: Can skip 4-edge computation when only Magnum Three is needed!
    
    This function caches the force closure analysis to avoid redundant checks in 
    both Magnum Three and Magnum Four algorithms.
    
    Strategy:
    1. Extract force signature for each edge
    2. Test all 3-edge combinations using check_three_edge_force_closure()
    3. (Optional) Test all 4-edge combinations by trying to reduce to 3-edge subsets
       - For each 4-edge combo, test all 6 possible 3-edge subsets
       - If ANY subset passes, the 4-edge combo is valid
    
    Args:
        obj: GenericObject instance
        force_magnitude: Magnitude of normal force for edge characterization
        verbose: If True, print detailed analysis
        include_4edge: If True, compute 4-edge combinations (needed for Magnum Four)
                       If False, skip 4-edge computation (faster for Magnum Three only)
    
    Returns:
        dict: {
            'edge_characterizer': EdgeCharacterizer instance,
            'edge_force_signatures': dict mapping edge_idx -> force_direction,
            'valid_3edge_combos': list of dicts with valid 3-edge combinations,
            'valid_4edge_combos': list of dicts (empty if include_4edge=False),
            'valid_3edge_set': set of tuples for O(1) lookup,
            'valid_4edge_set': set of tuples for O(1) lookup (empty if include_4edge=False),
            'num_edges': int,
            'statistics': dict with counts
        }
    """
    if verbose:
        print("\n" + "="*80)
        print("🔧 PREPROCESSING: Analyzing Object Force Closure Capabilities")
        if not include_4edge:
            print("   (3-edge only mode - skipping 4-edge computation)")
        print("="*80)
    
    # =========================================================================
    # STEP 1: Create edge characterizer and extract force signatures
    # =========================================================================
    if verbose:
        print(f"\n📐 Step 1: Analyzing edges and extracting force signatures...")
    
    edge_characterizer = EdgeCharacterizer(obj, force_magnitude=force_magnitude)
    num_edges = len(edge_characterizer.edges)
    
    if verbose:
        print(f"   Found {num_edges} edges")
    
    # Extract force signature for each edge
    edge_force_signatures = {}
    
    for edge_idx in range(num_edges):
        edge_name = f'edge_{edge_idx}'
        char = edge_characterizer.edge_characteristics[edge_name]
        
        # Extract force direction from fixed_fx and fixed_fy
        force_direction = np.array([char['fixed_fx'], char['fixed_fy']])
        
        # Normalize to unit vector
        force_magnitude_norm = np.linalg.norm(force_direction)
        if force_magnitude_norm > 1e-10:
            force_direction = force_direction / force_magnitude_norm
        
        edge_force_signatures[edge_idx] = force_direction
        
        if verbose:
            print(f"   Edge {edge_idx}: force_dir = ({force_direction[0]:7.4f}, {force_direction[1]:7.4f})")
    
    # =========================================================================
    # STEP 2: Test all 3-edge combinations
    # =========================================================================
    if verbose:
        print(f"\n🔎 Step 2: Testing all 3-edge combinations for force closure...")
    
    import itertools
    
    valid_3edge_combos = []
    valid_3edge_set = set()
    total_3edge_combos = 0
    
    for edge_combo in itertools.combinations(range(num_edges), 3):
        total_3edge_combos += 1
        
        e1, e2, e3 = edge_combo
        
        force_dirs = [
            edge_force_signatures[e1],
            edge_force_signatures[e2],
            edge_force_signatures[e3]
        ]
        
        # Check force closure
        result = check_three_edge_force_closure(
            force_dirs, 
            edge_indices=list(edge_combo),
            verbose=verbose
        )
        
        if result['valid']:
            valid_3edge_combos.append({
                'edge_indices': list(edge_combo),
                'force_directions': force_dirs,
                'coefficients': result['coefficients'],
                'residual': result['residual']
            })
            
            valid_3edge_set.add(tuple(sorted(edge_combo)))
    
    if verbose:
        print(f"\n   Total 3-edge combinations tested: {total_3edge_combos}")
        print(f"   Valid 3-edge combinations found: {len(valid_3edge_combos)}")
    
    # =========================================================================
    # STEP 3: Test all 4-edge combinations (OPTIONAL)
    # =========================================================================
    valid_4edge_combos = []
    valid_4edge_set = set()
    total_4edge_combos = 0
    
    if include_4edge:
        if verbose:
            print(f"\n🔎 Step 3: Testing all 4-edge combinations (by combining pairs)...")
        
        for edge_combo_4 in itertools.combinations(range(num_edges), 4):
            total_4edge_combos += 1
            
            e1, e2, e3, e4 = edge_combo_4
            
            # Get the 4 force directions
            f1 = edge_force_signatures[e1]
            f2 = edge_force_signatures[e2]
            f3 = edge_force_signatures[e3]
            f4 = edge_force_signatures[e4]
            
            # Test all 6 ways to combine 2 edges into 1 vector
            pair_combinations = [
                ((0, 1), (2, 3)),
                ((0, 2), (1, 3)),
                ((0, 3), (1, 2)),
                ((1, 2), (0, 3)),
                ((1, 3), (0, 2)),
                ((2, 3), (0, 1))
            ]
            
            forces = [f1, f2, f3, f4]
            edge_indices_list = [e1, e2, e3, e4]
            
            valid_reductions = []
            
            for pair_to_combine, remaining_singles in pair_combinations:
                i, j = pair_to_combine
                k, l = remaining_singles
                
                # Combine forces i and j
                f_combined = forces[i] + forces[j]
                
                # Create 3-vector system
                force_dirs_3 = [f_combined, forces[k], forces[l]]
                
                # Check force closure
                result = check_three_edge_force_closure(
                    force_dirs_3,
                    edge_indices=None,
                    verbose=False
                )
                
                if result['valid']:
                    valid_reductions.append({
                        'combined_edges': (edge_indices_list[i], edge_indices_list[j]),
                        'remaining_edges': (edge_indices_list[k], edge_indices_list[l]),
                        'combined_force': f_combined,
                        'coefficients': result['coefficients'],
                        'residual': result['residual']
                    })
            
            # If ANY reduction is valid, the 4-edge combo is valid
            if len(valid_reductions) > 0:
                valid_4edge_combos.append({
                    'edge_indices': list(edge_combo_4),
                    'valid_reductions': valid_reductions,
                    'num_valid_reductions': len(valid_reductions)
                })
                
                valid_4edge_set.add(tuple(sorted(edge_combo_4)))
                
                if verbose:
                    print(f"   4-edge combo {edge_combo_4}: ✅ VALID ({len(valid_reductions)}/6 reductions pass)")
        
        if verbose:
            print(f"\n   Total 4-edge combinations tested: {total_4edge_combos}")
            print(f"   Valid 4-edge combinations found: {len(valid_4edge_combos)}")
    else:
        if verbose:
            print(f"\n   Step 3: Skipped (include_4edge=False)")
    
    # =========================================================================
    # STEP 4: Compile statistics
    # =========================================================================
    statistics = {
        'num_edges': num_edges,
        'total_3edge_combos_tested': total_3edge_combos,
        'valid_3edge_combos_found': len(valid_3edge_combos),
        'total_4edge_combos_tested': total_4edge_combos,
        'valid_4edge_combos_found': len(valid_4edge_combos),
        'include_4edge': include_4edge
    }
    
    if verbose:
        print(f"\n📊 Preprocessing Summary:")
        print(f"   Edges: {num_edges}")
        print(f"   3-edge combos: {len(valid_3edge_combos)}/{total_3edge_combos} valid")
        if include_4edge:
            print(f"   4-edge combos: {len(valid_4edge_combos)}/{total_4edge_combos} valid")
        else:
            print(f"   4-edge combos: SKIPPED")
        print("="*80)
    
    return {
        'edge_characterizer': edge_characterizer,
        'edge_force_signatures': edge_force_signatures,
        'valid_3edge_combos': valid_3edge_combos,
        'valid_4edge_combos': valid_4edge_combos,
        'valid_3edge_set': valid_3edge_set,
        'valid_4edge_set': valid_4edge_set,
        'num_edges': num_edges,
        'statistics': statistics
    }



def is_valid_edge_combination(edge_indices, preprocess_result):
    """
    🆕 Quick O(1) check if an edge combination is valid for force closure.
    
    Handles special cases:
    - 3 edges: Direct lookup in valid_3edge_set
    - 4 edges with all distinct: Lookup in valid_4edge_set
    - 4 edges with 1 duplicate: Remove duplicate, lookup in valid_3edge_set
    - 4 edges with >1 duplicate: Return False (invalid configuration)
    
    Args:
        edge_indices: List or tuple of edge indices (3 or 4 edges)
        preprocess_result: Result dict from preprocess_object_force_closure()
    
    Returns:
        bool: True if valid for force closure, False otherwise
    """
    # Convert to list for easier manipulation
    edge_list = list(edge_indices)
    
    if len(edge_list) == 3:
        # Direct 3-edge lookup
        edge_tuple = tuple(sorted(edge_list))
        return edge_tuple in preprocess_result['valid_3edge_set']
    
    elif len(edge_list) == 4:
        # Count edge occurrences
        edge_counts = Counter(edge_list)
        max_count = max(edge_counts.values())
        
        if max_count > 2:
            # Edge appears more than twice - invalid configuration
            print(f"⚠️ Warning: Edge combination {edge_list} has edge appearing {max_count} times (max allowed: 2)")
            return False
        
        elif max_count == 2:
            # Exactly one edge appears twice - reduce to 3-edge check
            # Remove one duplicate to get the 3 distinct edges
            unique_edges = list(edge_counts.keys())
            
            if len(unique_edges) != 3:
                # Shouldn't happen with max_count==2, but safety check
                print(f"⚠️ Warning: Unexpected edge configuration {edge_list}")
                return False
            
            # Check in 3-edge set
            edge_tuple = tuple(sorted(unique_edges))
            return edge_tuple in preprocess_result['valid_3edge_set']
        
        else:
            # All 4 edges distinct (max_count == 1)
            edge_tuple = tuple(sorted(edge_list))
            return edge_tuple in preprocess_result['valid_4edge_set']
    
    else:
        # Invalid number of edges
        return False




def test_preprocessing_with_random_combos(obj_name='rectangle', num_tests=5, verbose=True):
    """
    🧪 Test function: Preprocess object, then test random edge combinations.
    
    Demonstrates:
    1. Preprocessing an object
    2. Quick O(1) validity checks for random 3-edge and 4-edge combinations
    
    Args:
        obj_name: Name of object from standard_objects
        num_tests: Number of random combinations to test
        verbose: If True, print detailed results
    
    Returns:
        dict: Test results and timing
    """
    import random
    import time
    
    if verbose:
        print("\n" + "="*80)
        print(f"🧪 TESTING PREPROCESSING WITH RANDOM COMBINATIONS")
        print(f"   Object: {obj_name}")
        print("="*80)
    
    # Get object
    standard_objects = create_standard_objects()
    obj = standard_objects[obj_name]
    
    # =========================================================================
    # STEP 1: Preprocessing
    # =========================================================================
    if verbose:
        print(f"\n⏱️  Running preprocessing...")
    
    start_time = time.time()
    preprocess_result = preprocess_object_force_closure(obj, verbose=verbose)
    preprocess_time = time.time() - start_time
    
    if verbose:
        print(f"\n   ✅ Preprocessing completed in {preprocess_time:.4f} seconds")
    
    num_edges = preprocess_result['num_edges']
    
    if num_edges < 3:
        print(f"\n   ⚠️ Object only has {num_edges} edges - cannot test 3-edge combinations")
        return {
            'success': False,
            'reason': 'insufficient_edges'
        }
    
    # =========================================================================
    # STEP 2: Test random 3-edge combinations
    # =========================================================================
    if verbose:
        print(f"\n🎲 Testing {num_tests} random 3-edge combinations...")
    
    test_3edge_results = []
    
    for i in range(num_tests):
        # Pick random 3 edges
        random_edges = tuple(sorted(random.sample(range(num_edges), 3)))
        
        # Quick O(1) lookup
        start = time.time()
        is_valid = is_valid_edge_combination(random_edges, preprocess_result)
        lookup_time = time.time() - start
        
        test_3edge_results.append({
            'edges': random_edges,
            'valid': is_valid,
            'lookup_time': lookup_time
        })
        
        if verbose:
            status = "✅ VALID" if is_valid else "❌ INVALID"
            print(f"   Test {i+1}: Edges {random_edges} → {status} (lookup: {lookup_time*1e6:.2f} μs)")
    
    # =========================================================================
    # STEP 3: Test random 4-edge combinations (if possible)
    # =========================================================================
    test_4edge_results = []
    
    if num_edges >= 4:
        if verbose:
            print(f"\n🎲 Testing {num_tests} random 4-edge combinations...")
        
        for i in range(num_tests):
            # Pick random 4 edges
            random_edges = tuple(sorted(random.sample(range(num_edges), 4)))
            
            # Quick O(1) lookup
            start = time.time()
            is_valid = is_valid_edge_combination(random_edges, preprocess_result)
            lookup_time = time.time() - start
            
            test_4edge_results.append({
                'edges': random_edges,
                'valid': is_valid,
                'lookup_time': lookup_time
            })
            
            if verbose:
                status = "✅ VALID" if is_valid else "❌ INVALID"
                print(f"   Test {i+1}: Edges {random_edges} → {status} (lookup: {lookup_time*1e6:.2f} μs)")
    else:
        if verbose:
            print(f"\n   ⚠️ Object only has {num_edges} edges - cannot test 4-edge combinations")
    
    # =========================================================================
    # STEP 4: Summary statistics
    # =========================================================================
    if verbose:
        print(f"\n📊 Test Summary:")
        print(f"   Preprocessing time: {preprocess_time:.4f} seconds")
        print(f"\n   3-edge tests:")
        valid_3 = sum(1 for r in test_3edge_results if r['valid'])
        avg_time_3 = np.mean([r['lookup_time'] for r in test_3edge_results]) * 1e6
        print(f"      Valid: {valid_3}/{num_tests}")
        print(f"      Avg lookup time: {avg_time_3:.2f} μs")
        
        if len(test_4edge_results) > 0:
            print(f"\n   4-edge tests:")
            valid_4 = sum(1 for r in test_4edge_results if r['valid'])
            avg_time_4 = np.mean([r['lookup_time'] for r in test_4edge_results]) * 1e6
            print(f"      Valid: {valid_4}/{num_tests}")
            print(f"      Avg lookup time: {avg_time_4:.2f} μs")
        
        print("="*80)
    
    return {
        'success': True,
        'obj_name': obj_name,
        'preprocess_time': preprocess_time,
        'preprocess_result': preprocess_result,
        'test_3edge_results': test_3edge_results,
        'test_4edge_results': test_4edge_results,
        'num_edges': num_edges
    }


# %%
# test_preprocessing_with_random_combos('rectangle', num_tests=5)
# test_preprocessing_with_random_combos('star', num_tests=5, verbose=False)



# %%
def _check_torque_closure_lp(contacts, verbose=False):
    """
    Option 1: LP test for 6 basic wrenches to verify torque closure.
    
    Tests if the grasp can generate:
    - +x, -x, +y, -y forces
    - +z, -z torques
    
    Args:
        contacts: List of ContactPoint objects
        verbose: If True, print detailed analysis
    
    Returns:
        dict: {
            'satisfied': bool,
            'test_results': list of individual wrench tests,
            'all_pass': bool
        }
    """
    n_contacts = len(contacts)
    
    # Build grasp matrix G
    G = np.zeros((3, n_contacts))
    
    for i, contact in enumerate(contacts):
        wrench = contact.calculate_contact_wrench(
            normal_force=1.0,
            tangential_force=0.0,
            friction_constraint=True
        )
        G[0, i] = wrench['force_x']
        G[1, i] = wrench['force_y']
        G[2, i] = wrench['torque']
    
    # Test 6 basic wrenches
    test_wrenches = [
        [1, 0, 0],   # +x force
        [-1, 0, 0],  # -x force
        [0, 1, 0],   # +y force
        [0, -1, 0],  # -y force
        [0, 0, 1],   # +z moment (torque)
        [0, 0, -1]   # -z moment (torque)
    ]
    
    test_results = []
    all_pass = True
    
    for w in test_wrenches:
        # Solve: G @ alpha = w, with alpha >= 0
        result = linprog(
            c=np.zeros(n_contacts),  # Minimize zero (just find feasibility)
            A_eq=G,
            b_eq=np.array(w),
            bounds=[(0, 1000)] * n_contacts,  # Max force per contact
            method='highs',
            options={'disp': False}
        )
        
        success = result.success
        test_results.append({
            'wrench': w,
            'success': success,
            'alpha': result.x if success else None
        })
        
        if not success:
            all_pass = False
        
        if verbose:
            status = '✓' if success else '✗'
            print(f"      w = {w}: {status}")
    
    return {
        'satisfied': all_pass,
        'test_results': test_results,
        'all_pass': all_pass
    }


def _check_torque_closure_convex_hull(contacts, epsilon_distance=0.1, verbose=False):
    """
    Option 2: Geometric test using 2D convex hull projections.
    
    Tests if origin is inside convex hull with epsilon margin in all 3 projections:
    - (Fx, Fy) plane
    - (Fy, τ) plane
    - (Fx, τ) plane
    
    Args:
        contacts: List of ContactPoint objects
        epsilon_distance: Minimum distance from origin to hull edge
        verbose: If True, print detailed analysis
    
    Returns:
        dict: {
            'satisfied': bool,
            'projection_results': dict with results for each projection,
            'all_pass': bool
        }
    """
    # Calculate wrench space
    wrench_visualizer = WrenchSpaceVisualizer()
    
    wrench_data = wrench_visualizer.calculate_wrench_space(
        contacts,
        force_ranges=[(0.0, 5.0)] * len(contacts),
        sampling_density=3,
        enable_tangent_forces=False
    )
    
    wrenches = wrench_data['wrenches']  # Shape: (N, 3) - [Fx, Fy, τ]
    
    # Test three 2D projections
    projections = {
        'Fx_Fy': (0, 1),      # (Fx, Fy) plane
        'Fy_Torque': (1, 2),  # (Fy, τ) plane
        'Fx_Torque': (0, 2)   # (Fx, τ) plane
    }
    
    projection_results = {}
    all_pass = True
    
    for proj_name, (axis1, axis2) in projections.items():
        # Project wrenches to 2D
        points_2d = wrenches[:, [axis1, axis2]]
        
        # Compute 2D convex hull
        try:
            hull = ConvexHull(points_2d)
            
            # Check if origin is inside convex hull
            origin = np.array([0.0, 0.0])
            hull_vertices_2d = points_2d[hull.vertices]
            
            origin_inside = _point_in_convex_hull_2d(origin, hull_vertices_2d)
            
            if not origin_inside:
                min_distance = 0.0
                contains_circle = False
            else:
                # Compute minimum distance from origin to hull edges
                min_distance = _min_distance_to_convex_hull_edges(origin, hull_vertices_2d)
                
                # Check if minimum distance >= epsilon_distance
                contains_circle = min_distance >= epsilon_distance
            
            projection_results[proj_name] = {
                'origin_inside': origin_inside,
                'min_distance': min_distance,
                'contains_circle': contains_circle,
                'hull_vertices': hull_vertices_2d
            }
            
            if not contains_circle:
                all_pass = False
            
            if verbose:
                status = '✓' if contains_circle else '✗'
                print(f"      {proj_name}: {status} (origin_inside={origin_inside}, min_dist={min_distance:.4f})")
        
        except Exception as e:
            # Degenerate hull (collinear points, etc.)
            if verbose:
                print(f"      {proj_name}: ✗ Failed (degenerate hull: {e})")
            
            projection_results[proj_name] = {
                'origin_inside': False,
                'min_distance': 0.0,
                'contains_circle': False,
                'error': str(e)
            }
            all_pass = False
    
    return {
        'satisfied': all_pass,
        'projection_results': projection_results,
        'all_pass': all_pass
    }


def _check_wrench_space_sufficiency_vs_limit_surface(
    contacts,
    obj,
    threshold: float = 1.0,
    n_ellipse_samples: int = 72,
    force_range_scalar: float = 2.0,
    enable_tangent_forces: bool = False,
    verbose: bool = False,
):
    """
    Check if the grasp wrench space (GWS) of a contact configuration is
    sufficient with respect to the object's Limit Surface (LS).

    Geometric interpretation:
    - GWS: Convex hull of achievable wrenches from current contacts
    - LS: Friction limit surface of the object (ellipsoid in (Fx, Fy, τ))
    - We require: GWS ⊇ threshold × LS

    Practical implementation:
    - Use existing `WrenchSpaceVisualizer.calculate_wrench_space()` to
      approximate GWS by sampling feasible contact forces.
    - Use existing `WrenchSpaceVisualizer.calculate_limit_surface()` to
      approximate LS and extract (f_max, m_max).
    - Project both spaces onto three planes:
        1) (Fx, Fy)
        2) (Fy, τ)
        3) (Fx, τ)
    - For each 2D projection, check that the projected convex hull of GWS
      contains a scaled ellipse corresponding to threshold × LS.

    NOTE:
    - This is an approximate geometric test based on sampling the ellipse
      boundary in each projection (n_ellipse_samples points).
    - The `threshold` parameter allows asking for stricter coverage than 100%
      of the LS (e.g. threshold=1.2).
    - The `force_range_scalar` multiplies the object's static friction limit
      to determine the maximum force range: [0, force_range_scalar × static_f_max]
    - The `enable_tangent_forces` flag is passed to `calculate_wrench_space`;
      if True, tangent forces are included and the GWS is larger (e.g. for fallback).
    """
    if verbose:
        print("\n" + "=" * 80)
        print("🔍 Wrench Space Sufficiency Check vs Limit Surface")
        print(f"   threshold = {threshold:.2f}")
        print(f"   force_range_scalar = {force_range_scalar:.2f}")
        print("=" * 80)

    if len(contacts) == 0:
        if verbose:
            print("   ❌ No contacts provided.")
        return {
            'satisfied': False,
            'reason': 'no_contacts',
            'projection_results': {},
        }

    # -------------------------------------------------------------------------
    # STEP 1: Compute approximate GWS via existing wrench space calculator
    # -------------------------------------------------------------------------
    wrench_visualizer = WrenchSpaceVisualizer()

    # Calculate maximum force based on object's static friction
    # static_f_max = static_friction * normal_force = static_friction * (mass * 9.81)
    normal_force = obj.mass * 9.81
    static_f_max = obj.static_friction * normal_force
    max_force = force_range_scalar * static_f_max

    if verbose:
        print(f"   Object static_f_max = {static_f_max:.4f} N")
        print(f"   Force range: [0.0, {max_force:.4f}] N (scalar = {force_range_scalar:.2f})")

    # Use computed force range based on object properties
    wrench_data = wrench_visualizer.calculate_wrench_space(
        contacts,
        force_ranges=[(0.0, max_force)] * len(contacts),
        sampling_density=3,
        enable_tangent_forces=enable_tangent_forces,
    )

    wrenches = wrench_data['wrenches']  # Shape: (N, 3) -> [Fx, Fy, τ]

    if wrenches.shape[0] == 0:
        if verbose:
            print("   ❌ No wrenches generated for this contact set.")
        return {
            'satisfied': False,
            'reason': 'empty_wrench_space',
            'projection_results': {},
        }

    # -------------------------------------------------------------------------
    # STEP 2: Compute Limit Surface for this object and scale by `threshold`
    # -------------------------------------------------------------------------
    ls_data = wrench_visualizer.calculate_limit_surface(
        obj,
        resolution=50,
        scaling_factor=threshold,
        grid_size=30,
    )

    f_max = ls_data['f_max']
    m_max = ls_data['m_max']

    if verbose:
        print(f"   Limit Surface (scaled by threshold): f_max={f_max:.4f}, m_max={m_max:.4f}")

    # -------------------------------------------------------------------------
    # STEP 3: Project GWS and LS onto 3 planes and test containment
    # -------------------------------------------------------------------------
    projections = {
        'Fx_Fy': (0, 1),      # (Fx, Fy)
        'Fy_Torque': (1, 2),  # (Fy, τ)
        'Fx_Torque': (0, 2),  # (Fx, τ)
    }

    projection_results = {}
    all_pass = True

    # Precompute ellipse samples (unit circle in parameter space)
    angles = np.linspace(0.0, 2.0 * np.pi, n_ellipse_samples, endpoint=False)

    for proj_name, (axis1, axis2) in projections.items():
        # 3.1: Project wrenches to 2D
        points_2d = wrenches[:, [axis1, axis2]]

        try:
            hull = ConvexHull(points_2d)
            hull_vertices_2d = points_2d[hull.vertices]
        except Exception as e:
            if verbose:
                print(f"   {proj_name}: ❌ Convex hull computation failed: {e}")

            projection_results[proj_name] = {
                'contains_scaled_ls': False,
                'reason': 'hull_failure',
                'min_margin': 0.0,
            }
            all_pass = False
            continue

        # 3.2: Build the projected LS ellipse in this plane
        if proj_name == 'Fx_Fy':
            # Circle of radius f_max in Fx-Fy
            ellipse_points = np.column_stack(
                [f_max * np.cos(angles), f_max * np.sin(angles)]
            )
        elif proj_name == 'Fx_Torque':
            # Ellipse in Fx-τ: (Fx/f_max)^2 + (τ/m_max)^2 = 1
            ellipse_points = np.column_stack(
                [f_max * np.cos(angles), m_max * np.sin(angles)]
            )
        elif proj_name == 'Fy_Torque':
            # Ellipse in Fy-τ: (Fy/f_max)^2 + (τ/m_max)^2 = 1
            ellipse_points = np.column_stack(
                [f_max * np.cos(angles), m_max * np.sin(angles)]
            )
        else:
            # Should not happen
            ellipse_points = np.zeros((len(angles), 2))

        # 3.3: Check if ALL ellipse boundary points lie inside the convex hull
        all_inside = True
        min_margin = float('inf')

        for pt in ellipse_points:
            inside = _point_in_convex_hull_2d(pt, hull_vertices_2d)
            if not inside:
                all_inside = False
                min_margin = 0.0
                break

            # Optional: track "margin" as distance to hull edges (approximate)
            dist = _min_distance_to_convex_hull_edges(pt, hull_vertices_2d)
            min_margin = min(min_margin, dist)

        projection_results[proj_name] = {
            'contains_scaled_ls': all_inside,
            'min_margin': float(min_margin if np.isfinite(min_margin) else 0.0),
            'hull_vertices': hull_vertices_2d,
        }

        if verbose:
            status = "✓" if all_inside else "✗"
            print(
                f"   {proj_name}: {status}  "
                f"(min margin to hull edges ≈ {projection_results[proj_name]['min_margin']:.4f})"
            )

        if not all_inside:
            all_pass = False

    if verbose:
        print("\n   RESULT:", "✅ SUFFICIENT" if all_pass else "❌ INSUFFICIENT")

    return {
        'satisfied': all_pass,
        'projection_results': projection_results,
        'f_max': f_max,
        'm_max': m_max,
        'threshold': threshold,
    }


def check_wrench_space_sufficiency(
    contacts,
    obj,
    threshold: float = 1.0,
    n_ellipse_samples: int = 72,
    force_range_scalar: float = 2.0,
    enable_tangent_forces: bool = False,
    verbose: bool = False,
):
    """
    Phase 1 core: public "sufficiency check" for Magnum Stochastic.

    Concept:
        Check whether the grasp wrench space (GWS) generated by `contacts`
        is sufficient with respect to the object's Limit Surface (LS).

    Geometric test:
        - Compute GWS via `WrenchSpaceVisualizer.calculate_wrench_space`
        - Compute LS ellipsoid via `WrenchSpaceVisualizer.calculate_limit_surface`
        - Project both onto three planes:
              (Fx, Fy), (Fy, τ), (Fx, τ)
        - In each 2D plane, check that the convex hull of the projected GWS
          contains a scaled ellipse corresponding to `threshold × LS`.

    Args:
        contacts: list of `ContactPoint` objects defining the grasp.
        obj: `GenericObject` instance (provides geometry + friction for LS).
        threshold: float ≥ 1.0 typically.
            - 1.0  → GWS must contain 100% of LS (default "good enough").
            - >1.0 → require margin beyond LS (stricter sufficiency).
        n_ellipse_samples: number of points sampled along each projected
            LS ellipse boundary when testing containment.
        force_range_scalar: float multiplier for maximum force range.
            - Force range = [0, force_range_scalar × static_f_max]
            - static_f_max = static_friction × (mass × 9.81)
            - Default 2.0 means robots can exert up to 2× the static friction limit
        enable_tangent_forces: if True, GWS is built with tangent forces enabled
            (larger wrench space; can be used as fallback when normal-only check fails).
        verbose: if True, prints a short textual summary of the test.

    Returns:
        dict with at least:
            'satisfied'          : bool  (True ⇔ GWS ⊇ threshold × LS in all 3 planes)
            'projection_results' : per‑projection diagnostics
            'f_max', 'm_max'     : LS parameters actually used (after scaling)
            'threshold'          : the threshold that was applied
    """
    result = _check_wrench_space_sufficiency_vs_limit_surface(
        contacts,
        obj,
        threshold=threshold,
        n_ellipse_samples=n_ellipse_samples,
        force_range_scalar=force_range_scalar,
        enable_tangent_forces=enable_tangent_forces,
        verbose=verbose,
    )

    if verbose:
        status = "SUFFICIENT ✅" if result['satisfied'] else "INSUFFICIENT ❌"
        print(
            f"\n[check_wrench_space_sufficiency] {status} "
            f"(threshold={result['threshold']:.2f}, "
            f"f_max={result['f_max']:.3f}, m_max={result['m_max']:.3f})"
        )

    return result



def _point_in_convex_hull_2d(point, hull_vertices):
    """
    Check if a point is inside a 2D convex hull using cross product test.
    
    Args:
        point: np.array([x, y]) - point to test
        hull_vertices: np.array of shape (N, 2) - ordered hull vertices
    
    Returns:
        bool: True if point is inside hull
    """
    n = len(hull_vertices)
    
    for i in range(n):
        v1 = hull_vertices[i]
        v2 = hull_vertices[(i + 1) % n]
        
        # Vector from v1 to v2
        edge = v2 - v1
        
        # Vector from v1 to point
        to_point = point - v1
        
        # Cross product (2D): edge × to_point
        cross = edge[0] * to_point[1] - edge[1] * to_point[0]
        
        # If cross product is negative, point is on wrong side of this edge
        if cross < -1e-10:  # Small tolerance for numerical errors
            return False
    
    return True


def _min_distance_to_convex_hull_edges(point, hull_vertices):
    """
    Compute minimum distance from a point to the edges of a 2D convex hull.
    
    Args:
        point: np.array([x, y]) - point (typically origin)
        hull_vertices: np.array of shape (N, 2) - ordered hull vertices
    
    Returns:
        float: minimum distance to any edge
    """
    n = len(hull_vertices)
    min_dist = float('inf')
    
    for i in range(n):
        v1 = hull_vertices[i]
        v2 = hull_vertices[(i + 1) % n]
        
        # Compute distance from point to line segment [v1, v2]
        dist = _distance_point_to_segment(point, v1, v2)
        min_dist = min(min_dist, dist)
    
    return min_dist



def _check_torque_closure_geometric_intersection(contacts, verbose=False):
    """
    Option 3: Geometric force closure test using force line intersections.
    
    Theoretical condition:
    For a valid grasp, there must exist a partition of 4 contacts into two pairs
    (i,j) and (k,l) such that:
    
    1. Three of the four force directions do NOT intersect at a common point or infinity
    2. Let P_ij = intersection of force lines i and j
       Let P_kl = intersection of force lines k and l
       Then: P_kl - P_ij = ±(α_i × dir_i + α_j × dir_j) = ±(α_k × dir_k + α_l × dir_l)
       where all α > 0
    
    The ± means: (α_i, α_j) have same sign, and (α_k, α_l) have same sign.
    
    Geometric interpretation:
    The segment P_ij → P_kl must point into and out of the two force cones.
    
    Args:
        contacts: List of 4 ContactPoint objects
        verbose: If True, print detailed analysis
    
    Returns:
        dict: {
            'satisfied': bool,
            'valid_partition': tuple or None,  # (i, j, k, l) if found
            'P_ij': np.array or None,
            'P_kl': np.array or None,
            'alphas': dict or None,  # {'alpha_i': float, 'alpha_j': float, ...}
            'reason': str
        }
    """
    
    if verbose:
        print(f"\n   🔷 Option 3: Geometric intersection test...")
    
    # Extract force directions (inward normals)
    force_dirs = [contact.normal_inward for contact in contacts]
    positions = [contact.position for contact in contacts]
    
    # Test all three possible pair partitions
    partitions = [
        (0, 1, 2, 3),  # Pair (0,1) vs Pair (2,3)
        (0, 2, 1, 3),  # Pair (0,2) vs Pair (1,3)
        (0, 3, 1, 2)   # Pair (0,3) vs Pair (1,2)
    ]
    
    for partition_idx, (i, j, k, l) in enumerate(partitions):
        if verbose:
            print(f"\n      Testing partition {partition_idx + 1}: Pair ({i},{j}) vs Pair ({k},{l})")
        
        # =====================================================================
        # STEP 1: Check condition 1 - no three forces intersect at common point
        # =====================================================================
        
        # For each triple of contacts, check if their force lines meet at a point
        triples = [
            (i, j, k),
            (i, j, l),
            (i, k, l),
            (j, k, l)
        ]
        
        has_common_intersection = False
        
        for triple in triples:
            # Check if these three force lines intersect at a common point
            common_point = _check_three_lines_common_intersection(
                positions, force_dirs, triple
            )
            
            if common_point is not None:
                has_common_intersection = True
                if verbose:
                    print(f"         ✗ Forces {triple} intersect at common point {common_point}")
                break
        
        if has_common_intersection:
            if verbose:
                print(f"         ✗ Condition 1 failed - three forces have common intersection")
            continue  # Try next partition
        
        if verbose:
            print(f"         ✓ Condition 1 passed - no three forces share common intersection")
        
        # =====================================================================
        # STEP 2: Find intersection points P_ij and P_kl
        # =====================================================================
        
        # P_ij: intersection of force lines from contacts i and j
        P_ij = _intersect_two_force_lines(
            positions[i], force_dirs[i],
            positions[j], force_dirs[j]
        )
        
        if P_ij is None:
            if verbose:
                print(f"         ✗ Forces {i} and {j} are parallel - cannot compute P_ij")
            continue
        
        # P_kl: intersection of force lines from contacts k and l
        P_kl = _intersect_two_force_lines(
            positions[k], force_dirs[k],
            positions[l], force_dirs[l]
        )
        
        if P_kl is None:
            if verbose:
                print(f"         ✗ Forces {k} and {l} are parallel - cannot compute P_kl")
            continue
        
        if verbose:
            print(f"         P_ij = {P_ij}")
            print(f"         P_kl = {P_kl}")
        
        # =====================================================================
        # STEP 3: Check condition 2 - vector equation with positive alphas
        # =====================================================================
        
        # Vector from P_ij to P_kl
        segment_vec = P_kl - P_ij
        
        if verbose:
            print(f"         Segment vector P_kl - P_ij = {segment_vec}")
        
        # We need to solve for α_i, α_j, α_k, α_l > 0 such that:
        # segment_vec = s_ij × (α_i × dir_i + α_j × dir_j)
        #             = s_kl × (α_k × dir_k + α_l × dir_l)
        # where s_ij, s_kl ∈ {+1, -1} and (s_ij = s_kl) OR (s_ij = -s_kl)
        
        # This gives us 4 cases to check:
        # Case 1: + (pair_ij) = + (pair_kl)
        # Case 2: + (pair_ij) = - (pair_kl)
        # Case 3: - (pair_ij) = + (pair_kl)
        # Case 4: - (pair_ij) = - (pair_kl)
        
        cases = [
            (+1, +1),  # Case 1
            (+1, -1),  # Case 2
            (-1, +1),  # Case 3
            (-1, -1)   # Case 4
        ]
        
        solution_found = False
        best_alphas = None
        
        for case_idx, (sign_ij, sign_kl) in enumerate(cases):
            # Solve linear system for this case
            # LHS: segment_vec = sign_ij × (α_i × dir_i + α_j × dir_j)
            # RHS: segment_vec = sign_kl × (α_k × dir_k + α_l × dir_l)
            
            # Build matrix equation: A × α = b
            # where α = [α_i, α_j, α_k, α_l]
            
            # From LHS: sign_ij × α_i × dir_i + sign_ij × α_j × dir_j = segment_vec
            # From RHS: sign_kl × α_k × dir_k + sign_kl × α_l × dir_l = segment_vec
            
            # This gives 2 equations (Fx and Fy) from each side
            # Total: 4 equations, 4 unknowns
            
            # However, we only need 2 equations (Fx and Fy)
            # Use LHS: sign_ij × (α_i × dir_i + α_j × dir_j) = segment_vec
            
            A = np.column_stack([
                sign_ij * force_dirs[i],
                sign_ij * force_dirs[j]
            ])  # Shape: (2, 2)
            
            b = segment_vec  # Shape: (2,)
            
            try:
                # Solve for α_i, α_j
                alphas_ij = np.linalg.solve(A, b)
                alpha_i = alphas_ij[0]
                alpha_j = alphas_ij[1]
                
                # Check if both positive
                if alpha_i > 0 and alpha_j > 0:
                    # Now solve for α_k, α_l using RHS
                    A_kl = np.column_stack([
                        sign_kl * force_dirs[k],
                        sign_kl * force_dirs[l]
                    ])
                    
                    alphas_kl = np.linalg.solve(A_kl, b)
                    alpha_k = alphas_kl[0]
                    alpha_l = alphas_kl[1]
                    
                    # Check if both positive
                    if alpha_k > 0 and alpha_l > 0:
                        # Valid solution found!
                        solution_found = True
                        best_alphas = {
                            'alpha_i': alpha_i,
                            'alpha_j': alpha_j,
                            'alpha_k': alpha_k,
                            'alpha_l': alpha_l,
                            'sign_ij': sign_ij,
                            'sign_kl': sign_kl
                        }
                        
                        if verbose:
                            print(f"\n         ✓ Valid solution found (Case {case_idx + 1})")
                            print(f"            Sign pair: ({sign_ij:+d}, {sign_kl:+d})")
                            print(f"            α_{i} = {alpha_i:.4f}, α_{j} = {alpha_j:.4f}")
                            print(f"            α_{k} = {alpha_k:.4f}, α_{l} = {alpha_l:.4f}")
                        
                        break
            
            except np.linalg.LinAlgError:
                # Singular matrix - directions are parallel
                continue
        
        if solution_found:
            # Return successful result
            return {
                'satisfied': True,
                'valid_partition': (i, j, k, l),
                'P_ij': P_ij,
                'P_kl': P_kl,
                'alphas': best_alphas,
                'segment_vector': segment_vec,
                'reason': 'geometric_intersection_satisfied'
            }
    
    # No valid partition found
    if verbose:
        print(f"\n      ✗ No valid partition found - geometric condition not satisfied")
    
    return {
        'satisfied': False,
        'valid_partition': None,
        'P_ij': None,
        'P_kl': None,
        'alphas': None,
        'reason': 'no_valid_partition'
    }


def _check_three_lines_common_intersection(positions, force_dirs, triple):
    """
    Check if three force lines intersect at a common point.
    
    Args:
        positions: list of contact positions
        force_dirs: list of force directions
        triple: tuple of 3 indices (i, j, k)
    
    Returns:
        np.array or None: common intersection point if exists, else None
    """
    i, j, k = triple
    
    # Find intersection of line i and line j
    P_ij = _intersect_two_force_lines(
        positions[i], force_dirs[i],
        positions[j], force_dirs[j]
    )
    
    if P_ij is None:
        return None  # Lines i and j are parallel
    
    # Check if line k passes through P_ij
    # Point-to-line distance
    pos_k = positions[k]
    dir_k = force_dirs[k]
    
    # Distance from P_ij to line k
    dist = _point_to_line_distance(P_ij, pos_k, dir_k)
    
    tolerance = 1e-6
    if dist < tolerance:
        return P_ij  # All three lines meet at P_ij
    
    return None


def _intersect_two_force_lines(pos1, dir1, pos2, dir2):
    """
    Find intersection point of two force lines.
    
    Line 1: p = pos1 + t1 × dir1
    Line 2: p = pos2 + t2 × dir2
    
    Args:
        pos1: np.array - position of contact 1
        dir1: np.array - force direction of contact 1
        pos2: np.array - position of contact 2
        dir2: np.array - force direction of contact 2
    
    Returns:
        np.array or None: intersection point if exists, else None (parallel lines)
    """
    # Solve: pos1 + t1 × dir1 = pos2 + t2 × dir2
    # Rearrange: t1 × dir1 - t2 × dir2 = pos2 - pos1
    
    # In 2D: [dir1_x, -dir2_x] [t1]   [pos2_x - pos1_x]
    #        [dir1_y, -dir2_y] [t2] = [pos2_y - pos1_y]
    
    A = np.column_stack([dir1, -dir2])
    b = pos2 - pos1
    
    try:
        t_values = np.linalg.solve(A, b)
        t1 = t_values[0]
        
        # Compute intersection point
        intersection = pos1 + t1 * dir1
        
        return intersection
    
    except np.linalg.LinAlgError:
        # Lines are parallel
        return None


def _point_to_line_distance(point, line_pos, line_dir):
    """
    Compute distance from a point to an infinite line.
    
    Args:
        point: np.array - point position
        line_pos: np.array - a point on the line
        line_dir: np.array - line direction (unit vector)
    
    Returns:
        float: perpendicular distance
    """
    # Vector from line_pos to point
    vec = point - line_pos
    
    # Project onto line direction
    projection_length = np.dot(vec, line_dir)
    
    # Perpendicular component
    projection = projection_length * line_dir
    perpendicular = vec - projection
    
    return np.linalg.norm(perpendicular)



# %%
def _check_force_and_torque_closure(contacts, edge_characterizer, preprocess_result,
                                     torque_method=3, epsilon_distance=0.1, verbose=False):
    """
    🆕 REFACTORED: Check force and torque closure using preprocessing + selected torque method.
    
    Force Closure Check:
    - Uses preprocess_result and is_valid_edge_combination() for O(1) lookup
    - If force closure fails, immediately return without torque check
    
    Torque Closure Check (only if force closure passes):
    - Method 1: LP test for 6 basic wrenches
    - Method 2: Geometric test (2D convex hull projections)
    - Method 3: Geometric intersection test (default)
    
    Args:
        contacts: List of ContactPoint objects
        edge_characterizer: EdgeCharacterizer instance
        preprocess_result: Result dict from preprocess_object_force_closure()
        torque_method: 1 (LP), 2 (convex hull), or 3 (geometric intersection)
        epsilon_distance: Minimum distance from origin to hull (for method 2)
        verbose: If True, print detailed analysis
    
    Returns:
        dict: {
            'force_closure': bool,
            'torque_closure': bool,
            'method': str,
            'force_check': dict,
            'torque_check': dict or None,
            'overall_pass': bool
        }
    """
    
    if verbose:
        print(f"\n🔍 Checking force and torque closure...")
        print(f"   Torque method: {torque_method}")
    
    # =========================================================================
    # STEP 1: FORCE CLOSURE CHECK (O(1) with preprocessing)
    # =========================================================================
    if verbose:
        print(f"\n   🔬 Step 1: Force closure check (via preprocessing)...")
    
    # Extract edge indices from contacts
    edge_indices = []
    for contact in contacts:
        t_param = contact.parameter
        
        # Find which edge this contact is on
        found_edge = False
        for edge_idx in range(preprocess_result['num_edges']):
            edge_name = f'edge_{edge_idx}'
            char = edge_characterizer.edge_characteristics[edge_name]
            edge_info = char['edge_info']
            
            if edge_info['start_param'] <= t_param <= edge_info['end_param']:
                edge_indices.append(edge_idx)
                found_edge = True
                break
        
        if not found_edge:
            # Shouldn't happen, but handle gracefully
            edge_indices.append(-1)
    
    if verbose:
        print(f"      Contact edge indices: {edge_indices}")
    
    # Quick O(1) validity check
    force_closure_valid = is_valid_edge_combination(edge_indices, preprocess_result)
    
    force_check_result = {
        'edge_indices': edge_indices,
        'valid': force_closure_valid,
        'method': 'preprocessing_lookup'
    }
    
    if not force_closure_valid:
        if verbose:
            print(f"      ❌ Force closure FAILED (invalid edge combination)")
        
        return {
            'force_closure': False,
            'torque_closure': False,
            'method': 'preprocessing_only',
            'force_check': force_check_result,
            'torque_check': None,
            'overall_pass': False
        }
    
    if verbose:
        print(f"      ✅ Force closure PASSED")
    
    # =========================================================================
    # STEP 2: TORQUE CLOSURE CHECK (only if force closure passed)
    # =========================================================================
    if verbose:
        print(f"\n   🔬 Step 2: Torque closure check (method {torque_method})...")
    
    if torque_method == 1:
        # Method 1: LP test
        if verbose:
            print(f"      Using LP test for 6 basic wrenches...")
        
        torque_check_result = _check_torque_closure_lp(contacts, verbose=verbose)
        torque_closure_valid = torque_check_result['satisfied']
    
    elif torque_method == 2:
        # Method 2: Convex hull projections
        if verbose:
            print(f"      Using 2D convex hull projections...")
        
        torque_check_result = _check_torque_closure_convex_hull(
            contacts, 
            epsilon_distance=epsilon_distance,
            verbose=verbose
        )
        torque_closure_valid = torque_check_result['satisfied']
    
    elif torque_method == 3:
        # Method 3: Geometric intersection
        if verbose:
            print(f"      Using geometric intersection test...")
        
        torque_check_result = _check_torque_closure_geometric_intersection(
            contacts,
            verbose=verbose
        )
        torque_closure_valid = torque_check_result.get('satisfied', False)
    
    else:
        # Invalid method
        if verbose:
            print(f"      ❌ Invalid torque method: {torque_method}")
        
        return {
            'force_closure': True,
            'torque_closure': False,
            'method': 'invalid_torque_method',
            'force_check': force_check_result,
            'torque_check': {'error': f'Invalid method {torque_method}'},
            'overall_pass': False
        }
    
    if verbose:
        if torque_closure_valid:
            print(f"      ✅ Torque closure PASSED")
        else:
            print(f"      ❌ Torque closure FAILED")
    
    # =========================================================================
    # STEP 3: OVERALL RESULT
    # =========================================================================
    overall_pass = force_closure_valid and torque_closure_valid
    
    if verbose:
        print(f"\n   📊 Overall: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    return {
        'force_closure': force_closure_valid,
        'torque_closure': torque_closure_valid,
        'method': f'preprocessing_force + method{torque_method}_torque',
        'force_check': force_check_result,
        'torque_check': torque_check_result,
        'overall_pass': overall_pass
    }

print("✅ _check_force_and_torque_closure() implemented!")
print("\n📝 Three-stage approach:")
print("   Stage 1: Null space test (quick rejection)")
print("   Stage 2: Three verification methods:")
print("      • Option 1: LP test for 6 basic wrenches")
print("      • Option 2: Geometric test (2D convex hull projections)")
print("      • Option 3: (Placeholder - to be implemented)")
print("\n🔍 Closure satisfied if:")
print("   • Passes Stage 1 (null space has positive combination)")
print("   • AND passes at least one Stage 2 method")




# %%
def _compute_grasp_quality_metrics(contacts, edge_characterizer, weighting_scheme='balanced', verbose=False):
    """
    Compute grasp quality metrics for ranking solutions.
    
    🆕 Added weighting schemes and wrench space radius metric.
    
    Args:
        contacts: List of 4 ContactPoint objects
        edge_characterizer: EdgeCharacterizer instance
        weighting_scheme: 'balanced', 'focus_translational', or 'focus_rotational'
        verbose: If True, print detailed metrics
    
    Returns:
        dict: {
            'overall_score': float,
            'individual_metrics': dict with breakdown,
            'weighting_scheme': str
        }
    """
    if verbose:
        print(f"\n📊 Computing grasp quality metrics (scheme: {weighting_scheme})...")
    
    # Extract contact information
    n_contacts = len(contacts)
    
    # Get edge indices for each contact
    edge_indices = []
    t_params = []
    force_directions = []
    positions = []
    
    for contact in contacts:
        # Find which edge this contact is on
        t_param = contact.parameter
        t_params.append(t_param)
        positions.append(contact.position)
        force_directions.append(contact.normal_inward)
        
        # Find edge index
        edge_idx = -1
        for i, edge_name in enumerate(edge_characterizer.edge_characteristics.keys()):
            char = edge_characterizer.edge_characteristics[edge_name]
            edge_info = char['edge_info']
            if edge_info['start_param'] <= t_param <= edge_info['end_param']:
                edge_idx = i
                break
        edge_indices.append(edge_idx)
    
    # Build grasp matrix G (3×4) - needed for multiple metrics
    G = np.zeros((3, n_contacts))
    
    for i, contact in enumerate(contacts):
        unit_wrench = contact.calculate_contact_wrench(
            normal_force=1.0,
            tangential_force=0.0,
            friction_constraint=True
        )
        
        G[0, i] = unit_wrench['force_x']
        G[1, i] = unit_wrench['force_y']
        G[2, i] = unit_wrench['torque']
    
    # =========================================================================
    # 🆕 METRIC 0: WRENCH SPACE INSCRIBED SPHERE RADIUS (MOST IMPORTANT!)
    # =========================================================================
    if verbose:
        print(f"   🌌 Wrench Space Inscribed Sphere Radius:")
    
    # Calculate wrench space using WrenchSpaceVisualizer
    wrench_visualizer = WrenchSpaceVisualizer()
    
    wrench_data = wrench_visualizer.calculate_wrench_space(
        contacts,
        force_ranges=[(0.0, 5.0)] * len(contacts),
        sampling_density=3,
        enable_tangent_forces=False
    )
    
    wrenches = wrench_data['wrenches']  # Shape: (N, 3) - [Fx, Fy, τ]
    
    # Test three 2D projections and compute minimum distance to hull edges
    projections = {
        'Fx_Fy': (0, 1),
        'Fy_Torque': (1, 2),
        'Fx_Torque': (0, 2)
    }
    
    min_distances = []
    
    for proj_name, (axis1, axis2) in projections.items():
        points_2d = wrenches[:, [axis1, axis2]]
        
        try:
            hull = ConvexHull(points_2d)
            origin = np.array([0.0, 0.0])
            hull_vertices_2d = points_2d[hull.vertices]
            
            # Check if origin is inside
            origin_inside = _point_in_convex_hull_2d(origin, hull_vertices_2d)
            
            if origin_inside:
                min_distance = _min_distance_to_convex_hull_edges(origin, hull_vertices_2d)
                min_distances.append(min_distance)
                
                if verbose:
                    print(f"      {proj_name}: min_dist={min_distance:.4f}")
            else:
                # Origin outside - set to 0
                min_distances.append(0.0)
                if verbose:
                    print(f"      {proj_name}: min_dist=0.0000 (origin outside)")
        
        except Exception as e:
            min_distances.append(0.0)
            if verbose:
                print(f"      {proj_name}: min_dist=0.0000 (error: {e})")
    
    # Maximum inscribed sphere radius = minimum of the three projection distances
    if len(min_distances) > 0:
        wrench_space_radius = min(min_distances)
    else:
        wrench_space_radius = 0.0
    
    # Normalize by typical force magnitude (5.0N) and moment arm (~0.5m)
    # Typical wrench magnitude = sqrt(Fx² + Fy² + τ²) ≈ sqrt(5² + 5² + (5×0.5)²) ≈ 7.5
    typical_wrench_magnitude = 7.5
    normalized_wrench_radius = wrench_space_radius / typical_wrench_magnitude
    
    # Convert to score (sigmoid-like): score = 1 - exp(-k × radius)
    # where k controls sensitivity (higher k = more sensitive to small radii)
    k_sensitivity = 5.0  # Tune this to adjust scoring curve, used to be 2
    wrench_space_score = 1.0 - np.exp(-k_sensitivity * normalized_wrench_radius)
    
    if verbose:
        print(f"      Max inscribed sphere radius: {wrench_space_radius:.4f}")
        print(f"      Normalized radius: {normalized_wrench_radius:.4f}")
        print(f"      Score: {wrench_space_score:.4f}")
    
    # =========================================================================
    # METRIC 1: TRANSLATIONAL ROBUSTNESS (Angular Diversity)
    # =========================================================================
    angles = []
    for force_dir in force_directions:
        angle = np.arctan2(force_dir[1], force_dir[0])
        angles.append(angle)
    
    angles_sorted = sorted(angles)
    
    angular_gaps = []
    n_angles = len(angles_sorted)
    for i in range(n_angles):
        next_i = (i + 1) % n_angles
        gap = angles_sorted[next_i] - angles_sorted[i]
        if gap < 0:
            gap += 2 * np.pi
        angular_gaps.append(gap)
    
    ideal_gap = 2 * np.pi / n_contacts
    gap_variance = sum((gap - ideal_gap)**2 for gap in angular_gaps)
    translational_robustness_score = np.exp(-gap_variance / (ideal_gap**2 + 1e-10))
    
    if verbose:
        print(f"   Translational Robustness:")
        print(f"      Angular gaps: {[f'{g*180/np.pi:.1f}°' for g in angular_gaps]}")
        print(f"      Score: {translational_robustness_score:.4f}")
    
    # =========================================================================
    # METRIC 2: ROTATIONAL ROBUSTNESS (Parameter Spread)
    # =========================================================================
    edge_contact_params = {}
    
    for edge_idx, t_param in zip(edge_indices, t_params):
        if edge_idx not in edge_contact_params:
            edge_contact_params[edge_idx] = []
        edge_contact_params[edge_idx].append(t_param)
    
    parameter_spread_scores = []
    
    for edge_idx, params_on_edge in edge_contact_params.items():
        if len(params_on_edge) < 2:
            continue
        
        edge_name = f'edge_{edge_idx}'
        char = edge_characterizer.edge_characteristics[edge_name]
        edge_info = char['edge_info']
        t_start = edge_info['start_param']
        t_end = edge_info['end_param']
        edge_range = t_end - t_start
        
        if edge_range < 1e-10:
            continue
        
        params_sorted = sorted(params_on_edge)
        max_gap = max(params_sorted[i+1] - params_sorted[i] 
                     for i in range(len(params_sorted) - 1))
        
        normalized_spread = max_gap / edge_range
        parameter_spread_scores.append(normalized_spread)
    
    if len(parameter_spread_scores) > 0:
        parameter_spread_score = np.mean(parameter_spread_scores)
    else:
        parameter_spread_score = 1.0
    
    if verbose:
        print(f"   Rotational Robustness (Parameter Spread):")
        print(f"      Overall score: {parameter_spread_score:.4f}")
    
    # =========================================================================
    # METRIC 3: ROTATIONAL ROBUSTNESS (Torque Diversity)
    # =========================================================================
    torque_capabilities = []
    
    for edge_idx, t_param in zip(edge_indices, t_params):
        edge_name = f'edge_{edge_idx}'
        char = edge_characterizer.edge_characteristics[edge_name]
        
        torque_slope = char['torque_slope']
        torque_offset = char['torque_offset']
        
        torque_capability = abs(torque_slope * t_param + torque_offset)
        torque_capabilities.append(torque_capability)
    
    torque_mean = np.mean(torque_capabilities)
    torque_std = np.std(torque_capabilities)
    
    if torque_mean > 1e-10:
        torque_diversity_score = min(torque_std / torque_mean, 1.0)
    else:
        torque_diversity_score = 0.0
    
    if verbose:
        print(f"   Rotational Robustness (Torque Diversity):")
        print(f"      Diversity score: {torque_diversity_score:.4f}")
    
    # =========================================================================
    # METRIC 4: GRASP MATRIX CONDITION NUMBER
    # =========================================================================
    singular_values = np.linalg.svd(G, compute_uv=False)
    
    max_sv = np.max(singular_values)
    min_sv = np.min(singular_values[singular_values > 1e-10])
    
    if min_sv > 1e-10:
        condition_number = max_sv / min_sv
        condition_score = 1.0 / (1.0 + np.log10(condition_number))
    else:
        condition_score = 0.0
    
    if verbose:
        print(f"   Grasp Matrix Condition Number:")
        print(f"      Condition score: {condition_score:.4f}")
    
    # =========================================================================
    # METRIC 5: CONTACT SPATIAL DISTRIBUTION
    # =========================================================================
    pairwise_distances = []
    for i in range(n_contacts):
        for j in range(i+1, n_contacts):
            dist = np.linalg.norm(positions[i] - positions[j])
            pairwise_distances.append(dist)
    
    mean_dist = np.mean(pairwise_distances)
    dist_variance = np.var(pairwise_distances)
    
    if mean_dist > 1e-10:
        spatial_distribution_score = np.exp(-dist_variance / (mean_dist**2 + 1e-10))
    else:
        spatial_distribution_score = 0.0
    
    if verbose:
        print(f"   Spatial Distribution:")
        print(f"      Distribution score: {spatial_distribution_score:.4f}")
    
    # =========================================================================
    # 🆕 COMBINE METRICS WITH SELECTABLE WEIGHTING SCHEMES
    # =========================================================================
    
    weighting_schemes = {
        'balanced': {
            'wrench_space_radius': 0.40,  # 🆕 Most important!
            'translational_robustness': 0.15,
            'parameter_spread': 0.15,
            'torque_diversity': 0.15,
            'condition_number': 0.10,
            'spatial_distribution': 0.05
        },
        'focus_translational': {
            'wrench_space_radius': 0.35,
            'translational_robustness': 0.30,  # Focus on force directions
            'parameter_spread': 0.10,
            'torque_diversity': 0.10,
            'condition_number': 0.10,
            'spatial_distribution': 0.05
        },
        'focus_rotational': {
            'wrench_space_radius': 0.65,
            'translational_robustness': 0.05,
            'parameter_spread': 0.1,  # Focus on parameter spread
            'torque_diversity': 0.1,  # Focus on torque diversity
            'condition_number': 0.05,
            'spatial_distribution': 0.05
        }
    }
    
    # Select weights
    if weighting_scheme not in weighting_schemes:
        print(f"⚠️ Unknown weighting scheme '{weighting_scheme}', using 'balanced'")
        weighting_scheme = 'balanced'
    
    weights = weighting_schemes[weighting_scheme]
    
    overall_score = (
        weights['wrench_space_radius'] * wrench_space_score +
        weights['translational_robustness'] * translational_robustness_score +
        weights['parameter_spread'] * parameter_spread_score +
        weights['torque_diversity'] * torque_diversity_score +
        weights['condition_number'] * condition_score +
        weights['spatial_distribution'] * spatial_distribution_score
    )
    
    if verbose:
        print(f"\n   📊 Overall Quality Score ({weighting_scheme}): {overall_score:.4f}")
        print(f"      Breakdown:")
        print(f"         🌌 Wrench Radius:  {wrench_space_score:.3f} (weight: {weights['wrench_space_radius']}) 🆕")
        print(f"         Translational:    {translational_robustness_score:.3f} (weight: {weights['translational_robustness']})")
        print(f"         Param Spread:     {parameter_spread_score:.3f} (weight: {weights['parameter_spread']})")
        print(f"         Torque Div:       {torque_diversity_score:.3f} (weight: {weights['torque_diversity']})")
        print(f"         Condition:        {condition_score:.3f} (weight: {weights['condition_number']})")
        print(f"         Spatial Dist:     {spatial_distribution_score:.3f} (weight: {weights['spatial_distribution']})")
    
    return {
        'overall_score': overall_score,
        'weighting_scheme': weighting_scheme,
        'individual_metrics': {
            'wrench_space_radius': wrench_space_score,  # 🆕 New metric
            'wrench_space_radius_raw': wrench_space_radius,  # 🆕 Raw value
            'translational_robustness': translational_robustness_score,
            'parameter_spread': parameter_spread_score,
            'torque_diversity': torque_diversity_score,
            'condition_number': condition_score,
            'spatial_distribution': spatial_distribution_score,
            'angular_gaps': angular_gaps,
            'singular_values': singular_values,
            'grasp_matrix': G,
            'projection_min_distances': min_distances  # 🆕 Individual projection distances
        }
    }



def _rank_solutions_by_quality(solutions, verbose=False):
    """
    Rank solutions by grasp quality metrics.
    
    Multi-criteria ranking:
    1. Primary: Overall quality score
    2. Tie-breaker 1: Translational robustness
    3. Tie-breaker 2: Condition number score
    
    Args:
        solutions: List of solution dicts (each with 'grasp_quality' key)
        verbose: If True, print ranking details
    
    Returns:
        list: Solutions sorted by quality (best first)
    """
    if verbose:
        print(f"\n📊 Ranking {len(solutions)} solutions by quality...")
    
    # Sort by overall score (descending - higher is better)
    ranked = sorted(
        solutions,
        key=lambda sol: (
            sol['grasp_quality']['overall_score'],
            sol['grasp_quality']['individual_metrics']['translational_robustness'],
            sol['grasp_quality']['individual_metrics']['condition_number']
        ),
        reverse=True
    )
    
    if verbose and len(ranked) > 0:
        print(f"\n   Top 5 solutions:")
        for i, sol in enumerate(ranked[:5]):
            score = sol['grasp_quality']['overall_score']
            trans = sol['grasp_quality']['individual_metrics']['translational_robustness']
            cond = sol['grasp_quality']['individual_metrics']['condition_number']
            
            print(f"      #{i+1}: Score={score:.4f} (Trans={trans:.3f}, Cond={cond:.3f})")
            print(f"            Edges: {sol['edge_indices']}")
            print(f"            Points: {sol['point_descriptions']}")
    
    return ranked

def _visualize_magnum_four_solution(solution, obj, max_inscribed_circles, edge_characterizer=None):
    """
    Visualize the Magnum Four solution with inscribed circles and grasp quality indicators.
    Two-panel layout: main visualization (top) + detailed metrics table (bottom).
    
    Args:
        solution: Solution dict containing contacts, closure_result, grasp_quality
        obj: GenericObject instance
        max_inscribed_circles: List of circle dicts from inscribed circle finder
        edge_characterizer: Optional EdgeCharacterizer (for edge labeling)
    """
    fig, (ax_main, ax_metrics) = plt.subplots(2, 1, figsize=(14, 16), 
                                               gridspec_kw={'height_ratios': [2, 1]})
    
    # =========================================================================
    # TOP PANEL: MAIN VISUALIZATION
    # =========================================================================
    
    # Draw object
    obj.visualize(ax=ax_main, alpha=0.3, facecolor='lightcyan', show_frame=True)
    
    # Extract solution data
    contacts = solution['contacts']
    edge_indices = solution['edge_indices']
    point_descriptions = solution['point_descriptions']
    grasp_quality = solution['grasp_quality']
    
    # Color map for edges
    edge_colors = ['red', 'blue', 'green', 'purple']
    
    # Force vector scale
    force_scale = 0.08
    
    # Draw contact points and force vectors
    for i, (contact, edge_idx) in enumerate(zip(contacts, edge_indices)):
        pos = contact.position
        normal = contact.normal_inward
        
        # Choose color based on edge
        color = edge_colors[edge_idx % len(edge_colors)]
        
        # Draw contact point (numbered marker)
        ax_main.plot(pos[0], pos[1], 'o', 
                     color=color, 
                     markersize=12,
                     markeredgecolor='black',
                     markeredgewidth=2,
                     alpha=0.9,
                     label=f'Edge {edge_idx}' if i == edge_indices.index(edge_idx) else '')
        
        # Add number label
        ax_main.text(pos[0], pos[1], str(i+1),
                     fontsize=10, fontweight='bold',
                     ha='center', va='center',
                     color='white')
        
        # Draw force vector
        force_vec = normal * force_scale
        ax_main.arrow(pos[0], pos[1], 
                      force_vec[0], force_vec[1],
                      head_width=0.02, 
                      head_length=0.02,
                      linewidth=2.5,
                      fc=color, 
                      ec='black',
                      alpha=0.8)
    
    # Draw maximum inscribed circles
    for i, circle in enumerate(max_inscribed_circles):
        center = circle['center']
        radius = circle['radius']
        
        circle_patch = MPLCircle(
            center, radius,
            fill=False,
            edgecolor='orange',
            linewidth=2,
            linestyle='--',
            alpha=0.7,
            label='Max Inscribed Circle' if i == 0 else ''
        )
        ax_main.add_patch(circle_patch)
        
        ax_main.plot(center[0], center[1], 
                     '*', 
                     color='orange',
                     markersize=12,
                     markeredgecolor='black',
                     markeredgewidth=1)
    
    # Add centroid
    centroid = obj.get_centroid()
    ax_main.plot(centroid.x, centroid.y,
                 'x',
                 color='black',
                 markersize=10,
                 markeredgewidth=2,
                 label='Centroid')
    
    # Title
    closure_result = solution['closure_result']
    force_closure = closure_result.get('force_closure', False)
    torque_closure = closure_result.get('torque_closure', False)
    quality_score = grasp_quality['overall_score']
    
    title_text = "🏆 THE MAGNUM FOUR - Contact Configuration\n"
    title_text += f"Force Closure: {'✓' if force_closure else '✗'} | "
    title_text += f"Torque Closure: {'✓' if torque_closure else '✗'} | "
    title_text += f"Quality Score: {quality_score:.3f}"
    
    ax_main.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
    ax_main.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal')
    
    # =========================================================================
    # BOTTOM PANEL: DETAILED METRICS TABLE
    # =========================================================================
    
    ax_metrics.axis('off')
    
    # Build detailed text content
    metrics = grasp_quality['individual_metrics']
    t_params = solution['t_parameters']
    grasp_matrix = metrics['grasp_matrix']
    
    # Create formatted text
    text_content = []
    
    # Section 1: Contact Details
    text_content.append("=" * 80)
    text_content.append("📍 CONTACT POINT DETAILS")
    text_content.append("=" * 80)
    for i in range(len(contacts)):
        text_content.append(
            f"Contact {i+1}: t={t_params[i]:.6f} (Edge {edge_indices[i]}, {point_descriptions[i]})"
        )
        pos = contacts[i].position
        normal = contacts[i].normal_inward
        text_content.append(
            f"           Position: ({pos[0]:7.4f}, {pos[1]:7.4f}), Normal: ({normal[0]:7.4f}, {normal[1]:7.4f})"
        )
    
    # Section 2: Grasp Matrix
    text_content.append("\n" + "=" * 80)
    text_content.append("📊 GRASP MATRIX G (3×4)")
    text_content.append("=" * 80)
    text_content.append(f"Fx: [{grasp_matrix[0,0]:7.4f}, {grasp_matrix[0,1]:7.4f}, {grasp_matrix[0,2]:7.4f}, {grasp_matrix[0,3]:7.4f}]")
    text_content.append(f"Fy: [{grasp_matrix[1,0]:7.4f}, {grasp_matrix[1,1]:7.4f}, {grasp_matrix[1,2]:7.4f}, {grasp_matrix[1,3]:7.4f}]")
    text_content.append(f"τ : [{grasp_matrix[2,0]:7.4f}, {grasp_matrix[2,1]:7.4f}, {grasp_matrix[2,2]:7.4f}, {grasp_matrix[2,3]:7.4f}]")
    
    # Section 3: Quality Metrics
    text_content.append("\n" + "=" * 80)
    text_content.append("📈 QUALITY METRICS")
    text_content.append("=" * 80)
    text_content.append(f"Overall Score:           {quality_score:.4f} (Weighting: {grasp_quality['weighting_scheme']})")
    text_content.append(f"")
    text_content.append(f"Wrench Space Radius:     {metrics['wrench_space_radius']:.4f} (raw: {metrics['wrench_space_radius_raw']:.6f})")
    text_content.append(f"Translational Robustness: {metrics['translational_robustness']:.4f}")
    text_content.append(f"Parameter Spread:        {metrics['parameter_spread']:.4f}")
    text_content.append(f"Torque Diversity:        {metrics['torque_diversity']:.4f}")
    text_content.append(f"Condition Number:        {metrics['condition_number']:.4f}")
    text_content.append(f"Spatial Distribution:    {metrics['spatial_distribution']:.4f}")
    text_content.append(f"")
    text_content.append(f"Iteration Found:         {solution.get('iteration_found', 'N/A')}")
    
    # Section 4: Closure Analysis Summary
    text_content.append("\n" + "=" * 80)
    text_content.append("🔬 CLOSURE ANALYSIS SUMMARY")
    text_content.append("=" * 80)
    
    if 'stage1_null_space' in closure_result:
        null_result = closure_result['stage1_null_space']
        text_content.append(f"Stage 1 (Null Space):    Kernel Dim={null_result['kernel_dimension']}, Rank={null_result['rank']}")
    
    if 'stage2_lp_test' in closure_result:
        lp_result = closure_result['stage2_lp_test']
        lp_pass = lp_result['all_wrenches_pass']
        text_content.append(f"Stage 2 - LP Test:       {'✓ PASS' if lp_pass else '✗ FAIL'} (All basic wrenches reachable: {lp_pass})")
    
    if 'stage2_geometric_test' in closure_result and closure_result['stage2_geometric_test'] is not None:
        geom_result = closure_result['stage2_geometric_test']
        geom_pass = geom_result['all_projections_pass']
        text_content.append(f"Stage 2 - Geometric:     {'✓ PASS' if geom_pass else '✗ FAIL'} (All projections pass: {geom_pass})")
        
        for proj_name, proj_result in geom_result['projection_results'].items():
            contains = proj_result.get('contains_circle', False)
            min_dist = proj_result.get('min_distance', 0.0)
            text_content.append(f"   {proj_name:15s} {'✓' if contains else '✗'} min_dist={min_dist:.4f}")
    
    if 'stage2_option3_test' in closure_result:
        opt3_result = closure_result['stage2_option3_test']
        opt3_pass = opt3_result.get('satisfied', False)
        text_content.append(f"Stage 2 - Intersection:  {'✓ PASS' if opt3_pass else '✗ FAIL'} (Geometric intersection: {opt3_pass})")
        
        if opt3_pass:
            partition = opt3_result['valid_partition']
            text_content.append(f"   Valid partition: {partition}")
    
    # Display all text in metrics panel
    full_text = '\n'.join(text_content)
    ax_metrics.text(0.05, 0.95, full_text,
                    transform=ax_metrics.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='left',
                    family='monospace',
                    bbox=dict(boxstyle='round,pad=0.8',
                             facecolor='white',
                             edgecolor='black',
                             linewidth=2,
                             alpha=0.95))
    
    plt.tight_layout()
    plt.show()
    
    # =========================================================================
    # CONSOLE OUTPUT: Full detailed analysis
    # =========================================================================
    
    print("\n" + "="*80)
    print("🔬 DETAILED CLOSURE ANALYSIS (Console Output)")
    print("="*80)
    
    # Stage 1: Null space
    if 'stage1_null_space' in closure_result:
        null_space_result = closure_result['stage1_null_space']
        print(f"\n📐 Stage 1: Null Space Test")
        print(f"   Kernel dimension: {null_space_result['kernel_dimension']}")
        print(f"   Rank: {null_space_result['rank']}")
        print(f"   Has positive combination: {null_space_result['has_positive_combination']}")
    
    # Stage 2 methods
    if 'stage2_lp_test' in closure_result:
        lp_result = closure_result['stage2_lp_test']
        print(f"\n🧪 Stage 2 - Option 1: LP Test")
        print(f"   All basic wrenches reachable: {lp_result['all_wrenches_pass']}")
        
        for test in lp_result['test_results']:
            status = '✓' if test['success'] else '✗'
            print(f"   {status} Wrench {test['wrench']}")
    
    if 'stage2_geometric_test' in closure_result and closure_result['stage2_geometric_test'] is not None:
        geom_result = closure_result['stage2_geometric_test']
        print(f"\n🔷 Stage 2 - Option 2: Geometric Test")
        print(f"   All projections pass: {geom_result['all_projections_pass']}")
        
        for proj_name, proj_result in geom_result['projection_results'].items():
            status = '✓' if proj_result.get('contains_circle', False) else '✗'
            min_dist = proj_result.get('min_distance', 0.0)
            print(f"   {status} {proj_name}: min_dist={min_dist:.4f}")
    
    if 'stage2_option3_test' in closure_result:
        opt3_result = closure_result['stage2_option3_test']
        print(f"\n🔷 Stage 2 - Option 3: Geometric Intersection Test")
        print(f"   Force closure satisfied: {opt3_result.get('satisfied', False)}")
        
        if opt3_result.get('satisfied', False):
            partition = opt3_result['valid_partition']
            print(f"   Valid partition: {partition}")
            alphas = opt3_result['alphas']
            print(f"   Alpha values: {alphas}")
    
    print("\n" + "="*80)

# %%
def find_the_magnum_four_v3(obj, verbose=True, visualize=True, force_magnitude=1.0,
                            weighting_scheme='balanced', torque_method=3,
                            preprocess_result=None, robot_radius: Optional[float] = None):
    """
    🆕 OPTIMIZED v3: Find four contact points with preprocessing and smart edge filtering.
    
    KEY OPTIMIZATIONS:
    1. **Preprocessing**: Use preprocess_object_force_closure() to cache valid edge combos
    2. **Smart Edge Filtering**: Only test 4-edge combinations where:
       - At most 1 edge has 2 contacts (ensures ≥3 distinct edges)
       - Valid for force closure (O(1) lookup via preprocessing)
    3. **Early Exit**: Skip invalid edge combos before point sampling
    4. **Fast Torque Check**: Force closure already validated, only check torque
    
    Args:
        obj: GenericObject instance
        verbose: If True, print detailed search information
        visualize: If True, visualize the solution and wrench space
        force_magnitude: Magnitude of normal forces for analysis
        weighting_scheme: 'balanced', 'focus_translational', or 'focus_rotational'
        torque_method: 1 (LP), 2 (convex hull), or 3 (geometric intersection)
        preprocess_result: Optional preprocessing result (to avoid redundant computation)
        robot_radius: Optional radius of the circular robot. If provided, strategic
            contact samples will be filtered using C-space reachability so that
            only physically reachable boundary points are considered.
    
    Returns:
        dict: Solution containing contact points, closure metrics, timing, and statistics
    """
    import time
    import itertools
    
    print("\n" + "="*80)
    print("🔍 SEARCHING FOR THE MAGNUM FOUR (v3 - OPTIMIZED)")
    print("="*80)
    
    timing = {}
    
    # =========================================================================
    # STEP 0: PREPROCESSING
    # =========================================================================
    if preprocess_result is None:
        print(f"\n⏱️  Step 0: Preprocessing object for force closure...")
        start_time = time.time()
        
        preprocess_result = preprocess_object_force_closure(
            obj, 
            force_magnitude=force_magnitude, 
            verbose=verbose,
            include_4edge=True  # Need 4-edge combos
        )
        
        timing['preprocessing'] = time.time() - start_time
        print(f"   ✅ Preprocessing completed in {timing['preprocessing']:.4f} seconds")
    else:
        if verbose:
            print(f"\n⏱️  Step 0: Using provided preprocessing result")
        timing['preprocessing'] = 0.0  # Already done
    
    edge_characterizer = preprocess_result['edge_characterizer']
    num_edges = preprocess_result['num_edges']
    obj_min_edge_length = _get_object_min_edge_length(obj)
    
    if num_edges < 3:
        print("❌ Need at least 3 edges for meaningful contact configuration")
        return {'success': False, 'reason': 'insufficient_edges', 'timing': timing}
    
    # =========================================================================
    # STEP 1: FIND MAXIMUM INSCRIBED CIRCLE(S)
    # =========================================================================
    print(f"\n🔵 Step 1: Finding maximum inscribed circle(s)...")
    start_time = time.time()
    
    max_inscribed_circles = _find_max_inscribed_circles(
        obj, 
        edge_characterizer,
        method='auto'
    )
    
    if len(max_inscribed_circles) > 1:
        max_inscribed_circles = _rank_and_filter_circles(
            max_inscribed_circles, 
            obj,
            max_circles=4
        )
    
    timing['inscribed_circles'] = time.time() - start_time
    
    if verbose:
        for i, circle in enumerate(max_inscribed_circles):
            print(f"   Circle {i}: center=({circle['center'][0]:.3f}, {circle['center'][1]:.3f}), "
                  f"radius={circle['radius']:.3f}, tangency_points={circle['num_tangents']}")

    # Optional: compute reachable boundary intervals in t-space for a circular robot
    reachable_intervals = None
    if robot_radius is not None:
        try:
            print(f"\n🔍 Step 1b: Computing reachable boundary intervals for robot radius={robot_radius:.4f}...")
            reachable_intervals = get_reachable_contact_intervals(
                obj.geometry,
                robot_radius=robot_radius,
                n_samples=2048,
            )
            if verbose:
                if len(reachable_intervals) == 0:
                    print("   ⚠️ No reachable boundary intervals found (all boundary treated as unreachable)")
                else:
                    print(f"   Reachable intervals in t-space (0-1):")
                    for idx, (t0, t1) in enumerate(reachable_intervals):
                        print(f"     [{idx}] t ∈ [{t0:.4f}, {t1:.4f}]")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to compute reachable intervals: {e}")
            reachable_intervals = None
    
    # =========================================================================
    # STEP 2: GENERATE STRATEGIC CONTACT POINT SAMPLES ON EACH EDGE
    # =========================================================================
    print(f"\n📍 Step 2: Sampling strategic contact points on each edge...")
    start_time = time.time()
    
    edge_sample_points = generate_strategic_contact_samples(
        edge_characterizer,
        max_inscribed_circles,
        verbose=verbose
    )

    # Optional: filter out samples that lie on unreachable parts of the boundary
    if robot_radius is not None and reachable_intervals:
        def _t_is_reachable(t_val: float) -> bool:
            for t0, t1 in reachable_intervals:
                if t0 <= t_val <= t1:
                    return True
            return False

        filtered_edge_sample_points = {}
        removed_total = 0
        for edge_idx, samples in edge_sample_points.items():
            filtered = []
            for t_val, desc in samples:
                if _t_is_reachable(t_val):
                    filtered.append((t_val, desc))
                else:
                    removed_total += 1
            filtered_edge_sample_points[edge_idx] = filtered
        edge_sample_points = filtered_edge_sample_points
        if verbose:
            print(f"   ✅ Reachability filter removed {removed_total} unreachable samples")
    
    total_samples = sum(len(samples) for samples in edge_sample_points.values())
    timing['sampling'] = time.time() - start_time
    print(f"   Total strategic samples: {total_samples}")
    
    # =========================================================================
    # STEP 3: 🆕 SMART EDGE COMBINATION FILTERING
    # =========================================================================
    print(f"\n🎯 Step 3: Smart filtering of 4-edge combinations...")
    start_time = time.time()
    
    valid_edge_combos = []
    total_4edge_combos = 0
    
    # Case 1: Generate all 4-distinct-edge combinations
    for edge_combo in itertools.combinations(range(num_edges), 4):
        total_4edge_combos += 1
        
        # Filter 1: Check force closure via preprocessing (O(1) lookup)
        if not is_valid_edge_combination(edge_combo, preprocess_result):
            continue
        
        # Filter 2 passed: Force closure is valid
        valid_edge_combos.append(edge_combo)

    # Case 2: One edge appears twice, two others distinct
    # Choose 1 edge to duplicate, then choose 2 other distinct edges
    for dup_edge in range(num_edges):
        # Choose 2 other distinct edges (excluding dup_edge)
        other_edges = [e for e in range(num_edges) if e != dup_edge]
        
        for edge_pair in itertools.combinations(other_edges, 2):
            e2, e3 = edge_pair
            
            # Create combination: [dup_edge, dup_edge, e2, e3]
            edge_combo_list = [dup_edge, dup_edge, e2, e3]
            
            total_4edge_combos += 1
            
            # Filter: Check force closure via preprocessing (O(1) lookup)
            if is_valid_edge_combination([dup_edge, e2, e3], preprocess_result):
                valid_edge_combos.append(edge_combo_list)
    
    timing['edge_filtering'] = time.time() - start_time
    
    print(f"   Total 4-edge combinations: {total_4edge_combos}")
    print(f"   Valid combinations (force closure): {len(valid_edge_combos)}")
    print(f"   Filtering completed in {timing['edge_filtering']:.4f} seconds")
    
    if len(valid_edge_combos) == 0:
        print(f"\n❌ No valid 4-edge combinations found for force closure")
        return {
            'success': False,
            'reason': 'no_valid_edge_combos',
            'timing': timing,
            'statistics': {
                'total_4edge_combos': total_4edge_combos,
                'valid_4edge_combos': 0
            }
        }
    
    # =========================================================================
    # STEP 4: 🆕 OPTIMIZED EXHAUSTIVE SEARCH (only on valid edge combos)
    # =========================================================================
    print(f"\n🔎 Step 4: Exhaustive search on {len(valid_edge_combos)} valid edge combinations...")
    start_time = time.time()
    
    # Estimate total point combinations for valid edges
    total_point_combos = 0
    for edge_combo in valid_edge_combos:
        e1, e2, e3, e4 = edge_combo
        total_point_combos += (
            len(edge_sample_points[e1]) * 
            len(edge_sample_points[e2]) * 
            len(edge_sample_points[e3]) * 
            len(edge_sample_points[e4])
        )
    
    print(f"   Point combinations to test: {total_point_combos:,}")
    
    valid_solutions = []
    iteration_count = 0
    
    pruned_count = {
        'duplicate_points': 0,
        'parallel_normals': 0,
        'quick_force_closure_fail': 0,
        'torque_closure_fail': 0,
        'insufficient_robot_spacing': 0
    }
    
    # Compute epsilon for spatial distinctness check
    epsilon = compute_epsilon(max_inscribed_circles, edge_characterizer)
    
    # 🆕 Only iterate through VALID edge combinations
    for edge_combo in valid_edge_combos:
        e1, e2, e3, e4 = edge_combo
        edge_indices = [e1, e2, e3, e4]
        
        # Generate all point combinations on these 4 edges
        for t1_info in edge_sample_points[e1]:
            for t2_info in edge_sample_points[e2]:
                for t3_info in edge_sample_points[e3]:
                    for t4_info in edge_sample_points[e4]:
                        
                        iteration_count += 1
                        
                        if verbose and iteration_count % 5000 == 0:
                            print(f"   Tested {iteration_count:,} combinations...")
                        
                        # Extract t parameters
                        t_params = [t1_info[0], t2_info[0], t3_info[0], t4_info[0]]
                        
                        # Build contact points
                        contacts = _build_contact_points(
                            edge_indices,
                            t_params,
                            edge_characterizer
                        )
                        
                        # =====================================================
                        # EARLY PRUNING CHECKS
                        # =====================================================
                        
                        # Prune 1: Check all points are spatially distinct
                        if not _check_points_distinct(contacts, tolerance=epsilon):
                            pruned_count['duplicate_points'] += 1
                            continue
                        
                        # Prune 2: Check normals not all parallel
                        if not _check_normals_not_parallel(contacts):
                            pruned_count['parallel_normals'] += 1
                            continue
                        
                        # Prune 3: Quick force closure check
                        if not _quick_force_closure_check(contacts):
                            pruned_count['quick_force_closure_fail'] += 1
                            continue
                        
                        # Prune 4: Check robot center spacing (if robot_radius provided)
                        if robot_radius is not None:
                            if not _check_enough_space_for_robots(
                                contacts, robot_radius, min_edge_length=obj_min_edge_length
                            ):
                                pruned_count['insufficient_robot_spacing'] += 1
                                continue
                        
                        # =====================================================
                        # 🆕 OPTIMIZED CLOSURE CHECK (force closure already validated!)
                        # =====================================================
                        
                        closure_result = _check_force_and_torque_closure(
                            contacts,
                            edge_characterizer,
                            preprocess_result,  # 🆕 Pass preprocessing data
                            torque_method=torque_method,
                            verbose=False
                        )
                        
                        # Force closure should always pass (already validated by edge combo)
                        if not closure_result['force_closure']:
                            # This shouldn't happen, but log it
                            if verbose:
                                print(f"   ⚠️ Warning: Force closure failed despite valid edge combo")
                            continue
                        
                        if not closure_result['torque_closure']:
                            pruned_count['torque_closure_fail'] += 1
                            continue
                        
                        # =====================================================
                        # VALID SOLUTION FOUND!
                        # =====================================================
                        
                        solution = {
                            'contacts': contacts,
                            'edge_indices': edge_indices,
                            't_parameters': t_params,
                            'point_descriptions': [t1_info[1], t2_info[1], t3_info[1], t4_info[1]],
                            'closure_result': closure_result,
                            'iteration_found': iteration_count
                        }
                        
                        # Compute grasp quality metrics
                        solution['grasp_quality'] = _compute_grasp_quality_metrics(
                            contacts,
                            edge_characterizer,
                            weighting_scheme=weighting_scheme,
                            verbose=False
                        )
                        
                        valid_solutions.append(solution)
                        
                        if verbose and len(valid_solutions) % 10 == 0:
                            print(f"   Found {len(valid_solutions)} valid solutions so far...")
    
    timing['exhaustive_search'] = time.time() - start_time
    timing['total'] = sum(timing.values())
    
    # =========================================================================
    # STEP 5: RANK SOLUTIONS AND SELECT BEST
    # =========================================================================
    print(f"\n📊 Step 5: Ranking solutions...")
    print(f"   Total iterations: {iteration_count:,}")
    print(f"   Valid solutions found: {len(valid_solutions)}")
    print(f"\n   Pruning statistics:")
    for reason, count in pruned_count.items():
        pct = count / total_point_combos * 100 if total_point_combos > 0 else 0
        print(f"      {reason}: {count:,} ({pct:.1f}%)")
    
    print(f"\n⏱️  Timing breakdown:")
    for stage, duration in timing.items():
        if stage != 'total':
            pct = duration / timing['total'] * 100
            print(f"      {stage}: {duration:.4f}s ({pct:.1f}%)")
    print(f"      TOTAL: {timing['total']:.4f}s")
    
    if len(valid_solutions) == 0:
        print(f"\n❌ No valid Magnum Four configuration found")
        return {
            'success': False,
            'reason': 'no_solution_found',
            'iterations_tested': iteration_count,
            'pruning_stats': pruned_count,
            'timing': timing,
            'max_inscribed_circles': max_inscribed_circles,
            'preprocess_result': preprocess_result
        }
    
    # Rank solutions by grasp quality
    start_time = time.time()
    ranked_solutions = _rank_solutions_by_quality(valid_solutions, verbose=verbose)
    timing['ranking'] = time.time() - start_time
    timing['total'] += timing['ranking']
    
    best_solution = ranked_solutions[0]
    
    print(f"\n🏆 Best solution selected:")
    print(f"   Edges: {best_solution['edge_indices']}")
    print(f"   Points: {best_solution['point_descriptions']}")
    print(f"   Grasp quality score: {best_solution['grasp_quality']['overall_score']:.4f}")
    print(f"   Found at iteration: {best_solution['iteration_found']}")
    
    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    if visualize:
        print(f"\n🎨 Visualizing solution...")
        _visualize_magnum_four_solution(
            best_solution, 
            obj, 
            max_inscribed_circles,
            edge_characterizer=edge_characterizer
        )
        
        print(f"\n🌌 Visualizing wrench space...")
        contact_points = best_solution['contacts']
        wrench_visualizer = WrenchSpaceVisualizer()
        
        wrench_data = wrench_visualizer.calculate_wrench_space(
            contact_points,
            force_ranges=[(0.0, 5.0)] * len(contact_points),
            sampling_density=3,
            enable_tangent_forces=False
        )
        
        fig = wrench_visualizer.visualize_wrench_space_with_limit_surface(
            wrench_data,
            contact_points,
            obj,
            title=f"Wrench Space - Best Solution (v3 Optimized)\n"
                  f"Quality Score: {best_solution['grasp_quality']['overall_score']:.4f}"
        )
        fig.show()
    
    return {
        'success': True,
        'best_solution': best_solution,
        'all_solutions': ranked_solutions,
        'num_solutions_found': len(valid_solutions),
        'iterations_tested': iteration_count,
        'pruning_stats': pruned_count,
        'timing': timing,
        'max_inscribed_circles': max_inscribed_circles,
        'preprocess_result': preprocess_result,
        'statistics': {
            'total_4edge_combos': total_4edge_combos,
            'valid_4edge_combos': len(valid_edge_combos),
            'total_point_combos': total_point_combos,
            'valid_solutions': len(valid_solutions)
        }
    }



# %%
if __name__ == "__main__":
    standard_objects = create_standard_objects()
    # obj = standard_objects['l_shape']
    # obj = standard_objects['boot']
    obj = standard_objects['fat_triangle']
    # obj = standard_objects['u_shape']
    # obj = standard_objects['star']

    # find_magnum_result = find_the_magnum_four_v3(obj, verbose=True, visualize=True, weighting_scheme='balanced')
    # find_magnum_result = find_the_magnum_four_v3(obj, verbose=True, visualize=True, weighting_scheme='focus_translational')
    find_magnum_result = find_the_magnum_four_v3(obj, verbose=True, visualize=True, weighting_scheme='focus_rotational')

    


# =============================================================================
# PLACEHOLDER UTILITIES FOR STOCHASTIC SEARCH (LHS / CEM)
# =============================================================================

def _estimate_max_lhs_batches(total_combinations: int, batch_size: int) -> int:
    """
    Placeholder: Estimate the maximum number of LHS batches to use.

    Design choice (from high-level plan):
    - Total number of LHS batches is capped at ≈ 30% of the total number of
      exhaustive combinations (rounded up).
    - This function only encodes that policy; the actual LHS sampler will be
      implemented in a later phase.
    """
    if total_combinations <= 0 or batch_size <= 0:
        return 0

    # 30% of total combinations, then divided by batch size
    max_samples = int(np.ceil(0.3 * float(total_combinations)))
    max_batches = int(np.ceil(max_samples / float(batch_size)))
    return max_batches


def _sample_contacts_lhs_placeholder(
    edge_characterizer,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
):
    """
    Placeholder for Latin Hypercube Sampling (LHS) over contact parameters.

    Intended behavior (to be implemented in Phase 2):
    - Uniformly sample contact configurations across:
        * Edge indices
        * Contact parameters t ∈ [0, 1] along each edge
    - Return a list of candidate configurations:
        [
            {
                'edge_indices': [...],
                't_parameters': [...],
            },
            ...
        ]

    Current behavior:
    - Returns an empty list; used only as a logical placeholder so that
      algorithm structure and interfaces can be nailed down first.
    """
    _ = edge_characterizer  # Unused for now
    _ = num_samples
    _ = rng
    return []


def _cem_update_distribution_placeholder(
    elite_solutions,
    current_distribution,
):
    """
    Placeholder for Cross-Entropy Method (CEM) update step.

    Intended behavior (to be implemented in Phase 3):
    - Take the best ("elite") contact configurations from the previous batch
      and update a parametric sampling distribution (e.g., over edge indices
      and contact parameters) to focus subsequent samples near promising areas.

    Current behavior:
    - Returns `current_distribution` unchanged.
    """
    _ = elite_solutions
    return current_distribution


def find_the_magnum_stochastic(
    obj,
    threshold: float = 1.0,
    timeout: float = 10.0,
    n_ellipse_samples: int = 72,
    force_range_scalar: float = 2.0,
    robot_radius: Optional[float] = None,
    used_tangent_as_fallback: bool = False,
    tangent_required: bool = False,
    theory_mode: bool = False,
    verbose: bool = True,
):
    """
    Phase 2: Latin Square-based stochastic search with early termination.

    High-level behavior:
        - Generate strategic contact point samples on all edges
        - Create a Latin square: 4 columns (robots), n_strategic_points rows
          Each column is a permutation of [0, 1, ..., n_strategic_points-1]
        - For each row (combination):
              1) Map 4 indices to strategic points (edge_idx, t_param)
              2) Build ContactPoint objects
              3) Apply pruning checks (engineering mode only: robot spacing,
                 parallel normals, quick force closure; always: distinct points)
              4) Run `check_wrench_space_sufficiency(...)`
          If the test passes, RETURN IMMEDIATELY (anytime algorithm).
        - If no configuration passes before the first-pass time budget,
          and `used_tangent_as_fallback` is True (and `tangent_required` is False),
          re-run the search with `enable_tangent_forces=True` and quick FC pruning
          disabled; if that finds a config, return it with `used_tangent_fallback=True`.
          When tangent fallback is enabled without `tangent_required`, the total
          `timeout` is split evenly between the normal-force pass and the tangent-force pass.
        - If `tangent_required` is True, skip the normal-only pass and use the full
          `timeout` on a tangent-force search (quick FC prune off).
        - Otherwise return `success=False`.

    Design:
        - "Good enough" is defined purely by LS containment:
              GWS(contacts) ⊇ threshold × LS
        - Uses strategic sampling (near corners, midpoints, tangency points, etc.)
        - Latin square ensures uniform coverage without clustering

    Args:
        obj:          `GenericObject` instance.
        threshold:    LS coverage factor (1.0 = 100% LS; >1.0 = stricter).
        timeout:      Search time budget in seconds (default 10.0), measured after
            setup (edge characterization, inscribed circles, strategic sampling).
            Batches continue until this budget is exhausted.
        n_ellipse_samples: Boundary samples per LS projection ellipse.
        force_range_scalar: Multiplier for maximum force range.
            Force range = [0, force_range_scalar × static_f_max]
            where static_f_max = static_friction × (mass × 9.81)
            Default 2.0 means robots can exert up to 2× static friction limit.
        robot_radius: Optional robot radius for spacing checks (engineering mode).
        used_tangent_as_fallback: If True, on first-pass failure re-run the search
            with `enable_tangent_forces=True` and quick FC pruning disabled.
            Total `timeout` is split evenly across both passes (ignored when
            `tangent_required` is True).
        tangent_required: If True, skip the normal-only pass and search with tangent
            forces for the full `timeout` (e.g. D/σ₃ screening already mandates friction).
        theory_mode: If True, validate Latin-square search with minimal pruning:
            keeps distinct-point check and `check_wrench_space_sufficiency` (use a
            small non-zero `threshold`), but skips robot-spacing and quick
            force-closure heuristics. `robot_radius` is ignored for pruning.
        verbose:      If True, prints batch-level progress and summary.

    Returns:
        dict with keys:
            'success'          : bool
            'found_by'         : 'stochastic' (if success) or None
            'contacts'         : list[ContactPoint] (if success)
            'threshold'        : float used
            'batches_tested'   : int
            'configs_tested'   : int
            'pruned_count'     : dict of pruning statistics
            'sufficiency_result': result dict from last sufficiency check
            'used_tangent_fallback': bool (if success via tangent-force fallback)
            'theory_mode'        : bool (whether theory validation mode was used)
    """
    import time

    apply_robot_spacing_check = (robot_radius is not None) and (not theory_mode)
    apply_quick_prune = not theory_mode

    if verbose:
        print("\n" + "=" * 80)
        mode_label = "THEORY" if theory_mode else "ENGINEERING"
        print(f"🎲 STOCHASTIC MAGNUM SEARCH (Phase 2 - Latin Square + Early Termination) [{mode_label}]")
        print("=" * 80)

    # ---------------------------------------------------------------------
    # STEP 0: Setup – edge characterization and strategic sampling
    # (not counted against search timeout)
    # ---------------------------------------------------------------------
    setup_start_time = time.time()

    edge_characterizer = EdgeCharacterizer(obj, force_magnitude=1.0)
    num_edges = len(edge_characterizer.edges)

    if num_edges == 0:
        if verbose:
            print("❌ Object has no identifiable edges – cannot sample contacts.")
        return {
            'success': False,
            'found_by': None,
            'reason': 'no_edges',
            'threshold': threshold,
            'theory_mode': theory_mode,
            'batches_tested': 0,
            'configs_tested': 0,
            'pruned_count': {},
            'sufficiency_result': None,
        }

    # Find max inscribed circles for strategic sampling
    max_inscribed_circles = _find_max_inscribed_circles(
        obj, edge_characterizer, method='auto', samples_per_edge=50, tolerance=1e-6
    )
    
    # Generate strategic contact samples
    edge_sample_points = generate_strategic_contact_samples(
        edge_characterizer, max_inscribed_circles, verbose=False
    )
    
    # Flatten strategic samples into a single list: [(edge_idx, t_param), ...]
    strategic_points = []
    for edge_idx, samples in edge_sample_points.items():
        for t_param, description in samples:
            strategic_points.append((edge_idx, t_param))
    
    n_strategic_points = len(strategic_points)
    
    if n_strategic_points < 4:
        if verbose:
            print(f"❌ Not enough strategic points ({n_strategic_points}) – need at least 4.")
        return {
            'success': False,
            'found_by': None,
            'reason': 'insufficient_strategic_points',
            'threshold': threshold,
            'theory_mode': theory_mode,
            'batches_tested': 0,
            'configs_tested': 0,
            'pruned_count': {},
            'sufficiency_result': None,
        }
    
    # Compute epsilon for pruning
    epsilon = compute_epsilon(max_inscribed_circles, edge_characterizer)
    obj_min_edge_length = _get_object_min_edge_length(obj)
    robot_spacing_buffer = (
        _compute_robot_spacing_buffer(robot_radius, obj_min_edge_length)
        if robot_radius is not None else None
    )
    
    if verbose:
        print(f"\n📐 Edges found           : {num_edges}")
        print(f"📍 Strategic points      : {n_strategic_points}")
        print(f"📏 Epsilon (pruning)     : {epsilon:.6f}")
        if tangent_required:
            print(f"⏱️  Timeout (tangent-only) : {timeout:.1f} s")
        elif used_tangent_as_fallback:
            print(f"⏱️  Timeout (total budget): {timeout:.1f} s ({timeout / 2.0:.1f} s per pass)")
        else:
            print(f"⏱️  Timeout (search budget): {timeout:.1f} s")
        print(f"🎯 Threshold (LS scale)  : {threshold:.2f}")
        if theory_mode:
            print("🧪 Theory mode          : ON (skip robot spacing + quick FC prune)")
        if tangent_required:
            print("🔄 Tangent required     : ON (skip normal-only pass)")
        elif used_tangent_as_fallback:
            print("🔄 Tangent fallback     : ON (2nd pass skips quick FC prune)")
        if apply_robot_spacing_check:
            print(f"🤖 Robot radius          : {robot_radius:.3f}")
            print(f"📏 Min edge length       : {obj_min_edge_length:.6f}")
            print(f"📏 Robot spacing buffer  : {robot_spacing_buffer:.6f}")

    setup_elapsed = time.time() - setup_start_time
    if verbose:
        print(f"⏱️  Setup time (excluded) : {setup_elapsed:.2f} s")

    pass_timeout = (
        timeout
        if tangent_required
        else (timeout / 2.0 if used_tangent_as_fallback else timeout)
    )
    rng = np.random.default_rng()

    def _create_latin_square(n: int, n_cols: int = 4):
        """
        Create a Latin square: n rows × n_cols columns.
        Each column is a random permutation of [0, 1, ..., n-1].
        
        Returns:
            (n, n_cols) array where each column is a permutation
        """
        square = np.zeros((n, n_cols), dtype=int)
        for col in range(n_cols):
            square[:, col] = rng.permutation(n)
        return square

    def _run_one_pass(
        enable_tangent_forces: bool,
        apply_robot_spacing: bool = True,
        apply_quick_prune: bool = True,
    ):
        """
        Run one full stochastic pass (batches of Latin square rows).

        Pruning:
            - Distinct contact points: always applied.
            - Robot spacing / quick FC heuristics: controlled by flags (off in theory_mode).
        Each call resets the per-pass search timer (`pass_timeout`).
        Returns (success_result_dict, None) on success, else (None, failure_stats).
        failure_stats = (batches_tested, configs_tested, pruned_count, last_sufficiency_result).
        """
        search_start_time = time.time()
        local_configs_tested = 0
        local_pruned_count = {
            'duplicate_points': 0,
            'parallel_normals': 0,
            'quick_force_closure_fail': 0,
            'insufficient_robot_spacing': 0,
        }
        local_last_sufficiency_result = None
        batch_idx = 0
        while True:
            elapsed_time = time.time() - search_start_time
            if elapsed_time >= pass_timeout:
                if verbose:
                    print(f"\n⏱️  Pass timeout reached ({pass_timeout:.1f} s)")
                break

            if verbose:
                print(f"\n--- Batch {batch_idx + 1} (search elapsed: {elapsed_time:.2f} s) ---")

            latin_square = _create_latin_square(n_strategic_points, n_cols=4)
            timed_out_mid_batch = False

            for row_idx in range(n_strategic_points):
                elapsed_time = time.time() - search_start_time
                if elapsed_time >= pass_timeout:
                    if verbose:
                        print(f"   ⏱️  Pass timeout reached during batch {batch_idx + 1}")
                    timed_out_mid_batch = True
                    break

                point_indices = latin_square[row_idx, :]
                edge_indices = []
                t_params = []
                for idx in point_indices:
                    edge_idx, t_param = strategic_points[idx]
                    edge_indices.append(edge_idx)
                    t_params.append(t_param)

                contacts = _build_contact_points(edge_indices, t_params, edge_characterizer)
                local_configs_tested += 1

                if not _check_points_distinct(contacts, tolerance=epsilon):
                    local_pruned_count['duplicate_points'] += 1
                    continue

                if apply_robot_spacing:
                    if not _check_enough_space_for_robots(
                        contacts, robot_radius, min_edge_length=obj_min_edge_length
                    ):
                        local_pruned_count['insufficient_robot_spacing'] += 1
                        continue

                if apply_quick_prune:

                    if not _check_normals_not_parallel(contacts):
                        local_pruned_count['parallel_normals'] += 1
                        continue
                    if not _quick_force_closure_check(contacts):
                        local_pruned_count['quick_force_closure_fail'] += 1
                        continue


                local_last_sufficiency_result = check_wrench_space_sufficiency(
                    contacts,
                    obj,
                    threshold=threshold,
                    n_ellipse_samples=n_ellipse_samples,
                    force_range_scalar=force_range_scalar,
                    enable_tangent_forces=enable_tangent_forces,
                    verbose=False,
                )

                if local_last_sufficiency_result.get('satisfied', False):
                    search_elapsed = time.time() - search_start_time
                    total_elapsed = time.time() - setup_start_time
                    if verbose:
                        print(f"\n✅ SUFFICIENT CONFIGURATION FOUND (stochastic)")
                        print(f"   Batch index     : {batch_idx + 1}")
                        print(f"   Config in batch : {row_idx + 1}")
                        print(f"   Total configs   : {local_configs_tested}")
                        print(f"   Search elapsed  : {search_elapsed:.3f} s")
                        print(f"   Total elapsed   : {total_elapsed:.3f} s")
                        print(f"   Pruning stats   : {local_pruned_count}")

                    return (
                        {
                            'success': True,
                            'found_by': 'stochastic',
                            'contacts': contacts,
                            'threshold': threshold,
                            'batches_tested': batch_idx + 1,
                            'configs_tested': local_configs_tested,
                            'pruned_count': local_pruned_count.copy(),
                            'sufficiency_result': local_last_sufficiency_result,
                            'theory_mode': theory_mode,
                        },
                        None,
                    )

            batch_idx += 1
            if timed_out_mid_batch:
                break

        return (
            None,
            (batch_idx, local_configs_tested, local_pruned_count, local_last_sufficiency_result),
        )

    # ---------------------------------------------------------------------
    # STEP 1: Latin square batches with early termination
    # ---------------------------------------------------------------------
    if tangent_required:
        if verbose:
            print("\n🚀 Starting tangent-force Latin square batches (D/σ₃ gate)...")
        result, fail_stats = _run_one_pass(
            enable_tangent_forces=True,
            apply_robot_spacing=apply_robot_spacing_check,
            apply_quick_prune=False,
        )
        if result is not None:
            result['used_tangent_fallback'] = True
            return result
    else:
        if verbose:
            print("\n🚀 Starting Latin square batches...")

        result, fail_stats = _run_one_pass(
            enable_tangent_forces=False,
            apply_robot_spacing=apply_robot_spacing_check,
            apply_quick_prune=apply_quick_prune,
        )
        if result is not None:
            return result

    # ---------------------------------------------------------------------
    # STEP 2: No stochastic success – optional tangent-force fallback
    # ---------------------------------------------------------------------
    if used_tangent_as_fallback and not tangent_required:
        if verbose:
            print(
                f"\n🔄 Fallback: re-running with tangent forces enabled "
                f"({pass_timeout:.1f} s, quick FC prune off)..."
            )
        result, tangent_fail_stats = _run_one_pass(
            enable_tangent_forces=True,
            apply_robot_spacing=apply_robot_spacing_check,
            apply_quick_prune=False,
        )
        if result is not None:
            result['used_tangent_fallback'] = True
            return result

        b1, c1, p1, l1 = fail_stats
        b2, c2, p2, l2 = tangent_fail_stats
        fail_stats = (
            b1 + b2,
            c1 + c2,
            {k: p1[k] + p2[k] for k in p1},
            l2 if l2 is not None else l1,
        )

    batches_tested, configs_tested, pruned_count, last_sufficiency_result = fail_stats
    total_elapsed = time.time() - setup_start_time
    if verbose:
        print(f"\n❌ No sufficient configuration found by stochastic search")
        print(f"   Batches tested    : {batches_tested}")
        print(f"   Configs tested    : {configs_tested}")
        print(f"   Search timeout    : {timeout:.1f} s")
        print(f"   Total elapsed     : {total_elapsed:.3f} s")
        print(f"   Pruning stats     : {pruned_count}")

    return {
        'success': False,
        'found_by': None,
        'contacts': None,
        'threshold': threshold,
        'theory_mode': theory_mode,
        'batches_tested': batches_tested,
        'configs_tested': configs_tested,
        'pruned_count': pruned_count,
        'sufficiency_result': last_sufficiency_result,
    }


# %%
def find_the_magnum_three_v3(obj, verbose=True, visualize=True, force_magnitude=1.0, 
                                   weighting_scheme='balanced', preprocess_result=None,
                                   robot_radius: Optional[float] = None):
    """
    🆕 OPTIMIZED v3 (NO TIMING LOGS): Fast version for fair performance comparison.
    
    Uses preprocessing with include_4edge=False for maximum speed.
    
    Args:
        obj: GenericObject instance
        verbose: If True, print basic progress (no detailed timing)
        visualize: If True, visualize the solution
        force_magnitude: Magnitude of normal forces for analysis
        weighting_scheme: 'balanced', 'focus_translational', or 'focus_rotational'
        preprocess_result: Optional preprocessing result (to avoid redundant computation)
    
    Returns:
        dict: Solution containing contact points and statistics
    """
    if verbose:
        print("\n" + "="*80)
        print("🔍 SEARCHING FOR THE MAGNUM THREE (v3 - FAST)")
        print("="*80)
    
    # STEP 0: Preprocessing (3-edge only)
    if preprocess_result is None:
        preprocess_result = preprocess_object_force_closure(
            obj, 
            force_magnitude=force_magnitude, 
            verbose=verbose,
            include_4edge=False  # 🆕 Skip 4-edge for speed!
        )
    else:
        if verbose:
            print(f"\n⏱️  Using provided preprocessing result")
    
    edge_characterizer = preprocess_result['edge_characterizer']
    num_edges = preprocess_result['num_edges']
    obj_min_edge_length = _get_object_min_edge_length(obj)
    
    if len(preprocess_result['valid_3edge_combos']) == 0:
        if verbose:
            print(f"\n❌ No valid 3-edge combination found")
        return None
    
    # STEP 1: Find maximum inscribed circles
    max_inscribed_circles = _find_max_inscribed_circles(obj, edge_characterizer, method='auto')
    
    if len(max_inscribed_circles) > 1:
        max_inscribed_circles = _rank_and_filter_circles(max_inscribed_circles, obj, max_circles=4)
    
    # Optional: compute reachable boundary intervals in t-space for a circular robot
    reachable_intervals = None
    if robot_radius is not None:
        try:
            if verbose:
                print(f"\n🔍 Magnum Three: Computing reachable boundary intervals for robot radius={robot_radius:.4f}...")
            reachable_intervals = get_reachable_contact_intervals(
                obj.geometry,
                robot_radius=robot_radius,
                n_samples=2048,
            )
            if verbose and reachable_intervals is not None:
                if len(reachable_intervals) == 0:
                    print("   ⚠️ No reachable boundary intervals found (all boundary treated as unreachable)")
                else:
                    print(f"   Reachable intervals in t-space (0-1):")
                    for idx, (t0, t1) in enumerate(reachable_intervals):
                        print(f"     [{idx}] t ∈ [{t0:.4f}, {t1:.4f}]")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to compute reachable intervals for Magnum Three: {e}")
            reachable_intervals = None
    
    # STEP 2: Generate strategic samples
    edge_sample_points = generate_strategic_contact_samples(
        edge_characterizer,
        max_inscribed_circles,
        verbose=False
    )

    # Optional: filter out samples that lie on unreachable parts of the boundary
    if robot_radius is not None and reachable_intervals:
        def _t_is_reachable_three(t_val: float) -> bool:
            for t0, t1 in reachable_intervals:
                if t0 <= t_val <= t1:
                    return True
            return False

        filtered_edge_sample_points = {}
        removed_total = 0
        for edge_idx, samples in edge_sample_points.items():
            filtered = []
            for t_val, desc in samples:
                if _t_is_reachable_three(t_val):
                    filtered.append((t_val, desc))
                else:
                    removed_total += 1
            filtered_edge_sample_points[edge_idx] = filtered
        edge_sample_points = filtered_edge_sample_points
        if verbose:
            print(f"   ✅ Magnum Three reachability filter removed {removed_total} unreachable samples")
    
    # STEP 3: Sample on valid edge combinations
    valid_edge_combinations = preprocess_result['valid_3edge_combos']
    valid_solutions = []
    epsilon = compute_epsilon(max_inscribed_circles, edge_characterizer)
    
    for edge_combo_info in valid_edge_combinations:
        edge_indices = edge_combo_info['edge_indices']
        e1, e2, e3 = edge_indices
        
        for t1_info in edge_sample_points[e1]:
            for t2_info in edge_sample_points[e2]:
                for t3_info in edge_sample_points[e3]:
                    
                    t_params = [t1_info[0], t2_info[0], t3_info[0]]
                    contacts = _build_contact_points(edge_indices, t_params, edge_characterizer)
                    
                    if not _check_points_distinct(contacts, tolerance=epsilon):
                        continue
                    
                    # Check robot center spacing (if robot_radius provided)
                    if robot_radius is not None:
                        if not _check_enough_space_for_robots(
                            contacts, robot_radius, min_edge_length=obj_min_edge_length
                        ):
                            continue
                    
                    solution = {
                        'contacts': contacts,
                        'edge_indices': edge_indices,
                        't_parameters': t_params,
                        'point_descriptions': [t1_info[1], t2_info[1], t3_info[1]],
                        'edge_combo_info': edge_combo_info
                    }
                    
                    solution['grasp_quality'] = _compute_grasp_quality_metrics_three_v2(
                        contacts, edge_characterizer, weighting_scheme, verbose=False
                    )
                    
                    valid_solutions.append(solution)
    
    if len(valid_solutions) == 0:
        if verbose:
            print(f"\n❌ No valid solutions found")
        return None
    
    # Rank solutions
    ranked_solutions = _rank_solutions_by_quality_three(valid_solutions, verbose=False)
    best_solution = ranked_solutions[0]
    
    if verbose:
        print(f"\n🏆 Best solution: {len(valid_solutions)} total solutions found")
        print(f"   Edges: {best_solution['edge_indices']}")
        print(f"   Quality score: {best_solution['grasp_quality']['overall_score']:.4f}")
    
    if visualize:
        _visualize_magnum_three_solution(best_solution, obj, max_inscribed_circles, edge_characterizer)
    
    return {
        'success': True,
        'best_solution': best_solution,
        'all_solutions': ranked_solutions,
        'num_solutions_found': len(valid_solutions)
    }



def find_the_magnum_three_v3_logtime(obj, verbose=True, visualize=True, force_magnitude=1.0, 
                              weighting_scheme='balanced', robot_radius: Optional[float] = None):
    """
    🆕 OPTIMIZED v3: Find three contact points using preprocessing for force closure.
    
    KEY OPTIMIZATIONS:
    1. **Preprocessing**: Use preprocess_object_force_closure() to cache valid 3-edge combos
    2. **O(1) Force Closure Check**: Direct lookup via is_valid_edge_combination()
    3. **No Redundant LP Solves**: Force closure already validated in preprocessing
    4. **Comprehensive Timing**: Track time for each stage
    
    This dramatically reduces:
    - Edge combination checks (O(1) lookup vs LP solve per combo)
    - Overall computation time
    
    Args:
        obj: GenericObject instance
        verbose: If True, print detailed search information
        visualize: If True, visualize the solution
        force_magnitude: Magnitude of normal forces for analysis
        weighting_scheme: 'balanced', 'focus_translational', or 'focus_rotational'
    
    Returns:
        dict: Solution containing contact points, closure metrics, timing, and statistics
              If no valid 3-edge combination found, returns None (caller should use Magnum Four)
    """
    import time
    
    print("\n" + "="*80)
    print("🔍 SEARCHING FOR THE MAGNUM THREE (v3 - OPTIMIZED)")
    print("="*80)
    
    timing = {}
    
    # =========================================================================
    # STEP 0: PREPROCESSING
    # =========================================================================
    print(f"\n⏱️  Step 0: Preprocessing object for force closure...")
    start_time = time.time()
    
    preprocess_result = preprocess_object_force_closure(obj, force_magnitude=force_magnitude, verbose=verbose)
    
    timing['preprocessing'] = time.time() - start_time
    print(f"   ✅ Preprocessing completed in {timing['preprocessing']:.4f} seconds")
    
    edge_characterizer = preprocess_result['edge_characterizer']
    num_edges = preprocess_result['num_edges']
    obj_min_edge_length = _get_object_min_edge_length(obj)
    
    if num_edges < 2:
        print("❌ Need at least 2 edges for meaningful contact configuration")
        return None
    
    # Quick check: Are there any valid 3-edge combinations?
    num_valid_3edge = len(preprocess_result['valid_3edge_combos'])
    
    if num_valid_3edge == 0:
        print(f"\n❌ No valid 3-edge combination found for force closure")
        print(f"   ➡️  Caller should fall back to Magnum Four (4 contacts)")
        return None
    
    print(f"   Found {num_valid_3edge} valid 3-edge combinations")
    
    # =========================================================================
    # STEP 1: FIND MAXIMUM INSCRIBED CIRCLE(S)
    # =========================================================================
    print(f"\n🔵 Step 1: Finding maximum inscribed circle(s)...")
    start_time = time.time()
    
    max_inscribed_circles = _find_max_inscribed_circles(
        obj, 
        edge_characterizer,
        method='auto'
    )
    
    if len(max_inscribed_circles) > 1:
        max_inscribed_circles = _rank_and_filter_circles(
            max_inscribed_circles, 
            obj,
            max_circles=4
        )
    
    timing['inscribed_circles'] = time.time() - start_time
    
    if verbose:
        for i, circle in enumerate(max_inscribed_circles):
            print(f"   Circle {i}: center=({circle['center'][0]:.3f}, {circle['center'][1]:.3f}), "
                  f"radius={circle['radius']:.3f}, tangency_points={circle['num_tangents']}")

    # Optional: compute reachable boundary intervals in t-space for a circular robot
    reachable_intervals = None
    if robot_radius is not None:
        try:
            print(f"\n🔍 Magnum Three (logtime): Computing reachable boundary intervals for robot radius={robot_radius:.4f}...")
            reachable_intervals = get_reachable_contact_intervals(
                obj.geometry,
                robot_radius=robot_radius,
                n_samples=2048,
            )
            if verbose:
                if len(reachable_intervals) == 0:
                    print("   ⚠️ No reachable boundary intervals found (all boundary treated as unreachable)")
                else:
                    print(f"   Reachable intervals in t-space (0-1):")
                    for idx, (t0, t1) in enumerate(reachable_intervals):
                        print(f"     [{idx}] t ∈ [{t0:.4f}, {t1:.4f}]")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Failed to compute reachable intervals for Magnum Three (logtime): {e}")
            reachable_intervals = None
    
    # =========================================================================
    # STEP 2: GENERATE STRATEGIC CONTACT POINT SAMPLES ON EACH EDGE
    # =========================================================================
    print(f"\n📍 Step 2: Sampling strategic contact points on each edge...")
    start_time = time.time()
    
    edge_sample_points = generate_strategic_contact_samples(
        edge_characterizer,
        max_inscribed_circles,
        verbose=False
    )

    # Optional: filter out samples that lie on unreachable parts of the boundary
    if robot_radius is not None and reachable_intervals:
        def _t_is_reachable_three_log(t_val: float) -> bool:
            for t0, t1 in reachable_intervals:
                if t0 <= t_val <= t1:
                    return True
            return False

        filtered_edge_sample_points = {}
        removed_total = 0
        for edge_idx, samples in edge_sample_points.items():
            filtered = []
            for t_val, desc in samples:
                if _t_is_reachable_three_log(t_val):
                    filtered.append((t_val, desc))
                else:
                    removed_total += 1
            filtered_edge_sample_points[edge_idx] = filtered
        edge_sample_points = filtered_edge_sample_points
        if verbose:
            print(f"   ✅ Magnum Three (logtime) reachability filter removed {removed_total} unreachable samples")
    
    total_samples = sum(len(samples) for samples in edge_sample_points.values())
    timing['sampling'] = time.time() - start_time
    print(f"   Total strategic samples: {total_samples}")
    
    # =========================================================================
    # STEP 3: 🆕 USE PREPROCESSING FOR VALID EDGE COMBINATIONS
    # =========================================================================
    print(f"\n🎯 Step 3: Using preprocessed valid 3-edge combinations...")
    start_time = time.time()
    
    # Extract valid edge combinations from preprocessing
    valid_edge_combinations = preprocess_result['valid_3edge_combos']
    
    timing['edge_filtering'] = time.time() - start_time
    print(f"   Valid 3-edge combinations: {len(valid_edge_combinations)}")
    print(f"   Filtering completed in {timing['edge_filtering']:.4f} seconds (instant via preprocessing!)")
    
    # =========================================================================
    # STEP 4: 🆕 OPTIMIZED SAMPLING (only on valid edge combos)
    # =========================================================================
    print(f"\n🔎 Step 4: Sampling contact points on {len(valid_edge_combinations)} valid edge combinations...")
    start_time = time.time()
    
    # Estimate total point combinations
    total_point_combos = 0
    for edge_combo_info in valid_edge_combinations:
        e1, e2, e3 = edge_combo_info['edge_indices']
        total_point_combos += (
            len(edge_sample_points[e1]) * 
            len(edge_sample_points[e2]) * 
            len(edge_sample_points[e3])
        )
    
    print(f"   Point combinations to test: {total_point_combos:,}")
    
    valid_solutions = []
    iteration_count = 0
    pruned_count = {
        'duplicate_points': 0,
        'insufficient_robot_spacing': 0
    }
    
    # Compute epsilon for spatial distinctness check
    epsilon = compute_epsilon(max_inscribed_circles, edge_characterizer)
    
    # Iterate through valid edge combinations
    for edge_combo_info in valid_edge_combinations:
        edge_indices = edge_combo_info['edge_indices']
        e1, e2, e3 = edge_indices
        
        # Iterate through all point combinations on these 3 edges
        for t1_info in edge_sample_points[e1]:
            for t2_info in edge_sample_points[e2]:
                for t3_info in edge_sample_points[e3]:
                    
                    iteration_count += 1
                    
                    if verbose and iteration_count % 1000 == 0:
                        print(f"   Tested {iteration_count:,} combinations...")
                    
                    # Extract t parameters
                    t_params = [t1_info[0], t2_info[0], t3_info[0]]
                    
                    # Build contact points
                    contacts = _build_contact_points(
                        edge_indices,
                        t_params,
                        edge_characterizer
                    )
                    
                    # Check points are spatially distinct
                    if not _check_points_distinct(contacts, tolerance=epsilon):
                        pruned_count['duplicate_points'] += 1
                        continue
                    
                    # Check robot center spacing (if robot_radius provided)
                    if robot_radius is not None:
                        if not _check_enough_space_for_robots(
                            contacts, robot_radius, min_edge_length=obj_min_edge_length
                        ):
                            pruned_count['insufficient_robot_spacing'] += 1
                            continue
                    
                    # Valid solution found! (force closure already validated)
                    solution = {
                        'contacts': contacts,
                        'edge_indices': edge_indices,
                        't_parameters': t_params,
                        'point_descriptions': [t1_info[1], t2_info[1], t3_info[1]],
                        'iteration_found': iteration_count,
                        'edge_combo_info': edge_combo_info  # Store coefficients and residual
                    }
                    
                    # Compute grasp quality metrics
                    solution['grasp_quality'] = _compute_grasp_quality_metrics_three_v2(
                        contacts,
                        edge_characterizer,
                        weighting_scheme=weighting_scheme,
                        verbose=False
                    )
                    
                    valid_solutions.append(solution)
                    
                    if verbose and len(valid_solutions) % 50 == 0:
                        print(f"   Found {len(valid_solutions)} valid solutions so far...")
    
    timing['point_sampling'] = time.time() - start_time
    timing['total'] = sum(timing.values())
    
    # =========================================================================
    # STEP 5: RANK SOLUTIONS AND SELECT BEST
    # =========================================================================
    print(f"\n📊 Step 5: Ranking solutions...")
    print(f"   Total iterations: {iteration_count:,}")
    print(f"   Valid solutions found: {len(valid_solutions)}")
    print(f"\n   Pruning statistics:")
    for reason, count in pruned_count.items():
        pct = count / total_point_combos * 100 if total_point_combos > 0 else 0
        print(f"      {reason}: {count:,} ({pct:.1f}%)")
    
    print(f"\n⏱️  Timing breakdown:")
    for stage, duration in timing.items():
        if stage != 'total':
            pct = duration / timing['total'] * 100
            print(f"      {stage}: {duration:.4f}s ({pct:.1f}%)")
    print(f"      TOTAL: {timing['total']:.4f}s")
    
    if len(valid_solutions) == 0:
        print(f"\n❌ No valid contact point configurations found")
        return None
    
    # Rank solutions
    start_time = time.time()
    ranked_solutions = _rank_solutions_by_quality_three(valid_solutions, verbose=verbose)
    timing['ranking'] = time.time() - start_time
    timing['total'] += timing['ranking']
    
    best_solution = ranked_solutions[0]
    
    print(f"\n🏆 Best solution selected:")
    print(f"   Edges: {best_solution['edge_indices']}")
    print(f"   Points: {best_solution['point_descriptions']}")
    print(f"   Grasp quality score: {best_solution['grasp_quality']['overall_score']:.4f}")
    print(f"   Torque range: {best_solution['grasp_quality']['individual_metrics']['torque_range_raw']:.4f}")
    print(f"   Force closure coefficients: {best_solution['edge_combo_info']['coefficients']}")
    print(f"   Found at iteration: {best_solution['iteration_found']}")
    
    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    if visualize:
        print(f"\n🎨 Visualizing solution...")
        _visualize_magnum_three_solution(
            best_solution, 
            obj, 
            max_inscribed_circles,
            edge_characterizer=edge_characterizer
        )
    
    return {
        'success': True,
        'best_solution': best_solution,
        'all_solutions': ranked_solutions,
        'num_solutions_found': len(valid_solutions),
        'iterations_tested': iteration_count,
        'pruning_stats': pruned_count,
        'timing': timing,
        'valid_edge_combinations': valid_edge_combinations,
        'max_inscribed_circles': max_inscribed_circles,
        'preprocess_result': preprocess_result,
        'statistics': {
            'total_3edge_combos': preprocess_result['statistics']['total_3edge_combos_tested'],
            'valid_3edge_combos': len(valid_edge_combinations),
            'total_point_combos': total_point_combos,
            'valid_solutions': len(valid_solutions)
        }
    }


def _compute_grasp_quality_metrics_three_v2(contacts, edge_characterizer, 
                                             weighting_scheme='balanced',
                                             verbose=False):
    """
    Compute grasp quality metrics for 3-contact configurations (force closure only).
    
    🆕 v2 SIMPLIFIED: Focus on two key metrics:
    - Metric 1: Force Space Radius (translational capability)
    - Metric 5: Torque Range (rotational capability)
    
    Weighting schemes encode torque range preference:
    - 'balanced': Balance force space and torque range
    - 'focus_translational': Maximize force space, MINIMIZE torque range (uniform torque)
    - 'focus_rotational': Balance force space with MAXIMIZE torque range (diverse torque)
    
    Args:
        contacts: List of 3 ContactPoint objects
        edge_characterizer: EdgeCharacterizer instance
        weighting_scheme: 'balanced', 'focus_translational', or 'focus_rotational'
        verbose: If True, print detailed metrics
    
    Returns:
        dict: Quality metrics
    """
    if verbose:
        print(f"\n📊 Computing grasp quality metrics for 3 contacts (scheme: {weighting_scheme})...")
    
    n_contacts = len(contacts)
    
    # Extract information
    force_directions = []
    positions = []
    
    for contact in contacts:
        positions.append(contact.position)
        force_directions.append(contact.normal_inward)
    
    # Build force matrix F (2×3)
    F = np.zeros((2, n_contacts))
    
    for i, contact in enumerate(contacts):
        F[0, i] = contact.normal_inward[0]
        F[1, i] = contact.normal_inward[1]
    
    # Build full grasp matrix G (3×3) for torque analysis
    G = np.zeros((3, n_contacts))
    
    for i, contact in enumerate(contacts):
        wrench = contact.calculate_contact_wrench(
            normal_force=1.0,
            tangential_force=0.0,
            friction_constraint=True
        )
        G[0, i] = wrench['force_x']
        G[1, i] = wrench['force_y']
        G[2, i] = wrench['torque']
    
    # =========================================================================
    # METRIC 1: FORCE SPACE RADIUS (Translational Capability)
    # =========================================================================
    force_visualizer = WrenchSpaceVisualizer()
    
    wrench_data = force_visualizer.calculate_wrench_space(
        contacts,
        force_ranges=[(0.0, 5.0)] * n_contacts,
        sampling_density=3,
        enable_tangent_forces=False
    )
    
    forces_2d = wrench_data['wrenches'][:, :2]  # Only Fx, Fy
    
    try:
        hull = ConvexHull(forces_2d)
        origin = np.array([0.0, 0.0])
        hull_vertices_2d = forces_2d[hull.vertices]
        
        origin_inside = _point_in_convex_hull_2d(origin, hull_vertices_2d)
        
        if origin_inside:
            force_space_radius = _min_distance_to_convex_hull_edges(origin, hull_vertices_2d)
        else:
            force_space_radius = 0.0
    except:
        force_space_radius = 0.0
    
    # Normalize
    typical_force_magnitude = 5.0
    normalized_force_radius = force_space_radius / typical_force_magnitude
    k_sensitivity = 5.0
    force_space_score = 1.0 - np.exp(-k_sensitivity * normalized_force_radius)
    
    if verbose:
        print(f"   Metric 1 - Force Space Radius: {force_space_radius:.4f}, Score: {force_space_score:.4f}")
    
    # =========================================================================
    # METRIC 5: TORQUE RANGE (Rotational Capability)
    # =========================================================================
    # Extract torque values from grasp matrix (third row)
    torque_values = G[2, :]
    
    torque_max = np.max(torque_values)
    torque_min = np.min(torque_values)
    torque_range = torque_max - torque_min
    
    # Normalize by typical moment arm (~0.5m) and force (5.0N)
    typical_torque = 5.0 * 0.5  # 2.5 N·m
    normalized_torque_range = torque_range / typical_torque
    
    # Score based on weighting scheme
    # 'focus_translational': MINIMIZE torque range (uniform distribution)
    # 'focus_rotational': MAXIMIZE torque range (diverse capability)
    # 'balanced': Balanced approach
    
    k_torque = 2.0
    
    if weighting_scheme == 'focus_translational':
        # Smaller range is better (more uniform torque)
        torque_range_score = np.exp(-k_torque * normalized_torque_range)
        torque_preference = 'minimize'
    elif weighting_scheme == 'focus_rotational':
        # Larger range is better (more diverse torque)
        torque_range_score = 1.0 - np.exp(-k_torque * normalized_torque_range)
        torque_preference = 'maximize'
    else:  # 'balanced'
        # Balanced: moderate range is good, but not too extreme
        # Use a Gaussian-like scoring: peak at moderate range
        optimal_normalized_range = 0.5  # Optimal is moderate range
        range_deviation = abs(normalized_torque_range - optimal_normalized_range)
        torque_range_score = np.exp(-k_torque * range_deviation)
        torque_preference = 'balanced'
    
    if verbose:
        print(f"   Metric 5 - Torque Range: {torque_range:.4f} (min={torque_min:.4f}, max={torque_max:.4f})")
        print(f"              Preference: {torque_preference}, Score: {torque_range_score:.4f}")
    
    # =========================================================================
    # COMBINE METRICS WITH WEIGHTING SCHEMES
    # =========================================================================
    weighting_schemes = {
        'balanced': {
            'force_space_radius': 0.50,   # Equal weight
            'torque_range': 0.50          # Equal weight
        },
        'focus_translational': {
            'force_space_radius': 0.70,   # Prioritize force space
            'torque_range': 0.30          # Minimize torque range (uniform)
        },
        'focus_rotational': {
            'force_space_radius': 0.40,   # Still important
            'torque_range': 0.60          # Prioritize torque diversity
        }
    }
    
    if weighting_scheme not in weighting_schemes:
        print(f"⚠️ Unknown weighting scheme '{weighting_scheme}', using 'balanced'")
        weighting_scheme = 'balanced'
    
    weights = weighting_schemes[weighting_scheme]
    
    overall_score = (
        weights['force_space_radius'] * force_space_score +
        weights['torque_range'] * torque_range_score
    )
    
    if verbose:
        print(f"\n   📊 Overall Quality Score ({weighting_scheme}): {overall_score:.4f}")
        print(f"      Breakdown:")
        print(f"         Force Space Radius: {force_space_score:.3f} (weight: {weights['force_space_radius']:.2f})")
        print(f"         Torque Range:       {torque_range_score:.3f} (weight: {weights['torque_range']:.2f}) [{torque_preference}]")
    
    return {
        'overall_score': overall_score,
        'weighting_scheme': weighting_scheme,
        'torque_preference': torque_preference,
        'individual_metrics': {
            'force_space_radius': force_space_score,
            'force_space_radius_raw': force_space_radius,
            'torque_range': torque_range_score,
            'torque_range_raw': torque_range,
            'torque_min': torque_min,
            'torque_max': torque_max,
            'force_matrix': F,
            'grasp_matrix': G
        }
    }

def _visualize_magnum_three_solution(solution, obj, max_inscribed_circles, edge_characterizer):
    """
    Visualize the Magnum Three solution.
    
    Similar to _visualize_magnum_four_solution but adapted for 3 contacts.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Circle as MPLCircle
    
    fig, (ax_main, ax_metrics) = plt.subplots(2, 1, figsize=(14, 12),
                                               gridspec_kw={'height_ratios': [3, 1]})
    
    # =========================================================================
    # TOP PANEL: CONTACT CONFIGURATION
    # =========================================================================
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3)
    
    # Plot object boundary
    obj.visualize(ax=ax_main, alpha=0.3, facecolor='lightsteelblue', show_frame=True)
    
    # Plot contact points
    contacts = solution['contacts']
    point_descriptions = solution['point_descriptions']
    edge_indices = solution['edge_indices']
    grasp_quality = solution['grasp_quality']
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green
    force_scale = 0.8
    
    for i, contact in enumerate(contacts):
        pos = contact.position
        normal = contact.normal_inward
        color = colors[i]
        
        # Contact point
        ax_main.plot(pos[0], pos[1], 'o',
                     color=color,
                     markersize=14,
                     markeredgecolor='white',
                     markeredgewidth=2,
                     zorder=10)
        
        # Number label
        ax_main.text(pos[0], pos[1], str(i+1),
                     fontsize=10, fontweight='bold',
                     ha='center', va='center',
                     color='white')
        
        # Force vector
        force_vec = normal * force_scale
        ax_main.arrow(pos[0], pos[1], 
                      force_vec[0], force_vec[1],
                      head_width=0.02, 
                      head_length=0.02,
                      linewidth=2.5,
                      fc=color, 
                      ec='black',
                      alpha=0.8)
    
    # Draw maximum inscribed circles
    for i, circle in enumerate(max_inscribed_circles):
        center = circle['center']
        radius = circle['radius']
        
        circle_patch = MPLCircle(
            center, radius,
            fill=False,
            edgecolor='orange',
            linewidth=2,
            linestyle='--',
            alpha=0.7,
            label='Max Inscribed Circle' if i == 0 else ''
        )
        ax_main.add_patch(circle_patch)
        
        ax_main.plot(center[0], center[1], 
                     '*', 
                     color='orange',
                     markersize=12,
                     markeredgecolor='black',
                     markeredgewidth=1)
    
    # Add centroid
    centroid = obj.get_centroid()
    ax_main.plot(centroid.x, centroid.y,
                 'x',
                 color='black',
                 markersize=10,
                 markeredgewidth=2,
                 label='Centroid')
    
    # Title
    quality_score = grasp_quality['overall_score']
    torque_pref = grasp_quality.get('torque_preference', 'unknown')
    
    title_text = "🏆 THE MAGNUM THREE - Force Closure Configuration\n"
    title_text += f"Quality Score: {quality_score:.3f} | "
    title_text += f"Torque Preference: {torque_pref}"
    
    ax_main.set_title(title_text, fontsize=14, fontweight='bold', pad=15)
    ax_main.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal')
    
    # =========================================================================
    # BOTTOM PANEL: DETAILED METRICS
    # =========================================================================
    ax_metrics.axis('off')
    
    metrics = grasp_quality['individual_metrics']
    t_params = solution['t_parameters']
    force_matrix = metrics['force_matrix']
    grasp_matrix = metrics['grasp_matrix']
    
    text_content = []
    
    # Contact details
    text_content.append("=" * 80)
    text_content.append("📍 CONTACT POINT DETAILS")
    text_content.append("=" * 80)
    for i in range(len(contacts)):
        text_content.append(
            f"Contact {i+1}: t={t_params[i]:.6f} (Edge {edge_indices[i]}, {point_descriptions[i]})"
        )
        pos = contacts[i].position
        normal = contacts[i].normal_inward
        text_content.append(
            f"           Position: ({pos[0]:7.4f}, {pos[1]:7.4f}), Normal: ({normal[0]:7.4f}, {normal[1]:7.4f})"
        )
    
    # Force matrix
    text_content.append("\n" + "=" * 80)
    text_content.append("📊 FORCE MATRIX F (2×3)")
    text_content.append("=" * 80)
    text_content.append(f"Fx: [{force_matrix[0,0]:7.4f}, {force_matrix[0,1]:7.4f}, {force_matrix[0,2]:7.4f}]")
    text_content.append(f"Fy: [{force_matrix[1,0]:7.4f}, {force_matrix[1,1]:7.4f}, {force_matrix[1,2]:7.4f}]")
    
    # Grasp matrix (full with torques)
    text_content.append("\n" + "=" * 80)
    text_content.append("📊 GRASP MATRIX G (3×3)")
    text_content.append("=" * 80)
    text_content.append(f"Fx: [{grasp_matrix[0,0]:7.4f}, {grasp_matrix[0,1]:7.4f}, {grasp_matrix[0,2]:7.4f}]")
    text_content.append(f"Fy: [{grasp_matrix[1,0]:7.4f}, {grasp_matrix[1,1]:7.4f}, {grasp_matrix[1,2]:7.4f}]")
    text_content.append(f"τ : [{grasp_matrix[2,0]:7.4f}, {grasp_matrix[2,1]:7.4f}, {grasp_matrix[2,2]:7.4f}]")
    
    # Quality metrics (simplified for 2 metrics)
    text_content.append("\n" + "=" * 80)
    text_content.append("📈 QUALITY METRICS (Magnum Three)")
    text_content.append("=" * 80)
    text_content.append(f"Overall Score:           {quality_score:.4f} (Scheme: {grasp_quality['weighting_scheme']})")
    text_content.append(f"")
    text_content.append(f"Metric 1 - Force Space Radius:")
    text_content.append(f"   Score:     {metrics['force_space_radius']:.4f}")
    text_content.append(f"   Raw value: {metrics['force_space_radius_raw']:.6f}")
    text_content.append(f"")
    text_content.append(f"Metric 5 - Torque Range ({grasp_quality['torque_preference']}):")
    text_content.append(f"   Score:     {metrics['torque_range']:.4f}")
    text_content.append(f"   Raw value: {metrics['torque_range_raw']:.6f}")
    text_content.append(f"   Min/Max:   [{metrics['torque_min']:.4f}, {metrics['torque_max']:.4f}]")
    text_content.append(f"")
    
    # Force closure info
    if 'edge_combo_info' in solution:
        edge_info = solution['edge_combo_info']
        text_content.append("Force Closure Coefficients:")
        coeffs = edge_info['coefficients']
        text_content.append(f"   a = [{coeffs[0]:.3f}, {coeffs[1]:.3f}, {coeffs[2]:.3f}]")
        text_content.append(f"   Residual: {edge_info['residual']:.2e}")
    
    text_content.append(f"")
    text_content.append(f"Iteration Found:         {solution.get('iteration_found', 'N/A')}")
    
    full_text = '\n'.join(text_content)
    ax_metrics.text(0.05, 0.95, full_text,
                    transform=ax_metrics.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='left',
                    family='monospace',
                    bbox=dict(boxstyle='round,pad=0.8',
                             facecolor='white',
                             edgecolor='black',
                             linewidth=2,
                             alpha=0.95))
    
    plt.tight_layout()
    plt.show()

def _rank_solutions_by_quality_three(solutions, verbose=False):
    """
    Rank solutions by grasp quality metrics for 3-contact configurations.
    
    Simplified ranking for Magnum Three (only 2 metrics):
    1. Primary: Overall quality score (weighted combination of force space + torque range)
    2. Tie-breaker 1: Force space radius (translational capability)
    3. Tie-breaker 2: Torque range raw value (depends on preference in weighting scheme)
    
    Args:
        solutions: List of solution dicts (each with 'grasp_quality' key)
        verbose: If True, print ranking details
    
    Returns:
        list: Solutions sorted by quality (best first)
    """
    if verbose:
        print(f"\n📊 Ranking {len(solutions)} solutions by quality...")
    
    # Sort by overall score (descending - higher is better)
    ranked = sorted(
        solutions,
        key=lambda sol: (
            sol['grasp_quality']['overall_score'],              # Primary: overall score
            sol['grasp_quality']['individual_metrics']['force_space_radius'],  # Tie-breaker 1
            sol['grasp_quality']['individual_metrics']['torque_range']         # Tie-breaker 2
        ),
        reverse=True
    )
    
    if verbose and len(ranked) > 0:
        print(f"\n   Top 5 solutions:")
        for i, sol in enumerate(ranked[:5]):
            quality = sol['grasp_quality']
            overall = quality['overall_score']
            force_space = quality['individual_metrics']['force_space_radius']
            torque_range = quality['individual_metrics']['torque_range']
            torque_range_raw = quality['individual_metrics']['torque_range_raw']
            
            print(f"      #{i+1}: Score={overall:.4f}")
            print(f"            Force Space={force_space:.3f}, Torque Range={torque_range:.3f} (raw={torque_range_raw:.4f})")
            print(f"            Edges: {sol['edge_indices']}")
            print(f"            Points: {sol['point_descriptions']}")
    
    return ranked



print("✅ _compute_grasp_quality_metrics_three_v2() SIMPLIFIED!")
print("\n📝 Key changes:")
print("   • Only 2 metrics: Force Space Radius + Torque Range")
print("   • Removed: Translational Robustness, Condition Number, Spatial Distribution")
print("   • Torque range preference encoded in weighting scheme:")
print("      - 'balanced': Equal weights, moderate torque range preferred")
print("      - 'focus_translational': 70% force space, MINIMIZE torque range")
print("      - 'focus_rotational': 40% force space, MAXIMIZE torque range")
print("\n💡 Weighting schemes:")
print("   balanced           = {force_space: 0.50, torque_range: 0.50}")
print("   focus_translational = {force_space: 0.70, torque_range: 0.30} [minimize torque range]")
print("   focus_rotational    = {force_space: 0.40, torque_range: 0.60} [maximize torque range]")

# %%
if __name__ == "__main__":
    standard_objects = create_standard_objects()
    # obj = standard_objects['l_shape']
    # obj = standard_objects['boot']
    obj = standard_objects['fat_triangle']
    # obj = standard_objects['u_shape']
    # obj = standard_objects['star']

    # find_magnum_result = find_the_magnum_three_v2(obj, verbose=True, visualize=True, weighting_scheme='balanced')
    find_magnum_result = find_the_magnum_three_v3(obj, verbose=False, visualize=True, weighting_scheme='balanced')
    # find_magnum_result = find_the_magnum_three_v3(obj, verbose=True, visualize=True, weighting_scheme='focus_translational')
    # find_magnum_result = find_the_magnum_three_v3(obj, verbose=True, visualize=True, weighting_scheme='focus_rotational')

    

# %%
def check_minimum_contacts_required(obj, verbose=False, preprocess_result=None):
    """
    Determine the minimum number of contacts required for force closure.
    
    🆕 OPTIMIZED: Uses preprocessing for instant O(1) check!
    
    Strategy:
    - If preprocess_result provided → Check valid_3edge_set size (O(1))
    - Otherwise → Run preprocessing with include_4edge=False
    - If valid 3-edge combos exist → return 3
    - Otherwise → return 4
    
    Args:
        obj: GenericObject instance
        verbose: If True, print search progress
        preprocess_result: Optional preprocessing result (for O(1) check)
    
    Returns:
        int: 3 or 4 (minimum number of contacts required for force closure)
    """
    if verbose:
        print("\n" + "="*80)
        print("🔍 CHECKING MINIMUM CONTACTS REQUIRED FOR FORCE CLOSURE")
        print("="*80)
    
    # =========================================================================
    # STEP 1: Get or compute preprocessing
    # =========================================================================
    if preprocess_result is None:
        if verbose:
            print(f"\n📐 Running preprocessing (3-edge only)...")
        
        preprocess_result = preprocess_object_force_closure(
            obj, 
            force_magnitude=1.0,
            verbose=verbose,
            include_4edge=False  # Only need 3-edge check
        )
    
    # =========================================================================
    # STEP 2: Check if any valid 3-edge combinations exist (O(1))
    # =========================================================================
    num_valid_3edge = len(preprocess_result['valid_3edge_set'])
    
    if verbose:
        print(f"\n📊 Analysis:")
        print(f"   Total edges: {preprocess_result['num_edges']}")
        print(f"   Valid 3-edge combinations: {num_valid_3edge}")
    
    if num_valid_3edge > 0:
        # Force closure possible with 3 contacts
        if verbose:
            print(f"\n   ✅ Force closure achievable with 3 contacts")
            print(f"   📊 RESULT: Minimum contacts = 3")
        return 3
    else:
        # Need 4 contacts for force closure
        if verbose:
            print(f"\n   ❌ No valid 3-edge combinations")
            print(f"   📊 RESULT: Minimum contacts = 4")
        return 4


print("✅ check_minimum_contacts_required() implemented!")
print("\n📝 Usage:")
print("   min_contacts = check_minimum_contacts_required(obj, verbose=True)")
print("   # Returns 3 or 4")

if __name__ == "__main__":
    standard_objects = create_standard_objects()

    print(check_minimum_contacts_required(standard_objects['triangle'], verbose=False))
    print(check_minimum_contacts_required(standard_objects['rectangle'], verbose=False))
    print(check_minimum_contacts_required(standard_objects['boot'], verbose=False))

# %%
def find_on_demand_magnum_cps_v3(obj, desired_goal, verbose=True, visualize=True, 
                                  force_magnitude=1.0, weighting_scheme='balanced',
                                  torque_method=3, preprocess_result=None):
    """
    🆕 OPTIMIZED v3: Find contact points tailored to specific control goals using preprocessing.
    
    KEY IMPROVEMENTS:
    - Uses preprocessing to avoid redundant edge checks
    - Shares preprocessing between Magnum Three and Four when possible
    - Significantly faster than v2
    """
    # Validate inputs
    valid_goals = ['omega_only', 'position_only', 'full_pose']
    if desired_goal not in valid_goals:
        return {
            'success': False,
            'error': f"Invalid desired_goal '{desired_goal}'. Must be one of {valid_goals}"
        }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🎯 FINDING ON-DEMAND CONTACT POINTS (v3 - Optimized)")
        print(f"   Control Goal: {desired_goal}")
        print(f"{'='*80}")
    
    # =========================================================================
    # GOAL: FULL_POSE → Need Magnum Four
    # =========================================================================
    if desired_goal == 'full_pose':
        if verbose:
            print(f"\n📍 Goal: full_pose → Computing Magnum Four...")
        
        magnum_result = find_the_magnum_four_v3(
            obj,
            verbose=verbose,
            visualize=visualize,
            force_magnitude=force_magnitude,
            weighting_scheme=weighting_scheme,
            torque_method=torque_method,
            preprocess_result=preprocess_result  # 🆕 Pass preprocessing!
        )
        
        if not magnum_result['success']:
            return {
                'success': False,
                'error': 'Failed to find Magnum Four',
                'desired_goal': desired_goal
            }
        
        selected_contacts = magnum_result['best_solution']['contacts']
        selection_method = "full_pose: All 4 Magnum Four contacts"
        
        return {
            'success': True,
            'desired_goal': desired_goal,
            'contacts': selected_contacts,
            'num_contacts': len(selected_contacts),
            'selection_method': selection_method,
            'magnum_four_result': magnum_result,
            'preprocess_result': magnum_result['preprocess_result']
        }
    
    # =========================================================================
    # GOAL: POSITION_ONLY → Check if 3 or 4 needed
    # =========================================================================
    elif desired_goal == 'position_only':
        if verbose:
            print(f"\n📍 Goal: position_only → Checking minimum contacts required...")
        
        # Use or create preprocessing
        if preprocess_result is None:
            if verbose:
                print(f"   Running preprocessing...")
            
            preprocess_result = preprocess_object_force_closure(
                obj,
                force_magnitude=force_magnitude,
                verbose=verbose,
                include_4edge=True  # Need both 3 and 4 edge combos
            )
        
        # Check minimum contacts using preprocessing
        min_contacts = check_minimum_contacts_required(
            obj, 
            verbose=verbose,
            preprocess_result=preprocess_result
        )
        
        if min_contacts == 3:
            # Use Magnum Three
            if verbose:
                print(f"\n   ✅ Object can achieve force closure with 3 contacts")
                print(f"   🔍 Finding Magnum Three...")
            
            magnum_three_result = find_the_magnum_three_v3(
                obj,
                verbose=verbose,
                visualize=visualize,
                force_magnitude=force_magnitude,
                weighting_scheme=weighting_scheme,
                preprocess_result=preprocess_result  # 🆕 Pass preprocessing!
            )
            
            if not magnum_three_result or not magnum_three_result.get('success', False):
                return {
                    'success': False,
                    'error': 'Failed to find Magnum Three',
                    'desired_goal': desired_goal,
                    'min_contacts_required': min_contacts
                }
            
            selected_contacts = magnum_three_result['best_solution']['contacts']
            selection_method = "position_only: 3 contacts (Magnum Three)"
            
            return {
                'success': True,
                'desired_goal': desired_goal,
                'contacts': selected_contacts,
                'num_contacts': 3,
                'selection_method': selection_method,
                'min_contacts_required': min_contacts,
                'magnum_three_result': magnum_three_result,
                'preprocess_result': preprocess_result
            }
        
        else:  # min_contacts == 4
            # Use Magnum Four
            if verbose:
                print(f"\n   ✅ Object requires 4 contacts for force closure")
                print(f"   🔍 Finding Magnum Four...")
            
            magnum_four_result = find_the_magnum_four_v3(
                obj,
                verbose=verbose,
                visualize=visualize,
                force_magnitude=force_magnitude,
                weighting_scheme=weighting_scheme,
                torque_method=torque_method,
                preprocess_result=preprocess_result  # 🆕 Pass preprocessing!
            )
            
            if not magnum_four_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to find Magnum Four',
                    'desired_goal': desired_goal,
                    'min_contacts_required': min_contacts
                }
            
            selected_contacts = magnum_four_result['best_solution']['contacts']
            selection_method = "position_only: 4 contacts required (Magnum Four)"
            
            return {
                'success': True,
                'desired_goal': desired_goal,
                'contacts': selected_contacts,
                'num_contacts': 4,
                'selection_method': selection_method,
                'min_contacts_required': min_contacts,
                'magnum_four_result': magnum_four_result,
                'preprocess_result': magnum_four_result['preprocess_result']
            }
    
    # =========================================================================
    # GOAL: OMEGA_ONLY → Pick 2 near-corner points with opposite torque signs
    # =========================================================================
    elif desired_goal == 'omega_only':
        if verbose:
            print(f"\n📍 Goal: omega_only → Finding 2 contacts with opposite torque signs...")
        
        # Create edge characterizer (if not in preprocessing)
        if preprocess_result is None:
            edge_characterizer = EdgeCharacterizer(obj, force_magnitude=force_magnitude)
        else:
            edge_characterizer = preprocess_result['edge_characterizer']
        
        num_edges = len(edge_characterizer.edges)
        
        # Analyze torque signs on each edge
        edge_torque_info = []
        
        for edge_idx in range(num_edges):
            edge_name = f'edge_{edge_idx}'
            char = edge_characterizer.edge_characteristics[edge_name]
            
            torque_min = char['torque_min']
            torque_max = char['torque_max']
            
            has_positive = torque_max > 0
            has_negative = torque_min < 0
            
            edge_torque_info.append({
                'edge_idx': edge_idx,
                'torque_min': torque_min,
                'torque_max': torque_max,
                'has_positive': has_positive,
                'has_negative': has_negative,
                'has_both_signs': has_positive and has_negative
            })
            
            if verbose:
                sign_status = "both signs ✓" if has_positive and has_negative else \
                             "positive only" if has_positive else \
                             "negative only" if has_negative else "zero torque"
                print(f"   Edge {edge_idx}: τ ∈ [{torque_min:.4f}, {torque_max:.4f}] ({sign_status})")
        
        # Strategy 1: Find ONE edge with both torque signs
        edges_with_both = [e for e in edge_torque_info if e['has_both_signs']]
        
        # Get max inscribed circles for epsilon calculation
        max_inscribed_circles = _find_max_inscribed_circles(obj, edge_characterizer, method='auto')
        epsilon = compute_epsilon(max_inscribed_circles, edge_characterizer)
        
        if len(edges_with_both) > 0:
            # Pick the first edge with both signs
            selected_edge = edges_with_both[0]
            edge_idx = selected_edge['edge_idx']
            
            if verbose:
                print(f"\n   ✅ Strategy 1: Found edge {edge_idx} with both torque signs")
                print(f"      Picking 2 near-corner contacts from this edge")
            
            edge_info = edge_characterizer.edges[edge_idx]
            t_start = edge_info['start_param']
            t_end = edge_info['end_param']
            
            epsilon_param = epsilon / edge_info['length'] if edge_info['length'] > 0 else epsilon
            
            t1 = t_start + epsilon_param
            t2 = t_end - epsilon_param
            
            contacts = _build_contact_points([edge_idx, edge_idx], [t1, t2], edge_characterizer)
            selection_method = f"omega_only: 2 near-corner contacts from Edge {edge_idx} (has both torque signs)"
            
        else:
            # Strategy 2: Find TWO edges with opposite signs
            if verbose:
                print(f"\n   🔄 Strategy 2: No single edge has both signs")
                print(f"      Looking for 2 edges with opposite signs...")
            
            positive_edges = [e for e in edge_torque_info if e['has_positive'] and not e['has_negative']]
            negative_edges = [e for e in edge_torque_info if e['has_negative'] and not e['has_positive']]
            
            if len(positive_edges) > 0 and len(negative_edges) > 0:
                pos_edge = positive_edges[0]
                neg_edge = negative_edges[0]
                
                if verbose:
                    print(f"   ✅ Found Edge {pos_edge['edge_idx']} (positive) and Edge {neg_edge['edge_idx']} (negative)")
                    print(f"      Picking 1 near-corner contact from each")
                
                edge_info_pos = edge_characterizer.edges[pos_edge['edge_idx']]
                edge_info_neg = edge_characterizer.edges[neg_edge['edge_idx']]
                
                epsilon_param_pos = epsilon / edge_info_pos['length'] if edge_info_pos['length'] > 0 else epsilon
                epsilon_param_neg = epsilon / edge_info_neg['length'] if edge_info_neg['length'] > 0 else epsilon
                
                t_pos = edge_info_pos['start_param'] + epsilon_param_pos
                t_neg = edge_info_neg['start_param'] + epsilon_param_neg
                
                contacts = _build_contact_points(
                    [pos_edge['edge_idx'], neg_edge['edge_idx']],
                    [t_pos, t_neg],
                    edge_characterizer
                )
                
                selection_method = f"omega_only: 1 contact from Edge {pos_edge['edge_idx']} (+τ), 1 from Edge {neg_edge['edge_idx']} (-τ)"
            
            else:
                # Fallback: No opposite signs available
                if verbose:
                    print(f"\n   ❌ No edges with opposite torque signs found!")
                    print(f"      Falling back to first 2 edges")
                
                edge_info_0 = edge_characterizer.edges[0]
                edge_info_1 = edge_characterizer.edges[1] if num_edges > 1 else edge_info_0
                
                epsilon_param_0 = epsilon / edge_info_0['length'] if edge_info_0['length'] > 0 else epsilon
                epsilon_param_1 = epsilon / edge_info_1['length'] if edge_info_1['length'] > 0 else epsilon
                
                t0 = edge_info_0['start_param'] + epsilon_param_0
                t1 = edge_info_1['start_param'] + epsilon_param_1
                
                contacts = _build_contact_points([0, 1 if num_edges > 1 else 0], [t0, t1], edge_characterizer)
                selection_method = "omega_only: Fallback - 2 contacts from first edges (no opposite torques available)"
        
        # Visualize if requested
        if visualize:
            _visualize_omega_only_solution(contacts, obj, edge_characterizer, selection_method)
        
        return {
            'success': True,
            'desired_goal': desired_goal,
            'contacts': contacts,
            'num_contacts': 2,
            'selection_method': selection_method,
            'edge_torque_info': edge_torque_info,
            'preprocess_result': preprocess_result
        }


def _visualize_omega_only_solution(contacts, obj, edge_characterizer, selection_method):
    """Simple visualization for omega_only solution."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw object
    obj.visualize(ax=ax, alpha=0.3, facecolor='lightcyan', show_frame=True)
    
    # Draw contacts
    colors = ['red', 'blue']
    for i, contact in enumerate(contacts):
        pos = contact.position
        normal = contact.normal_inward
        
        # Contact point
        ax.plot(pos[0], pos[1], 'o', color=colors[i], markersize=12, 
               markeredgecolor='black', markeredgewidth=2)
        
        # Force arrow
        force_scale = 0.8
        ax.arrow(pos[0], pos[1], normal[0] * force_scale, normal[1] * force_scale,
                head_width=0.12, head_length=0.08, linewidth=3,
                fc=colors[i], ec='black', alpha=0.8)
        
        # Label
        ax.text(pos[0], pos[1] - 0.25, f'C{i+1}',
               ha='center', va='top', fontsize=12, fontweight='bold')
    
    ax.set_title(f"OMEGA-ONLY: 2 Contacts\n{selection_method}",
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


print("✅ find_on_demand_magnum_cps_v2() implemented - STREAMLINED VERSION!")
print("\n📊 KEY IMPROVEMENTS:")
print("   • No redundant calculations!")
print("   • omega_only: Direct edge analysis → pick 2 near-corner points")
print("   • position_only: check_minimum_contacts_required() → route to 3 or 4")
print("   • full_pose: find_the_magnum_four_v2()")
print("\n💡 Usage:")
print("   result = find_on_demand_magnum_cps_v2(obj, 'omega_only')")
print("   result = find_on_demand_magnum_cps_v2(obj, 'position_only')")
print("   result = find_on_demand_magnum_cps_v2(obj, 'full_pose')")




# %%
if __name__ == "__main__":
    standard_objects = create_standard_objects()

    # obj_type = 'fat_triangle'
    # obj_type = 'l_shape'
    # obj_type = 'boot'
    obj_type = 'rectangle'

    obj = standard_objects[obj_type]

    result = find_on_demand_magnum_cps_v3(obj, 'position_only')
    # result = find_on_demand_magnum_cps_v3(obj, 'position_only', selection_criterion='narrowest', magnum_four_result=dict_magnum_cache[obj_type])
    # result = find_on_demand_magnum_cps_v3(obj, 'position_only', magnum_four_result=dict_magnum_cache[obj_type])

    # result = find_on_demand_magnum_cps_v3(obj, 'full_pose', magnum_four_result=dict_magnum_cache[obj_type])



