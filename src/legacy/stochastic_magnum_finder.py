"""Stochastic AFC contact finder (Latin square search).

Dedicated module extracted from contact_optimizer_utils_test_ver.py.
Implements find_the_magnum_stochastic and its direct dependencies only.
Supports configurable n_contacts (default 4) via Latin-square column count.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, Voronoi
from shapely.geometry import Point

from object_utils import (
    ContactPoint,
    ContactPointParameterization,
    EdgeCharacterizer,
    WrenchSpaceVisualizer,
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


def _filter_edge_sample_points_by_reachability(
    edge_sample_points: Dict[int, list],
    geometry,
    robot_radius: float,
    *,
    verbose: bool = False,
) -> Tuple[Dict[int, list], int, Optional[list]]:
    """
    Drop strategic samples on boundary regions unreachable by a circular robot.

    Uses the same C-space buffer filter as find_the_magnum_four_v3 (object_utils).
    """
    if robot_radius is None or robot_radius <= 0:
        return edge_sample_points, 0, None

    try:
        reachable_intervals = get_reachable_contact_intervals(
            geometry,
            robot_radius=float(robot_radius),
            n_samples=2048,
        )
    except Exception as exc:
        if verbose:
            print(f"   ⚠️ Failed to compute reachable intervals: {exc}")
        return edge_sample_points, 0, None

    if not reachable_intervals:
        if verbose:
            print("   ⚠️ No reachable boundary intervals found (all boundary treated as unreachable)")
        return {edge_idx: [] for edge_idx in edge_sample_points}, sum(
            len(samples) for samples in edge_sample_points.values()
        ), reachable_intervals

    def _t_is_reachable(t_val: float) -> bool:
        t_val = float(t_val) % 1.0
        for t0, t1 in reachable_intervals:
            if t0 <= t_val <= t1:
                return True
        return False

    filtered: Dict[int, list] = {}
    removed_total = 0
    for edge_idx, samples in edge_sample_points.items():
        kept = []
        for t_val, desc in samples:
            if _t_is_reachable(t_val):
                kept.append((t_val, desc))
            else:
                removed_total += 1
        filtered[edge_idx] = kept

    if verbose:
        print(
            f"   ✅ Reachability filter (r={robot_radius:.4f} m) removed "
            f"{removed_total} unreachable samples"
        )
        print("   Reachable intervals in t-space (0-1):")
        for idx, (t0, t1) in enumerate(reachable_intervals):
            print(f"     [{idx}] t ∈ [{t0:.4f}, {t1:.4f}]")

    return filtered, removed_total, reachable_intervals


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
def find_the_magnum_stochastic(
    obj,
    threshold: float = 1.0,
    timeout: float = 10.0,
    n_ellipse_samples: int = 72,
    force_range_scalar: float = 2.0,
    robot_radius: Optional[float] = None,
    n_contacts: int = 4,
    used_tangent_as_fallback: bool = False,
    tangent_required: bool = False,
    theory_mode: bool = False,
    verbose: bool = True,
):
    """
    Phase 2: Latin Square-based stochastic search with early termination.

    High-level behavior:
        - Generate strategic contact point samples on all edges
        - Create a Latin square: n_contacts columns (robots), n_strategic_points rows
          Each column is a permutation of [0, 1, ..., n_strategic_points-1]
        - For each row (combination):
              1) Map n_contacts indices to strategic points (edge_idx, t_param)
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
        robot_radius: Optional robot radius for C-space reachability filtering and
            robot center spacing checks (engineering mode). When provided, strategic
            samples on unreachable boundary regions are removed before search.
        n_contacts:   Number of contact points / Latin-square columns (default 4).
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
            'n_contacts'         : int (number of contacts searched)
    """
    import time

    if n_contacts < 2:
        raise ValueError(f"n_contacts must be >= 2, got {n_contacts}")

    apply_robot_spacing_check = (robot_radius is not None) and (not theory_mode)
    apply_quick_prune = not theory_mode

    if verbose:
        print("\n" + "=" * 80)
        mode_label = "THEORY" if theory_mode else "ENGINEERING"
        print(
            f"🎲 STOCHASTIC AFC SEARCH (Phase 2 - Latin Square + Early Termination) "
            f"[{mode_label}, n_contacts={n_contacts}]"
        )
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
            'n_contacts': n_contacts,
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

    unreachable_samples_removed = 0
    if robot_radius is not None:
        if verbose:
            print(
                f"\n🔍 Computing reachable boundary intervals for robot radius="
                f"{robot_radius:.4f}..."
            )
        edge_sample_points, unreachable_samples_removed, _reachable_intervals = (
            _filter_edge_sample_points_by_reachability(
                edge_sample_points,
                obj.geometry,
                robot_radius,
                verbose=verbose,
            )
        )
    
    # Flatten strategic samples into a single list: [(edge_idx, t_param), ...]
    strategic_points = []
    for edge_idx, samples in edge_sample_points.items():
        for t_param, description in samples:
            strategic_points.append((edge_idx, t_param))
    
    n_strategic_points = len(strategic_points)
    
    if n_strategic_points < n_contacts:
        if verbose:
            print(
                f"❌ Not enough strategic points ({n_strategic_points}) – "
                f"need at least {n_contacts}."
            )
            if unreachable_samples_removed > 0:
                print(
                    f"   ({unreachable_samples_removed} samples removed by reachability filter)"
                )
        return {
            'success': False,
            'found_by': None,
            'reason': 'insufficient_strategic_points',
            'threshold': threshold,
            'theory_mode': theory_mode,
            'n_contacts': n_contacts,
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
        print(f"🤝 Contact count         : {n_contacts}")
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
        if unreachable_samples_removed > 0:
            print(f"🚫 Unreachable samples   : {unreachable_samples_removed} removed")

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

            latin_square = _create_latin_square(n_strategic_points, n_cols=n_contacts)
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
                            'n_contacts': n_contacts,
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
        'n_contacts': n_contacts,
        'batches_tested': batches_tested,
        'configs_tested': configs_tested,
        'pruned_count': pruned_count,
        'sufficiency_result': last_sufficiency_result,
    }
