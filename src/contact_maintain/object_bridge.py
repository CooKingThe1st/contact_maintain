"""Bridge between GenericObject (Shapely-based 2D) and PyBullet 3D objects.

Provides bidirectional conversion:
- generic_to_pybullet: Convert GenericObject to PyBullet body
- pybullet_to_generic: Convert PyBullet body back to GenericObject
- create_standard_pybullet_objects: Create all standard shapes in PyBullet
"""
import numpy as np
import pybullet as pyb
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

# Import from legacy object_utils
import sys
import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import GenericObject, create_standard_objects, estimate_realistic_mass, read_obj_to_vertices, dxf_to_generic
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_HEIGHT = 0.1  # Default extrusion height for 2D shapes
DEFAULT_COLOR = (0.4, 0.6, 0.8, 1.0)
GRAVITY = 9.81  # m/s^2


# ============================================================================
# PHYSICS PROPERTY HELPERS
# ============================================================================

def calculate_3d_inertia_from_2d(moment_of_inertia_2d: float, mass: float, height: float) -> Tuple[float, float, float]:
    """Calculate 3D inertia tensor diagonal from 2D moment of inertia.
    
    For a 2D shape extruded to height h, the 3D inertia tensor is:
    - Ixx: rotation around x-axis (involves height and y-extent)
    - Iyy: rotation around y-axis (involves height and x-extent)
    - Izz: rotation around z-axis (the original 2D moment of inertia)
    
    Parameters
    ----------
    moment_of_inertia_2d : float
        The 2D moment of inertia (Iz) from GenericObject.
    mass : float
        Object mass.
    height : float
        Extrusion height of the object.
    
    Returns
    -------
    tuple
        (Ixx, Iyy, Izz) inertia tensor diagonal.
    """
    # For a 2D shape extruded to height h:
    # Izz = I_2d (the planar moment of inertia)
    # Ixx ≈ Iyy ≈ I_2d/2 + m*h²/12 (parallel axis theorem for height)
    Izz = moment_of_inertia_2d
    
    # Approximation for Ixx, Iyy (assuming roughly symmetric shape)
    height_contribution = mass * height * height / 12.0
    Ixx = Izz / 2.0 + height_contribution
    Iyy = Izz / 2.0 + height_contribution
    
    return (Ixx, Iyy, Izz)


# ============================================================================
# CONVEX DECOMPOSITION UTILITIES
# ============================================================================

def is_convex(polygon: Polygon) -> bool:
    """Check if a polygon is convex.
    
    Parameters
    ----------
    polygon : Polygon
        Shapely Polygon to check.
    
    Returns
    -------
    bool
        True if convex, False otherwise.
    """
    coords = list(polygon.exterior.coords)[:-1]  # Remove duplicate closing point
    n = len(coords)
    
    if n < 3:
        return False
    
    sign = None
    for i in range(n):
        p1 = coords[i]
        p2 = coords[(i + 1) % n]
        p3 = coords[(i + 2) % n]
        
        # Cross product of edges
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign and abs(cross) > 1e-10:
            return False
    
    return True


def decompose_to_convex_parts(polygon: Polygon) -> List[Polygon]:
    """Decompose a non-convex polygon into convex parts.
    
    Supports:
    - Triangles (already convex, returned as-is)
    - Rectangles (already convex, returned as-is)
    - L-shapes (decomposed into 2 rectangles)
    - T-shapes (decomposed into 2 rectangles)
    
    For other shapes, falls back to convex hull approximation.
    
    ⚠️  KNOWN ISSUES:
        - L-shape: Physics and t-param compatibility are quite okay
        - T-shape: Has problems with t-param and cannot provide correct transformation
                   (physics behavior not tested yet)
        - Triangle: Physics is wrong (objects don't rotate when pushed at corners).
                   This is weird since triangles should be simpler than rectilinear shapes.
                   Double-check triangle creation vs rectilinear - the more complex shapes
                   (L/T) work but the simple triangle doesn't, which is funny.
    
    Parameters
    ----------
    polygon : Polygon
        Shapely Polygon to decompose.
    
    Returns
    -------
    list of Polygon
        List of convex polygons that approximate the original.
    """
    if is_convex(polygon):
        return [polygon]
    
    # For L-shape and T-shape, use manual decomposition
    coords = list(polygon.exterior.coords)[:-1]
    n = len(coords)
    
    # Triangles are already convex (3 vertices)
    if n == 3:
        return [polygon]
    
    # Try simple rectangle decomposition for L/T shapes (6-8 vertices)
    if 6 <= n <= 12:
        return _decompose_rectilinear(polygon)
    
    # Fallback: use convex hull (loses concave detail)
    return [polygon.convex_hull]


def _decompose_rectilinear(polygon: Polygon) -> List[Polygon]:
    """Decompose rectilinear polygon (L, T shapes) into rectangles.
    
    Simple, direct approach for L and T shapes only.
    Creates perfect tiling with no gaps.
    
    ⚠️  KNOWN ISSUES:
        - L-shape: Physics and t-param compatibility are quite okay
        - T-shape: Has problems with t-param and cannot provide correct transformation
                   (physics behavior not tested yet)
    """
    coords = list(polygon.exterior.coords)[:-1]
    n = len(coords)
    
    # Get unique x and y coordinates (sorted)
    xs = sorted(set(c[0] for c in coords))
    ys = sorted(set(c[1] for c in coords))
    minx, miny, maxx, maxy = polygon.bounds
    
    # For L-shape (6 vertices): always has 3 unique x and 3 unique y values
    # NOTE: L-shape has quite okay physics and t-param compatibility
    if n == 6:
        # L-shape decomposition: two rectangles that share a corner
        # The corner is always at one of the middle coordinates
        
        # Try Option 1: Horizontal base (bottom) + Vertical arm (left)
        # Base: xs[0] to xs[2], ys[0] to ys[1]
        # Arm: xs[0] to xs[1], ys[1] to ys[2]
        rect1 = Polygon([
            (xs[0], ys[0]), (xs[2], ys[0]), (xs[2], ys[1]), (xs[0], ys[1])
        ])
        rect2 = Polygon([
            (xs[0], ys[1]), (xs[1], ys[1]), (xs[1], ys[2]), (xs[0], ys[2])
        ])
        
        # Check if this decomposition works (rectangles are contained in polygon)
        if rect1.within(polygon.buffer(1e-6)) and rect2.within(polygon.buffer(1e-6)):
            # Verify they cover the polygon
            union_area = rect1.union(rect2).area
            if abs(union_area - polygon.area) < 1e-4:
                return [rect1, rect2]
        
        # Try Option 2: Vertical base (left) + Horizontal arm (bottom)
        # Base: xs[0] to xs[1], ys[0] to ys[2]
        # Arm: xs[1] to xs[2], ys[0] to ys[1]
        rect1 = Polygon([
            (xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[2]), (xs[0], ys[2])
        ])
        rect2 = Polygon([
            (xs[1], ys[0]), (xs[2], ys[0]), (xs[2], ys[1]), (xs[1], ys[1])
        ])
        
        if rect1.within(polygon.buffer(1e-6)) and rect2.within(polygon.buffer(1e-6)):
            union_area = rect1.union(rect2).area
            if abs(union_area - polygon.area) < 1e-4:
                return [rect1, rect2]
        
        # Try Option 3: Horizontal base (top) + Vertical arm (right)
        # Base: xs[0] to xs[2], ys[1] to ys[2]
        # Arm: xs[1] to xs[2], ys[0] to ys[1]
        rect1 = Polygon([
            (xs[0], ys[1]), (xs[2], ys[1]), (xs[2], ys[2]), (xs[0], ys[2])
        ])
        rect2 = Polygon([
            (xs[1], ys[0]), (xs[2], ys[0]), (xs[2], ys[1]), (xs[1], ys[1])
        ])
        
        if rect1.within(polygon.buffer(1e-6)) and rect2.within(polygon.buffer(1e-6)):
            union_area = rect1.union(rect2).area
            if abs(union_area - polygon.area) < 1e-4:
                return [rect1, rect2]
        
        # Try Option 4: Vertical base (right) + Horizontal arm (top)
        # Base: xs[1] to xs[2], ys[0] to ys[2]
        # Arm: xs[0] to xs[1], ys[1] to ys[2]
        rect1 = Polygon([
            (xs[1], ys[0]), (xs[2], ys[0]), (xs[2], ys[2]), (xs[1], ys[2])
        ])
        rect2 = Polygon([
            (xs[0], ys[1]), (xs[1], ys[1]), (xs[1], ys[2]), (xs[0], ys[2])
        ])
        
        if rect1.within(polygon.buffer(1e-6)) and rect2.within(polygon.buffer(1e-6)):
            union_area = rect1.union(rect2).area
            if abs(union_area - polygon.area) < 1e-4:
                return [rect1, rect2]
    
    # For T-shape (8 vertices): top bar + stem
    # ⚠️  WARNING: T-shape has problems with t-param and cannot provide correct transformation
    #     (physics behavior not tested yet)
    if n == 8:
        # Find the y coordinate where stem meets top bar
        # Look for the y value that has the most x coordinates (top bar)
        y_counts = {}
        for y in ys:
            count = sum(1 for c in coords if abs(c[1] - y) < 1e-6)
            y_counts[y] = count
        
        # The top bar typically has 4 x coordinates, stem has 2
        # Find the y with most coordinates (top bar)
        max_count_y = max(y_counts.items(), key=lambda x: x[1])[0]
        
        # The stem junction is typically the second highest y
        if len(ys) >= 3:
            mid_y = ys[-2]  # Second highest y
        else:
            mid_y = (miny + maxy) / 2
        
        # Find stem x boundaries from bottom coordinates
        bottom_coords = [c for c in coords if abs(c[1] - miny) < 1e-6]
        if len(bottom_coords) >= 2:
            x_at_bottom = sorted(set(c[0] for c in bottom_coords))
            if len(x_at_bottom) == 2:
                stem_x_min, stem_x_max = x_at_bottom[0], x_at_bottom[1]
            else:
                # Use middle x range
                stem_x_min = xs[len(xs)//2 - 1] if len(xs) >= 3 else xs[0]
                stem_x_max = xs[len(xs)//2] if len(xs) >= 3 else xs[-1]
        else:
            # Fallback: use center
            center_x = (minx + maxx) / 2
            stem_width = (maxx - minx) / 3
            stem_x_min = center_x - stem_width / 2
            stem_x_max = center_x + stem_width / 2
        
        # Top bar rectangle (full width)
        top_rect = Polygon([
            (minx, mid_y), (maxx, mid_y), (maxx, maxy), (minx, maxy)
        ])
        
        # Stem rectangle (narrow width)
        stem_rect = Polygon([
            (stem_x_min, miny),
            (stem_x_max, miny),
            (stem_x_max, mid_y),
            (stem_x_min, mid_y)
        ])
        
        # Verify decomposition
        if top_rect.within(polygon.buffer(1e-6)) and stem_rect.within(polygon.buffer(1e-6)):
            union_area = top_rect.union(stem_rect).area
            if abs(union_area - polygon.area) < 1e-4:
                return [top_rect, stem_rect]
    
    # Fallback: return convex hull
    return [polygon.convex_hull]


# ============================================================================
# GENERIC TO PYBULLET CONVERSION
# ============================================================================

# TODO: Alternative workflow for non-rectangular shapes (to fix physics issues):
#   Instead of creating shapes programmatically, use dedicated URDF files:
#   1. For shapes other than rectangle/square: load from URDF file
#   2. Load URDF into PyBullet first (proper physics properties in URDF)
#   3. Extract 2D vertices from top view (hardcoded for simplicity)
#   4. Create GenericObject from extracted vertices + mass/inertia from URDF
#   
#   This workflow is backward: PyBullet -> GenericObject (instead of GenericObject -> PyBullet)
#   But it ensures proper physics behavior since URDF files have correct collision meshes
#   and inertia properties.
#
#   Implementation plan:
#   - Create URDF files for: triangle, L-shape, T-shape, etc.
#   - Add function: urdf_to_generic(urdf_path, shape_name) -> GenericObject
#   - Modify generic_to_pybullet() to check if shape has URDF, use it if available
#   - Extract 2D boundary from URDF collision mesh or use hardcoded vertices

def generic_to_pybullet(
    generic_obj: GenericObject,
    height: float = DEFAULT_HEIGHT,
    position: Tuple[float, float, float] = (0, 0, 0),
    orientation: float = 0.0,
    color: Tuple[float, float, float, float] = DEFAULT_COLOR,
    use_compound: bool = True,
    ground_friction_mode: bool = False
) -> int:
    """Convert a GenericObject to a PyBullet body.
    
    Friction terminology in GenericObject:
    - kinetic_friction / static_friction: ground-object friction coefficients
    - lateral_friction: object-robot contact friction coefficient
    
    PyBullet computes effective friction between two bodies as the product
    of their individual lateralFriction values.
    
    Parameters
    ----------
    generic_obj : GenericObject
        The Shapely-based object to convert.
    height : float
        Height to extrude the 2D shape.
    position : tuple
        (x, y, z) position in world frame.
    orientation : float
        Rotation around z-axis in radians.
    color : tuple
        RGBA color for visualization.
    use_compound : bool
        If True, decompose non-convex shapes into compound bodies.
    ground_friction_mode : bool
        If True, set object's lateralFriction to kinetic_friction (for ground contact).
        If False, set to lateral_friction (for robot contact scenarios).
    
    Returns
    -------
    int
        PyBullet body UID.
    """
    polygon = generic_obj.geometry
    
    # Extract physics properties from GenericObject
    kinetic_friction = getattr(generic_obj, 'kinetic_friction', 0.2)
    
    # Choose which friction to use for the object's PyBullet lateralFriction
    if ground_friction_mode:
        # For ground contact validation: use kinetic_friction
        # Combined with ground's friction set to kinetic_friction, effective = kinetic * kinetic
        # To get effective = kinetic_friction, we set object to 1.0 and ground to kinetic_friction
        object_friction = 1.0  # Ground friction dominates
    else:
        # For robot contact scenarios: use lateral_friction
        object_friction = generic_obj.lateral_friction
    
    physics_props = {
        'mass': generic_obj.mass,
        'moment_of_inertia': generic_obj.moment_of_inertia,
        'object_friction': object_friction,  # What to set on PyBullet body
        'kinetic_friction': kinetic_friction,
        'static_friction': getattr(generic_obj, 'static_friction', 0.4),
        'lateral_friction': generic_obj.lateral_friction,
    }
    
    # Calculate 3D inertia from 2D moment of inertia
    inertia_3d = calculate_3d_inertia_from_2d(
        physics_props['moment_of_inertia'],
        physics_props['mass'],
        height
    )
    physics_props['inertia_3d'] = inertia_3d
    
    # Check if convex
    print(f"Polygon: {polygon} is convex: {is_convex(polygon)}")
    if is_convex(polygon):
        return _create_convex_body(
            polygon, physics_props, height, position, orientation, color
        )
    
    if use_compound:
        # Decompose into convex parts
        parts = decompose_to_convex_parts(polygon)
        print(f"Parts: {parts} and number of parts: {len(parts)}")
        if len(parts) == 1:
            return _create_convex_body(
                parts[0], physics_props, height, position, orientation, color
            )
        else:
            return _create_compound_body(
                parts, physics_props, height, position, orientation, color
            )
    else:
        # Use convex hull approximation
        return _create_convex_body(
            polygon.convex_hull, physics_props, height, position, orientation, color
        )


def _create_convex_body(
    polygon: Polygon,
    physics_props: Dict,
    height: float,
    position: Tuple[float, float, float],
    orientation: float,
    color: Tuple[float, float, float, float]
) -> int:
    """Create a single convex PyBullet body from a polygon.
    
    Uses appropriate shape type:
    - Triangles: GEOM_MESH (proper triangle mesh)
    - Circles: GEOM_CYLINDER
    - Rectangles/Squares: GEOM_BOX
    - Other: GEOM_MESH (extruded polygon)
    
    Parameters
    ----------
    polygon : Polygon
        Shapely Polygon (must be convex).
    physics_props : dict
        Physics properties including 'mass', 'inertia_3d', 'object_friction'.
    height : float
        Extrusion height.
    position : tuple
        (x, y, z) world position.
    orientation : float
        Z-axis rotation in radians.
    color : tuple
        RGBA color.
    
    Returns
    -------
    int
        PyBullet body UID.
    """
    mass = physics_props['mass']
    inertia_3d = physics_props['inertia_3d']  # (Ixx, Iyy, Izz)
    object_friction = physics_props['object_friction']
    
    # Get polygon coordinates (exterior ring, excluding duplicate closing point)
    coords = list(polygon.exterior.coords)[:-1]
    n_vertices = len(coords)
    
    # Check if it's roughly circular
    bounds = polygon.bounds
    half_extents = [
        (bounds[2] - bounds[0]) / 2,
        (bounds[3] - bounds[1]) / 2,
        height / 2
    ]
    centroid = polygon.centroid
    is_circular = polygon.buffer(0).equals(polygon.convex_hull)
    
    # For circles, use cylinder
    # if is_circular and abs(half_extents[0] - half_extents[1]) < 0.01:
    #     radius = (half_extents[0] + half_extents[1]) / 2
    #     collision_id = pyb.createCollisionShape(
    #         shapeType=pyb.GEOM_CYLINDER,
    #         radius=radius,
    #         height=height
    #     )
    #     visual_id = pyb.createVisualShape(
    #         shapeType=pyb.GEOM_CYLINDER,
    #         radius=radius,
    #         length=height,
    #         rgbaColor=color
    #     )
    # For triangles, create proper triangle mesh
    # ⚠️  WARNING: Triangle physics is wrong (objects don't rotate when pushed at corners).
    #     This is weird since triangles should be simpler than rectilinear shapes.
    #     Double-check triangle creation vs rectilinear - the more complex shapes (L/T) work
    #     but the simple triangle doesn't, which is funny. Check _create_triangle_mesh().
    if n_vertices == 3:
        vertices_3d, indices = _create_triangle_mesh(coords, height)
        collision_id = pyb.createCollisionShape(
            shapeType=pyb.GEOM_MESH,
            vertices=vertices_3d.tolist(),
            indices=indices.flatten().tolist()
        )
        visual_id = pyb.createVisualShape(
            shapeType=pyb.GEOM_MESH,
            vertices=vertices_3d.tolist(),
            indices=indices.flatten().tolist(),
            rgbaColor=color
        )
    # For rectangles/squares (4 vertices, axis-aligned), use box
    elif n_vertices == 4:
        # Check if it's axis-aligned (rectangle/square)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        x_unique = len(set(xs)) == 2
        y_unique = len(set(ys)) == 2
        
        if x_unique and y_unique:
            # Use box for rectangles/squares
            collision_id = pyb.createCollisionShape(
                shapeType=pyb.GEOM_BOX,
                halfExtents=half_extents
            )
            visual_id = pyb.createVisualShape(
                shapeType=pyb.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=color
            )
        else:
            # Non-axis-aligned quadrilateral, use mesh
            vertices_3d, indices = _create_polygon_mesh(coords, height)
            collision_id = pyb.createCollisionShape(
                shapeType=pyb.GEOM_MESH,
                vertices=vertices_3d.tolist(),
                indices=indices.flatten().tolist()
            )
            visual_id = pyb.createVisualShape(
                shapeType=pyb.GEOM_MESH,
                vertices=vertices_3d.tolist(),
                indices=indices.flatten().tolist(),
                rgbaColor=color
            )
    else:
        # For other polygons, use mesh
        vertices_3d, indices = _create_polygon_mesh(coords, height)
        collision_id = pyb.createCollisionShape(
            shapeType=pyb.GEOM_MESH,
            vertices=vertices_3d.tolist(),
            indices=indices.flatten().tolist()
        )
        visual_id = pyb.createVisualShape(
            shapeType=pyb.GEOM_MESH,
            vertices=vertices_3d.tolist(),
            indices=indices.flatten().tolist(),
            rgbaColor=color
        )
    
    # Create body at centroid
    centroid_coords = list(centroid.coords)[0]
    orn = pyb.getQuaternionFromEuler([0, 0, orientation])
    pos = [position[0] + centroid_coords[0], 
           position[1] + centroid_coords[1], 
           height / 2]
    
    # Create multibody with explicit inertia
    uid = pyb.createMultiBody(
        baseMass=mass,
        baseInertialFramePosition=[0, 0, 0],
        baseInertialFrameOrientation=[0, 0, 0, 1],
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=pos,
        baseOrientation=orn
    )
    
    # Set dynamics properties including inertia and friction
    pyb.changeDynamics(
        uid, -1,
        mass=mass,
        localInertiaDiagonal=list(inertia_3d),
        lateralFriction=object_friction,
        spinningFriction=0.01,
        rollingFriction=0.01,
        linearDamping=0.0,
        angularDamping=0.0
    )
    
    return uid


def _create_triangle_mesh(coords_2d, height):
    """Create a 3D triangle mesh from 2D coordinates.
    
    ⚠️  KNOWN ISSUE: Triangle physics is wrong (objects don't rotate when pushed at corners).
        This is weird since triangles should be simpler than rectilinear shapes.
        The more complex shapes (L/T) work but the simple triangle doesn't.
        Double-check this implementation vs rectilinear shape creation.
        Possible issues: mesh vertex ordering, inertia calculation, or mass distribution.
    
    Parameters
    ----------
    coords_2d : list
        List of 3 (x, y) coordinates
    height : float
        Extrusion height
    
    Returns
    -------
    tuple
        (vertices_3d, indices) arrays
    """
    if len(coords_2d) != 3:
        raise ValueError("Triangle must have exactly 3 vertices")
    
    vertices_3d = []
    indices = []
    
    # Bottom face vertices (z = 0)
    for x, y in coords_2d:
        vertices_3d.append([x, y, 0.0])
    
    # Top face vertices (z = height)
    for x, y in coords_2d:
        vertices_3d.append([x, y, height])
    
    # Bottom face (counter-clockwise when viewed from below)
    indices.append([0, 1, 2])
    
    # Top face (clockwise when viewed from above)
    indices.append([3, 5, 4])
    
    # Side faces
    indices.append([0, 3, 4])  # Side 0-1, triangle 1
    indices.append([0, 4, 1])  # Side 0-1, triangle 2
    indices.append([1, 4, 5])  # Side 1-2, triangle 1
    indices.append([1, 5, 2])  # Side 1-2, triangle 2
    indices.append([2, 5, 3])  # Side 2-0, triangle 1
    indices.append([2, 3, 0])  # Side 2-0, triangle 2
    
    return np.array(vertices_3d, dtype=np.float64), np.array(indices, dtype=np.int32)


def _create_polygon_mesh(coords_2d, height):
    """Create a 3D polygon mesh from 2D coordinates by extrusion.
    
    Parameters
    ----------
    coords_2d : list
        List of (x, y) coordinates
    height : float
        Extrusion height
    
    Returns
    -------
    tuple
        (vertices_3d, indices) arrays
    """
    n = len(coords_2d)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    
    vertices_3d = []
    indices = []
    
    # Bottom face vertices (z = 0)
    for x, y in coords_2d:
        vertices_3d.append([x, y, 0.0])
    
    # Top face vertices (z = height)
    for x, y in coords_2d:
        vertices_3d.append([x, y, height])
    
    # Bottom face: triangulate using fan from first vertex
    for i in range(n - 2):
        indices.append([0, i + 1, i + 2])
    
    # Top face: same triangulation but reversed winding
    top_base = n
    for i in range(n - 2):
        indices.append([top_base, top_base + i + 2, top_base + i + 1])
    
    # Side faces: connect bottom and top outlines
    for i in range(n):
        next_i = (i + 1) % n
        # Create two triangles for each side face
        indices.append([i, next_i, top_base + next_i])
        indices.append([i, top_base + next_i, top_base + i])
    
    return np.array(vertices_3d, dtype=np.float64), np.array(indices, dtype=np.int32)


def _create_compound_body(
    parts: List[Polygon],
    physics_props: Dict,
    height: float,
    position: Tuple[float, float, float],
    orientation: float,
    color: Tuple[float, float, float, float]
) -> int:
    """Create a compound PyBullet body from multiple convex polygons.
    
    Parameters
    ----------
    parts : list of Polygon
        List of convex polygons forming the compound shape.
    physics_props : dict
        Physics properties including 'mass', 'inertia_3d', 'lateral_friction'.
    height : float
        Extrusion height.
    position : tuple
        (x, y, z) world position.
    orientation : float
        Z-axis rotation in radians.
    color : tuple
        RGBA color.
    
    Returns
    -------
    int
        PyBullet body UID.
    """
    if len(parts) == 0:
        raise ValueError("No parts provided for compound body")
    
    total_mass = physics_props['mass']
    inertia_3d = physics_props['inertia_3d']
    object_friction = physics_props['object_friction']
    
    # Calculate mass distribution based on area
    total_area = sum(p.area for p in parts)
    masses = [total_mass * p.area / total_area for p in parts]
    
    # Create collision and visual shapes for each part
    collision_ids = []
    visual_ids = []
    positions = []
    
    # Use first part as base
    base_centroid = np.array(parts[0].centroid.coords[0])
    
    for i, (part, part_mass) in enumerate(zip(parts, masses)):
        bounds = part.bounds
        half_extents = [
            (bounds[2] - bounds[0]) / 2,
            (bounds[3] - bounds[1]) / 2,
            height / 2
        ]
        
        collision_id = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=half_extents
        )
        collision_ids.append(collision_id)
        
        visual_id = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color
        )
        visual_ids.append(visual_id)
        
        # Position relative to base centroid
        centroid = np.array(part.centroid.coords[0])
        rel_pos = centroid - base_centroid
        positions.append([rel_pos[0], rel_pos[1], 0])
    
    # Create multi-body
    orn = pyb.getQuaternionFromEuler([0, 0, orientation])
    pos = list(position)
    pos[0] += base_centroid[0]
    pos[1] += base_centroid[1]
    pos[2] = height / 2
    
    if len(parts) == 1:
        uid = pyb.createMultiBody(
            baseMass=total_mass,
            baseInertialFramePosition=[0, 0, 0],
            baseInertialFrameOrientation=[0, 0, 0, 1],
            baseCollisionShapeIndex=collision_ids[0],
            baseVisualShapeIndex=visual_ids[0],
            basePosition=pos,
            baseOrientation=orn
        )
    else:
        # Build link arrays
        link_masses = masses[1:]
        link_collision_ids = collision_ids[1:]
        link_visual_ids = visual_ids[1:]
        link_positions = positions[1:]
        link_orientations = [[0, 0, 0, 1]] * len(link_masses)
        link_inertial_positions = [[0, 0, 0]] * len(link_masses)
        link_inertial_orientations = [[0, 0, 0, 1]] * len(link_masses)
        link_parent_indices = [0] * len(link_masses)
        link_joint_types = [pyb.JOINT_FIXED] * len(link_masses)
        link_joint_axes = [[0, 0, 1]] * len(link_masses)
        
        uid = pyb.createMultiBody(
            baseMass=masses[0],
            baseInertialFramePosition=[0, 0, 0],
            baseInertialFrameOrientation=[0, 0, 0, 1],
            baseCollisionShapeIndex=collision_ids[0],
            baseVisualShapeIndex=visual_ids[0],
            basePosition=pos,
            baseOrientation=orn,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_ids,
            linkVisualShapeIndices=link_visual_ids,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_positions,
            linkInertialFrameOrientations=link_inertial_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes
        )
    
    # Set dynamics properties for base
    pyb.changeDynamics(
        uid, -1,
        localInertiaDiagonal=list(inertia_3d),
        lateralFriction=object_friction,
        spinningFriction=0.01,
        rollingFriction=0.01,
        linearDamping=0.0,
        angularDamping=0.0
    )
    
    # Set friction for all links
    for i in range(len(parts) - 1):
        pyb.changeDynamics(
            uid, i,
            lateralFriction=object_friction,
            spinningFriction=0.01,
            rollingFriction=0.01
        )
    
    return uid


# ============================================================================
# PYBULLET TO GENERIC CONVERSION
# ============================================================================

# TODO: Implement OBJ-based workflow here
#   Function: obj_to_generic(obj_path: str, shape_name: str, 
#                            position: Tuple, orientation: float) -> Tuple[GenericObject, int]
#   - Load OBJ file into PyBullet using createCollisionShape/createVisualShape
#   - Extract physics properties (mass, inertia) from PyBullet
#   - Use hardcoded 2D vertices for the shape (from top view)
#   - Create GenericObject with extracted properties
#   - Return (GenericObject, pybullet_uid)
#
#   This is the "backward" workflow: PyBullet -> GenericObject
#   But ensures correct physics since OBJ files have proper collision meshes

def pybullet_to_generic(
    uid: int,
    link_idx: int = -1,
    name: str = "PyBulletObject"
) -> GenericObject:
    """Convert a PyBullet body to a GenericObject.
    
    Extracts physics properties:
    - mass
    - moment_of_inertia (from Izz component)
    - lateral_friction
    
    Parameters
    ----------
    uid : int
        PyBullet body UID.
    link_idx : int
        Link index (-1 for base).
    name : str
        Name for the GenericObject.
    
    Returns
    -------
    GenericObject
        Shapely-based object with physics properties.
    """
    # Get collision shape data
    shape_data = pyb.getCollisionShapeData(uid, link_idx)
    
    if len(shape_data) == 0:
        raise ValueError(f"No collision shape found for body {uid}, link {link_idx}")
    
    shape_type = shape_data[0][2]
    dimensions = shape_data[0][3]
    
    # Get dynamics info
    # Returns: mass, lateral_friction, local_inertia_diagonal, local_inertial_pos, 
    #          local_inertial_orn, restitution, rolling_friction, spinning_friction,
    #          contact_damping, contact_stiffness, body_type, collision_margin
    dynamics = pyb.getDynamicsInfo(uid, link_idx)
    mass = dynamics[0]
    friction = dynamics[1]
    local_inertia = dynamics[2]  # (Ixx, Iyy, Izz)
    
    # Extract Izz as 2D moment of inertia
    moment_of_inertia = local_inertia[2] if len(local_inertia) >= 3 else 0.1
    
    # Get current pose
    if link_idx == -1:
        pos, orn = pyb.getBasePositionAndOrientation(uid)
    else:
        link_state = pyb.getLinkState(uid, link_idx)
        pos, orn = link_state[0], link_state[1]
    
    euler = pyb.getEulerFromQuaternion(orn)
    heading = euler[2]
    
    # Create geometry based on shape type
    if shape_type == pyb.GEOM_BOX:
        half_extents = dimensions
        vertices = [
            (-half_extents[0], -half_extents[1]),
            (half_extents[0], -half_extents[1]),
            (half_extents[0], half_extents[1]),
            (-half_extents[0], half_extents[1])
        ]
        geometry = Polygon(vertices)
        
    elif shape_type == pyb.GEOM_CYLINDER or shape_type == pyb.GEOM_SPHERE:
        radius = dimensions[1] if shape_type == pyb.GEOM_CYLINDER else dimensions[0]
        geometry = Point(0, 0).buffer(radius, resolution=32)
        
    elif shape_type == pyb.GEOM_MESH:
        # Try to extract vertices from mesh
        # This is limited - mesh data not easily accessible
        # Fallback to bounding box
        aabb_min, aabb_max = pyb.getAABB(uid, link_idx)
        half_x = (aabb_max[0] - aabb_min[0]) / 2
        half_y = (aabb_max[1] - aabb_min[1]) / 2
        vertices = [
            (-half_x, -half_y), (half_x, -half_y),
            (half_x, half_y), (-half_x, half_y)
        ]
        geometry = Polygon(vertices)
    else:
        # Unknown shape - use AABB
        aabb_min, aabb_max = pyb.getAABB(uid, link_idx)
        half_x = (aabb_max[0] - aabb_min[0]) / 2
        half_y = (aabb_max[1] - aabb_min[1]) / 2
        vertices = [
            (-half_x, -half_y), (half_x, -half_y),
            (half_x, half_y), (-half_x, half_y)
        ]
        geometry = Polygon(vertices)
    
    # Create GenericObject with all extracted physics properties
    obj = GenericObject(
        geometry=geometry,
        mass=mass if mass > 0 else 1.0,
        moment_of_inertia=moment_of_inertia,
        lateral_friction=friction,
        heading=heading,
        name=name
    )
    obj.position = np.array([pos[0], pos[1]])
    
    return obj


def print_physics_comparison(generic_obj: GenericObject, pybullet_uid: int, height: float = DEFAULT_HEIGHT):
    """Print comparison of physics properties between GenericObject and PyBullet body.
    
    Useful for debugging and verifying physics property transfer.
    
    Friction terminology:
    - kinetic_friction / static_friction: ground-object friction (in GenericObject)
    - lateral_friction: object-robot contact friction (in GenericObject)
    - PyBullet lateralFriction: general dry friction coefficient
    
    Parameters
    ----------
    generic_obj : GenericObject
        Original Shapely-based object.
    pybullet_uid : int
        PyBullet body UID.
    height : float
        Extrusion height used.
    """
    dynamics = pyb.getDynamicsInfo(pybullet_uid, -1)
    pyb_mass = dynamics[0]
    pyb_friction = dynamics[1]
    pyb_inertia = dynamics[2]
    
    kinetic_friction = getattr(generic_obj, 'kinetic_friction', 0.2)
    inertia_3d = calculate_3d_inertia_from_2d(generic_obj.moment_of_inertia, generic_obj.mass, height)
    
    print("\n" + "="*60)
    print(f"  PHYSICS PROPERTIES COMPARISON: {generic_obj.name}")
    print("="*60)
    print(f"  {'Property':<25} {'GenericObject':<15} {'PyBullet':<15}")
    print("-"*60)
    print(f"  {'Mass (kg)':<25} {generic_obj.mass:<15.4f} {pyb_mass:<15.4f}")
    print(f"  {'Contact Friction':<25} {generic_obj.lateral_friction:<15.4f} {pyb_friction:<15.4f}")
    print(f"  {'Ground Friction (μk)':<25} {kinetic_friction:<15.4f} {'(via ground)':<15}")
    print(f"  {'Moment of Inertia (2D)':<25} {generic_obj.moment_of_inertia:<15.4f} {pyb_inertia[2]:<15.4f}")
    print("-"*60)
    print(f"  {'Inertia Ixx':<25} {inertia_3d[0]:<15.4f} {pyb_inertia[0]:<15.4f}")
    print(f"  {'Inertia Iyy':<25} {inertia_3d[1]:<15.4f} {pyb_inertia[1]:<15.4f}")
    print(f"  {'Inertia Izz':<25} {inertia_3d[2]:<15.4f} {pyb_inertia[2]:<15.4f}")
    print("="*60)


def obj_to_generic(
    obj_path: str,
    shape_name: str,
    position: Tuple[float, float, float] = (0, 0, 0),
    orientation: float = 0.0,
    mass: float = 1.0,
    lateral_friction: float = 0.8,
    blind_test: bool = False,
    **kwargs
) -> Tuple[GenericObject, int]:
    """Load object from OBJ file and create GenericObject from it.
    
    This is the "backward" workflow: PyBullet -> GenericObject
    Used for shapes with physics issues (triangle, L-shape, T-shape).
    
    Workflow:
    1. Load OBJ file into PyBullet using createCollisionShape/createVisualShape
    2. Set physics properties (mass, inertia) manually or extract from PyBullet
    3. Extract 2D vertices directly from the OBJ file using read_obj_to_vertices (slices at z_min)
    4. Create GenericObject with extracted vertices and physics properties
    
    Parameters
    ----------
    obj_path : str
        Path to OBJ file (relative to package urdf directory or absolute)
    shape_name : str
        Name of the shape (used for object naming and color selection)
    position : tuple
        (x, y, z) position in world frame
    orientation : float
        Rotation around z-axis in radians
    mass : float
        Mass of the object (default: 1.0 kg)
    lateral_friction : float
        Lateral friction coefficient (default: 0.8)
    blind_test : bool
        If True, skip vertex matching validation when using DXF fallback.
        When OBJ polygon is invalid and DXF vertices don't match OBJ vertices,
        this flag allows using DXF geometry without raising an error (default: False)
    **kwargs
        Additional arguments (e.g., moment_of_inertia) for physics properties
    
    Returns
    -------
    tuple
        (GenericObject, pybullet_uid)
    """
    # Resolve OBJ path
    if not Path(obj_path).is_absolute():
        # Try package urdf directory first
        obj_full_path = Path(pkg_path) / "urdf" / obj_path
        if not obj_full_path.exists():
            # Fallback to provided path
            obj_full_path = Path(obj_path)
        obj_path = str(obj_full_path)
    
    if not Path(obj_path).exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    
    # Convert orientation to quaternion
    
    orientation_quat = pyb.getQuaternionFromEuler([0, 0, orientation])
    
    # Choose a simple per-shape color (MTL files are not reliably used by PyBullet
    # when loading raw OBJ meshes, so we assign colors explicitly here).
    shape_colors = {
        'right_triangle': (0.8, 0.3, 0.3, 1.0),
        # 'scalene_triangle': (0.3, 0.8, 0.3, 1.0),
        # 'equilateral_triangle': (0.3, 0.3, 0.8, 1.0),
        # 'l_shape': (0.8, 0.6, 0.2, 1.0),
        # 'asym_l_shape': (0.6, 0.2, 0.8, 1.0),
        # 't_shape': (0.2, 0.8, 0.8, 1.0),
        # 'asym_t_shape': (0.8, 0.2, 0.5, 1.0),
        # 'hourglass': (0.7, 0.7, 0.2, 1.0),
        'bolt': (0.2, 0.8, 0.8, 1.0),
        'pi': (0.9, 0.7, 0.3, 1.0),
        'root': (0.6, 0.9, 0.6, 1.0),
        'rectangle': (0.8, 0.8, 0.8, 1.0),
    }
    rgba = shape_colors.get(shape_name, (0.8, 0.8, 0.8, 1.0))

    # Load OBJ file into PyBullet (similar to test_display_obj.py)
    collision_shape_id = pyb.createCollisionShape(
        shapeType=pyb.GEOM_MESH,
        fileName=obj_path,
        flags=pyb.URDF_INITIALIZE_SAT_FEATURES
    )
    
    visual_shape_id = pyb.createVisualShape(
        shapeType=pyb.GEOM_MESH,
        fileName=obj_path,
        rgbaColor=rgba,
    )
    
    # Create multi-body
    body_uid = pyb.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position,
        baseOrientation=orientation_quat
    )
    
    # Set dynamics properties
    pyb.changeDynamics(
        body_uid, -1,
        lateralFriction=lateral_friction
    )
    
    # Extract physics properties from PyBullet (after setting them)
    dynamics_info = pyb.getDynamicsInfo(body_uid, -1)
    actual_mass = dynamics_info[0]
    actual_lateral_friction = dynamics_info[1]
    local_inertia_diagonal = dynamics_info[2]  # (Ixx, Iyy, Izz)
    moment_of_inertia = kwargs.get('moment_of_inertia', local_inertia_diagonal[2])  # Izz for 2D

    # 2D footprint: precomputed cache first, then OBJ/DXF slice fallback
    from contact_maintain.footprint_cache import vertices_for_shape

    vertices_2d = vertices_for_shape(shape_name)
    if not vertices_2d:
        try:
            vertices_2d = read_obj_to_vertices(obj_path)
        except Exception as e:
            raise ValueError(
                f"Failed to extract 2D vertices from OBJ file {obj_path}: {e}. "
                "Run scripts/test/preprocess_obj_footprints.py or ensure the OBJ can be sliced."
            )
    
    # Create polygon from extracted vertices
    geometry = Polygon(vertices_2d)
    if not geometry.is_valid or geometry.area <= 0.01:
        print(f"⚠ Invalid polygon from OBJ vertices (area={geometry.area}, is_valid={geometry.is_valid})")
        print(f"  Vertices from OBJ: {vertices_2d}")
        print(f"  Attempting fallback: using DXF file with reverse_sign=True...")
        
        # Fallback: try using DXF file with reverse_sign=True
        # Find corresponding DXF file (same name, different extension)
        obj_path_obj = Path(obj_path)
        dxf_path = obj_path_obj.with_suffix('.dxf')
        
        if not dxf_path.exists():
            # Try in the same directory as OBJ
            dxf_path = obj_path_obj.parent / f"{obj_path_obj.stem}.dxf"
        
        if not dxf_path.exists():
            raise ValueError(
                f"Invalid polygon from OBJ vertices and DXF file not found: {dxf_path}. "
                f"OBJ vertices may be in wrong order. Check OBJ file geometry."
            )
        
        # Use dxf_to_generic with reverse_sign=True
        try:
            dxf_generic_obj = dxf_to_generic(
                dxf_path=str(dxf_path),
                name=shape_name,
                reverse_sign=True
            )
            dxf_vertices = list(dxf_generic_obj.geometry.exterior.coords[:-1])  # Exclude duplicate last point
            
            # Check if DXF vertices match OBJ vertices (with tolerance)
            # Convert to sets of tuples for comparison (with tolerance)
            tolerance = 1e-3
            vertices_2d_set = set((round(v[0] / tolerance) * tolerance, round(v[1] / tolerance) * tolerance) for v in vertices_2d)
            dxf_vertices_set = set((round(v[0] / tolerance) * tolerance, round(v[1] / tolerance) * tolerance) for v in dxf_vertices)
            
            if vertices_2d_set != dxf_vertices_set:
                # Try with more lenient comparison (check if all OBJ vertices are close to some DXF vertex)
                all_match = True
                for obj_v in vertices_2d:
                    found_match = False
                    for dxf_v in dxf_vertices:
                        if np.linalg.norm(np.array(obj_v) - np.array(dxf_v)) < tolerance:
                            found_match = True
                            break
                    if not found_match:
                        all_match = False
                        break
                
                if not all_match:
                    if blind_test:
                        # In blind_test mode, skip validation and use DXF geometry anyway
                        print(f"⚠ blind_test=True: Vertices don't match, but using DXF geometry anyway")
                    else:
                        raise ValueError(
                            f"DXF vertices do not match OBJ vertices (tolerance={tolerance}). "
                            f"OBJ vertices: {vertices_2d}, DXF vertices: {dxf_vertices}"
                        )
            
            # Use geometry from DXF
            geometry = dxf_generic_obj.geometry
            print(f"✓ Using geometry from DXF file (area={geometry.area:.6f}, is_valid={geometry.is_valid})")
            
        except Exception as e:
            raise ValueError(
                f"Failed to use DXF fallback: {e}. "
                f"OBJ vertices may be in wrong order. Check OBJ file geometry."
            )
    
    # Estimate mass if not provided
    if actual_mass <= 0:
        actual_mass = estimate_realistic_mass(geometry) if mass <= 0 else mass
    
    # Create GenericObject with extracted geometry and physics properties
    generic_obj = GenericObject(
        geometry=geometry,
        mass=actual_mass,
        lateral_friction=actual_lateral_friction,
        heading=orientation,
        name=shape_name,
    )
    # New friction model: parameter is object material µ (PyBullet body).
    # Contact µ (material × bumper) is applied later by revised / spawn logic.
    generic_obj.set_material_friction(
        actual_lateral_friction,
        sync_legacy_lateral=True,
        sync_legacy_static=False,
    )
    generic_obj.position = np.array([position[0], position[1]])
    generic_obj.moment_of_inertia = moment_of_inertia

    return generic_obj, body_uid


# ============================================================================
# STANDARD OBJECTS FACTORY
# ============================================================================

def create_standard_pybullet_objects(
    height: float = DEFAULT_HEIGHT,
    spacing: float = 1.0
) -> Dict[str, Tuple[GenericObject, int]]:
    """Create all standard shapes in PyBullet.
    
    Parameters
    ----------
    height : float
        Extrusion height for all shapes.
    spacing : float
        Spacing between objects in the scene.
    
    Returns
    -------
    dict
        Mapping from shape name to (GenericObject, pybullet_uid) tuple.
    """
    standard_objects = create_standard_objects()
    result = {}
    
    # Colors for different shapes
    colors = {
        'rectangle': (0.8, 0.4, 0.2, 1.0),
        'circle': (0.2, 0.6, 0.8, 1.0),
        'ellipse': (0.6, 0.2, 0.8, 1.0),
        'triangle': (0.2, 0.8, 0.4, 1.0),
        'fat_triangle': (0.8, 0.8, 0.2, 1.0),
        'scalene': (0.8, 0.2, 0.6, 1.0),
        'l_shape': (0.4, 0.8, 0.8, 1.0),
        'asym_l_shape': (0.8, 0.6, 0.4, 1.0),
        't_shape': (0.2, 0.8, 0.2, 1.0),
        'plus_shape': (0.6, 0.6, 0.8, 1.0),
    }
    
    # Grid layout
    n_cols = 4
    col = 0
    row = 0
    
    for name, generic_obj in standard_objects.items():
        x = col * spacing
        y = row * spacing
        position = (x, y, 0)
        
        color = colors.get(name, DEFAULT_COLOR)
        
        try:
            uid = generic_to_pybullet(
                generic_obj, height=height, position=position, color=color
            )
            result[name] = (generic_obj, uid)
            print(f"  Created {name} at ({x:.1f}, {y:.1f})")
        except Exception as e:
            print(f"  Failed to create {name}: {e}")
        
        col += 1
        if col >= n_cols:
            col = 0
            row += 1
    
    return result


# ============================================================================
# WRAPPER CLASS FOR BRIDGED OBJECTS
# ============================================================================

class BridgedObject:
    """Wrapper that maintains both GenericObject and PyBullet representations.
    
    Keeps the two representations synchronized.
    """
    
    def __init__(self, generic_obj: GenericObject, pybullet_uid: int):
        """Initialize with both representations.
        
        Parameters
        ----------
        generic_obj : GenericObject
            The Shapely-based object.
        pybullet_uid : int
            The PyBullet body UID.
        """
        self.generic = generic_obj
        self.uid = pybullet_uid
    
    @classmethod
    def from_generic(cls, generic_obj: GenericObject, **kwargs) -> 'BridgedObject':
        """Create from GenericObject."""
        uid = generic_to_pybullet(generic_obj, **kwargs)
        return cls(generic_obj, uid)
    
    @classmethod
    def from_pybullet(cls, uid: int, **kwargs) -> 'BridgedObject':
        """Create from existing PyBullet body."""
        generic_obj = pybullet_to_generic(uid, **kwargs)
        return cls(generic_obj, uid)
    
    def sync_from_pybullet(self):
        """Update GenericObject state from PyBullet."""
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)
        euler = pyb.getEulerFromQuaternion(orn)
        
        self.generic.position = np.array([pos[0], pos[1]])
        self.generic.heading = euler[2]
    
    def sync_to_pybullet(self):
        """Update PyBullet state from GenericObject."""
        pos = [self.generic.position[0], self.generic.position[1], 0.05]
        orn = pyb.getQuaternionFromEuler([0, 0, self.generic.heading])
        pyb.resetBasePositionAndOrientation(self.uid, pos, orn)
    
    def get_pose(self) -> Tuple[np.ndarray, float]:
        """Get current pose (position, heading)."""
        self.sync_from_pybullet()
        return self.generic.position.copy(), self.generic.heading
    
    def get_velocity(self) -> Tuple[np.ndarray, float]:
        """Get current velocity (linear, angular)."""
        vel_lin, vel_ang = pyb.getBaseVelocity(self.uid)
        return np.array(vel_lin[:2]), vel_ang[2]
    
    def apply_force(self, force: np.ndarray, position: np.ndarray):
        """Apply force at a position on the object."""
        pos_3d = [position[0], position[1], 0.05]
        force_3d = [force[0], force[1], 0]
        pyb.applyExternalForce(
            self.uid, -1, force_3d, pos_3d, pyb.WORLD_FRAME
        )
    
    def apply_wrench(self, wrench: np.ndarray):
        """Apply wrench (Fx, Fy, tau) at center of mass."""
        force = [wrench[0], wrench[1], 0]
        torque = [0, 0, wrench[2]]
        pos, _ = pyb.getBasePositionAndOrientation(self.uid)
        pyb.applyExternalForce(self.uid, -1, force, pos, pyb.WORLD_FRAME)
        pyb.applyExternalTorque(self.uid, -1, torque, pyb.WORLD_FRAME)

