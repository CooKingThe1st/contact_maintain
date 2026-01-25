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
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from object_utils import GenericObject, create_standard_objects, estimate_realistic_mass
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
    
    Uses a simple ear-clipping based approach for basic shapes.
    For complex shapes, falls back to bounding box approximation.
    
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
    
    # Try simple rectangle decomposition for L/T shapes (6-8 vertices)
    if 6 <= n <= 12:
        return _decompose_rectilinear(polygon)
    
    # Fallback: use convex hull (loses concave detail)
    return [polygon.convex_hull]


def _decompose_rectilinear(polygon: Polygon) -> List[Polygon]:
    """Decompose rectilinear polygon (L, T shapes) into rectangles.
    
    Uses axis-aligned bounding box partitioning.
    """
    coords = list(polygon.exterior.coords)[:-1]
    
    # Get bounding box
    minx, miny, maxx, maxy = polygon.bounds
    
    # For L-shape (6 vertices): split into 2 rectangles
    if len(coords) == 6:
        # Find the "corner" of the L
        # The L-shape has one interior corner
        xs = sorted(set(c[0] for c in coords))
        ys = sorted(set(c[1] for c in coords))
        
        if len(xs) == 3 and len(ys) == 3:
            # Typical L-shape with 3 unique x and y values
            # Create two rectangles that cover the L
            rect1 = Polygon([
                (xs[0], ys[0]), (xs[2], ys[0]), (xs[2], ys[1]), (xs[0], ys[1])
            ])
            rect2 = Polygon([
                (xs[0], ys[1]), (xs[1], ys[1]), (xs[1], ys[2]), (xs[0], ys[2])
            ])
            
            # Only keep parts that intersect with original
            parts = []
            for rect in [rect1, rect2]:
                intersection = polygon.intersection(rect)
                if intersection.area > 0.001:
                    if intersection.geom_type == 'Polygon':
                        parts.append(intersection)
            
            if len(parts) >= 2:
                return parts
    
    # For T-shape (8 vertices): split into 2 rectangles
    if len(coords) == 8:
        xs = sorted(set(c[0] for c in coords))
        ys = sorted(set(c[1] for c in coords))
        
        if len(xs) >= 3 and len(ys) >= 3:
            # Try horizontal + vertical split
            mid_y = (miny + maxy) / 2
            
            # Top bar
            top_rect = Polygon([
                (minx, mid_y), (maxx, mid_y), (maxx, maxy), (minx, maxy)
            ])
            # Stem
            center_x = (minx + maxx) / 2
            stem_width = (maxx - minx) / 3
            stem_rect = Polygon([
                (center_x - stem_width, miny),
                (center_x + stem_width, miny),
                (center_x + stem_width, mid_y),
                (center_x - stem_width, mid_y)
            ])
            
            parts = []
            for rect in [top_rect, stem_rect]:
                intersection = polygon.intersection(rect)
                if intersection.area > 0.001:
                    if intersection.geom_type == 'Polygon':
                        parts.append(intersection)
            
            if len(parts) >= 2:
                return parts
    
    # Fallback: return convex hull
    return [polygon.convex_hull]


# ============================================================================
# GENERIC TO PYBULLET CONVERSION
# ============================================================================

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
    if is_convex(polygon):
        return _create_convex_body(
            polygon, physics_props, height, position, orientation, color
        )
    
    if use_compound:
        # Decompose into convex parts
        parts = decompose_to_convex_parts(polygon)
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
    
    Uses box approximation for simplicity and reliability.
    Properly sets mass, inertia, and friction from physics_props.
    
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
    
    # Get bounding box
    bounds = polygon.bounds
    half_extents = [
        (bounds[2] - bounds[0]) / 2,
        (bounds[3] - bounds[1]) / 2,
        height / 2
    ]
    
    # Check if it's roughly circular
    centroid = polygon.centroid
    is_circular = polygon.buffer(0).equals(polygon.convex_hull)
    
    # For circles, use cylinder
    if is_circular and abs(half_extents[0] - half_extents[1]) < 0.01:
        radius = (half_extents[0] + half_extents[1]) / 2
        collision_id = pyb.createCollisionShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            height=height
        )
        visual_id = pyb.createVisualShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color
        )
    else:
        # Use box approximation
        collision_id = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=half_extents
        )
        visual_id = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=half_extents,
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
    # Note: PyBullet automatically computes inertia from shape if not specified
    # We override with our calculated values for consistency with object_utils.py
    pyb.changeDynamics(
        uid, -1,
        mass=mass,
        localInertiaDiagonal=list(inertia_3d),
        lateralFriction=object_friction,
        spinningFriction=0.01,  # Small spinning friction
        rollingFriction=0.01,   # Small rolling friction
        linearDamping=0.0,      # No artificial damping
        angularDamping=0.0      # No artificial damping
    )
    
    return uid


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

