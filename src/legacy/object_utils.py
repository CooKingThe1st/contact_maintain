# %%
import numpy as np
import scipy.io as sio
import math
import time
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import transform, nearest_points, unary_union
from shapely.affinity import rotate, translate
from pathlib import Path
from typing import Union, Optional
import ezdxf
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

import socket
import struct
import pickle
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from itertools import product

# %%
def stream_figure_image(fig, format='png', dpi=150, quality=95, animation_params=None, host='localhost', port=42069):
    """
    🎯 Optimized figure image streaming for v4 server.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to stream
    format : str
        'png' (recommended) or 'jpg'
    dpi : int
        Image resolution (higher = better quality, larger size)
    quality : int
        JPEG quality (1-100, only for JPG format)
    animation_params : dict
        Animation parameters
    """
    if animation_params is None:
        animation_params = {'clear_figure': False}
    
    # Convert figure to image with optimized settings
    img_buffer = io.BytesIO()
    
    if format.lower() == 'png':
        fig.savefig(img_buffer, format='png', 
                   bbox_inches='tight', 
                   dpi=dpi,
                   facecolor='white',
                   edgecolor='none',
                   pad_inches=0.1)
    elif format.lower() in ['jpg', 'jpeg']:
        # 🔧 Fix: Use pil_kwargs for JPEG quality
        fig.savefig(img_buffer, format='jpeg',
                   bbox_inches='tight',
                   dpi=dpi,
                   pil_kwargs={'quality': quality},  # ✅ Correct way to pass quality
                   facecolor='white',
                   edgecolor='none',
                   pad_inches=0.1)
    
    # Encode to base64
    img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Package the data
    viz_data = {
        'type': 'figure_image',
        'image_data': img_data,
        'format': format.lower(),
        'dpi': dpi,
        'animation': animation_params,
        'timestamp': time.time()
    }
    
    # Send to server with optimized socket settings
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)  # 1MB send buffer
        sock.connect((host, port))
        
        serialized_data = pickle.dumps(viz_data)
        sock.sendall(struct.pack('>I', len(serialized_data)))
        sock.sendall(serialized_data)
        sock.close()
        
        return True
        
    except Exception as e:
        print(f"Error sending image to server: {e}")
        return False

# Convenience functions
def stream_figure(fig, **kwargs):
    """Stream a figure with default optimized settings."""
    return stream_figure_image(fig, format='png', dpi=150, **kwargs)

def stream_figure_hq(fig, **kwargs):
    """Stream a figure with high quality settings."""
    return stream_figure_image(fig, format='png', dpi=200, **kwargs)

def stream_figure_fast(fig, **kwargs):
    """Stream a figure with fast/low-bandwidth settings."""
    return stream_figure_image(fig, format='jpg', dpi=100, quality=85, **kwargs)

# %%
# =============================================================================
# STEP 1: GENERIC OBJECT DEFINITION SYSTEM
# =============================================================================
class GenericObject:
    """
    Generic object class that can represent any 2D shape (convex or non-convex)
    using Shapely geometry with physical properties and local frame.
    """
    
    def __init__(self, geometry, mass=3.0, moment_of_inertia=None, 
                 kinetic_friction=0.2, static_friction=0.4, lateral_friction=0.2, 
                 heading=0.0, name="GenericObject"):
        """
        Initialize a generic object.
        
        Args:
            geometry: Shapely geometry (Polygon, Point with buffer, etc.)
            mass: Object mass (kg)
            moment_of_inertia: Moment of inertia (kg⋅m²). If None, calculated automatically
            kinetic_friction: Kinetic friction coefficient with ground
            static_friction: Static friction coefficient with ground
            lateral_friction: Lateral friction coefficient for contact forces
            heading: Object heading/orientation (radians) - defines local frame
            name: Object identifier
        """
        self.geometry = geometry
        self.mass = mass
        self.kinetic_friction = kinetic_friction
        self.static_friction = static_friction
        self.lateral_friction = lateral_friction
        self.heading = heading  # Object's local frame orientation
        self.name = name

        # New friction model (additive; legacy fields above stay for old scripts):
        # - material_friction: single PyBullet lateralFriction on the object (floor pairing)
        # - contact_friction: effective robot–object Coulomb µ (= material × bumper)
        # Unset (_material_friction / _contact_friction = None) → fall back to lateral_friction.
        self._material_friction = None
        self._contact_friction = None
        
        # Calculate moment of inertia if not provided
        if moment_of_inertia is None:
            self.moment_of_inertia = self._calculate_moment_of_inertia()
        else:
            self.moment_of_inertia = moment_of_inertia
        

        # Cache boundary for efficiency
        self._boundary = None
        self._boundary_length = None
        
        # Store reference geometry (unrotated) for local frame calculations
        self.reference_geometry = geometry
        
        # Current pose (for transformed objects)
        self.position = np.array([0.0, 0.0])  # (x, y) position of centroid

    # ------------------------------------------------------------------
    # Friction model (material + effective contact). Legacy kinetic/static/
    # lateral_friction fields are unchanged; revised paths should prefer these.
    # ------------------------------------------------------------------
    @property
    def material_friction(self) -> float:
        """Object material µ for PyBullet body (and ground pairing in revised scenes)."""
        if self._material_friction is not None:
            return float(self._material_friction)
        return float(self.lateral_friction)

    @material_friction.setter
    def material_friction(self, value: float) -> None:
        self.set_material_friction(value)

    def set_material_friction(
        self,
        mu: float,
        *,
        sync_legacy_lateral: bool = True,
        sync_legacy_static: bool = False,
    ) -> None:
        """
        Set object material friction (PyBullet object lateralFriction).

        Parameters
        ----------
        mu : float
            Material coefficient >= 0.
        sync_legacy_lateral : bool
            If True, also set ``lateral_friction`` so older GWS paths see the
            same body µ until they migrate to ``get_contact_friction()``.
        sync_legacy_static : bool
            If True, also set ``static_friction`` (LS/ground scale). Use in
            revised scenes where material µ is the shared floor story.
        """
        mu = float(mu)
        if mu < 0.0:
            raise ValueError(f"material_friction must be >= 0, got {mu}")
        self._material_friction = mu
        if sync_legacy_lateral:
            self.lateral_friction = mu
        if sync_legacy_static:
            self.static_friction = mu

    @property
    def contact_friction(self) -> float:
        """
        Effective robot–object Coulomb µ used by AFC search / friction cone.

        Prefer the value set by ``set_contact_friction`` (product model).
        Falls back to ``lateral_friction`` for legacy callers.
        """
        return self.get_contact_friction()

    @contact_friction.setter
    def contact_friction(self, value: float) -> None:
        self.set_contact_friction(value)

    def get_contact_friction(self) -> float:
        if self._contact_friction is not None:
            return float(self._contact_friction)
        return float(self.lateral_friction)

    def set_contact_friction(self, mu: float, *, sync_legacy_lateral: bool = False) -> None:
        """
        Set effective contact µ for search (typically material × bumper).

        Does not change ``material_friction``. Optionally mirrors into
        ``lateral_friction`` so code that only reads that field stays aligned.
        """
        mu = float(mu)
        if mu < 0.0:
            raise ValueError(f"contact_friction must be >= 0, got {mu}")
        self._contact_friction = mu
        if sync_legacy_lateral:
            self.lateral_friction = mu

    def effective_contact_friction(self, bumper_mu: float) -> float:
        """Product model: µ_contact = material_friction × bumper_mu."""
        return float(self.material_friction) * float(bumper_mu)

    def apply_bumper_contact_model(
        self,
        bumper_mu: float,
        *,
        sync_search_to_legacy_lateral: bool = False,
    ) -> float:
        """
        Set contact_friction from material × bumper and return µ_contact.

        Call after ``set_material_friction`` and after choosing bumper µ.
        Does **not** overwrite ``lateral_friction`` by default (that remains the
        object body / material µ for PyBullet); GWS should use
        ``get_contact_friction()``.
        """
        mu_c = self.effective_contact_friction(bumper_mu)
        self.set_contact_friction(
            mu_c, sync_legacy_lateral=sync_search_to_legacy_lateral
        )
        return mu_c

    def _calculate_moment_of_inertia(self):
        """Calculate moment of inertia for the geometry (approximation)."""
        # For complex shapes, use bounding box approximation
        bounds = self.geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        # Approximate as rectangle: I = m*(w²+h²)/12
        return self.mass * (width**2 + height**2) / 12
    
    @property
    def boundary(self):
        """Get the boundary of the object as a LineString."""
        if self._boundary is None:
            self._boundary = LineString(self.geometry.exterior.coords)
        return self._boundary
    
    @property
    def boundary_length(self):
        """Get the total length of the boundary."""
        if self._boundary_length is None:
            self._boundary_length = self.boundary.length
        return self._boundary_length
    
    def get_centroid(self):
        """Get the centroid of the object."""
        return self.geometry.centroid
    
    def get_area(self):
        """Get the area of the object."""
        return self.geometry.area
    
    def get_local_frame_axes(self):
        """
        Get the local frame axes based on object heading.
        
        Returns:
            tuple: (x_axis, y_axis) - unit vectors for local frame
        """
        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)
        
        x_axis = np.array([cos_h, sin_h])  # Forward direction
        y_axis = np.array([-sin_h, cos_h])  # Left direction (90° CCW from forward)
        
        return x_axis, y_axis
    
    def world_to_local(self, world_point):
        """
        Convert a point from world coordinates to local object frame.
        
        Args:
            world_point: (x, y) coordinates in world frame
            
        Returns:
            numpy.ndarray: Point in local object frame
        """
        # Translate to object center
        centroid = self.get_centroid()
        translated = np.array(world_point) - np.array([centroid.x, centroid.y])
        
        # Rotate to local frame
        cos_h = np.cos(-self.heading)  # Negative for inverse rotation
        sin_h = np.sin(-self.heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        return rotation_matrix @ translated
    
    def local_to_world(self, local_point):
        """
        Convert a point from local object frame to world coordinates.
        
        Args:
            local_point: (x, y) coordinates in local frame
            
        Returns:
            numpy.ndarray: Point in world frame
        """
        # Rotate to world frame
        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        rotated = rotation_matrix @ np.array(local_point)
        
        # Translate from object center
        centroid = self.get_centroid()
        return rotated + np.array([centroid.x, centroid.y])
    def transform(self, x, y, theta):
        """
        Transform the object to a new pose.
        
        Args:
            x, y: Translation for the centroid (or CoM)
            theta: Rotation angle (radians) - updates heading
            
        Returns:
            New GenericObject with transformed geometry and updated heading
        """
        # Get current centroid
        centroid = self.get_centroid()
        centroid_pos = (centroid.x, centroid.y)
        
        # Rotate around the centroid, then translate so centroid is at (x, y)
        rotated_geom = rotate(self.geometry, theta, origin=centroid_pos, use_radians=True)
        
        # Calculate the new centroid position after rotation
        rotated_centroid = rotate(Point(centroid_pos), theta, origin=centroid_pos, use_radians=True)
        
        # Translate so the centroid ends up at (x, y)
        dx = x - rotated_centroid.x
        dy = y - rotated_centroid.y
        transformed_geom = translate(rotated_geom, xoff=dx, yoff=dy)
        
        # Update heading (cumulative rotation)
        new_heading = self.heading + theta
        
        new_obj = GenericObject(
            geometry=transformed_geom,
            mass=self.mass,
            moment_of_inertia=self.moment_of_inertia,
            kinetic_friction=self.kinetic_friction,
            static_friction=self.static_friction,
            lateral_friction=self.lateral_friction,
            heading=new_heading,
            name=self.name
        )
        new_obj._material_friction = self._material_friction
        new_obj._contact_friction = self._contact_friction
        
        # Update position to the target location
        new_obj.position = np.array([x, y])
        
        return new_obj
        
    def visualize(self, ax=None, show_frame=True, **kwargs):
        """Visualize the object with optional local frame axes."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        # Default styling
        style = {'facecolor': 'lightblue', 'edgecolor': 'blue', 'alpha': 0.7}
        style.update(kwargs)
        
        # Plot the geometry
        x, y = self.geometry.exterior.xy
        ax.fill(x, y, **style)
        
        # Mark centroid
        centroid = self.get_centroid()
        ax.plot(centroid.x, centroid.y, 'ro', markersize=8, label='Centroid')
        
        # Show local frame axes
        if show_frame:
            x_axis, y_axis = self.get_local_frame_axes()
            scale = 0.1  # Axis length
            
            # X-axis (forward, blue)
            ax.arrow(centroid.x, centroid.y, 
                    x_axis[0] * scale, x_axis[1] * scale,
                    head_width=0.02, head_length=0.02, 
                    fc='blue', ec='blue', linewidth=2,
                    alpha=0.2,
                    label='X-axis')
            
            # Y-axis (left, green)  
            ax.arrow(centroid.x, centroid.y,
                    y_axis[0] * scale, y_axis[1] * scale,
                    head_width=0.02, head_length=0.02,
                    fc='green', ec='green', linewidth=2,
                    alpha=0.2,
                    label='Y-axis')
        
        ax.set_aspect('equal')
        ax.grid(True)
        # ax.legend()
        ax.set_title(f"Object: {self.name} (heading: {np.degrees(self.heading):.1f}°)")
        
        return ax


def estimate_realistic_mass(geometry, material_density=300):
    """
    Estimate a realistic mass for an object based on its geometry.
    
    Args:
        geometry: Shapely geometry of the object
        material_density: Material density in kg/m³ (default: 300 - low density wood)
        
    Returns:
        float: Estimated mass in kg
    """
    # Get dimensions
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]  # meters
    height = bounds[3] - bounds[1]  # meters
    
    # Get area from geometry
    area = geometry.area  # square meters
    
    # Estimate a reasonable thickness as 1/3 of the sum of width and height
    thickness = (width + height) / 8
    
    # Calculate volume (area * thickness)
    volume = area * thickness  # cubic meters
    
    # Calculate mass (density * volume)
    mass = material_density * volume  # kg
    
    return mass

def create_standard_objects():
    """Factory function to create common object shapes with headings.
    
    Object sizes are designed to work well with small robots (radius ~0.06m).
    Most objects are in the 0.3-0.6m range, making them about 5-10x the robot size.
    
    Scale factor applied: ~2.5x from original sizes.
    """
    objects = {}
    
    # Scale factor for objects (2.5x larger than original)
    S = 2.5
    
    # Rectangle/Box (0.875m x 0.625m)
    box_vertices = [(-0.35*S, -0.25*S), (0.35*S, -0.25*S), (0.35*S, 0.25*S), (-0.35*S, 0.25*S)]
    rect_geom = Polygon(box_vertices)
    rect_mass = estimate_realistic_mass(rect_geom)
    objects['rectangle'] = GenericObject(
        geometry=rect_geom,
        mass=rect_mass,
        lateral_friction=0.3,
        heading=0.0,  # No initial rotation
        name="Rectangle"
    )
    
    # Circle (radius 0.5m)
    circle_geom = Point(0, 0).buffer(0.4 * S)
    circle_mass = estimate_realistic_mass(circle_geom)
    objects['circle'] = GenericObject(
        geometry=circle_geom,
        mass=circle_mass,
        lateral_friction=0.4,
        heading=0.0,
        name="True_Circle"
    )
    

    # Ellipse (stretched circle: 0.75m x 0.375m)
    ellipse_geom = Point(0, 0).buffer(1.0)
    # Scale to create ellipse
    ellipse_geom = translate(rotate(ellipse_geom, 0), xoff=0, yoff=0)
    from shapely.affinity import scale
    ellipse_geom = scale(ellipse_geom, xfact=0.3*S, yfact=0.15*S, origin=(0, 0))
    ellipse_mass = estimate_realistic_mass(ellipse_geom)
    objects['ellipse'] = GenericObject(
        geometry=ellipse_geom,
        mass=ellipse_mass,
        lateral_friction=0.35,
        heading=0.0,
        name="Ellipse"
    )

    # Triangle (scaled up)
    triangle_vertices = [(0, 0.25*S), (-0.22*S, -0.125*S), (0.22*S, -0.125*S)]
    triangle_geom = Polygon(triangle_vertices)
    triangle_mass = estimate_realistic_mass(triangle_geom)
    objects['triangle'] = GenericObject(
        geometry=triangle_geom,
        mass=triangle_mass,
        lateral_friction=0.35,
        heading=0.0,
        name="Triangle"
    )
    
    # Fat Triangle (non-symmetric, wide base)
    fat_triangle_vertices = [(0.05*S, 0.22*S), (-0.28*S, -0.15*S), (0.25*S, -0.15*S)]
    fat_triangle_geom = Polygon(fat_triangle_vertices)
    fat_triangle_mass = estimate_realistic_mass(fat_triangle_geom)
    objects['fat_triangle'] = GenericObject(
        geometry=fat_triangle_geom,
        mass=fat_triangle_mass,
        lateral_friction=0.33,
        heading=0.0,
        name="Fat Triangle"
    )

    # Narrow Triangle (symmetric, narrow base)
    narrow_triangle_vertices = [(0*S, 0.9*S), (-0.01*S, -0.9*S), (0.01*S, -0.9*S)]
    narrow_triangle_geom = Polygon(narrow_triangle_vertices)
    narrow_triangle_mass = estimate_realistic_mass(narrow_triangle_geom)
    objects['narrow_triangle'] = GenericObject(
        geometry=narrow_triangle_geom,
        mass=narrow_triangle_mass,
        lateral_friction=0.33,
        heading=0.0,
        name="Narrow Triangle"
    )

    # Obese Triangle (symmetric, wide base)
    obese_triangle_vertices = [(0*S, 0.05*S), (-0.9*S, -0.05*S), (0.9*S, -0.05*S)]
    obese_triangle_geom = Polygon(obese_triangle_vertices)
    obese_triangle_mass = estimate_realistic_mass(obese_triangle_geom)
    objects['obese_triangle'] = GenericObject(
        geometry=obese_triangle_geom,
        mass=obese_triangle_mass,
        lateral_friction=0.33,
        heading=0.0,
        name="Obese Triangle"
    )
    
    # Scalene Triangle (all sides different, non-symmetric)
    scalene_vertices = [(0.1*S, 0.25*S), (-0.25*S, -0.1*S), (0.2*S, -0.18*S)]
    scalene_geom = Polygon(scalene_vertices)
    scalene_mass = estimate_realistic_mass(scalene_geom)
    objects['scalene'] = GenericObject(
        geometry=scalene_geom,
        mass=scalene_mass,
        lateral_friction=0.34,
        heading=0.0,
        name="Scalene Triangle"
    )

    # L-shape (non-convex example)
    l_vertices = [
        (0, 0), (0.3*S, 0), (0.3*S, 0.1*S), (0.1*S, 0.1*S), 
        (0.1*S, 0.3*S), (0, 0.3*S), (0, 0)
    ]
    l_shape_geom = Polygon(l_vertices)
    l_shape_mass = estimate_realistic_mass(l_shape_geom)
    objects['l_shape'] = GenericObject(
        geometry=l_shape_geom,
        mass=l_shape_mass,
        lateral_friction=0.25,
        heading=0.0,
        name="L-Shape"
    )
    
    # Asymmetric L-shape (different arm lengths)
    asym_l_vertices = [
        (0, 0), (0.28*S, 0), (0.28*S, 0.12*S), (0.12*S, 0.12*S), 
        (0.12*S, 0.22*S), (0, 0.22*S), (0, 0)
    ]
    asym_l_geom = Polygon(asym_l_vertices)
    asym_l_mass = estimate_realistic_mass(asym_l_geom)
    objects['asym_l_shape'] = GenericObject(
        geometry=asym_l_geom,
        mass=asym_l_mass,
        lateral_friction=0.26,
        heading=0.0,
        name="Asymmetric L-Shape"
    )


    # Star shape (non-convex)
    angles = np.linspace(0, 2*np.pi, 11)  # 10 points + closing
    star_vertices = []
    for i in range(10):
        radius = (0.2 if i % 2 == 0 else 0.1) * S  # Alternate between outer and inner radius
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        star_vertices.append((x, y))
    
    star_geom = Polygon(star_vertices)
    star_mass = estimate_realistic_mass(star_geom)
    objects['star'] = GenericObject(
        geometry=star_geom,
        mass=star_mass,
        lateral_friction=0.3,
        heading=0.0,
        name="Star"
    )


    # T-shape (non-convex, symmetric)
    t_vertices = [
        (-0.25*S, 0.15*S), (0.25*S, 0.15*S), (0.25*S, 0.05*S),
        (0.08*S, 0.05*S), (0.08*S, -0.2*S), (-0.08*S, -0.2*S),
        (-0.08*S, 0.05*S), (-0.25*S, 0.05*S)
    ]
    t_shape_geom = Polygon(t_vertices)
    t_shape_mass = estimate_realistic_mass(t_shape_geom)
    objects['t_shape'] = GenericObject(
        geometry=t_shape_geom,
        mass=t_shape_mass,
        lateral_friction=0.28,
        heading=0.0,
        name="T-Shape"
    )
    
    # U-shape (non-convex)
    u_vertices = [
        (0, 0), (0.25*S, 0), (0.25*S, 0.3*S), (0.18*S, 0.3*S),
        (0.18*S, 0.07*S), (0.07*S, 0.07*S), (0.07*S, 0.3*S),
        (0, 0.3*S)
    ]
    u_shape_geom = Polygon(u_vertices)
    u_shape_mass = estimate_realistic_mass(u_shape_geom)
    objects['u_shape'] = GenericObject(
        geometry=u_shape_geom,
        mass=u_shape_mass,
        lateral_friction=0.27,
        heading=0.0,
        name="U-Shape"
    )
    
    # Star shape (non-convex) - duplicate removed, using the one above
    # (The original had duplicate star shape code)
    
    # Crescent (non-convex, non-symmetric)
    # Create by subtracting a smaller circle from a larger one
    large_circle = Point(0, 0).buffer(0.22 * S)
    small_circle = Point(0.08 * S, 0).buffer(0.18 * S)
    crescent_geom = large_circle.difference(small_circle)
    crescent_mass = estimate_realistic_mass(crescent_geom)
    objects['crescent'] = GenericObject(
        geometry=crescent_geom,
        mass=crescent_mass,
        lateral_friction=0.32,
        heading=0.0,
        name="Crescent"
    )


    # Create by subtracting two offset circles to create an asymmetric crescent
    large_circle = Point(0, 0).buffer(0.22 * S)
    small_circle = Point(0.09 * S, 0.08 * S).buffer(0.18 * S)  # Offset both x and y for asymmetry
    crescent_asym_geom = large_circle.difference(small_circle)

    # Ensure it's a single Polygon
    from shapely.ops import unary_union
    if crescent_asym_geom.geom_type == 'MultiPolygon':
        crescent_asym_geom = crescent_asym_geom.buffer(0.001).buffer(-0.001)

    crescent_asym_mass = estimate_realistic_mass(crescent_asym_geom)
    objects['crescent_asym'] = GenericObject(
        geometry=crescent_asym_geom,
        mass=crescent_asym_mass,
        lateral_friction=0.36,
        heading=0.0,
        name="Asymmetric Crescent"
    )   
    
    # Asymmetric Pentagon (non-symmetric)
    pentagon_vertices = [
        (0, 0.25*S), (0.2*S, 0.1*S), (0.15*S, -0.15*S), 
        (-0.1*S, -0.2*S), (-0.2*S, 0.05*S)
    ]
    pentagon_geom = Polygon(pentagon_vertices)
    pentagon_mass = estimate_realistic_mass(pentagon_geom)
    objects['pentagon_asym'] = GenericObject(
        geometry=pentagon_geom,
        mass=pentagon_mass,
        lateral_friction=0.31,
        heading=0.0,
        name="Asymmetric Pentagon"
    )
    
    # Trapezoid (non-symmetric orientation)
    trapezoid_vertices = [
        (-0.2*S, -0.1*S), (0.25*S, -0.1*S), (0.18*S, 0.15*S), (-0.15*S, 0.15*S)
    ]
    trapezoid_geom = Polygon(trapezoid_vertices)
    trapezoid_mass = estimate_realistic_mass(trapezoid_geom)
    objects['trapezoid'] = GenericObject(
        geometry=trapezoid_geom,
        mass=trapezoid_mass,
        lateral_friction=0.29,
        heading=0.0,
        name="Trapezoid"
    )
    
    # Asymmetric Quadrilateral (irregular 4-sided, non-symmetric)
    quad_vertices = [
        (-0.18*S, -0.12*S), (0.22*S, -0.08*S), (0.15*S, 0.18*S), (-0.12*S, 0.2*S)
    ]
    quad_geom = Polygon(quad_vertices)
    quad_mass = estimate_realistic_mass(quad_geom)
    objects['asym_quad'] = GenericObject(
        geometry=quad_geom,
        mass=quad_mass,
        lateral_friction=0.3,
        heading=0.0,
        name="Asymmetric Quadrilateral"
    )
    
    # Plus/Cross shape (non-convex, symmetric)
    plus_vertices = [
        (-0.07*S, 0.2*S), (0.07*S, 0.2*S), (0.07*S, 0.07*S),
        (0.2*S, 0.07*S), (0.2*S, -0.07*S), (0.07*S, -0.07*S),
        (0.07*S, -0.2*S), (-0.07*S, -0.2*S), (-0.07*S, -0.07*S),
        (-0.2*S, -0.07*S), (-0.2*S, 0.07*S), (-0.07*S, 0.07*S)
    ]
    plus_geom = Polygon(plus_vertices)
    plus_mass = estimate_realistic_mass(plus_geom)
    objects['plus'] = GenericObject(
        geometry=plus_geom,
        mass=plus_mass,
        lateral_friction=0.26,
        heading=0.0,
        name="Plus"
    )
    
    # Arrow (non-symmetric)
    arrow_vertices = [
        (0, 0.25*S), (0.15*S, 0.1*S), (0.08*S, 0.1*S),
        (0.08*S, -0.2*S), (-0.08*S, -0.2*S), (-0.08*S, 0.1*S),
        (-0.15*S, 0.1*S)
    ]
    arrow_geom = Polygon(arrow_vertices)
    arrow_mass = estimate_realistic_mass(arrow_geom)
    objects['arrow'] = GenericObject(
        geometry=arrow_geom,
        mass=arrow_mass,
        lateral_friction=0.3,
        heading=0.0,
        name="Arrow"
    )
    
    # # Comma/Hook shape (non-convex, non-symmetric) - FIXED
    # head_circle = Point(0, 0.15*S).buffer(0.12*S)
    # tail_vertices = [(-0.05*S, 0.05*S), (0.05*S, 0.05*S), (0.08*S, -0.18*S), (-0.02*S, -0.18*S)]
    # tail_poly = Polygon(tail_vertices)
    # hook_geom = head_circle.union(tail_poly)
    
    # # Ensure single Polygon
    # if hook_geom.geom_type == 'MultiPolygon':
    #     hook_geom = hook_geom.buffer(0.001).buffer(-0.001)
    
    # hook_mass = estimate_realistic_mass(hook_geom)
    # objects['hook'] = GenericObject(
    #     geometry=hook_geom,
    #     mass=hook_mass,
    #     lateral_friction=0.31,
    #     heading=0.0,
    #     name="Hook"
    # )
    
    
    # Wedge (right triangle, non-symmetric)
    wedge_vertices = [(0, 0), (0.3*S, 0), (0, 0.25*S)]
    wedge_geom = Polygon(wedge_vertices)
    wedge_mass = estimate_realistic_mass(wedge_geom)
    objects['wedge'] = GenericObject(
        geometry=wedge_geom,
        mass=wedge_mass,
        lateral_friction=0.32,
        heading=0.0,
        name="Wedge"
    )
    
    # Boot/Shoe shape (non-convex, non-symmetric)
    boot_vertices = [
        (0, 0), (0.25*S, 0), (0.25*S, 0.08*S), (0.15*S, 0.08*S),
        (0.15*S, 0.2*S), (0.08*S, 0.2*S), (0.08*S, 0.12*S), (0, 0.12*S)
    ]
    boot_geom = Polygon(boot_vertices)
    boot_mass = estimate_realistic_mass(boot_geom)
    objects['boot'] = GenericObject(
        geometry=boot_geom,
        mass=boot_mass,
        lateral_friction=0.28,
        heading=0.0,
        name="Boot"
    )
    
    return objects


def get_reachable_contact_points(
    geometry: Polygon,
    robot_radius: float,
    n_samples: int = 512,
):
    """
    Compute reachable contact points on the boundary of a 2D object for a
    circular robot of radius `robot_radius` starting from infinity.

    This uses a configuration-space formulation:
    - Buffer the object geometry by `robot_radius` to obtain the C-obstacle
      O ⊕ B(0, R).
    - Take only the exterior ring of the buffered polygon; this corresponds
      to the locus of robot centers that just touch the object from the
      unbounded component of free space.
    - Project these center points back to the original object boundary.

    Args:
        geometry: Shapely Polygon representing the object (workspace obstacle).
        robot_radius: Radius of the circular robot.
        n_samples: Number of samples along the reachable locus to return.

    Returns:
        np.ndarray of shape (N, 2): reachable contact points in world
        coordinates, ordered along the reachable locus.
    """
    if not isinstance(geometry, Polygon):
        raise TypeError(
            f"get_reachable_contact_points expects a Polygon, got {type(geometry)}"
        )
    if robot_radius <= 0:
        raise ValueError("robot_radius must be positive")

    # 1. Build configuration-space obstacle via Minkowski sum with a disk
    #    using a round buffer.
    buffered = geometry.buffer(distance=robot_radius, join_style=1)  # 1 == ROUND

    # Handle potential MultiPolygon from buffering non-convex shapes.
    if buffered.geom_type == "MultiPolygon":
        # Take largest component by area as the main C-obstacle.
        buffered = max(buffered.geoms, key=lambda g: g.area)
    elif buffered.geom_type != "Polygon":
        raise RuntimeError(
            f"Unsupported buffered geometry type: {buffered.geom_type}"
        )

    # 2. Extract the exterior ring of the buffered polygon: this is the path
    #    of robot centers that touch the object from the outside.
    center_path = buffered.exterior

    # 3. Parameterize the center path and map each point back to the original
    #    object boundary by closest-point projection.
    path_length = center_path.length
    if path_length <= 0 or n_samples <= 0:
        return np.zeros((0, 2), dtype=float)

    ts = np.linspace(0.0, path_length, int(n_samples), endpoint=False)
    boundary = LineString(geometry.exterior.coords)

    contact_points = []
    for t in ts:
        c = center_path.interpolate(t)
        # Use boundary.project + interpolate, which is more efficient than
        # nearest_points for many samples.
        s = boundary.project(c)
        contact = boundary.interpolate(s)
        contact_points.append((contact.x, contact.y))

    return np.asarray(contact_points, dtype=float)


def get_reachable_contact_intervals(
    geometry: Polygon,
    robot_radius: float,
    n_samples: int = 2048,
    gap_factor: float = 3.0,
):
    """
    Compute reachable intervals in boundary parameter space t ∈ [0, 1].

    This uses the same C-space buffering approach as get_reachable_contact_points,
    but returns contiguous intervals of boundary parameters where contact is
    reachable by a circular robot of radius `robot_radius` starting from infinity.

    Args:
        geometry: Shapely Polygon representing the object (workspace obstacle).
        robot_radius: Radius of the circular robot.
        n_samples: Number of samples along the reachable locus for interval
            estimation (higher = finer resolution).
        gap_factor: Multiplier on nominal parameter step to detect gaps between
            reachable regions. Larger values are more conservative.

    Returns:
        List of (t_start, t_end) tuples with 0 ≤ t_start ≤ t_end ≤ 1, covering
        the approximate reachable subset of the boundary parameterization.
    """
    # Sample reachable contact points along the boundary
    reachable_pts = get_reachable_contact_points(
        geometry, robot_radius, n_samples=n_samples
    )

    boundary = LineString(geometry.exterior.coords)
    boundary_length = boundary.length

    if reachable_pts.size == 0 or boundary_length <= 0.0:
        return []

    # Map each reachable point back to a boundary parameter t ∈ [0, 1]
    ts = []
    for x, y in reachable_pts:
        s = boundary.project(Point(x, y))  # arc-length along boundary
        t = s / boundary_length
        ts.append(t)

    ts = np.sort(np.asarray(ts, dtype=float))

    # Remove near-duplicates
    if ts.size == 0:
        return []
    unique_ts = [ts[0]]
    for val in ts[1:]:
        if abs(val - unique_ts[-1]) > 1e-6:
            unique_ts.append(val)
    ts = np.asarray(unique_ts, dtype=float)

    if ts.size == 0:
        return []

    # Nominal parameter step assuming uniform coverage
    nominal_step = 1.0 / max(ts.size, 1)
    gap_threshold = gap_factor * nominal_step

    intervals = []
    start_t = ts[0]
    prev_t = ts[0]

    for t in ts[1:]:
        if t - prev_t > gap_threshold:
            # Close previous interval
            intervals.append((start_t, prev_t))
            start_t = t
        prev_t = t

    # Close final interval
    intervals.append((start_t, prev_t))

    return intervals


def dxf_to_generic(
    dxf_path: Union[str, Path],
    name: Optional[str] = None,
    mass: Optional[float] = None,
    lateral_friction: float = 0.8,
    heading: float = 0.0,
    reverse_sign: bool = True,
) -> GenericObject:
    """
    Create a GenericObject from a DXF file containing a closed LWPOLYLINE.

    ⚠️  **DEPRECATED / BUGGY**: This function has a confirmed bug where it extracts
    vertices with incorrect signs: all x and y coordinates are negated (i.e., vertices
    are in (-x, -y) format instead of (x, y)). This was verified by comparing with
    Fusion 360's actual geometry.
    
    **DO NOT USE**: Use `read_obj_to_vertices` instead, which extracts 2D vertices
    directly from OBJ files and gives correct coordinates. DXF files are no longer
    needed in the pipeline.

    This is the new geometry pipeline for complex shapes (bolt, pi, root, etc.):
    we extract the 2D boundary directly from the DXF instead of hardcoding
    vertices or using an intermediate JSON file.
    
    **Bug Fix Required**: To fix this function, reverse the sign of all x and y
    values: `vertices = [(-x, -y) for x, y in vertices]` should become
    `vertices = [(x, y) for x, y in vertices]` (or negate the extracted values).

    Parameters
    ----------
    dxf_path : str | Path
        Path to the DXF file.
    name : str, optional
        Name of the object. If None, uses the stem of the DXF file.
    mass : float, optional
        If provided, use this mass. Otherwise, estimate a reasonable mass
        from the polygon area via estimate_realistic_mass.
    lateral_friction : float
        Lateral friction coefficient for the GenericObject.
    heading : float
        Initial heading (rotation) of the object in radians.
    reverse_sign : bool
        If True, reverse the sign of all x and y coordinates (default: True).
        This fixes the known bug where DXF extraction gives (-x, -y) instead of (x, y).

    Returns
    -------
    GenericObject
        The constructed object ready for use with ContactPointParameterization.
    """
    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    vertices = []

    # Extract the first non-empty LWPOLYLINE as the outer boundary
    for e in msp:
        if e.dxftype() == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            # DXF LWPOLYLINE may repeat the first point at the end; drop duplicate
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) >= 3:
                vertices = pts
                break

    if not vertices:
        raise ValueError(f"No valid LWPOLYLINE with >=3 points found in DXF: {dxf_path}")

    # Reverse sign of x and y if requested (to fix the known bug)
    if reverse_sign:
        vertices = [(-x, -y) for x, y in vertices]

    print(f"Vertices: {vertices}")
    geometry = Polygon(vertices)
    if not geometry.is_valid or geometry.area <= 0:
        raise ValueError(f"DXF polygon is invalid or has zero area: {dxf_path}")

    if mass is None:
        obj_mass = estimate_realistic_mass(geometry)
    else:
        obj_mass = mass

    obj_name = name if name is not None else dxf_path.stem

    generic_obj = GenericObject(
        geometry=geometry,
        mass=obj_mass,
        lateral_friction=lateral_friction,
        heading=heading,
        name=obj_name,
    )

    return generic_obj


def _vertices_from_path2d_entities(path2d) -> list:
    """Extract ordered closed polygons from each entity in a trimesh Path2D."""
    polys = []
    verts = path2d.vertices
    for ent in path2d.entities:
        if not hasattr(ent, "points") or len(ent.points) < 3:
            continue
        coords = [(float(verts[i][0]), float(verts[i][1])) for i in ent.points]
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if poly.is_valid and poly.area > 1e-8:
            polys.append(poly)
    return polys


def _polygon_to_vertex_list(poly: Polygon) -> list:
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if len(coords) < 3:
        raise ValueError("Polygon has fewer than 3 vertices")
    return [(float(x), float(y)) for x, y in coords]


def _read_dxf_vertices(dxf_path: Union[str, Path], reverse_sign: bool = True) -> list:
    """Read the outer LWPOLYLINE boundary from a sibling DXF footprint file."""
    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    vertices = []
    for e in msp:
        if e.dxftype() == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) >= 3:
                vertices = pts
                break

    if not vertices:
        raise ValueError(f"No valid LWPOLYLINE with >=3 points found in DXF: {dxf_path}")

    if reverse_sign:
        vertices = [(-x, -y) for x, y in vertices]

    geometry = Polygon(vertices)
    if not geometry.is_valid or geometry.area <= 0:
        raise ValueError(f"DXF polygon is invalid or has zero area: {dxf_path}")
    return vertices


def _mesh_xy_convex_hull_vertices(mesh) -> list:
    xy = mesh.vertices[:, :2]
    if len(xy) < 3:
        raise ValueError("Mesh has fewer than 3 vertices for convex hull")
    hull = Polygon(xy).convex_hull
    if hull.is_empty or hull.area <= 0:
        raise ValueError("Convex hull of mesh XY projection is degenerate")
    return _polygon_to_vertex_list(hull)


def read_obj_to_vertices(
    obj_path: Union[str, Path],
) -> list:
    """
    Extract 2D footprint vertices from an OBJ file.

    Uses a bottom cross-section when it yields a single valid polygon. For meshes
    whose slice is split (e.g. hourglass waist above the bottom plane) or whose
    section vertex order is ambiguous, falls back to a sibling ``.dxf`` file, then
    to the XY convex hull of the mesh.
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError(
            "trimesh is required for read_obj_to_vertices. "
            "Install it with: pip install trimesh"
        )

    obj_path = Path(obj_path)
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    try:
        mesh = trimesh.load(str(obj_path))
    except Exception as e:
        raise ValueError(f"Failed to load OBJ file {obj_path}: {e}")

    z_min = mesh.bounds[0][2]
    try:
        section = mesh.section(
            plane_origin=[0, 0, z_min],
            plane_normal=[0, 0, 1],
        )
    except Exception as e:
        section = None
        slice_err = e
    else:
        slice_err = None

    if section is not None:
        try:
            path2d, _ = section.to_planar()
            entity_polys = _vertices_from_path2d_entities(path2d)
            if len(entity_polys) == 1:
                return _polygon_to_vertex_list(entity_polys[0])
            if len(entity_polys) > 1:
                dxf_path = obj_path.with_suffix(".dxf")
                if dxf_path.exists():
                    return _read_dxf_vertices(dxf_path)
                merged = unary_union(entity_polys)
                if merged.geom_type == "Polygon" and merged.is_valid and merged.area > 1e-8:
                    return _polygon_to_vertex_list(merged)
                if merged.geom_type == "MultiPolygon":
                    largest = max(merged.geoms, key=lambda g: g.area)
                    if largest.is_valid and largest.area > 1e-8:
                        return _polygon_to_vertex_list(largest)
        except Exception:
            pass

    dxf_path = obj_path.with_suffix(".dxf")
    if dxf_path.exists():
        try:
            return _read_dxf_vertices(dxf_path)
        except Exception:
            pass

    if slice_err is not None:
        raise ValueError(f"Failed to create cross-section at z={z_min}: {slice_err}")

    return _mesh_xy_convex_hull_vertices(mesh)


def create_pybullet_objects():
    """Create a focused set of 9 object shapes for PyBullet testing.
    
    Creates only the essential shapes that work well with object_bridge.py:
    - Rectangle and Square (variations)
    - Right Triangle, Scalene Triangle, Equilateral Triangle (variations)
    - L-shape (symmetric and asymmetric variations)
    - T-shape (symmetric and asymmetric variations)
    
    Object sizes are designed to work well with small robots (radius ~0.06m).
    Most objects are in the 0.3-0.6m range, making them about 5-10x the robot size.
    
    Scale factor applied: ~2.5x from original sizes.
    
    Returns
    -------
    dict
        Dictionary mapping shape names to GenericObject instances.
    """
    objects = {}
    
    # Scale factor for objects (2.5x larger than original)
    S = 2.5
    
    # 1. RECTANGLE (0.875m x 0.625m)
    box_vertices = [(-0.35*S, -0.25*S), (0.35*S, -0.25*S), (0.35*S, 0.25*S), (-0.35*S, 0.25*S)]
    rect_geom = Polygon(box_vertices)
    rect_mass = estimate_realistic_mass(rect_geom)
    objects['rectangle'] = GenericObject(
        geometry=rect_geom,
        mass=rect_mass,
        lateral_friction=0.3,
        heading=0.0,
        name="Rectangle"
    )
    
    # 2. SQUARE (variation of rectangle, 0.625m x 0.625m)
    square_size = 0.25 * S
    square_vertices = [
        (-square_size, -square_size),
        (square_size, -square_size),
        (square_size, square_size),
        (-square_size, square_size)
    ]
    square_geom = Polygon(square_vertices)
    square_mass = estimate_realistic_mass(square_geom)
    objects['square'] = GenericObject(
        geometry=square_geom,
        mass=square_mass,
        lateral_friction=0.3,
        heading=0.0,
        name="Square"
    )
    
    # 3. RIGHT TRIANGLE (3-4-5 triangle)
    right_triangle_vertices = [
        (0.0, 0.0),           # Right angle at origin
        (0.4*S, 0.0),         # Base (4 units)
        (0.0, 0.3*S)          # Height (3 units)
    ]
    right_triangle_geom = Polygon(right_triangle_vertices)
    right_triangle_mass = estimate_realistic_mass(right_triangle_geom)
    objects['right_triangle'] = GenericObject(
        geometry=right_triangle_geom,
        mass=right_triangle_mass,
        lateral_friction=0.35,
        heading=0.0,
        name="Right Triangle"
    )
    
    # 4. SCALENE TRIANGLE (all sides different)
    scalene_vertices = [
        (0.1*S, 0.25*S),      # Top vertex
        (-0.25*S, -0.1*S),    # Bottom left
        (0.2*S, -0.18*S)      # Bottom right
    ]
    scalene_geom = Polygon(scalene_vertices)
    scalene_mass = estimate_realistic_mass(scalene_geom)
    objects['scalene_triangle'] = GenericObject(
        geometry=scalene_geom,
        mass=scalene_mass,
        lateral_friction=0.34,
        heading=0.0,
        name="Scalene Triangle"
    )
    
    # 5. EQUILATERAL TRIANGLE
    equilateral_size = 0.35 * S
    equilateral_vertices = [
        (0.0, equilateral_size * np.sqrt(3) / 3),                    # Top vertex
        (-equilateral_size / 2, -equilateral_size * np.sqrt(3) / 6),   # Bottom left
        (equilateral_size / 2, -equilateral_size * np.sqrt(3) / 6)    # Bottom right
    ]
    equilateral_geom = Polygon(equilateral_vertices)
    equilateral_mass = estimate_realistic_mass(equilateral_geom)
    objects['equilateral_triangle'] = GenericObject(
        geometry=equilateral_geom,
        mass=equilateral_mass,
        lateral_friction=0.35,
        heading=0.0,
        name="Equilateral Triangle"
    )
    
    # 6. L-SHAPE (symmetric)
    l_vertices = [
        (0, 0), (0.3*S, 0), (0.3*S, 0.1*S), (0.1*S, 0.1*S), 
        (0.1*S, 0.3*S), (0, 0.3*S)
    ]
    l_shape_geom = Polygon(l_vertices)
    l_shape_mass = estimate_realistic_mass(l_shape_geom)
    objects['l_shape'] = GenericObject(
        geometry=l_shape_geom,
        mass=l_shape_mass,
        lateral_friction=0.25,
        heading=0.0,
        name="L-Shape (Symmetric)"
    )
    
    # 7. L-SHAPE (asymmetric - different arm lengths)
    asym_l_vertices = [
        (0, 0), (0.28*S, 0), (0.28*S, 0.12*S), (0.12*S, 0.12*S), 
        (0.12*S, 0.22*S), (0, 0.22*S)
    ]
    asym_l_geom = Polygon(asym_l_vertices)
    asym_l_mass = estimate_realistic_mass(asym_l_geom)
    objects['asym_l_shape'] = GenericObject(
        geometry=asym_l_geom,
        mass=asym_l_mass,
        lateral_friction=0.26,
        heading=0.0,
        name="L-Shape (Asymmetric)"
    )
    
    # 8. T-SHAPE (symmetric)
    t_vertices = [
        (-0.25*S, 0.15*S), (0.25*S, 0.15*S), (0.25*S, 0.05*S),
        (0.08*S, 0.05*S), (0.08*S, -0.2*S), (-0.08*S, -0.2*S),
        (-0.08*S, 0.05*S), (-0.25*S, 0.05*S)
    ]
    t_shape_geom = Polygon(t_vertices)
    t_shape_mass = estimate_realistic_mass(t_shape_geom)
    objects['t_shape'] = GenericObject(
        geometry=t_shape_geom,
        mass=t_shape_mass,
        lateral_friction=0.28,
        heading=0.0,
        name="T-Shape (Symmetric)"
    )
    
    # 9. T-SHAPE (asymmetric - offset stem)
    asym_t_vertices = [
        (-0.25*S, 0.15*S), (0.25*S, 0.15*S), (0.25*S, 0.05*S),
        (0.12*S, 0.05*S), (0.12*S, -0.2*S), (-0.05*S, -0.2*S),
        (-0.05*S, 0.05*S), (-0.25*S, 0.05*S)
    ]
    asym_t_geom = Polygon(asym_t_vertices)
    asym_t_mass = estimate_realistic_mass(asym_t_geom)
    objects['asym_t_shape'] = GenericObject(
        geometry=asym_t_geom,
        mass=asym_t_mass,
        lateral_friction=0.28,
        heading=0.0,
        name="T-Shape (Asymmetric)"
    )
    
    return objects

# %%
# STEP 2: CONTACT POINT PARAMETERIZATION SYSTEM
# =============================================================================

class ContactPointParameterization:
    """
    System for parameterizing contact points on arbitrary object boundaries.
    Supports both convex and non-convex shapes.
    """
    
    def __init__(self, generic_object):
        """
        Initialize parameterization for a given object.
        
        Args:
            generic_object: GenericObject instance
        """
        self.object = generic_object
        self.boundary = generic_object.boundary
        self.total_length = generic_object.boundary_length
        
        # Extract boundary coordinates for efficient computation
        self.boundary_coords = np.array(self.boundary.coords)
        self.n_segments = len(self.boundary_coords) - 1  # Number of line segments
        
        # Pre-compute segment lengths and cumulative distances
        self._compute_segment_info()
        
        # Check polygon orientation (clockwise vs counter-clockwise)
        # This is needed to correctly determine normal vector directions
        self._is_clockwise = self._check_orientation()
    
    def _compute_segment_info(self):
        """Pre-compute segment lengths and cumulative distances along boundary."""
        self.segment_lengths = []
        self.cumulative_distances = [0.0]
        
        for i in range(self.n_segments):
            p1 = self.boundary_coords[i]
            p2 = self.boundary_coords[i + 1]
            length = np.linalg.norm(p2 - p1)
            self.segment_lengths.append(length)
            self.cumulative_distances.append(self.cumulative_distances[-1] + length)
    
    def _check_orientation(self):
        """
        Check if the polygon boundary is oriented clockwise or counter-clockwise.
        
        Uses the shoelace formula to calculate signed area.
        Positive signed area = counter-clockwise (CCW)
        Negative signed area = clockwise (CW)
        
        Returns:
            bool: True if clockwise, False if counter-clockwise
        """
        coords = self.boundary_coords
        n = len(coords) - 1  # Last point is duplicate of first
        
        # Calculate signed area using shoelace formula
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += (coords[j][0] - coords[i][0]) * (coords[j][1] + coords[i][1])
        
        # If signed_area > 0, boundary is clockwise (CW)
        # If signed_area < 0, boundary is counter-clockwise (CCW)
        return signed_area > 0
    
    def parameter_to_point(self, t):
        """
        Convert parameter t ∈ [0, 1] to a point on the boundary.
        t=0 starts at the first vertex, t=1 completes the loop.
        
        Args:
            t: Parameter value in [0, 1]
            
        Returns:
            tuple: (point, segment_index, local_t)
                point: (x, y) coordinates on boundary
                segment_index: Which line segment the point lies on
                local_t: Parameter within that segment [0, 1]
        """
        # Clamp parameter to [0, 1]
        t = max(0.0, min(1.0, t))
        
        # Convert to distance along boundary
        target_distance = t * self.total_length
        
        # Find which segment contains this distance
        segment_index = 0
        for i in range(self.n_segments):
            if target_distance <= self.cumulative_distances[i + 1]:
                segment_index = i
                break
        
        # Compute local parameter within the segment
        segment_start_dist = self.cumulative_distances[segment_index]
        remaining_distance = target_distance - segment_start_dist
        segment_length = self.segment_lengths[segment_index]
        
        if segment_length > 0:
            local_t = remaining_distance / segment_length
        else:
            local_t = 0.0
        
        # Interpolate within the segment
        p1 = self.boundary_coords[segment_index]
        p2 = self.boundary_coords[segment_index + 1]
        point = p1 + local_t * (p2 - p1)
        
        return point, segment_index, local_t

    def point_to_parameter(self, point):
        """
        Convert a 2D point to the parameter of its closest point on the boundary.
        
        Args:
            point: array-like (x, y) coordinates
            
        Returns:
            dict: Information about the closest boundary point
                - 'parameter': t value [0, 1] of closest point on boundary
                - 'closest_point': (x, y) coordinates of closest boundary point
                - 'distance': Distance from input point to boundary
                - 'segment_index': Which line segment contains the closest point
                - 'local_parameter': Parameter within that segment [0, 1]
        """
        point = np.array(point)
        min_distance = float('inf')
        best_t = 0.0
        best_point = None
        best_segment_index = 0
        best_local_t = 0.0
        
        # Check each line segment
        for i in range(self.n_segments):
            p1 = self.boundary_coords[i]
            p2 = self.boundary_coords[i + 1]
            
            # Find closest point on this segment
            segment_vec = p2 - p1
            segment_length_sq = np.dot(segment_vec, segment_vec)
            
            if segment_length_sq < 1e-10:
                # Degenerate segment (point)
                closest_on_segment = p1
                local_t = 0.0
            else:
                # Project point onto line segment
                t_proj = np.dot(point - p1, segment_vec) / segment_length_sq
                # Clamp to segment bounds [0, 1]
                local_t = max(0.0, min(1.0, t_proj))
                closest_on_segment = p1 + local_t * segment_vec
            
            # Calculate distance
            distance = np.linalg.norm(point - closest_on_segment)
            
            # Update if this is the closest so far
            if distance < min_distance:
                min_distance = distance
                best_point = closest_on_segment
                best_segment_index = i
                best_local_t = local_t
        
        # Convert segment index and local parameter to global parameter t
        segment_start_dist = self.cumulative_distances[best_segment_index]
        segment_length = self.segment_lengths[best_segment_index]
        distance_along_boundary = segment_start_dist + best_local_t * segment_length
        best_t = distance_along_boundary / self.total_length
        
        return {
            'parameter': best_t,
            'closest_point': best_point,
            'distance': min_distance,
            'segment_index': best_segment_index,
            'local_parameter': best_local_t
        }
    
    def get_tangent_vector(self, t):
        """
        Get the tangent vector at parameter t.
        
        Args:
            t: Parameter value in [0, 1]
            
        Returns:
            numpy.ndarray: Unit tangent vector (pointing in direction of increasing t)
        """
        point, segment_index, local_t = self.parameter_to_point(t)
        
        # Get segment direction
        p1 = self.boundary_coords[segment_index]
        p2 = self.boundary_coords[segment_index + 1]
        tangent = p2 - p1
        
        # Normalize
        tangent_length = np.linalg.norm(tangent)
        if tangent_length > 0:
            tangent = tangent / tangent_length
        else:
            tangent = np.array([1.0, 0.0])  # Default direction
            
        return tangent
    
    def get_normal_vector(self, t, outward=True):
        """
        Get the normal vector at parameter t.
        
        Args:
            t: Parameter value in [0, 1]
            outward: If True, return outward normal; if False, return inward normal
            
        Returns:
            numpy.ndarray: Unit normal vector
        """
        tangent = self.get_tangent_vector(t)
        
        # Get normal by rotating tangent 90 degrees
        # For outward normal: rotate clockwise (for standard CCW orientation)
        if outward:
            normal = np.array([tangent[1], -tangent[0]])  # Rotate 90° clockwise
        else:
            normal = np.array([-tangent[1], tangent[0]])  # Rotate 90° counter-clockwise
        
        # FIX: Check polygon orientation for ALL polygons (not just circles)
        # If the boundary is clockwise, we need to flip the normals
        # because the normal calculation assumes counter-clockwise traversal
        if self._is_clockwise:
            normal = -normal
            
        return normal
    
    def get_contact_info(self, t):
        """
        Get complete contact information at parameter t.
        
        Args:
            t: Parameter value in [0, 1]
            
        Returns:
            dict: Complete contact information
                - 'point': Contact point coordinates
                - 'tangent': Unit tangent vector
                - 'normal_outward': Outward unit normal vector
                - 'normal_inward': Inward unit normal vector
                - 'parameter': The input parameter t
                - 'segment_index': Which boundary segment
                - 'local_parameter': Parameter within segment
        """
        point, segment_index, local_t = self.parameter_to_point(t)
        tangent = self.get_tangent_vector(t)
        normal_outward = self.get_normal_vector(t, outward=True)
        normal_inward = self.get_normal_vector(t, outward=False)
        
        return {
            'point': point,
            'tangent': tangent,
            'normal_outward': normal_outward,
            'normal_inward': normal_inward,
            'parameter': t,
            'segment_index': segment_index,
            'local_parameter': local_t
        }
    
    def generate_uniform_parameters(self, n_points):
        """
        Generate n uniformly distributed parameters along the boundary.
        
        Args:
            n_points: Number of contact points to generate
            
        Returns:
            numpy.ndarray: Array of parameter values
        """
        if n_points <= 0:
            return np.array([])
        elif n_points == 1:
            return np.array([0.02])  # Start at first vertex
        else:
            # Distribute evenly around the boundary
            return np.linspace(0.02, 0.98, n_points)
    
    def visualize_parameterization(self, n_test_points=20, ax=None):
        """
        Visualize the parameterization with test points and vectors.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw the object
        self.object.visualize(ax=ax, alpha=0.3)
        
        # Generate test points
        test_params = np.linspace(0, 1, n_test_points, endpoint=False)
        
        for i, t in enumerate(test_params):
            contact_info = self.get_contact_info(t)
            point = contact_info['point']
            tangent = contact_info['tangent']
            normal = contact_info['normal_inward']
            
            # Plot contact point
            ax.plot(point[0], point[1], 'ro', markersize=6)
            
            # Plot tangent vector (green)
            scale = 0.05
            ax.arrow(point[0], point[1], 
                    tangent[0] * scale, tangent[1] * scale,
                    head_width=0.01, head_length=0.01, 
                    fc='green', ec='green', alpha=0.7)
            
            # Plot outward normal vector (red)
            ax.arrow(point[0], point[1], 
                    normal[0] * scale, normal[1] * scale,
                    head_width=0.01, head_length=0.01, 
                    fc='red', ec='red', alpha=0.7)
            
            # Add parameter label
            if i % 4 == 0:  # Only label every 4th point to avoid clutter
                ax.text(point[0] + 0.02, point[1] + 0.02, f't={t:.2f}', 
                       fontsize=8, alpha=0.8)
        
        ax.set_title(f'Contact Parameterization: {self.object.name}')
        ax.legend(['Contact Points', 'Tangent Vectors', 'Normal Vectors'])
        
        return ax


# %%
# STEP 3: GENERIC CONTACT POINT CALCULATION SYSTEM WITH WRENCH CALCULATION
# =============================================================================
class ContactPoint:
    """
    Represents a single contact point with all associated geometric information.
    """
    def __init__(self, position, tangent, normal_outward, normal_inward, 
                 parameter, force_direction=None, object_ref=None):
        """
        Initialize a contact point.
        
        Args:
            position: (x, y) coordinates of contact point
            tangent: Unit tangent vector at contact point
            normal_outward: Outward unit normal vector
            normal_inward: Inward unit normal vector  
            parameter: Parameter value t ∈ [0, 1] along boundary
            force_direction: Optional force direction (radians)
            object_ref: Reference to the GenericObject (for wrench calculations)
        """
        self.position = np.array(position)
        self.tangent = np.array(tangent)
        self.normal_outward = np.array(normal_outward)
        self.normal_inward = np.array(normal_inward)
        self.parameter = parameter
        self.force_direction = force_direction
        self.object_ref = object_ref
    
    def get_force_vector(self, normal_component=1.0, tangential_component=0.0, 
                        max_magnitude=None, enforce_friction=True):
        """
        Get force vector for pushing at this contact point with proper constraints.
        
        Args:
            normal_component: Force component along inward normal (positive = pushing into object)
            tangential_component: Force component along tangent
            max_magnitude: Maximum allowed force magnitude (uses robot limit if None)
            enforce_friction: If True, enforce friction cone constraint
            
        Returns:
            dict: Contains 'force_vector', 'is_valid', 'clamped_normal', 'clamped_tangential'
        """
        if self.object_ref is None:
            # Fallback to simple normal force if no object reference
            direction = self.normal_inward if normal_component >= 0 else -self.normal_inward
            return {
                'force_vector': abs(normal_component) * direction,
                'is_valid': True,
                'clamped_normal': normal_component,
                'clamped_tangential': 0.0
            }
        
        # Get maximum force from object or use provided value
        if max_magnitude is None:
            # Default robot max force (could be set from robot properties)
            max_magnitude = 5.0  # You can link this to robot_data_to_save['push_max_possible_magnitude']
        
        # Start with desired force components
        desired_normal = normal_component
        desired_tangential = tangential_component
        
        # Apply friction constraint: |tangential| ≤ μ * |normal|
        if enforce_friction and self.object_ref.get_contact_friction() > 0:
            max_tangential = abs(desired_normal) * self.object_ref.get_contact_friction()
            if abs(desired_tangential) > max_tangential:
                # Clamp tangential component to friction limit
                desired_tangential = np.sign(desired_tangential) * max_tangential
        
        # Calculate resulting force vector
        force_vector = (desired_normal * self.normal_inward + 
                       desired_tangential * self.tangent)
        
        # Apply magnitude constraint
        force_magnitude = np.linalg.norm(force_vector)
        if force_magnitude > max_magnitude:
            # Scale down both components proportionally
            scale_factor = max_magnitude / force_magnitude
            force_vector *= scale_factor
            desired_normal *= scale_factor
            desired_tangential *= scale_factor
        
        # Check if the force satisfies all constraints
        is_valid = True
        if enforce_friction and self.object_ref.get_contact_friction() > 0:
            friction_satisfied = abs(desired_tangential) <= abs(desired_normal) * self.object_ref.get_contact_friction() + 1e-6
            magnitude_satisfied = np.linalg.norm(force_vector) <= max_magnitude + 1e-6
            is_valid = friction_satisfied and magnitude_satisfied
        
        return {
            'force_vector': force_vector,
            'is_valid': is_valid,
            'clamped_normal': desired_normal,
            'clamped_tangential': desired_tangential,
            'magnitude': np.linalg.norm(force_vector),
            'max_magnitude': max_magnitude
        }
    
    def calculate_contact_wrench(self, normal_force, tangential_force, 
                               friction_constraint=True):
        """
        Calculate the wrench (force and torque) at object centroid from contact forces.
        
        Args:
            normal_force: Force component along inward normal (positive = pushing into object)
            tangential_force: Force component along tangent (constrained by friction)
            friction_constraint: If True, enforce friction cone constraint
            
        Returns:
            dict: Contains 'force_x', 'force_y', 'torque', 'is_valid'
                force_x, force_y: Net forces at centroid
                torque: Net torque about centroid (positive = CCW)
                is_valid: Whether the force satisfies friction constraints
        """
        if self.object_ref is None:
            raise ValueError("Object reference needed for wrench calculation")
        
        # Check friction constraint: |tangential_force| ≤ μ * |normal_force|
        is_valid = True
        if friction_constraint and self.object_ref.get_contact_friction() > 0:
            max_tangential = abs(normal_force) * self.object_ref.get_contact_friction()
            if abs(tangential_force) > max_tangential:
                if friction_constraint:
                    # Clamp to friction limit
                    tangential_force = np.sign(tangential_force) * max_tangential
                is_valid = False
        
        # Calculate contact force vector in world frame
        # Force = normal_force * normal_inward + tangential_force * tangent
        contact_force = (normal_force * self.normal_inward + 
                        tangential_force * self.tangent)
        
        # Get object centroid
        centroid = self.object_ref.get_centroid()
        centroid_pos = np.array([centroid.x, centroid.y])
        
        # Vector from centroid to contact point
        r_vector = self.position - centroid_pos
        
        # Calculate torque using cross product (2D: τ = r × F)
        # In 2D: τ = r_x * F_y - r_y * F_x
        torque = r_vector[0] * contact_force[1] - r_vector[1] * contact_force[0]
        
        return {
            'force_x': contact_force[0],
            'force_y': contact_force[1], 
            'torque': torque,
            'is_valid': is_valid,
            'normal_component': normal_force,
            'tangential_component': tangential_force,
            'contact_force_magnitude': np.linalg.norm(contact_force)
        }
    
    def get_friction_cone_limits(self):
        """
        Get the friction cone limits for this contact point.
        
        Returns:
            dict: Contains friction cone information
        """
        mu = self.object_ref.get_contact_friction() if self.object_ref is not None else 0.0
        if self.object_ref is None or mu <= 0:
            return {'friction_angle': 0, 'has_friction': False}
        
        friction_angle = np.arctan(mu)
        
        return {
            'friction_angle': friction_angle,
            'friction_coefficient': mu,
            'has_friction': True,
            'max_tangential_ratio': mu
        }
    
    def __repr__(self):
        return (f"ContactPoint(pos=({self.position[0]:.3f}, {self.position[1]:.3f}), "
                f"parameter={self.parameter:.3f})")

# =============================================================================
class GraspMatrixCalculator:
    """
    Utility class for calculating and manipulating grasp matrices.
    This implements a matrix-based approach for contact wrench analysis.
    """
    
    @staticmethod
    def build_wrench_matrix(contact_points):
        """
        Build the wrench matrix W where W[i,j] = wrench_j contribution from contact_i with unit force.
        W is a 3×n_contacts matrix where W @ α = wrench.
        
        Args:
            contact_points: List of ContactPoint objects
            
        Returns:
            numpy.ndarray: Wrench matrix of shape (3, n_contacts)
        """
        n_contacts = len(contact_points)
        wrench_matrix = np.zeros((3, n_contacts))
        
        for i, cp in enumerate(contact_points):
            # Calculate unit wrench (force magnitude = 1.0)
            unit_wrench = cp.calculate_contact_wrench(
                normal_force=1.0,
                tangential_force=0.0,
                friction_constraint=False
            )
            
            # Fill wrench matrix column i
            wrench_matrix[0, i] = unit_wrench['force_x']   # Fx contribution
            wrench_matrix[1, i] = unit_wrench['force_y']   # Fy contribution  
            wrench_matrix[2, i] = unit_wrench['torque']    # τ contribution
            
        return wrench_matrix
    
    @staticmethod
    def build_tangent_wrench_matrix(contact_points):
        """
        Build the tangent wrench matrix W_t where W_t[i,j] = tangent wrench contribution 
        from contact_i with unit tangent force.
        
        Args:
            contact_points: List of ContactPoint objects
            
        Returns:
            numpy.ndarray: Tangent wrench matrix of shape (3, n_contacts)
        """
        n_contacts = len(contact_points)
        tangent_wrench_matrix = np.zeros((3, n_contacts))
        
        for i, cp in enumerate(contact_points):
            # Calculate unit tangent wrench (tangent force magnitude = 1.0)
            unit_wrench = cp.calculate_contact_wrench(
                normal_force=0.0,
                tangential_force=1.0,
                friction_constraint=False  # No friction constraint for unit calculation
            )
            
            # Fill tangent wrench matrix column i
            tangent_wrench_matrix[0, i] = unit_wrench['force_x']
            tangent_wrench_matrix[1, i] = unit_wrench['force_y']
            tangent_wrench_matrix[2, i] = unit_wrench['torque']
            
        return tangent_wrench_matrix
    
    @staticmethod
    def compute_total_wrench(wrench_matrix, force_magnitudes):
        """
        Compute total wrench using the wrench matrix.
        W @ α = wrench
        
        Args:
            wrench_matrix: Wrench matrix of shape (3, n_contacts)
            force_magnitudes: Array of force magnitudes (α values)
            
        Returns:
            numpy.ndarray: Total wrench [force_x, force_y, torque]
        """
        return wrench_matrix @ force_magnitudes
    
    @staticmethod
    def visualize_wrench(contact_points, normal_forces, tangential_forces=None, 
                        ax=None, force_scale=0.1, show_friction_cones=True):
        """
        Visualize contact forces and resulting wrench.
        
        Args:
            contact_points: List of ContactPoint objects
            normal_forces: Array of normal force components
            tangential_forces: Array of tangential force components (optional)
            ax: Optional matplotlib axis
            force_scale: Scale factor for force visualization
            show_friction_cones: Whether to display friction cones
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        if tangential_forces is None:
            tangential_forces = [0.0] * len(normal_forces)
        
        # Get the object (assume all contact points belong to same object)
        if contact_points:
            obj = contact_points[0].object_ref
            if obj:
                obj.visualize(ax=ax, alpha=0.3, show_frame=True)
        
        # Build wrench matrices
        normal_wrench_matrix = GraspMatrixCalculator.build_wrench_matrix(contact_points)
        
        # Compute total wrench
        total_wrench = GraspMatrixCalculator.compute_total_wrench(
            normal_wrench_matrix, np.array(normal_forces))
        
        # Add tangential contribution if provided
        if any(abs(tf) > 1e-6 for tf in tangential_forces):
            tangent_wrench_matrix = GraspMatrixCalculator.build_tangent_wrench_matrix(contact_points)
            total_wrench += GraspMatrixCalculator.compute_total_wrench(
                tangent_wrench_matrix, np.array(tangential_forces))
        
        # Draw contact points and forces
        for i, cp in enumerate(contact_points):
            pos = cp.position
            normal_f = normal_forces[i] if i < len(normal_forces) else 0.0
            tangential_f = tangential_forces[i] if i < len(tangential_forces) else 0.0
            
            # Contact point
            ax.plot(pos[0], pos[1], 'ro', markersize=12, 
                   label='Contact Points' if i == 0 else '')
            
            # Show friction cone
            if show_friction_cones:
                friction_info = cp.get_friction_cone_limits()
                if friction_info['has_friction']:
                    cone_angle = friction_info['friction_angle']
                    # Draw friction cone
                    cone_length = 0.08
                    
                    # Cone boundaries
                    normal_angle = np.arctan2(cp.normal_inward[1], cp.normal_inward[0])
                    cone_left = normal_angle + cone_angle
                    cone_right = normal_angle - cone_angle
                    
                    for cone_dir in [cone_left, cone_right]:
                        ax.arrow(pos[0], pos[1],
                               cone_length * np.cos(cone_dir),
                               cone_length * np.sin(cone_dir),
                               head_width=0.01, head_length=0.01,
                               fc='orange', ec='orange', alpha=0.5,
                               linestyle='--')
            
            # Normal force (blue)
            if abs(normal_f) > 1e-6:
                force_vec = normal_f * cp.normal_inward * force_scale
                ax.arrow(pos[0], pos[1], force_vec[0], force_vec[1],
                        head_width=0.02, head_length=0.02, linewidth=2,
                        fc='blue', ec='blue', alpha=0.8,
                        label='Normal Force' if i == 0 else '')
            
            # Tangential force (green)
            if abs(tangential_f) > 1e-6:
                force_vec = tangential_f * cp.tangent * force_scale
                ax.arrow(pos[0], pos[1], force_vec[0], force_vec[1],
                        head_width=0.02, head_length=0.02, linewidth=2,
                        fc='green', ec='green', alpha=0.8,
                        label='Tangential Force' if i == 0 else '')
            
            # Label contact point
            ax.text(pos[0] + 0.03, pos[1] + 0.03, 
                   f'CP{i+1}\nN:{normal_f:.1f}\nT:{tangential_f:.1f}', 
                   fontsize=9, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        # Show total wrench at centroid
        if obj:
            centroid = obj.get_centroid()
            
            # Total force vector (red)
            total_force_vec = np.array([total_wrench[0], total_wrench[1]]) * force_scale
            if np.linalg.norm(total_force_vec) > 1e-6:
                ax.arrow(centroid.x, centroid.y, 
                        total_force_vec[0], total_force_vec[1],
                        head_width=0.03, head_length=0.03, linewidth=3,
                        fc='red', ec='darkred', alpha=0.9,
                        label='Total Force')
            
            # Torque indicator (purple circle)
            if abs(total_wrench[2]) > 1e-6:
                torque_radius = abs(total_wrench[2]) * 0.1
                torque_circle = plt.Circle((centroid.x, centroid.y), 
                                         torque_radius, 
                                         fill=False, color='purple', 
                                         linewidth=3, alpha=0.7)
                ax.add_patch(torque_circle)
                
                # Arrow to show rotation direction
                if total_wrench[2] > 0:  # CCW
                    ax.annotate('⟲', (centroid.x, centroid.y + torque_radius), 
                               fontsize=20, color='purple', ha='center')
                else:  # CW
                    ax.annotate('⟳', (centroid.x, centroid.y + torque_radius), 
                               fontsize=20, color='purple', ha='center')
        
        ax.set_title(f'Contact Wrench Analysis\n'
                    f'Total Force: ({total_wrench[0]:.2f}, {total_wrench[1]:.2f}) N\n'
                    f'Total Torque: {total_wrench[2]:.3f} N⋅m')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return ax, total_wrench



class GenericContactCalculator:
    """
    Generic contact point calculation system that works with any object shape.
    """
    
    def __init__(self, generic_object):
        """
        Initialize calculator for a specific object.
        
        Args:
            generic_object: GenericObject instance
        """
        self.object = generic_object
        self.parameterization = ContactPointParameterization(generic_object)
    
    def calculate_contact_points(self, n_contacts, strategy='uniform', 
                               custom_parameters=None, robot_angles=None):
        """
        Calculate contact points using the new generic system.
        
        Args:
            n_contacts: Number of contact points
            strategy: Contact distribution strategy
                - 'uniform': Evenly distributed along boundary
                - 'angle_based': Based on provided robot angles  
                - 'custom': Use custom parameter values
            custom_parameters: Custom parameter values (for 'custom' strategy)
            robot_angles: Robot approach angles (for 'angle_based' strategy)
            
        Returns:
            list: List of ContactPoint objects
        """
        if strategy == 'uniform':
            parameters = self.parameterization.generate_uniform_parameters(n_contacts)
            
        elif strategy == 'angle_based':
            if robot_angles is None:
                raise ValueError("robot_angles must be provided for 'angle_based' strategy")
            parameters = self._angles_to_parameters(robot_angles[:n_contacts])
            
        elif strategy == 'custom':
            if custom_parameters is None:
                raise ValueError("custom_parameters must be provided for 'custom' strategy")
            parameters = np.array(custom_parameters[:n_contacts])
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Generate contact points
        contact_points = []
        for i, t in enumerate(parameters):
            contact_info = self.parameterization.get_contact_info(t)
            
            # Determine force direction
            force_direction = None
            if robot_angles is not None and i < len(robot_angles):
                force_direction = robot_angles[i]
            
            contact_point = ContactPoint(
                position=contact_info['point'],
                tangent=contact_info['tangent'],
                normal_outward=contact_info['normal_outward'],
                normal_inward=contact_info['normal_inward'],
                parameter=t,
                force_direction=force_direction,
                object_ref=self.object  # Add object reference
            )
            
            contact_points.append(contact_point)
        
        return contact_points
    
    def _angles_to_parameters(self, angles):
        """
        Convert robot approach angles to boundary parameters.
        This is a heuristic mapping - can be refined based on specific needs.
        
        Args:
            angles: Array of angles in radians
            
        Returns:
            numpy.ndarray: Corresponding parameter values
        """
        # Simple mapping: normalize angles to [0, 1] range
        normalized_angles = (np.array(angles) % (2 * np.pi)) / (2 * np.pi)
        return normalized_angles

    def visualize_contact_solution(self, contact_points, normal_forces=None, tangential_forces=None, 
                                 ax=None, force_scale=0.1):
        """
        Visualize the contact solution using the enhanced GraspMatrixCalculator.
        
        Args:
            contact_points: List of ContactPoint objects
            normal_forces: Optional array of normal force magnitudes
            tangential_forces: Optional array of tangential force magnitudes
            ax: Optional matplotlib axis
            force_scale: Scale factor for force visualization
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use default forces if none provided
        if normal_forces is None:
            normal_forces = np.ones(len(contact_points))
        
        if tangential_forces is None:
            tangential_forces = np.zeros(len(contact_points))
        
        # Use the GraspMatrixCalculator for visualization
        return GraspMatrixCalculator.visualize_wrench(
            contact_points, normal_forces, tangential_forces,
            ax=ax, force_scale=force_scale, show_friction_cones=True
        )


# =============================================================================
# DEMONSTRATION: FULLY ENHANCED SYSTEM WITH IMPROVED FORCE HANDLING
# =============================================================================

def demonstrate_enhanced_wrench_system():
    """Demonstrate the enhanced system with improved force handling and additional shapes."""
    
    print("\n=== ENHANCED STEP 3: Improved Contact System with Force Constraints ===")
    
    # Test with multiple shapes including the problematic circle
    test_shapes = ['rectangle', 'circle', 'triangle']
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    for shape_idx, shape_name in enumerate(test_shapes):
        obj = standard_objects[shape_name]
        calculator = GenericContactCalculator(obj)
        
        # Calculate contact points
        contact_points = calculator.calculate_contact_points(n_contacts=3, strategy='uniform')
        
        print(f"\n=== {obj.name.upper()} - Enhanced Force Analysis ===")
        print(f"Lateral friction coefficient: {obj.lateral_friction}")
        
        # Test different force scenarios
        scenarios = [
            ("Pure Normal Forces", [(2.0, 0.0), (1.5, 0.0), (1.0, 0.0)]),
            ("Mixed Forces (friction constrained)", [(2.0, 0.4), (1.5, -0.3), (1.0, 0.5)])
        ]
        
        for scenario_idx, (scenario_name, force_components) in enumerate(scenarios):
            ax = axes[scenario_idx, shape_idx]
            
            print(f"\n--- {scenario_name} ---")
            
            # Extract normal and tangential components
            normal_forces = [fc[0] for fc in force_components]
            tangential_forces = [fc[1] for fc in force_components]
            
            # Test the enhanced force vector calculation
            enhanced_forces = []
            for i, cp in enumerate(contact_points):
                force_result = cp.get_force_vector(
                    normal_component=normal_forces[i],
                    tangential_component=tangential_forces[i],
                    max_magnitude=5.0,  # Robot max force
                    enforce_friction=True
                )
                enhanced_forces.append(force_result)
                
                print(f"CP{i+1}: Desired({normal_forces[i]:.1f}, {tangential_forces[i]:.1f}) "
                      f"→ Actual({force_result['clamped_normal']:.1f}, {force_result['clamped_tangential']:.1f}) "
                      f"Mag:{force_result['magnitude']:.2f} Valid:{force_result['is_valid']}")
            
            # Calculate total wrench using the enhanced forces
            actual_normal = [ef['clamped_normal'] for ef in enhanced_forces]
            actual_tangential = [ef['clamped_tangential'] for ef in enhanced_forces]
            
            # FIX: Using the correct approach to calculate total wrench
            # Build the wrench matrices for normal and tangent forces
            normal_wrench_matrix = GraspMatrixCalculator.build_wrench_matrix(contact_points)
            tangent_wrench_matrix = GraspMatrixCalculator.build_tangent_wrench_matrix(contact_points)
            
            # Compute normal and tangent contributions separately
            normal_contribution = GraspMatrixCalculator.compute_total_wrench(normal_wrench_matrix, actual_normal)
            tangent_contribution = GraspMatrixCalculator.compute_total_wrench(tangent_wrench_matrix, actual_tangential)
            
            # Combined total wrench
            total_wrench_vec = normal_contribution + tangent_contribution
            
            # Format as a dictionary for compatibility with rest of the code
            total_wrench = {
                'total_force_x': float(total_wrench_vec[0]),
                'total_force_y': float(total_wrench_vec[1]),
                'total_torque': float(total_wrench_vec[2])
            }
            
            print(f"Total Wrench: Force=({total_wrench['total_force_x']:.2f}, "
                  f"{total_wrench['total_force_y']:.2f}), "
                  f"Torque={total_wrench['total_torque']:.3f}")
            
            # Visualize with enhanced force information
            ax_result, visualized_wrench = calculator.visualize_contact_solution(
                contact_points, actual_normal, actual_tangential,
                ax=ax, force_scale=0.06)
            
            ax.set_title(f'{obj.name}\n{scenario_name}')
            
            # Add force constraint information to the plot
            constraint_text = f"μ_lateral = {obj.lateral_friction}\nF_max = 5.0 N"
            ax.text(0.02, 0.98, constraint_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Additional demonstration: Test force vector generation
    print("\n=== Force Vector Generation Test ===")
    circle_obj = standard_objects['circle']
    circle_calc = GenericContactCalculator(circle_obj)
    circle_contacts = circle_calc.calculate_contact_points(n_contacts=4, strategy='uniform')
    
    print(f"\nCircle Contact Points (testing normal vector fix):")
    for i, cp in enumerate(circle_contacts):
        print(f"CP{i+1}: pos=({cp.position[0]:.3f}, {cp.position[1]:.3f})")
        print(f"      normal_inward=({cp.normal_inward[0]:.3f}, {cp.normal_inward[1]:.3f})")
        print(f"      normal_outward=({cp.normal_outward[0]:.3f}, {cp.normal_outward[1]:.3f})")
        
        # Test force generation with different scenarios
        force_tests = [
            (2.0, 0.0, "Pure normal"),
            (2.0, 0.5, "Mixed within friction"),
            (2.0, 1.0, "Mixed at friction limit"),
            (2.0, 1.5, "Mixed exceeding friction")
        ]
        
        for normal, tangential, desc in force_tests:
            result = cp.get_force_vector(normal, tangential, enforce_friction=True)
            print(f"        {desc}: ({normal:.1f}, {tangential:.1f}) → "
                  f"({result['clamped_normal']:.1f}, {result['clamped_tangential']:.1f}) "
                  f"Valid: {result['is_valid']}")
    
    return contact_points, total_wrench_vec

# %%
class WrenchSpaceVisualizer:
    """
    A class for calculating and visualizing wrench spaces (zonotopes) from grasp matrices.
    Provides methods to calculate wrench space points and create interactive 3D visualizations.
    """
    
    def __init__(self):
        """Initialize the WrenchSpaceVisualizer."""
        pass
    
    def calculate_wrench_space(self, contact_points, force_ranges=None, sampling_density=3, enable_tangent_forces=False):
        """
        Calculate the achievable wrench space from a list of contact points.
        
        Args:
            contact_points: List of ContactPoint objects
            force_ranges: List of (min_force, max_force) tuples for each contact
            sampling_density: Number of force samples per contact point
            enable_tangent_forces: Whether to include tangent forces in calculation
            
        Returns:
            dict: Contains 'wrenches', 'feasible_mask', 'contact_forces'
        """

        if not contact_points:
            return {'wrenches': np.zeros((0, 3)), 'feasible_mask': np.array([]), 
                    'contact_forces': []}
        
        n_contacts = len(contact_points)
        
        # Default force ranges
        if force_ranges is None:
            force_ranges = [(0.0, 5.0)] * n_contacts
        
        # 🔧 FIX: Build grasp matrices correctly
        G_normal = np.zeros((3, n_contacts))
        G_tangent = np.zeros((3, n_contacts))
        
        for i, cp in enumerate(contact_points):
            # Normal wrench (force magnitude = 1.0)
            normal_wrench = cp.calculate_contact_wrench(
                normal_force=1.0, 
                tangential_force=0.0, 
                friction_constraint=True
            )
            G_normal[0, i] = normal_wrench['force_x']
            G_normal[1, i] = normal_wrench['force_y']
            G_normal[2, i] = normal_wrench['torque']
            
            # Tangent wrench (tangent force magnitude = 1.0)
            if enable_tangent_forces:
                tangent_wrench = cp.calculate_contact_wrench(
                    normal_force=0.0,
                    tangential_force=1.0,
                    friction_constraint=False  # Pure tangent
                )
                G_tangent[0, i] = tangent_wrench['force_x']
                G_tangent[1, i] = tangent_wrench['force_y']
                G_tangent[2, i] = tangent_wrench['torque']
        
        # 🔧 FIX: Generate force samples properly
        # For each contact, sample (α_normal, β_tangent) pairs
        force_combos_per_contact = []
        
        for i, cp in enumerate(contact_points):
            min_f, max_f = force_ranges[i]
            normal_samples = np.linspace(min_f, max_f, sampling_density)
            
            contact_forces = []
            
            if enable_tangent_forces and cp.object_ref and cp.object_ref.get_contact_friction() > 0:
                μ = cp.object_ref.get_contact_friction()
                
                for α in normal_samples:
                    # Friction constraint: |β| ≤ μ * α
                    max_tangent = μ * α
                    
                    if max_tangent > 1e-6:
                        # Sample tangent forces within cone
                        tangent_samples = np.linspace(-max_tangent, max_tangent, 
                                                    max(3, sampling_density//2))
                    else:
                        tangent_samples = [0.0]
                    
                    for β in tangent_samples:
                        contact_forces.append((α, β))
            else:
                # No friction - only normal forces
                contact_forces = [(α, 0.0) for α in normal_samples]
            
            force_combos_per_contact.append(contact_forces)
        
        # 🔧 FIX: Compute wrenches correctly
        all_combos = list(product(*force_combos_per_contact))
        
        # print(f"🎯 Computing wrench space with {len(all_combos)} force combinations...")
        
        wrenches = []
        feasible_mask = []
        contact_forces_log = []
        
        for combo in all_combos:
            # Extract normal and tangent forces as separate vectors
            α_vec = np.array([f[0] for f in combo])  # Normal forces (n_contacts,)
            β_vec = np.array([f[1] for f in combo])  # Tangent forces (n_contacts,)
            
            # 🔧 FIX: Correct wrench calculation
            wrench_normal = G_normal @ α_vec      # (3,) = (3, n) @ (n,)
            wrench_tangent = G_tangent @ β_vec    # (3,) = (3, n) @ (n,)
            
            total_wrench = wrench_normal + wrench_tangent
            
            # Feasibility check (already satisfied by sampling, but double-check)
            combo_feasible = True
            for i, (α, β) in enumerate(combo):
                if α < 0:  # Normal force must be non-negative
                    combo_feasible = False
                    break
                
                cp = contact_points[i]
                if cp.object_ref and cp.object_ref.get_contact_friction() > 0:
                    μ = cp.object_ref.get_contact_friction()
                    if abs(β) > μ * α + 1e-6:  # Friction cone violation
                        combo_feasible = False
                        break
            
            wrenches.append(total_wrench)
            feasible_mask.append(combo_feasible)
            contact_forces_log.append(combo)
        
        return {
            'wrenches': np.array(wrenches),
            'feasible_mask': np.array(feasible_mask),
            'contact_forces': contact_forces_log
        }
    
    def visualize_wrench_space_plotly(self, wrench_data, contact_points=None, 
                                     title="Wrench Space Visualization"):
        """
        Create an interactive Plotly visualization of the wrench space.
        
        Args:
            wrench_data: Output from calculate_wrench_space()
            contact_points: List of ContactPoint objects (for reference)
            title: Plot title
            
        Returns:
            plotly Figure object
        """
        wrenches = wrench_data['wrenches']
        feasible_mask = wrench_data['feasible_mask']
        
        if len(wrenches) == 0:
            # Return empty plot
            fig = go.Figure()
            fig.add_annotation(text="No wrench data to display", 
                              xref="paper", yref="paper", x=0.5, y=0.5)
            return fig
        
        # Create subplots: main 3D plot + 3 2D projections
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "scatter3d", "rowspan": 3}, {"type": "scatter"}],
                [None, {"type": "scatter"}],
                [None, {"type": "scatter"}]
            ],
            subplot_titles=("3D Wrench Space", "Fx vs Fy Projection", 
                           "Fx vs Torque Projection", "Fy vs Torque Projection"),
            horizontal_spacing=0.1,
            vertical_spacing=0.08
        )
        
        # Separate feasible and infeasible points
        feasible_wrenches = wrenches[feasible_mask]
        infeasible_wrenches = wrenches[~feasible_mask]
        
        # 3D scatter plot (main plot)
        if len(feasible_wrenches) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=feasible_wrenches[:, 0],
                    y=feasible_wrenches[:, 1], 
                    z=feasible_wrenches[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='green', opacity=0.6),
                    name='Feasible Wrenches',
                    hovertemplate='Fx: %{x:.3f}<br>Fy: %{y:.3f}<br>τ: %{z:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if len(infeasible_wrenches) > 0 and len(infeasible_wrenches) < 10000:  # Limit number for performance
            fig.add_trace(
                go.Scatter3d(
                    x=infeasible_wrenches[:, 0],
                    y=infeasible_wrenches[:, 1],
                    z=infeasible_wrenches[:, 2], 
                    mode='markers',
                    marker=dict(size=2, color='red', opacity=0.3),
                    name='Infeasible Wrenches',
                    hovertemplate='Fx: %{x:.3f}<br>Fy: %{y:.3f}<br>τ: %{z:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2D projections
        if len(feasible_wrenches) > 0:
            # Fx vs Fy projection
            fig.add_trace(
                go.Scatter(
                    x=feasible_wrenches[:, 0],
                    y=feasible_wrenches[:, 1],
                    mode='markers',
                    marker=dict(size=3, color='green', opacity=0.6),
                    name='Feasible (Fx-Fy)',
                    showlegend=False,
                    hovertemplate='Fx: %{x:.3f}<br>Fy: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Fx vs Torque projection  
            fig.add_trace(
                go.Scatter(
                    x=feasible_wrenches[:, 0],
                    y=feasible_wrenches[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='green', opacity=0.6),
                    name='Feasible (Fx-τ)',
                    showlegend=False,
                    hovertemplate='Fx: %{x:.3f}<br>τ: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # NEW: Fy vs Torque projection
            fig.add_trace(
                go.Scatter(
                    x=feasible_wrenches[:, 1],  # Fy
                    y=feasible_wrenches[:, 2],  # Torque
                    mode='markers',
                    marker=dict(size=3, color='green', opacity=0.6),
                    name='Feasible (Fy-τ)',
                    showlegend=False,
                    hovertemplate='Fy: %{x:.3f}<br>τ: %{y:.3f}<extra></extra>'
                ),
                row=3, col=2
            )
        
        # Add zero wrench point (origin)
        fig.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=6, color='black', symbol='diamond'),
                name='Origin (Zero Wrench)',
                hovertemplate='Origin (0,0,0)<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add convex hull outline for feasible points (optional)
        if len(feasible_wrenches) > 4:
            try:
                from scipy.spatial import ConvexHull
                
                # Fx-Fy projection hull
                hull_2d_xy = ConvexHull(feasible_wrenches[:, :2])  
                hull_points_xy = feasible_wrenches[hull_2d_xy.vertices, :2]
                hull_points_xy = np.vstack([hull_points_xy, hull_points_xy[0]])  # Close the loop
                
                fig.add_trace(
                    go.Scatter(
                        x=hull_points_xy[:, 0],
                        y=hull_points_xy[:, 1],
                        mode='lines',
                        line=dict(color='darkgreen', width=2),
                        name='Feasible Region (Fx-Fy)',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Fx-Torque projection hull
                hull_2d_xt = ConvexHull(feasible_wrenches[:, [0, 2]])
                hull_points_xt = feasible_wrenches[hull_2d_xt.vertices, :][:, [0, 2]]
                hull_points_xt = np.vstack([hull_points_xt, hull_points_xt[0]])  # Close the loop
                
                fig.add_trace(
                    go.Scatter(
                        x=hull_points_xt[:, 0],
                        y=hull_points_xt[:, 1],
                        mode='lines',
                        line=dict(color='darkgreen', width=2),
                        name='Feasible Region (Fx-τ)',
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                # NEW: Fy-Torque projection hull
                hull_2d_yt = ConvexHull(feasible_wrenches[:, [1, 2]])
                hull_points_yt = feasible_wrenches[hull_2d_yt.vertices, :][:, [1, 2]]
                hull_points_yt = np.vstack([hull_points_yt, hull_points_yt[0]])  # Close the loop
                
                fig.add_trace(
                    go.Scatter(
                        x=hull_points_yt[:, 0],
                        y=hull_points_yt[:, 1],
                        mode='lines',
                        line=dict(color='darkgreen', width=2),
                        name='Feasible Region (Fy-τ)',
                        showlegend=False
                    ),
                    row=3, col=2
                )
            except:
                pass  # Skip if convex hull fails
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sup>{len(feasible_wrenches)} feasible, {len(infeasible_wrenches)} infeasible points</sup>",
                font=dict(size=12)
            ),
            height=800,  # Increased height to accommodate the third 2D plot
            scene=dict(
                xaxis_title='Force X (N)',
                yaxis_title='Force Y (N)', 
                zaxis_title='Torque (N⋅m)',
                aspectmode='cube'
            )
        )
        
        # Update 2D subplot axes
        fig.update_xaxes(title_text="Force X (N)", row=1, col=2, title_font=dict(size=10))
        fig.update_yaxes(title_text="Force Y (N)", row=1, col=2, title_font=dict(size=10))
        
        fig.update_xaxes(title_text="Force X (N)", row=2, col=2, title_font=dict(size=10))
        fig.update_yaxes(title_text="Torque (N⋅m)", row=2, col=2, title_font=dict(size=10))
        
        # NEW: Update axes for Fy vs Torque projection
        fig.update_xaxes(title_text="Force Y (N)", row=3, col=2, title_font=dict(size=10))
        fig.update_yaxes(title_text="Torque (N⋅m)", row=3, col=2, title_font=dict(size=10))
        
        return fig
    
    def demonstrate_wrench_space_visualization(self, standard_objects):

        """
        Demonstrate interactive wrench space visualization using Plotly.
        
        Args:
            standard_objects: Dictionary of standard objects to visualize
            
        Returns:
            dict: Results containing wrench spaces for each object
        """
        print("\n" + "="*80)
        print("🚀 INTERACTIVE WRENCH SPACE VISUALIZATION WITH PLOTLY")
        print("="*80)
        
        # Test with different objects
        test_objects = ['rectangle', 'triangle', 'l_shape']
        
        results = {}
        
        for obj_name in test_objects:
            print(f"\n📊 Generating wrench space for {obj_name.upper()}")
            
            # Get object
            obj = standard_objects[obj_name]
            calculator = GenericContactCalculator(obj)
            
            # Generate contact points (adjust based on object complexity)
            n_contacts = 3 if obj_name == 'rectangle' else 4
            contact_points = calculator.calculate_contact_points(n_contacts=n_contacts, strategy='uniform')
            
            print(f"  Created {len(contact_points)} contact points")
            
            # Define force range based on object
            max_force = 3.0
            force_ranges = [(0.0, max_force)] * len(contact_points)
            
            # Calculate wrench space
            print(f"  Calculating wrench space (this may take a moment)...")
            wrench_data = self.calculate_wrench_space(
                contact_points, 
                force_ranges=force_ranges,
                sampling_density=4,  # Adjust for performance vs. detail
                enable_tangent_forces=True
            )
            
            # Save results
            results[obj_name] = {
                'contact_points': contact_points,
                'wrench_data': wrench_data
            }
            
            # Create visualization
            fig = self.visualize_wrench_space_plotly(
                wrench_data, 
                contact_points, 
                title=f"Wrench Space - {obj_name.title()}"
            )
            
            # Show the figure
            fig.show()
            
            # Calculate wrench space metrics
            feasible_wrenches = wrench_data['wrenches'][wrench_data['feasible_mask']]
            if len(feasible_wrenches) > 0:
                # Range of achievable wrenches
                wrench_min = np.min(feasible_wrenches, axis=0)
                wrench_max = np.max(feasible_wrenches, axis=0)
                wrench_ranges = wrench_max - wrench_min
                
                # Volume approximation (if more than 4 points)
                volume = 0
                if len(feasible_wrenches) > 4:
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(feasible_wrenches)
                        volume = hull.volume
                    except:
                        volume = "N/A (calculation failed)"
                
                print(f"\n  📈 Wrench Space Metrics for {obj_name.upper()}:")
                print(f"    Fx range: [{wrench_min[0]:.2f}, {wrench_max[0]:.2f}] N (width: {wrench_ranges[0]:.2f} N)")
                print(f"    Fy range: [{wrench_min[1]:.2f}, {wrench_max[1]:.2f}] N (width: {wrench_ranges[1]:.2f} N)")
                print(f"    Torque range: [{wrench_min[2]:.2f}, {wrench_max[2]:.2f}] N⋅m (width: {wrench_ranges[2]:.2f} N⋅m)")
                print(f"    Approximated volume: {volume}")
                print(f"    Feasible points: {len(feasible_wrenches)} / {len(wrench_data['wrenches'])}")
            else:
                print(f"  ⚠️ No feasible wrenches found for {obj_name}")
        
        print("\n" + "="*80)
        print("✅ WRENCH SPACE VISUALIZATION COMPLETE")
        print("="*80)
        
        return results
    
    def calculate_limit_surface(self, generic_object, resolution=50, scaling_factor=1.0, grid_size=30):
        """
        Calculate the limit surface (ellipsoid) for the object's friction model
        using numerical integration for accurate moment calculation.
        
        Args:
            generic_object: GenericObject to analyze
            resolution: Number of points to generate on ellipsoid surface
            scaling_factor: Optional scaling factor for visualization
            grid_size: Resolution for numerical integration
            
        Returns:
            dict: Contains surface points and parameters
        """
        # Extract object properties
        mass = generic_object.mass
        friction_coef = generic_object.static_friction  # Using static friction for limit surface
        normal_force = mass * 9.81  # Normal force (weight)
        
        # Calculate maximum friction force (pure translation)
        f_max = friction_coef * normal_force
        
        # Calculate maximum friction moment (pure rotation) using numerical integration
        m_max = self._calculate_max_moment_numerical(generic_object, friction_coef, normal_force, grid_size)
        
        # Scale if needed
        f_max *= scaling_factor
        m_max *= scaling_factor
        
        # Generate points on the ellipsoid surface
        # Parametric equations for ellipsoid
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        
        fx = f_max * np.outer(np.cos(u), np.sin(v))
        fy = f_max * np.outer(np.sin(u), np.sin(v))
        m = m_max * np.outer(np.ones_like(u), np.cos(v))
        
        return {
            'fx': fx,
            'fy': fy,
            'm': m,
            'f_max': f_max,
            'm_max': m_max,
            'friction_coef': friction_coef,
            'normal_force': normal_force
        }

    def _calculate_max_moment_numerical(self, generic_object, friction_coef, normal_force, grid_size=30):
        """
        Calculate maximum moment using numerical integration.
        
        Implementation of m_max = μ * ∫_A ||r||p(r) dA with uniform pressure.
        
        Args:
            generic_object: GenericObject instance
            friction_coef: Friction coefficient
            normal_force: Normal force
            grid_size: Grid resolution for numerical integration
            
        Returns:
            float: Maximum moment for limit surface
        """
        # Get object geometry and properties
        geometry = generic_object.geometry
        bounds = geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Object centroid
        centroid = generic_object.get_centroid()
        centroid_pos = np.array([centroid.x, centroid.y])
        
        # Create grid for numerical integration
        x_grid = np.linspace(bounds[0], bounds[2], grid_size)
        y_grid = np.linspace(bounds[1], bounds[3], grid_size)
        
        # Grid cell size
        dx = width / (grid_size - 1)
        dy = height / (grid_size - 1)
        cell_area = dx * dy
        
        # Total object area and uniform pressure
        total_area = generic_object.get_area()
        uniform_pressure = normal_force / total_area
        
        # Numerical integration
        moment_integral = 0.0
        
        for x in x_grid:
            for y in y_grid:
                point = Point(x, y)
                
                # Check if point is inside the object
                if geometry.contains(point):
                    # Distance from centroid
                    r_vector = np.array([x, y]) - centroid_pos
                    r_norm = np.linalg.norm(r_vector)
                    
                    # Contribution to moment integral: ||r|| * p * dA
                    moment_integral += r_norm * uniform_pressure * cell_area
        
        # Apply friction coefficient
        m_max = friction_coef * moment_integral
        
        return m_max
    
    def visualize_wrench_space_with_limit_surface(self, wrench_data, contact_points, 
                                                 generic_object, title="Wrench Space with Limit Surface"):
        """
        Visualize wrench space with the limit surface ellipsoid.
        
        Args:
            wrench_data: Output from calculate_wrench_space()
            contact_points: List of ContactPoint objects
            generic_object: GenericObject to calculate limit surface for
            title: Plot title
            
        Returns:
            plotly Figure object
        """
        # First create regular wrench space visualization
        fig = self.visualize_wrench_space_plotly(wrench_data, contact_points, title=title)
        
        # Calculate limit surface
        limit_surface = self.calculate_limit_surface(generic_object)
        
        # Add limit surface to 3D plot
        fig.add_trace(
            go.Surface(
                x=limit_surface['fx'],
                y=limit_surface['fy'],
                z=limit_surface['m'],
                opacity=0.3,
                colorscale='Blues',
                showscale=False,
                name="Limit Surface"
            ),
            row=1, col=1
        )
        
        # Add limit surface projections to 2D plots
        # For Fx-Fy projection (ellipse)
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = limit_surface['f_max'] * np.cos(theta)
        y_circle = limit_surface['f_max'] * np.sin(theta)
        
        fig.add_trace(
            go.Scatter(
                x=x_circle,
                y=y_circle,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Friction Limit (Fx-Fy)',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # For Fx-Torque projection (ellipse)
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        x_ellipse = limit_surface['f_max'] * np.cos(theta)
        z_ellipse = limit_surface['m_max'] * np.sin(theta)
        
        fig.add_trace(
            go.Scatter(
                x=x_ellipse,
                y=z_ellipse,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Friction Limit (Fx-τ)',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # For Fy-Torque projection (ellipse)
        fig.add_trace(
            go.Scatter(
                x=x_ellipse,  # Same shape as Fx-Torque
                y=z_ellipse,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Friction Limit (Fy-τ)',
                showlegend=False
            ),
            row=3, col=2
        )
        
        # Add legend entry for limit surface
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color='blue'),
                name='Limit Surface'
            ),
            row=1, col=1
        )
        
        # Add explanatory annotation
        fig.add_annotation(
            text=f"Friction Limit Surface<br>μ={limit_surface['friction_coef']:.2f}, f_max={limit_surface['f_max']:.2f}N, m_max={limit_surface['m_max']:.2f}Nm",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue",
            borderwidth=1,
            borderpad=4
        )
        
        return fig

# %%
class EdgeCharacterizer:
    """
    Comprehensive system for characterizing edge capabilities in wrench space.
    Includes physics-based edge decomposition and visualization.
    """
    
    def __init__(self, generic_object, force_magnitude=1.0):
        """
        Initialize edge characterizer for a specific object.
        
        Args:
            generic_object: GenericObject instance
            force_magnitude: Fixed normal force magnitude for characterization
        """
        self.object = generic_object
        self.force_magnitude = force_magnitude
        self.parameterization = ContactPointParameterization(generic_object)
        self.calculator = GenericContactCalculator(generic_object)
        
        # Identify edges
        self.edges = self.identify_edges()
        
        # Store edge characteristics (results)
        self.edge_characteristics = {}
        
        # Pre-compute edge matrices and characteristics
        self._characterize_edges()
    
    def identify_edges(self, min_edge_length=0.05):
        """
        Identify distinct edges of the object boundary.

        Segments shorter than `min_edge_length` are not dropped; they are merged
        into the next sufficiently long edge along the boundary (trailing shorts
        at the end of the loop attach to the last edge).

        The underlying polygon parameterization is unchanged — only the logical
        edge grouping used for characterization and sampling is affected.

        Args:
            min_edge_length: Minimum length for a segment to start a new edge
                chain on its own. Shorter segments are absorbed into a neighbor.

        Returns:
            list: Edge dicts with start_param, end_param, length, segment_index,
                and segment_indices (all raw segment indices in the chain).
        """
        param = self.parameterization
        n_segments = param.n_segments
        lengths = param.segment_lengths
        cumulative = param.cumulative_distances
        total_length = param.total_length

        if n_segments == 0 or total_length <= 0:
            return []

        chains = []
        pending_short = []

        for seg_idx in range(n_segments):
            seg_len = lengths[seg_idx]
            if seg_len >= min_edge_length:
                chains.append(pending_short + [seg_idx])
                pending_short = []
            else:
                pending_short.append(seg_idx)

        if pending_short:
            if chains:
                chains[-1].extend(pending_short)
            else:
                chains.append(pending_short)

        edges = []
        for edge_id, chain in enumerate(chains):
            start_dist = cumulative[chain[0]]
            end_dist = cumulative[chain[-1]] + lengths[chain[-1]]
            edge_length = end_dist - start_dist

            edges.append({
                'edge_id': edge_id,
                'start_param': start_dist / total_length,
                'end_param': end_dist / total_length,
                'length': edge_length,
                'segment_index': chain[0],
                'segment_indices': chain,
            })

        return edges
    
    def _characterize_edges(self):
        """
        Characterize each edge by analyzing its normal and tangent force properties.
        Collects and stores comprehensive edge characteristics.
        """
        print(f"🔍 Characterizing {len(self.edges)} edges with physics-based approach...")
        
        # Pre-compute characteristics for each edge
        for i, edge in enumerate(self.edges):
            edge_name = f'edge_{i}'
            
            # Analyze normal force characteristics (get slope, offset, etc.)
            normal_analysis = self._analyze_edge_forces(edge, force_type='normal')
            
            # Analyze tangent force characteristics
            tangent_analysis = self._analyze_edge_forces(edge, force_type='tangent')
            
            # Extract key parameters from normal force analysis
            wrench_array = normal_analysis['wrench_array']
            fx_values = wrench_array[:, 0]
            fy_values = wrench_array[:, 1]
            torque_values = wrench_array[:, 2]
            
            # Verify that force components are indeed constant
            fx_std = np.std(fx_values)
            fy_std = np.std(fy_values)
            
            # Fixed force components (take mean for robustness)
            fixed_fx = np.mean(fx_values)
            fixed_fy = np.mean(fy_values)
            
            # Torque range
            torque_min = np.min(torque_values)
            torque_max = np.max(torque_values)
            torque_range = torque_max - torque_min
            
            # Store comprehensive edge characteristics
            self.edge_characteristics[edge_name] = {
                'edge_info': edge,
                'edge_index': i,
                'fixed_fx': fixed_fx,
                'fixed_fy': fixed_fy,
                'torque_min': torque_min,
                'torque_max': torque_max,
                'torque_range': torque_range,
                'torque_slope': normal_analysis['torque_slope'],
                'torque_offset': normal_analysis['torque_offset'],
                'fx_std': fx_std,
                'fy_std': fy_std,
                'is_force_constant': (fx_std < 1e-6) and (fy_std < 1e-6),
                'wrench_array': wrench_array,
                'contact_points': normal_analysis['contact_points'],
                'tangent_fx': tangent_analysis['tangent_fx'],
                'tangent_fy': tangent_analysis['tangent_fy'],
                'tangent_torque': tangent_analysis['tangent_torque']
            }
            
            print(f"  {edge_name}: Fixed force=[{fixed_fx:.3f}, {fixed_fy:.3f}], "
                  f"Torque range=[{torque_min:.3f}, {torque_max:.3f}], "
                  f"Slope={normal_analysis['torque_slope']:.4f}, "
                  f"Offset={normal_analysis['torque_offset']:.4f}")
    
    def _analyze_edge_forces(self, edge_info, force_type='normal', n_samples=20):
        """
        Unified method to analyze edge forces (normal or tangent).
        
        Args:
            edge_info: Edge dictionary from identify_edges()
            force_type: Type of force to analyze ('normal' or 'tangent')
            n_samples: Number of contact points to sample along edge
            
        Returns:
            dict: Edge force analysis results
        """
        edge_id = edge_info['edge_id']
        start_param = edge_info['start_param']
        end_param = edge_info['end_param']
        
        # Sample contact points along the edge
        if n_samples == 1:
            sample_params = [0.5 * (start_param + end_param)]
        else:
            sample_params = np.linspace(start_param + 0.01, end_param - 0.01, n_samples)
        
        wrench_list = []
        contact_points = []
        
        for param in sample_params:
            # Get contact point at this parameter
            contact_info = self.parameterization.get_contact_info(param)
            
            contact_point = ContactPoint(
                position=contact_info['point'],
                tangent=contact_info['tangent'],
                normal_outward=contact_info['normal_outward'],
                normal_inward=contact_info['normal_inward'],
                parameter=param,
                object_ref=self.object
            )
            
            # Set force components based on force_type
            if force_type == 'normal':
                normal_force = self.force_magnitude
                tangent_force = 0.0
            else:  # tangent
                normal_force = 0.0
                tangent_force = self.force_magnitude
            
            # Calculate wrench
            wrench = contact_point.calculate_contact_wrench(
                normal_force=normal_force,
                tangential_force=tangent_force,
                friction_constraint=(force_type == 'normal')  # Only apply friction constraint for normal forces
            )
            
            wrench_vector = np.array([
                wrench['force_x'],
                wrench['force_y'], 
                wrench['torque']
            ])
            
            wrench_list.append(wrench_vector)
            contact_points.append(contact_point)
        
        wrench_array = np.array(wrench_list)
        
        # Prepare result structure
        result = {
            'wrench_array': wrench_array,
            'contact_points': contact_points,
            'sample_params': sample_params,
            'force_type': force_type,
            'force_magnitude': self.force_magnitude
        }
        
        # For normal forces, calculate torque slope and offset
        if force_type == 'normal':
            # Calculate torque slope and offset: τ = slope * t + offset
            params = [cp.parameter for cp in contact_points]
            torques = wrench_array[:, 2]
            
            if len(params) >= 2:
                # Linear regression: τ = slope*t + offset
                A = np.vstack([params, np.ones(len(params))]).T
                slope, offset = np.linalg.lstsq(A, torques, rcond=None)[0]
                result['torque_slope'] = slope
                result['torque_offset'] = offset
            else:
                # Default for single point
                result['torque_slope'] = 0.0
                result['torque_offset'] = torques[0] if len(torques) > 0 else 0.0
        
        # For tangent forces, calculate representative values (at edge midpoint)
        if force_type == 'tangent':
            # Use midpoint value as representative of tangent wrench
            mid_idx = len(wrench_array) // 2
            if len(wrench_array) > 0:
                result['tangent_fx'] = wrench_array[mid_idx, 0]
                result['tangent_fy'] = wrench_array[mid_idx, 1]
                result['tangent_torque'] = wrench_array[mid_idx, 2]
            else:
                result['tangent_fx'] = 0.0
                result['tangent_fy'] = 0.0
                result['tangent_torque'] = 0.0
        
        return result

    def visualize_edge_signatures(self, ax=None, show_edge_boundaries=True, force_type='normal'):
        """
        Visualize the signature signals (Fx, Fy, τ) for all edges.
        Shows how force components vary along edges.
        
        Args:
            ax: Optional axes array [3 subplots] for Fx, Fy, Torque
            show_edge_boundaries: If True, show vertical lines at edge boundaries
            force_type: Type of force to visualize ('normal' or 'tangent')
            
        Returns:
            fig, axes: Figure and axes array
        """
        if ax is None:
            fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        else:
            axes = ax
            fig = axes[0].figure
        
        # Colors for different edges
        edge_colors = plt.cm.tab10(np.linspace(0, 1, len(self.edges))) if force_type == 'normal' else \
                    plt.cm.tab20(np.linspace(0, 1, len(self.edges)))
        
        # Collect all data for global parameter space [0, 1]
        all_params = []
        all_fx = []
        all_fy = []
        all_torque = []
        edge_boundaries = [0]  # Start with boundary at 0
        
        for i, edge in enumerate(self.edges):
            edge_name = f'edge_{i}'
            char = self.edge_characteristics[edge_name]
            
            # Get edge parameter range
            start_param = char['edge_info']['start_param']
            end_param = char['edge_info']['end_param']
            edge_boundaries.append(end_param)
            
            # Sample more densely within this edge for smooth visualization
            n_dense_samples = 50
            edge_params = np.linspace(start_param, end_param, n_dense_samples)
            
            # Generate detailed wrench data for this edge
            edge_fx = []
            edge_fy = []
            edge_torque = []
            
            for param in edge_params:
                # Get contact point at this parameter
                contact_info = self.parameterization.get_contact_info(param)
                
                contact_point = ContactPoint(
                    position=contact_info['point'],
                    tangent=contact_info['tangent'],
                    normal_outward=contact_info['normal_outward'],
                    normal_inward=contact_info['normal_inward'],
                    parameter=param,
                    object_ref=self.object
                )
                
                # Calculate wrench based on force_type
                if force_type == 'normal':
                    wrench = contact_point.calculate_contact_wrench(
                        normal_force=self.force_magnitude,
                        tangential_force=0.0,
                        friction_constraint=True
                    )
                else:  # tangent
                    wrench = contact_point.calculate_contact_wrench(
                        normal_force=0.0,
                        tangential_force=self.force_magnitude,
                        friction_constraint=False
                    )
                
                edge_fx.append(wrench['force_x'])
                edge_fy.append(wrench['force_y'])
                edge_torque.append(wrench['torque'])
            
            # Store for global plot
            all_params.extend(edge_params)
            all_fx.extend(edge_fx)
            all_fy.extend(edge_fy)
            all_torque.extend(edge_torque)
            
            # Plot each edge with different color
            color = edge_colors[i]
            
            # Force X subplot
            axes[0].plot(edge_params, edge_fx, 'o-', color=color, 
                        linewidth=2, markersize=3, alpha=0.8,
                        label=f'Edge {i} (Fx={np.mean(edge_fx):.3f})')
            
            # Force Y subplot  
            axes[1].plot(edge_params, edge_fy, 's-', color=color,
                        linewidth=2, markersize=3, alpha=0.8,
                        label=f'Edge {i} (Fy={np.mean(edge_fy):.3f})')
            
            # Torque subplot
            axes[2].plot(edge_params, edge_torque, '^-', color=color,
                        linewidth=2, markersize=3, alpha=0.8,
                        label=f'Edge {i} (τ∈[{np.min(edge_torque):.3f}, {np.max(edge_torque):.3f}])')
        
        # Add edge boundaries
        if show_edge_boundaries:
            for boundary in edge_boundaries[1:-1]:  # Skip first (0) and last (1)
                for ax_i in axes:
                    ax_i.axvline(x=boundary, color='red', linestyle='--', 
                            alpha=0.7, linewidth=1.5)
        
        # Set labels and formatting based on force type
        force_prefix = "Tangent " if force_type == 'tangent' else ""
        
        axes[0].set_ylabel(f'{force_prefix}Force X (N)', fontsize=11, fontweight='bold')
        title_text = f'{force_prefix}Force Signatures - {self.object.name}'
        if force_type == 'normal':
            title_text += f'\nFixed Normal Force: {self.force_magnitude} N'
        else:
            title_text += f'\nFixed Tangent Force: {self.force_magnitude} N'
        axes[0].set_title(title_text, fontsize=12, fontweight='bold')
        
        axes[1].set_ylabel(f'{force_prefix}Force Y (N)', fontsize=11, fontweight='bold')
        
        axes[2].set_ylabel(f'{force_prefix}Torque (N⋅m)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Contact Point Parameter (t)', fontsize=11, fontweight='bold')
        
        # Make legends more compact
        for ax_i in axes:
            ax_i.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
            ax_i.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    

# %%
def demonstrate_restructured_system():
    """
    Demonstrate the restructured object library system with comprehensive visualizations.
    """
    print("\n" + "="*80)
    print("🚀 DEMONSTRATING RESTRUCTURED OBJECT LIBRARY")
    print("="*80)
    
    # Test with standard objects
    print("\n1️⃣ Creating standard objects...")
    standard_objects = create_standard_objects()
    
    # Select an object for demonstration
    obj = standard_objects['l_shape']
    print(f"\n2️⃣ Selected object: {obj.name}")
    
    # Step 2: Contact point parameterization
    print("\n3️⃣ Creating contact parameterization...")
    parameterization = ContactPointParameterization(obj)
    
    # Step 3: Generate contact points and compute wrenches
    print("\n4️⃣ Calculating contact points...")
    calculator = GenericContactCalculator(obj)
    contact_points = calculator.calculate_contact_points(n_contacts=3, strategy='uniform')
    
    # Calculate wrench matrix using GraspMatrixCalculator
    print("\n5️⃣ Building wrench matrix...")
    wrench_matrix = GraspMatrixCalculator.build_wrench_matrix(contact_points)
    
    print(f"Wrench matrix shape: {wrench_matrix.shape}")
    print("Wrench matrix:")
    print(wrench_matrix)
    
    # Test wrench calculation
    normal_forces = [1.0, 2.0, 1.5]
    print(f"\n6️⃣ Computing total wrench with forces: {normal_forces}")
    total_wrench = GraspMatrixCalculator.compute_total_wrench(wrench_matrix, normal_forces)
    
    print(f"Total wrench: {total_wrench}")
    
    # Step 4: Edge characterization
    print("\n7️⃣ Edge characterization...")
    edge_char = EdgeCharacterizer(obj, force_magnitude=2.0)
    
    print(f"Number of edges: {len(edge_char.edges)}")
    print("Edge characteristics:")
    for edge_name, char in edge_char.edge_characteristics.items():
        print(f"  {edge_name}: Fx={char['fixed_fx']:.3f}, Fy={char['fixed_fy']:.3f}")
        print(f"    Torque range: [{char['torque_min']:.3f}, {char['torque_max']:.3f}]")
        print(f"    Torque model: τ = {char['torque_slope']:.4f}*t + {char['torque_offset']:.4f}")
    
    # Visualize contact solution with GraspMatrixCalculator
    print("\n8️⃣ Visualizing contact solution...")
    fig, ax = plt.subplots(figsize=(10, 8))
    calculator.visualize_contact_solution(contact_points, normal_forces, ax=ax)
    plt.tight_layout()
    plt.show()
    
    # Visualize edge signatures - demonstrating both normal and tangent forces
    print("\n9️⃣ Visualizing edge signatures...")
    print("   a) Normal force signatures:")
    fig_normal, axes_normal = edge_char.visualize_edge_signatures(force_type='normal')
    plt.tight_layout()
    plt.show()
    
    print("   b) Tangent force signatures:")
    fig_tangent, axes_tangent = edge_char.visualize_edge_signatures(force_type='tangent')
    plt.tight_layout()
    plt.show()
    
    return {
        'object': obj,
        'contact_points': contact_points,
        'wrench_matrix': wrench_matrix,
        'edge_characterizer': edge_char,
        'normal_signature_fig': fig_normal,
        'tangent_signature_fig': fig_tangent
    }


# %%
class PlaceholderController:
    """
    Base controller class that interfaces between simulation and specific control algorithms.
    This serves as a placeholder that can be extended for various control strategies.
    """
    
    def __init__(self, object_model):
        """
        Initialize controller with reference to the object model.
        
        Args:
            object_model: DynamicObjectModel instance
        """
        self.object_model = object_model
        self.time = 0.0
        self.dt = 0.01  # Default time step
        self.contact_points = []
        self.force_magnitudes = []
        self.state_history = []
        self.data_history = []
        self.data_history_length = 50  # Number of recent data points to keep
        
        # Control history storage (optional)
        self.control_history_save = False
        self.contact_points_history = []
        self.force_magnitudes_history = []
        
    def initialize(self, **kwargs):
        """
        Initialize controller with specified parameters.
        Derived controllers can override this for custom initialization.
        
        Args:
            **kwargs: Arbitrary keyword arguments for controller initialization
                     (e.g., contact_points, initial_state, etc.)
        
        Returns:
            list: Initial contact points
        """
        # Process contact points if provided
        if 'contact_points' in kwargs:
            self.contact_points = kwargs['contact_points']
        else:
            # Default: create a single contact point using calculator
            calculator = GenericContactCalculator(self.object_model.object)
            self.contact_points = calculator.calculate_contact_points(n_contacts=1, strategy='uniform')
            
        return self.contact_points
    
    def update(self, state, dt):
        """
        Update controller state based on current object state.
        This processes and stores state information but doesn't generate actions.
        
        Args:
            state: Dictionary with object state information
            dt: Time step
        """
        # Update internal time
        self.time += dt
        self.dt = dt
        
        # Store state history
        self.state_history.append(state)
        
        # Call internal update (for subclass-specific processing)
        self.update_internal()
    
    def update_internal(self):
        """
        Process internal controller state after an update.
        Derived controllers can override this for custom internal processing.
        """
        pass  # Default implementation does nothing
    
# NEW method to provide access to simulation data after state update

    def post_update(self, latest_data):
        """
        Process the latest simulation data after state update.
        This gives the controller access to the most recent simulation values.
        
        Args:
            latest_data: Dictionary containing the latest simulation values (not the entire history)
        """
        # Store the latest data directly as a simple dictionary
        if not hasattr(self, 'latest_simulation_data'):
            self.latest_simulation_data = {}
        
        # Update with latest values - simple assignment
        self.latest_simulation_data.update(latest_data)
        
        # Optionally maintain a short history if needed (e.g., last 10 values)
        if not hasattr(self, 'data_history'):
            self.data_history = []
        
        # Keep only recent history (optional)
        self.data_history.append(latest_data.copy())
        if len(self.data_history) > self.data_history_length:  # Keep only last N entries
            self.data_history.pop(0)
        
        # Call internal post-update processing
        self.post_update_internal(latest_data)

    def post_update_internal(self, latest_data):
        """
        Process internal controller state after post_update.
        Derived controllers can override this for custom processing.
        
        Args:
            latest_data: Dictionary containing the latest simulation values
        """
        pass  # Default implementation does nothing

    def get_control_actions(self):
        """
        Generate control actions based on current controller state.
        Returns contact_points and force_magnitudes for the next timestep.
        
        Returns:
            tuple: (contact_points, force_magnitudes)
        """
        # Default implementation returns current contact points with zero forces
        self.force_magnitudes = [0.0] * len(self.contact_points)
        
        # Optionally save control history
        if self.control_history_save:
            self.contact_points_history.append(self.contact_points.copy())
            self.force_magnitudes_history.append(self.force_magnitudes.copy())
        
        return self.contact_points, self.force_magnitudes
    
    def reset(self):
        """Reset controller state"""
        self.time = 0.0
        self.state_history = []
        
        # Reset control history if enabled
        if self.control_history_save:
            self.contact_points_history = []
            self.force_magnitudes_history = []

# %%
class DynamicObjectModel:
    """
    Dynamic model for simulating object motion under applied forces and torques.
    
    Implements the following dynamic model:
    ẋ = vx cos(θ) - vy sin(θ)
    ẏ = vx sin(θ) + vy cos(θ)
    θ̇ = ω
    v̇x = (1/m)(Fx)
    v̇y = (1/m)(Fy)
    ω̇ = (1/Iz)(τ)
    [Fx, Fy, τ]T = G · Fmag + μ
    
    With friction effects modeled using limit surface:
    (fx/f_max)^2 + (fy/f_max)^2 + (m/m_max)^2 = 1
    """
    
    def __init__(self, generic_object, friction_noise_std=0.01, position_init=None, orientation_init=0.0, dt=0.01):
        """
        Initialize the dynamic model for a generic object.
        
        Args:
            generic_object: GenericObject instance to simulate
            friction_noise_std: Standard deviation for friction/disturbance noise
            position_init: Initial position [x, y], default is object's current position
            orientation_init: Initial orientation in radians
        """
        self.object = generic_object
        self.mass = generic_object.mass
        self.moment_of_inertia = generic_object.moment_of_inertia
        self.friction_noise_std = friction_noise_std

        # State variables
        if position_init is None:
            centroid = generic_object.get_centroid()
            self.position = np.array([centroid.x, centroid.y])
        else:
            self.position = np.array(position_init)
            
        self.orientation = orientation_init
        self.velocity_body = np.array([0.0, 0.0])  # [vx, vy] in body frame
        self.angular_velocity = 0.0  # ω
        
        # Track trajectory for visualization
        self.trajectory = [self.position.copy()]
        self.orientation_history = [self.orientation]
        self.time_history = [0.0]

        # Cached grasp matrix
        self._grasp_matrix_cache = {}
        
        # Instrument for debugging
        self.last_s_value = 0
        self.last_twist_magnitude = 0.0

        # Current simulation time
        self.time = 0.0
        # Add path parameter for optional contour following
        self.path_param = 0.0

        self.dt = dt
        # Pre-calculate limit surface parameters for both static and kinetic friction
        self._precalculate_limit_surfaces()
        
    def reset_state(self, position=None, orientation=0.0, velocity=None, angular_velocity=0.0):
        """
        Reset/set the object state to specified values.
        
        Args:
            position: [x, y] position, default keeps current position
            orientation: Orientation in radians, default is 0.0
            velocity: [vx, vy] body frame velocity, default is [0.0, 0.0]
            angular_velocity: Angular velocity in rad/s, default is 0.0
        """
        # Set position
        if position is not None:
            self.position = np.array(position, dtype=float)
        
        # Set orientation
        self.orientation = float(orientation)
        
        # Set body frame velocity
        if velocity is not None:
            self.velocity_body = np.array(velocity, dtype=float)
        else:
            self.velocity_body = np.array([0.0, 0.0])
        
        # Set angular velocity
        self.angular_velocity = float(angular_velocity)
        
        # Reset time and path parameter
        self.time = 0.0
        if hasattr(self, 'path_param'):
            self.path_param = 0.0
        
        # Clear trajectory history and restart
        self.trajectory = [self.position.copy()]
        self.orientation_history = [self.orientation]
        self.time_history = [0.0]
        
        print(f"🔄 State reset: pos=({self.position[0]:.3f}, {self.position[1]:.3f}), "
            f"θ={np.degrees(self.orientation):.1f}°, "
            f"vel=({self.velocity_body[0]:.3f}, {self.velocity_body[1]:.3f}), "
            f"ω={np.degrees(self.angular_velocity):.1f}°/s")

    def _precalculate_limit_surfaces(self):
        """Pre-calculate limit surface parameters for efficient computation."""
        # Normal force (weight)
        self.normal_force = self.mass * 9.81
        
        # Static friction parameters
        self.static_f_max = self.object.static_friction * self.normal_force
        self.static_m_max = self._calculate_max_moment(self.object.static_friction)
        self.static_c = self.static_m_max / self.static_f_max
        
        # Kinetic friction parameters
        self.kinetic_f_max = self.object.kinetic_friction * self.normal_force
        self.kinetic_m_max = self._calculate_max_moment(self.object.kinetic_friction)
        self.kinetic_c = self.kinetic_m_max / self.kinetic_f_max
        
        print(f"Limit surface parameters pre-calculated:")
        print(f"Static: f_max={self.static_f_max:.3f}N, m_max={self.static_m_max:.3f}Nm, c={self.static_c:.3f}")
        print(f"Kinetic: f_max={self.kinetic_f_max:.3f}N, m_max={self.kinetic_m_max:.3f}Nm, c={self.kinetic_c:.3f}")
    
    def _calculate_max_moment(self, friction_coef):
        """
        Calculate maximum friction moment using numerical integration.
        Reuses the approach from WrenchSpaceVisualizer.
        
        Args:
            friction_coef: Friction coefficient to use
            
        Returns:
            float: Maximum moment for limit surface
        """
        # Get object geometry and properties
        geometry = self.object.geometry
        bounds = geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Object centroid
        centroid = self.object.get_centroid()
        centroid_pos = np.array([centroid.x, centroid.y])
        
        # For simple shapes, use analytical approximation
        # For more complex shapes, we would use the numerical integration from WrenchSpaceVisualizer
        if hasattr(geometry, 'exterior') and len(list(geometry.exterior.coords)) > 8:
            # Complex shape - use numerical integration with grid
            grid_size = 30
            
            # Create grid for numerical integration
            x_grid = np.linspace(bounds[0], bounds[2], grid_size)
            y_grid = np.linspace(bounds[1], bounds[3], grid_size)
            
            # Grid cell size
            dx = width / (grid_size - 1)
            dy = height / (grid_size - 1)
            cell_area = dx * dy
            
            # Total object area and uniform pressure
            total_area = self.object.get_area()
            uniform_pressure = self.normal_force / total_area
            
            # Numerical integration
            moment_integral = 0.0
            
            for x in x_grid:
                for y in y_grid:
                    point = Point(x, y)
                    
                    # Check if point is inside the object
                    if geometry.contains(point):
                        # Distance from centroid
                        r_vector = np.array([x, y]) - centroid_pos
                        r_norm = np.linalg.norm(r_vector)
                        
                        # Contribution to moment integral: ||r|| * p * dA
                        moment_integral += r_norm * uniform_pressure * cell_area
            
            # Apply friction coefficient
            m_max = friction_coef * moment_integral
            
        else:
            # Simple shape - use analytical approximation
            # For rectangle/circle approximation: r_equivalent = sqrt(width^2 + height^2) / 4
            r_eq = np.sqrt(width**2 + height**2) / 4
            m_max = friction_coef * self.normal_force * r_eq
        
        return m_max
    
    def predict_next_state(self, state_vector, contour_param, contact_points, force_magnitudes, contour_speed, dt=0.01, friction_enabled=True, include_noise=False):
        """
        Predict next state given current state vector and applied forces.
        
        Args:
            state_vector: Current state vector [x, y, θ, vx_body, vy_body, ω]
            contour_param: Current contour/path parameter
            contact_points: List of ContactPoint objects
            force_magnitudes: List of force magnitudes or callable f(t, state)
            contour_speed: Speed along the contour/path
            dt: Time step size
            friction_enabled: Whether to apply friction model
            include_noise: Whether to include noise in the simulation
        Returns:
            np.array: Next state vector [x, y, θ, vx_body, vy_body, ω, contour_param]
        """
        
        # Unpack state vector
        x, y, theta, vx_b, vy_b, omega = state_vector
        
        # Calculate applied wrench (in body frame)
        applied_wrench = self._calculate_total_wrench(contact_points, force_magnitudes)
        
        # Friction and optional noise
        friction_wrench = self._calculate_friction(applied_wrench=applied_wrench, injected_velocity=(vx_b, vy_b, omega)) if friction_enabled else np.zeros(3)
        noise = np.random.normal(0, self.friction_noise_std, 3) if include_noise else np.zeros(3)
        
        # Add noise if specified
        noise = np.random.normal(0, self.friction_noise_std, 3) if include_noise else np.zeros(3)
        
        total_wrench = applied_wrench + friction_wrench + noise
        Fx, Fy, tau = total_wrench

        # Integrate velocities (body frame) and angular velocity
        vx_b_next = vx_b + (Fx / self.mass) * dt
        vy_b_next = vy_b + (Fy / self.mass) * dt
        omega_next = omega + (tau / self.moment_of_inertia) * dt

        # Integrate pose using previous velocities/orientation
        cos_th, sin_th = np.cos(theta), np.sin(theta)
        x_next = x + dt * (vx_b * cos_th - vy_b * sin_th)
        y_next = y + dt * (vx_b * sin_th + vy_b * cos_th)
        theta_next = theta + omega * dt
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))  # normalize

        # Optional contour/path parameter update
        if contour_speed is not None:
            contour_param = max(0.0, min(1.0, contour_param + contour_speed * dt))

        return np.array([x_next, y_next, theta_next, vx_b_next, vy_b_next, omega_next, contour_param])
        
    def update_state(self, contact_points, force_magnitudes, dt=0.01, friction_enabled=True, contour_speed=None, include_noise=False):
        """
        Explicit-Euler state update using previous-state snapshot.
        """
        # Snapshot previous state
        x_prev, y_prev = float(self.position[0]), float(self.position[1])
        theta_prev = float(self.orientation)
        vx_b_prev, vy_b_prev = float(self.velocity_body[0]), float(self.velocity_body[1])
        omega_prev = float(self.angular_velocity)

        # Wrench from contacts
        applied_wrench = self._calculate_total_wrench(contact_points, force_magnitudes)

        # Friction and optional noise
        friction_wrench = self._calculate_friction(applied_wrench = applied_wrench) if friction_enabled else np.zeros(3)
        noise = np.random.normal(0, self.friction_noise_std, 3) if include_noise else np.zeros(3)

        total_wrench = applied_wrench + friction_wrench + noise
        Fx, Fy, tau = total_wrench

        # Integrate velocities (body frame) and angular velocity
        vx_b_next = vx_b_prev + (Fx / self.mass) * dt
        vy_b_next = vy_b_prev + (Fy / self.mass) * dt
        omega_next = omega_prev + (tau / self.moment_of_inertia) * dt

        # Integrate pose using previous velocities/orientation
        cos_th, sin_th = np.cos(theta_prev), np.sin(theta_prev)
        x_next = x_prev + dt * (vx_b_prev * cos_th - vy_b_prev * sin_th)
        y_next = y_prev + dt * (vx_b_prev * sin_th + vy_b_prev * cos_th)
        theta_next = theta_prev + omega_prev * dt
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))  # normalize

        # Commit next state
        self.velocity_body = np.array([vx_b_next, vy_b_next])
        self.angular_velocity = omega_next
        self.position = np.array([x_next, y_next])
        self.orientation = theta_next
        self.time += dt

        # Optional contour/path parameter update
        if contour_speed is not None:
            self.path_param = max(0.0, min(1.0, self.path_param + contour_speed * dt))

        # History
        self.trajectory.append(self.position.copy())
        self.orientation_history.append(self.orientation)
        self.time_history.append(self.time)

        return {
            'position': self.position.copy(),
            'orientation': self.orientation,
            'velocity_body': self.velocity_body.copy(),
            'angular_velocity': self.angular_velocity,
            'time': self.time,
            'applied_wrench': applied_wrench,
            'friction_wrench': friction_wrench,
            'total_wrench': total_wrench,
            'path_param': getattr(self, 'path_param', None)
        }

    def update_state_rk(self, contact_points, force_magnitudes, dt=0.01, friction_enabled=True, contour_speed=None, include_noise=False):
        """
        Higher precision state update using 4th order Runge-Kutta integration.
        
        RK4 divides each timestep into four evaluations and weighs them to achieve 
        higher accuracy than simple Euler integration.
        
        Args:
            contact_points: List of ContactPoint objects
            force_magnitudes: List of force magnitudes or callable f(t, state)
            dt: Time step size
            friction_enabled: Whether to apply friction model
            contour_speed: Optional speed for path parameter update
            include_noise: Whether to include noise in the simulation
            
        Returns:
            dict: Updated state information
        """
        # Snapshot current state into a state vector for RK4
        x, y = float(self.position[0]), float(self.position[1])
        theta = float(self.orientation)
        vx_b, vy_b = float(self.velocity_body[0]), float(self.velocity_body[1])
        omega = float(self.angular_velocity)
        
        # Create state vector [x, y, θ, vx_body, vy_body, ω]
        state = np.array([x, y, theta, vx_b, vy_b, omega])
        
        # For noise, generate one sample for the entire step (if needed)
        noise = np.random.normal(0, self.friction_noise_std, 3) if include_noise else np.zeros(3)
        
        # Define a derivative function for RK4 
        def state_derivative(t, s, f_mag):
            # Extract state components
            sx, sy, stheta, svx_b, svy_b, somega = s
            
            # Force magnitudes might be time/state dependent
            if callable(f_mag):
                current_forces = f_mag(t, 
                                     np.array([sx, sy]),
                                     stheta,
                                     np.array([svx_b, svy_b]), 
                                     somega)
            else:
                current_forces = f_mag
                
            # Calculate applied wrench (in body frame)
            applied_wrench = self._calculate_total_wrench(contact_points, current_forces)
            
            # Add friction (in body frame)
            if friction_enabled:
                # Create a temporary state for friction calculation
                temp_state = state.copy()
                temp_state[0:6] = s[0:6]
                
                # Store current values to restore after friction calculation
                original_pos = self.position.copy()
                original_orient = self.orientation
                original_vel = self.velocity_body.copy()
                original_omega = self.angular_velocity
                
                # Set temporary state for friction calculation
                self.position = np.array([sx, sy])
                self.orientation = stheta
                self.velocity_body = np.array([svx_b, svy_b])
                self.angular_velocity = somega
                
                # Calculate friction wrench based on this temporary state
                friction_wrench = self._calculate_friction(applied_wrench)
                
                # Restore original state
                self.position = original_pos
                self.orientation = original_orient
                self.velocity_body = original_vel
                self.angular_velocity = original_omega
            else:
                friction_wrench = np.zeros(3)
            
            # Total wrench with noise
            total_wrench = applied_wrench + friction_wrench + noise
            Fx_b, Fy_b, tau = total_wrench
            
            # State derivatives
            cos_th, sin_th = np.cos(stheta), np.sin(stheta)
            
            # Position derivatives (in world frame)
            dx = svx_b * cos_th - svy_b * sin_th
            dy = svx_b * sin_th + svy_b * cos_th
            
            # Orientation derivative
            dtheta = somega
            
            # Velocity derivatives (in body frame)
            dvx_b = Fx_b / self.mass
            dvy_b = Fy_b / self.mass
            
            # Angular acceleration
            domega = tau / self.moment_of_inertia
            
            return np.array([dx, dy, dtheta, dvx_b, dvy_b, domega])
        
        # RK4 integration
        k1 = state_derivative(self.time, state, force_magnitudes)
        k2 = state_derivative(self.time + 0.5*dt, state + 0.5*dt*k1, force_magnitudes)
        k3 = state_derivative(self.time + 0.5*dt, state + 0.5*dt*k2, force_magnitudes)
        k4 = state_derivative(self.time + dt, state + dt*k3, force_magnitudes)
        
        # Weighted sum for next state
        state_next = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Update actual state variables
        x_next, y_next, theta_next, vx_b_next, vy_b_next, omega_next = state_next
        
        # Normalize angle to [-π, π]
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        # Calculate applied wrench at final state (for return info)
        if callable(force_magnitudes):
            final_forces = force_magnitudes(self.time + dt, 
                                         np.array([x_next, y_next]),
                                         theta_next,
                                         np.array([vx_b_next, vy_b_next]), 
                                         omega_next)
        else:
            final_forces = force_magnitudes
        
        applied_wrench = self._calculate_total_wrench(contact_points, final_forces)
        
        # Calculate friction at final state
        self.position = np.array([x_next, y_next])
        self.orientation = theta_next
        self.velocity_body = np.array([vx_b_next, vy_b_next])
        self.angular_velocity = omega_next
        
        friction_wrench = self._calculate_friction(applied_wrench) if friction_enabled else np.zeros(3)
        total_wrench = applied_wrench + friction_wrench + noise
        
        # Update time
        self.time += dt
        
        # Optional contour/path parameter update
        if contour_speed is not None:
            self.path_param = max(0.0, min(1.0, self.path_param + contour_speed * dt))
        
        # History
        self.trajectory.append(self.position.copy())
        self.orientation_history.append(self.orientation)
        self.time_history.append(self.time)
        
        return {
            'position': self.position.copy(),
            'orientation': self.orientation,
            'velocity_body': self.velocity_body.copy(),
            'angular_velocity': self.angular_velocity,
            'time': self.time,
            'applied_wrench': applied_wrench,
            'friction_wrench': friction_wrench,
            'total_wrench': total_wrench,
            'path_param': getattr(self, 'path_param', None)
        }
    
    def _calculate_total_wrench(self, contact_points, force_magnitudes):
        """
        Calculate total wrench from contact points and force magnitudes.
        [Fx, Fy, τ]T = G · Fmag
        
        Args:
            contact_points: List of ContactPoint objects
            force_magnitudes: List of force magnitudes
            
        Returns:
            numpy.ndarray: Total wrench [Fx, Fy, τ]
        """
        # Build the grasp matrix (G)
        key = tuple(cp.parameter for cp in contact_points)
        if key not in self._grasp_matrix_cache:
            self._grasp_matrix_cache[key] = GraspMatrixCalculator.build_wrench_matrix(contact_points)
        G = self._grasp_matrix_cache[key]
        
        # Calculate wrench: G · Fmag
        return G @ np.array(force_magnitudes)
 
    
    def _calculate_friction(self, applied_wrench, injected_velocity=None):
        """
        Friction via limit surface (LOCAL frame).
        - Static: if twist ~ 0 and wrench inside LS => fully cancel; if outside => saturate on LS.
        - Kinetic: oppose current velocity direction, scaled to LS.
        """
        fx, fy, m = map(float, applied_wrench)

        # Use c to weight angular rate in regime check (LOCAL twist)
        # Prefer static parameters for thresholding; guard small denominators.
        f_max_s = max(1e-9, self.static_f_max)
        m_max_s = max(1e-9, self.static_m_max)
        c_static = m_max_s / f_max_s

        if injected_velocity is None:
            vx_b, vy_b = float(self.velocity_body[0]), float(self.velocity_body[1])
            omega = float(self.angular_velocity)
        else:
            vx_b, vy_b, omega = injected_velocity
        
        # Calculate twist magnitude with proper scaling
        twist_magnitude = np.sqrt(vx_b**2 + vy_b**2 + (omega * c_static)**2)
        self.last_twist_magnitude = twist_magnitude
        threshold_velocity = 1e-2

        if twist_magnitude < threshold_velocity:
            # print("Static friction regime engaged.")
            
            # Special case: Very small velocity with no meaningful applied wrench
            # We should stop the object completely
            if np.linalg.norm(applied_wrench) < 1e-3 and twist_magnitude > 1e-6:
                # print("  Very small velocity with no applied wrench - applying stopping friction")
                
                # Calculate friction needed to stop the object in this timestep
                # friction = -(v * mass / dt)
                friction_fx_stop = -(vx_b * self.mass / self.dt)
                friction_fy_stop = -(vy_b * self.mass / self.dt)
                friction_m_stop = -(omega * self.moment_of_inertia / self.dt)
                
                friction_wrench_stop = np.array([friction_fx_stop, friction_fy_stop, friction_m_stop])
                
                # Verify this stopping friction is within static limit surface
                f_max = max(1e-9, self.static_f_max)
                m_max = max(1e-9, self.static_m_max)
                
                fx_scaled = friction_wrench_stop[0] / f_max
                fy_scaled = friction_wrench_stop[1] / f_max
                m_scaled = friction_wrench_stop[2] / m_max
                s_stop = np.sqrt(fx_scaled**2 + fy_scaled**2 + m_scaled**2)
                
                self.last_s_value = s_stop
                
                if s_stop <= 1.0 + 1e-12:
                    # Stopping friction is within static limit - use it to stop completely
                    # print(f"  Stopping friction within static limit (s={s_stop:.3f}) - object will stop")
                    return friction_wrench_stop - applied_wrench
                else:
                    # Stopping friction exceeds static limit - saturate to limit surface
                    # print(f"  Stopping friction exceeds static limit (s={s_stop:.3f}) - saturating")
                    return friction_wrench_stop / s_stop
            
            # Normal static regime: check if applied wrench exceeds static friction
            f_max = max(1e-9, self.static_f_max)
            m_max = max(1e-9, self.static_m_max)
            
            s = np.sqrt((fx/f_max)**2 + (fy/f_max)**2 + (m/m_max)**2)
            self.last_s_value = s
            
            if s <= 1.0 + 1e-12:
                # Inside LS: fully cancelled
                # print(f"  Applied wrench within static limit (s={s:.3f}) - fully cancelled")
                return -applied_wrench
            else:
                # Exceeded static limit: saturate on the surface along the applied-wrench direction
                # print(f"  Applied wrench exceeds static limit (s={s:.3f}) - saturating and transitioning")
                return -applied_wrench / s

        else:
            # Inside _calculate_friction method, replace the kinetic friction section:

            # print("Kinetic friction regime engaged.")

            # Special handling when applied wrench is very small
            if np.linalg.norm(applied_wrench) < 1e-3:
                # print("No meaningful applied wrench; checking for stop condition.")
                
                if twist_magnitude < 1e-1:  # Very small velocity
                    # Calculate friction that would stop the object within dt
                    # We want: v_new = v_old + (applied_wrench + friction_wrench) / mass * dt = 0
                    # Therefore: friction_wrench = -(v_old * mass / dt + applied_wrench)
                    
                    # For linear components
                    friction_fx_stop = -(vx_b * self.mass / self.dt) - applied_wrench[0]
                    friction_fy_stop = -(vy_b * self.mass / self.dt) - applied_wrench[1]
                    
                    # For angular component
                    friction_m_stop = -(omega * self.moment_of_inertia / self.dt) - applied_wrench[2]
                    
                    friction_wrench_stop = np.array([friction_fx_stop, friction_fy_stop, friction_m_stop])
                    
                    # Check if this stopping friction is within the limit surface
                    fx_scaled = friction_wrench_stop[0] / self.kinetic_f_max
                    fy_scaled = friction_wrench_stop[1] / self.kinetic_f_max
                    m_scaled = friction_wrench_stop[2] / self.kinetic_m_max
                    s_stop = np.sqrt(fx_scaled**2 + fy_scaled**2 + m_scaled**2)
                    
                    if s_stop <= 1.0:
                        # Stopping friction is within limit surface - use it
                        # print(f"  Stopping friction (s={s_stop:.3f} ≤ 1.0) - object will stop")
                        return friction_wrench_stop - applied_wrench
                    else:
                        # Stopping friction exceeds limit surface - use limit surface friction
                        # # print(f"  Stopping friction exceeds limit (s={s_stop:.3f} > 1.0) - use limit surface")
                        # Continue to limit surface calculation below
                        pass
                else:
                    # print("  Velocity not small enough; proceeding with kinetic friction.")
                    pass

            # Standard kinetic regime: Friction opposes the current velocity direction
            f_max = max(1e-9, self.kinetic_f_max)
            m_max = max(1e-9, self.kinetic_m_max)
            c_squared = (m_max / f_max)**2

            # Create normalized twist direction vector (opposite to motion)
            twist_dir = -np.array([vx_b, vy_b, omega * c_squared])

            # Normalize to limit surface
            fx_scaled = twist_dir[0] / f_max
            fy_scaled = twist_dir[1] / f_max
            m_scaled = twist_dir[2] / m_max

            # Calculate s for the entire vector
            s = np.sqrt(fx_scaled**2 + fy_scaled**2 + m_scaled**2)

            self.last_s_value = s

            # Guard against zero division 
            if s > 1e-6:  # If there is any meaningful motion
                # Check if limit surface friction would reverse velocity
                # Calculate what acceleration this friction would cause
                friction_on_ls = np.array([
                    fx_scaled * f_max / s,
                    fy_scaled * f_max / s,
                    m_scaled * m_max / s
                ])
                
                # Calculate resulting velocities after applying this friction
                total_wrench = applied_wrench + friction_on_ls
                
                ax = total_wrench[0] / self.mass
                ay = total_wrench[1] / self.mass
                alpha = total_wrench[2] / self.moment_of_inertia
                
                vx_new = vx_b + ax * self.dt
                vy_new = vy_b + ay * self.dt
                omega_new = omega + alpha * self.dt
                
                # Check if any velocity component would reverse sign
                velocity_reverses = (
                    (vx_b * vx_new < 0 and abs(vx_b) > 1e-6) or
                    (vy_b * vy_new < 0 and abs(vy_b) > 1e-6) or
                    (omega * omega_new < 0 and abs(omega) > 1e-6)
                )
                
                if velocity_reverses:
                    # Limit surface friction is too strong - calculate friction to stop exactly
                    # print(f"  Limit surface friction would reverse velocity - capping to stopping friction")
                    
                    # Calculate friction that brings velocity to zero (not reversing)
                    friction_fx_stop = -(vx_b * self.mass / self.dt) - applied_wrench[0]
                    friction_fy_stop = -(vy_b * self.mass / self.dt) - applied_wrench[1]
                    friction_m_stop = -(omega * self.moment_of_inertia / self.dt) - applied_wrench[2]
                    
                    friction_vector = np.array([friction_fx_stop, friction_fy_stop, friction_m_stop])
                    
                    # Verify this is inside limit surface
                    fx_scaled_stop = friction_vector[0] / f_max
                    fy_scaled_stop = friction_vector[1] / f_max
                    m_scaled_stop = friction_vector[2] / m_max
                    s_stop = np.sqrt(fx_scaled_stop**2 + fy_scaled_stop**2 + m_scaled_stop**2)
                    
                    # print(f"  Stopping friction: s={s_stop:.3f} (should be < 1.0)")
                    self.last_s_value = s_stop
                    
                    return friction_vector
                else:
                    # Normal case - use limit surface friction
                    friction_vector = friction_on_ls
                    return friction_vector
            else:
                # Fallback for extremely low velocities
                friction_vector = np.zeros(3)
                return friction_vector
        
    # The rest of the methods remain the same...
    def get_transformed_object(self):
        """
        Get the transformed object at current position and orientation.
        
        Returns:
            GenericObject: Transformed copy of the object
        """
        return self.object.transform(
            self.position[0], self.position[1], 
            self.orientation - self.object.heading
        )
    
    def visualize_state(self, ax=None, show_trajectory=True, show_velocities=True):
        """
        Visualize current object state with trajectory and velocities.
        
        Args:
            ax: Optional matplotlib axis
            show_trajectory: Whether to display trajectory
            show_velocities: Whether to display velocity vectors
            
        Returns:
            matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get transformed object and visualize it
        transformed_object = self.get_transformed_object()
        transformed_object.visualize(ax=ax, alpha=0.8)
        
        # Draw trajectory
        if show_trajectory and len(self.trajectory) > 1:
            trajectory = np.array(self.trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.5, label='Trajectory')
        
        # Draw velocity vectors
        if show_velocities:
            # Body velocities transformed to world frame
            cos_theta = np.cos(self.orientation)
            sin_theta = np.sin(self.orientation)
            
            vx_world = self.velocity_body[0] * cos_theta - self.velocity_body[1] * sin_theta
            vy_world = self.velocity_body[0] * sin_theta + self.velocity_body[1] * cos_theta
            
            # Scale factor for velocity visualization
            vel_scale = 0.2
            
            # Linear velocity (green)
            ax.arrow(self.position[0], self.position[1],
                    vx_world * vel_scale, vy_world * vel_scale,
                    head_width=0.02, head_length=0.02,
                    fc='green', ec='green', alpha=0.7,
                    label='Linear Velocity')
            
            # Angular velocity (blue spiral)
            if abs(self.angular_velocity) > 1e-6:
                omega_dir = np.sign(self.angular_velocity)
                omega_scale = min(0.1, abs(self.angular_velocity) * 0.05)
                
                # Draw a spiral or arc to represent angular velocity
                theta = np.linspace(0, omega_dir * np.pi, 20)
                spiral_radius = np.linspace(0.03, 0.08, 20) 
                spiral_x = self.position[0] + spiral_radius * np.cos(theta)
                spiral_y = self.position[1] + spiral_radius * np.sin(theta)
                
                ax.plot(spiral_x, spiral_y, 'b-', alpha=0.7, label='Angular Velocity')
                
                # Add arrowhead
                arrow_idx = len(spiral_x) - 3
                ax.arrow(spiral_x[arrow_idx], spiral_y[arrow_idx],
                        spiral_x[-1] - spiral_x[arrow_idx], spiral_y[-1] - spiral_y[arrow_idx],
                        head_width=0.02, head_length=0.02,
                        fc='blue', ec='blue', alpha=0.7)
        
        # Set equal aspect ratio and grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Set title with state info
        ax.set_title(f"Object State at t={self.time:.2f}s\n"
                    f"Position: ({self.position[0]:.2f}, {self.position[1]:.2f}), "
                    f"Orientation: {np.degrees(self.orientation):.1f}°\n"
                    f"Body Velocity: ({self.velocity_body[0]:.2f}, {self.velocity_body[1]:.2f}) m/s, "
                    f"Angular Velocity: {np.degrees(self.angular_velocity):.1f}°/s")
        
        return ax

    def simulate_and_animate(self, controller,
                        duration=5.0, dt=0.01, fps=30, stream=False):
        """
        Simulate object dynamics using a controller and create animation with enhanced data collection.
        
        Args:
            controller: A controller object for simulation
            duration: Total simulation duration (seconds)
            dt: Integration time step
            fps: Frames per second for visualization
            stream: Whether to stream animation frames
            collect_data: Whether to collect detailed simulation data. Default is True
            
        Returns:
            dict: Simulation results and collected data
        """
        # Reset simulation
        self.trajectory = [self.position.copy()]
        self.orientation_history = [self.orientation]
        self.time_history = [0.0]
        self.time = 0.0
        
        self.dt = dt  # Store dt for friction calculations
        # Reset controller
        controller.reset()
        
        # Create time steps for simulation and visualization
        sim_steps = int(duration / dt)
        # How many simulation steps between frames
        steps_per_frame = sim_steps if fps == 0 else max(1, int(1.0 / (dt * fps)))

        # Data collection containers
        state_history = []
        
        # Enhanced data collection (if enabled)
        collect_data = True
        # we do not need these instrumentation for normal simulation runs
        if collect_data:
            data = {
                'times': [],
                'forces': [],
                'velocities': [],
                'linear_accelerations': [],
                'angular_velocities': [],
                'angular_accelerations': [],
                'positions': [],
                'orientations': [],
                'frictions_fx': [],
                'frictions_fy': [],
                'frictions_fxy': [],
                'frictions_m': [],
                'applied_wrench': [],
                'total_wrench': [],
                'twist_magnitudes': [],
                's_values': []
            }
        
        # Main simulation loop
        for step in range(sim_steps):
            # Current state to pass to controller
            current_state = {
                'position': self.position.copy(),
                'orientation': self.orientation,
                'velocity_body': self.velocity_body.copy(),
                'angular_velocity': self.angular_velocity,
                'time': self.time
            }
            
            # Update controller with current state
            controller.update(current_state, dt)
            
            # Get control actions from controller
            contact_points, force_magnitudes = controller.get_control_actions()
            
            # Update state
            state = self.update_state(contact_points, force_magnitudes, dt)
            state_history.append(state)
            
            latest_simulation_data = {
                'time': self.time,
                'position': self.position.copy(),
                'orientation': self.orientation,
                'velocity_body': self.velocity_body.copy(),
                'angular_velocity': self.angular_velocity,
                'applied_wrench': state['applied_wrench'],
                'friction_wrench': state['friction_wrench'],
                'total_wrench': state['total_wrench']
            }

            # Enhanced data collection
            if collect_data:
                # Calculate linear velocity magnitude (in world frame)
                cos_theta = np.cos(self.orientation)
                sin_theta = np.sin(self.orientation)
                vx_world = self.velocity_body[0] * cos_theta - self.velocity_body[1] * sin_theta
                vy_world = self.velocity_body[0] * sin_theta + self.velocity_body[1] * cos_theta
                vel_magnitude = np.sqrt(vx_world**2 + vy_world**2)
                
                # Store basic data
                data['times'].append(self.time)
                data['positions'].append(self.position.copy())
                data['orientations'].append(self.orientation)
                data['velocities'].append(vel_magnitude)
                data['angular_velocities'].append(self.angular_velocity)
                
                # Store force data
                total_force_mag = sum(force_magnitudes) if len(force_magnitudes) > 0 else 0
                data['forces'].append(total_force_mag)
                
                # Store wrench components
                data['applied_wrench'].append(state['applied_wrench'])
                data['total_wrench'].append(state['total_wrench'])
                
                # Store friction wrench components
                friction_wrench = state['friction_wrench']
                data['frictions_fx'].append(friction_wrench[0])
                data['frictions_fy'].append(friction_wrench[1])
                data['frictions_fxy'].append(np.linalg.norm(friction_wrench[:2]))
                data['frictions_m'].append(friction_wrench[2])

                # Store twist magnitude and s value
                data['twist_magnitudes'].append(self.last_twist_magnitude)
                data['s_values'].append(self.last_s_value)
                
                # Calculate accelerations
                if step > 0:
                    dt_actual = data['times'][-1] - data['times'][-2]
                    data['linear_accelerations'].append((vel_magnitude - data['velocities'][-2]) / dt_actual)
                    data['angular_accelerations'].append(
                        (self.angular_velocity - data['angular_velocities'][-2]) / dt_actual
                    )
                else:
                    data['linear_accelerations'].append(0)
                    data['angular_accelerations'].append(0)
            
                latest_simulation_data.update({
                    'force_magnitude': total_force_mag,
                    'linear_velocity_world': vel_magnitude,
                    'linear_acceleration': data['linear_accelerations'][-1],
                    'angular_acceleration': data['angular_accelerations'][-1],
                    'friction_fx': friction_wrench[0],
                    'friction_fy': friction_wrench[1],
                    'friction_m': friction_wrench[2],
                    'friction_magnitude': np.linalg.norm(friction_wrench[:2]),
                    's_value': self.last_s_value,
                    'twist_magnitude': self.last_twist_magnitude
                })

            # Now pass only the latest data
            controller.post_update(latest_simulation_data)

            # Visualize at specified FPS
            if stream and step % steps_per_frame == 0:
                fig, ax = plt.subplots(figsize=(10, 8))
                self.visualize_state(ax=ax, show_trajectory=True)
                
                # Show contact points and forces
                if len(contact_points) > 0 and all(cp.object_ref is not None for cp in contact_points):
                    # Transform contact points to current object pose
                    transformed_cps = self._transform_contact_points(contact_points)
                    
                    # Visualize contact forces
                    calculator = GenericContactCalculator(self.object)
                    calculator.visualize_contact_solution(
                        transformed_cps, force_magnitudes, ax=ax, force_scale=0.1
                    )
                
                plt.tight_layout()
                
                # Stream the frame
                stream_figure(fig)
                plt.close(fig)
        
        # Convert lists to numpy arrays for numerical operations
        if collect_data:
            for key in data:
                if len(data[key]) > 0:
                    data[key] = np.array(data[key])
        
        print(f"Simulation completed: {sim_steps} steps ({duration:.1f}s)")
        
        # Return appropriate results
        if collect_data:
            return {
                'state_history': state_history,
                'data': data,
                'controller': controller  # Return controller for further analysis
            }
        else:
            return {
                'state_history': state_history,
                'controller': controller
            }

    def _transform_contact_points(self, contact_points):
        """
        Transform contact points to current object pose.
        
        Args:
            contact_points: List of original ContactPoint objects
            
        Returns:
            list: Transformed ContactPoint objects
        """
        transformed_cps = []
        
        # Get transformed object
        transformed_obj = self.get_transformed_object()
        
        for cp in contact_points:
            # Convert contact point to local coordinates relative to original object
            centroid = cp.object_ref.get_centroid()
            local_pos = cp.position - np.array([centroid.x, centroid.y])
            
            # Rotate to account for any initial heading
            cos_h = np.cos(-cp.object_ref.heading)
            sin_h = np.sin(-cp.object_ref.heading)
            rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            local_pos = rotation_matrix @ local_pos
            
            # Transform to world coordinates at current pose
            cos_new = np.cos(self.orientation)
            sin_new = np.sin(self.orientation)
            new_rotation = np.array([[cos_new, -sin_new], [sin_new, cos_new]])
            world_pos = new_rotation @ local_pos + self.position
            
            # Also transform direction vectors
            new_tangent = new_rotation @ rotation_matrix @ cp.tangent
            new_normal_in = new_rotation @ rotation_matrix @ cp.normal_inward
            new_normal_out = new_rotation @ rotation_matrix @ cp.normal_outward
            
            # Create new contact point
            new_cp = ContactPoint(
                position=world_pos,
                tangent=new_tangent,
                normal_outward=new_normal_out,
                normal_inward=new_normal_in,
                parameter=cp.parameter,
                force_direction=cp.force_direction,
                object_ref=transformed_obj
            )
            
            transformed_cps.append(new_cp)
        
        return transformed_cps

# %%
class FrictionTestController(PlaceholderController):
    """
    Specialized controller for testing friction model with predefined force profile.
    """
    
    def __init__(self, object_model):
        """Initialize friction test controller."""
        super().__init__(object_model)
        
        # Determine friction thresholds
        self.static_friction_threshold = object_model.static_f_max
        self.kinetic_friction_level = object_model.kinetic_f_max
    
    def initialize(self, **kwargs):
        """
        Initialize with provided parameters or defaults.
        
        Args:
            **kwargs: Initialization parameters
                contact_points: Optional predefined contact points
        
        Returns:
            list: Contact points
        """
        if 'contact_points' in kwargs:
            self.contact_points = kwargs['contact_points']
        else:
            # Create a single contact point on the side of the object
            calculator = GenericContactCalculator(self.object_model.object)
            self.contact_points = calculator.calculate_contact_points(n_contacts=1, strategy='uniform')
        
        return self.contact_points
    
    def update_internal(self):
        """Process internal controller state after update."""
        # No special processing needed in this controller
        pass
    
    def get_control_actions(self):
        """
        Apply the predefined force profile for testing friction behavior.
        
        Returns:
            tuple: (contact_points, force_magnitudes)
        """
        # Force profile with multiple phases
        # Phase durations
        phase1_duration = 3.0   # Ramp up to peak
        phase2_duration = 2.0   # Decrease to kinetic level
        phase3_duration = 2.0   # Hold at kinetic level
        phase4_duration = 2.0   # Decrease to quasi-static level
        phase5_duration = 2.0   # Hold at quasi-static level
        phase6_duration = 2.0   # Decrease to zero
        phase7_duration = 100.0  # Hold at zero (large value to handle any simulation duration)
        
        # Phase end times
        phase1_end = phase1_duration
        phase2_end = phase1_end + phase2_duration
        phase3_end = phase2_end + phase3_duration
        phase4_end = phase3_end + phase4_duration
        phase5_end = phase4_end + phase5_duration
        phase6_end = phase5_end + phase6_duration
        
        # Force levels
        peak_force = self.static_friction_threshold * 1.2
        kinetic_level = self.kinetic_friction_level * 1.0
        quasi_static_level = kinetic_level * 0.6
        
        t = self.time
        
        if t < phase1_end:
            # Phase 1: Linear increase to peak
            force = (t / phase1_duration) * peak_force
        elif t < phase2_end:
            # Phase 2: Linear decrease to kinetic level
            remaining_t = t - phase1_end
            force = peak_force - (remaining_t / phase2_duration) * (peak_force - kinetic_level)
        elif t < phase3_end:
            # Phase 3: Hold at kinetic level
            force = kinetic_level
        elif t < phase4_end:
            # Phase 4: Decrease to quasi-static level
            remaining_t = t - phase3_end
            force = kinetic_level - (remaining_t / phase4_duration) * (kinetic_level - quasi_static_level)
        elif t < phase5_end:
            # Phase 5: Hold at quasi-static level
            force = quasi_static_level
        elif t < phase6_end:
            # Phase 6: Decrease to zero
            remaining_t = t - phase5_end
            force = quasi_static_level * (1.0 - remaining_t / phase6_duration)
        else:
            # Phase 7: Hold at zero
            force = 0.0
        
        self.force_magnitudes = [force]
        
        # Optionally save control history if enabled
        if self.control_history_save:
            self.contact_points_history.append(self.contact_points.copy())
            self.force_magnitudes_history.append(self.force_magnitudes.copy())
            
        return self.contact_points, self.force_magnitudes


# %%
def test_friction_model_with_controller():
    """
    Test the friction model using the controller-based approach.
    """
    # Create rectangle object
    obj = standard_objects['rectangle']
    
    # Create dynamic model with lower noise
    dynamics = DynamicObjectModel(obj, friction_noise_std=0.001)
    
    # Create the specialized friction test controller
    controller = FrictionTestController(dynamics)
    
    # Create specific contact points if needed
    calculator = GenericContactCalculator(obj)
    contact_points = calculator.calculate_contact_points(n_contacts=1, strategy='uniform')
    
    # Initialize controller with custom contact points
    controller.initialize(contact_points=contact_points)
    
    # Enable control history saving for later analysis
    controller.control_history_save = True
    
    # Run simulation with enhanced data collection
    print("Running detailed friction model test with controller...")
    simulation_results = dynamics.simulate_and_animate(
        controller,
        duration=25.0,  # Extended duration
        dt=0.05,
        fps=0,  # No visualization during simulation
        stream=False,
    )
    

    data = simulation_results['data']
    
    # Create a comprehensive dashboard with subplots
    fig = plt.figure(figsize=(18, 12))
    grid = plt.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])
    
    # 1. Force and Friction Plot (top left)
    ax_force = fig.add_subplot(grid[0, 0])
    ax_force.plot(data['times'], data['forces'], 'b-', linewidth=2, label='Applied Force X')
    # ax_force.plot(data['times'], data['total_wrench'][:,0], 'b-', linewidth=2, label='Applied Force X')
    # ax_force.plot(data['times'], data['total_wrench'][:,1], 'g-', linewidth=2, label='Applied Force Y')
    # ax_force.plot(data['times'], data['total_wrench'][:,2], 'm-', linewidth=2, label='Applied Torque')
    ax_force.plot(data['times'], data['frictions_fxy'], 'r--', linewidth=2, label='Friction Force')
    ax_force.axhline(y=dynamics.static_f_max, color='k', linestyle='-', alpha=0.3, 
                     label=f'Static ({dynamics.static_f_max:.2f}N)')
    ax_force.axhline(y=dynamics.kinetic_f_max, color='k', linestyle=':', alpha=0.3,
                     label=f'Kinetic ({dynamics.kinetic_f_max:.2f}N)')
    ax_force.set_title('Forces')
    ax_force.set_ylabel('Force (N)')
    ax_force.legend(fontsize='small')
    ax_force.grid(True)
    
    # 2. Velocities Plot (top middle)
    ax_vel = fig.add_subplot(grid[0, 1], sharex=ax_force)
    ax_vel.plot(data['times'], data['velocities'], 'g-', linewidth=2, label='Linear')
    ax_vel.plot(data['times'], data['angular_velocities'], 'c-', linewidth=2, label='Angular')
    ax_vel.set_title('Velocities')
    ax_vel.set_ylabel('Velocity')
    ax_vel.legend(fontsize='small')
    ax_vel.grid(True)
    
    # 3. Accelerations Plot (top right)
    ax_accel = fig.add_subplot(grid[0, 2], sharex=ax_force)
    ax_accel.plot(data['times'], data['linear_accelerations'], 'm-', linewidth=2, label='Linear')
    ax_accel.plot(data['times'], data['angular_accelerations'], 'y-', linewidth=2, label='Angular')
    ax_accel.set_title('Accelerations')
    ax_accel.set_ylabel('Acceleration')
    ax_accel.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_accel.legend(fontsize='small')
    ax_accel.grid(True)
    
    # 4. Friction Components (middle left)
    ax_fric_comp = fig.add_subplot(grid[1, 0], sharex=ax_force)
    ax_fric_comp.plot(data['times'], data['frictions_fx'], 'r-', linewidth=2, label='Fx')
    ax_fric_comp.plot(data['times'], data['frictions_fy'], 'g-', linewidth=2, label='Fy')
    ax_fric_comp.plot(data['times'], data['frictions_m'], 'b-', linewidth=2, label='Moment')
    ax_fric_comp.set_title('Friction Components')
    ax_fric_comp.set_xlabel('Time (s)')
    ax_fric_comp.set_ylabel('Friction')
    ax_fric_comp.legend(fontsize='small')
    ax_fric_comp.grid(True)
    
    # 5. Twist Magnitude (middle middle)
    ax_twist = fig.add_subplot(grid[1, 1], sharex=ax_force)
    ax_twist.semilogy(data['times'], data['twist_magnitudes'], 'k-', linewidth=2)
    ax_twist.axhline(y=1e-4, color='r', linestyle='--', alpha=0.7, label='Threshold')
    ax_twist.set_title('Twist Magnitude')
    ax_twist.set_xlabel('Time (s)')
    ax_twist.set_ylabel('Log Magnitude')
    ax_twist.grid(True)
    
    # 6. S Value (middle right)
    ax_s = fig.add_subplot(grid[1, 2], sharex=ax_force)
    ax_s.semilogy(data['times'], data['s_values'], 'm-', linewidth=2)
    ax_s.axhline(y=1e-9, color='r', linestyle='--', alpha=0.7, label='Threshold')
    ax_s.set_title('S Value')
    ax_s.set_xlabel('Time (s)')
    ax_s.set_ylabel('Log Value')
    ax_s.grid(True)
    
    
    plt.tight_layout()
    plt.show()


    # We can also access control history from the controller
    print(f"Collected {len(controller.force_magnitudes_history)} control actions")
    
    return data


# # Run both tests
# print("TEST 1: FORCE RAMP")
# ramp_results = test_friction_model_ramp()

# %%
class StoppingVelocityCalculator:
    """
    Calculate maximum velocities that result in complete stop after X seconds
    with no applied force (only friction deceleration).
    
    This helps determine safe velocity bounds for control systems.
    """
    
    def __init__(self, dynamics_model):
        """
        Initialize with a DynamicObjectModel.
        
        Args:
            dynamics_model: DynamicObjectModel instance with friction properties
        """
        self.dynamics = dynamics_model
        
        # Extract friction parameters
        self.kinetic_f_max = dynamics_model.kinetic_f_max
        self.kinetic_m_max = dynamics_model.kinetic_m_max
        self.mass = dynamics_model.mass
        self.inertia = dynamics_model.moment_of_inertia
        
        # Scaling factor for unified treatment
        self.c_kinetic = self.kinetic_m_max / self.kinetic_f_max
        
    def calculate_max_stopping_velocity(self, stop_time=1.0, velocity_type='linear', 
                                       direction=None, tolerance=1e-6):
        """
        Calculate maximum velocity that stops completely after stop_time seconds.
        
        For linear motion:
            - Friction force opposes motion: f_friction = -f_max * (v / |v|)
            - Deceleration: a = -f_max / m
            - Stopping condition: v(t) = v_0 + a*t = 0 at t = stop_time
            - Therefore: v_max = f_max * stop_time / m
        
        For rotational motion:
            - Friction torque opposes rotation: τ_friction = -m_max * (ω / |ω|)
            - Angular deceleration: α = -m_max / I
            - Stopping condition: ω(t) = ω_0 + α*t = 0 at t = stop_time
            - Therefore: ω_max = m_max * stop_time / I
        
        For combined motion (general case):
            - Friction wrench lies on limit surface: (f_x/f_max)² + (f_y/f_max)² + (m/m_max)² = 1
            - Direction-dependent deceleration
            
        Args:
            stop_time: Time in seconds to come to complete stop
            velocity_type: 'linear', 'angular', or 'combined'
            direction: For combined motion, specify direction as [vx, vy, ω] unit vector
            tolerance: Numerical tolerance for stopping condition
            
        Returns:
            dict: Maximum velocities and related information
        """
        
        if velocity_type == 'linear':
            # Pure translational motion
            # Maximum deceleration = f_max / m
            max_deceleration = self.kinetic_f_max / self.mass
            
            # Maximum velocity that stops in stop_time
            v_max = max_deceleration * stop_time
            
            return {
                'velocity_type': 'linear',
                'v_max': v_max,  # m/s
                'stop_time': stop_time,
                'deceleration': max_deceleration,  # m/s²
                'distance_traveled': 0.5 * v_max * stop_time,  # Using s = v₀t - 0.5at²
                'interpretation': f'Any velocity ≤ {v_max:.3f} m/s will stop within {stop_time:.1f}s'
            }
            
        elif velocity_type == 'angular':
            # Pure rotational motion
            # Maximum angular deceleration = m_max / I
            max_angular_decel = self.kinetic_m_max / self.inertia
            
            # Maximum angular velocity that stops in stop_time
            omega_max = max_angular_decel * stop_time
            
            return {
                'velocity_type': 'angular',
                'omega_max': omega_max,  # rad/s
                'omega_max_deg': np.degrees(omega_max),  # deg/s
                'stop_time': stop_time,
                'angular_deceleration': max_angular_decel,  # rad/s²
                'angle_rotated': 0.5 * omega_max * stop_time,  # Using θ = ω₀t - 0.5αt²
                'angle_rotated_deg': np.degrees(0.5 * omega_max * stop_time),
                'interpretation': f'Any angular velocity ≤ {np.degrees(omega_max):.1f}°/s will stop within {stop_time:.1f}s'
            }
            
        elif velocity_type == 'combined':
            # Combined motion with specified direction
            if direction is None:
                raise ValueError("Direction vector [vx, vy, ω] required for combined motion")
            
            # Normalize direction vector
            direction = np.array(direction, dtype=float)
            
            # Scale angular component by c for unified treatment
            scaled_direction = np.array([direction[0], direction[1], direction[2] * self.c_kinetic])
            dir_magnitude = np.linalg.norm(scaled_direction)
            
            if dir_magnitude < 1e-12:
                raise ValueError("Direction vector cannot be zero")
            
            dir_unit = scaled_direction / dir_magnitude
            
            # Friction wrench on limit surface opposing this direction
            # friction = -[f_max * dir_x, f_max * dir_y, m_max * dir_z]
            friction_wrench = -np.array([
                self.kinetic_f_max * dir_unit[0],
                self.kinetic_f_max * dir_unit[1],
                self.kinetic_m_max * dir_unit[2]
            ])
            
            # Accelerations from friction
            accel_x = friction_wrench[0] / self.mass
            accel_y = friction_wrench[1] / self.mass
            accel_omega = friction_wrench[2] / self.inertia
            
            # Maximum initial velocities that stop at stop_time
            vx_max = -accel_x * stop_time
            vy_max = -accel_y * stop_time
            omega_max = -accel_omega * stop_time
            
            # Verify these are in the correct direction
            result_direction = np.array([vx_max, vy_max, omega_max])
            result_magnitude = np.linalg.norm([vx_max, vy_max, omega_max * self.c_kinetic])
            
            return {
                'velocity_type': 'combined',
                'direction': direction / np.linalg.norm([direction[0], direction[1], direction[2] * self.c_kinetic]),
                'vx_max': vx_max,
                'vy_max': vy_max,
                'omega_max': omega_max,
                'omega_max_deg': np.degrees(omega_max),
                'velocity_magnitude': result_magnitude,
                'stop_time': stop_time,
                'accelerations': [accel_x, accel_y, accel_omega],
                'distance_traveled': {
                    'x': 0.5 * vx_max * stop_time,
                    'y': 0.5 * vy_max * stop_time,
                    'rotation': 0.5 * omega_max * stop_time
                },
                'interpretation': f'Velocity [{vx_max:.3f}, {vy_max:.3f}, {np.degrees(omega_max):.1f}°/s] stops in {stop_time:.1f}s'
            }
        
        else:
            raise ValueError(f"Unknown velocity_type: {velocity_type}")
    
    def verify_stopping_time(self, initial_velocity, velocity_type='linear', 
                           num_steps=1000, dt=0.001):
        """
        Verify stopping time by simulating the motion with no applied forces.
        
        Args:
            initial_velocity: Initial velocity (scalar for linear/angular, array for combined)
            velocity_type: 'linear', 'angular', or 'combined'
            num_steps: Number of simulation steps
            dt: Time step size
            
        Returns:
            dict: Simulation results with actual stopping time
        """
        # Save original state
        original_position = self.dynamics.position.copy()
        original_orientation = self.dynamics.orientation
        original_velocity = self.dynamics.velocity_body.copy()
        original_omega = self.dynamics.angular_velocity
        
        # Set initial conditions based on velocity type
        if velocity_type == 'linear':
            # Assume motion in x direction
            self.dynamics.velocity_body = np.array([initial_velocity, 0.0])
            self.dynamics.angular_velocity = 0.0
        elif velocity_type == 'angular':
            self.dynamics.velocity_body = np.array([0.0, 0.0])
            self.dynamics.angular_velocity = initial_velocity
        elif velocity_type == 'combined':
            vx, vy, omega = initial_velocity
            self.dynamics.velocity_body = np.array([vx, vy])
            self.dynamics.angular_velocity = omega
        
        # Simulate with no applied forces
        times = []
        velocities = []
        angular_velocities = []
        positions = []
        
        # No contact points = no applied forces
        no_contacts = []
        no_forces = []
        
        for step in range(num_steps):
            # Record current state
            times.append(step * dt)
            velocities.append(np.linalg.norm(self.dynamics.velocity_body))
            angular_velocities.append(self.dynamics.angular_velocity)
            positions.append(self.dynamics.position.copy())
            
            # Check if stopped
            vel_magnitude = np.linalg.norm(self.dynamics.velocity_body)
            omega_magnitude = abs(self.dynamics.angular_velocity)
            
            if vel_magnitude < 1e-6 and omega_magnitude < 1e-6:
                break
            
            # Update with no forces (only friction acts)
            self.dynamics.update_state(no_contacts, no_forces, dt=dt, friction_enabled=True)
        
        # Find actual stopping time
        stopping_idx = len(times) - 1
        actual_stop_time = times[stopping_idx]
        
        # Restore original state
        self.dynamics.position = original_position
        self.dynamics.orientation = original_orientation
        self.dynamics.velocity_body = original_velocity
        self.dynamics.angular_velocity = original_omega
        
        return {
            'actual_stop_time': actual_stop_time,
            'times': np.array(times),
            'velocities': np.array(velocities),
            'angular_velocities': np.array(angular_velocities),
            'positions': np.array(positions),
            'final_position': positions[-1],
            'total_distance': np.linalg.norm(positions[-1] - positions[0])
        }
    
    def plot_stopping_analysis(self, stop_time=1.0, num_test_velocities=5):
        """
        Create comprehensive visualization of stopping behavior.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Test velocities from 0 to max
        linear_result = self.calculate_max_stopping_velocity(stop_time, 'linear')
        angular_result = self.calculate_max_stopping_velocity(stop_time, 'angular')
        
        v_max = linear_result['v_max']
        omega_max = angular_result['omega_max']
        
        test_v_values = np.linspace(0.5 * v_max, 1.5 * v_max, num_test_velocities)
        test_omega_values = np.linspace(0.5 * omega_max, 1.5 * omega_max, num_test_velocities)
        
        # 1. Linear velocity stopping (top left)
        ax1 = axes[0, 0]
        for v_test in test_v_values:
            result = self.verify_stopping_time(v_test, 'linear')
            label = f'v₀={v_test:.3f} m/s'
            if abs(result['actual_stop_time'] - stop_time) < 0.05:
                label += ' ✓'
            ax1.plot(result['times'], result['velocities'], label=label, linewidth=2)
        
        ax1.axvline(x=stop_time, color='red', linestyle='--', alpha=0.7, label=f'Target: {stop_time}s')
        ax1.axhline(y=v_max, color='green', linestyle=':', alpha=0.7, label=f'v_max={v_max:.3f}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Linear Velocity (m/s)')
        ax1.set_title('Linear Stopping Behavior')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Angular velocity stopping (top middle)
        ax2 = axes[0, 1]
        for omega_test in test_omega_values:
            result = self.verify_stopping_time(omega_test, 'angular')
            label = f'ω₀={np.degrees(omega_test):.1f}°/s'
            if abs(result['actual_stop_time'] - stop_time) < 0.05:
                label += ' ✓'
            ax2.plot(result['times'], np.degrees(result['angular_velocities']), 
                    label=label, linewidth=2)
        
        ax2.axvline(x=stop_time, color='red', linestyle='--', alpha=0.7, label=f'Target: {stop_time}s')
        ax2.axhline(y=np.degrees(omega_max), color='green', linestyle=':', alpha=0.7, 
                   label=f'ω_max={np.degrees(omega_max):.1f}°/s')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (°/s)')
        ax2.set_title('Angular Stopping Behavior')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Combined motion (top right) - test several directions
        ax3 = axes[0, 2]
        test_directions = [
            [1, 0, 0],      # Pure x
            [0, 1, 0],      # Pure y
            [1, 1, 0],      # Diagonal
            [1, 0, 0.5],    # x + rotation
            [0.5, 0.5, 1]   # Mixed
        ]
        
        for direction in test_directions:
            combined_result = self.calculate_max_stopping_velocity(
                stop_time, 'combined', direction=direction
            )
            
            initial_vel = [combined_result['vx_max'], 
                         combined_result['vy_max'], 
                         combined_result['omega_max']]
            
            result = self.verify_stopping_time(initial_vel, 'combined')
            
            # Plot velocity magnitude
            vel_mags = result['velocities']
            label = f'dir={direction}'
            ax3.plot(result['times'], vel_mags, label=label, linewidth=2)
        
        ax3.axvline(x=stop_time, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity Magnitude (m/s)')
        ax3.set_title('Combined Motion Stopping')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Distance traveled analysis (bottom left)
        ax4 = axes[1, 0]
        distances = []
        velocities_tested = np.linspace(0, 2 * v_max, 20)
        
        for v_test in velocities_tested:
            result = self.verify_stopping_time(v_test, 'linear')
            distances.append(result['total_distance'])
        
        ax4.plot(velocities_tested, distances, 'b-', linewidth=2)
        ax4.axvline(x=v_max, color='green', linestyle='--', alpha=0.7, label=f'v_max={v_max:.3f}')
        ax4.scatter([v_max], [linear_result['distance_traveled']], 
                   color='red', s=100, zorder=5, label='Predicted distance')
        ax4.set_xlabel('Initial Velocity (m/s)')
        ax4.set_ylabel('Distance Traveled (m)')
        ax4.set_title('Distance vs Initial Velocity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Summary table (bottom middle)
        ax5 = axes[1, 1]
        ax5.axis('off')
        
        table_data = [
            ['Parameter', 'Linear', 'Angular'],
            ['Max Velocity', f'{v_max:.3f} m/s', f'{np.degrees(omega_max):.1f}°/s'],
            ['Deceleration', f'{linear_result["deceleration"]:.3f} m/s²', 
             f'{np.degrees(angular_result["angular_deceleration"]):.1f}°/s²'],
            ['Distance/Angle', f'{linear_result["distance_traveled"]:.3f} m', 
             f'{angular_result["angle_rotated_deg"]:.1f}°'],
            ['Stop Time', f'{stop_time:.1f}s', f'{stop_time:.1f}s'],
        ]
        
        table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax5.set_title(f'Stopping Analysis Summary (Target: {stop_time}s)', 
                     fontsize=12, fontweight='bold', pad=20)
        
        # 6. Friction limit surface visualization (bottom right)
        ax6 = axes[1, 2]
        
        # Draw limit surface
        theta = np.linspace(0, 2*np.pi, 100)
        fx = self.kinetic_f_max * np.cos(theta)
        fy = self.kinetic_f_max * np.sin(theta)
        
        ax6.plot(fx, fy, 'k-', linewidth=2, label='Kinetic Limit Surface')
        ax6.fill(fx, fy, alpha=0.2, color='gray')
        
        # Show friction vectors for different velocity directions
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            vel_dir = np.array([np.cos(angle), np.sin(angle)])
            friction = -self.kinetic_f_max * vel_dir
            
            ax6.arrow(0, 0, friction[0]*0.8, friction[1]*0.8,
                     head_width=0.3, head_length=0.2, fc='blue', ec='blue', alpha=0.6)
        
        ax6.set_xlabel('Friction Force X (N)')
        ax6.set_ylabel('Friction Force Y (N)')
        ax6.set_title('Friction Force Directions')
        ax6.set_aspect('equal')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.suptitle(f'Stopping Velocity Analysis\n'
                    f'Object: m={self.mass:.2f}kg, I={self.inertia:.4f}kg⋅m², '
                    f'μ_k={self.dynamics.object.kinetic_friction:.2f}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()



# %%
def demo_stopping_velocity_calculator():
    """
    Demonstrate the StoppingVelocityCalculator with comprehensive analysis.
    """
    print("\n" + "="*80)
    print("🎯 PROBLEM 1: MAXIMUM STOPPING VELOCITY ANALYSIS")
    print("="*80)
    
    
    objects = create_standard_objects()
    obj = objects['star']  # Test with star
    
    obj.kinetic_friction = 0.4  # Moderate friction
    dynamics = DynamicObjectModel(obj)
    calculator = StoppingVelocityCalculator(dynamics)
    
    # Test different stop times
    stop_times = [0.5, 1.0, 2.0]
    
    print("\n📊 Testing different target stop times:")
    print("-" * 80)
    
    for stop_time in stop_times:
        print(f"\n⏱️  Target Stop Time: {stop_time}s")
        
        # Linear motion
        linear_result = calculator.calculate_max_stopping_velocity(stop_time, 'linear')
        print(f"   Linear: v_max = {linear_result['v_max']:.4f} m/s")
        print(f"           Distance traveled: {linear_result['distance_traveled']:.4f} m")
        print(f"           {linear_result['interpretation']}")
        
        # Angular motion
        angular_result = calculator.calculate_max_stopping_velocity(stop_time, 'angular')
        print(f"   Angular: ω_max = {angular_result['omega_max_deg']:.2f}°/s ({angular_result['omega_max']:.4f} rad/s)")
        print(f"            Angle rotated: {angular_result['angle_rotated_deg']:.2f}°")
        print(f"            {angular_result['interpretation']}")
        
        # Combined motion (several directions)
        print(f"   Combined motion examples:")
        test_directions = [
            ([1, 0, 0], "Pure X translation"),
            ([1, 1, 0], "Diagonal translation"),
            ([1, 0, 1], "X translation + rotation")
        ]
        
        for direction, description in test_directions:
            combined_result = calculator.calculate_max_stopping_velocity(
                stop_time, 'combined', direction=direction
            )
            print(f"      {description}: "
                  f"v=[{combined_result['vx_max']:.3f}, {combined_result['vy_max']:.3f}] m/s, "
                  f"ω={combined_result['omega_max_deg']:.1f}°/s")
    
    # Create comprehensive visualization
    print("\n📈 Creating visualization...")
    calculator.plot_stopping_analysis(stop_time=0.2, num_test_velocities=5)
    
    print("\n" + "="*80)
    print("✅ STOPPING VELOCITY ANALYSIS COMPLETE")
    print("="*80)
    print("\n💡 Key Insights:")
    print("   - Maximum safe velocities are directly proportional to stop time")
    print("   - Distance/angle traveled scales quadratically with initial velocity")
    print("   - Combined motion has direction-dependent stopping characteristics")
    print("   - These bounds are crucial for velocity controller design")


# %%
class BoundaryMotionPredictor:
    """
    Predict motion of boundary points and segments under applied wrench.
    
    REFACTORED to:
    - Use ContactPoint class with t_param for boundary points
    - Leverage DynamicObjectModel methods directly
    - Calculate heading (normal inward) in main prediction function
    - Return comprehensive motion data for visualization and analysis
    """
    
    def __init__(self, dynamics_model):
        """
        Initialize with a DynamicObjectModel.
        
        Args:
            dynamics_model: DynamicObjectModel instance
        """
        self.dynamics = dynamics_model
    
    def create_boundary_contact_point(self, t_param):
        """
        Create a ContactPoint at parameter t along the boundary.
        This ensures the point is exactly on the object boundary.
        
        Args:
            t_param: Parameter value t ∈ [0, 1] along boundary
            
        Returns:
            ContactPoint: Contact point at parameter t
        """
        parameterization = ContactPointParameterization(self.dynamics.object)
        contact_info = parameterization.get_contact_info(t_param)
        
        contact_point = ContactPoint(
            position=contact_info['point'],
            tangent=contact_info['tangent'],
            normal_outward=contact_info['normal_outward'],
            normal_inward=contact_info['normal_inward'],
            parameter=t_param,
            object_ref=self.dynamics.object
        )
        
        return contact_point
    
    def get_point_position_in_body_frame(self, contact_point, object_position, object_orientation):
        """
        Get contact point position in body frame relative to object centroid.
        
        Args:
            contact_point: ContactPoint instance
            object_position: Object centroid position in world frame
            object_orientation: Object orientation in radians
            
        Returns:
            np.ndarray: Position in body frame [x_body, y_body]
        """
        # Vector from centroid to contact point in world frame
        relative_world = contact_point.position - object_position
        
        # Rotate to body frame
        cos_theta = np.cos(-object_orientation)
        sin_theta = np.sin(-object_orientation)
        rotation_matrix = np.array([[cos_theta, -sin_theta], 
                                   [sin_theta, cos_theta]])
        
        return rotation_matrix @ relative_world
    
    def get_point_position_in_world_frame(self, point_body, object_position, object_orientation):
        """
        Transform body frame point to world frame.
        
        Args:
            point_body: Point in body frame [x_body, y_body]
            object_position: Object centroid position in world frame
            object_orientation: Object orientation in radians
            
        Returns:
            np.ndarray: Position in world frame [x_world, y_world]
        """
        # Rotate to world frame
        cos_theta = np.cos(object_orientation)
        sin_theta = np.sin(object_orientation)
        rotation_matrix = np.array([[cos_theta, -sin_theta], 
                                   [sin_theta, cos_theta]])
        
        return rotation_matrix @ point_body + object_position
    
    def calculate_point_velocity(self, point_world, object_position, object_velocity_body, 
                                object_angular_velocity, object_orientation):
        """
        Calculate velocity of a boundary point in world frame.
        
        Uses rigid body kinematics: v_point = v_object + ω × r
        
        Args:
            point_world: Point position in world frame
            object_position: Object centroid position in world frame
            object_velocity_body: Object velocity in body frame [vx, vy]
            object_angular_velocity: Object angular velocity (rad/s)
            object_orientation: Object orientation (rad)
            
        Returns:
            np.ndarray: Point velocity in world frame [vx, vy]
        """
        # Object velocity in world frame
        cos_theta = np.cos(object_orientation)
        sin_theta = np.sin(object_orientation)
        v_object_world = np.array([
            object_velocity_body[0] * cos_theta - object_velocity_body[1] * sin_theta,
            object_velocity_body[0] * sin_theta + object_velocity_body[1] * cos_theta
        ])
        
        # Vector from centroid to point
        r = point_world - object_position
        
        # Rotational contribution: ω × r (in 2D: perpendicular to r)
        v_rotation = object_angular_velocity * np.array([-r[1], r[0]])
        
        return v_object_world + v_rotation
    def calculate_heading_at_point(self, t_param, object_orientation):
        """
        Calculate the heading (normal inward direction) at a boundary point.
        (Updated to match ground truth method and handle object heading correctly)
        
        Args:
            t_param: Parameter value t ∈ [0, 1] along boundary
            object_orientation: Current object orientation in radians
            
        Returns:
            dict: {
                'heading_angle': Heading angle in world frame (radians),
                'heading_vector': Heading unit vector in world frame,
                'normal_body': Normal inward in body frame
            }
        """
        # Get contact info in body frame using ContactPointParameterization
        parameterization = ContactPointParameterization(self.dynamics.object)
        contact_info = parameterization.get_contact_info(t_param)
        normal_body = contact_info['normal_inward']
        
        # IMPORTANT: The normal from get_contact_info is in the object's canonical frame
        # We need to account for the object's current orientation MINUS its initial heading
        # because the object.heading is already baked into the ContactPointParameterization
        
        # Effective orientation = current orientation - object's initial heading
        effective_orientation = object_orientation - self.dynamics.object.heading
        
        # Transform to world frame using effective orientation
        cos_theta = np.cos(effective_orientation)
        sin_theta = np.sin(effective_orientation)
        R = np.array([[cos_theta, -sin_theta],
                    [sin_theta, cos_theta]])
        
        normal_world = R @ normal_body
        heading_angle = np.arctan2(normal_world[1], normal_world[0])
        
        return {
            'heading_angle': heading_angle,
            'heading_vector': normal_world,
            'normal_body': normal_body
        }
    def predict_point_motion(self, t_param, wrench, duration, dt=0.01,
                           initial_velocity=None, initial_omega=None):
        """
        Predict motion of a boundary point specified by t_param.
        
        NOW INCLUDES: Complete motion data with heading information.
        
        Args:
            t_param: Parameter value t ∈ [0, 1] along boundary
            wrench: Applied wrench [Fx, Fy, τ] (can be zero)
            duration: Prediction duration in seconds
            dt: Time step for prediction
            initial_velocity: Initial object velocity [vx_body, vy_body], default [0, 0]
            initial_omega: Initial angular velocity, default 0
            
        Returns:
            dict: Comprehensive motion prediction including:
                - times: Time array
                - point_trajectory: Point position over time (world frame)
                - point_velocities: Point velocity over time (world frame)
                - heading_angles: Heading angle over time (radians)
                - heading_vectors: Heading unit vectors over time
                - heading_angular_velocities: Rate of heading change (rad/s)
                - object_positions: Object centroid positions
                - object_orientations: Object orientations
                - object_velocities: Object velocities (body frame)
                - object_angular_velocities: Object angular velocities
                - contact_point: Initial ContactPoint
                - t_param: Parameter value
                - ... (additional summary statistics)
        """
        # Create contact point at parameter t
        contact_point = self.create_boundary_contact_point(t_param)
        
        # Save original state
        original_position = self.dynamics.position.copy()
        original_orientation = self.dynamics.orientation
        original_velocity = self.dynamics.velocity_body.copy()
        original_omega = self.dynamics.angular_velocity
        original_time = self.dynamics.time
        
        # Set initial conditions - FIXED: Don't shadow parameter names
        initial_velo = np.array([0.0, 0.0])
        if initial_velocity is not None:
            initial_velo = np.array(initial_velocity)
        
        initial_omega_value = 0.0  # ✅ Use different variable name
        if initial_omega is not None:
            initial_omega_value = initial_omega  # ✅ Now correctly copies the parameter
        
        self.dynamics.reset_state(
            position=original_position,
            orientation=original_orientation,
            velocity=initial_velo,
            angular_velocity=initial_omega_value  # ✅ Use the correct value
        )
        
        # Get point position in body frame (constant during motion)
        point_body = self.get_point_position_in_body_frame(
            contact_point, self.dynamics.position, self.dynamics.orientation
        )
        
        # Simulate motion and collect ALL data
        num_steps = int(duration / dt)
        times = []
        point_world_trajectory = []
        point_velocities_world = []
        heading_angles = []
        heading_vectors = []
        heading_angular_velocities = []
        object_positions = []
        object_orientations = []
        object_velocities = []
        object_angular_velocities = []
        
        prev_heading_angle = None
        
        for step in range(num_steps + 1):
            current_time = step * dt
            times.append(current_time)
            
            # Calculate current world position of the boundary point
            point_world = self.get_point_position_in_world_frame(
                point_body, self.dynamics.position, self.dynamics.orientation
            )
            point_world_trajectory.append(point_world)
            
            # Calculate point velocity in world frame
            v_point = self.calculate_point_velocity(
                point_world, self.dynamics.position, 
                self.dynamics.velocity_body, self.dynamics.angular_velocity,
                self.dynamics.orientation
            )
            point_velocities_world.append(v_point)
            
            # Calculate heading (normal inward direction) at this point
            heading_info = self.calculate_heading_at_point(t_param, self.dynamics.orientation)
            heading_angles.append(heading_info['heading_angle'])
            heading_vectors.append(heading_info['heading_vector'])
            
            # Calculate heading angular velocity
            if prev_heading_angle is not None and dt > 0:
                dheading = heading_info['heading_angle'] - prev_heading_angle
                # Unwrap angle difference to [-π, π]
                dheading = np.arctan2(np.sin(dheading), np.cos(dheading))
                heading_omega = dheading / dt
                heading_angular_velocities.append(heading_omega)
            else:
                heading_angular_velocities.append(0.0)
            
            prev_heading_angle = heading_info['heading_angle']
            
            # Record object state
            object_positions.append(self.dynamics.position.copy())
            object_orientations.append(self.dynamics.orientation)
            object_velocities.append(self.dynamics.velocity_body.copy())
            object_angular_velocities.append(self.dynamics.angular_velocity)
            
            # Update dynamics using DynamicObjectModel's method
            if step < num_steps:
                self._apply_wrench_step(wrench, dt)
        
        # Restore original state
        self.dynamics.reset_state(
            position=original_position,
            orientation=original_orientation,
            velocity=original_velocity,
            angular_velocity=original_omega
        )
        
        # Convert to numpy arrays
        times = np.array(times)
        point_world_trajectory = np.array(point_world_trajectory)
        point_velocities_world = np.array(point_velocities_world)
        heading_angles = np.array(heading_angles)
        heading_vectors = np.array(heading_vectors)
        heading_angular_velocities = np.array(heading_angular_velocities)
        object_positions = np.array(object_positions)
        object_orientations = np.array(object_orientations)
        object_velocities = np.array(object_velocities)
        object_angular_velocities = np.array(object_angular_velocities)
        
        # Calculate summary statistics
        total_displacement = np.linalg.norm(point_world_trajectory[-1] - point_world_trajectory[0])
        max_velocity = np.max(np.linalg.norm(point_velocities_world, axis=1))
        avg_velocity = np.mean(np.linalg.norm(point_velocities_world, axis=1))
        heading_range = np.max(heading_angles) - np.min(heading_angles)
        max_heading_omega = np.max(np.abs(heading_angular_velocities))
        avg_heading_omega = np.mean(np.abs(heading_angular_velocities))
        
        return {
            # Time series data
            'times': times,
            'point_trajectory': point_world_trajectory,
            'point_velocities': point_velocities_world,
            'heading_angles': heading_angles,
            'heading_vectors': heading_vectors,
            'heading_angular_velocities': heading_angular_velocities,
            'object_positions': object_positions,
            'object_orientations': object_orientations,
            'object_velocities': object_velocities,
            'object_angular_velocities': object_angular_velocities,
            
            # Initial state
            'contact_point': contact_point,
            't_param': t_param,
            'initial_point': point_world_trajectory[0],
            'initial_heading': heading_angles[0],
            
            # Final state
            'final_point': point_world_trajectory[-1],
            'final_heading': heading_angles[-1],
            
            # Summary statistics
            'total_displacement': total_displacement,
            'max_velocity': max_velocity,
            'avg_velocity': avg_velocity,
            'heading_range': heading_range,
            'max_heading_angular_velocity': max_heading_omega,
            'avg_heading_angular_velocity': avg_heading_omega,
            
            # Configuration
            'wrench_applied': wrench,
            'duration': duration,
            'dt': dt,
            'method': 'simulation'
        }
    
    def _apply_wrench_step(self, wrench, dt):
        """
        Apply wrench for one time step using DynamicObjectModel's friction calculation.
        
        Args:
            wrench: Applied wrench [Fx, Fy, τ]
            dt: Time step
        """
        # Use dynamics model's friction calculation
        friction_wrench = self.dynamics._calculate_friction(applied_wrench=wrench)
        
        # Add optional noise
        noise = np.random.normal(0, self.dynamics.friction_noise_std, 3)
        
        # Total wrench
        total_wrench = wrench + friction_wrench + noise
        Fx, Fy, tau = total_wrench
        
        # Update velocities (body frame)
        self.dynamics.velocity_body[0] += (Fx / self.dynamics.mass) * dt
        self.dynamics.velocity_body[1] += (Fy / self.dynamics.mass) * dt
        self.dynamics.angular_velocity += (tau / self.dynamics.moment_of_inertia) * dt
        
        # Update position and orientation
        cos_theta = np.cos(self.dynamics.orientation)
        sin_theta = np.sin(self.dynamics.orientation)
        
        vx_world = self.dynamics.velocity_body[0] * cos_theta - self.dynamics.velocity_body[1] * sin_theta
        vy_world = self.dynamics.velocity_body[0] * sin_theta + self.dynamics.velocity_body[1] * cos_theta
        
        self.dynamics.position[0] += vx_world * dt
        self.dynamics.position[1] += vy_world * dt
        self.dynamics.orientation += self.dynamics.angular_velocity * dt
        
        # Normalize orientation
        self.dynamics.orientation = np.arctan2(np.sin(self.dynamics.orientation), 
                                              np.cos(self.dynamics.orientation))
        
        self.dynamics.time += dt
    
    def predict_segment_motion(self, t_param_endpoints, wrench, duration, dt=0.01,
                              initial_velocity=None, initial_omega=None, num_points=10):
        """
        Predict motion of a boundary segment defined by two t_param values.
        
        Args:
            t_param_endpoints: Two parameter values [t1, t2] defining segment endpoints
            wrench: Applied wrench [Fx, Fy, τ]
            duration: Prediction duration in seconds
            dt: Time step
            initial_velocity: Initial object velocity [vx_body, vy_body]
            initial_omega: Initial angular velocity
            num_points: Number of points to sample along segment
            
        Returns:
            dict: Predicted trajectory of the segment with heading information
        """
        t1, t2 = t_param_endpoints
        
        # Sample t_params along the segment
        t_values = np.linspace(t1, t2, num_points)
        
        # Predict motion for each point
        point_predictions = []
        for t in t_values:
            prediction = self.predict_point_motion(
                t, wrench, duration, dt, initial_velocity, initial_omega
            )
            point_predictions.append(prediction)
        
        # Compile segment trajectory
        times = point_predictions[0]['times']
        segment_trajectories = []
        segment_headings = []  # Average heading of segment
        
        for time_idx in range(len(times)):
            segment_at_time = [pred['point_trajectory'][time_idx] for pred in point_predictions]
            segment_trajectories.append(np.array(segment_at_time))
            
            # Average heading of the segment
            headings_at_time = [pred['heading_angles'][time_idx] for pred in point_predictions]
            avg_heading = np.arctan2(
                np.mean([np.sin(h) for h in headings_at_time]),
                np.mean([np.cos(h) for h in headings_at_time])
            )
            segment_headings.append(avg_heading)
        
        # Calculate segment properties over time
        segment_lengths = []
        segment_angles = []
        segment_midpoints = []
        
        for seg in segment_trajectories:
            # Segment length (should be preserved for rigid body)
            seg_length = np.linalg.norm(seg[-1] - seg[0])
            segment_lengths.append(seg_length)
            
            # Segment angle
            seg_vector = seg[-1] - seg[0]
            seg_angle = np.arctan2(seg_vector[1], seg_vector[0])
            segment_angles.append(seg_angle)
            
            # Segment midpoint
            seg_midpoint = (seg[0] + seg[-1]) / 2
            segment_midpoints.append(seg_midpoint)
        
        # Get contact points for endpoints
        cp1 = self.create_boundary_contact_point(t1)
        cp2 = self.create_boundary_contact_point(t2)
        
        return {
            'times': times,
            'segment_trajectories': segment_trajectories,
            'segment_lengths': np.array(segment_lengths),
            'segment_angles': np.array(segment_angles),
            'segment_midpoints': np.array(segment_midpoints),
            'segment_headings': np.array(segment_headings),
            't_param_endpoints': t_param_endpoints,
            'contact_points': [cp1, cp2],
            'point_predictions': point_predictions,  # Full data for each point
            'initial_segment': segment_trajectories[0],
            'final_segment': segment_trajectories[-1],
            'length_change': segment_lengths[-1] - segment_lengths[0],
            'angle_change': segment_angles[-1] - segment_angles[0],
            'midpoint_displacement': np.linalg.norm(segment_midpoints[-1] - segment_midpoints[0]),
            'wrench_applied': wrench,
            'duration': duration,
            'method': 'simulation'
        }
    
    def compare_with_and_without_wrench(self, t_param, wrench, duration, dt=0.01,
                                       initial_velocity=None, initial_omega=None):
        """
        Compare motion with and without applied wrench (W = 0 vs W ≠ 0).
        
        Args:
            t_param: Parameter value for boundary point
            wrench: Applied wrench [Fx, Fy, τ]
            duration: Prediction duration
            dt: Time step
            initial_velocity: Initial object velocity
            initial_omega: Initial angular velocity
            
        Returns:
            dict: Comparison results with heading information
        """
        # Case 1: No wrench (W = 0)
        print(" Predicting motion WITHOUT wrench...")
        result_no_wrench = self.predict_point_motion(
            t_param, np.zeros(3), duration, dt, initial_velocity, initial_omega
        )
        
        print(" Predicting motion WITH wrench...")
        # Case 2: With wrench (W ≠ 0)
        result_with_wrench = self.predict_point_motion(
            t_param, wrench, duration, dt, initial_velocity, initial_omega
        )
        
        # Calculate differences
        trajectory_diff = result_with_wrench['point_trajectory'] - result_no_wrench['point_trajectory']
        velocity_diff = result_with_wrench['point_velocities'] - result_no_wrench['point_velocities']
        
        # Heading differences
        heading_diff = result_with_wrench['heading_angles'] - result_no_wrench['heading_angles']
        # Unwrap to [-π, π]
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        
        heading_omega_diff = result_with_wrench['heading_angular_velocities'] - result_no_wrench['heading_angular_velocities']
        
        return {
            'without_wrench': result_no_wrench,
            'with_wrench': result_with_wrench,
            'trajectory_difference': trajectory_diff,
            'velocity_difference': velocity_diff,
            'heading_difference': heading_diff,
            'heading_angular_velocity_difference': heading_omega_diff,
            'final_position_diff': np.linalg.norm(trajectory_diff[-1]),
            'max_position_diff': np.max(np.linalg.norm(trajectory_diff, axis=1)),
            'avg_velocity_diff': np.mean(np.linalg.norm(velocity_diff, axis=1)),
            'max_heading_diff': np.max(np.abs(heading_diff)),
            'avg_heading_diff': np.mean(np.abs(heading_diff))
        }
    
    def visualize_boundary_motion(self, t_param=None, t_param_segment=None,
                                 wrench=np.array([5.0, 3.0, 0.5]), duration=2.0,
                                 initial_velocity=None, initial_omega=None,
                                 num_object_poses=6, show_heading=True):
        """
        ENHANCED visualization showing object poses along the trajectory.
        NOW USES data from predict_point_motion directly (no redundant calculations).
        
        Args:
            t_param: Parameter value for point (if None, uses t=0.25)
            t_param_segment: Two parameter values [t1, t2] for segment (optional)
            wrench: Applied wrench
            duration: Prediction duration
            initial_velocity: Initial object velocity
            initial_omega: Initial angular velocity
            num_object_poses: Number of object poses to show along trajectory
            show_heading: If True, show heading (normal inward) analysis
        """
        # Default to t=0.25 if not specified
        if t_param is None:
            t_param = 0.25
        
        # Adjust figure size and layout based on whether we're showing heading
        if show_heading:
            fig = plt.figure(figsize=(20, 18))
            gs = plt.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(20, 14))
            gs = plt.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Run comparison to get both trajectories (with ALL data including heading)
        comparison = self.compare_with_and_without_wrench(
            t_param, wrench, duration, initial_velocity=initial_velocity, initial_omega=initial_omega
        )
        
        # Extract data from comparison results
        times = comparison['without_wrench']['times']
        traj_no_wrench = comparison['without_wrench']['point_trajectory']
        traj_with_wrench = comparison['with_wrench']['point_trajectory']
        contact_point = comparison['with_wrench']['contact_point']
        
        # Heading data (already calculated in predict_point_motion)
        heading_trajectory = comparison['with_wrench']['heading_angles']
        heading_angular_velocity = comparison['with_wrench']['heading_angular_velocities']
        heading_vectors = comparison['with_wrench']['heading_vectors']
        
        # Object state data
        obj_positions = comparison['with_wrench']['object_positions']
        obj_orientations = comparison['with_wrench']['object_orientations']
        
        # 1. MAIN PLOT: Trajectories with object poses (LARGER - spans 2x2)
        ax_main = fig.add_subplot(gs[:2, :2])
        
        # Plot trajectories
        ax_main.plot(traj_no_wrench[:, 0], traj_no_wrench[:, 1], 'b-', linewidth=2.5, 
                    label='No wrench (friction only)', alpha=0.7)
        ax_main.plot(traj_with_wrench[:, 0], traj_with_wrench[:, 1], 'r-', linewidth=2.5,
                    label='With wrench', alpha=0.7)
        
        # Show object poses along trajectory (with wrench case)
        # Select evenly spaced poses
        pose_indices = np.linspace(0, len(obj_positions)-1, num_object_poses, dtype=int)
        
        for i, idx in enumerate(pose_indices):
            pos = obj_positions[idx]
            orient = obj_orientations[idx]
            
            # Create transformed object at this pose
            transformed_obj = self.dynamics.object.transform(
                pos[0], pos[1], orient - self.dynamics.object.heading
            )
            
            # Color gradient from start (blue) to end (red)
            color = plt.cm.RdYlBu_r(i / max(1, num_object_poses-1))
            
            # Visualize object
            transformed_obj.visualize(
                ax=ax_main,
                facecolor=color,
                edgecolor='black',
                alpha=0.4,
                show_frame=False,
                linewidth=1.5
            )
            
            # Highlight the tracked boundary point on each pose
            point_at_idx = traj_with_wrench[idx]
            ax_main.plot(point_at_idx[0], point_at_idx[1], 'o', 
                        color=color, markersize=10, markeredgecolor='black', 
                        markeredgewidth=1.5, zorder=10)
            
            # Show heading (normal inward) at this pose if requested
            if show_heading:
                heading_vec = heading_vectors[idx]
                arrow_length = 0.15
                dx = arrow_length * heading_vec[0]
                dy = arrow_length * heading_vec[1]
                ax_main.arrow(point_at_idx[0], point_at_idx[1], dx, dy,
                             head_width=0.04, head_length=0.03, fc=color, ec='black',
                             linewidth=2, zorder=11, alpha=0.8)
            
            # Add time label
            time_label = f'{times[idx]:.2f}s'
            ax_main.text(pos[0], pos[1], time_label, 
                        fontsize=9, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=color, alpha=0.8, linewidth=2))
        
        # Mark initial point
        ax_main.scatter([contact_point.position[0]], [contact_point.position[1]], 
                       color='green', s=200, zorder=15, label=f'Initial point (t={t_param:.3f})', 
                       marker='*', edgecolors='black', linewidths=2)
        
        ax_main.set_xlabel('X Position (m)', fontsize=12)
        ax_main.set_ylabel('Y Position (m)', fontsize=12)
        heading_note = ' (with heading arrows)' if show_heading else ''
        ax_main.set_title('Object Motion with Boundary Point Tracking' + heading_note + '\n'
                         f'Tracked point: t={t_param:.3f}, Wrench: [{wrench[0]:.1f}, {wrench[1]:.1f}, {wrench[2]:.2f}]',
                         fontsize=13, fontweight='bold')
        ax_main.legend(loc='best', fontsize=11)
        ax_main.grid(True, alpha=0.3)
        ax_main.axis('equal')
        
        # 2. Point velocity (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        
        vel_mag_no_wrench = np.linalg.norm(comparison['without_wrench']['point_velocities'], axis=1)
        vel_mag_with_wrench = np.linalg.norm(comparison['with_wrench']['point_velocities'], axis=1)
        
        ax2.plot(times, vel_mag_no_wrench, 'b-', linewidth=2, label='No wrench')
        ax2.plot(times, vel_mag_with_wrench, 'r-', linewidth=2, label='With wrench')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Point Velocity (m/s)')
        ax2.set_title('Point Velocity Magnitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Position difference (middle right)
        ax3 = fig.add_subplot(gs[1, 2])
        
        position_diff = np.linalg.norm(comparison['trajectory_difference'], axis=1)
        ax3.plot(times, position_diff * 1000, 'purple', linewidth=2)  # Convert to mm
        ax3.fill_between(times, 0, position_diff * 1000, alpha=0.3, color='purple')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position Difference (mm)')
        ax3.set_title('Impact of Wrench on Position')
        ax3.grid(True, alpha=0.3)
        
        # 4. Object centroid trajectory + BOUNDARY POINT TRAJECTORY (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        
        ax4.plot(obj_positions[:, 0], obj_positions[:, 1], 'k-', linewidth=2, alpha=0.7, label='Centroid')
        
        # ADD BOUNDARY POINT TRAJECTORY
        ax4.plot(traj_with_wrench[:, 0], traj_with_wrench[:, 1], 'r--', linewidth=2, 
                 alpha=0.7, label='Boundary Point')
        
        # Mark poses with circles
        for i, idx in enumerate(pose_indices):
            pos = obj_positions[idx]
            color = plt.cm.RdYlBu_r(i / max(1, num_object_poses-1))
            ax4.plot(pos[0], pos[1], 'o', color=color, markersize=12, 
                    markeredgecolor='black', markeredgewidth=1.5)
        
        ax4.scatter([obj_positions[0, 0]], [obj_positions[0, 1]], color='green', s=150, 
                   marker='*', label='Start', edgecolors='black', linewidths=2)
        ax4.scatter([obj_positions[-1, 0]], [obj_positions[-1, 1]], color='red', s=150, 
                   marker='X', label='End', edgecolors='black', linewidths=2)
        
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Object Centroid + Boundary Point Trajectory')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        # 5. Velocity components (bottom middle)
        ax5 = fig.add_subplot(gs[2, 1])
        
        vel_x = comparison['with_wrench']['point_velocities'][:, 0]
        vel_y = comparison['with_wrench']['point_velocities'][:, 1]
        
        ax5.plot(times, vel_x, 'b-', linewidth=2, label='Vx')
        ax5.plot(times, vel_y, 'r-', linewidth=2, label='Vy')
        ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Velocity Component (m/s)')
        ax5.set_title('Point Velocity Components')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        max_diff = comparison['max_position_diff']
        final_diff = comparison['final_position_diff']
        avg_vel_diff = comparison['avg_velocity_diff']
        
        # Calculate some additional stats
        max_vel = comparison['with_wrench']['max_velocity']
        avg_vel = comparison['with_wrench']['avg_velocity']
        total_distance_no_wrench = comparison['without_wrench']['total_displacement']
        total_distance_with_wrench = comparison['with_wrench']['total_displacement']
        
        summary_text = f"""
BOUNDARY MOTION SUMMARY
{'═'*32}

Point: t={t_param:.4f}
Pos: ({contact_point.position[0]:.3f}, {contact_point.position[1]:.3f})

Wrench: [{wrench[0]:.1f}, {wrench[1]:.1f}, {wrench[2]:.2f}]
Duration: {duration:.2f}s

Without Wrench:
  Distance: {total_distance_no_wrench:.4f} m

With Wrench:
  Distance: {total_distance_with_wrench:.4f} m
  Max Vel: {max_vel:.3f} m/s
  Avg Vel: {avg_vel:.3f} m/s

Wrench Impact:
  Max Diff: {max_diff*1000:.2f} mm
  Final Diff: {final_diff*1000:.2f} mm
  Avg Vel Diff: {avg_vel_diff:.3f} m/s
"""
        
        if show_heading:
            heading_range = comparison['with_wrench']['heading_range']
            max_heading_vel = comparison['with_wrench']['max_heading_angular_velocity']
            avg_heading_vel = comparison['with_wrench']['avg_heading_angular_velocity']
            
            summary_text += f"""
Heading (Normal Inward):
  Range: {np.rad2deg(heading_range):.1f}°
  Max ω: {np.rad2deg(max_heading_vel):.1f}°/s
  Avg ω: {np.rad2deg(avg_heading_vel):.1f}°/s
"""
        
        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # If show_heading is enabled, add additional plots in row 4
        if show_heading:
            # 7. Heading angle over time (bottom left, row 4)
            ax7 = fig.add_subplot(gs[3, 0])
            
            ax7.plot(times, np.rad2deg(heading_trajectory), 'g-', linewidth=2.5)
            ax7.set_xlabel('Time (s)', fontsize=11)
            ax7.set_ylabel('Heading Angle (deg)', fontsize=11)
            ax7.set_title('Point Heading (Normal Inward Direction)', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            
            # Mark poses
            for i, idx in enumerate(pose_indices):
                color = plt.cm.RdYlBu_r(i / max(1, num_object_poses-1))
                ax7.plot(times[idx], np.rad2deg(heading_trajectory[idx]), 'o', 
                        color=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            
            # 8. Heading angular velocity over time (bottom middle, row 4)
            ax8 = fig.add_subplot(gs[3, 1])
            
            ax8.plot(times, np.rad2deg(heading_angular_velocity), 'orange', linewidth=2.5)
            ax8.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax8.set_xlabel('Time (s)', fontsize=11)
            ax8.set_ylabel('Heading Angular Velocity (deg/s)', fontsize=11)
            ax8.set_title('Rate of Heading Change', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3)
            
            # Mark poses
            for i, idx in enumerate(pose_indices):
                color = plt.cm.RdYlBu_r(i / max(1, num_object_poses-1))
                ax8.plot(times[idx], np.rad2deg(heading_angular_velocity[idx]), 'o', 
                        color=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            
            # 9. Heading trajectory in polar coordinates (bottom right, row 4)
            ax9 = fig.add_subplot(gs[3, 2], projection='polar')
            
            # Plot heading as polar plot
            scatter = ax9.scatter(heading_trajectory, times, c=times, cmap='viridis', 
                                 s=50, alpha=0.6, edgecolors='black', linewidths=1)
            
            # Mark start and end
            ax9.scatter([heading_trajectory[0]], [times[0]], color='green', s=200, 
                       marker='*', edgecolors='black', linewidths=2, zorder=10, label='Start')
            ax9.scatter([heading_trajectory[-1]], [times[-1]], color='red', s=200, 
                       marker='X', edgecolors='black', linewidths=2, zorder=10, label='End')
            
            ax9.set_title('Heading Direction Over Time\n(Polar View)', fontsize=12, fontweight='bold', pad=20)
            ax9.set_ylabel('Time (s)', labelpad=30)
            ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax9, pad=0.1, shrink=0.8)
            cbar.set_label('Time (s)', rotation=270, labelpad=15)
        
        plt.suptitle(f'Boundary Motion Prediction for {self.dynamics.object.__class__.__name__}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.show()
        
        # Return results for further analysis
        return comparison
# ============================================================================
# SIMPLIFIED DEMO FUNCTION
# ============================================================================

def demo_boundary_motion_predictor():

    """
    Simplified demo - just call visualize_boundary_motion with different scenarios.
    """
    print("\n" + "="*80)
    print("🎯 PROBLEM 2: BOUNDARY MOTION PREDICTION DEMO")
    print("="*80)
    
    
    objects = create_standard_objects()
    
    # Test with different objects and scenarios
    test_scenarios = [
        # {
        #     'object': 'fat_triangle',
        #     't_param': 0.25,
        #     'wrench': np.array([5.0, 3.0, 0.5]),
        #     'initial_velocity': [0.1, 0.05],
        #     'initial_omega': 0.2,
        #     'duration': 2.0,
        #     'description': 'Triangle with combined motion'
        # },
        # {
        #     'object': 'rectangle',
        #     't_param': 0.48,
        #     'wrench': np.array([8.0, 0.0, 1.0]),
        #     'initial_velocity': [0.1, 0.0],
        #     'initial_omega': 0.0,
        #     'duration': 2.0,
        #     'description': 'Rectangle from rest with torque-heavy wrench'
        # },
        {
            'object': 'l_shape',
            't_param': 0.33,
            'wrench': np.zeros(3),  # Friction only
            'initial_velocity': [0.03, 1],
            'initial_omega': 5,
            'duration': 7,
            'description': 'L-shape decelerating by friction only'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'─'*80}")
        print(f"Scenario {i}/{len(test_scenarios)}: {scenario['description']}")
        print(f"{'─'*80}")
        
        obj = objects[scenario['object']]
        dynamics = DynamicObjectModel(obj)
        predictor = BoundaryMotionPredictor(dynamics)
        
        print(f"  Object: {scenario['object']}")
        print(f"  Tracking point: t={scenario['t_param']:.3f}")
        print(f"  Wrench: {scenario['wrench']}")
        print(f"  Initial velocity: {scenario['initial_velocity']} m/s")
        print(f"  Initial ω: {np.degrees(scenario['initial_omega']) if scenario['initial_omega'] else 0:.1f}°/s")
        print(f"  Duration: {scenario['duration']}s")
        
        # Run visualization
        result = predictor.visualize_boundary_motion(
            t_param=scenario['t_param'],
            wrench=scenario['wrench'],
            duration=scenario['duration'],
            initial_velocity=scenario['initial_velocity'],
            initial_omega=scenario['initial_omega'],
            num_object_poses=6
        )
        
        print(f"\n  ✅ Visualization complete")
    
    print("\n" + "="*80)
    print("✅ ALL SCENARIOS COMPLETE")
    print("="*80)
    print("\n💡 Key Observations:")
    print("   ✓ Gray object poses show rigid body motion")
    print("   ✓ Tracked boundary point (colored dots) follows object")
    print("   ✓ Color gradient shows time progression (blue→red)")
    print("   ✓ Comparison shows wrench impact clearly")
    print("   ✓ All points stay on boundary (rigid body constraint)")


# %%
# ============================================================================
# DUMMY CONTROLLER FOR GROUND TRUTH VALIDATION (FIXED)
# ============================================================================

class DummyBoundaryPointController:
    """
    Dummy controller that applies a fixed wrench to the object.
    Used for ground truth validation with simulate_and_animate.
    
    This controller:
    - Applies a constant wrench (either zero or fixed value)
    - Uses 2×E contact points (2 per edge) for guaranteed wrench space coverage
    - Tracks a specific boundary point specified by t_param
    - Provides ground truth for validating prediction methods
    """
    
    def __init__(self, dynamics_model, t_param, fixed_wrench=None):
        """
        Initialize dummy controller.
        
        Args:
            dynamics_model: DynamicObjectModel instance
            t_param: Parameter value for boundary point to track
            fixed_wrench: Fixed wrench to apply [Fx, Fy, τ], or None for W=0
        """
        self.dynamics = dynamics_model
        self.object_model = dynamics_model  # Alias for compatibility
        self.t_param = t_param
        
        # Fixed wrench to apply
        if fixed_wrench is None:
            self.fixed_wrench = np.zeros(3)
        else:
            self.fixed_wrench = np.array(fixed_wrench)
        
        # Create the boundary contact point for tracking
        parameterization = ContactPointParameterization(dynamics_model.object)
        contact_info = parameterization.get_contact_info(t_param)
        self.contact_point = ContactPoint(
            position=contact_info['point'],
            tangent=contact_info['tangent'],
            normal_outward=contact_info['normal_outward'],
            normal_inward=contact_info['normal_inward'],
            parameter=t_param,
            object_ref=dynamics_model.object
        )
        
        # Get point position in body frame (constant throughout motion)
        initial_pos = dynamics_model.position.copy()
        initial_orient = dynamics_model.orientation
        relative_world = self.contact_point.position - initial_pos
        cos_theta = np.cos(-initial_orient)
        sin_theta = np.sin(-initial_orient)
        R_inv = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        self.point_body = R_inv @ relative_world
        
        # Create dummy contact configuration (2 per edge)
        self._setup_dummy_contacts()
        
        # History tracking
        self.history = {
            'times': [],
            'point_positions': [],
            'point_velocities': [],
            'heading_angles': [],
            'heading_vectors': [],
            'object_positions': [],
            'object_orientations': [],
            'object_velocities': [],
            'object_angular_velocities': [],
            'wrenches_applied': []
        }
        
        self.current_state = None
        self.dt = None
    
    def _setup_dummy_contacts(self):
        """
        Create dummy contact points: 2 per edge (2×E total).
        Places contacts near endpoints of each edge for guaranteed wrench space coverage.
        Uses ContactPointParameterization to determine edge boundaries and 
        GenericContactCalculator.calculate_contact_points() with t_param values.
        """
        obj = self.dynamics.object
        
        # Use ContactPointParameterization to get edge information
        parameterization = ContactPointParameterization(obj)
        num_edges = parameterization.n_segments
        
        # Get cumulative distances to determine t_param boundaries for each edge
        cumulative_distances = parameterization.cumulative_distances
        total_length = parameterization.total_length
        
        # Collect t_param values for contact points: 2 per edge
        t_params = []
        offset_ratio = 0.15  # Place contacts at 15% and 85% along each edge
        
        for edge_idx in range(num_edges):
            # Get the cumulative distance at start and end of this edge
            dist_start = cumulative_distances[edge_idx]
            dist_end = cumulative_distances[edge_idx + 1]
            
            # Convert to t_param (normalized by total length)
            t_start = dist_start / total_length
            t_end = dist_end / total_length
            
            # Create 2 t_param values per edge at offset positions
            for offset in [offset_ratio, 1.0 - offset_ratio]:
                t_param = t_start + offset * (t_end - t_start)
                t_params.append(t_param)
        
        print(f"Generated {len(t_params)} t_param values (2 per {num_edges} edges)")
        
        # Use GenericContactCalculator to create ContactPoints from t_params
        calculator = GenericContactCalculator(obj)
        self.dummy_contacts = calculator.calculate_contact_points(
            n_contacts=len(t_params),
            strategy='custom',
            custom_parameters=t_params
        )
        
        print(f"Created {len(self.dummy_contacts)} dummy contact points")
        
        # Calculate the grasp matrix and force distribution
        self._calculate_force_distribution()

    def _calculate_force_distribution(self):
        """
        Calculate how to distribute forces across dummy contacts to achieve desired wrench.
        Uses pseudo-inverse of grasp matrix: forces = G_pinv @ wrench
        """
        # Build grasp matrix G such that wrench = G @ forces
        # Each column corresponds to one contact point
        centroid = self.dynamics.object.get_centroid()
        cx, cy = centroid.x, centroid.y
        
        G = []
        for cp in self.dummy_contacts:
            # Force direction (normal inward)
            fx, fy = cp.normal_inward
            # Moment arm from centroid
            rx = cp.position[0] - cx
            ry = cp.position[1] - cy
            # Torque = r × F (in 2D: rx*fy - ry*fx)
            tau = rx * fy - ry * fx
            
            G.append([fx, fy, tau])
        
        self.grasp_matrix = np.array(G).T  # 3 × (2E) matrix
        
        # Compute pseudo-inverse for wrench -> forces mapping
        # We want minimum norm solution: forces = G_pinv @ wrench
        self.grasp_matrix_pinv = np.linalg.pinv(self.grasp_matrix)
        
        print(f"Grasp matrix shape: {self.grasp_matrix.shape}")
        print(f"Pseudo-inverse shape: {self.grasp_matrix_pinv.shape}")
        
        # Verify the configuration can achieve desired wrench
        test_wrench = np.array([1.0, 1.0, 1.0])
        test_forces = self.grasp_matrix_pinv @ test_wrench
        reconstructed_wrench = self.grasp_matrix @ test_forces
        error = np.linalg.norm(reconstructed_wrench - test_wrench)
        
        if error < 1e-6:
            print(f"✓ Dummy contact configuration verified (reconstruction error: {error:.2e})")
        else:
            print(f"⚠️  Warning: Large reconstruction error ({error:.2e})")
    
    def reset(self):
        """Reset controller state."""
        self.history = {
            'times': [],
            'point_positions': [],
            'point_velocities': [],
            'heading_angles': [],
            'heading_vectors': [],
            'object_positions': [],
            'object_orientations': [],
            'object_velocities': [],
            'object_angular_velocities': [],
            'wrenches_applied': []
        }
        self.current_state = None
    
    def initialize(self, **kwargs):
        """
        Initialize controller (required by interface).
        Returns the dummy contact configuration.
        """
        return self.dummy_contacts
    
    def update(self, state, dt):
        """
        Update controller with current state.
        
        Args:
            state: Dictionary with current object state
            dt: Time step
        """
        self.current_state = state
        self.dt = dt
    
    def get_control_actions(self):
        """
        Return control actions that produce the desired wrench.
        
        Converts the fixed wrench into contact forces using minimum-norm solution.
        
        Returns:
            tuple: (contact_points, force_magnitudes)
        """
        # Special case: If wrench is zero, return no contacts
        # This lets friction act naturally without contact complications
        if np.linalg.norm(self.fixed_wrench) < 1e-10:
            return self.dummy_contacts, [0.0] * len(self.dummy_contacts)
        

        # Calculate force magnitudes needed to produce desired wrench
        # Using minimum-norm solution: f = G_pinv @ W
        force_vector = self.grasp_matrix_pinv @ self.fixed_wrench
        force_magnitudes = force_vector.tolist()
        
        return self.dummy_contacts, force_magnitudes
    
    def post_update(self, simulation_data):
        """
        Process data after state update and collect tracking information.
        
        Args:
            simulation_data: Dictionary with simulation state
        """
        if self.current_state is None:
            return
        
        # Get current object state
        pos = simulation_data['position']
        orient = simulation_data['orientation']
        vel_body = simulation_data['velocity_body']
        omega = simulation_data['angular_velocity']
        time = simulation_data['time']
        
        # Transform boundary point to world frame
        cos_theta = np.cos(orient)
        sin_theta = np.sin(orient)
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        point_world = R @ self.point_body + pos
        
        # Calculate point velocity (v_point = v_object + ω × r)
        r_world = point_world - pos
        v_obj_world = R @ vel_body
        v_rotation = omega * np.array([-r_world[1], r_world[0]])
        v_point = v_obj_world + v_rotation
        
        # Calculate heading (normal inward in world frame)
        parameterization = ContactPointParameterization(self.dynamics.object)
        contact_info = parameterization.get_contact_info(self.t_param)
        normal_body = contact_info['normal_inward']
        normal_world = R @ normal_body
        heading_angle = np.arctan2(normal_world[1], normal_world[0])
        
        # Store history
        self.history['times'].append(time)
        self.history['point_positions'].append(point_world.copy())
        self.history['point_velocities'].append(v_point.copy())
        self.history['heading_angles'].append(heading_angle)
        self.history['heading_vectors'].append(normal_world.copy())
        self.history['object_positions'].append(pos.copy())
        self.history['object_orientations'].append(orient)
        self.history['object_velocities'].append(vel_body.copy())
        self.history['object_angular_velocities'].append(omega)
        self.history['wrenches_applied'].append(self.fixed_wrench.copy())
    
    def update_internal(self):
        """Internal update (required by interface but not used)."""
        pass
    
    def get_results(self):
        """
        Get collected results in format matching predictor output.
        
        Returns:
            dict: Results with same structure as BoundaryMotionPredictor
        """
        # Convert lists to numpy arrays
        times = np.array(self.history['times'])
        point_traj = np.array(self.history['point_positions'])
        point_vels = np.array(self.history['point_velocities'])
        heading_angles = np.array(self.history['heading_angles'])
        heading_vectors = np.array(self.history['heading_vectors'])
        obj_pos = np.array(self.history['object_positions'])
        obj_orient = np.array(self.history['object_orientations'])
        obj_vel = np.array(self.history['object_velocities'])
        obj_omega = np.array(self.history['object_angular_velocities'])
        
        # Calculate heading angular velocities
        heading_omega = np.zeros_like(heading_angles)
        for i in range(1, len(heading_angles)):
            dt = times[i] - times[i-1]
            dheading = heading_angles[i] - heading_angles[i-1]
            dheading = np.arctan2(np.sin(dheading), np.cos(dheading))  # Unwrap
            heading_omega[i] = dheading / dt if dt > 0 else 0.0
        
        # Calculate summary statistics
        total_displacement = np.linalg.norm(point_traj[-1] - point_traj[0]) if len(point_traj) > 0 else 0.0
        max_velocity = np.max(np.linalg.norm(point_vels, axis=1)) if len(point_vels) > 0 else 0.0
        avg_velocity = np.mean(np.linalg.norm(point_vels, axis=1)) if len(point_vels) > 0 else 0.0
        heading_range = np.max(heading_angles) - np.min(heading_angles) if len(heading_angles) > 0 else 0.0
        max_heading_omega = np.max(np.abs(heading_omega)) if len(heading_omega) > 0 else 0.0
        avg_heading_omega = np.mean(np.abs(heading_omega)) if len(heading_omega) > 0 else 0.0
        
        return {
            # Time series data
            'times': times,
            'point_trajectory': point_traj,
            'point_velocities': point_vels,
            'heading_angles': heading_angles,
            'heading_vectors': heading_vectors,
            'heading_angular_velocities': heading_omega,
            'object_positions': obj_pos,
            'object_orientations': obj_orient,
            'object_velocities': obj_vel,
            'object_angular_velocities': obj_omega,
            
            # Initial state
            'contact_point': self.contact_point,
            't_param': self.t_param,
            'initial_point': point_traj[0] if len(point_traj) > 0 else np.zeros(2),
            'initial_heading': heading_angles[0] if len(heading_angles) > 0 else 0.0,
            
            # Final state
            'final_point': point_traj[-1] if len(point_traj) > 0 else np.zeros(2),
            'final_heading': heading_angles[-1] if len(heading_angles) > 0 else 0.0,
            
            # Summary statistics
            'total_displacement': total_displacement,
            'max_velocity': max_velocity,
            'avg_velocity': avg_velocity,
            'heading_range': heading_range,
            'max_heading_angular_velocity': max_heading_omega,
            'avg_heading_angular_velocity': avg_heading_omega,
            
            # Configuration
            'wrench_applied': self.fixed_wrench,
            'duration': times[-1] if len(times) > 0 else 0.0,
            'dt': times[1] - times[0] if len(times) > 1 else 0.01,
            'method': 'ground_truth'
        }



def demo_ground_truth_with_dummy_controller():

    print("\n" + "="*80)
    print("🎯 GROUND TRUTH: DUMMY CONTROLLER WITH simulate_and_animate")
    print("="*80)
    
    objects = create_standard_objects()
    
    # Test scenarios
    test_scenarios = [
        # {
        #     'name': 'Rectangle - From rest with wrench',
        #     'object': objects['rectangle'],
        #     't_param': 0.25,
        #     'wrench': np.array([5.0, 3.0, 0.5]),
        #     'initial_velocity': None,
        #     'initial_omega': None,
        #     'duration': 2.0
        # },
        # {
        #     'name': 'Triangle - Moving with wrench',
        #     'object': objects['triangle'],
        #     't_param': 0.3,
        #     'wrench': np.array([8.0, 5.0, 1.2]),
        #     'initial_velocity': [0.15, 0.08],
        #     'initial_omega': 0.3,
        #     'duration': 2.0
        # },
        {
            'name': 'L-Shape - Friction only (W=0)',
            'object': objects['l_shape'],
            't_param': 0.74,
            'wrench': np.zeros(3),
            'initial_velocity': [0.25, -0.15],
            'initial_omega': -0.4,
            'duration': 2.5
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'─'*80}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"{'─'*80}")
        
        # Create dynamics model
        dynamics = DynamicObjectModel(scenario['object'])
        
        # Set initial conditions
        if scenario['initial_velocity'] is not None:
            dynamics.velocity_body = np.array(scenario['initial_velocity'])
        if scenario['initial_omega'] is not None:
            dynamics.angular_velocity = scenario['initial_omega']
        
        # Create dummy controller
        controller = DummyBoundaryPointController(
            dynamics, 
            t_param=scenario['t_param'],
            fixed_wrench=scenario['wrench']
        )
        
        print(f"\n📋 Configuration:")
        print(f"   Object: {scenario['object'].__class__.__name__}")
        print(f"   t_param: {scenario['t_param']}")
        print(f"   Wrench: {scenario['wrench']}")
        print(f"   Initial velocity: {scenario['initial_velocity']}")
        print(f"   Initial ω: {np.rad2deg(scenario['initial_omega']) if scenario['initial_omega'] else 0:.1f}°/s")
        print(f"   Duration: {scenario['duration']}s")
        
        # Run simulation with dummy controller
        print(f"\n⚙️  Running simulate_and_animate (ground truth)...")
        sim_results = dynamics.simulate_and_animate(
            controller,
            duration=scenario['duration'],
            dt=0.01,
            fps=0,  # No animation
            stream=False
        )
        
        # Get controller results
        ground_truth = controller.get_results()
        
        print(f"\n✅ Ground truth collected!")
        print(f"   Time steps: {len(ground_truth['times'])}")
        print(f"   Final point position: {ground_truth['final_point']}")
        print(f"   Total displacement: {ground_truth['total_displacement']:.4f} m")
        print(f"   Max velocity: {ground_truth['max_velocity']:.4f} m/s")
        print(f"   Heading range: {np.rad2deg(ground_truth['heading_range']):.2f}°")
        
        # ENHANCED Visualization with object poses and heading vectors
        fig = plt.figure(figsize=(20, 14))
        gs = plt.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        times = ground_truth['times']
        
        # 1. MAIN PLOT: Trajectory with object poses (LARGER - spans 2x2)
        ax_main = fig.add_subplot(gs[:2, :2])
        
        # Plot boundary point trajectory
        ax_main.plot(ground_truth['point_trajectory'][:, 0], 
                    ground_truth['point_trajectory'][:, 1],
                    'g-', linewidth=2.5, label='Boundary Point', alpha=0.7)
        
        # Show object poses along trajectory
        obj_positions = ground_truth['object_positions']
        obj_orientations = ground_truth['object_orientations']
        
        # Select evenly spaced poses (6 poses)
        num_object_poses = 6
        pose_indices = np.linspace(0, len(obj_positions)-1, num_object_poses, dtype=int)
        
        for idx_i, idx in enumerate(pose_indices):
            pos = obj_positions[idx]
            orient = obj_orientations[idx]
            
            # Create transformed object at this pose
            transformed_obj = dynamics.object.transform(
                pos[0], pos[1], orient - dynamics.object.heading
            )
            
            # Color gradient from start (blue) to end (red)
            color = plt.cm.RdYlBu_r(idx_i / max(1, num_object_poses-1))
            
            # Visualize object
            transformed_obj.visualize(
                ax=ax_main,
                facecolor=color,
                edgecolor='black',
                alpha=0.4,
                show_frame=False,
                linewidth=1.5
            )
            
            # Highlight the tracked boundary point on each pose
            point_at_idx = ground_truth['point_trajectory'][idx]
            ax_main.plot(point_at_idx[0], point_at_idx[1], 'o', 
                        color=color, markersize=10, markeredgecolor='black', 
                        markeredgewidth=1.5, zorder=10)
            
            # Show heading (normal inward) at this pose
            heading_vec = ground_truth['heading_vectors'][idx]
            arrow_length = 0.15
            dx = arrow_length * heading_vec[0]
            dy = arrow_length * heading_vec[1]
            ax_main.arrow(point_at_idx[0], point_at_idx[1], dx, dy,
                         head_width=0.04, head_length=0.03, fc=color, ec='black',
                         linewidth=2, zorder=11, alpha=0.8)
            
            # Add time label at object centroid
            time_label = f'{times[idx]:.2f}s'
            ax_main.text(pos[0], pos[1], time_label, 
                        fontsize=9, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=color, alpha=0.8, linewidth=2))
        
        # Mark initial point
        ax_main.scatter([ground_truth['initial_point'][0]], 
                       [ground_truth['initial_point'][1]], 
                       color='green', s=200, zorder=15, 
                       label=f'Initial point (t={scenario["t_param"]:.3f})', 
                       marker='*', edgecolors='black', linewidths=2)
        
        ax_main.set_xlabel('X Position (m)', fontsize=12)
        ax_main.set_ylabel('Y Position (m)', fontsize=12)
        ax_main.set_title('Ground Truth: Object Motion with Boundary Point Tracking\n'
                         f'Tracked point: t={scenario["t_param"]:.3f}, Wrench: {scenario["wrench"]}',
                         fontsize=13, fontweight='bold')
        ax_main.legend(loc='best', fontsize=11)
        ax_main.grid(True, alpha=0.3)
        ax_main.axis('equal')
        
        # 2. Point velocity
        ax2 = fig.add_subplot(gs[0, 2])
        vel_mag = np.linalg.norm(ground_truth['point_velocities'], axis=1)
        ax2.plot(times, vel_mag, 'g-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity Magnitude (m/s)')
        ax2.set_title('Point Velocity')
        ax2.grid(True, alpha=0.3)
        
        # 3. Heading angle
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.plot(times, np.rad2deg(ground_truth['heading_angles']), 'g-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Heading Angle (deg)')
        ax3.set_title('Point Heading (Normal Inward)')
        ax3.grid(True, alpha=0.3)
        
        # Mark poses on heading plot
        for idx_i, idx in enumerate(pose_indices):
            color = plt.cm.RdYlBu_r(idx_i / max(1, num_object_poses-1))
            ax3.plot(times[idx], np.rad2deg(ground_truth['heading_angles'][idx]), 'o', 
                    color=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        
        # 4. Object centroid + boundary point trajectory
        ax4 = fig.add_subplot(gs[2, 0])
        obj_pos = ground_truth['object_positions']
        ax4.plot(obj_pos[:, 0], obj_pos[:, 1], 'k-', linewidth=2, alpha=0.7, label='Centroid')
        ax4.plot(ground_truth['point_trajectory'][:, 0],
                ground_truth['point_trajectory'][:, 1],
                'g--', linewidth=2, alpha=0.7, label='Boundary Point')
        
        # Mark poses with circles
        for idx_i, idx in enumerate(pose_indices):
            pos = obj_positions[idx]
            color = plt.cm.RdYlBu_r(idx_i / max(1, num_object_poses-1))
            ax4.plot(pos[0], pos[1], 'o', color=color, markersize=12, 
                    markeredgecolor='black', markeredgewidth=1.5)
        
        ax4.scatter([obj_pos[0, 0]], [obj_pos[0, 1]], color='green', s=150, 
                   marker='*', label='Start', edgecolors='black', linewidths=2)
        ax4.scatter([obj_pos[-1, 0]], [obj_pos[-1, 1]], color='red', s=150, 
                   marker='X', label='End', edgecolors='black', linewidths=2)
        
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Object Centroid + Boundary Point Trajectory')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        # 5. Velocity components
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(times, ground_truth['point_velocities'][:, 0], 'b-', linewidth=2, label='Vx')
        ax5.plot(times, ground_truth['point_velocities'][:, 1], 'r-', linewidth=2, label='Vy')
        ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Velocity Component (m/s)')
        ax5.set_title('Point Velocity Components')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Object orientation and angular velocity
        ax6 = fig.add_subplot(gs[2, 2])
        ax6_twin = ax6.twinx()
        
        line1 = ax6.plot(times, np.rad2deg(ground_truth['object_orientations']), 
                        'b-', linewidth=2, label='Orientation')
        line2 = ax6_twin.plot(times, np.rad2deg(ground_truth['object_angular_velocities']), 
                             'r-', linewidth=2, label='Angular Velocity')
        
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Orientation (deg)', color='b')
        ax6_twin.set_ylabel('Angular Velocity (deg/s)', color='r')
        ax6.tick_params(axis='y', labelcolor='b')
        ax6_twin.tick_params(axis='y', labelcolor='r')
        ax6.set_title('Object Rotation')
        ax6.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='best')
        
        plt.suptitle(f'Ground Truth: {scenario["name"]}\n'
                    f'Using simulate_and_animate with DummyBoundaryPointController',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    print(f"\n{'='*80}")
    print("✅ GROUND TRUTH DEMO COMPLETE")
    print(f"{'='*80}\n")


# %%
# ============================================================================
# VALIDATION COMPARISON FUNCTION (SINGLE PREDICTOR)
# ============================================================================

def compare_predictor_vs_ground_truth():
    """
    Compare BoundaryMotionPredictor against ground truth from simulate_and_animate.
    This validates the predictor against the established simulation standard.
    """
    import time
    
    print("\n" + "="*80)
    print("🔬 VALIDATION: BoundaryMotionPredictor vs GROUND TRUTH")
    print("="*80)
    
    objects = create_standard_objects()
    
    # Test scenarios
    test_cases = [
        # {
        #     'name': 'Rectangle - From rest with wrench',
        #     'object': objects['rectangle'],
        #     't_param': 0.25,
        #     'wrench': np.array([5.0, 3.0, 5]),
        #     'velocity': [0.0, 0.1],
        #     'omega': None,
        #     'duration': 1.5
        # },
        # {
        #     'name': 'Triangle - Moving with wrench',
        #     'object': objects['triangle'],
        #     't_param': 0.3,
        #     'wrench': np.array([8.0, 5.0, 4.0]),
        #     'velocity': [0.15, 0.08],
        #     'omega': 0.3,
        #     'duration': 1.5
        # },
        {
            'name': 'L-Shape - Friction only (W=0)',
            'object': objects['l_shape'],
            't_param': 0.3,
            'wrench': np.zeros(3),
            'velocity': [0.2, 0.1],
            'omega': 5,
            'duration': 30
        }
    ]
    
    all_results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"Test Case {i}/{len(test_cases)}: {test['name']}")
        print(f"{'─'*80}")
        
        print(f"\n📋 Configuration:")
        print(f"   Duration: {test['duration']}s")
        print(f"   t_param: {test['t_param']}")
        print(f"   Wrench: {test['wrench']}")
        print(f"   Initial velocity: {test['velocity']}")
        print(f"   Initial ω: {np.rad2deg(test['omega']) if test['omega'] else 0:.1f}°/s")
        
        # 1. GROUND TRUTH: simulate_and_animate + DummyController
        print(f"\n🎯 Collecting GROUND TRUTH...")
        
        # Reset dynamics for ground truth run
        dynamics_gt = DynamicObjectModel(test['object'])
        if test['velocity'] is not None:
            dynamics_gt.velocity_body = np.array(test['velocity'])
        if test['omega'] is not None:
            dynamics_gt.angular_velocity = test['omega']
        controller_gt = DummyBoundaryPointController(dynamics_gt, test['t_param'], test['wrench'])
        
        t_start = time.time()
        sim_results = dynamics_gt.simulate_and_animate(
            controller_gt, duration=test['duration'], dt=0.01, fps=0, stream=False
        )
        t_ground_truth = time.time() - t_start
        ground_truth = controller_gt.get_results()
        print(f"   ✓ Ground truth: {t_ground_truth:.4f}s ({len(ground_truth['times'])} steps)")
        
        # DEBUG: Check if object is actually rotating
        obj_orient = ground_truth['object_orientations']
        orient_change = np.rad2deg(obj_orient[-1] - obj_orient[0])
        obj_omega = ground_truth['object_angular_velocities']
        max_omega = np.max(np.abs(obj_omega))
        
        print(f"\n   🔍 GROUND TRUTH DEBUG:")
        print(f"      Wrench applied: {test['wrench']}")
        print(f"      Initial orientation: {np.rad2deg(obj_orient[0]):.2f}°")
        print(f"      Final orientation: {np.rad2deg(obj_orient[-1]):.2f}°")
        print(f"      Total rotation: {orient_change:.2f}°")
        print(f"      Max angular velocity: {np.rad2deg(max_omega):.2f}°/s")
        
        if abs(orient_change) < 0.1 and abs(test['wrench'][2]) > 0.01:
            print(f"      ⚠️  WARNING: Torque applied but no rotation detected!")
            print(f"      ⚠️  This suggests DummyController may not be applying wrench correctly")
        
        # 2. PREDICTOR: BoundaryMotionPredictor (numerical integration)
        print(f"\n⏱️  Testing BoundaryMotionPredictor...")
        dynamics_p1 = DynamicObjectModel(test['object'])
        predictor = BoundaryMotionPredictor(dynamics_p1)
        
        t_start = time.time()
        result_pred = predictor.predict_point_motion(
            test['t_param'], test['wrench'], test['duration'], dt=0.01,
            initial_velocity=test['velocity'], initial_omega=test['omega']
        )
        t_pred = time.time() - t_start
        print(f"   ✓ Predictor: {t_pred:.4f}s ({len(result_pred['times'])} steps)")
        
        # DEBUG: Check predictor rotation
        pred_orient = result_pred['object_orientations']
        pred_orient_change = np.rad2deg(pred_orient[-1] - pred_orient[0])
        pred_omega = result_pred['object_angular_velocities']
        pred_max_omega = np.max(np.abs(pred_omega))
        
        print(f"\n   🔍 PREDICTOR DEBUG:")
        print(f"      Initial orientation: {np.rad2deg(pred_orient[0]):.2f}°")
        print(f"      Final orientation: {np.rad2deg(pred_orient[-1]):.2f}°")
        print(f"      Total rotation: {pred_orient_change:.2f}°")
        print(f"      Max angular velocity: {np.rad2deg(pred_max_omega):.2f}°/s")
        
        # TRIM to match the shortest length
        min_length = min(len(ground_truth['times']), len(result_pred['times']))
        print(f"\n   ⚙️  Trimming all results to {min_length} steps for comparison")
        
        # Trim ground truth
        gt_trimmed = {
            'times': ground_truth['times'][:min_length],
            'point_trajectory': ground_truth['point_trajectory'][:min_length],
            'point_velocities': ground_truth['point_velocities'][:min_length],
            'heading_angles': ground_truth['heading_angles'][:min_length],
            'heading_vectors': ground_truth['heading_vectors'][:min_length],
            'object_positions': ground_truth['object_positions'][:min_length],
            'object_orientations': ground_truth['object_orientations'][:min_length],
            'object_angular_velocities': ground_truth['object_angular_velocities'][:min_length]
        }
        
        # Trim predictor
        pred_trimmed = {
            'times': result_pred['times'][:min_length],
            'point_trajectory': result_pred['point_trajectory'][:min_length],
            'point_velocities': result_pred['point_velocities'][:min_length],
            'heading_angles': result_pred['heading_angles'][:min_length],
            'heading_vectors': result_pred['heading_vectors'][:min_length],
            'object_positions': result_pred['object_positions'][:min_length],
            'object_orientations': result_pred['object_orientations'][:min_length],
            'object_angular_velocities': result_pred['object_angular_velocities'][:min_length]
        }
        
        # Calculate errors vs ground truth
        print(f"\n📊 Error Analysis:")
        
        # Position error
        pos_error = np.linalg.norm(pred_trimmed['point_trajectory'] - gt_trimmed['point_trajectory'], axis=1)
        vel_error = np.linalg.norm(pred_trimmed['point_velocities'] - gt_trimmed['point_velocities'], axis=1)
        heading_error = np.abs(pred_trimmed['heading_angles'] - gt_trimmed['heading_angles'])
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Orientation error
        orient_error = np.abs(pred_trimmed['object_orientations'] - gt_trimmed['object_orientations'])
        orient_error = np.arctan2(np.sin(orient_error), np.cos(orient_error))
        omega_error = np.abs(pred_trimmed['object_angular_velocities'] - gt_trimmed['object_angular_velocities'])
        
        print(f"\n   Boundary Point Errors:")
        print(f"      Position:  Mean={np.mean(pos_error)*1000:.3f}mm, Max={np.max(pos_error)*1000:.3f}mm")
        print(f"      Velocity:  Mean={np.mean(vel_error):.6f}m/s, Max={np.max(vel_error):.6f}m/s")
        print(f"      Heading:   Mean={np.rad2deg(np.mean(np.abs(heading_error))):.3f}°, Max={np.rad2deg(np.max(np.abs(heading_error))):.3f}°")
        
        print(f"\n   Object State Errors:")
        print(f"      Orientation:  Mean={np.rad2deg(np.mean(np.abs(orient_error))):.3f}°, Max={np.rad2deg(np.max(np.abs(orient_error))):.3f}°")
        print(f"      Angular vel:  Mean={np.rad2deg(np.mean(omega_error)):.3f}°/s, Max={np.rad2deg(np.max(omega_error)):.3f}°/s")
        
        # Determine accuracy
        mean_pos_error = np.mean(pos_error)
        
        if mean_pos_error < 0.001:
            status = "✅ EXCELLENT"
        elif mean_pos_error < 0.01:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ POOR - NEEDS INVESTIGATION"
        
        print(f"\n   Status: {status}")
        
        # Store results
        all_results.append({
            'name': test['name'],
            'ground_truth': ground_truth,
            'predictor': result_pred,
            'timing': {
                'ground_truth': t_ground_truth,
                'predictor': t_pred
            },
            'errors': {
                'pos_mean': np.mean(pos_error),
                'pos_max': np.max(pos_error),
                'vel_mean': np.mean(vel_error),
                'orient_mean': np.mean(np.abs(orient_error)),
                'orient_max': np.max(np.abs(orient_error)),
                'omega_mean': np.mean(omega_error)
            },
            'rotation_debug': {
                'gt_total_rotation': orient_change,
                'gt_max_omega': np.rad2deg(max_omega),
                'pred_total_rotation': pred_orient_change,
                'pred_max_omega': np.rad2deg(pred_max_omega)
            }
        })
        
        # Visualize comparison (use trimmed data)
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        times = gt_trimmed['times']
        
        # 1. Trajectory comparison
        ax1 = axes[0, 0]
        ax1.plot(gt_trimmed['point_trajectory'][:, 0], gt_trimmed['point_trajectory'][:, 1],
                'g-', linewidth=3, label='Ground Truth', alpha=0.8)
        ax1.plot(pred_trimmed['point_trajectory'][:, 0], pred_trimmed['point_trajectory'][:, 1],
                'b--', linewidth=2, label='Predictor', alpha=0.7)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Boundary Point Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. Position error vs time
        ax2 = axes[0, 1]
        ax2.plot(times, pos_error * 1000, 'b-', linewidth=2)
        ax2.fill_between(times, 0, pos_error * 1000, alpha=0.3, color='blue')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (mm)')
        ax2.set_title(f'Position Error\nMean: {np.mean(pos_error)*1000:.3f}mm')
        ax2.grid(True, alpha=0.3)
        
        # 3. Velocity comparison
        ax3 = axes[0, 2]
        vel_mag_gt = np.linalg.norm(gt_trimmed['point_velocities'], axis=1)
        vel_mag_pred = np.linalg.norm(pred_trimmed['point_velocities'], axis=1)
        ax3.plot(times, vel_mag_gt, 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
        ax3.plot(times, vel_mag_pred, 'b--', linewidth=2, label='Predictor', alpha=0.7)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity Magnitude (m/s)')
        ax3.set_title('Point Velocity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Object orientation comparison
        ax4 = axes[1, 0]
        ax4.plot(times, np.rad2deg(gt_trimmed['object_orientations']), 'g-', linewidth=3,
                label='Ground Truth', alpha=0.8)
        ax4.plot(times, np.rad2deg(pred_trimmed['object_orientations']), 'b--', linewidth=2,
                label='Predictor', alpha=0.7)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Orientation (deg)')
        ax4.set_title('Object Orientation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Angular velocity comparison
        ax5 = axes[1, 1]
        ax5.plot(times, np.rad2deg(gt_trimmed['object_angular_velocities']), 'g-', linewidth=3,
                label='Ground Truth', alpha=0.8)
        ax5.plot(times, np.rad2deg(pred_trimmed['object_angular_velocities']), 'b--', linewidth=2,
                label='Predictor', alpha=0.7)
        ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Angular Velocity (deg/s)')
        ax5.set_title('Object Angular Velocity')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Orientation error
        ax6 = axes[1, 2]
        ax6.plot(times, np.rad2deg(np.abs(orient_error)), 'r-', linewidth=2)
        ax6.fill_between(times, 0, np.rad2deg(np.abs(orient_error)), alpha=0.3, color='red')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Orientation Error (deg)')
        ax6.set_title(f'Orientation Error\nMean: {np.rad2deg(np.mean(np.abs(orient_error))):.3f}°')
        ax6.grid(True, alpha=0.3)
        
        # 7. Heading comparison
        ax7 = axes[2, 0]
        ax7.plot(times, np.rad2deg(gt_trimmed['heading_angles']), 'g-', linewidth=3,
                label='Ground Truth', alpha=0.8)
        ax7.plot(times, np.rad2deg(pred_trimmed['heading_angles']), 'b--', linewidth=2,
                label='Predictor', alpha=0.7)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Heading Angle (deg)')
        ax7.set_title('Point Heading (Normal Inward)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Object centroid trajectory
        ax8 = axes[2, 1]
        ax8.plot(gt_trimmed['object_positions'][:, 0], gt_trimmed['object_positions'][:, 1],
                'g-', linewidth=3, label='Ground Truth', alpha=0.8)
        ax8.plot(pred_trimmed['object_positions'][:, 0], pred_trimmed['object_positions'][:, 1],
                'b--', linewidth=2, label='Predictor', alpha=0.7)
        ax8.set_xlabel('X Position (m)')
        ax8.set_ylabel('Y Position (m)')
        ax8.set_title('Object Centroid Trajectory')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.axis('equal')
        
        # 9. Summary
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        summary_text = f"""
VALIDATION SUMMARY
{'═'*32}

Ground Truth:
  Method: simulate_and_animate
  Time: {t_ground_truth:.4f}s
  Rotation: {orient_change:.2f}°

Predictor:
  BoundaryMotionPredictor
  Time: {t_pred:.4f}s
  Rotation: {pred_orient_change:.2f}°

Errors:
  Position: {mean_pos_error*1000:.3f}mm
  Velocity: {np.mean(vel_error):.6f}m/s
  Orient: {np.rad2deg(np.mean(np.abs(orient_error))):.3f}°
  Ang Vel: {np.rad2deg(np.mean(omega_error)):.3f}°/s

Status: {status}

Compared: {min_length} steps

{'⚠️ Ground truth shows no rotation!' if abs(orient_change) < 0.1 and abs(test['wrench'][2]) > 0.01 else '✓ Rotation detected'}
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.suptitle(f'Validation: {test["name"]}\nPredictor vs Ground Truth (simulate_and_animate)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # Overall summary
    print(f"\n{'='*80}")
    print("📊 OVERALL VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(all_results, 1):
        print(f"{i}. {result['name']}")
        print(f"   Position error: {result['errors']['pos_mean']*1000:.3f}mm")
        print(f"   Orientation error: {np.rad2deg(result['errors']['orient_mean']):.3f}°")
        print(f"   Ground truth rotation: {result['rotation_debug']['gt_total_rotation']:.2f}°")
        print(f"   Predictor rotation: {result['rotation_debug']['pred_total_rotation']:.2f}°")
        
        if abs(result['rotation_debug']['gt_total_rotation']) < 0.1:
            print(f"   ⚠️  WARNING: Ground truth shows no rotation despite torque!")
    
    avg_pos_error = np.mean([r['errors']['pos_mean'] for r in all_results])
    avg_orient_error = np.mean([r['errors']['orient_mean'] for r in all_results])
    
    print(f"\n💡 Key Findings:")
    print(f"   Average position error: {avg_pos_error*1000:.3f}mm")
    print(f"   Average orientation error: {np.rad2deg(avg_orient_error):.3f}°")
    
    # Check for suspicious cases
    suspicious_cases = [r for r in all_results if abs(r['rotation_debug']['gt_total_rotation']) < 0.1 
                       and abs(r['ground_truth']['wrench_applied'][2]) > 0.01]
    
    if suspicious_cases:
        print(f"\n⚠️  SUSPICIOUS CASES DETECTED:")
        print(f"   {len(suspicious_cases)} case(s) have torque but no rotation in ground truth")
        print(f"   → DummyController may not be applying wrench correctly")
        print(f"   → Check how simulate_and_animate handles direct wrench application")
    else:
        print(f"\n✅ All test cases show expected rotation behavior")
    
    if avg_pos_error < 0.005:
        print(f"\n✅ BoundaryMotionPredictor matches ground truth well!")
    else:
        print(f"\n⚠️  BoundaryMotionPredictor has discrepancies - needs investigation")
    
    print(f"\n   Ground truth established via simulate_and_animate")
    print(f"   This is your codebase standard - predictor should match it")
    
    print(f"\n{'='*80}")
    print("✅ VALIDATION COMPLETE")
    print(f"{'='*80}\n")
    
    return all_results

