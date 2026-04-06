# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from scipy.io import loadmat, savemat
from scipy.spatial import KDTree
from collections import defaultdict

import osqp
import scipy.sparse as sparse

import copy
import heapq
import socket
import struct
import pickle

from scipy.interpolate import CubicSpline, interp1d

from object_utils import (
    GenericObject, 
    create_standard_objects,
    stream_figure
)

# %%
# ============================================================================
# HYBRID PATH SYSTEM
# ============================================================================
# This system allows combining multiple component paths (spline, straight, arc)
# into a single hybrid path. Focus is on curvature, especially at intersections.
# ============================================================================

class ComponentPath:
    """
    Abstract base class for path components in a hybrid path.
    Each component path must provide position, tangent, and curvature information.
    Uses arc length s ∈ [0, path_length] as the parameter.
    """
    
    def __init__(self, start_point, end_point):
        """
        Initialize a component path.
        
        Args:
            start_point: (x, y) starting point
            end_point: (x, y) ending point
        """
        self.start_point = np.array(start_point[:2])
        self.end_point = np.array(end_point[:2])
        self.path_length = None  # Will be computed by subclass (arc length in meters)
    
    def get_point_at_arc_length(self, s):
        """
        Get point at arc length s ∈ [0, path_length] along this component.
        
        Args:
            s: Arc length parameter [0, path_length]
            
        Returns:
            (x, y) point
        """
        raise NotImplementedError
    
    def get_tangent_at_arc_length(self, s):
        """
        Get unit tangent vector at arc length s.
        
        Args:
            s: Arc length parameter [0, path_length]
            
        Returns:
            (dx, dy) unit tangent vector
        """
        raise NotImplementedError
    
    def get_curvature_at_arc_length(self, s):
        """
        Get curvature K(s) at arc length s.
        
        Args:
            s: Arc length parameter [0, path_length]
            
        Returns:
            curvature value (1/m)
        """
        raise NotImplementedError
    
    def get_first_derivative_at_arc_length(self, s):
        """
        Get first derivative with respect to arc length s.
        For paths with constant curvature (straight, arc), this should be zero.
        
        Args:
            s: Arc length parameter [0, path_length]
            
        Returns:
            (dx/ds, dy/ds) first derivative vector
        """
        raise NotImplementedError
    
    def get_second_derivative_at_arc_length(self, s):
        """
        Get second derivative with respect to arc length s.
        For paths with constant curvature (straight, arc), this should be zero.
        
        Args:
            s: Arc length parameter [0, path_length]
            
        Returns:
            (d²x/ds², d²y/ds²) second derivative vector
        """
        raise NotImplementedError
    
    def get_path_length(self):
        """Get the arc length (path_length) of this component path."""
        if self.path_length is None:
            raise ValueError("path_length must be set by subclass")
        return self.path_length
    
    # Legacy methods for backward compatibility (convert to arc length)
    def get_point_at_parameter(self, t):
        """Legacy: Get point at normalized parameter t ∈ [0, 1]"""
        s = t * self.get_path_length()
        return self.get_point_at_arc_length(s)
    
    def get_tangent_at_parameter(self, t):
        """Legacy: Get tangent at normalized parameter t ∈ [0, 1]"""
        s = t * self.get_path_length()
        return self.get_tangent_at_arc_length(s)
    
    def get_curvature_at_parameter(self, t):
        """Legacy: Get curvature at normalized parameter t ∈ [0, 1]"""
        s = t * self.get_path_length()
        return self.get_curvature_at_arc_length(s)
    
    def get_first_derivative_at_parameter(self, t):
        """Legacy: Get first derivative at normalized parameter t ∈ [0, 1]"""
        s = t * self.get_path_length()
        return self.get_first_derivative_at_arc_length(s)
    
    def get_second_derivative_at_parameter(self, t):
        """Legacy: Get second derivative at normalized parameter t ∈ [0, 1]"""
        s = t * self.get_path_length()
        return self.get_second_derivative_at_arc_length(s)
    
    def get_length(self):
        """Legacy: Alias for get_path_length()"""
        return self.get_path_length()


class SplineComponentPath(ComponentPath):
    """
    Component path using cubic spline interpolation (wraps SplineReferencePath).
    Uses numerical differentiation with respect to arc length s.
    """
    
    def __init__(self, waypoints):
        """
        Initialize spline component path.
        
        Args:
            waypoints: List of (x, y) or (x, y, theta) points
        """
        # Convert to (x, y, theta) format if needed
        waypoints_array = np.array(waypoints)
        if waypoints_array.shape[-1] == 2:
            # Add dummy theta values
            waypoints_3d = np.zeros((len(waypoints), 3))
            waypoints_3d[:, :2] = waypoints_array
            waypoints = waypoints_3d.tolist()
        
        super().__init__(waypoints[0], waypoints[-1])
        self.spline_path = SplineReferencePath(waypoints, orientation_mode="follow_path")
        self.path_length = self.spline_path.total_path_length
        
        # Numerical differentiation step size
        self.ds = 1e-6  # Small step for numerical differentiation
    
    def s_to_t(self, s):
        """
        Convert arc length s to normalized parameter t ∈ [0, 1].
        
        Args:
            s: Arc length [0, path_length]
            
        Returns:
            t: Normalized parameter [0, 1]
        """
        s = np.clip(s, 0.0, self.path_length)
        if self.path_length > 1e-10:
            return s / self.path_length
        return 0.0
    
    def t_to_s(self, t):
        """
        Convert normalized parameter t ∈ [0, 1] to arc length s.
        
        Args:
            t: Normalized parameter [0, 1]
            
        Returns:
            s: Arc length [0, path_length]
        """
        t = np.clip(t, 0.0, 1.0)
        return t * self.path_length
    
    def get_point_at_arc_length(self, s):
        """Get point at arc length s ∈ [0, path_length]"""
        t = self.s_to_t(s)
        point = self.spline_path.get_point_at_parameter(t)
        return point[:2]
    
    def get_tangent_at_arc_length(self, s):
        """Get unit tangent vector at arc length s"""
        t = self.s_to_t(s)
        return self.spline_path.get_tangent_at_parameter(t)
    
    def get_curvature_at_arc_length(self, s):
        """Get curvature K(s) at arc length s"""
        t = self.s_to_t(s)
        return self.spline_path.get_curvature_at_parameter(t)
    
    def get_first_derivative_at_arc_length(self, s):
        """
        Get first derivative with respect to arc length s using numerical differentiation.
        dP/ds = (P(s+ds) - P(s-ds)) / (2*ds)
        """
        s = np.clip(s, 0.0, self.path_length)
        ds = self.ds
        
        # Handle boundaries
        if s < ds:
            # Forward difference at start
            s1 = s
            s2 = min(s + ds, self.path_length)
            p1 = self.get_point_at_arc_length(s1)
            p2 = self.get_point_at_arc_length(s2)
            if s2 - s1 > 1e-10:
                return (p2 - p1) / (s2 - s1)
            else:
                return np.array([0.0, 0.0])
        elif s > self.path_length - ds:
            # Backward difference at end
            s1 = max(s - ds, 0.0)
            s2 = s
            p1 = self.get_point_at_arc_length(s1)
            p2 = self.get_point_at_arc_length(s2)
            if s2 - s1 > 1e-10:
                return (p2 - p1) / (s2 - s1)
            else:
                return np.array([0.0, 0.0])
        else:
            # Central difference
            s1 = s - ds
            s2 = s + ds
            p1 = self.get_point_at_arc_length(s1)
            p2 = self.get_point_at_arc_length(s2)
            return (p2 - p1) / (2 * ds)
    
    def get_second_derivative_at_arc_length(self, s):
        """
        Get second derivative with respect to arc length s using numerical differentiation.
        d²P/ds² = (P(s+ds) - 2*P(s) + P(s-ds)) / ds²
        """
        s = np.clip(s, 0.0, self.path_length)
        ds = self.ds
        
        # Handle boundaries
        if s < ds:
            # Use forward difference approximation
            s0 = s
            s1 = min(s + ds, self.path_length)
            s2 = min(s + 2*ds, self.path_length)
            p0 = self.get_point_at_arc_length(s0)
            p1 = self.get_point_at_arc_length(s1)
            p2 = self.get_point_at_arc_length(s2)
            if s2 - s0 > 1e-10:
                return (p2 - 2*p1 + p0) / (ds**2)
            else:
                return np.array([0.0, 0.0])
        elif s > self.path_length - ds:
            # Use backward difference approximation
            s0 = max(s - 2*ds, 0.0)
            s1 = max(s - ds, 0.0)
            s2 = s
            p0 = self.get_point_at_arc_length(s0)
            p1 = self.get_point_at_arc_length(s1)
            p2 = self.get_point_at_arc_length(s2)
            if s2 - s0 > 1e-10:
                return (p2 - 2*p1 + p0) / (ds**2)
            else:
                return np.array([0.0, 0.0])
        else:
            # Central difference
            s1 = s - ds
            s0 = s
            s2 = s + ds
            p1 = self.get_point_at_arc_length(s1)
            p0 = self.get_point_at_arc_length(s0)
            p2 = self.get_point_at_arc_length(s2)
            return (p2 - 2*p0 + p1) / (ds**2)


class StraightComponentPath(ComponentPath):
    """
    Straight line component path between two points.
    K(s) = 0 for all s.
    """
    
    def __init__(self, start_point, end_point):
        """
        Initialize straight component path.
        
        Args:
            start_point: (x, y) starting point
            end_point: (x, y) ending point
        """
        super().__init__(start_point, end_point)
        self.direction = self.end_point - self.start_point
        self.path_length = np.linalg.norm(self.direction)
        if self.path_length > 1e-10:
            self.unit_direction = self.direction / self.path_length
        else:
            self.unit_direction = np.array([1.0, 0.0])
            self.path_length = 0.0
    
    def get_point_at_arc_length(self, s):
        """Get point at arc length s ∈ [0, path_length]"""
        s = np.clip(s, 0.0, self.path_length)
        if self.path_length > 1e-10:
            t = s / self.path_length
            return self.start_point + t * self.direction
        return self.start_point
    
    def get_tangent_at_arc_length(self, s):
        """Get unit tangent vector at arc length s"""
        return self.unit_direction
    
    def get_curvature_at_arc_length(self, s):
        """K(s) = 0 for straight line"""
        return 0.0
    
    def get_first_derivative_at_arc_length(self, s):
        """First derivative with respect to s: zero for constant curvature paths"""
        return np.array([0.0, 0.0])
    
    def get_second_derivative_at_arc_length(self, s):
        """Second derivative with respect to s: zero for constant curvature paths"""
        return np.array([0.0, 0.0])


class ArcComponentPath(ComponentPath):
    """
    Circular arc component path defined by center, radius, start/end angles, and direction.
    K(s) = 1/radius (constant curvature).
    """
    
    def __init__(self, center, radius, start_angle, end_angle, clockwise=True):
        """
        Initialize arc component path.
        
        Args:
            center: (x, y) center of the circle
            radius: Radius of the arc (positive)
            start_angle: Starting angle in radians
            end_angle: Ending angle in radians
            clockwise: True for clockwise, False for counterclockwise
        """
        self.center = np.array(center)
        self.radius = float(radius)
        self.start_angle = float(start_angle)
        self.end_angle = float(end_angle)
        self.clockwise = bool(clockwise)
        
        # Calculate start and end points
        start_point = self.center + self.radius * np.array([np.cos(start_angle), np.sin(start_angle)])
        end_point = self.center + self.radius * np.array([np.cos(end_angle), np.sin(end_angle)])
        
        super().__init__(start_point, end_point)
        
        # Calculate arc length
        angle_diff = end_angle - start_angle
        if clockwise:
            # Ensure we go the shorter way clockwise
            if angle_diff > 0:
                angle_diff = angle_diff - 2 * np.pi
        else:
            # Counterclockwise
            if angle_diff < 0:
                angle_diff = angle_diff + 2 * np.pi
        
        self.angle_diff = angle_diff
        self.path_length = abs(self.radius * self.angle_diff)
        self.curvature = 1.0 / self.radius if self.radius > 1e-10 else 0.0
    
    def get_point_at_arc_length(self, s):
        """Get point at arc length s ∈ [0, path_length]"""
        s = np.clip(s, 0.0, self.path_length)
        if self.path_length > 1e-10:
            # Convert arc length to angle
            angle = self.start_angle + (s / self.path_length) * self.angle_diff
        else:
            angle = self.start_angle
        return self.center + self.radius * np.array([np.cos(angle), np.sin(angle)])
    
    def get_tangent_at_arc_length(self, s):
        """Get unit tangent vector at arc length s"""
        s = np.clip(s, 0.0, self.path_length)
        if self.path_length > 1e-10:
            angle = self.start_angle + (s / self.path_length) * self.angle_diff
        else:
            angle = self.start_angle
        # Tangent is perpendicular to radius vector
        if self.clockwise:
            tangent = np.array([np.sin(angle), -np.cos(angle)])
        else:
            tangent = np.array([-np.sin(angle), np.cos(angle)])
        # Normalize
        norm = np.linalg.norm(tangent)
        if norm > 1e-10:
            return tangent / norm
        return tangent
    
    def get_curvature_at_arc_length(self, s):
        """K(s) = 1/radius (constant curvature)"""
        return self.curvature
    
    def get_first_derivative_at_arc_length(self, s):
        """First derivative with respect to s: zero for constant curvature paths"""
        return np.array([0.0, 0.0])
    
    def get_second_derivative_at_arc_length(self, s):
        """Second derivative with respect to s: zero for constant curvature paths"""
        return np.array([0.0, 0.0])



class HybridPath:
    """
    Hybrid path consisting of multiple component paths that connect at their endpoints.
    Focus is on curvature, especially at intersections between components.
    """
    
    def __init__(self, components):
        """
        Initialize hybrid path from component paths.
        
        Args:
            components: List of ComponentPath objects that must connect end-to-end
        """
        if len(components) == 0:
            raise ValueError("HybridPath must have at least one component")
        
        # Verify connectivity
        for i in range(len(components) - 1):
            end_point = components[i].end_point
            next_start = components[i+1].start_point
            dist = np.linalg.norm(end_point - next_start)
            if dist > 1e-6:
                raise ValueError(
                    f"Component {i} end point {end_point} does not connect to "
                    f"component {i+1} start point {next_start} (distance: {dist:.6f})"
                )
        
        self.components = components
        self.num_components = len(components)
        
        # Calculate cumulative arc lengths for parameter mapping
        self.component_lengths = [comp.get_path_length() for comp in self.components]
        self.cumulative_lengths = np.zeros(self.num_components + 1)
        for i in range(self.num_components):
            self.cumulative_lengths[i+1] = self.cumulative_lengths[i] + self.component_lengths[i]
        
        self.total_length = self.cumulative_lengths[-1]
        
        # Pre-compute transition point curvatures
        self.transition_curvatures = self._compute_transition_curvatures()


    def _global_s_to_component(self, s):
        """
        Convert global arc length s ∈ [0, total_length] to (component_index, local_s).
        
        Args:
            s: Global arc length [0, total_length]
            
        Returns:
            (component_index, local_s) where local_s ∈ [0, component_path_length]
        """
        s = np.clip(s, 0.0, self.total_length)
        
        # Find which component this arc length belongs to
        for i in range(self.num_components):
            if s <= self.cumulative_lengths[i+1]:
                # This arc length belongs to component i
                local_s = s - self.cumulative_lengths[i]
                local_s = np.clip(local_s, 0.0, self.component_lengths[i])
                return i, local_s
        
        # Should not reach here, but handle edge case
        return self.num_components - 1, self.component_lengths[-1]
    
    def _compute_transition_curvatures(self, delta_s=1e-2):
        """
        Compute curvature at transition points using three-point method.
        
        Args:
            delta_s: Small arc length step for computing transition curvature
            
        Returns:
            List of transition curvature values, one for each transition (length = num_components - 1)
        """
        transition_curvatures = []
        
        for i in range(self.num_components - 1):
            # Get three points: before transition, at transition, after transition
            # Point before (end of component i)
            s_before = self.cumulative_lengths[i+1] - delta_s
            if s_before < self.cumulative_lengths[i]:
                s_before = self.cumulative_lengths[i] + 1e-6  # Very close to transition
            
            # Point at transition (end of component i = start of component i+1)
            s_transition = self.cumulative_lengths[i+1]
            
            # Point after (start of component i+1)
            s_after = self.cumulative_lengths[i+1] + delta_s
            if s_after > self.cumulative_lengths[i+2]:
                s_after = self.cumulative_lengths[i+2] - 1e-6  # Very close to next transition
            
            # Get points
            p_before = self.get_point_at_arc_length(s_before)
            p_transition = self.get_point_at_arc_length(s_transition)
            p_after = self.get_point_at_arc_length(s_after)
            
            # Get tangent vectors
            t_before = self.get_tangent_at_arc_length(s_before)
            t_after = self.get_tangent_at_arc_length(s_after)
            
            # Calculate delta_theta using atan2 formula
            # delta_theta = atan2(t1x*t2y - t1y*t2x, t1x*t2x + t1y*t2y)
            t1x, t1y = t_before[0], t_before[1]
            t2x, t2y = t_after[0], t_after[1]
            
            delta_theta = np.arctan2(t1x*t2y - t1y*t2x, t1x*t2x + t1y*t2y)
            
            # Calculate delta_s (arc length between before and after points)
            ds_total = s_after - s_before
            
            # Curvature = delta_theta / delta_s
            if ds_total > 1e-10:
                curvature = delta_theta / ds_total
            else:
                curvature = 0.0
            
            transition_curvatures.append(curvature)
        
        return transition_curvatures
    
    def get_point_at_arc_length(self, s):
        """
        Get point at global arc length s ∈ [0, total_length].
        
        Args:
            s: Global arc length [0, total_length]
            
        Returns:
            (x, y) point
        """
        comp_idx, local_s = self._global_s_to_component(s)
        return self.components[comp_idx].get_point_at_arc_length(local_s)
    
    def get_tangent_at_arc_length(self, s):
        """
        Get unit tangent vector at global arc length s.
        
        Args:
            s: Global arc length [0, total_length]
            
        Returns:
            (dx, dy) unit tangent vector
        """
        comp_idx, local_s = self._global_s_to_component(s)
        return self.components[comp_idx].get_tangent_at_arc_length(local_s)
    
    def get_curvature_at_arc_length(self, s):
        """
        Get curvature K(s) at global arc length s.
        Accounts for transition point curvatures.
        
        Args:
            s: Global arc length [0, total_length]
            
        Returns:
            curvature value (1/m)
        """
        # Check if s is at a transition point
        for i in range(self.num_components - 1):
            transition_s = self.cumulative_lengths[i+1]
            if abs(s - transition_s) < 1e-8:
                # At transition point, return transition curvature
                return self.transition_curvatures[i]
        
        # Not at transition, get from component
        comp_idx, local_s = self._global_s_to_component(s)
        return self.components[comp_idx].get_curvature_at_arc_length(local_s)
    
    def get_first_derivative_at_arc_length(self, s):
        """
        Get first derivative with respect to arc length s.
        Accounts for transition point discontinuities.
        
        Args:
            s: Global arc length [0, total_length]
            
        Returns:
            (dx/ds, dy/ds) first derivative vector
        """
        # Check if s is at a transition point
        for i in range(self.num_components - 1):
            transition_s = self.cumulative_lengths[i+1]
            if abs(s - transition_s) < 1e-8:
                # At transition point, use numerical differentiation
                ds = 1e-6
                s1 = max(s - ds, 0.0)
                s2 = min(s + ds, self.total_length)
                p1 = self.get_point_at_arc_length(s1)
                p2 = self.get_point_at_arc_length(s2)
                if s2 - s1 > 1e-10:
                    return (p2 - p1) / (s2 - s1)
                else:
                    return np.array([0.0, 0.0])
        
        # Not at transition, get from component
        comp_idx, local_s = self._global_s_to_component(s)
        return self.components[comp_idx].get_first_derivative_at_arc_length(local_s)
    
    def get_second_derivative_at_arc_length(self, s):
        """
        Get second derivative with respect to arc length s.
        Accounts for transition point discontinuities.
        
        Args:
            s: Global arc length [0, total_length]
            
        Returns:
            (d²x/ds², d²y/ds²) second derivative vector
        """
        # Check if s is at a transition point
        for i in range(self.num_components - 1):
            transition_s = self.cumulative_lengths[i+1]
            if abs(s - transition_s) < 1e-8:
                # At transition point, use numerical differentiation
                ds = 1e-6
                s1 = max(s - ds, 0.0)
                s0 = s
                s2 = min(s + ds, self.total_length)
                p1 = self.get_point_at_arc_length(s1)
                p0 = self.get_point_at_arc_length(s0)
                p2 = self.get_point_at_arc_length(s2)
                if s2 - s1 > 1e-10:
                    return (p2 - 2*p0 + p1) / (ds**2)
                else:
                    return np.array([0.0, 0.0])
        
        # Not at transition, get from component
        comp_idx, local_s = self._global_s_to_component(s)
        return self.components[comp_idx].get_second_derivative_at_arc_length(local_s)
    
    # Legacy methods for backward compatibility (convert to arc length)
    def get_point_at_parameter(self, t):
        """Legacy: Get point at normalized parameter t ∈ [0, 1]"""
        s = t * self.total_length
        return self.get_point_at_arc_length(s)
    
    def get_tangent_at_parameter(self, t):
        """Legacy: Get tangent at normalized parameter t ∈ [0, 1]"""
        s = t * self.total_length
        return self.get_tangent_at_arc_length(s)
    
    def get_curvature_at_parameter(self, t):
        """Legacy: Get curvature at normalized parameter t ∈ [0, 1]"""
        s = t * self.total_length
        return self.get_curvature_at_arc_length(s)
    
    def get_first_derivative_at_parameter(self, t):
        """Legacy: Get first derivative at normalized parameter t ∈ [0, 1]"""
        s = t * self.total_length
        return self.get_first_derivative_at_arc_length(s)
    
    def get_second_derivative_at_parameter(self, t):
        """Legacy: Get second derivative at normalized parameter t ∈ [0, 1]"""
        s = t * self.total_length
        return self.get_second_derivative_at_arc_length(s)
    
    def get_intersection_curvatures(self):
        """
        Get curvature values at all intersections between components.
        
        Returns:
            List of tuples (intersection_index, curvature_before, curvature_after)
            where intersection_index is the index of the component that ends at the intersection
        """
        curvatures = []
        for i in range(self.num_components - 1):
            # Curvature at end of component i
            curv_before = self.components[i].get_curvature_at_parameter(1.0)
            # Curvature at start of component i+1
            curv_after = self.components[i+1].get_curvature_at_parameter(0.0)
            curvatures.append((i, curv_before, curv_after))
        return curvatures
    
    def check_curvature_continuity(self, tolerance=1e-3):
        """
        Check if curvature is continuous at intersections, including transition curvature.
        
        We consider three values at each intersection i:
          - κ_before: curvature at the end of component i
          - κ_after:  curvature at the start of component i+1
          - κ_trans:  transition curvature from the three-point method (if available)
        
        The effective curvature jump is taken as the maximum of:
          |κ_before - κ_trans|, |κ_trans - κ_after|, |κ_after - κ_before|
        (if κ_trans is not available, we fall back to |κ_after - κ_before|).
        
        Args:
            tolerance: Maximum allowed curvature jump
            
        Returns:
            List of tuples (intersection_index, curvature_jump, is_continuous)
        """
        results = []
        intersections = self.get_intersection_curvatures()
        
        for idx, (inter_idx, curv_before, curv_after) in enumerate(intersections):
            # Base jump between before and after
            jumps = [abs(curv_after - curv_before)]
            
            # Include transition curvature if available
            if hasattr(self, "transition_curvatures") and idx < len(self.transition_curvatures):
                k_trans = self.transition_curvatures[idx]
                jumps.append(abs(k_trans - curv_before))
                jumps.append(abs(curv_after - k_trans))
            
            curvature_jump = max(jumps)
            is_continuous = curvature_jump <= tolerance
            results.append((inter_idx, curvature_jump, is_continuous))
        
        return results
    
    def get_normal_at_parameter(self, s):
        """
        Get unit normal vector at global parameter s (right-hand perpendicular to tangent).
        
        Args:
            s: Global parameter [0, 1]
            
        Returns:
            (dx, dy) unit normal vector
        """
        tangent = self.get_tangent_at_parameter(s)
        # Rotate tangent 90° clockwise to get normal
        return np.array([tangent[1], -tangent[0]])
    
    def get_contour_error(self, query_point):
        """
        Calculate the contour error (lateral and longitudinal) for controllers.
        
        Args:
            query_point: (x, y) or (x, y, theta) point
            
        Returns:
            contour_error: dict with keys:
                - 'lateral': Signed lateral error (negative = left, positive = right)
                - 'longitudinal': Signed longitudinal error (negative = behind)
                - 'path_param': Path parameter s at closest point
                - 'closest_point': (x, y) point on path
        """
        x, y = query_point[0], query_point[1]
        
        # Find closest point on path using numerical search
        s_samples = np.linspace(0, 1, 100)
        min_dist = float('inf')
        best_s = 0
        
        for s in s_samples:
            point = self.get_point_at_parameter(s)
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_s = s
        
        # Refine search using local optimization
        from scipy.optimize import minimize
        def distance_to_path(s):
            point = self.get_point_at_parameter(float(s))
            return (x - point[0])**2 + (y - point[1])**2
        
        bounds = [(0, 1)]
        result = minimize(distance_to_path, best_s, bounds=bounds, method='L-BFGS-B')
        optimal_s = result.x[0]
        closest_point = self.get_point_at_parameter(optimal_s)
        
        # Get tangent and normal at closest point
        tangent = self.get_tangent_at_parameter(optimal_s)
        normal = self.get_normal_at_parameter(optimal_s)
        
        # Calculate vector from closest point to query point
        error_vector = np.array([x - closest_point[0], y - closest_point[1]])
        
        # Project error onto tangent and normal
        longitudinal_error = np.dot(error_vector, tangent)
        lateral_error = np.dot(error_vector, normal)
        
        return {
            'lateral': lateral_error,
            'longitudinal': longitudinal_error,
            'path_param': optimal_s,
            'closest_point': closest_point
        }
    
    def t_to_s(self, s):
        """
        Convert path parameter s to arc length.
        For HybridPath, parameter s is already normalized, so this is just s * total_length.
        
        Args:
            s: Path parameter [0, 1]
            
        Returns:
            Arc length in meters
        """
        return s * self.total_length
    
    @staticmethod
    def create_from_trajectory(trajectory, orientation_mode="follow_path"):
        """
        Create a HybridPath from trajectory data using a single SplineComponentPath.
        
        Args:
            trajectory: List of points in one of these formats:
                - [(x1, y1), (x2, y2), ...] (positions only)
                - [(x1, y1, theta1), (x2, y2, theta2), ...] (positions with orientations)
            orientation_mode: How to handle orientation
                - "explicit": Use provided orientations (required if not provided in trajectory)
                - "follow_path": Calculate orientations based on path tangent
        
        Returns:
            HybridPath: A hybrid path created from the trajectory (single SplineComponentPath)
        """
        # Convert trajectory to numpy array if it's not already
        trajectory = np.array(trajectory)
        
        # Check trajectory format and dimensions
        if len(trajectory.shape) != 2:
            raise ValueError("Trajectory must be a 2D array/list of points")
        
        # Handle different input formats
        if trajectory.shape[1] == 2:  # Only (x, y) provided
            # Add placeholder orientations (they'll be recalculated in follow_path mode)
            positions = np.zeros((len(trajectory), 3))
            positions[:, :2] = trajectory
            
            # If we're in explicit mode without orientations, we need to calculate them
            if orientation_mode == "explicit":
                orientation_mode = "follow_path"
                
        elif trajectory.shape[1] >= 3:  # (x, y, theta) or more provided
            # Use the first three columns as (x, y, theta)
            positions = trajectory[:, :3].copy()
        else:
            raise ValueError("Trajectory points must have at least 2 dimensions (x, y)")
        
        # Create a single SplineComponentPath from the trajectory
        spline_component = SplineComponentPath(positions.tolist())
        
        # Create HybridPath with single component
        return HybridPath([spline_component])


# ============================================================================
# ORIENTATION PATH SYSTEM
# ============================================================================
# Separate orientation path that is independent of position path.
# Orientation is a 1D value over path length with multiple parameterization options.
# ============================================================================

class OrientationPath:
    """
    Separate orientation path that is independent of position path.
    Orientation is parameterized as a 1D value over path length.
    
    Parameterization methods:
    1. "curvature": Orientation equals path curvature (integrated)
    2. "fourier": Frequency-based using Fourier transform
    3. "linear_interp": Desired values at intersections + linear interpolation (default)
    """
    
    def __init__(self, path_length, parameterization="linear_interp", **kwargs):
        """
        Initialize orientation path.
        
        Args:
            path_length: Total length of the path (m)
            parameterization: Method for parameterization
                - "curvature": Orientation follows curvature
                - "fourier": Frequency-based (requires frequencies/amplitudes)
                - "linear_interp": Linear interpolation between waypoints (default)
            **kwargs: Additional parameters based on parameterization method
                - For "linear_interp": intersection_orientations (list of desired orientations at intersections)
                - For "fourier": frequencies, amplitudes (for Fourier synthesis)
        """
        self.path_length = path_length
        self.parameterization = parameterization
        
        if parameterization == "linear_interp":
            # Default: linear interpolation between intersection orientations
            intersection_orientations = kwargs.get('intersection_orientations', [0.0])
            self._setup_linear_interpolation(intersection_orientations)
        elif parameterization == "curvature":
            # Will be set up when position path is provided
            self.position_path = None
        elif parameterization == "fourier":
            frequencies = kwargs.get('frequencies', [])
            amplitudes = kwargs.get('amplitudes', [])
            self._setup_fourier(frequencies, amplitudes)
        else:
            raise ValueError(f"Unknown parameterization method: {parameterization}")
    
    def _setup_linear_interpolation(self, intersection_orientations):
        """Setup linear interpolation between intersection orientations."""
        self.intersection_orientations = np.array(intersection_orientations)
        # For now, assume orientations are specified at each intersection
        # If only one value, use constant orientation
        if len(self.intersection_orientations) == 1:
            self.constant_orientation = self.intersection_orientations[0]
            self.use_constant = True
        else:
            self.use_constant = False
            # Create interpolation function
            # Assume orientations are evenly spaced along path
            num_intersections = len(self.intersection_orientations)
            if num_intersections > 1:
                s_values = np.linspace(0, 1, num_intersections)
                # Unwrap angles for interpolation
                orientations_unwrapped = np.unwrap(self.intersection_orientations)
                from scipy.interpolate import interp1d
                self.orientation_interp = interp1d(
                    s_values, orientations_unwrapped, 
                    kind='linear', bounds_error=False, fill_value="extrapolate"
                )
            else:
                self.use_constant = True
                self.constant_orientation = self.intersection_orientations[0]
    
    def _setup_fourier(self, frequencies, amplitudes):
        """Setup Fourier-based parameterization."""
        self.frequencies = np.array(frequencies)
        self.amplitudes = np.array(amplitudes)
        if len(self.frequencies) != len(self.amplitudes):
            raise ValueError("Frequencies and amplitudes must have same length")
    
    def set_position_path_for_curvature(self, position_path):
        """
        Set position path for curvature-based orientation.
        
        Args:
            position_path: HybridPath or SplineReferencePath instance
        """
        if self.parameterization != "curvature":
            raise ValueError("set_position_path_for_curvature only valid for 'curvature' parameterization")
        self.position_path = position_path
    
    def get_orientation_at_parameter(self, s):
        """
        Get orientation at path parameter s ∈ [0, 1].
        
        Args:
            s: Path parameter [0, 1]
            
        Returns:
            Orientation in radians
        """
        if self.parameterization == "linear_interp":
            if self.use_constant:
                return self.constant_orientation
            else:
                return self.orientation_interp(s) % (2 * np.pi)
        
        elif self.parameterization == "curvature":
            if self.position_path is None:
                raise ValueError("Position path must be set for curvature-based orientation")
            # Integrate curvature to get orientation
            # For now, use numerical integration
            s_samples = np.linspace(0, s, 100)
            total_orientation = 0.0
            for i in range(1, len(s_samples)):
                ds = s_samples[i] - s_samples[i-1]
                curv = self.position_path.get_curvature_at_parameter(s_samples[i-1])
                total_orientation += curv * ds * self.path_length
            return total_orientation % (2 * np.pi)
        
        elif self.parameterization == "fourier":
            # Fourier synthesis
            arc_length = s * self.path_length
            orientation = 0.0
            for freq, amp in zip(self.frequencies, self.amplitudes):
                orientation += amp * np.sin(2 * np.pi * freq * arc_length)
            return orientation % (2 * np.pi)
        
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")


class SplineReferencePath:
    def __init__(self, waypoints, orientation_mode="explicit"):
        """
        Initialize reference path from waypoints using cubic spline interpolation.
        
        Args:
            waypoints: List of (x, y, theta) points or numpy array with shape (n, 3)
            orientation_mode: How to handle orientation
                - "explicit": Use the provided orientations in waypoints
                - "follow_path": Calculate orientation based on path tangent
        """
        self.waypoints = np.array(waypoints)
        self.orientation_mode = orientation_mode
        self.length = len(waypoints)
        
        # Validate waypoints input
        if len(waypoints) == 0:
            raise ValueError("Waypoints cannot be empty")
        
        waypoints_array = np.array(waypoints)
        if waypoints_array.shape[-1] != 3:
            raise ValueError("Each waypoint must have exactly 3 elements (x, y, theta)")
        
        self.waypoints = waypoints_array
        # Extract x and y coordinates
        self.points_x = self.waypoints[:, 0]
        self.points_y = self.waypoints[:, 1]
        self.points_theta = self.waypoints[:, 2]
        
        # Create parameter t along the path (normalized arc length approximation)
        self._create_path_parameter()
        
        # Create spline representation
        self._create_splines()
        
        # If using follow_path mode, recalculate orientations based on path direction
        if orientation_mode == "follow_path":
            self._calculate_path_orientations()
    
        # By this point, we have the spline functions for x(t), y(t), and theta(t)
        # along with their derivatives (except theta)

        # Calculate path metrics (lengths, etc.)
        self._compute_path_metrics()

    def _create_path_parameter(self):
        """Create a parameter t that approximates normalized arc length"""
        # Start with chord-length parameterization
        dt = np.zeros(self.length)
        for i in range(1, self.length):
            dx = self.points_x[i] - self.points_x[i-1]
            dy = self.points_y[i] - self.points_y[i-1]
            dt[i] = np.sqrt(dx*dx + dy*dy)
            
        # Cumulative parameter (normalized later)
        self.t = np.zeros(self.length)
        for i in range(1, self.length):
            self.t[i] = self.t[i-1] + dt[i]
        
        # # **FIX: Ensure strictly increasing by adding small epsilon where needed**
        # epsilon = 1e-6
        # for i in range(1, self.length):
        #     if self.t[i] <= self.t[i-1]:
        #         self.t[i] = self.t[i-1] + epsilon

        # # Normalize to [0, 1]
        # if self.t[-1] > 0:
        #     self.t = self.t / self.t[-1]

        # # **ADDITIONAL CHECK: Ensure normalized values are also strictly increasing**
        # for i in range(1, self.length):
        #     if self.t[i] <= self.t[i-1]:
        #         self.t[i] = self.t[i-1] + epsilon
    

        # **IMPROVED: Adaptive epsilon based on path scale**
        if self.t[-1] > 0:
            # Use epsilon proportional to total path length
            epsilon = self.t[-1] * 1e-6  # Adaptive epsilon
            
            # Ensure strictly increasing
            for i in range(1, self.length):
                if self.t[i] <= self.t[i-1]:
                    self.t[i] = self.t[i-1] + epsilon
            
            # Normalize to [0, 1]
            self.t = self.t / self.t[-1]
            
            # **ADDITIONAL: Check if normalization broke monotonicity**
            for i in range(1, self.length):
                if self.t[i] <= self.t[i-1]:
                    # Use normalized epsilon
                    self.t[i] = self.t[i-1] + 1e-8
        else:
            # Handle degenerate case
            self.t = np.linspace(0, 1, self.length)

    def _create_splines(self):
        """Create cubic splines for x(t) and y(t)"""
        self.spline_x = CubicSpline(self.t, self.points_x)
        self.spline_y = CubicSpline(self.t, self.points_y)
        
        # Create derivatives for tangent calculation
        self.spline_dx = self.spline_x.derivative()
        self.spline_dy = self.spline_y.derivative()
        
        # Create second derivatives for curvature
        self.spline_d2x = self.spline_dx.derivative()
        self.spline_d2y = self.spline_dy.derivative()
        
        # Create orientation spline (only used in explicit mode)
        if self.orientation_mode == "explicit":
            # Ensure angles are continuous (unwrapped) for interpolation
            theta_unwrapped = np.unwrap(self.points_theta)
            self.spline_theta = interp1d(self.t, theta_unwrapped, kind='linear', 
                                        bounds_error=False, fill_value="extrapolate")
    
    def _calculate_path_orientations(self):
        """Calculate orientation at each waypoint based on path tangent"""
        # Sample points along the path
        t_samples = np.linspace(0, 1, 100)
        
        # Get tangent at each point and calculate orientation
        orientations = np.zeros_like(t_samples)
        for i, t in enumerate(t_samples):
            dx = self.spline_dx(t)
            dy = self.spline_dy(t)
            orientations[i] = np.arctan2(dy, dx)
        
        # Create interpolation function for orientation
        self.spline_theta = interp1d(t_samples, orientations, kind='linear',
                                    bounds_error=False, fill_value="extrapolate")
    
    def _compute_path_metrics(self):
        """Compute path length and other metrics using numerical integration"""
        # Sample points for length calculation
        n_samples = 1000
        t_samples = np.linspace(0, 1, n_samples)
        
        # Calculate approximate path length using numerical integration
        self.segment_lengths = np.zeros(n_samples-1)
        self.cumulative_distances = np.zeros(n_samples)
        
        for i in range(1, n_samples):
            t1, t2 = t_samples[i-1], t_samples[i]
            
            # Get points at t1 and t2
            x1, y1 = self.spline_x(t1), self.spline_y(t1)
            x2, y2 = self.spline_x(t2), self.spline_y(t2)
            
            # Calculate segment length
            dx, dy = x2 - x1, y2 - y1
            self.segment_lengths[i-1] = np.sqrt(dx*dx + dy*dy)
            
            # Update cumulative distance
            self.cumulative_distances[i] = self.cumulative_distances[i-1] + self.segment_lengths[i-1]
        
        # Store total path length
        self.total_path_length = self.cumulative_distances[-1]
        
        # Create mapping between parameter t and arc length
        self.t_to_s = interp1d(t_samples, self.cumulative_distances, kind='linear',
                             bounds_error=False, fill_value="extrapolate")
        
        # Create mapping between arc length and parameter t (for inverse lookup)
        self.s_to_t = interp1d(self.cumulative_distances, t_samples, kind='linear',
                             bounds_error=False, fill_value="extrapolate")
    
    def get_point_at_parameter(self, t):
        """Get point at parameter value t ∈ [0,1]"""
        x = self.spline_x(t)
        y = self.spline_y(t)
        
        # Get orientation based on mode
        if self.orientation_mode == "explicit":
            theta = self.spline_theta(t) % (2*np.pi)  # Normalize to [0, 2π]
        else:
            # Calculate orientation from path tangent
            dx = self.spline_dx(t)
            dy = self.spline_dy(t)
            theta = np.arctan2(dy, dx)
            
        return np.array([x, y, theta])
    
    def get_tangent_at_parameter(self, t):
        """Get unit tangent vector at parameter t"""
        dx = self.spline_dx(t)
        dy = self.spline_dy(t)
        
        # Normalize to unit vector
        norm = np.sqrt(dx*dx + dy*dy)
        if norm < 1e-10:
            # Avoid division by zero
            return np.array([1.0, 0.0])
            
        return np.array([dx/norm, dy/norm])
    
    def get_normal_at_parameter(self, t):
        """Get unit normal vector at parameter t (right-hand perpendicular to tangent)"""
        tangent = self.get_tangent_at_parameter(t)
        # Rotate tangent 90° clockwise to get normal
        return np.array([tangent[1], -tangent[0]])
    
    def get_curvature_at_parameter(self, t):
        """Calculate curvature at parameter t"""
        dx = self.spline_dx(t)
        dy = self.spline_dy(t)
        d2x = self.spline_d2x(t)
        d2y = self.spline_d2y(t)
        
        # Curvature formula: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = dx * d2y - dy * d2x
        denominator = (dx*dx + dy*dy)**(1.5)
        
        if denominator < 1e-10:
            return 0.0
            
        return numerator / denominator
    
    def find_closest_point(self, query_point):
        """
        Find the closest point on the path to the query_point.
        
        Args:
            query_point: (x, y) or (x, y, theta) point
            
        Returns:
            closest_point: (x, y, theta) point on path
            path_param: Parameter t of the closest point
            progress: Normalized progress along path [0-1]
        """
        x, y = query_point[0], query_point[1]
        
        # Sample points along the path for initial guess
        t_samples = np.linspace(0, 1, 100)
        min_dist = float('inf')
        best_t = 0
        
        # Find closest sample point for initial guess
        for t in t_samples:
            point = self.get_point_at_parameter(t)
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                best_t = t
        
        # Refine search using local optimization
        def distance_to_path(t):
            point = self.get_point_at_parameter(float(t))
            return (x - point[0])**2 + (y - point[1])**2
        
        # Constrain t to [0, 1]
        bounds = [(0, 1)]
        result = minimize(distance_to_path, best_t, bounds=bounds, method='L-BFGS-B')
        
        # Get optimized parameter and corresponding point
        optimal_t = result.x[0]
        closest_point = self.get_point_at_parameter(optimal_t)
        
        # Calculate progress along path (as normalized arc length)
        arc_length = self.t_to_s(optimal_t)
        progress = arc_length / self.total_path_length
        
        return closest_point, optimal_t, progress
    
    def get_lookahead_point(self, current_point, lookahead_distance):
        """
        Get a point on the path that is lookahead_distance ahead of the closest point.
        
        Args:
            current_point: Current (x, y) or (x, y, theta) position
            lookahead_distance: Distance to look ahead on path
            
        Returns:
            target_point: (x, y, theta) point on path
            target_t: Parameter t of the target point
        """
        # Find closest point on path
        closest_point, closest_t, progress = self.find_closest_point(current_point)
        
        # Calculate arc length at closest point
        arc_length_closest = self.t_to_s(closest_t)
        
        # Add lookahead distance
        target_arc_length = min(arc_length_closest + lookahead_distance, self.total_path_length)
        
        # Convert back to parameter t
        target_t = self.s_to_t(target_arc_length)
        
        # Get target point
        target_point = self.get_point_at_parameter(target_t)
        
        return target_point, target_t
    
    def get_contour_error(self, query_point):
        """
        Calculate the contour error (lateral and longitudinal) for controllers.
        
        Args:
            query_point: (x, y) or (x, y, theta) point
            
        Returns:
            contour_error: dict with keys:
                - 'lateral': Signed lateral error (negative = left, positive = right)
                - 'longitudinal': Signed longitudinal error (negative = behind)
                - 'path_param': Path parameter t at closest point
                - 'closest_point': (x, y, theta) point on path
        """
        x, y = query_point[0], query_point[1]
        
        # Find closest point on path
        closest_point, path_param, _ = self.find_closest_point(query_point)
        
        # Get tangent and normal at closest point
        tangent = self.get_tangent_at_parameter(path_param)
        normal = self.get_normal_at_parameter(path_param)
        
        # Calculate vector from closest point to query point
        error_vector = np.array([x - closest_point[0], y - closest_point[1]])
        
        # Project error onto tangent and normal
        longitudinal_error = np.dot(error_vector, tangent)
        lateral_error = np.dot(error_vector, normal)
        
        return {
            'lateral': lateral_error,
            'longitudinal': longitudinal_error,
            'path_param': path_param,
            'closest_point': closest_point
        }
    
    def get_curvature(self, s):
        """Calculate the curvature at path parameter s"""
        # For numerical stability, use central difference
        ds = 1e-6
        
        # Get tangents at neighboring points
        t_minus = self.get_tangent_at_parameter(max(0, s - ds))
        t_plus = self.get_tangent_at_parameter(min(1.0, s + ds))
        
        # Calculate tangent derivative (approximation)
        tangent_derivative = (t_plus - t_minus) / (2 * ds)
        
        # Curvature formula for 2D: κ = |T'|/|T|
        # Since |T| = 1 (unit tangent), κ = |T'|
        tangent_deriv_norm = np.linalg.norm(tangent_derivative)
        
        return tangent_deriv_norm
    
    def get_tangent_derivative(self, s):
        """Calculate the derivative of the tangent vector at path parameter s"""
        # Use central difference for numerical stability
        ds = 1e-6
        
        # Get tangents at neighboring points
        t_minus = self.get_tangent_at_parameter(max(0, s - ds))
        t_plus = self.get_tangent_at_parameter(min(1.0, s + ds))
        
        # Calculate tangent derivative
        tangent_derivative = (t_plus - t_minus) / (2 * ds)
        
        return tangent_derivative



# %%
class HybridPathVisualizer:
    """Visualization tools for hybrid paths with focus on curvature and derivatives."""
    
    @staticmethod
    def plot_hybrid_path(hybrid_path, num_samples=200, show_components=True):
        """
        Plot the hybrid path with component boundaries marked.
        
        Args:
            hybrid_path: HybridPath instance
            num_samples: Number of points to sample
            show_components: If True, mark component boundaries
        """
        s_samples = np.linspace(0, 1, num_samples)
        points = np.array([hybrid_path.get_point_at_parameter(s) for s in s_samples])
        
        plt.figure(figsize=(10, 8))
        plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Hybrid Path')
        
        # Mark component boundaries
        if show_components:
            for i in range(hybrid_path.num_components):
                comp_start = hybrid_path.components[i].start_point
                plt.plot(comp_start[0], comp_start[1], 'ro', markersize=8, 
                        label='Component Start' if i == 0 else '')
            
            # Mark last end point
            comp_end = hybrid_path.components[-1].end_point
            plt.plot(comp_end[0], comp_end[1], 'go', markersize=8, label='Path End')
        
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Hybrid Path Visualization')
        plt.legend()
        plt.show()
    
    
    @staticmethod
    def plot_curvature_and_derivatives(hybrid_path, num_samples=500, save_path=None):
        """
        Plot curvature, first derivative, and second derivative along the path.
        Focus especially on intersections between components.
        
        Args:
            hybrid_path: HybridPath instance
            num_samples: Number of points to sample
        """
        # Sample in arc length directly
        arc_lengths = np.linspace(0.0, hybrid_path.total_length, num_samples)

        # Calculate curvature using arc-length-based API
        curvatures = np.array([hybrid_path.get_curvature_at_arc_length(s) for s in arc_lengths])
        
        # Calculate first derivatives
        first_derivs = np.array([hybrid_path.get_first_derivative_at_arc_length(s) for s in arc_lengths])
        first_deriv_magnitudes = np.linalg.norm(first_derivs, axis=1)
        
        # Calculate second derivatives
        second_derivs = np.array([hybrid_path.get_second_derivative_at_arc_length(s) for s in arc_lengths])
        second_deriv_magnitudes = np.linalg.norm(second_derivs, axis=1)
        
        # Mark intersection points
        intersection_arc_lengths = []
        for i in range(hybrid_path.num_components - 1):
            intersection_arc_lengths.append(hybrid_path.cumulative_lengths[i+1])
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # --- Subplot 1: Curvature ---
        ax1.plot(arc_lengths, curvatures, 'b-', linewidth=2, label='Curvature')
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.fill_between(arc_lengths, 0, curvatures, alpha=0.3, color='blue')
        
        # Mark intersections and plot transition-point curvature explicitly
        for idx, inter_s in enumerate(intersection_arc_lengths):
            ax1.axvline(inter_s, color='r', linestyle='--', alpha=0.5, linewidth=1)
            
            # Curvature immediately before and after intersection (from neighbour components)
            s_before = max(inter_s - 1e-6, 0.0)
            s_after = min(inter_s + 1e-6, hybrid_path.total_length)
            curv_before = hybrid_path.get_curvature_at_arc_length(s_before)
            curv_after = hybrid_path.get_curvature_at_arc_length(s_after)
            
            ax1.plot(inter_s, curv_before, 'ro', markersize=6, zorder=5, label='Before transition' if idx == 0 else '')
            ax1.plot(inter_s, curv_after, 'go', markersize=6, zorder=5, label='After transition' if idx == 0 else '')
            
            # Transition-point curvature from three-point method
            if hasattr(hybrid_path, 'transition_curvatures') and idx < len(hybrid_path.transition_curvatures):
                k_tr = hybrid_path.transition_curvatures[idx]
                ax1.plot(inter_s, k_tr, 'kx', markersize=8, zorder=6,
                         label='Transition curvature' if idx == 0 else '')
        
        ax1.set_title('Path Curvature (κ)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Arc Length (m)')
        ax1.set_ylabel('Curvature (1/m)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add statistics
        mean_curv = np.mean(np.abs(curvatures))
        max_curv = np.max(np.abs(curvatures))
        ax1.text(0.02, 0.98, 
                f'Mean |κ|: {mean_curv:.4f}\nMax |κ|: {max_curv:.4f}',
                transform=ax1.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # --- Subplot 2: First Derivative Magnitude ---
        ax2.plot(arc_lengths, first_deriv_magnitudes, 'g-', linewidth=2, label='|dP/ds|')
        
        # Mark intersections and show derivative values around transitions
        for idx, inter_s in enumerate(intersection_arc_lengths):
            ax2.axvline(inter_s, color='r', linestyle='--', alpha=0.5, linewidth=1)
            
            s_before = max(inter_s - 1e-6, 0.0)
            s_after = min(inter_s + 1e-6, hybrid_path.total_length)
            d_before = np.linalg.norm(hybrid_path.get_first_derivative_at_arc_length(s_before))
            d_after = np.linalg.norm(hybrid_path.get_first_derivative_at_arc_length(s_after))
            d_trans = np.linalg.norm(hybrid_path.get_first_derivative_at_arc_length(inter_s))
            
            ax2.plot(inter_s, d_before, 'ro', markersize=6, zorder=5, label='|dP/ds| before' if idx == 0 else '')
            ax2.plot(inter_s, d_after, 'go', markersize=6, zorder=5, label='|dP/ds| after' if idx == 0 else '')
            ax2.plot(inter_s, d_trans, 'kx', markersize=7, zorder=6, label='|dP/ds| at transition' if idx == 0 else '')
        
        ax2.set_title('First Derivative Magnitude (|dP/ds|)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Arc Length (m)')
        ax2.set_ylabel('|dP/ds| (m)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # --- Subplot 3: Second Derivative Magnitude ---
        ax3.plot(arc_lengths, second_deriv_magnitudes, 'r-', linewidth=2, label='|d²P/ds²|')
        
        # Mark intersections and show second derivative values around transitions
        for idx, inter_s in enumerate(intersection_arc_lengths):
            ax3.axvline(inter_s, color='r', linestyle='--', alpha=0.5, linewidth=1)
            
            s_before = max(inter_s - 1e-6, 0.0)
            s_after = min(inter_s + 1e-6, hybrid_path.total_length)
            d2_before = np.linalg.norm(hybrid_path.get_second_derivative_at_arc_length(s_before))
            d2_after = np.linalg.norm(hybrid_path.get_second_derivative_at_arc_length(s_after))
            d2_trans = np.linalg.norm(hybrid_path.get_second_derivative_at_arc_length(inter_s))
            
            ax3.plot(inter_s, d2_before, 'ro', markersize=6, zorder=5, label='|d²P/ds²| before' if idx == 0 else '')
            ax3.plot(inter_s, d2_after, 'go', markersize=6, zorder=5, label='|d²P/ds²| after' if idx == 0 else '')
            ax3.plot(inter_s, d2_trans, 'kx', markersize=7, zorder=6, label='|d²P/ds²| at transition' if idx == 0 else '')
        
        ax3.set_title('Second Derivative Magnitude (|d²P/ds²|)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Arc Length (m)')
        ax3.set_ylabel('|d²P/ds²| (m)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        # Save to file if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        
        # Print intersection curvature information
        print("\n" + "="*60)
        print("INTERSECTION CURVATURE ANALYSIS")
        print("="*60)
        intersections = hybrid_path.get_intersection_curvatures()
        continuity = hybrid_path.check_curvature_continuity()
        
        for idx, ((inter_idx, curv_before, curv_after), (_, jump, is_cont)) in enumerate(zip(intersections, continuity)):
            print(f"\nIntersection {inter_idx} (between component {inter_idx} and {inter_idx+1}):")
            print(f"  Curvature before (end of comp {inter_idx}):   {curv_before:.6f} 1/m")
            print(f"  Curvature after  (start of comp {inter_idx+1}): {curv_after:.6f} 1/m")
            
            # Transition curvature from three-point method (if available)
            if hasattr(hybrid_path, 'transition_curvatures') and idx < len(hybrid_path.transition_curvatures):
                k_tr = hybrid_path.transition_curvatures[idx]
                print(f"  Transition curvature (three-point):          {k_tr:.6f} 1/m")
            
            print(f"  Curvature jump (|after-before|):             {jump:.6f} 1/m")
            print(f"  Continuous (by threshold):                   {'✓' if is_cont else '✗'}")
        
        print("\n" + "="*60)


# %%
# Helper functions for creating common hybrid paths

def create_rectangle_hybrid_path(corner1, corner2, corner3, corner4, use_arcs=False, arc_radius=None):
    """
    Create a perfect rectangle hybrid path from four corners.
    
    Args:
        corner1, corner2, corner3, corner4: (x, y) corners in order
        use_arcs: If True, use arc segments at corners instead of sharp turns
        arc_radius: Radius for corner arcs (only used if use_arcs=True)
        
    Returns:
        HybridPath instance
    """
    corners = [np.array(corner1), np.array(corner2), np.array(corner3), np.array(corner4)]
    
    if use_arcs and arc_radius is not None:
        # Create rectangle with arc corners
        components = []
        # This is more complex - would need to calculate arc centers and angles
        # For now, implement straight version
        raise NotImplementedError("Arc corners not yet implemented")
    else:
        # Simple rectangle with straight segments
        components = [
            StraightComponentPath(corners[0], corners[1]),
            StraightComponentPath(corners[1], corners[2]),
            StraightComponentPath(corners[2], corners[3]),
            StraightComponentPath(corners[3], corners[0]),
        ]
    
    return HybridPath(components)


def create_square_hybrid_path(center, side_length, use_arcs=False, arc_radius=None):
    """
    Create a perfect square hybrid path.
    
    Args:
        center: (x, y) center of square
        side_length: Length of each side
        use_arcs: If True, use arc segments at corners
        arc_radius: Radius for corner arcs
        
    Returns:
        HybridPath instance
    """
    half_side = side_length / 2.0
    corners = [
        np.array([center[0] + half_side, center[1] + half_side]),  # Top-right
        np.array([center[0] + half_side, center[1] - half_side]),  # Bottom-right
        np.array([center[0] - half_side, center[1] - half_side]),  # Bottom-left
        np.array([center[0] - half_side, center[1] + half_side]),  # Top-left
    ]
    return create_rectangle_hybrid_path(corners[0], corners[1], corners[2], corners[3], 
                                        use_arcs, arc_radius)


def create_p_trajectory_hybrid_path(start_point, stem_height, arc_radius, arc_center_offset):
    """
    Create a P-shaped hybrid path (like the letter "P").
    
    Args:
        start_point: (x, y) starting point (bottom of stem)
        stem_height: Height of the vertical stem
        arc_radius: Radius of the arc forming the top of the P
        arc_center_offset: Horizontal offset of arc center from start_point
        
    Returns:
        HybridPath instance
    """
    start = np.array(start_point)
    
    # Stem: vertical line upward
    top_stem = start + np.array([0, stem_height])
    
    # Arc center: positioned so arc starts at top_stem
    # For arc to start at top_stem with start_angle=0 (pointing right):
    # top_stem = arc_center + radius * [cos(0), sin(0)] = arc_center + [radius, 0]
    # So: arc_center = top_stem - [radius, 0] = [top_stem_x - radius, top_stem_y]
    # The arc_center_offset parameter is interpreted as: where we want the arc center to be
    # relative to start_point, but we adjust it to ensure the arc starts at top_stem.
    # Actually, to satisfy both constraints, we need: start_x + arc_center_offset = top_stem_x - radius
    # But if that doesn't hold, we prioritize the constraint that arc starts at top_stem.
    # So: arc_center_x = top_stem_x - radius
    #     arc_center_y = top_stem_y
    arc_center = np.array([top_stem[0] - arc_radius, top_stem[1]])
    
    # Arc: semicircle from top of stem, going right and down
    # Start angle: pointing right (0 radians) - arc starts at top_stem
    # End angle: pointing down (-pi/2 radians, going clockwise)
    start_angle = 0.0
    end_angle = np.pi / 2.0
    
    components = [
        StraightComponentPath(start, top_stem),  # Vertical stem
        ArcComponentPath(arc_center, arc_radius, start_angle, end_angle, clockwise=True),  # Top arc
    ]
    
    return HybridPath(components)


def create_catenary_hybrid_path(x_start, x_end, a, y_offset=0.0, num_points=20):
    """
    Create a catenary curve hybrid path.
    Catenary equation: y = a * cosh((x - x_center) / a) + y_offset
    
    Args:
        x_start: Starting x coordinate
        x_end: Ending x coordinate
        a: Catenary parameter (high a = flatter curve, low a = deeper curve)
        y_offset: Vertical offset
        num_points: Number of waypoints for spline interpolation
        
    Returns:
        HybridPath instance (single SplineComponentPath)
    """
    x_center = (x_start + x_end) / 2.0
    x_range = np.linspace(x_start, x_end, num_points)
    
    # Generate catenary waypoints
    waypoints = []
    for x in x_range:
        y = a * np.cosh((x - x_center) / a) + y_offset
        # Normalize so the lowest point is at y_offset
        y_min = a * np.cosh((x_center - x_center) / a) + y_offset  # = a + y_offset
        y = y - (y_min - y_offset)  # Shift so minimum is at y_offset
        waypoints.append([x, y, 0.0])  # Orientation will be calculated from path
    
    # Create single SplineComponentPath
    spline_comp = SplineComponentPath(waypoints)
    return HybridPath([spline_comp])


# %%
def demo_three_hybrid_paths():
    """
    Demo function that creates three hybrid paths:
    1. Rectangle path
    2. P trajectory
    3. Catenary curve with high alpha
    
    Plots each path with its 1st and 2nd derivatives and saves to /tmp/hybrid_path
    """
    import os
    from pathlib import Path
    
    # Create output directory
    output_dir = Path("/tmp/hybrid_path")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 1. Create rectangle path
    print("\n" + "="*60)
    print("1. Creating Rectangle Hybrid Path")
    print("="*60)
    rectangle_path = create_rectangle_hybrid_path(
        corner1=[0, 0],
        corner2=[3, 0],
        corner3=[3, 2],
        corner4=[0, 2]
    )
    print(f"   Total length: {rectangle_path.total_length:.4f} m")
    print(f"   Components: {rectangle_path.num_components}")
    
    # 2. Create P trajectory
    print("\n" + "="*60)
    print("2. Creating P Trajectory Hybrid Path")
    print("="*60)
    p_path = create_p_trajectory_hybrid_path(
        start_point=[0, 0],
        stem_height=2.0,
        arc_radius=1.0,
        arc_center_offset=1.0
    )
    print(f"   Total length: {p_path.total_length:.4f} m")
    print(f"   Components: {p_path.num_components}")
    
    # 3. Create catenary curve with high alpha (flatter)
    print("\n" + "="*60)
    print("3. Creating Catenary Curve Hybrid Path (high alpha)")
    print("="*60)
    catenary_path = create_catenary_hybrid_path(
        x_start=-2.0,
        x_end=2.0,
        a=5.0,  # High alpha for flatter curve
        y_offset=0.0,
        num_points=100
    )
    print(f"   Total length: {catenary_path.total_length:.4f} m")
    print(f"   Components: {catenary_path.num_components}")
    
    # Plot and save each path using the HybridPathVisualizer
    paths = [
        ("rectangle", rectangle_path),
        ("p_trajectory", p_path),
        ("catenary", catenary_path)
    ]
    
    for name, path in paths:
        print(f"\n{'='*60}")
        print(f"Plotting {name} path and derivatives...")
        print(f"{'='*60}")
        
        # Save curvature & derivatives plot using the dedicated visualizer
        analysis_path = output_dir / f"{name}_analysis.png"
        HybridPathVisualizer.plot_curvature_and_derivatives(
            path,
            num_samples=500,
            save_path=analysis_path
        )
        print(f"   Saved curvature/derivatives plot: {analysis_path}")
    
    print(f"\n{'='*60}")
    print("Demo completed! All plots saved to:", output_dir)
    print(f"{'='*60}\n")
    
    return rectangle_path, p_path, catenary_path


# %%
class PathError:
    """Utility class for path error calculations"""
    
    @staticmethod
    def normalize_theta_diff(target, start):
        """
        Signed angle, not the smallest angle difference.
        Normalize the difference from start to target, to the range [-pi, pi].
        """
        diff = target - start
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff
    
    @staticmethod
    def absolute_theta_diff(target, start):
        """
        Absolute angle difference, always the smallest angle difference, should be non-negative.
        Normalize the difference from start to target, to the range [0, pi].
        """
        return min(abs(PathError.normalize_theta_diff(target, start)), 
                   abs(PathError.normalize_theta_diff(start, target)))
    
    @staticmethod
    def calculate_contour_error(reference_path, query_point):
        """Calculate contour error between a point and reference path"""
        return reference_path.get_contour_error(query_point)
    


# %%
class PathVisualizer:
    """Tools for visualizing paths and trajectory tracking.
    Supports both SplineReferencePath (legacy) and HybridPath (new system).
    """
    # General visualization functions
    
    @staticmethod
    def _is_hybrid_path(path):
        """Check if path is a HybridPath instance"""
        return isinstance(path, HybridPath)
    
    @staticmethod
    def plot_path(reference_path, num_points=100):
        """
        Plot the reference path.
        Supports both SplineReferencePath and HybridPath.
        """
        # Sample points along the path
        t_samples = np.linspace(0, 1, num_points)
        points = np.array([reference_path.get_point_at_parameter(t) for t in t_samples])
        
        plt.figure(figsize=(10, 6))
        plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Path')
        
        # Plot orientation arrows at intervals
        num_arrows = min(20, num_points)
        arrow_indices = np.linspace(0, num_points-1, num_arrows, dtype=int)
        for idx in arrow_indices:
            point = reference_path.get_point_at_parameter(t_samples[idx])
            if len(point) >= 2:
                x, y = point[0], point[1]
                # Get orientation from tangent
                tangent = reference_path.get_tangent_at_parameter(t_samples[idx])
                theta = np.arctan2(tangent[1], tangent[0])
                plt.arrow(x, y, 0.1*np.cos(theta), 0.1*np.sin(theta), 
                         head_width=0.05, head_length=0.07, fc='b', ec='b', alpha=0.6)
        
        # Plot waypoints if available (only for SplineReferencePath)
        if not PathVisualizer._is_hybrid_path(reference_path):
            if hasattr(reference_path, 'points_x') and hasattr(reference_path, 'points_y'):
                plt.plot(reference_path.points_x, reference_path.points_y, 'ro', markersize=6, label='Waypoints')
        else:
            # For HybridPath, mark component boundaries
            for i in range(reference_path.num_components):
                comp_start = reference_path.components[i].start_point
                plt.plot(comp_start[0], comp_start[1], 'ro', markersize=6, 
                        label='Component Start' if i == 0 else '')
            comp_end = reference_path.components[-1].end_point
            plt.plot(comp_end[0], comp_end[1], 'go', markersize=6, label='Path End')
        
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Reference Path')
        plt.legend()
        plt.show()


    @staticmethod
    def plot_gradient(reference_path, num_samples=200, use_degrees=True):
        """
        Plot positional and rotational gradients along the reference path.
        Supports both SplineReferencePath and HybridPath.
        
        Args:
            reference_path: SplineReferencePath or HybridPath instance
            num_samples: Number of points to sample along the path for gradient calculation
            use_degrees: If True, display rotational gradient in degrees/m instead of rad/m
        """
        # Sample points along the path
        t_samples = np.linspace(0, 1, num_samples)
            
        # Calculate curvature (positional gradient) at each point
        curvatures = []
        for t in t_samples:
            curvature = reference_path.get_curvature_at_parameter(t)
            curvatures.append(curvature)
        
        curvatures = np.array(curvatures)
        
        # Calculate orientation change (rotational gradient) at each point
        orientations = []
        for t in t_samples:
            point = reference_path.get_point_at_parameter(t)
            # For HybridPath, get orientation from tangent
            if len(point) >= 3:
                orientations.append(point[2])  # theta from point
            else:
                # Calculate from tangent
                tangent = reference_path.get_tangent_at_parameter(t)
                theta = np.arctan2(tangent[1], tangent[0])
                orientations.append(theta)
        
        orientations = np.array(orientations)
        
        # **FIX: Unwrap angles to remove 2π discontinuities**
        orientations_unwrapped = np.unwrap(orientations)
        
        # **CRITICAL FIX: Convert path parameter to arc length for proper gradient calculation**
        # Both SplineReferencePath and HybridPath have t_to_s method
        arc_lengths = np.array([reference_path.t_to_s(t) for t in t_samples])
        
        # Calculate the rate of change of orientation (rotational gradient)
        # NOW using arc length differences instead of parameter differences
        rotational_gradient = np.zeros_like(orientations_unwrapped)
        
        for i in range(1, len(orientations_unwrapped) - 1):
            # Central difference using ARC LENGTH
            ds = arc_lengths[i+1] - arc_lengths[i-1]  # Arc length difference
            if ds > 1e-10:  # Avoid division by zero
                rotational_gradient[i] = (orientations_unwrapped[i+1] - orientations_unwrapped[i-1]) / ds
        
        # Forward difference for first point (using arc length)
        ds_forward = arc_lengths[1] - arc_lengths[0]
        if ds_forward > 1e-10:
            rotational_gradient[0] = (orientations_unwrapped[1] - orientations_unwrapped[0]) / ds_forward
        
        # Backward difference for last point (using arc length)
        ds_backward = arc_lengths[-1] - arc_lengths[-2]
        if ds_backward > 1e-10:
            rotational_gradient[-1] = (orientations_unwrapped[-1] - orientations_unwrapped[-2]) / ds_backward
        
        # Convert to degrees if requested
        if use_degrees:
            rotational_gradient_display = np.degrees(rotational_gradient)
            unit_label = '°/m'
            unit_name = 'degrees per meter'
        else:
            rotational_gradient_display = rotational_gradient
            unit_label = 'rad/m'
            unit_name = 'radians per meter'
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # --- First subplot: Positional Gradient (Curvature) ---
        ax1.plot(arc_lengths, curvatures, 'b-', linewidth=2)
        ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax1.fill_between(arc_lengths, 0, curvatures, alpha=0.3, color='blue')
        ax1.set_title('Positional Gradient (Path Curvature)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Arc Length (m)')
        ax1.set_ylabel('Curvature (1/m)')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_curvature = np.mean(np.abs(curvatures))
        max_curvature = np.max(np.abs(curvatures))
        ax1.text(0.02, 0.98, 
                f'Mean |curvature|: {mean_curvature:.4f}\nMax |curvature|: {max_curvature:.4f}',
                transform=ax1.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Highlight regions of high curvature
        high_curvature_threshold = np.percentile(np.abs(curvatures), 75)
        high_curvature_mask = np.abs(curvatures) > high_curvature_threshold
        if np.any(high_curvature_mask):
            ax1.scatter(arc_lengths[high_curvature_mask], 
                    curvatures[high_curvature_mask],
                    color='red', s=20, alpha=0.6, 
                    label=f'High curvature (>{high_curvature_threshold:.3f})')
            ax1.legend(loc='upper right')
        
        # --- Second subplot: Rotational Gradient (FIXED with arc length) ---
        ax2.plot(arc_lengths, rotational_gradient_display, 'r-', linewidth=2)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.fill_between(arc_lengths, 0, rotational_gradient_display, alpha=0.3, color='red')
        ax2.set_title(f'Rotational Gradient (Rate of Orientation Change) [{unit_name.upper()}]', 
                    fontsize=12, fontweight='bold')
        ax2.set_xlabel('Arc Length (m)')
        ax2.set_ylabel(f'dθ/ds ({unit_label})')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_rot_gradient = np.mean(np.abs(rotational_gradient_display))
        max_rot_gradient = np.max(np.abs(rotational_gradient_display))
        
        # Also show in radians for reference
        mean_rot_rad = np.mean(np.abs(rotational_gradient))
        max_rot_rad = np.max(np.abs(rotational_gradient))
        
        ax2.text(0.02, 0.98,
                f'Mean |dθ/ds|: {mean_rot_gradient:.2f} {unit_label} ({mean_rot_rad:.4f} rad/m)\n'
                f'Max |dθ/ds|: {max_rot_gradient:.2f} {unit_label} ({max_rot_rad:.4f} rad/m)',
                transform=ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        # Highlight regions of high rotational gradient
        high_rot_threshold = np.percentile(np.abs(rotational_gradient_display), 75)
        high_rot_mask = np.abs(rotational_gradient_display) > high_rot_threshold
        if np.any(high_rot_mask):
            ax2.scatter(arc_lengths[high_rot_mask],
                    rotational_gradient_display[high_rot_mask],
                    color='darkred', s=20, alpha=0.6,
                    label=f'High rotation rate (>{high_rot_threshold:.2f} {unit_label})')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("PATH GRADIENT ANALYSIS SUMMARY")
        print("="*60)
        # Both path types have total_path_length (HybridPath) or total_path_length (SplineReferencePath)
        path_length = reference_path.total_length if PathVisualizer._is_hybrid_path(reference_path) else reference_path.total_path_length
        print(f"Total path length: {path_length:.3f} m")
        print(f"\nPositional Gradient (Curvature):")
        print(f"  Mean absolute curvature: {mean_curvature:.4f} 1/m")
        print(f"  Maximum absolute curvature: {max_curvature:.4f} 1/m")
        print(f"  Standard deviation: {np.std(curvatures):.4f}")
        print(f"\nRotational Gradient (dθ/ds):")
        print(f"  Mean absolute rate: {mean_rot_gradient:.2f} {unit_label} ({mean_rot_rad:.4f} rad/m)")
        print(f"  Maximum absolute rate: {max_rot_gradient:.2f} {unit_label} ({max_rot_rad:.4f} rad/m)")
        print(f"  Standard deviation: {np.std(rotational_gradient):.4f} rad/m")
        print("="*60)


    @staticmethod
    def plot_double_path(path_a, path_b, num_points=100):
        """
        Plot two reference paths together for visual comparison.
        Supports both SplineReferencePath and HybridPath.
        
        Args:
            path_a: First path (SplineReferencePath or HybridPath)
            path_b: Second path (SplineReferencePath or HybridPath)
            num_points: Number of points to sample along each path
        """
        # Sample points along both paths
        t_samples = np.linspace(0, 1, num_points)
        points_a = np.array([path_a.get_point_at_parameter(t) for t in t_samples])
        points_b = np.array([path_b.get_point_at_parameter(t) for t in t_samples])
        
        plt.figure(figsize=(12, 8))
        
        # Plot path A
        plt.plot(points_a[:, 0], points_a[:, 1], 'b-', linewidth=2, label='Path A')
        # Plot waypoints if available (only for SplineReferencePath)
        if not PathVisualizer._is_hybrid_path(path_a):
            if hasattr(path_a, 'points_x') and hasattr(path_a, 'points_y'):
                plt.plot(path_a.points_x, path_a.points_y, 'bo', markersize=4, alpha=0.6)
        
        # Plot path B
        plt.plot(points_b[:, 0], points_b[:, 1], 'r-', linewidth=2, label='Path B')
        # Plot waypoints if available (only for SplineReferencePath)
        if not PathVisualizer._is_hybrid_path(path_b):
            if hasattr(path_b, 'points_x') and hasattr(path_b, 'points_y'):
                plt.plot(path_b.points_x, path_b.points_y, 'ro', markersize=4, alpha=0.6)
        
        # Plot orientation arrows at intervals for both paths
        num_arrows = min(10, num_points)
        arrow_indices = np.linspace(0, num_points-1, num_arrows, dtype=int)
        
        for idx in arrow_indices:
            # Path A arrows
            point_a = points_a[idx]
            x, y = point_a[0], point_a[1]
            tangent_a = path_a.get_tangent_at_parameter(t_samples[idx])
            theta_a = np.arctan2(tangent_a[1], tangent_a[0])
            plt.arrow(x, y, 0.1*np.cos(theta_a), 0.1*np.sin(theta_a), 
                    head_width=0.05, head_length=0.07, fc='b', ec='b', alpha=0.6)
            
            # Path B arrows
            point_b = points_b[idx]
            x, y = point_b[0], point_b[1]
            tangent_b = path_b.get_tangent_at_parameter(t_samples[idx])
            theta_b = np.arctan2(tangent_b[1], tangent_b[0])
            plt.arrow(x, y, 0.1*np.cos(theta_b), 0.1*np.sin(theta_b), 
                    head_width=0.05, head_length=0.07, fc='r', ec='r', alpha=0.6)
        
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Comparison of Two Reference Paths')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_cross_track_error_overtime(path_a, path_b, mode='both', num_samples=100):
        """
        Plot cross-track and/or along-track errors between two paths.
        
        Args:
            path_a: Reference path (errors are measured from path_b to path_a)
            path_b: Test path
            mode: Error plotting mode
                - 'both': Plot both lateral and longitudinal errors
                - 'lateral': Plot only lateral (cross-track) errors
                - 'longitudinal': Plot only longitudinal (along-track) errors
            num_samples: Number of points to sample along path_b
        """
        # Sample points along path_b
        t_samples = np.linspace(0, 1, num_samples)
        points_b = np.array([path_b.get_point_at_parameter(t) for t in t_samples])
        
        # Calculate errors from each point on path_b to path_a
        lateral_errors = []
        longitudinal_errors = []
        
        for point in points_b:
            error_info = path_a.get_contour_error(point)
            lateral_errors.append(error_info['lateral'])
            longitudinal_errors.append(error_info['longitudinal'])
        
        # Convert to numpy arrays
        lateral_errors = np.array(lateral_errors)
        longitudinal_errors = np.array(longitudinal_errors)
        
        # Calculate error statistics
        mean_lateral = np.mean(np.abs(lateral_errors))
        max_lateral = np.max(np.abs(lateral_errors))
        mean_longitudinal = np.mean(np.abs(longitudinal_errors))
        max_longitudinal = np.max(np.abs(longitudinal_errors))
        
        # Print statistics
        print(f"Error Statistics:")
        print(f"  Mean absolute lateral error: {mean_lateral:.4f}")
        print(f"  Maximum absolute lateral error: {max_lateral:.4f}")
        print(f"  Mean absolute longitudinal error: {mean_longitudinal:.4f}")
        print(f"  Maximum absolute longitudinal error: {max_longitudinal:.4f}")
        
        # Create plots based on mode
        if mode == 'both':
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot lateral errors
            ax1.plot(t_samples, lateral_errors, 'b-', linewidth=2)
            ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax1.axhline(mean_lateral, color='g', linestyle='--', 
                    label=f'Mean abs: {mean_lateral:.4f}')
            ax1.axhline(-mean_lateral, color='g', linestyle='--')
            ax1.grid(True)
            ax1.set_title('Lateral (Cross-Track) Error')
            ax1.set_ylabel('Error')
            ax1.legend()
            
            # Plot longitudinal errors
            ax2.plot(t_samples, longitudinal_errors, 'r-', linewidth=2)
            ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax2.axhline(mean_longitudinal, color='g', linestyle='--', 
                    label=f'Mean abs: {mean_longitudinal:.4f}')
            ax2.axhline(-mean_longitudinal, color='g', linestyle='--')
            ax2.grid(True)
            ax2.set_title('Longitudinal (Along-Track) Error')
            ax2.set_xlabel('Path Parameter of Path B')
            ax2.set_ylabel('Error')
            ax2.legend()
            
        elif mode == 'lateral':
            # Plot only lateral errors
            plt.figure(figsize=(12, 6))
            plt.plot(t_samples, lateral_errors, 'b-', linewidth=2)
            plt.axhline(0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(mean_lateral, color='g', linestyle='--', 
                    label=f'Mean abs: {mean_lateral:.4f}')
            plt.axhline(-mean_lateral, color='g', linestyle='--')
            plt.grid(True)
            plt.title('Lateral (Cross-Track) Error')
            plt.xlabel('Path Parameter of Path B')
            plt.ylabel('Error')
            plt.legend()
            
        elif mode == 'longitudinal':
            # Plot only longitudinal errors
            plt.figure(figsize=(12, 6))
            plt.plot(t_samples, longitudinal_errors, 'r-', linewidth=2)
            plt.axhline(0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(mean_longitudinal, color='g', linestyle='--', 
                    label=f'Mean abs: {mean_longitudinal:.4f}')
            plt.axhline(-mean_longitudinal, color='g', linestyle='--')
            plt.grid(True)
            plt.title('Longitudinal (Along-Track) Error')
            plt.xlabel('Path Parameter of Path B')
            plt.ylabel('Error')
            plt.legend()
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'both', 'lateral', or 'longitudinal'")
        
        plt.tight_layout()
        plt.show()

    # Object trajectory visualization functions
    # wait for implementation

    @staticmethod
    def plot_path_with_object(reference_path, generic_object, num_path_points=100, num_object_poses=6):
        """
        Visualize a reference path with a generic object positioned at multiple points along the path.
        Supports both SplineReferencePath and HybridPath.
        
        Args:
            reference_path: SplineReferencePath or HybridPath instance
            generic_object: GenericObject instance from ObjectLib
            num_path_points: Number of points to sample along the path
            num_object_poses: Number of object poses to display along the path
            
        Returns:
            matplotlib.axes: The plot axes
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Sample points along the path
        t_samples = np.linspace(0, 1, num_path_points)
        points = np.array([reference_path.get_point_at_parameter(t) for t in t_samples])
        
        # Plot the path
        ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Path')
        
        # Plot waypoints if available (only for SplineReferencePath)
        if not PathVisualizer._is_hybrid_path(reference_path):
            if hasattr(reference_path, 'points_x') and hasattr(reference_path, 'points_y'):
                ax.plot(reference_path.points_x, reference_path.points_y, 'ro', markersize=6, label='Waypoints')
        
        # Sample positions for object placement
        object_t_samples = np.linspace(0, 1, num_object_poses)
        
        # Get object centroid in its local coordinates to calculate offset
        local_centroid = generic_object.get_centroid()
        centroid_offset_x = local_centroid.x
        centroid_offset_y = local_centroid.y
        
        # Plot the object at each sampled position
        for i, t in enumerate(object_t_samples):
            # Get position and orientation from path
            path_point = reference_path.get_point_at_parameter(t)
            x, y = path_point[0], path_point[1]
            # Get orientation from tangent
            tangent = reference_path.get_tangent_at_parameter(t)
            theta = np.arctan2(tangent[1], tangent[0])
            
            # Calculate offset in world coordinates to align centroid with path
            offset_x = -centroid_offset_x * np.cos(theta) + centroid_offset_y * np.sin(theta)
            offset_y = -centroid_offset_x * np.sin(theta) - centroid_offset_y * np.cos(theta)
            
            # Create a transformed copy of the object, adjusting position to align centroid with path
            transformed_object = generic_object.transform(
                x + offset_x, 
                y + offset_y, 
                theta - generic_object.heading
            )
            
            # Determine color based on position (gradient from start to end)
            color_param = i / max(1, (num_object_poses - 1))
            color = plt.cm.viridis(color_param)
            
            # Plot the object with reduced alpha for better visibility
            alpha = 0.7 if i == 0 or i == num_object_poses-1 else 0.5
            label = 'Start' if i == 0 else ('End' if i == num_object_poses-1 else None)
            
            # Visualize the object
            transformed_object.visualize(
                ax=ax, 
                facecolor=color, 
                edgecolor=color, 
                alpha=alpha,
                show_frame=False
            )
            
            # Add a marker for the object position (at the path point)
            ax.plot(x, y, 'o', color=color, markersize=8, alpha=alpha, label=label)
            
            # Optionally add orientation arrows for clearer visualization
            arrow_length = 0.1
            ax.arrow(x, y, 
                    arrow_length * np.cos(theta), 
                    arrow_length * np.sin(theta),
                    head_width=0.03, 
                    head_length=0.05, 
                    fc=color, 
                    ec=color, 
                    alpha=alpha)
        
        # Add annotations for the first and last objects
        first_point = reference_path.get_point_at_parameter(object_t_samples[0])
        last_point = reference_path.get_point_at_parameter(object_t_samples[-1])
        
        ax.annotate('Start', 
                (first_point[0], first_point[1]),
                xytext=(-30, -30),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                color='green',
                fontweight='bold')
        
        ax.annotate('End', 
                (last_point[0], last_point[1]),
                xytext=(30, 30),
                textcoords='offset points', 
                arrowprops=dict(arrowstyle='->', color='darkviolet'),
                color='darkviolet',
                fontweight='bold')
        
        # Add path tangent and normal vectors at a few points for reference
        num_vectors = min(5, num_path_points)
        vector_indices = np.linspace(0, num_path_points-1, num_vectors, dtype=int)
        
        for idx in vector_indices:
            t = t_samples[idx]
            x, y, _ = reference_path.get_point_at_parameter(t)
            
            # Get tangent and normal
            tangent = reference_path.get_tangent_at_parameter(t)
            normal = reference_path.get_normal_at_parameter(t)
            
            # Plot tangent vector (green)
            ax.arrow(x, y, 
                    tangent[0] * 0.15, tangent[1] * 0.15,
                    head_width=0.03, head_length=0.04, 
                    fc='green', ec='green', alpha=0.6,
                    label='Tangent' if idx == vector_indices[0] else '')
            
            # Plot normal vector (red)
            ax.arrow(x, y, 
                    normal[0] * 0.15, normal[1] * 0.15,
                    head_width=0.03, head_length=0.04, 
                    fc='red', ec='red', alpha=0.6,
                    label='Normal' if idx == vector_indices[0] else '')
        
        # Set labels and grid
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title(f'Path with {generic_object.name} Object')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Create a legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.tight_layout()
        return ax

    @staticmethod
    def visualize_object_following_path(reference_path, generic_object, 
                                    num_path_points=100, 
                                    animation_frames=None,
                                    show_path_details=True,
                                    show_object_frames=True,
                                    title=None,
                                    stream_animation=False):
        """
        Create detailed visualization of an object following a path, with animation option.
        Supports both SplineReferencePath and HybridPath.
        
        Args:
            reference_path: SplineReferencePath or HybridPath instance
            generic_object: GenericObject instance from ObjectLib
            num_path_points: Number of points to sample along the path
            animation_frames: If provided, creates animation with this many frames
            show_path_details: Whether to show path details like normals and tangents
            show_object_frames: Whether to show object coordinate frames
            title: Custom title for the plot
            stream_animation: Whether to stream the animation (requires setup)
            
        Returns:
            matplotlib.axes or animation.FuncAnimation: The plot or animation
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Sample points along the path
        t_samples = np.linspace(0, 1, num_path_points)
        points = np.array([reference_path.get_point_at_parameter(t) for t in t_samples])
        
        # Plot the path
        path_line, = ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Path')
        
        # Plot waypoints (only for SplineReferencePath)
        if not PathVisualizer._is_hybrid_path(reference_path):
            if hasattr(reference_path, 'points_x') and hasattr(reference_path, 'points_y'):
                waypoints = ax.plot(reference_path.points_x, reference_path.points_y, 'ro', 
                            markersize=6, label='Waypoints')
            else:
                waypoints = []
        else:
            waypoints = []
        
        # Path details (tangents, normals, etc.)
        path_details = []
        if show_path_details:
            # Sample a few points for vectors
            num_vectors = min(5, num_path_points)
            vector_indices = np.linspace(0, num_path_points-1, num_vectors, dtype=int)
            
            for idx in vector_indices:
                t = t_samples[idx]
                x, y, _ = reference_path.get_point_at_parameter(t)
                
                # Get tangent and normal
                tangent = reference_path.get_tangent_at_parameter(t)
                normal = reference_path.get_normal_at_parameter(t)
                
                # Plot tangent vector (green)
                tangent_arrow = ax.arrow(x, y, 
                                    tangent[0] * 0.15, tangent[1] * 0.15,
                                    head_width=0.03, head_length=0.04, 
                                    fc='green', ec='green', alpha=0.6,
                                    label='Tangent' if idx == vector_indices[0] else '')
                
                # Plot normal vector (red)
                normal_arrow = ax.arrow(x, y, 
                                    normal[0] * 0.15, normal[1] * 0.15,
                                    head_width=0.03, head_length=0.04, 
                                    fc='red', ec='red', alpha=0.6,
                                    label='Normal' if idx == vector_indices[0] else '')
                
                path_details.extend([tangent_arrow, normal_arrow])
        
        # Set labels and grid
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(f'Path Following with {generic_object.name}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Get object centroid in its local coordinates to calculate offset
        local_centroid = generic_object.get_centroid()
        centroid_offset_x = local_centroid.x
        centroid_offset_y = local_centroid.y
        
        # Static case (no animation)
        if animation_frames is None:
            # Sample positions for object placement
            num_object_poses = 6  # Number of static object poses to show
            object_t_samples = np.linspace(0, 1, num_object_poses)
            
            # Plot the object at each sampled position
            for i, t in enumerate(object_t_samples):
                # Get position and orientation from path
                path_point = reference_path.get_point_at_parameter(t)
                x, y = path_point[0], path_point[1]
                # Get orientation from tangent
                tangent = reference_path.get_tangent_at_parameter(t)
                theta = np.arctan2(tangent[1], tangent[0])
                
                # Calculate offset in world coordinates to align centroid with path
                offset_x = -centroid_offset_x * np.cos(theta) + centroid_offset_y * np.sin(theta)
                offset_y = -centroid_offset_x * np.sin(theta) - centroid_offset_y * np.cos(theta)
                
                # Create a transformed copy of the object, adjusting position to align centroid with path
                transformed_object = generic_object.transform(
                    x + offset_x, 
                    y + offset_y, 
                    theta - generic_object.heading
                )
                
                # Determine color based on position
                color_param = i / max(1, (num_object_poses - 1))
                color = plt.cm.viridis(color_param)
                
                # Plot the object with reduced alpha for better visibility
                alpha = 0.8 if i == 0 or i == num_object_poses-1 else 0.5
                label = 'Start' if i == 0 else ('End' if i == num_object_poses-1 else None)
                
                # Visualize the object
                transformed_object.visualize(
                    ax=ax, 
                    facecolor=color, 
                    edgecolor=color, 
                    alpha=alpha,
                    show_frame=show_object_frames
                )
                
                # Add a marker for the object position (at the path point)
                ax.plot(x, y, 'o', color=color, markersize=8, alpha=alpha, label=label)
            
            # Create a legend with unique entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best')
            
            plt.tight_layout()
            return ax
        
        # Animation case
        else:
            # Create animation frames
            object_artists = []
            progress_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                verticalalignment='top', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Animation function
            def animate(frame):
                # Clear previous object
                for artist in object_artists:
                    if isinstance(artist, list):
                        for a in artist:
                            a.remove() if hasattr(a, 'remove') else None
                    else:
                        artist.remove() if hasattr(artist, 'remove') else None
                object_artists.clear()
                
                # Get path parameter for this frame
                t = frame / (animation_frames - 1) if animation_frames > 1 else 0.5
                
                # Get position and orientation from path
                path_point = reference_path.get_point_at_parameter(t)
                x, y = path_point[0], path_point[1]
                # Get orientation from tangent
                tangent = reference_path.get_tangent_at_parameter(t)
                theta = np.arctan2(tangent[1], tangent[0])
                
                # Calculate offset in world coordinates to align centroid with path
                offset_x = -centroid_offset_x * np.cos(theta) + centroid_offset_y * np.sin(theta)
                offset_y = -centroid_offset_x * np.sin(theta) - centroid_offset_y * np.cos(theta)
                
                # Create a transformed copy of the object, adjusting position to align centroid with path
                transformed_object = generic_object.transform(
                    x + offset_x, 
                    y + offset_y, 
                    theta - generic_object.heading
                )
                
                # Get transformed object geometry for plotting
                if hasattr(transformed_object.geometry, 'exterior'):
                    exterior_x, exterior_y = transformed_object.geometry.exterior.xy
                    poly = ax.fill(exterior_x, exterior_y, 
                                facecolor='lightblue', edgecolor='blue', alpha=0.8)
                    object_artists.append(poly)
                
                # Draw a marker at the path point for reference
                path_marker = ax.plot(x, y, 'o', color='red', markersize=6)
                object_artists.append(path_marker)
                
                # Optionally show object frame
                if show_object_frames:
                    centroid = transformed_object.get_centroid()
                    x_axis, y_axis = transformed_object.get_local_frame_axes()
                    
                    # X-axis (forward, blue)
                    x_arrow = ax.arrow(centroid.x, centroid.y, 
                                    x_axis[0] * 0.1, x_axis[1] * 0.1,
                                    head_width=0.02, head_length=0.02, 
                                    fc='blue', ec='blue', linewidth=2)
                    
                    # Y-axis (left, green)
                    y_arrow = ax.arrow(centroid.x, centroid.y,
                                    y_axis[0] * 0.1, y_axis[1] * 0.1,
                                    head_width=0.02, head_length=0.02,
                                    fc='green', ec='green', linewidth=2)
                    
                    object_artists.extend([x_arrow, y_arrow])
                
                # Update progress text
                arc_length = reference_path.t_to_s(t)
                progress_text.set_text(f'Progress: {t:.2f} ({arc_length:.2f} m)')
                object_artists.append(progress_text)
                
                # Return all artists that need to be updated
                return [path_line] + waypoints + path_details + object_artists
            
            # Create animation
            from matplotlib.animation import FuncAnimation
            anim = FuncAnimation(fig, animate, frames=animation_frames, 
                                interval=50, blit=True)
            
            # Stream animation if requested
            if stream_animation:
                for frame in range(animation_frames):
                    animate(frame)
                    # Use the imported stream_figure function directly
                    stream_figure(fig)
                return None
            
            return anim

    # MPCC specific visualization functions
    @staticmethod
    def MPCC_plot_cross_track_error_over_time(trajectory_states, reference_path, dt):
        """Plot cross-track error over time"""
        if not trajectory_states:
            print("No trajectory provided to plot.")
            return
        
        # Calculate cross-track error at each state
        cross_track_errors = []
        for state in trajectory_states:
            if hasattr(state, 'object_state'):  # Handle MPCC state objects
                query_point = [state.object_state.x, state.object_state.y, state.object_state.theta]
            else:  # Handle simple position tuples/arrays
                query_point = state[:3] if len(state) >= 3 else state[:2] + [0]
                
            error_info = reference_path.get_contour_error(query_point)
            cross_track_errors.append(abs(error_info['lateral']))
        
        # Create time array
        time_array = np.arange(len(cross_track_errors)) * dt
        
        # Calculate mean error
        mean_error = np.mean(cross_track_errors)
        print(f"Mean Cross-Track Error: {mean_error:.4f} m")
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(time_array, cross_track_errors, label='Cross-Track Error', color='blue')
        plt.title('Cross-Track Error Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Cross-Track Error (m)')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True)
        plt.legend()
        plt.show()
        
    @staticmethod
    def visualize_trajectory_detailed_mpcc(trajectory_states, initial_state, target_state,
                                     reference_path, **kwargs):
        """Create detailed visualization of trajectory and path tracking"""
        # Extract optional parameters
        object_dimensions = kwargs.get('object_dimensions', (0.3, 0.5))
        dt = kwargs.get('dt', 0.1)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), 
                                     gridspec_kw={'height_ratios': [2, 1]})
        
        # Extract trajectory coordinates
        if hasattr(trajectory_states[0], 'object_state'):
            x_coords = [s.object_state.x for s in trajectory_states]
            y_coords = [s.object_state.y for s in trajectory_states]
            thetas = [s.object_state.theta for s in trajectory_states]
        else:
            x_coords = [s[0] for s in trajectory_states]
            y_coords = [s[1] for s in trajectory_states]
            thetas = [s[2] if len(s) > 2 else 0 for s in trajectory_states]
        
        # --- First subplot: Path and trajectory visualization ---
        # Plot reference path
        path_points = []
        for t in np.linspace(0, 1, 100):
            point = reference_path.get_point_at_parameter(t)
            path_points.append(point[:2])  # Only x, y
        path_points = np.array(path_points)
        ax1.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=2, label='Reference Path')
        
        # Plot the trajectory
        ax1.plot(x_coords, y_coords, 'g-', linewidth=1.5, label='Actual Trajectory')
        
        # Draw start and goal
        if initial_state is not None:
            if hasattr(initial_state, 'object_state'):
                ax1.plot(initial_state.object_state.x, initial_state.object_state.y, 'go', markersize=10, label='Start')
            else:
                ax1.plot(initial_state[0], initial_state[1], 'go', markersize=10, label='Start')
                
        if target_state is not None:
            if hasattr(target_state, 'object_state'):
                ax1.plot(target_state.object_state.x, target_state.object_state.y, 'ro', markersize=10, label='Goal')
            else:
                ax1.plot(target_state[0], target_state[1], 'ro', markersize=10, label='Goal')
        
        # Draw the object shapes along the trajectory
        width, length = object_dimensions
        num_shapes = min(20, len(trajectory_states))
        shape_indices = np.linspace(0, len(trajectory_states)-1, num_shapes, dtype=int)
        
        for i, idx in enumerate(shape_indices):
            # Get position and orientation
            x, y, theta = x_coords[idx], y_coords[idx], thetas[idx]
            
            # Create rectangle corners (centered at origin)
            corners = [
                (-length/2, -width/2),
                (length/2, -width/2),
                (length/2, width/2),
                (-length/2, width/2),
                (-length/2, -width/2)  # Close the polygon
            ]
            
            # Rotate and translate corners
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rotated_corners = []
            for px, py in corners:
                # Rotate
                xr = px * cos_theta - py * sin_theta
                yr = px * sin_theta + py * cos_theta
                # Translate
                xr += x
                yr += y
                rotated_corners.append((xr, yr))
            
            # Draw the rectangle
            rect_xs, rect_ys = zip(*rotated_corners)
            color = plt.cm.Blues(0.2 + 0.8 * (i / num_shapes))
            ax1.plot(rect_xs, rect_ys, '-', color=color, linewidth=1.5)
            
            # Draw orientation arrow
            ax1.arrow(x, y, 0.1*np.cos(theta), 0.1*np.sin(theta),
                    head_width=0.05, fc=color, ec=color, length_includes_head=True)
        
        ax1.set_title('Path Following Trajectory')
        ax1.set_xlabel('X position (m)')
        ax1.set_ylabel('Y position (m)')
        ax1.grid(True)
        ax1.axis('equal')
        ax1.legend(loc='best')
        
        # --- Second subplot: Cross-track error ---
        # Calculate cross-track error at each state
        cross_track_errors = []
        for idx, _ in enumerate(trajectory_states):
            query_point = [x_coords[idx], y_coords[idx], thetas[idx]]
            error_info = reference_path.get_contour_error(query_point)
            cross_track_errors.append(error_info['lateral'])
        
        # Create time array
        time_array = np.arange(len(cross_track_errors)) * dt
        
        # Plot cross-track error
        ax2.plot(time_array, cross_track_errors, 'b-', linewidth=1.5)
        ax2.set_title('Cross-Track Error Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Cross-Track Error (m)')
        ax2.grid(True)
        ax2.axhline(0, color='r', linestyle='--', alpha=0.7)
        
        # Show mean cross-track error
        mean_error = np.mean(np.abs(cross_track_errors))
        ax2.text(0.02, 0.95, f'Mean absolute error: {mean_error:.4f} m', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def small_visualizer_with_path_and_tracking_mpcc(trajectory_states, reference_path):
        """Simple visualization of trajectory and path tracking"""
        plt.figure(figsize=(10, 6))
        
        # Plot reference path
        path_points = []
        for t in np.linspace(0, 1, 100):
            point = reference_path.get_point_at_parameter(t)
            path_points.append(point[:2])  # Only x, y
        path_points = np.array(path_points)
        plt.plot(path_points[:, 0], path_points[:, 1], 'b-', label='Reference Path')
        
        # Handle different object types
        if hasattr(trajectory_states[0], 'object_state'):
            # Extract positions from MPCC states
            x_coords = [state.object_state.x for state in trajectory_states]
            y_coords = [state.object_state.y for state in trajectory_states]
            
            # Plot the approximation of the path based on path parameters
            approx_path_points = []
            for state in trajectory_states:
                current_path_param = state.path_param
                point = reference_path.get_point_at_parameter(current_path_param)
                approx_path_points.append(point)
            approx_path_points = np.array(approx_path_points)
            plt.plot(approx_path_points[:, 0], approx_path_points[:, 1], 'ro', 
                    label='Path Parameter Points', alpha=0.7, markersize=4)
            
            # Plot orientation arrows
            for i, state in enumerate(trajectory_states):
                if i % 3 == 0:  # Plot every 3rd state to avoid clutter
                    x, y, theta = state.object_state.x, state.object_state.y, state.object_state.theta
                    plt.arrow(x, y, 0.1*np.cos(theta), 0.1*np.sin(theta),
                             head_width=0.05, head_length=0.07, fc='g', ec='g', alpha=0.6)
        else:
            # For simple position arrays/tuples
            x_coords = [s[0] for s in trajectory_states]
            y_coords = [s[1] for s in trajectory_states]
        
        # Plot actual trajectory
        plt.plot(x_coords, y_coords, 'g.-', label='Object Trajectory')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Path Tracking Visualization')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


# %%
class PathDerivatives:
    """Advanced path derivative and gradient calculations"""
    
    @staticmethod
    def get_path_gradient(path, param, position):
        """Calculate the gradient of the path at a specific parameter"""
        tangent = path.get_tangent_at_parameter(param)
        normal = path.get_normal_at_parameter(param)
        return tangent, normal
    
    @staticmethod
    def getdphivirt_dtheta(theta_virt, pathinfo):
        """Compute d(phi_virt)/d(theta) evaluated at theta_virt"""
        dxdth = pathinfo.spline_dx(theta_virt)  # d x_virt / d theta
        dydth = pathinfo.spline_dy(theta_virt)    # d y_virt / d theta
        d2xdth2 = pathinfo.spline_d2x(theta_virt)  # d2 x_virt / d theta^2
        d2ydth2 = pathinfo.spline_d2y(theta_virt)   # d2 y_virt / d theta^2

        numer = dxdth * d2ydth2 - dydth * d2xdth2
        denom = dxdth**2 + dydth**2
        
        if denom < 1e-10:
            return 0.0

        dphivirt_dtheta = numer / denom
        return dphivirt_dtheta
    
    @staticmethod
    def getderror_dtheta(pathinfo, theta_virt, x_phys, y_phys):
        """Calculate derivatives of error with respect to path parameter"""
        dxvirt_dtheta = pathinfo.spline_dx(theta_virt)  # d x_virt / d theta
        dyvirt_dtheta = pathinfo.spline_dy(theta_virt)    # d y_virt / d theta

        phi_virt = np.arctan2(dyvirt_dtheta, dxvirt_dtheta)  # orientation of virtual position
        # virtual positions
        x_virt = pathinfo.spline_x(theta_virt)
        y_virt = pathinfo.spline_y(theta_virt)

        # difference in position between virtual and physical
        Dx = x_phys - x_virt
        Dy = y_phys - y_virt

        dphivirt_dtheta = PathDerivatives.getdphivirt_dtheta(theta_virt, pathinfo)

        cos_phi_virt = np.cos(phi_virt)
        sin_phi_virt = np.sin(phi_virt)

        tmp1 = np.array([dphivirt_dtheta, 1])         # 1x2 row vector
        tmp2 = np.array([cos_phi_virt, sin_phi_virt]).reshape(2,1)  # 2x1 column vector

        MC = np.array([[Dx, Dy],
                      [dyvirt_dtheta, -dxvirt_dtheta]])
        ML = np.array([[-Dy, Dx],
                      [dxvirt_dtheta, dyvirt_dtheta]])

        # Calculate the dot product
        deC_dtheta = tmp1.dot(MC).dot(tmp2).item()
        deL_dtheta = tmp1.dot(ML).dot(tmp2).item()

        return deC_dtheta, deL_dtheta, cos_phi_virt, sin_phi_virt
    
    @staticmethod
    def getErrorGradient(pathinfo, theta_virt, model_params, x_phys, y_phys):
        """Calculate error gradients for optimization"""
        deC_dtheta, deL_dtheta, cos_phi_virt, sin_phi_virt = PathDerivatives.getderror_dtheta(
            pathinfo, theta_virt, x_phys, y_phys)

        # Create gradient vectors with zeros for velocities
        n_states = model_params.nx if hasattr(model_params, 'nx') else 7
        
        # For contouring error gradient
        grad_eC = np.zeros(n_states)
        grad_eC[0] = sin_phi_virt        # x
        grad_eC[1] = -cos_phi_virt       # y
        grad_eC[-1] = deC_dtheta         # path parameter
        
        # For lag error gradient
        grad_eL = np.zeros(n_states)
        grad_eL[0] = -cos_phi_virt       # x
        grad_eL[1] = -sin_phi_virt       # y  
        grad_eL[-1] = deL_dtheta         # path parameter

        return grad_eC, grad_eL

# %%
# Additional utility functions
# DEPRECATED: Use HybridPath.create_from_trajectory() instead
def create_path_from_trajectory(trajectory, orientation_mode="follow_path"):
    """
    DEPRECATED: This function is deprecated. Use HybridPath.create_from_trajectory() instead.
    
    Create a reference path from trajectory data.
    
    Args:
        trajectory: List of points in one of these formats:
            - [(x1, y1), (x2, y2), ...] (positions only)
            - [(x1, y1, theta1), (x2, y2, theta2), ...] (positions with orientations)
        orientation_mode: How to handle orientation
            - "explicit": Use provided orientations (required if not provided in trajectory)
            - "follow_path": Calculate orientations based on path tangent
    
    Returns:
        HybridPath: A hybrid path created from the trajectory (single SplineComponentPath)
    """
    return HybridPath.create_from_trajectory(trajectory, orientation_mode=orientation_mode)
 

def create_reference_path_from_waypoints(waypoints, mode="follow_path"):
    """Create a reference path from waypoints"""
    return SplineReferencePath(waypoints, orientation_mode=mode)

def calculate_path_length(path):
    """Calculate the total length of a path"""
    return path.total_path_length

def resample_path_by_distance(path, spacing):
    """Resample a path with equal distance spacing"""
    # Calculate how many points we need
    total_length = path.total_path_length
    num_points = max(2, int(total_length / spacing) + 1)
    
    # Get equally spaced arc lengths
    arc_lengths = np.linspace(0, total_length, num_points)
    
    # Convert arc lengths to path parameters
    t_values = [path.s_to_t(s) for s in arc_lengths]
    
    # Get points at these parameters
    new_points = [path.get_point_at_parameter(t) for t in t_values]
    
    # Create a new path from these points
    return SplineReferencePath(new_points, orientation_mode=path.orientation_mode)

def compare_paths(path1, path2, metric="rmse"):
    """Compare two paths using specified metric"""
    # Sample points along both paths
    t_samples = np.linspace(0, 1, 100)
    points1 = np.array([path1.get_point_at_parameter(t)[:2] for t in t_samples])
    
    # Calculate distances from points on path1 to path2
    distances = []
    for point in points1:
        closest, _, _ = path2.find_closest_point(point)
        dist = np.linalg.norm(point - closest[:2])
        distances.append(dist)
    
    # Calculate specified metric
    if metric == "rmse":
        return np.sqrt(np.mean(np.array(distances)**2))
    elif metric == "max":
        return np.max(distances)
    elif metric == "mean":
        return np.mean(distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")
