#!/usr/bin/env python3
"""
Path Velocity Planner - Segment-Based Trapezoidal Velocity Planning

This module implements a velocity planner for hybrid paths that:
- Decomposes paths into segments (Straight, Arc, Spline)
- Assigns velocity caps based on curvature constraints
- Generates trapezoidal/triangular velocity profiles per segment
- Supports look-ahead mode for smooth transitions between segments
"""

import numpy as np
import sys


from pathlib import Path

# Add paths to import paths_lib

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src" / "legacy"))

from paths_lib import (
    HybridPath,
    ComponentPath,
    StraightComponentPath,
    ArcComponentPath,
    SplineComponentPath
)

class Segment:
    """
    Wrapper for a path segment with velocity planning properties.
    
    Attributes:
        component: ComponentPath instance (Straight, Arc, or Spline)
        length: Arc length of the segment (m)
        k_max: Maximum curvature magnitude on the segment (1/m)
        v_user_max: User/system speed limit (m/s)
        v_cap: Computed velocity cap for this segment (m/s)
        v_start: Velocity at segment start (m/s)
        v_end: Velocity at segment end (m/s)
        v_cruise: Cruise velocity for this segment (m/s)
    """
    
    def __init__(self, component, v_user_max):
        """
        Initialize a segment from a ComponentPath.
        
        Args:
            component: ComponentPath instance
            v_user_max: Maximum user/system speed limit (m/s)
        """
        self.component = component
        self.length = component.get_path_length()
        self.v_user_max = v_user_max
        self.v_cap = None  # Will be computed by planner
        self.v_start = 0.0
        self.v_end = 0.0
        self.v_cruise = 0.0
        
        # Compute maximum curvature on this segment
        self.k_max = self._compute_max_curvature()
    
    def _compute_max_curvature(self):
        """
        Compute the maximum curvature magnitude on this segment.
        
        Returns:
            Maximum |curvature| on the segment (1/m)
        """
        if self.length < 1e-10:
            return 0.0
        
        # Sample curvature along the segment
        num_samples = max(10, int(self.length * 100))  # ~100 samples per meter
        s_samples = np.linspace(0.0, self.length, num_samples)
        
        curvatures = []
        for s in s_samples:
            k = self.component.get_curvature_at_arc_length(s)
            curvatures.append(abs(k))
        
        return max(curvatures) if curvatures else 0.0
    
    def get_type(self):
        """Get the type of this segment as a string."""
        if isinstance(self.component, StraightComponentPath):
            return "Straight"
        elif isinstance(self.component, ArcComponentPath):
            return "Arc"
        elif isinstance(self.component, SplineComponentPath):
            return "Spline"
        else:
            return "Unknown"


class PathVelocityPlanner:
    """
    Path velocity planner with segment-based trapezoidal profiles.
    
    Supports three modes:
    - look_ahead=0: Independent trapezoids (robot stops at every boundary)
    - look_ahead=1: Blended motion (continuous transitions, ignores transition curvature)
    - look_ahead=2: Smart blended motion (considers transition curvature at corners)
    """
    
    def __init__(self, hybrid_path, a_max, a_lat_max, v_user_max, look_ahead=1):
        """
        Initialize the velocity planner.
        
        Args:
            hybrid_path: HybridPath instance to plan velocity for
            a_max: Maximum linear acceleration (m/s²)
            a_lat_max: Maximum lateral acceleration (m/s²)
            v_user_max: Maximum user/system speed limit (m/s)
            look_ahead: Look-ahead mode (0=independent, 1=blended naive, 2=blended smart)
        """
        self.hybrid_path = hybrid_path
        self.a_max = float(a_max)
        self.a_lat_max = float(a_lat_max)
        self.v_user_max = float(v_user_max)
        self.look_ahead = int(look_ahead)
        
        if self.look_ahead not in [0, 1, 2]:
            raise ValueError("look_ahead must be 0, 1, or 2")
        
        # Extract segments from hybrid path
        self.segments = []
        for component in hybrid_path.components:
            segment = Segment(component, self.v_user_max)
            self.segments.append(segment)
        
        self.num_segments = len(self.segments)
        
        # Velocity profile data (computed by plan())
        self.boundary_velocities = None  # List of velocities at segment boundaries
        self.profiles = None  # List of profile dictionaries per segment
        
        # Plan the velocity profile
        self.plan()
    
    def compute_caps(self):
        """
        Step 1: Compute velocity caps for each segment.
        
        For each segment:
            v_cap = min(v_user_max, v_max_curvature)
        
        where v_max_curvature = sqrt(a_lat_max / |k_max|)
        """
        for segment in self.segments:
            if segment.k_max == 0:
                # Straight line: no curvature limit
                v_curve = float('inf')
            else:
                # Curvature constraint: v_max = sqrt(a_lat_max / |k|)
                v_curve = np.sqrt(self.a_lat_max / abs(segment.k_max))
            
            segment.v_cap = min(segment.v_user_max, v_curve)
    
    def compute_boundaries(self):
        """
        Step 2: Determine boundary velocities between segments.
        
        Mode 0: All internal boundaries = 0 (stop at every corner)
        Mode 1: Boundary = min(v_cap[i], v_cap[i+1]) - ignores transition curvature
        Mode 2: Boundary = min(v_cap[i], v_cap[i+1], v_transition) - considers transition curvature
        
        Returns:
            List of boundary velocities [v_0, v_1, ..., v_n] where n = num_segments
        """
        n = self.num_segments
        boundaries = [0.0] * (n + 1)
        
        # First boundary: start of path (usually 0, but could be non-zero)
        boundaries[0] = 0.0
        
        # Last boundary: end of path (usually 0, but could be non-zero)
        boundaries[n] = 0.0
        
        # Get transition curvatures from hybrid path (for mode 2)
        # transition_curvatures[i] is the curvature at boundary i+1 (between segment i and i+1)
        transition_curvatures = []
        if self.look_ahead == 2 and hasattr(self.hybrid_path, 'transition_curvatures'):
            transition_curvatures = self.hybrid_path.transition_curvatures
        
        # Intermediate boundaries
        for i in range(n - 1):
            if self.look_ahead == 0:
                # Independent mode: stop at every boundary
                boundaries[i + 1] = 0.0
            elif self.look_ahead == 1:
                # Naive look-ahead mode: blend with next segment (ignores transition curvature)
                boundaries[i + 1] = min(self.segments[i].v_cap, 
                                       self.segments[i + 1].v_cap)
            else:
                # Smart look-ahead mode (mode 2): consider transition curvature
                # Start with segment caps
                v_boundary = min(self.segments[i].v_cap, self.segments[i + 1].v_cap)
                
                # Apply transition curvature constraint: v_transition = sqrt(a_lat_max / |k_transition|)
                if i < len(transition_curvatures):
                    k_transition = abs(transition_curvatures[i])
                    if k_transition > 1e-6:  # Significant curvature at transition
                        v_transition = np.sqrt(self.a_lat_max / k_transition)
                        v_boundary = min(v_boundary, v_transition)
                
                boundaries[i + 1] = v_boundary
        
        return boundaries

    def check_acceleration_feasibility(self, boundaries):
        """
        Step 3: Ensure acceleration feasibility.
        
        Forward check: v[i+1] ≤ sqrt(v[i]^2 + 2 * a_max * L)
        Backward check: v[i] ≤ sqrt(v[i+1]^2 + 2 * a_max * L)
        
        Respects fixed boundaries (start, end, and Mode 0 internal boundaries).
        
        Args:
            boundaries: List of boundary velocities
            
        Returns:
            Adjusted boundaries that satisfy acceleration constraints
        """
        adjusted = boundaries.copy()
        n = self.num_segments
        
        # Mark which boundaries are fixed and should not be changed
        fixed = [False] * (n + 1)
        fixed[0] = True   # Start is always fixed
        fixed[n] = True   # End is always fixed
        
        if self.look_ahead == 0:
            # In Mode 0, all internal boundaries are fixed at 0
            for i in range(1, n):
                fixed[i] = True
        
        # Forward pass: ensure we can accelerate to next boundary
        for i in range(n):
            v_start = adjusted[i]
            v_end = adjusted[i + 1]
            L = self.segments[i].length
            
            if L > 1e-10:
                # Maximum achievable velocity at end of segment
                v_max_achievable = np.sqrt(v_start**2 + 2 * self.a_max * L)
                
                # Only adjust if boundary is not fixed
                if not fixed[i + 1] and v_end > v_max_achievable:
                    adjusted[i + 1] = v_max_achievable
        
        # Backward pass: ensure we can decelerate from previous boundary
        for i in range(n - 1, -1, -1):
            v_start = adjusted[i]
            v_end = adjusted[i + 1]
            L = self.segments[i].length
            
            if L > 1e-10:
                # Maximum achievable velocity at start of segment
                v_max_achievable = np.sqrt(v_end**2 + 2 * self.a_max * L)
                
                # Only adjust if boundary is not fixed
                if not fixed[i] and v_start > v_max_achievable:
                    adjusted[i] = v_max_achievable
        
        print(f"Incoming boundaries: {boundaries}")
        print(f"Adjusted boundaries: {adjusted}")
        return adjusted
    
    def build_segment_profile(self, segment, v_start, v_end):
        """
        Step 4: Build trapezoidal velocity profile for a segment.
        
        ALWAYS returns a trapezoid (never triangle) for stability.
        Uses analytical formula to cap cruise velocity such that
        there's always a non-negative cruise distance.
        
        Args:
            segment: Segment instance
            v_start: Velocity at segment start (m/s)
            v_end: Velocity at segment end (m/s)
            
        Returns:
            Dictionary with profile information:
            {
                'type': 'trapezoid',
                'v_start': float,
                'v_end': float,
                'v_cruise': float,
                't_accel': float,  # Time to accelerate
                't_cruise': float,  # Time at cruise
                't_decel': float,   # Time to decelerate
                's_accel': float,   # Distance during acceleration
                's_cruise': float,  # Distance at cruise
                's_decel': float    # Distance during deceleration
            }
        """
        v_cap = segment.v_cap
        L = segment.length
        
        # Calculate maximum cruise velocity that ensures trapezoid profile
        # Derived from: s_accel + s_decel ≤ L
        # Formula: v_cruise_max = sqrt(a*L + (v_start² + v_end²)/2)
        # This guarantees s_cruise ≥ 0
        if self.a_max > 1e-10 and L > 1e-10:
            v_cruise_max = np.sqrt(self.a_max * L + (v_start**2 + v_end**2) / 2.0)
            v_cruise = min(v_cap, v_cruise_max)
        else:
            # Edge case: no acceleration or zero length
            v_cruise = min(v_start, v_end, v_cap)
        
        # Calculate acceleration phase
        if v_cruise > v_start:
            dv_accel = v_cruise - v_start
            t_accel = dv_accel / self.a_max
            s_accel = v_start * t_accel + 0.5 * self.a_max * t_accel**2
        else:
            t_accel = 0.0
            s_accel = 0.0
        
        # Calculate deceleration phase
        if v_cruise > v_end:
            dv_decel = v_cruise - v_end
            t_decel = dv_decel / self.a_max
            s_decel = v_cruise * t_decel - 0.5 * self.a_max * t_decel**2
        else:
            t_decel = 0.0
            s_decel = 0.0
        
        # Cruise phase (guaranteed to be non-negative)
        s_cruise = L - s_accel - s_decel
        t_cruise = s_cruise / v_cruise if v_cruise > 1e-10 else 0.0
        
        # Numerical safety check (should not be needed with correct formula)
        if s_cruise < -1e-6:
            print(f"WARNING: Negative cruise distance {s_cruise:.6f} m, clamping to 0")
            s_cruise = 0.0
            t_cruise = 0.0
        
        return {
            'type': 'trapezoid',
            'v_start': v_start,
            'v_end': v_end,
            'v_cruise': v_cruise,
            't_accel': t_accel,
            't_cruise': t_cruise,
            't_decel': t_decel,
            's_accel': s_accel,
            's_cruise': s_cruise,
            's_decel': s_decel
        }
    
    def plan(self):
        """
        Execute the complete velocity planning process.
        """
        # Step 1: Compute velocity caps
        self.compute_caps()
        
        # Step 2: Determine boundary velocities
        boundaries = self.compute_boundaries()
        
        # Step 3: Check and adjust for acceleration feasibility
        self.boundary_velocities = self.check_acceleration_feasibility(boundaries)
        
        # Step 4: Build profiles for each segment
        self.profiles = []
        for i, segment in enumerate(self.segments):
            v_start = self.boundary_velocities[i]
            v_end = self.boundary_velocities[i + 1]
            
            profile = self.build_segment_profile(segment, v_start, v_end)
            self.profiles.append(profile)
            
            # Store in segment for convenience
            segment.v_start = v_start
            segment.v_end = v_end
            segment.v_cruise = profile['v_cruise']
    
    def get_velocity_at_arc_length(self, s):
        """
        Get planned velocity at a given arc length along the path.
        
        Args:
            s: Global arc length [0, total_length] (m)
            
        Returns:
            Velocity at arc length s (m/s)
        """
        s = np.clip(s, 0.0, self.hybrid_path.total_length)
        
        # Find which segment this arc length belongs to
        cumulative_s = 0.0
        for i, segment in enumerate(self.segments):
            if s <= cumulative_s + segment.length:
                # This arc length is in segment i
                local_s = s - cumulative_s
                profile = self.profiles[i]
                
                # Get velocity based on profile phase
                if local_s <= profile['s_accel']:
                    # Acceleration phase: v^2 = v_start^2 + 2*a*s
                    if self.a_max > 1e-10:
                        v_squared = profile['v_start']**2 + 2 * self.a_max * local_s
                        v = np.sqrt(max(0.0, v_squared))
                    else:
                        v = profile['v_start']
                elif local_s <= profile['s_accel'] + profile['s_cruise']:
                    # Cruise phase
                    v = profile['v_cruise']
                else:
                    # Deceleration phase: v^2 = v_cruise^2 - 2*a*s
                    s_decel_start = profile['s_accel'] + profile['s_cruise']
                    s_decel_local = local_s - s_decel_start
                    if self.a_max > 1e-10 and s_decel_local > 0:
                        v_cruise = profile['v_cruise']
                        v_squared = v_cruise**2 - 2 * self.a_max * s_decel_local
                        v = np.sqrt(max(profile['v_end']**2, v_squared))
                    else:
                        v = profile['v_end']
                
                return max(0.0, v)
            
            cumulative_s += segment.length
        
        # Should not reach here, but return end velocity
        return self.boundary_velocities[-1]
    
    def get_total_time(self):
        """
        Calculate total time to traverse the path.
        
        Returns:
            Total time (s)
        """
        total_time = 0.0
        for profile in self.profiles:
            total_time += profile['t_accel'] + profile['t_cruise'] + profile['t_decel']
        return total_time
    
    def print_summary(self):
        """Print a summary of the velocity plan."""
        print("\n" + "="*80)
        print("PATH VELOCITY PLANNER SUMMARY")
        print("="*80)
        print(f"Total path length: {self.hybrid_path.total_length:.4f} m")
        print(f"Number of segments: {self.num_segments}")
        mode_names = {
            0: 'Independent (stops at boundaries)',
            1: 'Blended naive (ignores transition curvature)',
            2: 'Blended smart (considers transition curvature)'
        }
        print(f"Look-ahead mode: {mode_names.get(self.look_ahead, 'Unknown')}")
        print(f"Total traversal time: {self.get_total_time():.4f} s")
        print(f"\nDynamic Limits:")
        print(f"  Max linear acceleration: {self.a_max:.2f} m/s²")
        print(f"  Max lateral acceleration: {self.a_lat_max:.2f} m/s²")
        print(f"  Max user speed: {self.v_user_max:.2f} m/s")
        
        print(f"\n{'='*80}")
        print("SEGMENT DETAILS")
        print("="*80)
        
        cumulative_s = 0.0
        for i, (segment, profile) in enumerate(zip(self.segments, self.profiles)):
            print(f"\nSegment {i} ({segment.get_type()}):")
            print(f"  Arc length: {segment.length:.4f} m")
            print(f"  Max curvature: {segment.k_max:.4f} 1/m")
            print(f"  Velocity cap: {segment.v_cap:.4f} m/s")
            print(f"  Profile type: {profile['type']}")
            print(f"  Start velocity: {profile['v_start']:.4f} m/s")
            print(f"  Cruise velocity: {profile['v_cruise']:.4f} m/s")
            print(f"  End velocity: {profile['v_end']:.4f} m/s")
            print(f"  Acceleration phase: {profile['s_accel']:.4f} m ({profile['t_accel']:.4f} s)")
            print(f"  Cruise phase: {profile['s_cruise']:.4f} m ({profile['t_cruise']:.4f} s)")
            print(f"  Deceleration phase: {profile['s_decel']:.4f} m ({profile['t_decel']:.4f} s)")
            print(f"  Cumulative arc length: {cumulative_s:.4f} - {cumulative_s + segment.length:.4f} m")
            
            cumulative_s += segment.length
        
        print("\n" + "="*80)


class PathDirectionProvider:
    """
    Provides velocity direction (vx, vy, omega) at any arc length along a HybridPath.
    
    For different component types:
    - StraightComponentPath: Direction is tangent, omega = 0
    - ArcComponentPath: Direction is tangent, omega = v/r (with sign for direction)
    - SplineComponentPath: Uses point tracking (placeholder)
    
    Combined with PathVelocityPlanner:
    - PathVelocityPlanner provides magnitude (speed)
    - PathDirectionProvider provides direction
    - Full command: velocity = speed * direction
    """
    
    def __init__(self, hybrid_path, goal_orientation=0.0):
        """
        Initialize the direction provider.
        
        Args:
            hybrid_path: HybridPath instance
            goal_orientation: Target orientation at end of path (for orientation control)
        """
        self.hybrid_path = hybrid_path
        self.goal_orientation = goal_orientation
        
        # Spline tracking parameters (for SplineComponentPath)
        self.kp_pos = 0.5  # Position gain
        self.kp_orient = 0.8  # Orientation gain
        self.kd_vel = 0.3  # Velocity damping
    
    def _global_s_to_component(self, s):
        """Convert global arc length to (component_index, local_s)."""
        s = np.clip(s, 0.0, self.hybrid_path.total_length)
        
        for i, component in enumerate(self.hybrid_path.components):
            component_start = self.hybrid_path.cumulative_lengths[i]
            component_end = self.hybrid_path.cumulative_lengths[i + 1]
            
            if s <= component_end or i == self.hybrid_path.num_components - 1:
                local_s = s - component_start
                local_s = np.clip(local_s, 0.0, component.get_path_length())
                return i, local_s
        
        # Fallback to last component
        return self.hybrid_path.num_components - 1, self.hybrid_path.components[-1].get_path_length()
    
    def get_velocity_direction_at_arc_length(self, s, speed):
        """
        Get velocity command (vx, vy, omega) at arc length s with given speed.
        
        For an arc with center (c_x, c_y), radius ρ, and angular velocity ω:
            v_x = -ω · ρ · sin(θ)  = speed · tangent[0]
            v_y =  ω · ρ · cos(θ)  = speed · tangent[1]
        
        The Scaling Property (from Lemma 1):
            The arc shape depends only on the ratio ||v||/|ω|
            If you scale both linear and angular velocity by k:
                - Same radius: ρ = ||v||/|ω| = k·||v||/(k·|ω|)
                - Same arc shape, just traced faster/slower
        
        This means: omega = speed / radius = speed * curvature
        
        Args:
            s: Arc length along the path (m)
            speed: Desired speed (m/s) from PathVelocityPlanner
            
        Returns:
            np.array([vx, vy, omega]) - velocity command in world frame
        """
        component_idx, local_s = self._global_s_to_component(s)
        component = self.hybrid_path.components[component_idx]
        
        # Get tangent direction at this point
        tangent = component.get_tangent_at_arc_length(local_s)
        
        # Linear velocity: speed along tangent
        vx = speed * tangent[0]
        vy = speed * tangent[1]
        
        # Angular velocity depends on component type
        if isinstance(component, StraightComponentPath):
            # Straight line: no rotation needed (κ = 0)
            omega = 0.0
            
        elif isinstance(component, ArcComponentPath):
            # Arc: ω = v / ρ = v · κ
            # From the scaling property: ρ = ||v|| / |ω| ⟹ |ω| = ||v|| / ρ = ||v|| · κ
            # Sign convention: clockwise = negative ω, counterclockwise = positive ω
            curvature = component.curvature  # κ = 1/ρ, always positive
            if component.clockwise:
                omega = -speed * curvature  # Clockwise = negative rotation
            else:
                omega = speed * curvature   # Counterclockwise = positive rotation
                
        elif isinstance(component, SplineComponentPath):
            # Spline: use instantaneous curvature to approximate ω
            # This is a placeholder - more sophisticated spline following
            # would use look-ahead or tracking controllers
            curvature = component.get_curvature_at_arc_length(local_s)
            omega = speed * curvature  # ω = v · κ (sign from curvature)
            
        else:
            # Unknown type: no rotation
            omega = 0.0
        
        return np.array([vx, vy, omega])
    
    def get_velocity_direction_with_tracking(
        self, 
        s, 
        speed,
        current_position,
        current_orientation,
        current_velocity,
        current_angular_velocity,
    ):
        """
        Get velocity command with point tracking correction.
        
        This is useful for SplineComponentPath or when precise tracking is needed.
        Combines feed-forward from path tangent with feedback from position error.
        
        Args:
            s: Arc length along the path (m)
            speed: Desired speed (m/s)
            current_position: Current object position (x, y)
            current_orientation: Current object orientation (rad)
            current_velocity: Current object velocity (vx, vy)
            current_angular_velocity: Current angular velocity (rad/s)
            
        Returns:
            np.array([vx, vy, omega]) - velocity command
        """
        # Get target point on path
        target_point = self.hybrid_path.get_point_at_arc_length(s)
        target_tangent = self.hybrid_path.get_tangent_at_arc_length(s)
        
        # Target orientation: direction of tangent
        target_orientation = np.arctan2(target_tangent[1], target_tangent[0])
        
        # Position error
        position_error = target_point - current_position
        
        # Orientation error (wrap to [-pi, pi])
        orientation_error = target_orientation - current_orientation
        orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))
        
        # Feed-forward: speed along tangent
        vx_ff = speed * target_tangent[0]
        vy_ff = speed * target_tangent[1]
        
        # Feedback: position correction
        vx_fb = self.kp_pos * position_error[0] - self.kd_vel * current_velocity[0]
        vy_fb = self.kp_pos * position_error[1] - self.kd_vel * current_velocity[1]
        
        # Combined linear velocity
        vx = vx_ff + vx_fb
        vy = vy_ff + vy_fb
        
        # Angular velocity: feedback on orientation error
        omega = self.kp_orient * orientation_error - self.kd_vel * current_angular_velocity
        
        return np.array([vx, vy, omega])


class PathFollowingController:
    """
    High-level controller that combines PathVelocityPlanner and PathDirectionProvider.
    
    This controller:
    1. Tracks progress along the path using TIME-BASED integration
    2. Gets velocity magnitude from PathVelocityPlanner
    3. Gets velocity direction from PathDirectionProvider
    4. Outputs complete velocity command (vx, vy, omega)
    
    IMPORTANT: The velocity profile is defined over position (s), but we track 
    elapsed time and compute s from the profile's time equations. This avoids
    the "stuck at zero" problem where v(s=0)=0 leads to s never updating.
    
    Usage:
        controller = PathFollowingController(hybrid_path, a_max, a_lat_max, v_user_max)
        
        # In control loop:
        cmd = controller.compute_velocity(dt=0.01)  # dt in seconds
    """
    
    def __init__(
        self, 
        hybrid_path, 
        a_max, 
        a_lat_max, 
        v_user_max, 
        look_ahead=2,
        use_tracking=False,
    ):
        """
        Initialize the path following controller.
        
        Args:
            hybrid_path: HybridPath to follow
            a_max: Max linear acceleration (m/s²)
            a_lat_max: Max lateral acceleration (m/s²)
            v_user_max: Max user speed (m/s)
            look_ahead: Look-ahead mode (0, 1, or 2)
            use_tracking: Use point tracking for velocity direction
        """
        self.hybrid_path = hybrid_path
        self.use_tracking = use_tracking
        self.a_max = a_max
        
        # Create velocity planner (magnitude)
        self.velocity_planner = PathVelocityPlanner(
            hybrid_path=hybrid_path,
            a_max=a_max,
            a_lat_max=a_lat_max,
            v_user_max=v_user_max,
            look_ahead=look_ahead,
        )
        
        # Create direction provider
        self.direction_provider = PathDirectionProvider(hybrid_path)
        
        # State tracking - TIME-BASED
        self.elapsed_time = 0.0  # Total elapsed time
        self.current_segment_idx = 0  # Current segment index
        self.time_in_segment = 0.0  # Time spent in current segment
        self.current_s = 0.0  # Current arc length along path
        self.completed = False
        
        # Precompute segment start times
        self._compute_segment_times()
    
    def _compute_segment_times(self):
        """Compute cumulative start times for each segment."""
        self.segment_start_times = [0.0]
        self.segment_start_s = [0.0]
        
        cumulative_time = 0.0
        cumulative_s = 0.0
        
        for i, profile in enumerate(self.velocity_planner.profiles):
            segment_time = profile['t_accel'] + profile['t_cruise'] + profile['t_decel']
            segment_s = self.velocity_planner.segments[i].length
            
            cumulative_time += segment_time
            cumulative_s += segment_s
            
            self.segment_start_times.append(cumulative_time)
            self.segment_start_s.append(cumulative_s)
        
        self.total_time = cumulative_time
    
    def reset(self):
        """Reset the controller to start of path."""
        self.elapsed_time = 0.0
        self.current_segment_idx = 0
        self.time_in_segment = 0.0
        self.current_s = 0.0
        self.completed = False
    
    def _time_to_s_in_segment(self, segment_idx, t_local):
        """
        Convert local time within a segment to local arc length.
        
        Uses the trapezoidal profile equations:
        - Accel phase: s = v_start * t + 0.5 * a * t²
        - Cruise phase: s = s_accel + v_cruise * (t - t_accel)
        - Decel phase: s = s_accel + s_cruise + v_cruise * (t - t_accel - t_cruise) - 0.5 * a * (t - t_accel - t_cruise)²
        
        Args:
            segment_idx: Index of the segment
            t_local: Time since entering this segment (s)
            
        Returns:
            Local arc length within the segment (m)
        """
        profile = self.velocity_planner.profiles[segment_idx]
        
        t_accel = profile['t_accel']
        t_cruise = profile['t_cruise']
        t_decel = profile['t_decel']
        v_start = profile['v_start']
        v_cruise = profile['v_cruise']
        s_accel = profile['s_accel']
        s_cruise = profile['s_cruise']
        a = self.a_max
        
        t_local = max(0.0, t_local)
        
        if t_local <= t_accel:
            # Acceleration phase: s = v_start * t + 0.5 * a * t²
            s = v_start * t_local + 0.5 * a * t_local**2
        elif t_local <= t_accel + t_cruise:
            # Cruise phase
            t_in_cruise = t_local - t_accel
            s = s_accel + v_cruise * t_in_cruise
        else:
            # Deceleration phase
            t_in_decel = t_local - t_accel - t_cruise
            t_in_decel = min(t_in_decel, t_decel)  # Clamp
            # s = s_accel + s_cruise + v_cruise * t - 0.5 * a * t²
            s = s_accel + s_cruise + v_cruise * t_in_decel - 0.5 * a * t_in_decel**2
        
        # Clamp to segment length
        segment_length = self.velocity_planner.segments[segment_idx].length
        return min(s, segment_length)
    
    def _time_to_velocity_in_segment(self, segment_idx, t_local):
        """
        Get velocity at local time within a segment.
        
        Args:
            segment_idx: Index of the segment
            t_local: Time since entering this segment (s)
            
        Returns:
            Velocity (m/s)
        """
        profile = self.velocity_planner.profiles[segment_idx]
        
        t_accel = profile['t_accel']
        t_cruise = profile['t_cruise']
        t_decel = profile['t_decel']
        v_start = profile['v_start']
        v_cruise = profile['v_cruise']
        v_end = profile['v_end']
        a = self.a_max
        
        t_local = max(0.0, t_local)
        
        if t_local <= t_accel:
            # Acceleration phase: v = v_start + a * t
            return v_start + a * t_local
        elif t_local <= t_accel + t_cruise:
            # Cruise phase
            return v_cruise
        else:
            # Deceleration phase: v = v_cruise - a * (t - t_accel - t_cruise)
            t_in_decel = t_local - t_accel - t_cruise
            v = v_cruise - a * t_in_decel
            return max(v_end, v)
    
    def _update_state_from_time(self):
        """Update current_s and segment index based on elapsed_time."""
        if self.completed:
            return
        
        # Check if we've exceeded total time
        if self.elapsed_time >= self.total_time:
            self.elapsed_time = self.total_time
            self.current_s = self.hybrid_path.total_length
            self.current_segment_idx = self.velocity_planner.num_segments - 1
            self.completed = True
            return
        
        # Find which segment we're in based on time
        for i in range(self.velocity_planner.num_segments):
            if self.elapsed_time < self.segment_start_times[i + 1]:
                self.current_segment_idx = i
                self.time_in_segment = self.elapsed_time - self.segment_start_times[i]
                
                # Compute s from time within this segment
                local_s = self._time_to_s_in_segment(i, self.time_in_segment)
                self.current_s = self.segment_start_s[i] + local_s
                return
        
        # Fallback: at end of path
        self.current_segment_idx = self.velocity_planner.num_segments - 1
        self.current_s = self.hybrid_path.total_length
        self.completed = True
    
    def compute_velocity(
        self, 
        current_position=None,
        current_orientation=0.0,
        current_velocity=None,
        current_angular_velocity=0.0,
        dt=None,
    ):
        """
        Compute velocity command for path following.
        
        Args:
            current_position: Current object position (x, y) - optional for tracking
            current_orientation: Current object orientation (rad) - optional for tracking
            current_velocity: Current velocity (vx, vy) - optional for tracking
            current_angular_velocity: Current angular velocity (rad/s) - optional for tracking
            dt: Time step for progress update - REQUIRED for time-based progress
            
        Returns:
            np.array([vx, vy, omega]) - velocity command
        """
        if self.completed:
            return np.array([0.0, 0.0, 0.0])
        
        # Update elapsed time and recompute state
        if dt is not None and dt > 0:
            self.elapsed_time += dt
            self._update_state_from_time()
        
        if self.completed:
            return np.array([0.0, 0.0, 0.0])
        
        # Get speed from time-based velocity (more accurate than position-based)
        speed = self._time_to_velocity_in_segment(self.current_segment_idx, self.time_in_segment)
        
        # Get direction from direction provider
        if self.use_tracking and current_position is not None and current_velocity is not None:
            # Use tracking mode
            cmd = self.direction_provider.get_velocity_direction_with_tracking(
                s=self.current_s,
                speed=speed,
                current_position=current_position,
                current_orientation=current_orientation,
                current_velocity=current_velocity,
                current_angular_velocity=current_angular_velocity,
            )
        else:
            # Use simple direction mode
            cmd = self.direction_provider.get_velocity_direction_at_arc_length(
                s=self.current_s,
                speed=speed,
            )
        
        return cmd
    
    def get_current_s(self):
        """Get current arc length progress."""
        return self.current_s
    
    def get_progress_fraction(self):
        """Get progress as fraction of total path length."""
        return self.current_s / self.hybrid_path.total_length
    
    def is_completed(self):
        """Check if path following is completed."""
        return self.completed
    
    def get_target_point(self):
        """Get the target point on path at current s."""
        return self.hybrid_path.get_point_at_arc_length(self.current_s)
    
    def get_target_tangent(self):
        """Get the target tangent direction at current s."""
        return self.hybrid_path.get_tangent_at_arc_length(self.current_s)
    
    def get_elapsed_time(self):
        """Get total elapsed time."""
        return self.elapsed_time
    
    def get_estimated_remaining_time(self):
        """Get estimated remaining time to complete the path."""
        return max(0.0, self.total_time - self.elapsed_time)


def demo_velocity_planning():
    """
    Demo function that creates three hybrid paths and plans velocities for them.
    """
    # Import here to avoid matplotlib issues when just importing the module
    try:
        from paths_lib import (
            demo_three_hybrid_paths,
            create_rectangle_hybrid_path,
            create_p_trajectory_hybrid_path,
            create_catenary_hybrid_path
        )
    except ImportError as e:
        print(f"Warning: Could not import paths_lib: {e}")
        print("This is expected if matplotlib is not available in the environment.")
        return
    
    # Create the three demo paths
    print("Creating hybrid paths...")
    rectangle_path, p_path, catenary_path = demo_three_hybrid_paths()
    
    # Planning parameters
    a_max = 2.0  # m/s²
    a_lat_max = 1.5  # m/s²
    v_user_max = 3.0  # m/s
    
    # Test both look-ahead modes
    for look_ahead in [0, 1]:
        print(f"\n{'='*80}")
        print(f"VELOCITY PLANNING - Look-ahead mode: {look_ahead}")
        print(f"{'='*80}")
        
        paths = [
            ("Rectangle", rectangle_path),
            ("P Trajectory", p_path),
            ("Catenary", catenary_path)
        ]
        
        for name, path in paths:
            print(f"\n{'-'*80}")
            print(f"Planning for: {name}")
            print(f"{'-'*80}")
            
            planner = PathVelocityPlanner(
                hybrid_path=path,
                a_max=a_max,
                a_lat_max=a_lat_max,
                v_user_max=v_user_max,
                look_ahead=look_ahead
            )
            
            planner.print_summary()


if __name__ == "__main__":
    demo_velocity_planning()
