"""Reconfiguration Planners for Different Navigation Schemes.

Each navigation scheme has its own reconfiguration planner that optimizes
the transition to new target positions based on the scheme's characteristics.

Author: Contact Maintain Team
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
from itertools import permutations


class ReconfigurationPlanner(ABC):
    """Base class for reconfiguration planners.
    
    Each navigation scheme implements its own planner to optimize
    the transition to new target positions.
    """
    
    @abstractmethod
    def plan(
        self,
        new_t_params: Dict[str, float],
        monitors: Dict[str, Any],  # DistributedMonitor instances
        generic_object: Any,
    ) -> Dict[str, float]:
        """Plan reconfiguration to new target positions.
        
        Parameters
        ----------
        new_t_params : Dict[str, float]
            New target t_params for each robot
        monitors : Dict[str, DistributedMonitor]
            Current monitor instances
        generic_object : GenericObject
            Object model for boundary parameterization
            
        Returns
        -------
        Dict[str, float]
            Optimized assignment of new_t_params to robots
            (may be different from input if optimization is performed)
        """
        pass


class APFReconfigurationPlanner(ReconfigurationPlanner):
    """APF Reconfiguration Planner.
    
    All robots move simultaneously to new positions.
    No optimization - direct assignment.
    """
    
    def plan(
        self,
        new_t_params: Dict[str, float],
        monitors: Dict[str, Any],
        generic_object: Any,
    ) -> Dict[str, float]:
        """Plan APF reconfiguration - direct assignment."""
        # APF: all robots move simultaneously, no optimization needed
        return new_t_params.copy()


class StaticSingleReconfigurationPlanner(ReconfigurationPlanner):
    """Static Single Reconfiguration Planner.
    
    Computes optimal assignment (robot → target) using permutation check
    to minimize total travel distance. Robots will move sequentially.
    """
    
    def plan(
        self,
        new_t_params: Dict[str, float],
        monitors: Dict[str, Any],
        generic_object: Any,
    ) -> Dict[str, float]:
        """Plan static single reconfiguration - optimal assignment."""
        robot_names = list(monitors.keys())
        target_t_params = list(new_t_params.values())
        
        if len(robot_names) != len(target_t_params):
            # Mismatch - return direct assignment
            return new_t_params.copy()
        
        # Get current positions
        current_positions = {}
        object_position = np.zeros(2)  # Will be updated from monitor
        object_orientation = 0.0
        
        for name, monitor in monitors.items():
            current_positions[name] = monitor.local_state.position
            # Get object state from monitor (approximate)
            # In practice, this should be passed in
        
        # Compute distance matrix: robot -> target
        # For simplicity, use t_param arc distance as proxy
        # In practice, should compute actual travel distance
        from object_utils import ContactPointParameterization
        param = ContactPointParameterization(generic_object)
        
        current_t_params = {}
        for name, monitor in monitors.items():
            current_t_params[name] = monitor.local_state.current_t_param
        
        # Try all permutations to find minimum total distance
        min_total_distance = float('inf')
        best_assignment = None
        
        for perm in permutations(range(len(robot_names))):
            total_distance = 0.0
            assignment = {}
            
            for i, robot_idx in enumerate(perm):
                robot_name = robot_names[robot_idx]
                target_t = target_t_params[i]
                current_t = current_t_params.get(robot_name, 0.0)
                
                # Compute arc distance (t_param difference)
                raw_diff = target_t - current_t
                if raw_diff >= 0:
                    arc_dist = min(raw_diff, 1.0 - raw_diff)
                else:
                    arc_dist = min(-raw_diff, 1.0 + raw_diff)
                
                total_distance += arc_dist
                assignment[robot_name] = target_t
            
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_assignment = assignment
        
        return best_assignment if best_assignment is not None else new_t_params.copy()


class DivideConquerReconfigurationPlanner(ReconfigurationPlanner):
    """Divide-n-Conquer Reconfiguration Planner.
    
    Reassigns edge segments to robots based on new target positions.
    May need prep phase if edge assignments change significantly.
    """
    
    def plan(
        self,
        new_t_params: Dict[str, float],
        monitors: Dict[str, Any],
        generic_object: Any,
    ) -> Dict[str, float]:
        """Plan divide-n-conquer reconfiguration - reassign edges."""
        robot_names = list(monitors.keys())
        target_t_params = list(new_t_params.values())
        
        if len(robot_names) != len(target_t_params):
            return new_t_params.copy()
        
        # Sort targets by t_param to assign consecutive edges
        sorted_targets = sorted(enumerate(target_t_params), key=lambda x: x[1])
        num_robots = len(robot_names)
        segment_size = 1.0 / num_robots
        
        # Create assignment with edge segments
        assignment = {}
        edge_assignments = {}
        
        for i, (orig_idx, target_t) in enumerate(sorted_targets):
            robot_name = robot_names[orig_idx]
            assignment[robot_name] = target_t
            
            # Assign edge segment
            t_start = i * segment_size
            t_end = ((i + 1) * segment_size) % 1.0
            
            # Ensure target is within segment (with margin)
            if t_start <= t_end:
                if not (t_start <= target_t <= t_end):
                    # Expand segment to include target
                    if target_t < t_start:
                        t_start = target_t
                    elif target_t > t_end:
                        t_end = target_t
            else:
                # Wrap-around case
                if not ((target_t >= t_start) or (target_t <= t_end)):
                    # Expand segment
                    if target_t > t_end and target_t < t_start:
                        t_start = target_t
                        t_end = target_t
            
            edge_assignments[robot_name] = (t_start, t_end)
        
        # Update edge assignments in navigation controllers
        for name, monitor in monitors.items():
            if hasattr(monitor.navigation_controller, 'set_edge_assignment'):
                t_start, t_end = edge_assignments.get(name, (0.0, 1.0))
                monitor.navigation_controller.set_edge_assignment(name, t_start, t_end)
        
        return assignment


# Factory function for creating reconfiguration planners
def create_reconfiguration_planner(
    navigation_scheme: str,
) -> ReconfigurationPlanner:
    """Create a reconfiguration planner based on navigation scheme.
    
    Parameters
    ----------
    navigation_scheme : str
        Navigation scheme: 'apf', 'static_single', or 'divide_conquer'
        
    Returns
    -------
    ReconfigurationPlanner
        Reconfiguration planner instance
    """
    if navigation_scheme == 'apf':
        return APFReconfigurationPlanner()
    elif navigation_scheme == 'static_single':
        return StaticSingleReconfigurationPlanner()
    elif navigation_scheme == 'divide_conquer':
        return DivideConquerReconfigurationPlanner()
    else:
        raise ValueError(f"Unknown navigation scheme: {navigation_scheme}")
