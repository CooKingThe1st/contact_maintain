"""ORCA (Optimal Reciprocal Collision Avoidance) Navigation Module.

This module provides ORCA-based navigation using the RVO2 library for
decentralized multi-robot collision avoidance.

Author: Contact Maintain Team
"""
import numpy as np
from typing import List, Optional, Tuple
import rvo2


class ORCANavigator:
    """ORCA-based navigation with obstacle avoidance.
    
    Uses RVO2 library for optimal reciprocal collision avoidance between
    robots and static obstacles.
    
    Parameters
    ----------
    time_step : float
        Simulation time step (seconds).
    neighbor_dist : float
        Maximum distance to consider neighbors (meters).
    max_neighbors : int
        Maximum number of neighbors to consider.
    time_horizon : float
        Time horizon for collision avoidance (seconds).
    time_horizon_obst : float
        Time horizon for obstacle avoidance (seconds).
    radius : float
        Robot radius (meters).
    max_speed : float
        Maximum speed (m/s).
    """
    
    def __init__(
        self,
        time_step: float = 1.0 / 60.0,
        neighbor_dist: float = 2.0,
        max_neighbors: int = 10,
        time_horizon: float = 2.0,
        time_horizon_obst: float = 0.5,
        radius: float = 0.06,
        max_speed: float = 0.5,
    ):
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.radius = radius
        self.max_speed = max_speed
        
        # RVO2 simulator (created per agent group)
        self.sim = None
        self.agent_id = None
        # Store the original agent order for consistent indexing
        self.agent_order = None
    
    def initialize_simulator(
        self,
        agent_positions: List[np.ndarray],
        obstacles: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> int:
        """Initialize RVO2 simulator with agents and obstacles.
        
        Parameters
        ----------
        agent_positions : list
            List of agent positions as (x, y) arrays.
        obstacles : list, optional
            List of obstacles, each as list of (x, y) vertices.
        
        Returns
        -------
        int
            Agent ID in simulator (for this navigator's robot).
        """
        # Create simulator
        self.sim = rvo2.PyRVOSimulator(
            self.time_step,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.radius,
            self.max_speed,
        )
        
        # Add obstacles
        if obstacles:
            for obstacle in obstacles:
                # Convert to list of tuples
                obstacle_verts = [(float(v[0]), float(v[1])) for v in obstacle]
                self.sim.addObstacle(obstacle_verts)
            self.sim.processObstacles()
        
        # Add agents (all robots)
        agent_ids = []
        for pos in agent_positions:
            agent_id = self.sim.addAgent((float(pos[0]), float(pos[1])))
            agent_ids.append(agent_id)
        
        # Store agent order (positions as tuples for hashing)
        self.agent_order = [tuple(pos) for pos in agent_positions]
        
        # Return first agent ID (this navigator's robot)
        self.agent_id = agent_ids[0] if agent_ids else None
        return self.agent_id
    
    def compute_velocity(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray,
        other_robot_positions: List[np.ndarray],
        obstacles: Optional[List[List[Tuple[float, float]]]] = None,
        other_robot_targets: Optional[List[np.ndarray]] = None,
        all_robot_positions: Optional[List[np.ndarray]] = None,
        all_robot_targets: Optional[List[np.ndarray]] = None,
        current_robot_index: Optional[int] = None,
    ) -> np.ndarray:
        """Compute velocity using ORCA.
        
        Parameters
        ----------
        current_position : np.ndarray
            Current robot position (x, y).
        target_position : np.ndarray
            Target position (x, y).
        other_robot_positions : list
            Positions of other robots.
        obstacles : list, optional
            Static obstacles as list of vertex lists.
        other_robot_targets : list, optional
            Target positions for other robots. If None, other robots
            will have zero preferred velocity (stay in place).
        all_robot_positions : list, optional
            All robot positions in consistent order (for proper agent indexing).
            If provided, this will be used instead of [current] + other_positions.
        all_robot_targets : list, optional
            All robot targets in same order as all_robot_positions.
        current_robot_index : int, optional
            Index of current robot in all_robot_positions/all_robot_targets.
        
        Returns
        -------
        np.ndarray
            Velocity command (vx, vy).
        """
        # Use all_robot_positions if provided (for consistent ordering)
        if all_robot_positions is not None and current_robot_index is not None:
            all_positions = all_robot_positions
            self.agent_id = current_robot_index
        else:
            # Fallback to old behavior
            all_positions = [current_position] + other_robot_positions
            current_robot_index = 0
        
        if self.sim is None:
            self.initialize_simulator(all_positions, obstacles)
            # Update agent_id if we used all_robot_positions
            if current_robot_index is not None:
                self.agent_id = current_robot_index
        else:
            # Check if number of agents changed
            num_agents_needed = len(all_positions)
            num_agents_current = self.sim.getNumAgents()
            
            if num_agents_needed != num_agents_current:
                # Reinitialize
                self.initialize_simulator(all_positions, obstacles)
                if current_robot_index is not None:
                    self.agent_id = current_robot_index
            else:
                # Update agent positions
                for i, pos in enumerate(all_positions):
                    self.sim.setAgentPosition(i, (float(pos[0]), float(pos[1])))
        
        # Set preferred velocities for all agents (matching dummy_test_rvo2.py pattern)
        # In dummy test: normalize direction vector, RVO2 handles speed clamping
        if self.agent_id is not None and self.sim is not None:
            # Set preferred velocity for this agent (toward target)
            v_pref = target_position - current_position
            if np.linalg.norm(v_pref) > 1e-3:
                v_pref = v_pref / np.linalg.norm(v_pref)
            else:
                v_pref = np.zeros(2)
            
            self.sim.setAgentPrefVelocity(
                self.agent_id,
                (float(v_pref[0]), float(v_pref[1]))
            )
            
            # Set preferred velocities for other agents
            if all_robot_targets is not None and len(all_robot_targets) == len(all_positions):
                # Use all_robot_targets (consistent ordering)
                for agent_idx in range(self.sim.getNumAgents()):
                    if agent_idx != self.agent_id:
                        other_pos = all_positions[agent_idx]
                        other_target = all_robot_targets[agent_idx]
                        other_v_pref = other_target - other_pos
                        if np.linalg.norm(other_v_pref) > 1e-3:
                            other_v_pref = other_v_pref / np.linalg.norm(other_v_pref)
                        else:
                            other_v_pref = np.zeros(2)
                        self.sim.setAgentPrefVelocity(
                            agent_idx,
                            (float(other_v_pref[0]), float(other_v_pref[1]))
                        )
            elif other_robot_targets is not None and len(other_robot_targets) == len(other_robot_positions):
                # Fallback: use other_robot_targets (need to map indices)
                # Build mapping: find which agent index corresponds to each other robot
                for i, (other_pos, other_target) in enumerate(zip(other_robot_positions, other_robot_targets)):
                    # Find this position in all_positions
                    other_pos_tuple = tuple(other_pos)
                    try:
                        agent_idx = all_positions.index(other_pos) if hasattr(all_positions[0], '__iter__') else None
                        # Try to find by matching
                        for idx, pos in enumerate(all_positions):
                            if idx != self.agent_id and np.allclose(pos, other_pos, atol=1e-3):
                                agent_idx = idx
                                break
                        if agent_idx is not None and agent_idx < self.sim.getNumAgents():
                            other_v_pref = other_target - other_pos
                            if np.linalg.norm(other_v_pref) > 1e-3:
                                other_v_pref = other_v_pref / np.linalg.norm(other_v_pref)
                            else:
                                other_v_pref = np.zeros(2)
                            self.sim.setAgentPrefVelocity(
                                agent_idx,
                                (float(other_v_pref[0]), float(other_v_pref[1]))
                            )
                    except (ValueError, AttributeError):
                        pass
            else:
                # Default: other agents want to stay in place
                for i in range(self.sim.getNumAgents()):
                    if i != self.agent_id:
                        self.sim.setAgentPrefVelocity(i, (0.0, 0.0))
            
            # Store for return value
            pref_vel = v_pref
        else:
            # Fallback if simulator not initialized
            v_pref = target_position - current_position
            if np.linalg.norm(v_pref) > 1e-3:
                pref_vel = v_pref / np.linalg.norm(v_pref)
            else:
                pref_vel = np.zeros(2)
        
        # Step simulation
        if self.sim is not None:
            self.sim.doStep()
            
            # Get computed velocity
            if self.agent_id is not None:
                vel = self.sim.getAgentVelocity(self.agent_id)
                return np.array([vel[0], vel[1]])
        
        return pref_vel
    
    def reset(self):
        """Reset the simulator."""
        self.sim = None
        self.agent_id = None
        self.agent_order = None

