#!/usr/bin/env python3
"""
APF Navigation Test (Distributed)

Tests Artificial Potential Field with:
- Hierarchical Waiting
- Tangential Sliding (wall following)
- Local sensing (IR simulation)

Usage:
    python test_apf_navigation.py --test formation
    python test_apf_navigation.py --test obstacles
    python test_apf_navigation.py --test all
"""
import argparse
import sys
import time
from pathlib import Path
from enum import Enum
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

import pybullet as pyb
import pybullet_data

import rospkg
rospack = rospkg.RosPack()
pkg_path = rospack.get_path("contact_maintain")
sys.path.insert(0, str(Path(pkg_path) / "src"))

from contact_maintain.robot_factory import create_robot


TIMESTEP = 1.0 / 240.0
CTRL_FREQ = 60
CTRL_STEP = int((1.0 / CTRL_FREQ) / TIMESTEP)

# Robot parameters
ROBOT_RADIUS = 0.06
SAFETY_RADIUS = ROBOT_RADIUS + 0.15
MAX_SPEED = 0.3
SENSING_RADIUS = 1.0  # IR sensing range
COLLISION_HORIZON = 2.0  # seconds

# APF parameters
K_GOAL = 1.0
K_OBSTACLE = 0.5
K_WALL = 0.3
REPULSION_DISTANCE = 0.5

# State machine parameters
BASE_WAIT_TIME = 0.5  # seconds
SCALE_WAIT_BASE = 0.1
GAMMA = 0.2  # conflict counter scaling

# Open-space bias
NUM_DIRECTIONS = 16  # number of directions to sample



class RobotState(Enum):
    NORMAL = "NORMAL"
    WAITING = "WAITING"
    YIELDING = "YIELDING"


class APFRobot:
    """Robot with APF controller and state machine"""
    
    def __init__(self, robot, robot_id: int, start_pos: np.ndarray, goal_pos: np.ndarray):
        self.robot = robot
        self.robot_id = robot_id
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        self.state = RobotState.NORMAL
        self.wait_timer = 0.0
        self.conflict_counter = 0
        
        # For open-space bias
        self.directions = np.array([
            [np.cos(angle), np.sin(angle)]
            for angle in np.linspace(0, 2 * np.pi, NUM_DIRECTIONS, endpoint=False)
        ])
    
    def get_position(self) -> np.ndarray:
        """Get current 2D position"""
        pos, _, _ = self.robot.get_state()
        return pos[:2]
    
    def get_velocity(self) -> np.ndarray:
        """Get current 2D velocity"""
        _, _, vel = self.robot.get_state()
        return vel[:2]
    
    def sense_neighbors(
        self,
        all_robots: List['APFRobot'],
        obstacles: List[List[Tuple[float, float]]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate IR sensing - returns list of (relative_position, relative_velocity)
        for neighbors within sensing radius.
        """
        neighbors = []
        my_pos = self.get_position()
        my_vel = self.get_velocity()
        
        for other in all_robots:
            if other.robot_id == self.robot_id:
                continue
            
            other_pos = other.get_position()
            other_vel = other.get_velocity()
            
            rel_pos = other_pos - my_pos
            rel_vel = other_vel - my_vel
            dist = np.linalg.norm(rel_pos)
            
            if dist < SENSING_RADIUS:
                neighbors.append((rel_pos, rel_vel))
        
        return neighbors
    
    def predict_collision(
        self,
        neighbors: List[Tuple[np.ndarray, np.ndarray]],
        obstacles: List[List[Tuple[float, float]]],
        intended_vel: Optional[np.ndarray] = None
    ) -> bool:
        """
        Predict collision within horizon. Allows parallel movement.
        Only predicts head-on collisions, not parallel or perpendicular movement.
        
        Args:
            intended_vel: If provided, use this velocity instead of current velocity
                         (useful for checking if a planned velocity would cause collision)
        """
        my_pos = self.get_position()
        my_vel = intended_vel if intended_vel is not None else self.get_velocity()
        vel_mag = np.linalg.norm(my_vel)
        
        # Check robot-robot collisions - only if moving toward each other
        for rel_pos, rel_vel_base in neighbors:
            rel_dist = np.linalg.norm(rel_pos)
            
            # If already too close, predict collision
            if rel_dist < 2 * SAFETY_RADIUS:
                return True
            
            # Compute relative velocity using intended velocity if provided
            # rel_vel_base is relative velocity from other robot's perspective
            # We need: rel_vel = my_vel - other_vel = -rel_vel_base + my_vel
            # Actually, rel_vel_base = other_vel - my_current_vel
            # So: other_vel = rel_vel_base + my_current_vel
            # If using intended_vel: rel_vel = intended_vel - other_vel = intended_vel - (rel_vel_base + my_current_vel)
            # But we don't have my_current_vel here. Let's use rel_vel_base as approximation
            # Or better: compute rel_vel = my_vel - (rel_vel_base + my_current_vel) + my_current_vel
            # Actually simpler: if intended_vel is provided, use it; otherwise rel_vel_base is already relative to current vel
            # For now, use rel_vel_base as is (it's relative to current velocity)
            # If intended_vel differs significantly, we should recompute, but that requires other robot's velocity
            # For simplicity, assume other robot's velocity doesn't change much
            rel_vel = rel_vel_base
            if intended_vel is not None:
                # Approximate: adjust relative velocity by difference in my velocity
                my_current_vel = self.get_velocity()
                vel_diff = intended_vel - my_current_vel
                rel_vel = rel_vel_base - vel_diff
            
            # Check if robots are moving toward each other (head-on collision)
            rel_vel_mag = np.linalg.norm(rel_vel)
            if rel_vel_mag > 0.01 and rel_dist < SENSING_RADIUS:
                # Normalize relative position
                rel_pos_norm = rel_pos / (rel_dist + 1e-6)
                # Check if relative velocity points toward each other
                rel_vel_dir = rel_vel / (rel_vel_mag + 1e-6)
                
                # If relative velocity is opposite to relative position, they're approaching
                if np.dot(rel_vel_dir, -rel_pos_norm) > 0.3:  # Threshold for "moving toward"
                    # Predict future collision
                    for t in np.linspace(0.1, COLLISION_HORIZON, 10):
                        future_rel_pos = rel_pos + t * rel_vel
                        future_dist = np.linalg.norm(future_rel_pos)
                        if future_dist < 2 * SAFETY_RADIUS:
                            return True
        
        # Check obstacle collisions - only if moving directly into obstacle
        for obstacle in obstacles:
            for i in range(len(obstacle)):
                p1 = np.array(obstacle[i])
                p2 = np.array(obstacle[(i + 1) % len(obstacle)])
                
                seg_vec = p2 - p1
                seg_len = np.linalg.norm(seg_vec)
                if seg_len < 1e-6:
                    continue
                
                # Find closest point on segment
                to_robot = my_pos - p1
                proj = np.dot(to_robot, seg_vec) / (seg_len ** 2)
                proj = np.clip(proj, 0, 1)
                closest = p1 + proj * seg_vec
                dist = np.linalg.norm(my_pos - closest)
                
                # If already very close, predict collision
                if dist < SAFETY_RADIUS:
                    return True
                
                # Only predict collision if moving directly toward obstacle
                if vel_mag > 0.01:
                    # Vector from robot to closest point on obstacle
                    to_obstacle = closest - my_pos
                    to_obstacle_dist = np.linalg.norm(to_obstacle)
                    
                    if to_obstacle_dist < REPULSION_DISTANCE:
                        # Check if velocity is pointing toward obstacle
                        vel_dir = my_vel / vel_mag
                        to_obstacle_dir = to_obstacle / (to_obstacle_dist + 1e-6)
                        
                        # Only predict collision if moving directly toward it (not parallel)
                        if np.dot(vel_dir, to_obstacle_dir) > 0.5:  # Threshold for "moving toward"
                            # Predict future collision
                            for t in np.linspace(0.1, min(COLLISION_HORIZON, to_obstacle_dist / vel_mag), 10):
                                future_pos = my_pos + t * my_vel
                                to_robot_future = future_pos - p1
                                proj_future = np.dot(to_robot_future, seg_vec) / (seg_len ** 2)
                                proj_future = np.clip(proj_future, 0, 1)
                                closest_future = p1 + proj_future * seg_vec
                                dist_future = np.linalg.norm(future_pos - closest_future)
                                
                                if dist_future < SAFETY_RADIUS:
                                    return True
        
        return False

    def apf_control(
        self,
        neighbors: List[Tuple[np.ndarray, np.ndarray]],
        obstacles: List[List[Tuple[float, float]]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        my_pos = self.get_position()
        
        # 1. Goal attraction
        to_goal = self.goal_pos - my_pos
        dist_goal = np.linalg.norm(to_goal)
        F_goal = K_GOAL * to_goal / (dist_goal + 1e-6)

        F_obs = np.zeros(2) # Neighbor repulsion
        F_wall = np.zeros(2) # Wall repulsion + Tangential Slide
        
        # 2. Neighbor Repulsion (Standard APF)
        for rel_pos, _ in neighbors:
            dist = np.linalg.norm(rel_pos)
            if dist < REPULSION_DISTANCE and dist > 1e-3:
                F_obs += K_OBSTACLE * (1.0/dist - 1.0/REPULSION_DISTANCE) * (-rel_pos / dist)

        # 3. Wall Repulsion & Tangential Sliding
        for obstacle in obstacles:
            for i in range(len(obstacle)):
                p1, p2 = np.array(obstacle[i]), np.array(obstacle[(i + 1) % len(obstacle)])
                seg_vec = p2 - p1
                seg_len = np.linalg.norm(seg_vec)
                if seg_len < 1e-6: continue
                
                proj = np.clip(np.dot(my_pos - p1, seg_vec) / (seg_len ** 2), 0, 1)
                closest = p1 + proj * seg_vec
                dist = np.linalg.norm(my_pos - closest)
                
                if dist < REPULSION_DISTANCE:
                    # Normal vector (pointing away from wall)
                    n = (my_pos - closest) / (dist + 1e-6)
                    
                    # REPEL: Standard repulsion gain (increased for walls)
                    repel_gain = 2.0 * K_WALL * (1.0/dist - 1.0/REPULSION_DISTANCE)
                    F_wall += repel_gain * n
                    
                    # TANGENTIAL KICK: Check if goal is "behind" the wall
                    # If dot product of F_goal and n is negative, the wall is in the way
                    if np.dot(F_goal, n) < 0.2: 
                        # Create consistent tangent: rotate normal 90 degrees clockwise
                        # T = [n_y, -n_x]
                        tangent = np.array([n[1], -n[0]])
                        
                        # Consistent direction: if we want to be even smarter, 
                        # we pick the tangent that aligns better with the goal
                        # if np.dot(tangent, F_goal) < 0:
                        #     tangent = -tangent
                        
                        # Apply sliding force proportional to how much the goal is blocked
                        slide_strength = K_GOAL * 1.5 
                        F_wall += slide_strength * tangent

        F_total = F_goal + F_obs + F_wall
        return F_total, F_goal, F_obs, F_wall
    
    def get_wall_tangent(
        self,
        obstacles: List[List[Tuple[float, float]]]
    ) -> Optional[np.ndarray]:
        """
        Get wall tangent direction if robot is near a wall.
        Returns None if not near any wall.
        """
        my_pos = self.get_position()
        
        # Compute goal direction
        to_goal = self.goal_pos - my_pos
        dist_goal = np.linalg.norm(to_goal)
        F_goal = K_GOAL * to_goal / (dist_goal + 1e-6)
        
        for obstacle in obstacles:
            for i in range(len(obstacle)):
                p1, p2 = np.array(obstacle[i]), np.array(obstacle[(i + 1) % len(obstacle)])
                seg_vec = p2 - p1
                seg_len = np.linalg.norm(seg_vec)
                if seg_len < 1e-6:
                    continue
                
                proj = np.clip(np.dot(my_pos - p1, seg_vec) / (seg_len ** 2), 0, 1)
                closest = p1 + proj * seg_vec
                dist = np.linalg.norm(my_pos - closest)
                
                if dist < REPULSION_DISTANCE:
                    # Normal vector (pointing away from wall)
                    n = (my_pos - closest) / (dist + 1e-6)
                    
                    # Check if goal is "behind" the wall
                    if np.dot(F_goal, n) < 0.2:
                        # Create consistent tangent: rotate normal 90 degrees clockwise
                        tangent = np.array([n[1], -n[0]])
                        return tangent
        
        return None
    
    def open_space_bias(
        self,
        neighbors: List[Tuple[np.ndarray, np.ndarray]],
        obstacles: List[List[Tuple[float, float]]]
    ) -> np.ndarray:
        """
        Compute open-space bias direction.
        Returns direction with maximum clearance.
        Prioritizes wall tangent direction if available.
        """
        my_pos = self.get_position()
        
        # First, check if there's a wall tangent direction (higher priority)
        wall_tangent = self.get_wall_tangent(obstacles)
        if wall_tangent is not None:
            # Use wall tangent as preferred direction
            # Normalize it
            wall_tangent_norm = wall_tangent / (np.linalg.norm(wall_tangent) + 1e-6)
            return wall_tangent_norm
        
        # Otherwise, use standard open-space bias
        best_dir = None
        best_score = -1.0
        
        for direction in self.directions:
            # Cast ray in this direction
            max_dist = SENSING_RADIUS
            min_dist = max_dist
            
            # Check neighbors
            for rel_pos, _ in neighbors:
                # Project neighbor onto direction
                proj = np.dot(rel_pos, direction)
                if proj > 0:  # In front
                    perp_dist = np.linalg.norm(rel_pos - proj * direction)
                    if perp_dist < 2 * SAFETY_RADIUS:
                        min_dist = min(min_dist, proj)
            
            # Check obstacles - find minimum distance to obstacle in this direction
            for obstacle in obstacles:
                for i in range(len(obstacle)):
                    p1 = np.array(obstacle[i])
                    p2 = np.array(obstacle[(i + 1) % len(obstacle)])
                    
                    seg_vec = p2 - p1
                    seg_len = np.linalg.norm(seg_vec)
                    if seg_len < 1e-6:
                        continue
                    
                    # Find closest point on segment to robot
                    to_p1 = my_pos - p1
                    proj = np.dot(to_p1, seg_vec) / (seg_len ** 2)
                    proj = np.clip(proj, 0, 1)
                    closest_point = p1 + proj * seg_vec
                    
                    # Vector from robot to closest point
                    to_closest = closest_point - my_pos
                    dist_to_segment = np.linalg.norm(to_closest)
                    
                    # Project onto direction
                    proj_on_dir = np.dot(to_closest, direction)
                    
                    if proj_on_dir > 0:  # Obstacle is in front
                        # Check if direction passes close to obstacle
                        perp_dist = np.linalg.norm(to_closest - proj_on_dir * direction)
                        if perp_dist < SAFETY_RADIUS + 0.1:
                            # Ray would hit obstacle at this distance
                            min_dist = min(min_dist, proj_on_dir)
            
            if min_dist > best_score:
                best_score = min_dist
                best_dir = direction
        
        if best_dir is None:
            return np.array([1.0, 0.0])  # Default forward
        
        return best_dir
    
    def compute_control(
        self,
        all_robots: List['APFRobot'],
        obstacles: List[List[Tuple[float, float]]],
        dt: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Main control loop with state machine.
        Returns (desired_velocity, force_info_dict) for analysis.
        force_info contains: F_total, F_goal, F_obs, F_wall, state
        """
        neighbors = self.sense_neighbors(all_robots, obstacles)
        collision_predicted = self.predict_collision(neighbors, obstacles)
        
        force_info = {
            'F_total': np.zeros(2),
            'F_goal': np.zeros(2),
            'F_obs': np.zeros(2),
            'F_wall': np.zeros(2),
            'F_open_space': np.zeros(2),
            'state': self.state.value,
            'collision_predicted': collision_predicted
        }
        
        if self.state == RobotState.NORMAL:
            if collision_predicted:
                self.state = RobotState.WAITING
                scale_wait = SCALE_WAIT_BASE * (1 + GAMMA * self.conflict_counter)
                self.wait_timer = BASE_WAIT_TIME + scale_wait * self.robot_id
                self.conflict_counter += 1
                force_info['state'] = self.state.value
                return np.zeros(2), force_info
            else:
                # Normal APF control
                F_total, F_goal, F_obs, F_wall = self.apf_control(neighbors, obstacles)
                force_info.update({
                    'F_total': F_total,
                    'F_goal': F_goal,
                    'F_obs': F_obs,
                    'F_wall': F_wall,
                    'F_open_space': np.zeros(2)
                })
                vel = np.clip(F_total, -MAX_SPEED, MAX_SPEED)
                return vel, force_info
        
        elif self.state == RobotState.WAITING:
            self.wait_timer -= dt
            # print(f"wait_timer: {self.wait_timer} for robot_id: {self.robot_id}")
            if not collision_predicted or self.wait_timer <= 0:
                self.state = RobotState.YIELDING
                force_info['state'] = self.state.value
            
            return np.zeros(2), force_info
        
        elif self.state == RobotState.YIELDING:


            # APF control (tangential sliding already in F_wall)
            F_total_base, F_goal, F_obs, F_wall = self.apf_control(neighbors, obstacles)
            
            # Add open-space bias (prioritizes wall tangent if available)
            bias_dir = self.open_space_bias(neighbors, obstacles)
            epsilon = 0.3  # Open-space bias strength
            F_open_space = epsilon * MAX_SPEED * bias_dir
            F_total = F_total_base + F_open_space
            
            force_info.update({
                'F_total': F_total,
                'F_goal': F_goal,
                'F_obs': F_obs,
                'F_wall': F_wall,
                'F_open_space': F_open_space
            })
            
            vel = np.clip(F_total, -MAX_SPEED, MAX_SPEED)
            
            # Check if the INTENDED velocity would cause collision
            # This allows robots to find safe velocities even if current state predicts collision


            collision_with_intended = self.predict_collision(neighbors, obstacles, intended_vel=vel)

            # print(f"F_total: {F_total} for robot_id: {self.robot_id} in YIELDING state and velocity: {vel} from f_total_base: {F_total_base} and f_open_space: {F_open_space} and collision_with_intended: {collision_with_intended}")

            # print(f" YIELDING state for robot_id: {self.robot_id} and intended velocity: {vel}")

            if not collision_with_intended:
                # Safe velocity found - transition back to NORMAL
                self.state = RobotState.NORMAL
                force_info['state'] = self.state.value
            else:
                # Still predicting collision with intended velocity
                # But allow transition if velocity is very small (robot is stuck)
                if np.linalg.norm(vel) < 0.05:
                    self.wait_timer -= dt
                    if self.wait_timer <= 0:
                        # Force transition to allow robot to try again
                        self.state = RobotState.NORMAL
                        self.wait_timer = BASE_WAIT_TIME
                        force_info['state'] = self.state.value
            
            return vel, force_info
        
        return np.zeros(2), force_info


def test_formation(gui=True):
    """Test robots in 2 side lines with opposite goals."""
    print("\n" + "=" * 60)
    print("  APF FORMATION TEST")
    print("=" * 60)
    
    # Setup PyBullet
    if gui:
        client_id = pyb.connect(pyb.GUI)
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )
    
    # Create robots in 2 side lines (same as ORCA test)
    num_robots_per_side = 4
    line_spacing = 0.6
    line_offset = 1.5
    
    robots = {}
    start_positions = []
    target_positions = []
    
    # Left side robots
    for i in range(num_robots_per_side):
        name = f"L_{i+1:02d}"
        y = (i - num_robots_per_side/2 + 0.5) * line_spacing
        start_pos = (-line_offset, y)
        target_pos = (line_offset, -y)
        
        robot = create_robot('holonomic', 'dummy', position=start_pos, orientation=0.0, name=name)
        robots[name] = robot
        start_positions.append(np.array(start_pos))
        target_positions.append(np.array(target_pos))
    
    # Right side robots
    for i in range(num_robots_per_side):
        name = f"R_{i+1:02d}"
        y = (i - num_robots_per_side/2 + 0.5) * line_spacing
        start_pos = (line_offset, y)
        target_pos = (-line_offset, -y)
        
        robot = create_robot('holonomic', 'dummy', position=start_pos, orientation=np.pi, name=name)
        robots[name] = robot
        start_positions.append(np.array(start_pos))
        target_positions.append(np.array(target_pos))
    
    print(f"  Created {len(robots)} robots")
    
    # Create APF robots
    apf_robots = []
    robot_names = list(robots.keys())
    for i, name in enumerate(robot_names):
        apf_robot = APFRobot(robots[name], i, start_positions[i], target_positions[i])
        apf_robots.append(apf_robot)
    
    # No obstacles for formation test
    obstacles = []
    
    # Run simulation
    max_steps = int(30.0 / TIMESTEP)
    step_count = 0
    trajectories = {name: [] for name in robots}
    
    print("\n  Running simulation...")
    
    for step in range(max_steps):
        # Get current positions
        for name, robot in robots.items():
            pos, _, _ = robot.get_state()
            trajectories[name].append(pos.copy())
        
        # Control at lower frequency
        if step_count % CTRL_STEP == 0:
            dt = 1.0 / CTRL_FREQ
            
            for apf_robot in apf_robots:
                vel_2d, _ = apf_robot.compute_control(apf_robots, obstacles, dt)
                
                # Check if reached goal
                current_pos = apf_robot.get_position()
                goal = apf_robot.goal_pos
                if np.linalg.norm(current_pos - goal) < 0.1:
                    vel_2d = np.zeros(2)
                
                apf_robot.robot.command_velocity(np.array([vel_2d[0], vel_2d[1], 0.0]))
        
        pyb.stepSimulation()
        step_count += 1
        
        if gui:
            time.sleep(TIMESTEP * 0.3)
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))
    for (name, traj), color in zip(trajectories.items(), colors):
        traj_arr = np.array(traj)
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], '-', color=color, linewidth=1.5, label=name)
        ax.plot(traj_arr[0, 0], traj_arr[0, 1], 'o', color=color, markersize=8)
        ax.plot(traj_arr[-1, 0], traj_arr[-1, 1], 's', color=color, markersize=8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('APF Formation Test - Robot Trajectories')
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/apf_formation_test.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to /tmp/apf_formation_test.png")
    
    if gui:
        print("\n  Press Enter to exit...")
        input()
    
    pyb.disconnect()
    plt.close()


def test_obstacles(gui=True):
    """Test APF with static obstacles."""
    print("\n" + "=" * 60)
    print("  APF OBSTACLE TEST")
    print("=" * 60)
    
    # Setup PyBullet
    if gui:
        client_id = pyb.connect(pyb.GUI)
    else:
        client_id = pyb.connect(pyb.DIRECT)
    
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(TIMESTEP)
    pyb.setRealTimeSimulation(0)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground = pyb.loadURDF("plane.urdf", [0, 0, 0])
    
    if gui:
        pyb.resetDebugVisualizerCamera(
            cameraDistance=6.0,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=[0, 0, 0]
        )
    
    # Create robots (same as ORCA test)
    num_robots = 4
    robots = {}
    start_positions = []
    target_positions = []
    
    for i in range(num_robots):
        name = f"R_{i+1:02d}"
        angle = 2 * np.pi * i / num_robots
        start_pos = (2.0 * np.cos(angle), 2.0 * np.sin(angle))
        target_pos = (2.0 * np.cos(angle + np.pi), 2.0 * np.sin(angle + np.pi))
        
        robot = create_robot('holonomic', 'dummy', position=start_pos, orientation=angle, name=name)
        robots[name] = robot
        start_positions.append(np.array(start_pos))
        target_positions.append(np.array(target_pos))
    
    # Define obstacle (square)
    obstacle = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    ]
    
    # Create visual obstacle
    obstacle_height = 0.5
    obstacle_visual_shape = pyb.createVisualShape(
        shapeType=pyb.GEOM_BOX,
        halfExtents=[1.0, 1.0, obstacle_height/2],
        rgbaColor=[0.5, 0.5, 0.5, 0.8]
    )
    obstacle_collision_shape = pyb.createCollisionShape(
        shapeType=pyb.GEOM_BOX,
        halfExtents=[1.0, 1.0, obstacle_height/2]
    )
    obstacle_body = pyb.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=obstacle_collision_shape,
        baseVisualShapeIndex=obstacle_visual_shape,
        basePosition=[0, 0, obstacle_height/2]
    )
    
    print(f"  Created {len(robots)} robots")
    print(f"  Obstacle: square from (-1, -1) to (1, 1)")
    
    # Create APF robots
    apf_robots = []
    robot_names = list(robots.keys())
    for i, name in enumerate(robot_names):
        apf_robot = APFRobot(robots[name], i, start_positions[i], target_positions[i])
        apf_robots.append(apf_robot)
    
    # Obstacles as list of polygons
    obstacles = [obstacle]
    
    # Run simulation
    max_steps = int(30.0 / TIMESTEP)
    step_count = 0
    trajectories = {name: [] for name in robots}
    
    # Force and state history for analysis
    force_history = {name: [] for name in robots}
    state_history = {name: [] for name in robots}
    time_history = []
    
    print("\n  Running simulation...")
    
    for step in range(max_steps):
        current_positions = {}
        for name, robot in robots.items():
            pos, _, _ = robot.get_state()
            trajectories[name].append(pos.copy())
        
        if step_count % CTRL_STEP == 0:
            dt = 1.0 / CTRL_FREQ
            current_time = step_count * TIMESTEP
            
            for i, apf_robot in enumerate(apf_robots):
                name = robot_names[i]
                
                # Debug: Print collision prediction for robots 1 and 3 (indices 1 and 3)
                if i in [1, 3] and step_count % (CTRL_STEP * 10) == 0:  # Print every 10 control steps
                    neighbors = apf_robot.sense_neighbors(apf_robots, obstacles)
                    collision_pred = apf_robot.predict_collision(neighbors, obstacles)
                    pos = apf_robot.get_position()
                    vel = apf_robot.get_velocity()
                    print(f"  DEBUG {name}: pos=({pos[0]:.2f}, {pos[1]:.2f}), "
                          f"vel=({vel[0]:.2f}, {vel[1]:.2f}), "
                          f"collision_pred={collision_pred}, state={apf_robot.state.value}, "
                          f"neighbors={len(neighbors)}")
                
                vel_2d, force_info = apf_robot.compute_control(apf_robots, obstacles, dt)
                
                # Store force and state history
                if len(time_history) == 0 or current_time != time_history[-1]:
                    time_history.append(current_time)
                
                # Ensure history lists are same length
                while len(force_history[name]) < len(time_history):
                    force_history[name].append({
                        'F_total': np.zeros(2),
                        'F_goal': np.zeros(2),
                        'F_obs': np.zeros(2),
                        'F_wall': np.zeros(2),
                        'F_open_space': np.zeros(2),
                        'state': 'NORMAL'
                    })
                
                force_history[name][-1] = force_info.copy()
                state_history[name].append(apf_robot.state.value)
                
                # Check if reached goal
                current_pos = apf_robot.get_position()
                goal = apf_robot.goal_pos
                if np.linalg.norm(current_pos - goal) < 0.1:
                    vel_2d = np.zeros(2)
                
                apf_robot.robot.command_velocity(np.array([vel_2d[0], vel_2d[1], 0.0]))
        
        pyb.stepSimulation()
        step_count += 1
        
        if gui:
            time.sleep(TIMESTEP * 0.3)
    
    # Plot 1: Trajectories
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    # Draw obstacle
    obs_x = [v[0] for v in obstacle] + [obstacle[0][0]]
    obs_y = [v[1] for v in obstacle] + [obstacle[0][1]]
    ax1.fill(obs_x, obs_y, alpha=0.3, color='gray', label='Obstacle')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(robots)))
    for (name, traj), color in zip(trajectories.items(), colors):
        traj_arr = np.array(traj)
        ax1.plot(traj_arr[:, 0], traj_arr[:, 1], '-', color=color, linewidth=1.5, label=name)
        ax1.plot(traj_arr[0, 0], traj_arr[0, 1], 'o', color=color, markersize=8)
        ax1.plot(traj_arr[-1, 0], traj_arr[-1, 1], 's', color=color, markersize=8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('APF Obstacle Test - Robot Trajectories')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/apf_obstacle_test.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved trajectory plot to /tmp/apf_obstacle_test.png")
    
    # Plot 2: Force component magnitudes over time
    fig2, axes2 = plt.subplots(len(robots), 1, figsize=(14, 4 * len(robots)))
    if len(robots) == 1:
        axes2 = [axes2]
    
    for i, (name, color) in enumerate(zip(robot_names, colors)):
        ax = axes2[i]
        hist = force_history[name]
        
        # Extract force magnitudes
        times = np.array(time_history[:len(hist)])
        F_total_mag = [np.linalg.norm(f['F_total']) for f in hist]
        F_goal_mag = [np.linalg.norm(f['F_goal']) for f in hist]
        F_obs_mag = [np.linalg.norm(f['F_obs']) for f in hist]
        F_wall_mag = [np.linalg.norm(f['F_wall']) for f in hist]
        F_open_mag = [np.linalg.norm(f['F_open_space']) for f in hist]
        
        ax.plot(times, F_total_mag, 'k-', linewidth=2, label='F_total')
        ax.plot(times, F_goal_mag, 'g-', linewidth=1.5, label='F_goal')
        ax.plot(times, F_obs_mag, 'r-', linewidth=1.5, label='F_obs (neighbors)')
        ax.plot(times, F_wall_mag, 'b-', linewidth=1.5, label='F_wall (obstacles)')
        ax.plot(times, F_open_mag, 'm-', linewidth=1.5, label='F_open_space')
        
        # Add state transitions as vertical lines
        prev_state = None
        for j, state in enumerate(state_history[name]):
            if state != prev_state and j < len(times):
                ax.axvline(times[j], color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            prev_state = state
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force Magnitude')
        ax.set_title(f'{name} - Force Components Over Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/apf_obstacle_forces_time.png', dpi=150, bbox_inches='tight')
    print(f"  Saved force-time plot to /tmp/apf_obstacle_forces_time.png")
    
    # Plot 3: Force vector visualization at key moments
    # Sample every N control steps
    sample_rate = max(1, len(time_history) // 20)
    sample_indices = list(range(0, len(time_history), sample_rate))
    
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    
    # Draw obstacle
    ax3.fill(obs_x, obs_y, alpha=0.3, color='gray', label='Obstacle')
    
    # Plot trajectories
    for (name, traj), color in zip(trajectories.items(), colors):
        traj_arr = np.array(traj)
        ax3.plot(traj_arr[:, 0], traj_arr[:, 1], '-', color=color, linewidth=1, alpha=0.3)
    
    # Draw force vectors at sampled points
    scale_factor = 0.3  # Scale for visualization
    for idx in sample_indices:
        if idx >= len(time_history):
            continue
        
        for i, (name, color) in enumerate(zip(robot_names, colors)):
            if idx >= len(force_history[name]):
                continue
            
            # Get position at this time (approximate from trajectory)
            traj = trajectories[name]
            traj_idx = min(idx * CTRL_STEP, len(traj) - 1)
            pos = traj[traj_idx][:2]
            
            force_info = force_history[name][idx]
            
            # Draw force components
            F_goal = force_info['F_goal']
            F_obs = force_info['F_obs']
            F_wall = force_info['F_wall']
            F_open = force_info['F_open_space']
            F_total = force_info['F_total']
            
            # Goal force (green)
            if np.linalg.norm(F_goal) > 0.01:
                ax3.arrow(pos[0], pos[1], F_goal[0] * scale_factor, F_goal[1] * scale_factor,
                         head_width=0.05, head_length=0.05, fc='green', ec='green', alpha=0.6, linewidth=1)
            
            # Obstacle repulsion (red)
            if np.linalg.norm(F_obs) > 0.01:
                ax3.arrow(pos[0], pos[1], F_obs[0] * scale_factor, F_obs[1] * scale_factor,
                         head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6, linewidth=1)
            
            # Wall repulsion (blue)
            if np.linalg.norm(F_wall) > 0.01:
                ax3.arrow(pos[0], pos[1], F_wall[0] * scale_factor, F_wall[1] * scale_factor,
                         head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.6, linewidth=1)
            
            # Open space bias (magenta)
            if np.linalg.norm(F_open) > 0.01:
                ax3.arrow(pos[0], pos[1], F_open[0] * scale_factor, F_open[1] * scale_factor,
                         head_width=0.05, head_length=0.05, fc='magenta', ec='magenta', alpha=0.6, linewidth=1)
            
            # Total force (black, thicker)
            if np.linalg.norm(F_total) > 0.01:
                ax3.arrow(pos[0], pos[1], F_total[0] * scale_factor, F_total[1] * scale_factor,
                         head_width=0.08, head_length=0.08, fc='black', ec='black', alpha=0.8, linewidth=2)
    
    # Add legend
    from matplotlib.patches import FancyArrowPatch
    legend_elements = [
        plt.Line2D([0], [0], color='green', linewidth=2, label='F_goal'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='F_obs (neighbors)'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='F_wall (obstacles)'),
        plt.Line2D([0], [0], color='magenta', linewidth=2, label='F_open_space'),
        plt.Line2D([0], [0], color='black', linewidth=3, label='F_total')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('APF Obstacle Test - Force Vector Visualization')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/apf_obstacle_forces_vectors.png', dpi=150, bbox_inches='tight')
    print(f"  Saved force vector plot to /tmp/apf_obstacle_forces_vectors.png")
    
    # Plot 4: State transitions
    fig4, axes4 = plt.subplots(len(robots), 1, figsize=(14, 2 * len(robots)))
    if len(robots) == 1:
        axes4 = [axes4]
    
    state_colors = {'NORMAL': 'green', 'WAITING': 'orange', 'YIELDING': 'red'}
    
    for i, (name, color) in enumerate(zip(robot_names, colors)):
        ax = axes4[i]
        states = state_history[name]
        times_state = np.linspace(0, max(time_history) if time_history else 0, len(states))
        
        # Map states to numbers for plotting
        state_map = {'NORMAL': 0, 'WAITING': 1, 'YIELDING': 2}
        state_nums = [state_map.get(s, 0) for s in states]
        
        ax.plot(times_state, state_nums, 'o-', color=color, markersize=4, linewidth=1.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['NORMAL', 'WAITING', 'YIELDING'])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('State')
        ax.set_title(f'{name} - State Transitions')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 2.2)
    
    plt.tight_layout()
    plt.savefig('/tmp/apf_obstacle_states.png', dpi=150, bbox_inches='tight')
    print(f"  Saved state plot to /tmp/apf_obstacle_states.png")
    
    # Plot 5: Force component analysis (summary statistics)
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Average force magnitudes per robot
    ax = axes5[0, 0]
    robot_avg_forces = {name: {
        'F_goal': 0, 'F_obs': 0, 'F_wall': 0, 'F_open_space': 0, 'F_total': 0
    } for name in robot_names}
    
    for name in robot_names:
        hist = force_history[name]
        if len(hist) > 0:
            for key in robot_avg_forces[name]:
                values = [np.linalg.norm(f[key]) for f in hist]
                robot_avg_forces[name][key] = np.mean(values) if values else 0
    
    x = np.arange(len(robot_names))
    width = 0.15
    for i, key in enumerate(['F_goal', 'F_obs', 'F_wall', 'F_open_space', 'F_total']):
        values = [robot_avg_forces[name][key] for name in robot_names]
        ax.bar(x + i * width, values, width, label=key, alpha=0.8)
    
    ax.set_xlabel('Robot')
    ax.set_ylabel('Average Force Magnitude')
    ax.set_title('Average Force Components per Robot')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(robot_names)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Force cancellation analysis: dot product between F_goal and (F_obs + F_wall)
    ax = axes5[0, 1]
    cancellation_angles = {name: [] for name in robot_names}
    for name in robot_names:
        hist = force_history[name]
        for f in hist:
            F_goal = f['F_goal']
            F_repulsive = f['F_obs'] + f['F_wall']
            if np.linalg.norm(F_goal) > 0.01 and np.linalg.norm(F_repulsive) > 0.01:
                cos_angle = np.dot(F_goal, F_repulsive) / (np.linalg.norm(F_goal) * np.linalg.norm(F_repulsive))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                cancellation_angles[name].append(angle * 180 / np.pi)
    
    for name, color in zip(robot_names, colors):
        if cancellation_angles[name]:
            ax.hist(cancellation_angles[name], bins=20, alpha=0.6, label=name, color=color)
    
    ax.set_xlabel('Angle between F_goal and F_repulsive (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Force Cancellation Analysis\n(180° = complete cancellation)')
    ax.axvline(180, color='red', linestyle='--', linewidth=2, label='Complete cancellation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Time spent in each state
    ax = axes5[1, 0]
    state_times = {name: {'NORMAL': 0, 'WAITING': 0, 'YIELDING': 0} for name in robot_names}
    dt_control = 1.0 / CTRL_FREQ
    for name in robot_names:
        for state in state_history[name]:
            state_times[name][state] += dt_control
    
    x = np.arange(len(robot_names))
    width = 0.25
    states_list = ['NORMAL', 'WAITING', 'YIELDING']
    state_colors_bar = {'NORMAL': 'green', 'WAITING': 'orange', 'YIELDING': 'red'}
    for i, state in enumerate(states_list):
        values = [state_times[name][state] for name in robot_names]
        ax.bar(x + i * width, values, width, label=state, color=state_colors_bar[state], alpha=0.8)
    
    ax.set_xlabel('Robot')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time Spent in Each State')
    ax.set_xticks(x + width)
    ax.set_xticklabels(robot_names)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Force magnitude distribution
    ax = axes5[1, 1]
    for name, color in zip(robot_names, colors):
        hist = force_history[name]
        F_total_mags = [np.linalg.norm(f['F_total']) for f in hist]
        if F_total_mags:
            ax.hist(F_total_mags, bins=30, alpha=0.6, label=name, color=color)
    
    ax.set_xlabel('F_total Magnitude')
    ax.set_ylabel('Frequency')
    ax.set_title('Total Force Magnitude Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/apf_obstacle_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved analysis plot to /tmp/apf_obstacle_analysis.png")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("  FORCE ANALYSIS SUMMARY")
    print("=" * 60)
    for name in robot_names:
        hist = force_history[name]
        if len(hist) > 0:
            F_goal_avg = np.mean([np.linalg.norm(f['F_goal']) for f in hist])
            F_obs_avg = np.mean([np.linalg.norm(f['F_obs']) for f in hist])
            F_wall_avg = np.mean([np.linalg.norm(f['F_wall']) for f in hist])
            F_open_avg = np.mean([np.linalg.norm(f['F_open_space']) for f in hist])
            F_total_avg = np.mean([np.linalg.norm(f['F_total']) for f in hist])
            
            # Count cancellation events (F_goal and F_repulsive pointing opposite)
            cancellation_count = 0
            for f in hist:
                F_goal = f['F_goal']
                F_repulsive = f['F_obs'] + f['F_wall']
                if np.linalg.norm(F_goal) > 0.01 and np.linalg.norm(F_repulsive) > 0.01:
                    cos_angle = np.dot(F_goal, F_repulsive) / (np.linalg.norm(F_goal) * np.linalg.norm(F_repulsive))
                    if cos_angle < -0.5:  # Angle > 120 degrees
                        cancellation_count += 1
            
            print(f"\n  {name}:")
            print(f"    Avg F_goal:      {F_goal_avg:.3f}")
            print(f"    Avg F_obs:       {F_obs_avg:.3f}")
            print(f"    Avg F_wall:      {F_wall_avg:.3f}")
            print(f"    Avg F_open_space: {F_open_avg:.3f}")
            print(f"    Avg F_total:     {F_total_avg:.3f}")
            print(f"    Cancellation events: {cancellation_count}/{len(hist)} ({100*cancellation_count/len(hist):.1f}%)")
            
            # State statistics
            state_counts = {}
            for state in state_history[name]:
                state_counts[state] = state_counts.get(state, 0) + 1
            print(f"    States: NORMAL={state_counts.get('NORMAL', 0)}, "
                  f"WAITING={state_counts.get('WAITING', 0)}, "
                  f"YIELDING={state_counts.get('YIELDING', 0)}")
    
    print("\n" + "=" * 60)
    print("  DIAGNOSTIC INSIGHTS:")
    print("=" * 60)
    print("  - If F_wall >> F_goal: Obstacle repulsion too strong")
    print("  - If cancellation events > 50%: Forces canceling frequently")
    print("  - If F_open_space is small: Open-space bias not helping")
    print("  - If robots stuck in YIELDING: Need stronger bias or different strategy")
    print("=" * 60)
    
    plt.close('all')
    
    if gui:
        print("\n  Press Enter to exit...")
        input()
    
    pyb.disconnect()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="APF Navigation Test")
    parser.add_argument("--test", type=str, default="formation",
                       choices=['formation', 'obstacles', 'all'],
                       help="Test to run")
    parser.add_argument("--no-gui", action="store_true",
                       help="Run without GUI")
    
    args = parser.parse_args()
    
    if args.test in ['formation', 'all']:
        test_formation(gui=not args.no_gui)
    
    if args.test in ['obstacles', 'all']:
        test_obstacles(gui=not args.no_gui)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
