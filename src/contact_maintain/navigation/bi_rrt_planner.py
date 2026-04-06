"""Bidirectional RRT Planner for 2D Navigation.

This module implements a simple bi-directional RRT (Bi-RRT) planner for
2D point robots with polygonal obstacles. It is intended to be used by
navigation schemes such as:

- Static Single: step 1 (navigate from current pose to the ring)
- Divide-n-Conquer: step 1 (navigate from current pose to some ring point)

The planner returns a sequence of waypoints in world coordinates that
can be followed by a simple P-controller.

Author: Contact Maintain Team
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union


@dataclass
class RRTConfig:
    """Configuration parameters for Bi-RRT."""

    step_size: float = 0.15          # max extension step (m)
    max_iterations: int = 2000       # max tree expansions
    goal_sample_rate: float = 0.1    # probability of sampling goal directly
    min_clearance: float = 0.02      # minimum distance from obstacles (m)
    connection_threshold: float = 0.2  # distance to attempt tree connection


@dataclass
class RRTNode:
    """Node in an RRT tree."""

    position: np.ndarray             # shape (2,)
    parent: Optional[int]            # index of parent node in tree list


class BiRRTPlanner:
    """Simple 2D Bi-RRT planner with polygon obstacles."""

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        obstacles: Sequence[Polygon],
        config: Optional[RRTConfig] = None,
    ):
        """
        Parameters
        ----------
        bounds : (xmin, xmax, ymin, ymax)
            Sampling bounds for the planner.
        obstacles : sequence of shapely.Polygon
            Obstacles in world coordinates.
        config : RRTConfig, optional
            Planner parameters.
        """
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.obstacles = list(obstacles)
        self.obstacle_union = unary_union(self.obstacles) if self.obstacles else None
        self.config = config or RRTConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """Plan a path from start to goal using Bi-RRT.

        Parameters
        ----------
        start : np.ndarray
            Start position (x, y).
        goal : np.ndarray
            Goal position (x, y).

        Returns
        -------
        list of np.ndarray or None
            Sequence of waypoints from start to goal (including both).
            Returns None if planning fails.
        """
        start = np.array(start, dtype=float).reshape(2)
        goal = np.array(goal, dtype=float).reshape(2)

        if self._in_collision(start) or self._in_collision(goal):
            return None

        tree_a: List[RRTNode] = [RRTNode(start, None)]
        tree_b: List[RRTNode] = [RRTNode(goal, None)]

        for it in range(self.config.max_iterations):
            # Alternate which tree we grow from
            if it % 2 == 0:
                tree_from, tree_to = tree_a, tree_b
            else:
                tree_from, tree_to = tree_b, tree_a

            # Sample a random point (with goal bias)
            rnd = self._sample_random_point(goal if tree_from is tree_a else start)

            # Extend tree_from towards rnd
            idx_near = self._nearest_node_index(tree_from, rnd)
            new_pos = self._steer(tree_from[idx_near].position, rnd)
            if self._segment_in_collision(tree_from[idx_near].position, new_pos):
                continue

            tree_from.append(RRTNode(new_pos, idx_near))
            idx_new = len(tree_from) - 1

            # Try to connect tree_to towards new node
            idx_near_to = self._nearest_node_index(tree_to, new_pos)
            pos_near_to = tree_to[idx_near_to].position
            if np.linalg.norm(pos_near_to - new_pos) < self.config.connection_threshold:
                # Check direct connection between new_pos and pos_near_to
                if not self._segment_in_collision(new_pos, pos_near_to):
                    # Trees are connected - reconstruct full path
                    if tree_from is tree_a:
                        path_a = self._reconstruct_path(tree_a, idx_new, forward=True)
                        path_b = self._reconstruct_path(tree_b, idx_near_to, forward=False)
                    else:
                        path_a = self._reconstruct_path(tree_b, idx_near_to, forward=True)
                        path_b = self._reconstruct_path(tree_a, idx_new, forward=False)

                    path = path_a + path_b
                    path = self._shortcut_path(path)
                    return path

        # Failed
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sample_random_point(self, goal: np.ndarray) -> np.ndarray:
        """Sample random point in bounds, with goal bias."""
        if np.random.rand() < self.config.goal_sample_rate:
            return goal
        x = np.random.uniform(self.xmin, self.xmax)
        y = np.random.uniform(self.ymin, self.ymax)
        return np.array([x, y], dtype=float)

    def _nearest_node_index(self, tree: List[RRTNode], point: np.ndarray) -> int:
        """Return index of nearest node in tree to point."""
        dists = [np.linalg.norm(n.position - point) for n in tree]
        return int(np.argmin(dists))

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Move from from_pos towards to_pos by at most step_size."""
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-9:
            return from_pos.copy()
        step = min(self.config.step_size, dist)
        return from_pos + direction / dist * step

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------
    def _in_collision(self, point: np.ndarray) -> bool:
        """Check if point is inside or too close to obstacles."""
        if self.obstacle_union is None:
            return False
        p = Point(float(point[0]), float(point[1]))
        # Add clearance margin
        if self.config.min_clearance > 0:
            inflated = self.obstacle_union.buffer(self.config.min_clearance)
            return inflated.contains(p)
        return self.obstacle_union.contains(p)

    def _segment_in_collision(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if segment [p1, p2] intersects obstacles with clearance."""
        if self.obstacle_union is None:
            return False
        segment = LineString([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))])
        inflated = (
            self.obstacle_union.buffer(self.config.min_clearance)
            if self.config.min_clearance > 0
            else self.obstacle_union
        )
        return inflated.intersects(segment)

    # ------------------------------------------------------------------
    # Path reconstruction and shortcutting
    # ------------------------------------------------------------------
    def _reconstruct_path(
        self,
        tree: List[RRTNode],
        idx: int,
        forward: bool = True,
    ) -> List[np.ndarray]:
        """Reconstruct path from root to idx (or reverse)."""
        path: List[np.ndarray] = []
        curr = idx
        while curr is not None:
            path.append(tree[curr].position)
            curr = tree[curr].parent
        if forward:
            path.reverse()
        return path

    def _shortcut_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Simple path-shortcutting to remove unnecessary waypoints."""
        if len(path) <= 2 or self.obstacle_union is None:
            return path
        shortened: List[np.ndarray] = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            # Find the furthest point we can connect to directly
            while j > i + 1:
                if not self._segment_in_collision(path[i], path[j]):
                    break
                j -= 1
            shortened.append(path[j])
            i = j
        return shortened


# ----------------------------------------------------------------------
# Convenience helper: plan to ring point
# ----------------------------------------------------------------------

def compute_ring_target(
    generic_object,
    t_param: float,
    object_position: np.ndarray,
    object_orientation: float,
    ring_offset: float,
) -> np.ndarray:
    """Compute a target point on the ring for given t_param.

    This mirrors the ring definition used in DistributedMonitor:
    - boundary point at t_param
    - outward normal
    - offset by ring_offset
    """
    from object_utils import ContactPointParameterization

    param = ContactPointParameterization(generic_object)
    result = param.parameter_to_point(t_param)
    local_point = np.array([result[0][0], result[0][1]], dtype=float)

    c, s = np.cos(object_orientation), np.sin(object_orientation)
    R = np.array([[c, -s], [s, c]])
    boundary_pos = R @ local_point + object_position

    normal_local = param.get_normal_vector(t_param, outward=True)
    normal_world = R @ normal_local

    return boundary_pos + normal_world * ring_offset


def plan_to_ring_point(
    start: np.ndarray,
    generic_object,
    t_param: float,
    object_position: np.ndarray,
    object_orientation: float,
    robot_radius: float = 0.06,
    extra_margin: float = 0.2,
    bounds_margin: float = 2.0,
) -> Optional[List[np.ndarray]]:
    """Plan a path from start to a ring point above t_param.

    This is a convenience wrapper used by navigation schemes for step 1:
    moving the robot from its current position to the ring.
    """
    # Ring offset: 1.5 * robot_radius + small margin
    ring_offset = 1.5 * robot_radius + extra_margin

    # Compute ring target
    target = compute_ring_target(
        generic_object,
        t_param,
        object_position,
        object_orientation,
        ring_offset,
    )

    # Derive simple bounds from object geometry
    geom = generic_object.geometry
    minx, miny, maxx, maxy = geom.bounds
    xmin = float(minx) - bounds_margin
    xmax = float(maxx) + bounds_margin
    ymin = float(miny) - bounds_margin
    ymax = float(maxy) + bounds_margin

    # Obstacle is the object's polygon itself
    obstacles = [Polygon(geom.exterior.coords)]

    planner = BiRRTPlanner(bounds=(xmin, xmax, ymin, ymax), obstacles=obstacles)
    path = planner.plan(start=start, goal=target)
    return path

