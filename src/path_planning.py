"""Path planning module implementing graph search algorithms for 2D grid environments.

Provides deterministic search planners:
1. AStarPlanner: Heuristic A* search using admissible Manhattan distance.
2. DijkstraPlanner: Uniform-cost search (A* with h(n) = 0).
3. BFSPlanner: Breadth-First Search evaluating level-by-level using a FIFO queue.

Decoupled entirely from visualization and robot mechanics. Computes optimal routes
using strictly four-directional transitions.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import heapq
import time
from typing import Dict, List, Optional, Tuple, Set

from src.environment import GridEnvironment, EnvironmentError


@dataclass
class PathResult:
    """Encapsulates the outcome of a path planning query.

    Attributes:
        success: True if an obstacle-free route was found, False otherwise.
        path: List of (row, col) coordinates from start to goal inclusive, or None if failed.
        cost: Total movement cost of the path (number of steps), or infinity if no path.
        nodes_expanded: Number of unique grid cells evaluated during the search.
        execution_time_ms: Computation duration in milliseconds.
        error_message: Explanatory error details if planning failed.
        algorithm: Name of algorithm used to compute result.
    """

    success: bool
    path: Optional[List[Tuple[int, int]]] = None
    cost: float = float("inf")
    nodes_expanded: int = 0
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    algorithm: str = "A*"

    @property
    def path_length(self) -> int:
        """Return the number of waypoints in the path, or 0 if no path."""
        return len(self.path) if self.path is not None else 0


class BasePlanner(ABC):
    """Abstract base class for discrete 2D grid path planners."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the canonical name of the planning algorithm."""
        pass

    @abstractmethod
    def plan(
        self,
        environment: GridEnvironment,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> PathResult:
        """Compute a collision-free path from start to goal.

        Args:
            environment: GridEnvironment containing occupancy data.
            start: Logical (row, column) start coordinate.
            goal: Logical (row, column) goal coordinate.

        Returns:
            PathResult encapsulating success, path, cost, and telemetry.
        """
        pass

    def find_path(
        self,
        environment: GridEnvironment,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Convenience method returning either the path coordinate list or None."""
        result = self.plan(environment, start, goal)
        return result.path if result.success else None

    @staticmethod
    def _validate_positions(
        environment: GridEnvironment,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[str]:
        """Validate start and goal coordinates. Returns error message if invalid."""
        try:
            environment.validate_position(start, name="Start")
            environment.validate_position(goal, name="Goal")
            return None
        except EnvironmentError as err:
            return str(err)

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Backtrack from goal node to start to reconstruct sequential route."""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path


class AStarPlanner(BasePlanner):
    """Deterministic A* path planner for 2D grid environments.

    Uses strictly 4-directional transitions (up, down, left, right) with uniform
    cost 1.0 per transition. The Manhattan distance heuristic is utilized, which is
    provably admissible and consistent on a 4-connected grid.
    """

    @property
    def name(self) -> str:
        return "A*"

    @staticmethod
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate the Manhattan (L1) distance heuristic between two grid cells."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def plan(
        self,
        environment: GridEnvironment,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> PathResult:
        start_time = time.perf_counter()

        # Step 1: Validate coordinates
        err = self._validate_positions(environment, start, goal)
        if err is not None:
            return PathResult(
                success=False,
                path=None,
                cost=float("inf"),
                nodes_expanded=0,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                error_message=err,
                algorithm=self.name,
            )

        # Step 2: Base case — start equals goal
        if start == goal:
            return PathResult(
                success=True,
                path=[start],
                cost=0.0,
                nodes_expanded=0,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                algorithm=self.name,
            )

        # Step 3: Priority queue initialization (f_score, tie_breaker_counter, cell)
        counter: int = 0
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self.manhattan_distance(start, goal), counter, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed_set: Set[Tuple[int, int]] = set()
        nodes_expanded: int = 0

        # Step 4: Search Loop
        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return PathResult(
                    success=True,
                    path=path,
                    cost=float(len(path) - 1),
                    nodes_expanded=nodes_expanded,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                    algorithm=self.name,
                )

            current_g = g_score[current]

            for neighbor in environment.get_neighbors(current[0], current[1]):
                if neighbor in closed_set:
                    continue

                tentative_g = current_g + 1.0

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.manhattan_distance(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        # Step 5: Exhausted open set without finding goal
        return PathResult(
            success=False,
            path=None,
            cost=float("inf"),
            nodes_expanded=nodes_expanded,
            execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            error_message="No obstacle-free path exists between start and goal.",
            algorithm=self.name,
        )


class DijkstraPlanner(BasePlanner):
    """Uniform-cost search planner (Dijkstra's Algorithm).

    Equivalent to A* with h(n) = 0. Explores radially in order of distance from start.
    Guaranteed optimal for non-negative edge costs.
    """

    @property
    def name(self) -> str:
        return "Dijkstra"

    def plan(
        self,
        environment: GridEnvironment,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> PathResult:
        start_time = time.perf_counter()

        # Step 1: Validate coordinates
        err = self._validate_positions(environment, start, goal)
        if err is not None:
            return PathResult(
                success=False,
                path=None,
                cost=float("inf"),
                nodes_expanded=0,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                error_message=err,
                algorithm=self.name,
            )

        # Step 2: Base case — start equals goal
        if start == goal:
            return PathResult(
                success=True,
                path=[start],
                cost=0.0,
                nodes_expanded=0,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                algorithm=self.name,
            )

        # Step 3: Priority queue initialization (cost, counter, cell)
        counter: int = 0
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, counter, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        dist: Dict[Tuple[int, int], float] = {start: 0.0}
        closed_set: Set[Tuple[int, int]] = set()
        nodes_expanded: int = 0

        # Step 4: Search Loop
        while open_set:
            current_cost, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return PathResult(
                    success=True,
                    path=path,
                    cost=float(len(path) - 1),
                    nodes_expanded=nodes_expanded,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                    algorithm=self.name,
                )

            for neighbor in environment.get_neighbors(current[0], current[1]):
                if neighbor in closed_set:
                    continue

                tentative_cost = current_cost + 1.0

                if tentative_cost < dist.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    dist[neighbor] = tentative_cost
                    counter += 1
                    heapq.heappush(open_set, (tentative_cost, counter, neighbor))

        return PathResult(
            success=False,
            path=None,
            cost=float("inf"),
            nodes_expanded=nodes_expanded,
            execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            error_message="No obstacle-free path exists between start and goal.",
            algorithm=self.name,
        )


class BFSPlanner(BasePlanner):
    """Breadth-First Search (BFS) path planner.

    Explores nodes level-by-level using a FIFO queue. Guaranteed to find the
    shortest path in terms of transition step count on unweighted grid graphs.
    """

    @property
    def name(self) -> str:
        return "BFS"

    def plan(
        self,
        environment: GridEnvironment,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> PathResult:
        start_time = time.perf_counter()

        # Step 1: Validate coordinates
        err = self._validate_positions(environment, start, goal)
        if err is not None:
            return PathResult(
                success=False,
                path=None,
                cost=float("inf"),
                nodes_expanded=0,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                error_message=err,
                algorithm=self.name,
            )

        # Step 2: Base case — start equals goal
        if start == goal:
            return PathResult(
                success=True,
                path=[start],
                cost=0.0,
                nodes_expanded=0,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                algorithm=self.name,
            )

        # Step 3: Queue initialization (FIFO)
        queue: deque = deque([start])
        visited: Set[Tuple[int, int]] = {start}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        nodes_expanded: int = 0

        # Step 4: Search Loop
        while queue:
            current = queue.popleft()
            nodes_expanded += 1

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return PathResult(
                    success=True,
                    path=path,
                    cost=float(len(path) - 1),
                    nodes_expanded=nodes_expanded,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                    algorithm=self.name,
                )

            for neighbor in environment.get_neighbors(current[0], current[1]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

        return PathResult(
            success=False,
            path=None,
            cost=float("inf"),
            nodes_expanded=nodes_expanded,
            execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            error_message="No obstacle-free path exists between start and goal.",
            algorithm=self.name,
        )


SUPPORTED_ALGORITHMS: List[str] = ["A*", "Dijkstra", "BFS"]


def get_planner(algorithm_name: str) -> BasePlanner:
    """Factory retrieving a path planner instance by algorithm name.

    Args:
        algorithm_name: One of 'A*', 'Dijkstra', 'BFS' (case-insensitive).

    Returns:
        An instance of BasePlanner.

    Raises:
        ValueError: If algorithm_name is unrecognized.
    """
    clean_name = algorithm_name.strip().upper().replace("-", "").replace(" ", "").replace("_", "")
    if "A*" in clean_name or "ASTAR" in clean_name or clean_name == "A":
        return AStarPlanner()
    elif "DIJKSTRA" in clean_name:
        return DijkstraPlanner()
    elif "BFS" in clean_name:
        return BFSPlanner()
    raise ValueError(
        f"Unknown planning algorithm: '{algorithm_name}'. Supported algorithms: {SUPPORTED_ALGORITHMS}"
    )
