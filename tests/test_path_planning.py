"""Unit tests for the A* Path Planning module.

Verifies:
1. Simple map with known valid route.
2. Map where no route exists (blocked path).
3. Start equals goal (zero-cost trivial path).
4. Path never enters an occupied cell.
5. Path never leaves grid boundaries.
6. Strictly four-directional movement (Manhattan distance = 1 between successive steps).
7. Planning rejection on invalid start or goal positions.
8. Dynamic replanning when an obstacle blocks a route.
"""

from typing import List, Tuple
import pytest

from src.environment import GridEnvironment
from src.path_planning import (
    AStarPlanner,
    DijkstraPlanner,
    BFSPlanner,
    PathResult,
    get_planner,
    SUPPORTED_ALGORITHMS,
)
from src.robot import Robot, RobotStatus
from src.navigation import NavigationController


class TestAStarPlanner:
    """Core test suite for AStarPlanner algorithms and invariants."""

    @pytest.fixture
    def planner(self) -> AStarPlanner:
        return AStarPlanner()

    def test_simple_map_with_known_valid_route(self, planner: AStarPlanner) -> None:
        """Finds a valid optimal path in a simple 5x5 grid with a central obstacle."""
        env = GridEnvironment(rows=5, cols=5)
        # Wall at column 2, rows 1, 2, 3
        env.set_obstacle(1, 2)
        env.set_obstacle(2, 2)
        env.set_obstacle(3, 2)

        start = (2, 0)
        goal = (2, 4)

        result: PathResult = planner.plan(env, start, goal)

        assert result.success is True
        assert result.path is not None
        assert result.path[0] == start
        assert result.path[-1] == goal
        assert result.cost > 0
        assert result.cost == float(len(result.path) - 1)

    def test_map_where_no_route_exists(self, planner: AStarPlanner) -> None:
        """Returns explicit failure when the goal is completely sealed off by walls."""
        env = GridEnvironment(rows=5, cols=5)
        # Wall completely encircling corner goal (4, 4)
        env.set_obstacle(3, 4)
        env.set_obstacle(4, 3)
        env.set_obstacle(3, 3)

        start = (0, 0)
        goal = (4, 4)

        result: PathResult = planner.plan(env, start, goal)

        assert result.success is False
        assert result.path is None
        assert result.cost == float("inf")
        assert "No obstacle-free path exists" in (result.error_message or "")

    def test_start_equals_goal(self, planner: AStarPlanner) -> None:
        """When start == goal, returns success with a single-element path [start] and 0 cost."""
        env = GridEnvironment(rows=5, cols=5)
        pos = (2, 3)

        result: PathResult = planner.plan(env, pos, pos)

        assert result.success is True
        assert result.path == [pos]
        assert result.cost == 0.0
        assert result.nodes_expanded == 0

    def test_path_never_enters_occupied_cell(self, planner: AStarPlanner) -> None:
        """Invariant: No waypoint in the computed route may coincide with an obstacle."""
        env = GridEnvironment(rows=10, cols=10)
        # Place several obstacles
        obstacles = [(2, 2), (2, 3), (2, 4), (5, 5), (5, 6), (7, 1)]
        for r, c in obstacles:
            env.set_obstacle(r, c)

        start = (0, 0)
        goal = (8, 8)

        result = planner.plan(env, start, goal)

        assert result.success is True
        assert result.path is not None
        for r, c in result.path:
            assert env.is_obstacle(r, c) is False, f"Path entered obstacle at ({r}, {c})"

    def test_path_never_leaves_grid(self, planner: AStarPlanner) -> None:
        """Invariant: Every coordinate along the path must reside strictly within bounds."""
        env = GridEnvironment(rows=6, cols=6)
        start = (0, 0)
        goal = (5, 5)

        result = planner.plan(env, start, goal)

        assert result.success is True
        assert result.path is not None
        for r, c in result.path:
            assert 0 <= r < env.rows, f"Row {r} is outside grid boundaries"
            assert 0 <= c < env.cols, f"Column {c} is outside grid boundaries"

    def test_four_directional_movement_only(self, planner: AStarPlanner) -> None:
        """Invariant: Consecutive steps must only differ by 1 in row OR column, NEVER both (no diagonals)."""
        env = GridEnvironment(rows=8, cols=8)
        # Scatter obstacles to force turning
        env.set_obstacle(2, 2)
        env.set_obstacle(3, 2)
        env.set_obstacle(4, 2)

        start = (1, 1)
        goal = (5, 5)

        result = planner.plan(env, start, goal)
        assert result.success is True
        assert result.path is not None

        path = result.path
        for i in range(len(path) - 1):
            curr = path[i]
            nxt = path[i + 1]
            dr = abs(curr[0] - nxt[0])
            dc = abs(curr[1] - nxt[1])

            # Manhattan step must be exactly 1
            manhattan_step = dr + dc
            assert manhattan_step == 1, (
                f"Illegal transition from {curr} to {nxt}: "
                f"dr={dr}, dc={dc}, manhattan_step={manhattan_step} (diagonals forbidden)"
            )

    def test_start_inside_obstacle_fails_gracefully(self, planner: AStarPlanner) -> None:
        """Planner returns failed PathResult if start cell is occupied."""
        env = GridEnvironment(rows=5, cols=5)
        env.set_obstacle(1, 1)

        result = planner.plan(env, start=(1, 1), goal=(4, 4))
        assert result.success is False
        assert result.path is None
        assert "inside an obstacle" in (result.error_message or "")

    def test_goal_inside_obstacle_fails_gracefully(self, planner: AStarPlanner) -> None:
        """Planner returns failed PathResult if goal cell is occupied."""
        env = GridEnvironment(rows=5, cols=5)
        env.set_obstacle(4, 4)

        result = planner.plan(env, start=(0, 0), goal=(4, 4))
        assert result.success is False
        assert result.path is None
        assert "inside an obstacle" in (result.error_message or "")

    def test_out_of_bounds_positions_fail_gracefully(self, planner: AStarPlanner) -> None:
        """Planner returns failed PathResult if coordinates fall outside grid."""
        env = GridEnvironment(rows=5, cols=5)

        result_start = planner.plan(env, start=(-1, 0), goal=(3, 3))
        assert result_start.success is False

        result_goal = planner.plan(env, start=(0, 0), goal=(10, 10))
        assert result_goal.success is False


class TestNavigationAndReplanning:
    """Tests for integration between Robot, NavigationController, and dynamic replanning."""

    def test_robot_traversal_to_goal(self) -> None:
        """Robot successfully executes steps to reach goal."""
        env = GridEnvironment(rows=4, cols=4)
        planner = AStarPlanner()
        robot = Robot(initial_position=(0, 0))
        controller = NavigationController(env, planner, robot)

        result = controller.set_goal((0, 2))
        assert result.success is True
        assert robot.status == RobotStatus.NAVIGATING

        # Step 1
        status = controller.update_step()
        assert status == RobotStatus.NAVIGATING
        assert robot.position == (0, 1)

        # Step 2
        status = controller.update_step()
        assert status == RobotStatus.REACHED_GOAL
        assert robot.position == (0, 2)

    def test_dynamic_replanning_when_blocked(self) -> None:
        """When an obstacle is placed directly in front of the robot, navigation triggers replanning."""
        env = GridEnvironment(rows=5, cols=5)
        planner = AStarPlanner()
        robot = Robot(initial_position=(0, 0))
        controller = NavigationController(env, planner, robot)

        # Goal is (0, 3). Initial straight route is (0,0) -> (0,1) -> (0,2) -> (0,3)
        res = controller.set_goal((0, 3))
        assert res.success is True

        # Advance to (0, 1)
        controller.update_step()
        assert robot.position == (0, 1)

        # Place dynamic obstacle at (0, 2), directly in front of robot
        env.set_obstacle(0, 2)

        # Update cycle detects obstacle and replans detour through row 1
        status = controller.update_step()
        assert status == RobotStatus.NAVIGATING
        assert robot.position != (0, 2), "Robot must not step into obstacle"
        # Verify robot took a detour (e.g. moved down to (1, 1))
        assert robot.position == (1, 1)

    def test_known_optimal_lengths_on_small_maps(self) -> None:
        """Verify exact optimal path length and coordinates on small known grids."""
        planner = AStarPlanner()

        # Case 1: 2x2 open grid from (0,0) to (1,1). Known optimal cost is 2 steps (length 3).
        env2 = GridEnvironment(rows=2, cols=2)
        res2 = planner.plan(env2, start=(0, 0), goal=(1, 1))
        assert res2.success is True
        assert res2.cost == 2.0
        assert res2.path_length == 3

        # Case 2: 3x3 grid, start (0,0), goal (0,2), wall at (0,1).
        # Direct path is blocked. Optimal detour must go through row 1: cost 4, length 5.
        env3 = GridEnvironment(rows=3, cols=3)
        env3.set_obstacle(0, 1)
        res3 = planner.plan(env3, start=(0, 0), goal=(0, 2))
        assert res3.success is True
        assert res3.cost == 4.0
        assert res3.path_length == 5
        assert res3.path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]

    def test_start_equals_goal_navigation_controller_and_metrics(self) -> None:
        """Verify navigation controller and metrics when start == goal."""
        from src.metrics import MetricsTracker
        from src.simulation import Simulation

        env = GridEnvironment(rows=4, cols=4)
        sim = Simulation(env, start_position=(2, 2), goal_position=(2, 2))
        metrics = sim.run_headless()

        assert metrics.success is True
        assert metrics.final_robot_status == "REACHED_GOAL"
        assert metrics.steps_executed == 0
        assert metrics.initial_path_length == 1
        assert metrics.replan_count == 0

    def test_unreachable_goal_results_in_failure_state(self) -> None:
        """When the destination is completely boxed in, controller transitions robot to FAILED."""
        from src.metrics import MetricsTracker

        env = GridEnvironment(rows=4, cols=4)
        # Wall off corner (3, 3)
        env.set_obstacle(2, 3)
        env.set_obstacle(3, 2)

        robot = Robot(initial_position=(0, 0))
        metrics = MetricsTracker()
        controller = NavigationController(env, AStarPlanner(), robot, metrics)

        res = controller.set_goal((3, 3))
        assert res.success is False
        assert robot.status == RobotStatus.FAILED
        assert metrics.metrics.success is False
        assert metrics.metrics.final_robot_status == "FAILED"

    def test_replanning_when_all_alternate_routes_blocked(self) -> None:
        """When an obstacle blocks the active route and no detour exists, robot stops in BLOCKED state."""
        env = GridEnvironment(rows=3, cols=3)
        # Narrow corridor: only row 0 is open initially
        env.set_obstacle(1, 0)
        env.set_obstacle(1, 1)
        env.set_obstacle(1, 2)

        robot = Robot(initial_position=(0, 0))
        controller = NavigationController(env, AStarPlanner(), robot)
        res = controller.set_goal((0, 2))
        assert res.success is True

        # Advance to (0, 1)
        controller.update_step()
        assert robot.position == (0, 1)

        # Block the destination (0, 2) and the retreat cell (0, 0)
        env.set_obstacle(0, 2)
        env.set_obstacle(0, 0)

        # Step detects blockage, attempts replanning, finds no route, stops safely
        status = controller.update_step()
        assert status == RobotStatus.BLOCKED
        assert robot.position == (0, 1)
        assert robot.status == RobotStatus.BLOCKED


class TestDijkstraPlanner:
    """Test suite for DijkstraPlanner (uniform-cost search)."""

    @pytest.fixture
    def planner(self) -> DijkstraPlanner:
        return DijkstraPlanner()

    def test_dijkstra_finds_valid_route(self, planner: DijkstraPlanner) -> None:
        """Dijkstra planner successfully discovers optimal path around an obstacle."""
        env = GridEnvironment(rows=5, cols=5)
        env.set_obstacle(1, 2)
        env.set_obstacle(2, 2)
        env.set_obstacle(3, 2)

        start = (2, 0)
        goal = (2, 4)
        res = planner.plan(env, start, goal)

        assert res.success is True
        assert res.path is not None
        assert res.path[0] == start
        assert res.path[-1] == goal
        assert res.algorithm == "Dijkstra"
        assert res.cost == float(len(res.path) - 1)

    def test_dijkstra_start_equals_goal(self, planner: DijkstraPlanner) -> None:
        """Dijkstra handles start == goal with 0 cost."""
        env = GridEnvironment(rows=3, cols=3)
        res = planner.plan(env, (1, 1), (1, 1))
        assert res.success is True
        assert res.cost == 0.0
        assert res.path == [(1, 1)]

    def test_dijkstra_blocked_goal(self, planner: DijkstraPlanner) -> None:
        """Dijkstra returns failure when goal is blocked."""
        env = GridEnvironment(rows=3, cols=3)
        env.set_obstacle(0, 1)
        env.set_obstacle(1, 0)
        res = planner.plan(env, (0, 0), (2, 2))
        assert res.success is False
        assert res.path is None


class TestBFSPlanner:
    """Test suite for BFSPlanner (breadth-first search)."""

    @pytest.fixture
    def planner(self) -> BFSPlanner:
        return BFSPlanner()

    def test_bfs_finds_valid_route(self, planner: BFSPlanner) -> None:
        """BFS discovers valid path with minimum hop count."""
        env = GridEnvironment(rows=5, cols=5)
        env.set_obstacle(1, 2)
        env.set_obstacle(2, 2)
        env.set_obstacle(3, 2)

        start = (2, 0)
        goal = (2, 4)
        res = planner.plan(env, start, goal)

        assert res.success is True
        assert res.path is not None
        assert res.path[0] == start
        assert res.path[-1] == goal
        assert res.algorithm == "BFS"
        assert res.cost == float(len(res.path) - 1)

    def test_bfs_start_equals_goal(self, planner: BFSPlanner) -> None:
        """BFS handles start == goal with 0 cost."""
        env = GridEnvironment(rows=3, cols=3)
        res = planner.plan(env, (0, 0), (0, 0))
        assert res.success is True
        assert res.cost == 0.0
        assert res.path == [(0, 0)]

    def test_bfs_blocked_goal(self, planner: BFSPlanner) -> None:
        """BFS returns failure when goal is walled off."""
        env = GridEnvironment(rows=3, cols=3)
        env.set_obstacle(0, 1)
        env.set_obstacle(1, 0)
        res = planner.plan(env, (0, 0), (2, 2))
        assert res.success is False


class TestPlannerFactory:
    """Test suite for the planner factory get_planner()."""

    @pytest.mark.parametrize("name,expected_cls", [
        ("A*", AStarPlanner),
        ("astar", AStarPlanner),
        ("A-Star", AStarPlanner),
        ("Dijkstra", DijkstraPlanner),
        ("dijkstra", DijkstraPlanner),
        ("BFS", BFSPlanner),
        ("bfs", BFSPlanner),
    ])
    def test_get_planner_valid(self, name: str, expected_cls: type) -> None:
        planner = get_planner(name)
        assert isinstance(planner, expected_cls)

    def test_get_planner_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            get_planner("NonExistentAlgorithm")


class TestAlgorithmComparison:
    """Compare A*, Dijkstra, and BFS behavior on identical environments."""

    def test_all_algorithms_produce_same_optimal_cost(self) -> None:
        """A*, Dijkstra, and BFS must find routes with identical minimal cost."""
        env = GridEnvironment(rows=10, cols=10)
        # Place barrier with passage
        for r in range(2, 8):
            if r != 5:
                env.set_obstacle(r, 4)

        start = (1, 1)
        goal = (8, 8)

        astar_res = AStarPlanner().plan(env, start, goal)
        dijkstra_res = DijkstraPlanner().plan(env, start, goal)
        bfs_res = BFSPlanner().plan(env, start, goal)

        assert astar_res.success is True
        assert dijkstra_res.success is True
        assert bfs_res.success is True

        # All 3 algorithms must find identical minimal step cost
        assert astar_res.cost == dijkstra_res.cost
        assert dijkstra_res.cost == bfs_res.cost

        # A* should expand fewer or equal nodes than Dijkstra because of Manhattan heuristic
        assert astar_res.nodes_expanded <= dijkstra_res.nodes_expanded


