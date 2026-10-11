"""Navigation module coordinating route execution, path monitoring, and replanning.

Acts as the middle-tier controller linking the logical environment, the A* planner,
the robot state machine, and metrics tracking. Decoupled from rendering.
"""

from typing import List, Optional, Tuple

from src.environment import GridEnvironment
from src.path_planning import BasePlanner, AStarPlanner, PathResult
from src.robot import Robot, RobotStatus
from src.metrics import MetricsTracker


class NavigationController:
    """Coordinates route planning, active path monitoring, and dynamic replanning.

    Attributes:
        environment: GridEnvironment containing occupancy data.
        planner: BasePlanner instance used to compute collision-free routes.
        robot: Robot instance tracking physical/logical location.
        metrics: Optional MetricsTracker recording trial events.
        goal: Target (row, column) coordinate.
    """

    def __init__(
        self,
        environment: GridEnvironment,
        planner: BasePlanner,
        robot: Robot,
        metrics: Optional[MetricsTracker] = None,
    ) -> None:
        """Initialize the NavigationController.

        Args:
            environment: Active occupancy grid map.
            planner: Path planning algorithm engine (A*, Dijkstra, BFS).
            robot: Physical/simulated robot entity.
            metrics: Optional performance metrics tracker.
        """
        self.environment: GridEnvironment = environment
        self.planner: BasePlanner = planner
        self.robot: Robot = robot
        self.metrics: Optional[MetricsTracker] = metrics
        self.goal: Optional[Tuple[int, int]] = None

    def set_goal(self, goal: Tuple[int, int]) -> PathResult:
        """Set a navigation destination and plan an initial route from robot position.

        Args:
            goal: Target (row, column) coordinates.

        Returns:
            PathResult of the initial planning attempt.
        """
        self.goal = (int(goal[0]), int(goal[1]))

        if self.metrics:
            self.metrics.start_trial()

        result = self.planner.plan(self.environment, self.robot.position, self.goal)

        if result.success and result.path is not None:
            self.robot.set_path(result.path)
            if self.metrics:
                self.metrics.record_initial_plan(
                    planning_time_ms=result.execution_time_ms,
                    path_length=result.path_length,
                    nodes_expanded=result.nodes_expanded,
                    success=True,
                )
                if self.robot.status == RobotStatus.REACHED_GOAL:
                    self.metrics.end_trial(final_status=self.robot.status.value, success=True)
        else:
            self.robot.set_path([])
            self.robot.status = RobotStatus.FAILED
            if self.metrics:
                self.metrics.record_initial_plan(
                    planning_time_ms=result.execution_time_ms,
                    path_length=0,
                    nodes_expanded=result.nodes_expanded,
                    success=False,
                )
                self.metrics.end_trial(final_status=self.robot.status.value, success=False)

        return result

    def is_route_clear(self) -> bool:
        """Inspect the remaining unvisited waypoints along the robot's current path.

        Returns:
            True if all remaining waypoints remain free of obstacles, False otherwise.
        """
        if not self.robot.path or self.robot.current_step_index >= len(self.robot.path):
            return True

        remaining_waypoints = self.robot.path[self.robot.current_step_index :]
        for r, c in remaining_waypoints:
            if self.environment.is_obstacle(r, c):
                return False
        return True

    def replan(self) -> PathResult:
        """Attempt to compute an alternative path from the current robot position to the goal.

        Returns:
            PathResult representing the replanning outcome.
        """
        if self.goal is None:
            return PathResult(
                success=False,
                error_message="Cannot replan: No active navigation goal has been set.",
            )

        result = self.planner.plan(self.environment, self.robot.position, self.goal)

        if result.success and result.path is not None:
            self.robot.set_path(result.path)
            if self.metrics:
                self.metrics.record_replan(
                    position=self.robot.position,
                    duration_ms=result.execution_time_ms,
                    nodes_expanded=result.nodes_expanded,
                    new_path_length=result.path_length,
                    success=True,
                )
        else:
            self.robot.set_path([])
            self.robot.stop_blocked()
            if self.metrics:
                self.metrics.record_replan(
                    position=self.robot.position,
                    duration_ms=result.execution_time_ms,
                    nodes_expanded=result.nodes_expanded,
                    new_path_length=0,
                    success=False,
                )
                self.metrics.end_trial(
                    final_status=self.robot.status.value, success=False
                )

        return result

    def update_step(self) -> RobotStatus:
        """Execute a single navigation cycle.

        1. Checks whether the next cell is blocked by an obstacle.
        2. If blocked, triggers dynamic replanning.
        3. If a clear path exists, advances the robot one step.
        4. Updates status and records step metrics.

        Returns:
            Current RobotStatus after the cycle.
        """
        if self.robot.status in (RobotStatus.IDLE, RobotStatus.REACHED_GOAL, RobotStatus.FAILED):
            return self.robot.status

        # Check if next waypoint is blocked
        next_pos = self.robot.peek_next_position()
        if next_pos is not None and self.environment.is_obstacle(next_pos[0], next_pos[1]):
            # Dynamic blockage detected -> Replan!
            replan_res = self.replan()
            if not replan_res.success:
                if self.metrics:
                    self.metrics.end_trial(final_status=self.robot.status.value, success=False)
                return self.robot.status

        # If still navigating, advance one step
        if self.robot.status == RobotStatus.NAVIGATING:
            moved = self.robot.step_forward()
            if moved and self.metrics:
                self.metrics.record_step()

            if self.robot.status == RobotStatus.REACHED_GOAL and self.metrics:
                self.metrics.end_trial(final_status=self.robot.status.value, success=True)

        return self.robot.status
