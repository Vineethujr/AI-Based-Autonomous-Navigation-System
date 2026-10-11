"""Performance metrics module for autonomous navigation evaluation.

Records timing, search expansions, trajectory length, step efficiency,
and replanning events. Provides summary reporting and JSON serialization.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional


@dataclass
class ReplanRecord:
    """Details of a single replanning event."""

    timestamp: float
    trigger_position: tuple
    new_path_length: int
    replan_duration_ms: float
    nodes_expanded: int
    success: bool


@dataclass
class NavigationMetrics:
    """Aggregated numerical metrics for a single navigation trial.

    Attributes:
        success: True if the robot successfully reached the goal.
        initial_planning_time_ms: Milliseconds spent on initial A* search.
        initial_path_length: Number of cells in initial route.
        initial_nodes_expanded: Search nodes expanded during initial planning.
        steps_executed: Total physical steps taken by the robot.
        replan_count: Number of dynamic replanning events triggered.
        total_replan_time_ms: Cumulative duration of all replanning computations.
        total_execution_time_ms: Total runtime from start of trial to termination.
        final_robot_status: String representation of final RobotStatus.
    """

    success: bool = False
    initial_planning_time_ms: float = 0.0
    initial_path_length: int = 0
    initial_nodes_expanded: int = 0
    steps_executed: int = 0
    replan_count: int = 0
    total_replan_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    final_robot_status: str = "IDLE"
    replan_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def path_efficiency_ratio(self) -> float:
        """Ratio of theoretical initial path length to actual steps executed.

        Ideal is 1.0; < 1.0 indicates detours due to obstacles/replanning.
        """
        if self.steps_executed == 0:
            return 1.0 if self.initial_path_length <= 1 else 0.0
        # Optimal steps would be initial_path_length - 1
        optimal_steps = max(1, self.initial_path_length - 1)
        return round(optimal_steps / self.steps_executed, 3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a Python dictionary."""
        data = asdict(self)
        data["path_efficiency_ratio"] = self.path_efficiency_ratio
        return data


class MetricsTracker:
    """Manages recording, computing, and exporting navigation performance metrics."""

    def __init__(self) -> None:
        """Initialize a fresh MetricsTracker session."""
        self.metrics = NavigationMetrics()
        self._trial_start_time: Optional[float] = None

    def start_trial(self) -> None:
        """Record the start of a navigation trial."""
        self.metrics = NavigationMetrics()
        self._trial_start_time = time.perf_counter()

    def record_initial_plan(
        self,
        planning_time_ms: float,
        path_length: int,
        nodes_expanded: int,
        success: bool,
    ) -> None:
        """Record metrics from the initial A* planning phase."""
        self.metrics.initial_planning_time_ms = round(planning_time_ms, 3)
        self.metrics.initial_path_length = path_length
        self.metrics.initial_nodes_expanded = nodes_expanded
        self.metrics.success = success

    def record_step(self) -> None:
        """Increment count of discrete steps executed by the robot."""
        self.metrics.steps_executed += 1

    def record_replan(
        self,
        position: tuple,
        duration_ms: float,
        nodes_expanded: int,
        new_path_length: int,
        success: bool,
    ) -> None:
        """Record a dynamic replanning occurrence."""
        self.metrics.replan_count += 1
        self.metrics.total_replan_time_ms += duration_ms
        self.metrics.total_replan_time_ms = round(self.metrics.total_replan_time_ms, 3)
        record = ReplanRecord(
            timestamp=round(time.time(), 3),
            trigger_position=position,
            new_path_length=new_path_length,
            replan_duration_ms=round(duration_ms, 3),
            nodes_expanded=nodes_expanded,
            success=success,
        )
        self.metrics.replan_history.append(asdict(record))

    def end_trial(self, final_status: str, success: bool) -> NavigationMetrics:
        """Finalize the trial recording and compute total execution time.

        Args:
            final_status: String name of final robot status.
            success: Whether the trial achieved its goal.

        Returns:
            Completed NavigationMetrics instance.
        """
        if self._trial_start_time is not None:
            self.metrics.total_execution_time_ms = round(
                (time.perf_counter() - self._trial_start_time) * 1000.0, 3
            )
        self.metrics.final_robot_status = final_status
        self.metrics.success = success
        return self.metrics

    def export_json(self, filepath: str) -> None:
        """Save metrics summary to a JSON file.

        Args:
            filepath: Target file path (creates parent directories if needed).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

    def generate_report(self) -> str:
        """Generate a human-readable text report of the session metrics."""
        m = self.metrics
        lines = [
            "=" * 50,
            "      AUTONOMOUS NAVIGATION PERFORMANCE REPORT    ",
            "=" * 50,
            f"Goal Reached:             {'YES' if m.success else 'NO'}",
            f"Final Robot Status:       {m.final_robot_status}",
            f"Initial Path Length:      {m.initial_path_length} cells",
            f"Initial Planning Time:    {m.initial_planning_time_ms:.3f} ms",
            f"Initial Nodes Expanded:   {m.initial_nodes_expanded}",
            f"Steps Executed:           {m.steps_executed}",
            f"Replan Count:             {m.replan_count}",
            f"Total Replan Time:        {m.total_replan_time_ms:.3f} ms",
            f"Total Execution Time:     {m.total_execution_time_ms:.3f} ms",
            f"Path Efficiency Ratio:    {m.path_efficiency_ratio:.3f}",
            "=" * 50,
        ]
        return "\n".join(lines)
