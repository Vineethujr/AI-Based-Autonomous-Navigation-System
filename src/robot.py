"""Robot module defining state, heading, and discrete movement primitives.

Decoupled from path planning and rendering. The Robot tracks its own position,
target route, history of traversed cells, and current operational status.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class RobotStatus(str, Enum):
    """Operational states of the autonomous robot."""

    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    REACHED_GOAL = "REACHED_GOAL"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class Robot:
    """Represents the simulated autonomous robot in the 2D grid environment.

    Attributes:
        position: Current logical (row, column) coordinate.
        status: Current RobotStatus enum.
        path: Planned sequence of (row, col) waypoints to follow.
        current_step_index: Index in path representing the next waypoint.
        history: Chronological list of logical positions visited by the robot.
    """

    def __init__(self, initial_position: Tuple[int, int]) -> None:
        """Initialize the Robot.

        Args:
            initial_position: Starting (row, column) coordinate.
        """
        self.position: Tuple[int, int] = (int(initial_position[0]), int(initial_position[1]))
        self.status: RobotStatus = RobotStatus.IDLE
        self.path: List[Tuple[int, int]] = []
        self.current_step_index: int = 0
        self.history: List[Tuple[int, int]] = [self.position]

    @property
    def row(self) -> int:
        """Current logical row coordinate."""
        return self.position[0]

    @property
    def col(self) -> int:
        """Current logical column coordinate."""
        return self.position[1]

    def set_path(self, path: List[Tuple[int, int]]) -> None:
        """Assign a newly planned path to the robot.

        Args:
            path: Sequential list of (row, col) coordinates starting from robot's position.
        """
        if not path:
            self.path = []
            self.current_step_index = 0
            self.status = RobotStatus.FAILED
            return

        self.path = list(path)
        # If the first waypoint is the robot's current position, next target is index 1
        if self.path[0] == self.position and len(self.path) > 1:
            self.current_step_index = 1
        elif len(self.path) == 1 and self.path[0] == self.position:
            self.current_step_index = 0
            self.status = RobotStatus.REACHED_GOAL
            return
        else:
            self.current_step_index = 0

        self.status = RobotStatus.NAVIGATING

    def step_forward(self) -> bool:
        """Advance the robot to the next waypoint along its assigned path.

        Returns:
            True if moved successfully, False if already at goal or no path.
        """
        if self.status != RobotStatus.NAVIGATING:
            return False

        if not self.path or self.current_step_index >= len(self.path):
            self.status = RobotStatus.REACHED_GOAL
            return False

        next_pos = self.path[self.current_step_index]
        self.position = next_pos
        self.history.append(self.position)
        self.current_step_index += 1

        if self.current_step_index >= len(self.path):
            self.status = RobotStatus.REACHED_GOAL

        return True

    def peek_next_position(self) -> Optional[Tuple[int, int]]:
        """Look ahead at the next planned waypoint without advancing.

        Returns:
            Next (row, col) coordinate if available, or None.
        """
        if not self.path or self.current_step_index >= len(self.path):
            return None
        return self.path[self.current_step_index]

    def stop_blocked(self) -> None:
        """Set robot status to BLOCKED when an obstacle impedes the route."""
        self.status = RobotStatus.BLOCKED

    def reset(self, new_position: Optional[Tuple[int, int]] = None) -> None:
        """Reset robot state and route memory.

        Args:
            new_position: Optional new starting position. If None, resets to original position.
        """
        if new_position is not None:
            self.position = (int(new_position[0]), int(new_position[1]))
        elif self.history:
            self.position = self.history[0]

        self.path = []
        self.current_step_index = 0
        self.history = [self.position]
        self.status = RobotStatus.IDLE


@dataclass(frozen=True)
class RobotState:
    """Immutable snapshot of the robot's physical and operational state.

    Useful for metrics logging, visualization frames, and replay streams.
    """

    position: Tuple[int, int]
    status: RobotStatus
    step_index: int
    total_steps_in_path: int
    traversed_count: int
