"""AI-Based Autonomous Navigation System in a 2D Simulation Environment.

Modular package architecture separating environment, path planning,
robot state, navigation control, metrics, and visualization.
"""

from src.config import SimulationConfig, GridConfig
from src.environment import GridEnvironment
from src.path_planning import (
    BasePlanner,
    AStarPlanner,
    DijkstraPlanner,
    BFSPlanner,
    PathResult,
    get_planner,
    SUPPORTED_ALGORITHMS,
)
from src.robot import Robot, RobotState, RobotStatus
from src.navigation import NavigationController
from src.metrics import MetricsTracker

__all__ = [
    "SimulationConfig",
    "GridConfig",
    "GridEnvironment",
    "BasePlanner",
    "AStarPlanner",
    "DijkstraPlanner",
    "BFSPlanner",
    "PathResult",
    "get_planner",
    "SUPPORTED_ALGORITHMS",
    "Robot",
    "RobotState",
    "RobotStatus",
    "NavigationController",
    "MetricsTracker",
]
