"""Configuration module for the Autonomous Navigation System.

Defines constants, data structures, and configuration presets for
grid dimensions, display parameters, robot speeds, and color schemes.
"""

from dataclasses import dataclass
from typing import Tuple


# Cell occupancy constants
FREE_CELL: int = 0
OCCUPIED_CELL: int = 1

# Four-directional cardinal moves: (delta_row, delta_column)
# Strictly 4-directional; diagonal movement is not permitted in this version.
CARDINAL_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),  # North / Up
    (1, 0),   # South / Down
    (0, -1),  # West / Left
    (0, 1),   # East / Right
)


@dataclass(frozen=True)
class GridConfig:
    """Configuration settings for the logical occupancy grid.

    Attributes:
        rows: Number of grid rows (height).
        cols: Number of grid columns (width).
        cell_size: Rendering size of each square cell in pixels.
    """

    rows: int = 20
    cols: int = 20
    cell_size: int = 32

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError(f"Grid rows must be positive, got {self.rows}")
        if self.cols <= 0:
            raise ValueError(f"Grid cols must be positive, got {self.cols}")
        if self.cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {self.cell_size}")

    @property
    def screen_width(self) -> int:
        """Calculate screen width in pixels based on columns and cell size."""
        return self.cols * self.cell_size

    @property
    def screen_height(self) -> int:
        """Calculate screen height in pixels based on rows and cell size."""
        return self.rows * self.cell_size


@dataclass(frozen=True)
class ColorConfig:
    """RGB color definitions for Pygame visualization."""

    BACKGROUND: Tuple[int, int, int] = (245, 247, 250)
    GRID_LINE: Tuple[int, int, int] = (218, 224, 233)
    OBSTACLE: Tuple[int, int, int] = (44, 62, 80)
    FREE: Tuple[int, int, int] = (255, 255, 255)
    START: Tuple[int, int, int] = (46, 204, 113)
    GOAL: Tuple[int, int, int] = (231, 76, 60)
    PATH: Tuple[int, int, int] = (52, 152, 219)
    PATH_LINE: Tuple[int, int, int] = (41, 128, 185)
    ROBOT: Tuple[int, int, int] = (155, 89, 182)
    VISITED: Tuple[int, int, int] = (235, 245, 251)
    PANEL_BG: Tuple[int, int, int] = (30, 39, 46)
    TEXT_COLOR: Tuple[int, int, int] = (236, 240, 241)


@dataclass
class SimulationConfig:
    """Master simulation configuration aggregating grid, display, and timing.

    Attributes:
        grid: Grid dimension and scaling configuration.
        colors: Color palette configuration.
        fps: Target frames per second for Pygame rendering loop.
        robot_step_delay: Seconds between logical robot steps along route.
        window_title: Pygame window caption.
    """

    grid: GridConfig = GridConfig()
    colors: ColorConfig = ColorConfig()
    fps: int = 60
    robot_step_delay: float = 0.15
    window_title: str = "AI Autonomous Navigation System (2D Grid Simulation)"
