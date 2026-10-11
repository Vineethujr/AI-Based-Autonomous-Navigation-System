"""Environment module for 2D Grid Occupancy Map Representation.

Provides the GridEnvironment class handling map initialization, boundary checks,
occupancy queries, obstacle modification, and coordinate conversions between
logical grid space (row, column) and screen space (x, y).
"""

from typing import List, Tuple, Optional
import numpy as np

from src.config import FREE_CELL, OCCUPIED_CELL, CARDINAL_DIRECTIONS


class EnvironmentError(Exception):
    """Base exception for environment-related errors."""


class InvalidGridError(EnvironmentError):
    """Raised when an occupancy grid is malformed, has invalid dimensions, or illegal values."""


class OutOfBoundsError(EnvironmentError):
    """Raised when coordinates fall outside the logical grid boundaries."""


class ObstacleCollisionError(EnvironmentError):
    """Raised when start, goal, or waypoint coincides with an occupied obstacle cell."""


class GridEnvironment:
    """Manages the 2D occupancy grid map and spatial queries.

    Data conventions:
        - Logical coordinates are strictly (row, column) integers.
        - rows index from 0 to height - 1 (vertical axis, top to bottom).
        - columns index from 0 to width - 1 (horizontal axis, left to right).
        - Cell value 0 represents a FREE cell.
        - Cell value 1 represents an OCCUPIED obstacle cell.

    Attributes:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        grid: 2D NumPy ndarray of dtype np.int8 containing 0s and 1s.
    """

    def __init__(self, rows: int, cols: int, initial_grid: Optional[np.ndarray] = None) -> None:
        """Initialize the GridEnvironment.

        Args:
            rows: Number of grid rows (must be > 0).
            cols: Number of grid columns (must be > 0).
            initial_grid: Optional 2D NumPy array with values in {0, 1} matching (rows, cols).

        Raises:
            InvalidGridError: If dimensions are <= 0 or provided grid is malformed.
        """
        if rows <= 0 or cols <= 0:
            raise InvalidGridError(f"Grid dimensions must be positive integers, got rows={rows}, cols={cols}")

        self.rows: int = int(rows)
        self.cols: int = int(cols)

        if initial_grid is not None:
            self.grid: np.ndarray = self._validate_and_format_grid(initial_grid)
        else:
            self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)

    def _validate_and_format_grid(self, grid_input: np.ndarray) -> np.ndarray:
        """Validate input array dimensions and cell values.

        Args:
            grid_input: NumPy array or array-like to validate.

        Returns:
            Formatted np.ndarray of np.int8.

        Raises:
            InvalidGridError: If dimensions or cell values are invalid.
        """
        arr = np.asarray(grid_input)

        if arr.ndim != 2:
            raise InvalidGridError(f"Grid must be a 2-dimensional array, got shape {arr.shape} ({arr.ndim}D)")

        if arr.shape != (self.rows, self.cols):
            raise InvalidGridError(
                f"Grid shape mismatch: expected ({self.rows}, {self.cols}), got {arr.shape}"
            )

        # Check for non-integer or unexpected numeric values
        unique_vals = np.unique(arr)
        for val in unique_vals:
            if val not in (FREE_CELL, OCCUPIED_CELL):
                raise InvalidGridError(
                    f"Unsupported cell value '{val}'. Only {FREE_CELL} (free) and {OCCUPIED_CELL} (occupied) are allowed."
                )

        return arr.astype(np.int8, copy=True)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "GridEnvironment":
        """Factory method to construct an environment directly from a 2D numpy array.

        Args:
            array: 2D array of {0, 1}.

        Returns:
            A new GridEnvironment instance.
        """
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise InvalidGridError(f"Array must be 2D, got ndim={arr.ndim}")
        rows, cols = arr.shape
        return cls(rows=rows, cols=cols, initial_grid=arr)

    def is_in_bounds(self, row: int, col: int) -> bool:
        """Check if logical coordinate (row, col) is within grid bounds.

        Args:
            row: Logical row index.
            col: Logical column index.

        Returns:
            True if inside grid boundaries, False otherwise.
        """
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_obstacle(self, row: int, col: int) -> bool:
        """Check if cell (row, col) contains an obstacle.

        Args:
            row: Logical row index.
            col: Logical column index.

        Returns:
            True if cell is occupied (1) or out of bounds, False if free (0).
        """
        if not self.is_in_bounds(row, col):
            return True
        return bool(self.grid[row, col] == OCCUPIED_CELL)

    def is_free(self, row: int, col: int) -> bool:
        """Check if cell (row, col) is within bounds and free of obstacles.

        Args:
            row: Logical row index.
            col: Logical column index.

        Returns:
            True if valid and free (0), False otherwise.
        """
        if not self.is_in_bounds(row, col):
            return False
        return bool(self.grid[row, col] == FREE_CELL)

    def set_obstacle(self, row: int, col: int) -> None:
        """Mark cell (row, col) as occupied by an obstacle.

        Args:
            row: Logical row index.
            col: Logical column index.

        Raises:
            OutOfBoundsError: If (row, col) is outside the grid.
        """
        if not self.is_in_bounds(row, col):
            raise OutOfBoundsError(f"Cannot set obstacle at out-of-bounds position ({row}, {col})")
        self.grid[row, col] = OCCUPIED_CELL

    def clear_cell(self, row: int, col: int) -> None:
        """Mark cell (row, col) as free.

        Args:
            row: Logical row index.
            col: Logical column index.

        Raises:
            OutOfBoundsError: If (row, col) is outside the grid.
        """
        if not self.is_in_bounds(row, col):
            raise OutOfBoundsError(f"Cannot clear cell at out-of-bounds position ({row}, {col})")
        self.grid[row, col] = FREE_CELL

    def validate_position(self, pos: Tuple[int, int], name: str = "Position") -> None:
        """Validate that a position is strictly within bounds and not on an obstacle.

        Args:
            pos: (row, col) tuple.
            name: Identifier for error messaging (e.g. 'Start', 'Goal').

        Raises:
            OutOfBoundsError: If position is outside grid boundaries.
            ObstacleCollisionError: If position is on an obstacle.
        """
        row, col = pos
        if not self.is_in_bounds(row, col):
            raise OutOfBoundsError(
                f"{name} ({row}, {col}) is out of bounds for grid shape ({self.rows}, {self.cols})"
            )
        if self.is_obstacle(row, col):
            raise ObstacleCollisionError(
                f"{name} ({row}, {col}) is located inside an obstacle"
            )

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Retrieve valid, traversable 4-directional adjacent neighbors.

        Movement is strictly 4-directional (North, South, West, East).
        Diagonal moves are prohibited in this architecture.

        Args:
            row: Center cell row.
            col: Center cell column.

        Returns:
            List of (r, c) neighboring coordinates that are in bounds and free.
        """
        neighbors: List[Tuple[int, int]] = []
        for dr, dc in CARDINAL_DIRECTIONS:
            nr, nc = row + dr, col + dc
            if self.is_free(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def copy(self) -> "GridEnvironment":
        """Return a deep copy of the environment."""
        return GridEnvironment(self.rows, self.cols, self.grid.copy())

    @staticmethod
    def grid_to_screen(row: int, col: int, cell_size: int) -> Tuple[int, int]:
        """Convert logical grid coordinate (row, column) to screen coordinate (x, y) top-left.

        Conversion convention:
            screen_x = col * cell_size
            screen_y = row * cell_size

        Args:
            row: Grid row index (vertical).
            col: Grid column index (horizontal).
            cell_size: Size of square cell in pixels.

        Returns:
            (x, y) pixel coordinates of the cell's top-left corner.
        """
        return col * cell_size, row * cell_size

    @staticmethod
    def grid_to_screen_center(row: int, col: int, cell_size: int) -> Tuple[int, int]:
        """Convert logical grid coordinate (row, column) to center pixel coordinate (x, y).

        Args:
            row: Grid row index.
            col: Grid column index.
            cell_size: Size of square cell in pixels.

        Returns:
            (center_x, center_y) pixel coordinates.
        """
        x = col * cell_size + cell_size // 2
        y = row * cell_size + cell_size // 2
        return x, y

    @staticmethod
    def screen_to_grid(x: int, y: int, cell_size: int) -> Tuple[int, int]:
        """Convert screen coordinate (x, y) in pixels to logical grid coordinate (row, column).

        Conversion convention:
            col = x // cell_size
            row = y // cell_size

        Args:
            x: Screen pixel x coordinate (horizontal).
            y: Screen pixel y coordinate (vertical).
            cell_size: Size of square cell in pixels.

        Returns:
            (row, col) logical grid coordinate.
        """
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        col = int(x) // cell_size
        row = int(y) // cell_size
        return row, col
