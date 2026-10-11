"""Unit tests for the GridEnvironment module.

Verifies grid construction, dimension validation, cell value constraints,
boundary checks, obstacle querying, coordinate transformations, and position validation.
"""

import numpy as np
import pytest

from src.environment import (
    GridEnvironment,
    InvalidGridError,
    OutOfBoundsError,
    ObstacleCollisionError,
)
from src.config import FREE_CELL, OCCUPIED_CELL


class TestGridCreation:
    """Tests for grid initialization and shape validation."""

    def test_valid_grid_creation_default(self) -> None:
        """Test creating an empty grid with valid positive dimensions."""
        env = GridEnvironment(rows=10, cols=15)
        assert env.rows == 10
        assert env.cols == 15
        assert env.grid.shape == (10, 15)
        assert np.all(env.grid == FREE_CELL)

    def test_valid_grid_from_array(self) -> None:
        """Test constructing an environment from an existing 2D NumPy array."""
        custom_grid = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.int8)
        env = GridEnvironment.from_array(custom_grid)
        assert env.rows == 3
        assert env.cols == 3
        assert env.is_obstacle(0, 1) is True
        assert env.is_free(0, 0) is True

    @pytest.mark.parametrize("rows,cols", [
        (0, 10),
        (-5, 10),
        (10, 0),
        (10, -3),
        (-1, -1),
    ])
    def test_invalid_grid_dimensions_raise_error(self, rows: int, cols: int) -> None:
        """Reject non-positive row or column counts."""
        with pytest.raises(InvalidGridError):
            GridEnvironment(rows=rows, cols=cols)

    def test_invalid_grid_non_2d_raises_error(self) -> None:
        """Reject 1D or 3D arrays."""
        arr_1d = np.array([0, 1, 0])
        with pytest.raises(InvalidGridError):
            GridEnvironment.from_array(arr_1d)

        arr_3d = np.zeros((3, 3, 3))
        with pytest.raises(InvalidGridError):
            GridEnvironment.from_array(arr_3d)

    @pytest.mark.parametrize("bad_val", [2, -1, 99, 10])
    def test_invalid_cell_values_raise_error(self, bad_val: int) -> None:
        """Reject occupancy arrays containing values outside {0, 1}."""
        bad_grid = np.array([
            [0, bad_val],
            [1, 0],
        ])
        with pytest.raises(InvalidGridError):
            GridEnvironment(rows=2, cols=2, initial_grid=bad_grid)


class TestPositionValidation:
    """Tests for validating coordinates against boundaries and obstacles."""

    @pytest.fixture
    def sample_env(self) -> GridEnvironment:
        env = GridEnvironment(rows=5, cols=5)
        env.set_obstacle(2, 2)
        return env

    def test_valid_position(self, sample_env: GridEnvironment) -> None:
        """Legitimate free coordinates should pass validation without error."""
        sample_env.validate_position((0, 0))
        sample_env.validate_position((4, 4))
        sample_env.validate_position((1, 2))

    @pytest.mark.parametrize("out_pos", [
        (-1, 0),
        (0, -1),
        (5, 2),
        (2, 5),
        (10, 10),
        (-5, -5),
    ])
    def test_out_of_bounds_start_and_goal(self, sample_env: GridEnvironment, out_pos: tuple) -> None:
        """Reject out-of-bounds start and goal coordinates."""
        with pytest.raises(OutOfBoundsError):
            sample_env.validate_position(out_pos, name="Start")

        with pytest.raises(OutOfBoundsError):
            sample_env.validate_position(out_pos, name="Goal")

    def test_start_inside_obstacle(self, sample_env: GridEnvironment) -> None:
        """Reject start coordinates placed on occupied obstacle cells."""
        with pytest.raises(ObstacleCollisionError):
            sample_env.validate_position((2, 2), name="Start")

    def test_goal_inside_obstacle(self, sample_env: GridEnvironment) -> None:
        """Reject goal coordinates placed on occupied obstacle cells."""
        with pytest.raises(ObstacleCollisionError):
            sample_env.validate_position((2, 2), name="Goal")


class TestObstacleOperations:
    """Tests for mutating and querying cell occupancy."""

    def test_set_and_clear_obstacle(self) -> None:
        env = GridEnvironment(rows=4, cols=4)
        assert env.is_free(1, 1) is True

        env.set_obstacle(1, 1)
        assert env.is_obstacle(1, 1) is True
        assert env.is_free(1, 1) is False

        env.clear_cell(1, 1)
        assert env.is_free(1, 1) is True
        assert env.is_obstacle(1, 1) is False

    def test_set_obstacle_out_of_bounds_raises(self) -> None:
        env = GridEnvironment(rows=4, cols=4)
        with pytest.raises(OutOfBoundsError):
            env.set_obstacle(10, 10)

    def test_clear_cell_out_of_bounds_raises(self) -> None:
        env = GridEnvironment(rows=4, cols=4)
        with pytest.raises(OutOfBoundsError):
            env.clear_cell(-1, 0)

    def test_get_neighbors_respects_boundaries_and_obstacles(self) -> None:
        env = GridEnvironment(rows=3, cols=3)
        # In a 3x3 grid, corner (0, 0) should have only 2 neighbors: (1, 0) and (0, 1)
        neighbors = env.get_neighbors(0, 0)
        assert set(neighbors) == {(1, 0), (0, 1)}

        # Center (1, 1) should have 4 neighbors
        center_neighbors = env.get_neighbors(1, 1)
        assert set(center_neighbors) == {(0, 1), (2, 1), (1, 0), (1, 2)}

        # Add obstacle at (1, 2)
        env.set_obstacle(1, 2)
        center_neighbors_after = env.get_neighbors(1, 1)
        assert (1, 2) not in center_neighbors_after
        assert set(center_neighbors_after) == {(0, 1), (2, 1), (1, 0)}


class TestCoordinateTransforms:
    """Tests for conversions between logical grid (row, col) and screen space (x, y)."""

    def test_grid_to_screen(self) -> None:
        cell_size = 32
        x, y = GridEnvironment.grid_to_screen(row=3, col=5, cell_size=cell_size)
        assert x == 5 * 32  # column corresponds to x
        assert y == 3 * 32  # row corresponds to y

    def test_screen_to_grid(self) -> None:
        cell_size = 32
        row, col = GridEnvironment.screen_to_grid(x=160, y=96, cell_size=cell_size)
        assert col == 160 // 32  # 5
        assert row == 96 // 32   # 3

    def test_grid_to_screen_center(self) -> None:
        cell_size = 40
        cx, cy = GridEnvironment.grid_to_screen_center(row=2, col=3, cell_size=cell_size)
        assert cx == 3 * 40 + 20
        assert cy == 2 * 40 + 20
