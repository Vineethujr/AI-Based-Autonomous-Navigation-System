"""Integration tests for Streamlit helper logic and preset environments.

Tests:
1. Preset environment creation and boundary properties.
2. Matplotlib figure rendering for clean Streamlit display without memory leaks.
3. Verification that app module components import cleanly.
4. Strict JSON serialization for dashboard and file metrics exports.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import app
from app import (
    add_dynamic_obstacle,
    apply_grid_click,
    advance_navigation,
    clear_obstacles,
    clear_navigation_state,
    create_preset_environment,
    render_grid_figure,
    render_interactive_grid,
    _selected_grid_cell,
    set_navigation_playing,
    start_navigation,
)
from src.environment import GridEnvironment
from src.metrics import MetricsTracker, NavigationMetrics
from src.path_planning import AStarPlanner, get_planner
from src.robot import RobotStatus


class TestStreamlitPresets:
    """Test preset map generation used in the Streamlit app."""

    @pytest.mark.parametrize("preset_name", [
        "Open Field",
        "Random Obstacles",
        "Maze-like Environment",
        "U-Shaped Obstacle Trap",
        "Unreachable Goal",
        "Demo Barriers / Maze (Default)",
        "Open Grid (No Obstacles)",
        "Narrow Corridor",
        "U-Shaped Trap / Local Minimum",
        "Dense Random Clutter",
        "Walled-Off Goal (Unreachable Test)",
    ])
    def test_presets_generate_valid_environments(self, preset_name: str) -> None:
        """All presets must generate environments with valid dimensions and start/goal positions."""
        rows, cols = 15, 15
        env, start, goal = create_preset_environment(preset_name, rows=rows, cols=cols)

        assert isinstance(env, GridEnvironment)
        assert env.rows == rows
        assert env.cols == cols
        assert env.is_in_bounds(start[0], start[1])
        assert env.is_in_bounds(goal[0], goal[1])
        env.validate_position(start, name="Start")
        env.validate_position(goal, name="Goal")

    def test_demo_barrier_has_valid_route(self) -> None:
        """Default preset produces a solvable scenario with A*."""
        env, start, goal = create_preset_environment(
            "Demo Barriers / Maze (Default)", rows=20, cols=20
        )
        planner = AStarPlanner()
        res = planner.plan(env, start, goal)

        assert res.success is True
        assert res.path is not None
        assert res.path_length > 0

    def test_walled_off_preset_is_unsolvable(self) -> None:
        """Walled-off goal preset correctly fails route planning."""
        env, start, goal = create_preset_environment(
            "Walled-Off Goal (Unreachable Test)", rows=15, cols=15
        )
        planner = AStarPlanner()
        res = planner.plan(env, start, goal)

        assert res.success is False
        assert res.path is None

    def test_unreachable_goal_scenario_is_unreachable(self) -> None:
        env, start, goal = create_preset_environment(
            "Unreachable Goal", rows=15, cols=15, seed=42
        )

        result = AStarPlanner().plan(env, start, goal)

        assert result.success is False
        assert result.path is None

    def test_open_field_has_no_obstacles(self) -> None:
        env, start, goal = create_preset_environment(
            "Open Field", rows=12, cols=12, seed=44, obstacle_density=0.8
        )

        assert not np.any(env.grid)
        env.validate_position(start, name="Start")
        env.validate_position(goal, name="Goal")

    def test_random_obstacles_are_reproducible_and_seeded(self) -> None:
        first, _, _ = create_preset_environment(
            "Random Obstacles", rows=20, cols=20, seed=42, obstacle_density=0.3
        )
        same_seed, _, _ = create_preset_environment(
            "Random Obstacles", rows=20, cols=20, seed=42, obstacle_density=0.3
        )
        different_seed, _, _ = create_preset_environment(
            "Random Obstacles", rows=20, cols=20, seed=43, obstacle_density=0.3
        )

        assert np.array_equal(first.grid, same_seed.grid)
        assert not np.array_equal(first.grid, different_seed.grid)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rows": 4, "cols": 10},
            {"rows": 10, "cols": 4},
            {"rows": 10, "cols": 10, "obstacle_density": -0.1},
            {"rows": 10, "cols": 10, "obstacle_density": 1.1},
            {"rows": 10, "cols": 10, "obstacle_density": float("nan")},
            {"rows": 10, "cols": 10, "seed": -1},
        ],
    )
    def test_scenario_generation_rejects_invalid_parameters(
        self, kwargs: dict
    ) -> None:
        with pytest.raises(ValueError):
            create_preset_environment("Random Obstacles", **kwargs)


class TestStreamlitFigureRendering:
    """Test Matplotlib figure generation for Streamlit visualizer."""

    def test_render_grid_figure_returns_valid_matplotlib_figure(self) -> None:
        """Figure generation succeeds and returns a closed figure."""
        env, start, goal = create_preset_environment(
            "Demo Barriers / Maze (Default)", rows=10, cols=10
        )
        planner = AStarPlanner()
        res = planner.plan(env, start, goal)

        fig = render_grid_figure(
            env=env,
            start=start,
            goal=goal,
            path=res.path,
            robot_pos=start,
            history=[start],
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMetricsJsonExport:
    """Test JSON payloads and download configuration for navigation metrics."""

    def test_dashboard_download_is_strict_json_with_original_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        metrics = NavigationMetrics(
            success=True,
            initial_planning_time_ms=1.25,
            initial_path_length=3,
            initial_nodes_expanded=4,
            steps_executed=2,
            replan_count=1,
            total_replan_time_ms=0.5,
            total_execution_time_ms=2.0,
            final_robot_status="REACHED_GOAL",
            replan_history=[
                {
                    "timestamp": 123.5,
                    "trigger_position": (1, 2),
                    "new_path_length": 2,
                    "replan_duration_ms": 0.5,
                    "nodes_expanded": 1,
                    "success": True,
                }
            ],
        )
        captured: dict = {}
        monkeypatch.setattr(
            app.st, "download_button", lambda **kwargs: captured.update(kwargs)
        )

        app._render_metrics_download(metrics)

        payload = json.loads(captured["data"])
        assert payload == {
            "success": True,
            "initial_planning_time_ms": 1.25,
            "initial_path_length": 3,
            "initial_nodes_expanded": 4,
            "steps_executed": 2,
            "replan_count": 1,
            "total_replan_time_ms": 0.5,
            "total_execution_time_ms": 2.0,
            "final_robot_status": "REACHED_GOAL",
            "replan_history": [
                {
                    "timestamp": 123.5,
                    "trigger_position": [1, 2],
                    "new_path_length": 2,
                    "replan_duration_ms": 0.5,
                    "nodes_expanded": 1,
                    "success": True,
                }
            ],
            "path_efficiency_ratio": 1.0,
        }
        assert type(payload["success"]) is bool
        assert type(payload["initial_planning_time_ms"]) is float
        assert type(payload["initial_path_length"]) is int
        assert type(payload["final_robot_status"]) is str
        assert captured["file_name"] == "navigation_metrics.json"
        assert captured["mime"] == "application/json"

    def test_dashboard_download_rejects_non_finite_metrics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        metrics = NavigationMetrics(initial_planning_time_ms=float("nan"))
        monkeypatch.setattr(
            app.st,
            "download_button",
            lambda **kwargs: pytest.fail("Invalid JSON must not be sent as a download"),
        )

        with pytest.raises(ValueError, match="Out of range float values"):
            app._render_metrics_download(metrics)

    def test_metrics_tracker_json_export_remains_valid(self, tmp_path) -> None:
        tracker = MetricsTracker()
        tracker.start_trial()
        tracker.record_initial_plan(1.25, 3, 4, success=True)
        tracker.record_step()
        tracker.end_trial("REACHED_GOAL", success=True)
        filepath = tmp_path / "metrics.json"

        tracker.export_json(str(filepath))

        exported = json.loads(filepath.read_text(encoding="utf-8"))
        assert exported["success"] is True
        assert exported["initial_path_length"] == 3
        assert exported["steps_executed"] == 1
        assert exported["final_robot_status"] == "REACHED_GOAL"
        assert "path_efficiency_ratio" in exported


class TestLiveNavigationState:
    """Test dashboard controls through the shared navigation controller."""

    @staticmethod
    def make_state() -> dict:
        env = GridEnvironment(rows=7, cols=7)
        return {
            "env": env,
            "start": (0, 0),
            "goal": (6, 6),
            "robot": None,
            "controller": None,
            "metrics": None,
            "path_result": None,
            "current_step": 0,
            "playing": False,
            "obstacle_events": 0,
        }

    def test_manual_step_pause_and_reset_state(self) -> None:
        state = self.make_state()

        result = start_navigation(state, "A*")
        assert result is not None and result.success
        assert state["robot"].position == state["start"]
        assert state["robot"].status == RobotStatus.NAVIGATING
        assert state["current_step"] == 0

        assert set_navigation_playing(state, True) is True
        assert set_navigation_playing(state, False) is False
        first_position = state["robot"].position
        advance_navigation(state)
        assert state["robot"].position != first_position
        assert state["metrics"].steps_executed == 1

        reset_result = start_navigation(state, "A*")
        assert reset_result is not None and reset_result.success
        assert state["robot"].position == state["start"]
        assert state["metrics"].steps_executed == 0
        assert state["playing"] is False

        clear_navigation_state(state)
        assert state["path_result"] is None
        assert state["controller"] is None
        assert state["robot"].position == state["start"]

    def test_dynamic_obstacle_triggers_replan_without_collision(self) -> None:
        state = self.make_state()
        result = start_navigation(state, "A*")
        assert result is not None and result.success
        blocked_cell = state["robot"].peek_next_position()
        assert blocked_cell is not None

        add_dynamic_obstacle(state, blocked_cell)

        assert state["env"].is_obstacle(*blocked_cell)
        assert state["robot"].position != blocked_cell
        assert not state["env"].is_obstacle(*state["robot"].position)
        assert state["metrics"].replan_count == 1
        assert state["obstacle_events"] == 1

    def test_dynamic_obstacle_cannot_be_placed_on_robot(self) -> None:
        state = self.make_state()
        start_navigation(state, "A*")

        with pytest.raises(ValueError, match="current position"):
            add_dynamic_obstacle(state, state["robot"].position)

    def test_navigation_stops_when_dynamic_obstacle_removes_all_routes(self) -> None:
        env = GridEnvironment(rows=3, cols=5)
        env.set_obstacle(1, 0)
        env.set_obstacle(1, 1)
        env.set_obstacle(1, 2)
        env.set_obstacle(1, 3)
        env.set_obstacle(1, 4)
        state = self.make_state()
        state.update(
            env=env,
            start=(0, 0),
            goal=(0, 4),
        )
        result = start_navigation(state, "A*")
        assert result is not None and result.success
        blocked_cell = state["robot"].peek_next_position()
        assert blocked_cell is not None

        add_dynamic_obstacle(state, blocked_cell)

        assert state["robot"].status == RobotStatus.BLOCKED
        assert state["playing"] is False
        assert state["metrics"].success is False


class TestInteractiveGridControls:
    """Verify cell selection, goal placement, and safe obstacle editing."""

    @staticmethod
    def make_state() -> dict:
        env = GridEnvironment(rows=7, cols=7)
        return {
            "env": env,
            "start": (0, 0),
            "goal": (6, 6),
            "robot": None,
            "controller": None,
            "metrics": None,
            "path_result": None,
            "current_step": 0,
            "playing": False,
            "obstacle_events": 0,
        }

    def test_dataframe_selection_maps_to_row_and_column_and_checks_bounds(self) -> None:
        assert _selected_grid_cell(
            {"selection": {"cells": [(4, "2")]}}, rows=7, cols=7
        ) == (4, 2)
        assert _selected_grid_cell(
            {"selection": {"cells": [(7, "2")]}}, rows=7, cols=7
        ) is None
        assert _selected_grid_cell(
            {"selection": {"cells": []}}, rows=7, cols=7
        ) is None

    def test_grid_uses_streamlit_single_cell_selection(self, monkeypatch) -> None:
        env = GridEnvironment(rows=5, cols=5)
        captured = {}
        monkeypatch.setattr(
            app.st, "dataframe", lambda data, **kwargs: captured.update(kwargs)
        )

        render_interactive_grid(
            env, (0, 0), (4, 4), (0, 0), None, widget_key="test-grid"
        )

        assert captured["on_select"] is app._on_grid_selection
        assert captured["selection_mode"] == "single-cell"
        assert captured["key"] == "test-grid"

    def test_goal_click_updates_goal_without_resetting_robot(self) -> None:
        state = self.make_state()
        start_navigation(state, "A*")
        advance_navigation(state)
        current_position = state["robot"].position
        set_navigation_playing(state, True)

        notice = apply_grid_click(state, (3, 4), "Set Goal", "A*")

        assert state["goal"] == (3, 4)
        assert state["robot"].position == current_position
        assert state["playing"] is False
        assert state["path_result"].success
        assert state["path_result"].path[0] == current_position
        assert "route recalculated" in notice

    def test_goal_click_rejects_an_obstacle_and_out_of_bounds_selection(self) -> None:
        state = self.make_state()
        state["env"].set_obstacle(3, 4)
        original_goal = state["goal"]

        blocked_notice = apply_grid_click(state, (3, 4), "Set Goal", "A*")
        bounds_notice = apply_grid_click(state, (8, 2), "Set Goal", "A*")

        assert state["goal"] == original_goal
        assert "Goal not changed" in blocked_notice
        assert "outside the grid" in bounds_notice

    def test_obstacle_click_replans_around_blocked_path_without_moving(self) -> None:
        state = self.make_state()
        result = start_navigation(state, "A*")
        blocked_cell = result.path[1]
        current_position = state["robot"].position

        notice = apply_grid_click(state, blocked_cell, "Place Obstacles", "A*")

        assert state["env"].is_obstacle(*blocked_cell)
        assert state["robot"].position == current_position
        assert state["path_result"].success
        assert blocked_cell not in state["path_result"].path
        assert state["metrics"].replan_count == 1
        assert state["obstacle_events"] == 1
        assert "route recalculated" in notice

    def test_obstacle_click_toggles_existing_obstacle_off_and_replans(self) -> None:
        state = self.make_state()
        start_navigation(state, "A*")
        cell = (2, 2)
        state["env"].set_obstacle(*cell)

        notice = apply_grid_click(state, cell, "Place Obstacles", "A*")

        assert not state["env"].is_obstacle(*cell)
        assert state["path_result"].success
        assert state["metrics"].replan_count == 1
        assert "removed" in notice

    @pytest.mark.parametrize("position", [(0, 0), (6, 6)])
    def test_obstacle_cannot_be_placed_on_start_or_goal(self, position) -> None:
        state = self.make_state()
        start_navigation(state, "A*")
        with pytest.raises(ValueError):
            add_dynamic_obstacle(state, position)

    def test_clearing_obstacles_preserves_robot_and_replans(self) -> None:
        state = self.make_state()
        start_navigation(state, "A*")
        advance_navigation(state)
        current_position = state["robot"].position
        state["env"].set_obstacle(2, 2)
        state["env"].set_obstacle(3, 3)

        removed = clear_obstacles(state)

        assert removed == 2
        assert not np.any(state["env"].grid)
        assert state["robot"].position == current_position
        assert state["path_result"].success
        assert state["metrics"].replan_count == 1

    def test_navigation_reaches_goal_and_never_enters_a_blocked_cell(self) -> None:
        state = self.make_state()
        result = start_navigation(state, "A*")
        obstacle = result.path[1]
        add_dynamic_obstacle(state, obstacle)

        for _ in range(state["env"].rows * state["env"].cols):
            if state["robot"].status != RobotStatus.NAVIGATING:
                break
            next_position = state["robot"].peek_next_position()
            if next_position is not None:
                assert state["env"].is_in_bounds(*next_position)
                assert state["env"].is_free(*next_position)
            advance_navigation(state)

        assert state["robot"].status == RobotStatus.REACHED_GOAL
        assert state["robot"].position == state["goal"]
        assert obstacle not in state["robot"].history

    def test_failed_replan_discards_the_stale_path(self) -> None:
        state = self.make_state()
        assert start_navigation(state, "A*").success
        state["goal"] = (6, 6)
        state["env"].set_obstacle(5, 6)
        state["env"].set_obstacle(6, 5)

        result = start_navigation(state, "A*", start_from_current=True)

        assert not result.success
        assert state["robot"].path == []
        assert state["robot"].status == RobotStatus.FAILED
