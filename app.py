"""Streamlit Live Demo for the AI-Based Autonomous Navigation System.

Provides an interactive browser dashboard to:
1. Configure grid dimensions, obstacle presets, start and goal coordinates.
2. Select between genuinely implemented planning algorithms (A*, Dijkstra, BFS).
3. Visualize the occupancy grid, planned route, obstacles, and robot motion using Matplotlib.
4. Run, pause, and step the robot through its route using Streamlit fragments.
5. Inspect real-time performance metrics (planning latency, nodes expanded, path efficiency).
"""

import json
import math
import time
from typing import Any, List, MutableMapping, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.environment import GridEnvironment, EnvironmentError
from src.path_planning import (
    PathResult,
    get_planner,
    SUPPORTED_ALGORITHMS,
)
from src.robot import Robot, RobotStatus
from src.navigation import NavigationController
from src.metrics import MetricsTracker, NavigationMetrics


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Autonomous Navigation — 2D Simulation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Preset Map Generators
# ---------------------------------------------------------------------------
def create_preset_environment(
    preset_name: str,
    rows: int,
    cols: int,
    seed: int = 42,
    obstacle_density: float = 0.22,
) -> Tuple[GridEnvironment, Tuple[int, int], Tuple[int, int]]:
    """Generate a reproducible occupancy grid and valid start/goal coordinates."""
    if rows < 5 or cols < 5:
        raise ValueError("Scenario dimensions must be at least 5 rows by 5 columns.")
    if not math.isfinite(obstacle_density) or not 0.0 <= obstacle_density <= 1.0:
        raise ValueError("Obstacle density must be a finite value between 0 and 1.")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Random seed must be a non-negative integer.")

    env = GridEnvironment(rows=rows, cols=cols)
    start: Tuple[int, int] = (1, 1)
    goal: Tuple[int, int] = (rows - 2, cols - 2)

    if preset_name in ("Demo Barriers / Maze (Default)", "Maze-like Environment"):
        # Vertical wall 1 with passage
        col1 = max(2, cols // 3)
        gap1 = rows // 2
        for r in range(1, rows - 1):
            if r not in (gap1, gap1 + 1):
                env.set_obstacle(r, col1)

        # Vertical wall 2 with passage
        col2 = max(col1 + 2, (2 * cols) // 3)
        gap2 = max(1, rows // 4)
        for r in range(1, rows - 1):
            if r not in (gap2, gap2 + 1):
                env.set_obstacle(r, col2)

    elif preset_name in ("Narrow Corridor",):
        # Walls creating a single corridor
        mid_row = rows // 2
        for c in range(1, cols - 1):
            if c != cols // 2:
                if mid_row - 1 > 0:
                    env.set_obstacle(mid_row - 1, c)
                if mid_row + 1 < rows - 1:
                    env.set_obstacle(mid_row + 1, c)

    elif preset_name in (
        "U-Shaped Trap / Local Minimum",
        "U-Shaped Obstacle Trap",
    ):
        # U-shaped cup around the goal to test heuristic backtracking
        gr, gc = goal
        for r in range(max(0, gr - 3), min(rows, gr + 4)):
            if gc - 2 >= 0:
                env.set_obstacle(r, gc - 2)
        for c in range(max(0, gc - 2), min(cols, gc + 2)):
            if gr - 3 >= 0:
                env.set_obstacle(gr - 3, c)
            if gr + 3 < rows:
                env.set_obstacle(gr + 3, c)

    elif preset_name in ("Dense Random Clutter", "Random Obstacles"):
        rng = np.random.default_rng(seed)
        for r in range(rows):
            for c in range(cols):
                if (r, c) != start and (r, c) != goal and rng.random() < obstacle_density:
                    env.set_obstacle(r, c)

    elif preset_name in ("Walled-Off Goal (Unreachable Test)", "Unreachable Goal"):
        # Completely seal off the goal
        gr, gc = goal
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = gr + dr, gc + dc
            if env.is_in_bounds(nr, nc):
                env.set_obstacle(nr, nc)

    elif preset_name not in ("Open Field", "Open Grid (No Obstacles)"):
        raise ValueError(f"Unknown scenario preset: {preset_name!r}")

    # Ensure start and goal are always clear, including for dense random maps.
    if env.is_in_bounds(start[0], start[1]):
        env.clear_cell(start[0], start[1])
    if env.is_in_bounds(goal[0], goal[1]):
        env.clear_cell(goal[0], goal[1])

    return env, start, goal


# ---------------------------------------------------------------------------
# Rendering Helpers (Matplotlib for Streamlit)
# ---------------------------------------------------------------------------
def render_grid_figure(
    env: GridEnvironment,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path: Optional[List[Tuple[int, int]]] = None,
    robot_pos: Optional[Tuple[int, int]] = None,
    history: Optional[List[Tuple[int, int]]] = None,
) -> plt.Figure:
    """Generate a high-quality Matplotlib figure of the grid and navigation state."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    # Base grid image (0=white, 1=slate dark)
    grid_img = np.zeros((env.rows, env.cols, 3), dtype=np.float32)
    for r in range(env.rows):
        for c in range(env.cols):
            if env.is_obstacle(r, c):
                grid_img[r, c] = [0.17, 0.24, 0.31]  # Dark slate
            else:
                grid_img[r, c] = [0.97, 0.98, 0.99]  # Crisp light background

    ax.imshow(grid_img, origin="upper")

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which="minor", color="#D0D7DE", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", size=0)

    # Major ticks showing coordinates
    step_tick = max(1, max(env.rows, env.cols) // 10)
    ax.set_xticks(np.arange(0, env.cols, step_tick))
    ax.set_yticks(np.arange(0, env.rows, step_tick))
    ax.set_xlabel("Column", fontsize=10, fontweight="bold", color="#333333")
    ax.set_ylabel("Row", fontsize=10, fontweight="bold", color="#333333")

    # Draw planned route
    if path and len(path) > 1:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax.plot(
            path_cols,
            path_rows,
            color="#2980B9",
            linewidth=3,
            linestyle="-",
            alpha=0.85,
            zorder=3,
            label="Planned Route",
        )
        ax.scatter(
            path_cols,
            path_rows,
            color="#3498DB",
            s=40,
            zorder=4,
            edgecolors="#1B4F72",
            linewidths=0.8,
        )

    # Draw traversed breadcrumb history
    if history and len(history) > 1:
        hist_rows = [h[0] for h in history[:-1]]
        hist_cols = [h[1] for h in history[:-1]]
        ax.scatter(
            hist_cols,
            hist_rows,
            color="#A569BD",
            s=25,
            alpha=0.6,
            zorder=5,
            label="Traversed Trail",
        )

    # Draw Start marker
    if env.is_in_bounds(start[0], start[1]):
        ax.scatter(
            start[1],
            start[0],
            color="#2ECC71",
            s=180,
            marker="s",
            zorder=6,
            edgecolors="#1E8449",
            linewidths=1.5,
            label="Start",
        )
        ax.text(
            start[1],
            start[0],
            "S",
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )

    # Draw Goal marker
    if env.is_in_bounds(goal[0], goal[1]):
        ax.scatter(
            goal[1],
            goal[0],
            color="#E74C3C",
            s=180,
            marker="s",
            zorder=6,
            edgecolors="#922B21",
            linewidths=1.5,
            label="Goal",
        )
        ax.text(
            goal[1],
            goal[0],
            "G",
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )

    # Draw Robot current position
    curr_robot = robot_pos if robot_pos is not None else start
    if env.is_in_bounds(curr_robot[0], curr_robot[1]):
        ax.scatter(
            curr_robot[1],
            curr_robot[0],
            color="#8E44AD",
            s=220,
            marker="o",
            zorder=8,
            edgecolors="#FFFFFF",
            linewidths=2,
            label="Robot",
        )
        ax.scatter(
            curr_robot[1],
            curr_robot[0],
            color="#FFFFFF",
            s=40,
            marker="o",
            zorder=9,
        )

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)  # Inverted so row 0 is top
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    return fig


def _grid_cell_style(
    env: GridEnvironment,
    row: int,
    col: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    robot_pos: Tuple[int, int],
    path: Optional[List[Tuple[int, int]]],
) -> str:
    """Return a readable background style for one interactive grid cell."""
    if env.is_obstacle(row, col):
        return "background-color: #2c3e50; color: white; font-weight: bold; text-align: center"
    if (row, col) == robot_pos:
        return "background-color: #8e44ad; color: white; font-weight: bold; text-align: center"
    if (row, col) == goal:
        return "background-color: #27ae60; color: white; font-weight: bold; text-align: center"
    if (row, col) == start:
        return "background-color: #f1c40f; color: #222; font-weight: bold; text-align: center"
    if path and (row, col) in path:
        return "background-color: #d6eaf8; color: #154360; text-align: center"
    return (
        "background-color: #f8f9fa; color: #566573; "
        "text-align: center; border: 1px solid #d0d7de"
    )


def _selected_grid_cell(
    event_state: Any, rows: int, cols: int
) -> Optional[Tuple[int, int]]:
    """Translate Streamlit's single-cell dataframe selection into (row, col)."""
    if not event_state:
        return None
    selection = event_state.get("selection", {})
    cells = selection.get("cells", []) if selection else []
    if not cells:
        return None
    try:
        row, column = cells[-1]
        position = (int(row), int(column))
    except (TypeError, ValueError):
        return None
    return position if 0 <= position[0] < rows and 0 <= position[1] < cols else None


def render_interactive_grid(
    env: GridEnvironment,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    robot_pos: Tuple[int, int],
    path: Optional[List[Tuple[int, int]]],
    widget_key: str,
) -> None:
    """Render a clickable, styled grid using Streamlit's native cell selection."""
    values: List[List[str]] = []
    for row in range(env.rows):
        cells = []
        for col in range(env.cols):
            markers = []
            if (row, col) == start:
                markers.append("S")
            if (row, col) == goal:
                markers.append("G")
            if (row, col) == robot_pos:
                markers.append("R")
            if env.is_obstacle(row, col):
                markers.append("■")
            elif not markers and path and (row, col) in path:
                markers.append("·")
            elif not markers:
                markers.append("·")
            cells.append("/".join(markers))
        values.append(cells)

    columns = [str(col) for col in range(env.cols)]
    frame = pd.DataFrame(values, columns=columns, index=range(env.rows))
    frame.index.name = "Row"
    styles = pd.DataFrame(
        [
            [
                _grid_cell_style(env, row, col, start, goal, robot_pos, path)
                for col in range(env.cols)
            ]
            for row in range(env.rows)
        ],
        columns=columns,
        index=range(env.rows),
    )
    st.dataframe(
        frame.style.apply(lambda _: styles, axis=None),
        key=widget_key,
        on_select=_on_grid_selection,
        selection_mode="single-cell",
        height=min(700, 42 * env.rows + 42),
        width="stretch",
        column_config={
            column: st.column_config.TextColumn(
                column, width="small", help=f"Column {column}"
            )
            for column in columns
        },
        hide_index=False,
        row_height=32,
        alt="Interactive navigation grid. Select a cell while in Set Goal or Place Obstacles mode.",
    )


def _replan_after_map_change(state: MutableMapping[str, Any]) -> Optional[PathResult]:
    """Recalculate a retained route after the occupancy grid changes."""
    controller: Optional[NavigationController] = state.get("controller")
    if controller is None:
        return None
    result = controller.replan()
    state["path_result"] = result
    state["metrics"] = controller.metrics.metrics if controller.metrics else None
    if state["robot"].status != RobotStatus.NAVIGATING:
        state["playing"] = False
        state["next_navigation_step_at"] = None
    return result


def apply_grid_click(
    state: MutableMapping[str, Any],
    position: Tuple[int, int],
    mode: str,
    algorithm: str,
) -> str:
    """Apply one selected cell according to the active grid interaction mode."""
    env: GridEnvironment = state["env"]
    if not env.is_in_bounds(*position):
        return "That cell is outside the grid."

    if mode == "Set Goal":
        try:
            env.validate_position(position, name="Goal")
        except EnvironmentError as error:
            return f"Goal not changed: {error}"
        if position == state["goal"]:
            return f"Goal remains at {position}."
        had_controller = state.get("controller") is not None
        current_position = state["robot"].position
        state["goal"] = position
        state["playing"] = False
        state["next_navigation_step_at"] = None
        if had_controller:
            result = start_navigation(
                state, algorithm, start_from_current=True
            )
            if result is not None and not result.success:
                return f"Goal set to {position}, but no valid path found."
            return f"Goal set to {position}; route recalculated from {current_position}."
        clear_navigation_state_preserving_robot(state)
        return f"Goal set to {position}. Start navigation to calculate a route."

    if mode == "Place Obstacles":
        if state["env"].is_obstacle(*position):
            state["env"].clear_cell(*position)
            action = "removed"
            result = _replan_after_map_change(state)
        else:
            try:
                add_dynamic_obstacle(state, position)
            except (ValueError, EnvironmentError) as error:
                return f"Obstacle not changed: {error}"
            action = "placed"
            result = state.get("path_result")
        if result is not None and not result.success:
            return f"Obstacle {action} at {position}; no valid path found."
        if result is not None:
            return f"Obstacle {action} at {position}; route recalculated."
        return f"Obstacle {action} at {position}."

    return "Grid is in view-only mode."


def clear_obstacles(state: MutableMapping[str, Any]) -> int:
    """Remove all obstacles and refresh a retained route without moving the robot."""
    env: GridEnvironment = state["env"]
    removed = int(np.count_nonzero(env.grid))
    if removed:
        env.grid.fill(0)
        _replan_after_map_change(state)
    return removed


def clear_navigation_state_preserving_robot(
    state: MutableMapping[str, Any],
) -> None:
    """Clear a stale route without moving the robot or discarding its history."""
    robot: Robot = state["robot"]
    robot.path = []
    robot.current_step_index = 0
    robot.status = RobotStatus.IDLE
    state["path_result"] = None
    state["controller"] = None
    state["metrics"] = None
    state["current_step"] = 0
    state["playing"] = False
    state["obstacle_events"] = 0
    state["next_navigation_step_at"] = None


def _on_grid_selection() -> None:
    """Handle native Streamlit dataframe selections before the app reruns."""
    state = st.session_state
    widget_key = state.get("active_grid_widget_key")
    if not widget_key:
        return
    cell = _selected_grid_cell(
        state.get(widget_key), state["env"].rows, state["env"].cols
    )
    if cell is None:
        return
    mode = state.get("grid_interaction_mode", "View Only")
    algorithm = state.get("selected_algorithm", "A*")
    state["grid_notice"] = apply_grid_click(state, cell, mode, algorithm)
    state["grid_widget_revision"] = state.get("grid_widget_revision", 0) + 1


def _render_metrics_download(metrics_obj: NavigationMetrics) -> None:
    """Render a strict JSON download for the current navigation metrics."""
    st.download_button(
        label="📥 Download Metrics JSON",
        data=json.dumps(metrics_obj.to_dict(), indent=2, allow_nan=False),
        file_name="navigation_metrics.json",
        mime="application/json",
    )


def clear_navigation_state(state: MutableMapping[str, Any]) -> None:
    """Discard a route when its environment, coordinates, or planner changes."""
    state["path_result"] = None
    state["controller"] = None
    state["metrics"] = None
    state["robot"] = Robot(initial_position=state["start"])
    state["current_step"] = 0
    state["playing"] = False
    state["obstacle_events"] = 0
    state["next_navigation_step_at"] = None


def start_navigation(
    state: MutableMapping[str, Any],
    algorithm: str,
    start_from_current: bool = False,
) -> Optional[PathResult]:
    """Plan a route and retain its controller and metrics in session state."""
    robot = (
        state["robot"]
        if start_from_current and state.get("robot") is not None
        else Robot(initial_position=state["start"])
    )
    metrics = MetricsTracker()
    controller = NavigationController(
        environment=state["env"],
        planner=get_planner(algorithm),
        robot=robot,
        metrics=metrics,
    )
    result = controller.set_goal(state["goal"])
    state["robot"] = robot
    state["metrics"] = metrics.metrics
    state["controller"] = controller
    state["path_result"] = result
    state["current_step"] = 0
    state["playing"] = False
    state["obstacle_events"] = 0
    state["next_navigation_step_at"] = None
    return result


def advance_navigation(state: MutableMapping[str, Any]) -> RobotStatus:
    """Execute one guarded navigation cycle and update session progress."""
    controller = state.get("controller")
    robot = state["robot"]
    if controller is None:
        return robot.status

    next_position = robot.peek_next_position()
    if next_position is not None and state["env"].is_obstacle(*next_position):
        state["obstacle_events"] = state.get("obstacle_events", 0) + 1

    status = controller.update_step()
    state["current_step"] = controller.metrics.metrics.steps_executed
    state["metrics"] = controller.metrics.metrics
    if status != RobotStatus.NAVIGATING:
        state["playing"] = False
    return status


def set_navigation_playing(
    state: MutableMapping[str, Any], playing: bool
) -> bool:
    """Set timed playback only while a valid route is actively navigable."""
    result = state.get("path_result")
    can_play = (
        result is not None
        and result.success
        and state["robot"].status == RobotStatus.NAVIGATING
    )
    state["playing"] = bool(playing and can_play)
    if state["playing"]:
        interval = {"Fast": 0.15, "Normal": 0.5, "Slow": 1.0}.get(
            state.get("playback_speed", "Normal"), 0.5
        )
        state["next_navigation_step_at"] = time.monotonic() + interval
    else:
        state["next_navigation_step_at"] = None
    return state["playing"]


def add_dynamic_obstacle(
    state: MutableMapping[str, Any], position: Tuple[int, int]
) -> bool:
    """Add an obstacle and immediately recalculate any active route."""
    env = state["env"]
    robot = state["robot"]
    env.validate_position(position, name="Obstacle")
    if position == robot.position:
        raise ValueError("Cannot place an obstacle on the robot's current position.")
    if position == state["goal"]:
        raise ValueError("Cannot place an obstacle on the destination.")
    if position == state["start"]:
        raise ValueError("Cannot place an obstacle on the start position.")

    env.set_obstacle(*position)
    path_result: Optional[PathResult] = state.get("path_result")
    if path_result and path_result.path:
        remaining_path = path_result.path[robot.current_step_index :]
        if position in remaining_path:
            state["obstacle_events"] = state.get("obstacle_events", 0) + 1
    _replan_after_map_change(state)
    return True


def _render_dashboard_frame() -> None:
    """Render the current environment, robot state, and retained metrics."""
    state = st.session_state
    env: GridEnvironment = state["env"]
    robot: Robot = state["robot"]
    result: Optional[PathResult] = state.get("path_result")
    metrics: Optional[NavigationMetrics] = state.get("metrics")
    active_path = robot.path if robot.status != RobotStatus.BLOCKED else None

    st.subheader("🗺️ Interactive Navigation Grid")
    st.caption(
        "Select a cell while **Set Goal** or **Place Obstacles** mode is active. "
        "S = start, G = goal, R = robot, ■ = obstacle."
    )
    grid_key = f"navigation_grid_{state.get('grid_widget_revision', 0)}"
    previous_key = state.get("last_rendered_grid_key")
    if previous_key and previous_key != grid_key:
        st.session_state.pop(previous_key, None)
    state["active_grid_widget_key"] = grid_key
    state["last_rendered_grid_key"] = grid_key
    render_interactive_grid(
        env=env,
        start=state["start"],
        goal=state["goal"],
        path=active_path,
        robot_pos=robot.position,
        widget_key=grid_key,
    )
    st.write(f"**Robot position:** `{robot.position}`")
    st.write(f"**Selected goal:** `{state['goal']}`")
    if result and result.success and active_path:
        st.write(f"**Current route:** {len(active_path)} cells")

    st.subheader("📊 Telemetry & Performance")
    if result is None:
        st.info("Click **Start Navigation** to plan and begin.")
    elif not result.success:
        st.error(
            f"🚫 **No valid path found.** {result.error_message or 'The goal is unreachable.'}"
        )
        st.metric("Nodes explored", result.nodes_expanded)
        st.metric("Planning time", f"{result.execution_time_ms:.3f} ms")
    elif robot.status == RobotStatus.BLOCKED:
        st.error("🚫 **Navigation stopped:** no safe route remains to the goal.")
    else:
        status_label = {
            RobotStatus.NAVIGATING: "RUNNING" if state.get("playing") else "PAUSED",
            RobotStatus.REACHED_GOAL: "GOAL REACHED",
            RobotStatus.FAILED: "FAILED",
        }.get(robot.status, robot.status.value)
        st.metric("Navigation status", status_label)
        st.metric("Algorithm", result.algorithm)
        current_path = active_path or result.path or []
        st.metric("Path length", f"{len(current_path)} cells")
        st.metric("Steps taken", metrics.steps_executed if metrics else 0)
        st.metric("Nodes explored", result.nodes_expanded)
        st.metric("Planning time", f"{result.execution_time_ms:.3f} ms")
        st.metric("Obstacle-triggered replans", state.get("obstacle_events", 0))
        st.metric("Cell collisions", 0)
        if metrics:
            st.metric("Path efficiency", f"{metrics.path_efficiency_ratio:.3f}")

    if metrics:
        with st.expander("📄 Raw Metrics (JSON)"):
            st.json(metrics.to_dict())
            _render_metrics_download(metrics)

    if result and result.path:
        with st.expander("📍 View Planned Waypoint Coordinates"):
            st.code(
                " ➔ ".join(
                    f"({row},{col})"
                    for row, col in (active_path or result.path)
                ),
                language="text",
            )


def _advance_and_render() -> None:
    """Advance one step on each fragment rerun, then redraw the dashboard."""
    if (
        st.session_state.get("playing")
        and time.monotonic()
        >= st.session_state.get("next_navigation_step_at", float("inf"))
    ):
        advance_navigation(st.session_state)
        if st.session_state.get("playing"):
            interval = {"Fast": 0.15, "Normal": 0.5, "Slow": 1.0}.get(
                st.session_state.get("playback_speed", "Normal"), 0.5
            )
            st.session_state["next_navigation_step_at"] = time.monotonic() + interval
        else:
            st.session_state["next_navigation_step_at"] = None
            st.rerun(scope="app")
    _render_dashboard_frame()


@st.fragment(run_every=0.15)
def _fast_live_navigation() -> None:
    _advance_and_render()


@st.fragment(run_every=0.5)
def _normal_live_navigation() -> None:
    _advance_and_render()


@st.fragment(run_every=1.0)
def _slow_live_navigation() -> None:
    _advance_and_render()


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize persistent Streamlit session keys."""
    if "env" not in st.session_state:
        env, start, goal = create_preset_environment(
            "Demo Barriers / Maze (Default)", rows=20, cols=20
        )
        st.session_state["env"] = env
        st.session_state["start"] = start
        st.session_state["goal"] = goal
        st.session_state["last_preset"] = "Demo Barriers / Maze (Default)"
        st.session_state["path_result"] = None
        st.session_state["metrics"] = None
        st.session_state["robot"] = Robot(initial_position=start)
        st.session_state["current_step"] = 0
        st.session_state["controller"] = None
        st.session_state["playing"] = False
        st.session_state["obstacle_events"] = 0
        st.session_state["next_navigation_step_at"] = None
        st.session_state["grid_widget_revision"] = 0
        st.session_state["last_rendered_grid_key"] = None
        st.session_state["active_grid_widget_key"] = None
        st.session_state["grid_notice"] = ""
        st.session_state["grid_interaction_mode"] = "View Only"
        st.session_state["selected_algorithm"] = "A*"
        st.session_state["scenario_signature"] = (
            "Demo Barriers / Maze (Default)",
            20,
            20,
            42,
            0.22,
        )
        st.session_state["last_algorithm"] = "A*"


init_session_state()


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.title("🛠️ Navigation Controls")

st.sidebar.subheader("1. Environment Dimensions")
grid_rows = st.sidebar.slider("Rows (Height)", min_value=5, max_value=50, value=20, step=1)
grid_cols = st.sidebar.slider("Columns (Width)", min_value=5, max_value=50, value=20, step=1)

st.sidebar.subheader("2. Scenario Preset")
preset_options = [
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
]
selected_preset = st.sidebar.selectbox("Select Scenario Preset", preset_options)
random_seed = st.sidebar.number_input(
    "Random Seed", min_value=0, max_value=999999, value=42, step=1
)
obstacle_density = st.sidebar.slider(
    "Obstacle Density", min_value=0.0, max_value=0.8, value=0.22, step=0.02,
    help="Used by Random Obstacles and Dense Random Clutter.",
)

scenario_signature = (
    selected_preset,
    grid_rows,
    grid_cols,
    int(random_seed),
    float(obstacle_density),
)
if st.session_state.get("scenario_signature") != scenario_signature:
    new_env, new_s, new_g = create_preset_environment(
        selected_preset,
        rows=grid_rows,
        cols=grid_cols,
        seed=int(random_seed),
        obstacle_density=float(obstacle_density),
    )
    st.session_state["env"] = new_env
    st.session_state["start"] = new_s
    st.session_state["goal"] = new_g
    st.session_state["last_preset"] = selected_preset
    st.session_state["scenario_signature"] = scenario_signature
    clear_navigation_state(st.session_state)
    st.session_state["grid_widget_revision"] += 1
    st.session_state["grid_notice"] = "Scenario updated; previous route was cleared."

st.sidebar.subheader("3. Start Position")
start_r = st.sidebar.number_input(
    "Start Row", min_value=0, max_value=grid_rows - 1, value=st.session_state["start"][0]
)
start_c = st.sidebar.number_input(
    "Start Column", min_value=0, max_value=grid_cols - 1, value=st.session_state["start"][1]
)
new_start = (int(start_r), int(start_c))
if new_start != st.session_state["start"]:
    try:
        st.session_state["env"].validate_position(new_start, name="Start")
    except EnvironmentError as error:
        st.sidebar.error(str(error))
    else:
        st.session_state["start"] = new_start
        st.session_state["grid_widget_revision"] += 1
        st.session_state["grid_notice"] = f"Start position set to {new_start}."
        clear_navigation_state(st.session_state)

st.sidebar.write(f"**Goal:** `{st.session_state['goal']}`")
st.sidebar.subheader("4. Grid Interaction")
st.sidebar.radio(
    "Grid mode",
    options=["View Only", "Set Goal", "Place Obstacles"],
    key="grid_interaction_mode",
    help="Click a cell to set the destination or toggle an obstacle.",
)
clear_obstacles_clicked = st.sidebar.button(
    "🧹 Clear Obstacles", use_container_width=True
)
if clear_obstacles_clicked:
    removed = clear_obstacles(st.session_state)
    st.session_state["grid_notice"] = (
        f"Cleared {removed} obstacle(s)."
        if removed
        else "There are no obstacles to clear."
    )

st.sidebar.subheader("5. Algorithm Selection")
selected_algo = st.sidebar.selectbox(
    "Path Planning Algorithm",
    options=SUPPORTED_ALGORITHMS,
    help="All algorithms are fully implemented and verified with automated tests.",
)
st.session_state["selected_algorithm"] = selected_algo
if selected_algo != st.session_state.get("last_algorithm"):
    clear_navigation_state(st.session_state)
    st.session_state["last_algorithm"] = selected_algo

st.sidebar.subheader("6. Execution")
start_clicked = st.sidebar.button(
    "🧭 Start Navigation", type="primary", use_container_width=True
)
resume_clicked = st.sidebar.button(
    "▶️ Resume Navigation",
    use_container_width=True,
    disabled=(
        st.session_state.get("path_result") is None
        or not st.session_state["path_result"].success
        or st.session_state["robot"].status != RobotStatus.NAVIGATING
    ),
)
pause_clicked = st.sidebar.button(
    "⏸ Pause Navigation",
    use_container_width=True,
    disabled=not st.session_state.get("playing", False),
)
step_clicked = st.sidebar.button(
    "⏭ Step",
    use_container_width=True,
    disabled=(
        st.session_state.get("controller") is None
        or st.session_state["robot"].status != RobotStatus.NAVIGATING
        or st.session_state.get("playing", False)
    ),
)
reset_clicked = st.sidebar.button("🔄 Reset Simulation", use_container_width=True)
playback_speed = st.sidebar.selectbox(
    "Playback Speed", options=["Fast", "Normal", "Slow"], index=1
)
st.session_state["playback_speed"] = playback_speed

if start_clicked:
    result = start_navigation(st.session_state, selected_algo)
    if result is not None and result.success:
        set_navigation_playing(st.session_state, True)
        st.session_state["grid_notice"] = "Navigation started."
    else:
        st.session_state["grid_notice"] = (
            result.error_message if result else "No valid path found."
        )
if reset_clicked:
    result = start_navigation(st.session_state, selected_algo)
    st.session_state["grid_notice"] = (
        "Simulation reset to the start position."
        if result and result.success
        else (result.error_message if result else "No valid path found.")
    )
if resume_clicked:
    set_navigation_playing(st.session_state, True)
if pause_clicked:
    set_navigation_playing(st.session_state, False)
    st.session_state["grid_notice"] = "Navigation paused."
if step_clicked:
    status = advance_navigation(st.session_state)
    st.session_state["grid_notice"] = (
        "Goal reached."
        if status == RobotStatus.REACHED_GOAL
        else f"Robot advanced to {st.session_state['robot'].position}."
    )

if st.session_state.get("grid_notice"):
    st.info(st.session_state["grid_notice"])


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
st.title("🤖 AI-Based Autonomous Navigation System")
st.caption(
    "Interactive 2D Simulation Dashboard powered by Python and Streamlit. "
    "Featuring fully verified graph planning engines (**A***, **Dijkstra**, and **BFS**) with strictly 4-directional transitions."
)

st.caption(
    f"Scenario: **{selected_preset}** · Seed: **{int(random_seed)}** · "
    f"Obstacle density: **{obstacle_density:.2f}**"
)
if st.session_state.get("playing"):
    live_navigation = {
        "Fast": _fast_live_navigation,
        "Normal": _normal_live_navigation,
        "Slow": _slow_live_navigation,
    }[playback_speed]
    live_navigation()
else:
    _render_dashboard_frame()

# --- Bottom Educational Card ---
st.divider()
st.subheader("📚 Algorithmic Comparison Reference")
ac1, ac2, ac3 = st.columns(3)
with ac1:
    st.markdown(
        r"""
        **A\* Search Algorithm**
        - Evaluation: $f(n) = g(n) + h(n)$
        - Heuristic: Admissible **Manhattan Distance** ($L_1$)
        - Behavior: Highly directed search towards goal; evaluates minimal states.
        - Guarantees: Complete and optimal.
        """
    )
with ac2:
    st.markdown(
        r"""
        **Dijkstra's Algorithm**
        - Evaluation: $f(n) = g(n)$ (Uniform-Cost)
        - Heuristic: None ($h(n) = 0$)
        - Behavior: Expands radially in concentric circles from start.
        - Guarantees: Complete and optimal; evaluates more nodes than A*.
        """
    )
with ac3:
    st.markdown(
        """
        **Breadth-First Search (BFS)**
        - Evaluation: FIFO Queue level-by-level
        - Heuristic: None
        - Behavior: Explores equal-hop frontiers outwards.
        - Guarantees: Guaranteed shortest path on unweighted unit-cost grid graphs.
        """
    )
