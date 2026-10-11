# Project Architecture: AI-Based Autonomous Navigation System

## 1. Project Objective

The objective of this project is to build an extensible, modular, and beginner-friendly simulation architecture for an autonomous ground robot navigating a 2D grid environment.

The system is designed to:
- Model physical space using a discrete 2D occupancy grid.
- Compute mathematically optimal, collision-free paths from a starting position to a goal destination using the A* heuristic search algorithm.
- Autonomously drive a simulated robot along the planned path in discrete steps.
- Detect path blockages dynamically when new obstacles emerge in the environment.
- Trigger real-time replanning when an existing route is obstructed.
- Track, calculate, and log rigorous performance metrics (e.g., search expansion, execution time, path optimality, replan count).
- Decouple logical path planning and decision-making from graphical rendering and physical modeling.

---

## 2. High-Level Block Diagram

```
+-------------------------------------------------------------------------------------------------+
|                                     CONFIGURATION LAYER                                         |
|  - Grid dimensions (rows, cols)   - Display parameters (cell_size, colors, FPS)                 |
|  - Movement primitives (cardinal moves only)                                                    |
+-------------------------------------------------------------------------------------------------+
                                               |
                                               v
+-------------------------------------------------------------------------------------------------+
|                                    ENVIRONMENT LAYER (src/environment.py)                       |
|  - 2D NumPy Occupancy Grid: 0 = Free, 1 = Occupied                                              |
|  - Coordinate Validation (In-bounds, Obstacle collision)                                        |
|  - 4-Directional Neighbor Query Engine                                                          |
|  - Coordinate Transforms: Grid (row, col) <---> Screen (x, y)                                   |
+-------------------------------------------------------------------------------------------------+
                       |                                               |
         Map / Query   |                                 Occupancy     | Dynamic Obstacle
          Interfaces   |                                  Telemetry    | Detection
                       v                                               v
+------------------------------------------+      +-----------------------------------------------+
|      PATH PLANNING (src/path_planning.py)|      |        NAVIGATION LOGIC (src/navigation.py)   |
|  - Deterministic A* Search               |      |  - Mission controller                         |
|  - Admissible Manhattan Distance         | <--- |  - Goal setting & path assignment             |
|  - Strictly 4-Directional Transitions    |      |  - Active path blockage detection             |
|  - Output: PathResult (path, cost, stats)| ---> |  - Dynamic replanning trigger                 |
+------------------------------------------+      +-----------------------------------------------+
                                                                       |
                                                           Step        | State
                                                           Forward     | Updates
                                                                       v
+------------------------------------------+      +-----------------------------------------------+
|     PERFORMANCE METRICS (src/metrics.py) |      |          ROBOT ENTITY (src/robot.py)          |
|  - Search expansion count                |      |  - Logical position (row, col)                |
|  - Planning and replanning latency (ms)  | <--- |  - State machine: IDLE, NAVIGATING,           |
|  - Path length & execution step count    |      |    REACHED_GOAL, BLOCKED, FAILED              |
|  - Path optimality ratio                 |      |  - Waypoint traversal & history breadcrumbs   |
+------------------------------------------+      +-----------------------------------------------+
                       |                                               |
                       v                                               v
+-------------------------------------------------------------------------------------------------+
|                              SIMULATION & VISUALIZATION LAYER (src/simulation.py)               |
|  - Decoupled Pygame Renderer (Grid, Path Trail, Markers, Robot body, HUD telemetry)            |
|  - Interactive Event Loop: Space (Pause), R (Reset), Left-Click (Toggle Obstacle), ESC (Quit)   |
|  - Headless Execution Mode for CI/automated testing & batch benchmarks                         |
+-------------------------------------------------------------------------------------------------+
```

---

## 3. Input & Configuration Module (`src/config.py`)

The configuration layer establishes single-source-of-truth configuration dataclasses and constants:
- **`FREE_CELL = 0` / `OCCUPIED_CELL = 1`**: Standard integer occupancy states.
- **`CARDINAL_DIRECTIONS`**: Fixed tuple `((-1, 0), (1, 0), (0, -1), (0, 1))` enforcing strict 4-connected transitions (North, South, West, East). Diagonal transitions are forbidden.
- **`GridConfig`**: Holds `rows`, `cols`, and `cell_size` (pixels). Calculates derived window dimensions `screen_width` and `screen_height`.
- **`ColorConfig`**: Curated RGB color palette for rendering background, grid lines, obstacles, start/goal markers, planned path lines, visited breadcrumbs, and status panels.
- **`SimulationConfig`**: Consolidates grid settings, colors, simulation target FPS (default 60), robot step delay (seconds per tick), and window metadata.

---

## 4. Environment & Map Module (`src/environment.py`)

The environment represents the physical space in which navigation occurs:
- **Internal Storage**: `self.grid`, a 2-dimensional NumPy array of data type `np.int8`.
- **Dimensions**: Defined by `rows` (vertical axis, 0 to `rows - 1`) and `cols` (horizontal axis, 0 to `cols - 1`).
- **Coordinate Convention**: Every logical location is represented as a 2-tuple: `(row, column)`.
- **Factory Construction**: Can be created with dimensions (defaulting to all zeros) or loaded from an arbitrary 2D array via `GridEnvironment.from_array(array)`.
- **Traversability & Neighbor Generation**: `get_neighbors(row, col)` evaluates the 4 cardinal directions and returns only cells that are in-bounds and free (`grid[r, c] == 0`).

---

## 5. Preprocessing & Validation

Defensive validation guards all system boundaries:
1. **Grid Dimensional Validation**: Rejects non-positive rows or columns (`rows <= 0` or `cols <= 0`), raising `InvalidGridError`.
2. **Dimension Rank Check**: Rejects arrays that are not strictly 2-dimensional (e.g., 1D vectors or 3D volumes), raising `InvalidGridError`.
3. **Value Whitelist**: Inspects all elements in the occupancy array. Any value other than `0` (free) or `1` (occupied) raises `InvalidGridError`.
4. **Boundary Validation**: `is_in_bounds(row, col)` verifies `0 <= row < rows` and `0 <= col < cols`. Coordinates falling outside raise `OutOfBoundsError`.
5. **Obstacle Collision Check**: `validate_position(pos)` verifies that start, goal, or waypoints are neither out of bounds nor positioned upon an occupied obstacle cell (`grid[r, c] == 1`). If occupied, an `ObstacleCollisionError` is raised.

---

## 6. Computer Vision as a FUTURE Optional Module

> [!NOTE]
> **Status: Deliberately Deferred (Future Work)**
> Computer vision, camera feeds, YOLO, OpenCV, deep learning models, ROS 2, and Gazebo are **NOT** implemented in this task.

### Future Architecture Vision:
When perception hardware (e.g., camera or LiDAR) is introduced in future tasks:
1. A raw image or point-cloud perception pipeline will detect physical objects in a continuous environment.
2. A perception adapter will map bounding boxes or segmented point clouds into discrete grid occupancy beliefs.
3. The generated occupancy grid will be passed into `GridEnvironment` via the existing `from_array()` interface.
4. The path planning and navigation modules will consume the resulting grid without requiring architectural redesign.

---

## 7. Obstacle & Map Interface

The environment exposes an explicit public API for querying and modifying map state:
- `is_in_bounds(row: int, col: int) -> bool`: Checks if coordinates lie within grid boundaries.
- `is_obstacle(row: int, col: int) -> bool`: Returns `True` if the cell contains an obstacle or is out of bounds; `False` if free.
- `is_free(row: int, col: int) -> bool`: Returns `True` if the cell is in bounds and free (value 0).
- `set_obstacle(row: int, col: int) -> None`: Sets cell value to `1`.
- `clear_cell(row: int, col: int) -> None`: Sets cell value to `0`.
- `get_neighbors(row: int, col: int) -> List[Tuple[int, int]]`: Returns list of valid 4-directional adjacent free coordinates.
- `copy() -> GridEnvironment`: Creates a deep copy of the current environment state.

---

## 8. Path-Planning Module (`src/path_planning.py`)

The path planner is completely decoupled from rendering and robot mechanics:
- **Base Interface (`BasePlanner`)**:
  All planners implement `plan(environment, start, goal) -> PathResult` and `find_path(environment, start, goal) -> Optional[List[Tuple[int, int]]]`.
- **Supported Algorithms**:
  1. **A\* Search (`AStarPlanner`)**:
     - Cost Function: $f(n) = g(n) + h(n)$.
     - $g(n)$: Exact movement cost from start to node $n$ (uniform $1.0$ per cardinal transition).
     - $h(n)$: Admissible & consistent Manhattan distance heuristic:
       $$h(n) = |r_n - r_g| + |c_n - c_g|$$
     - Uses `heapq` priority queue with deterministic FIFO integer counter tie-breaking.
  2. **Dijkstra's Algorithm (`DijkstraPlanner`)**:
     - Uniform-cost search ($f(n) = g(n)$, equivalent to A* with $h(n) = 0$).
     - Explores radially in concentric rings in order of distance from start.
     - Complete and optimal.
  3. **Breadth-First Search (`BFSPlanner`)**:
     - FIFO queue search exploring level-by-level without heuristics.
     - Guaranteed to find paths with minimal step transitions on unweighted grid graphs.
- **Factory Access**: `get_planner("A*" | "Dijkstra" | "BFS")` provides unified instantiation.
- **Return Type (`PathResult`)**:
  - `success`: Boolean indicator.
  - `path`: Ordered `List[Tuple[int, int]]` starting at start and ending at goal, or `None`.
  - `cost`: Float step count, or `float('inf')`.
  - `nodes_expanded`: Search efficiency counter.
  - `execution_time_ms`: Algorithm duration in milliseconds.
  - `error_message`: Explanatory failure reason if no path exists or coordinates are invalid.
  - `algorithm`: Name of the planning engine used.
- **Edge Cases Handled**:
  - Start equals goal: immediately returns `success=True`, `path=[start]`, `cost=0.0`.
  - Start or goal inside obstacle / out of bounds: returns `success=False` with descriptive error.
  - Unreachable goal (walled off): exhausts open set / queue, returns `success=False`, `path=None`.

---

## 9. Navigation Logic (`src/navigation.py`)

The `NavigationController` coordinates mission execution and active path monitoring:
- **Goal Setting**: Calls `planner.plan(env, robot.pos, goal)`. If successful, sets robot's path and initializes metrics.
- **Active Route Monitoring (`is_route_clear`)**: Inspects remaining unvisited waypoints on the robot's planned path.
- **Blocked Path Detection**: Before taking a step, checks `environment.is_obstacle(next_pos)`.
- **Dynamic Replanning**: If an obstacle appears on the path, the controller immediately halts forward motion, re-queries `planner.plan(env, current_robot_pos, goal)`, updates the robot's path if an alternate route exists, and logs the replanning event. If no alternate path exists, transitions robot to `RobotStatus.BLOCKED`.
- **Step Execution (`update_step`)**: Advances robot position by 1 cell per tick when path is clear and updates metrics.

---

## 10. Robot Movement Module (`src/robot.py`)

The `Robot` class represents the physical state of the autonomous agent:
- **State Machine**:
  - `IDLE`: Initialized, awaiting goal.
  - `NAVIGATING`: Actively advancing along planned route.
  - `REACHED_GOAL`: Robot has reached destination.
  - `BLOCKED`: Path obstructed and no replanned route available.
  - `FAILED`: Initialization or planning failed.
- **Waypoints & History**: Maintains `path: List[Tuple[int, int]]`, `current_step_index: int`, and `history: List[Tuple[int, int]]` (complete breadcrumb trail of traversed cells).
- **Movement Primitives**:
  - `step_forward() -> bool`: Advances one cell along path.
  - `peek_next_position() -> Optional[Tuple[int, int]]`: Non-destructive lookahead.
  - `stop_blocked()`: Transitions status to `BLOCKED`.
  - `reset(new_position)`: Clears route memory and returns robot to specified position.
- **Decoupling**: The robot does not interact with UI rendering or make path decisions; it strictly executes motion assigned by the navigation controller.

---

## 11. Visualization & Simulation Layers

The system provides two decoupled presentation interfaces on top of the shared navigation engine:

### A. Streamlit Live Web Dashboard (`app.py`)
- **Browser Accessibility**: Runs locally or over a network at `http://localhost:8501`.
- **Matplotlib Render Pipeline**: Uses non-interactive `Agg` backend to generate crisp grid plots, obstacle blocks, start/goal badges, route lines, and robot circles.
- **Interactive Scrubber**: User can scrub through the trajectory step-by-step using a slider to inspect coordinates at any waypoint.
- **Live Frame Animation**: Iterates through waypoints in real time inside a dedicated `st.empty()` container with configurable playback speeds (Slow, Normal, Fast).
- **Scenario Presets**: Supports 6 one-click scenarios: Demo Barriers, Open Grid, Narrow Corridor, U-Shaped Trap, Dense Random Clutter, and Walled-Off Goal.
- **Session State Management**: Preserves grid dimensions, selected algorithms, and latest telemetry across user interactions without retaining stale data.

### B. Pygame Desktop Simulation (`src/simulation.py` & `main.py`)
- **Desktop Window**: Hardware-accelerated desktop window for 60 FPS rendering.
- **Interactive Obstacle Placement**: Left-click on any grid cell to dynamically add or remove obstacles mid-simulation, demonstrating real-time A* replanning.
- **Headless CLI Mode (`--headless`)**: Executes simulation logic without opening a display surface, ideal for automated test runners and continuous integration.

---

## 12. Performance Metrics Module (`src/metrics.py`)

Captures empirical performance data during navigation trials:
- **Metrics Tracked**:
  - `initial_planning_time_ms`: Algorithm duration for first route computation.
  - `initial_path_length`: Length of initial planned route.
  - `initial_nodes_expanded`: Nodes evaluated by A*.
  - `steps_executed`: Total discrete steps taken by robot.
  - `replan_count`: Number of dynamic replanning events triggered.
  - `total_replan_time_ms`: Cumulative latency of all replanning computations.
  - `total_execution_time_ms`: Total navigation elapsed time.
  - `path_efficiency_ratio`: Ratio of theoretical optimal steps to actual steps executed ($L_{\text{optimal}} / N_{\text{steps}}$). A ratio of $1.0$ indicates no detour; lower values quantify the cost of dynamic detours.
- **Export & Reporting**:
  - `generate_report() -> str`: Formatted console summary table.
  - `export_json(filepath)`: Saves complete telemetry and replanning logs to disk (e.g., `results/metrics.json`).

---

## 13. Data Flow Between Modules

```
1. Configuration Setup:
   SimulationConfig ---> Environment & SimulationVisualizer

2. Problem Formulation:
   (start_pos, goal_pos) ---> NavigationController.set_goal()

3. Route Computation:
   GridEnvironment + start_pos + goal_pos ---> AStarPlanner.plan()
   AStarPlanner ---> PathResult ---> NavigationController

4. Route Assignment:
   PathResult.path ---> Robot.set_path()
   PathResult telemetry ---> MetricsTracker.record_initial_plan()

5. Navigation Execution Loop:
   NavigationController.update_step()
     |-- Query next waypoint in GridEnvironment
     |-- If free: Robot.step_forward() ---> MetricsTracker.record_step()
     |-- If blocked: AStarPlanner.plan(current_pos, goal) ---> Dynamic Replan
           |-- If new path found: Robot.set_path(new_path)
           |-- If no path exists: Robot.stop_blocked()

6. Visualization Refresh (if graphical):
   Environment + Robot + Path + Metrics ---> SimulationVisualizer.render()

7. Post-Mission Telemetry:
   MetricsTracker.end_trial() ---> Console Report & results/metrics.json
```

---

## 14. Data Conventions & Coordinate Transformations

### Coordinate Conventions:
- **Logical Grid**: Strictly 2D integer tuples `(row, column)`.
  - `row`: 0 (top) to `rows - 1` (bottom).
  - `column`: 0 (left) to `cols - 1` (right).
- **Screen / Pygame Coordinates**: Continuous pixel coordinates `(x, y)`.
  - `x`: Horizontal axis (increases to the right).
  - `y`: Vertical axis (increases downward).
- **Transformations**:
  - Cell top-left to screen:
    $$x = \text{column} \times \text{cell\_size}$$
    $$y = \text{row} \times \text{cell\_size}$$
  - Cell center to screen:
    $$x_{\text{center}} = \text{column} \times \text{cell\_size} + \lfloor \text{cell\_size} / 2 \rfloor$$
    $$y_{\text{center}} = \text{row} \times \text{cell\_size} + \lfloor \text{cell\_size} / 2 \rfloor$$
  - Screen pixel to grid cell:
    $$\text{column} = \lfloor x / \text{cell\_size} \rfloor$$
    $$\text{row} = \lfloor y / \text{cell\_size} \rfloor$$

### Movement Restrictions:
- Strictly 4-directional: $\Delta \text{row} + \Delta \text{col} = 1$.
- Diagonal movements ($\Delta \text{row} = \pm 1, \Delta \text{col} = \pm 1$) are explicitly forbidden in this architecture.

---

## 15. Error and Failure States

| Failure Condition | Module Detecting Error | Action Taken |
| :--- | :--- | :--- |
| Negative or zero grid dimensions | `GridEnvironment` | Raises `InvalidGridError` |
| Non-2D array or illegal cell values ($\ne 0, 1$) | `GridEnvironment` | Raises `InvalidGridError` |
| Start or goal outside grid dimensions | `GridEnvironment.validate_position` | Raises `OutOfBoundsError` (or `PathResult.success = False`) |
| Start or goal placed on an obstacle cell | `GridEnvironment.validate_position` | Raises `ObstacleCollisionError` (or `PathResult.success = False`) |
| Goal completely surrounded by obstacles | `AStarPlanner.plan` | Open set exhausts; returns `PathResult(success=False, path=None, cost=inf)` |
| Obstacle placed on robot's next step | `NavigationController.update_step` | Halts forward motion; triggers immediate dynamic replan |
| Obstacle blocks all alternate routes | `NavigationController.replan` | Replan fails; transitions robot to `RobotStatus.BLOCKED` |
| Pygame display unavailable (headless system) | `Simulation.run` | Gracefully catches missing display and executes `run_headless()` |

---

## 16. Planned Future Extensions

| Feature | Architectural Placement | Status |
| :--- | :--- | :--- |
| **8-Connected Diagonal Movement** | `CARDINAL_DIRECTIONS` in `config.py` + Euclidean/Octile heuristic in `path_planning.py` | Future Task |
| **Continuous Robot Kinematics** | Differential-drive kinematic equations in `robot.py` (replacing discrete grid steps) | Future Task |
| **LiDAR / Sonar Raycasting** | Distance sensor range simulation in `environment.py` | Future Task |
| **Dynamic Moving Obstacles** | Timed moving obstacle agents with velocity vectors in `environment.py` | Future Task |
| **Computer Vision Object Detection** | Camera stream with bounding boxes converted to occupancy in new `src/perception/` package | Future Task |
| **D* Lite / Lifelong Planning A*** | Incremental replanning module to avoid re-running full A* from scratch | Future Task |
| **ROS 2 Navigation Stack Integration** | ROS 2 Nav2 wrapper nodes around `src/` modular components | Future Task |
