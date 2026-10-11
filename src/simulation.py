"""Simulation and visualization module using Pygame.

Coordinates the main simulation loop, renders grid cells, obstacles, planned routes,
and robot motion. Decoupled from path planning decisions.
"""

import sys
import time
from typing import Optional, Tuple

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from src.config import SimulationConfig, GridConfig, ColorConfig
from src.environment import GridEnvironment
from src.robot import Robot, RobotStatus
from src.navigation import NavigationController
from src.metrics import MetricsTracker, NavigationMetrics


class SimulationVisualizer:
    """Handles Pygame rendering of the occupancy grid, path trails, and robot entity."""

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize Pygame display surface and fonts.

        Args:
            config: SimulationConfig defining dimensions and color palette.
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is not installed. Install pygame to use graphical visualization.")

        pygame.init()
        self.config: SimulationConfig = config
        self.grid_cfg: GridConfig = config.grid
        self.colors: ColorConfig = config.colors

        # Extra height for bottom status panel
        self.panel_height: int = 70
        self.window_width: int = self.grid_cfg.screen_width
        self.window_height: int = self.grid_cfg.screen_height + self.panel_height

        self.screen: pygame.Surface = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        pygame.display.set_caption(self.config.window_title)
        self.clock = pygame.time.Clock()

        # Initialize fonts
        pygame.font.init()
        self.font = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 16)
        self.font_bold = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 18, bold=True)

    def draw_grid(self, env: GridEnvironment) -> None:
        """Render all grid cells and obstacle blocks."""
        cs = self.grid_cfg.cell_size

        for r in range(env.rows):
            for c in range(env.cols):
                rect = (c * cs, r * cs, cs, cs)
                if env.is_obstacle(r, c):
                    pygame.draw.rect(self.screen, self.colors.OBSTACLE, rect)
                else:
                    pygame.draw.rect(self.screen, self.colors.FREE, rect)
                pygame.draw.rect(self.screen, self.colors.GRID_LINE, rect, width=1)

    def draw_path(self, path: list) -> None:
        """Draw planned route waypoints and connecting lines."""
        if not path or len(path) < 2:
            return

        cs = self.grid_cfg.cell_size
        points = [
            (c * cs + cs // 2, r * cs + cs // 2)
            for r, c in path
        ]

        # Draw connecting line
        pygame.draw.lines(self.screen, self.colors.PATH_LINE, False, points, width=3)

        # Draw waypoint nodes
        for pt in points:
            pygame.draw.circle(self.screen, self.colors.PATH, pt, radius=max(2, cs // 6))

    def draw_start_and_goal(self, start: Tuple[int, int], goal: Optional[Tuple[int, int]]) -> None:
        """Draw distinct markers for start and target destination."""
        cs = self.grid_cfg.cell_size

        # Start cell
        sr, sc = start
        s_rect = (sc * cs + 2, sr * cs + 2, cs - 4, cs - 4)
        pygame.draw.rect(self.screen, self.colors.START, s_rect, border_radius=4)

        # Goal cell
        if goal is not None:
            gr, gc = goal
            g_rect = (gc * cs + 2, gr * cs + 2, cs - 4, cs - 4)
            pygame.draw.rect(self.screen, self.colors.GOAL, g_rect, border_radius=4)

    def draw_robot(self, robot: Robot) -> None:
        """Draw robot entity and past movement history trail."""
        cs = self.grid_cfg.cell_size

        # Past breadcrumbs
        for hr, hc in robot.history[:-1]:
            hx = hc * cs + cs // 2
            hy = hr * cs + cs // 2
            pygame.draw.circle(self.screen, self.colors.VISITED, (hx, hy), radius=max(2, cs // 8))

        # Robot body
        rx = robot.col * cs + cs // 2
        ry = robot.row * cs + cs // 2
        pygame.draw.circle(self.screen, self.colors.ROBOT, (rx, ry), radius=max(4, cs // 3))
        # Inner accent core
        pygame.draw.circle(self.screen, (255, 255, 255), (rx, ry), radius=max(2, cs // 6))

    def draw_status_panel(
        self,
        robot_status: RobotStatus,
        step_count: int,
        replan_count: int,
        paused: bool,
    ) -> None:
        """Render HUD panel at the bottom with real-time state telemetry."""
        panel_rect = (0, self.grid_cfg.screen_height, self.window_width, self.panel_height)
        pygame.draw.rect(self.screen, self.colors.PANEL_BG, panel_rect)

        # Top border line for panel
        pygame.draw.line(
            self.screen,
            (50, 65, 85),
            (0, self.grid_cfg.screen_height),
            (self.window_width, self.grid_cfg.screen_height),
            width=2,
        )

        status_text = f"Status: {robot_status.value}"
        if paused:
            status_text += " [PAUSED]"

        info_text = f"Steps: {step_count} | Replans: {replan_count} | Controls: [SPACE] Pause | [R] Reset | [Click] Toggle Obstacle"

        txt_status = self.font_bold.render(status_text, True, self.colors.TEXT_COLOR)
        txt_info = self.font.render(info_text, True, (189, 195, 199))

        self.screen.blit(txt_status, (12, self.grid_cfg.screen_height + 8))
        self.screen.blit(txt_info, (12, self.grid_cfg.screen_height + 36))

    def refresh(self) -> None:
        """Update the Pygame display buffer."""
        pygame.display.flip()


class Simulation:
    """Master orchestrator combining environment, navigation logic, and visualization loop."""

    def __init__(
        self,
        environment: GridEnvironment,
        start_position: Tuple[int, int],
        goal_position: Tuple[int, int],
        config: Optional[SimulationConfig] = None,
        planner: Optional[object] = None,
    ) -> None:
        """Initialize simulation environment, planner, robot, and controller.

        Args:
            environment: Initialized GridEnvironment.
            start_position: Initial logical (row, col) position for the robot.
            goal_position: Target logical (row, col) destination.
            config: Optional custom SimulationConfig.
            planner: Optional BasePlanner instance (defaults to AStarPlanner).
        """
        self.config: SimulationConfig = config or SimulationConfig()
        self.environment: GridEnvironment = environment
        self.start_position: Tuple[int, int] = start_position
        self.goal_position: Tuple[int, int] = goal_position

        # Internal subsystems
        from src.path_planning import AStarPlanner, BasePlanner

        self.planner = planner if isinstance(planner, BasePlanner) else AStarPlanner()
        self.robot = Robot(initial_position=self.start_position)
        self.metrics = MetricsTracker()
        self.controller = NavigationController(
            environment=self.environment,
            planner=self.planner,
            robot=self.robot,
            metrics=self.metrics,
        )

        self.paused: bool = False
        self.running: bool = False

    def reset(self) -> None:
        """Reset robot and re-plan path to goal."""
        self.robot.reset(self.start_position)
        self.controller.set_goal(self.goal_position)

    def run_headless(self, max_steps: int = 200) -> NavigationMetrics:
        """Execute the simulation headlessly without a graphical Pygame window.

        Useful for unit tests, benchmarking, and batch performance evaluation.

        Args:
            max_steps: Maximum step iterations before emergency cutoff.

        Returns:
            Final NavigationMetrics summary.
        """
        self.controller.set_goal(self.goal_position)

        steps = 0
        while steps < max_steps:
            if self.robot.status in (
                RobotStatus.REACHED_GOAL,
                RobotStatus.BLOCKED,
                RobotStatus.FAILED,
            ):
                break
            self.controller.update_step()
            steps += 1

        if self.metrics and self.metrics.metrics.final_robot_status == "IDLE" and self.robot.status != RobotStatus.IDLE:
            self.metrics.end_trial(
                final_status=self.robot.status.value,
                success=(self.robot.status == RobotStatus.REACHED_GOAL),
            )

        return self.metrics.metrics

    def run(self) -> NavigationMetrics:
        """Start the interactive Pygame visualization loop.

        Returns:
            Final NavigationMetrics summary upon completion or exit.
        """
        if not PYGAME_AVAILABLE:
            print("[WARN] Pygame not available. Falling back to headless execution.")
            return self.run_headless()

        visualizer = SimulationVisualizer(self.config)
        self.controller.set_goal(self.goal_position)
        self.running = True

        last_step_time = time.time()

        while self.running:
            # Handle user input events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_R or event.key == pygame.K_r:
                        self.reset()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Left click toggles obstacle at clicked grid cell
                    mx, my = pygame.mouse.get_pos()
                    if my < self.config.grid.screen_height:
                        r, c = GridEnvironment.screen_to_grid(mx, my, self.config.grid.cell_size)
                        if (r, c) != self.robot.position and (r, c) != self.start_position and (r, c) != self.goal_position:
                            if self.environment.is_obstacle(r, c):
                                self.environment.clear_cell(r, c)
                            else:
                                self.environment.set_obstacle(r, c)

            # Step autonomous navigation if not paused
            now = time.time()
            if not self.paused and (now - last_step_time) >= self.config.robot_step_delay:
                self.controller.update_step()
                last_step_time = now

            # Render scene
            visualizer.screen.fill(self.config.colors.BACKGROUND)
            visualizer.draw_grid(self.environment)
            visualizer.draw_path(self.robot.path)
            visualizer.draw_start_and_goal(self.start_position, self.goal_position)
            visualizer.draw_robot(self.robot)
            visualizer.draw_status_panel(
                robot_status=self.robot.status,
                step_count=self.metrics.metrics.steps_executed,
                replan_count=self.metrics.metrics.replan_count,
                paused=self.paused,
            )

            visualizer.refresh()
            visualizer.clock.tick(self.config.fps)

        pygame.quit()
        return self.metrics.metrics
