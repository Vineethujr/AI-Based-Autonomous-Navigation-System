"""Main entrypoint for the AI-Based Autonomous Navigation System.

Initializes the occupancy grid environment, places sample obstacles,
configures start and goal positions, and launches either the interactive
Pygame visualization or a headless simulation session.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from src.config import SimulationConfig, GridConfig
from src.environment import GridEnvironment
from src.simulation import Simulation


def build_demo_environment(rows: int = 20, cols: int = 20) -> Tuple[GridEnvironment, Tuple[int, int], Tuple[int, int]]:
    """Build a sample occupancy grid with non-trivial obstacles for demonstration.

    Args:
        rows: Grid row count.
        cols: Grid column count.

    Returns:
        Tuple of (environment, start_pos, goal_pos).
    """
    env = GridEnvironment(rows=rows, cols=cols)

    # Place sample wall barriers with navigable gaps to showcase A* planning
    # Vertical wall 1 with gap
    for r in range(4, 16):
        if r not in (8, 9):  # Gap at rows 8 and 9
            env.set_obstacle(r, 6)

    # Vertical wall 2 with gap
    for r in range(3, 17):
        if r not in (13, 14):  # Gap at rows 13 and 14
            env.set_obstacle(r, 13)

    # Horizontal barrier block
    for c in range(8, 12):
        env.set_obstacle(10, c)

    start_pos: Tuple[int, int] = (2, 2)
    goal_pos: Tuple[int, int] = (17, 17)

    return env, start_pos, goal_pos


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Based Autonomous Navigation System (2D Grid Simulation)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless CLI mode without opening a Pygame graphical window.",
    )
    parser.add_argument(
        "--save-metrics",
        type=str,
        default="results/metrics.json",
        help="Path where session metrics should be saved as JSON (default: results/metrics.json).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Grid rows (default: 20).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=20,
        help="Grid cols (default: 20).",
    )
    return parser.parse_args()


def main() -> int:
    """Application main function."""
    args = parse_arguments()

    print("=" * 60)
    print(" AI Autonomous Navigation System - 2D Simulation")
    print(" Architecture: Modular Grid Perception & A* Search")
    print("=" * 60)

    # 1. Environment & Problem setup
    env, start_pos, goal_pos = build_demo_environment(rows=args.rows, cols=args.cols)
    print(f"[*] Grid size: {env.rows}x{env.cols}")
    print(f"[*] Start coordinate: {start_pos} (row, col)")
    print(f"[*] Goal coordinate:  {goal_pos} (row, col)")

    # 2. Configuration
    config = SimulationConfig(grid=GridConfig(rows=args.rows, cols=args.cols, cell_size=32))

    # 3. Initialize simulation orchestrator
    sim = Simulation(
        environment=env,
        start_position=start_pos,
        goal_position=goal_pos,
        config=config,
    )

    # 4. Run interactive or headless
    if args.headless:
        print("[*] Running simulation in HEADLESS mode...")
        metrics = sim.run_headless()
    else:
        print("[*] Launching interactive Pygame simulation window...")
        print("[*] Interaction hints:")
        print("    - SPACE: Pause / Resume simulation")
        print("    - R:     Reset simulation to start")
        print("    - Left Click on cell: Toggle obstacle to test replanning!")
        print("    - ESC:   Exit simulation")
        metrics = sim.run()

    # 5. Print metrics and export
    print()
    print(sim.metrics.generate_report())

    if args.save_metrics:
        sim.metrics.export_json(args.save_metrics)
        print(f"[*] Metrics successfully exported to '{args.save_metrics}'")

    return 0 if metrics.success else 1


if __name__ == "__main__":
    sys.exit(main())
