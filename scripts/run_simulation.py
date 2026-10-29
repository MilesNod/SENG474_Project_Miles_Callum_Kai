"""Run the original 500-step experiment using the reusable API."""

import argparse
from pathlib import Path
import sys

# Allow both `python scripts/run_simulation.py` and `python -m scripts.run_simulation`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.simulation import NavierStokesSolver, SolverConfig


def main() -> None:
    """Run the cavity, optionally saving a plot or avoiding plotting entirely."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true", help="Run without Matplotlib.")
    parser.add_argument("--save-plot", type=Path, help="Save a plot instead of opening a window.")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.no_plot and args.save_plot is not None:
        parser.error("--no-plot and --save-plot cannot be used together")

    solver = NavierStokesSolver(SolverConfig())
    state = solver.initial_state()
    for _ in range(args.steps):
        state = solver.step(state)
    print(f"Completed {args.steps} steps; state shape: {state.to_array().shape}")
    for name, field in (("u", state.u), ("v", state.v), ("p", state.p)):
        print(f"{name}: min={field.min():.8g}, max={field.max():.8g}, finite={np.isfinite(field).all()}")

    if not args.no_plot:
        if args.save_plot is not None:
            import matplotlib
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from src.simulation.visualization import visualize

        figure, _ = visualize(state, solver.config.domain_size)
        if args.save_plot is None:
            plt.show()
        else:
            figure.savefig(args.save_plot, dpi=150, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved plot to {args.save_plot}")


if __name__ == "__main__":
    main()
