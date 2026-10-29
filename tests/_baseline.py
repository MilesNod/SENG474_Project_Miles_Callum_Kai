"""Execute the existing baseline's numerical code without plotting or tqdm.

This reads the current working-tree file, not a copied solver implementation.
Only imports and the final plotting statements are omitted. The main function's
numerical statements, including all boundary assignments, are compiled unchanged;
an appended return exposes the final arrays for regression comparisons.
"""

import ast
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.simulation import SolverConfig


def run_baseline(
    config: SolverConfig, num_steps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run the original numerical loop using the requested matching parameters."""
    path = Path(__file__).resolve().parents[1] / "nav-stokes-equation-solver" / "nav-stokes-eq.py"
    module = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    plot_start = next(
        index for index, node in enumerate(main.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "plt"
    )
    main.body = main.body[:plot_start] + [
        ast.Return(value=ast.Tuple(
            elts=[ast.Name(id=name, ctx=ast.Load()) for name in ("u_prev", "v_prev", "p_prev")],
            ctx=ast.Load(),
        ))
    ]
    module.body = [node for node in module.body if isinstance(node, ast.Assign)] + [main]
    ast.fix_missing_locations(module)
    namespace = {"np": np, "tqdm": lambda iterable: iterable}
    exec(compile(module, str(path), "exec"), namespace)
    namespace.update(
        N_POINTS=config.n_points,
        DOMAIN_SIZE=config.domain_size,
        N_ITERATIONS=num_steps,
        TIME_STEP_LENGTH=config.dt,
        KINEMATIC_VISCOSITY=config.viscosity,
        DENSITY=config.density,
        HORIZONTAL_VELOCITY_TOP=config.top_wall_velocity,
        N_PRESSURE_POISSON_ITERATION=config.pressure_iterations,
        STABILITY_SAFETY_FACTOR=config.stability_safety_factor,
    )
    return namespace["main"]()
