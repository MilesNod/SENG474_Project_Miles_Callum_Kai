# Reusable cavity solver

This package adds a reusable API alongside the existing scripts. The original
files are retained unchanged. The solver uses NumPy only; plotting is optional.
Python 3.10+ and NumPy are required. The demonstration plot also needs Matplotlib.
Tests use the Python standard library's `unittest`.

Run commands from the repository root:

```bash
python scripts/run_simulation.py --no-plot
python scripts/run_simulation.py
python scripts/run_simulation.py --save-plot cavity.png
python -m unittest discover -s tests -v
```

The demonstration defaults to the original 41 x 41 grid and 500 timesteps.
`--steps N` changes the run length. The solver itself never opens a window.

```python
from src.simulation import FluidState, NavierStokesSolver, SolverConfig

solver = NavierStokesSolver(SolverConfig())
state = solver.initial_state()
state_1 = solver.step(state)
state_2 = solver.step(state_1)

channels = state_2.to_array()       # (3, 41, 41), ordered [u, v, p]
restored = FluidState.from_array(channels)
trajectory = solver.run(state, num_steps=500)  # [S_0, ..., S_500]
```

Each state owns float64 arrays of equal shape. Conversion to/from channel arrays
copies storage. `step()` reads the input arrays and returns independent arrays.
`run()` includes a copy of the initial state and invokes `step()` for each
transition. For long trajectories, use a loop over `step()` if storing every state
would consume too much memory. State arrays are mutable so callers can supply
initial conditions; all three fields must match the configured square grid.

## Preserved numerical behavior

All fields use the baseline's collocated representation. Array axis 0 increases
in y and axis 1 increases in x. The last row is the moving lid. The implementation
preserves:

- Interior central differences with zero derivative entries at boundaries.
- The five-point Laplacian and explicit convection/diffusion momentum step.
- Velocity boundary assignments, with the moving lid winning at top corners.
- Tentative-velocity divergence as the pressure Poisson right-hand side.
- The fixed number of Jacobi pressure iterations, seeded from the previous
  state's pressure, and the original pressure boundary assignment order.
- Pressure-gradient velocity correction and reapplication of velocity boundaries.
- The diffusion stability check:
  `dt <= stability_safety_factor * 0.5 * (domain_size / (n_points - 1))**2 / viscosity`.

`initial_state()` returns zero velocities and pressure, including the lid, just
as the original script does before entering its loop. The lid is applied during
the first timestep. Applying it at initialization would change the trajectory.
Pressure is part of the timestep state because a finite number of pressure
iterations depends on the previous iterate.

Configurations are immutable. Grid sizes below 3, nonpositive physical parameters,
zero pressure iterations, nonfinite values, and safety factors outside (0, 1] are
rejected. Unstable diffusion timesteps raise `ValueError` during solver creation.
The inherited stability bound does not check advective CFL limits for arbitrary
lid speeds or initial velocities. No new CFL rule or numerical scheme is added.

## Baseline verification

`tests/_baseline.py` reads the current `Nav-Stokes-Solver-From-CodeAlong/nav-stokes-eq.py`
and executes its numerical statements, removing imports/final plotting and
appending a return of the computed arrays. Its equations are not rewritten or
replaced with the new solver. Only the run constants are set from the test
configuration; the progress wrapper becomes an identity function.

Regression tests compare all three fields using `numpy.allclose` with
`rtol=atol=1e-12` at 0, 1, 2, 10, and 500 steps under the original configuration,
and after 30 steps under three alternative configurations. Other tests cover
array conversion/ownership, boundary values, input nonmutation, finite outputs,
trajectory behavior, and configuration/stability validation.

The baseline attributes its numerical example to
https://www.youtube.com/watch?v=BQLvNLgMTQE. This package retains that attribution.
