# Notes on the fluid solver and its functions

Read this alongside `src/simulation/solver.py` and `tests/test_solver.py`.
The solver advances the fluid one timestep; the tests check both its API and
whether it still computes the same results as the original script.

## What the arrays mean

One `FluidState` holds three two-dimensional arrays:

| Field | Meaning |
| --- | --- |
| `u` | Horizontal velocity at each grid point. |
| `v` | Vertical velocity at each grid point. |
| `p` | Pressure at each grid point. |

The solver stores all three quantities on the same grid. This is what
**collocated** means here. For the default configuration, each array is
`(41, 41)`. An entry is addressed as `[row, column]`, or `[y_index, x_index]`.
Row `0` is the bottom wall, and row `-1` is the top wall.

The problem is a lid-driven cavity: the top wall moves horizontally and drags
the fluid, while the other walls stay still.

Some NumPy slices used throughout the code are:

| Slice | Meaning |
| --- | --- |
| `[1:-1, 1:-1]` | Interior points, excluding all four walls. |
| `[1:-1, 2:]` | Right neighbors of the interior points. |
| `[1:-1, :-2]` | Left neighbors of the interior points. |
| `[2:, 1:-1]` | Neighbors one row above the interior points. |
| `[:-2, 1:-1]` | Neighbors one row below the interior points. |
| `[-1, :]` | The entire top row. |
| `[:, 0]` | The entire left column. |

These slices calculate values across the grid at once, instead of looping over
individual points in Python.

## `src/simulation/state.py`

### `FluidState.__post_init__()`

The dataclass calls this automatically after construction. It checks that `u`,
`v`, and `p` are nonempty 2D arrays of the same shape, containing finite real
numbers. `NaN` and infinity are rejected.

It then copies the arrays into `float64` storage. Copying means changing one
state will not accidentally change another state that was built from its arrays.
Converting integer arrays to floating point also allows fractional velocities
and pressures to be stored.

### `FluidState.to_array()`

Stacks the three fields into a new array with shape `(3, H, W)`:

```python
channels = state.to_array()
# channels[0] is u, channels[1] is v, channels[2] is p.
```

This will be a useful format for future model inputs. The returned array owns
separate storage, so modifying it does not modify the state.

### `FluidState.from_array(array)`

Performs the reverse conversion. It checks that the input has shape `(3, H, W)`,
then passes its three channels into the `FluidState` constructor. The constructor
does the remaining validation and copying.

It is a `classmethod`, so you call `FluidState.from_array(array)` without first
creating a state object.

## `src/simulation/solver.py`

### `SolverConfig.__post_init__()`

Validates the settings when a configuration is created. It checks that the grid
has at least three points per axis, pressure iterations are a positive integer,
and physical parameters are finite and positive. The lid velocity may be zero or
negative: those values mean a stationary lid or movement in the opposite direction.
The safety factor must be greater than zero and at most one.

The dataclass uses `frozen=True`, so settings cannot be changed after creation.
To use different settings, construct another configuration and solver.

### `NavierStokesSolver.__init__(config)`

Stores the configuration and calculates the spacing between adjacent grid points:

```text
h = domain_size / (n_points - 1)
```

There are `n_points - 1` gaps between `n_points` points. The constructor also
checks the original solver's diffusion timestep bound:

```text
dt <= stability_safety_factor * 0.5 * h**2 / viscosity
```

A timestep above that bound raises `ValueError`. This preserves the baseline's
check; it does not check every possible cause of instability, such as very large
advective velocities.

### `NavierStokesSolver.config`

A property exposing the configuration as `solver.config`. You can inspect it
without calling a method, but cannot replace it through this property.

### `NavierStokesSolver.initial_state()`

Creates fresh, all-zero `u`, `v`, and `p` arrays of the configured grid size.

The top row is initially zero too. The moving-lid velocity is first imposed
inside `step()`. This timing matches the original script and matters when
comparing its early timesteps.

### `NavierStokesSolver._validate_state(state)`

Checks that all three arrays match this solver's grid size and contain finite
values. A valid `FluidState` could have a different grid size, and callers can
edit state arrays, so the solver checks again before using them.

Names beginning with `_` are internal helpers. Normal usage only needs
`initial_state()`, `step()`, and optionally `run()`.

### `NavierStokesSolver._central_difference_x(field)`

Approximates how a field changes in the horizontal direction:

```text
df/dx at a point ~= (value to the right - value to the left) / (2*h)
```

For velocity, this supplies part of the convection or divergence calculation.
For pressure, it supplies a pressure gradient. The output has the same shape as
the input. Only interior entries are computed; boundary entries stay zero,
matching the baseline. Those zeros are a detail of this numerical helper, not
a statement that every field has zero physical derivative at every wall.

### `NavierStokesSolver._central_difference_y(field)`

Does the same calculation vertically, using the values one row above and below.
Rows increase with y in this solver, so the subtraction is above minus below.

### `NavierStokesSolver._laplace(field)`

Approximates the sum of the second derivatives in x and y:

```text
Laplacian(field) ~= (left + below - 4*center + right + above) / h**2
```

This measures local curvature. In the momentum update, multiplying it by
viscosity gives the diffusion term, which smooths variations in velocity.
Only interior values are computed.

### `NavierStokesSolver._apply_velocity_boundary_conditions(u, v)`

Writes wall values directly into the supplied velocity arrays. Horizontal
velocity is zero on the bottom and side walls, then set to `top_wall_velocity`
on the lid. Vertical velocity is zero on every wall.

The order matters: assigning the lid last gives its horizontal velocity to both
top corners. This helper mutates its arguments, but `step()` only passes newly
calculated arrays to it, keeping the input state unchanged.

### `NavierStokesSolver._solve_pressure(pressure, rhs)`

Approximately solves the pressure Poisson equation using a fixed number of
Jacobi iterations. Here, `rhs` means the right-hand side of the equation.
It is calculated from the tentative velocity's divergence.

Each iteration builds a fresh pressure array. An interior value comes from
the four neighboring values in the previous pressure iterate and the local RHS:

```text
p_new = (p_left + p_below + p_right + p_above - h**2 * rhs) / 4
```

After each iteration, it copies neighboring interior values onto the right,
bottom, and left pressure boundaries. This implements the baseline's zero
normal pressure-gradient conditions. The top pressure is then set to zero.
The boundary assignment order is preserved, including its effect on corners.

The first iterate is the input state's pressure, so pressure information carries
between timesteps. The method does not overwrite that input array: each new
iterate uses separate storage. The loop count is `pressure_iterations`; these
iterations improve the pressure estimate within one timestep and do not advance
simulation time.

### `NavierStokesSolver.step(state)`

Advances the fluid by exactly `config.dt` and returns a new `FluidState`.
The original state is left unchanged. The calculation has these stages:

1. Validate the state and calculate velocity derivatives and Laplacians.
2. Estimate the new velocities using convection and viscosity, temporarily
   leaving out the pressure gradient:

   ```text
   u_tentative = u + dt * (-(u*du/dx + v*du/dy) + viscosity*Laplacian(u))
   v_tentative = v + dt * (-(u*dv/dx + v*dv/dy) + viscosity*Laplacian(v))
   ```

   Convection describes velocity transported by the moving fluid. Diffusion
   describes viscosity smoothing the velocity field.

3. Impose the wall velocities on these tentative arrays.
4. Calculate their divergence and construct the pressure RHS:

   ```text
   rhs = density / dt * (du_tentative/dx + dv_tentative/dy)
   ```

   Divergence measures local net outflow. The incompressible-flow target is
   zero divergence.

5. Solve for pressure, starting from the previous state's pressure.
6. Calculate the pressure gradient and correct the tentative velocities:

   ```text
   u_next = u_tentative - dt / density * dp/dx
   v_next = v_tentative - dt / density * dp/dy
   ```

7. Reapply wall velocities and return the new velocities and pressure together.

The pressure correction targets incompressibility. A finite pressure iteration
count and the inherited discrete operators do not guarantee exactly zero
divergence; the refactor preserves the original numerical behavior.

### `NavierStokesSolver.run(initial_state, num_steps)`

Calls `step()` repeatedly and returns `[S_0, S_1, ..., S_n]`. The list contains
`num_steps + 1` entries because it includes the initial state.

The initial entry is a copy, and later entries come from successive calls to
`step()`. This preserves separate snapshots for future training pairs. Passing
zero steps returns a list containing only the initial snapshot. For large runs,
storing every snapshot uses more memory than keeping just the latest state.

## Visualization and the demonstration

### `visualize(state, domain_size)` in `src/simulation/visualization.py`

Creates coordinates for the grid, draws pressure as filled contours, and overlays
velocity streamlines. It returns the Matplotlib figure and axes so the caller
can display, save, or customize them. It does not call `plt.show()`.

`TYPE_CHECKING` lets the module declare Matplotlib return types without importing
those classes at runtime. The plotting import occurs inside `visualize()`.
The numerical solver never imports this visualization module.

### `main()` in `scripts/run_simulation.py`

Parses the command-line options, builds a default solver, and repeatedly calls
`step()`. It prints the final array shape, ranges, and finite-value checks.
`--steps` changes the run length, `--no-plot` skips plotting, and `--save-plot`
saves an image using the noninteractive `Agg` backend. With neither plotting
option, the script shows the figure.

The `if __name__ == "__main__":` block runs `main()` when the file is executed
as a script. Importing the file does not start a simulation.

## `tests/_baseline.py`

### `run_baseline(config, num_steps)`

Runs the original script's numerical loop and returns its final `(u, v, p)`
arrays. This supplies a reference independent of the new solver methods.

The original function normally finishes by plotting and does not return arrays.
The helper adapts it in memory using Python's abstract syntax tree, or AST:

1. Read the original file and parse its source into Python syntax nodes.
2. Locate `main()` and its final plotting section.
3. Keep the numerical statements and append a return of the final arrays.
4. Keep the original constant assignments and the adapted function, leaving out
   imports and the script's automatic call to `main()`.
5. Compile and execute this temporary version with NumPy available and a `tqdm`
   replacement that simply returns its input iterable.
6. Set the original constants to match the requested configuration and run it.

The helper changes no files on disk. Its `namespace` dictionary supplies the
globals used by the temporary function. Replacing the progress wrapper removes
progress output without changing the iteration count.

This helper depends on the original script's current function and variable
names and plotting structure. It is a regression-test adapter for that file,
not a general loader for arbitrary simulation scripts.

## `tests/test_solver.py`

Each `test_...` method is discovered by `unittest`. Assertions describe a required
behavior; a failed assertion makes that test fail. `subTest(...)` labels multiple
cases within one test so failures identify the relevant input.

### State tests

| Function | What it checks and why |
| --- | --- |
| `test_channel_order_round_trip_and_storage_independence()` | Packs and unpacks distinct values to check shape, channel order, and copying. `np.shares_memory` checks for shared storage. |
| `test_constructor_copies_and_promotes_integer_fields()` | Starts all three fields from one integer array, then changes `u`. Checks that the source, `v`, and `p` stay unchanged and fractional values are retained. |
| `test_invalid_states()` | Checks that invalid dimensions, mismatched field shapes, nonfinite values, complex values, and text raise `ValueError`. |

### Solver tests

| Function | What it checks and why |
| --- | --- |
| `setUp()` | Creates a fresh default solver before each test in `SolverTests`. This prevents tests from depending on another test's solver. |
| `test_initial_state_is_zero_with_expected_shape()` | Confirms a default state packs into `(3, 41, 41)` and starts entirely at zero. |
| `test_boundary_conditions_and_corner_precedence()` | Runs ten steps with a negative lid speed and checks wall velocities, top-corner precedence, and pressure boundary values. |
| `test_step_accepts_read_only_input_and_returns_independent_arrays()` | Makes a nonzero input read-only, runs a step, and checks unchanged values and separate output storage. An accidental write to the input would also raise an error. |
| `test_run_calls_step_and_preserves_independent_snapshots()` | Uses a mock with `wraps` to count real `step()` calls. Compares trajectory values with repeated steps, then edits snapshots to check storage independence. Also checks zero steps. |
| `test_invalid_run_length_and_input_state()` | Checks rejection of negative, fractional, or boolean step counts, mismatched grid sizes, and a state edited to contain `NaN`. |
| `test_stationary_lid_preserves_rest()` | Sets lid speed to zero and confirms an initially resting fluid remains at rest. |
| `test_stability_bound_and_immutable_configuration()` | Checks acceptance exactly at the timestep bound and rejection just above it. Also checks that configuration fields and the solver's configuration property cannot be reassigned. |
| `test_invalid_configuration()` | Exercises invalid values for each setting, including invalid integers, nonpositive parameters, nonfinite values, and invalid safety factors. |

In the stability test, `replace(config, dt=...)` creates a new dataclass instance.
`np.nextafter(limit, np.inf)` gives the next representable floating-point number
above the limit, testing the boundary precisely.

### Baseline comparison tests

| Function | What it checks and why |
| --- | --- |
| `assert_matches_baseline(config, steps)` | A shared helper that runs both implementations with matching settings, checks new states are finite, and compares final `u`, `v`, and `p`. |
| `test_original_experiment_and_early_timesteps()` | Calls the comparison helper for 0, 1, 2, 10, and 500 steps using the original settings. Early steps catch initialization differences; 500 steps checks the original experiment. |
| `test_alternative_configurations()` | Compares three other configurations after 30 steps, exercising different grids, physical parameters, pressure iteration counts, and lid velocities. Includes the minimum supported grid. |

The comparison uses `np.allclose(actual, expected, rtol=1e-12, atol=1e-12)`.
For each value, the permitted difference is approximately
`1e-12 + 1e-12 * abs(expected)`. This allows tiny floating-point differences
while requiring close agreement. A failure reports the largest absolute error.

Matching the baseline checks that the refactor preserved its computations.
It does not independently establish the physical accuracy of the original
discretization.

## Package files

`src/simulation/__init__.py` re-exports `FluidState`, `SolverConfig`, and
`NavierStokesSolver`, allowing `from src.simulation import ...`.
Its `__all__` lists those public names. `src/__init__.py` and
`tests/__init__.py` mark their directories as packages and contain descriptive
module docstrings; they do not run simulations or tests.

## A small example connecting the functions

```python
from src.simulation import NavierStokesSolver, SolverConfig

solver = NavierStokesSolver(SolverConfig())  # Validate settings and timestep.
state_0 = solver.initial_state()            # Fresh zero arrays.
state_1 = solver.step(state_0)              # First timestep; impose the lid.
state_2 = solver.step(state_1)              # Second timestep; carry pressure forward.

input_array = state_1.to_array()            # Shape (3, 41, 41).
target_array = state_2.to_array()           # The next-state target for this input.
```

Both states remain available because `step()` returns new arrays. A future model
can use such consecutive pairs to learn the next-state mapping.
