"""
The baseline credits https://www.youtube.com/watch?v=BQLvNLgMTQE as a similar structure is used
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from .state import FluidState


@dataclass(frozen=True)
class SolverConfig:
    """
    This solver represents a square lid-driven cavity - just like the baseline credit

    The stability safety factor must be in (0, 1]. The solver checks the
    baseline's diffusion timestep bound; this is not an advective CFL check.
    """

    # Class Variables
    n_points: int = 41
    domain_size: float = 1.0
    dt: float = 0.001
    viscosity: float = 0.1
    density: float = 1.0
    pressure_iterations: int = 50
    top_wall_velocity: float = 1.0
    stability_safety_factor: float = 0.5

    def __post_init__(self) -> None:
        for name, minimum in (("n_points", 3), ("pressure_iterations", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in (
            "domain_size", "dt", "viscosity", "density",
            "top_wall_velocity", "stability_safety_factor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
            if name != "top_wall_velocity" and value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.stability_safety_factor > 1:
            raise ValueError("stability_safety_factor must be <= 1.")


class NavierStokesSolver:
    """Advance collocated u, v, p with the baseline projection method.

    Set-up/relavent info:
    u -> Horizontal velocity at each grid point
    v -> vertical velocity at each point
    p -> the pressure at each point
    All points are on the same grid, thus collocated

    A point in the simulation is accessed via [row, col]
    Row 0 is the bottom wall(floor); the final row(top) is the moving lid. 
    Boundary assignments intentionally give the lid precedence at both top corners.
    The pressure iterate starts from the input state's pressure each timestep.
    """

    def __init__(self, config: SolverConfig = SolverConfig()) -> None:
        self._config = config
        self._element_length = config.domain_size / (config.n_points - 1)
        maximum_dt = (
            config.stability_safety_factor
            * (0.5 * self._element_length**2 / config.viscosity)
        )
        if config.dt > maximum_dt:
            raise ValueError(
                f"Unstable timestep dt={config.dt:g}: the baseline diffusion "
                f"stability bound requires dt <= {maximum_dt:g}."
            )

    @property
    def config(self) -> SolverConfig:
        """Return the immutable parameters used by this solver."""
        return self._config

    def initial_state(self) -> FluidState:
        """
        Return all-zero u, v, p, matching the baseline before its first step.

        The moving-lid boundary is first imposed during step(), not here.
        So if we want to remove the lid thing, we want to do that during step
        """
        shape = (self.config.n_points, self.config.n_points)
        return FluidState(np.zeros(shape), np.zeros(shape), np.zeros(shape))

    def _validate_state(self, state: FluidState) -> None:
        expected = (self.config.n_points, self.config.n_points)
        for field in (state.u, state.v, state.p):
            if field.shape != expected:
                raise ValueError(f"State components must have shape {expected}.")
            if not np.isfinite(field).all():
                raise ValueError("State components must contain finite values.")

    # All of these functions interior logic are the same as nav-stokes-eq/baseline solver
    # We may want to fiddle with the nums to match what Kai has, but the logic is the same
    def _central_difference_x(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        derivative = np.zeros_like(field)
        derivative[1:-1, 1:-1] = (
            field[1:-1, 2:] - field[1:-1, :-2]
        ) / (2 * self._element_length)
        return derivative

    def _central_difference_y(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        derivative = np.zeros_like(field)
        derivative[1:-1, 1:-1] = (
            field[2:, 1:-1] - field[:-2, 1:-1]
        ) / (2 * self._element_length)
        return derivative

    def _laplace(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        derivative = np.zeros_like(field)
        derivative[1:-1, 1:-1] = (
            field[1:-1, :-2] # left neighbours of int points
            + field[:-2, 1:-1] # this slice is the neighbors on row below int points
            - 4 * field[1:-1, 1:-1] # [1:-1, 1:-1] is the interior points
            + field[1:-1, 2:] # right neighbours to int points
            + field[2:, 1:-1] # neightbours one row abv int points
        ) / self._element_length**2
        return derivative

    def _apply_velocity_boundary_conditions(
        self, u: NDArray[np.float64], v: NDArray[np.float64]
    ) -> None:
        # Stationary walls + moving lid (including its corners).
        u[0, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = self.config.top_wall_velocity # This is the lid part, so if we may need to change this
        v[0, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        v[-1, :] = 0.0

    def _solve_pressure(
        self, pressure: NDArray[np.float64], rhs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # this is the same as what nav-stokes-eq has(although its not a function)
        # may want to change it to 
        for _ in range(self.config.pressure_iterations):
            pressure_next = np.zeros_like(pressure)
            pressure_next[1:-1, 1:-1] = 1 / 4 * (
                + pressure[1:-1, :-2]
                + pressure[:-2, 1:-1]
                + pressure[1:-1, 2:]
                + pressure[2:, 1:-1]
                - self._element_length**2 * rhs[1:-1, 1:-1]
            )
            # Homogeneous Neumann on right, bottom, left; zero pressure at lid.
            # Preserve the baseline order, which also determines corner values.
            pressure_next[:, -1] = pressure_next[:, -2]
            pressure_next[0, :] = pressure_next[1, :]
            pressure_next[:, 0] = pressure_next[:, 1]
            pressure_next[-1, :] = 0.0
            pressure = pressure_next
        return pressure

    def step(self, state: FluidState) -> FluidState:
        """Perform exactly one timestep and return a new, independent state.
        In other words, advance fluid by config.dt and return the new fluid state

        """
        self._validate_state(state)
        config = self.config
        u, v = state.u, state.v
        du_dx = self._central_difference_x(u)
        du_dy = self._central_difference_y(u)
        dv_dx = self._central_difference_x(v)
        dv_dy = self._central_difference_y(v)
        laplace_u = self._laplace(u)
        laplace_v = self._laplace(v)

        # Solve momentum without the pressure gradient for tentative velocity.
        u_tentative = u + config.dt * (
            -(u * du_dx + v * du_dy) + config.viscosity * laplace_u
        )
        v_tentative = v + config.dt * (
            -(u * dv_dx + v * dv_dy) + config.viscosity * laplace_v
        )
        self._apply_velocity_boundary_conditions(u_tentative, v_tentative)

        # Pressure Poisson RHS from tentative-velocity divergence.
        rhs = config.density / config.dt * (
            self._central_difference_x(u_tentative)
            + self._central_difference_y(v_tentative)
        )
        pressure_next = self._solve_pressure(state.p, rhs)
        dp_dx = self._central_difference_x(pressure_next)
        dp_dy = self._central_difference_y(pressure_next)

        # Correct velocity using the pressure gradient, then restore wall values.
        u_next = u_tentative - config.dt / config.density * dp_dx
        v_next = v_tentative - config.dt / config.density * dp_dy
        self._apply_velocity_boundary_conditions(u_next, v_next)
        return FluidState(u_next, v_next, pressure_next)

    def run(self, initial_state: FluidState, num_steps: int) -> list[FluidState]:
        """Return [S_0, ..., S_n], with an owned snapshot of the initial state.

        Each transition calls step(). 
        Long runs that only need a final state, call step() in a loop to avoid storing the full traj in our mem.
        """
        if isinstance(num_steps, bool) or not isinstance(num_steps, Integral) or num_steps < 0:
            raise ValueError("num_steps must be a nonnegative integer.")
        self._validate_state(initial_state)
        states = [FluidState(initial_state.u, initial_state.v, initial_state.p)]
        for _ in range(num_steps):
            states.append(self.step(states[-1]))
        return states
