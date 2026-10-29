"""API checks and numerical comparisons against the unchanged working baseline."""

from dataclasses import FrozenInstanceError, replace
import unittest
from unittest.mock import patch

import numpy as np

from src.simulation import FluidState, NavierStokesSolver, SolverConfig
from tests._baseline import run_baseline


class FluidStateTests(unittest.TestCase):
    def test_channel_order_round_trip_and_storage_independence(self) -> None:
        channels = np.arange(3 * 4 * 6, dtype=float).reshape(3, 4, 6)
        state = FluidState.from_array(channels)
        packed = state.to_array()
        self.assertEqual(packed.shape, (3, 4, 6))
        np.testing.assert_array_equal(packed, channels)
        for field, channel in zip((state.u, state.v, state.p), channels):
            np.testing.assert_array_equal(field, channel)
            self.assertFalse(np.shares_memory(field, channels))
            self.assertFalse(np.shares_memory(field, packed))

    def test_constructor_copies_and_promotes_integer_fields(self) -> None:
        source = np.zeros((3, 3), dtype=int)
        state = FluidState(source, source, source)
        state.u[1, 1] = 0.25
        self.assertEqual(source[1, 1], 0)
        self.assertEqual(state.v[1, 1], 0)
        self.assertEqual(state.p[1, 1], 0)
        self.assertEqual(state.u.dtype, np.float64)

    def test_invalid_states(self) -> None:
        for shape in ((3, 4), (2, 3, 4), (3, 0, 4), (3, 2, 3, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                FluidState.from_array(np.zeros(shape))
        with self.assertRaises(ValueError):
            FluidState(np.zeros((3, 3)), np.zeros((4, 4)), np.zeros((3, 3)))
        for value in (np.nan, np.inf, 1j, "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FluidState.from_array(np.full((3, 3, 3), value))


class SolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = NavierStokesSolver()

    def test_initial_state_is_zero_with_expected_shape(self) -> None:
        state = self.solver.initial_state()
        self.assertEqual(state.to_array().shape, (3, 41, 41))
        np.testing.assert_array_equal(state.to_array(), 0)

    def test_boundary_conditions_and_corner_precedence(self) -> None:
        solver = NavierStokesSolver(SolverConfig(top_wall_velocity=-0.7))
        state = solver.initial_state()
        for _ in range(10):
            state = solver.step(state)
        np.testing.assert_array_equal(state.u[0, :], 0)
        np.testing.assert_array_equal(state.u[:-1, 0], 0)
        np.testing.assert_array_equal(state.u[:-1, -1], 0)
        np.testing.assert_array_equal(state.u[-1, :], -0.7)
        for edge in (state.v[0, :], state.v[-1, :], state.v[:, 0], state.v[:, -1]):
            np.testing.assert_array_equal(edge, 0)
        np.testing.assert_array_equal(state.p[:, -1], state.p[:, -2])
        np.testing.assert_array_equal(state.p[0, :], state.p[1, :])
        np.testing.assert_array_equal(state.p[:, 0], state.p[:, 1])
        np.testing.assert_array_equal(state.p[-1, :], 0)

    def test_step_accepts_read_only_input_and_returns_independent_arrays(self) -> None:
        rng = np.random.default_rng(42)
        state = FluidState.from_array(rng.normal(scale=0.01, size=(3, 41, 41)))
        before = state.to_array()
        for field in (state.u, state.v, state.p):
            field.setflags(write=False)
        result = self.solver.step(state)
        np.testing.assert_array_equal(state.to_array(), before)
        self.assertIsNot(result, state)
        for field in (result.u, result.v, result.p):
            for original in (state.u, state.v, state.p):
                self.assertFalse(np.shares_memory(field, original))

    def test_run_calls_step_and_preserves_independent_snapshots(self) -> None:
        initial = self.solver.initial_state()
        with patch.object(self.solver, "step", wraps=self.solver.step) as step:
            trajectory = self.solver.run(initial, 3)
            self.assertEqual(step.call_count, 3)
        self.assertEqual(len(trajectory), 4)
        expected = initial
        for actual in trajectory:
            np.testing.assert_array_equal(actual.to_array(), expected.to_array())
            expected = self.solver.step(expected)
        initial.u[1, 1] = 9
        self.assertEqual(trajectory[0].u[1, 1], 0)
        snapshot = trajectory[1].to_array()
        trajectory[-1].u[:] = 9
        np.testing.assert_array_equal(trajectory[1].to_array(), snapshot)
        self.assertEqual(len(self.solver.run(self.solver.initial_state(), 0)), 1)

    def test_invalid_run_length_and_input_state(self) -> None:
        for value in (-1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.solver.run(self.solver.initial_state(), value)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.solver.step(FluidState.from_array(np.zeros((3, 5, 5))))
        state = self.solver.initial_state()
        state.p[1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.solver.step(state)

    def test_stationary_lid_preserves_rest(self) -> None:
        solver = NavierStokesSolver(SolverConfig(top_wall_velocity=0))
        state = solver.initial_state()
        for _ in range(5):
            state = solver.step(state)
        np.testing.assert_array_equal(state.to_array(), 0)

    def test_stability_bound_and_immutable_configuration(self) -> None:
        config = SolverConfig()
        element_length = config.domain_size / (config.n_points - 1)
        limit = config.stability_safety_factor * (0.5 * element_length**2 / config.viscosity)
        NavierStokesSolver(replace(config, dt=limit))
        with self.assertRaisesRegex(ValueError, "Unstable timestep"):
            NavierStokesSolver(replace(config, dt=np.nextafter(limit, np.inf)))
        with self.assertRaises(FrozenInstanceError):
            config.dt = 1.0
        with self.assertRaises(AttributeError):
            self.solver.config = replace(config, n_points=5)

    def test_invalid_configuration(self) -> None:
        invalid = {
            "n_points": (2, 4.5, True),
            "domain_size": (0, -1, np.inf),
            "dt": (0, -1, np.nan),
            "viscosity": (0, -0.1),
            "density": (0, -1),
            "pressure_iterations": (0, -1, 1.5, True),
            "top_wall_velocity": (np.nan, np.inf),
            "stability_safety_factor": (0, -1, 1.1),
        }
        for name, values in invalid.items():
            for value in values:
                with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                    SolverConfig(**{name: value})


class BaselineRegressionTests(unittest.TestCase):
    def assert_matches_baseline(self, config: SolverConfig, steps: int) -> None:
        solver = NavierStokesSolver(config)
        state = solver.initial_state()
        for _ in range(steps):
            state = solver.step(state)
            self.assertTrue(np.isfinite(state.to_array()).all())
        reference = run_baseline(config, steps)
        for name, actual, expected in zip(("u", "v", "p"), (state.u, state.v, state.p), reference):
            self.assertTrue(
                np.allclose(actual, expected, rtol=1e-12, atol=1e-12),
                f"{name}: maximum absolute error {np.max(np.abs(actual - expected))}",
            )

    def test_original_experiment_and_early_timesteps(self) -> None:
        for steps in (0, 1, 2, 10, 500):
            with self.subTest(steps=steps):
                self.assert_matches_baseline(SolverConfig(), steps)

    def test_alternative_configurations(self) -> None:
        configurations = (
            SolverConfig(n_points=9, domain_size=2, dt=0.002, viscosity=0.04,
                         density=1.7, pressure_iterations=13, top_wall_velocity=-0.3,
                         stability_safety_factor=0.4),
            SolverConfig(n_points=17, domain_size=0.8, dt=0.0005, viscosity=0.2,
                         density=2, pressure_iterations=7, top_wall_velocity=0.7),
            SolverConfig(n_points=3, pressure_iterations=1),
        )
        for config in configurations:
            with self.subTest(config=config):
                self.assert_matches_baseline(config, 30)


if __name__ == "__main__":
    unittest.main()
