"""Compare a saved model with the numerical solver on new initial conditions."""

import argparse
from pathlib import Path

import numpy as np
import torch

from data import make_random_initial_state
from model import FluidModel
from src.simulation import NavierStokesSolver, SolverConfig


def field_mse(predicted, expected):
    """Avg squared errors over samples and grid points, separately for u/v/p."""
    error = predicted.astype(np.float64) - expected.astype(np.float64)
    return np.square(error).mean(axis=(0, 2, 3))


def save_comparison(expected, predicted, path, horizon):
    """Plot the final rollout for the first test trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained")
    for channel, name in enumerate(("u", "v", "p")):
        truth = expected[channel].astype(np.float64)
        estimate = predicted[channel].astype(np.float64)
        minimum = min(float(truth.min()), float(estimate.min()))
        maximum = max(float(truth.max()), float(estimate.max()))
        for row, (label, field) in enumerate((("Solver", truth), ("Model", estimate))):
            plotted = axes[row, channel].imshow(
                field, origin="lower", cmap="coolwarm", vmin=minimum, vmax=maximum
            )
            axes[row, channel].set_title(f"{label}: {name}")
            figure.colorbar(plotted, ax=axes[row, channel], shrink=0.8)
        plotted = axes[2, channel].imshow(
            np.abs(estimate - truth), origin="lower", cmap="magma", vmin=0
        )
        axes[2, channel].set_title(f"Absolute error: {name}")
        figure.colorbar(plotted, ax=axes[2, channel], shrink=0.8)
    figure.suptitle(f"First unseen trajectory after {horizon} model steps")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved comparison: {path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(__file__).resolve().parents[1] / "weights" / "newest_model.pt",
    )
    parser.add_argument("--seed", type=int, default=4, help="Training currently uses seed 3.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rollout-start", type=int, default=25)
    parser.add_argument("--rollout-steps", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--save-plot", type=Path)
    args = parser.parse_args()
    for name in ("runs", "steps", "batch_size", "rollout_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.rollout_start < 0 or args.rollout_start + args.rollout_steps > args.steps:
        parser.error("rollout-start must be nonnegative and rollout must fit within --steps")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    config = SolverConfig()
    solver = NavierStokesSolver(config)
    rng = np.random.default_rng(args.seed)
    print(f"Generating {args.runs} test trajectories; seed={args.seed}, noise=0", flush=True)
    print("Using the same solver parms, random-state amplitude (0.1) as training.")
    if args.seed == 3:
        print("Seed 3 matches current training data; this is not a held-out evaluation.")
    trajectories = []
    for _ in range(args.runs):
        initial = make_random_initial_state(solver, rng, amplitude=0.1)
        states = solver.run(initial, args.steps)
        trajectories.append(np.stack([state.to_array() for state in states]))
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if not np.isfinite(trajectories).all():
        raise ValueError("The generated reference trajectories contain nonfinite values.")

    shape = trajectories.shape[2:]
    inputs = trajectories[:, :-1].reshape(-1, *shape)
    targets = trajectories[:, 1:].reshape(-1, *shape)
    print(f"Loading {args.checkpoint.name}; device={device}", flush=True)
    model = FluidModel(torch.Size(shape))
    model.load(str(args.checkpoint))
    model.to(device)
    model.eval()

    model_error = np.zeros(3)
    baseline_error = np.zeros(3)
    with torch.inference_mode():
        for start in range(0, len(inputs), args.batch_size):
            batch = inputs[start:start + args.batch_size]
            expected = targets[start:start + args.batch_size]
            predicted = model(torch.from_numpy(batch).to(device)).cpu().numpy().reshape(batch.shape)
            if not np.isfinite(predicted).all():
                raise ValueError("One-step predictions contain nonfinite values.")
            model_error += field_mse(predicted, expected) * len(batch)
            baseline_error += field_mse(batch, expected) * len(batch)
        model_error /= len(inputs)
        baseline_error /= len(inputs)

        print(f"\nOne-step MSE across {len(inputs)} test transitions (lower is better)")
        print(f"{'Field':<10}{'Model':>15}{'No-change baseline':>22}")
        for name, learned, baseline in zip(("u", "v", "p", "Overall"),
                                          [*model_error, model_error.mean()],
                                          [*baseline_error, baseline_error.mean()]):
            print(f"{name:<10}{learned:>15.6g}{baseline:>22.6g}")
        if baseline_error.mean() > 0:
            improvement = 100 * (1 - model_error.mean() / baseline_error.mean())
            print(f"Overall improvement over no change: {improvement:.2f}% (negative means worse)")

        initial = trajectories[:, args.rollout_start]
        prediction = torch.from_numpy(initial).to(device)
        predicted = initial
        completed_steps = 0
        horizons = {1, 10, 25, args.rollout_steps}
        print(f"\nRollout from solver timestep {args.rollout_start}; feeding predictions back into model")
        print(f"{'Steps':<10}{'Model MSE':>15}{'No-change MSE':>22}")
        for step in range(1, args.rollout_steps + 1):
            prediction = model(prediction)
            next_prediction = prediction.cpu().numpy().reshape(initial.shape)
            if not np.isfinite(next_prediction).all():
                print(f"UNSTABLE: nonfinite predictions after {step} model steps.")
                print(f"Stopped at the last finite prediction (step {completed_steps}).")
                break
            predicted = next_prediction
            completed_steps = step
            if step in horizons:
                expected = trajectories[:, args.rollout_start + step]
                learned = field_mse(predicted, expected).mean()
                baseline = field_mse(initial, expected).mean()
                print(f"{step:<10}{learned:>15.6g}{baseline:>22.6g}")

    if args.save_plot is not None:
        expected = trajectories[0, args.rollout_start + completed_steps]
        save_comparison(expected, predicted[0], args.save_plot, completed_steps)


if __name__ == "__main__":
    main()
