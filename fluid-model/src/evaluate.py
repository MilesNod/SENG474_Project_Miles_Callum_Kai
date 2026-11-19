"""Evaluate frozen development refits on the reserved final-test trajectories."""

# Run this AFTER train.py has finished selecting/refitting its models.
# Steps: check saved run -> score true-state inputs -> roll predictions forward -> save results.
# One-step: we give the model a true solver state for every prediction.
# Rollout: we give it a true start once, then keep feeding its own output back in.

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from data import TrajectoryStore, write_json
from experiment import DEFAULT_RUN, ErrorTotals, FIELDS, environment, resolve_device, score, write_csv
from model import load_predictor
from plots import plot_rollouts, save_comparison
from src.simulation import FluidState, NavierStokesSolver, SolverConfig


def predict(model, states, parameters, device, batch_size):
    # Used by rollouts and timing. Process the supplied states in batches to limit GPU memory.
    # model=None is persistence, so its prediction is a copy of the unchanged current state.
    if model is None:
        return states.copy()
    batches = []
    with torch.inference_mode():
        # No weight updates here; tensors only need a forward pass, not gradient tracking.
        for start in range(0, len(states), batch_size):
            state = torch.from_numpy(states[start:start + batch_size]).to(device)
            physical = torch.from_numpy(parameters[start:start + batch_size]).to(device)
            batches.append(model(state, physical).cpu().numpy())
    return np.concatenate(batches)


def rollouts(models, store, ids, std, start, horizons, device, batch_size, output):
    # Called after one-step scoring to test whether each model can advance a simulation itself.
    # start: solver timestep to begin from; horizons: how many predicted steps to score.
    initial = np.stack([store.trajs[run, start] for run in ids])
    parameters = store.parameters[ids]
    rows, per_trajectory = [], []
    comparison_saved = False
    for name, model in models.items():
        # Every method starts from the same true states and physical parameters.
        prediction = initial.copy()
        unstable_at = None
        for step in range(1, max(horizons) + 1):
            if unstable_at is None:
                # After step 1 this input is our previous prediction, not a fresh solver frame.
                with np.errstate(over="ignore", invalid="ignore"):
                    prediction = predict(model, prediction, parameters, device, batch_size)
                if not np.isfinite(prediction).all():
                    # Mark the entire method/horizon as failed if any trajectory becomes invalid.
                    # Dropping only the failed trajectories would make its average look too good.
                    unstable_at = step
            if step not in horizons:
                # Still predict every intermediate step, but only measure the requested horizons.
                continue
            row = {"model": name, "horizon": step, "status": "nonfinite" if unstable_at else "finite",
                   "first_nonfinite_step": unstable_at, "primary": None}
            for metric in ("mae", "rmse", "normalized_rmse"):
                row.update({f"{metric}_{field}": None for field in FIELDS})
            if unstable_at is None:
                # Compare step h against the stored solver state at start + h.
                expected = np.stack([store.trajs[run, start + step] for run in ids])
                totals = ErrorTotals(std)
                totals.update(prediction, expected)
                metrics = totals.finish()
                row["primary"] = metrics["primary"]
                for metric in ("mae", "rmse", "normalized_rmse"):
                    row.update({f"{metric}_{field}": metrics[metric][i] for i, field in enumerate(FIELDS)})
                for i, run in enumerate(ids):
                    item = ErrorTotals(std)
                    item.update(prediction[i:i + 1], expected[i:i + 1])
                    per_trajectory.append({"model": name, "horizon": step, "id": run, **item.finish()})
                if name == "cnn" and step == max(horizons):
                    # Only make the final-horizon heatmap when that prediction is actually finite.
                    # If a rollout failed earlier, there is no valid comparison.png for that horizon.
                    save_comparison(expected[0], prediction[0], output / "comparison.png", step)
                    comparison_saved = True
            rows.append(row)
    write_csv(output / "rollout.csv", rows)
    write_json(output / "rollout_per_trajectory.json", per_trajectory)
    plot_rollouts(rows, output / "rollout.png")
    return rows, comparison_saved


def benchmark(models, store, run, start, device, repeats):
    """Single-state wall time including host/device transfers, with warmup."""
    # Optional evaluation step. Repeat one fixed input so we measure prediction/solver latency,
    # not model training, data generation, or an entire evolving trajectory.
    state = store.trajs[run, start].copy()[None]
    parameters = store.parameters[run:run + 1]
    solver = NavierStokesSolver(SolverConfig(**store.metadata["trajectories"][run]["config"]))
    reference = FluidState.from_array(state[0])
    functions = {name: (lambda model=model: predict(model, state, parameters, device, 1))
                 for name, model in models.items()}
    functions["solver"] = lambda: solver.step(reference)
    results = []
    for name, function in functions.items():
        # Warm up first: the first call can include GPU initialization or kernel setup.
        for _ in range(5):
            function()
        durations = []
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            # Synchronize GPU work before/after timing so we do not just time queuing it.
            begin = time.perf_counter()
            function()
            if device == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - begin)
        results.append({"method": name, "median_seconds_per_step": float(np.median(durations)),
                        "repeats": repeats, "batch_size": 1})
    return results


def evaluate_run(args):
    # Main evaluation work, called by main. Only the already-selected development refits
    # are allowed here; tuning against these final-test results would leak test information.
    run_path = Path(args.run).resolve()
    info = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    if info["status"] != "complete":
        raise ValueError("Complete cross-validation and development refitting before final evaluation.")
    store = TrajectoryStore(args.data or info["data_path"])
    if store.metadata["dataset_id"] != info["dataset_id"] or store.splits != info["splits"]:
        # Same file location is not enough: its dataset identity and split IDs must agree too.
        raise ValueError("Dataset/splits differ from the frozen training run.")
    if args.verify_data:
        store.verify_content()
    if args.rollout_start < 0 or args.rollout_start + max(args.horizons) > store.num_steps:
        raise ValueError(f"Rollouts must fit within the {store.num_steps} stored transitions.")
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    output = Path(args.output).resolve() if args.output else run_path / "test"
    if output.exists() and any(output.iterdir()) and not args.repeat:
        raise FileExistsError(f"Evaluation already exists: {output}. --repeat explicitly reruns the final test.")
    models = {"persistence": None}
    # Load the final CNN/ridge, not the individual fold models or whichever weights ran last.
    for name in ("cnn", "ridge"):
        model, metadata = load_predictor(run_path / "final" / name / "best.pt", device)
        if (metadata.get("dataset_id") != info["dataset_id"]
                or sorted(metadata["train_ids"]) != sorted(store.splits["development"])
                or metadata["validation_ids"]
                or metadata["config"] != info["selected"][name]["config"]):
            raise ValueError(f"{name} checkpoint does not match the frozen development refit.")
        models[name] = model
    std = models["cnn"].field_std.detach().cpu().numpy().reshape(3)
    # Use one shared development-set scale for all final-test methods, including persistence.
    if not np.allclose(std, models["ridge"].field_std.detach().cpu().numpy().reshape(3)):
        raise ValueError("Models must share development-set normalization for comparison.")
    output.mkdir(parents=True, exist_ok=True)
    ids = store.splits["test"]
    scores = {name: score(model, store, ids, std, device, args.batch_size) for name, model in models.items()}
    # These one-step scores average all true-state -> next-state pairs in the test trajectories.
    rows = []
    for name, metrics in scores.items():
        baseline = scores["persistence"]["primary"]
        # Relative reduction in normalized error. A negative improvement means worse than persistence.
        row = {"model": name, "primary": metrics["primary"],
               "improvement_percent": 100 * (1 - metrics["primary"] / baseline) if baseline > 0 else None}
        for metric in ("mse", "mae", "rmse", "normalized_rmse"):
            row.update({f"{metric}_{field}": metrics[metric][i] for i, field in enumerate(FIELDS)})
        rows.append(row)
        print(f"{name:<12} final-test NRMSE={metrics['primary']:.6g}; per-field RMSE={metrics['rmse']}", flush=True)
    write_csv(output / "one_step.csv", rows)
    rollout_rows, comparison_saved = rollouts(models, store, ids, std, args.rollout_start,
                                             sorted(set(args.horizons)), device, args.batch_size, output)
    # Timing is optional; using 0 repeats skips it without affecting prediction metrics.
    timings = benchmark(models, store, ids[0], args.rollout_start, device, args.benchmark_repeats) if args.benchmark_repeats else []
    if timings:
        write_csv(output / "timing.csv", timings)
    result = {"dataset_id": info["dataset_id"], "test_ids": ids, "environment": environment(device),
              "arguments": vars(args), "one_step": scores, "rollouts": rollout_rows,
              "normalization_std": std.tolist(), "timings": timings,
              "timing_protocol": "Batch 1, five warmups; wall time includes transfers; NumPy solver float64, models float32.",
              "comparison_saved": comparison_saved}
    write_json(output / "results.json", result)
    if args.save_plot and comparison_saved:
        import shutil
        target = Path(args.save_plot)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != (output / "comparison.png").resolve():
            shutil.copyfile(output / "comparison.png", target)
    print(f"Saved final-test tables, per-trajectory results and plots to {output}", flush=True)
    return result


def main():
    # Command-line entry: validate sizes/horizons before opening the final test and writing outputs.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--data", help="Override the recorded dataset path after moving the artifacts.")
    parser.add_argument("--output", help="Default: <run>/test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rollout-start", type=int, default=0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 25, 50, 100])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    parser.add_argument("--verify-data", action="store_true")
    parser.add_argument("--repeat", action="store_true", help="Explicitly permit repeating final-test evaluation.")
    parser.add_argument("--save-plot", help="Also copy the CNN comparison plot to this path.")
    args = parser.parse_args()
    if min(args.batch_size, args.threads, *args.horizons) < 1 or args.benchmark_repeats < 0:
        parser.error("Sizes/horizons must be positive; benchmark repeats must be nonnegative.")
    try:
        evaluate_run(args)
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
