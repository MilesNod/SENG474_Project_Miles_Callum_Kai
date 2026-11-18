"""Tune CNN/ridge models with grouped CV, then refit on development data only."""

# Run this AFTER generating data. This chooses configurations without opening the final test.
# Steps:
# 1. Read saved trajectory splits, or recover an interrupted run.
# 2. Fit/validate each candidate across the same folds.
# 3. Pick the lowest mean validation error and refit on all development runs.
# 4. Optionally repeat the selected CNN with smaller training subsets for a learning curve.

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import sys
import uuid

import numpy as np
import torch

from data import DEFAULT_DATA, TrajectoryStore, fit_normalization, write_json
from experiment import (DEFAULT_RUN, FIELDS, capture_sources, check_fit_metadata, environment, fit_ridge, resolve_device,
                        ridge_statistics, score, summarize_folds, train_cnn, write_csv)
from model import load_predictor


def restore_run_arguments(args, explicit_options):
    """Resume with the original research settings, allowing runtime/path overrides."""
    # Called by main when --resume is given. We must not silently use today's defaults
    # if the original run searched different learning rates or dataset sizes.
    if not args.resume:
        return args
    path = Path(args.output) / "run.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    runtime = {"data", "output", "device", "threads", "quiet", "verify_data", "resume", "dry_run"}
    # Runtime/path options can change; experimental settings must stay with the saved run.
    for key, value in info["arguments"].items():
        if key in runtime:
            continue
        if f"--{key.replace('_', '-')}" in explicit_options:
            actual = getattr(args, key)
            equivalent = sorted(actual) == sorted(value) if isinstance(value, list) else actual == value
            if not equivalent:
                raise ValueError(f"--resume must keep the saved {key}; use a new --output for a different experiment.")
        setattr(args, key, value)
    if "--data" not in explicit_options:
        args.data = info["data_path"]
    return args


def source_matches_saved(current, recorded, expected_hash):
    # Used by resume below: adding comments must not invalidate a completed experiment.
    # Check the recorded snapshot's hash first, then compare parsed Python statements.
    # Python's AST ignores comments/formatting but still contains model and solver logic.
    current_bytes = Path(current).read_bytes()
    if hashlib.sha256(current_bytes).hexdigest() == expected_hash:
        return True
    recorded = Path(recorded)
    if not recorded.is_file():
        return False
    recorded_bytes = recorded.read_bytes()
    if hashlib.sha256(recorded_bytes).hexdigest() != expected_hash:
        return False
    try:
        return ast.dump(ast.parse(current_bytes), include_attributes=False) == ast.dump(
            ast.parse(recorded_bytes), include_attributes=False)
    except (SyntaxError, UnicodeError):
        return False


@contextmanager
def run_lock(output):
    """An OS lock is automatically released on process exit, including interruptions."""
    # Held for a whole training run so two terminals cannot overwrite the same checkpoints.
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".training.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            # Windows and Linux expose different locking APIs, but both lock this one byte.
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError(f"Another training process is using {output}.") from error
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError(f"Another training process is using {output}.") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_experiment(args):
    # Public entry used by main/tests: protect existing output, then lock and run the work.
    output = Path(args.output).resolve()
    existing = output.is_dir() and any(p.name != ".training.lock" for p in output.iterdir())
    if existing and not args.resume:
        raise FileExistsError(f"Run directory is not empty: {output}. Use --resume to continue it or choose a new --output.")
    if args.dry_run:
        # A progress check only reads files, so it should not create a lock or new run folder.
        return _run_experiment(args)
    with run_lock(output):
        return _run_experiment(args)


def _run_experiment(args):
    # Actual experiment orchestration, called after run_experiment handles output protection.
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    store = TrajectoryStore(args.data)
    if args.verify_data:
        store.verify_content()
    max_curve_size = min(len(f["train"]) for f in store.splits["folds"])
    # Every requested learning-curve size must fit even the smallest training fold.
    if any(n < 1 or n > max_curve_size for n in args.learning_curve_sizes):
        raise ValueError(f"Learning-curve sizes must be in [1, {max_curve_size}] for this split.")
    output = Path(args.output).resolve()
    cnn_configs = [{"hidden_dim": width, "learning_rate": rate}
                   for width in sorted(set(args.hidden_dims)) for rate in sorted(set(args.learning_rates))]
    ridge_configs = [{"alpha": alpha} for alpha in sorted(set(args.ridge_alphas))]
    if args.resume:
        # Reuse the same dataset and numerical/model code; comments alone are harmless.
        info = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if info["dataset_id"] != store.metadata["dataset_id"] or info["splits"] != store.splits:
            raise ValueError("Cannot resume with a different dataset or split manifest.")
        for name in ("fluid-model/src/model.py", "src/simulation/solver.py", "src/simulation/state.py"):
            saved_hash = info.get("source_sha256", {}).get(name)
            if saved_hash and not source_matches_saved(Path(__file__).resolve().parents[2] / name,
                                                        output / "source" / name, saved_hash):
                raise ValueError(f"Cannot mix changed numerical/model code into the existing run: {name}")
    if args.dry_run:
        # Report which fits can be reused, resumed, or restarted, without launching training.
        counts = {"completed": 0, "resumable": 0, "older_partial": 0, "pending": 0}
        for fold in range(1, len(store.splits["folds"]) + 1):
            for config in cnn_configs:
                directory = output / "cv" / f"cnn_h{config['hidden_dim']}_lr{config['learning_rate']}" / f"fold_{fold}"
                category = ("completed" if (directory / "fit.json").is_file() else
                            "resumable" if (directory / "training_state.pt").is_file() else
                            "older_partial" if (directory / "latest.pt").is_file() else "pending")
                counts[category] += 1
        print(f"Dataset ready: {len(store.trajs)} trajectories, {store.metadata['transitions']} transitions.")
        print(f"CNN CV fits: {counts}. Older partial fits require a restart of that fit only.")
        print(f"Epoch budget: {args.epochs}; learning rates: {args.learning_rates}; learning-curve sizes: {args.learning_curve_sizes}.")
        print("Dry run: no training started and no artifacts changed.")
        return counts
    if args.resume and info["status"] == "complete":
        print(f"Training is already complete: {output}. Continue with evaluate.py --run {output}")
        return info
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        session = output / "resume_sessions" / uuid.uuid4().hex[:12]
        # Keep the original source snapshot intact; resumed code gets its own snapshot.
        info.setdefault("resume_sessions", []).append({"environment": environment(device),
                                                       "source_directory": str(session / "source"),
                                                       "source_sha256": capture_sources(session)})
    else:
        info = {"format_version": 2, "status": "cross_validation", "data_path": str(store.path),
            "dataset_id": store.metadata["dataset_id"], "arguments": vars(args),
            "environment": environment(device), "source_sha256": capture_sources(output), "splits": store.splits,
            "primary_metric": "Mean of per-field RMSE divided by training-input channel standard deviation.",
            "cnn_configs": cnn_configs, "ridge_configs": ridge_configs}
    write_json(output / "run.json", info)
    write_json(output / "dataset_metadata.json", store.metadata)
    write_json(output / "splits.json", store.splits)
    rows = json.loads((output / "cv_results.json").read_text(encoding="utf-8")) if args.resume and (output / "cv_results.json").is_file() else []
    # One cache key per fold/candidate means resuming cannot append duplicate result rows.
    cache = {(row["fold"], row["candidate"]): row for row in rows}

    def record(row):
        # Save after EACH fit, so an interruption does not lose a whole fold's result records.
        cache[(row["fold"], row["candidate"])] = row
        rows[:] = list(cache.values())
        write_json(output / "cv_results.json", rows)

    fits = len(store.splits["folds"]) * len(cnn_configs)
    print(f"Device: {device}. {fits} CNN CV fits; {len(ridge_configs)} ridge configurations per fold.", flush=True)
    for fold_index, fold in enumerate(store.splits["folds"], start=1):
        # Phase 1: use these same whole-run training/validation IDs for every candidate.
        train_ids, validation_ids = fold["train"], fold["validation"]
        seed = args.seed + fold_index
        _, std = fit_normalization(store, train_ids)
        if (fold_index, "persistence") not in cache:
            # No fitting is needed for persistence, only measuring its validation error.
            baseline = score(None, store, validation_ids, std, device, args.batch_size)
            record({"candidate": "persistence", "kind": "persistence", "config": {},
                    "fold": fold_index, "best_epoch": None, "training_seconds": 0.0, **baseline})
        for config in cnn_configs:
            # train_cnn either starts fresh, resumes its epoch state, or returns a completed fit.
            candidate = f"cnn_h{config['hidden_dim']}_lr{config['learning_rate']}"
            print(f"Fold {fold_index}: {candidate}", flush=True)
            result = train_cnn(store, train_ids, validation_ids, config, args.epochs, args.batch_size,
                               device, seed, output / "cv" / candidate / f"fold_{fold_index}",
                               args.patience, args.noise, args.quiet, resume=args.resume)
            record({"candidate": candidate, "kind": "cnn", "config": config, "fold": fold_index,
                         "best_epoch": result["best_epoch"], "training_seconds": result["training_seconds"],
                         **result["metrics"]})
        statistics, sampling_seconds = None, 0.0
        # Build ridge's sampled X^T X / X^T Y once, then try every penalty on that same sample.
        for config in ridge_configs:
            candidate = f"ridge_alpha{config['alpha']}"
            directory = output / "cv" / candidate / f"fold_{fold_index}"
            expected = {"dataset_id": store.metadata["dataset_id"], "train_ids": train_ids,
                        "validation_ids": validation_ids, "config": config, "seed": seed}
            if args.resume and (fold_index, candidate) in cache and (directory / "best.pt").is_file():
                _, metadata = load_predictor(directory / "best.pt", "cpu")
                check_fit_metadata(metadata, expected)
                print(f"Fold {fold_index}: reusing {candidate}", flush=True)
                continue
            if statistics is None:
                started = time.perf_counter()
                statistics = ridge_statistics(store, train_ids, seed, args.ridge_max_points)
                sampling_seconds = time.perf_counter() - started
            started = time.perf_counter()
            model = fit_ridge(statistics, config["alpha"])
            model.save(directory / "best.pt",
                       {"dataset_id": store.metadata["dataset_id"], "train_ids": train_ids,
                        "validation_ids": validation_ids, "config": config, "seed": seed,
                        "sampled_pixels": statistics[-1]})
            training_seconds = sampling_seconds + time.perf_counter() - started
            metrics = score(model.to(device), store, validation_ids, std, device, args.batch_size)
            record({"candidate": candidate, "kind": "ridge", "config": config, "fold": fold_index,
                         "best_epoch": None, "training_seconds": training_seconds, **metrics})
            print(f"Fold {fold_index}: {candidate} validation NRMSE={metrics['primary']:.6g}", flush=True)
    summary = summarize_folds(rows)
    # Phase 2: aggregate validation results. The final test has not been used for selection.
    write_json(output / "cv_summary.json", summary)
    write_csv(output / "cv_summary.csv", [{**r, "config": json.dumps(r["config"])} for r in summary])
    flat = []
    for row in rows:
        entry = {k: row[k] for k in ("candidate", "kind", "fold", "best_epoch", "training_seconds", "primary")}
        for metric in ("mse", "mae", "rmse", "normalized_rmse"):
            entry.update({f"{metric}_{field}": row[metric][channel] for channel, field in enumerate(FIELDS)})
        flat.append(entry)
    write_csv(output / "fold_metrics.csv", flat)
    selected = {kind: min((r for r in summary if r["kind"] == kind), key=lambda r: r["mean_primary"])
                for kind in ("cnn", "ridge")}
    info.update(status="development_refit", selected=selected)
    write_json(output / "run.json", info)

    # Freeze choices before touching the final test data. No test states are read here.
    best_cnn = selected["cnn"]
    best_epochs = [r["best_epoch"] for r in rows if r["candidate"] == best_cnn["candidate"]]
    # Phase 3: use the median chosen epoch as a fixed budget for the final refit.
    # There is no held-out validation fold left when training on all development runs.
    final_epochs = max(1, int(np.median(best_epochs)))
    development = store.splits["development"]
    print(f"Refitting selected CNN on development trajectories for {final_epochs} epochs.", flush=True)
    train_cnn(store, development, [], best_cnn["config"], final_epochs, args.batch_size, device,
              args.seed, output / "final" / "cnn", patience=0, noise=args.noise, quiet=args.quiet, resume=args.resume)
    statistics = ridge_statistics(store, development, args.seed, args.ridge_max_points)
    ridge = fit_ridge(statistics, selected["ridge"]["config"]["alpha"])
    ridge.save(output / "final" / "ridge" / "best.pt",
               {"dataset_id": store.metadata["dataset_id"], "train_ids": development,
                "validation_ids": [], "config": selected["ridge"]["config"],
                "seed": args.seed, "sampled_pixels": statistics[-1]})

    curve_rows = []
    # Phase 4: keep the validation sets fixed and vary only the number of training runs.
    for size in sorted(set(args.learning_curve_sizes)):
        for fold_index, fold in enumerate(store.splits["folds"], start=1):
            # Nested subsets: a larger point contains all training IDs of smaller points.
            order = np.random.default_rng(args.seed + fold_index).permutation(fold["train"]).tolist()
            print(f"Learning curve: {size} training trajectories, fold {fold_index}", flush=True)
            result = train_cnn(store, order[:size], fold["validation"], best_cnn["config"],
                               args.epochs, args.batch_size, device, args.seed + fold_index,
                               output / "learning_curve" / f"size_{size}" / f"fold_{fold_index}",
                               args.patience, args.noise, args.quiet,
                               metric_std=fit_normalization(store, fold["train"])[1], resume=args.resume)
            # Fixed full-fold metric scales make scores comparable across subset sizes.
            curve_rows.append({"training_trajectories": size, "fold": fold_index,
                               "primary": result["metrics"]["primary"], "best_epoch": result["best_epoch"]})
            write_csv(output / "learning_curve_folds.csv", curve_rows)
    if curve_rows:
        curve_summary = []
        for size in sorted(set(args.learning_curve_sizes)):
            values = [r["primary"] for r in curve_rows if r["training_trajectories"] == size]
            curve_summary.append({"training_trajectories": size, "mean_primary": float(np.mean(values)),
                                  "std_primary": float(np.std(values, ddof=1))})
        write_csv(output / "learning_curve.csv", curve_summary)
    from plots import plot_cross_validation
    plot_cross_validation(output)
    info.update(status="complete", final_epochs=final_epochs)
    write_json(output / "run.json", info)
    print(f"Saved CV results and development refits to {output}\nFinal test set has not been evaluated.", flush=True)
    return info


def main():
    # Command-line entry. Read options, restore saved settings if requested, then start the run.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_RUN))
    parser.add_argument("--resume", action="store_true", help="Continue --output using its original settings and completed fits.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and show saved progress without starting training.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience; 0 disables it.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.001])
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[0.0001, 0.01, 1.0])
    parser.add_argument("--ridge-max-points", type=int, default=100000)
    parser.add_argument("--learning-curve-sizes", type=int, nargs="*", default=[])
    parser.add_argument("--noise", type=float, default=0.0, help="Training-only noise as a fraction of channel std.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--verify-data", action="store_true", help="Also check every stored trajectory byte against SHA256.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        args = restore_run_arguments(args, {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")})
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if (min(args.epochs, args.batch_size, args.ridge_max_points, args.threads, *args.hidden_dims) < 1
            or args.patience < 0 or args.noise < 0 or not np.isfinite(args.noise)
            or any(not np.isfinite(v) or v <= 0 for v in args.learning_rates + args.ridge_alphas)):
        parser.error("Epochs, sizes, learning rates and ridge penalties must be positive; patience/noise nonnegative.")
    try:
        run_experiment(args)
    except (ValueError, FileExistsError, FileNotFoundError, FloatingPointError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
