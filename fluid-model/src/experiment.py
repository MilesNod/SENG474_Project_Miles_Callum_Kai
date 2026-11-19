"""Shared fitting and scoring code for cross-validation and final evaluation."""

# train.py decides WHICH fold/configuration runs; this file does the actual fitting.
# evaluate.py also uses the same scoring code so the metrics have the same meaning.
# Main steps: fit training stats -> train CNN / solve ridge -> score -> save progress.
# A fit is one model/configuration trained on one particular set of run IDs.

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import FluidDataset, ROOT, atomic_write, fit_normalization, write_json
from model import FluidModel, LocalRidge, load_predictor

FIELDS = ("u", "v", "p")
DEFAULT_RUN = ROOT / "artifacts" / "runs" / "cavity_cv"


def capture_sources(output):
    """Preserve the code actually used, including changes not yet committed to Git."""
    # Called when a run or resume session starts, so we can recover the code behind its results.
    paths = [*ROOT.glob("fluid-model/src/*.py"), *ROOT.glob("src/simulation/*.py"),
             *ROOT.glob("requirements*.txt"), *ROOT.glob("reports/*.tex"),
             *ROOT.glob("scripts/run_experiment.py")]
    hashes = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        target = Path(output) / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def seed_everything(seed):
    # Called before making each CNN. Seeds control weights, augmentation, and shuffling.
    # cuDNN chooses GPU convolution algorithms; deterministic settings reduce run-to-run changes.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested="auto"):
    # Used by train/evaluate before moving tensors. "auto" prefers CUDA if it is available.
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch build.")
    return requested


def environment(device):
    # Saved with results so a timing/accuracy number has its software and hardware context.
    def git(*args):
        # Read the commit/status if this is a Git checkout; missing Git should not stop a fit.
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "matplotlib", "tqdm")},
            "device": device, "cpu": platform.processor(), "torch_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "cuda_version": torch.version.cuda, "git_commit": git("rev-parse", "HEAD"),
            "git_status": git("status", "--short"),
            "determinism": "Seeded RNGs and deterministic cuDNN; cross-platform bitwise equality is not guaranteed."}


def write_csv(path, rows):
    # Save histories and result tables in a form we can open in a spreadsheet.
    if not rows:
        return
    def write(pending):
        # atomic_write supplies the temporary path; the first row gives the column names.
        with pending.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_write(path, write)


def save_torch(path, value):
    # Used for model/training checkpoints; only replace the old checkpoint after saving succeeds.
    atomic_write(path, lambda pending: torch.save(value, pending))


def check_fit_metadata(metadata, expected):
    # Before resume reuses weights, check they belong to these exact runs and settings.
    for key in ("dataset_id", "train_ids", "validation_ids", "config", "seed", "noise"):
        # Ridge checkpoints do not carry a noise setting; CNN checkpoints always do.
        if key in expected and metadata.get(key) != expected[key]:
            raise ValueError(f"Cannot reuse checkpoint: {key} differs from the saved experiment.")
    if "fit_settings" in metadata and metadata["fit_settings"] != expected.get("fit_settings"):
        raise ValueError("Cannot resume with different epochs, batch size, patience, or metric scales.")


def capture_rng(generator):
    # Called after each completed epoch. Saving just the seed would replay randomness
    # from the beginning; these states tell each generator where it had reached.
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state(), "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}


def restore_rng(state, generator):
    # Called after loading a partial fit, so its next batch/noise matches an uninterrupted run.
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; use the original hardware to restore GPU RNG state.")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


def publish_training_state(output, state):
    # Turn the complete recovery checkpoint into the normal best/latest/history files.
    # Also repairs these smaller files if interruption happened after the state was committed.
    for filename, weights, metadata in (("best.pt", state["best_model"], state["best_metadata"]),
                                        ("latest.pt", state["model"], state["metadata"])):
        save_torch(output / filename, {"format_version": 2, "architecture": state["architecture"],
                                      "state_dict": weights, "metadata": metadata})
    write_csv(output / "history.csv", state["history"])


class ErrorTotals:
    # Add errors over batches first, then divide once; small last batches must not get extra weight.
    def __init__(self, std):
        # std is from training data, and lets us compare the three fields on a common scale.
        self.std = np.asarray(std, dtype=np.float64).reshape(3)
        self.squared, self.absolute, self.count = np.zeros(3), np.zeros(3), 0

    def update(self, predicted, expected):
        # Called once per scored batch; arrays are (batch, 3, H, W).
        if not np.isfinite(predicted).all():
            raise FloatingPointError("Predictions contain nonfinite values.")
        error = predicted.astype(np.float64) - expected.astype(np.float64)
        # Accumulate in float64 even though stored states/model outputs are float32.
        if not np.isfinite(error).all():
            raise FloatingPointError("Reference or prediction errors contain nonfinite values.")
        self.squared += np.square(error).sum(axis=(0, 2, 3))
        # Sum batch and grid axes, keeping channel axis 1 separate for u/v/p errors.
        self.absolute += np.abs(error).sum(axis=(0, 2, 3))
        self.count += error.shape[0] * error.shape[2] * error.shape[3]

    def finish(self):
        # Called after all batches: MSE = squared error / points; RMSE = sqrt(MSE).
        if not self.count:
            raise ValueError("Cannot score an empty dataset.")
        mse = self.squared / self.count
        rmse = np.sqrt(mse)
        normalized = rmse / self.std
        # The primary score is the mean of the 3 normalized RMSEs, not raw mixed-unit errors.
        return {"mse": mse.tolist(), "rmse": rmse.tolist(),
                "mae": (self.absolute / self.count).tolist(),
                "normalized_rmse": normalized.tolist(), "primary": float(normalized.mean())}


def score(model, store, ids, std, device, batch_size):
    # Used for fold validation and final one-step tests. Every input is a TRUE solver state.
    # model=None means persistence: predict the current input without changing it.
    dataset = FluidDataset(store, ids)
    totals = ErrorTotals(std)
    per_run = {run: ErrorTotals(std) for run in ids}
    if model is not None:
        model.eval()
    offset = 0
    with torch.inference_mode():
        # We only measure errors here, so gradients and optimizer updates are unnecessary.
        for state, target, parameters in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            predicted = (state if model is None else model(state.to(device), parameters.to(device))).cpu().numpy()
            expected = target.numpy()
            totals.update(predicted, expected)
            groups = np.arange(offset, offset + len(state)) // store.num_steps
            # A batch can cross run boundaries; map each sample back to its trajectory
            # so we save per-run scores as well as one combined score.
            for local_run in np.unique(groups):
                mask = groups == local_run
                per_run[ids[int(local_run)]].update(predicted[mask], expected[mask])
            offset += len(state)
    result = totals.finish()
    result["per_trajectory"] = [{"id": int(run), **item.finish()} for run, item in per_run.items()]
    return result


def train_cnn(store, train_ids, validation_ids, config, epochs, batch_size, device, seed,
              output, patience=10, noise=0.0, quiet=False, metric_std=None, resume=False):
    """Validation controls checkpoint selection; final refits have a fixed epoch budget."""
    # Called for every CNN CV fit, learning-curve point, and the final development refit.
    # validation_ids=[] means a fixed-length final refit, with no test-based stopping.
    # Steps: check/recover saved fit -> prepare model/loader -> train -> validate -> save.
    seed_everything(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mean, std = fit_normalization(store, train_ids)
    # Learning-curve models normalize their own subset, but share full-fold metric scales.
    selection_std = std if metric_std is None else np.asarray(metric_std)
    expected = {"dataset_id": store.metadata["dataset_id"], "train_ids": list(train_ids),
                "validation_ids": list(validation_ids), "config": config, "seed": seed, "noise": noise,
                "fit_settings": {"epochs": epochs, "batch_size": batch_size, "patience": patience,
                                 "metric_std": selection_std.tolist()}}
    state_path = output / "training_state.pt"
    saved = torch.load(state_path, weights_only=True, map_location="cpu") if resume and state_path.is_file() else None
    if saved:
        if saved.get("format_version") != 1:
            raise ValueError("Unsupported training-state format.")
        check_fit_metadata(saved["metadata"], expected)
    if resume and (output / "fit.json").is_file():
        # fit.json marks a finished fit. Reuse its result instead of spending more GPU time.
        if saved and not (output / "best.pt").is_file():
            publish_training_state(output, saved)
        cached, metadata = load_predictor(output / "best.pt", "cpu")
        check_fit_metadata(metadata, expected)
        if (not np.array_equal(cached.field_mean.numpy().reshape(3), mean)
                or not np.array_equal(cached.field_std.numpy().reshape(3), std)):
            raise ValueError("Saved checkpoint normalization differs from the current training data.")
        print(f"  Reusing completed fit: {output}", flush=True)
        return json.loads((output / "fit.json").read_text(encoding="utf-8"))
    if saved:
        publish_training_state(output, saved)
    if not resume and any((output / name).exists() for name in ("best.pt", "latest.pt", "history.csv", "training_state.pt")):
        raise FileExistsError(f"Fit already exists: {output}; use --resume.")
    if resume and saved is None and any((output / name).exists() for name in ("best.pt", "latest.pt", "history.csv")):
        # Older weights alone cannot restore Adam's history. Keep a copy before restarting that fit.
        checkpoint_path = output / "latest.pt" if (output / "latest.pt").is_file() else output / "best.pt"
        _, metadata = load_predictor(checkpoint_path, "cpu")
        check_fit_metadata(metadata, expected)
        backup = output / "legacy_backups" / uuid.uuid4().hex[:12]
        backup.mkdir(parents=True)
        for name in ("best.pt", "latest.pt", "history.csv"):
            if (output / name).is_file():
                shutil.copyfile(output / name, backup / name)
        print(f"  Older partial fit lacks optimizer/RNG state; backed up to {backup}. Restarting only this fit.", flush=True)
    model = FluidModel(hidden_dim=config["hidden_dim"], mean=mean, std=std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # A separate generator preserves the exact shuffled sample order across resumes.
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(FluidDataset(store, train_ids, noise, std), batch_size=batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    history, best, best_epoch, stale = [], float("inf"), 0, 0
    first_epoch, previous_seconds, best_model, best_metadata = 1, 0.0, None, None
    if saved:
        # Restore optimizer momentum, best validation score, patience counter, and RNG position.
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_epoch, stale = saved["history"], saved["best"], saved["best_epoch"], saved["stale"]
        first_epoch, previous_seconds = saved["epoch"] + 1, saved["elapsed_seconds"]
        best_model, best_metadata = saved["best_model"], saved["best_metadata"]
        restore_rng(saved["rng"], generator)
        print(f"  Resuming after epoch {saved['epoch']}/{epochs}: {output}", flush=True)
    started = time.perf_counter()
    epoch_range = () if validation_ids and patience and stale >= patience else range(first_epoch, epochs + 1)
    # If the saved epoch had already met early stopping, only the final bookkeeping remains.
    for epoch in epoch_range:
        model.train()
        total, count = 0.0, 0
        for state, target, parameters in loader:
            # Move the input, clean next-state target, and physical parameters to the same device.
            state, target, parameters = state.to(device), target.to(device), parameters.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Clear the previous batch's gradients; PyTorch otherwise adds new gradients to them.
            predicted = model(state, parameters)
            loss = torch.square((predicted - target) / model.field_std).mean()
            # Scale each field's error before squaring, so pressure does not dominate by units alone.
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in {output}, epoch {epoch}.")
            loss.backward()
            # backward calculates weight gradients; step applies Adam's weight updates.
            optimizer.step()
            total += float(loss.detach()) * len(state)
            count += len(state)
        validation = score(model, store, validation_ids, selection_std, device, batch_size) if validation_ids else None
        # Validation, not training loss, decides which weights count as best.
        value = validation["primary"] if validation else None
        history.append({"epoch": epoch, "training_normalized_mse": total / count,
                        "validation_primary": value})
        metadata = {**expected, "epoch": epoch, "validation_primary": value,
                    "normalization_source": "training input states only",
                    "validation_metric_std": selection_std.tolist()}
        if validation is None or value < best:
            # Clone the tensors: keeping references would let later updates change the saved best.
            best, best_epoch, stale = value if validation else 0.0, epoch, 0
            best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metadata = metadata
        else:
            stale += 1
            # stale counts epochs without validation improvement; patience sets the limit.
        state = {"format_version": 1, "architecture": model.architecture,
                 "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                 "best_model": best_model, "metadata": metadata, "best_metadata": best_metadata,
                 "optimizer": optimizer.state_dict(), "rng": capture_rng(generator), "history": history,
                 "best": best, "best_epoch": best_epoch, "stale": stale, "epoch": epoch,
                 "elapsed_seconds": previous_seconds + time.perf_counter() - started}
        # Commit the complete state first; best/latest/history can be reconstructed if interrupted.
        save_torch(state_path, state)
        publish_training_state(output, state)
        if not quiet:
            suffix = f" validation NRMSE={value:.6g}" if value is not None else " (development refit)"
            print(f"  epoch {epoch}/{epochs}: train normalized MSE={total / count:.6g}{suffix}", flush=True)
        if validation is not None and patience and stale >= patience:
            break
    elapsed = previous_seconds + time.perf_counter() - started
    model, _ = load_predictor(output / "best.pt", device)
    # Score the chosen checkpoint, which may be earlier than the last epoch we ran.
    result = {"best_epoch": best_epoch, "epochs_completed": len(history), "training_seconds": elapsed,
              "metrics": score(model, store, validation_ids, selection_std, device, batch_size) if validation_ids else None}
    write_json(output / "fit.json", result)
    return result


def ridge_statistics(store, ids, seed, max_points=100000, batch_size=8):
    """Sample training pixels uniformly, accumulating small sufficient-statistic matrices."""
    # Called once per fold, then reused for every ridge penalty so the comparison is fair.
    # X: local stencil inputs plus intercept; Y: normalized next-state changes.
    mean, std = fit_normalization(store, ids)
    model = LocalRidge(mean, std).eval()
    dataset = FluidDataset(store, ids)
    pixels = store.trajs.shape[-1] * store.trajs.shape[-2]
    # Select up to 128 pixels per sampled transition, spread over the entire training set.
    per_transition = min(128, pixels, max_points)
    transitions = min(len(dataset), max(1, int(np.ceil(max_points / per_transition))))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(dataset), size=transitions, replace=False)
    xtx, xty, count = np.zeros((176, 176)), np.zeros((176, 3)), 0
    # Keep X^T X and X^T Y instead of storing the full large regression matrix.
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            samples = [dataset[int(i)] for i in selected[start:start + batch_size]]
            state, target, parameters = [torch.stack(items) for items in zip(*samples)]
            design = model.design(state, parameters).reshape(len(state), pixels, -1)
            delta = ((target - state) / model.field_std).permute(0, 2, 3, 1).reshape(len(state), pixels, 3)
            # Move channels to the last axis so each grid point has one [du, dv, dp] target row.
            for run in range(len(state)):
                n = min(per_transition, max_points - count)
                if n <= 0:
                    break
                indices = rng.choice(pixels, size=n, replace=False)
                x = np.column_stack((design[run, indices].numpy().astype(np.float64), np.ones(n)))
                # The column of ones fits an intercept, i.e. an offset independent of the inputs.
                y = delta[run, indices].numpy().astype(np.float64)
                xtx += x.T @ x
                xty += x.T @ y
                count += n
    return mean, std, xtx / count, xty / count, count


def fit_ridge(statistics, alpha):
    # Called for each alpha using the same accumulated training statistics.
    # Solve (X^T X / n + alpha * I) weights = X^T Y / n, with no penalty on the intercept.
    mean, std, xtx, xty, _ = statistics
    penalty = np.eye(len(xtx)) * alpha
    penalty[-1, -1] = 0  # Do not penalize the intercept.
    weights = np.linalg.solve(xtx + penalty, xty)
    model = LocalRidge(mean, std)
    with torch.no_grad():
        # Put the solved stencil into Conv2D weights so inference can use the same device as CNN.
        model.linear.weight.copy_(torch.from_numpy(weights[:-1].T.reshape(3, 7, 5, 5).astype(np.float32)))
        model.linear.bias.copy_(torch.from_numpy(weights[-1].astype(np.float32)))
    return model.eval()


def summarize_folds(rows):
    # Called after CV to combine each candidate's fold scores into cv_summary.json/csv.
    summaries = []
    for candidate in dict.fromkeys(row["candidate"] for row in rows):
        subset = [r for r in rows if r["candidate"] == candidate]
        values = [r["primary"] for r in subset]
        result = {"candidate": candidate, "kind": subset[0]["kind"], "config": subset[0]["config"],
                  "mean_primary": float(np.mean(values)), "std_primary": float(np.std(values, ddof=1)),
                  "folds": len(subset)}
        # ddof=1 gives sample standard deviation across folds; this is not a confidence interval.
        for metric in ("mae", "rmse", "normalized_rmse"):
            for channel, field in enumerate(FIELDS):
                numbers = [r[metric][channel] for r in subset]
                result[f"mean_{metric}_{field}"] = float(np.mean(numbers))
                result[f"std_{metric}_{field}"] = float(np.std(numbers, ddof=1))
        summaries.append(result)
    return summaries
