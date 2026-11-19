"""Trajectory storage, training-only statistics, and reproducible grouped splits."""

# This is the shared data code used by generation, training, and evaluation.
# Steps:
# 1. Generate separate fluid runs and save all their states.
# 2. Split whole runs into development, validation folds, and final test data.
# 3. Let PyTorch ask for pairs of states: current state -> next state.
# u: horizontal velocity; v: vertical velocity; p: pressure.
# A trajectory is one full run, so its neighbouring frames should stay together.

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
# The ML scripts live two folders down; this makes src.simulation importable from there.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import FluidState, NavierStokesSolver, SolverConfig

DEFAULT_DATA = ROOT / "artifacts" / "data" / "cavity"


def atomic_write(path, writer):
    """Publish a complete file with a same-directory rename, preserving the previous file on failure."""
    # Used when saving progress. Write a temporary file first so an interruption
    # does not leave half a checkpoint where we expected a complete one.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def write_json(path, value):
    # Used for settings, split IDs, and results; JSON lets us inspect these by hand.
    # allow_nan=False catches invalid numbers rather than putting them into a report.
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    atomic_write(path, lambda pending: pending.write_text(encoded, encoding="utf-8"))


def fingerprint(value):
    # Used to identify a dataset by its metadata. Sorting the keys means changing
    # the order of a dictionary does not make it look like a different dataset.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_random_initial_state(solver, rng, amplitude, family="random"):
    """Synthetic random or smooth velocities; the solver applies the lid at step 1.

    These starts are not guaranteed divergence-free. The solver supplies the
    reference states; this limitation is recorded in the dataset description.
    """
    # Called once per trajectory by generate_dataset, before the first solver step.
    # amplitude controls how much initial motion we add; rng makes it repeatable.
    shape = (solver.config.n_points,) * 2
    fields = []
    for _ in range(2):
        field = rng.normal(size=shape)
        if family == "smooth":
            # Average each point with its 8 neighbours to remove the sharp random jumps.
            for _ in range(4):
                padded = np.pad(field, 1, mode="reflect")
                field = sum(padded[i:i + shape[0], j:j + shape[1]]
                            for i in range(3) for j in range(3)) / 9
            # Smoothing shrinks the values, so put the spread back before applying amplitude.
            field /= max(float(field.std()), 1e-12)
        elif family != "random":
            raise ValueError(f"Unknown initial-condition family: {family}")
        field *= amplitude
        # Start with stationary edges and zero pressure; the solver applies the lid later.
        field[0, :] = field[-1, :] = 0
        field[:, 0] = field[:, -1] = 0
        fields.append(field)
    return FluidState(*fields, np.zeros(shape))


def make_configs(n, seed):
    """Legacy helper; generated datasets can instead record parameter ranges."""
    #current generator does not call this helper
    # it changes parameters inside generate_dataset only when ranges are supplied.
    rng = np.random.default_rng(seed)
    return [replace(SolverConfig(), viscosity=float(rng.uniform(0.05, 0.1)),
                    top_wall_velocity=float(rng.uniform(0.3, 2) * rng.choice([-1, 1])))
            for _ in range(n)]


def make_splits(num_runs, folds=5, test_fraction=0.2, seed=17):
    # Called during generation to save the exact same partitions for every model.
    
    # We shuffle RUN IDs here, not individual timesteps, to avoid leaking nearby frames.
    if not 0 < test_fraction < 1 or folds < 2:
        raise ValueError("Use 0 < test_fraction < 1 and at least two folds.")
    order = np.random.default_rng(seed).permutation(num_runs)
    test_count = max(1, int(np.ceil(num_runs * test_fraction)))
    test, development = order[:test_count], order[test_count:]
    # Hold the final test aside first; cross-validation only uses development runs.
    if len(development) < folds:
        raise ValueError("Not enough development trajectories for the requested folds.")
    validation_groups = np.array_split(development, folds)
    # Each group validates once; all the other groups train that fold's model.
    result = {
        "seed": seed, "test_fraction": test_fraction,
        "test": test.tolist(), "development": development.tolist(),
        "folds": [{"train": np.concatenate([g for j, g in enumerate(validation_groups) if j != i]).tolist(),
                   "validation": group.tolist()}
                  for i, group in enumerate(validation_groups)],
    }
    validate_splits(result, num_runs)
    return result


def validate_splits(splits, num_runs):
    # Called when splits are made and when a saved dataset is opened.
    # Check that no run is missing, duplicated, or present on both sides of a split.
    development, test = splits["development"], splits["test"]
    if (not development or not test or len(set(development + test)) != num_runs
            or sorted(development + test) != list(range(num_runs))):
        raise ValueError("Development/test IDs must partition the dataset without duplicates.")
    seen = []
    if len(splits["folds"]) < 2:
        raise ValueError("At least two folds are required.")
    for fold in splits["folds"]:
        train, validation = fold["train"], fold["validation"]
        if (not train or not validation or set(train) & set(validation)
                or sorted(train + validation) != sorted(development)):
            raise ValueError("Each fold must partition development trajectories without overlap.")
        seen.extend(validation)
    if sorted(seen) != sorted(development):
        # Across all folds, every development run must appear in validation exactly once.
        raise ValueError("Each development trajectory must be validated exactly once.")


def generate_dataset(output, num_runs=300, num_steps=250, seed=3, folds=5,
                     test_fraction=0.2, split_seed=17, config=None,
                     amplitudes=(0.03, 0.15), families=("random", "smooth"),
                     viscosity_range=None, lid_speed_range=None, progress=True):
    """Write one simulation at a time to an uncompressed, memory-mapped NPY file."""
    # Called by generate_data.py. This is where the actual numerical simulations run.
    # num_runs: independent trajectories; num_steps: transitions within each run.
    # Steps: validate settings -> allocate disk array -> simulate -> save metadata/splits.
    config = config or SolverConfig()
    if num_runs < 1 or num_steps < 1:
        raise ValueError("num_runs and num_steps must be positive.")
    for name, bounds, positive in (("amplitude", amplitudes, True),
                                  ("viscosity", viscosity_range, True),
                                  ("lid speed", lid_speed_range, False)):
        if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all()
                                   or bounds[0] > bounds[1] or (positive and bounds[0] <= 0)):
            raise ValueError(f"Invalid {name} range.")
    if not families or not set(families) <= {"random", "smooth"}:
        raise ValueError("Choose initial-condition families from random and smooth.")
    NavierStokesSolver(config)
    # Larger viscosity gives a tighter diffusion limit, so check that end of the range too.
    if viscosity_range:
        NavierStokesSolver(replace(config, viscosity=float(viscosity_range[1])))
    splits = make_splits(num_runs, folds, test_fraction, split_seed)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Dataset directory is not empty: {output}. Reuse the existing dataset by skipping generation, "
                              "or choose a new --output for different data.")
    output.mkdir(parents=True, exist_ok=True)
    shape = (num_runs, num_steps + 1, 3, config.n_points, config.n_points)
    # The extra state is S_0: 250 transitions require states S_0 through S_250.
    # A memory map writes into the disk file without building the whole dataset in RAM.
    pending_path = output / "trajectories.partial.npy"
    trajectories = np.lib.format.open_memmap(pending_path, mode="w+", dtype="float32", shape=shape)
    digest, records = hashlib.sha256(), []
    total, squares = np.zeros(3), np.zeros(3)
    # Per-channel sums and squared sums let us calculate descriptive statistics as we go.
    minima, maxima = np.full(3, np.inf), np.full(3, -np.inf)
    count = 0
    try:
        for run, child in enumerate(tqdm(np.random.SeedSequence(seed).spawn(num_runs),
                                         desc="Generating trajectories", disable=not progress)):
            # Give each run its own seed, rather than resetting every run to the same start.
            trajectory_seed = int(child.generate_state(1, dtype=np.uint64)[0])
            rng = np.random.default_rng(trajectory_seed)
            cfg = replace(config,
                          viscosity=float(rng.uniform(*viscosity_range)) if viscosity_range else config.viscosity,
                          top_wall_velocity=float(rng.uniform(*lid_speed_range)) if lid_speed_range else config.top_wall_velocity)
            # With no parameter ranges, viscosity/lid speed stay fixed across all runs.
            family = str(rng.choice(families))
            amplitude = float(rng.uniform(*amplitudes))
            solver = NavierStokesSolver(cfg)
            state = make_random_initial_state(solver, rng, amplitude, family)
            for step in range(num_steps + 1):
                snapshot = state.to_array().astype(np.float32)
                if not np.isfinite(snapshot).all():
                    raise ValueError(f"Nonfinite reference at trajectory {run}, step {step}.")
                trajectories[run, step] = snapshot
                digest.update(snapshot.tobytes())
                # This checksum records the actual state values, not just the file's name.
                values = snapshot.astype(np.float64)
                total += values.sum(axis=(1, 2))
                squares += np.square(values).sum(axis=(1, 2))
                minima = np.minimum(minima, values.min(axis=(1, 2)))
                maxima = np.maximum(maxima, values.max(axis=(1, 2)))
                count += config.n_points ** 2
                if step < num_steps:
                    dx = cfg.domain_size / (cfg.n_points - 1)
                    cfl = cfg.dt * (np.abs(state.u).max() + np.abs(state.v).max()) / dx
                    # CFL estimates how far motion travels across the grid in one step.
                    # Passing this screen alone does not prove the simulation is accurate.
                    if cfl > 1:
                        raise ValueError(f"Advective CFL exceeds 1 in trajectory {run}; lower dt/amplitude/speed.")
                    state = solver.step(state)
            records.append({"id": run, "seed": trajectory_seed, "family": family,
                            "amplitude": amplitude, "config": asdict(cfg)})
        trajectories.flush()
    finally:
        del trajectories
    pending_path.replace(output / "trajectories.npy")
    # Only publish the finished filename after all runs have been written.
    mean = total / count
    metadata = {
        # These full-dataset stats describe the data; training fits its own stats below.
        "format_version": 2, "seed": seed, "shape": list(shape), "dtype": "float32",
        "channels": ["u", "v", "p"], "transitions": num_runs * num_steps,
        "num_steps": num_steps, "base_config": asdict(config), "trajectories": records,
        "ranges": {"amplitude": list(amplitudes), "families": list(families),
                   "viscosity": viscosity_range, "lid_speed": lid_speed_range},
        "content_sha256": digest.hexdigest(),
        "statistics": {"mean": mean.tolist(), "std": np.sqrt(np.maximum(squares / count - mean ** 2, 0)).tolist(),
                       "min": minima.tolist(), "max": maxima.tolist(), "nonfinite_values": 0},
        "source": "Synthetic lid-driven cavity trajectories from src/simulation/solver.py",
        "limitations": "Initial velocities are not guaranteed divergence-free; only one grid/domain/dt per dataset. "
                        "CFL <= 1 is a screening check, not a proof of numerical accuracy/stability.",
    }
    metadata["dataset_id"] = fingerprint(metadata)
    splits["dataset_id"] = metadata["dataset_id"]
    write_json(output / "metadata.json", metadata)
    write_json(output / "splits.json", splits)
    return TrajectoryStore(output)


class TrajectoryStore:
    # Open one saved dataset once, then share it between the different fold views.
    def __init__(self, path):
        # Called by train.py/evaluate.py; verify that states, metadata, and splits agree.
        self.path = Path(path).resolve()
        self.metadata = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        self.splits = json.loads((self.path / "splits.json").read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != 2:
            raise ValueError("Unsupported dataset format; regenerate with generate_data.py.")
        without_id = {k: v for k, v in self.metadata.items() if k != "dataset_id"}
        if self.metadata["dataset_id"] != fingerprint(without_id):
            raise ValueError("Dataset metadata fingerprint does not match its contents.")
        self.trajs = np.load(self.path / "trajectories.npy", mmap_mode="r", allow_pickle=False)
        # Read-only memory map: training can read frames without loading every run at once.
        if (list(self.trajs.shape) != self.metadata["shape"] or self.trajs.dtype != np.float32
                or len(self.metadata["trajectories"]) != len(self.trajs)):
            raise ValueError("Trajectory file does not match metadata.")
        if self.splits.get("dataset_id") != self.metadata["dataset_id"]:
            raise ValueError("Split manifest belongs to a different dataset.")
        validate_splits(self.splits, len(self.trajs))
        self.num_steps = self.trajs.shape[1] - 1
        self.parameters = np.asarray([[r["config"]["viscosity"], r["config"]["top_wall_velocity"]]
                                      for r in self.metadata["trajectories"]], dtype=np.float32)

    def verify_content(self):
        # Only used when --verify-data is requested; reading every state costs more time.
        # Recalculate the checksum to catch changes to the array after generation.
        digest = hashlib.sha256()
        for trajectory in self.trajs:
            digest.update(trajectory.tobytes())
        if digest.hexdigest() != self.metadata["content_sha256"]:
            raise ValueError("Trajectory content checksum mismatch.")


def fit_normalization(store, trajectory_ids):
    """Streaming input-state statistics, calculated ONLY from supplied training IDs."""
    # Called before a fold trains, and again before the final development refit.
    # The supplied IDs must be the training runs, otherwise validation/test data leaks in.
    if not len(trajectory_ids):
        raise ValueError("Normalization requires training trajectories.")
    total, squares, count = np.zeros(3), np.zeros(3), 0
    for run in trajectory_ids:
        for start in range(0, store.num_steps, 32):
            # Read small time chunks. The last stored state is only a target, not an input.
            values = store.trajs[run, start:min(start + 32, store.num_steps)].astype(np.float64)
            total += values.sum(axis=(0, 2, 3))
            squares += np.square(values).sum(axis=(0, 2, 3))
            count += values.shape[0] * values.shape[2] * values.shape[3]
    mean = total / count
    # variance = mean(x^2) - mean(x)^2; floor std so a constant field cannot divide by zero.
    std = np.maximum(np.sqrt(np.maximum(squares / count - mean ** 2, 0)), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


class FluidDataset(Dataset):
    """A view of selected whole trajectories; returns (state, target, nu/lid)."""
    def __init__(self, store, trajectory_ids, noise=0.0, channel_std=None):
        # DataLoader uses this view to see only the runs assigned to its current fold.
        # Noise is optional training augmentation, scaled with training-only channel std.
        self.store = store
        self.trajectory_ids = list(trajectory_ids)
        if (not self.trajectory_ids or len(set(self.trajectory_ids)) != len(self.trajectory_ids)
                or min(self.trajectory_ids) < 0 or max(self.trajectory_ids) >= len(store.trajs)):
            raise ValueError("Supply distinct, valid trajectory IDs.")
        if not np.isfinite(noise) or noise < 0:
            raise ValueError("Noise must be finite and nonnegative.")
        if noise and channel_std is None:
            raise ValueError("Training noise requires training-fold channel standard deviations.")
        self.noise = noise
        self.channel_std = torch.as_tensor(channel_std if channel_std is not None else np.ones(3),
                                           dtype=torch.float32).reshape(3, 1, 1)

    def __len__(self):
        # PyTorch asks this for the sample count: runs in this view * transitions per run.
        return len(self.trajectory_ids) * self.store.num_steps

    def __getitem__(self, index):
        # PyTorch calls this for one training/validation pair, not a whole trajectory.
        if not 0 <= index < len(self):
            raise IndexError(index)
        local_run, step = divmod(index, self.store.num_steps)
        # Example: with 250 steps, index 251 means local run 1, timestep 1.
        run = self.trajectory_ids[local_run]
        state = torch.from_numpy(self.store.trajs[run, step].copy())
        # Copy because the disk array is read-only; augmentation must not change saved data.
        target = torch.from_numpy(self.store.trajs[run, step + 1].copy())
        if self.noise:
            # Only perturb the input. The clean next state remains the target to learn.
            state = state + self.noise * self.channel_std * torch.randn_like(state)
        return state, target, torch.from_numpy(self.store.parameters[run].copy())
