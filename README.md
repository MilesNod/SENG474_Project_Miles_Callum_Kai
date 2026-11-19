# Fluid Prediction Model

Predict the next horizontal velocity (`u`), vertical velocity (`v`), and pressure
(`p`) fields of a lid-driven cavity solver. Compare a CNN against persistence
(no change) and a learned local ridge regressor, using independent simulation
trajectories, grouped cross-validation, and a reserved final test set.

Run the commands below from the repository root in PowerShell. Python 3.10+ is
required. New datasets, checkpoints, figures, and experiment outputs go under
`artifacts/`, which Git ignores. Existing `.npz` data and old weights are preserved;
the old dense-model weights are incompatible with the new models and must be retrained.

If a dataset/run already exists, continue with
`.\.venv\Scripts\python scripts/run_experiment.py --resume` rather than repeating
generation. See **Continue an interrupted experiment** below for recovery details.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

An existing working environment needs no additional packages. `requirements-tested.txt`
records the tested package versions; the tested environment used Python 3.14.5 and
PyTorch 2.11.0+cu128. CUDA training requires a compatible CUDA-enabled PyTorch build.
Training/evaluation automatically use CUDA when available; `--device cpu` overrides it.

## Quick end-to-end check

This is a small pipeline check, with reduced grid size and pressure iterations.
Its scores are not suitable for the project evaluation. Each dataset/run directory
must be new; use different names when repeating the commands.

```powershell
.\.venv\Scripts\python fluid-model/src/generate_data.py --output artifacts/data/quick_test --runs 12 --steps 12 --grid-size 9 --pressure-iterations 5 --folds 3
.\.venv\Scripts\python fluid-model/src/train.py --data artifacts/data/quick_test --output artifacts/runs/quick_test --epochs 2 --hidden-dims 8 --learning-rates 0.001 --ridge-alphas 0.01 --ridge-max-points 1000 --learning-curve-sizes 2 4 --batch-size 16
.\.venv\Scripts\python fluid-model/src/evaluate.py --run artifacts/runs/quick_test --horizons 1 5 10
.\.venv\Scripts\python fluid-model/src/report.py --run artifacts/runs/quick_test
```

Inspect `artifacts/runs/quick_test/test/one_step.csv`, `rollout.png`, and
`comparison.png`. The report draft is `artifacts/runs/quick_test/report/evaluation.tex`.
The first command freezes a separate 20% final test set; the training command
never evaluates it. The evaluation command explicitly opens the final test set.

## Generate the expanded dataset

```powershell
.\.venv\Scripts\python fluid-model/src/generate_data.py
```

Defaults: **300 independent trajectories, 250 transitions each, 41 x 41 grid**,
five CV folds, 80% development / 20% final test, generation seed 3, split seed 17.
This produces 75,000 transitions, compared with 4,950 in the former training script.
It takes approximately **1.42 GiB** for trajectory storage. Generation writes one
state at a time to disk; the training loader reads the array through a memory map.
An interrupted/failed generation leaves a `.partial.npy` for diagnosis; rerun into
a new output directory rather than reusing the incomplete dataset.

Initial velocities use random and smoothed patterns with amplitudes sampled from
`[0.03, 0.15]`. Initial pressure is zero. The default physical configuration remains
viscosity 0.1, lid speed 1, domain side 1, dt 0.001, and 50 pressure iterations.
Initial velocities are not guaranteed divergence-free: these are synthetic starts
for learning this numerical solver. Finite-value, diffusion-timestep, and CFL
screening checks do not establish independent physical/numerical accuracy.

For a broader physical distribution, generate a **separate** dataset:

```powershell
.\.venv\Scripts\python fluid-model/src/generate_data.py --output artifacts/data/varied --viscosity-range 0.05 0.1 --lid-speed-range -1 1
```

Both learned methods receive viscosity and lid speed as explicit input channels.
One dataset uses one grid/domain/dt/pressure-iteration count. Models accept different
grid dimensions, but this alone does not demonstrate generalization to new grids.
Increasing grid resolution or viscosity may require lowering `--dt`.

Each dataset directory contains:

- `trajectories.npy`: float32 `(runs, steps + 1, 3, H, W)` states, including initial states.
- `metadata.json`: per-run IDs, seeds, initial-condition families/amplitudes, physical
  parameters, channel statistics, generation procedure, and content checksum.
- `splits.json`: fixed development/test trajectory IDs and fold assignments, tied
  to the dataset fingerprint. Every development trajectory validates exactly once.

The original `src/data/data_generation.py` now delegates to this same generator.
The historical `datasetForTestingDefault.npz` is not consumed by the ML pipeline.

## Cross-validation and hyperparameter study

```powershell
.\.venv\Scripts\python fluid-model/src/train.py --learning-rates 0.0003 0.001 --learning-curve-sizes 25 50 100
```

This uses the expanded dataset at `artifacts/data/cavity` and saves to
`artifacts/runs/cavity_cv`. Defaults search widths `{16, 32, 64}`; the command adds
two learning rates, giving **30 CNN CV fits** over five folds. The three requested
learning-curve sizes add 15 CNN fits, plus one development refit. Ridge penalties
default to `{0.0001, 0.01, 1}`. Each fold reuses the same training-pixel sample across
ridge penalties, with at most 100,000 sampled pixels. Ridge samples training pixels
to keep its fit practical; the CNN trains on all training transitions.

Every CNN starts fresh with a seeded initialization, uses Adam and batched CPU/CUDA
training, and runs for at most 50 epochs with early-stopping patience 10. Configure
these with `--epochs`, `--patience`, and `--batch-size`. `--noise 0.01` adds optional
training-only CNN input noise, scaled using that fold's channel standard deviations.
The ridge baseline fits clean inputs. Noise defaults to zero.

Splits operate on **whole trajectories**, never individual frames. Normalization is
fitted separately from each fold's training input states. The CNN and ridge predict
signed state increments; the CNN starts as a no-change predictor. Both receive the
three normalized fields, coordinates, and physical parameters, and both have a
5 x 5 receptive-field extent. No final output ReLU clamps negative velocities/pressure.

The selection metric is **mean channel-normalized RMSE**:

`mean(RMSE_u / std_u, RMSE_v / std_v, RMSE_p / std_p)`

Channel standard deviations come only from the training inputs, with a small floor
for constant fields. Training minimizes normalized MSE; checkpoint selection and
hyperparameter selection minimize validation normalized RMSE. Raw MAE, MSE, and RMSE
are also saved separately for each field. This avoids averaging pressure and velocity
errors in incompatible units. CV mean/standard deviation use equal weight per fold;
fold standard deviation is not a confidence interval.

## Continue an interrupted experiment

If `artifacts/data/cavity` already contains a complete dataset, skip generation.
If training created `artifacts/runs/cavity_cv` and stopped before finishing, run:

```powershell
.\.venv\Scripts\python fluid-model/src/train.py --resume
```

This restores the run's original hyperparameters and learning-curve sizes, reuses
completed fits (including fits not yet recorded in the CV summary), and continues
unfinished work. For another run directory, add `--output PATH`. Use
`--resume --dry-run` to inspect progress without starting training; `--verify-data`
also checks the stored trajectory bytes against the content checksum. Scientific
settings cannot be changed during a resume; use a new run directory for those.

New CNN fits save optimizer state, epoch, early-stopping state, random-number
generators, and shuffle state in `training_state.pt` after each complete epoch.
They resume from that epoch. Older partial fits created before this support lack
optimizer/RNG state: their weights/history are copied into `legacy_backups/`, and
only that incomplete fit restarts from epoch 1. Completed older fits are reused.
Research inputs and normalization are checked before reuse. An OS lock prevents
two training processes from writing the same run concurrently. Original source
snapshots are retained; resumed code/environment snapshots go in `resume_sessions/`.

Wait for training to report completion, then run evaluation and reporting. A
pasted list of PowerShell commands continues even when an earlier Python command
fails. The following Python runner checks exit codes and stops at the failed stage:

```powershell
.\.venv\Scripts\python scripts/run_experiment.py --resume
```

It reuses the saved dataset, resumes training, evaluates only when test results
are absent, then generates the report while preserving an existing edited LaTeX draft.
Without `--resume`, it starts a fresh run
with the full search settings shown above; use `--run` and `--data` for custom paths,
and `--device cpu` to force CPU execution. Add `--dry-run` to check saved progress
without training. Fresh experiments still require a new output directory.

After selecting CNN/ridge configurations, training refits both on all development
trajectories. The final CNN's fixed epoch budget is the median best epoch from its
selected CV configuration. It uses no test-based stopping. Optional learning curves
train the selected CNN configuration on nested subsets within each fold, leaving
the fold's validation IDs unchanged; sizes must fit the smallest training fold.
Each subset fits its own model normalizer, while learning-curve validation scores
share the full training fold's channel scales so the metric is comparable across sizes.

Useful outputs:

| Artifact | Purpose |
| --- | --- |
| `run.json`, `splits.json`, `dataset_metadata.json` | Configuration, split IDs, package/hardware versions, Git revision, seeds, selected settings |
| `source/` | Snapshot of the actual Python/report sources and requirements, including uncommitted changes |
| `cv_summary.csv`, `cv_summary.json` | Configuration means and fold standard deviations |
| `fold_metrics.csv`, `cv_results.json` | Individual fold metrics and per-trajectory errors |
| `cv/<candidate>/fold_N/` | Independent checkpoints; CNN epoch histories and fit summaries |
| `final/cnn/best.pt`, `final/ridge/best.pt` | Selected models refitted on development trajectories |
| `hyperparameters.png` | Width/learning-rate sensitivity figure |
| `learning_curve.csv`, `learning_curve.png` | Optional dataset-size study |

Checkpoints include architecture, normalizer buffers, training IDs, configuration,
and dataset identity. RNGs are seeded and cuDNN uses deterministic settings; bitwise
agreement across hardware/software versions is not guaranteed. `--verify-data`
additionally scans trajectory bytes against the saved SHA256 checksum. Metadata
fingerprints and split consistency are always checked. To relocate artifacts, pass
the dataset's new path to evaluation using `--data`.

## Final test and rollouts

Only do this after fixing the experimental choices:

```powershell
.\.venv\Scripts\python fluid-model/src/evaluate.py --benchmark-repeats 30
```

The evaluator loads the development refits and evaluates the frozen test IDs. It
rejects mismatched datasets, splits, or checkpoints. It saves one-step errors and
autoregressive errors at horizons 1, 10, 25, 50, and 100, starting at solver timestep 0.
Change `--rollout-start` and `--horizons` before your final evaluation to match the
research questions; horizons must fit the saved trajectory. Later start times can
separately assess behavior after initial transients. Training and test trajectories
have the same temporal extent, so results do not silently mix in a later time range.

`test/` contains `results.json`, `one_step.csv`, `rollout.csv`, per-trajectory results,
`comparison.png`, and `rollout.png`. Nonfinite rollouts are explicitly marked as
failures with null metrics; failed trajectories are not averaged away. Finite but
inaccurate predictions still receive their measured errors.

Optional timing measures batch-one wall time after five warmups, including host/device
transfers. It compares float32 learned models with the float64 NumPy solver and records
the hardware; it is not a hardware-independent speed claim. `--save-plot PATH` also
copies the CNN comparison figure to a chosen path.

Existing final-test outputs are protected; `--repeat` explicitly permits another
evaluation. Repeated inspection/tuning against these results invalidates their role
as an untouched final test. Use CV for further development.

## Evaluation report

```powershell
.\.venv\Scripts\python fluid-model/src/report.py --authors "Miles Fullname;Callum Fullname;Kai Fullname" --repo-url https://github.com/OWNER/REPOSITORY
```

Replace the example author names and repository URL with your own. This fills the
ACM `sigconf` template in `reports/evaluation_template.tex` with the **actual saved
measurements**. It includes research questions, dataset statistics, baseline/setup
descriptions, CV/final-test tables, hyperparameter/rollout figures, and optional
learning-curve/timing discussion. Review and edit the discussion and citations.

With an ACM-capable LaTeX installation providing `pdflatex` and `acmart`:

```powershell
.\.venv\Scripts\python fluid-model/src/report.py --use-existing --package
```

`--use-existing` preserves manual edits to `report/evaluation.tex`. Packaging compiles
twice, verifies **exactly two pages**, and creates `4_Evaluation_Miles_Callum_Kai.zip`
with source, PDF, and required figures. `--package-name` changes the archive filename.
If LaTeX is unavailable, the generated source/figures can be uploaded to Overleaf.
The two-page layout must be checked after compilation and condensed if necessary.
Order the authors as required by the course, with any graduate student last.

## Tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Tests cover solver regression against the attributed original implementation, exact
trajectory counts, independent generation, reproducibility/checksums, leakage-free
splits and normalization, noise shapes/immutability, signed fields and flexible grids,
checkpoint round trips, validation-based selection, ridge fitting, batch-independent
metrics, and explicit rollout failures. Use `--help` on any script for all options.
