"""Standalone figures built from the saved experimental measurements."""

# Training/evaluation call these after saving their numbers; plotting does not train a model.
# Agg writes figures to files, so running scripts does not need to open a plot window.

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def plot_cross_validation(output):
    # Called at the end of train.py. Read completed CV summaries and plot mean +/- fold std.
    output = Path(output)
    summaries = json.loads((output / "cv_summary.json").read_text(encoding="utf-8"))
    cnn = [r for r in summaries if r["kind"] == "cnn"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for rate in sorted({r["config"]["learning_rate"] for r in cnn}):
        # Left panel: vary width while keeping learning rate fixed for each line.
        rows = sorted((r for r in cnn if r["config"]["learning_rate"] == rate), key=lambda r: r["config"]["hidden_dim"])
        axes[0].errorbar([r["config"]["hidden_dim"] for r in rows], [r["mean_primary"] for r in rows],
                         yerr=[r["std_primary"] for r in rows], marker="o", capsize=3, label=f"lr={rate:g}")
    for width in sorted({r["config"]["hidden_dim"] for r in cnn}):
        # Right panel: vary learning rate while keeping width fixed for each line.
        rows = sorted((r for r in cnn if r["config"]["hidden_dim"] == width), key=lambda r: r["config"]["learning_rate"])
        axes[1].errorbar([r["config"]["learning_rate"] for r in rows], [r["mean_primary"] for r in rows],
                         yerr=[r["std_primary"] for r in rows], marker="o", capsize=3, label=f"width={width}")
    for axis, xlabel in zip(axes, ("Hidden channels", "Learning rate")):
        axis.set(xlabel=xlabel, ylabel="Mean channel-normalized RMSE", title="CV mean ± fold standard deviation")
        axis.legend()
        axis.grid(alpha=0.3)
    axes[1].set_xscale("log")
    axes[0].set_xticks(sorted({r["config"]["hidden_dim"] for r in cnn}))
    figure.savefig(output / "hyperparameters.png", dpi=180)
    plt.close(figure)
    curve = output / "learning_curve.csv"
    # This study is optional, so do not expect its CSV for runs without subset-size experiments.
    if curve.is_file():
        with curve.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        figure, axis = plt.subplots(figsize=(5, 4), layout="constrained")
        axis.errorbar([int(r["training_trajectories"]) for r in rows], [float(r["mean_primary"]) for r in rows],
                       yerr=[float(r["std_primary"]) for r in rows], marker="o", capsize=3)
        axis.set(xlabel="Independent training trajectories per fold", ylabel="Validation NRMSE", title="Dataset learning curve")
        axis.grid(alpha=0.3)
        figure.savefig(output / "learning_curve.png", dpi=180)
        plt.close(figure)


def save_comparison(expected, predicted, path, horizon):
    # Called for the first test trajectory when its final CNN rollout is finite.
    # Rows: solver truth, CNN prediction, absolute error. Columns: u, v, p.
    figure, axes = plt.subplots(3, 3, figsize=(10, 8), layout="constrained")
    for channel, name in enumerate(("u", "v", "p")):
        truth, estimate = expected[channel], predicted[channel]
        minimum, maximum = min(float(truth.min()), float(estimate.min())), max(float(truth.max()), float(estimate.max()))
        # Share the colour scale between truth/prediction; separate scales can hide large errors.
        for row, label, field in ((0, "Solver", truth), (1, "CNN", estimate)):
            plotted = axes[row, channel].imshow(field, origin="lower", cmap="coolwarm", vmin=minimum, vmax=maximum)
            axes[row, channel].set_title(f"{label}: {name}")
            figure.colorbar(plotted, ax=axes[row, channel], shrink=0.8)
        plotted = axes[2, channel].imshow(abs(estimate - truth), origin="lower", cmap="magma", vmin=0)
        axes[2, channel].set_title(f"Absolute error: {name}")
        figure.colorbar(plotted, ax=axes[2, channel], shrink=0.8)
    figure.suptitle(f"First test trajectory after {horizon} predicted steps")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_rollouts(rows, path):
    # Called by evaluate.py to show how errors change as predictions feed back into the model.
    figure, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    for name in dict.fromkeys(row["model"] for row in rows):
        finite = [r for r in rows if r["model"] == name and r["status"] == "finite"]
        # Failed horizons stay in the CSV, but NaN/Inf cannot be plotted as numerical points.
        # Even a finite point can have a huge error; finite does not mean the rollout is useful.
        if finite:
            axis.plot([r["horizon"] for r in finite], [r["primary"] for r in finite], marker="o", label=name)
    axis.set(xlabel="Autoregressive steps", ylabel="Mean channel-normalized RMSE", title="Held-out trajectory rollouts")
    axis.set_yscale("symlog", linthresh=0.01)
    # A logarithmic scale outside a small interval keeps small errors and exploding errors visible.
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)
