"""Run the experiment stages in order, stopping immediately if a stage fails."""

# This is the one-command entry point from the README.
# Steps: generate/reuse data -> train/resume -> evaluate -> make the report source.
# The individual scripts can still be run separately while inspecting/debugging a stage.

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    # Used for CLI paths: relative paths are relative to this repository, not some other terminal folder.
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv=None):
    # argv is optional so the tests can supply flags without opening a real terminal session.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data", help="Default: the saved dataset when resuming, otherwise artifacts/data/cavity.")
    parser.add_argument("--run", default="artifacts/runs/cavity_cv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Check training inputs/progress without starting any stage.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 25, 50, 100])
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    args = parser.parse_args(argv)
    run = project_path(args.run)
    if args.resume:
        # Find the dataset recorded by the existing run, unless --data explicitly relocates it.
        manifest = run / "run.json"
        if not manifest.is_file():
            parser.error(f"No saved run at {run}; omit --resume to start a new experiment.")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        data = project_path(args.data or saved["data_path"])
    else:
        data = project_path(args.data or "artifacts/data/cavity")
        if run.exists() and any(p.name != ".training.lock" for p in run.iterdir()):
            parser.error(f"Run directory already exists: {run}. Use --resume or a new --run.")

    def stage(script, *options):
        # Start each stage with this same Python environment. check=True stops the sequence
        # on a nonzero exit code, avoiding the cascading errors from pasted shell commands.
        command = [sys.executable, "-B", str(ROOT / "fluid-model" / "src" / script), *map(str, options)]
        subprocess.run(command, cwd=ROOT, check=True)

    try:
        dataset_ready = (data / "metadata.json").is_file() and (data / "trajectories.npy").is_file()
        if args.dry_run and not dataset_ready:
            parser.error(f"Dataset is missing at {data}; a dry run cannot generate it.")
        if dataset_ready:
            # A complete dataset can be reused; train.py checks its metadata and split identity.
            print(f"Reusing saved dataset: {data}", flush=True)
        else:
            stage("generate_data.py", "--output", data)
        training = ["--data", data, "--output", run, "--device", args.device]
        if args.resume:
            training.append("--resume")
        else:
            training.extend(["--learning-rates", "0.0003", "0.001", "--learning-curve-sizes", "25", "50", "100"])
        if args.dry_run:
            # Ask training to validate/show progress, then stop before evaluation or reporting.
            stage("train.py", *training, "--dry-run")
            return 0
        stage("train.py", *training)
        if (run / "test" / "results.json").is_file():
            # Reusing final-test results avoids rerunning the test whenever the runner resumes.
            print("Reusing existing final-test results.", flush=True)
        else:
            stage("evaluate.py", "--run", run, "--data", data, "--device", args.device,
                  "--benchmark-repeats", args.benchmark_repeats, "--horizons", *args.horizons)
        report_options = ["--run", run]
        if (run / "report" / "evaluation.tex").is_file():
            # Do not replace a paper draft we may have already edited by hand.
            report_options.append("--use-existing")
        stage("report.py", *report_options)
        return 0
    except subprocess.CalledProcessError as error:
        print(f"Experiment stage failed (exit {error.returncode}). Later stages were not started.", file=sys.stderr)
        return error.returncode or 1
    except KeyboardInterrupt:
        # Ctrl+C ends this invocation; completed-epoch checkpoints are available for --resume.
        print("Experiment interrupted. Use --resume to continue from saved progress.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
