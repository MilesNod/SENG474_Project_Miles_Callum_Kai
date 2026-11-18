"""Build an ACM evaluation draft from measured results; optionally compile/package it."""

# Run this after evaluate.py. It turns the saved numbers into a paper draft;
# it does not rerun the final test or make up results to fill missing sections.
# Steps: read artifacts -> fill dataset/setup/results text -> insert into template -> optionally compile/zip.

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from data import ROOT
from experiment import DEFAULT_RUN


def latex(value):
    # Used for names and other plain text that goes into LaTeX. Characters like & and _
    # have special meanings there, so escape them before putting them in the document.
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(character, character) for character in str(value))


def number(value):
    # Used throughout the draft to keep numerical tables/text at 4 significant figures.
    return f"{value:.4g}"


def table(caption, header, rows, columns):
    # Build the repeated LaTeX table structure from actual saved result rows.
    # columns describes alignment, e.g. "lrrrr" is one text column and four number columns.
    return ("\\begin{table}[ht]\n\\centering\n\\small\n\\caption{" + caption + "}\n"
            "\\begin{tabular}{" + columns + "}\n\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n"
            + "\n".join(" & ".join(row) + " \\\\" for row in rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def generate_report(args):
    # Called by main (or the pipeline runner). Missing training/test results stop us here
    # instead of producing a report with placeholder performance numbers.
    run = Path(args.run).resolve()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    info, data = read(run / "run.json"), read(run / "dataset_metadata.json")
    if info["status"] != "complete":
        raise ValueError(f"Training is incomplete. Continue it with train.py --resume --output {run} before generating the report.")
    test = Path(args.test_output).resolve() if args.test_output else run / "test"
    if not (test / "results.json").is_file():
        raise ValueError(f"Final-test results are missing. Run evaluate.py --run {run} after training completes.")
    results = read(test / "results.json")
    if info["status"] != "complete" or results["dataset_id"] != info["dataset_id"]:
        raise ValueError("Report requires completed, matching training and final-test results.")
    if results["test_ids"] != info["splits"]["test"]:
        raise ValueError("Evaluation test IDs differ from the run manifest.")
    output = Path(args.output).resolve() if args.output else run / "report"
    output.mkdir(parents=True, exist_ok=True)
    for source in (run / "hyperparameters.png", test / "rollout.png"):
        # Put figures beside evaluation.tex so it can be moved into Overleaf as one folder.
        shutil.copyfile(source, output / source.name)
    shape, config, arguments = data["shape"], data["base_config"], info["arguments"]
    authors = "\n".join("\\author{" + latex(name.strip()) + "}\n"
                        "\\affiliation{\\institution{University of Victoria}\\country{Canada}}"
                        for name in args.authors.split(";") if name.strip())
    if not authors:
        raise ValueError("Supply at least one author.")
    # Dataset section: counts, grid size, physical settings, and the initial-condition families.
    counts = {family: sum(r["family"] == family for r in data["trajectories"])
              for family in sorted({r["family"] for r in data["trajectories"]})}
    parameter_text = []
    for key, label in (("viscosity", "Viscosity"), ("top_wall_velocity", "Lid speed")):
        values = [r["config"][key] for r in data["trajectories"]]
        parameter_text.append(f"{label}: [{number(min(values))}, {number(max(values))}]")
    dataset = (f"We generated {shape[0]} independent trajectories on a ${shape[3]}\\times{shape[4]}$ grid, "
               f"with {data['num_steps']} transitions per trajectory ({data['transitions']} examples total), "
               f"$\\Delta t={number(config['dt'])}$, domain side {number(config['domain_size'])}, and "
               f"{config['pressure_iterations']} pressure iterations per step. "
               + "; ".join(parameter_text) + ". Initial-condition counts: "
               + ", ".join(f"{latex(family)}={count}" for family, count in counts.items()) + ". "
               f"Amplitude range: [{number(data['ranges']['amplitude'][0])}, {number(data['ranges']['amplitude'][1])}]. "
               f"Generation seed: {data['seed']}; split seed: {info['splits']['seed']}. "
               f"We reserve {len(info['splits']['test'])} trajectories for the final test and "
               f"{len(info['splits']['development'])} for development.")
    folds = len(info["splits"]["folds"])
    # Setup section: describe what actually ran, including selection and final refitting.
    cnn, ridge = info["selected"]["cnn"], info["selected"]["ridge"]
    setup = (f"All methods use the same {folds} trajectory-level folds; no simulation contributes frames "
             "to both sides of a fold. Each fold has an independently fitted normalizer and a fresh model. "
             f"CNN widths are {latex(arguments['hidden_dims'])}, learning rates {latex(arguments['learning_rates'])}, "
             f"batch size {arguments['batch_size']}, maximum epochs {arguments['epochs']}, and early-stopping "
             f"patience {arguments['patience']} (zero disables stopping). Training noise is {number(arguments['noise'])} "
             f"times channel standard deviation. Ridge penalties are {latex(arguments['ridge_alphas'])}; "
             f"at most {arguments['ridge_max_points']} training pixels are sampled per fold. "
             f"The selected CNN has width {cnn['config']['hidden_dim']} and learning rate "
             f"{number(cnn['config']['learning_rate'])}; selected ridge penalty is {number(ridge['config']['alpha'])}. "
             f"After selection, we refit on all development trajectories; CNN duration is the median best "
             f"fold epoch ({info['final_epochs']} epochs). The final test is excluded from selection.")
    summary = read(run / "cv_summary.json")
    # RQ1 tables: keep development CV variation separate from the final-test measurements.
    selected_rows = [next(r for r in summary if r["kind"] == "persistence"), cnn, ridge]
    cvtable = table("Selected configurations: development CV error (mean $\\pm$ fold SD).",
                    ["Method", "Normalized RMSE"],
                    [[latex(row["kind"]), f"${number(row['mean_primary'])} \\pm {number(row['std_primary'])}$"]
                     for row in selected_rows], "lr")
    scores = results["one_step"]
    testtable = table("Final-test one-step errors. Lower is better; raw fields have different units.",
                      ["Method", "$u$ RMSE", "$v$ RMSE", "$p$ RMSE", "$E$"],
                      [[latex(name), *[number(value) for value in metrics["rmse"]], number(metrics["primary"])]
                       for name, metrics in scores.items()], "lrrrr")
    best_method = min(scores, key=lambda name: scores[name]["primary"])
    # Derive "lower/higher" from the numbers rather than assuming our CNN must win.
    improvement = 100 * (1 - scores["cnn"]["primary"] / scores["persistence"]["primary"]) if scores["persistence"]["primary"] else None
    comparison = f"The lowest final-test normalized error is obtained by {latex(best_method)}. "
    if improvement is not None:
        comparison += (f"CNN error is {number(abs(improvement))}\\% "
                       + ("lower" if improvement >= 0 else "higher") + " than persistence. ")
    comparison += "This comparison addresses predictive agreement with this numerical reference; it does not establish physical accuracy."
    cnn_rows = [row for row in summary if row["kind"] == "cnn"]
    # RQ2: summarize the tested configuration range; the figure shows the individual settings.
    sensitivity = (f"Across the {len(cnn_rows)} tested CNN configurations, mean validation $E$ ranges "
                   f"from {number(min(r['mean_primary'] for r in cnn_rows))} to {number(max(r['mean_primary'] for r in cnn_rows))}. "
                   "The selected configuration minimizes mean fold error. The plot shows the measured search; "
                   "claims about settings outside this search are not supported.")
    finite = [row for row in results["rollouts"] if row["model"] == "cnn" and row["status"] == "finite"]
    # RQ3: report usable numerical entries AND failures. A finite error can still be enormous.
    if finite:
        first, last = finite[0], finite[-1]
        rollout = (f"Starting from solver timestep {results['arguments']['rollout_start']}, CNN $E$ is "
                   f"{number(first['primary'])} at horizon {first['horizon']} and {number(last['primary'])} "
                   f"at horizon {last['horizon']}. These predictions use model outputs as subsequent inputs. ")
    else:
        rollout = "The CNN produces nonfinite rollout values at the measured horizons. "
    failed = [row for row in results["rollouts"] if row["status"] != "finite"]
    rollout += (f"There are {len(failed)} failed method/horizon entries; no failed trajectories are silently excluded."
                if failed else "All measured method/horizon entries remain finite; finite predictions can still have large errors.")
    learning_rq, learning_result = "", ""
    # Only add the dataset-size research question when that experiment was actually run.
    if (run / "learning_curve.csv").is_file():
        with (run / "learning_curve.csv").open(newline="", encoding="utf-8") as handle:
            curve = list(csv.DictReader(handle))
        learning_rq = "\\textbf{RQ4:} How does validation error change with the number of independent training trajectories?"
        learning_result = ("\\subsection{RQ4: Dataset Size}\n"
                           f"Nested training subsets range from {curve[0]['training_trajectories']} to "
                           f"{curve[-1]['training_trajectories']} trajectories per fold, with unchanged validation IDs. "
                           "Subset models fit their own normalizers; metric scales are shared within each full training fold. "
                           f"Mean validation $E$ changes from {number(float(curve[0]['mean_primary']))} to "
                           f"{number(float(curve[-1]['mean_primary']))}. The complete learning curve is saved with the artifacts.")
    timings = results["timings"]
    # Efficiency is optional and uses different CPU/GPU hardware and numerical precision.
    timing = ""
    if timings:
        timing = ("Optional batch-one timing includes host/device transfers and uses five warmups. "
                  + "; ".join(f"{latex(row['method'])}: {number(row['median_seconds_per_step'] * 1000)} ms"
                              for row in timings) + ". Models use float32 and the CPU solver uses float64.")
    repo = ("Source and reproducibility artifacts: \\url{" + args.repo_url.replace("}", "%7D").replace("{", "%7B") + "}."
            if args.repo_url else "The source repository URL must be added before submission.")
    replacements = {"AUTHORS": authors, "LEARNINGRQ": learning_rq, "DATASET": dataset, "SETUP": setup,
                    "CVTABLE": cvtable, "TESTTABLE": testtable, "COMPARISON": comparison,
                    "SENSITIVITY": sensitivity, "ROLLOUT": rollout, "LEARNINGRESULT": learning_result,
                    "TIMING": timing, "REPO": repo}
    source = (ROOT / "reports" / "evaluation_template.tex").read_text(encoding="utf-8")
    # Template markers such as @@DATASET@@ connect the calculated text to its paper section.
    for key, value in replacements.items():
        source = source.replace(f"@@{key}@@", value)
    if "@@" in source:
        raise ValueError("Report template contains unresolved placeholders.")
    if args.use_existing:
        # Once we manually edit a draft, compiling/packaging should keep those edits.
        if not (output / "evaluation.tex").is_file():
            raise FileNotFoundError("--use-existing requires an existing evaluation.tex in the report directory.")
    else:
        (output / "evaluation.tex").write_text(source, encoding="utf-8")
    (output / "REVIEW_BEFORE_SUBMISSION.txt").write_text(
        "Review the numerical interpretation, full author names/order, repository URL, citations, and dataset justification.\n"
        "The source is a measured-results draft, not a substitute for your discussion of the findings.\n"
        "Exactly two pages are required; compilation/packaging checks that page count.\n", encoding="utf-8")
    print(f"Generated measured-results LaTeX draft and figures: {output}")
    if args.compile or args.package:
        # Real LaTeX compilation requires pdflatex/acmart on the machine running this script.
        executable = shutil.which("pdflatex")
        if not executable:
            raise RuntimeError("pdflatex is unavailable. Sources were generated; compile using an ACM-capable TeX installation/Overleaf.")
        for _ in range(2):
            # A second pass resolves cross-references/citations left by the first pass.
            completed = subprocess.run([executable, "-interaction=nonstopmode", "-halt-on-error", "evaluation.tex"],
                                       cwd=output, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"LaTeX compilation failed; inspect {output / 'evaluation.log'}\n{completed.stdout[-2000:]}")
        log = (output / "evaluation.log").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Output written on .*?\((\d+) pages?", log, re.DOTALL)
        # The assignment requires exactly two pages; inspect the real compiler output to check.
        if not match or int(match.group(1)) != 2:
            raise RuntimeError("The compiled report is not exactly two pages. Edit/condense evaluation.tex before submission.")
        print("Compiled PDF verified: exactly two pages.")
    if args.package:
        # Only package after compilation/page-count checks succeed; include every figure used.
        archive = output / args.package_name
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
            for name in ("evaluation.tex", "evaluation.pdf", "hyperparameters.png", "rollout.png"):
                package.write(output / name, arcname=name)
        print(f"Submission archive: {archive}")
    return output


def main():
    # Command-line entry: choose a run, author details, and whether to write source or compile/zip.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--test-output")
    parser.add_argument("--output")
    parser.add_argument("--authors", default="Miles;Callum;Kai", help="Semicolon-separated full names in submission order.")
    parser.add_argument("--repo-url", default="")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use-existing", action="store_true", help="Preserve manual edits to evaluation.tex when compiling/packaging.")
    parser.add_argument("--package", action="store_true", help="Compile, verify two pages, then zip source/PDF/figures.")
    parser.add_argument("--package-name", default="4_Evaluation_Miles_Callum_Kai.zip")
    args = parser.parse_args()
    if args.repo_url and not args.repo_url.startswith("https://"):
        parser.error("Use an HTTPS repository URL.")
    if Path(args.package_name).name != args.package_name or not args.package_name.endswith(".zip"):
        parser.error("Package name must be a ZIP filename without a directory.")
    try:
        generate_report(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
