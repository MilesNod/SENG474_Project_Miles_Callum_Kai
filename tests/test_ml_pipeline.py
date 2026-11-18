"""Regression checks for leakage, signed fields, fitting, checkpoints, and metrics."""

# Run with: python -m unittest discover -s tests -v
# These use tiny temporary datasets/checkpoints, not our actual training outputs.
# The point is to catch changes to the pipeline without needing another full GPU experiment.

import gc
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from argparse import Namespace
import zipfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fluid-model" / "src"))
from data import (FluidDataset, atomic_write, fit_normalization, generate_dataset,
                  make_splits, validate_splits)
from evaluate import rollouts
from experiment import ErrorTotals, capture_sources, fit_ridge, ridge_statistics, score, train_cnn
from model import FluidModel, load_predictor
from report import generate_report
from train import restore_run_arguments, run_experiment, run_lock, source_matches_saved
from scripts.run_experiment import main as run_pipeline
from src.simulation import SolverConfig


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # unittest calls this once for the class; share a tiny 6-run dataset to keep tests quick.
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        cls.store = generate_dataset(cls.path / "data", num_runs=6, num_steps=3, folds=2,
                                     config=SolverConfig(n_points=5, pressure_iterations=2), progress=False)

    @classmethod
    def tearDownClass(cls):
        # Close the memory map before removing temporary files; Windows keeps open files locked.
        cls.store.trajs._mmap.close()
        del cls.store
        gc.collect()
        cls.temp.cleanup()

    def test_trajectory_counts_independence_reproducibility_and_checksum(self):
        # Catch the old missing-run bug, identical starts, or generation that changes for the same seed.
        self.assertEqual(self.store.trajs.shape, (6, 4, 3, 5, 5))
        self.assertEqual(len(FluidDataset(self.store, [0, 1])), 6)
        self.assertFalse(np.array_equal(self.store.trajs[0], self.store.trajs[1]))
        with tempfile.TemporaryDirectory() as directory:
            duplicate = generate_dataset(Path(directory) / "data", num_runs=6, num_steps=3, folds=2,
                                         config=SolverConfig(n_points=5, pressure_iterations=2), progress=False)
            try:
                self.assertEqual(duplicate.metadata["dataset_id"], self.store.metadata["dataset_id"])
                duplicate.verify_content()
            finally:
                duplicate.trajs._mmap.close()

    def test_splits_are_disjoint_and_each_run_is_validated_once(self):
        # A good score is not useful if a trajectory appears in both training and validation/test.
        splits = make_splits(23, folds=5, seed=9)
        seen = []
        for fold in splits["folds"]:
            self.assertFalse(set(fold["train"]) & set(fold["validation"]))
            self.assertFalse(set(fold["train"] + fold["validation"]) & set(splits["test"]))
            seen.extend(fold["validation"])
        self.assertEqual(sorted(seen), sorted(splits["development"]))
        splits["folds"][0]["train"].append(splits["test"][0])
        with self.assertRaises(ValueError):
            validate_splits(splits, 23)

    def test_normalization_excludes_held_out_trajectories_and_last_target(self):
        # Huge held-out values would change the answer if the normalizer accidentally reads them.
        values = np.zeros((3, 3, 3, 4, 4), dtype=np.float32)
        values[0, 0] = 2
        values[0, 1] = 4
        values[0, 2] = 500
        values[1:] = 10000
        mean, std = fit_normalization(SimpleNamespace(trajs=values, num_steps=2), [0])
        np.testing.assert_allclose(mean, [3, 3, 3])
        np.testing.assert_allclose(std, [1, 1, 1])

    def test_training_noise_has_correct_shape_and_does_not_mutate_targets(self):
        # Augmentation should perturb only the returned input, not its target or the saved disk data.
        original = self.store.trajs[0].copy()
        noisy = FluidDataset(self.store, [0], noise=0.1, channel_std=[1, 2, 3])
        state, target, parameters = noisy[0]
        self.assertEqual(tuple(state.shape), (3, 5, 5))
        self.assertEqual(tuple(parameters.shape), (2,))
        self.assertFalse(np.array_equal(state.numpy(), original[0]))
        np.testing.assert_array_equal(target.numpy(), original[1])
        np.testing.assert_array_equal(self.store.trajs[0], original)
        with self.assertRaises(ValueError):
            FluidDataset(self.store, [0], noise=0.1)

    def test_signed_predictions_dynamic_grid_and_physical_inputs(self):
        # Negative fluid values must survive the output; the model should not assume a 41x41 grid.
        model = FluidModel(hidden_dim=4)
        state = torch.full((2, 3, 7, 9), -2.0)
        parameters = torch.tensor([[0.05, -1.0], [0.1, 1.0]])
        torch.testing.assert_close(model(state, parameters), state)
        features = model.features(state, parameters)
        self.assertFalse(torch.equal(features[0, 5:], features[1, 5:]))
        with self.assertRaises(ValueError):
            model(state, torch.zeros(2, 3))

    def test_checkpoint_round_trip_includes_normalization_and_configuration(self):
        # A saved model needs its architecture/stats too, otherwise loading weights can change predictions.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = FluidModel(hidden_dim=5, mean=[1, 2, 3], std=[2, 3, 4])
            model.save(path, {"epoch": 7, "train_ids": [1, 2]})
            restored, metadata = load_predictor(path)
            self.assertEqual(metadata["epoch"], 7)
            self.assertEqual(restored.architecture["hidden_dim"], 5)
            torch.testing.assert_close(model.field_std, restored.field_std)
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "Legacy"):
                load_predictor(path)

    def test_errors_use_pixel_weighting_and_training_channel_scales(self):
        # Choose known errors so the primary metric should be exactly 1 after normalization.
        totals = ErrorTotals([1, 2, 4])
        expected = np.zeros((3, 3, 2, 2))
        predicted = np.ones_like(expected) * np.array([1, 2, 4])[None, :, None, None]
        totals.update(predicted[:2], expected[:2])
        totals.update(predicted[2:], expected[2:])
        result = totals.finish()
        np.testing.assert_allclose(result["rmse"], [1, 2, 4])
        self.assertEqual(result["primary"], 1.0)
        with self.assertRaises(FloatingPointError):
            totals.update(np.full_like(expected, np.inf), expected)

    def test_scoring_does_not_depend_on_batch_partition(self):
        # Changing inference batch size must not change how much weight the last samples get.
        ids = self.store.splits["folds"][0]["validation"]
        _, std = fit_normalization(self.store, self.store.splits["folds"][0]["train"])
        first = score(None, self.store, ids, std, "cpu", 1)
        second = score(None, self.store, ids, std, "cpu", 4)
        self.assertAlmostEqual(first["primary"], second["primary"], places=12)
        self.assertEqual(len(first["per_trajectory"]), len(ids))

    def test_best_checkpoint_is_selected_by_validation_not_training_loss(self):
        # Force validation to get worse in epoch 2 and check that epoch 1's weights are still selected.
        fold = self.store.splits["folds"][0]
        with tempfile.TemporaryDirectory() as directory:
            with patch("experiment.score", side_effect=[{"primary": 0.1}, {"primary": 0.9}, {"primary": 0.1}]):
                result = train_cnn(self.store, fold["train"], fold["validation"],
                                   {"hidden_dim": 3, "learning_rate": 0.001}, 2, 4,
                                   "cpu", 8, directory, quiet=True)
            _, metadata = load_predictor(Path(directory) / "best.pt")
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(metadata["epoch"], 1)
            self.assertEqual(metadata["validation_primary"], 0.1)

    def test_ridge_learns_a_signed_linear_update_and_reloads(self):
        # Use a known linear next-state rule to check the regression solve and stencil reshaping.
        rng = np.random.default_rng(1)
        values = rng.normal(size=(12, 2, 3, 7, 7)).astype(np.float32)
        values[:, 1] = -0.5 * values[:, 0] - 0.2
        fake = SimpleNamespace(trajs=values, num_steps=1,
                               parameters=np.tile([0.1, 1.0], (12, 1)).astype(np.float32))
        statistics = ridge_statistics(fake, list(range(12)), 2, max_points=588)
        model = fit_ridge(statistics, 1e-8)
        state = torch.from_numpy(rng.normal(size=(2, 3, 7, 7)).astype(np.float32))
        parameters = torch.tensor([[0.1, 1.0], [0.1, 1.0]])
        with torch.inference_mode():
            prediction = model(state, parameters)
        self.assertLess(float(torch.square(prediction - (-0.5 * state - 0.2)).mean()), 1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ridge.pt"
            model.save(path)
            restored, _ = load_predictor(path)
            with torch.inference_mode():
                torch.testing.assert_close(restored(state, parameters), prediction)

    def test_failed_rollout_is_reported_without_dropping_trajectories(self):
        # Deliberately return infinity and check that evaluation records failure rather than a good average.
        class Nonfinite(torch.nn.Module):
            def forward(self, state, parameters):
                # Test-only broken predictor; the real models do not intentionally return infinity.
                return torch.full_like(state, float("inf"))
        with tempfile.TemporaryDirectory() as directory:
            rows, comparison = rollouts({"persistence": None, "cnn": Nonfinite()}, self.store,
                                       [0, 1], np.ones(3), 0, [1, 2], "cpu", 2, Path(directory))
            failures = [r for r in rows if r["model"] == "cnn"]
            self.assertTrue(all(r["status"] == "nonfinite" and r["primary"] is None for r in failures))
            self.assertTrue(all(r["first_nonfinite_step"] == 1 for r in failures))
            self.assertFalse(comparison)

    def test_generation_does_not_overwrite_an_existing_dataset(self):
        # Repeating the generator command should protect the data already used by an experiment.
        with self.assertRaises(FileExistsError):
            generate_dataset(self.store.path, num_runs=6, num_steps=3, folds=2, progress=False)

    def test_source_snapshot_preserves_the_code_used_by_a_run(self):
        # Copied source bytes must match the hashes recorded for reproducing that run.
        with tempfile.TemporaryDirectory() as directory:
            hashes = capture_sources(directory)
            self.assertIn("fluid-model/src/train.py", hashes)
            for path, expected in hashes.items():
                self.assertEqual(hashlib.sha256((Path(directory) / "source" / path).read_bytes()).hexdigest(), expected)

    @patch("builtins.print")
    def test_interrupted_training_resumes_optimizer_rng_and_shuffle_exactly(self, _print):
        # Compare uninterrupted training against interruption + resume, including optional noise.
        # Matching weights/history checks more than just loading a checkpoint without crashing.
        fold = self.store.splits["folds"][0]
        config = {"hidden_dim": 3, "learning_rate": 0.001}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_cnn(self.store, fold["train"], fold["validation"], config, 3, 2, "cpu", 19,
                      root / "full", patience=0, noise=0.05, quiet=True)
            # Interrupt after the complete epoch state is committed, before best/latest/history.
            with patch("experiment.publish_training_state", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    train_cnn(self.store, fold["train"], fold["validation"], config, 3, 2, "cpu", 19,
                              root / "interrupted", patience=0, noise=0.05, quiet=True)
            self.assertFalse((root / "interrupted" / "best.pt").exists())
            state = torch.load(root / "interrupted" / "training_state.pt", weights_only=True)
            self.assertEqual(state["epoch"], 1)
            self.assertTrue(state["optimizer"]["state"])
            train_cnn(self.store, fold["train"], fold["validation"], config, 3, 2, "cpu", 19,
                      root / "interrupted", patience=0, noise=0.05, quiet=True, resume=True)
            full, _ = load_predictor(root / "full" / "latest.pt")
            resumed, _ = load_predictor(root / "interrupted" / "latest.pt")
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            self.assertEqual((root / "full" / "history.csv").read_text(),
                             (root / "interrupted" / "history.csv").read_text())

    @patch("builtins.print")
    def test_completed_fit_is_reused_and_changed_settings_are_rejected(self, _print):
        # Reuse must not call Adam again, change the completed checkpoint, or accept a new epoch budget.
        fold = self.store.splits["folds"][0]
        config = {"hidden_dim": 3, "learning_rate": 0.001}
        with tempfile.TemporaryDirectory() as directory:
            result = train_cnn(self.store, fold["train"], fold["validation"], config, 1, 4,
                               "cpu", 9, directory, quiet=True)
            original = (Path(directory) / "best.pt").read_bytes()
            with patch("torch.optim.Adam.step", side_effect=AssertionError("Completed fit was retrained")):
                cached = train_cnn(self.store, fold["train"], fold["validation"], config, 1, 4,
                                   "cpu", 9, directory, quiet=True, resume=True)
            self.assertEqual(result, cached)
            self.assertEqual(original, (Path(directory) / "best.pt").read_bytes())
            with self.assertRaisesRegex(ValueError, "different epochs"):
                train_cnn(self.store, fold["train"], fold["validation"], config, 2, 4,
                          "cpu", 9, directory, quiet=True, resume=True)

    @patch("builtins.print")
    def test_older_partial_fit_is_backed_up_before_restarting(self, _print):
        # Mimic old weight-only checkpoints: preserve the old files before the one partial fit restarts.
        fold = self.store.splits["folds"][0]
        config = {"hidden_dim": 3, "learning_rate": 0.001}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            train_cnn(self.store, fold["train"], fold["validation"], config, 1, 4,
                      "cpu", 9, output, quiet=True)
            original = (output / "latest.pt").read_bytes()
            (output / "fit.json").unlink()
            (output / "training_state.pt").unlink()
            train_cnn(self.store, fold["train"], fold["validation"], config, 1, 4,
                      "cpu", 9, output, quiet=True, resume=True)
            backups = list((output / "legacy_backups").glob("*/latest.pt"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_resume_restores_original_search_and_rejects_changed_options(self):
        # CLI defaults may differ from the old run; --resume must recover the saved research settings.
        with tempfile.TemporaryDirectory() as directory:
            saved = {"arguments": {"epochs": 50, "learning_rates": [0.0003, 0.001],
                                    "learning_curve_sizes": [25, 50, 100]}, "data_path": "saved/data"}
            (Path(directory) / "run.json").write_text(json.dumps(saved), encoding="utf-8")
            args = Namespace(resume=True, output=directory, data="default/data", epochs=50,
                             learning_rates=[0.001], learning_curve_sizes=[])
            restored = restore_run_arguments(args, {"--resume"})
            self.assertEqual(restored.learning_rates, [0.0003, 0.001])
            self.assertEqual(restored.learning_curve_sizes, [25, 50, 100])
            self.assertEqual(restored.data, "saved/data")
            restored.learning_rates = [0.003]
            with self.assertRaisesRegex(ValueError, "saved learning_rates"):
                restore_run_arguments(restored, {"--resume", "--learning-rates"})

    def test_source_check_accepts_comments_but_rejects_logic_or_snapshot_changes(self):
        # Comments should not force retraining, but changed calculations or an altered archive should fail.
        with tempfile.TemporaryDirectory() as directory:
            recorded = Path(directory) / "recorded.py"
            current = Path(directory) / "current.py"
            original = b"def predict(x):\n    return x + 1\n"
            recorded.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            current.write_text("# explanation\ndef predict(x):\n    # same calculation\n    return x+1\n", encoding="utf-8")
            self.assertTrue(source_matches_saved(current, recorded, expected))
            current.write_text("def predict(x):\n    return x + 2\n", encoding="utf-8")
            self.assertFalse(source_matches_saved(current, recorded, expected))
            current.write_text("# explanation\ndef predict(x):\n    return x + 1\n", encoding="utf-8")
            recorded.write_bytes(original + b"# archive was altered\n")
            self.assertFalse(source_matches_saved(current, recorded, expected))
            recorded.write_bytes(original)
            current.write_text("def predict(:\n", encoding="utf-8")
            self.assertFalse(source_matches_saved(current, recorded, expected))

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        # An interrupted write should leave the previous complete file and no half-written replacement.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.json"
            target.write_text("previous complete checkpoint", encoding="utf-8")

            def interrupted(pending):
                # Simulate the write failing after only part of the new file reached disk.
                pending.write_text("partial new checkpoint", encoding="utf-8")
                raise OSError("Interrupted write")

            with self.assertRaises(OSError):
                atomic_write(target, interrupted)
            self.assertEqual(target.read_text(), "previous complete checkpoint")
            self.assertEqual([p.name for p in Path(directory).iterdir()], [target.name])

    def test_run_lock_prevents_concurrent_writers_and_releases(self):
        # A second training process should be refused; after release a later resume must be allowed.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with run_lock(output):
                with self.assertRaisesRegex(ValueError, "Another training process"):
                    with run_lock(output):
                        self.fail("A second writer acquired the lock")
            with run_lock(output):
                pass

    @patch("builtins.print")
    def test_pipeline_stops_after_a_failed_training_stage(self, _print):
        # This is the pasted-PowerShell-commands failure case: evaluation/report must not run after a failure.
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run.json").write_text(json.dumps({"data_path": str(self.store.path)}), encoding="utf-8")
            with patch("scripts.run_experiment.subprocess.run", side_effect=subprocess.CalledProcessError(2, "train")) as command:
                self.assertEqual(run_pipeline(["--resume", "--run", str(run)]), 2)
            self.assertEqual(command.call_count, 1)
            self.assertTrue(any(str(arg).endswith("train.py") for arg in command.call_args.args[0]))

    @patch("builtins.print")
    def test_run_resume_recovers_fit_missing_from_cv_results_without_duplicates(self, _print):
        # Interrupt after a fit finishes but before its result row saves; recover from its fit.json.
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(data=str(self.store.path), output=directory, epochs=1, patience=0,
                             batch_size=4, hidden_dims=[3], learning_rates=[0.001], ridge_alphas=[0.1],
                             ridge_max_points=50, learning_curve_sizes=[], noise=0.0, seed=42,
                             device="cpu", threads=2, verify_data=False, quiet=True, resume=False, dry_run=False)
            from data import write_json as real_write

            def interrupt_before_record(path, value):
                # Leave the trained checkpoint intact but stop the orchestrator before recording its score.
                if Path(path).name == "cv_results.json" and any(row["kind"] == "cnn" for row in value):
                    raise KeyboardInterrupt
                real_write(path, value)

            with patch("train.write_json", side_effect=interrupt_before_record):
                with self.assertRaises(KeyboardInterrupt):
                    run_experiment(args)
            checkpoint = Path(directory) / "cv" / "cnn_h3_lr0.001" / "fold_1" / "best.pt"
            before = checkpoint.read_bytes()
            self.assertEqual(len(json.loads((Path(directory) / "cv_results.json").read_text())), 1)
            args.resume = True
            result = run_experiment(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(checkpoint.read_bytes(), before)
            rows = json.loads((Path(directory) / "cv_results.json").read_text())
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({(row["fold"], row["candidate"]) for row in rows}), 6)

    @patch("builtins.print")
    def test_report_uses_measurements_and_checks_two_pages_before_packaging(self, _print):
        # Use tiny fake measurements and a mocked compiler; no real pdflatex installation is needed.
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            test = run / "test"
            test.mkdir()
            cnn = {"kind": "cnn", "mean_primary": 0.4, "std_primary": 0.02,
                   "config": {"hidden_dim": 8, "learning_rate": 0.001}}
            ridge = {"kind": "ridge", "mean_primary": 0.3, "std_primary": 0.03, "config": {"alpha": 0.1}}
            info = {"status": "complete", "dataset_id": self.store.metadata["dataset_id"],
                    "splits": self.store.splits, "selected": {"cnn": cnn, "ridge": ridge}, "final_epochs": 2,
                    "arguments": {"hidden_dims": [8], "learning_rates": [0.001], "batch_size": 4,
                                  "epochs": 2, "patience": 1, "noise": 0, "ridge_alphas": [0.1], "ridge_max_points": 100}}
            metric = {"rmse": [1.25, 2.5, 3.75], "primary": 0.45}
            results = {"dataset_id": info["dataset_id"], "test_ids": info["splits"]["test"],
                       "one_step": {"persistence": metric, "cnn": metric, "ridge": metric},
                       "rollouts": [{"model": "cnn", "status": "finite", "horizon": 1, "primary": 0.6}],
                       "arguments": {"rollout_start": 0}, "timings": []}
            fixtures = {"run.json": info, "dataset_metadata.json": self.store.metadata,
                        "cv_summary.json": [{"kind": "persistence", "mean_primary": 0.5, "std_primary": 0.01}, cnn, ridge],
                        "test/results.json": results}
            for path, value in fixtures.items():
                (run / path).write_text(json.dumps(value), encoding="utf-8")
            (run / "hyperparameters.png").write_bytes(b"figure placeholder")
            (test / "rollout.png").write_bytes(b"figure placeholder")
            args = SimpleNamespace(run=str(run), test_output=None, output=None, authors="A & B;C",
                                   repo_url="https://example.com/repo", compile=False, package=False,
                                   use_existing=False, package_name="submission.zip")
            output = generate_report(args)
            source = output / "evaluation.tex"
            content = source.read_text(encoding="utf-8")
            self.assertNotIn("@@", content)
            self.assertIn("1.25", content)
            self.assertIn(r"A \& B", content)
            source.write_text(content + "\n% manual edit\n", encoding="utf-8")
            args.package, args.use_existing = True, True

            def compile_result(*unused, **kwargs):
                # Pretend compilation succeeded with 3 pages, which must still block submission packaging.
                (output / "evaluation.pdf").write_bytes(b"mock compiled PDF")
                (output / "evaluation.log").write_text("Output written on evaluation.pdf (3 pages, 123 bytes).", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="")

            with patch("report.shutil.which", return_value="pdflatex"), patch("report.subprocess.run", side_effect=compile_result):
                with self.assertRaisesRegex(RuntimeError, "not exactly two pages"):
                    generate_report(args)
            self.assertFalse((output / "submission.zip").exists())
            self.assertIn("% manual edit", source.read_text(encoding="utf-8"))

            def two_page_result(*unused, **kwargs):
                # Now report 2 pages so the same packaging code can include the expected source/figures/PDF.
                (output / "evaluation.log").write_text("Output written on evaluation.pdf (2 pages, 123 bytes).", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="")

            with patch("report.shutil.which", return_value="pdflatex"), patch("report.subprocess.run", side_effect=two_page_result):
                generate_report(args)
            with zipfile.ZipFile(output / "submission.zip") as package:
                self.assertEqual(set(package.namelist()), {"evaluation.tex", "evaluation.pdf", "hyperparameters.png", "rollout.png"})


if __name__ == "__main__":
    unittest.main()
