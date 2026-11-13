"""Core training, direct loss calls, and evaluation."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat
import torch

from sciml._artifacts import load_checkpoint
from sciml.config import Config
from sciml.evaluate import evaluate
from sciml.losses import forward_objective, inverse_objective
from sciml.model import PINN
from sciml.train import prepare_training_data, train



class TrainingTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_adam(self):
        config = replace(self.config, adam_steps=4, checkpoint_every=2)
        full = train(replace(config, name="full"), self.data, self.root / "runs")
        original_step = torch.optim.Adam.step
        calls = 0

        def interrupt(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise KeyboardInterrupt
            return original_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.Adam, "step", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                train(config, self.data, self.root / "runs")
        run = self.root / "runs" / config.name
        self.assertEqual(load_checkpoint(run / "checkpoint.pt")["progress"]["adam_steps"], 2)
        train(config, self.data, self.root / "runs", resume=True)
        expected = load_checkpoint(full / "checkpoint.pt")["model"]
        actual = load_checkpoint(run / "checkpoint.pt")["model"]
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        xx, yy = np.meshgrid(np.linspace(1, 8, 5), np.linspace(-2, 2, 4))
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        times = np.array([0., .1, .2, .3])
        x, y, t = xy[:, 0, None], xy[:, 1, None], times[None, :]
        self.raw = {"X_star": xy, "t": times[:, None],
                    "U_star": np.stack((np.sin(x)+.1*y+.2*t, np.cos(y)+.1*x-.1*t), axis=1),
                    "p_star": x*y+.3*t}
        self.data = self.root / "reference.mat"
        savemat(self.data, self.raw)
        self.config = Config(name="run", device="cpu", purpose="smoke", adam_steps=3, lbfgs_steps=2,
                             hidden_layers=2, width=7, n_observations=6, n_collocation=5,
                             test_per_time=2, physics_batch_size=8, eval_batch_size=7,
                             log_every=1, checkpoint_every=1)

    def test_training_matches_one_direct_adam_step(self):
        for mode in ("known", "inverse"):
            with self.subTest(mode=mode):
                config = replace(self.config, name="direct_" + mode, mode=mode, adam_steps=1,
                                 w_initial=2, w_boundary=.5, w_observation=.7, w_residual=3, w_gauge=.3)
                prepared = prepare_training_data(config, self.data)
                torch.manual_seed(config.seed)
                model = PINN(prepared.samples.lower, prepared.samples.upper, config)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)
                optimizer.zero_grad()
                if mode == "known":
                    losses = forward_objective(model, **prepared.loss_inputs, **config.forward_loss_weights())
                else:
                    losses = inverse_objective(model, **prepared.loss_inputs, **config.inverse_loss_weights())
                losses["total"].backward()
                optimizer.step()
                run = train(config, self.data, self.root / "runs")
                saved = load_checkpoint(run / "checkpoint.pt")
                self.assertEqual(saved["progress"]["lbfgs_steps"], 0)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(saved["model"][name], value, rtol=0, atol=0)

    def test_both_modes_complete_and_evaluation_reloads_saved_indices(self):
        for mode in ("known", "inverse"):
            with self.subTest(mode=mode):
                config = replace(self.config, name=mode, mode=mode)
                run = train(config, self.data, self.root / "runs")
                for name in ("config.json", "samples.npz", "history.csv", "checkpoint.pt",
                             "metrics.json", "predictions.npz"):
                    self.assertTrue((run / name).exists())
                before = json.loads((run / "metrics.json").read_text())
                after = evaluate(run, self.data)
                for key in ("relative_l2_u", "relative_l2_v", "relative_l2_pressure_centered"):
                    self.assertEqual(before[key], after[key])
                with np.load(run / "samples.npz") as samples, np.load(run / "predictions.npz") as predictions:
                    np.testing.assert_array_equal(samples["test_indices"], predictions["test_indices"])
                with self.assertRaises(FileExistsError):
                    train(config, self.data, self.root / "runs")

    def test_status_records_completed_and_interrupted_runs(self):
        run = train(self.config, self.data, self.root / "runs")
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["state"], status["checkpoint_adam_steps"]), ("completed", 3))
        with patch.object(torch.optim.Adam, "step", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                train(replace(self.config, name="stopped"), self.data, self.root / "runs")
        status = json.loads((self.root / "runs" / "stopped" / "status.json").read_text())
        self.assertEqual((status["state"], status["checkpoint_adam_steps"]), ("interrupted", 0))

    def test_nonfinite_loss_stops_and_keeps_last_finite_checkpoint(self):
        config = replace(self.config, name="diverged", mode="inverse", adam_steps=6,
                         checkpoint_every=2, log_every=2)
        calls = 0

        def diverging(*args, **kwargs):
            nonlocal calls
            calls += 1
            losses = inverse_objective(*args, **kwargs)
            if calls >= 7:  # initial log, steps 1-3, logs at steps 1 and 2, then step 4
                losses["total"] = losses["total"] * float("nan")
            return losses

        with patch("sciml.train.inverse_objective", diverging):
            with self.assertRaises(FloatingPointError):
                train(config, self.data, self.root / "runs")
        run = self.root / "runs" / config.name
        status = json.loads((run / "status.json").read_text())
        self.assertEqual(status["state"], "failed")
        self.assertIn("Nonfinite loss at Adam step 4", status["reason"])
        saved = load_checkpoint(run / "checkpoint.pt")
        self.assertEqual(saved["progress"]["adam_steps"], 2)
        self.assertEqual(status["checkpoint_adam_steps"], 2)
        for value in saved["model"].values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertFalse((run / "metrics.json").exists())
        # The kept checkpoint is still a valid starting point.
        train(config, self.data, self.root / "runs", resume=True)
        self.assertEqual(json.loads((run / "status.json").read_text())["state"], "completed")

    def test_nonfinite_weights_are_never_checkpointed(self):
        config = replace(self.config, name="bad_weights", mode="inverse", adam_steps=2)
        original_step = torch.optim.Adam.step

        def corrupt(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("inf"))
            return result

        with patch.object(torch.optim.Adam, "step", corrupt):
            with self.assertRaises(FloatingPointError):
                train(config, self.data, self.root / "runs")
        run = self.root / "runs" / config.name
        self.assertEqual(load_checkpoint(run / "checkpoint.pt")["progress"]["adam_steps"], 0)
        self.assertEqual(json.loads((run / "status.json").read_text())["state"], "failed")

    def test_history_plot_is_written_for_both_modes(self):
        from sciml.plotting import plot_run
        for mode in ("known", "inverse"):
            with self.subTest(mode=mode):
                run = train(replace(self.config, name="plot_" + mode, mode=mode),
                            self.data, self.root / "runs")
                figure = plot_run(run, self.root / "figures")
                self.assertEqual(figure.name, f"plot_{mode}_history.png")
                self.assertGreater(figure.stat().st_size, 1000)

    def test_legacy_configs_keep_their_fingerprint_and_stay_adam_only(self):
        legacy = self.config.to_dict()
        for key in ("adam_lr_final", "use_lbfgs", "init_from"):
            legacy.pop(key)
        import hashlib
        expected = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
        self.assertEqual(self.config.fingerprint, expected)
        self.assertEqual(Config.from_dict(legacy).fingerprint, expected)
        for change in ({"adam_lr_final": 1e-4}, {"use_lbfgs": True}, {"init_from": "other"}):
            self.assertNotEqual(replace(self.config, **change).fingerprint, expected)
        # lbfgs_steps=2 without use_lbfgs is the pre-existing meaning: ignored.
        run = train(self.config, self.data, self.root / "runs")
        self.assertEqual(load_checkpoint(run / "checkpoint.pt")["progress"]["lbfgs_steps"], 0)

    def test_learning_rate_decays_geometrically_and_resume_is_exact(self):
        from sciml.train import adam_learning_rate
        config = replace(self.config, adam_steps=5, adam_lr=1e-2, adam_lr_final=1e-4, checkpoint_every=2)
        rates = [adam_learning_rate(config, step) for step in range(1, 6)]
        self.assertAlmostEqual(rates[0], 1e-2)
        self.assertAlmostEqual(rates[-1], 1e-4)
        for a, b in zip(rates, rates[1:]):
            self.assertAlmostEqual(b / a, rates[1] / rates[0])
        self.assertEqual(adam_learning_rate(self.config, 3), self.config.adam_lr)
        full = train(replace(config, name="decay_full"), self.data, self.root / "runs")
        constant = train(replace(config, name="constant", adam_lr_final=None), self.data, self.root / "runs")
        original_step = torch.optim.Adam.step
        calls = 0

        def interrupt(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise KeyboardInterrupt
            return original_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.Adam, "step", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                train(config, self.data, self.root / "runs")
        run = train(config, self.data, self.root / "runs", resume=True)
        expected = load_checkpoint(full / "checkpoint.pt")["model"]
        actual = load_checkpoint(run / "checkpoint.pt")["model"]
        other = load_checkpoint(constant / "checkpoint.pt")["model"]
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(expected["net.0.weight"], other["net.0.weight"]))

    def test_lbfgs_stage_lowers_loss_logs_and_resumes(self):
        for mode in ("known", "inverse"):
            with self.subTest(mode=mode):
                config = replace(self.config, name="lbfgs_" + mode, mode=mode, adam_steps=3,
                                 use_lbfgs=True, lbfgs_steps=6, log_every=2, checkpoint_every=1)
                run = train(config, self.data, self.root / "runs")
                with open(run / "history.csv", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                lbfgs_rows = [row for row in rows if row["stage"] == "lbfgs"]
                self.assertEqual([int(row["step"]) for row in lbfgs_rows], [5, 7, 9])
                adam_end = float([row for row in rows if row["stage"] == "adam"][-1]["total"])
                self.assertLess(float(lbfgs_rows[-1]["total"]), adam_end)
                progress = load_checkpoint(run / "checkpoint.pt")["progress"]
                self.assertEqual((progress["adam_steps"], progress["lbfgs_steps"]), (3, 6))
                metrics = json.loads((run / "metrics.json").read_text())
                self.assertEqual(metrics["lbfgs_steps"], 6)
                status = json.loads((run / "status.json").read_text())
                self.assertEqual((status["state"], status["checkpoint_lbfgs_steps"],
                                  status["target_lbfgs_steps"]), ("completed", 6, 6))
                from sciml.plotting import plot_run
                self.assertTrue(plot_run(run, self.root / "figures").exists())
        # Interrupt inside the second L-BFGS chunk; the first chunk's checkpoint resumes.
        config = replace(self.config, name="lbfgs_stopped", mode="inverse", adam_steps=3,
                         use_lbfgs=True, lbfgs_steps=6, log_every=2, checkpoint_every=1)
        original_step = torch.optim.LBFGS.step
        calls = 0

        def interrupt(optimizer, closure):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return original_step(optimizer, closure)

        with patch.object(torch.optim.LBFGS, "step", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                train(config, self.data, self.root / "runs")
        run = self.root / "runs" / config.name
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["state"], status["checkpoint_adam_steps"],
                          status["checkpoint_lbfgs_steps"]), ("interrupted", 3, 2))
        train(config, self.data, self.root / "runs", resume=True)
        progress = load_checkpoint(run / "checkpoint.pt")["progress"]
        self.assertEqual((progress["adam_steps"], progress["lbfgs_steps"]), (3, 6))
        with open(run / "history.csv", newline="") as stream:
            steps = [int(row["step"]) for row in csv.DictReader(stream)]
        self.assertEqual(steps, sorted(set(steps)))

    def test_lbfgs_stops_early_at_its_tolerance(self):
        config = replace(self.config, name="lbfgs_converged", mode="inverse", adam_steps=1,
                         use_lbfgs=True, lbfgs_steps=50, log_every=10)
        original = torch.optim.LBFGS.__init__

        def loose(optimizer, *args, **kwargs):
            kwargs["tolerance_grad"] = 1e9  # every gradient already counts as converged
            original(optimizer, *args, **kwargs)

        with patch.object(torch.optim.LBFGS, "__init__", loose):
            run = train(config, self.data, self.root / "runs")
        status = json.loads((run / "status.json").read_text())
        self.assertEqual(status["state"], "completed")
        self.assertIn("met its tolerance after 0 of 50", status["reason"])
        self.assertTrue((run / "metrics.json").exists())
        # A finished run is not optimized further by another resume.
        before = load_checkpoint(run / "checkpoint.pt")["model"]
        train(config, self.data, self.root / "runs", resume=True)
        after = load_checkpoint(run / "checkpoint.pt")["model"]
        for key in before:
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)

    def test_warm_start_continues_from_another_run(self):
        first = replace(self.config, name="stage1", mode="inverse", adam_steps=4)
        source = train(first, self.data, self.root / "runs")
        source_model = load_checkpoint(source / "checkpoint.pt")["model"]
        second = replace(first, name="stage2", init_from="stage1", adam_steps=2, lbfgs_steps=0)
        run = train(second, self.data, self.root / "runs")
        with open(run / "history.csv", newline="") as stream:
            initial = next(csv.DictReader(stream))
        # The new run's step-0 row is the source model, not a fresh initialization.
        self.assertEqual(float(initial["lambda1"]), source_model["lambda1"].item())
        checkpoint = load_checkpoint(run / "checkpoint.pt")
        self.assertEqual(checkpoint["init"]["run"], "stage1")
        self.assertEqual(checkpoint["init"]["adam_steps"], 4)
        self.assertEqual(json.loads((run / "metrics.json").read_text())["init"]["run"], "stage1")
        self.assertEqual(json.loads((run / "status.json").read_text())["init"]["adam_steps"], 4)
        # Constant-rate Adam continued through a warm start equals one longer run.
        long = train(replace(first, name="long", adam_steps=6), self.data, self.root / "runs")
        expected = load_checkpoint(long / "checkpoint.pt")["model"]
        for key in expected:
            torch.testing.assert_close(checkpoint["model"][key], expected[key], rtol=0, atol=0)
        # The source run itself is left untouched.
        for key in source_model:
            torch.testing.assert_close(load_checkpoint(source / "checkpoint.pt")["model"][key],
                                       source_model[key], rtol=0, atol=0)

    def test_warm_start_rejects_unusable_sources_without_creating_a_run(self):
        train(replace(self.config, name="stage1", mode="inverse"), self.data, self.root / "runs")
        base = replace(self.config, mode="inverse", init_from="stage1")
        cases = {"missing": (replace(base, name="a", init_from="absent"), FileNotFoundError),
                 "width": (replace(base, name="b", width=9), ValueError),
                 "mode": (replace(base, name="c", mode="known"), ValueError),
                 "window": (replace(base, name="d", time_end=0.2), ValueError)}
        for label, (config, error) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(error):
                    train(config, self.data, self.root / "runs")
                self.assertFalse((self.root / "runs" / config.name).exists())
        for bad in ("../stage1", "runs/stage1", ""):
            with self.assertRaises(ValueError):
                replace(base, name="e", init_from=bad).validate()
        with self.assertRaises(ValueError):
            replace(base, name="stage1").validate()

    def test_pressure_truth_does_not_change_training(self):
        config = replace(self.config, adam_steps=2, lbfgs_steps=0)
        first = train(config, self.data, self.root / "runs")
        changed = self.root / "different_pressure.mat"
        savemat(changed, {**self.raw, "p_star": self.raw["p_star"]*100+50})
        second = train(replace(config, name="different_pressure"), changed, self.root / "runs")
        a, b = load_checkpoint(first / "checkpoint.pt"), load_checkpoint(second / "checkpoint.pt")
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
