"""Core training, direct loss calls, and evaluation."""

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

    def test_pressure_truth_does_not_change_training(self):
        config = replace(self.config, adam_steps=2, lbfgs_steps=0)
        first = train(config, self.data, self.root / "runs")
        changed = self.root / "different_pressure.mat"
        savemat(changed, {**self.raw, "p_star": self.raw["p_star"]*100+50})
        second = train(replace(config, name="different_pressure"), changed, self.root / "runs")
        a, b = load_checkpoint(first / "checkpoint.pt"), load_checkpoint(second / "checkpoint.pt")
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
