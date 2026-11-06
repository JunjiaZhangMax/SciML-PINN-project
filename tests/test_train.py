"""Core training, direct loss calls, and evaluation."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

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

    def test_pressure_truth_does_not_change_training(self):
        config = replace(self.config, adam_steps=2, lbfgs_steps=0)
        first = train(config, self.data, self.root / "runs")
        changed = self.root / "different_pressure.mat"
        savemat(changed, {**self.raw, "p_star": self.raw["p_star"]*100+50})
        second = train(replace(config, name="different_pressure"), changed, self.root / "runs")
        a, b = load_checkpoint(first / "checkpoint.pt"), load_checkpoint(second / "checkpoint.pt")
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
