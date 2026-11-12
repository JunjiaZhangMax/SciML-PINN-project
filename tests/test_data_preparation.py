"""Synthetic indexed data verifies labels, geometry, leakage, and loss integration."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat
import torch

from sciml.config import Config
from sciml.data import WakeData, coordinates_at, load_data, prepare_inverse_data as prepare_data, prepare_forward_data, velocities_at
from sciml.losses import forward_objective, inverse_objective
from sciml.model import PINN
from sciml.train import prepare_training_data


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        xx, yy = np.meshgrid(np.linspace(1, 8, 5), np.linspace(-2, 2, 4))
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        times = np.array([0.0, 0.1, 0.2, 0.3])
        s, t = np.arange(len(xy))[:, None], np.arange(len(times))[None, :]
        velocity = np.stack((100*s + t, -10*s - 2*t), axis=1).astype(np.float64)
        self.data = WakeData(xy, times, velocity, np.zeros((len(xy), len(times))))
        self.config = Config(device="cpu", n_observations=4, n_collocation=5,
                             test_per_time=1, hidden_layers=2, width=7)

    def test_forward_labels_geometry_and_disjoint_test_set(self):
        prepared = prepare_forward_data(self.data, self.config)
        self.assertEqual(prepared.initial_coordinates.shape, (20, 3))
        self.assertTrue(np.all(prepared.initial_coordinates[:, 2] == 0))
        np.testing.assert_array_equal(prepared.initial_velocity, self.data.velocity[:, :, 0])
        self.assertEqual(prepared.boundary_coordinates.shape, (14*3, 3))
        self.assertEqual(len(np.unique(prepared.boundary_indices)), 14*3)
        xy = prepared.boundary_coordinates[:, :2]
        self.assertTrue(np.all((xy[:, 0] == 1) | (xy[:, 0] == 8)
                               | (xy[:, 1] == -2) | (xy[:, 1] == 2)))
        self.assertTrue(np.all(prepared.boundary_coordinates[:, 2] > 0))
        np.testing.assert_array_equal(prepared.boundary_velocity,
                                      velocities_at(self.data, prepared.boundary_indices))
        collocation = prepared.collocation_coordinates
        self.assertTrue(np.all((collocation[:, 0] > 1) & (collocation[:, 0] < 8)
                               & (collocation[:, 1] > -2) & (collocation[:, 1] < 2)
                               & (collocation[:, 2] > 0)))
        for key in ("initial_indices", "boundary_indices", "collocation_indices"):
            self.assertEqual(np.intersect1d(getattr(prepared, key), prepared.test_indices).size, 0)
        self.assertEqual(len(prepared.test_indices), 3)
        np.testing.assert_array_equal(prepared.gauge_coordinates[:, :2], np.tile([4.5, 0], (4, 1)))
        np.testing.assert_array_equal(prepared.gauge_coordinates[:, 2], self.data.times)
        changed = replace(self.data, pressure=np.full_like(self.data.pressure, 999))
        other = prepare_forward_data(changed, self.config)
        np.testing.assert_array_equal(other.initial_velocity, prepared.initial_velocity)
        np.testing.assert_array_equal(other.boundary_velocity, prepared.boundary_velocity)

    def test_time_window_is_selected_before_sampling(self):
        config = replace(self.config, time_end=0.2)
        prepared = prepare_forward_data(self.data, config)
        np.testing.assert_array_equal(prepared.time_indices, [0, 1, 2])
        self.assertEqual(prepared.upper[2], 0.2)
        for a in (prepared.boundary_coordinates, prepared.collocation_coordinates, prepared.gauge_coordinates):
            self.assertTrue(np.all(a[:, 2] <= 0.2))
        self.assertEqual(prepared.boundary_coordinates.shape, (14*2, 3))

    def test_inverse_pairing_nesting_and_no_label_leakage(self):
        config = replace(self.config, mode="inverse", collocation="fixed", noise=0.05)
        small = prepare_data(self.data, config)
        large = prepare_data(self.data, replace(config, n_observations=8))
        np.testing.assert_array_equal(small.observation_indices, large.observation_indices[:4])
        np.testing.assert_array_equal(small.base_noise, large.base_noise[:4])
        np.testing.assert_array_equal(small.collocation_indices, large.collocation_indices)
        np.testing.assert_array_equal(small.test_indices, large.test_indices)
        np.testing.assert_array_equal(small.observation_coordinates,
                                      coordinates_at(self.data, small.observation_indices))
        np.testing.assert_array_equal(small.clean_velocity, velocities_at(self.data, small.observation_indices))
        np.testing.assert_allclose(small.observed_velocity,
                                   small.clean_velocity + .05*small.clean_velocity.std(0)*small.base_noise)
        noisy = prepare_data(self.data, replace(config, noise=0.10))
        np.testing.assert_array_equal(noisy.observation_indices, small.observation_indices)
        np.testing.assert_array_equal(noisy.base_noise, small.base_noise)
        np.testing.assert_allclose(noisy.observed_velocity-small.clean_velocity,
                                   2*(small.observed_velocity-small.clean_velocity))
        self.assertEqual(np.intersect1d(small.test_indices, small.observation_indices).size, 0)
        self.assertEqual(np.intersect1d(small.test_indices, small.collocation_indices).size, 0)
        at_observations = prepare_data(self.data, replace(config, collocation="observations"))
        np.testing.assert_array_equal(at_observations.collocation_indices, at_observations.observation_indices)

    def test_mat_loading_tensor_preparation_and_real_losses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"reference.mat"
            savemat(path, {"X_star": self.data.xy, "t": self.data.times[:, None],
                           "U_star": self.data.velocity, "p_star": self.data.pressure})
            loaded = load_data(path)
            np.testing.assert_array_equal(loaded.velocity, self.data.velocity)
            for mode in ("known", "inverse"):
                with self.subTest(mode=mode):
                    config = replace(self.config, mode=mode)
                    prepared = prepare_training_data(config, path)
                    self.assertFalse(hasattr(prepared.samples, "pressure"))
                    for name, value in prepared.loss_inputs.items():
                        self.assertEqual(value.dtype, torch.float64)
                        self.assertEqual(value.device.type, "cpu")
                        self.assertEqual(value.requires_grad,
                                         name.endswith("_coordinates") and name != "gauge_coordinates")
                    model = PINN(prepared.samples.lower, prepared.samples.upper, config)
                    if mode == "known":
                        losses = forward_objective(model, **prepared.loss_inputs, **config.forward_loss_weights())
                    else:
                        losses = inverse_objective(model, **prepared.loss_inputs, **config.inverse_loss_weights())
                    self.assertTrue(torch.isfinite(losses["total"]).item())
                    losses["total"].backward()
                    self.assertIsNotNone(model.net[0].weight.grad)

    def test_download_verifies_hash_and_never_overwrites(self):
        import hashlib
        import json
        from sciml.data import download
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mat"
            savemat(source, {"X_star": self.data.xy, "t": self.data.times[:, None],
                             "U_star": self.data.velocity, "p_star": self.data.pressure})
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = root / "raw" / "wake.mat"
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                download(target, source.as_uri(), expected_sha256="0" * 64)
            self.assertEqual(list(target.parent.iterdir()), [])
            self.assertEqual(download(target, source.as_uri(), expected_sha256=digest), target)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            record = json.loads(target.with_suffix(".source.json").read_text())
            self.assertEqual((record["sha256"], record["url"]), (digest, source.as_uri()))
            # A second call only re-verifies; an unreachable URL proves nothing is fetched.
            download(target, "file:///does/not/exist.mat", expected_sha256=digest)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                download(target, source.as_uri(), expected_sha256="0" * 64)
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_invalid_window_or_oversampling_is_rejected(self):
        for updates in ({"time_end": 0.0}, {"time_end": 1.0}, {"test_per_time": 6},
                        {"n_collocation": 99999}, {"noise": .05}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                prepare_forward_data(self.data, replace(self.config, **updates))
