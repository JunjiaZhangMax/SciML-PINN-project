"""Pressure metrics must respect the time-dependent gauge ambiguity."""

import unittest

import numpy as np

from sciml.evaluate import centered_pressure, pressure_error, relative_l2


class EvaluationTests(unittest.TestCase):
    def test_pressure_metric_ignores_per_time_offsets_in_unsorted_groups(self):
        times = np.array([0, 1, 0, 1, 2, 2])
        reference = np.array([1., 10., 3., 14., -2., 2.])
        prediction = reference + np.array([100., 50., 100., 50., -20., -20.])
        self.assertAlmostEqual(pressure_error(prediction, reference, times), 0.0)
        prediction[0] += .5
        self.assertAlmostEqual(pressure_error(prediction, reference, times), 1/12)
        centered = centered_pressure(prediction, times)
        for group in np.unique(times):
            self.assertAlmostEqual(centered[times == group].mean(), 0.0)

    def test_zero_reference_norm_is_undefined(self):
        self.assertIsNone(relative_l2(np.ones(3), np.zeros(3)))
        self.assertIsNone(pressure_error(np.array([1., 2.]), np.ones(2), np.zeros(2)))

    def test_invalid_values_or_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            relative_l2(np.ones(2), np.ones((2, 1)))
        with self.assertRaises(ValueError):
            relative_l2(np.array([float("nan")]), np.ones(1))
        with self.assertRaises(ValueError):
            centered_pressure(np.ones(3), np.zeros(2))
