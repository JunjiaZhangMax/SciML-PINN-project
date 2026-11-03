"""Implement each scientific test, then remove its skip. No tests are implemented yet."""

import unittest


class ScientificAcceptance(unittest.TestCase):
    @unittest.skip("TODO: analytic derivatives, velocity signs and coordinate scaling")
    def test_analytic_derivatives(self):
        self.fail("Use analytic fields independent of the network implementation")

    @unittest.skip("TODO: momentum residuals and divergence against analytic expressions")
    def test_analytic_residuals(self):
        self.fail("Check f/g and continuity against independently derived expressions")

    @unittest.skip("TODO: gradients reach network and both learned PDE coefficients")
    def test_parameter_gradients(self):
        self.fail("Exercise an inverse-mode objective with nondegenerate inputs")

    @unittest.skip("TODO: flattening, holdout, paired noise, nesting, shared PDE points")
    def test_sampling_and_noise(self):
        self.fail("Use a synthetic cube with identifiable space/time values")

    @unittest.skip("TODO: pressure metric invariant to arbitrary additive c(t)")
    def test_pressure_gauge(self):
        self.fail("Also check undefined denominator and unsorted time groups")

    @unittest.skip("TODO: checkpoint reload/resume matches uninterrupted tiny run")
    def test_checkpoint_roundtrip(self):
        self.fail("Check predictions and optimizer state")

    @unittest.skip("TODO: summary preserves failures and excludes pilot/smoke runs")
    def test_summary_protocol(self):
        self.fail("Check sample SD, actual n, and failure reporting")
