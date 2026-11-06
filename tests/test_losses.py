"""Independent analytic checks for forward and observation-based losses."""

import unittest

import torch
from torch import nn

from sciml.config import Config
from sciml.losses import forward_objective, objective, physics_loss, pressure_gauge_loss, velocity_loss
from sciml.model import PINN


class AnalyticFlow(nn.Module):
    """psi=x^3*y^3+t*x*y, p=x^2*y+t*y^2+2; derivatives are known by hand."""

    def __init__(self):
        super().__init__()
        self.register_buffer("lambda1", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("lambda2", torch.tensor(0.01, dtype=torch.float64))

    def forward(self, coordinates):
        x, y, t = coordinates.unbind(1)
        return torch.stack((x**3 * y**3 + t*x*y, x**2*y + t*y**2 + 2), dim=1)


def analytic_velocity(coordinates):
    x, y, t = coordinates.unbind(1)
    return torch.stack((3*x**3*y**2 + t*x, -3*x**2*y**3 - t*y), dim=1)


def analytic_residual_loss(coordinates):
    x, y, t = coordinates.unbind(1)
    u, v = analytic_velocity(coordinates).unbind(1)
    f = (x + u*(9*x**2*y**2 + t) + v*(6*x**3*y) + 2*x*y
         - 0.01*(18*x*y**2 + 6*x**3))
    g = (-y + u*(-6*x*y**3) + v*(-9*x**2*y**2 - t) + x**2 + 2*t*y
         - 0.01*(-6*y**3 - 18*x**2*y))
    return (f.square() + g.square()).mean()


class LossTests(unittest.TestCase):
    def setUp(self):
        self.model = AnalyticFlow()
        self.initial = torch.tensor([[0.6, 0.7, 0.0], [1.2, 0.4, 0.0]], dtype=torch.float64)
        self.boundary = torch.tensor([[1.0, 0.6, 0.2], [1.0, 0.8, 0.5]], dtype=torch.float64)
        self.collocation = torch.tensor([[0.4, 0.9, 0.3], [0.8, 0.5, 0.7], [1.1, 0.6, 0.2]], dtype=torch.float64)
        self.gauge = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.4]], dtype=torch.float64)

    def test_forward_components_and_weighted_total(self):
        initial_targets = analytic_velocity(self.initial) + torch.tensor([1.0, 2.0])
        boundary_targets = analytic_velocity(self.boundary) + torch.tensor([2.0, 3.0])
        config = Config(w_initial=1.5, w_boundary=0.5, w_residual=2, w_gauge=3)
        result = forward_objective(self.model, self.initial, initial_targets,
                                   self.boundary, boundary_targets, self.collocation, self.gauge,
                                   **config.forward_loss_weights())
        self.assertAlmostEqual(result["initial"].item(), 5.0)
        self.assertAlmostEqual(result["boundary"].item(), 13.0)
        # At (x_ref, y_ref)=(1, 0), p=2 for every sampled time, hence mean(p^2)=4.
        self.assertAlmostEqual(result["gauge"].item(), 4.0)
        expected_residual = analytic_residual_loss(self.collocation)
        torch.testing.assert_close(result["residual"], expected_residual)
        torch.testing.assert_close(result["total"], 1.5*5 + 0.5*13 + 2*expected_residual + 3*4)

    def test_inverse_uses_collocation_not_observation_coordinates(self):
        targets = analytic_velocity(self.initial) + torch.tensor([1.0, 2.0])
        result = objective(self.model, self.initial, targets, self.collocation)
        expected = analytic_residual_loss(self.collocation)
        torch.testing.assert_close(result["residual"], expected)
        torch.testing.assert_close(result["total"], expected + 5)
        self.assertEqual(result["initial"].item(), 0.0)
        self.assertEqual(result["boundary"].item(), 0.0)
        other = physics_loss(self.model, self.boundary)
        self.assertGreater(abs(other.item() - result["residual"].item()), 1e-4)

    def test_optional_initial_and_boundary_conditions(self):
        result = objective(self.model, self.initial, analytic_velocity(self.initial),
                           self.collocation, self.initial, analytic_velocity(self.initial),
                           self.boundary, analytic_velocity(self.boundary))
        self.assertAlmostEqual(result["observation"].item(), 0.0)
        self.assertAlmostEqual(result["initial"].item(), 0.0)
        self.assertAlmostEqual(result["boundary"].item(), 0.0)
        torch.testing.assert_close(result["total"], analytic_residual_loss(self.collocation))

    def test_configured_inverse_weights_apply_to_all_terms(self):
        config = Config(w_observation=2, w_initial=3, w_boundary=4, w_residual=0.5)
        observed = analytic_velocity(self.initial) + torch.tensor([1.0, 2.0])
        initial = analytic_velocity(self.initial) + torch.tensor([2.0, 3.0])
        boundary = analytic_velocity(self.boundary) + torch.tensor([3.0, 4.0])
        result = objective(self.model, self.initial, observed, self.collocation,
                           self.initial, initial, self.boundary, boundary,
                           **config.inverse_loss_weights())
        self.assertAlmostEqual(result["observation"].item(), 5.0)
        self.assertAlmostEqual(result["initial"].item(), 13.0)
        self.assertAlmostEqual(result["boundary"].item(), 25.0)
        expected = 2*5 + 3*13 + 4*25 + 0.5*analytic_residual_loss(self.collocation)
        torch.testing.assert_close(result["total"], expected)

    def test_zero_weight_removes_physics_gradient_contribution(self):
        torch.manual_seed(17)
        model = PINN([0, 0, 0], [2, 2, 1], Config(mode="inverse", hidden_layers=2, width=7))
        config = Config(w_observation=1, w_residual=0)
        result = objective(model, self.initial.clone(), torch.zeros(2, 2, dtype=torch.float64),
                           self.collocation.clone(), **config.inverse_loss_weights())
        torch.testing.assert_close(result["total"], result["observation"])
        result["total"].backward()
        self.assertEqual(model.lambda1.grad.item(), 0.0)
        self.assertEqual(model.lambda2.grad.item(), 0.0)
        self.assertGreater(model.net[0].weight.grad.norm().item(), 0.0)

    def test_backpropagation_with_real_network(self):
        for mode in ("known", "inverse"):
            with self.subTest(mode=mode):
                torch.manual_seed(17)
                model = PINN([0, 0, 0], [2, 2, 1], Config(mode=mode, hidden_layers=2, width=7))
                result = forward_objective(model, self.initial.clone(), torch.zeros(2, 2, dtype=torch.float64),
                                           self.boundary.clone(), torch.zeros(2, 2, dtype=torch.float64),
                                           self.collocation.clone(), self.gauge.clone())
                self.assertTrue(result["total"].requires_grad)
                result["total"].backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                if mode == "inverse":
                    self.assertGreater(model.lambda1.grad.abs().item(), 0)
                    self.assertGreater(model.lambda2.grad.abs().item(), 0)
                else:
                    self.assertNotIn("lambda1", dict(model.named_parameters()))

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            velocity_loss(self.model, self.initial, torch.zeros(2, 1))
        with self.assertRaises(ValueError):
            physics_loss(self.model, torch.empty(0, 3))
        with self.assertRaises(ValueError):
            pressure_gauge_loss(self.model, self.boundary)
        with self.assertRaises(ValueError):
            objective(self.model, self.initial, analytic_velocity(self.initial),
                      self.collocation, initial_coordinates=self.initial)
        with self.assertRaises(ValueError):
            forward_objective(self.model, self.initial, analytic_velocity(self.initial),
                              self.boundary, analytic_velocity(self.boundary),
                              self.collocation, self.gauge, w_initial=-1)
        for weight in ("w_observation", "w_initial", "w_boundary", "w_residual"):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(weight=weight, value=value), self.assertRaises(ValueError):
                    objective(self.model, self.initial, analytic_velocity(self.initial),
                              self.collocation, **{weight: value})
