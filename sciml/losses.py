"""Differentiable forward/inverse losses; optimizer steps belong in train.py.

Prepare float64 tensors on the model's device before calling these functions.
Coordinate columns are [x, y, t]; velocity target columns are [u, v].
"""

import math

import torch

from .model import PINN
from .physics import fields


def _check_coordinates(coordinates: torch.Tensor) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0:
        raise ValueError("coordinates must have nonempty shape (N, 3)")


def _check_weights(*weights: float) -> None:
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Loss weights must be finite and nonnegative")


def velocity_loss(model: PINN, coordinates: torch.Tensor,
                  target_velocity: torch.Tensor) -> torch.Tensor:
    """Mean squared u/v error, used for observations, initial data, or boundary data."""
    _check_coordinates(coordinates)
    if target_velocity.shape != (len(coordinates), 2):
        raise ValueError("target_velocity must have shape (N, 2), matching coordinates")
    prediction = fields(model, coordinates, residuals=False)
    return (
        (prediction["u"] - target_velocity[:, 0:1]).square()
        + (prediction["v"] - target_velocity[:, 1:2]).square()
    ).mean()


def physics_loss(model: PINN, collocation_coordinates: torch.Tensor) -> torch.Tensor:
    """Mean squared momentum residuals at unlabeled PDE collocation points."""
    _check_coordinates(collocation_coordinates)
    prediction = fields(model, collocation_coordinates, residuals=True)
    return (prediction["f"].square() + prediction["g"].square()).mean()


def pressure_gauge_loss(model: PINN, gauge_coordinates: torch.Tensor) -> torch.Tensor:
    """Set p(x_ref, y_ref, t)=0 at sampled times without pressure truth labels.
    """
    _check_coordinates(gauge_coordinates)
    xy = gauge_coordinates[:, :2]
    if not torch.allclose(xy, xy[:1].expand_as(xy)):
        raise ValueError("Pressure gauge must use one fixed spatial reference point")
    pressure = model(gauge_coordinates)[:, 1:2]
    return pressure.square().mean()


def forward_objective(
    model: PINN,
    initial_coordinates: torch.Tensor,
    initial_velocity: torch.Tensor,
    boundary_coordinates: torch.Tensor,
    boundary_velocity: torch.Tensor,
    collocation_coordinates: torch.Tensor,
    gauge_coordinates: torch.Tensor,
    *,
    w_initial: float = 1.0,
    w_boundary: float = 1.0,
    w_residual: float = 1.0,
    w_gauge: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Forward problem: initial/boundary velocities + PDE + pressure reference.
    """
    _check_weights(w_initial, w_boundary, w_residual, w_gauge)
    initial = velocity_loss(model, initial_coordinates, initial_velocity)
    boundary = velocity_loss(model, boundary_coordinates, boundary_velocity)
    residual = physics_loss(model, collocation_coordinates)
    gauge = pressure_gauge_loss(model, gauge_coordinates)
    total = (w_initial * initial + w_boundary * boundary
             + w_residual * residual + w_gauge * gauge)
    return {"total": total, "initial": initial, "boundary": boundary,
            "residual": residual, "gauge": gauge}


def inverse_objective(
    model: PINN,
    observation_coordinates: torch.Tensor,
    observed_velocity: torch.Tensor,
    collocation_coordinates: torch.Tensor,
    initial_coordinates: torch.Tensor | None = None,
    initial_velocity: torch.Tensor | None = None,
    boundary_coordinates: torch.Tensor | None = None,
    boundary_velocity: torch.Tensor | None = None,
    w_initial=1.0,
    w_boundary=1.0,
    w_observation=1.0,
    w_residual=1.0,
) -> dict[str, torch.Tensor]:
    """Observation-based reconstruction/inverse objective, preserving the old entry.
    """
    _check_weights(w_initial, w_boundary, w_observation, w_residual)
    if (initial_coordinates is None) != (initial_velocity is None):
        raise ValueError("Provide both initial_coordinates and initial_velocity")
    if (boundary_coordinates is None) != (boundary_velocity is None):
        raise ValueError("Provide both boundary_coordinates and boundary_velocity")
    observation = velocity_loss(model, observation_coordinates, observed_velocity)
    residual = physics_loss(model, collocation_coordinates)
    initial = (velocity_loss(model, initial_coordinates, initial_velocity)
               if initial_coordinates is not None and initial_velocity is not None
               else observation.new_zeros(()))
    boundary = (velocity_loss(model, boundary_coordinates, boundary_velocity)
                if boundary_coordinates is not None and boundary_velocity is not None
                else observation.new_zeros(()))
    total = (w_observation * observation + w_residual * residual + w_initial * initial + w_boundary * boundary)
    return {"total": total, "observation": observation, "residual": residual,
            "initial": initial, "boundary": boundary}


# Backwards-compatible name for older examples and tests.
objective = inverse_objective
