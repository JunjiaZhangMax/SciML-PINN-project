"""Automatic differentiation, velocity recovery and PDE residuals."""

import torch

from .model import PINN


def derivative(value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """(B,1) scalar field -> (B,3) derivatives [d/dx,d/dy,d/dt].

    Must support higher derivatives and zero derivatives of analytic fields.
    """


    
def fields(model: PINN, coordinates: torch.Tensor, residuals: bool = False) -> dict[str, torch.Tensor]:
    """Return u/v/p (B,1); optionally f/g/divergence (B,1).

    u=psi_y, v=-psi_x. Momentum residuals are specified in docs/DESIGN.md.
    Velocity prediction still requires coordinate gradients during evaluation.
    """
    