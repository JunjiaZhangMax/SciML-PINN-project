"""Automatic differentiation, velocity recovery and PDE residuals."""

import torch

from .model import PINN


def derivative(value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """(B,1) scalar field -> (B,3) derivatives [d/dx,d/dy,d/dt].

    Must support higher derivatives and zero derivatives of analytic fields.
    """
    derivatives = torch.autograd.grad(
        value,
        coordinates,
        grad_outputs=torch.ones_like(value),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if derivatives is None:
        return coordinates * 0.0
    
    return derivatives + coordinates * 0.0  # ensure same device and dtype as coordinates

def fields(model: PINN, coordinates: torch.Tensor, residuals: bool = False) -> dict[str, torch.Tensor]:
    """Return u/v/p (B,1); optionally f/g/divergence (B,1).

    u=psi_y, v=-psi_x. Momentum residuals are specified in docs/DESIGN.md.
    Velocity prediction still requires coordinate gradients during evaluation.
    """
    fields = {}
    coordinates.requires_grad_(True)
    psi, p = model(coordinates).split(1, dim=1)
    psi_x, psi_y, _ = derivative(psi, coordinates).split(1, dim=1)
    u = psi_y
    v = -psi_x
    fields["u"] = u
    fields["v"] = v
    fields["p"] = p
    if residuals:
        p_x, p_y, _ = derivative(p, coordinates).split(1, dim=1)
        u_x, u_y, u_t = derivative(u, coordinates).split(1, dim=1)
        u_xx, _ , _ = derivative(u_x, coordinates).split(1, dim=1)
        _ , u_yy, _ = derivative(u_y, coordinates).split(1, dim=1)
        v_x, v_y, v_t = derivative(v, coordinates).split(1, dim=1)
        v_xx, _ , _ = derivative(v_x, coordinates).split(1, dim=1)
        _ , v_yy, _ = derivative(v_y, coordinates).split(1, dim=1)
        lambda1 = model.lambda1
        lambda2 = model.lambda2
        f = u_t + lambda1 * (u * u_x + v * u_y) + p_x - lambda2 * (u_xx + u_yy)
        g = v_t + lambda1 * (u * v_x + v * v_y) + p_y - lambda2 * (v_xx + v_yy)
        divergence = u_x + v_y
        fields["f"] = f
        fields["g"] = g
        fields["divergence"] = divergence
    return fields

    