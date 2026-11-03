"""TODO: differentiable loss; optimizer stepping belongs in train.py."""

import torch

from .model import PINN


def objective(model: PINN, observation_coordinates: torch.Tensor,
              observed_velocity: torch.Tensor, collocation_coordinates: torch.Tensor
              ) -> dict[str, torch.Tensor]:
    """TODO: scalar tensors loss, data_loss, physics_loss; keep their graphs.

    loss=mean((u-u_obs)^2+(v-v_obs)^2)+mean(f^2+g^2), each mean over its own
    points. No optimizer step/backward here. PDE chunking must preserve gradients.
    """
    raise NotImplementedError("Implement data and physics losses")
