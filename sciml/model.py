"""TODO: architecture and the two PDE coefficients."""

import numpy as np
import torch
from torch import nn

from .config import Config


class PINN(nn.Module):
    """Physical [x,y,t] -> [psi,p]. Default: 8x20 tanh, float64.

    Expose scalar tensors lambda1/lambda2: fixed at 1/0.01 in known mode,
    trainable in inverse mode. Normalize inputs inside forward so derivatives
    remain with respect to physical coordinates.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray, config: Config):
        super().__init__()
        raise NotImplementedError("Implement network and coefficients in sciml/model.py")

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """TODO: (batch,3) coordinates -> (batch,2) stream function/pressure."""
        raise NotImplementedError("Implement PINN.forward")
