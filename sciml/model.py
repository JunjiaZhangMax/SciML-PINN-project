"""Stream-function network and fixed or trainable Navier-Stokes coefficients."""

import numpy as np
import torch
from torch import nn

from .config import Config


class PINN(nn.Module):
    """Physical [x,y,t] -> [psi,p]. Default: 8x20 tanh, float64.
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray, config: Config):
        super().__init__()
        # Own copies prevent later edits to the input arrays from changing the model.
        lower_tensor = torch.as_tensor(lower, dtype=torch.float64).detach().clone()
        upper_tensor = torch.as_tensor(upper, dtype=torch.float64).detach().clone()
        if lower_tensor.shape != (3,) or upper_tensor.shape != (3,):
            raise ValueError("lower and upper must have shape (3,) in [x, y, t] order")
        if not torch.isfinite(lower_tensor).all() or not torch.isfinite(upper_tensor).all():
            raise ValueError("Domain bounds must be finite")
        if not torch.all(upper_tensor > lower_tensor):
            raise ValueError("Each upper bound must be greater than its lower bound")
        self.register_buffer("lower", lower_tensor)
        self.register_buffer("upper", upper_tensor)
        self.config = config

        # Known coefficients can be used for a forward solve or reconstruction check.
        if config.mode == "known":
            self.register_buffer("lambda1", torch.tensor(1.0, dtype=torch.float64))
            self.register_buffer("lambda2", torch.tensor(0.01, dtype=torch.float64))

        elif config.mode == "inverse":
            self.lambda1 = nn.Parameter(torch.tensor(config.lambda1_init, dtype=torch.float64))
            self.lambda2 = nn.Parameter(torch.tensor(config.lambda2_init, dtype=torch.float64))

        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        layers = []
        input_size = 3
        for _ in range(config.hidden_layers):
            layers.extend((nn.Linear(input_size, config.width, dtype=torch.float64), nn.Tanh()))
            input_size = config.width
        layers.append(nn.Linear(input_size, 2, dtype=torch.float64))
        self.net = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Map physical (batch, 3) coordinates to (batch, 2) [psi, p]
        """
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (batch, 3) in [x, y, t] order")
        lower = self.get_buffer("lower")
        upper = self.get_buffer("upper")
        normalized = 2.0 * (coordinates - lower) / (upper - lower) - 1.0
        return self.net(normalized)
