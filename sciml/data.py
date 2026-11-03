"""TODO: data loading and preparation; no downloads or computations on import."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config

DATA_URL = "https://raw.githubusercontent.com/maziarraissi/PINNs/master/main/Data/cylinder_nektar_wake.mat"
DEFAULT_DATA = Path("data/raw/cylinder_nektar_wake.mat")


@dataclass
class WakeData:
    """Raw arrays: xy=(S,2), times=(T,), velocity=(S,2,T), pressure=(S,T).

    Pressure is evaluation-only; do not pass its labels to the training loss.
    """

    xy: np.ndarray
    times: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray


@dataclass
class PreparedData:
    """Preprocessing output. Coordinates: [x,y,t]; arrays: float64 except indices.

    Flattened index = spatial_index * T + time_index. Standard deviations use
    clean training observations per channel (ddof=0), before adding noise.
    """

    observation_coordinates: np.ndarray  # (N,3)
    clean_velocity: np.ndarray           # (N,2), provenance only
    observed_velocity: np.ndarray        # (N,2), training targets
    collocation_coordinates: np.ndarray  # (N_f,3), coordinates only
    observation_indices: np.ndarray      # (N,)
    collocation_indices: np.ndarray      # (N_f,)
    test_indices: np.ndarray             # (N_test,), shared across runs
    base_noise: np.ndarray               # (N,2), standard-normal draws
    noise_std: np.ndarray                # (2,)
    lower: np.ndarray                    # (3,), domain bounds
    upper: np.ndarray                    # (3,)


def download(path: Path = DEFAULT_DATA) -> Path:
    """TODO: fetch/validate raw MAT; preserve originals; record URL, SHA256, metadata."""
    raise NotImplementedError("Implement dataset acquisition in sciml/data.py")


def load_data(path: Path = DEFAULT_DATA) -> WakeData:
    """TODO: read X_star/t/U_star/p_star and validate shapes, bounds and finite values."""
    raise NotImplementedError("Implement MAT loading in sciml/data.py")


def prepare_data(data: WakeData, config: Config) -> PreparedData:
    """TODO: holdout, nested samples, paired noise, collocation; see docs/DESIGN.md."""
    raise NotImplementedError("Implement sampling and noise in sciml/data.py")
