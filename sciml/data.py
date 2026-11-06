"""Reference data loading and reproducible forward/inverse experiment preparation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

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


@dataclass
class ForwardData:
    """Known-coefficient window problem. No interior velocity/pressure labels."""

    initial_coordinates: np.ndarray      # (S, 3), first reference time
    initial_velocity: np.ndarray         # (S, 2)
    boundary_coordinates: np.ndarray     # boundary points at later times
    boundary_velocity: np.ndarray        # prescribed reference boundary velocities
    collocation_coordinates: np.ndarray  # interior, later times; no labels
    gauge_coordinates: np.ndarray        # one fixed spatial point, every selected time
    initial_indices: np.ndarray
    boundary_indices: np.ndarray
    collocation_indices: np.ndarray
    test_indices: np.ndarray             # interior, later times, held-out labels
    time_indices: np.ndarray             # selected original time indices
    lower: np.ndarray
    upper: np.ndarray


def download(path: Path = DEFAULT_DATA) -> Path:
    """TODO: fetch/validate raw MAT; preserve originals; record URL, SHA256, metadata."""
    raise NotImplementedError("Implement dataset acquisition in sciml/data.py")


def load_data(path: Path = DEFAULT_DATA) -> WakeData:
    """Read X_star/t/U_star/p_star without changing the raw MAT file."""
    raw = loadmat(path)
    names = ("X_star", "t", "U_star", "p_star")
    missing = set(names) - raw.keys()
    if missing:
        raise ValueError(f"Missing MAT arrays: {sorted(missing)}")
    data = WakeData(
        xy=np.asarray(raw["X_star"], dtype=np.float64),
        times=np.asarray(raw["t"], dtype=np.float64).reshape(-1),
        velocity=np.asarray(raw["U_star"], dtype=np.float64),
        pressure=np.asarray(raw["p_star"], dtype=np.float64),
    )
    n, nt = len(data.xy), len(data.times)
    if (data.xy.shape != (n, 2) or data.velocity.shape != (n, 2, nt)
            or data.pressure.shape != (n, nt)):
        raise ValueError("Expected X_star=(S,2), U_star=(S,2,T), p_star=(S,T)")
    if n < 3 or nt < 2 or not np.all(np.diff(data.times) > 0):
        raise ValueError("Need at least three spatial points and increasing time samples")
    if not all(np.isfinite(a).all() for a in (data.xy, data.times, data.velocity, data.pressure)):
        raise ValueError("Reference data contains nonfinite values")
    if np.any(np.ptp(data.xy, axis=0) <= 0):
        raise ValueError("Both spatial coordinates must have a nonzero range")
    return data


def _selected_times(data: WakeData, config: Config) -> np.ndarray:
    """Use original frames up to time_end; no temporal interpolation."""
    if config.time_end is None:
        return np.arange(len(data.times))
    end = config.time_end
    if end <= data.times[0] or (end > data.times[-1] and not np.isclose(end, data.times[-1])):
        raise ValueError("time_end must be after the first time and within the reference range")
    selected = np.flatnonzero((data.times <= end)
                              | np.isclose(data.times, end, rtol=0, atol=1e-10))
    if len(selected) < 2:
        raise ValueError("The selected time window must contain at least two reference frames")
    return selected


def coordinates_at(data: WakeData, indices: np.ndarray) -> np.ndarray:
    """Decode original flattened indices: index = spatial_index * T + time_index."""
    space, time = np.divmod(indices, len(data.times))
    return np.column_stack((data.xy[space], data.times[time]))


def velocities_at(data: WakeData, indices: np.ndarray) -> np.ndarray:
    space, time = np.divmod(indices, len(data.times))
    return np.column_stack((data.velocity[space, 0, time], data.velocity[space, 1, time]))


def _split_pool(data: WakeData, space: np.ndarray, times: np.ndarray,
                config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Time-stratified label holdout; neither observations nor PDE points use it."""
    if config.test_per_time >= len(space):
        raise ValueError("test_per_time must leave spatial samples for training at every time")
    nt = len(data.times)
    rng = np.random.default_rng(config.split_seed)
    test = np.concatenate([
        rng.choice(space, config.test_per_time, replace=False) * nt + t for t in times
    ])
    candidates = (space[:, None] * nt + times[None, :]).ravel()
    pool = np.setdiff1d(candidates, test, assume_unique=True)
    return pool, test


def _bounds(data: WakeData, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.r_[data.xy.min(axis=0), data.times[times[0]]],
            np.r_[data.xy.max(axis=0), data.times[times[-1]]])


def prepare_forward_data(data: WakeData, config: Config) -> ForwardData:
    """Initial velocities + four-edge velocities; interior labels are held out.

    All spatial points at the first time provide IC labels. Each distinct edge
    point is used once per subsequent selected time for BC labels (corners are
    not duplicated). PDE points are sampled from interior reference coordinates
    at later times, excluding the test set; their velocity labels are not read.
    """
    config.validate()
    if config.mode != "known":
        raise ValueError("Forward preparation requires mode='known'")
    if config.noise != 0:
        raise ValueError("Forward initial/boundary data are noise-free; use noise=0")
    times = _selected_times(data, config)
    lower, upper = _bounds(data, times)
    boundary_mask = np.any(
        np.isclose(data.xy, lower[:2], rtol=0, atol=1e-10)
        | np.isclose(data.xy, upper[:2], rtol=0, atol=1e-10), axis=1
    )
    boundary_space = np.flatnonzero(boundary_mask)
    interior_space = np.flatnonzero(~boundary_mask)
    nt = len(data.times)
    initial_indices = np.arange(len(data.xy)) * nt + times[0]
    boundary_indices = (boundary_space[:, None] * nt + times[None, 1:]).ravel()
    if len(boundary_indices) == 0 or len(interior_space) == 0:
        raise ValueError("The rectangular window needs both boundary and interior samples")
    pool, test_indices = _split_pool(data, interior_space, times[1:], config)
    if config.n_collocation > len(pool):
        raise ValueError("n_collocation exceeds the interior non-test pool in this time window")
    collocation_indices = np.random.default_rng(config.collocation_seed).choice(
        pool, config.n_collocation, replace=False
    )
    reference_xy = (lower[:2] + upper[:2]) / 2
    gauge_coordinates = np.column_stack((
        np.tile(reference_xy, (len(times), 1)), data.times[times]
    ))
    return ForwardData(
        initial_coordinates=coordinates_at(data, initial_indices),
        initial_velocity=velocities_at(data, initial_indices),
        boundary_coordinates=coordinates_at(data, boundary_indices),
        boundary_velocity=velocities_at(data, boundary_indices),
        collocation_coordinates=coordinates_at(data, collocation_indices),
        gauge_coordinates=gauge_coordinates,
        initial_indices=initial_indices, boundary_indices=boundary_indices,
        collocation_indices=collocation_indices, test_indices=test_indices,
        time_indices=times, lower=lower, upper=upper,
    )


def prepare_inverse_data(data: WakeData, config: Config) -> PreparedData:
    """Observation-based reconstruction/inverse data with paired noise and nesting.

    This is separate from prepare_forward_data, even when coefficients are known.
    The reference pressure array is never used for constructing training labels.
    """
    config.validate()
    times = _selected_times(data, config)
    pool, test_indices = _split_pool(data, np.arange(len(data.xy)), times, config)
    if config.n_observations > len(pool):
        raise ValueError("n_observations exceeds the non-test pool")
    observation_indices = np.random.default_rng(config.observation_seed).permutation(pool)[:config.n_observations]
    clean = velocities_at(data, observation_indices)
    base_noise = np.random.default_rng(config.noise_seed).standard_normal((config.n_observations, 2))
    noise_std = clean.std(axis=0, ddof=0)
    observed = clean + config.noise * noise_std * base_noise
    if config.collocation == "observations":
        collocation_indices = observation_indices.copy()
    else:
        if config.n_collocation > len(pool):
            raise ValueError("n_collocation exceeds the non-test pool")
        collocation_indices = np.random.default_rng(config.collocation_seed).choice(
            pool, config.n_collocation, replace=False
        )
    lower, upper = _bounds(data, times)
    return PreparedData(
        observation_coordinates=coordinates_at(data, observation_indices),
        clean_velocity=clean, observed_velocity=observed,
        collocation_coordinates=coordinates_at(data, collocation_indices),
        observation_indices=observation_indices, collocation_indices=collocation_indices,
        test_indices=test_indices, base_noise=base_noise, noise_std=noise_std,
        lower=lower, upper=upper,
    )
