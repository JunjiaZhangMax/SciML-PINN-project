"""Evaluate saved checkpoints on the original held-out points, without retraining."""

from pathlib import Path
import time

import numpy as np
import torch

from ._artifacts import load_checkpoint, save_arrays, sha256, write_json
from .config import Config
from .data import DEFAULT_DATA, coordinates_at, load_data, velocities_at
from .model import PINN
from .physics import fields


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float | None:
    """Return relative L2 error, or None for an identically zero reference."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must have identical shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("Metrics require finite predictions and references")
    denominator = np.linalg.norm(reference.ravel())
    if denominator == 0:
        return None
    result = float(np.linalg.norm((prediction - reference).ravel()) / denominator)
    if not np.isfinite(result):
        raise FloatingPointError("Nonfinite relative error")
    return result


def centered_pressure(values: np.ndarray, time_indices: np.ndarray) -> np.ndarray:
    """Remove each time group's spatial arithmetic mean, including unsorted groups."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    time_indices = np.asarray(time_indices).reshape(-1)
    if len(values) != len(time_indices):
        raise ValueError("Pressure values and time indices must have matching lengths")
    _, inverse = np.unique(time_indices, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    means = sums / np.bincount(inverse)
    return values - means[inverse]


def pressure_error(prediction: np.ndarray, reference: np.ndarray,
                   time_indices: np.ndarray) -> float | None:
    """Relative pressure error after independently centering each frame."""
    return relative_l2(centered_pressure(prediction, time_indices),
                       centered_pressure(reference, time_indices))


def _predict(model: PINN, coordinates: np.ndarray, batch_size: int,
             device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    velocity, pressure = [], []
    model.eval()
    # u/v are coordinate derivatives, so evaluation still needs autograd.
    with torch.enable_grad():
        for start in range(0, len(coordinates), batch_size):
            x = torch.tensor(coordinates[start:start + batch_size], dtype=torch.float64,
                             device=device, requires_grad=True)
            prediction = fields(model, x, residuals=False)
            velocity.append(torch.cat((prediction["u"], prediction["v"]), dim=1).detach().cpu().numpy())
            pressure.append(prediction["p"].detach().cpu().numpy().reshape(-1))
    uv, p = np.concatenate(velocity), np.concatenate(pressure)
    if not np.isfinite(uv).all() or not np.isfinite(p).all():
        raise FloatingPointError("Nonfinite evaluation predictions")
    return uv, p


def evaluate(run_dir: Path, data_path: Path = DEFAULT_DATA) -> dict:
    """Use saved indices/checkpoint; export held-out predictions and three snapshots.

    Snapshot frames span the actual training window and use all spatial points.
    They are illustrations, not held-out test metrics. No test labels influence
    training, checkpoint selection, or stopping.
    """
    started = time.perf_counter()
    run_dir, data_path = Path(run_dir), Path(data_path)
    config = Config.load(run_dir / "config.json")
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                          else "cpu" if config.device == "auto" else config.device)
    checkpoint = load_checkpoint(run_dir / "checkpoint.pt", device)
    if checkpoint["config_fingerprint"] != config.fingerprint:
        raise ValueError("Saved config does not match the checkpoint")
    if sha256(data_path) != checkpoint["data_sha256"]:
        raise ValueError("Reference dataset does not match the checkpoint")
    if sha256(run_dir / "samples.npz") != checkpoint["samples_sha256"]:
        raise ValueError("Saved samples have changed or are corrupted")
    with np.load(run_dir / "samples.npz", allow_pickle=False) as saved:
        test_indices = saved["test_indices"].copy()
        lower, upper = saved["lower"].copy(), saved["upper"].copy()
    data = load_data(data_path)
    nt = len(data.times)
    test_coordinates = coordinates_at(data, test_indices)
    reference_uv = velocities_at(data, test_indices)
    space, time_indices = np.divmod(test_indices, nt)
    reference_p = data.pressure[space, time_indices]
    model = PINN(lower, upper, config).to(device)
    model.load_state_dict(checkpoint["model"])
    predicted_uv, predicted_p = _predict(model, test_coordinates, config.eval_batch_size, device)

    selected = np.flatnonzero((data.times >= lower[2] - 1e-10) & (data.times <= upper[2] + 1e-10))
    frames = np.unique(selected[[0, len(selected) // 2, -1]])
    snapshot_indices = (np.arange(len(data.xy))[:, None] * nt + frames[None, :]).ravel()
    snapshot_coordinates = coordinates_at(data, snapshot_indices)
    snapshot_uv, snapshot_p = _predict(model, snapshot_coordinates, config.eval_batch_size, device)
    snapshot_reference_uv = velocities_at(data, snapshot_indices)
    snapshot_space, snapshot_times = np.divmod(snapshot_indices, nt)
    snapshot_reference_p = data.pressure[snapshot_space, snapshot_times]

    lambda1, lambda2 = float(model.lambda1.detach()), float(model.lambda2.detach())
    metrics = {
        "purpose": config.purpose, "mode": config.mode,
        "task": "forward" if config.mode == "known" else "inverse",
        "lambda1": lambda1, "lambda2": lambda2,
        "relative_error_lambda1": abs(lambda1 - 1.0),
        "relative_error_lambda2": abs(lambda2 - 0.01) / 0.01,
        "relative_l2_u": relative_l2(predicted_uv[:, 0], reference_uv[:, 0]),
        "relative_l2_v": relative_l2(predicted_uv[:, 1], reference_uv[:, 1]),
        "relative_l2_pressure_centered": pressure_error(predicted_p, reference_p, time_indices),
        "test_count": len(test_indices), "test_time_count": len(np.unique(time_indices)),
        "pressure_reference": "prediction and reference spatially centered separately at each test time",
        "relative_error_units": "fractions, not percentages",
        "nonphysical_viscosity": lambda2 < 0,
        "training_seconds": checkpoint["progress"].get("training_seconds"),
        "adam_steps": checkpoint["progress"]["adam_steps"],
        "lbfgs_steps": checkpoint["progress"]["lbfgs_steps"],
        "snapshot_time_indices": frames.tolist(),
        "snapshot_scope": "full-spatial illustrations, not held-out metrics",
        "evaluation_seconds": time.perf_counter() - started,
    }
    metrics["undefined_metrics"] = [key for key, value in metrics.items()
                                    if key.startswith("relative_") and value is None]
    save_arrays(
        run_dir / "predictions.npz",
        test_indices=test_indices, test_coordinates=test_coordinates, test_time_indices=time_indices,
        test_velocity_predicted=predicted_uv, test_velocity_reference=reference_uv,
        test_pressure_predicted=predicted_p, test_pressure_reference=reference_p,
        test_pressure_predicted_centered=centered_pressure(predicted_p, time_indices),
        test_pressure_reference_centered=centered_pressure(reference_p, time_indices),
        snapshot_indices=snapshot_indices, snapshot_coordinates=snapshot_coordinates,
        snapshot_time_indices=frames, snapshot_velocity_predicted=snapshot_uv,
        snapshot_velocity_reference=snapshot_reference_uv,
        snapshot_pressure_predicted=snapshot_p, snapshot_pressure_reference=snapshot_reference_p,
    )
    write_json(run_dir / "metrics.json", metrics)
    return metrics
