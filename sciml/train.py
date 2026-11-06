"""Prepare data, train with Adam, then save and evaluate."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch
from torch import optim

from ._artifacts import save_arrays, save_checkpoint, sha256
from .config import Config
from .data import DEFAULT_DATA, ForwardData, PreparedData, load_data, prepare_data as prepare_inverse_data, prepare_forward_data
from .evaluate import evaluate
from .losses import forward_objective, inverse_objective
from .model import PINN


@dataclass
class TrainingData:
    """Loss tensors plus NumPy samples/indices for provenance; no pressure labels."""

    loss_inputs: dict[str, torch.Tensor]
    samples: ForwardData | PreparedData
    device: torch.device

def prepare_training_data(config: Config, data_path: Path = DEFAULT_DATA) -> TrainingData:
    """known -> window IC/BC forward problem; inverse -> sparse observations."""
    config.validate()
    device_name = config.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    reference = load_data(data_path)
    if config.mode == "known":
        samples = prepare_forward_data(reference, config)
        arrays = {"initial_coordinates": samples.initial_coordinates,
                  "initial_velocity": samples.initial_velocity,
                  "boundary_coordinates": samples.boundary_coordinates,
                  "boundary_velocity": samples.boundary_velocity,
                  "collocation_coordinates": samples.collocation_coordinates,
                  "gauge_coordinates": samples.gauge_coordinates}
    else:
        samples = prepare_inverse_data(reference, config)
        arrays = {"observation_coordinates": samples.observation_coordinates,
                  "observed_velocity": samples.observed_velocity,
                  "collocation_coordinates": samples.collocation_coordinates}
    tensors = {}
    for name, array in arrays.items():
        tensor = torch.as_tensor(array, dtype=torch.float64, device=device)
        if name.endswith("_coordinates") and name != "gauge_coordinates":
            tensor.requires_grad_(True)
        tensors[name] = tensor
    return TrainingData(tensors, samples, device)


def train(config: Config, data_path: Path = DEFAULT_DATA,
          runs_dir: Path = Path("runs")) -> Path:
    """Run one experiment. Use a new config.name for each run."""
    # 1. Prepare the output directory, data, and model.
    config.validate()
    run_dir = Path(runs_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=False)
    config.save(run_dir / "config.json")
    prepared = prepare_training_data(config, data_path)
    save_arrays(run_dir / "samples.npz", **asdict(prepared.samples))
    torch.manual_seed(config.seed)
    model = PINN(prepared.samples.lower, prepared.samples.upper, config).to(prepared.device)
    model.train()
    started = time.perf_counter()
    if config.mode == "known":
        objective = forward_objective
        weights = config.forward_loss_weights()
    else:
        objective = inverse_objective
        weights = config.inverse_loss_weights()

    def record(stage, step):
        # Log losses at the current weights, after the optimizer update.
        losses = objective(model, **prepared.loss_inputs, **weights)
        values = {name: value.item() for name, value in losses.items()}
        row = {"stage": stage, "step": step, **values,
               "lambda1": model.lambda1.item(), "lambda2": model.lambda2.item(),
               "elapsed_seconds": time.perf_counter() - started}
        history_path = run_dir / "history.csv"
        new_file = not history_path.exists()
        with open(history_path, "a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"[{stage}] step={step} loss={values['total']:.6g} "
              f"lambda=({row['lambda1']:.6g}, {row['lambda2']:.6g})", flush=True)

    # 2. Adam: clear gradients, compute gradients, update weights.
    record("initial", 0)
    adam = optim.Adam(model.parameters(), lr=config.adam_lr)
    for step in range(1, config.adam_steps + 1):
        adam.zero_grad(set_to_none=True)
        # Coordinate gradients are not optimized; clear them between steps.
        for tensor in prepared.loss_inputs.values():
            tensor.grad = None
        losses = objective(model, **prepared.loss_inputs, **weights)
        losses["total"].backward()
        adam.step()
        del losses
        if step == 1 or step % config.log_every == 0 or step == config.adam_steps:
            record("adam", step)

    # 3. Save the final model and evaluate on the held-out test set.
    save_checkpoint(run_dir / "checkpoint.pt", {
        "model": model.state_dict(),
        "config_fingerprint": config.fingerprint,
        "data_sha256": sha256(Path(data_path)),
        "samples_sha256": sha256(run_dir / "samples.npz"),
        "progress": {"adam_steps": config.adam_steps, "lbfgs_steps": 0,
                     "training_seconds": time.perf_counter() - started},
    })
    evaluate(run_dir, data_path)
    print(f"Saved run: {run_dir}", flush=True)
    return run_dir
