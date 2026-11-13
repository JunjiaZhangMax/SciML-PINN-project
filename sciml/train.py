"""Prepare data, train with Adam and optional L-BFGS, then save and evaluate."""

import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import torch
from torch import optim

from ._artifacts import load_checkpoint, save_arrays, save_checkpoint, sha256, write_json
from .config import Config
from .data import DEFAULT_DATA, ForwardData, PreparedData, load_data, prepare_inverse_data, prepare_forward_data
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


def adam_learning_rate(config: Config, step: int) -> float:
    """Rate for 1-based Adam step; a pure function of step, so resume is exact."""
    if config.adam_lr_final is None or config.adam_steps == 1:
        return config.adam_lr
    fraction = (step - 1) / (config.adam_steps - 1)
    return config.adam_lr * (config.adam_lr_final / config.adam_lr) ** fraction


def _load_initial_state(config: Config, runs_dir: Path, data_sha256: str,
                        model: PINN) -> tuple[dict, dict]:
    """Checkpoint of the run named by init_from, checked against this run's setup."""
    source_dir = Path(runs_dir) / config.init_from
    source_path = source_dir / "checkpoint.pt"
    if not source_path.exists():
        raise FileNotFoundError(f"init_from run has no checkpoint: {source_path}")
    source = load_checkpoint(source_path)
    source_config = Config.load(source_dir / "config.json")
    if source["config_fingerprint"] != source_config.fingerprint:
        raise ValueError("init_from config does not match its checkpoint")
    if source["data_sha256"] != data_sha256:
        raise ValueError("init_from run was trained on a different dataset")
    for key in ("mode", "hidden_layers", "width"):
        if getattr(source_config, key) != getattr(config, key):
            raise ValueError(f"init_from run has a different {key}")
    for key in ("lower", "upper"):
        if not torch.equal(source["model"][key].cpu(), model.get_buffer(key).cpu()):
            raise ValueError("init_from run covers a different coordinate range")
    return source, {"run": config.init_from, "checkpoint_sha256": sha256(source_path),
                    "adam_steps": source["progress"]["adam_steps"],
                    "lbfgs_steps": source["progress"]["lbfgs_steps"],
                    "training_seconds": source["progress"].get("training_seconds")}


def train(config: Config, data_path: Path = DEFAULT_DATA,
          runs_dir: Path = Path("runs"), resume: bool = False) -> Path:
    """Adam, then optional L-BFGS; save periodically. Resume needs the same config and data."""
    # 1. Prepare the output directory, data, and model.
    config.validate()
    run_dir = Path(runs_dir) / config.name
    checkpoint_path = run_dir / "checkpoint.pt"
    identity = {"config_fingerprint": config.fingerprint, "data_sha256": sha256(Path(data_path))}
    saved = load_checkpoint(checkpoint_path) if resume else None
    if saved:
        if any(saved[key] != value for key, value in identity.items()):
            raise ValueError("Resume requires the original config and dataset")
        if sha256(run_dir / "samples.npz") != saved["samples_sha256"]:
            raise ValueError("Saved samples have changed")
    elif run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    prepared = prepare_training_data(config, data_path)
    torch.manual_seed(config.seed)
    model = PINN(prepared.samples.lower, prepared.samples.upper, config).to(prepared.device)
    model.train()
    # The starting run is checked before a directory is created that would block a retry.
    initial = None
    if saved:
        identity["init"] = saved.get("init")
    elif config.init_from:
        initial, identity["init"] = _load_initial_state(config, runs_dir, identity["data_sha256"], model)
    else:
        identity["init"] = None
    if not saved:
        run_dir.mkdir(parents=True, exist_ok=False)
        config.save(run_dir / "config.json")
        save_arrays(run_dir / "samples.npz", **asdict(prepared.samples))
    identity["samples_sha256"] = sha256(run_dir / "samples.npz")
    started = time.perf_counter()
    previous_seconds = saved["progress"]["training_seconds"] if saved else 0.0
    if config.mode == "known":
        objective = forward_objective
        weights = config.forward_loss_weights()
    else:
        objective = inverse_objective
        weights = config.inverse_loss_weights()
    lbfgs_target = config.lbfgs_steps if config.use_lbfgs else 0

    def record(stage, step):
        # Log losses at the current weights, after the optimizer update.
        losses = objective(model, **prepared.loss_inputs, **weights)
        values = {name: value.item() for name, value in losses.items()}
        row = {"stage": stage, "step": step, **values,
               "lambda1": model.lambda1.item(), "lambda2": model.lambda2.item(),
               "elapsed_seconds": previous_seconds + time.perf_counter() - started}
        history_path = run_dir / "history.csv"
        new_file = not history_path.exists()
        with open(history_path, "a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"[{stage}] step={step} loss={values['total']:.6g} "
              f"lambda=({row['lambda1']:.6g}, {row['lambda2']:.6g})", flush=True)
        return values["total"]

    # 2. Adam: clear gradients, compute gradients, update weights.
    adam = optim.Adam(model.parameters(), lr=config.adam_lr)
    step = lbfgs_done = 0
    lbfgs_converged = False
    if initial:
        model.load_state_dict(initial["model"])
        adam.load_state_dict(initial["adam"])
        print(f"Starting from {config.init_from} (Adam {identity['init']['adam_steps']}, "
              f"L-BFGS {identity['init']['lbfgs_steps']})", flush=True)
    if saved:
        model.load_state_dict(saved["model"])
        adam.load_state_dict(saved["adam"])
        step = saved["progress"]["adam_steps"]
        lbfgs_done = saved["progress"]["lbfgs_steps"]
        lbfgs_converged = saved["progress"].get("lbfgs_converged", False)
        # Discard log entries newer than the last successfully saved checkpoint.
        history_path = run_dir / "history.csv"
        with open(history_path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = [row for row in reader if int(row["step"]) <= step + lbfgs_done]
        with open(history_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Resuming at Adam step {step}, L-BFGS iteration {lbfgs_done}", flush=True)

    saved_progress = (step, lbfgs_done)

    def persist():
        nonlocal saved_progress
        # A diverged model must never replace the last usable checkpoint.
        if not all(torch.isfinite(value).all() for value in model.parameters()):
            raise FloatingPointError(
                f"Nonfinite model parameters after Adam step {step}, L-BFGS iteration {lbfgs_done}")
        save_checkpoint(checkpoint_path, {
            **identity, "model": model.state_dict(), "adam": adam.state_dict(),
            "progress": {"adam_steps": step, "lbfgs_steps": lbfgs_done,
                         "lbfgs_converged": lbfgs_converged,
                         "training_seconds": previous_seconds + time.perf_counter() - started},
        })
        saved_progress = (step, lbfgs_done)

    def set_status(state, reason=None):
        write_json(run_dir / "status.json", {
            "state": state, "reason": reason,
            "checkpoint_adam_steps": saved_progress[0], "target_adam_steps": config.adam_steps,
            "checkpoint_lbfgs_steps": saved_progress[1], "target_lbfgs_steps": lbfgs_target,
            "init": identity["init"]})

    def clear_gradients(optimizer):
        optimizer.zero_grad(set_to_none=True)
        # Coordinate gradients are not optimized; clear them between steps.
        for tensor in prepared.loss_inputs.values():
            tensor.grad = None

    note = None
    try:
        if not resume:
            record("initial", 0)
            persist()
        set_status("running")
        for step in range(step + 1, config.adam_steps + 1):
            for group in adam.param_groups:
                group["lr"] = adam_learning_rate(config, step)
            clear_gradients(adam)
            losses = objective(model, **prepared.loss_inputs, **weights)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Nonfinite loss at Adam step {step}")
            losses["total"].backward()
            adam.step()
            del losses
            if step == 1 or step % config.log_every == 0 or step == config.adam_steps:
                record("adam", step)
            if step % config.checkpoint_every == 0 or step == config.adam_steps:
                persist()

        # 3. Optional L-BFGS in chunks of log_every iterations, each logged and saved.
        # Curvature memory is rebuilt after a resume; the weights are what is kept.
        if lbfgs_done < lbfgs_target and not lbfgs_converged:
            lbfgs = optim.LBFGS(model.parameters(), lr=1.0, history_size=config.lbfgs_history,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-15)

            def closure():
                clear_gradients(lbfgs)
                total = objective(model, **prepared.loss_inputs, **weights)["total"]
                total.backward()
                return total

            def iterations():
                state = lbfgs.state_dict()["state"]
                return state[0].get("n_iter", 0) if state else 0

            while lbfgs_done < lbfgs_target and not lbfgs_converged:
                chunk = min(config.log_every, lbfgs_target - lbfgs_done)
                before = iterations()
                for group in lbfgs.param_groups:
                    # A generous evaluation cap, so only the tolerances end a chunk early.
                    group["max_iter"], group["max_eval"] = chunk, 30 * chunk
                lbfgs.step(closure)
                taken = iterations() - before
                lbfgs_done += taken
                lbfgs_converged = taken < chunk
                if not math.isfinite(record("lbfgs", step + lbfgs_done)):
                    raise FloatingPointError(f"Nonfinite loss at L-BFGS iteration {lbfgs_done}")
                persist()
            if lbfgs_converged:
                note = f"L-BFGS met its tolerance after {lbfgs_done} of {lbfgs_target} iterations"
                print(note, flush=True)

        # 4. Evaluate the final model on the held-out test set.
        evaluate(run_dir, data_path)
    except KeyboardInterrupt:
        set_status("interrupted", "KeyboardInterrupt")
        raise
    except Exception as error:
        # Keep the failure on record; the last finite checkpoint stays resumable.
        set_status("failed", f"{type(error).__name__}: {error}")
        raise
    set_status("completed", note)
    print(f"Saved run: {run_dir}", flush=True)
    return run_dir
