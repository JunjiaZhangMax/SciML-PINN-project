"""Diagnose a completed forward run; no optimization or checkpoint changes."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from sciml._artifacts import load_checkpoint, sha256
from sciml.config import Config
from sciml.data import coordinates_at, load_data
from sciml.model import PINN
from sciml.physics import fields


def residual_values(model, coordinates, device):
    values = []
    for start in range(0, len(coordinates), 256):
        x = torch.tensor(coordinates[start:start+256], dtype=torch.float64, device=device,
                         requires_grad=True)
        result = fields(model, x, residuals=True)
        values.append((result["f"].square() + result["g"].square()).detach().cpu().numpy().ravel())
    return np.concatenate(values)


def analytic_check():
    # Decaying Taylor-Green solution on an unrestricted coordinate sample.
    class TaylorGreen(torch.nn.Module):
        lambda1, lambda2 = 1., .01

        def forward(self, xyz):
            x, y, t = xyz.split(1, dim=1)
            decay = torch.exp(-.02*t)
            psi = torch.sin(x)*torch.sin(y)*decay
            p = .25*(torch.cos(2*x)+torch.cos(2*y))*decay.square()
            return torch.cat((psi, p), dim=1)

    rng = np.random.default_rng(12)
    xyz = torch.tensor(rng.uniform(-2, 2, (100, 3)), dtype=torch.float64, requires_grad=True)
    result = fields(TaylorGreen(), xyz, residuals=True)
    x, y, t = xyz.split(1, dim=1)
    torch.testing.assert_close(result["u"], torch.sin(x)*torch.cos(y)*torch.exp(-.02*t))
    torch.testing.assert_close(result["v"], -torch.cos(x)*torch.sin(y)*torch.exp(-.02*t))
    maximum = max(float(result[key].detach().abs().max()) for key in ("f", "g", "divergence"))
    assert maximum < 1e-12
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    output = run / "diagnostics"
    output.mkdir(exist_ok=True)
    cfg = Config.load(run / "config.json")
    checkpoint = load_checkpoint(run / "checkpoint.pt")
    data_path = PROJECT / "data/raw/cylinder_nektar_wake.mat"
    assert sha256(data_path) == checkpoint["data_sha256"]
    metadata = json.loads((PROJECT / "output/forward_comparison.source.json").read_text())
    assert metadata["checkpoint_sha256"] == sha256(run / "checkpoint.pt")
    with np.load(run / "animation_fields.npz") as a:
        xy, times, prediction, reference = a["xy"], a["times"], a["predicted"], a["reference"]
    with np.load(run / "samples.npz") as a:
        collocation = a["collocation_coordinates"]
        test = a["test_indices"]
        lower, upper = a["lower"], a["upper"]
    history = np.genfromtxt(run / "history.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    rel = np.sqrt(np.sum((prediction-reference)**2, axis=0)/np.sum(reference**2, axis=0))
    vrms_pred = np.sqrt(np.mean(prediction[:, :, 1]**2, axis=0))
    vrms_ref = np.sqrt(np.mean(reference[:, :, 1]**2, axis=0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN(lower, upper, cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    data = load_data(data_path)
    rng = np.random.default_rng(381)
    # 25 previously held-out interior points from each of the 199 later frames.
    heldout = np.concatenate([rng.choice(test[test % len(data.times) == t], 25, replace=False)
                              for t in np.unique(test % len(data.times))])
    check_coordinates = coordinates_at(data, heldout)
    print("Checking PDE residuals at training and held-out coordinates...", flush=True)
    train_residual = residual_values(model, collocation, device)
    test_residual = residual_values(model, check_coordinates, device)
    residual_bins = []
    for low, high in ((0, 5), (5, 10), (10, 15), (15, 20)):
        tr = train_residual[(collocation[:, 2] >= low) & (collocation[:, 2] < high)]
        te = test_residual[(check_coordinates[:, 2] >= low) & (check_coordinates[:, 2] < high)]
        residual_bins.append({"time_start": low, "time_end_exclusive": high,
                              "train_mse": float(tr.mean()), "heldout_mse": float(te.mean())})
    frame_rows = []
    for i, time in enumerate(times):
        frame_rows.append({"time": float(time), "full_field_relative_l2_u": float(rel[i, 0]),
                           "full_field_relative_l2_v": float(rel[i, 1]),
                           "full_field_relative_l2_p_centered": float(rel[i, 2]),
                           "v_spatial_rms_predicted": float(vrms_pred[i]),
                           "v_spatial_rms_reference": float(vrms_ref[i]),
                           "v_rms_ratio": float(vrms_pred[i]/vrms_ref[i])})
    with open(output / "per_time.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(frame_rows)
    boundary = {}
    for name, axis, position in (("inlet", 0, lower[0]), ("outlet", 0, upper[0]),
                                 ("bottom", 1, lower[1]), ("top", 1, upper[1])):
        mask = np.isclose(xy[:, axis], position)
        error = prediction[mask, 1:, :2] - reference[mask, 1:, :2]
        boundary[name] = {"uv_mse": float(np.mean(np.sum(error**2, axis=-1))),
                          "v_relative_l2": float(np.linalg.norm(error[:, :, 1])/np.linalg.norm(reference[mask, 1:, 1]))}
        boundary[name]["time_bins"] = []
        for low, high in ((0, 5), (5, 10), (10, 15), (15, 20)):
            selected = (times >= low) & (times < high) & (times > 0)
            predicted_rms = float(np.sqrt(np.mean(prediction[mask][:, selected, 1]**2)))
            reference_rms = float(np.sqrt(np.mean(reference[mask][:, selected, 1]**2)))
            boundary[name]["time_bins"].append({"time_start": low, "time_end_exclusive": high,
                                                "v_rms_predicted": predicted_rms,
                                                "v_rms_reference": reference_rms,
                                                "v_rms_ratio": predicted_rms/reference_rms})
    summary = {
        "run": str(run), "checkpoint_sha256": sha256(run / "checkpoint.pt"),
        "analytic_ns_max_residual": analytic_check(),
        "final_loss": {key: float(history[-1][key]) for key in ("total", "initial", "boundary", "residual", "gauge")},
        "loss_at_steps": {str(step): float(history[history["step"] == step]["total"][0])
                          for step in (10000, 20000, 30000, 40000, 45000, 50000)},
        "selected_frames": [frame_rows[i] for i in (0, 25, 50, 100, 150, 199)],
        "residual_train_mse": float(train_residual.mean()),
        "residual_heldout_mse": float(test_residual.mean()),
        "residual_heldout_to_train_ratio": float(test_residual.mean()/train_residual.mean()),
        "residual_time_bins": residual_bins, "boundary_errors": boundary,
        "scope": "Full-field time metrics include IC/BC locations; PDE held-out diagnostic uses 25 test points/time. v RMS measures transverse speed, not vorticity or fluctuation energy.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    with plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for key in ("initial", "boundary", "residual"):
            axes[0, 0].semilogy(history["step"], history[key], label=key)
        axes[0, 0].set(title="Training losses", xlabel="Adam step", ylabel="Mean squared loss")
        for i, label in enumerate(("u", "v", "p (centered)")):
            axes[0, 1].plot(times, 100*rel[:, i], label=label)
        axes[0, 1].set(title="Full-field relative error (includes IC/BC points)", xlabel="Time", ylabel="Relative L2 error (%)")
        axes[1, 0].plot(times, vrms_ref, color="#0072B2", label="Reference")
        axes[1, 0].plot(times, vrms_pred, color="#D55E00", linestyle="--", label="PINN")
        axes[1, 0].set(title="Transverse velocity amplitude", xlabel="Time", ylabel="Spatial RMS of v", ylim=(0, None))
        centers = [b["time_start"]+2.5 for b in residual_bins]
        for key, label, style in (("train_mse", "Training collocation points", "o-"),
                                   ("heldout_mse", "Held-out interior points", "s--")):
            axes[1, 1].semilogy(centers, [b[key] for b in residual_bins], style, label=label)
        axes[1, 1].set(title="Momentum residual, 5-unit time bins", xlabel="Time-bin center", ylabel="Mean(f squared + g squared)")
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle("Forward PINN diagnosis | 50,000 Adam steps | fixed coefficients")
        fig.savefig(output / "diagnosis.png", dpi=150)
        plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
