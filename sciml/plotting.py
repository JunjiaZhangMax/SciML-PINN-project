"""Figures backed by saved experiment outputs."""

import csv
from pathlib import Path

from .config import Config

REFERENCE_COEFFICIENTS = {"lambda1": 1.0, "lambda2": 0.01}


def plot_run(run_dir: Path, output_dir: Path) -> Path:
    """Plot history.csv: every loss term, and both coefficients for inverse runs.

    Reads saved logs only; nothing is retrained or re-evaluated. Field
    comparisons are produced by scripts/animate_comparison.py.
    TODO: sampling map and three-time field/error panels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    config = Config.load(run_dir / "config.json")
    with open(run_dir / "history.csv", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No logged steps in {run_dir / 'history.csv'}")
    steps = [int(row["step"]) for row in rows]
    skipped = {"stage", "step", "elapsed_seconds", *REFERENCE_COEFFICIENTS}
    series = {name: [float(row[name]) for row in rows] for name in rows[0] if name not in skipped}
    inverse = config.mode == "inverse"
    fig, axes = plt.subplots(1, 3 if inverse else 1, figsize=(15 if inverse else 6, 4.2),
                             layout="constrained", squeeze=False)
    loss_axis = axes[0, 0]
    for name, values in series.items():
        # Terms that are not part of this objective are logged as exact zeros.
        if any(value > 0 for value in values):
            loss_axis.plot(steps, values, label=name, linewidth=2 if name == "total" else 1)
    loss_axis.set(yscale="log", xlabel="Adam step", ylabel="loss",
                  title="Training losses (terms unweighted, total weighted)")
    loss_axis.legend()
    if inverse:
        for axis, (name, reference) in zip(axes[0, 1:], REFERENCE_COEFFICIENTS.items()):
            values = [float(row[name]) for row in rows]
            axis.plot(steps, values, label="estimate")
            axis.axhline(reference, color="black", linestyle="--", linewidth=1,
                         label=f"reference {reference:g}")
            error = abs(values[-1] - reference) / reference
            axis.set(xlabel="Adam step", ylabel=name,
                     title=f"{name}: final {values[-1]:.5g} ({100 * error:.2f}% error)")
            axis.legend()
    fig.suptitle(f"{config.name} | {config.purpose} | step {steps[-1]} of {config.adam_steps}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{config.name}_history.png"
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path
