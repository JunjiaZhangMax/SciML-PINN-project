"""TODO: figures backed by saved experiment outputs."""

from pathlib import Path


def plot_run(run_dir: Path, output_dir: Path) -> None:
    """TODO: sampling, loss/coefficient curves, and three-time field comparisons.

    Save PNG/SVG plus provenance; common truth/prediction color limits, centered
    pressure, explicit units and separate error scales. No synthetic results.
    """
    raise NotImplementedError("Implement single-run figures in sciml/plotting.py")


def plot_summary(runs_dir: Path, output_dir: Path) -> None:
    """TODO: noise/sparsity means +/- sample SD with individual seed points.

    Exclude smoke/pilot runs; do not pool protocols/budgets. Explicitly report
    incomplete/failed runs, actual n, and the definition of uncertainty.
    """
    raise NotImplementedError("Implement sensitivity/statistics figures")
