"""TODO: checkpoint evaluation and gauge-aware metrics."""

from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """TODO: norm(prediction-reference)/norm(reference); handle zero norm explicitly."""
    raise NotImplementedError("Implement relative L2 error")


def pressure_error(prediction: np.ndarray, reference: np.ndarray,
                   time_indices: np.ndarray) -> float:
    """TODO: center prediction/reference separately per time group, then relative L2.

    Adding arbitrary c(t) must not change the metric. A zero denominator is
    undefined, not a zero error. Center over spatial test samples at each time.
    """
    raise NotImplementedError("Implement gauge-invariant pressure error")


def evaluate(run_dir: Path, data_path: Path = DEFAULT_DATA) -> dict:
    """TODO: reload config/indices/checkpoint; save metrics and predictions.

    No resampling or retraining. Separate held-out metrics from full-domain
    illustrative snapshots at the three fixed time indices.
    """
    raise NotImplementedError("Implement checkpoint evaluation in sciml/evaluate.py")
