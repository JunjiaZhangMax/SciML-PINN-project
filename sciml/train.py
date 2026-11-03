"""TODO: optimizers, logging, checkpoint/resume and failure records."""

from pathlib import Path

from .config import Config
from .data import DEFAULT_DATA


def train(config: Config, data_path: Path = DEFAULT_DATA,
          runs_dir: Path = Path("runs"), resume: bool = False) -> Path:
    """TODO: prepare_data -> PINN -> objective -> Adam/L-BFGS; return run directory.

    Save artifacts per docs/DESIGN.md. Refuse accidental overwrites. Resume must
    validate config/data identity. Never select checkpoints/budgets using test
    labels or pressure labels. Record exceptions and interruptions explicitly.
    """
    raise NotImplementedError("Implement training in sciml/train.py")
