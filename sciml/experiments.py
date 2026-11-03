"""TODO: staged execution and baseline gates. Definitions live in config.py."""

from pathlib import Path

from .data import DEFAULT_DATA


def run_batch(config_dir: Path, data_path: Path = DEFAULT_DATA,
              runs_dir: Path = Path("runs"), stage: str = "known") -> None:
    """TODO: known -> reference -> sensitivity; preserve failures and logs.

    Record baseline review/criteria before sensitivity. Respect manifest order;
    only reuse completed runs after config/data identity checks. No automatic
    advancement past an unverified baseline.
    """
    raise NotImplementedError("Implement staged batch execution in sciml/experiments.py")
