"""Small artifact helpers shared by training and evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_arrays(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_checkpoint(path: Path, device="cpu") -> dict:
    # These are trusted local checkpoints created by this project's trainer.
    return torch.load(path, map_location=device, weights_only=False)
