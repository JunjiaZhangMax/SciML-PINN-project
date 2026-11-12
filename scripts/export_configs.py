"""Export the 25 planned experiments when needed. Run from the repository root."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciml.config import suite


def main():
    directory = Path("configs/generated")
    configs = suite()
    names = [f"{c.name}.json" for c in configs]
    if any((directory / name).exists() for name in [*names, "manifest.json"]):
        raise SystemExit("Output exists in configs/generated; existing configs were kept.")
    directory.mkdir(parents=True, exist_ok=True)
    for config, name in zip(configs, names):
        config.save(directory / name)
    (directory / "manifest.json").write_text(json.dumps({"order": names}, indent=2), encoding="utf-8")
    print(f"Saved {len(configs)} configs to {directory}")


if __name__ == "__main__":
    main()
