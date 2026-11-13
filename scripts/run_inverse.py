"""Preview or train a grid of inverse runs using one base config."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciml.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("configs/pilot_inverse_n5000_noise0_seed0.json"))
    parser.add_argument("--observations", nargs="+", type=int, required=True)
    parser.add_argument("--noise", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, help="Override the Adam budget in the base config")
    parser.add_argument("--runs", type=Path, default=Path("runs/inverse_grid"))
    parser.add_argument("--run", action="store_true", help="Start training; default is preview only")
    args = parser.parse_args()
    base = Config.load(args.base)
    if base.mode != "inverse":
        parser.error("Choose a base config with mode=inverse")
    configs = []
    for count in dict.fromkeys(args.observations):
        for noise in dict.fromkeys(args.noise):
            for seed in dict.fromkeys(args.seeds):
                config = replace(base, name=f"n{count}_noise{noise}_seed{seed}",
                                 group="sensitivity", n_observations=count, noise=noise,
                                 seed=seed, observation_seed=seed, noise_seed=10000 + seed,
                                 collocation="fixed",
                                 adam_steps=base.adam_steps if args.steps is None else args.steps)
                configs.append(config.validate())
    print(f"{len(configs)} runs, {configs[0].adam_steps} Adam steps each, "
          f"{base.n_collocation} fixed PDE points; output: {args.runs}")
    for config in configs:
        print(config.name)
    if not args.run:
        return

    from sciml.train import train
    # Check all destinations before starting the first job.
    for config in configs:
        folder = args.runs / config.name
        if folder.exists():
            if not (folder / "config.json").is_file() or Config.load(folder / "config.json") != config:
                parser.error(f"Conflicting config in {folder}; choose a new --runs directory")
            if not (folder / "checkpoint.pt").is_file():
                parser.error(f"No checkpoint in {folder}; choose a new --runs directory")
    rows = []
    for config in configs:
        folder = args.runs / config.name
        status = folder / "status.json"
        complete = status.exists() and json.loads(status.read_text())["state"] == "completed"
        if not complete or not (folder / "metrics.json").exists():
            train(config, runs_dir=args.runs, resume=folder.exists())
        metrics = json.loads((folder / "metrics.json").read_text())
        row = dict(name=config.name, observations=config.n_observations, noise=config.noise, seed=config.seed)
        for key in ("adam_steps", "lambda1", "lambda2", "relative_error_lambda1", "relative_error_lambda2",
                    "relative_l2_u", "relative_l2_v", "relative_l2_pressure_centered", "training_seconds"):
            row[key] = metrics[key]
        rows.append(row)
    summary = args.runs / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(summary)


if __name__ == "__main__":
    main()
