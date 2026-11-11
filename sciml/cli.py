"""Project CLI for configurations, training, evaluation, and remaining scaffolds."""

import argparse
import json
from pathlib import Path

from .config import Config, suite


def main(argv=None):
    parser = argparse.ArgumentParser(description="SciML cylinder PINN training and evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="List 25 configurations without executing them")
    export = commands.add_parser("init-configs", help="Export JSON configs; refuse overwrites")
    export.add_argument("--output", type=Path, default=Path("configs/generated"))
    download = commands.add_parser("download", help="Fetch the reference MAT and verify its SHA256")
    download.add_argument("--data", type=Path, default=Path("data/raw/cylinder_nektar_wake.mat"))
    train = commands.add_parser("train", help="Train one experiment; save logs, checkpoint, and evaluation")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--data", type=Path, default=Path("data/raw/cylinder_nektar_wake.mat"))
    train.add_argument("--runs", type=Path, default=Path("runs"))
    train.add_argument("--resume", action="store_true", help="Continue from the last Adam checkpoint")
    evaluate = commands.add_parser("evaluate", help="Re-evaluate a saved checkpoint on its fixed test set")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, default=Path("data/raw/cylinder_nektar_wake.mat"))
    plot = commands.add_parser("plot", help="Loss/coefficient curves for --run; --runs summary is TODO")
    source = plot.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--runs", type=Path)
    plot.add_argument("--output", type=Path, default=Path("output/figures"))
    batch = commands.add_parser("batch", help="TODO: staged experiment execution")
    batch.add_argument("--configs", type=Path, default=Path("configs/generated"))
    batch.add_argument("--stage", choices=["known", "reference", "sensitivity"], default="known")
    batch.add_argument("--data", type=Path, default=Path("data/raw/cylinder_nektar_wake.mat"))
    batch.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            for index, config in enumerate(suite(), 1):
                print(f"{index:02d}. {config.name} | {config.mode} | PDE: {config.collocation}")
            print("25 configurations. No training has been started.")
        elif args.command == "init-configs":
            configs = suite()
            paths = [args.output / f"{c.name}.json" for c in configs]
            manifest = args.output / "manifest.json"
            if any(p.exists() for p in [*paths, manifest]):
                parser.error("Output exists; choose a new configuration directory.")
            args.output.mkdir(parents=True, exist_ok=True)
            for config, path in zip(configs, paths):
                config.save(path)
            manifest.write_text(json.dumps({"order": [p.name for p in paths]}, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {len(configs)} configurations and manifest to {args.output}")
        elif args.command == "download":
            from .data import download
            download(args.data)
        elif args.command == "train":
            from .train import train
            print(train(Config.load(args.config), args.data, args.runs, resume=args.resume))
        elif args.command == "evaluate":
            from .evaluate import evaluate
            print(json.dumps(evaluate(args.run, args.data), indent=2))
        elif args.command == "plot":
            from .plotting import plot_run, plot_summary
            if args.run:
                print(plot_run(args.run, args.output))
            else:
                plot_summary(args.runs, args.output)
        elif args.command == "batch":
            from .experiments import run_batch
            run_batch(args.configs, args.data, args.runs, args.stage)
    except NotImplementedError as exc:
        parser.exit(2, f"TODO: {exc}\n")
