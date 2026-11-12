"""Download, train, evaluate, and plot a cylinder-wake PINN."""

import argparse
import json
from pathlib import Path

from .config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description="SciML cylinder PINN training and evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
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
    plot = commands.add_parser("plot", help="Plot losses and inverse coefficients for one run")
    plot.add_argument("--run", type=Path, required=True)
    plot.add_argument("--output", type=Path, default=Path("output/figures"))
    args = parser.parse_args(argv)
    if args.command == "download":
        from .data import download
        download(args.data)
    elif args.command == "train":
        from .train import train
        print(train(Config.load(args.config), args.data, args.runs, resume=args.resume))
    elif args.command == "evaluate":
        from .evaluate import evaluate
        print(json.dumps(evaluate(args.run, args.data), indent=2))
    elif args.command == "plot":
        from .plotting import plot_run
        print(plot_run(args.run, args.output))
