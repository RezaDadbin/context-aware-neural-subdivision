#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write NeuralSubdiv hyperparameters.json for one style.")
    parser.add_argument("style", help="Style name, e.g. bunny or gear_16t")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-subd", type=int, default=2)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--train-pkl")
    parser.add_argument("--valid-pkl")
    args = parser.parse_args()

    if bool(args.train_pkl) != bool(args.valid_pkl):
        parser.error("--train-pkl and --valid-pkl must be provided together")

    root = Path(__file__).resolve().parents[1]
    neural = root / "external/neuralSubdiv"
    job = neural / "jobs" / f"net_style_{args.style}"
    job.mkdir(parents=True, exist_ok=True)

    train_pkl = (str(Path(args.train_pkl).expanduser().resolve())
                 if args.train_pkl else
                 f"./data_PKL/style_{args.style}_train.pkl")
    valid_pkl = (str(Path(args.valid_pkl).expanduser().resolve())
                 if args.valid_pkl else
                 f"./data_PKL/style_{args.style}_valid.pkl")
    data = {
        "train_pkl": train_pkl,
        "valid_pkl": valid_pkl,
        "output_path": f"./jobs/net_style_{args.style}/",
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "Din": 6,
        "Dout": 32,
        "h_initNet": [32, 32],
        "h_edgeNet": [32, 32],
        "h_vertexNet": [32, 32],
        "numSubd": args.num_subd,
    }
    if args.initial_checkpoint:
        data["initial_checkpoint"] = str(
            Path(args.initial_checkpoint).expanduser().resolve())

    path = job / "hyperparameters.json"
    path.write_text(json.dumps(data, indent=2))
    print("Wrote", path)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
