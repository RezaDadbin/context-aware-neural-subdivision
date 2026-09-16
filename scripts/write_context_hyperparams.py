#!/usr/bin/env python3
"""Write a context-model job without selecting a dataset implicitly."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="job name")
    parser.add_argument("--train-pkl", required=True)
    parser.add_argument("--valid-pkl", required=True)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--context-lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"),
                        default="cpu")
    parser.add_argument("--num-subd", type=int, default=2)
    parser.add_argument("--local-layers", type=int, default=6)
    parser.add_argument("--global-layers", type=int, default=2)
    parser.add_argument("--global-chunk-size", type=int, default=256)
    parser.add_argument("--core-checkpoint")
    args = parser.parse_args()

    if "/" in args.name or "\\" in args.name:
        parser.error("name must not contain path separators")
    if args.local_layers < 0 or args.global_layers < 0:
        parser.error("attention layer counts must be non-negative")
    if args.global_chunk_size < 0:
        parser.error("global chunk size must be non-negative")

    root = Path(__file__).resolve().parents[1]
    neural = root / "external" / "neuralSubdiv"
    job = neural / "jobs" / ("net_context_" + args.name)
    job.mkdir(parents=True, exist_ok=True)

    data = {
        "train_pkl": str(Path(args.train_pkl).expanduser().resolve()),
        "valid_pkl": str(Path(args.valid_pkl).expanduser().resolve()),
        "output_path": "./jobs/net_context_%s/" % args.name,
        "epochs": args.epochs,
        "lr": args.lr,
        "context_lr": args.context_lr,
        "seed": args.seed,
        "device": args.device,
        "Din": 6,
        "Dout": 32,
        "h_initNet": [32, 32],
        "h_edgeNet": [32, 32],
        "h_vertexNet": [32, 32],
        "numSubd": args.num_subd,
        "context_dim": 64,
        "context_hidden_dim": 64,
        "context_heads": 4,
        "context_local_layers": args.local_layers,
        "context_global_layers": args.global_layers,
        "context_feed_forward_dim": 128,
        "context_dropout": 0.0,
        "context_global_chunk_size": args.global_chunk_size,
        "context_geometry_eps": 1e-8,
    }
    if args.core_checkpoint:
        data["core_checkpoint"] = str(
            Path(args.core_checkpoint).expanduser().resolve())

    path = job / "hyperparameters.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    print("Wrote", path)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
