#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and validate NeuralSubdiv train/valid/test PKLs.")
    parser.add_argument("style", help="Style name, e.g. bunny or gear_16t")
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-valid", type=int, default=20)
    parser.add_argument("--num-test", type=int, default=20)
    parser.add_argument(
        "--smoke-meshes", type=int, default=3,
        help="meshes per split used for Phase 0/Phase 1 gradient preflight")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    neural = root / "external/neuralSubdiv"
    os.chdir(neural)
    sys.path.insert(0, str(neural))

    from include import TrainMeshes

    train_folder = f"./data_meshes/style_{args.style}_{args.num_train}/"
    valid_folder = f"./data_meshes/style_{args.style}_valid_{args.num_valid}/"
    test_folder = f"./data_meshes/style_{args.style}_test_{args.num_test}/"
    out_dir = Path("./data_PKL")
    out_dir.mkdir(exist_ok=True)

    print("============================================================")
    print("Creating PKLs")
    print("style:", args.style)
    print("train folder:", train_folder)
    print("valid folder:", valid_folder)
    print("test folder:", test_folder)
    print("============================================================")

    train = TrainMeshes([train_folder])
    valid = TrainMeshes([valid_folder])
    test = TrainMeshes([test_folder])

    train_pkl = out_dir / f"style_{args.style}_train.pkl"
    valid_pkl = out_dir / f"style_{args.style}_valid.pkl"
    test_pkl = out_dir / f"style_{args.style}_test.pkl"
    for path, dataset in ((train_pkl, train), (valid_pkl, valid),
                          (test_pkl, test)):
        with path.open("wb") as handle:
            pickle.dump(dataset, handle)
        subprocess.run([
            sys.executable, "validate_dataset.py", str(path.resolve())],
            check=True)
        smoke_path = path.with_name(path.stem + "_smoke.json")
        subprocess.run([
            sys.executable, "smoke_test_dataset.py", str(path.resolve()),
            "--max-meshes", str(args.smoke_meshes),
            "--output", str(smoke_path.resolve())], check=True)

    print("Wrote", train_pkl)
    print("Wrote", valid_pkl)
    print("Wrote", test_pkl)


if __name__ == "__main__":
    main()
