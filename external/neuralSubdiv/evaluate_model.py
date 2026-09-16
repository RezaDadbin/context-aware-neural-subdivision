"""Evaluate Phase 0 or context-aware models on held-out PKL chains."""

import argparse
import json
import os
import pickle

import numpy as np
import torch

from context_models import ContextSubdNet
from mesh_metrics import prediction_metrics
from models import SubdNet
from train_resume import NETPARAMS, torch_load
from validate_dataset import require_valid_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", help="job folder containing hyperparameters.json")
    parser.add_argument("pkl", help="held-out PKL chains")
    parser.add_argument("--model", choices=("context", "phase0"),
                        default="context")
    parser.add_argument("--checkpoint", help="override the job netparams.dat")
    parser.add_argument("--modes", nargs="+",
                        choices=("context", "disabled", "shuffled"),
                        default=("context",))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--surface-samples", type=int, default=10000)
    parser.add_argument("--max-meshes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    return parser.parse_args()


def fixed_rotation(dtype, device):
    axis = torch.tensor([1.0, 2.0, -0.5], dtype=dtype, device=device)
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.73, dtype=dtype, device=device)
    cross = torch.stack((
        torch.stack((axis[0] * 0.0, -axis[2], axis[1])),
        torch.stack((axis[2], axis[0] * 0.0, -axis[0])),
        torch.stack((-axis[1], axis[0], axis[0] * 0.0)),
    ))
    identity = torch.eye(3, dtype=dtype, device=device)
    return (identity * torch.cos(angle) +
            (1.0 - torch.cos(angle)) * axis[:, None] * axis[None, :] +
            torch.sin(angle) * cross)


def run_model(model, dataset, mesh_index, model_type, mode):
    inputs = dataset.getInputData(mesh_index)
    if model_type == "phase0":
        return model(inputs, mesh_index, dataset.hfList,
                     dataset.poolMats, dataset.dofs)
    return model(
        inputs, mesh_index, dataset.hfList, dataset.poolMats, dataset.dofs,
        context_enabled=mode != "disabled", shuffle_context=mode == "shuffled")


def rotation_errors(model, dataset, mesh_index, model_type, mode, reference):
    inputs = dataset.getInputData(mesh_index).clone()
    rotation = fixed_rotation(inputs.dtype, inputs.device)
    translation = torch.tensor(
        [[0.6, -1.2, 2.3]], dtype=inputs.dtype, device=inputs.device)
    inputs[:, :3] = inputs[:, :3].mm(rotation.t()) + translation
    inputs[:, 3:] = inputs[:, 3:].mm(rotation.t())
    if model_type == "phase0":
        moved = model(inputs, mesh_index, dataset.hfList,
                      dataset.poolMats, dataset.dofs)
    else:
        moved = model(
            inputs, mesh_index, dataset.hfList, dataset.poolMats, dataset.dofs,
            context_enabled=mode != "disabled",
            shuffle_context=mode == "shuffled")
    errors = []
    for original, transformed in zip(reference, moved):
        restored = (transformed - translation).mm(rotation)
        difference = restored - original
        errors.append({
            "rmse": float(torch.sqrt(difference.square().mean()).cpu()),
            "maximum": float(torch.linalg.vector_norm(
                difference, dim=1).max().cpu()),
        })
    return errors


def aggregate(records):
    by_mode_level = {}
    excluded = {"vertices", "faces", "unique_edges", "boundary_edges",
                "nonmanifold_edges", "connected_components",
                "euler_characteristic", "surface_samples",
                "smooth_region_vertices", "detail_region_vertices"}
    for record in records:
        key = "%s_level_%d" % (record["mode"], record["level"])
        destination = by_mode_level.setdefault(key, {})
        for name, value in record["metrics"].items():
            if name not in excluded and isinstance(value, (int, float)):
                destination.setdefault(name, []).append(value)
    return {
        key: {name: float(np.mean(values)) for name, values in metrics.items()}
        for key, metrics in by_mode_level.items()
    }


def main():
    args = parse_args()
    job = args.job if args.job.endswith(os.sep) else args.job + os.sep
    with open(job + "hyperparameters.json", "r") as handle:
        params = json.load(handle)
    if args.device:
        params["device"] = args.device
    device = params["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")

    with open(args.pkl, "rb") as handle:
        dataset = pickle.load(handle)
    require_valid_dataset(dataset, "evaluation", params["numSubd"])
    dataset.computeParameters()
    dataset.toDevice(device)
    model = (ContextSubdNet(params) if args.model == "context"
             else SubdNet(params)).to(device)
    checkpoint = args.checkpoint or job + NETPARAMS
    model.load_state_dict(torch_load(checkpoint, device))
    model.eval()

    modes = args.modes if args.model == "context" else ("phase0",)
    count = dataset.nM if args.max_meshes <= 0 else min(
        dataset.nM, args.max_meshes)
    records = []
    rotations = []
    with torch.no_grad():
        for mode_index, mode in enumerate(modes):
            for mesh_index in range(count):
                mode_seed = args.seed + mode_index * 100000 + mesh_index
                torch.manual_seed(mode_seed)
                outputs = run_model(
                    model, dataset, mesh_index, args.model, mode)
                target_all = dataset.meshes[mesh_index][params["numSubd"]].V
                for level, output in enumerate(outputs):
                    target = target_all[:output.size(0)]
                    faces = dataset.meshes[mesh_index][level].F
                    metrics = prediction_metrics(
                        output, target, faces, args.surface_samples,
                        args.seed + mesh_index * 100 + level)
                    records.append({
                        "mode": mode,
                        "mesh": mesh_index,
                        "level": level,
                        "metrics": metrics,
                    })
                if mesh_index == 0:
                    torch.manual_seed(mode_seed)
                    rotations.append({
                        "mode": mode,
                        "levels": rotation_errors(
                            model, dataset, mesh_index, args.model, mode,
                            outputs),
                    })

    report = {
        "model": args.model,
        "job": os.path.abspath(job),
        "checkpoint": os.path.abspath(checkpoint),
        "pkl": os.path.abspath(args.pkl),
        "device": device,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "meshes_evaluated": count,
        "surface_samples_per_mesh": args.surface_samples,
        "aggregate": aggregate(records),
        "rotation_equivariance": rotations,
        "records": records,
    }
    output = args.output or os.path.join(job, "evaluation.json")
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("evaluated %d meshes in modes: %s" % (count, ", ".join(modes)))
    print("report: %s" % output)


if __name__ == "__main__":
    main()
