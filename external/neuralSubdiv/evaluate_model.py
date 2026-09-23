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


CONTEXT_MODES = (
    "context", "disabled", "shuffled", "uniform",
    "fixed_1", "fixed_2", "fixed_3", "fixed_4",
    "fixed_5", "fixed_6", "fixed_7", "fixed_8",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", help="job folder containing hyperparameters.json")
    parser.add_argument("pkl", help="held-out PKL chains")
    parser.add_argument("--model", choices=("context", "phase0"),
                        default="context")
    parser.add_argument("--checkpoint", help="override the job netparams.dat")
    parser.add_argument("--modes", nargs="+",
                        choices=CONTEXT_MODES,
                        default=("context",))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--surface-samples", type=int, default=10000)
    parser.add_argument("--max-meshes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument(
        "--selector-output",
        help="optional compressed NPZ containing learned per-vertex weights")
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


def selector_mode(mode):
    if mode == "uniform" or mode.startswith("fixed_"):
        return mode
    return "learned"


def run_model(model, dataset, mesh_index, model_type, mode,
              return_selector_weights=False):
    inputs = dataset.getInputData(mesh_index)
    if model_type == "phase0":
        return model(inputs, mesh_index, dataset.hfList,
                     dataset.poolMats, dataset.dofs)
    return model(
        inputs, mesh_index, dataset.hfList, dataset.poolMats, dataset.dofs,
        context_enabled=mode != "disabled", shuffle_context=mode == "shuffled",
        local_context_mode=selector_mode(mode),
        return_selector_weights=return_selector_weights)


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
            shuffle_context=mode == "shuffled",
            local_context_mode=selector_mode(mode))
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


def selector_metrics(weights):
    scale_count = weights.size(1)
    radii = torch.arange(
        1, scale_count + 1, dtype=weights.dtype, device=weights.device)
    effective_radius = (weights * radii.unsqueeze(0)).sum(dim=1)
    safe_weights = weights.clamp_min(1e-12)
    entropy = -(safe_weights * safe_weights.log()).sum(dim=1)
    if scale_count > 1:
        entropy = entropy / np.log(float(scale_count))
    dominant = torch.argmax(weights, dim=1)
    fractions = torch.bincount(
        dominant, minlength=scale_count).to(weights.dtype) / weights.size(0)
    return {
        "vertices": int(weights.size(0)),
        "mean_weights": weights.mean(dim=0).cpu().tolist(),
        "std_weights": weights.std(dim=0, unbiased=False).cpu().tolist(),
        "mean_effective_radius": float(effective_radius.mean().cpu()),
        "std_effective_radius": float(
            effective_radius.std(unbiased=False).cpu()),
        "mean_normalized_entropy": float(entropy.mean().cpu()),
        "dominant_radius_fractions": fractions.cpu().tolist(),
    }


def aggregate_selector(records):
    grouped = {}
    for record in records:
        key = "%s_stage_%d" % (record["mode"], record["stage"])
        grouped.setdefault(key, []).append(record["metrics"])

    result = {}
    for key, values in grouped.items():
        result[key] = {
            "meshes": len(values),
            "vertices_per_mesh_mean": float(np.mean([
                value["vertices"] for value in values])),
            "mean_weights": np.mean([
                value["mean_weights"] for value in values], axis=0).tolist(),
            "mean_weight_std_within_mesh": np.mean([
                value["std_weights"] for value in values], axis=0).tolist(),
            "mean_effective_radius": float(np.mean([
                value["mean_effective_radius"] for value in values])),
            "mean_effective_radius_std_within_mesh": float(np.mean([
                value["std_effective_radius"] for value in values])),
            "mean_normalized_entropy": float(np.mean([
                value["mean_normalized_entropy"] for value in values])),
            "dominant_radius_fractions": np.mean([
                value["dominant_radius_fractions"] for value in values],
                axis=0).tolist(),
        }
    return result


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
    selector_records = []
    selector_arrays = {}
    with torch.no_grad():
        for mode_index, mode in enumerate(modes):
            for mesh_index in range(count):
                mode_seed = args.seed + mode_index * 100000 + mesh_index
                torch.manual_seed(mode_seed)
                collect_selector = (
                    args.model == "context" and mode != "disabled" and
                    model.context_block.adaptive_local)
                result = run_model(
                    model, dataset, mesh_index, args.model, mode,
                    return_selector_weights=collect_selector)
                if collect_selector:
                    outputs, selector_history = result
                    for stage, weights in enumerate(selector_history):
                        if args.selector_output and mode == "context":
                            key = "mesh_%03d_stage_%d" % (mesh_index, stage)
                            selector_arrays[key] = weights.cpu().numpy()
                        selector_records.append({
                            "mode": mode,
                            "mesh": mesh_index,
                            "stage": stage,
                            "metrics": selector_metrics(weights),
                        })
                else:
                    outputs = result
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
        "selector_aggregate": aggregate_selector(selector_records),
        "selector_weights_file": (
            os.path.abspath(args.selector_output)
            if args.selector_output else None),
        "rotation_equivariance": rotations,
        "selector_records": selector_records,
        "records": records,
    }
    output = args.output or os.path.join(job, "evaluation.json")
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if args.selector_output:
        np.savez_compressed(args.selector_output, **selector_arrays)
        print("selector weights: %s" % args.selector_output)
    print("evaluated %d meshes in modes: %s" % (count, ", ".join(modes)))
    print("report: %s" % output)


if __name__ == "__main__":
    main()
