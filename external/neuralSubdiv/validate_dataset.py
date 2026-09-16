"""Validate Neural Subdivision PKL chains before expensive training."""

import argparse
import json
import os
import pickle
from types import SimpleNamespace

import numpy as np

from mesh_metrics import (
    core_frame_metrics,
    mesh_quality_metrics,
    midpoint_subdivision_faces,
    topology_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl", help="training, validation, or test PKL")
    parser.add_argument("--output", help="JSON report path")
    parser.add_argument("--max-meshes", type=int, default=0)
    parser.add_argument("--max-narrow-ratio", type=float, default=0.1)
    parser.add_argument("--prefix-tolerance", type=float, default=1e-6)
    parser.add_argument("--min-core-edge-ratio", type=float, default=1e-5)
    parser.add_argument("--min-core-face-area-ratio", type=float, default=1e-10)
    parser.add_argument("--min-core-normal-sum", type=float, default=1e-3)
    parser.add_argument("--allow-boundary", action="store_true")
    return parser.parse_args()


def validation_settings(max_narrow_ratio=0.1, prefix_tolerance=1e-6,
                        min_core_edge_ratio=1e-5,
                        min_core_face_area_ratio=1e-10,
                        min_core_normal_sum=1e-3, allow_boundary=False):
    return SimpleNamespace(
        max_narrow_ratio=max_narrow_ratio,
        prefix_tolerance=prefix_tolerance,
        min_core_edge_ratio=min_core_edge_ratio,
        min_core_face_area_ratio=min_core_face_area_ratio,
        min_core_normal_sum=min_core_normal_sum,
        allow_boundary=allow_boundary,
    )


def validate_chain(chain, chain_index, args):
    failures = []
    levels = []
    finest_vertices = chain[-1].V.detach().cpu().numpy()
    for level_index, mesh in enumerate(chain):
        quality = mesh_quality_metrics(mesh.V, mesh.F)
        quality.update(core_frame_metrics(
            mesh.V, mesh.F, args.min_core_edge_ratio,
            args.min_core_face_area_ratio, args.min_core_normal_sum))
        # Only level 0 is supplied as geometry to the autoregressive model.
        # Later frames are built from predictions, not from the GT levels.
        quality["core_frame_used_by_model"] = level_index == 0
        quality["level"] = level_index
        current_vertices = mesh.V.detach().cpu().numpy()
        if (current_vertices.size > 0 and
                current_vertices.shape[0] <= finest_vertices.shape[0]):
            prefix_error = float(np.max(np.abs(
                current_vertices - finest_vertices[:current_vertices.shape[0]])))
        else:
            prefix_error = None
        quality["finest_prefix_max_error"] = prefix_error
        levels.append(quality)

        prefix = "chain %d level %d" % (chain_index, level_index)
        if not quality["finite_vertices"]:
            failures.append(prefix + ": non-finite vertices")
        if not quality["valid_face_indices"]:
            failures.append(prefix + ": invalid face index")
        if quality["nonmanifold_edges"] != 0:
            failures.append(prefix + ": non-manifold edges")
        if quality["inconsistent_oriented_edges"] != 0:
            failures.append(prefix + ": inconsistent face orientation")
        if not args.allow_boundary and quality["boundary_edges"] != 0:
            failures.append(prefix + ": boundary edges")
        if quality["connected_components"] != 1:
            failures.append(prefix + ": disconnected topology")
        if quality["isolated_vertices"] != 0:
            failures.append(prefix + ": isolated vertices")
        if quality["degenerate_face_ratio"] > 0.0:
            failures.append(prefix + ": degenerate faces")
        if quality["narrow_face_ratio"] > args.max_narrow_ratio:
            failures.append(prefix + ": excessive narrow faces")
        if (quality["core_frame_used_by_model"] and
                quality["ill_conditioned_core_frame_ratio"] > 0.0):
            failures.append(prefix + ": ill-conditioned Neural Subdivision frame")
        if prefix_error is None or prefix_error > args.prefix_tolerance:
            failures.append(prefix + ": target prefix correspondence mismatch")

        if level_index > 0:
            previous = chain[level_index - 1]
            previous_topology = topology_metrics(
                previous.F, previous.V.size(0))
            expected_vertices = (
                previous.V.size(0) + previous_topology["unique_edges"])
            if mesh.F.size(0) != previous.F.size(0) * 4:
                failures.append(prefix + ": face count is not 4x previous level")
            if mesh.V.size(0) != expected_vertices:
                failures.append(prefix + ": vertex count does not match edge splits")
            try:
                expected_faces = midpoint_subdivision_faces(
                    previous.F, previous.V.size(0))
                actual_faces = mesh.F.detach().cpu().numpy()
                if not np.array_equal(actual_faces, expected_faces):
                    failures.append(
                        prefix + ": face connectivity/order is not official midpoint subdivision")
            except ValueError:
                failures.append(prefix + ": previous level has invalid faces")
    return {"chain": chain_index, "levels": levels, "failures": failures}


def require_valid_dataset(dataset, label, expected_subdivisions,
                          settings=None):
    if dataset.nM <= 0:
        raise ValueError("%s dataset contains no mesh chains" % label)
    settings = settings or validation_settings()
    failures = []
    for chain_index, chain in enumerate(dataset.meshes):
        if len(chain) != expected_subdivisions + 1:
            failures.append(
                "chain %d: expected %d levels, found %d" %
                (chain_index, expected_subdivisions + 1, len(chain)))
            continue
        failures.extend(validate_chain(
            chain, chain_index, settings)["failures"])
    if failures:
        preview = "\n".join(failures[:20])
        raise ValueError(
            "%s dataset failed pretraining validation (%d failures):\n%s" %
            (label, len(failures), preview))
    print("validated %s dataset: %d chains" % (label, dataset.nM), flush=True)


def main():
    args = parse_args()
    with open(args.pkl, "rb") as handle:
        dataset = pickle.load(handle)
    count = dataset.nM if args.max_meshes <= 0 else min(
        dataset.nM, args.max_meshes)
    settings = validation_settings(
        args.max_narrow_ratio, args.prefix_tolerance,
        args.min_core_edge_ratio, args.min_core_face_area_ratio,
        args.min_core_normal_sum, args.allow_boundary)
    chains = [validate_chain(dataset.meshes[index], index, settings)
              for index in range(count)]
    failures = [failure for chain in chains for failure in chain["failures"]]
    report = {
        "pkl": os.path.abspath(args.pkl),
        "chains_checked": count,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "chains": chains,
        "note": "Self-intersection requires a separate CGAL or visual check.",
    }
    output = args.output or os.path.splitext(args.pkl)[0] + "_validation.json"
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("checked %d chains: %s" % (
        count, "PASS" if report["passed"] else "FAIL"))
    print("report: %s" % output)
    if failures:
        for failure in failures[:20]:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
