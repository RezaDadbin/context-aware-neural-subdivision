#!/usr/bin/env python3
"""Run 8K-to-32K recursive inference and render held-out comparisons."""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
NEURAL = ROOT / "external" / "neuralSubdiv"
sys.path.insert(0, str(NEURAL))
sys.path.insert(0, str(ROOT / "scripts"))

from context_models import ContextSubdNet  # noqa: E402
from include import TestMeshes  # noqa: E402
from mesh_metrics import prediction_metrics  # noqa: E402
from models import SubdNet  # noqa: E402
from render_final_comparisons import (  # noqa: E402
    METHOD_COLORS,
    METHODS,
    OBJECTS,
    box_and_crop,
    centered_text,
    choose_improvement_region,
    compose_detail_board,
    compose_full_board,
    font,
    render_mesh,
    transform_vertices,
    write_obj,
)
from train_resume import torch_load  # noqa: E402


STEP4 = ROOT / "artifacts" / "final_visual_comparisons"
OUTPUT = ROOT / "artifacts" / "recursive_32k_comparisons"
GENERATOR_ROOT = ROOT / "external" / "surface_multigrid_code" / "09_random_subdiv_remesh"
GENERATOR = GENERATOR_ROOT / "build" / "random_subdiv_remesh_bin"
SOURCES = {
    "bimba_head": ROOT / "data_candidates" / "compact_context" / "01_bimba_head.obj",
    "fat_dragon": ROOT / "data_candidates" / "compact_context" / "02_fat_dragon.obj",
    "gear16": ROOT / "data_candidates" / "compact_context" / "03_gear16.obj",
}
LABELS_32K = {
    "coarse": "Coarse input (500 faces)",
    "phase0": "Phase 0 recursive (32K)",
    "phase1": "Phase 1 recursive (32K)",
    "gt": "Ground truth (32K)",
}
PROGRESSION = (
    ("coarse", "Coarse\n500"),
    ("phase0_8k", "Phase 0\n8K trained level"),
    ("phase0_32k", "Phase 0\nrecursive 32K"),
    ("phase1_8k", "Phase 1\n8K trained level"),
    ("phase1_32k", "Phase 1\nrecursive 32K"),
    ("gt_32k", "Ground truth\n32K"),
)
PROGRESSION_COLORS = {
    "coarse": METHOD_COLORS["coarse"],
    "phase0_8k": METHOD_COLORS["phase0"],
    "phase0_32k": "#898b90",
    "phase1_8k": METHOD_COLORS["phase1"],
    "phase1_32k": "#ab985e",
    "gt_32k": METHOD_COLORS["gt"],
}


def accepted_test_seed(slug: str, chain_number: int) -> int:
    path = NEURAL / "data_meshes" / f"style_{slug}_seeds.tsv"
    with path.open() as handle:
        header = next(handle)
        if not header.startswith("split\tindex\tseed"):
            raise RuntimeError(f"unexpected seed manifest: {path}")
        for line in handle:
            split, index, seed, status, *_ = line.rstrip("\n").split("\t")
            if split == "test" and status == "accepted" and int(index) == chain_number:
                return int(seed)
    raise RuntimeError(f"no accepted test seed for {slug} chain {chain_number:03d}")


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False, force="mesh")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int64)


def generate_true_32k(slug: str, chain_number: int, destination: Path) -> dict:
    seed = accepted_test_seed(slug, chain_number)
    command = [str(GENERATOR), str(SOURCES[slug]), "500", "3", str(seed)]
    completed = subprocess.run(
        command, cwd=GENERATOR_ROOT / "build", check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    prefix_errors = []
    for level in range(3):
        generated = GENERATOR_ROOT / f"output_s{level}.obj"
        stored = (NEURAL / "data_meshes" / f"style_{slug}_test_20" /
                  f"subd{level}" / f"{chain_number:03d}.obj")
        generated_v, generated_f = load_mesh(generated)
        stored_v, stored_f = load_mesh(stored)
        if generated_v.shape != stored_v.shape or not np.array_equal(generated_f, stored_f):
            raise RuntimeError(f"regenerated topology mismatch: {slug} level {level}")
        error = float(np.max(np.abs(generated_v - stored_v)))
        if error > 1e-12:
            raise RuntimeError(
                f"regenerated vertex mismatch: {slug} level {level}, {error}")
        prefix_errors.append(error)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GENERATOR_ROOT / "output_s3.obj", destination)
    vertices, faces = load_mesh(destination)
    if len(faces) != 32000 or len(vertices) <= 0:
        raise RuntimeError(f"invalid 32K target shape: {destination}")
    return {
        "seed": seed,
        "prefix_max_vertex_errors": prefix_errors,
        "generator_output": completed.stdout.strip(),
    }


def load_recursive_model(job_name: str, context: bool):
    job = NEURAL / "jobs" / job_name
    params = json.loads((job / "hyperparameters.json").read_text())
    params["device"] = "cpu"
    params["numSubd"] = 1
    model = ContextSubdNet(params) if context else SubdNet(params)
    model.load_state_dict(torch_load(str(job / "netparams.dat"), "cpu"))
    model.eval()
    return model


def recursive_inference(model, context: bool, source_8k: Path) -> tuple[np.ndarray, np.ndarray]:
    test = TestMeshes([str(source_8k)], 1)
    test.computeParameters()
    test.toDevice("cpu")
    with torch.no_grad():
        inputs = test.getInputData(0)
        if context:
            outputs = model(inputs, 0, test.hfList, test.poolMats, test.dofs)
        else:
            outputs = model(inputs, 0, test.hfList, test.poolMats, test.dofs)
    vertices = outputs[1].detach().cpu().numpy()
    faces = test.meshes[0][1].F.detach().cpu().numpy().astype(np.int64)
    return vertices, faces


def compose_progression_board(paths: dict[str, Path], output: Path, title: str,
                              subtitle: str) -> None:
    panel = 560
    header = 155
    label_height = 92
    footer = 68
    board = Image.new("RGB", (panel * len(PROGRESSION),
                              header + label_height + panel + footer), "white")
    draw = ImageDraw.Draw(board)
    centered_text(draw, (board.width // 2, 20), title, font(38, True))
    centered_text(draw, (board.width // 2, 76), subtitle, font(24), "#55585d")
    for column, (method, label) in enumerate(PROGRESSION):
        left = column * panel
        draw.rectangle((left, header, left + panel, header + label_height),
                       fill=PROGRESSION_COLORS[method])
        lines = label.split("\n")
        for line_index, line in enumerate(lines):
            centered_text(draw, (left + panel // 2, header + 12 + line_index * 32),
                          line, font(22, True), "#ffffff")
        image = Image.open(paths[method]).convert("RGB").resize(
            (panel, panel), Image.Resampling.LANCZOS)
        board.paste(image, (left, header + label_height))
    centered_text(
        draw, (board.width // 2, board.height - 47),
        "The 32K columns are generated by refeeding each model's own 8K prediction; no 32K training was used.",
        font(21), "#55585d")
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, quality=95)


def render_case(slug: str, settings: dict, source_record: dict,
                phase0_model, phase1_model) -> dict:
    kind = source_record["kind"]
    chain = int(source_record["chain_number"])
    case_name = f"{kind}_chain_{chain:03d}"
    source_case = STEP4 / "all_renders" / slug / case_name
    case_root = OUTPUT / "all_renders" / slug / case_name
    mesh_root = case_root / "meshes"
    image_root = case_root / "individual"
    board_root = case_root / "boards"
    mesh_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    board_root.mkdir(parents=True, exist_ok=True)

    for name in ("coarse", "phase0", "phase1", "gt"):
        shutil.copy2(source_case / "meshes" / f"{name}.obj",
                     mesh_root / f"{name}_8k.obj" if name != "coarse" else
                     mesh_root / "coarse_500.obj")
    gt_info = generate_true_32k(slug, chain, mesh_root / "gt_32k.obj")
    phase0_32k, phase0_faces = recursive_inference(
        phase0_model, False, mesh_root / "phase0_8k.obj")
    phase1_32k, phase1_faces = recursive_inference(
        phase1_model, True, mesh_root / "phase1_8k.obj")
    gt_32k, gt_faces = load_mesh(mesh_root / "gt_32k.obj")
    if not np.array_equal(phase0_faces, gt_faces) or not np.array_equal(phase1_faces, gt_faces):
        raise RuntimeError(f"recursive topology does not match 32K GT for {slug} {chain:03d}")
    write_obj(mesh_root / "phase0_32k.obj", phase0_32k, phase0_faces)
    write_obj(mesh_root / "phase1_32k.obj", phase1_32k, phase1_faces)

    coarse, coarse_faces = load_mesh(mesh_root / "coarse_500.obj")
    phase0_8k, faces_8k = load_mesh(mesh_root / "phase0_8k.obj")
    phase1_8k, phase1_faces_8k = load_mesh(mesh_root / "phase1_8k.obj")
    if not np.array_equal(faces_8k, phase1_faces_8k):
        raise RuntimeError("Phase 0 and Phase 1 8K topology mismatch")
    metrics = {
        "phase0": prediction_metrics(phase0_32k, gt_32k, gt_faces, 50000, chain),
        "phase1": prediction_metrics(phase1_32k, gt_32k, gt_faces, 50000, chain),
    }
    baseline_mse = metrics["phase0"]["correspondence_mse"]
    improvement = 100.0 * (
        baseline_mse - metrics["phase1"]["correspondence_mse"]) / baseline_mse

    render_data = {
        "coarse": coarse,
        "phase0": phase0_32k,
        "phase1": phase1_32k,
        "gt": gt_32k,
    }
    progression_data = {
        "coarse": (coarse, coarse_faces, METHOD_COLORS["coarse"], True),
        "phase0_8k": (phase0_8k, faces_8k, PROGRESSION_COLORS["phase0_8k"], False),
        "phase0_32k": (phase0_32k, phase0_faces, PROGRESSION_COLORS["phase0_32k"], False),
        "phase1_8k": (phase1_8k, phase1_faces_8k, PROGRESSION_COLORS["phase1_8k"], False),
        "phase1_32k": (phase1_32k, phase1_faces, PROGRESSION_COLORS["phase1_32k"], False),
        "gt_32k": (gt_32k, gt_faces, PROGRESSION_COLORS["gt_32k"], False),
    }
    transformed = {
        name: transform_vertices(vertices, settings["y_up"])
        for name, vertices in render_data.items()
    }
    transformed_progression = {
        name: (transform_vertices(values[0], settings["y_up"]), *values[1:])
        for name, values in progression_data.items()
    }
    all_bounds = np.vstack(tuple(transformed.values()))
    center = (all_bounds.min(axis=0) + all_bounds.max(axis=0)) * 0.5
    radius = max(float(np.ptp(all_bounds, axis=0).max()) * 0.51, 1e-6)
    region = choose_improvement_region(
        render_data, gt_faces, settings["y_up"], settings["views"]["primary"])

    images_by_view = {}
    points_by_view = {}
    progression_primary = {}
    for view_name, camera in settings["views"].items():
        images_by_view[view_name] = {}
        points_by_view[view_name] = {}
        for method in METHODS:
            path = image_root / f"{method}_32k_{view_name}.png"
            points_by_view[view_name][method] = render_mesh(
                path, transformed[method],
                coarse_faces if method == "coarse" else gt_faces,
                METHOD_COLORS[method], camera, (center, radius), region["point"],
                show_edges=method == "coarse")
            images_by_view[view_name][method] = path
        compose_full_board(
            images_by_view[view_name], board_root / f"comparison_32k_{view_name}.png",
            f'{settings["label"]}: recursive 32K comparison',
            f'{kind.title()} held-out chain {chain:03d} | Phase 1 MSE improvement: {improvement:.1f}%',
            labels=LABELS_32K)

        if view_name == "primary":
            for method, (vertices, faces, color, edges) in transformed_progression.items():
                path = image_root / f"{method}_progression_primary.png"
                render_mesh(path, vertices, faces, color, camera, (center, radius),
                            region["point"], show_edges=edges)
                progression_primary[method] = path

    compose_detail_board(
        images_by_view["primary"], points_by_view["primary"],
        board_root / "context_improvement_32k_redbox.png",
        f'{settings["label"]}: recursive 32K context detail',
        region["improvement_percent"], labels=LABELS_32K)
    compose_progression_board(
        progression_primary, board_root / "progression_8k_to_32k.png",
        f'{settings["label"]}: trained through 8K, recursively inferred at 32K',
        f'{kind.title()} held-out chain {chain:03d}')

    return {
        "object": slug,
        "label": settings["label"],
        "kind": kind,
        "chain_number": chain,
        "seed": gt_info["seed"],
        "gt_prefix_max_vertex_errors": gt_info["prefix_max_vertex_errors"],
        "mse_improvement_percent": improvement,
        "local_improvement_percent": region["improvement_percent"],
        "metrics": metrics,
        "paths": {
            "directory": str(case_root.relative_to(ROOT)),
            "primary_board": str((board_root / "comparison_32k_primary.png").relative_to(ROOT)),
            "secondary_board": str((board_root / "comparison_32k_secondary.png").relative_to(ROOT)),
            "redbox_board": str((board_root / "context_improvement_32k_redbox.png").relative_to(ROOT)),
            "progression_board": str((board_root / "progression_8k_to_32k.png").relative_to(ROOT)),
        },
    }


def organize_report_ready(records: list[dict]) -> None:
    all_boards = OUTPUT / "report_ready" / "all_boards"
    recommended = OUTPUT / "report_ready" / "recommended"
    all_boards.mkdir(parents=True, exist_ok=True)
    recommended.mkdir(parents=True, exist_ok=True)
    counter = 1
    path_keys = (
        ("primary", "primary_board"),
        ("secondary", "secondary_board"),
        ("redbox", "redbox_board"),
        ("progression", "progression_board"),
    )
    for record in records:
        for suffix, key in path_keys:
            source = ROOT / record["paths"][key]
            name = (f'{counter:02d}_{record["object"]}_{record["kind"]}_'
                    f'chain_{record["chain_number"]:03d}_{suffix}.png')
            shutil.copy2(source, all_boards / name)
            counter += 1
            if ((record["kind"] == "representative" and suffix in ("primary", "progression")) or
                    (record["kind"] == "strong" and suffix == "redbox")):
                shutil.copy2(source, recommended / name)


def write_readme(records: list[dict], elapsed: float) -> None:
    lines = [
        "# Recursive 32K Comparisons",
        "",
        "The models were trained only on 500 -> 2K -> 8K chains. For this step,",
        "each model's own 8K prediction was normalized through the released test",
        "preprocessor, fed back into the unchanged model, and subdivided once to 32K.",
        "No 32K target was used for training.",
        "",
        "The official C++ generator was rerun with each original held-out seed and",
        "three levels. Its regenerated 500/2K/8K prefixes had zero difference from",
        "the stored test chain, establishing level 3 as the corresponding 32K GT.",
        "",
        "- `report_ready/recommended`: nine concise figures recommended for reporting.",
        "- `report_ready/all_boards`: all 36 boards in one flat folder.",
        "- `all_renders`: meshes, individual renders, and boards by object/case.",
        "- `manifest.json`: exact seeds, metrics, and file paths.",
        "",
        f"Runtime: {elapsed:.1f} seconds.",
        "",
        "| Object | Case | Chain | Phase 1 32K MSE improvement | Boxed improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f'| {record["label"]} | {record["kind"]} | {record["chain_number"]:03d} '
            f'| {record["mse_improvement_percent"]:.2f}% '
            f'| {record["local_improvement_percent"]:.2f}% |')
    lines.append("")
    (OUTPUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    start = time.monotonic()
    if not GENERATOR.is_file():
        raise FileNotFoundError(f"build the official C++ generator first: {GENERATOR}")
    source_manifest_path = STEP4 / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("run scripts/render_final_comparisons.py first")
    source_manifest = json.loads(source_manifest_path.read_text())
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    records = []
    for slug, settings in OBJECTS.items():
        phase0_model = load_recursive_model(settings["phase0"], context=False)
        phase1_model = load_recursive_model(settings["phase1"], context=True)
        selected = [record for record in source_manifest["records"]
                    if record["object"] == slug]
        for source_record in selected:
            record = render_case(
                slug, settings, source_record, phase0_model, phase1_model)
            records.append(record)
            print(
                f'{slug} {record["kind"]} chain {record["chain_number"]:03d}: '
                f'Phase 1 32K MSE improvement {record["mse_improvement_percent"]:.2f}%',
                flush=True)

    organize_report_ready(records)
    elapsed = time.monotonic() - start
    manifest = {
        "schema": 1,
        "inference": "trained 8K prediction -> released normalization/initialization -> one recursive subdivision -> 32K",
        "ground_truth": "official C++ generator, original held-out seed, numSubd=3",
        "runtime_seconds": elapsed,
        "records": records,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write_readme(records, elapsed)
    print(f"wrote recursive 32K visual package: {OUTPUT}")


if __name__ == "__main__":
    main()
