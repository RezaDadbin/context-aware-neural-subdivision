#!/usr/bin/env python3
"""Render additional report-ready views of the existing 8K comparisons."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_final_comparisons import (  # noqa: E402
    METHOD_COLORS,
    METHODS,
    OBJECTS,
    choose_improvement_region,
    compose_detail_board,
    compose_full_board,
    render_mesh,
    transform_vertices,
)


SOURCE = ROOT / "artifacts" / "final_visual_comparisons"
OUTPUT = SOURCE / "additional_angles"
ADDITIONAL_VIEWS = {
    "bimba_head": {
        "left_three_quarter": (5, -5),
        "right_profile": (5, -95),
        "elevated_front": (42, -50),
    },
    "fat_dragon": {
        "left_side": (16, -15),
        "right_side": (16, -105),
        "elevated_front": (48, -60),
    },
    "gear16": {
        "top": (72, -45),
        "low_side": (5, -45),
        "opposite_top": (38, 135),
    },
}

# Keep every board in the archive, but only promote views that visibly expose
# meaningful geometry differences to the report-ready recommendation folder.
RECOMMENDED_VIEWS = {
    "bimba_head": {"left_three_quarter", "right_profile"},
    "fat_dragon": {"left_side", "right_side", "elevated_front"},
    "gear16": {"top", "opposite_top"},
}


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False, force="mesh")
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int64)


def render_case(record: dict) -> list[dict]:
    slug = record["object"]
    settings = OBJECTS[slug]
    case_name = f'{record["kind"]}_chain_{record["chain_number"]:03d}'
    source_meshes = SOURCE / "all_renders" / slug / case_name / "meshes"
    meshes = {}
    faces_by_method = {}
    for method in METHODS:
        meshes[method], faces_by_method[method] = load_mesh(
            source_meshes / f"{method}.obj")
    faces = faces_by_method["gt"]
    for method in ("phase0", "phase1"):
        if not np.array_equal(faces_by_method[method], faces):
            raise RuntimeError(f"8K topology mismatch for {slug} {case_name}")

    transformed = {
        method: transform_vertices(vertices, settings["y_up"])
        for method, vertices in meshes.items()
    }
    all_bounds = np.vstack(tuple(transformed.values()))
    center = (all_bounds.min(axis=0) + all_bounds.max(axis=0)) * 0.5
    radius = max(float(np.ptp(all_bounds, axis=0).max()) * 0.51, 1e-6)
    case_root = OUTPUT / "all_renders" / slug / case_name
    results = []

    for view_name, camera in ADDITIONAL_VIEWS[slug].items():
        view_root = case_root / view_name
        individual_root = view_root / "individual"
        individual_root.mkdir(parents=True, exist_ok=True)
        region = choose_improvement_region(meshes, faces, settings["y_up"], camera)
        paths = {}
        points = {}
        for method in METHODS:
            path = individual_root / f"{method}.png"
            points[method] = render_mesh(
                path, transformed[method], faces_by_method[method],
                METHOD_COLORS[method], camera, (center, radius), region["point"],
                show_edges=method == "coarse")
            paths[method] = path

        comparison = view_root / "comparison.png"
        redbox = view_root / "context_improvement_redbox.png"
        title_view = view_name.replace("_", " ").title()
        subtitle = (
            f'{record["kind"].title()} held-out chain {record["chain_number"]:03d} | '
            f'8K MSE improvement: {record["mse_improvement_percent"]:.1f}%')
        compose_full_board(
            paths, comparison,
            f'{settings["label"]}: {title_view} 8K comparison', subtitle)
        compose_detail_board(
            paths, points, redbox,
            f'{settings["label"]}: {title_view} context detail',
            region["improvement_percent"])
        results.append({
            "object": slug,
            "label": settings["label"],
            "kind": record["kind"],
            "chain_number": record["chain_number"],
            "view": view_name,
            "camera": list(camera),
            "mse_improvement_percent": record["mse_improvement_percent"],
            "local_improvement_region": {
                key: value for key, value in region.items() if key != "point"
            },
            "comparison": str(comparison.relative_to(ROOT)),
            "redbox": str(redbox.relative_to(ROOT)),
        })
        print(f"rendered {slug} {case_name} {view_name}", flush=True)
    return results


def organize(records: list[dict]) -> None:
    all_boards = OUTPUT / "report_ready" / "all_boards"
    recommended = OUTPUT / "report_ready" / "recommended"
    all_boards.mkdir(parents=True, exist_ok=True)
    recommended.mkdir(parents=True, exist_ok=True)
    counter = 1
    for record in records:
        for board_type in ("comparison", "redbox"):
            source = ROOT / record[board_type]
            name = (
                f'{counter:02d}_{record["object"]}_{record["kind"]}_'
                f'chain_{record["chain_number"]:03d}_{record["view"]}_{board_type}.png')
            shutil.copy2(source, all_boards / name)
            counter += 1
            selected_view = record["view"] in RECOMMENDED_VIEWS[record["object"]]
            selected_board = (
                (record["kind"] == "representative" and board_type == "comparison") or
                (record["kind"] == "strong" and board_type == "redbox")
            )
            if selected_view and selected_board:
                shutil.copy2(source, recommended / name)


def write_readme(records: list[dict]) -> None:
    lines = [
        "# Additional 8K Comparison Angles",
        "",
        "These figures reuse the exact held-out 8K meshes from the final visual",
        "comparison package. No new inference, training, or mesh modification occurs.",
        "Each camera receives its own visible, detail-weighted improvement region.",
        "",
        "- `report_ready/recommended`: visually reviewed representative full views and strong red-box views.",
        "- `report_ready/all_boards`: all additional comparison and red-box boards.",
        "- `all_renders`: individual views organized by object, case, and camera.",
        "- `manifest.json`: camera parameters, metrics, and exact paths.",
        "",
        "| Object | Case | Chain | View | Local boxed improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        local = record["local_improvement_region"]["improvement_percent"]
        lines.append(
            f'| {record["label"]} | {record["kind"]} '
            f'| {record["chain_number"]:03d} | {record["view"]} | {local:.2f}% |')
    lines.append("")
    (OUTPUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    manifest_path = SOURCE / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run scripts/render_final_comparisons.py first")
    source_manifest = json.loads(manifest_path.read_text())
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    records = []
    for source_record in source_manifest["records"]:
        records.extend(render_case(source_record))
    organize(records)
    manifest = {
        "schema": 1,
        "source": str(manifest_path.relative_to(ROOT)),
        "scope": "additional 8K held-out views only; recursive 32K excluded",
        "records": records,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write_readme(records)
    print(f"wrote additional 8K views: {OUTPUT}")


if __name__ == "__main__":
    main()
