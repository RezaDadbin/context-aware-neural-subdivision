#!/usr/bin/env python3
"""Render representative generated chains for visual pretraining approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OBJECTS = {
    "bimba_head": {
        "label": "Bimba Head",
        "source": "01_bimba_head.obj",
        "camera": (-5, -50),
        "y_up": True,
    },
    "fat_dragon": {
        "label": "Fat Dragon",
        "source": "02_fat_dragon.obj",
        "camera": (20, -60),
        "y_up": False,
    },
    "gear16": {
        "label": "Gear16",
        "source": "03_gear16.obj",
        "camera": (35, -45),
        "y_up": False,
    },
}

SPLITS = {
    "train": ("style_{style}_200", 200),
    "valid": ("style_{style}_valid_20", 20),
    "test": ("style_{style}_test_20", 20),
}


def load_mesh(path: Path, y_up: bool) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, force="mesh")
    if y_up:
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(
            ((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=float)
        mesh.apply_transform(transform)
    return mesh


def add_mesh(ax, mesh: trimesh.Trimesh, elevation: float, azimuth: float) -> None:
    triangles = mesh.vertices[mesh.faces]
    light = np.asarray([0.25, -0.35, 0.9])
    light /= np.linalg.norm(light)
    shade = np.clip(0.62 + 0.32 * (mesh.face_normals @ light), 0.28, 0.98)
    colors = np.column_stack(
        (0.77 * shade, 0.74 * shade, 0.64 * shade, np.ones(len(shade))))
    collection = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors=(0.17, 0.16, 0.14, 0.25),
        linewidths=0.06,
        rasterized=True,
    )
    ax.add_collection3d(collection)
    center = mesh.bounds.mean(axis=0)
    radius = max(float(mesh.extents.max()) * 0.56, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()


def selected_indices(count: int) -> list[int]:
    return [1, (count + 1) // 2, count]


def render_source(source: Path, output: Path, settings: dict) -> None:
    mesh = load_mesh(source, settings["y_up"])
    figure = plt.figure(figsize=(6, 6), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    add_mesh(axis, mesh, *settings["camera"])
    axis.set_title(
        f'{settings["label"]}: high-resolution source ({len(mesh.faces):,} faces)',
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_split(
    style: str,
    split: str,
    folder: Path,
    count: int,
    output: Path,
    settings: dict,
    reverse: bool,
) -> list[dict]:
    elevation, azimuth = settings["camera"]
    view_name = "opposite" if reverse else "primary"
    if reverse:
        azimuth += 160
        elevation = -elevation + 10

    indices = selected_indices(count)
    levels = ((0, "500 faces"), (1, "2,000 faces"), (2, "8,000 faces"))
    figure = plt.figure(figsize=(13, 11), facecolor="white")
    records = []
    for row, index in enumerate(indices):
        for column, (level, heading) in enumerate(levels):
            path = folder / f"subd{level}" / f"{index:03d}.obj"
            if not path.is_file():
                raise FileNotFoundError(path)
            mesh = load_mesh(path, settings["y_up"])
            axis = figure.add_subplot(3, 3, row * 3 + column + 1, projection="3d")
            add_mesh(axis, mesh, elevation, azimuth)
            if row == 0:
                axis.set_title(heading, fontsize=12)
            if column == 0:
                axis.text2D(
                    -0.05,
                    0.5,
                    f"chain {index:03d}",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
            if not reverse:
                records.append({
                    "split": split,
                    "chain": index,
                    "level": level,
                    "path": str(path.resolve()),
                    "vertices": int(len(mesh.vertices)),
                    "faces": int(len(mesh.faces)),
                })
    figure.suptitle(
        f'{settings["label"]} - {split} - {view_name} view', fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/dataset_audit"),
        help="output directory relative to the codebase root")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    neural = root / "external" / "neuralSubdiv"
    candidate_root = root / "data_candidates" / "compact_context"
    output_root = args.output
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    images = []
    for style, settings in OBJECTS.items():
        source_output = output_root / f"{style}_source.png"
        render_source(candidate_root / settings["source"], source_output, settings)
        images.append(str(source_output.resolve()))
        for split, (pattern, count) in SPLITS.items():
            folder = neural / "data_meshes" / pattern.format(style=style)
            for reverse in (False, True):
                view = "opposite" if reverse else "primary"
                output = output_root / f"{style}_{split}_{view}.png"
                records.extend(render_split(
                    style, split, folder, count, output, settings, reverse))
                images.append(str(output.resolve()))
                print(f"rendered {style} {split} {view}: {output}", flush=True)

    manifest = {
        "selection": "first, middle, and last chain in every split",
        "views": ["primary", "opposite"],
        "images": images,
        "meshes": records,
        "manual_check_required": [
            "self-intersections or folded surfaces",
            "grossly stretched triangles",
            "loss of recognizable source structure",
        ],
    }
    manifest_path = output_root / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote audit manifest: {manifest_path}")


if __name__ == "__main__":
    main()
