#!/usr/bin/env python3
"""Render the selected compact Context Block source/coarse/target meshes."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


OBJECTS = (
    ("bimba_head", "Bimba head", "01_bimba_head.obj", "bimba_head_seed_10", -5, -50, True),
    ("fat_dragon", "Fat dragon", "02_fat_dragon.obj", "fat_dragon_seed_01", 20, -60, False),
    ("gear16", "Gear16", "03_gear16.obj", "gear16_repaired_001", 35, -45, False),
)


def add_mesh(ax, path, elevation, azimuth, show_edges, y_up):
    mesh = trimesh.load(path, process=False)
    if y_up:
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
        mesh.apply_transform(transform)
    triangles = mesh.vertices[mesh.faces]
    light = np.asarray([0.25, -0.35, 0.9])
    light /= np.linalg.norm(light)
    shade = np.clip(0.62 + 0.32 * (mesh.face_normals @ light), 0.3, 0.98)
    colors = np.column_stack((0.78 * shade, 0.74 * shade, 0.63 * shade, np.ones(len(shade))))
    edge_color = (0.22, 0.21, 0.18, 0.28) if show_edges else "none"
    collection = Poly3DCollection(
        triangles,
        facecolors=colors,
        edgecolors=edge_color,
        linewidths=0.08 if show_edges else 0.0,
        rasterized=True,
    )
    ax.add_collection3d(collection)
    center = mesh.bounds.mean(axis=0)
    radius = mesh.extents.max() * 0.55
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()


def render_single(path, output, title, elevation, azimuth, show_edges, y_up):
    figure = plt.figure(figsize=(6, 6), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    add_mesh(axis, path, elevation, azimuth, show_edges, y_up)
    axis.set_title(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    args = parser.parse_args()
    root = args.candidate_root.resolve()
    render_root = root / "renders"
    render_root.mkdir(parents=True, exist_ok=True)

    columns = (
        ("source", "High-resolution source GT", lambda source, chain: source, False),
        ("coarse", "Coarse input (500 faces)", lambda source, chain: chain / "output_s0.obj", True),
        ("target", "Mapped target (8,000 faces)", lambda source, chain: chain / "output_s2.obj", True),
    )
    figure = plt.figure(figsize=(15, 15), facecolor="white")
    for row, (key, label, source_name, chain_name, elevation, azimuth, y_up) in enumerate(OBJECTS):
        source = root / source_name
        chain = root / "representative_chains" / chain_name
        for column, (column_key, heading, resolver, show_edges) in enumerate(columns):
            path = resolver(source, chain)
            axis = figure.add_subplot(3, 3, row * 3 + column + 1, projection="3d")
            add_mesh(axis, path, elevation, azimuth, show_edges, y_up)
            if row == 0:
                axis.set_title(heading, fontsize=14)
            if column == 0:
                axis.text2D(-0.08, 0.5, label, transform=axis.transAxes,
                            rotation=90, va="center", fontsize=14, fontweight="bold")
            render_single(
                path,
                render_root / f"{key}_{column_key}.png",
                f"{label}: {heading}",
                elevation,
                azimuth,
                show_edges,
                y_up,
            )
    figure.tight_layout()
    figure.savefig(
        render_root / "compact_context_source_coarse_target.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
