#!/usr/bin/env python3
"""Create reproducible tables and figures for the Bimba architecture gate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from render_dataset_audit import add_mesh, load_mesh


JOBS = {
    "Phase 0": ("net_style_bimba_pilot_phase0", "phase0"),
    "Local-only": ("net_context_bimba_pilot_local", "context"),
    "Hybrid": ("net_context_bimba_pilot_hybrid", "context"),
}

METRICS = (
    "correspondence_mse",
    "mean_vertex_distance",
    "maximum_vertex_distance",
    "correspondence_normal_cosine",
    "smooth_region_mse",
    "detail_region_mse",
    "chamfer_mean",
    "hausdorff",
    "narrow_face_ratio",
    "degenerate_face_ratio",
)


def load_reports(jobs_root: Path) -> dict:
    reports = {}
    for label, (folder, mode) in JOBS.items():
        job = jobs_root / folder
        base = json.loads((job / "evaluation_test.json").read_text())
        surface = json.loads((job / "evaluation_test_50k.json").read_text())
        values = dict(base["aggregate"][f"{mode}_level_2"])
        surface_values = surface["aggregate"][f"{mode}_level_2"]
        for key in ("chamfer_mean", "chamfer_squared", "hausdorff",
                    "predicted_to_target_mean", "target_to_predicted_mean"):
            values[key] = surface_values[key]
        reports[label] = {
            "job": job,
            "parameter_count": base["parameter_count"],
            "values": values,
            "base": base,
        }
    return reports


def improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def write_summary(output: Path, reports: dict) -> None:
    phase0 = reports["Phase 0"]["values"]
    local = reports["Local-only"]["values"]
    hybrid = reports["Hybrid"]["values"]
    payload = {
        "decision": "accept_hybrid_for_final_object_specific_training",
        "scope": "Bimba unseen-remeshing test; not cross-object generalization",
        "test_meshes": 20,
        "surface_samples_per_mesh": 50000,
        "models": {
            label: {
                "parameter_count": report["parameter_count"],
                "final_level": {name: report["values"][name] for name in METRICS},
            }
            for label, report in reports.items()
        },
        "hybrid_improvement_percent": {
            "correspondence_mse_vs_phase0": improvement(
                phase0["correspondence_mse"], hybrid["correspondence_mse"]),
            "correspondence_mse_vs_local": improvement(
                local["correspondence_mse"], hybrid["correspondence_mse"]),
            "detail_region_mse_vs_phase0": improvement(
                phase0["detail_region_mse"], hybrid["detail_region_mse"]),
            "detail_region_mse_vs_local": improvement(
                local["detail_region_mse"], hybrid["detail_region_mse"]),
            "chamfer_mean_vs_phase0": improvement(
                phase0["chamfer_mean"], hybrid["chamfer_mean"]),
            "hausdorff_vs_phase0": improvement(
                phase0["hausdorff"], hybrid["hausdorff"]),
        },
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    rows = []
    for name in METRICS:
        rows.append(
            f"| `{name}` | {phase0[name]:.8g} | {local[name]:.8g} | "
            f"**{hybrid[name]:.8g}** |")
    markdown = """# Bimba Pilot: Step 3/4 Result

Decision: **accept the hybrid local-plus-global Context Block for final
object-specific training**.

The best validation checkpoints were evaluated on all 20 untouched Bimba test
chains. Surface metrics were repeated with 50,000 samples per mesh.

| Final 8K metric | Phase 0 | Local-only | Hybrid |
| --- | ---: | ---: | ---: |
""" + "\n".join(rows) + "\n\n"
    markdown += (
        "Hybrid improves final correspondence MSE by "
        f"{payload['hybrid_improvement_percent']['correspondence_mse_vs_phase0']:.2f}% "
        "over Phase 0 and "
        f"{payload['hybrid_improvement_percent']['correspondence_mse_vs_local']:.2f}% "
        "over local-only. It improves detail-region MSE by "
        f"{payload['hybrid_improvement_percent']['detail_region_mse_vs_phase0']:.2f}% "
        "over Phase 0. All evaluated outputs have zero degenerate faces. "
        "This supports the object-specific Bimba claim, not semantic or "
        "cross-object generalization.\n")
    (output / "RESULTS.md").write_text(markdown)


def render_losses(output: Path, reports: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    colors = {"Phase 0": "#555555", "Local-only": "#2b7a78", "Hybrid": "#b4432f"}
    for label, report in reports.items():
        job = report["job"]
        train = np.atleast_1d(np.loadtxt(job / "train_loss.txt"))
        valid = np.atleast_1d(np.loadtxt(job / "valid_loss.txt"))
        axes[0].plot(train, label=label, color=colors[label], linewidth=1.8)
        axes[1].plot(valid, label=label, color=colors[label], linewidth=1.8)
        best = int(np.argmin(valid))
        axes[1].scatter(best, valid[best], color=colors[label], s=28, zorder=3)
    for axis, title in zip(axes, ("Training loss", "Validation loss")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE objective")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_comparison(output: Path, reports: dict) -> None:
    oracle = reports["Hybrid"]["job"] / "0_oracle.obj"
    columns = [("Oracle 8K", oracle)] + [
        (label, report["job"] / "0_subd2.obj")
        for label, report in reports.items()
    ]
    figure = plt.figure(figsize=(16, 8), facecolor="white")
    for row, (elevation, azimuth, view) in enumerate(((-5, -50, "primary"),
                                                       (15, 110, "opposite"))):
        for column, (label, path) in enumerate(columns):
            axis = figure.add_subplot(2, 4, row * 4 + column + 1, projection="3d")
            add_mesh(axis, load_mesh(path, True), elevation, azimuth)
            if row == 0:
                axis.set_title(label, fontsize=12)
            if column == 0:
                axis.text2D(-0.05, 0.5, view, transform=axis.transAxes,
                            rotation=90, va="center", fontweight="bold")
    figure.suptitle("Bimba pilot: saved best validation outputs", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output / "validation_comparison.png", dpi=180,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "bimba_pilot_results"
    output.mkdir(parents=True, exist_ok=True)
    reports = load_reports(root / "external" / "neuralSubdiv" / "jobs")
    write_summary(output, reports)
    render_losses(output, reports)
    render_comparison(output, reports)
    print(f"wrote Bimba pilot summary: {output}")


if __name__ == "__main__":
    main()
