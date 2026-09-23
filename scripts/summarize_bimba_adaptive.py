#!/usr/bin/env python3
"""Compare Adaptive V1 Bimba results with the fixed-six-depth model."""

import json
from pathlib import Path

import numpy as np


METRICS = (
    ("correspondence_mse", "lower"),
    ("detail_region_mse", "lower"),
    ("smooth_region_mse", "lower"),
    ("correspondence_normal_cosine", "higher"),
    ("chamfer_mean", "lower"),
    ("hausdorff", "lower"),
    ("narrow_face_ratio", "lower"),
    ("degenerate_face_ratio", "lower"),
)


def improvement(reference, candidate, direction):
    if direction == "higher":
        return 100.0 * (candidate - reference) / abs(reference)
    return 100.0 * (reference - candidate) / abs(reference)


def best_validation_epoch(job):
    values = np.atleast_1d(np.loadtxt(job / "valid_loss.txt"))
    index = int(np.nanargmin(values))
    return index + 1, float(values[index])


def main():
    root = Path(__file__).resolve().parents[1]
    jobs = root / "external" / "neuralSubdiv" / "jobs"
    fixed_job = jobs / "net_context_bimba_pilot_hybrid"
    adaptive_job = jobs / "net_context_bimba_adaptive_hybrid"
    fixed_report_path = fixed_job / "evaluation_final_700_test_50k.json"
    adaptive_report_path = adaptive_job / "evaluation_adaptive_test_50k.json"

    for path in (fixed_report_path, adaptive_report_path):
        if not path.is_file():
            raise FileNotFoundError("missing evaluation report: %s" % path)

    fixed_report = json.loads(fixed_report_path.read_text())
    adaptive_report = json.loads(adaptive_report_path.read_text())
    fixed = fixed_report["aggregate"]["context_level_2"]
    adaptive = adaptive_report["aggregate"]["context_level_2"]
    fixed_epoch, fixed_valid = best_validation_epoch(fixed_job)
    adaptive_epoch, adaptive_valid = best_validation_epoch(adaptive_job)

    comparisons = {}
    for name, direction in METRICS:
        comparisons[name] = {
            "direction": direction,
            "fixed_6_depth": fixed[name],
            "adaptive_1_to_6_depth": adaptive[name],
            "adaptive_improvement_percent": improvement(
                fixed[name], adaptive[name], direction)
            if fixed[name] != 0.0 else None,
        }

    payload = {
        "scope": (
            "Controlled Bimba held-out comparison; this split previously "
            "participated in architecture selection and is not a pristine "
            "independent final test set."
        ),
        "fixed_model": {
            "job": str(fixed_job),
            "parameters": fixed_report["parameter_count"],
            "best_validation_epoch": fixed_epoch,
            "best_validation_loss": fixed_valid,
        },
        "adaptive_model": {
            "job": str(adaptive_job),
            "parameters": adaptive_report["parameter_count"],
            "best_validation_epoch": adaptive_epoch,
            "best_validation_loss": adaptive_valid,
        },
        "final_8k_comparison": comparisons,
        "selector_aggregate": adaptive_report.get("selector_aggregate", {}),
    }

    output = root / "artifacts" / "bimba_adaptive_context"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    rows = []
    for name, _ in METRICS:
        item = comparisons[name]
        change = item["adaptive_improvement_percent"]
        change_text = "n/a" if change is None else "%.2f%%" % change
        rows.append(
            "| `%s` | %.8g | %.8g | %s |" %
            (name, item["fixed_6_depth"], item["adaptive_1_to_6_depth"],
             change_text))

    selector_lines = []
    for stage in range(3):
        key = "context_stage_%d" % stage
        values = payload["selector_aggregate"].get(key)
        if values:
            selector_lines.append(
                "- Stage %d: effective depth %.3f, normalized entropy %.3f, "
                "mean weights %s" %
                (stage, values["mean_effective_radius"],
                 values["mean_normalized_entropy"],
                 ", ".join("%.3f" % value
                           for value in values["mean_weights"])))

    markdown = """# Bimba Adaptive V1 Comparison

This is a controlled comparison with the saved fixed-six-depth Phase 1 model.
The Bimba held-out split previously participated in architecture selection, so
these values are not a completely independent final test estimate.

- Fixed Phase 1 best validation: epoch %d, `%.8g`
- Adaptive V1 best validation: epoch %d, `%.8g`
- Fixed parameters: `%d`
- Adaptive V1 parameters: `%d`

| Final 8K metric | Fixed depth 6 | Adaptive depths 1--6 | Improvement |
| --- | ---: | ---: | ---: |
%s

Positive improvement means the adaptive model is better for that metric.

## Learned Selector

%s

Spatial variation must also be checked in the per-mesh selector records. A
nearly identical distribution everywhere does not demonstrate adaptive
context, even when reconstruction metrics improve.
""" % (
        fixed_epoch, fixed_valid, adaptive_epoch, adaptive_valid,
        fixed_report["parameter_count"], adaptive_report["parameter_count"],
        "\n".join(rows), "\n".join(selector_lines) or "No selector data found.")
    (output / "RESULTS.md").write_text(markdown)
    print("wrote Adaptive V1 Bimba comparison: %s" % output)


if __name__ == "__main__":
    main()
