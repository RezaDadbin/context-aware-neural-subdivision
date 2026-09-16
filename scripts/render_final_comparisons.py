#!/usr/bin/env python3
"""Render report-ready Phase 0 versus Phase 1 held-out comparisons."""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from matplotlib import colors as mpl_colors
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
NEURAL = ROOT / "external" / "neuralSubdiv"
sys.path.insert(0, str(NEURAL))

from context_models import ContextSubdNet  # noqa: E402
from mesh_metrics import vertex_curvature_scores  # noqa: E402
from models import SubdNet  # noqa: E402
from train_resume import torch_load  # noqa: E402
from validate_dataset import require_valid_dataset  # noqa: E402


OUTPUT = ROOT / "artifacts" / "final_visual_comparisons"
METHODS = ("coarse", "phase0", "phase1", "gt")
METHOD_LABELS = {
    "coarse": "Coarse input (500 faces)",
    "phase0": "Phase 0: Neural Subdivision",
    "phase1": "Phase 1: Hybrid context",
    "gt": "Ground truth (8,000 faces)",
}
METHOD_COLORS = {
    "coarse": "#9aa78f",
    "phase0": "#a7a7aa",
    "phase1": "#c7b991",
    "gt": "#a8bac3",
}
OBJECTS = {
    "bimba_head": {
        "label": "Bimba Head",
        "pkl": "style_bimba_head_test.pkl",
        "mesh_folder": "style_bimba_head_test_20",
        "phase0": "net_style_bimba_pilot_phase0",
        "phase1": "net_context_bimba_pilot_hybrid",
        "views": {"primary": (-5, -50), "secondary": (15, 110)},
        "y_up": True,
    },
    "fat_dragon": {
        "label": "Fat Dragon",
        "pkl": "style_fat_dragon_test.pkl",
        "mesh_folder": "style_fat_dragon_test_20",
        "phase0": "net_style_fat_dragon_final_phase0",
        "phase1": "net_context_fat_dragon_final_hybrid",
        "views": {"primary": (20, -60), "secondary": (-10, 100)},
        "y_up": False,
    },
    "gear16": {
        "label": "Gear16",
        "pkl": "style_gear16_test.pkl",
        "mesh_folder": "style_gear16_test_20",
        "phase0": "net_style_gear16_final_phase0",
        "phase1": "net_context_gear16_final_hybrid",
        "views": {"primary": (35, -45), "secondary": (-25, 115)},
        "y_up": False,
    },
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
             "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/HelveticaNeue.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def read_level2_records(path: Path) -> dict[int, dict]:
    report = json.loads(path.read_text())
    records = {}
    for record in report["records"]:
        if record["level"] == 2:
            records[int(record["mesh"])] = record["metrics"]
    if len(records) != 20:
        raise RuntimeError(f"expected 20 level-2 records in {path}, got {len(records)}")
    return records


def select_candidates(phase0: dict[int, dict], phase1: dict[int, dict]) -> list[dict]:
    rows = []
    for index in sorted(phase0):
        baseline = phase0[index]["correspondence_mse"]
        improvement = 100.0 * (baseline - phase1[index]["correspondence_mse"]) / baseline
        rows.append({"mesh_index": index, "mse_improvement_percent": improvement})
    ordered = sorted(rows, key=lambda row: row["mse_improvement_percent"])
    choices = [
        ("strong", ordered[-1]),
        ("representative", min(rows, key=lambda row: abs(
            row["mse_improvement_percent"] - np.median(
                [item["mse_improvement_percent"] for item in rows])))),
        ("difficult", ordered[0]),
    ]
    result = []
    used = set()
    for kind, row in choices:
        if row["mesh_index"] in used:
            row = next(item for item in ordered if item["mesh_index"] not in used)
        used.add(row["mesh_index"])
        result.append({"kind": kind, **row})
    return result


def load_model(job_name: str, context: bool):
    job = NEURAL / "jobs" / job_name
    params = json.loads((job / "hyperparameters.json").read_text())
    params["device"] = "cpu"
    model = ContextSubdNet(params) if context else SubdNet(params)
    model.load_state_dict(torch_load(str(job / "netparams.dat"), "cpu"))
    model.eval()
    return model, params


def numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)


def transform_vertices(vertices: np.ndarray, y_up: bool) -> np.ndarray:
    if not y_up:
        return vertices.copy()
    rotation = np.asarray(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=float)
    return vertices @ rotation.T


def smooth_vertex_values(values: np.ndarray, faces: np.ndarray, passes: int = 4) -> np.ndarray:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    source = np.concatenate((edges[:, 0], edges[:, 1]))
    target = np.concatenate((edges[:, 1], edges[:, 0]))
    smoothed = values.copy()
    degree = np.bincount(source, minlength=len(values)).clip(min=1)
    for _ in range(passes):
        neighbor_sum = np.bincount(source, weights=smoothed[target], minlength=len(values))
        smoothed = 0.55 * smoothed + 0.45 * neighbor_sum / degree
    return smoothed


def choose_improvement_region(meshes: dict[str, np.ndarray], faces: np.ndarray,
                              y_up: bool, camera: tuple[float, float]) -> dict:
    gt = meshes["gt"]
    phase0_error = np.linalg.norm(meshes["phase0"] - gt, axis=1)
    phase1_error = np.linalg.norm(meshes["phase1"] - gt, axis=1)
    improvement = phase0_error - phase1_error
    local_improvement = smooth_vertex_values(improvement, faces)
    curvature = vertex_curvature_scores(gt, faces)
    curvature_scale = np.quantile(curvature, 0.95)
    curvature_weight = 0.5 + np.clip(curvature / max(curvature_scale, 1e-12), 0, 1)

    transformed = transform_vertices(gt, y_up)
    tri = transformed[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    center = transformed.mean(axis=0)
    outward = np.einsum("ij,ij->i", normals, tri.mean(axis=1) - center)
    if np.median(outward) < 0:
        normals *= -1
    elevation, azimuth = np.radians(camera)
    camera_direction = np.asarray((
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ))
    visible_faces = np.einsum("ij,j->i", normals, camera_direction) > 0.05
    visible_vertices = np.zeros(len(gt), dtype=bool)
    visible_vertices[np.unique(faces[visible_faces])] = True

    detail = curvature >= np.quantile(curvature, 0.55)
    candidates = visible_vertices & detail & (local_improvement > 0)
    score = local_improvement * curvature_weight
    if not np.any(candidates):
        candidates = local_improvement > 0
    score[~candidates] = -np.inf
    vertex = int(np.argmax(score))

    radius = 0.13 * np.linalg.norm(np.ptp(gt, axis=0))
    region = np.linalg.norm(gt - gt[vertex], axis=1) <= radius
    phase0_local = float(phase0_error[region].mean())
    phase1_local = float(phase1_error[region].mean())
    local_percent = 100.0 * (phase0_local - phase1_local) / max(phase0_local, 1e-12)
    return {
        "vertex": vertex,
        "point": transformed[vertex],
        "vertices_in_region": int(region.sum()),
        "phase0_mean_vertex_error": phase0_local,
        "phase1_mean_vertex_error": phase1_local,
        "improvement_percent": local_percent,
    }


def render_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray,
                color: str, camera: tuple[float, float], bounds: tuple[np.ndarray, float],
                roi_point: np.ndarray, show_edges: bool) -> tuple[int, int]:
    center, radius = bounds
    figure = plt.figure(figsize=(6, 6), dpi=180, facecolor="white")
    axis = figure.add_axes((0.01, 0.01, 0.98, 0.98), projection="3d")
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    elevation, azimuth = camera
    light_elev, light_azim = np.radians((50, -40))
    light = np.asarray((np.cos(light_elev) * np.cos(light_azim),
                        np.cos(light_elev) * np.sin(light_azim),
                        np.sin(light_elev)))
    shade = np.clip(0.67 + 0.30 * (normals @ light), 0.34, 1.0)
    rgb = np.asarray(mpl_colors.to_rgb(color))
    face_colors = np.column_stack((shade[:, None] * rgb[None, :], np.ones(len(shade))))
    collection = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors=(0.12, 0.12, 0.12, 0.25) if show_edges else "none",
        linewidths=0.10 if show_edges else 0.0,
        rasterized=True,
    )
    axis.add_collection3d(collection)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()
    figure.canvas.draw()
    projected = proj3d.proj_transform(*roi_point, axis.get_proj())
    display_x, display_y = axis.transData.transform(projected[:2])
    width, height = figure.canvas.get_width_height()
    pixel = (int(np.clip(display_x, 0, width - 1)),
             int(np.clip(height - display_y, 0, height - 1)))
    figure.savefig(path, dpi=figure.dpi, facecolor="white")
    plt.close(figure)
    return pixel


def centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                  text_font, fill: str = "#202124") -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text,
              font=text_font, fill=fill)


def compose_full_board(paths: dict[str, Path], output: Path, title: str,
                       subtitle: str, labels: dict[str, str] | None = None) -> None:
    labels = labels or METHOD_LABELS
    panel_size = 760
    header = 150
    label_height = 78
    footer = 72
    board = Image.new("RGB", (panel_size * 4, header + label_height + panel_size + footer), "white")
    draw = ImageDraw.Draw(board)
    centered_text(draw, (board.width // 2, 22), title, font(38, True))
    centered_text(draw, (board.width // 2, 76), subtitle, font(25), "#55585d")
    for column, method in enumerate(METHODS):
        left = column * panel_size
        draw.rectangle((left, header, left + panel_size, header + label_height),
                       fill=METHOD_COLORS[method])
        centered_text(draw, (left + panel_size // 2, header + 20),
                      labels[method], font(24, True), "#ffffff")
        image = Image.open(paths[method]).convert("RGB").resize(
            (panel_size, panel_size), Image.Resampling.LANCZOS)
        board.paste(image, (left, header + label_height))
    centered_text(draw, (board.width // 2, header + label_height + panel_size + 16),
                  "All predictions use the same held-out coarse input and camera.",
                  font(22), "#55585d")
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, quality=95)


def box_and_crop(image: Image.Image, point: tuple[int, int], box_size: int,
                 crop_size: int = 360) -> tuple[Image.Image, Image.Image]:
    x, y = point
    half_box = box_size // 2
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for offset in range(6):
        draw.rectangle((x - half_box - offset, y - half_box - offset,
                        x + half_box + offset, y + half_box + offset),
                       outline="#e31b23", width=2)
    half_crop = crop_size // 2
    left = int(np.clip(x - half_crop, 0, max(0, image.width - crop_size)))
    top = int(np.clip(y - half_crop, 0, max(0, image.height - crop_size)))
    crop = image.crop((left, top, left + crop_size, top + crop_size))
    crop = crop.resize((720, 720), Image.Resampling.LANCZOS)
    crop_draw = ImageDraw.Draw(crop)
    crop_draw.rectangle((2, 2, crop.width - 3, crop.height - 3),
                        outline="#e31b23", width=7)
    return annotated, crop


def compose_detail_board(paths: dict[str, Path], points: dict[str, tuple[int, int]],
                         output: Path, title: str, local_percent: float,
                         labels: dict[str, str] | None = None) -> None:
    labels = labels or METHOD_LABELS
    if local_percent >= 0:
        comparison = f"Phase 1 mean vertex error is {local_percent:.1f}% lower"
    else:
        comparison = f"Phase 1 mean vertex error is {abs(local_percent):.1f}% higher"
    panel = 720
    header = 165
    label_height = 72
    gap = 18
    board = Image.new(
        "RGB", (panel * 4, header + label_height + panel * 2 + gap + 65), "white")
    draw = ImageDraw.Draw(board)
    centered_text(draw, (board.width // 2, 20), title, font(38, True))
    centered_text(
        draw, (board.width // 2, 76),
        f"Selected local region: {comparison}",
        font(25), "#55585d")
    centered_text(draw, (board.width // 2, 116),
                  "Red boxes mark the same 3D region in every column.",
                  font(21), "#73767a")

    for column, method in enumerate(METHODS):
        left = column * panel
        draw.rectangle((left, header, left + panel, header + label_height),
                       fill=METHOD_COLORS[method])
        centered_text(draw, (left + panel // 2, header + 18),
                      labels[method], font(23, True), "#ffffff")
        original = Image.open(paths[method]).convert("RGB")
        boxed, crop = box_and_crop(original, points[method], 250)
        boxed = boxed.resize((panel, panel), Image.Resampling.LANCZOS)
        board.paste(boxed, (left, header + label_height))
        board.paste(crop, (left, header + label_height + panel + gap))
    centered_text(draw, (board.width // 2, board.height - 46),
                  "Magnified comparison of the boxed region", font(23, True), "#35373a")
    output.parent.mkdir(parents=True, exist_ok=True)
    board.save(output, quality=95)


def run_object(slug: str, settings: dict) -> list[dict]:
    jobs = NEURAL / "jobs"
    phase0_records = read_level2_records(
        jobs / settings["phase0"] / "evaluation_final_700_test_50k.json")
    phase1_records = read_level2_records(
        jobs / settings["phase1"] / "evaluation_final_700_test_50k.json")
    candidates = select_candidates(phase0_records, phase1_records)

    pkl_path = NEURAL / "data_PKL" / settings["pkl"]
    with pkl_path.open("rb") as handle:
        dataset = pickle.load(handle)
    require_valid_dataset(dataset, f"{slug} test", 2)
    dataset.computeParameters()
    dataset.toDevice("cpu")
    phase0_model, _ = load_model(settings["phase0"], context=False)
    phase1_model, _ = load_model(settings["phase1"], context=True)

    records = []
    with torch.no_grad():
        for candidate in candidates:
            index = candidate["mesh_index"]
            inputs = dataset.getInputData(index)
            phase0_output = phase0_model(
                inputs, index, dataset.hfList, dataset.poolMats, dataset.dofs)[2]
            phase1_output = phase1_model(
                inputs, index, dataset.hfList, dataset.poolMats, dataset.dofs)[2]
            coarse = numpy(dataset.meshes[index][0].V)
            coarse_faces = numpy(dataset.meshes[index][0].F).astype(np.int64)
            target = numpy(dataset.meshes[index][2].V)
            faces = numpy(dataset.meshes[index][2].F).astype(np.int64)
            meshes = {
                "coarse": coarse,
                "phase0": numpy(phase0_output),
                "phase1": numpy(phase1_output),
                "gt": target,
            }

            case_name = f'{candidate["kind"]}_chain_{index + 1:03d}'
            case_root = OUTPUT / "all_renders" / slug / case_name
            mesh_root = case_root / "meshes"
            image_root = case_root / "individual"
            board_root = case_root / "boards"
            image_root.mkdir(parents=True, exist_ok=True)
            board_root.mkdir(parents=True, exist_ok=True)
            for method, vertices in meshes.items():
                write_obj(mesh_root / f"{method}.obj", vertices,
                          coarse_faces if method == "coarse" else faces)

            transformed = {
                method: transform_vertices(vertices, settings["y_up"])
                for method, vertices in meshes.items()
            }
            all_bounds = np.vstack(tuple(transformed.values()))
            center = (all_bounds.min(axis=0) + all_bounds.max(axis=0)) * 0.5
            radius = max(float(np.ptp(all_bounds, axis=0).max()) * 0.51, 1e-6)
            primary_camera = settings["views"]["primary"]
            region = choose_improvement_region(
                meshes, faces, settings["y_up"], primary_camera)
            images_by_view = {}
            points_by_view = {}
            for view_name, camera in settings["views"].items():
                images_by_view[view_name] = {}
                points_by_view[view_name] = {}
                for method in METHODS:
                    image_path = image_root / f"{method}_{view_name}.png"
                    points_by_view[view_name][method] = render_mesh(
                        image_path, transformed[method],
                        coarse_faces if method == "coarse" else faces,
                        METHOD_COLORS[method], camera, (center, radius),
                        region["point"], show_edges=method == "coarse")
                    images_by_view[view_name][method] = image_path
                full_path = board_root / f"comparison_{view_name}.png"
                compose_full_board(
                    images_by_view[view_name], full_path,
                    f'{settings["label"]}: Phase 0 versus Phase 1',
                    f'{candidate["kind"].title()} held-out chain {index + 1:03d} | '
                    f'8K correspondence MSE improvement: '
                    f'{candidate["mse_improvement_percent"]:.1f}%')

            detail_path = board_root / "context_improvement_redbox.png"
            compose_detail_board(
                images_by_view["primary"], points_by_view["primary"], detail_path,
                f'{settings["label"]}: context improvement detail',
                region["improvement_percent"])

            record = {
                **candidate,
                "chain_number": index + 1,
                "paths": {
                    "directory": str(case_root.relative_to(ROOT)),
                    "primary_board": str((board_root / "comparison_primary.png").relative_to(ROOT)),
                    "secondary_board": str((board_root / "comparison_secondary.png").relative_to(ROOT)),
                    "redbox_board": str(detail_path.relative_to(ROOT)),
                },
                "metrics": {
                    "phase0": phase0_records[index],
                    "phase1": phase1_records[index],
                    "local_improvement_region": {
                        key: value for key, value in region.items() if key != "point"
                    },
                },
            }
            records.append(record)
            print(f"rendered {slug} {case_name}", flush=True)
    return records


def organize_report_ready(records: list[dict]) -> None:
    report_root = OUTPUT / "report_ready"
    all_boards = report_root / "all_boards"
    recommended = report_root / "recommended"
    all_boards.mkdir(parents=True, exist_ok=True)
    recommended.mkdir(parents=True, exist_ok=True)
    counter = 1
    for record in records:
        slug = record["object"]
        kind = record["kind"]
        chain = record["chain_number"]
        for board_kind, path_key in (
                ("primary", "primary_board"),
                ("secondary", "secondary_board"),
                ("redbox", "redbox_board")):
            source = ROOT / record["paths"][path_key]
            name = f"{counter:02d}_{slug}_{kind}_chain_{chain:03d}_{board_kind}.png"
            shutil.copy2(source, all_boards / name)
            counter += 1
            if ((kind == "representative" and board_kind == "primary") or
                    (kind == "strong" and board_kind == "redbox")):
                shutil.copy2(source, recommended / name)


def write_readme(records: list[dict]) -> None:
    lines = [
        "# Final Visual Comparisons",
        "",
        "These figures compare the best 700-epoch Phase 0 and Phase 1 checkpoints",
        "on held-out test chains. No training or validation meshes are rendered here.",
        "",
        "- `report_ready/recommended`: six concise figures recommended for the report.",
        "- `report_ready/all_boards`: every composed comparison board in one flat folder.",
        "- `all_renders`: source OBJ exports, individual views, and boards by object/case.",
        "- `manifest.json`: exact chain choices, metrics, and paths.",
        "",
        "Candidate policy: `strong` is the largest Phase 1 correspondence-MSE gain,",
        "`representative` is nearest the median gain, and `difficult` is the smallest gain.",
        "Red-box centers are selected from visible, detailed vertices after smoothing the",
        "per-vertex Phase 0 minus Phase 1 error difference over mesh adjacency.",
        "",
        "## Selected Chains",
        "",
        "| Object | Case | Chain | 8K MSE improvement | Local boxed improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        local = record["metrics"]["local_improvement_region"]["improvement_percent"]
        lines.append(
            f'| {record["label"]} | {record["kind"]} | {record["chain_number"]:03d} '
            f'| {record["mse_improvement_percent"]:.2f}% | {local:.2f}% |')
    lines.append("")
    (OUTPUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    all_records = []
    for slug, settings in OBJECTS.items():
        object_records = run_object(slug, settings)
        for record in object_records:
            record["object"] = slug
            record["label"] = settings["label"]
        all_records.extend(object_records)
    organize_report_ready(all_records)
    manifest = {
        "schema": 1,
        "selection": "strong, representative, and difficult held-out test chains",
        "checkpoints": "best validation netparams.dat from each 700-epoch run",
        "records": all_records,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write_readme(all_records)
    print(f"wrote final visual package: {OUTPUT}")


if __name__ == "__main__":
    main()
