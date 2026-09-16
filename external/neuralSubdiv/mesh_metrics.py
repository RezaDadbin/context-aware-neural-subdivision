"""Deterministic geometry and topology metrics for subdivision experiments."""

import math

import numpy as np
import torch
from scipy.spatial import cKDTree


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def topology_metrics(faces, vertex_count):
    faces = _numpy(faces).astype(np.int64, copy=False)
    valid_shape = faces.ndim == 2 and faces.shape[1] == 3
    valid_indices = bool(
        valid_shape and faces.size > 0 and faces.min() >= 0 and
        faces.max() < vertex_count)
    if not valid_indices:
        return {
            "vertices": int(vertex_count),
            "faces": int(faces.shape[0]) if faces.ndim > 0 else 0,
            "unique_edges": 0,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "inconsistent_oriented_edges": 0,
            "connected_components": 0,
            "isolated_vertices": int(vertex_count),
            "euler_characteristic": int(vertex_count),
            "valid_face_indices": False,
        }

    directed_edges = np.concatenate((
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.sort(directed_edges, axis=1)
    unique_edges, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True)
    direction = np.where(
        directed_edges[:, 0] < directed_edges[:, 1], 1, -1)
    direction_sum = np.bincount(
        inverse, weights=direction, minlength=unique_edges.shape[0])

    parents = np.arange(vertex_count, dtype=np.int64)

    def find(vertex):
        while parents[vertex] != vertex:
            parents[vertex] = parents[parents[vertex]]
            vertex = parents[vertex]
        return vertex

    for first, second in unique_edges:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    used_vertices = np.unique(faces)
    components = len({find(int(vertex)) for vertex in used_vertices})
    return {
        "vertices": int(vertex_count),
        "faces": int(faces.shape[0]),
        "unique_edges": int(unique_edges.shape[0]),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "inconsistent_oriented_edges": int(np.count_nonzero(
            (counts == 2) & (direction_sum != 0))),
        "connected_components": int(components),
        "isolated_vertices": int(vertex_count - used_vertices.size),
        "euler_characteristic": int(
            vertex_count - unique_edges.shape[0] + faces.shape[0]),
        "valid_face_indices": True,
    }


def mesh_quality_metrics(vertices, faces, narrow_threshold=0.2, eps=1e-12):
    vertices = _numpy(vertices).astype(np.float64, copy=False)
    faces = _numpy(faces).astype(np.int64, copy=False)
    result = topology_metrics(faces, vertices.shape[0])
    result["finite_vertices"] = bool(np.isfinite(vertices).all())
    if not result["finite_vertices"] or not result["valid_face_indices"]:
        result.update({
            "bbox_diagonal": None,
            "surface_area": None,
            "minimum_triangle_quality": None,
            "mean_triangle_quality": None,
            "narrow_face_ratio": 1.0,
            "degenerate_face_ratio": 1.0,
        })
        return result

    triangles = vertices[faces]
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_02 = triangles[:, 2] - triangles[:, 0]
    edge_12 = triangles[:, 2] - triangles[:, 1]
    double_area = np.linalg.norm(np.cross(edge_01, edge_02), axis=1)
    edge_square_sum = (
        np.einsum("ij,ij->i", edge_01, edge_01) +
        np.einsum("ij,ij->i", edge_02, edge_02) +
        np.einsum("ij,ij->i", edge_12, edge_12))
    quality = np.divide(
        2.0 * math.sqrt(3.0) * double_area,
        edge_square_sum,
        out=np.zeros_like(double_area),
        where=edge_square_sum > eps)
    bbox_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    degenerate_limit = eps * max(bbox_diagonal * bbox_diagonal, 1.0)
    result.update({
        "bbox_diagonal": bbox_diagonal,
        "surface_area": float(0.5 * double_area.sum()),
        "minimum_triangle_quality": float(quality.min()),
        "mean_triangle_quality": float(quality.mean()),
        "narrow_face_ratio": float(np.mean(quality < narrow_threshold)),
        "degenerate_face_ratio": float(np.mean(double_area <= degenerate_limit)),
    })
    return result


def midpoint_subdivision_faces(faces, vertex_count):
    """Return the exact face/vertex ordering used by the official generator."""
    faces = _numpy(faces).astype(np.int64, copy=False)
    if (faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0 or
            faces.min() < 0 or faces.max() >= vertex_count):
        raise ValueError("faces must be a non-empty valid triangle array")

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                            faces[:, [2, 0]]), axis=0)
    unique_edges = np.unique(np.sort(edges, axis=1), axis=0)
    midpoint = {
        (int(edge[0]), int(edge[1])): vertex_count + index
        for index, edge in enumerate(unique_edges)
    }

    def edge_vertex(first, second):
        return midpoint[tuple(sorted((int(first), int(second))))]

    first_corners = []
    second_corners = []
    third_corners = []
    centers = []
    for first, second, third in faces:
        edge_01 = edge_vertex(first, second)
        edge_12 = edge_vertex(second, third)
        edge_20 = edge_vertex(third, first)
        first_corners.append((first, edge_01, edge_20))
        second_corners.append((second, edge_12, edge_01))
        third_corners.append((third, edge_20, edge_12))
        centers.append((edge_12, edge_20, edge_01))
    return np.asarray(
        first_corners + second_corners + third_corners + centers,
        dtype=np.int64)


def core_frame_metrics(vertices, faces, edge_ratio=1e-5,
                       face_area_ratio=1e-10, normal_sum=1e-3):
    """Measure the denominators used by Neural Subdivision local frames."""
    vertices = _numpy(vertices).astype(np.float64, copy=False)
    faces = _numpy(faces).astype(np.int64, copy=False)
    topology = topology_metrics(faces, vertices.shape[0])
    if not topology["valid_face_indices"] or not np.isfinite(vertices).all():
        return {
            "interior_core_edges": 0,
            "minimum_core_edge_ratio": None,
            "minimum_core_face_area_ratio": None,
            "minimum_core_normal_sum": None,
            "ill_conditioned_core_frame_ratio": 1.0,
        }

    triangles = vertices[faces]
    face_cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(face_cross, axis=1)
    unit_normals = np.divide(
        face_cross, double_area[:, None], out=np.zeros_like(face_cross),
        where=double_area[:, None] > 0.0)
    diagonal = float(np.linalg.norm(
        vertices.max(axis=0) - vertices.min(axis=0)))
    length_scale = max(diagonal, np.finfo(np.float64).eps)

    incidents = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]),
                              (face[2], face[0])):
            key = tuple(sorted((int(first), int(second))))
            incidents.setdefault(key, []).append(face_index)

    edge_values = []
    area_values = []
    normal_values = []
    for (first, second), face_indices in incidents.items():
        if len(face_indices) != 2:
            continue
        edge_values.append(
            np.linalg.norm(vertices[second] - vertices[first]) / length_scale)
        area_values.append(
            min(double_area[face_indices[0]], double_area[face_indices[1]]) /
            (length_scale * length_scale))
        normal_values.append(np.linalg.norm(
            unit_normals[face_indices[0]] + unit_normals[face_indices[1]]))

    if not edge_values:
        return {
            "interior_core_edges": 0,
            "minimum_core_edge_ratio": None,
            "minimum_core_face_area_ratio": None,
            "minimum_core_normal_sum": None,
            "ill_conditioned_core_frame_ratio": 1.0,
        }
    edge_values = np.asarray(edge_values)
    area_values = np.asarray(area_values)
    normal_values = np.asarray(normal_values)
    ill_conditioned = (
        (edge_values <= edge_ratio) | (area_values <= face_area_ratio) |
        (normal_values <= normal_sum))
    return {
        "interior_core_edges": int(edge_values.size),
        "minimum_core_edge_ratio": float(edge_values.min()),
        "minimum_core_face_area_ratio": float(area_values.min()),
        "minimum_core_normal_sum": float(normal_values.min()),
        "ill_conditioned_core_frame_ratio": float(ill_conditioned.mean()),
    }


def correspondence_metrics(predicted, target, faces, eps=1e-12):
    predicted = _numpy(predicted).astype(np.float64, copy=False)
    target = _numpy(target).astype(np.float64, copy=False)
    faces = _numpy(faces).astype(np.int64, copy=False)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target vertices must have identical shape")

    difference = predicted - target
    vertex_distance = np.linalg.norm(difference, axis=1)
    predicted_cross = np.cross(
        predicted[faces[:, 1]] - predicted[faces[:, 0]],
        predicted[faces[:, 2]] - predicted[faces[:, 0]])
    target_cross = np.cross(
        target[faces[:, 1]] - target[faces[:, 0]],
        target[faces[:, 2]] - target[faces[:, 0]])
    predicted_length = np.linalg.norm(predicted_cross, axis=1)
    target_length = np.linalg.norm(target_cross, axis=1)
    valid = (predicted_length > eps) & (target_length > eps)
    if np.any(valid):
        cosine = np.einsum(
            "ij,ij->i", predicted_cross[valid], target_cross[valid])
        cosine /= predicted_length[valid] * target_length[valid]
        normal_cosine = float(np.clip(cosine, -1.0, 1.0).mean())
    else:
        normal_cosine = float("nan")
    return {
        "correspondence_mse": float(np.mean(difference * difference)),
        "correspondence_rmse": float(np.sqrt(np.mean(difference * difference))),
        "mean_vertex_distance": float(vertex_distance.mean()),
        "maximum_vertex_distance": float(vertex_distance.max()),
        "correspondence_normal_cosine": normal_cosine,
    }


def vertex_curvature_scores(vertices, faces, eps=1e-12):
    """Compute an invariant dihedral-based detail score per target vertex."""
    vertices = _numpy(vertices).astype(np.float64, copy=False)
    faces = _numpy(faces).astype(np.int64, copy=False)
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(
        normals, lengths[:, None], out=np.zeros_like(normals),
        where=lengths[:, None] > eps)

    directed_edges = np.concatenate((
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.sort(directed_edges, axis=1)
    _, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True)
    face_indices = np.tile(np.arange(faces.shape[0]), 3)
    score_sum = np.zeros(vertices.shape[0], dtype=np.float64)
    score_count = np.zeros(vertices.shape[0], dtype=np.float64)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts[:-1])))
    interior = np.flatnonzero(counts == 2)
    first_occurrence = order[starts[interior]]
    second_occurrence = order[starts[interior] + 1]
    first_face = face_indices[first_occurrence]
    second_face = face_indices[second_occurrence]
    cosine = np.einsum(
        "ij,ij->i", normals[first_face], normals[second_face])
    scores = 1.0 - np.clip(cosine, -1.0, 1.0)
    endpoints = edges[first_occurrence]
    np.add.at(score_sum, endpoints[:, 0], scores)
    np.add.at(score_sum, endpoints[:, 1], scores)
    np.add.at(score_count, endpoints[:, 0], 1.0)
    np.add.at(score_count, endpoints[:, 1], 1.0)
    return np.divide(
        score_sum, score_count, out=np.zeros_like(score_sum),
        where=score_count > 0.0)


def region_correspondence_metrics(predicted, target, faces):
    """Report errors on low- and high-curvature target vertex quartiles."""
    predicted = _numpy(predicted).astype(np.float64, copy=False)
    target = _numpy(target).astype(np.float64, copy=False)
    faces = _numpy(faces).astype(np.int64, copy=False)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target vertices must have identical shape")
    scores = vertex_curvature_scores(target, faces)
    smooth_limit, detail_limit = np.quantile(scores, (0.25, 0.75))
    smooth = scores <= smooth_limit
    detail = scores >= detail_limit
    squared_error = np.mean((predicted - target) ** 2, axis=1)
    distance = np.linalg.norm(predicted - target, axis=1)
    return {
        "smooth_region_vertices": int(np.count_nonzero(smooth)),
        "detail_region_vertices": int(np.count_nonzero(detail)),
        "smooth_region_mse": float(squared_error[smooth].mean()),
        "detail_region_mse": float(squared_error[detail].mean()),
        "smooth_region_mean_vertex_distance": float(distance[smooth].mean()),
        "detail_region_mean_vertex_distance": float(distance[detail].mean()),
    }


def sample_surface(vertices, faces, count, seed):
    vertices = _numpy(vertices).astype(np.float64, copy=False)
    faces = _numpy(faces).astype(np.int64, copy=False)
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0]), axis=1)
    area_sum = areas.sum()
    if not np.isfinite(area_sum) or area_sum <= 0.0:
        raise ValueError("cannot sample a mesh with zero or non-finite area")

    rng = np.random.default_rng(seed)
    selected = rng.choice(faces.shape[0], size=count, p=areas / area_sum)
    selected_triangles = triangles[selected]
    root_u = np.sqrt(rng.random(count))
    v = rng.random(count)
    return (
        (1.0 - root_u)[:, None] * selected_triangles[:, 0] +
        (root_u * (1.0 - v))[:, None] * selected_triangles[:, 1] +
        (root_u * v)[:, None] * selected_triangles[:, 2])


def surface_distance_metrics(predicted, target, faces, samples=10000, seed=0):
    predicted_samples = sample_surface(predicted, faces, samples, seed)
    target_samples = sample_surface(target, faces, samples, seed)
    predicted_to_target = cKDTree(target_samples).query(
        predicted_samples, workers=-1)[0]
    target_to_predicted = cKDTree(predicted_samples).query(
        target_samples, workers=-1)[0]
    return {
        "surface_samples": int(samples),
        "chamfer_mean": float(0.5 * (
            predicted_to_target.mean() + target_to_predicted.mean())),
        "chamfer_squared": float(0.5 * (
            np.mean(predicted_to_target ** 2) +
            np.mean(target_to_predicted ** 2))),
        "hausdorff": float(max(
            predicted_to_target.max(), target_to_predicted.max())),
        "predicted_to_target_mean": float(predicted_to_target.mean()),
        "target_to_predicted_mean": float(target_to_predicted.mean()),
    }


def prediction_metrics(predicted, target, faces, surface_samples=0, seed=0):
    result = correspondence_metrics(predicted, target, faces)
    result.update(region_correspondence_metrics(predicted, target, faces))
    result.update(mesh_quality_metrics(predicted, faces))
    if surface_samples > 0:
        result.update(surface_distance_metrics(
            predicted, target, faces, surface_samples, seed))
    return result


__all__ = [
    "correspondence_metrics",
    "core_frame_metrics",
    "mesh_quality_metrics",
    "midpoint_subdivision_faces",
    "prediction_metrics",
    "region_correspondence_metrics",
    "sample_surface",
    "surface_distance_metrics",
    "topology_metrics",
    "vertex_curvature_scores",
]
