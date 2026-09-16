#!/usr/bin/env python3
"""Extract and cap the largest portion of a mesh on one side of a plane."""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import triangulate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mesh", type=Path)
    parser.add_argument("output_mesh", type=Path)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--minimum", type=float)
    parser.add_argument(
        "--normal",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional arbitrary keep-side plane normal.",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Point on the arbitrary cut plane.",
    )
    parser.add_argument(
        "--boundary-vertices",
        type=int,
        default=0,
        help=(
            "Reduce only the cut boundary to this many vertices before "
            "capping; the retained source surface is otherwise unchanged."
        ),
    )
    parser.add_argument(
        "--voxel-pitch",
        type=float,
        default=0.0,
        help="Optional watertight remeshing pitch used after capping.",
    )
    return parser.parse_args()


def boundary_loop(mesh):
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    adjacency = {}
    for first, second in boundary:
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("cut must produce one or more simple boundary loops")

    start = min(adjacency)
    loop = [start]
    previous = None
    current = start
    while True:
        first, second = adjacency[current]
        following = first if first != previous else second
        if following == start:
            break
        if following in loop:
            raise ValueError("cut boundary is not a simple loop")
        loop.append(following)
        previous, current = current, following

    if len(loop) != len(adjacency):
        raise ValueError("cut produced multiple boundary loops")
    return np.asarray(loop, dtype=np.int64)


def cap_cut(mesh, normal):
    plane_normal = np.asarray(normal, dtype=np.float64)
    plane_normal /= np.linalg.norm(plane_normal)
    reference = np.eye(3)[np.argmin(np.abs(plane_normal))]
    first_axis = np.cross(plane_normal, reference)
    first_axis /= np.linalg.norm(first_axis)
    second_axis = np.cross(plane_normal, first_axis)
    loop = boundary_loop(mesh)
    points_2d = np.column_stack((
        mesh.vertices[loop] @ first_axis,
        mesh.vertices[loop] @ second_axis,
    ))
    polygon = Polygon(points_2d)
    if not polygon.is_valid or polygon.area <= 0.0:
        raise ValueError("cut boundary does not form a valid cap polygon")

    tree = cKDTree(points_2d)
    cap_faces = []
    for triangle in triangulate(polygon):
        if not polygon.covers(triangle.representative_point()):
            continue
        coordinates = np.asarray(triangle.exterior.coords)[:3]
        distances, local_indices = tree.query(coordinates)
        if distances.max() > 1e-7:
            raise ValueError("cap triangulation inserted an unexpected vertex")
        face = loop[local_indices]
        vertices = mesh.vertices[face]
        face_normal = np.cross(
            vertices[1] - vertices[0], vertices[2] - vertices[0])
        if face_normal @ plane_normal > 0.0:
            face = face[[0, 2, 1]]
        cap_faces.append(face)

    if not cap_faces:
        raise ValueError("cap triangulation produced no faces")
    faces = np.vstack((mesh.faces, np.asarray(cap_faces, dtype=np.int64)))
    capped = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=faces, process=True)
    capped.remove_unreferenced_vertices()
    capped.fix_normals(multibody=False)
    return capped


def simplify_boundary(mesh, target_vertices):
    """Collapse contiguous cut-loop groups without changing interior vertices."""
    loop = boundary_loop(mesh)
    if target_vertices <= 0 or len(loop) <= target_vertices:
        return mesh
    if target_vertices < 3:
        raise ValueError("boundary-vertices must be zero or at least three")

    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    replacement = np.arange(len(vertices) + target_vertices, dtype=np.int64)
    new_vertices = []
    for new_index, group in enumerate(
            np.array_split(np.arange(len(loop)), target_vertices),
            start=len(vertices)):
        boundary_indices = loop[group]
        new_vertices.append(vertices[boundary_indices].mean(axis=0))
        replacement[boundary_indices] = new_index

    vertices = np.vstack((vertices, np.asarray(new_vertices)))
    faces = replacement[faces]
    nondegenerate = (
        (faces[:, 0] != faces[:, 1]) &
        (faces[:, 1] != faces[:, 2]) &
        (faces[:, 2] != faces[:, 0]))
    simplified = trimesh.Trimesh(
        vertices=vertices, faces=faces[nondegenerate], process=True)
    simplified.remove_unreferenced_vertices()
    boundary_loop(simplified)
    return simplified


def main():
    args = parse_args()
    if (args.normal is None) != (args.origin is None):
        raise ValueError("normal and origin must be supplied together")
    if args.normal is None and args.minimum is None:
        raise ValueError("minimum is required for an axis-aligned cut")

    axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]
    source = trimesh.load(args.input_mesh, process=True)
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError("input must contain exactly one triangle mesh")

    if args.normal is None:
        normal = np.zeros(3)
        normal[axis_index] = 1.0
        origin = np.zeros(3)
        origin[axis_index] = args.minimum
    else:
        normal = np.asarray(args.normal, dtype=np.float64)
        normal_length = np.linalg.norm(normal)
        if normal_length <= np.finfo(np.float64).eps:
            raise ValueError("normal must be nonzero")
        normal /= normal_length
        origin = np.asarray(args.origin, dtype=np.float64)
    sliced = trimesh.intersections.slice_mesh_plane(
        source, plane_normal=normal, plane_origin=origin, cap=False)
    sliced.merge_vertices(digits_vertex=8)
    sliced.remove_unreferenced_vertices()
    sliced = max(
        sliced.split(only_watertight=False), key=lambda part: len(part.faces))
    sliced = simplify_boundary(sliced, args.boundary_vertices)
    result = cap_cut(sliced, normal)
    if args.voxel_pitch > 0.0:
        voxels = result.voxelized(
            pitch=args.voxel_pitch, method="subdivide").fill()
        result = voxels.marching_cubes
        result.apply_transform(voxels.transform)
        result.process(validate=True)
        result.fix_normals(multibody=False)

    if not result.is_watertight or not result.is_winding_consistent:
        raise ValueError("extracted mesh is not a consistently oriented closed mesh")
    if len(result.split(only_watertight=False)) != 1:
        raise ValueError("extracted mesh is not a single connected component")

    args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.output_mesh)
    print(
        f"wrote {args.output_mesh}: {len(result.vertices)} vertices, "
        f"{len(result.faces)} faces, Euler {result.euler_number}")


if __name__ == "__main__":
    main()
