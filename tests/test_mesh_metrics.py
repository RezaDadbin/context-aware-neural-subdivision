import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
NEURAL_SUBDIV = ROOT / "external" / "neuralSubdiv"
os.chdir(NEURAL_SUBDIV)
sys.path.insert(0, str(NEURAL_SUBDIV))

from mesh_metrics import (  # noqa: E402
    correspondence_metrics,
    core_frame_metrics,
    mesh_quality_metrics,
    midpoint_subdivision_faces,
    region_correspondence_metrics,
    surface_distance_metrics,
    topology_metrics,
)
from validate_dataset import require_valid_dataset, validate_chain  # noqa: E402


class MeshMetricTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ])
        self.faces = np.array([
            [0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3],
        ])

    def test_closed_tetrahedron_quality(self):
        result = mesh_quality_metrics(self.vertices, self.faces)
        self.assertTrue(result["finite_vertices"])
        self.assertEqual(result["boundary_edges"], 0)
        self.assertEqual(result["nonmanifold_edges"], 0)
        self.assertEqual(result["connected_components"], 1)
        self.assertEqual(result["degenerate_face_ratio"], 0.0)

    def test_identical_correspondence_is_exact(self):
        result = correspondence_metrics(
            self.vertices, self.vertices.copy(), self.faces)
        self.assertEqual(result["correspondence_mse"], 0.0)
        self.assertEqual(result["maximum_vertex_distance"], 0.0)
        self.assertAlmostEqual(result["correspondence_normal_cosine"], 1.0)

        regions = region_correspondence_metrics(
            self.vertices, self.vertices.copy(), self.faces)
        self.assertEqual(regions["smooth_region_mse"], 0.0)
        self.assertEqual(regions["detail_region_mse"], 0.0)
        self.assertGreater(regions["smooth_region_vertices"], 0)
        self.assertGreater(regions["detail_region_vertices"], 0)

    def test_surface_sampling_is_deterministic(self):
        result = surface_distance_metrics(
            self.vertices, self.vertices.copy(), self.faces,
            samples=1000, seed=17)
        self.assertEqual(result["chamfer_mean"], 0.0)
        self.assertEqual(result["hausdorff"], 0.0)

    def test_invalid_indices_are_reported_without_crashing(self):
        invalid = self.faces.copy()
        invalid[0, 0] = 100
        result = mesh_quality_metrics(self.vertices, invalid)
        self.assertFalse(result["valid_face_indices"])
        self.assertEqual(result["degenerate_face_ratio"], 1.0)

    def test_core_frame_metric_detects_opposite_adjacent_normals(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        faces = np.array([[0, 1, 2], [1, 0, 3]])
        result = core_frame_metrics(vertices, faces)
        self.assertEqual(result["minimum_core_normal_sum"], 0.0)
        self.assertGreater(result["ill_conditioned_core_frame_ratio"], 0.0)

    def test_official_midpoint_connectivity_is_checked_exactly(self):
        fine_faces = midpoint_subdivision_faces(self.faces, len(self.vertices))
        edges = np.concatenate((
            self.faces[:, [0, 1]], self.faces[:, [1, 2]],
            self.faces[:, [2, 0]]), axis=0)
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        fine_vertices = np.concatenate((
            self.vertices,
            0.5 * (self.vertices[edges[:, 0]] + self.vertices[edges[:, 1]]),
        ))
        finest = SimpleNamespace(
            V=torch.tensor(fine_vertices), F=torch.tensor(fine_faces))
        coarse = SimpleNamespace(
            V=torch.tensor(self.vertices), F=torch.tensor(self.faces))
        args = SimpleNamespace(
            min_core_edge_ratio=1e-5,
            min_core_face_area_ratio=1e-10,
            min_core_normal_sum=1e-3,
            allow_boundary=False,
            max_narrow_ratio=0.1,
            prefix_tolerance=1e-6,
        )
        result = validate_chain([coarse, finest], 0, args)
        self.assertEqual(result["failures"], [])
        require_valid_dataset(
            SimpleNamespace(nM=1, meshes=[[coarse, finest]]),
            "test", expected_subdivisions=1, settings=args)

        damaged_faces = fine_faces.copy()
        damaged_faces[[0, 1]] = damaged_faces[[1, 0]]
        damaged = SimpleNamespace(
            V=torch.tensor(fine_vertices), F=torch.tensor(damaged_faces))
        result = validate_chain([coarse, damaged], 0, args)
        self.assertTrue(any(
            "connectivity/order" in failure for failure in result["failures"]))
        with self.assertRaises(ValueError):
            require_valid_dataset(
                SimpleNamespace(nM=1, meshes=[[coarse, damaged]]),
                "test", expected_subdivisions=1, settings=args)

    def test_topology_reports_inconsistent_orientation(self):
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        result = topology_metrics(faces, 4)
        self.assertEqual(result["inconsistent_oriented_edges"], 1)

    def test_target_frame_warning_does_not_reject_autoregressive_chain(self):
        fine_faces = midpoint_subdivision_faces(self.faces, len(self.vertices))
        edges = np.concatenate((
            self.faces[:, [0, 1]], self.faces[:, [1, 2]],
            self.faces[:, [2, 0]]), axis=0)
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        fine_vertices = np.concatenate((
            self.vertices,
            0.5 * (self.vertices[edges[:, 0]] + self.vertices[edges[:, 1]]),
        ))
        coarse = SimpleNamespace(
            V=torch.tensor(self.vertices), F=torch.tensor(self.faces))
        fine = SimpleNamespace(
            V=torch.tensor(fine_vertices), F=torch.tensor(fine_faces))
        args = SimpleNamespace(
            min_core_edge_ratio=1e-5,
            min_core_face_area_ratio=1e-10,
            min_core_normal_sum=1e-3,
            allow_boundary=False,
            max_narrow_ratio=0.1,
            prefix_tolerance=1e-6,
        )
        original_metric = core_frame_metrics

        def target_frame_warning(vertices, faces, *thresholds):
            result = original_metric(vertices, faces, *thresholds)
            if len(vertices) > len(self.vertices):
                result["ill_conditioned_core_frame_ratio"] = 1.0
            return result

        with patch("validate_dataset.core_frame_metrics",
                   side_effect=target_frame_warning):
            result = validate_chain([coarse, fine], 0, args)

        self.assertTrue(result["levels"][0]["core_frame_used_by_model"])
        self.assertFalse(result["levels"][1]["core_frame_used_by_model"])
        self.assertFalse(any(
            "ill-conditioned" in failure for failure in result["failures"]))


if __name__ == "__main__":
    unittest.main()
