import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
NEURAL_SUBDIV = ROOT / "external" / "neuralSubdiv"
os.chdir(NEURAL_SUBDIV)
sys.path.insert(0, str(NEURAL_SUBDIV))

from context_models import (  # noqa: E402
    ContextSubdNet,
    HalfFlapContextProjection,
    InvariantVertexGeometry,
)
from include import computeFlapList  # noqa: E402
from models import SubdNet  # noqa: E402
from train_context import context_gradient_l1, initialize_model  # noqa: E402
from train_resume import (  # noqa: E402
    initialize_phase0_model,
    prepare_training_run,
)


def make_octahedron_problem(num_subd=1):
    vertices = torch.tensor([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    faces = torch.tensor([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ], dtype=torch.long)
    _, generated_half_flaps = computeFlapList(vertices, faces, num_subd)
    half_flaps = generated_half_flaps[:-1]

    pool_matrices = []
    degrees = []
    for level_half_flaps in half_flaps:
        vertex_count = int(level_half_flaps[:, 0].max().item()) + 1
        row = level_half_flaps[:, 0]
        column = torch.arange(level_half_flaps.size(0))
        indices = torch.stack((row, column), dim=0)
        values = torch.ones(level_half_flaps.size(0))
        pool = torch.sparse_coo_tensor(
            indices, values, (vertex_count, level_half_flaps.size(0)))
        pool_matrices.append(pool)
        degrees.append(torch.sparse.sum(pool, dim=1).to_dense())

    first_half_flaps = half_flaps[0]
    edge_differences = (
        vertices[first_half_flaps[:, 0]] -
        vertices[first_half_flaps[:, 1]])
    laplace = torch.sparse.mm(pool_matrices[0], edge_differences)
    laplace = laplace / degrees[0].unsqueeze(1)
    inputs = torch.cat((vertices, laplace), dim=1)
    return inputs, [half_flaps], [pool_matrices], [degrees]


def make_params(num_subd=1):
    return {
        "Din": 6,
        "Dout": 32,
        "h_initNet": [32, 32],
        "h_edgeNet": [32, 32],
        "h_vertexNet": [32, 32],
        "numSubd": num_subd,
        "context_dim": 64,
        "context_hidden_dim": 64,
        "context_heads": 4,
        "context_local_layers": 2,
        "context_global_layers": 1,
        "context_feed_forward_dim": 96,
        "context_dropout": 0.0,
        "context_global_chunk_size": 8,
    }


def fixed_rotation():
    axis = torch.tensor([1.0, 2.0, -0.5])
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.73)
    cross = torch.tensor([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    identity = torch.eye(3)
    return (identity * torch.cos(angle) +
            (1.0 - torch.cos(angle)) * axis[:, None] * axis[None, :] +
            torch.sin(angle) * cross)


class ContextModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.inputs, cls.half_flaps, cls.pool_matrices, cls.degrees = (
            make_octahedron_problem())
        cls.params = make_params()

    def test_invariant_geometry_ignores_rigid_motion(self):
        geometry = InvariantVertexGeometry()
        base, _, _, _ = geometry(
            self.inputs[:, :3], self.half_flaps[0][0])

        rotation = fixed_rotation()
        translation = torch.tensor([[0.6, -1.2, 2.3]])
        moved_positions = self.inputs[:, :3].mm(rotation.t()) + translation
        moved, _, _, _ = geometry(
            moved_positions, self.half_flaps[0][0])
        torch.testing.assert_close(base, moved, rtol=2e-5, atol=2e-5)

    def test_planar_dihedral_features_have_finite_gradients(self):
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ], requires_grad=True)
        half_flap = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        features, _, _, _ = InvariantVertexGeometry()(positions, half_flap)
        features.sum().backward()
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.isfinite(positions.grad).all())

    def test_global_radius_uses_object_scale(self):
        geometry = InvariantVertexGeometry()
        _, local_scale, global_scale, radial = geometry(
            self.inputs[:, :3], self.half_flaps[0][0])
        self.assertGreater(float(local_scale), 0.0)
        self.assertGreater(float(global_scale), 0.0)
        torch.testing.assert_close(
            radial.square().mean(), torch.tensor(1.0), rtol=1e-6, atol=1e-6)

    def test_context_block_is_vertex_permutation_equivariant(self):
        torch.manual_seed(9)
        model = ContextSubdNet(self.params).eval()
        positions = self.inputs[:, :3]
        half_flaps = self.half_flaps[0][0]
        permutation = torch.tensor([3, 5, 1, 0, 4, 2])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())

        with torch.no_grad():
            expected = model.context_block(positions, half_flaps)
            actual = model.context_block(
                positions[permutation], inverse[half_flaps])
        torch.testing.assert_close(
            expected[permutation], actual, rtol=2e-5, atol=2e-5)

    def test_adaptive_selector_is_permutation_equivariant(self):
        torch.manual_seed(10)
        params = dict(self.params, context_adaptive_local=True)
        model = ContextSubdNet(params).eval()
        positions = self.inputs[:, :3]
        half_flaps = self.half_flaps[0][0]
        permutation = torch.tensor([3, 5, 1, 0, 4, 2])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())

        with torch.no_grad():
            expected, expected_weights = model.context_block(
                positions, half_flaps, return_scale_weights=True)
            actual, actual_weights = model.context_block(
                positions[permutation], inverse[half_flaps],
                return_scale_weights=True)
        torch.testing.assert_close(
            expected[permutation], actual, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(
            expected_weights[permutation], actual_weights,
            rtol=2e-5, atol=2e-5)

    def test_adaptive_selector_starts_uniform_and_supports_fixed_depths(self):
        torch.manual_seed(12)
        params = dict(self.params, context_adaptive_local=True)
        model = ContextSubdNet(params).eval()
        positions = self.inputs[:, :3]
        half_flaps = self.half_flaps[0][0]

        with torch.no_grad():
            _, learned = model.context_block(
                positions, half_flaps, return_scale_weights=True)
            _, fixed_first = model.context_block(
                positions, half_flaps, local_context_mode="fixed_1",
                return_scale_weights=True)
            _, fixed_last = model.context_block(
                positions, half_flaps, local_context_mode="fixed_last",
                return_scale_weights=True)

        self.assertEqual(tuple(learned.shape), (positions.size(0), 2))
        torch.testing.assert_close(
            learned, torch.full_like(learned, 0.5), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            learned.sum(dim=1), torch.ones(positions.size(0)),
            rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            fixed_first[:, 0], torch.ones(positions.size(0)),
            rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            fixed_last[:, 1], torch.ones(positions.size(0)),
            rtol=0.0, atol=0.0)

    def test_adaptive_model_preserves_common_seeded_initialization(self):
        torch.manual_seed(14)
        fixed = ContextSubdNet(self.params)
        torch.manual_seed(14)
        adaptive = ContextSubdNet(
            dict(self.params, context_adaptive_local=True))
        adaptive_state = adaptive.state_dict()
        for name, value in fixed.state_dict().items():
            torch.testing.assert_close(
                value, adaptive_state[name], rtol=0.0, atol=0.0)

    def test_context_disabled_matches_original_core(self):
        torch.manual_seed(11)
        core = SubdNet(self.params).eval()
        model = ContextSubdNet(self.params).eval()
        model.copy_core_weights(core)

        with torch.no_grad():
            expected = core(self.inputs.clone(), 0, self.half_flaps,
                            self.pool_matrices, self.degrees)
            disabled = model(
                self.inputs.clone(), 0, self.half_flaps,
                self.pool_matrices, self.degrees, context_enabled=False)
            zero_conditioned = model(
                self.inputs.clone(), 0, self.half_flaps,
                self.pool_matrices, self.degrees, context_enabled=True)

        self.assertEqual(len(expected), len(disabled))
        for core_output, disabled_output, conditioned_output in zip(
                expected, disabled, zero_conditioned):
            torch.testing.assert_close(core_output, disabled_output,
                                       rtol=0.0, atol=0.0)
            torch.testing.assert_close(core_output, conditioned_output,
                                       rtol=1e-6, atol=1e-7)

    def test_indexed_pool_matches_sparse_pool(self):
        torch.manual_seed(13)
        model = SubdNet(self.params)
        values = torch.randn(self.half_flaps[0][0].size(0), 7)
        sparse = self.pool_matrices[0][0]
        targets = self.half_flaps[0][0][:, 0]
        expected = model.oneRingPool(values, sparse, self.degrees[0][0])
        actual = model.oneRingPool(values, targets, self.degrees[0][0])
        torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)

    def test_phase0_can_load_context_matching_initial_state(self):
        context_model, initial_state = initialize_model(self.params)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "phase0_initial_state.dat")
            torch.save(initial_state, path)
            params = dict(self.params, initial_checkpoint=path)
            phase0_model = initialize_phase0_model(params)

        expected = SubdNet(self.params)
        expected.load_state_dict(initial_state)
        for expected_parameter, actual_parameter in zip(
                expected.parameters(), phase0_model.parameters()):
            torch.testing.assert_close(
                expected_parameter, actual_parameter, rtol=0.0, atol=0.0)
        self.assertGreater(
            sum(parameter.numel() for parameter in context_model.parameters()),
            sum(parameter.numel() for parameter in phase0_model.parameters()))

    def test_context_core_checkpoint_accepts_full_training_checkpoint(self):
        core = SubdNet(self.params)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "checkpoint.pt")
            torch.save({"net_state_dict": core.state_dict()}, path)
            model, initial_state = initialize_model(
                dict(self.params, core_checkpoint=path))
        for name, value in core.state_dict().items():
            torch.testing.assert_close(
                value, initial_state[name], rtol=0.0, atol=0.0)
        self.assertIsInstance(model, ContextSubdNet)

    def test_context_gradient_activates_on_second_step(self):
        torch.manual_seed(31)
        model = ContextSubdNet(self.params)
        model.zero_context_input_weights()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        gradients = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                self.inputs.clone(), 0, self.half_flaps,
                self.pool_matrices, self.degrees)
            sum(output.square().mean() for output in outputs).backward()
            gradients.append(context_gradient_l1(model))
            optimizer.step()
        self.assertEqual(gradients[0], 0.0)
        self.assertGreater(gradients[1], 0.0)

    def test_run_manifest_rejects_changed_training_data(self):
        with tempfile.TemporaryDirectory() as folder:
            train_path = os.path.join(folder, "train.pkl")
            valid_path = os.path.join(folder, "valid.pkl")
            with open(train_path, "wb") as handle:
                handle.write(b"train-v1")
            with open(valid_path, "wb") as handle:
                handle.write(b"valid-v1")
            params = dict(
                self.params,
                output_path=folder,
                train_pkl=train_path,
                valid_pkl=valid_path,
                lr=0.002,
                seed=4,
            )
            first = prepare_training_run(folder, params, "context")
            second = prepare_training_run(
                folder, dict(params, epochs=100), "context")
            self.assertEqual(first, second)
            with open(train_path, "ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(RuntimeError):
                prepare_training_run(folder, params, "context")

    def test_default_parameter_count_stays_mac_sized(self):
        params = make_params(num_subd=2)
        params.update({
            "context_local_layers": 6,
            "context_global_layers": 2,
            "context_feed_forward_dim": 128,
        })
        core = SubdNet(params)
        context = ContextSubdNet(params)
        self.assertEqual(sum(p.numel() for p in core.parameters()), 15104)
        self.assertEqual(sum(p.numel() for p in context.parameters()), 313696)

        adaptive = ContextSubdNet(dict(params, context_adaptive_local=True))
        self.assertEqual(sum(p.numel() for p in adaptive.parameters()), 314759)

    def test_default_attention_depth_matches_first_configuration(self):
        params = make_params()
        del params["context_local_layers"]
        del params["context_global_layers"]
        model = ContextSubdNet(params)
        self.assertEqual(len(model.context_block.local_layers), 6)
        self.assertEqual(len(model.context_block.global_layers), 2)

    def test_active_context_preserves_rigid_motion_behavior(self):
        torch.manual_seed(19)
        model = ContextSubdNet(self.params).eval()
        with torch.no_grad():
            for predictor in (model.net_init, model.net_vertex, model.net_edge):
                predictor.layerIn.weight[:, predictor.input_dim:].normal_(
                    mean=0.0, std=0.02)

            original = model(
                self.inputs.clone(), 0, self.half_flaps,
                self.pool_matrices, self.degrees)

            rotation = fixed_rotation()
            translation = torch.tensor([[0.6, -1.2, 2.3]])
            moved_input = self.inputs.clone()
            moved_input[:, :3] = moved_input[:, :3].mm(rotation.t()) + translation
            moved_input[:, 3:] = moved_input[:, 3:].mm(rotation.t())
            moved = model(
                moved_input, 0, self.half_flaps,
                self.pool_matrices, self.degrees)

        for original_output, moved_output in zip(original, moved):
            expected = original_output.mm(rotation.t()) + translation
            torch.testing.assert_close(expected, moved_output,
                                       rtol=5e-5, atol=5e-5)

    def test_adaptive_context_preserves_rigid_motion_behavior(self):
        torch.manual_seed(20)
        params = dict(self.params, context_adaptive_local=True)
        model = ContextSubdNet(params).eval()
        with torch.no_grad():
            for predictor in (model.net_init, model.net_vertex, model.net_edge):
                predictor.layerIn.weight[:, predictor.input_dim:].normal_(
                    mean=0.0, std=0.02)

            original = model(
                self.inputs.clone(), 0, self.half_flaps,
                self.pool_matrices, self.degrees)
            rotation = fixed_rotation()
            translation = torch.tensor([[0.6, -1.2, 2.3]])
            moved_input = self.inputs.clone()
            moved_input[:, :3] = moved_input[:, :3].mm(rotation.t()) + translation
            moved_input[:, 3:] = moved_input[:, 3:].mm(rotation.t())
            moved = model(
                moved_input, 0, self.half_flaps,
                self.pool_matrices, self.degrees)

        for original_output, moved_output in zip(original, moved):
            expected = original_output.mm(rotation.t()) + translation
            torch.testing.assert_close(expected, moved_output,
                                       rtol=5e-5, atol=5e-5)

    def test_rollout_shapes_context_history_and_gradients(self):
        torch.manual_seed(23)
        model = ContextSubdNet(self.params)
        with torch.no_grad():
            model.net_init.layerIn.weight[:, model.net_init.input_dim:].fill_(0.01)
            model.net_vertex.layerIn.weight[:, model.net_vertex.input_dim:].fill_(0.01)
            model.net_edge.layerIn.weight[:, model.net_edge.input_dim:].fill_(0.01)

        outputs, contexts = model(
            self.inputs.clone(), 0, self.half_flaps,
            self.pool_matrices, self.degrees, return_context=True)
        self.assertEqual([tuple(output.shape) for output in outputs],
                         [(6, 3), (18, 3)])
        self.assertEqual(len(contexts), 2)
        self.assertEqual(tuple(contexts[0].shape), (24, 64))
        self.assertEqual(tuple(contexts[1].shape), (24, 64))
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs))

        sum(output.square().mean() for output in outputs).backward()
        gradient = model.context_block.geometry_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_adaptive_rollout_returns_weights_and_trains_selector(self):
        torch.manual_seed(24)
        params = dict(self.params, context_adaptive_local=True)
        model = ContextSubdNet(params)
        with torch.no_grad():
            for predictor in (model.net_init, model.net_vertex, model.net_edge):
                predictor.layerIn.weight[:, predictor.input_dim:].fill_(0.01)

        outputs, selectors = model(
            self.inputs.clone(), 0, self.half_flaps,
            self.pool_matrices, self.degrees, return_selector_weights=True)
        self.assertEqual(len(selectors), 2)
        self.assertTrue(all(selector is not None for selector in selectors))
        self.assertEqual(tuple(selectors[0].shape), (6, 2))
        self.assertEqual(tuple(selectors[1].shape), (6, 2))
        for selector in selectors:
            self.assertTrue(torch.isfinite(selector).all())
            self.assertTrue(bool((selector >= 0.0).all()))
            torch.testing.assert_close(
                selector.sum(dim=1), torch.ones(selector.size(0)),
                rtol=1e-6, atol=1e-6)

        sum(output.square().mean() for output in outputs).backward()
        gradient = model.context_block.local_selector.scorer[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_half_flap_projection_uses_canonical_order(self):
        projection = HalfFlapContextProjection(vertex_context_dim=1,
                                               context_dim=1)
        with torch.no_grad():
            projection.network[0].weight.copy_(
                torch.tensor([[1.0, 10.0, 100.0, 1000.0]]))
            projection.network[0].bias.zero_()
            projection.network[2].weight.fill_(1.0)
            projection.network[2].bias.zero_()

        vertex_context = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        half_flap = torch.tensor([[2, 0, 3, 1]])
        result = projection(vertex_context, half_flap)
        self.assertEqual(float(result.item()), 2413.0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_context_forward_and_backward(self):
        device = torch.device("mps")
        model = ContextSubdNet(self.params).to(device)
        half_flaps = [[value.to(device) for value in self.half_flaps[0]]]
        pool_indices = [[value[:, 0].contiguous() for value in half_flaps[0]]]
        degrees = [[value.to(device) for value in self.degrees[0]]]
        outputs = model(
            self.inputs.to(device), 0, half_flaps, pool_indices, degrees)
        loss = sum(output.square().mean() for output in outputs)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss).cpu()))
        self.assertTrue(all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().cpu())
            for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
