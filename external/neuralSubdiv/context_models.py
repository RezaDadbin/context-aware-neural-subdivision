"""Context-conditioned extension of the released Neural Subdivision model.

The subdivision operators and canonical half-flap processing remain in
``models.SubdNet``.  This module only computes invariant mesh context and adds
that context to the input of the existing I/V/E predictors.
"""

import math

import torch

from models import SubdNet


def _scatter_sum(values, index, size):
    output_shape = (size,) + tuple(values.shape[1:])
    output = values.new_zeros(output_shape)
    output.index_add_(0, index, values)
    return output


def _scatter_mean(values, index, size, eps):
    summed = _scatter_sum(values, index, size)
    counts = values.new_zeros(size)
    counts.index_add_(0, index, values.new_ones(index.numel()))
    denominator = counts.clamp_min(eps)
    while denominator.dim() < summed.dim():
        denominator = denominator.unsqueeze(-1)
    return summed / denominator


def _scatter_mean_std(values, index, size, eps):
    mean = _scatter_mean(values, index, size, eps)
    residual = values - mean[index]
    variance = _scatter_mean(residual * residual, index, size, eps)
    return mean, torch.sqrt(variance + eps)


def _segment_softmax(scores, segment_index, segment_count):
    """Softmax over rows that share a segment, independently per head."""
    expanded_index = segment_index.unsqueeze(1).expand_as(scores)

    if hasattr(torch.Tensor, "scatter_reduce_"):
        maxima = scores.new_full((segment_count, scores.size(1)), -torch.inf)
        maxima.scatter_reduce_(0, expanded_index, scores, reduce="amax",
                               include_self=True)
        stabilized = scores - maxima[segment_index]
        exponentials = torch.exp(stabilized)
        denominators = scores.new_zeros((segment_count, scores.size(1)))
        denominators.scatter_add_(0, expanded_index, exponentials)
        return exponentials / denominators[segment_index].clamp_min(1e-12)

    # Compatibility path for old PyTorch releases used by the original code.
    probabilities = torch.empty_like(scores)
    for segment in range(segment_count):
        mask = segment_index == segment
        probabilities[mask] = torch.softmax(scores[mask], dim=0)
    return probabilities


class InvariantVertexGeometry(torch.nn.Module):
    """Build rigid-motion-invariant scalar descriptors for each vertex."""

    feature_dim = 13

    def __init__(self, eps=1e-8):
        super(InvariantVertexGeometry, self).__init__()
        self.eps = eps

    def _boundary_state(self, half_flaps, vertex_count, dtype):
        undirected = torch.sort(half_flaps[:, :2], dim=1).values
        edge_keys = undirected[:, 0] * vertex_count + undirected[:, 1]
        _, inverse, counts = torch.unique(
            edge_keys, return_inverse=True, return_counts=True)
        boundary_edge = (counts[inverse] == 1).to(dtype)
        boundary_sum = _scatter_sum(
            torch.cat((boundary_edge, boundary_edge), dim=0),
            torch.cat((undirected[:, 0], undirected[:, 1]), dim=0),
            vertex_count)
        return (boundary_sum > 0).to(dtype)

    def forward(self, positions, half_flaps):
        if positions.dim() != 2 or positions.size(1) != 3:
            raise ValueError("positions must have shape [num_vertices, 3]")
        if half_flaps.dim() != 2 or half_flaps.size(1) != 4:
            raise ValueError("half_flaps must have shape [num_half_flaps, 4]")

        vertex_count = positions.size(0)
        target = half_flaps[:, 0]
        neighbor = half_flaps[:, 1]
        edge_vectors = positions[neighbor] - positions[target]
        edge_lengths = torch.linalg.vector_norm(edge_vectors, dim=1)
        local_scale = edge_lengths.mean().clamp_min(self.eps)

        degree = _scatter_sum(edge_lengths.new_ones(edge_lengths.shape), target,
                              vertex_count)
        mean_degree = degree.mean().clamp_min(self.eps)
        neighbor_mean = _scatter_mean(positions[neighbor], target, vertex_count,
                                      self.eps)
        differential_magnitude = torch.linalg.vector_norm(
            positions - neighbor_mean, dim=1) / local_scale

        edge_mean, edge_std = _scatter_mean_std(
            edge_lengths / local_scale, target, vertex_count, self.eps)

        centered = positions - positions.mean(dim=0, keepdim=True)
        global_scale = torch.sqrt(
            centered.square().sum(dim=1).mean()).clamp_min(self.eps)
        radial_distance = torch.linalg.vector_norm(
            centered, dim=1) / global_scale

        p0 = positions[half_flaps[:, 0]]
        p1 = positions[half_flaps[:, 1]]
        p2 = positions[half_flaps[:, 2]]
        p3 = positions[half_flaps[:, 3]]

        normal_a_raw = torch.cross(p1 - p0, p2 - p0, dim=1)
        normal_b_raw = torch.cross(p0 - p1, p3 - p1, dim=1)
        area = 0.5 * torch.linalg.vector_norm(normal_a_raw, dim=1)
        area_vertex_index = torch.cat((half_flaps[:, 0], half_flaps[:, 1],
                                       half_flaps[:, 2]), dim=0)
        area_samples = torch.cat((area, area, area), dim=0) / (
            local_scale * local_scale)
        area_mean, area_std = _scatter_mean_std(
            area_samples, area_vertex_index, vertex_count, self.eps)

        normal_a = torch.nn.functional.normalize(normal_a_raw, dim=1,
                                                 eps=self.eps)
        normal_b = torch.nn.functional.normalize(normal_b_raw, dim=1,
                                                 eps=self.eps)
        dihedral_cosine = (normal_a * normal_b).sum(dim=1).clamp(-1.0, 1.0)
        angle_margin = max(self.eps, torch.finfo(positions.dtype).eps * 8.0)
        dihedral_angle = torch.acos(dihedral_cosine.clamp(
            -1.0 + angle_margin, 1.0 - angle_margin)) / math.pi
        cosine_mean, cosine_std = _scatter_mean_std(
            dihedral_cosine, target, vertex_count, self.eps)
        angle_mean, angle_std = _scatter_mean_std(
            dihedral_angle, target, vertex_count, self.eps)

        boundary = self._boundary_state(half_flaps, vertex_count,
                                        positions.dtype)
        features = torch.stack((
            degree / mean_degree,
            torch.log1p(degree),
            boundary,
            edge_mean,
            edge_std,
            differential_magnitude,
            radial_distance,
            area_mean,
            area_std,
            cosine_mean,
            cosine_std,
            angle_mean,
            angle_std,
        ), dim=1)
        return features, local_scale, global_scale, radial_distance


class RelativeGeometryBias(torch.nn.Module):
    """Map invariant pair geometry to one scalar bias per attention head."""

    def __init__(self, heads, hidden_dim=16):
        super(RelativeGeometryBias, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, heads),
        )

    def forward(self, pair_features):
        return self.network(pair_features)


class TopologyAttentionLayer(torch.nn.Module):
    """Multi-head self-attention restricted to one-ring mesh connectivity."""

    def __init__(self, hidden_dim, heads, feed_forward_dim, dropout):
        super(TopologyAttentionLayer, self).__init__()
        if hidden_dim % heads != 0:
            raise ValueError("context hidden dimension must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relative_bias = RelativeGeometryBias(heads)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm_attention = torch.nn.LayerNorm(hidden_dim)
        self.norm_feed_forward = torch.nn.LayerNorm(hidden_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, feed_forward_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feed_forward_dim, hidden_dim),
        )

    def forward(self, features, positions, half_flaps, local_scale,
                radial_distance):
        vertex_count = features.size(0)
        vertices = torch.arange(vertex_count, device=features.device)
        target = torch.cat((half_flaps[:, 0], vertices), dim=0)
        source = torch.cat((half_flaps[:, 1], vertices), dim=0)

        query = self.query(features).view(vertex_count, self.heads, self.head_dim)
        key = self.key(features).view(vertex_count, self.heads, self.head_dim)
        value = self.value(features).view(vertex_count, self.heads, self.head_dim)

        distance = torch.linalg.vector_norm(
            positions[target] - positions[source], dim=1) / local_scale
        topology_distance = torch.cat((
            features.new_ones(half_flaps.size(0)),
            features.new_zeros(vertex_count),
        ), dim=0)
        radial_difference = torch.abs(
            radial_distance[target] - radial_distance[source])
        pair_features = torch.stack(
            (distance, topology_distance, radial_difference), dim=1)

        scores = (query[target] * key[source]).sum(dim=2)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + self.relative_bias(pair_features)
        attention = _segment_softmax(scores, target, vertex_count)
        attention = self.dropout(attention)

        messages = attention.unsqueeze(2) * value[source]
        aggregated = _scatter_sum(messages, target, vertex_count)
        aggregated = aggregated.reshape(vertex_count, self.hidden_dim)
        features = self.norm_attention(
            features + self.dropout(self.output(aggregated)))
        features = self.norm_feed_forward(
            features + self.dropout(self.feed_forward(features)))
        return features


class GlobalAttentionLayer(torch.nn.Module):
    """Full same-mesh attention with chunked query evaluation."""

    def __init__(self, hidden_dim, heads, feed_forward_dim, dropout,
                 chunk_size=256):
        super(GlobalAttentionLayer, self).__init__()
        if hidden_dim % heads != 0:
            raise ValueError("context hidden dimension must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.chunk_size = chunk_size
        self.query = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relative_bias = RelativeGeometryBias(heads)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm_attention = torch.nn.LayerNorm(hidden_dim)
        self.norm_feed_forward = torch.nn.LayerNorm(hidden_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, feed_forward_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feed_forward_dim, hidden_dim),
        )

    def _topology_relation(self, start, end, vertex_count, half_flaps, dtype):
        relation = torch.full(
            (end - start, vertex_count), 2.0,
            dtype=dtype, device=half_flaps.device)
        local_rows = torch.arange(end - start, device=half_flaps.device)
        global_rows = torch.arange(start, end, device=half_flaps.device)
        relation[local_rows, global_rows] = 0.0

        target = half_flaps[:, 0]
        mask = (target >= start) & (target < end)
        relation[target[mask] - start, half_flaps[mask, 1]] = 1.0
        return relation

    def forward(self, features, positions, half_flaps, global_scale,
                radial_distance):
        vertex_count = features.size(0)
        query = self.query(features).view(vertex_count, self.heads, self.head_dim)
        key = self.key(features).view(vertex_count, self.heads, self.head_dim)
        value = self.value(features).view(vertex_count, self.heads, self.head_dim)
        chunks = []

        chunk_size = self.chunk_size if self.chunk_size > 0 else vertex_count
        for start in range(0, vertex_count, chunk_size):
            end = min(start + chunk_size, vertex_count)
            pair_delta = positions[start:end, None, :] - positions[None, :, :]
            distance = torch.linalg.vector_norm(
                pair_delta, dim=2) / global_scale
            topology_relation = self._topology_relation(
                start, end, vertex_count, half_flaps, features.dtype)
            radial_difference = torch.abs(
                radial_distance[start:end, None] - radial_distance[None, :])
            pair_features = torch.stack(
                (distance, topology_relation, radial_difference), dim=2)
            bias = self.relative_bias(pair_features).permute(2, 0, 1)

            scores = torch.einsum("qhd,khd->hqk", query[start:end], key)
            scores = scores / math.sqrt(self.head_dim)
            attention = torch.softmax(scores + bias, dim=2)
            attention = self.dropout(attention)
            output = torch.einsum("hqk,khd->qhd", attention, value)
            chunks.append(output.reshape(end - start, self.hidden_dim))

        attended = torch.cat(chunks, dim=0)
        features = self.norm_attention(
            features + self.dropout(self.output(attended)))
        features = self.norm_feed_forward(
            features + self.dropout(self.feed_forward(features)))
        return features


class MeshContextBlock(torch.nn.Module):
    """Produce one invariant contextual embedding per mesh vertex."""

    def __init__(self, latent_dim, hidden_dim=64, heads=4, local_layers=6,
                 global_layers=2, feed_forward_dim=128, dropout=0.0,
                 global_chunk_size=256, eps=1e-8):
        super(MeshContextBlock, self).__init__()
        self.geometry = InvariantVertexGeometry(eps=eps)
        self.geometry_projection = torch.nn.Linear(
            self.geometry.feature_dim, hidden_dim)
        self.latent_projection = None
        if latent_dim > 0:
            self.latent_projection = torch.nn.Linear(
                latent_dim, hidden_dim, bias=False)
        self.input_norm = torch.nn.LayerNorm(hidden_dim)
        self.local_layers = torch.nn.ModuleList([
            TopologyAttentionLayer(hidden_dim, heads, feed_forward_dim, dropout)
            for _ in range(local_layers)
        ])
        self.global_layers = torch.nn.ModuleList([
            GlobalAttentionLayer(hidden_dim, heads, feed_forward_dim, dropout,
                                 global_chunk_size)
            for _ in range(global_layers)
        ])
        self.output_norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, positions, half_flaps, invariant_latent=None):
        geometry, local_scale, global_scale, radial_distance = self.geometry(
            positions, half_flaps)
        features = self.geometry_projection(geometry)
        if invariant_latent is not None:
            if self.latent_projection is None:
                raise ValueError("this Context Block was created without latent inputs")
            features = features + self.latent_projection(invariant_latent)
        features = self.input_norm(features)

        for layer in self.local_layers:
            features = layer(features, positions, half_flaps, local_scale,
                             radial_distance)
        for layer in self.global_layers:
            features = layer(features, positions, half_flaps, global_scale,
                             radial_distance)
        return self.output_norm(features)


class HalfFlapContextProjection(torch.nn.Module):
    """Gather four vertex embeddings in canonical half-flap order."""

    def __init__(self, vertex_context_dim=64, context_dim=64):
        super(HalfFlapContextProjection, self).__init__()
        self.vertex_context_dim = vertex_context_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4 * vertex_context_dim, context_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(context_dim, context_dim),
        )

    def forward(self, vertex_context, half_flaps):
        gathered = vertex_context[half_flaps]
        gathered = gathered.reshape(half_flaps.size(0),
                                    4 * self.vertex_context_dim)
        return self.network(gathered)


class ContextConditionedMLP(torch.nn.Module):
    """An original MLP whose first layer also accepts half-flap context."""

    def __init__(self, input_dim, hidden_dims, output_dim, context_dim):
        super(ContextConditionedMLP, self).__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one width")
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.layerIn = torch.nn.Linear(input_dim + context_dim, hidden_dims[0])
        self.hidden = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dims[index], hidden_dims[index + 1])
            for index in range(len(hidden_dims) - 1)
        ])
        self.layerOut = torch.nn.Linear(hidden_dims[-1], output_dim)
        self.relu = torch.nn.ReLU()

    @classmethod
    def from_core(cls, core_mlp, context_dim):
        hidden_dims = [core_mlp.layerIn.out_features]
        hidden_dims.extend(layer.out_features for layer in core_mlp.hidden)
        conditioned = cls(core_mlp.layerIn.in_features, hidden_dims,
                          core_mlp.layerOut.out_features, context_dim)
        conditioned.copy_core_weights(core_mlp)
        return conditioned

    def copy_core_weights(self, core_mlp):
        if core_mlp.layerIn.in_features != self.input_dim:
            raise ValueError("core MLP input dimension does not match")
        with torch.no_grad():
            self.layerIn.weight[:, :self.input_dim].copy_(core_mlp.layerIn.weight)
            self.layerIn.weight[:, self.input_dim:].zero_()
            self.layerIn.bias.copy_(core_mlp.layerIn.bias)
            if len(self.hidden) != len(core_mlp.hidden):
                raise ValueError("core MLP hidden layout does not match")
            for destination, source in zip(self.hidden, core_mlp.hidden):
                destination.weight.copy_(source.weight)
                destination.bias.copy_(source.bias)
            self.layerOut.weight.copy_(core_mlp.layerOut.weight)
            self.layerOut.bias.copy_(core_mlp.layerOut.bias)

    def zero_context_weights(self):
        with torch.no_grad():
            self.layerIn.weight[:, self.input_dim:].zero_()

    def forward(self, half_flap_features, context=None):
        if context is None:
            features = torch.nn.functional.linear(
                half_flap_features,
                self.layerIn.weight[:, :self.input_dim],
                self.layerIn.bias)
        else:
            if context.size(0) != half_flap_features.size(0):
                raise ValueError("context and half-flap counts do not match")
            features = self.layerIn(torch.cat((half_flap_features, context), dim=1))
        features = self.relu(features)
        for layer in self.hidden:
            features = self.relu(layer(features))
        return self.layerOut(features)


class ContextSubdNet(SubdNet):
    """Neural Subdivision with context supplied to the original predictors."""

    def __init__(self, params):
        super(ContextSubdNet, self).__init__(params)
        self.params = dict(params)
        context_dim = int(params.get("context_dim", 64))
        hidden_dim = int(params.get("context_hidden_dim", 64))
        latent_dim = int(params["Dout"]) - 3

        core_init = self.net_init
        core_edge = self.net_edge
        core_vertex = self.net_vertex
        self.net_init = ContextConditionedMLP.from_core(core_init, context_dim)
        self.net_edge = ContextConditionedMLP.from_core(core_edge, context_dim)
        self.net_vertex = ContextConditionedMLP.from_core(core_vertex, context_dim)

        self.context_block = MeshContextBlock(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            heads=int(params.get("context_heads", 4)),
            local_layers=int(params.get("context_local_layers", 6)),
            global_layers=int(params.get("context_global_layers", 2)),
            feed_forward_dim=int(params.get("context_feed_forward_dim", 128)),
            dropout=float(params.get("context_dropout", 0.0)),
            global_chunk_size=int(params.get("context_global_chunk_size", 256)),
            eps=float(params.get("context_geometry_eps", 1e-8)),
        )
        self.context_projection = HalfFlapContextProjection(hidden_dim, context_dim)

    def copy_core_weights(self, core_model):
        self.net_init.copy_core_weights(core_model.net_init)
        self.net_edge.copy_core_weights(core_model.net_edge)
        self.net_vertex.copy_core_weights(core_model.net_vertex)

    def load_core_state_dict(self, state_dict, strict=True):
        core_model = SubdNet(self.params)
        core_model.load_state_dict(state_dict, strict=strict)
        self.copy_core_weights(core_model)

    def zero_context_input_weights(self):
        self.net_init.zero_context_weights()
        self.net_edge.zero_context_weights()
        self.net_vertex.zero_context_weights()

    def _compute_half_flap_context(self, vertex_features, half_flaps,
                                   use_learned_features, shuffle_context):
        invariant_latent = None
        if use_learned_features and vertex_features.size(1) > 3:
            invariant_latent = vertex_features[:, 3:]
        vertex_context = self.context_block(
            vertex_features[:, :3], half_flaps, invariant_latent)
        if shuffle_context:
            permutation = torch.randperm(
                vertex_context.size(0), device=vertex_context.device)
            vertex_context = vertex_context[permutation]
        return self.context_projection(vertex_context, half_flaps)

    def forward(self, fv, mIdx, HFs, poolMats, DOFs, context_enabled=True,
                shuffle_context=False, return_context=False):
        outputs = []
        context_history = []

        initial_half_flaps = HFs[mIdx][0]
        initial_context = None
        if context_enabled:
            initial_context = self._compute_half_flap_context(
                fv, initial_half_flaps, use_learned_features=False,
                shuffle_context=shuffle_context)
        context_history.append(initial_context)

        fv_input_pos = fv[:, :3]
        fhf, local_frames = self.v2hf_initNet(fv, initial_half_flaps)
        fhf = self.net_init(fhf, initial_context)
        fhf = self.local2Global(fhf, local_frames)
        fv = self.oneRingPool(fhf, poolMats[mIdx][0], DOFs[mIdx][0])
        fv[:, :3] += fv_input_pos
        outputs.append(fv[:, :3])

        for level in range(self.numSubd):
            half_flaps = HFs[mIdx][level]
            transition_context = None
            if context_enabled:
                transition_context = self._compute_half_flap_context(
                    fv, half_flaps, use_learned_features=True,
                    shuffle_context=shuffle_context)
            context_history.append(transition_context)

            previous_position = fv[:, :3]
            fhf, local_frames = self.v2hf(fv, half_flaps)
            fhf = self.net_vertex(fhf, transition_context)
            fhf = self.local2Global(fhf, local_frames)
            fv = self.oneRingPool(
                fhf, poolMats[mIdx][level], DOFs[mIdx][level])
            fv[:, :3] += previous_position
            fv_even = fv

            edge_midpoint = self.edgeMidPoint(fv, half_flaps)
            fhf, local_frames = self.v2hf(fv, half_flaps)
            fv_odd = self.net_edge(fhf, transition_context)
            fv_odd = self.local2Global(fv_odd, local_frames)
            fv_odd = self.halfEdgePool(fv_odd)
            fv_odd[:, :3] += edge_midpoint

            fv = torch.cat((fv_even, fv_odd), dim=0)
            outputs.append(fv[:, :3])

        if return_context:
            return outputs, context_history
        return outputs


__all__ = [
    "ContextConditionedMLP",
    "ContextSubdNet",
    "GlobalAttentionLayer",
    "HalfFlapContextProjection",
    "InvariantVertexGeometry",
    "MeshContextBlock",
    "TopologyAttentionLayer",
]
