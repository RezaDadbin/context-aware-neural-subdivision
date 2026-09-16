"""Run finite forward/backward preflights for both experiment models."""

from __future__ import print_function

import argparse
import json
import pickle

import numpy as np
import torch

from context_models import ContextSubdNet
from models import SubdNet
from validate_dataset import require_valid_dataset


def smoke_params(num_subd):
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
        "context_local_layers": 6,
        "context_global_layers": 2,
        "context_feed_forward_dim": 128,
        "context_dropout": 0.0,
        "context_global_chunk_size": 512,
    }


def selected_indices(count, maximum):
    amount = min(count, maximum)
    return sorted(set(np.linspace(0, count - 1, amount, dtype=int).tolist()))


def require_finite_model(model, dataset, indices, num_subd, label):
    loss_function = torch.nn.MSELoss()
    checked = []
    for mesh_index in indices:
        model.zero_grad(set_to_none=True)
        outputs = model(
            dataset.getInputData(mesh_index), mesh_index,
            dataset.hfList, dataset.poolMats, dataset.dofs)
        target = dataset.meshes[mesh_index][num_subd].V
        if len(outputs) != num_subd + 1:
            raise RuntimeError("%s returned the wrong number of levels" % label)
        if not all(torch.isfinite(output).all().item() for output in outputs):
            raise FloatingPointError(
                "%s produced non-finite output for mesh %d" %
                (label, mesh_index))
        loss = sum(loss_function(output, target[:output.size(0)])
                   for output in outputs)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                "%s produced non-finite loss for mesh %d" %
                (label, mesh_index))
        loss.backward()
        for name, parameter in model.named_parameters():
            if (parameter.grad is not None and
                    not torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(
                    "%s produced non-finite gradient in %s for mesh %d" %
                    (label, name, mesh_index))
        checked.append({"mesh": mesh_index, "loss": float(loss.detach())})
    return checked


def smoke_test(dataset, maximum):
    num_subd = dataset.nS - 1
    require_valid_dataset(dataset, "smoke test", num_subd)
    dataset.computeParameters()
    dataset.toDevice("cpu")
    indices = selected_indices(dataset.nM, maximum)
    params = smoke_params(num_subd)

    torch.manual_seed(1729)
    core = SubdNet(params)
    phase0 = require_finite_model(core, dataset, indices, num_subd, "phase0")

    torch.manual_seed(1729)
    context = ContextSubdNet(params)
    context.copy_core_weights(core)
    context.zero_context_input_weights()
    # Exercise gradients through the context path, not only core-parity mode.
    with torch.no_grad():
        for predictor in (context.net_init, context.net_vertex,
                          context.net_edge):
            predictor.layerIn.weight[:, predictor.input_dim:].fill_(1e-4)
    phase1 = require_finite_model(
        context, dataset, indices, num_subd, "phase1")
    context_gradient = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in context.named_parameters()
        if parameter.grad is not None and
        (name.startswith("context_block.") or
         name.startswith("context_projection.")))
    if not np.isfinite(context_gradient) or context_gradient <= 0.0:
        raise FloatingPointError(
            "phase1 context path did not produce a finite non-zero gradient")
    return {
        "meshes": dataset.nM,
        "subdivisions": num_subd,
        "indices": indices,
        "phase0": phase0,
        "phase1": phase1,
        "phase1_context_path_active": True,
        "phase1_context_gradient_l1": context_gradient,
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl")
    parser.add_argument("--max-meshes", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_meshes <= 0:
        raise ValueError("--max-meshes must be positive")
    with open(args.pkl, "rb") as handle:
        dataset = pickle.load(handle)
    result = smoke_test(dataset, args.max_meshes)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
