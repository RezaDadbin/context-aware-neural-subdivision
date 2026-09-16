"""Run a trained context-conditioned model on one closed OBJ mesh."""

from __future__ import print_function

import argparse
import json
import os

import torch

from context_models import ContextSubdNet
from include import TestMeshes, random3DRotation, tgp
from train_resume import NETPARAMS, torch_load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", help="folder containing hyperparameters.json")
    parser.add_argument("mesh", help="closed triangular OBJ to subdivide")
    parser.add_argument("--num-subd", type=int, default=2)
    context_mode = parser.add_mutually_exclusive_group()
    context_mode.add_argument("--disable-context", action="store_true")
    context_mode.add_argument("--shuffle-context", action="store_true")
    return parser.parse_args()


def write_outputs(prefix, outputs, meshes, output_path):
    for level, output in enumerate(outputs):
        tgp.writeOBJ(
            os.path.join(output_path, "%s_subd%d.obj" % (prefix, level)),
            output.detach().cpu(), meshes[level].F.cpu())


def main():
    args = parse_args()
    job = args.job if args.job.endswith("/") else args.job + "/"
    with open(job + "hyperparameters.json", "r") as handle:
        params = json.load(handle)
    params["numSubd"] = args.num_subd
    if params["device"] == "cuda" and not torch.cuda.is_available():
        params["device"] = "cpu"
    if params["device"] == "mps" and not torch.backends.mps.is_available():
        params["device"] = "cpu"

    test_data = TestMeshes([args.mesh], params["numSubd"])
    test_data.computeParameters()
    test_data.toDevice(params["device"])

    model = ContextSubdNet(params).to(params["device"])
    state = torch_load(params["output_path"] + NETPARAMS, params["device"])
    model.load_state_dict(state)
    model.eval()

    context_enabled = not args.disable_context
    mode = "context"
    if args.disable_context:
        mode = "core"
    elif args.shuffle_context:
        mode = "context_shuffled"
    mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]

    with torch.no_grad():
        inputs = test_data.getInputData(0)
        outputs = model(
            inputs, 0, test_data.hfList, test_data.poolMats, test_data.dofs,
            context_enabled=context_enabled,
            shuffle_context=args.shuffle_context)
        write_outputs(
            "%s_%s" % (mesh_name, mode), outputs, test_data.meshes[0],
            params["output_path"])

        rotation = random3DRotation().to(params["device"])
        translation = torch.rand(1, 3, device=params["device"])
        moved_inputs = test_data.getInputData(0).clone()
        moved_inputs[:, :3] = moved_inputs[:, :3].mm(rotation.t()) + translation
        moved_inputs[:, 3:] = moved_inputs[:, 3:].mm(rotation.t())
        moved_outputs = model(
            moved_inputs, 0, test_data.hfList, test_data.poolMats,
            test_data.dofs, context_enabled=context_enabled,
            shuffle_context=args.shuffle_context)
        write_outputs(
            "%s_%s_rot" % (mesh_name, mode), moved_outputs,
            test_data.meshes[0], params["output_path"])

    print("wrote %s outputs to %s" % (mode, params["output_path"]))


if __name__ == "__main__":
    main()
