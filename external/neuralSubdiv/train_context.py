"""Resumable training entry point for the context-conditioned model."""

from __future__ import print_function

import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

from context_models import ContextSubdNet
from models import SubdNet
from validate_dataset import require_valid_dataset
from train_resume import (
    CHECKPOINT,
    NETPARAMS,
    atomic_torch_save,
    load_best_model_if_available,
    optimizer_to,
    model_state,
    prepare_training_run,
    require_checkpoint_signature,
    save_checkpoint,
    set_rng_state,
    torch_load,
    write_loss_history,
    write_validation_outputs,
)


CORE_INITIAL_STATE = "phase0_initial_state.dat"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_model(params):
    """Create context and Phase-0 models with an exactly shared core start."""
    seed_everything(int(params.get("seed", 0)))

    def initialize_linear(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)

    phase0_model = SubdNet(params)
    phase0_model.apply(initialize_linear)

    core_checkpoint = params.get("core_checkpoint")
    if core_checkpoint:
        phase0_model.load_state_dict(model_state(
            torch_load(core_checkpoint, "cpu")))

    model = ContextSubdNet(params)
    model.copy_core_weights(phase0_model)
    model.zero_context_input_weights()
    return model, phase0_model.state_dict()


def compute_mesh_loss(model, dataset, mesh_index, params, loss_function):
    inputs = dataset.getInputData(mesh_index)
    outputs = model(
        inputs, mesh_index, dataset.hfList, dataset.poolMats, dataset.dofs)
    target = dataset.meshes[mesh_index][params["numSubd"]].V.to(
        params["device"])

    loss = outputs[0].new_zeros(())
    for output in outputs:
        loss = loss + loss_function(output, target[:output.size(0), :])
    return loss


def require_finite_gradients(model, epoch, mesh_index):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(
                "non-finite gradient in %s at epoch %d, mesh %d" %
                (name, epoch, mesh_index))


def context_gradient_l1(model):
    total = 0.0
    parameters = list(model.context_block.parameters())
    parameters.extend(model.context_projection.parameters())
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum().cpu())
    return total


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python train_context.py /path/to/job/")

    folder = sys.argv[1]
    if not folder.endswith("/"):
        folder += "/"
    with open(folder + "hyperparameters.json", "r") as handle:
        params = json.load(handle)
    os.makedirs(params["output_path"], exist_ok=True)
    run_signature = prepare_training_run(folder, params, "context")

    if params["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("hyperparameters.json requests cuda, but CUDA is unavailable")
    if params["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("hyperparameters.json requests mps, but MPS is unavailable")

    with open(params["train_pkl"], "rb") as handle:
        train_data = pickle.load(handle)
    require_valid_dataset(train_data, "training", params["numSubd"])
    train_data.computeParameters()
    train_data.toDevice(params["device"])

    with open(params["valid_pkl"], "rb") as handle:
        valid_data = pickle.load(handle)
    require_valid_dataset(valid_data, "validation", params["numSubd"])
    valid_data.computeParameters()
    valid_data.toDevice(params["device"])

    if train_data.nM == 0 or valid_data.nM == 0:
        raise ValueError("training and validation PKLs must both contain meshes")

    model, phase0_initial_state = initialize_model(params)
    model = model.to(params["device"])
    loss_function = torch.nn.MSELoss().to(params["device"])
    core_parameters = []
    for predictor in (model.net_init, model.net_vertex, model.net_edge):
        core_parameters.extend(predictor.parameters())
    context_parameters = list(model.context_block.parameters())
    context_parameters.extend(model.context_projection.parameters())
    optimizer = torch.optim.Adam([
        {"params": core_parameters, "lr": params["lr"]},
        {"params": context_parameters,
         "lr": params.get("context_lr", params["lr"] * 0.1)},
    ])

    checkpoint_path = params["output_path"] + CHECKPOINT
    core_initial_path = params["output_path"] + CORE_INITIAL_STATE
    train_history = []
    valid_history = []
    best_loss = np.inf
    start_epoch = 0
    optimizer_steps = 0
    context_activated = False

    if os.path.exists(checkpoint_path):
        checkpoint = torch_load(checkpoint_path, params["device"])
        require_checkpoint_signature(checkpoint, run_signature)
        model.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_to(optimizer, params["device"])
        set_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint["next_epoch"])
        best_loss = checkpoint["best_loss"]
        train_history = list(checkpoint.get("train_loss_his", []))
        valid_history = list(checkpoint.get("valid_loss_his", []))
        optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        context_activated = bool(checkpoint.get("context_activated", False))
        print("resuming context training: epoch %d / %d, best valid %.6e" %
              (start_epoch, params["epochs"], best_loss), flush=True)
    else:
        atomic_torch_save(phase0_initial_state, core_initial_path)
        print("starting fresh context training", flush=True)
        print("matching Phase-0 initial state saved: %s" % core_initial_path,
              flush=True)

    try:
        for epoch in range(start_epoch, params["epochs"]):
            started = time.time()
            model.train()
            train_error = 0.0
            for mesh_index in range(train_data.nM):
                loss = compute_mesh_loss(
                    model, train_data, mesh_index, params, loss_function)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        "non-finite train loss at epoch %d, mesh %d" %
                        (epoch, mesh_index))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                require_finite_gradients(model, epoch, mesh_index)
                if optimizer_steps > 0 and not context_activated:
                    if context_gradient_l1(model) == 0.0:
                        raise RuntimeError(
                            "Context Block still has zero gradient on the "
                            "second optimizer step; refusing ineffective training")
                    context_activated = True
                    print("Context Block gradient activation verified", flush=True)
                optimizer.step()
                optimizer_steps += 1
                train_error += float(loss.detach().cpu().item())
            train_history.append(train_error / train_data.nM)

            model.eval()
            valid_error = 0.0
            with torch.no_grad():
                for mesh_index in range(valid_data.nM):
                    loss = compute_mesh_loss(
                        model, valid_data, mesh_index, params, loss_function)
                    loss_value = float(loss.cpu().item())
                    if not math.isfinite(loss_value):
                        raise FloatingPointError(
                            "non-finite valid loss at epoch %d, mesh %d" %
                            (epoch, mesh_index))
                    valid_error += loss_value
            valid_mean = valid_error / valid_data.nM
            valid_history.append(valid_mean)

            if valid_mean < best_loss:
                best_loss = valid_mean
                atomic_torch_save(
                    model.state_dict(), params["output_path"] + NETPARAMS)

            write_loss_history(params, train_history, valid_history)
            save_checkpoint(
                checkpoint_path, model, optimizer, epoch + 1, best_loss,
                train_history, valid_history, run_signature=run_signature,
                extra_state={
                    "optimizer_steps": optimizer_steps,
                    "context_activated": context_activated,
                })

            seconds_left = int(round(
                (params["epochs"] - epoch - 1) * (time.time() - started)))
            print("epoch %d, train loss %.6e, valid loss %.6e, remain time: %d" %
                  (epoch, train_history[-1], valid_history[-1], seconds_left),
                  flush=True)
            print("checkpoint saved: %s" % checkpoint_path, flush=True)

    except KeyboardInterrupt:
        print("interrupted; the latest completed epoch is saved at %s" %
              checkpoint_path, flush=True)
        raise
    except FloatingPointError as error:
        print("training aborted: %s" % error, flush=True)
        print("the previous completed-epoch checkpoint remains at %s" %
              checkpoint_path, flush=True)
        raise

    write_loss_history(params, train_history, valid_history)
    save_checkpoint(
        checkpoint_path, model, optimizer, params["epochs"], best_loss,
        train_history, valid_history, completed=True,
        run_signature=run_signature, extra_state={
            "optimizer_steps": optimizer_steps,
            "context_activated": context_activated,
        })
    load_best_model_if_available(params, model)
    write_validation_outputs(params, valid_data, model)
    print("context training complete: %s" % checkpoint_path, flush=True)


if __name__ == "__main__":
    main()
